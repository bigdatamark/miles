# [MODULE DOC] Worker provider for rollout engines that were started outside this run (external
# SGLang servers given by address). At init it parses and normalizes the configured addresses,
# discovers each engine by fetching its server info (with retries up to 600 seconds), cross-checks
# the discovered GPU counts and prefill/decode mode against launch args, and presents every engine
# as one cell of a single fixed pool with one worker. All cells are reported alive and static.
import asyncio
import ipaddress
import logging
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils.http_utils import wrap_ipv6
from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

logger = logging.getLogger(__name__)

DISCOVERY_TIMEOUT_SECONDS = 600.0

_EXTERNAL_ENGINE_POOL_ID = "external-inference-engine-0"
_EXTERNAL_MODEL_ID = "default"


# ================================== provider ==================================


def static_inference_engine_provider(
    args: Any, *, capability: BackendCapability
) -> "StaticInferenceEngineWorkerProvider":
    # [FUNCTION DOC] Factory used by backend wiring: returns a provider built from args only.
    # KNOWN ISSUE: the capability argument is unused.
    return StaticInferenceEngineWorkerProvider(args=args)


@dataclass(frozen=True)
class _ExternalCell:
    # [CLASS DOC] HELPER: Frozen pairing of a cell's reconcile info and the named host/port
    # addresses of its single engine worker.
    cell_info: CellInfo
    addrs: NamedHostAndPorts


class StaticInferenceEngineWorkerProvider(BaseWorkerProvider):
    # [CLASS DOC] Provider with a fixed, discovered set of cells, one per external engine. Cells are
    # unset until init runs; any accessor used before init raises an assertion error. Also keeps a
    # reverse index from worker name to cell id.

    def __init__(self, *, args: Any) -> None:
        # [FUNCTION DOC] Keep args; leave cells unset (not yet discovered) and the worker-name index
        # empty.
        self._args = args
        self._cells: dict[str, _ExternalCell] | None = None
        self._cell_id_by_worker_name: dict[str, str] = {}

    async def init(self) -> None:
        # [FUNCTION DOC] Discover the external engines: compute normalized urls from args (asserting
        # non-empty and unique), fetch server info from all concurrently (using the sglang api key),
        # assert the discovered engines match launch args, build cells, build the worker name to
        # cell id index, and log a description. Network I/O; raises timeout error if an engine
        # never answers, assertion error on mismatches.
        urls = _compute_external_engine_urls(self._args)
        engines = await _discover_external_engines(urls, api_key=self._args.sglang_api_key)
        _assert_engines_match_args(self._args, engines=engines)

        self._cells = _compute_cells(args=self._args, engines=engines)
        self._cell_id_by_worker_name = {
            worker_name: cell_id
            for cell_id, cell in self._cells.items()
            for worker_name in cell.cell_info.worker_names
        }

        logger.info(f"Discovered external inference engines: {_describe_cells(self._cells)}")

    @property
    def cell_infos(self) -> list[CellInfo]:
        # [FUNCTION DOC] Property: the reconcile info of every discovered cell, in discovery order.
        # Requires init to have run.
        return [cell.cell_info for cell in self._initialized_cells.values()]

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] Return the named addresses (primary, and disaggregation bootstrap when the
        # engine has one) of the cell owning the worker. Raises assertion error for an unknown worker
        # name or before init.
        assert worker_name in self._cell_id_by_worker_name, (
            f"{worker_name} is not one of the external engines this provider was given "
            f"({sorted(self._cell_id_by_worker_name)})"
        )
        return self._initialized_cells[self._cell_id_by_worker_name[worker_name]].addrs

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        # [FUNCTION DOC] For each requested cell id (in order) return its list of worker infos: one
        # per worker name, generation always zero, the cell's addresses, and an empty GPU id list
        # (external engines' GPUs are unknown to this run). Unknown cell ids raise a key error.
        cells = self._initialized_cells
        return [
            [
                WorkerInfo(
                    name=worker_name,
                    generation=0,
                    self_addrs=cells[cell_id].addrs,
                    gpu_ids=[],
                )
                for worker_name in cells[cell_id].cell_info.worker_names
            ]
            for cell_id in cell_ids
        ]

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        # [FUNCTION DOC] Static provider: call the reconcile callback once per cell with its id and
        # info (awaiting each in order), then return a stop function that does nothing. No further
        # change notifications are ever delivered.
        for cell_id, cell in self._initialized_cells.items():
            await reconcile(cell_id, cell.cell_info)

        async def _stop() -> None:
            # [FUNCTION DOC] HELPER: No-op stop callback; nothing to unsubscribe.
            pass

        return _stop

    def expected_num_cells(self, *, group_id: str) -> int:
        # [FUNCTION DOC] Count of cells whose model id metadata equals the given group id. Since all
        # cells carry the model id "default", this equals the cell count for that id and zero for
        # any other.
        return sum(1 for cell in self._initialized_cells.values() if cell.cell_info.meta["model_id"] == group_id)

    @property
    def _initialized_cells(self) -> dict[str, _ExternalCell]:
        # [FUNCTION DOC] HELPER: Property returning the cell mapping (cell id to cell); raises
        # assertion error if init has not populated it.
        assert self._cells is not None, "the provider discovers its engines in init(), which has not run"
        return self._cells


# ============================= cell construction ==============================


@dataclass(frozen=True)
class _ExternalEngineInfo:
    # [CLASS DOC] HELPER: Frozen facts about one discovered engine: normalized url, host, port,
    # worker type (regular, prefill or decode), GPU count (world size), and the optional
    # prefill/decode bootstrap port.
    url: str
    host: str
    port: int
    worker_type: Literal["regular", "prefill", "decode"]
    num_gpus: int
    disaggregation_bootstrap_port: int | None


def _compute_cells(*, args: Any, engines: list[_ExternalEngineInfo]) -> dict[str, _ExternalCell]:
    # [FUNCTION DOC] HELPER: Build one cell per engine in order (cell index = position) in the
    # external pool. Cell id and worker name come from the naming helpers; the cell is alive with a
    # single worker, workers hash equal to the engine url, and metadata: model id "default", worker
    # type, GPUs per engine, a running GPU offset (cumulative GPU count of earlier engines), the
    # sglang api key from args, no offload needed, and weight updates enabled. Returns an ordered
    # mapping from cell id to cell.
    cells: dict[str, _ExternalCell] = {}
    gpu_offset = 0
    for cell_index, engine in enumerate(engines):
        cell_id = compute_cell_id(pool_id=_EXTERNAL_ENGINE_POOL_ID, cell_index=cell_index)
        worker_name = compute_worker_name(pool_id=_EXTERNAL_ENGINE_POOL_ID, cell_index=cell_index)
        cells[cell_id] = _ExternalCell(
            cell_info=CellInfo(
                cell_id=cell_id,
                pool_id=_EXTERNAL_ENGINE_POOL_ID,
                alive=True,
                worker_names=[worker_name],
                workers_hash=engine.url,
                meta=dict(
                    model_id=_EXTERNAL_MODEL_ID,
                    worker_type=engine.worker_type,
                    num_gpus_per_engine=engine.num_gpus,
                    gpu_offset=gpu_offset,
                    sglang_api_key=args.sglang_api_key,
                    needs_offload=False,
                    update_weights=True,
                ),
            ),
            addrs=_compute_engine_addrs(engine),
        )
        gpu_offset += engine.num_gpus
    return cells


def _compute_engine_addrs(engine: _ExternalEngineInfo) -> NamedHostAndPorts:
    # [FUNCTION DOC] HELPER: Mapping with a primary address (host, port) and, only when the engine
    # has a bootstrap port, a disaggregation bootstrap address on the same host.
    addrs: NamedHostAndPorts = dict(primary=HostAndPort(host=engine.host, port=engine.port))
    if engine.disaggregation_bootstrap_port is not None:
        addrs["disaggregation_bootstrap"] = HostAndPort(host=engine.host, port=engine.disaggregation_bootstrap_port)
    return addrs


def _describe_cells(cells: dict[str, _ExternalCell]) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Summaries for logging: per cell a mapping of cell id, url, worker type,
    # GPUs per engine and GPU offset.
    return [
        dict(
            cell_id=cell.cell_info.cell_id,
            url=cell.cell_info.workers_hash,
            worker_type=cell.cell_info.meta["worker_type"],
            num_gpus_per_engine=cell.cell_info.meta["num_gpus_per_engine"],
            gpu_offset=cell.cell_info.meta["gpu_offset"],
        )
        for cell in cells.values()
    ]


# ============================ address book parsing ============================


def _compute_external_engine_urls(args: Any) -> list[str]:
    # [FUNCTION DOC] HELPER: Normalize every configured external engine address from args into an
    # http url, preserving order. Asserts the list is non-empty and has no duplicates after
    # normalization (a duplicate would register one engine twice with the router and weight-update
    # group).
    addrs: list[str] = args.rollout_external_engine_addrs
    assert addrs, "--rollout-external-engine-addrs must name at least one engine"

    urls = [_normalize_external_engine_url(addr) for addr in addrs]

    duplicates = sorted({url for url in urls if urls.count(url) > 1})
    assert not duplicates, (
        f"--rollout-external-engine-addrs lists {duplicates} more than once; one engine would be registered "
        f"with the router twice and would take two rank slots in the weight-update group"
    )
    return urls


def _normalize_external_engine_url(addr: str) -> str:
    # [FUNCTION DOC] HELPER: Turn "host:port" or an http url into canonical "http://host:port":
    # prepend the http scheme when no scheme separator is present, drop trailing slashes, parse and
    # validate, and rebuild with the canonicalized host (IPv6 bracketed). Raises assertion error
    # for invalid addresses.
    url = addr if "://" in addr else f"http://{addr}"
    addr_and_port = _parse_external_engine_url(url.rstrip("/"), source=addr)
    return f"http://{addr_and_port.host}:{addr_and_port.port}"


def _parse_external_engine_url(url: str, *, source: str | None = None) -> HostAndPort:
    # [FUNCTION DOC] HELPER: Parse an http url into a host and port. Asserts the scheme is http and
    # that host and port are present (error message quotes the original source text if given).
    # The host is canonicalized and wrapped in brackets if it is an IPv6 literal.
    parsed = urlsplit(url)
    assert parsed.scheme == "http" and parsed.hostname is not None and parsed.port is not None, (
        f"invalid external engine address {source or url!r}: use host:port or http://host:port "
        f"(bracket IPv6 literals)"
    )
    return HostAndPort(host=wrap_ipv6(_canonicalize_host(parsed.hostname)), port=parsed.port)


def _canonicalize_host(hostname: str) -> str:
    # [FUNCTION DOC] HELPER: If the host is an IP address, return its compressed canonical text form
    # (so equivalent IPv6 spellings compare equal); otherwise return it unchanged.
    try:
        return ipaddress.ip_address(hostname).compressed
    except ValueError:
        return hostname


# ================================= discovery ==================================


async def _discover_external_engines(urls: list[str], *, api_key: str | None) -> list[_ExternalEngineInfo]:
    # [FUNCTION DOC] HELPER: Discover all engines concurrently; result order matches the input url
    # order. The first failure propagates.
    return list(await asyncio.gather(*[_discover_external_engine(url, api_key=api_key) for url in urls]))


async def _discover_external_engine(url: str, *, api_key: str | None) -> _ExternalEngineInfo:
    # [FUNCTION DOC] HELPER: Fetch the engine's server info (with retry) and derive: worker type
    # from its disaggregation mode; GPU count from the world size of the first internal state
    # entry; and the optional disaggregation bootstrap port converted to an integer.
    addr = _parse_external_engine_url(url)
    server_info = await _fetch_server_info_with_retry(url, api_key=api_key)

    return _ExternalEngineInfo(
        url=url,
        host=addr.host,
        port=addr.port,
        worker_type=_infer_worker_type(server_info),
        num_gpus=int(server_info["internal_states"][0]["world_size"]),
        disaggregation_bootstrap_port=(
            int(x) if (x := server_info.get("disaggregation_bootstrap_port")) is not None else None
        ),
    )


async def _fetch_server_info_with_retry(
    url: str, *, api_key: str | None, timeout_seconds: float = DISCOVERY_TIMEOUT_SECONDS
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Repeatedly request the engine's server info until success or the total
    # timeout (default 600 seconds), retrying on HTTP errors and OS-level connection errors, with
    # log fields naming the operation and url. If the deadline is exhausted raises a timeout error
    # mentioning the url and the timeout, chained from the last error.
    async def _attempt(_remaining_seconds: float) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: One attempt: create a fresh api client for the url and key and
        # return its server info mapping. The remaining-seconds argument is ignored.
        return await SGLangApiClient(server_url=url, api_key=api_key).get_server_info()

    try:
        return await retry_until_deadline(
            _attempt,
            total_seconds=timeout_seconds,
            retry_on=(httpx.HTTPError, OSError),
            log_fields=dict(op="discover_external_engine", url=url),
        )
    except (httpx.HTTPError, OSError) as e:
        raise TimeoutError(f"External engine {url} did not answer /server_info within {timeout_seconds}s") from e


def _infer_worker_type(server_info: dict[str, Any]) -> Literal["regular", "prefill", "decode"]:
    # [FUNCTION DOC] HELPER: Return the disaggregation mode from server info if it is prefill or
    # decode; anything else (missing, null, other values) maps to regular.
    mode = server_info.get("disaggregation_mode")
    if mode in ("prefill", "decode"):
        return mode
    return "regular"


# ============================== args cross-check ==============================


def _assert_engines_match_args(args: Any, *, engines: list[_ExternalEngineInfo]) -> None:
    # [FUNCTION DOC] HELPER: Cross-check discovered engines against launch args, raising assertion
    # error with explanatory text on any violation: (1) total GPUs equals the rollout GPU count;
    # (2) the set of per-engine GPU counts is exactly the single configured GPUs per engine;
    # (3) whether any engine is prefill/decode matches the external-router PD flag; (4) when the PD
    # flag is set, the set of worker types is exactly prefill and decode.
    discovered_total = sum(engine.num_gpus for engine in engines)
    assert discovered_total == args.rollout_num_gpus, (
        f"the external engines report {discovered_total} gpus in total "
        f"({[(engine.url, engine.num_gpus) for engine in engines]}), but --rollout-num-gpus is "
        f"{args.rollout_num_gpus}. That argument sizes the placement group and the router, so let it "
        f"describe the fleet that is actually running"
    )

    reported_gpus = {engine.num_gpus for engine in engines}
    assert reported_gpus == {args.rollout_num_gpus_per_engine}, (
        f"external engines report {sorted(reported_gpus)} gpus each, expected "
        f"{args.rollout_num_gpus_per_engine} (--rollout-num-gpus-per-engine)"
    )

    pd_engine_urls = [engine.url for engine in engines if engine.worker_type != "regular"]
    assert bool(pd_engine_urls) == args.rollout_external_router_pd, (
        f"the router is launched in PD mode iff --rollout-external-router-pd is set "
        f"({args.rollout_external_router_pd}), but the engines reporting prefill/decode are "
        f"{pd_engine_urls or 'none'}"
    )

    assert not args.rollout_external_router_pd or {engine.worker_type for engine in engines} == {
        "prefill",
        "decode",
    }, (
        f"--rollout-external-router-pd needs prefill and decode engines, got "
        f"{[(engine.url, engine.worker_type) for engine in engines]}"
    )
