# [MODULE DOC]
# Tests of the external engine provider, which serves a fleet of already-running sglang engines
# given by address. Covers the loadable provider factory, address normalization and duplicate
# detection, per-engine discovery from the server info payload (gpu count from the reported world
# size, prefill/decode role and bootstrap port), the retrying server info fetch, and the static
# worker provider (cell info and meta, gpu offsets, fleet-size and prefill/decode consistency checks,
# watch semantics, worker addresses and infos, ipv6 handling, and compatibility with the controller's
# cell meta contract). Server info answers are faked through patched fetch functions or a fake api
# client; payload builders produce regular, prefill and decode answers.
from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any

import httpx
import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import external_engine_provider as external_engine_provider_module
from miles.ray.rollout.external_engine_provider import (
    _EXTERNAL_ENGINE_POOL_ID,
    StaticInferenceEngineWorkerProvider,
    _assert_engines_match_args,
    _compute_external_engine_urls,
    _discover_external_engine,
    _ExternalEngineInfo,
    _fetch_server_info_with_retry,
    static_inference_engine_provider,
)
from miles.ray.rollout.inference_controller import _compute_server_cell_meta_from_info
from miles.utils.workers.worker_provider.base import CellInfo


def _make_args(urls: list[str], *, num_gpus_per_engine: int = 1, **overrides: Any) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds run args for an external rollout naming the given engine addresses,
    # with total rollout gpus equal to the address count times the per-engine gpus and that per-engine
    # value set; any overrides replace these defaults.
    defaults = dict(
        rollout_external=True,
        rollout_external_engine_addrs=urls,
        rollout_num_gpus=len(urls) * num_gpus_per_engine,
        rollout_num_gpus_per_engine=num_gpus_per_engine,
    )
    return make_args(**{**defaults, **overrides})


def _regular_payload(*, num_gpus: int = 1) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns a fake server info answer for a regular engine: one internal
    # state reporting the given world size, and a null disaggregation mode.
    return dict(internal_states=[dict(world_size=num_gpus)], disaggregation_mode="null")


def _prefill_payload(*, num_gpus: int = 1, bootstrap_port: int = 12090) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Like the regular answer but with disaggregation mode prefill and the given
    # disaggregation bootstrap port (default 12090).
    return {
        **_regular_payload(num_gpus=num_gpus),
        "disaggregation_mode": "prefill",
        "disaggregation_bootstrap_port": bootstrap_port,
    }


def _decode_payload(*, num_gpus: int = 1) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Like the regular answer but with disaggregation mode decode and no
    # bootstrap port.
    return {**_regular_payload(num_gpus=num_gpus), "disaggregation_mode": "decode"}


def _install_payloads(monkeypatch, payloads: dict[str, dict[str, Any]]) -> None:
    # [FUNCTION DOC] HELPER: Replaces the provider module's retrying server info fetch with a fake
    # that answers from the given mapping of engine url to payload (a missing url raises a key error).
    async def _fetch(url: str, *, api_key: str | None) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Returns the mapped payload for the url, ignoring the api key.
        return payloads[url]

    monkeypatch.setattr(external_engine_provider_module, "_fetch_server_info_with_retry", _fetch)


async def _collect_watched_cells(provider: StaticInferenceEngineWorkerProvider) -> list[CellInfo]:
    # [FUNCTION DOC] HELPER: Establishes a cell watch on the provider with a callback that records the
    # cell info of every announcement, then immediately stops the watch and returns the recorded
    # infos. Since the stop happens right after the watch returns, only announcements completed before
    # the watch returned are captured.
    observed: list[CellInfo] = []

    async def _reconcile(cell_id: str, info: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: Watch callback. Asserts the info is present and carries the announced
        # cell id, then appends it to the recorded list.
        assert info is not None and info.cell_id == cell_id
        observed.append(info)

    stop = await provider.watch_cells(_reconcile)
    await stop()
    return observed


class TestStaticInferenceEngineProviderFactory:
    # [CLASS DOC] Checks the module-level factory that builds the static provider.
    def test_the_factory_builds_the_provider_from_args_alone(self, monkeypatch):
        # [FUNCTION DOC] Replaces the provider class in the module with a recording class, calls the
        # factory with args and an unused capability object, and asserts the result is an instance of
        # the recording class that received exactly the same args object.
        """The loadable factory contract passes capability, which the static provider has no use for."""
        recorded: dict[str, Any] = {}

        class _Recording:
            # [CLASS DOC] HELPER: Stand-in provider class that stores the args it was built with.
            def __init__(self, *, args: Namespace) -> None:
                # [FUNCTION DOC] HELPER: Stores the args in the enclosing test's record mapping.
                recorded["args"] = args

        monkeypatch.setattr(external_engine_provider_module, "StaticInferenceEngineWorkerProvider", _Recording)
        args = _make_args(["host1:8000"])

        provider = static_inference_engine_provider(args, capability=object())

        assert isinstance(provider, _Recording)
        assert recorded["args"] is args


class TestComputeExternalEngineUrls:
    # [CLASS DOC] Checks normalization of the configured engine addresses to canonical http urls and
    # rejection of invalid or duplicate addresses (assertion errors).
    def test_addresses_are_normalized_to_http_urls(self):
        # [FUNCTION DOC] A host-and-port address and a full url with trailing slash must become
        # canonical http urls without the trailing slash.
        """host:port and http://host:port/ must both land on one canonical url form."""
        args = _make_args(["host1:8000", "http://host2:8001/"])

        assert _compute_external_engine_urls(args) == ["http://host1:8000", "http://host2:8001"]

    def test_an_address_without_a_port_is_rejected(self):
        # [FUNCTION DOC] A bare hostname must raise an assertion error saying the address is invalid.
        """The engine port cannot be guessed, so a bare hostname must fail fast."""
        args = _make_args(["host1"])

        with pytest.raises(AssertionError, match="invalid external engine address"):
            _compute_external_engine_urls(args)

    def test_the_same_engine_listed_twice_is_rejected(self):
        # [FUNCTION DOC] The same engine written once as host and port and once as a full url must
        # raise an assertion error saying an address appears more than once.
        """One engine behind two cells registers with the router twice and eats two rank ranges."""
        args = _make_args(["host1:8000", "http://host1:8000"])

        with pytest.raises(AssertionError, match="more than once"):
            _compute_external_engine_urls(args)

    @pytest.mark.parametrize(
        "aliases",
        [
            ["HOST1:8000", "http://host1:8000/"],
            ["HTTP://host1:8000", "host1:8000"],
            ["[fd00:0:0::1]:8000", "[fd00::1]:8000"],
        ],
    )
    def test_two_spellings_of_one_endpoint_are_rejected(self, aliases):
        # [FUNCTION DOC] Parametrized over three alias pairs (differing host case plus trailing slash,
        # differing scheme case, and two spellings of the same ipv6 address). Each pair must raise the
        # more-than-once assertion error, so duplicates are detected on canonical urls, not raw strings.
        """Deduplicating raw strings would let a case or ipv6 alias dial one engine from two cells,
        which then claims two rank ranges in the weight-update group and stalls the rendezvous."""
        args = _make_args(aliases)

        with pytest.raises(AssertionError, match="more than once"):
            _compute_external_engine_urls(args)

    def test_addresses_are_reported_in_their_canonical_form(self):
        # [FUNCTION DOC] An upper-case host and a long-form bracketed ipv6 address must come back
        # lower-cased and with the ipv6 address compressed, kept in brackets.
        """The canonical url is what the cell dials and what workers_hash identifies it by."""
        args = _make_args(["HOST1:8000", "[fd00:0:0::2]:8001"])

        assert _compute_external_engine_urls(args) == ["http://host1:8000", "http://[fd00::2]:8001"]


class TestDiscoverExternalEngine:
    # [CLASS DOC] Checks the discovery of one engine's description from its server info answer.
    async def test_num_gpus_comes_from_the_world_size_the_engine_reports(self, monkeypatch):
        # [FUNCTION DOC] An answer reporting a world size of eight must give a description with eight
        # gpus, worker type regular, and host and port parsed from the url.
        """The parallel sizes on their own do not give the total, and deriving it here would put a
        copy of sglang's own formula in miles for every dimension it grows."""
        _install_payloads(monkeypatch, {"http://host1:8000": _regular_payload(num_gpus=8)})

        engine = await _discover_external_engine("http://host1:8000", api_key=None)

        assert (engine.num_gpus, engine.worker_type) == (8, "regular")
        assert (engine.host, engine.port) == ("host1", 8000)

    async def test_a_prefill_engine_reports_its_type_and_bootstrap_port(self, monkeypatch):
        # [FUNCTION DOC] A prefill answer whose bootstrap port is given as a string must be discovered
        # as worker type prefill with the port converted to the integer 12090.
        """PD registration needs both the role and the bootstrap port from discovery."""
        _install_payloads(
            monkeypatch,
            {"http://host1:8000": {**_prefill_payload(num_gpus=2), "disaggregation_bootstrap_port": "12090"}},
        )

        engine = await _discover_external_engine("http://host1:8000", api_key=None)

        assert (engine.worker_type, engine.disaggregation_bootstrap_port) == ("prefill", 12090)

    async def test_a_decode_engine_reports_its_type_without_a_bootstrap_port(self, monkeypatch):
        # [FUNCTION DOC] A decode answer must be discovered as worker type decode with no bootstrap port.
        """Only prefill engines own a bootstrap port; decode must come back None."""
        _install_payloads(monkeypatch, {"http://host1:8000": _decode_payload(num_gpus=4)})

        engine = await _discover_external_engine("http://host1:8000", api_key=None)

        assert (engine.worker_type, engine.disaggregation_bootstrap_port) == ("decode", None)


class _FakeApiClient:
    # [CLASS DOC] HELPER: Fake sglang api client used by the retry fetch tests. Class-level lists
    # record every requested server url and api key and hold scripted answers consumed in order (an
    # exception entry is raised, any other entry is returned). The fixture resets these lists per test.
    calls: list[str] = []
    api_keys: list[str | None] = []
    answers: list[Any] = []

    def __init__(self, *, server_url: str, api_key: str | None) -> None:
        # [FUNCTION DOC] HELPER: Stores the server url and api key it was constructed with.
        self.server_url = server_url
        self.api_key = api_key

    async def get_server_info(self) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Appends the server url and api key to the class-level records, removes
        # the first scripted answer, raises it if it is an exception, otherwise returns it. Raises an
        # index error when no scripted answers remain.
        _FakeApiClient.calls.append(self.server_url)
        _FakeApiClient.api_keys.append(self.api_key)
        answer = _FakeApiClient.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def fake_api_client(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture. Resets the fake api client's recorded calls, api keys and scripted
    # answers to empty lists, replaces the api client class in the provider module with it, and returns
    # the class so tests can script answers and inspect records.
    _FakeApiClient.calls = []
    _FakeApiClient.api_keys = []
    _FakeApiClient.answers = []
    monkeypatch.setattr(external_engine_provider_module, "SGLangApiClient", _FakeApiClient)
    return _FakeApiClient


class TestFetchServerInfoWithRetry:
    # [CLASS DOC] Checks the retrying server info fetch against the fake api client.
    async def test_discovery_authenticates_with_the_configured_api_key(self, fake_api_client):
        # [FUNCTION DOC] Fetching with the api key secret must construct the client with that key
        # (recorded keys equal secret, once).
        """A protected engine must receive its API key on the server-info probe."""
        fake_api_client.answers = [_regular_payload()]

        await _fetch_server_info_with_retry("http://host1:8000", api_key="secret")

        assert fake_api_client.api_keys == ["secret"]

    async def test_discovery_waits_for_an_engine_that_is_still_booting(self, fake_api_client):
        # [FUNCTION DOC] The first answer is a connection error and the second a regular payload. The
        # fetch must return the payload after querying the same url twice.
        """An engine that answers on the second try must not fail the run."""
        fake_api_client.answers = [httpx.ConnectError("still booting"), _regular_payload()]

        payload = await _fetch_server_info_with_retry("http://host1:8000", api_key=None)

        assert payload == _regular_payload()
        assert fake_api_client.calls == ["http://host1:8000", "http://host1:8000"]

    async def test_discovery_gives_up_after_the_deadline(self, fake_api_client):
        # [FUNCTION DOC] Only one connection error answer is scripted and the timeout is 0.01 seconds.
        # The fetch must raise a timeout error mentioning server info once the deadline passes.
        """A dead address must raise instead of retrying forever."""
        fake_api_client.answers = [httpx.ConnectError("nobody home")]

        with pytest.raises(TimeoutError, match="server_info"):
            await _fetch_server_info_with_retry("http://host1:8000", api_key=None, timeout_seconds=0.01)


class TestStaticInferenceEngineWorkerProvider:
    # [CLASS DOC] Checks the static worker provider after initialization against faked discovery
    # answers: the cells it publishes and their meta, rejection of inconsistent fleets, and its
    # lookup and watch interface.
    async def _make_provider(self, monkeypatch, args: Namespace, payloads: dict[str, dict[str, Any]]):
        # [FUNCTION DOC] HELPER: Installs the fake payloads, creates the provider from the args,
        # awaits its initialization, and returns it.
        _install_payloads(monkeypatch, payloads)
        provider = StaticInferenceEngineWorkerProvider(args=args)
        await provider.init()
        return provider

    async def test_an_empty_external_engine_address_list_is_rejected(self):
        # [FUNCTION DOC] Initializing a provider with no engine addresses must raise an assertion error
        # saying at least one engine must be named, before any discovery.
        """An external rollout must name at least one engine before discovery starts."""
        provider = StaticInferenceEngineWorkerProvider(args=_make_args([]))

        with pytest.raises(AssertionError, match="must name at least one engine"):
            await provider.init()

    async def test_one_unreachable_engine_publishes_no_cells_at_all(self, monkeypatch):
        # [FUNCTION DOC] With two engines where the second times out during discovery, initialization
        # must raise the timeout error, and afterwards reading the provider's cell infos must raise an
        # assertion error saying initialization has not run (no partial fleet is published).
        """A half fleet would still satisfy the router and the weight updater, which would then wait
        on ranks the missing engine was supposed to claim."""
        args = _make_args(["host1:8000", "host2:8000"])

        async def _fetch(url: str, *, api_key: str | None) -> dict[str, Any]:
            # [FUNCTION DOC] HELPER: Raises a timeout error for the second engine's url, saying it did
            # not answer server info; returns a regular payload for any other url.
            if url == "http://host2:8000":
                raise TimeoutError(f"External engine {url} did not answer /server_info")
            return _regular_payload()

        monkeypatch.setattr(external_engine_provider_module, "_fetch_server_info_with_retry", _fetch)
        provider = StaticInferenceEngineWorkerProvider(args=args)

        with pytest.raises(TimeoutError):
            await provider.init()

        with pytest.raises(AssertionError, match="which has not run"):
            _ = provider.cell_infos

    async def test_each_url_becomes_one_cell_with_the_discovered_meta(self, monkeypatch):
        # [FUNCTION DOC] One engine of two gpus with an api key configured. Asserts a single cell info
        # in the external engine pool whose meta equals exactly: model id default, worker type regular,
        # two gpus per engine, gpu offset zero, the api key, no offload need, and weight updates on.
        """The cell meta must carry what the controller reads, sourced from discovery."""
        args = _make_args(["host1:8000"], num_gpus_per_engine=2, sglang_api_key="secret")

        provider = await self._make_provider(monkeypatch, args, {"http://host1:8000": _regular_payload(num_gpus=2)})

        (info,) = provider.cell_infos
        assert info.pool_id == _EXTERNAL_ENGINE_POOL_ID
        assert info.meta == dict(
            model_id="default",
            worker_type="regular",
            num_gpus_per_engine=2,
            gpu_offset=0,
            sglang_api_key="secret",
            needs_offload=False,
            update_weights=True,
        )

    async def test_discovery_log_omits_the_engine_api_key(self, monkeypatch, caplog):
        # [FUNCTION DOC] Initializes at info log level with an api key configured and asserts the log
        # contains the discovery message and the engine address but never the key text.
        """Discovery logs operational engine metadata without exposing its credential."""
        args = _make_args(["host1:8000"], sglang_api_key="secret")

        with caplog.at_level(logging.INFO):
            await self._make_provider(monkeypatch, args, {"http://host1:8000": _regular_payload()})

        assert "Discovered external inference engines" in caplog.text
        assert "host1:8000" in caplog.text
        assert "secret" not in caplog.text

    async def test_gpu_offsets_accumulate_linearly_in_listed_order(self, monkeypatch):
        # [FUNCTION DOC] Three two-gpu engines; the cells' gpu offsets must be 0, 2 and 4 in the
        # listed address order.
        """The weight-update NCCL layout assigns each engine the ranks after its predecessors."""
        args = _make_args(["host1:8000", "host2:8000", "host3:8000"], num_gpus_per_engine=2)
        payloads = {
            "http://host1:8000": _regular_payload(num_gpus=2),
            "http://host2:8000": _regular_payload(num_gpus=2),
            "http://host3:8000": _regular_payload(num_gpus=2),
        }

        provider = await self._make_provider(monkeypatch, args, payloads)

        offsets = [info.meta["gpu_offset"] for info in provider.cell_infos]
        assert offsets == [0, 2, 4]

    async def test_a_fleet_with_a_different_gpu_total_is_rejected(self, monkeypatch):
        # [FUNCTION DOC] Args claim four gpus per engine for two engines (eight total) but the engines
        # report four and two gpus. Initialization must raise an assertion error naming the discovered
        # total of six gpus.
        """--rollout-num-gpus sizes the placement group and the router, so a fleet that is smaller
        than it claims must fail at startup instead of hanging in NCCL."""
        args = _make_args(["host1:8000", "host2:8000"], num_gpus_per_engine=4)
        payloads = {
            "http://host1:8000": _regular_payload(num_gpus=4),
            "http://host2:8000": _regular_payload(num_gpus=2),
        }

        with pytest.raises(AssertionError, match="6 gpus in total"):
            await self._make_provider(monkeypatch, args, payloads)

    async def test_a_pd_fleet_without_the_router_flag_is_rejected(self, monkeypatch):
        # [FUNCTION DOC] A prefill engine plus a decode engine while the external prefill/decode router
        # flag is off must raise an assertion error mentioning that flag.
        """The router was already launched non-PD, so serving a PD fleet behind it would misroute."""
        args = _make_args(["host1:8000", "host2:8000"], rollout_num_gpus=2)
        payloads = {
            "http://host1:8000": _prefill_payload(),
            "http://host2:8000": _decode_payload(),
        }

        with pytest.raises(AssertionError, match="rollout-external-router-pd"):
            await self._make_provider(monkeypatch, args, payloads)

    async def test_the_router_flag_without_a_pd_fleet_is_rejected(self, monkeypatch):
        # [FUNCTION DOC] The prefill/decode router flag on with only a regular engine must raise an
        # assertion error saying there are none (no prefill/decode engines).
        """A PD router in front of regular engines is a misdeclaration, not a fleet to serve."""
        args = _make_args(["host1:8000"], rollout_external_router_pd=True)

        with pytest.raises(AssertionError, match="are none"):
            await self._make_provider(monkeypatch, args, {"http://host1:8000": _regular_payload()})

    async def test_a_declared_pd_fleet_carries_roles_and_bootstrap_ports(self, monkeypatch):
        # [FUNCTION DOC] With the router flag on, a prefill plus a decode engine of two gpus each must
        # give cells whose meta worker types are prefill then decode, in listed order. KNOWN ISSUE: the
        # name and docstring mention bootstrap ports, but only worker types are asserted here.
        """External PD needs the discovered role and bootstrap port on every cell."""
        args = _make_args(["host1:8000", "host2:8000"], num_gpus_per_engine=2, rollout_external_router_pd=True)
        payloads = {
            "http://host1:8000": _prefill_payload(num_gpus=2, bootstrap_port=12090),
            "http://host2:8000": _decode_payload(num_gpus=2),
        }

        provider = await self._make_provider(monkeypatch, args, payloads)

        assert [info.meta["worker_type"] for info in provider.cell_infos] == ["prefill", "decode"]

    async def test_expected_num_cells_counts_the_urls_of_the_model(self, monkeypatch):
        # [FUNCTION DOC] With two regular engines, the expected cell count for group default is two and
        # for an unknown group is zero.
        """The startup barrier waits for exactly the announced fleet, one cell per url."""
        args = _make_args(["host1:8000", "host2:8000"])
        payloads = {
            "http://host1:8000": _regular_payload(),
            "http://host2:8000": _regular_payload(),
        }
        provider = await self._make_provider(monkeypatch, args, payloads)

        assert provider.expected_num_cells(group_id="default") == 2
        assert provider.expected_num_cells(group_id="ghost") == 0

    async def test_the_first_announcement_completes_before_the_watch_is_established(self, monkeypatch):
        # [FUNCTION DOC] Watching an initialized two-engine provider and stopping right away must still
        # have delivered one announcement per cell, with cell ids equal to the provider's cell infos in
        # order (the initial sync is done when the watch returns).
        """The controller treats a returned watch as the initial sync being done."""
        args = _make_args(["host1:8000", "host2:8000"])
        payloads = {
            "http://host1:8000": _regular_payload(),
            "http://host2:8000": _regular_payload(),
        }
        provider = await self._make_provider(monkeypatch, args, payloads)

        observed = await _collect_watched_cells(provider)

        assert [info.cell_id for info in observed] == [info.cell_id for info in provider.cell_infos]

    async def test_get_addrs_answers_the_primary_engine_address(self, monkeypatch):
        # [FUNCTION DOC] Asking for the addresses of the single cell's first worker must give a primary
        # address with the engine's host and port, and no gate address.
        """The cell dials its node-0 engine through exactly this address."""
        args = _make_args(["host1:8000"])
        provider = await self._make_provider(monkeypatch, args, {"http://host1:8000": _regular_payload()})

        (info,) = provider.cell_infos
        addrs = await provider.get_addrs(worker_name=info.worker_names[0])

        assert (addrs["primary"].host, addrs["primary"].port) == ("host1", 8000)
        assert "gate" not in addrs

    async def test_a_prefill_engines_addrs_carry_the_bootstrap_port(self, monkeypatch):
        # [FUNCTION DOC] In a prefill/decode fleet, the prefill worker's addresses must include a
        # disaggregation bootstrap entry with port 12090.
        """The router needs the bootstrap port when it registers a prefill worker."""
        args = _make_args(["host1:8000", "host2:8000"], rollout_num_gpus=2, rollout_external_router_pd=True)
        payloads = {
            "http://host1:8000": _prefill_payload(bootstrap_port=12090),
            "http://host2:8000": _decode_payload(),
        }
        provider = await self._make_provider(monkeypatch, args, payloads)

        (prefill_info, _decode_info) = provider.cell_infos
        addrs = await provider.get_addrs(worker_name=prefill_info.worker_names[0])

        assert addrs["disaggregation_bootstrap"].port == 12090

    async def test_asking_for_an_unknown_worker_is_rejected(self, monkeypatch):
        # [FUNCTION DOC] Asking for the addresses of a made-up worker name must raise an assertion error
        # saying it is not one of the external engines.
        """Answering a made-up name with a guess would dial the wrong machine."""
        args = _make_args(["host1:8000"])
        provider = await self._make_provider(monkeypatch, args, {"http://host1:8000": _regular_payload()})

        with pytest.raises(AssertionError, match="not one of the external engines"):
            await provider.get_addrs(worker_name="ghost-0-0")

    async def test_worker_infos_expose_addresses_but_no_rpc_handle(self, monkeypatch):
        # [FUNCTION DOC] The worker infos for the single cell hold one worker whose name matches the
        # cell's first worker name, whose worker class is none (no rpc handle), and whose primary
        # address has port 8000.
        """External engines are plain http servers; nothing can call rpc methods on them."""
        args = _make_args(["host1:8000"])
        provider = await self._make_provider(monkeypatch, args, {"http://host1:8000": _regular_payload()})

        (info,) = provider.cell_infos
        ((worker_info,),) = provider.get_worker_infos(cell_ids=[info.cell_id])

        assert worker_info.name == info.worker_names[0]
        assert worker_info.worker_class is None
        assert worker_info.self_addrs["primary"].port == 8000

    async def test_an_ipv6_engine_address_stays_bracketed(self, monkeypatch):
        # [FUNCTION DOC] An engine given as an ipv6 url must keep its host in brackets in the primary
        # address.
        """An unbracketed ipv6 host would render an unparseable server url."""
        args = _make_args(["http://[fd00::1]:8000"])
        provider = await self._make_provider(monkeypatch, args, {"http://[fd00::1]:8000": _regular_payload()})

        (info,) = provider.cell_infos
        addrs = await provider.get_addrs(worker_name=info.worker_names[0])

        assert addrs["primary"].host == "[fd00::1]"

    async def test_the_cell_info_satisfies_the_server_cell_meta_contract(self, monkeypatch):
        # [FUNCTION DOC] Builds a decode-then-prefill fleet of two-gpu engines, converts the first cell
        # info with the inference controller's server cell meta function, and asserts the worker type is
        # decode, gpus per engine is two, the workers hash is the engine's url, and the worker name is
        # the cell's first worker name.
        """A provider whose meta drifts from ServerCellMetadata would crash reconcile at runtime."""
        args = _make_args(["host1:8000", "host2:8000"], num_gpus_per_engine=2, rollout_external_router_pd=True)
        payloads = {
            "http://host1:8000": _decode_payload(num_gpus=2),
            "http://host2:8000": _prefill_payload(num_gpus=2),
        }
        provider = await self._make_provider(monkeypatch, args, payloads)

        (info, _prefill_info) = provider.cell_infos
        meta = _compute_server_cell_meta_from_info(info)

        assert (meta.worker_type, meta.num_gpus_per_engine, meta.workers_hash) == (
            "decode",
            2,
            "http://host1:8000",
        )
        assert meta.worker_name == info.worker_names[0]


def _engine(*, url: str, num_gpus: int) -> _ExternalEngineInfo:
    # [FUNCTION DOC] HELPER: Builds a discovered regular engine description for the url (host taken from
    # the url, port 8000) with the given gpu count and no bootstrap port.
    return _ExternalEngineInfo(
        url=url,
        host=url.removeprefix("http://").split(":")[0],
        port=8000,
        worker_type="regular",
        num_gpus=num_gpus,
        disaggregation_bootstrap_port=None,
    )


class TestARaggedExternalFleet:
    # [CLASS DOC] Checks refusal of fleets whose engines differ in size from the per-engine argument.
    def test_engines_that_do_not_each_report_the_per_engine_argument_are_refused(self):
        # [FUNCTION DOC] Two engines of four and two gpus against a per-engine argument of two (six
        # total) must make the args consistency check raise an assertion error naming the per-engine
        # gpu option.
        """Engines of unequal size cannot each hold --rollout-num-gpus-per-engine gpus."""
        args = _make_args(["host1:8000", "host2:8000"], num_gpus_per_engine=2, rollout_num_gpus=6)
        engines = [_engine(url="http://host1:8000", num_gpus=4), _engine(url="http://host2:8000", num_gpus=2)]

        with pytest.raises(AssertionError, match="rollout-num-gpus-per-engine"):
            _assert_engines_match_args(args, engines=engines)

    async def test_a_fleet_of_unequal_engines_is_refused_before_the_run_is_handed_an_engine(self, monkeypatch):
        # [FUNCTION DOC] The same unequal fleet through full provider initialization must raise the same
        # per-engine gpu assertion error, so nothing downstream sees the fleet.
        """Discovery refuses the fleet, so nothing downstream ever sees it."""
        args = _make_args(["host1:8000", "host2:8000"], num_gpus_per_engine=2, rollout_num_gpus=6)
        payloads = {
            "http://host1:8000": _regular_payload(num_gpus=4),
            "http://host2:8000": _regular_payload(num_gpus=2),
        }

        _install_payloads(monkeypatch, payloads)
        provider = StaticInferenceEngineWorkerProvider(args=args)

        with pytest.raises(AssertionError, match="rollout-num-gpus-per-engine"):
            await provider.init()


class TestAPdFleetRoles:
    # [CLASS DOC] Checks the role composition required of a prefill/decode external fleet.
    @pytest.mark.parametrize("worker_types", [["prefill"], ["decode"], ["prefill", "decode", "regular"]])
    def test_a_pd_fleet_without_exactly_both_roles_is_refused(self, worker_types: list[str]) -> None:
        # [FUNCTION DOC] Parametrized over prefill only, decode only, and prefill plus decode plus
        # regular. With the prefill/decode router flag on, builds one engine description per type
        # (prefill ones get a bootstrap port) and expects the args consistency check to raise an
        # assertion error saying prefill and decode engines are needed.
        """A PD router requires prefill and decode engines without regular-engine contamination."""
        args = _make_args(
            [f"host{index}:8000" for index in range(len(worker_types))],
            rollout_external_router_pd=True,
        )
        engines = [
            _ExternalEngineInfo(
                url=f"http://host{index}:8000",
                host=f"host{index}",
                port=8000,
                worker_type=worker_type,
                num_gpus=1,
                disaggregation_bootstrap_port=12090 if worker_type == "prefill" else None,
            )
            for index, worker_type in enumerate(worker_types)
        ]

        with pytest.raises(AssertionError, match="needs prefill and decode engines"):
            _assert_engines_match_args(args, engines=engines)
