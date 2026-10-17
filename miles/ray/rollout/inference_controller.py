# [MODULE DOC] Inference controller: the single actor-like component that owns all rollout (inference)
# servers of a run, one per served model. It creates the servers and routers, tracks engine cells
# through a provider (reconciling cell add/remove with observed state), runs a periodic tick over
# cells, coordinates offload/onload and weight updates under one context lock, exposes eval fleet
# info and cell status, and accepts engine registrations from external reporters.
# Timing constants: tick interval 5 s, per-cell tick timeout 120 s, cells-ready poll 2 s and
# cells-ready timeout 3600 s.
import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.eval_fleet import EvalFleetInfo, EvalFleetPin, InferenceControllerEvalFleet
from miles.ray.rollout.rollout_server import RolloutServer, create_rollout_servers
from miles.ray.rollout.router_manager import resolve_router_addrs
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils import async_utils
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.context_lock import (
    ContextLock,
    acquires_lock,
    enforce_lock_discipline,
    lock_exempt,
    releases_lock,
    requires_lock,
    with_lock,
)
from miles.utils.ft_utils.api_server.models import CellStatus
from miles.utils.init_once import InitOnce, init_once
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import SimpleTicker
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.registration.hub import RegistrationHub
from miles.utils.workers.registration.models import RegistrationSnapshot
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.utils import apply_cell_observation

logger = logging.getLogger(__name__)

TICK_INTERVAL_SECONDS = 5.0
CELL_TICK_TIMEOUT_SECONDS = 120.0
CELLS_READY_POLL_INTERVAL_SECONDS = 2.0
CELLS_READY_TIMEOUT_SECONDS = 3600.0


@enforce_lock_discipline
class InferenceController:
    # [CLASS DOC] Controller for all inference servers. Lock discipline is enforced by decorators:
    # methods marked with-lock hold the controller's context lock for their whole body; requires-lock
    # methods must be called with it held; acquires-lock/releases-lock methods start/end a lock
    # span across calls (used by the weight update start/end pair); lock-exempt methods never need
    # it. State: args, engine provider, router providers, the context lock, a mapping from model id
    # to rollout server, an optional eval fleet (only when eval GPUs are configured), a list of
    # watcher disposers and an optional ticker. Initialization happens once (idempotent init).

    @lock_exempt
    def __init__(
        self,
        args,
        *,
        engine_provider: BaseWorkerProvider,
        router_providers: Sequence[BaseWorkerProvider],
    ) -> None:
        # [FUNCTION DOC] Store args and providers, create the init-once guard, the context lock named
        # for this controller, and empty server mapping / no eval fleet / no watchers / no ticker.
        # No I/O.
        self._init_once = InitOnce(type(self).__name__)
        self.args = args
        self._engine_provider = engine_provider
        self._router_providers = router_providers
        self.context_lock = ContextLock("InferenceController")
        self.servers: dict[str, RolloutServer] = {}
        self._eval_fleet: InferenceControllerEvalFleet | None = None
        self._watcher_disposers: list[StopWatchFn] = []
        self._ticker: SimpleTicker | None = None

    @lock_exempt
    @init_once
    async def init(self) -> None:
        # [FUNCTION DOC] One-time startup (guarded so repeated calls do nothing). Configures process
        # logging with the inference-controller identity. If the run does not start inference
        # engines, stops there. Otherwise: init the engine provider; resolve router addresses from
        # the router providers; create the rollout servers (mapping model id to server); if eval
        # GPUs > 0 create the eval fleet over the "eval" server (KeyError if absent); subscribe to the
        # provider's cell watch with the reconcile callback and remember its disposer; create the
        # periodic ticker running the cell tick; register the router with the dashboard; and finally
        # wait until every server has its expected number of cells (up to 3600 s default).
        configure_logger(self.args, source=SimpleProcessIdentity(component="inference_controller"))

        if not self.args.starts_inference_engines:
            return

        await self._engine_provider.init()
        router_addrs = await resolve_router_addrs(self.args, router_providers=self._router_providers)
        self.servers = await create_rollout_servers(
            self.args,
            context_lock=self.context_lock,
            engine_provider=self._engine_provider,
            router_addrs=router_addrs,
        )
        if self.args.eval_num_gpus > 0:
            self._eval_fleet = InferenceControllerEvalFleet(self.args, srv=self.servers["eval"])

        self._watcher_disposers.append(await self._engine_provider.watch_cells(self._reconcile))
        self._ticker = SimpleTicker(self._tick_cells, interval_seconds=TICK_INTERVAL_SECONDS)

        dashboard_hooks.register_router(self.args)

        await self.wait_expected_num_cells()

    # TEMPORARY: exists only so a suspend can take this lock, reverted with the weight-update fault tolerance work
    @with_lock
    async def stop_cell_between_weight_updates(self, cell_id: str) -> None:
        # [FUNCTION DOC] TEMPORARY hook: holding the controller lock, ask the engine provider to stop
        # the single cell with the given id. Exists so a suspend serializes against weight updates.
        await self._engine_provider.stop_cells(cell_ids=[cell_id])

    # TEMPORARY: exists only so fault injection can take this lock, reverted with the weight-update fault tolerance work
    @with_lock
    async def inject_fault_between_weight_updates(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        # [FUNCTION DOC] TEMPORARY hook: holding the controller lock, inject a fault into one worker
        # of a cell. Finds the server containing the cell id (raises key error if none); raises
        # runtime error if that server's health checker is inactive (cell offloaded, since colocated
        # trainers own the GPUs). Otherwise calls the provider's worker manager actor to inject a
        # fault of the given mode (its string value) into the worker at the given index within the
        # cell, awaiting the remote call. Reaches into a private attribute of the provider.
        # TEMPORARY: colocate cannot kill rollout workers while trainer ranks own the shared GPUs
        server = next((srv for srv in self.servers.values() if cell_id in srv.server_cells), None)
        if server is None:
            raise KeyError(f"Unknown rollout cell {cell_id!r}")
        if not server.health_checker_activeness.get().active:
            raise RuntimeError(f"Rollout cell {cell_id!r} is offloaded; refusing fault injection")

        await self._engine_provider._worker_manager_handle.inject_fault.remote(
            cell_id,
            mode=mode.value,
            worker_in_cell_index=sub_index,
        )

    # -------------------------- take over -----------------------------

    @lock_exempt
    async def is_initialized(self) -> bool:
        # [FUNCTION DOC] Whether init has completed; used by a new orchestrator to detect a
        # controller left from a previous run (take over).
        return self._init_once.is_initialized()

    @lock_exempt
    async def wait_expected_num_cells(self, timeout: float = CELLS_READY_TIMEOUT_SECONDS) -> None:
        # [FUNCTION DOC] Concurrently wait for every server to reach its expected cell count, each
        # with the given timeout in seconds. Exceptions from a server propagate.
        await asyncio.gather(*[srv.wait_init_expected_num_cells(timeout=timeout) for srv in self.servers.values()])

    @with_lock
    async def abort_all(self) -> None:
        # [FUNCTION DOC] Abort all in-flight generation requests on every server concurrently; all
        # are awaited and the first error is raised afterwards.
        await async_utils.gather_and_raise_first([srv.abort_all() for srv in self.servers.values()])

    # -------------------------- registration -----------------------------

    @lock_exempt
    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        # [FUNCTION DOC] Accept a snapshot of cells announced by an external reporter (not holding the
        # lock). Raises assertion error if: no servers exist yet (controller not ready; reporter will
        # re-announce); the engine provider is not a registration hub (this run deploys its own
        # engines); or any cell's model id is missing from or not among the served model ids. Then
        # hands the snapshot to the hub's ingest.
        assert self.servers, (
            f"this inference controller is not ready: reporter {snapshot.reporter_id} announced its cells before "
            f"the orchestration script built this controller's servers, so it does not know yet which models this "
            f"run serves; the reporter announces them again, and they are taken in once the controller is up"
        )
        assert isinstance(provider := self._engine_provider, RegistrationHub), (
            f"run {self.args.run_uuid} deploys its own engines and takes no registration "
            f"(reporter {snapshot.reporter_id})"
        )
        for cell in snapshot.cells:
            assert (
                model_id := cell.info.meta.get("model_id")
            ) in self.servers, (
                f"cell {cell.info.cell_id} serves model {model_id!r}, this run serves {sorted(self.servers)}"
            )
        await provider.ingest(snapshot)

    # -------------------------- rollout lifecycle hooks -----------------------------

    @with_lock
    async def prepare_rollout(self, rollout_id: int, model_id: str | None = None) -> None:
        # [FUNCTION DOC] Before a rollout: resume health monitoring for the model's servers (all if no
        # model id) and register the engines with the dashboard. The rollout id is unused.
        await self._health_monitoring_resume(model_id)
        await dashboard_hooks.register_engines(self.servers, provider=self._engine_provider)

    @with_lock
    async def prepare_eval(self, model_id: str | None = None) -> None:
        # [FUNCTION DOC] Before an evaluation: resume health monitoring for the model's servers.
        await self._health_monitoring_resume(model_id)

    @with_lock
    async def dispose(self) -> None:
        # [FUNCTION DOC] Shut down: clear and dispose the ticker if present, call and then clear all
        # watcher disposers, and dispose every server in order. Not idempotent for servers.
        if (ticker := self._ticker) is not None:
            self._ticker = None
            await ticker.dispose()

        for disposer in self._watcher_disposers:
            await disposer()
        self._watcher_disposers = []

        for srv in self.servers.values():
            await srv.dispose()

    # -------------------------- offload/onload -----------------------------

    # TODO may parallelly execute offload/onload across services
    @with_lock
    async def offload(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC] Offload GPU memory of all servers for the given memory tags (all kinds if
        # none), pausing health monitoring first (see offload helper).
        await self._offload(tags=tags)

    @with_lock
    async def onload(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC] Onload GPU memory of all servers for the given tags (all kinds if none).
        await self._onload(tags=tags)

    @with_lock
    async def onload_weights(self) -> None:
        # [FUNCTION DOC] Onload only the weights memory kind, and only if the configured rollout
        # offload level includes weights; otherwise a no-op.
        if "weight" not in self.args.offload_rollout_level:
            return
        await self._onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    @with_lock
    async def onload_kv(self) -> None:
        # [FUNCTION DOC] Onload the KV cache and CUDA graph memory kinds unconditionally (not gated by
        # the configured offload level).
        await self._onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    @with_lock
    async def offload_kv(self) -> None:
        # [FUNCTION DOC] Offload the CUDA graph memory kind always, plus KV cache memory if the
        # configured rollout offload level includes the kv cache.
        tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
        if "kv_cache" in self.args.offload_rollout_level:
            tags.append(GPU_MEMORY_TYPE_KV_CACHE)
        await self._offload(tags=tags)

    @with_lock
    async def offload_weights(self) -> None:
        # [FUNCTION DOC] Offload only the weights memory kind, and only if the configured rollout
        # offload level includes weights; otherwise a no-op.
        if "weight" not in self.args.offload_rollout_level:
            return
        await self._offload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    @requires_lock
    async def _offload(self, tags: list[str] | None):
        # [FUNCTION DOC] HELPER: Pause health monitoring on all servers (so offloaded engines are not
        # flagged unhealthy), then offload each server sequentially for the tags.
        await self._health_monitoring_pause(None)
        for srv in self.servers.values():
            await srv.offload(tags=tags)

    @requires_lock
    async def _onload(self, tags: list[str] | None):
        # [FUNCTION DOC] HELPER: Onload each server sequentially for the tags. Does not resume health
        # monitoring (that happens in the prepare hooks).
        for srv in self.servers.values():
            await srv.onload(tags)

    # -------------------------- engine management -----------------------------

    @acquires_lock
    async def start_update_weights(self, model_id: str | None = None) -> "UpdatableEngines":
        # [FUNCTION DOC] Begin a weight update; acquires the controller lock and keeps holding it until
        # end or abort of the update is called. Steps: pause health monitoring for the model's
        # servers; wait until its cells are all pending-weights or serving (see ensure helper);
        # select the single updatable server. Returns an updatable-engines record: if no server is
        # updatable, all lists/mappings are empty; else the server's API clients, per-engine GPU
        # counts and offsets, and a mapping from each cell id to its current workers hash (a snapshot
        # later used to decide which cells to mark ready). Raises value error if several servers are
        # updatable and no model id is given, assertion error for an unknown or frozen model id,
        # timeout error if cells are not ready in time.
        """Return engines eligible for weight updates."""
        await self._health_monitoring_pause(model_id)
        await self._ensure_cells_ready(model_id=model_id)

        srv = self._get_updatable_server(model_id=model_id)
        if not srv:
            return UpdatableEngines(
                rollout_engines=[],
                engine_gpu_counts=[],
                engine_gpu_offsets=[],
                snapshot_cell_id_to_hashes={},
            )

        return UpdatableEngines(
            rollout_engines=srv.api_clients,
            engine_gpu_counts=srv.engine_gpu_counts,
            engine_gpu_offsets=srv.engine_gpu_offsets,
            snapshot_cell_id_to_hashes={cell_id: cell.meta.workers_hash for cell_id, cell in srv.server_cells.items()},
        )

    @releases_lock
    async def abort_update_weights(self) -> None:
        # [FUNCTION DOC] Abandon a weight update: only releases the lock acquired by the start call.
        # No cell is marked weights-ready.
        pass

    @releases_lock
    async def end_update_weights(self, snapshot_cell_id_to_hashes: dict[str, str]) -> None:
        # [FUNCTION DOC] Finish a weight update and release the lock held since the start call. For
        # every cell (across all servers) that is in the snapshot mapping, whose workers hash still
        # equals the snapshot's hash (i.e. the cell was not replaced meanwhile) and that is pending
        # weights, concurrently mark its weights ready so it can serve. Cells that appeared or were
        # replaced during the update stay pending.
        await asyncio.gather(
            *[
                cell.mark_weights_ready()
                for srv in self.servers.values()
                for cell_id, cell in srv.server_cells.items()
                if cell_id in snapshot_cell_id_to_hashes
                and snapshot_cell_id_to_hashes[cell_id] == cell.meta.workers_hash
                and cell.is_pending_weights
            ]
        )

    @requires_lock
    async def _ensure_cells_ready(self, model_id: str | None = None) -> None:
        # [FUNCTION DOC] HELPER: Loop until all cells of the model's servers are pending-weights or
        # serving, with an overall deadline of 3600 s (raises timeout error with counts). Each
        # iteration re-reads the cells; in colocate mode first concurrently initializes uninitialized
        # cells; if some remain not ready, logs and sleeps 2 s with the lock temporarily released
        # (so other operations such as reconcile can progress) before retrying.
        deadline = time.monotonic() + CELLS_READY_TIMEOUT_SECONDS
        while True:
            cells = [cell for srv in self._get_servers_of_model_id(model_id) for cell in srv.server_cells.values()]
            if self.args.colocate:
                await asyncio.gather(*[cell.init() for cell in cells if cell.is_uninitialized])
            pending = [cell for cell in cells if not cell.is_pending_weights_or_serving]
            if not pending:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {CELLS_READY_TIMEOUT_SECONDS}s waiting for "
                    f"{len(pending)}/{len(cells)} cells to become ready"
                )
            logger.info(f"Waiting for {len(pending)}/{len(cells)} cells to become ready...")
            async with self.context_lock.with_released():
                await asyncio.sleep(CELLS_READY_POLL_INTERVAL_SECONDS)

    @requires_lock
    def _get_servers_of_model_id(self, model_id: str | None) -> list[RolloutServer]:
        # [FUNCTION DOC] HELPER: All servers when no model id; otherwise a one-element list with the
        # server of that id, asserting it exists.
        if model_id is None:
            return list(self.servers.values())
        srv = self.servers.get(model_id)
        assert srv is not None, f"No server for model_id {model_id!r}, known ids: {sorted(self.servers)}"
        return [srv]

    @requires_lock
    def _get_updatable_server(self, model_id: str | None = None) -> RolloutServer | None:
        # [FUNCTION DOC] HELPER: With a model id: that model's server, asserting it is not frozen (its
        # update-weights flag set). Without: the only server whose update-weights flag is set;
        # nothing if there is none; raises value error naming them if several.
        if model_id is not None:
            [srv] = self._get_servers_of_model_id(model_id)
            assert srv.update_weights, f"Server for model_id {model_id!r} is frozen (update_weights=False)"
            return srv

        updatable = [srv for srv in self.servers.values() if srv.update_weights]
        match updatable:
            case []:
                return None
            case [srv]:
                return srv
            case _:
                raise ValueError(
                    f"Multiple servers have update_weights=True: {[srv.model_name for srv in updatable]}. "
                    f"Pass model_id to update exactly one of them."
                )

    # -------------------------- eval fleet -----------------------------

    @lock_exempt
    async def get_eval_fleet_info(self) -> EvalFleetInfo | None:
        # [FUNCTION DOC] The eval fleet description (router address and GPU counts), or nothing if this
        # run has no dedicated eval fleet.
        return self._eval_fleet.info if self._eval_fleet is not None else None

    @lock_exempt
    async def pin_eval_fleet(self, checkpoint_dir: str, weight_version: str) -> EvalFleetPin:
        # [FUNCTION DOC] Delegate to the eval fleet to load the checkpoint onto all eval engines at
        # the weight version; returns its pin result. Raises assertion error if no eval fleet exists.
        # Runs without the controller lock (the fleet works with the eval server's own lock).
        assert (
            self._eval_fleet is not None
        ), "this run deploys no eval fleet, so nothing asked this controller to pin one"
        return await self._eval_fleet.pin(checkpoint_dir=checkpoint_dir, weight_version=weight_version)

    # -------------------------- misc APIs -----------------------------

    @lock_exempt
    async def get_cell_statuses(self) -> dict[str, CellStatus]:
        # [FUNCTION DOC] Mapping from cell id to its status record across all servers. Iterates over
        # copies of the server and cell containers so concurrent reconcile changes are safe.
        return {
            cell_id: cell.cell_status()
            for srv in list(self.servers.values())
            for cell_id, cell in list(srv.server_cells.items())
        }

    @with_lock
    async def check_weights(
        self,
        action: str,
        allow_quant_error: bool = False,
        selector: str = "all",
        skip_list: list[str] | None = None,
        model_id: str | None = None,
    ) -> list[Any]:
        # [FUNCTION DOC] Run a weight consistency check on the updatable server's engines. The action
        # (e.g. checksum), quantization-error tolerance flag, selector and skip list are forwarded
        # unchanged. Returns the server's list of results, or an empty list if no updatable server
        # exists. Errors from server selection (multiple servers, frozen model) propagate.
        # Only the updatable model is re-synced; a frozen model would always mismatch.
        srv = self._get_updatable_server(model_id=model_id)
        if srv is None:
            return []
        return await srv.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )

    # -------------------------- tick -----------------------------

    @with_lock
    async def _tick_cells(self) -> None:
        # [FUNCTION DOC] HELPER: Periodic job (runs under the lock): tick every cell of every server
        # concurrently, each limited to 120 s; failures and timeouts are logged as errors with their
        # exception and never raised.
        cells = [cell for srv in list(self.servers.values()) for cell in list(srv.server_cells.values())]
        results = await asyncio.gather(
            *[asyncio.wait_for(cell.tick(), timeout=CELL_TICK_TIMEOUT_SECONDS) for cell in cells],
            return_exceptions=True,
        )
        for cell, result in zip(cells, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(f"Ticking cell {cell.meta.cell_id} failed", exc_info=result)

    # -------------------------- reconcile -----------------------------

    @with_lock
    async def _reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: Callback from the engine provider's cell watch: bring the actual cell
        # (found by searching all servers for the id; none if absent) in line with the observed info
        # (none means the cell vanished). Delegates the decision to the shared observation helper,
        # given the actual workers hash. Add: converts the observed info to cell metadata and adds the
        # cell to the server of the metadata's model id (key error if unknown). Remove: removes the
        # cell from the server that currently holds it.
        actual_srv: RolloutServer | None = None
        actual_cell: ServerCell | None = None
        for srv in self.servers.values():
            if (c := srv.server_cells.get(cell_id)) is not None:
                actual_srv, actual_cell = srv, c
                break

        async def _add(_cell_id: str, observed_info: CellInfo) -> None:
            # [FUNCTION DOC] HELPER: Add the cell described by the observed info to its model's server.
            observed_cell_meta = _compute_server_cell_meta_from_info(observed_info)
            await self.servers[observed_cell_meta.model_id].add_cell(observed_cell_meta)

        async def _remove(remove_cell_id: str) -> None:
            # [FUNCTION DOC] HELPER: Remove the cell from the server found in the enclosing lookup.
            await actual_srv.remove_cell(remove_cell_id)

        await apply_cell_observation(
            cell_id=cell_id,
            observed=observed,
            actual_workers_hash=actual_cell.meta.workers_hash if actual_cell is not None else None,
            add=_add,
            remove=_remove,
        )

    # -------------------------- utils -----------------------------

    @requires_lock
    async def _health_monitoring_pause(self, model_id: str | None) -> None:
        # [FUNCTION DOC] HELPER: For each selected server, bump the health checker activeness to
        # inactive.
        for srv in self._get_servers_of_model_id(model_id):
            srv.health_checker_activeness.bump_active(False)

    @requires_lock
    async def _health_monitoring_resume(self, model_id: str | None) -> None:
        # [FUNCTION DOC] HELPER: For each selected server, bump the health checker activeness to
        # active.
        for srv in self._get_servers_of_model_id(model_id):
            srv.health_checker_activeness.bump_active(True)


@dataclass(frozen=True)
class UpdatableEngines:
    # [CLASS DOC] Frozen result of starting a weight update: the API clients of the engines to update,
    # per-engine GPU counts and GPU offsets (parallel lists, same order), and a mapping from cell id
    # to the workers hash at snapshot time.
    rollout_engines: list[SGLangApiClient]
    engine_gpu_counts: list[int]
    engine_gpu_offsets: list[int]
    snapshot_cell_id_to_hashes: dict[str, str]


# TODO may move and generalize later
def _compute_server_cell_meta_from_info(info: CellInfo) -> ServerCellMetadata:
    # [FUNCTION DOC] HELPER: Convert a provider cell info to server cell metadata: copies model id,
    # worker type, GPUs per engine, GPU offset, API key, needs-offload and update-weights flags from
    # the info's metadata mapping (key error if missing), the cell id, the workers hash, and uses the
    # first worker name as the cell's worker name (extra workers ignored).
    return ServerCellMetadata(
        model_id=info.meta["model_id"],
        worker_type=info.meta["worker_type"],
        cell_id=info.cell_id,
        num_gpus_per_engine=info.meta["num_gpus_per_engine"],
        gpu_offset=info.meta["gpu_offset"],
        sglang_api_key=info.meta["sglang_api_key"],
        worker_name=info.worker_names[0],
        needs_offload=info.meta["needs_offload"],
        update_weights=info.meta["update_weights"],
        workers_hash=info.workers_hash,
    )
