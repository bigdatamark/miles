# [MODULE DOC]
# Unit tests for the inference controller: the driver-side actor that owns the rollout servers and
# their cells, the weight-update lock window, per-model health-check activeness, memory offload and
# onload fan-out, fault injection / cell suspension, the eval fleet and the registration endpoint.
# Everything runs against in-memory fakes: a recording server stands in for a rollout server, a fake
# worker provider stands in for the engine and router providers, and most controllers are built
# without running the constructor and given only the attributes the method under test reads.
# No Ray, GPU or network is involved; asynchronous tests rely on the pytest asyncio plugin.
# Test classes cover: abort fan-out, health-check activeness (single and per model), fault injection
# guards, reconcile of observed cells (add / remove / replace / no-op), reconcile after a failed cell
# init, init subscription order, the engine meta contract with the real spec, the weight-update lock
# window, lock sharing with servers, updatable model selection and per-model scoping, memory
# lifecycle fan-out, the updatable-engines payload, the init lifecycle, cell operations serialized
# against weight updates, init running exactly once, waiting for the whole fleet, the eval fleet
# rpc surface, cell readiness scoping, and the registration snapshot endpoint.
import asyncio
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
from tests.fast.ray.rollout.conftest import make_args

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout import inference_controller as inference_controller_module
from miles.ray.rollout.eval_fleet import EvalFleetInfo, EvalFleetPin
from miles.ray.rollout.inference_controller import (
    InferenceController,
    UpdatableEngines,
    _compute_server_cell_meta_from_info,
)
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.ray.specs.inference import compute_engine_pool_ids, compute_router_pool_id, specs_inference_engine
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.registration.hub import RegistrationHub
from miles.utils.workers.registration.models import RegisteredCellInfo, RegistrationSnapshot
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts, WorkerMetaContext

_RUN_UUID = "run-uuid-1"


def _make_cell_info(
    *,
    cell_id: str = "inference-engine-0-0-0",
    workers_hash: str = "pseudo-hash-0",
    alive: bool = True,
    model_id: str = "model-a",
    pool_id: str = "inference-engine-0-0",
) -> CellInfo:
    # [FUNCTION DOC] HELPER: Builds a cell info record for an engine cell. All arguments are optional
    # keyword-only values (cell id, workers hash, alive flag, model id, pool id) with fixed defaults.
    # The record has one worker named after the cell with a zero suffix and a meta mapping with
    # worker type regular, one GPU per engine, GPU offset zero, no api key, no offload and weight
    # updates enabled. Returns the cell info; no side effects.
    return CellInfo(
        cell_id=cell_id,
        pool_id=pool_id,
        alive=alive,
        worker_names=[f"{cell_id}-0"],
        workers_hash=workers_hash,
        meta=dict(
            model_id=model_id,
            worker_type="regular",
            num_gpus_per_engine=1,
            gpu_offset=0,
            sglang_api_key=None,
            needs_offload=False,
            update_weights=True,
        ),
    )


def _make_cell_meta(info: CellInfo) -> ServerCellMetadata:
    # [FUNCTION DOC] HELPER: Converts a cell info record into the server cell metadata a rollout
    # server would hold for it, copying each field from the meta mapping and taking the cell id,
    # the first worker name and the workers hash from the record. Returns the metadata; pure.
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


class _RecordingServer:
    # [CLASS DOC] HELPER: Fake rollout server that records the calls the controller makes on it.
    # Exposes the attributes the controller reads (cell mapping, health-check activeness tracker,
    # update-weights flag, model name, router address, api clients, engine GPU counts and offsets)
    # and records offload and onload tags, weight-check keyword arguments, add and remove cell
    # calls, the timeouts passed to the wait for expected cells, and the dispose count. Waiting for
    # the expected number of cells optionally blocks on a caller supplied event (a startup gate).
    def __init__(
        self,
        server_cells: dict | None = None,
        *,
        model_name: str = "model",
        update_weights: bool = False,
        cells_gate: asyncio.Event | None = None,
    ):
        # [FUNCTION DOC] HELPER: Stores the optional cell mapping (empty when none or empty is
        # given), the model name, the update-weights flag and the optional startup gate event.
        # Creates an active health-check activeness tracker and empty recording lists; the router
        # address defaults to 10.0.0.9 port 31000.
        self.server_cells = server_cells or {}
        self.health_checker_activeness = ActivenessTracker(active=True)
        self.update_weights = update_weights
        self.model_name = model_name
        self.calls: list[tuple] = []
        self.router_ip: str = "10.0.0.9"
        self.router_port: int = 31000
        self.api_clients: list = []
        self.engine_gpu_counts: list[int] = []
        self.engine_gpu_offsets: list[int] = []
        self.offload_tags: list = []
        self.onload_tags: list = []
        self.check_weights_kwargs: list[dict] = []
        self.waited_init_expected_num_cells = 0
        self.cells_timeouts: list[float] = []
        self.dispose_count = 0
        self._cells_gate = cells_gate

    async def offload(self, tags=None):
        # [FUNCTION DOC] HELPER: Records an offload call in the ordered call list and keeps the
        # given tags in the offload tags list. Returns nothing.
        self.calls.append(("offload",))
        self.offload_tags.append(tags)

    async def onload(self, tags=None):
        # [FUNCTION DOC] HELPER: Keeps the given tags in the onload tags list (not in the ordered
        # call list). Returns nothing.
        self.onload_tags.append(tags)

    async def dispose(self):
        # [FUNCTION DOC] HELPER: Increments the dispose counter. Returns nothing.
        self.dispose_count += 1

    async def check_weights(self, action, allow_quant_error=False, selector="all", skip_list=None):
        # [FUNCTION DOC] HELPER: Records a check-weights call (action in the ordered call list, all
        # four arguments in the keyword record list). Returns a list holding only this server's
        # model name, so callers can tell which servers were reached.
        self.calls.append(("check_weights", action))
        self.check_weights_kwargs.append(
            dict(action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list)
        )
        return [self.model_name]

    async def add_cell(self, cell_meta: ServerCellMetadata):
        # [FUNCTION DOC] HELPER: Records an add call with the cell id and stores a lightweight
        # object carrying the metadata in the cell mapping under that id.
        self.calls.append(("add", cell_meta.cell_id))
        self.server_cells[cell_meta.cell_id] = SimpleNamespace(meta=cell_meta)

    async def remove_cell(self, cell_id: str):
        # [FUNCTION DOC] HELPER: Records a remove call with the cell id and deletes the entry from
        # the cell mapping. Raises a key error when the cell is not tracked.
        self.calls.append(("remove", cell_id))
        del self.server_cells[cell_id]

    async def wait_init_expected_num_cells(self, timeout: float = 3600.0) -> None:
        # [FUNCTION DOC] HELPER: Blocks on the startup gate event when one was supplied, then counts
        # the call and records the timeout it was given (default 3600 seconds). Returns nothing.
        if self._cells_gate is not None:
            await self._cells_gate.wait()
        self.waited_init_expected_num_cells += 1
        self.cells_timeouts.append(timeout)


class _AbortRecordingServer:
    # [CLASS DOC] HELPER: Fake rollout server whose only behavior is aborting all requests. It
    # appends its name to a shared started list on entry, yields to the event loop a configurable
    # number of times, then either raises a configured error or appends its name to a shared
    # completed list.
    def __init__(
        self,
        name: str,
        *,
        started: list[str],
        completed: list[str],
        error: Exception | None = None,
        scheduling_turns: int = 1,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the server name, the two shared lists, the optional error
        # to raise and the number of event-loop yields (default one) to perform while aborting.
        self.name = name
        self.started = started
        self.completed = completed
        self.error = error
        self.scheduling_turns = scheduling_turns

    async def abort_all(self) -> None:
        # [FUNCTION DOC] HELPER: Records the start, yields to the event loop the configured number
        # of times, then raises the configured error if any; otherwise records completion.
        self.started.append(self.name)
        for _ in range(self.scheduling_turns):
            await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        self.completed.append(self.name)


class _RecordingRemoteMethod:
    # [CLASS DOC] HELPER: Stand-in for a remote actor method that exposes an asynchronous remote
    # entry point and records the positional and keyword arguments of every call.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the empty list of recorded calls (pairs of positional
        # arguments and keyword arguments).
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def remote(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Appends the received positional and keyword arguments to the
        # recorded calls. Returns nothing.
        self.calls.append((args, kwargs))


class _FakeUpdatableCell:
    # [CLASS DOC] HELPER: Fake server cell for weight-update tests. Carries a meta object with a
    # workers hash, always reports pending-weights and pending-or-serving as true unless a test
    # overrides the flag, and counts how many times it was marked weights-ready.
    def __init__(self, workers_hash: str):
        # [FUNCTION DOC] HELPER: Stores the workers hash in a meta object and initializes the
        # marked-ready counter to zero and both pending flags to true.
        self.meta = SimpleNamespace(workers_hash=workers_hash)
        self.marked_ready = 0
        self.is_pending_weights = True
        self.is_pending_weights_or_serving = True

    async def mark_weights_ready(self) -> None:
        # [FUNCTION DOC] HELPER: Increments the marked-ready counter. Returns nothing.
        self.marked_ready += 1


class _TickingCell:
    # [CLASS DOC] HELPER: Fake server cell that only counts how many times it was ticked.
    def __init__(self, cell_id: str = "engine-0"):
        # [FUNCTION DOC] HELPER: Stores a meta object carrying the cell id (default engine-0) and
        # sets the tick counter to zero.
        self.meta = SimpleNamespace(cell_id=cell_id)
        self.tick_count = 0

    async def tick(self) -> None:
        # [FUNCTION DOC] HELPER: Increments the tick counter. Returns nothing.
        self.tick_count += 1


class _RecordingEvalFleet:
    # [CLASS DOC] HELPER: Fake eval fleet substituted for the real fleet class in the controller
    # module. It remembers the arguments namespace and the server it was constructed with so tests
    # can check what the controller handed over; disposing it does nothing.
    def __init__(self, args: Namespace, *, srv):
        # [FUNCTION DOC] HELPER: Stores the arguments namespace and the server (keyword-only).
        self.args = args
        self.srv = srv

    async def dispose(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None


class _FakeWorkerProvider(BaseWorkerProvider):
    # [CLASS DOC] HELPER: In-memory worker provider. Holds a fixed list of cell infos and a list of
    # pool ids it serves; when watched it first requires init to have run, remembers the pool ids
    # it was asked to watch and reports every cell whose pool is one of its pools through the
    # reconcile callback, then returns a stop function that counts its calls. Asking it for worker
    # addresses is a test failure; it has a recording fake of the worker manager fault injection.
    def __init__(self, cell_infos: list[CellInfo], *, pool_ids: list[str] | None = None) -> None:
        # [FUNCTION DOC] HELPER: Stores the cell infos and the pool ids (empty list when none are
        # given), creates a worker manager handle whose fault injection is a recording remote
        # method, and initializes the watched pool ids to unset, the initialized flag to false and
        # the stop-watch counter to zero.
        self._cell_infos = cell_infos
        self._pools = pool_ids or []
        self._worker_manager_handle = SimpleNamespace(inject_fault=_RecordingRemoteMethod())
        self.watched_pool_ids: list[str] | None = None
        self.initialized = False
        self.stop_watch_calls = 0

    async def init(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the provider as initialized.
        self.initialized = True

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] HELPER: Always raises an assertion error naming the worker, because the
        # controller under test must never ask this fake for worker addresses.
        raise AssertionError(f"the controller must not ask this fake for {worker_name}")

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        # [FUNCTION DOC] HELPER: Returns one empty worker list per requested cell id.
        return [[] for _ in cell_ids]

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        # [FUNCTION DOC] HELPER: Asserts that init ran first (assertion error otherwise), records a
        # copy of its pool ids as the watched pools, awaits the reconcile callback once for every
        # stored cell whose pool id is among its pools (passing cell id and cell info), and returns
        # an asynchronous stop function that increments the stop-watch counter.
        assert self.initialized, "the controller must init the provider before observing its cells"
        self.watched_pool_ids = list(self._pools)
        for info in self._cell_infos:
            if info.pool_id in self._pools:
                await reconcile(info.cell_id, info)

        async def _stop_watch() -> None:
            # [FUNCTION DOC] HELPER: Increments the provider's stop-watch counter.
            self.stop_watch_calls += 1

        return _stop_watch


class _RecordingInferenceControllerEvalFleet:
    # [CLASS DOC] HELPER: Fake eval fleet installed on a controller to test its eval-fleet rpc
    # methods. Holds a fleet info record and records every pin request.
    def __init__(self, info: EvalFleetInfo):
        # [FUNCTION DOC] HELPER: Stores the fleet info and an empty list of recorded pins.
        self.info = info
        self.pins: list[dict] = []

    async def pin(self, checkpoint_dir: str, weight_version: str) -> EvalFleetPin:
        # [FUNCTION DOC] HELPER: Records the checkpoint directory and weight version as a pin and
        # returns a pin result with no skip reason (meaning the pin succeeded).
        self.pins.append(dict(checkpoint_dir=checkpoint_dir, weight_version=weight_version))
        return EvalFleetPin(skip_reason=None)


def _make_controller(
    servers: dict,
    *,
    engine_provider: BaseWorkerProvider | None = None,
    registration_hub: RegistrationHub | None = None,
) -> InferenceController:
    # [FUNCTION DOC] HELPER: Builds an inference controller without running its constructor. The
    # engine source is the registration hub if given, else the engine provider if given, else an
    # empty fake provider. Sets minimal arguments (no debug-train-only, no fault tolerance, no ci
    # test, no colocate, offload levels kv cache and weight, the fixed run uuid), the given servers
    # mapping, a fresh context lock named after the controller, the engine provider, and a single
    # empty fake router provider. Returns the controller; no other attributes exist.
    engines = registration_hub if registration_hub is not None else engine_provider
    engines = engines if engines is not None else _FakeWorkerProvider([])

    controller = InferenceController.__new__(InferenceController)
    controller.args = SimpleNamespace(
        debug_train_only=False,
        use_fault_tolerance=False,
        ci_test=False,
        colocate=False,
        offload_rollout_level=["kv_cache", "weight"],
        run_uuid=_RUN_UUID,
    )
    controller.servers = servers
    controller.context_lock = ContextLock("InferenceController")
    controller._engine_provider = engines
    controller._router_providers = [_FakeWorkerProvider([])]
    return controller


class TestAbortAll:
    # [CLASS DOC] Tests the controller's abort-all fan-out across its rollout servers.
    async def test_abort_all_reaches_every_server_before_propagating_the_first_failure(self) -> None:
        # [FUNCTION DOC] Builds a controller with two abort-recording servers: the first raises a
        # runtime error right after starting, the second needs two scheduling turns and completes.
        # Asserts that abort-all re-raises the very same error object, that both servers started,
        # and that the second one still ran to completion (the failure is only propagated after
        # every server finished).
        """Every server is aborted to completion before the first fleet failure is propagated."""
        started: list[str] = []
        completed: list[str] = []
        first_failure = RuntimeError("first server refused abort")
        controller = _make_controller(
            {
                "first": _AbortRecordingServer(
                    "first",
                    started=started,
                    completed=completed,
                    error=first_failure,
                ),
                "second": _AbortRecordingServer(
                    "second",
                    started=started,
                    completed=completed,
                    scheduling_turns=2,
                ),
            }
        )

        with pytest.raises(RuntimeError) as exc_info:
            await controller.abort_all()

        assert exc_info.value is first_failure
        assert set(started) == {"first", "second"}
        assert completed == ["second"]


class TestHealthCheckerActiveness:
    # [CLASS DOC] Tests that the controller toggles the health-check activeness tracker of its
    # server around offload, weight updates, rollout preparation and eval preparation (single
    # server case).
    @pytest.mark.asyncio
    async def test_offload_pauses_probing_before_putting_engines_to_sleep(self):
        # [FUNCTION DOC] Offloads through the controller with one recording server and asserts the
        # tracker is no longer active and the server saw exactly one offload call.
        """A slept engine cannot answer /health_generate, so probing must stop first."""
        srv = _RecordingServer()
        controller = _make_controller({"default": srv})

        await controller.offload()

        assert not srv.health_checker_activeness.get().active
        assert srv.calls == [("offload",)]

    @pytest.mark.asyncio
    async def test_starting_a_weight_update_pauses_probing(self):
        # [FUNCTION DOC] Starts then ends a weight update with one recording server and asserts
        # the tracker is inactive afterwards (probing stays paused until a rollout is prepared).
        """Engines are unusable while their weights are being replaced."""
        srv = _RecordingServer()
        controller = _make_controller({"default": srv})

        info = await controller.start_update_weights()
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

        assert not srv.health_checker_activeness.get().active

    @pytest.mark.asyncio
    async def test_preparing_a_rollout_resumes_probing(self):
        # [FUNCTION DOC] Pauses the tracker manually, prepares rollout zero, and asserts the
        # tracker is active again.
        """Probing comes back exactly when the engines start serving traffic again."""
        srv = _RecordingServer()
        controller = _make_controller({"default": srv})
        srv.health_checker_activeness.bump_active(False)

        await controller.prepare_rollout(rollout_id=0)

        assert srv.health_checker_activeness.get().active

    @pytest.mark.asyncio
    async def test_preparing_a_rollout_awaits_the_dashboard_engine_registration(self, monkeypatch):
        # [FUNCTION DOC] Replaces the dashboard hook that registers engines with a recording
        # coroutine, prepares a rollout on a controller holding one server and a specific engine
        # provider, and asserts the hook was awaited exactly once with the servers mapping and the
        # engine provider (keyword argument).
        """The dashboard hook is a coroutine, so prepare_rollout must await it instead of leaving it unscheduled."""
        awaited: list[tuple[dict, _FakeWorkerProvider]] = []

        async def _record(servers: dict, *, provider: _FakeWorkerProvider) -> None:
            # [FUNCTION DOC] HELPER: Records the servers mapping and the provider it was called with.
            awaited.append((servers, provider))

        monkeypatch.setattr(dashboard_hooks, "register_engines", _record)
        servers = {"default": _RecordingServer()}
        engine_provider = _FakeWorkerProvider([])
        controller = _make_controller(servers, engine_provider=engine_provider)

        await controller.prepare_rollout(rollout_id=0)

        assert awaited == [(servers, engine_provider)]

    @pytest.mark.asyncio
    async def test_preparing_an_eval_resumes_probing(self):
        # [FUNCTION DOC] Pauses the tracker manually, prepares an eval, and asserts the tracker is
        # active again.
        """Eval drives the same engines as a rollout does."""
        srv = _RecordingServer()
        controller = _make_controller({"default": srv})
        srv.health_checker_activeness.bump_active(False)

        await controller.prepare_eval()

        assert srv.health_checker_activeness.get().active


class TestRolloutFaultInjectionWindow:
    # [CLASS DOC] Tests the guards of the controller's fault injection between weight updates:
    # the cell must be tracked and its server must still be probing (not offloaded).
    async def test_fault_injection_refuses_an_unknown_rollout_cell(self) -> None:
        # [FUNCTION DOC] Injects a kill-signal fault into a cell id no server tracks and asserts
        # a key error mentioning the id is raised and the worker manager received no injection.
        """An unknown rollout cell raises its identifier without reaching the worker manager."""
        cell_id = "unknown-rollout-cell"
        provider = _FakeWorkerProvider([])
        controller = _make_controller(
            {"default": _RecordingServer(server_cells={"inference-engine-0-0-0": object()})},
            engine_provider=provider,
        )

        with pytest.raises(KeyError) as exc_info:
            await controller.inject_fault_between_weight_updates(
                cell_id=cell_id,
                mode=FailureMode.SIGKILL,
                sub_index=0,
            )

        assert cell_id in str(exc_info.value)
        assert provider._worker_manager_handle.inject_fault.calls == []

    @pytest.mark.asyncio
    async def test_fault_injection_reaches_a_serving_rollout_cell(self) -> None:
        # [FUNCTION DOC] Injects a kill-signal fault into a tracked cell of an active server and
        # asserts the worker manager's inject-fault method received the cell id positionally and
        # the mode as the lower-case string sigkill plus worker-in-cell index zero as keywords.
        """A serving rollout cell accepts a fault through the Ray worker manager."""
        cell_id = "inference-engine-0-0-0"
        server = _RecordingServer(server_cells={cell_id: object()})
        provider = _FakeWorkerProvider([])
        controller = _make_controller({"default": server}, engine_provider=provider)

        await controller.inject_fault_between_weight_updates(
            cell_id=cell_id,
            mode=FailureMode.SIGKILL,
            sub_index=0,
        )

        assert provider._worker_manager_handle.inject_fault.calls == [
            ((cell_id,), {"mode": "sigkill", "worker_in_cell_index": 0})
        ]

    @pytest.mark.asyncio
    async def test_fault_injection_refuses_an_offloaded_rollout_cell(self) -> None:
        # [FUNCTION DOC] Marks the server's tracker inactive (offloaded), attempts injection and
        # asserts a runtime error saying the cell is offloaded and fault injection is refused,
        # with no call reaching the worker manager.
        """Colocate must not kill rollout processes while trainer ranks own the shared GPUs."""
        cell_id = "inference-engine-0-0-0"
        server = _RecordingServer(server_cells={cell_id: object()})
        server.health_checker_activeness.bump_active(False)
        provider = _FakeWorkerProvider([])
        controller = _make_controller({"default": server}, engine_provider=provider)

        with pytest.raises(RuntimeError, match="is offloaded; refusing fault injection"):
            await controller.inject_fault_between_weight_updates(
                cell_id=cell_id,
                mode=FailureMode.SIGKILL,
                sub_index=0,
            )

        assert provider._worker_manager_handle.inject_fault.calls == []


class TestReconcile:
    # [CLASS DOC] Tests the controller's reconcile callback, which receives a cell id and the
    # observed cell info (or nothing when the cell vanished) and adds, removes or replaces the
    # cell in the owning server, chosen by the model id in the cell meta. Uses two recording
    # servers, model-a and model-b.
    @pytest.fixture
    def servers(self) -> dict[str, _RecordingServer]:
        # [FUNCTION DOC] HELPER: Fixture returning a mapping with a fresh recording server for
        # model-a and another for model-b.
        return {"model-a": _RecordingServer(), "model-b": _RecordingServer()}

    @pytest.mark.asyncio
    async def test_an_observed_untracked_cell_is_added_to_its_model_server(self, servers):
        # [FUNCTION DOC] Reconciles a new default cell (model-a) and asserts model-a's server
        # recorded exactly one add for that cell id.
        """A newly observed engine cell lands in the server named by its model_id meta."""
        controller = _make_controller(servers)
        info = _make_cell_info()

        await controller._reconcile(info.cell_id, info)

        assert servers["model-a"].calls == [("add", info.cell_id)]

    @pytest.mark.asyncio
    async def test_a_second_models_cell_is_routed_to_that_models_server(self, servers):
        # [FUNCTION DOC] Reconciles a new cell whose model id is model-b and asserts only model-b's
        # server recorded an add while model-a's recorded nothing.
        """Routing is by model_id, so model-b's cell must not be absorbed by the first server."""
        controller = _make_controller(servers)
        info = _make_cell_info(cell_id="inference-engine-1-0-0", model_id="model-b", pool_id="inference-engine-1-0")

        await controller._reconcile(info.cell_id, info)

        assert servers["model-a"].calls == []
        assert servers["model-b"].calls == [("add", info.cell_id)]

    @pytest.mark.asyncio
    async def test_a_disappeared_tracked_cell_is_removed(self, servers):
        # [FUNCTION DOC] Pre-registers a cell in model-a's server, reconciles it with no info, and
        # asserts a single remove was recorded and the cell mapping is empty.
        """A tracked cell reported as gone is removed even though no meta is observable."""
        info = _make_cell_info()
        servers["model-a"].server_cells[info.cell_id] = SimpleNamespace(meta=_make_cell_meta(info))
        controller = _make_controller(servers)

        await controller._reconcile(info.cell_id, None)

        assert servers["model-a"].calls == [("remove", info.cell_id)]
        assert servers["model-a"].server_cells == {}

    @pytest.mark.asyncio
    async def test_a_disappeared_cell_is_removed_from_its_owning_server(self, servers):
        # [FUNCTION DOC] Pre-registers a model-b cell only in model-b's server, reconciles it as
        # gone, and asserts the remove went to model-b (not to the first server) and its mapping
        # is empty.
        """The owner scan must find the server that actually tracks the cell, not the first one."""
        info = _make_cell_info(cell_id="inference-engine-1-0-0", model_id="model-b", pool_id="inference-engine-1-0")
        servers["model-b"].server_cells[info.cell_id] = SimpleNamespace(meta=_make_cell_meta(info))
        controller = _make_controller(servers)

        await controller._reconcile(info.cell_id, None)

        assert servers["model-a"].calls == []
        assert servers["model-b"].calls == [("remove", info.cell_id)]
        assert servers["model-b"].server_cells == {}

    @pytest.mark.asyncio
    async def test_a_workers_hash_change_replaces_the_cell(self, servers):
        # [FUNCTION DOC] Tracks a cell with one workers hash, reconciles an observation of the same
        # cell with a different hash, and asserts model-a's server recorded a remove followed by
        # an add (in that order) and model-b's recorded nothing.
        """A relaunched cell (new workers_hash) is removed then re-added, in that order."""
        old_info = _make_cell_info(workers_hash="pseudo-hash-0")
        servers["model-a"].server_cells[old_info.cell_id] = SimpleNamespace(meta=_make_cell_meta(old_info))
        controller = _make_controller(servers)
        new_info = _make_cell_info(workers_hash="pseudo-hash-1")

        await controller._reconcile(new_info.cell_id, new_info)

        assert servers["model-a"].calls == [("remove", new_info.cell_id), ("add", new_info.cell_id)]
        assert servers["model-b"].calls == []

    @pytest.mark.asyncio
    async def test_an_unchanged_tracked_cell_is_a_noop(self, servers):
        # [FUNCTION DOC] Tracks a cell and reconciles an observation with the identical workers
        # hash; asserts no server call was recorded.
        """A tracked cell observed with the same workers_hash triggers no bookkeeping change."""
        info = _make_cell_info()
        servers["model-a"].server_cells[info.cell_id] = SimpleNamespace(meta=_make_cell_meta(info))
        controller = _make_controller(servers)

        await controller._reconcile(info.cell_id, info)

        assert servers["model-a"].calls == []

    @pytest.mark.asyncio
    async def test_a_disappeared_untracked_cell_is_a_noop(self, servers):
        # [FUNCTION DOC] Reconciles a router cell id as gone although no server tracks it, and
        # asserts neither server recorded any call.
        """A vanished cell that was never tracked (e.g. a router) triggers nothing."""
        controller = _make_controller(servers)

        await controller._reconcile("miles-router-0-0", None)

        assert servers["model-a"].calls == []
        assert servers["model-b"].calls == []


def _patch_init(monkeypatch: pytest.MonkeyPatch, *, servers: dict[str, _RecordingServer]) -> None:
    # [FUNCTION DOC] HELPER: Patches the controller module so init does not build real servers:
    # the server factory is replaced by a coroutine returning the given servers mapping, and the
    # router address resolver by one returning host 10.0.0.1 port 30000 for each server name.
    # Side effect: monkeypatches two module attributes for the duration of the test.
    async def _fake_create_rollout_servers(args: Namespace, **kwargs: Any) -> dict[str, _RecordingServer]:
        # [FUNCTION DOC] HELPER: Ignores its arguments and returns the captured servers mapping.
        return servers

    async def _fake_resolve_router_addrs(args: Namespace, **kwargs: Any) -> dict[str, HostAndPort]:
        # [FUNCTION DOC] HELPER: Ignores its arguments and returns a router address per server name.
        return {name: HostAndPort(host="10.0.0.1", port=30000) for name in servers}

    monkeypatch.setattr(inference_controller_module, "create_rollout_servers", _fake_create_rollout_servers)
    monkeypatch.setattr(inference_controller_module, "resolve_router_addrs", _fake_resolve_router_addrs)


class _RefusingWorkerProvider(_FakeWorkerProvider):
    # [CLASS DOC] HELPER: Fake provider with no cells that fails the test if it is initialized or
    # watched, used to prove a debug train-only run never touches providers.
    """A provider a run must never touch, so touching it is the failure."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the base fake with no cells.
        super().__init__([])

    async def init(self) -> None:
        # [FUNCTION DOC] HELPER: Always raises an assertion error (providers must not be initialized).
        raise AssertionError("debug_train_only must not init any worker provider")

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        # [FUNCTION DOC] HELPER: Always raises an assertion error (cells must not be watched).
        raise AssertionError("debug_train_only must not watch cells")


async def _init_controller(args: Namespace, *, engine_provider: _FakeWorkerProvider) -> None:
    # [FUNCTION DOC] HELPER: Constructs a real controller from the arguments, the given engine
    # provider and one empty fake router provider, awaits its init, then awaits its dispose.
    # Returns nothing; propagates any init error.
    controller = InferenceController(args, engine_provider=engine_provider, router_providers=[_FakeWorkerProvider([])])
    await controller.init()
    await controller.dispose()


class TestReconcileAfterAFailedInit:
    # [CLASS DOC] Tests reconcile against a real rollout server whose cell initialization fails
    # (the cell init method is monkeypatched), covering retry on the next identical observation
    # and the status surface.
    @pytest.mark.asyncio
    async def test_an_unchanged_observation_rebuilds_a_cell_whose_init_failed(self, monkeypatch):
        # [FUNCTION DOC] Wires a real rollout server (no colocate, no ft components) to the
        # controller. First patches the cell init to record and raise a runtime error, reconciles a
        # cell and expects that error; then patches init to just record and reconciles the identical
        # observation again. Asserts init ran twice for that cell and the server tracks it exactly
        # once, i.e. the failed cell was dropped so the second sweep retries. Finally disposes the
        # server under the controller's lock.
        """A cell kept after a failed init keeps matching its own observation, so the identical next sweep never retries it."""
        init_calls: list[str] = []

        async def _record_then_raise(cell: ServerCell) -> None:
            # [FUNCTION DOC] HELPER: Records the cell id, then raises a runtime error
            # "injected init failure".
            init_calls.append(cell.meta.cell_id)
            raise RuntimeError("injected init failure")

        async def _record(cell: ServerCell) -> None:
            # [FUNCTION DOC] HELPER: Records the cell id and succeeds.
            init_calls.append(cell.meta.cell_id)

        controller = _make_controller({})
        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=False, ft_components=[]),
            context_lock=controller.context_lock,
            engine_provider=_FakeWorkerProvider([]),
        )
        controller.servers = {"model-a": srv}
        info = _make_cell_info()
        monkeypatch.setattr(ServerCell, "init", _record_then_raise)

        with pytest.raises(RuntimeError, match="injected init failure"):
            await controller._reconcile(info.cell_id, info)

        monkeypatch.setattr(ServerCell, "init", _record)
        await controller._reconcile(info.cell_id, info)

        assert init_calls == [info.cell_id, info.cell_id]
        assert list(srv.server_cells) == [info.cell_id]
        async with controller.context_lock:
            await srv.dispose()

    @pytest.mark.asyncio
    async def test_a_cell_whose_init_failed_is_not_reported_as_a_live_cell(self, monkeypatch):
        # [FUNCTION DOC] Patches cell init to always raise, reconciles a cell (expecting the
        # error), and asserts the controller's cell statuses mapping is empty.
        # KNOWN ISSUE: the real rollout server created here is never disposed in this test.
        """A dropped cell must vanish from the status surface, otherwise the dashboard shows an engine nobody owns."""
        controller = _make_controller({})
        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=False, ft_components=[]),
            context_lock=controller.context_lock,
            engine_provider=_FakeWorkerProvider([]),
        )
        controller.servers = {"model-a": srv}
        info = _make_cell_info()
        monkeypatch.setattr(ServerCell, "init", _raise_async)

        with pytest.raises(RuntimeError, match="injected init failure"):
            await controller._reconcile(info.cell_id, info)

        assert await controller.get_cell_statuses() == {}


class TestPerModelHealthCheckerActiveness:
    # [CLASS DOC] Tests that with several policies (models) the controller pauses and resumes
    # probing only for the policy it is told about, and for all policies when no model is named.
    @staticmethod
    def _controller(*model_ids: str) -> tuple[InferenceController, dict[str, _RecordingServer]]:
        # [FUNCTION DOC] HELPER: Builds one recording server per model id (named by the id) and a
        # controller over them. Returns the controller and the servers mapping.
        servers = {model_id: _RecordingServer(model_name=model_id) for model_id in model_ids}
        return _make_controller(servers), servers

    @pytest.mark.asyncio
    async def test_a_weight_update_pauses_only_the_probing_of_its_own_policy(self):
        # [FUNCTION DOC] Marks the solver server updatable, starts a weight update for solver, and
        # asserts solver's tracker is inactive while verifier's tracker value is unchanged.
        """The other policy keeps serving through this window, and an unprobed engine is not healed."""
        controller, servers = self._controller("solver", "verifier")
        servers["solver"].update_weights = True
        before = servers["verifier"].health_checker_activeness.get()

        await controller.start_update_weights(model_id="solver")

        assert not servers["solver"].health_checker_activeness.get().active
        assert servers["verifier"].health_checker_activeness.get() == before

    @pytest.mark.asyncio
    async def test_a_rollout_of_one_policy_does_not_resume_the_probing_of_another(self):
        # [FUNCTION DOC] Pauses solver's monitoring via the controller's private pause method
        # under the lock, prepares a rollout for verifier only, and asserts solver stays paused
        # while verifier is active.
        """This is the race: verifier's next round used to un-pause probing of solver's engines
        mid-broadcast, so the checker reported a live cell unhealthy and recycled it."""
        controller, servers = self._controller("solver", "verifier")
        async with controller.context_lock:
            await controller._health_monitoring_pause("solver")

        await controller.prepare_rollout(rollout_id=0, model_id="verifier")

        assert not servers["solver"].health_checker_activeness.get().active
        assert servers["verifier"].health_checker_activeness.get().active

    @pytest.mark.asyncio
    async def test_a_policy_that_finished_training_leaves_its_engines_probed_again(self):
        # [FUNCTION DOC] Runs a start and end weight update for solver (with an empty snapshot),
        # then prepares an eval for solver, and asserts both solver and verifier are active.
        """Its last round ends with a weight update, and no later rollout of its own would ever un-pause it."""
        controller, servers = self._controller("solver", "verifier")
        servers["solver"].update_weights = True
        await controller.start_update_weights(model_id="solver")
        await controller.end_update_weights({})

        await controller.prepare_eval(model_id="solver")

        assert servers["solver"].health_checker_activeness.get().active
        assert servers["verifier"].health_checker_activeness.get().active

    @pytest.mark.asyncio
    async def test_a_rollout_without_a_model_id_resumes_every_policy(self):
        # [FUNCTION DOC] Pauses all trackers, prepares a rollout naming no model, and asserts every
        # tracker is active.
        """A single policy run names no model, and must keep resuming the whole fleet."""
        controller, servers = self._controller("solver", "verifier")
        for srv in servers.values():
            srv.health_checker_activeness.bump_active(False)

        await controller.prepare_rollout(rollout_id=0)

        assert all(srv.health_checker_activeness.get().active for srv in servers.values())

    @pytest.mark.asyncio
    async def test_offloading_pauses_every_policy(self):
        # [FUNCTION DOC] Offloads through the controller and asserts no server's tracker is active.
        """Colocate offloads the whole fleet at once, so every model stops answering probes."""
        controller, servers = self._controller("solver", "verifier")

        await controller.offload()

        assert not any(srv.health_checker_activeness.get().active for srv in servers.values())

    @pytest.mark.asyncio
    async def test_an_eval_resumes_every_policy(self):
        # [FUNCTION DOC] Pauses all trackers, prepares an eval naming no model, and asserts every
        # tracker is active.
        """Eval names no model either; it drives whatever fleet the run deployed."""
        controller, servers = self._controller("solver", "verifier")
        for srv in servers.values():
            srv.health_checker_activeness.bump_active(False)

        await controller.prepare_eval()

        assert all(srv.health_checker_activeness.get().active for srv in servers.values())


class TestInitSubscription:
    # [CLASS DOC] Tests how the controller's init uses the engine provider: ordering of provider
    # init, router address resolution, server creation and cell watching, and which pools are
    # watched.
    @pytest.mark.asyncio
    async def test_init_initializes_the_provider_before_reading_anything_from_it(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] Uses a provider subclass and patched factory functions that all append a
        # label to a shared list, runs init and dispose, and asserts the exact order: provider
        # init, router address resolution, server creation, then watching cells.
        """A provider that discovers its engines in init() answers an empty fleet until then, so the
        router addresses and the startup barrier would both be sized against nothing."""
        order: list[str] = []

        class _OrderRecordingProvider(_FakeWorkerProvider):
            # [CLASS DOC] HELPER: Fake provider that appends a label to the shared order list
            # when initialized and when watched, then defers to the base fake.
            async def init(self) -> None:
                # [FUNCTION DOC] HELPER: Appends the label init, then runs the base init.
                order.append("init")
                await super().init()

            async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
                # [FUNCTION DOC] HELPER: Appends the label watch_cells, then returns the base result.
                order.append("watch_cells")
                return await super().watch_cells(reconcile)

        async def _fake_create_rollout_servers(args: Namespace, **kwargs: Any) -> dict[str, _RecordingServer]:
            # [FUNCTION DOC] HELPER: Appends the label create_rollout_servers and returns a mapping
            # with a single default recording server.
            order.append("create_rollout_servers")
            return {"default": _RecordingServer()}

        async def _fake_resolve_router_addrs(args: Namespace, **kwargs: Any) -> dict[str, HostAndPort]:
            # [FUNCTION DOC] HELPER: Appends the label resolve_router_addrs and returns a default
            # router address (host 10.0.0.1, port 30000).
            order.append("resolve_router_addrs")
            return {"default": HostAndPort(host="10.0.0.1", port=30000)}

        monkeypatch.setattr(inference_controller_module, "create_rollout_servers", _fake_create_rollout_servers)
        monkeypatch.setattr(inference_controller_module, "resolve_router_addrs", _fake_resolve_router_addrs)
        args = make_args()
        provider = _OrderRecordingProvider([], pool_ids=compute_engine_pool_ids(args))

        await _init_controller(args, engine_provider=provider)

        assert order == ["init", "resolve_router_addrs", "create_rollout_servers", "watch_cells"]

    @pytest.mark.asyncio
    async def test_init_watches_the_engine_provider_it_was_handed(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Runs init and dispose with a fake provider serving the engine pools
        # computed from the arguments and asserts the provider was asked to watch exactly those
        # pool ids, and neither the first router pool nor the session-server pool.
        """The pools are the provider's own, so the controller may only open a watch on what it was given."""
        args = make_args()
        provider = _FakeWorkerProvider([], pool_ids=compute_engine_pool_ids(args))
        _patch_init(monkeypatch, servers={"default": _RecordingServer()})

        await _init_controller(args, engine_provider=provider)

        assert provider.watched_pool_ids == compute_engine_pool_ids(args)
        assert compute_router_pool_id(0) not in provider.watched_pool_ids
        assert "session-server" not in provider.watched_pool_ids

    @pytest.mark.asyncio
    async def test_init_subscribes_narrowly_enough_to_never_see_a_router_cell(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Gives the fake provider both a router cell (empty meta, router pool) and
        # an engine cell, but only the engine pools as its pool list. After init and dispose,
        # asserts the recording server received only the add of the engine cell (a router cell
        # would have broken reading engine meta).
        """A router cell carries no engine meta, so reading one as engine meta would kill startup; the
        controller is safe because it subscribes to the engine pools alone."""
        args = make_args()
        assert compute_router_pool_id(0) not in compute_engine_pool_ids(args)

        router_info = CellInfo(
            cell_id="inference-router-0-0",
            pool_id=compute_router_pool_id(0),
            alive=True,
            worker_names=["inference-router-0-0-0"],
            workers_hash="pseudo-hash-router",
            meta={},
        )
        engine_info = _make_cell_info(model_id="default", pool_id=compute_engine_pool_ids(args)[0])
        provider = _FakeWorkerProvider([router_info, engine_info], pool_ids=compute_engine_pool_ids(args))
        srv = _RecordingServer()
        _patch_init(monkeypatch, servers={"default": srv})

        await _init_controller(args, engine_provider=provider)

        assert srv.calls == [("add", engine_info.cell_id)]


class TestEngineMetaContract:
    # [CLASS DOC] Pins the contract between the engine spec's meta mapping and the controller's
    # reader that turns a cell info into server cell metadata.
    def test_the_real_spec_meta_roundtrips_into_server_cell_metadata(self, tmp_path: Path):
        # [FUNCTION DOC] Writes a sglang config file with one model named default and one decode
        # server group (4 GPUs, 2 per engine), builds arguments with it, 4 rollout GPUs and an api
        # key, takes the single engine spec, builds a cell info for cell index one whose meta comes
        # from the spec, and asserts the controller's metadata computation yields: model default,
        # worker type decode, the cell id, 2 GPUs per engine, GPU offset 2, the api key from the
        # arguments, the first worker name, no offload, weight updates on, and the same hash.
        """The engine spec's meta dict and the driver-side reader share one key set, pinned end to end."""
        config_path: Path = tmp_path / "sglang.yaml"
        config_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: decode\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 2\n"
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4, sglang_api_key="from-args")
        (spec,) = specs_inference_engine(args)

        info = CellInfo(
            cell_id="inference-engine-0-0-1",
            pool_id=spec.name,
            alive=True,
            worker_names=["inference-engine-0-0-1-0"],
            workers_hash="pseudo-hash-0",
            meta=spec.meta(WorkerMetaContext(cell_index=1)),
        )

        assert _compute_server_cell_meta_from_info(info) == ServerCellMetadata(
            model_id="default",
            worker_type="decode",
            cell_id="inference-engine-0-0-1",
            num_gpus_per_engine=2,
            gpu_offset=2,
            sglang_api_key="from-args",
            worker_name="inference-engine-0-0-1-0",
            needs_offload=False,
            update_weights=True,
            workers_hash="pseudo-hash-0",
        )


class TestUpdateWeightsLockWindow:
    # [CLASS DOC] Tests that the context lock is held between starting and ending (or aborting)
    # a weight update and blocks concurrent reconcile meanwhile.
    @pytest.mark.asyncio
    async def test_the_lock_is_held_from_start_until_end_update_weights(self):
        # [FUNCTION DOC] Starts a weight update on a server-less controller, asserts the lock is
        # held, ends the update with the returned snapshot, and asserts the lock is released.
        """start_update_weights opens a lock window that only end_update_weights closes."""
        controller = _make_controller({})

        info = await controller.start_update_weights()
        assert controller.context_lock.locked

        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)
        assert not controller.context_lock.locked

    @pytest.mark.asyncio
    async def test_abort_update_weights_closes_the_window_without_marking_weights_ready(self):
        # [FUNCTION DOC] Starts then aborts a weight update and asserts the lock is held in
        # between and released afterwards.
        # KNOWN ISSUE: the name and docstring promise that cells stay pending (not marked ready),
        # but the test has no servers and never asserts anything about cell readiness.
        """A sync that failed on the trainer must release the lock while leaving every cell pending."""
        controller = _make_controller({})

        await controller.start_update_weights()
        assert controller.context_lock.locked

        await controller.abort_update_weights()
        assert not controller.context_lock.locked

    @pytest.mark.asyncio
    async def test_reconcile_waits_while_the_update_weights_window_is_open(self):
        # [FUNCTION DOC] Opens the update window, starts a reconcile of a router cell id as a
        # background task, yields to the event loop five times and asserts the task is still not
        # done; then ends the update and awaits the task to completion.
        """A concurrent reconcile must not mutate the engine set mid weight update."""
        controller = _make_controller({})
        info = await controller.start_update_weights()

        reconcile_task = asyncio.create_task(controller._reconcile("miles-router-0-0", None))
        for _ in range(5):
            await asyncio.sleep(0)
        assert not reconcile_task.done()

        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)
        await reconcile_task

    @pytest.mark.asyncio
    async def test_a_plain_locked_call_does_not_leave_the_lock_held(self):
        # [FUNCTION DOC] Prepares an eval on a server-less controller and asserts the lock is
        # released afterwards.
        """Ordinary controller methods release the lock when they return."""
        controller = _make_controller({})
        await controller.prepare_eval()
        assert not controller.context_lock.locked


class TestServersShareTheControllerLock:
    # [CLASS DOC] Tests that real rollout servers must share the controller's context lock.
    @pytest.mark.asyncio
    async def test_reconcile_can_drive_the_server_it_owns(self):
        # [FUNCTION DOC] Builds a real rollout server using the controller's own lock, reconciles
        # a cell as gone, and asserts the server ends with no cells (no lock error).
        """The controller lock is the very lock its servers require, so reconcile works end to end."""
        controller = _make_controller({})
        srv = RolloutServer(
            server_cells={},
            args=SimpleNamespace(),
            context_lock=controller.context_lock,
            engine_provider=_FakeWorkerProvider([]),
        )
        controller.servers = {"default": srv}
        info = _make_cell_info()

        await controller._reconcile(info.cell_id, None)
        assert srv.server_cells == {}

    @pytest.mark.asyncio
    async def test_a_server_holding_a_foreign_lock_is_rejected(self):
        # [FUNCTION DOC] Builds a real rollout server with a different lock object of the same
        # name, then offloads through the controller and expects an assertion error whose message
        # contains "must be called with".
        """A server wired up with its own lock instead of the controller's is a wiring bug."""
        controller = _make_controller({})
        srv = RolloutServer(
            server_cells={},
            args=SimpleNamespace(),
            context_lock=ContextLock("InferenceController"),
            engine_provider=_FakeWorkerProvider([]),
        )
        controller.servers = {"default": srv}

        with pytest.raises(AssertionError, match="must be called with"):
            await controller.offload()


class TestUpdatableModelSelection:
    # [CLASS DOC] Tests which servers a weight update or weight check targets when no model id is
    # given: exactly the one server flagged updatable, none if none is, an error if several are.
    @staticmethod
    def _controller(*servers: _RecordingServer) -> InferenceController:
        # [FUNCTION DOC] HELPER: Builds a controller whose servers mapping is keyed by each
        # server's model name.
        return _make_controller({srv.model_name: srv for srv in servers})

    @pytest.mark.asyncio
    async def test_only_the_updatable_models_engines_receive_weights(self):
        # [FUNCTION DOC] Two servers, actor (updatable) and ref (frozen), each with one api client.
        # Starts a weight update and asserts the returned engine clients are only the actor's.
        """A frozen reference model handed the trainer's weights stops being the baseline the
        KL term is measured against."""
        actor = _RecordingServer(model_name="actor", update_weights=True)
        actor.api_clients = ["actor-client"]
        ref = _RecordingServer(model_name="ref", update_weights=False)
        ref.api_clients = ["ref-client"]

        updatable = await self._controller(actor, ref).start_update_weights()

        assert updatable.rollout_engines == ["actor-client"]

    @pytest.mark.asyncio
    async def test_an_inference_only_deployment_updates_nothing(self):
        # [FUNCTION DOC] With only a frozen server, starts a weight update and asserts the engine
        # clients list and the snapshot mapping are both empty.
        """No model is being trained, so there is no engine to push weights into; returning a
        frozen model's engines here would overwrite it."""
        updatable = await self._controller(_RecordingServer(model_name="ref")).start_update_weights()

        assert updatable.rollout_engines == []
        assert updatable.snapshot_cell_id_to_hashes == {}

    @pytest.mark.asyncio
    async def test_two_updatable_models_are_refused_by_name(self):
        # [FUNCTION DOC] With two updatable servers and no model id, starting a weight update
        # raises a value error whose message says multiple servers have update weights enabled.
        """Picking one arbitrarily would silently train one model and leave the other stale."""
        controller = self._controller(
            _RecordingServer(model_name="a", update_weights=True),
            _RecordingServer(model_name="b", update_weights=True),
        )

        with pytest.raises(ValueError, match="Multiple servers have update_weights=True"):
            await controller.start_update_weights()

    @pytest.mark.asyncio
    async def test_the_ambiguity_points_at_the_way_out_of_it(self):
        # [FUNCTION DOC] Same setup as the two-updatable test, but asserts the value error message
        # tells the caller to pass a model id to update exactly one of them.
        """Naming a model id is the supported way to update one of several trainable models."""
        controller = self._controller(
            _RecordingServer(model_name="a", update_weights=True),
            _RecordingServer(model_name="b", update_weights=True),
        )

        with pytest.raises(ValueError, match="Pass model_id to update exactly one of them"):
            await controller.start_update_weights()

    @pytest.mark.asyncio
    async def test_the_weight_checker_skips_the_frozen_models(self):
        # [FUNCTION DOC] Runs a snapshot weight check over an updatable actor and a frozen ref;
        # asserts the result lists only the actor and the ref server recorded no call.
        """reset_tensors on a model nobody will rewrite scrambles it for the rest of the run."""
        actor = _RecordingServer(model_name="actor", update_weights=True)
        ref = _RecordingServer(model_name="ref", update_weights=False)

        assert await self._controller(actor, ref).check_weights(action="snapshot") == ["actor"]
        assert ref.calls == []

    @pytest.mark.asyncio
    async def test_the_weight_checker_is_a_noop_without_an_updatable_model(self):
        # [FUNCTION DOC] With only a frozen server, a compare weight check returns an empty list
        # and the server records no call.
        """Nothing was updated, so there is nothing to compare against."""
        ref = _RecordingServer(model_name="ref")

        assert await self._controller(ref).check_weights(action="compare") == []
        assert ref.calls == []

    @pytest.mark.asyncio
    async def test_check_weights_forwards_all_selection_arguments(self):
        # [FUNCTION DOC] Calls the controller's weight check with action compare, quantization
        # error allowed, selector first and a skip list holding lm_head; asserts the updatable
        # server received exactly those four values.
        """Losing the selector or the skip list here would compare tensors the caller asked to leave alone."""
        actor = _RecordingServer(model_name="actor", update_weights=True)

        await self._controller(actor).check_weights(
            action="compare", allow_quant_error=True, selector="first", skip_list=["lm_head"]
        )

        assert actor.check_weights_kwargs == [
            dict(action="compare", allow_quant_error=True, selector="first", skip_list=["lm_head"])
        ]


class TestGetServersOfModelId:
    # [CLASS DOC] Tests how an explicit model id scopes weight updates and weight checks to one
    # server, and the errors for an unknown or frozen model.
    @staticmethod
    def _controller(*servers: _RecordingServer) -> InferenceController:
        # [FUNCTION DOC] HELPER: Builds a controller whose servers mapping is keyed by each
        # server's model name.
        return _make_controller({srv.model_name: srv for srv in servers})

    @pytest.mark.asyncio
    async def test_a_named_policy_scopes_the_update_to_its_own_server(self):
        # [FUNCTION DOC] Two updatable servers with distinct api clients; starts an update for
        # beta and asserts only beta's client is returned.
        """Several policies train at once, so the ambiguity a single-policy run never hits is the normal case."""
        alpha = _RecordingServer(model_name="alpha", update_weights=True)
        alpha.api_clients = ["alpha-client"]
        beta = _RecordingServer(model_name="beta", update_weights=True)
        beta.api_clients = ["beta-client"]

        updatable = await self._controller(alpha, beta).start_update_weights(model_id="beta")

        assert updatable.rollout_engines == ["beta-client"]

    @pytest.mark.asyncio
    async def test_an_unknown_model_id_is_refused(self):
        # [FUNCTION DOC] Starting an update for a model id no server has raises an assertion error
        # naming the unknown id and listing the known ids alpha and beta.
        """Guessing a server here would push one policy's weights into another policy's engines."""
        controller = self._controller(
            _RecordingServer(model_name="alpha", update_weights=True),
            _RecordingServer(model_name="beta", update_weights=True),
        )

        with pytest.raises(AssertionError, match=r"No server for model_id 'nope'.*\['alpha', 'beta'\]"):
            await controller.start_update_weights(model_id="nope")

    @pytest.mark.asyncio
    async def test_a_frozen_server_named_explicitly_is_refused(self):
        # [FUNCTION DOC] Starting an update for a model whose server is frozen raises an assertion
        # error saying the server for that model id is frozen.
        """A reference model is frozen on purpose; overwriting it destroys the baseline the KL term uses."""
        controller = self._controller(
            _RecordingServer(model_name="alpha", update_weights=True),
            _RecordingServer(model_name="ref", update_weights=False),
        )

        with pytest.raises(AssertionError, match="Server for model_id 'ref' is frozen"):
            await controller.start_update_weights(model_id="ref")

    @pytest.mark.asyncio
    async def test_the_weight_checker_compares_only_the_named_policys_engines(self):
        # [FUNCTION DOC] With two updatable servers, a checksum weight check naming alpha returns
        # only alpha and beta's server records no call.
        """A checksum taken against the other policy's engines would report a mismatch on every step."""
        alpha = _RecordingServer(model_name="alpha", update_weights=True)
        beta = _RecordingServer(model_name="beta", update_weights=True)

        assert await self._controller(alpha, beta).check_weights(action="checksum", model_id="alpha") == ["alpha"]
        assert beta.calls == []


class TestEnsureCellsReady:
    # [CLASS DOC] Tests that starting a weight update waits only for the cells of the targeted
    # model (or of all models when none is named) to be pending-weights or serving.
    @staticmethod
    def _controller() -> InferenceController:
        # [FUNCTION DOC] HELPER: Builds a controller with alpha (updatable, one ready cell with
        # hash hash-a) and beta (frozen, one cell that is not yet pending or serving, hash hash-b).
        """One policy is serving, the other still has a cell coming up."""
        ready = _FakeUpdatableCell("hash-a")
        pending = _FakeUpdatableCell("hash-b")
        pending.is_pending_weights_or_serving = False
        return _make_controller(
            {
                "alpha": _RecordingServer({"alpha-0": ready}, model_name="alpha", update_weights=True),
                "beta": _RecordingServer({"beta-0": pending}, model_name="beta", update_weights=False),
            }
        )

    @pytest.mark.asyncio
    async def test_only_the_cells_of_the_named_policy_are_waited_for(self, monkeypatch):
        # [FUNCTION DOC] Sets the cells-ready timeout constant to zero, starts an update for alpha
        # and asserts it succeeds with a snapshot of just alpha's cell and hash.
        """One policy must not be blocked from updating by another policy's cells still coming up."""
        monkeypatch.setattr(inference_controller_module, "CELLS_READY_TIMEOUT_SECONDS", 0)

        updatable = await self._controller().start_update_weights(model_id="alpha")

        assert updatable.snapshot_cell_id_to_hashes == {"alpha-0": "hash-a"}

    @pytest.mark.asyncio
    async def test_an_unscoped_update_still_waits_for_every_cell_of_the_run(self, monkeypatch):
        # [FUNCTION DOC] With the timeout set to zero, an update naming no model raises a timeout
        # error mentioning that it waited for one of two cells.
        """The scope is what makes the wait short, so the unscoped path must keep covering all engines."""
        monkeypatch.setattr(inference_controller_module, "CELLS_READY_TIMEOUT_SECONDS", 0)

        with pytest.raises(TimeoutError, match="waiting for 1/2 cells"):
            await self._controller().start_update_weights()


class TestMemoryLifecycleFanOut:
    # [CLASS DOC] Tests that the controller's memory offload and onload entry points fan out to
    # every server with the right memory-type tags.
    @pytest.mark.asyncio
    async def test_memory_lifecycle_entrypoints_fan_out_with_exact_tags(self):
        # [FUNCTION DOC] With two servers, calls offload with the kv-cache tag, onload with the
        # cuda-graph tag, onload-weights and onload-kv. Asserts each server saw one offload with
        # the kv-cache tag and three onloads in order: cuda graph; weights; then kv cache plus
        # cuda graph.
        """Every server must be told exactly which memory pools to release and to reclaim."""
        first, second = _RecordingServer(model_name="a"), _RecordingServer(model_name="b")
        controller = _make_controller({"a": first, "b": second})

        await controller.offload(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        await controller.onload(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH])
        await controller.onload_weights()
        await controller.onload_kv()

        for srv in (first, second):
            assert srv.offload_tags == [[GPU_MEMORY_TYPE_KV_CACHE]]
            assert srv.onload_tags == [
                [GPU_MEMORY_TYPE_CUDA_GRAPH],
                [GPU_MEMORY_TYPE_WEIGHTS],
                [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH],
            ]


class TestUpdatableEnginesPayload:
    # [CLASS DOC] Tests the payload returned by start-update-weights and how end-update-weights
    # uses its generation snapshot.
    @pytest.mark.asyncio
    async def test_start_update_weights_returns_clients_gpu_layout_and_generation_snapshot(self):
        # [FUNCTION DOC] An updatable server with two cells, api clients, GPU counts 2 and 4 and
        # offsets 0 and 2, plus a frozen server. Starts and ends the update, and asserts the
        # payload equals clients, counts, offsets and a cell-id to workers-hash snapshot.
        """The trainer indexes these four lists in parallel, so swapping or dropping one misplaces every shard."""
        srv = _RecordingServer(
            {"engine-0": _FakeUpdatableCell("hash-a"), "engine-1": _FakeUpdatableCell("hash-b")},
            model_name="actor",
            update_weights=True,
        )
        srv.api_clients = ["client-0", "client-1"]
        srv.engine_gpu_counts = [2, 4]
        srv.engine_gpu_offsets = [0, 2]
        controller = _make_controller({"actor": srv, "ref": _RecordingServer(model_name="ref")})

        updatable = await controller.start_update_weights()
        await controller.end_update_weights(snapshot_cell_id_to_hashes=updatable.snapshot_cell_id_to_hashes)

        assert updatable == UpdatableEngines(
            rollout_engines=["client-0", "client-1"],
            engine_gpu_counts=[2, 4],
            engine_gpu_offsets=[0, 2],
            snapshot_cell_id_to_hashes={"engine-0": "hash-a", "engine-1": "hash-b"},
        )

    @pytest.mark.asyncio
    async def test_end_update_weights_skips_a_cell_from_a_different_worker_generation(self):
        # [FUNCTION DOC] Two cells; ends the update with a snapshot whose hash for the first cell
        # differs from the cell's current hash. Asserts the mismatched cell was not marked ready
        # and the matching cell was marked ready once.
        """A cell relaunched during the update runs new processes that never received these weights."""
        relaunched, untouched = _FakeUpdatableCell("hash-new"), _FakeUpdatableCell("hash-b")
        srv = _RecordingServer(
            {"engine-0": relaunched, "engine-1": untouched}, model_name="actor", update_weights=True
        )
        controller = _make_controller({"actor": srv})

        await controller.start_update_weights()
        await controller.end_update_weights(snapshot_cell_id_to_hashes={"engine-0": "hash-old", "engine-1": "hash-b"})

        assert (relaunched.marked_ready, untouched.marked_ready) == (0, 1)


class TestInitLifecycle:
    # [CLASS DOC] Tests the constructed controller's init and dispose: the debug train-only
    # shortcut, the lock handed to the server factory, eval fleet creation, startup gates,
    # initialized reporting and ownership of the watcher and ticker.
    @staticmethod
    def _controller(args: Namespace, *, engine_provider: _FakeWorkerProvider | None = None) -> InferenceController:
        # [FUNCTION DOC] HELPER: Constructs a real controller from the arguments, the given engine
        # provider (default an empty fake) and one empty fake router provider.
        return InferenceController(
            args,
            engine_provider=engine_provider if engine_provider is not None else _FakeWorkerProvider([]),
            router_providers=[_FakeWorkerProvider([])],
        )

    @pytest.mark.asyncio
    async def test_debug_train_only_init_has_no_rollout_side_effects(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Patches server creation, router address resolution and the dashboard
        # router registration to fail the test if called, and uses refusing providers. Runs init
        # with debug train-only on and asserts: no servers, no eval fleet, no watcher disposers,
        # no ticker, and the engine provider never initialized.
        """A train-only debug run owns no engines, so init must not reach any rollout machinery."""

        async def _no_servers(args: Namespace, **kwargs: Any) -> dict:
            # [FUNCTION DOC] HELPER: Always raises an assertion error.
            raise AssertionError("debug_train_only must not create rollout servers")

        async def _no_router_addrs(args: Namespace, **kwargs: Any) -> dict:
            # [FUNCTION DOC] HELPER: Always raises an assertion error.
            raise AssertionError("debug_train_only must not resolve any router")

        monkeypatch.setattr(inference_controller_module, "create_rollout_servers", _no_servers)
        monkeypatch.setattr(inference_controller_module, "resolve_router_addrs", _no_router_addrs)
        monkeypatch.setattr(
            dashboard_hooks, "register_router", lambda args: pytest.fail("debug_train_only has no router")
        )
        provider = _RefusingWorkerProvider()
        controller = InferenceController(
            make_args(debug_train_only=True),
            engine_provider=provider,
            router_providers=[_RefusingWorkerProvider()],
        )

        await controller.init()

        assert controller.servers == {}
        assert controller._eval_fleet is None
        assert controller._watcher_disposers == []
        assert controller._ticker is None
        assert provider.initialized is False

    @pytest.mark.asyncio
    async def test_init_passes_its_exact_context_lock_to_the_server_factory(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Captures the keyword arguments the server factory receives during init and
        # dispose, and asserts the context lock passed is the identical object the controller owns.
        """A server built on a second lock would let engine work run inside the controller's own window."""
        received: dict[str, Any] = {}

        async def _fake_create_rollout_servers(args: Namespace, **kwargs: Any) -> dict[str, _RecordingServer]:
            # [FUNCTION DOC] HELPER: Copies the keyword arguments into the captured mapping and
            # returns a mapping with a default recording server.
            received.update(kwargs)
            return {"default": _RecordingServer()}

        _patch_init(monkeypatch, servers={"default": _RecordingServer()})
        monkeypatch.setattr(inference_controller_module, "create_rollout_servers", _fake_create_rollout_servers)
        controller = self._controller(make_args())

        await controller.init()
        await controller.dispose()

        assert received["context_lock"] is controller.context_lock

    @pytest.mark.asyncio
    async def test_init_creates_the_eval_fleet_from_the_eval_server(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Substitutes the recording eval fleet class, provides servers default and
        # eval, and runs init and dispose with two eval GPUs. Asserts the controller's eval fleet is
        # the recording fleet, built from the eval server and the controller's own arguments.
        """The eval fleet drives the dedicated eval engines, so it must be handed that server and no other."""
        monkeypatch.setattr(inference_controller_module, "InferenceControllerEvalFleet", _RecordingEvalFleet)
        default, eval_srv = _RecordingServer(model_name="default"), _RecordingServer(model_name="eval")
        _patch_init(monkeypatch, servers={"default": default, "eval": eval_srv})
        controller = self._controller(make_args(eval_num_gpus=2))

        await controller.init()
        await controller.dispose()

        assert isinstance(controller._eval_fleet, _RecordingEvalFleet)
        assert controller._eval_fleet.srv is eval_srv
        assert controller._eval_fleet.args is controller.args

    @pytest.mark.asyncio
    async def test_init_without_eval_gpus_creates_no_eval_fleet(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Makes the eval fleet class fail the test if constructed, runs init and
        # dispose with zero eval GPUs, and asserts the controller has no eval fleet.
        """A run without dedicated eval engines has no eval server to build a fleet from."""
        monkeypatch.setattr(
            inference_controller_module,
            "InferenceControllerEvalFleet",
            lambda *args, **kwargs: pytest.fail("no eval fleet without eval gpus"),
        )
        _patch_init(monkeypatch, servers={"default": _RecordingServer()})
        controller = self._controller(make_args(eval_num_gpus=0))

        await controller.init()
        await controller.dispose()

        assert controller._eval_fleet is None

    @pytest.mark.asyncio
    async def test_init_registers_routing_and_waits_for_every_startup_gate(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Records dashboard router registrations, provides one ready server and one
        # whose expected-cells wait is blocked on an event. Runs init as a background task, yields
        # twenty times and asserts it is not finished while the ready server was waited on once;
        # then opens the gate, awaits init (five second limit) and disposes. Asserts the arguments
        # were registered once and the blocked server was also waited on once.
        """Returning before every server has its cells would start a rollout against engines that are not up."""
        registered: list[Namespace] = []
        monkeypatch.setattr(dashboard_hooks, "register_router", registered.append)
        gate = asyncio.Event()
        ready, blocked = _RecordingServer(), _RecordingServer(cells_gate=gate)
        _patch_init(monkeypatch, servers={"default": ready, "frozen": blocked})
        args = make_args()
        controller = self._controller(args)

        task = asyncio.create_task(controller.init())
        for _ in range(20):
            await asyncio.sleep(0)
        assert not task.done()
        assert ready.waited_init_expected_num_cells == 1
        gate.set()
        await asyncio.wait_for(task, timeout=5)
        await controller.dispose()

        assert registered == [args]
        assert blocked.waited_init_expected_num_cells == 1

    @pytest.mark.asyncio
    async def test_init_does_not_report_itself_initialized_before_its_fleet_is_in(self, monkeypatch):
        # [FUNCTION DOC] Starts init with a server blocked on its startup gate, yields twenty
        # times and asserts the controller reports not initialized; opens the gate, awaits init
        # and asserts it reports initialized; then disposes.
        """A take-over reads this answer, and a controller still short of its cells is not one to take over."""
        gate = asyncio.Event()
        _patch_init(monkeypatch, servers={"frozen": _RecordingServer(cells_gate=gate)})
        controller = self._controller(make_args())

        task = asyncio.create_task(controller.init())
        for _ in range(20):
            await asyncio.sleep(0)
        assert await controller.is_initialized() is False

        gate.set()
        await asyncio.wait_for(task, timeout=5)
        assert await controller.is_initialized() is True

        await controller.dispose()

    @pytest.mark.asyncio
    async def test_init_and_dispose_own_the_cell_watch_and_ticker_lifetimes(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Sets the tick interval to ten milliseconds, runs init with a ticking cell,
        # waits fifty milliseconds and asserts the cell was ticked, the ticker uses the module's
        # interval and one watcher disposer exists. After dispose and a short wait, asserts the
        # provider's stop-watch ran once, the disposers list is empty, the ticker is cleared and
        # the cell's tick count no longer grows.
        """A watch or tick loop outliving the controller keeps dialing engines that nobody owns any more."""
        monkeypatch.setattr(inference_controller_module, "TICK_INTERVAL_SECONDS", 0.01)
        cell = _TickingCell()
        provider = _FakeWorkerProvider([])
        _patch_init(monkeypatch, servers={"default": _RecordingServer({"engine-0": cell})})
        controller = self._controller(make_args(), engine_provider=provider)

        await controller.init()
        await asyncio.sleep(0.05)
        assert cell.tick_count > 0
        assert controller._ticker._interval_seconds == inference_controller_module.TICK_INTERVAL_SECONDS
        assert len(controller._watcher_disposers) == 1

        await controller.dispose()
        ticks_at_dispose = cell.tick_count
        await asyncio.sleep(0.01)

        assert provider.stop_watch_calls == 1
        assert controller._watcher_disposers == []
        assert controller._ticker is None
        assert cell.tick_count == ticks_at_dispose


async def _raise_async(cell: ServerCell) -> None:
    # [FUNCTION DOC] HELPER: Replacement for the cell init method that always raises a runtime
    # error "injected init failure".
    raise RuntimeError("injected init failure")


_ROLLOUT_CELL_ID = "inference-engine-0-0-0"


class _StoppingWorkerProvider(_FakeWorkerProvider):
    # [CLASS DOC] HELPER: Fake provider that records stop-cells requests, signals an event when
    # one arrives and optionally waits for a caller supplied future before finishing the stop.
    def __init__(self, *, completion: asyncio.Future | None = None) -> None:
        # [FUNCTION DOC] HELPER: Initializes the base fake with no cells, an empty stopped-cells
        # list, a stop-requested event and the optional completion future.
        super().__init__([])
        self.stopped_cells: list[list[str]] = []
        self.stop_requested = asyncio.Event()
        self._completion = completion

    async def stop_cells(self, *, cell_ids: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Records a copy of the cell ids, sets the stop-requested event, and
        # if a completion future was given waits for it before returning.
        self.stopped_cells.append(list(cell_ids))
        self.stop_requested.set()
        if self._completion is not None:
            await self._completion


def _make_cell_operations_controller(
    provider: _StoppingWorkerProvider, *, probing_paused: bool = False
) -> InferenceController:
    # [FUNCTION DOC] HELPER: Builds a controller with one default recording server tracking the
    # fixed rollout cell id and using the given stopping provider as engine provider. When
    # probing_paused is true the server's activeness tracker is switched off (offloaded state).
    server = _RecordingServer(server_cells={_ROLLOUT_CELL_ID: object()})
    if probing_paused:
        server.health_checker_activeness.bump_active(False)
    return _make_controller({"default": server}, engine_provider=provider)


async def _hold_context_lock(controller: InferenceController) -> tuple[asyncio.Task, asyncio.Event]:
    # [FUNCTION DOC] HELPER: Starts a background task that takes the controller's context lock and
    # holds it until released, and waits until the lock is taken. Returns the holder task and the
    # event that lets it release the lock.
    entered, may_finish = asyncio.Event(), asyncio.Event()

    async def _hold() -> None:
        # [FUNCTION DOC] HELPER: Under the lock, signals entry and waits for the release event.
        async with controller.context_lock:
            entered.set()
            await may_finish.wait()

    holder = asyncio.create_task(_hold())
    await entered.wait()
    return holder, may_finish


class TestCellOperations:
    # [CLASS DOC] Tests that stopping a cell and injecting a fault are serialized through the
    # controller's context lock (so they cannot overlap a weight update) and forwarded to the
    # engine provider.
    @pytest.mark.asyncio
    async def test_stop_cell_between_weight_updates_is_forwarded_to_the_engine_provider(self):
        # [FUNCTION DOC] Stops the fixed cell through the controller and asserts the provider
        # received exactly one stop request listing that cell.
        """The provider owns the processes, so the controller only serializes the suspension."""
        provider = _StoppingWorkerProvider()
        controller = _make_cell_operations_controller(provider)

        await controller.stop_cell_between_weight_updates(_ROLLOUT_CELL_ID)

        assert provider.stopped_cells == [[_ROLLOUT_CELL_ID]]

    @pytest.mark.asyncio
    async def test_stop_cell_between_weight_updates_waits_until_the_weight_update_window_closes(self):
        # [FUNCTION DOC] Holds the lock in a background task, starts a stop as a task and yields
        # once; asserts the provider has no stop request yet. Releases the lock, awaits both and
        # asserts the stop reached the provider.
        """Suspending a cell mid-broadcast leaves the trainer waiting on an engine that is being torn down."""
        provider = _StoppingWorkerProvider()
        controller = _make_cell_operations_controller(provider)
        holder, may_finish = await _hold_context_lock(controller)

        stopping = asyncio.create_task(controller.stop_cell_between_weight_updates(_ROLLOUT_CELL_ID))
        await asyncio.sleep(0)

        assert provider.stopped_cells == []

        may_finish.set()
        await holder
        await stopping

        assert provider.stopped_cells == [[_ROLLOUT_CELL_ID]]

    @pytest.mark.asyncio
    async def test_inject_fault_between_weight_updates_waits_until_the_weight_update_window_closes(self):
        # [FUNCTION DOC] Holds the lock, starts a kill-signal injection as a task and yields once;
        # asserts the worker manager has no injection yet. Releases the lock, awaits both and
        # asserts one injection with the cell id, mode string sigkill and worker index zero.
        """Injection racing a broadcast is the same hazard as suspension, so it takes the same turn."""
        provider = _StoppingWorkerProvider()
        controller = _make_cell_operations_controller(provider)
        holder, may_finish = await _hold_context_lock(controller)

        injecting = asyncio.create_task(
            controller.inject_fault_between_weight_updates(_ROLLOUT_CELL_ID, mode=FailureMode.SIGKILL, sub_index=0)
        )
        await asyncio.sleep(0)

        assert provider._worker_manager_handle.inject_fault.calls == []

        may_finish.set()
        await holder
        await injecting

        assert provider._worker_manager_handle.inject_fault.calls == [
            ((_ROLLOUT_CELL_ID,), {"mode": "sigkill", "worker_in_cell_index": 0})
        ]

    @pytest.mark.asyncio
    async def test_stop_cell_between_weight_updates_is_allowed_while_probing_is_paused(self):
        # [FUNCTION DOC] With the server's probing paused, stopping the cell still reaches the
        # provider (only fault injection checks the pause).
        """An offloaded cell is the one a heal loop most needs to suspend, so only injection is refused."""
        provider = _StoppingWorkerProvider()
        controller = _make_cell_operations_controller(provider, probing_paused=True)

        await controller.stop_cell_between_weight_updates(_ROLLOUT_CELL_ID)

        assert provider.stopped_cells == [[_ROLLOUT_CELL_ID]]

    @pytest.mark.asyncio
    async def test_inject_fault_between_weight_updates_refuses_a_pause_that_began_while_it_waited(self):
        # [FUNCTION DOC] A background task takes the lock and later pauses probing under it. An
        # injection is started while the lock is held and then the holder is released. Asserts the
        # injection fails with a runtime error "refusing fault injection" (the pause is read after
        # the lock is taken) and the worker manager got no call.
        """Reading the pause before taking the lock would kill a cell the offload has since put to sleep."""
        provider = _StoppingWorkerProvider()
        controller = _make_cell_operations_controller(provider)
        server = controller.servers["default"]
        entered, may_finish = asyncio.Event(), asyncio.Event()

        async def _pause_probing_under_the_lock() -> None:
            # [FUNCTION DOC] HELPER: Under the lock, signals entry, waits for the release event,
            # then switches the server's activeness tracker off before releasing the lock.
            async with controller.context_lock:
                entered.set()
                await may_finish.wait()
                server.health_checker_activeness.bump_active(False)

        holder = asyncio.create_task(_pause_probing_under_the_lock())
        await entered.wait()
        injecting = asyncio.create_task(
            controller.inject_fault_between_weight_updates(_ROLLOUT_CELL_ID, mode=FailureMode.SIGKILL, sub_index=0)
        )
        await asyncio.sleep(0)
        may_finish.set()
        await holder

        with pytest.raises(RuntimeError, match="refusing fault injection"):
            await injecting

        assert provider._worker_manager_handle.inject_fault.calls == []

    @pytest.mark.asyncio
    async def test_a_refused_injection_leaves_the_weight_update_lock_free(self):
        # [FUNCTION DOC] With probing paused, a refused injection raises a runtime error; asserts
        # the lock is not held afterwards and a following stop-cell call succeeds and reaches the
        # provider.
        """A refusal that kept the lock would hang the next weight update instead of only skipping the injection."""
        provider = _StoppingWorkerProvider()
        controller = _make_cell_operations_controller(provider, probing_paused=True)

        with pytest.raises(RuntimeError, match="refusing fault injection"):
            await controller.inject_fault_between_weight_updates(
                _ROLLOUT_CELL_ID, mode=FailureMode.SIGKILL, sub_index=0
            )

        assert not controller.context_lock.locked

        await controller.stop_cell_between_weight_updates(_ROLLOUT_CELL_ID)

        assert provider.stopped_cells == [[_ROLLOUT_CELL_ID]]

    @pytest.mark.asyncio
    async def test_a_weight_update_cannot_start_while_a_suspension_is_still_running(self):
        # [FUNCTION DOC] The provider's stop blocks on a future. Starts a stop, waits until the
        # provider received it, then starts a task that takes the lock; asserts that task has not
        # entered. Completes the future, awaits both and asserts the lock taker then entered.
        """Releasing the lock before the provider has torn the cell down reopens the very race this serializes."""
        completion: asyncio.Future = asyncio.get_running_loop().create_future()
        provider = _StoppingWorkerProvider(completion=completion)
        controller = _make_cell_operations_controller(provider)
        weight_update_started = asyncio.Event()

        async def _start_weight_update() -> None:
            # [FUNCTION DOC] HELPER: Takes the context lock and marks the weight update as started.
            async with controller.context_lock:
                weight_update_started.set()

        stopping = asyncio.create_task(controller.stop_cell_between_weight_updates(_ROLLOUT_CELL_ID))
        await provider.stop_requested.wait()
        weight_update = asyncio.create_task(_start_weight_update())
        await asyncio.sleep(0)

        assert not weight_update_started.is_set()

        completion.set_result(None)
        await stopping
        await weight_update

        assert weight_update_started.is_set()


def _raise_configure_logger(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement for the logger configuration function; ignores its
    # arguments and always raises a value error "configure_logger blew up".
    raise ValueError("configure_logger blew up")


class TestInitRunsExactlyOnce:
    # [CLASS DOC] Tests the initialized flag and the refusal of a second init, using a real
    # controller in debug train-only mode (so init is cheap) unless stated otherwise.
    @staticmethod
    def _controller() -> InferenceController:
        # [FUNCTION DOC] HELPER: Builds a real controller with debug train-only arguments and empty
        # fake providers.
        return InferenceController(
            make_args(debug_train_only=True),
            engine_provider=_FakeWorkerProvider([]),
            router_providers=[_FakeWorkerProvider([])],
        )

    @pytest.mark.asyncio
    async def test_a_controller_that_never_ran_init_reports_itself_uninitialized(self):
        # [FUNCTION DOC] A fresh controller reports not initialized.
        """A restarted script asks the controller it found running whether to initialize it or to take it over."""
        assert await self._controller().is_initialized() is False

    @pytest.mark.asyncio
    async def test_a_controller_that_ran_init_reports_itself_initialized(self):
        # [FUNCTION DOC] After init, the controller reports initialized.
        """The take-over path has to see the controller the previous script built as built."""
        controller = self._controller()

        await controller.init()

        assert await controller.is_initialized() is True

    @pytest.mark.asyncio
    async def test_a_second_init_is_refused(self):
        # [FUNCTION DOC] After one init, a second init raises an assertion error mentioning a
        # stale worker.
        """Re-initializing a live controller would rebuild a fleet the previous script is still driving."""
        controller = self._controller()
        await controller.init()

        with pytest.raises(AssertionError, match="stale worker"):
            await controller.init()

    @pytest.mark.asyncio
    async def test_a_controller_whose_init_raised_reports_itself_uninitialized(self, monkeypatch):
        # [FUNCTION DOC] Makes the logger configuration raise a value error, expects init to
        # propagate it, and asserts the controller reports not initialized.
        """A controller with no watcher and no ticker must not be taken over as a built one."""
        controller = self._controller()
        monkeypatch.setattr(inference_controller_module, "configure_logger", _raise_configure_logger)

        with pytest.raises(ValueError, match="configure_logger blew up"):
            await controller.init()

        assert await controller.is_initialized() is False

    @pytest.mark.asyncio
    async def test_a_controller_whose_init_raised_still_refuses_a_second_init(self, monkeypatch):
        # [FUNCTION DOC] Makes the first init fail with a value error, undoes the patch, and
        # asserts a second init raises an assertion error about a stale worker.
        """A half-built controller cannot be rebuilt over its own state, so it has to fail loudly instead."""
        controller = self._controller()
        monkeypatch.setattr(inference_controller_module, "configure_logger", _raise_configure_logger)
        with pytest.raises(ValueError):
            await controller.init()
        monkeypatch.undo()

        with pytest.raises(AssertionError, match="stale worker"):
            await controller.init()

    @pytest.mark.asyncio
    async def test_a_second_init_is_refused_after_a_full_init(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Builds a non-debug controller with patched server creation, runs a full
        # init, asserts a second init raises an assertion error, then disposes.
        """The train-only shortcut returns early, so the refusal has to hold for a controller that built a fleet."""
        _patch_init(monkeypatch, servers={"default": _RecordingServer()})
        controller = InferenceController(
            make_args(), engine_provider=_FakeWorkerProvider([]), router_providers=[_FakeWorkerProvider([])]
        )
        await controller.init()

        with pytest.raises(AssertionError):
            await controller.init()

        await controller.dispose()


class TestWaitingForTheWholeFleet:
    # [CLASS DOC] Tests the controller's wait for the expected number of cells across servers.
    @pytest.mark.asyncio
    async def test_the_budget_the_caller_gives_reaches_every_server(self):
        # [FUNCTION DOC] Waits with a 12 second timeout over two servers and asserts each server
        # recorded exactly that timeout.
        """A take-over waits inside its own bounded gate, and a server ignoring that budget hangs the whole gate."""
        controller = _make_controller({"a": _RecordingServer(model_name="a"), "b": _RecordingServer(model_name="b")})

        await controller.wait_expected_num_cells(timeout=12.0)

        assert [srv.cells_timeouts for srv in controller.servers.values()] == [[12.0], [12.0]]

    @pytest.mark.asyncio
    async def test_a_caller_that_names_no_budget_gets_the_fleet_wide_one(self):
        # [FUNCTION DOC] Waits with no timeout and asserts the server received the module's
        # cells-ready timeout constant.
        """Init waits without naming a budget, and an unbounded wait there hangs the run with no diagnosis."""
        controller = _make_controller({"a": _RecordingServer(model_name="a")})

        await controller.wait_expected_num_cells()

        assert [srv.cells_timeouts for srv in controller.servers.values()] == [
            [inference_controller_module.CELLS_READY_TIMEOUT_SECONDS]
        ]


class TestEvalFleetSurface:
    # [CLASS DOC] Tests the controller's eval fleet surface: remote-procedure-call exposure,
    # wire serialization, and behavior with and without a fleet.
    def test_the_eval_fleet_is_an_rpc_method_rather_than_an_attribute(self):
        # [FUNCTION DOC] Builds an rpc handle for the controller class (no network use) and asserts
        # the get-eval-fleet-info and pin-eval-fleet methods are callable while asking for an
        # eval fleet method raises an attribute error saying there is no such rpc method.
        """A handle resolves rpc methods only, so reading the fleet off it reaches nothing."""
        handle = RpcWorkerHandle(InferenceController, server_url="http://10.0.0.1:1234")

        assert callable(handle.get_eval_fleet_info)
        assert callable(handle.pin_eval_fleet)
        with pytest.raises(AttributeError, match="no rpc method 'eval_fleet'"):
            handle.eval_fleet()

    def test_the_fleet_description_survives_the_wire(self):
        # [FUNCTION DOC] Takes the serializer of the get-eval-fleet-info rpc method and asserts a
        # fleet info (router address, GPU counts) and a missing value both survive an encode and
        # decode of the result.
        """The executor retargets its eval args to what it decodes, so every field must round-trip."""
        serializer = collect_rpc_method_specs(InferenceController)["get_eval_fleet_info"].serializer
        info = EvalFleetInfo(router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=2, num_gpus_per_engine=1)

        assert serializer.decode_result(serializer.encode_result(info)) == info
        assert serializer.decode_result(serializer.encode_result(None)) is None

    def test_a_pin_and_its_skip_reason_survive_the_wire(self):
        # [FUNCTION DOC] Takes the serializer of the pin-eval-fleet rpc method; asserts a query
        # (checkpoint directory, weight version) round-trips, and so do pin results with no skip
        # reason and with the reason unhealthy.
        """A skipped point must arrive as a skip with its reason, not as a remote crash."""
        spec = collect_rpc_method_specs(InferenceController)["pin_eval_fleet"]
        query = dict(checkpoint_dir="/snap/step_5", weight_version="5")

        assert spec.serializer.decode_query(spec.serializer.encode_query(query)) == query
        for pin in (EvalFleetPin(skip_reason=None), EvalFleetPin(skip_reason="unhealthy")):
            assert spec.serializer.decode_result(spec.serializer.encode_result(pin)) == pin

    @pytest.mark.asyncio
    async def test_a_run_without_a_fleet_answers_nothing_to_wire_up(self):
        # [FUNCTION DOC] With no eval fleet set, get-eval-fleet-info returns nothing.
        """--eval-num-gpus 0 deploys no fleet, and the executor must be told so rather than guess."""
        controller = _make_controller({})
        controller._eval_fleet = None

        assert await controller.get_eval_fleet_info() is None

    @pytest.mark.asyncio
    async def test_pinning_a_fleet_that_is_not_deployed_is_refused(self):
        # [FUNCTION DOC] With no eval fleet set, pin-eval-fleet raises an assertion error saying
        # there is no eval fleet.
        """Nobody can reach this call without a fleet, so answering a skip would hide a wiring bug."""
        controller = _make_controller({})
        controller._eval_fleet = None

        with pytest.raises(AssertionError, match="no eval fleet"):
            await controller.pin_eval_fleet(checkpoint_dir="/snap/step_5", weight_version="5")

    @pytest.mark.asyncio
    async def test_the_fleet_answers_and_pins_through_the_controller(self):
        # [FUNCTION DOC] Installs a recording fleet; asserts the controller's get-eval-fleet-info
        # returns the fleet's info, pin-eval-fleet returns a pin without skip reason, and the fleet
        # recorded the checkpoint directory and weight version.
        """The fleet lives beside its engines: the executor only ever addresses it through the controller."""
        controller = _make_controller({})
        info = EvalFleetInfo(router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=2, num_gpus_per_engine=1)
        controller._eval_fleet = _RecordingInferenceControllerEvalFleet(info)

        assert await controller.get_eval_fleet_info() == info
        assert await controller.pin_eval_fleet(checkpoint_dir="/snap/step_5", weight_version="5") == EvalFleetPin(
            skip_reason=None
        )
        assert controller._eval_fleet.pins == [dict(checkpoint_dir="/snap/step_5", weight_version="5")]


class TestCellsReadyIsScopedToTheTargetedModel:
    # [CLASS DOC] Tests scoping of the cells-ready wait and the server lookup by model id.
    @pytest.mark.asyncio
    async def test_a_named_model_does_not_wait_for_another_models_cells(self):
        # [FUNCTION DOC] Model a has a ready cell, model b has a cell that is not ready. Starting
        # an update for a succeeds right away and returns a's client only.
        """Different model ids are independent, so a sick engine of B must not stall A's weight update."""
        a = _RecordingServer(model_name="a", update_weights=True)
        a.api_clients = ["a-client"]
        a.server_cells = {
            "a-0": SimpleNamespace(
                is_pending_weights_or_serving=True,
                is_uninitialized=False,
                meta=SimpleNamespace(workers_hash="hash-a"),
            )
        }
        b = _RecordingServer(model_name="b", update_weights=True)
        b.server_cells = {
            "b-0": SimpleNamespace(
                is_pending_weights_or_serving=False,
                is_uninitialized=False,
                meta=SimpleNamespace(workers_hash="hash-b"),
            )
        }
        controller = _make_controller({"a": a, "b": b})

        updatable = await controller.start_update_weights(model_id="a")

        assert updatable.rollout_engines == ["a-client"]

    @pytest.mark.asyncio
    async def test_an_unnamed_update_still_waits_for_every_model(self):
        # [FUNCTION DOC] Under the lock, asserts the controller's server lookup returns the single
        # server for both no model id and the matching model id.
        # KNOWN ISSUE: the name says an unnamed update waits for every model, but the test only
        # checks the server lookup helper and never starts an update or a wait.
        """A single policy run has one server, so scoping must not change what it waits for."""
        srv = _RecordingServer(model_name="a", update_weights=True)
        controller = _make_controller({"a": srv})

        async with controller.context_lock:
            assert controller._get_servers_of_model_id(None) == [srv]
            assert controller._get_servers_of_model_id("a") == [srv]


def _registration_snapshot(*, model_id: str = "model-a", run_uuid: str = _RUN_UUID) -> RegistrationSnapshot:
    # [FUNCTION DOC] HELPER: Builds a registration snapshot from reporter west with sequence number
    # one and a single registered engine cell (cell and worker names prefixed with west, one
    # worker at 10.0.0.5 port 8000 on GPU zero, workers hash hash-1) whose meta names the given
    # model id. The run uuid defaults to the module constant.
    meta = dict(
        model_id=model_id,
        worker_type="regular",
        num_gpus_per_engine=1,
        gpu_offset=0,
        sglang_api_key=None,
        needs_offload=False,
        update_weights=True,
    )

    cell = RegisteredCellInfo(
        reporter_id="west",
        info=CellInfo(
            cell_id="west-inference-engine-0-0-00000",
            pool_id="west-inference-engine-0-0",
            alive=True,
            worker_names=["west-inference-engine-0-0-00000-00000"],
            workers_hash="hash-1",
            meta=meta,
        ),
        workers=[
            WorkerInfo(
                name="west-inference-engine-0-0-00000-00000",
                generation=0,
                self_addrs={"primary": HostAndPort(host="10.0.0.5", port=8000)},
                gpu_ids=[0],
            )
        ],
    )
    return RegistrationSnapshot(
        run_uuid=run_uuid,
        reporter_id="west",
        sequence_number=1,
        cells=[cell],
    )


class TestRegistrationSnapshotEndpoint:
    # [CLASS DOC] Tests the controller's registration-ingest endpoint used by engines of another
    # deployment to join a run through a registration hub.
    def test_a_snapshot_survives_the_wire(self):
        # [FUNCTION DOC] Asserts a query holding a registration snapshot round-trips through the
        # registration-ingest rpc serializer.
        """The reporter is in another cluster, so the whole membership has to be wire typed both ways."""
        spec = collect_rpc_method_specs(InferenceController)["registration_ingest"]
        query = dict(snapshot=_registration_snapshot())

        assert spec.serializer.decode_query(spec.serializer.encode_query(query)) == query

    @pytest.mark.asyncio
    async def test_a_run_holding_a_registry_takes_the_snapshot_in(self):
        # [FUNCTION DOC] Controller with a registration hub as engine source and a model-a server;
        # ingests a snapshot and asserts the hub now knows the snapshot's cell id.
        """This endpoint is the only way an engine of another deployment ever joins the run."""
        registry = RegistrationHub(run_uuid=_RUN_UUID)
        controller = _make_controller({"model-a": _RecordingServer(model_name="model-a")}, registration_hub=registry)

        await controller.registration_ingest(snapshot=_registration_snapshot())

        assert sorted(registry._cell_of_id) == ["west-inference-engine-0-0-00000"]

    @pytest.mark.asyncio
    async def test_a_controller_the_script_has_not_initialized_yet_says_it_is_not_ready(self):
        # [FUNCTION DOC] With a hub but no servers, ingesting raises an assertion error saying the
        # controller is not ready, and the hub stays empty.
        """Until init this controller knows no model of the run, and blaming the reporter for that misleads."""
        registry = RegistrationHub(run_uuid=_RUN_UUID)
        controller = _make_controller({}, registration_hub=registry)

        with pytest.raises(AssertionError, match="not ready"):
            await controller.registration_ingest(snapshot=_registration_snapshot())

        assert registry._cell_of_id == {}

    @pytest.mark.asyncio
    async def test_a_run_serving_engines_of_its_own_refuses_a_snapshot(self):
        # [FUNCTION DOC] With a server but a plain provider instead of a hub, ingesting raises an
        # assertion error saying the run takes no registration.
        """It would take the cells in and never wait for them, so the reporter has to hear that it is unwanted."""
        controller = _make_controller({"model-a": _RecordingServer(model_name="model-a")})

        with pytest.raises(AssertionError, match="takes no registration"):
            await controller.registration_ingest(snapshot=_registration_snapshot())

    @pytest.mark.asyncio
    async def test_a_cell_serving_a_model_this_run_does_not_serve_is_refused(self):
        # [FUNCTION DOC] With a hub and a model-a server, ingesting a snapshot whose cell serves
        # model other raises an assertion error saying the run does not serve model other.
        """No router of this run would ever send it a request, so counting it would stall the wait for cells."""
        registry = RegistrationHub(run_uuid=_RUN_UUID)
        controller = _make_controller({"model-a": _RecordingServer(model_name="model-a")}, registration_hub=registry)

        with pytest.raises(AssertionError, match="serves model 'other'"):
            await controller.registration_ingest(snapshot=_registration_snapshot(model_id="other"))
