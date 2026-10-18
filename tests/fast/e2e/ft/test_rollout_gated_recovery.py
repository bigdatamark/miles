# [MODULE DOC]
# Asynchronous tests of rollout-engine fault recovery in a colocated two-engine deployment, built
# from the real inference controller, rollout server, server cells, cell handler, and mini fault
# tolerance controller, with only the outside world faked: the worker manager (cell start/stop and
# info), the sglang router client, per-cell health checkers, the engine api client, the launch gate
# and the server health probe. A harness object wires everything, can open a weight-update window,
# crash a cell, run one mini-controller healing poll, and observe the api-server-shaped cell
# objects. The tests assert phases (Running, Suspended, Pending), the Serving condition, router
# membership, gate activation, and that the e2e recovery witness accepts or rejects the observed
# sequences.
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import pytest
from tests.e2e.ft.conftest_ft.fault_injection import state, views
from tests.fast.e2e.ft.fault_injection.utils import note_injected
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import inference_controller as inference_controller_module
from miles.ray.rollout import rollout_server as rollout_server_module
from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import CellAddrInfo
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.ft_utils.mini_ft_controller import _compute_cell_snapshot, _MiniFTController
from miles.utils.workers.cell_operations.ray import RayCellOperations
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import NamedHostAndPorts

_POOL_ID = "inference-engine-0"
_CELL_IDS = ["inference-engine-0-0-0", "inference-engine-0-0-1"]


class _StubProvider:
    # [CLASS DOC] HELPER: Engine provider stub whose address lookup must never be used.
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] HELPER: Always raises an assertion error, because the tests patch address
        # computation on server cells instead.
        raise AssertionError(f"this module addresses cells through a patched _compute_addr_info ({worker_name=})")


class _FakeRouter:
    # [CLASS DOC] HELPER: Fake sglang router client that keeps the ordered list of registered worker urls.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with no registered worker urls.
        self.worker_urls: list[str] = []

    async def add_worker(
        self, *, worker_url: str, worker_type: str, use_legacy_api: bool, bootstrap_port: int | None
    ) -> None:
        # [FUNCTION DOC] HELPER: Appends the worker url (other arguments ignored; duplicates are not prevented).
        self.worker_urls.append(worker_url)

    async def remove_worker(self, *, worker_url: str, use_legacy_api: bool) -> None:
        # [FUNCTION DOC] HELPER: Removes the first occurrence of the url; raises a value error if absent.
        self.worker_urls.remove(worker_url)


class _FakeHealthChecker:
    # [CLASS DOC] HELPER: Fake cell health checker with a settable tri-state status (default true)
    # and a started flag.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Status starts true, not started.
        self.status: TriState = TriState.TRUE
        self.started: bool = False

    def start(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the checker as started.
        self.started = True

    def stop(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the checker as stopped.
        self.started = False


class _FakeEngineApiClient:
    # [CLASS DOC] HELPER: Fake sglang engine api client: stores its url and key, makes memory
    # release/resume no-ops, and returns fixed server info (a model path and one internal state
    # with rank zero environment variables).
    def __init__(self, server_url: str, api_key: str | None = None) -> None:
        # [FUNCTION DOC] HELPER: Stores the server url and optional api key.
        self.server_url = server_url
        self.api_key = api_key

    async def release_memory_occupation(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC] HELPER: No-op.
        return None

    async def resume_memory_occupation(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC] HELPER: No-op.
        return None

    async def get_server_info(self) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Returns a fixed mapping with a fake model path and one internal state with RANK 0.
        return {"model_path": "/fake/model", "internal_states": [{"env_vars": {"RANK": "0"}}]}


class _RemoteMethod:
    # [CLASS DOC] HELPER: Wraps a function so it can be invoked through a remote-call method like a Ray actor method.
    def __init__(self, fn: Callable[..., Any]) -> None:
        # [FUNCTION DOC] HELPER: Stores the wrapped function.
        self._fn = fn

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Calls the wrapped function directly with the arguments and returns
        # its result (for async functions, a coroutine to await).
        return self._fn(*args, **kwargs)


class _FakeWorkerManager:
    # [CLASS DOC] HELPER: Fake worker manager for the two fixed cell ids. Tracks per-cell alive flags
    # and generation counters (the generation changes the workers hash), builds cell info objects
    # for one pool, and notifies a reconcile callback whenever a cell arrives, stops (info absent)
    # or starts again (new generation). Exposes remote-style methods for cell infos, stop and start.
    def __init__(self, *, cell_ids: list[str], reconcile: Callable[[str, CellInfo | None], Any]) -> None:
        # [FUNCTION DOC] HELPER: Marks all cells alive at generation zero, stores the reconcile
        # callback, and creates remote-style wrappers for get-cell-infos, stop-cells, start-cells.
        self._alive: dict[str, bool] = {cell_id: True for cell_id in cell_ids}
        self._generation: dict[str, int] = {cell_id: 0 for cell_id in cell_ids}
        self._reconcile = reconcile
        self.get_cell_infos = _RemoteMethod(self._get_cell_infos)
        self.stop_cells = _RemoteMethod(self._stop_cells)
        self.start_cells = _RemoteMethod(self._start_cells)

    def cell_info(self, cell_id: str) -> CellInfo:
        # [FUNCTION DOC] HELPER: Builds the cell info of a known cell: pool id, alive flag, one
        # worker named after the cell, a workers hash made from the cell id and current generation,
        # and metadata for a regular default-model engine with one gpu, gpu offset equal to the cell's
        # position, no api key, offload needed and weight update enabled. Raises a value error for
        # an unknown cell id.
        index = _CELL_IDS.index(cell_id)
        return CellInfo(
            cell_id=cell_id,
            pool_id=_POOL_ID,
            alive=self._alive[cell_id],
            worker_names=[f"{cell_id}-0"],
            workers_hash=f"hash-{cell_id}-{self._generation[cell_id]}",
            meta={
                "model_id": "default",
                "worker_type": "regular",
                "num_gpus_per_engine": 1,
                "gpu_offset": index,
                "sglang_api_key": None,
                "needs_offload": True,
                "update_weights": True,
            },
        )

    async def arrive_all(self) -> None:
        # [FUNCTION DOC] HELPER: Reports every cell's current info to the reconcile callback in order.
        for cell_id in _CELL_IDS:
            await self._reconcile(cell_id, self.cell_info(cell_id))

    async def _get_cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        # [FUNCTION DOC] HELPER: Returns a mapping from cell id to current info for all cells; the pool ids are ignored.
        return {cell_id: self.cell_info(cell_id) for cell_id in _CELL_IDS}

    async def _stop_cells(self, cell_ids: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Marks each cell not alive and tells the reconcile callback the cell is gone.
        for cell_id in cell_ids:
            self._alive[cell_id] = False
            await self._reconcile(cell_id, None)

    async def _start_cells(self, cell_ids: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Marks each cell alive, increments its generation (new workers
        # hash), and reports its new info to the reconcile callback.
        for cell_id in cell_ids:
            self._alive[cell_id] = True
            self._generation[cell_id] += 1
            await self._reconcile(cell_id, self.cell_info(cell_id))


class _FakeStopCellController:
    # [CLASS DOC] HELPER: Fake inference controller exposing only stop-between-weight-updates, delegating to the fake worker manager.
    def __init__(self, *, worker_manager: _FakeWorkerManager) -> None:
        # [FUNCTION DOC] HELPER: Stores the worker manager.
        self._worker_manager = worker_manager

    async def stop_cell_between_weight_updates(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Stops the single cell through the worker manager's remote-style stop method.
        await self._worker_manager.stop_cells.remote([cell_id])


class _Harness:
    # [CLASS DOC] HELPER: Wiring of a real inference controller, rollout server, cell handler and
    # mini ft controller around fakes. Public pieces: router, health checkers keyed by cell id,
    # activated gate urls, args, controller, worker manager, handler, ft controller. Methods:
    # observe cells, open a weight update window, crash a cell, run one healing poll.
    def __init__(self, *, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] HELPER: Patches the server cell module (gate activation, health probe
        # that always reports healthy, engine api client, health checker factory that stores fakes
        # keyed by the cell id without the name prefix), the cell address computation (per-index
        # fake host, server port 30000, gate port 13000, no bootstrap port), the rollout server's
        # router client, and the controller's ready-poll interval (zero). Then builds test args
        # (colocated, fault tolerance on for rollout), a controller created without running its
        # constructor and populated with the attributes it needs, one default rollout server
        # expecting both cells, the fake worker manager reconciling into the controller, a cell
        # handler over Ray cell operations, and a mini ft controller with zero poll interval and delay.
        self.router = _FakeRouter()
        self.health_checkers: dict[str, _FakeHealthChecker] = {}
        self.activated_gate_urls: list[str] = []

        async def _activate(gate_url: str) -> None:
            # [FUNCTION DOC] HELPER: Records the gate url that was activated.
            self.activated_gate_urls.append(gate_url)

        async def _probe(server_url: str, api_key: str | None, timeout: float = 5.0) -> bool:
            # [FUNCTION DOC] HELPER: Reports every server as healthy.
            return True

        async def _compute_addr_info(cell: ServerCell) -> CellAddrInfo:
            # [FUNCTION DOC] HELPER: Derives fake server and gate urls from the cell's index among
            # the known cell ids.
            index = _CELL_IDS.index(cell.meta.cell_id)
            return CellAddrInfo(
                server_url=f"http://10.0.0.{index}:30000",
                bootstrap_port=None,
                gate_url=f"http://10.0.0.{index}:13000",
            )

        def _create_health_checker(
            *, args: Any, name: str, get_api_client: Any, get_activeness: Any
        ) -> _FakeHealthChecker:
            # [FUNCTION DOC] HELPER: Creates a fake checker and stores it under the cell name with
            # the rollout-cell prefix removed; returns it.
            checker = _FakeHealthChecker()
            self.health_checkers[name.removeprefix("rollout-cell-")] = checker
            return checker

        monkeypatch.setattr(server_cell_module, "activate_launch_gate", _activate)
        monkeypatch.setattr(server_cell_module, "probe_server_healthy", _probe)
        monkeypatch.setattr(server_cell_module, "SGLangApiClient", _FakeEngineApiClient)
        monkeypatch.setattr(server_cell_module, "create_rollout_cell_health_checker", _create_health_checker)
        monkeypatch.setattr(ServerCell, "_compute_addr_info", _compute_addr_info)
        monkeypatch.setattr(rollout_server_module, "SGLangRouterApiClient", lambda router_url: self.router)
        monkeypatch.setattr(inference_controller_module, "CELLS_READY_POLL_INTERVAL_SECONDS", 0.0)

        self.args = make_args(colocate=True, use_fault_tolerance=True, ft_components=["rollout"])
        self.controller = InferenceController.__new__(InferenceController)
        self.controller.args = self.args
        self.controller.context_lock = ContextLock("InferenceController")
        self.controller._watcher_disposers = []
        self.controller._ticker = None
        self.controller._health_checker_activeness = ActivenessTracker(active=True)
        self.controller.servers = {
            "default": RolloutServer(
                server_cells={},
                args=self.args,
                context_lock=self.controller.context_lock,
                engine_provider=_StubProvider(),
                router_ip="10.0.0.9",
                router_port=20000,
                model_name="default",
                update_weights=True,
                init_expected_num_cells=len(_CELL_IDS),
            )
        }
        self.worker_manager = _FakeWorkerManager(cell_ids=_CELL_IDS, reconcile=self.controller._reconcile)
        self.handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=self.worker_manager,
                resolve_inference_controller=lambda: _FakeStopCellController(worker_manager=self.worker_manager),
            ),
            controllers=[self.controller],
            pool_ids=[_POOL_ID],
        )
        self.ft_controller = _MiniFTController(
            get_cells=self._get_cell_snapshots,
            suspend_cell=self.handler.suspend,
            resume_cell=self.handler.resume,
            poll_interval=0.0,
            resume_delay=0.0,
        )

    async def observe(self) -> dict[str, dict]:
        # [FUNCTION DOC] HELPER: Lists the cells through the handler and returns a mapping from
        # cell name to the cell serialized as a json-compatible dictionary.
        cells = await self.handler.list_cells()
        return {cell.metadata.name: cell.model_dump(mode="json") for cell in cells}

    async def open_weight_update_window(self, *, mark_weights_ready: bool = True) -> None:
        # [FUNCTION DOC] HELPER: While a background task keeps ticking the controller's cells,
        # starts a weight update (waiting at most 5 seconds) and then ends it, passing the
        # snapshot of cell ids to hashes when weights are to be marked ready, or an empty mapping
        # otherwise. Always cancels the ticking task afterwards and swallows its cancellation.
        tick_task = asyncio.create_task(self._tick_forever())
        try:
            engines = await asyncio.wait_for(self.controller.start_update_weights(), timeout=5)
            await self.controller.end_update_weights(engines.snapshot_cell_id_to_hashes if mark_weights_ready else {})
        finally:
            tick_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tick_task

    def crash(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Makes the cell's fake health checker report a false status, so
        # the ft controller considers the cell failed.
        self.health_checkers[cell_id].status = TriState.FALSE

    async def run_ft_controller_once(self) -> None:
        # [FUNCTION DOC] HELPER: Runs one poll-and-heal iteration of the mini ft controller.
        await self.ft_controller._poll_and_heal()

    async def _get_cell_snapshots(self) -> list[Any]:
        # [FUNCTION DOC] HELPER: Lists cells through the handler and converts each to the snapshot
        # form the mini ft controller consumes.
        return [_compute_cell_snapshot(cell) for cell in await self.handler.list_cells()]

    async def _tick_forever(self) -> None:
        # [FUNCTION DOC] HELPER: Loops forever, ticking the controller's cells and yielding to the
        # event loop once per iteration; ends only when cancelled.
        while True:
            await self.controller._tick_cells()
            await asyncio.sleep(0)


@pytest.fixture
async def harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    # [FUNCTION DOC] HELPER: Async fixture that builds the harness, has all cells arrive, and opens
    # one weight update window with weights marked ready, returning the harness.
    """A colocated 2-engine rollout deployment wired to a fake worker manager, up and serving."""
    harness = _Harness(monkeypatch=monkeypatch)
    await harness.worker_manager.arrive_all()
    await harness.open_weight_update_window()
    return harness


def _serving_condition(cell: dict) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the status string of the cell's Serving condition, or none if
    # the cell has no such condition.
    return next((cond["status"] for cond in cell["status"]["conditions"] if cond["type"] == "Serving"), None)


async def test_a_healthy_deployment_reports_every_engine_as_serving(harness: _Harness) -> None:
    # [FUNCTION DOC] Observes the healthy harness; asserts both cells are Running with a true
    # Serving condition and the router holds both engine urls.
    """The soak's terminal criterion is Serving, so a fault-free deployment must report exactly that."""
    cells = await harness.observe()

    assert [cell["status"]["phase"] for cell in cells.values()] == ["Running", "Running"]
    assert [_serving_condition(cell) for cell in cells.values()] == ["True", "True"]
    assert sorted(harness.router.worker_urls) == ["http://10.0.0.0:30000", "http://10.0.0.1:30000"]


async def test_a_crashed_engine_is_suspended_and_withdrawn_from_the_router(harness: _Harness) -> None:
    # [FUNCTION DOC] Crashes the first cell and suspends it through the handler; asserts its phase
    # is Suspended and the router holds only the second engine url.
    """Healing starts by tearing the engine down; leaving it registered would keep routing at a dead engine."""
    harness.crash(_CELL_IDS[0])

    await harness.handler.suspend(_CELL_IDS[0])
    cells = await harness.observe()

    assert cells[_CELL_IDS[0]]["status"]["phase"] == "Suspended"
    assert harness.router.worker_urls == ["http://10.0.0.1:30000"]


async def test_the_mini_ft_controller_heals_exactly_the_crashed_engine(harness: _Harness) -> None:
    # [FUNCTION DOC] Crashes the first cell and runs one healing poll; asserts the first cell is
    # Pending, the second still Running, and the router holds only the second engine url.
    """A heal that also cycled the healthy engines would take the whole deployment down at once."""
    harness.crash(_CELL_IDS[0])

    await harness.run_ft_controller_once()
    cells = await harness.observe()

    assert cells[_CELL_IDS[0]]["status"]["phase"] == "Pending"
    assert cells[_CELL_IDS[1]]["status"]["phase"] == "Running"
    assert harness.router.worker_urls == ["http://10.0.0.1:30000"]


async def test_the_replacement_engine_stays_gated_until_the_next_window(harness: _Harness) -> None:
    # [FUNCTION DOC] Crashes the first cell, remembers the activated gates, runs one healing poll;
    # asserts the first server cell is uninitialized, no new gate was activated, and its phase is Pending.
    """Under colocation the trainer owns the gpus mid-step, so the relaunched engine must not claim memory yet."""
    harness.crash(_CELL_IDS[0])
    gates_before = list(harness.activated_gate_urls)

    await harness.run_ft_controller_once()
    cells = await harness.observe()

    assert harness.controller.servers["default"].server_cells[_CELL_IDS[0]].is_uninitialized
    assert harness.activated_gate_urls == gates_before
    assert cells[_CELL_IDS[0]]["status"]["phase"] == "Pending"


async def test_the_replacement_engine_is_not_serving_before_it_gets_weights(harness: _Harness) -> None:
    # [FUNCTION DOC] Crashes and heals the first cell, then opens a weight update window without
    # marking weights ready; asserts the cell is Running with a false Serving condition and still
    # absent from the router.
    """Regression: a relaunched engine holding stale weights reads Running, which alone proves no recovery."""
    harness.crash(_CELL_IDS[0])
    await harness.run_ft_controller_once()

    await harness.open_weight_update_window(mark_weights_ready=False)
    cells = await harness.observe()

    assert cells[_CELL_IDS[0]]["status"]["phase"] == "Running"
    assert _serving_condition(cells[_CELL_IDS[0]]) == "False"
    assert harness.router.worker_urls == ["http://10.0.0.1:30000"]


async def test_the_next_window_puts_the_replacement_engine_back_in_the_router(harness: _Harness) -> None:
    # [FUNCTION DOC] Crashes and heals the first cell, then opens a normal weight update window;
    # asserts the cell serves again and the router holds both engine urls.
    """This is the recovery the soak asserts: the replaced engine ends up serving again."""
    harness.crash(_CELL_IDS[0])
    await harness.run_ft_controller_once()

    await harness.open_weight_update_window()
    cells = await harness.observe()

    assert _serving_condition(cells[_CELL_IDS[0]]) == "True"
    assert sorted(harness.router.worker_urls) == ["http://10.0.0.0:30000", "http://10.0.0.1:30000"]


async def test_the_observed_sequence_satisfies_the_soak_recovery_witness(harness: _Harness) -> None:
    # [FUNCTION DOC] Records into an event log: an observation of the healthy cells, an injection
    # note on the first cell, the crash and heal poll observation, then the post-window
    # observation. Asserts the e2e views report no cells failing to serve after injection, with zero grace.
    """The fast-layer stand-in is only worth anything if the e2e witness accepts the sequence it produces."""
    log = state.EventLog()
    log.observe(list((await harness.observe()).values()))
    note_injected(log, _CELL_IDS[0])
    harness.crash(_CELL_IDS[0])

    await harness.run_ft_controller_once()
    log.observe(list((await harness.observe()).values()))
    await harness.open_weight_update_window()
    log.observe(list((await harness.observe()).values()))

    assert views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0) == {}


async def test_the_witness_rejects_a_replacement_that_never_reaches_the_router(harness: _Harness) -> None:
    # [FUNCTION DOC] Same sequence as the accepting test but the weight window does not mark
    # weights ready; asserts the offender set is exactly the first cell id.
    """A weight update that silently skips the replaced cell leaves it Running forever, and must fail the soak."""
    log = state.EventLog()
    log.observe(list((await harness.observe()).values()))
    note_injected(log, _CELL_IDS[0])
    harness.crash(_CELL_IDS[0])

    await harness.run_ft_controller_once()
    log.observe(list((await harness.observe()).values()))
    await harness.open_weight_update_window(mark_weights_ready=False)
    log.observe(list((await harness.observe()).values()))

    offenders = views.compute_cells_not_serving_after_injection(log.events, cell_type="rollout", grace_seconds=0.0)
    assert set(offenders) == {_CELL_IDS[0]}
