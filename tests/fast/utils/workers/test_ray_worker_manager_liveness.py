# [MODULE DOC] Tests for the per-cell liveness scan of RayWorkerManager, run on the fake ray
# cluster. A scan probes every worker of a live cell with a readiness method; a proven death (the
# actor is gone or the probe returned a task error) stops the whole cell so the membership drops
# it and it can be started again, while a probe that merely does not answer in time is not
# treated as death. Also covers races with stop/start, the self-rescheduling scan loop (one scan
# task per cell generation, surviving scan failures) and propagation of a self-death to a ray
# worker provider watching the manager through a small shim.
from __future__ import annotations

import asyncio

import pytest
import ray
from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.fake_ray import EVENT_KILL, READINESS_METHOD, FakeRayCluster

from miles.utils.workers import ray_worker_manager
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec


class DemoWorker:
    # [CLASS DOC] HELPER: Trivial serve worker class referenced by import path in the specs; stores
    # its keyword arguments.
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


_WORKER_CLASS_PATH = f"{DemoWorker.__module__}.{DemoWorker.__qualname__}"


def _make_spec(name: str, *, num_cells: int = 1, num_workers_per_cell: int = 1) -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a serve worker spec with the given cell shape, no gpus, one
    # master-mode dynamic port, empty environment and constructor arguments, using DemoWorker.
    return ServeWorkerSpec(
        name=name,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master", allow_dynamic=True)],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=num_cells, num_workers_per_cell=num_workers_per_cell, num_gpus_per_worker=0
        ),
        worker_class=_WORKER_CLASS_PATH,
        ctor_kwargs=lambda _ctx: {},
    )


async def _launch(specs: list[ServeWorkerSpec], *, comm_backend: WorkerCommBackend) -> RayWorkerManager:
    # [FUNCTION DOC] HELPER: Creates a manager and initialises it with the specs, no placement
    # groups and the given communication backend; returns it.
    manager = RayWorkerManager()
    await manager.init(worker_manager_args(), specs, {}, comm_backend=comm_backend)
    return manager


async def _scan_all_live_cells(manager: RayWorkerManager) -> None:
    # [FUNCTION DOC] HELPER: Runs one liveness scan on every cell currently alive, one after
    # another, awaiting each.
    for cell in manager._all_cells():
        if cell.alive:
            await cell._scan_liveness_once()


def _kill_worker_process(cluster: FakeRayCluster, *, handle_index: int) -> None:
    # [FUNCTION DOC] HELPER: Simulates a dead worker process by making the readiness method of the
    # fake handle at that index raise a ray actor error.
    cluster.handles[handle_index].failing_methods[READINESS_METHOD] = ray.exceptions.RayActorError()


@pytest.fixture
def instant_scans(monkeypatch) -> None:
    # [FUNCTION DOC] HELPER: Fixture setting the manager module's liveness scan interval to zero
    # seconds for the test, so the background loop rescans immediately.
    monkeypatch.setattr(ray_worker_manager, "_LIVENESS_SCAN_INTERVAL_SECONDS", 0.0)


class TestScanLivenessOnce:
    # [CLASS DOC] Tests of a single liveness scan pass over live cells (rpc and ray wires).
    async def test_a_cell_whose_workers_all_answer_stays_alive(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Two healthy cells; after a scan all cells are still alive.
        """The scan must not tear down a healthy cell, or every run would restart itself forever."""
        manager = await _launch([_make_spec("engine", num_cells=2)], comm_backend=WorkerCommBackend.RPC)

        await _scan_all_live_cells(manager)

        assert all(info.alive for info in manager.get_cell_infos(pool_ids=["engine"]).values())

    async def test_a_cell_that_lost_a_worker_stops_being_alive(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Worker 0 (cell 0) is made dead; after a scan cell 0 is not alive and cell 1
        # is alive.
        """A worker that exits on its own must reach the membership, or nobody ever replaces it."""
        manager = await _launch([_make_spec("engine", num_cells=2)], comm_backend=WorkerCommBackend.RPC)
        _kill_worker_process(fake_ray_cluster, handle_index=0)

        await _scan_all_live_cells(manager)

        infos = manager.get_cell_infos(pool_ids=["engine"])
        assert not infos["engine-00000"].alive
        assert infos["engine-00001"].alive

    async def test_the_whole_cell_goes_when_one_of_its_workers_dies(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] In a 2-worker cell worker 1 is made dead; after a scan the cell is not
        # alive and both actors were killed.
        """A cell is the unit of recovery, so a surviving sibling of a dead rank must be reclaimed too."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)], comm_backend=WorkerCommBackend.RPC)
        _kill_worker_process(fake_ray_cluster, handle_index=1)

        await _scan_all_live_cells(manager)

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive
        assert [handle.killed for handle in fake_ray_cluster.handles] == [True, True]

    async def test_a_dropped_cell_can_be_started_again(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] After a scan drops the cell, a start brings it alive with a second actor
        # created (2 handles in total), without a stop in between.
        """Reporting the death is only useful if the cell is then restartable without a stop first."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        _kill_worker_process(fake_ray_cluster, handle_index=0)
        await _scan_all_live_cells(manager)

        await manager.start_cells(["engine-00000"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive
        assert len(fake_ray_cluster.handles) == 2

    async def test_a_restarted_cell_reports_a_new_workers_hash(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Scan-drop then start; the cell's workers hash differs from the original.
        """The consumer rebuilds its handles off the hash, so a self-death must move it as a stop does."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        before = manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].workers_hash
        _kill_worker_process(fake_ray_cluster, handle_index=0)

        await _scan_all_live_cells(manager)
        await manager.start_cells(["engine-00000"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].workers_hash != before

    async def test_a_ray_wire_cell_is_scanned_the_same_way(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Same as the death test but with the ray communication backend; the cell is
        # dropped.
        """Liveness is a property of the actor process, not of the wire its methods travel on."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RAY)
        _kill_worker_process(fake_ray_cluster, handle_index=0)

        await _scan_all_live_cells(manager)

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive

    async def test_a_stopped_cell_probes_nothing(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] After stopping a cell and clearing the call log, a scan of that cell makes
        # no readiness call and does not raise.
        """A scan racing a suspend finds no actors to probe, and must answer that instead of raising."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        cell = manager._find_cell("engine-00000")
        await manager.stop_cells(["engine-00000"])
        fake_ray_cluster.calls.clear()

        await cell._scan_liveness_once()

        assert fake_ray_cluster.calls_of(READINESS_METHOD) == []


class TestScanLivenessRacesWithMembershipChanges:
    # [CLASS DOC] Tests of a scan overlapping a stop or restart; the readiness probe of a dead
    # worker is made to hang for 0.2 seconds so the race window is open.
    async def test_a_cell_stopped_while_it_is_probed_is_not_stopped_twice(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Starts a scan as a task, stops the cell 50 ms later, awaits the scan;
        # asserts the cell is not alive and the kill happened exactly once.
        """A suspend landing mid-probe already killed the actors the scan was about to declare dead."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        cell = manager._find_cell("engine-00000")
        _kill_worker_process(fake_ray_cluster, handle_index=0)
        fake_ray_cluster.handles[0].hanging_methods[READINESS_METHOD] = 0.2

        scan = asyncio.create_task(cell._scan_liveness_once())
        await asyncio.sleep(0.05)
        await manager.stop_cells(["engine-00000"])
        await scan

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive
        assert fake_ray_cluster.events.count(EVENT_KILL) == 1

    async def test_a_cell_restarted_while_being_probed_survives(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Same race but the cell is stopped and started during the probe; after the
        # scan finishes the new generation is still alive.
        """The dead workers the scan saw belong to the old generation, so the new one must not pay for them."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        cell = manager._find_cell("engine-00000")
        _kill_worker_process(fake_ray_cluster, handle_index=0)
        fake_ray_cluster.handles[0].hanging_methods[READINESS_METHOD] = 0.2

        scan = asyncio.create_task(cell._scan_liveness_once())
        await asyncio.sleep(0.05)
        await manager.stop_cells(["engine-00000"])
        await manager.start_cells(["engine-00000"])
        await scan

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive


class TestScanLivenessOnlyTrustsAProvenDeath:
    # [CLASS DOC] Tests of which probe outcomes count as death.
    async def test_a_worker_that_does_not_answer_in_time_is_kept(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The readiness probe hangs for an hour; after a scan the cell is still
        # alive.
        """A busy worker must not be declared dead, or a slow train step would kill its own cell."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        fake_ray_cluster.handles[0].hanging_methods[READINESS_METHOD] = 3600.0

        await _scan_all_live_cells(manager)

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive

    async def test_an_application_error_from_the_probe_is_treated_as_death(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The probe raises a ray task error (built without its constructor); after a
        # scan the cell is not alive.
        """A worker answering its readiness probe with a task error is as unusable as a missing one."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        fake_ray_cluster.handles[0].failing_methods[READINESS_METHOD] = ray.exceptions.RayTaskError.__new__(
            ray.exceptions.RayTaskError
        )

        await _scan_all_live_cells(manager)

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive


class TestScanLivenessLoop:
    # [CLASS DOC] Tests of the background scan loop that each launched cell runs on its own; uses
    # the instant_scans fixture so the loop spins without waiting.
    async def test_a_launched_cell_scans_itself(self, fake_ray_cluster: FakeRayCluster, instant_scans: None):
        # [FUNCTION DOC] Makes the worker dead without any explicit scan and waits until the cell
        # reports not alive.
        """The scan is what makes a death nobody reported reach the membership, so it must run unasked."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        _kill_worker_process(fake_ray_cluster, handle_index=0)

        await _wait_until(lambda: not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive)

    async def test_each_cell_scans_on_its_own(self, fake_ray_cluster: FakeRayCluster, instant_scans: None):
        # [FUNCTION DOC] Cell 0's probe hangs and cell 1's worker is dead; waits until cell 1 is not
        # alive and asserts cell 0 is still alive.
        """One worker hanging its probe must not delay how fast another cell's death is noticed."""
        manager = await _launch([_make_spec("engine", num_cells=2)], comm_backend=WorkerCommBackend.RPC)
        fake_ray_cluster.handles[0].hanging_methods[READINESS_METHOD] = 3600.0
        _kill_worker_process(fake_ray_cluster, handle_index=1)

        await _wait_until(lambda: not manager.get_cell_infos(pool_ids=["engine"])["engine-00001"].alive)

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive

    async def test_a_stopped_cell_stops_scanning(self, fake_ray_cluster: FakeRayCluster, instant_scans: None):
        # [FUNCTION DOC] After stopping the cell, waits until its liveness scan task is done.
        """A suspended cell that kept scanning would keep a task per suspend for the rest of the run."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        cell = manager._find_cell("engine-00000")

        await manager.stop_cells(["engine-00000"])

        await _wait_until(lambda: cell.liveness_scan_task.done())

    async def test_a_restarted_cell_leaves_only_its_new_scan_running(
        self, fake_ray_cluster: FakeRayCluster, instant_scans: None
    ):
        # [FUNCTION DOC] Remembers the first scan task, stops and starts the cell, waits for the
        # first task to finish and asserts the cell's current scan task is still running.
        """A restart that stacked one more scan per generation would probe the same actors N times."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        first_scan = manager._find_cell("engine-00000").liveness_scan_task

        await manager.stop_cells(["engine-00000"])
        await manager.start_cells(["engine-00000"])

        await _wait_until(lambda: first_scan.done())
        assert not manager._find_cell("engine-00000").liveness_scan_task.done()

    async def test_a_failing_scan_does_not_end_the_loop(self, fake_ray_cluster: FakeRayCluster, instant_scans: None):
        # [FUNCTION DOC] Replaces the cell's single-scan method with one that raises on its first
        # call; waits until at least 3 scans ran and asserts the loop task is not done.
        """One bad scan must not silently leave the cell without any liveness reporting at all."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        cell = manager._find_cell("engine-00000")
        scans: list[int] = []

        async def scan_once() -> None:
            # [FUNCTION DOC] HELPER: Counts the call and raises a runtime error on the first one.
            scans.append(len(scans))
            if len(scans) == 1:
                raise RuntimeError("scan failed")

        cell._scan_liveness_once = scan_once

        await _wait_until(lambda: len(scans) >= 3)

        assert not cell.liveness_scan_task.done()


class _ManagerHandleShim:
    # [CLASS DOC] HELPER: Adapter exposing the manager's cell-info query with a remote-style call
    # whose result is awaitable, as a ray actor handle would.
    """The provider talks to the manager as a ray actor; here it is the object itself."""

    def __init__(self, manager: RayWorkerManager) -> None:
        self.get_cell_infos = _ShimMethod(manager.get_cell_infos)


class _ShimMethod:
    # [CLASS DOC] HELPER: Wraps a function so that calling remote with keyword arguments invokes it
    # and returns an awaitable of the result.
    def __init__(self, fn) -> None:
        self._fn = fn

    def remote(self, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the wrapped function synchronously and returns a coroutine
        # that yields its result.
        return _resolved(self._fn(**kwargs))


async def _resolved(value):
    # [FUNCTION DOC] HELPER: Coroutine that returns the value unchanged.
    return value


class _RecordingReconciler:
    # [CLASS DOC] HELPER: Reconcile callback for the provider's cell watcher; records each
    # (cell id, info or none) call in order.
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []

    async def __call__(self, cell_id: str, info) -> None:
        # [FUNCTION DOC] HELPER: Appends the cell id and info (none means the cell is gone).
        self.calls.append((cell_id, info))


class TestTheDeathReachesTheProvider:
    # [CLASS DOC] End-to-end tests that a scan-detected death is seen by a ray worker provider
    # watching the manager (polling every millisecond) and reconciled.
    async def test_a_worker_that_died_alone_is_reconciled_away(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Starts watching, kills a worker, scans, waits until the last reconcile has
        # no info, then stops the watch. Asserts two reconcile calls, both for cell engine-00000
        # (appearance then disappearance).
        """Noticing the death is only worth anything if the watcher that rebuilds cells hears about it."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        provider = RayWorkerProvider(
            worker_manager_handle=_ManagerHandleShim(manager),
            pool_ids=["engine"],
            poll_interval_seconds=0.001,
        )
        reconciler = _RecordingReconciler()
        stop = await provider.watch_cells(reconciler)
        try:
            _kill_worker_process(fake_ray_cluster, handle_index=0)
            await _scan_all_live_cells(manager)

            await _wait_until(lambda: reconciler.calls[-1][1] is None)
        finally:
            await stop()

        assert [cell_id for cell_id, _ in reconciler.calls] == ["engine-00000", "engine-00000"]

    async def test_the_rebuilt_cell_is_reconciled_back_with_a_new_hash(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Records the first reconciled workers hash, kills a worker and scans, waits
        # for the removal, starts the cell again, waits for a reconcile with info, and asserts its
        # workers hash differs from the first.
        """A cell dropped by the scan must be startable again and look new, or nothing reconnects to it."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        provider = RayWorkerProvider(
            worker_manager_handle=_ManagerHandleShim(manager),
            pool_ids=["engine"],
            poll_interval_seconds=0.001,
        )
        reconciler = _RecordingReconciler()
        stop = await provider.watch_cells(reconciler)
        try:
            first_hash = reconciler.calls[0][1].workers_hash
            _kill_worker_process(fake_ray_cluster, handle_index=0)
            await _scan_all_live_cells(manager)
            await _wait_until(lambda: reconciler.calls[-1][1] is None)

            await manager.start_cells(["engine-00000"])

            await _wait_until(lambda: reconciler.calls[-1][1] is not None)
        finally:
            await stop()

        assert reconciler.calls[-1][1].workers_hash != first_hash


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    # [FUNCTION DOC] HELPER: Polls the predicate every 10 ms on the running event loop until true;
    # fails an assertion ("the condition never became true") after timeout seconds (default 5).
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "the condition never became true"
        await asyncio.sleep(0.01)
