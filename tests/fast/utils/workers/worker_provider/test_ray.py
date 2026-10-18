# [MODULE DOC] Tests for the ray worker provider, which talks to the ray worker manager actor
# through a handle. Fakes stand in for the manager's remote methods (each remote call returns an
# awaitable or value) and for ray's result getter. Covered: address lookup is never cached;
# stopping cells waits for the remote stop; worker infos for several cells come back in request
# order; the cell watch (initial sync reconciles every cell before returning and propagates
# failures, requires pool ids, asks only for its pools, polls and reconciles only on change using
# the workers hash, reports a not-alive cell as gone exactly once, retries failed polls and failed
# reconciles, isolates failures per cell, stops cleanly cancelling in-flight reconciles and not
# swallowing the stopper's own cancellation); rpc handles for served workers (class path known)
# versus actor handles for command-communication workers, with the generation check refusing a
# cell restarted between the two manager calls; and batched fetching of actor handles in one
# round trip.
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import pytest

import miles.utils.workers.worker_provider.ray as ray_worker_provider_mod
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort


@dataclass
class _FakeRemoteMethod:
    # [CLASS DOC] HELPER: Fake remote address lookup method returning the scripted answers in call
    # order, and recording the worker names asked for.
    answers: list[dict[str, HostAndPort]]
    requested_names: list[str] = field(default_factory=list)

    def remote(self, worker_name: str) -> Any:
        # [FUNCTION DOC] HELPER: Records the name and returns an awaitable of the answer at the
        # position of this call (raises an index error beyond the scripted answers).
        self.requested_names.append(worker_name)
        return _resolved(self.answers[len(self.requested_names) - 1])


@dataclass
class _FakeManagerHandle:
    # [CLASS DOC] HELPER: Fake manager handle exposing only the address lookup method.
    get_worker_addrs: _FakeRemoteMethod


async def _resolved(value: dict[str, HostAndPort]) -> dict[str, HostAndPort]:
    # [FUNCTION DOC] HELPER: Coroutine returning the value unchanged.
    return value


def _make_handle(*answers: dict[str, HostAndPort]) -> _FakeManagerHandle:
    # [FUNCTION DOC] HELPER: Builds a fake manager handle answering address lookups with the given
    # maps in order.
    return _FakeManagerHandle(get_worker_addrs=_FakeRemoteMethod(answers=list(answers)))


class TestRayWorkerProviderAddressLookup:
    # [CLASS DOC] Tests of address lookup through the manager handle.
    async def test_every_lookup_asks_the_manager_again(self):
        # [FUNCTION DOC] Two lookups of the same worker get two different scripted answers (ports
        # 15000 then 15001) and the manager was asked twice with that name (no caching).
        """Addresses are never cached, so a relaunched worker is not answered with a stale endpoint."""
        handle = _make_handle(
            {"primary": HostAndPort(host="10.0.0.7", port=15000)},
            {"primary": HostAndPort(host="10.0.0.7", port=15001)},
        )
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])

        first = (await provider.get_addrs(worker_name="router-0-0"))["primary"]
        second = (await provider.get_addrs(worker_name="router-0-0"))["primary"]

        assert (first.port, second.port) == (15000, 15001)
        assert handle.get_worker_addrs.requested_names == ["router-0-0", "router-0-0"]


class TestRayWorkerProviderGetAddrs:
    # [CLASS DOC] Tests of the shape of the address answer.
    async def test_returns_every_named_port_of_the_worker(self):
        # [FUNCTION DOC] The returned map equals the manager's full map (two named ports).
        """Consumers that need more than the primary endpoint get the worker's whole address map."""
        addrs = {
            "primary": HostAndPort(host="10.0.0.7", port=15000),
            "disaggregation_bootstrap": HostAndPort(host="10.0.0.7", port=15001),
        }
        handle = _make_handle(addrs)
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])

        assert await provider.get_addrs(worker_name="engine-0-0") == addrs


class _FakeStopCellsMethod:
    # [CLASS DOC] HELPER: Fake remote stop method that signals entry and blocks until released.
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def remote(self, cell_ids: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Sets the entered event, then waits (no timeout) for the release
        # event. Ignores the cell ids.
        self.entered.set()
        await self.release.wait()


@dataclass
class _StoppingManagerHandle:
    # [CLASS DOC] HELPER: Fake manager handle exposing only the stop method.
    stop_cells: _FakeStopCellsMethod


class TestRayWorkerProviderStopCells:
    # [CLASS DOC] Tests of the provider's stop operation.
    async def test_stop_cells_does_not_report_completion_while_the_remote_stop_is_pending(self) -> None:
        # [FUNCTION DOC] Starts the provider's stop as a task, waits until the remote stop is
        # entered, asserts the task is not done, then releases the remote stop and awaits the task.
        """Stopping cells remains pending until the remote operation has completed."""
        handle = _StoppingManagerHandle(stop_cells=_FakeStopCellsMethod())
        provider = RayWorkerProvider(worker_manager_handle=handle)

        stopping = asyncio.create_task(provider.stop_cells(cell_ids=["cell-a"]))
        await handle.stop_cells.entered.wait()
        assert not stopping.done()

        handle.stop_cells.release.set()
        await stopping


@dataclass
class _FakeObjectRef:
    # [CLASS DOC] HELPER: Fake ray object reference holding its resolved worker infos.
    infos: list[WorkerInfo]


@dataclass
class _FakeWorkerInfosMethod:
    # [CLASS DOC] HELPER: Fake remote worker-infos method backed by a mapping from cell id to
    # infos; records the requested cell ids.
    infos_by_cell_id: dict[str, list[WorkerInfo]]
    requested_cell_ids: list[str] = field(default_factory=list)

    def remote(self, cell_id: str) -> _FakeObjectRef:
        # [FUNCTION DOC] HELPER: Records the cell id and returns a reference holding that cell's
        # infos (key error for an unknown cell).
        self.requested_cell_ids.append(cell_id)
        return _FakeObjectRef(infos=self.infos_by_cell_id[cell_id])


@dataclass
class _WorkerInfosManagerHandle:
    # [CLASS DOC] HELPER: Fake manager handle exposing only the worker-infos method.
    get_worker_infos: _FakeWorkerInfosMethod


class _FakeRayModule:
    # [CLASS DOC] HELPER: Stand-in for the ray module offering only a getter that resolves fake
    # references.
    @staticmethod
    def get(refs: list[_FakeObjectRef]) -> list[list[WorkerInfo]]:
        # [FUNCTION DOC] HELPER: Returns the infos of each reference, in order.
        return [ref.infos for ref in refs]


def _worker_infos(cell_id: str, *, count: int) -> list[WorkerInfo]:
    # [FUNCTION DOC] HELPER: Builds count worker infos named <cell id>-<index>, generation 1, with a
    # primary address 10.0.0.7 at port 15000 plus the index, and gpu ids [index].
    return [
        WorkerInfo(
            name=f"{cell_id}-{worker_index}",
            generation=1,
            self_addrs={"primary": HostAndPort(host="10.0.0.7", port=15000 + worker_index)},
            gpu_ids=[worker_index],
        )
        for worker_index in range(count)
    ]


class TestRayWorkerProviderGetWorkerInfos:
    # [CLASS DOC] Tests of worker info retrieval for several cells.
    def test_multiple_cell_ids_return_worker_info_groups_in_request_order(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Replaces the provider module's ray with the fake; asking for cells c, a, b
        # queries the manager in that order and returns the groups in that order.
        """Every requested cell gets its own worker group, positioned as the caller asked for it."""
        infos_by_cell_id = {
            "cell-a": _worker_infos("cell-a", count=1),
            "cell-b": _worker_infos("cell-b", count=2),
            "cell-c": _worker_infos("cell-c", count=3),
        }
        handle = _WorkerInfosManagerHandle(get_worker_infos=_FakeWorkerInfosMethod(infos_by_cell_id=infos_by_cell_id))
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])
        monkeypatch.setattr(ray_worker_provider_mod, "ray", _FakeRayModule)

        groups = provider.get_worker_infos(cell_ids=["cell-c", "cell-a", "cell-b"])

        assert handle.get_worker_infos.requested_cell_ids == ["cell-c", "cell-a", "cell-b"]
        assert groups == [infos_by_cell_id["cell-c"], infos_by_cell_id["cell-a"], infos_by_cell_id["cell-b"]]


@dataclass
class _FakeCellInfosMethod:
    # [CLASS DOC] HELPER: Fake remote cell-infos method. Answers are scripted per call (the last
    # repeats); an exception instance as an answer makes that call's awaitable raise it. Records
    # the pool ids of each call.
    answers: list[Any]
    calls: list[list[str]] = field(default_factory=list)

    def remote(self, *, pool_ids: list[str]) -> Any:
        # [FUNCTION DOC] HELPER: Records a copy of the pool ids, picks the answer for this call
        # index (clamped to the last), and returns an awaitable that raises it if it is an
        # exception, otherwise resolves to it.
        self.calls.append(list(pool_ids))
        answer = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        if isinstance(answer, Exception):
            return _raised(answer)
        return _resolved(answer)


@dataclass
class _WatchingManagerHandle:
    # [CLASS DOC] HELPER: Fake manager handle exposing only the cell-infos method.
    get_cell_infos: _FakeCellInfosMethod


async def _raised(error: Exception) -> Any:
    # [FUNCTION DOC] HELPER: Coroutine that raises the given exception.
    raise error


def _make_watching_handle(*answers: Any) -> _WatchingManagerHandle:
    # [FUNCTION DOC] HELPER: Builds a watching handle whose successive polls return the given
    # answers (mappings of cell id to info, or exceptions to raise).
    return _WatchingManagerHandle(get_cell_infos=_FakeCellInfosMethod(answers=list(answers)))


def _cell_info(cell_id: str, *, alive: bool = True, workers_hash: str = "hash-0") -> CellInfo:
    # [FUNCTION DOC] HELPER: Builds a cell info of pool inference-engine-0-0 with one worker named
    # <cell id>-0, the given aliveness (default alive) and workers hash (default hash-0), empty
    # metadata.
    return CellInfo(
        cell_id=cell_id,
        pool_id="inference-engine-0-0",
        alive=alive,
        worker_names=[f"{cell_id}-0"],
        workers_hash=workers_hash,
        meta={},
    )


async def _wait_until(predicate, *, timeout_seconds: float = 2.0) -> None:
    # [FUNCTION DOC] HELPER: Polls the predicate every millisecond until true; fails an assertion
    # ("timed out waiting for the watch loop") after the timeout (default 2 seconds).
    deadline = time.monotonic() + timeout_seconds
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the watch loop"
        await asyncio.sleep(0.001)


class _RecordingReconciler:
    # [CLASS DOC] HELPER: Reconcile callback that records every (cell id, info or none) call.
    def __init__(self) -> None:
        self.calls: list[tuple[str, CellInfo | None]] = []

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: Appends the call to the record.
        self.calls.append((cell_id, info))


class _FailingOnceReconciler(_RecordingReconciler):
    # [CLASS DOC] HELPER: Recording reconciler that raises a runtime error the first time it is
    # called for a given cell id (without recording that call) and records normally afterwards.
    def __init__(self, *, failing_cell_id: str) -> None:
        super().__init__()
        self._failing_cell_id: str | None = failing_cell_id

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: For the failing cell id on its first call, disarms itself and
        # raises "reconcile failed"; otherwise records the call.
        if cell_id == self._failing_cell_id:
            self._failing_cell_id = None
            raise RuntimeError("reconcile failed")
        await super().__call__(cell_id, info)


class _FailingForCellReconciler(_RecordingReconciler):
    # [CLASS DOC] HELPER: Recording reconciler that records every call, then raises a runtime error
    # whenever the cell id is the configured failing one (every time).
    def __init__(self, *, failing_cell_id: str) -> None:
        super().__init__()
        self._failing_cell_id = failing_cell_id

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: Records the call, then raises "reconcile rejected <cell id>" for the
        # failing cell.
        await super().__call__(cell_id, info)
        if cell_id == self._failing_cell_id:
            raise RuntimeError(f"reconcile rejected {cell_id}")


class _AlwaysFailingReconciler(_RecordingReconciler):
    # [CLASS DOC] HELPER: Recording reconciler that raises after recording on every call.
    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: Records the call and raises "reconcile rejected the cell".
        await super().__call__(cell_id, info)
        raise RuntimeError("reconcile rejected the cell")


class _BlockingReconciler(_RecordingReconciler):
    # [CLASS DOC] HELPER: Recording reconciler that blocks forever after recording, signalling entry
    # and noting if it was cancelled.
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.cancelled = False

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: Records the call, sets the entered event, waits forever; on
        # cancellation sets the cancelled flag and re-raises the cancellation.
        await super().__call__(cell_id, info)
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _StuckOnCancelReconciler(_RecordingReconciler):
    # [CLASS DOC] HELPER: Recording reconciler that blocks forever after recording, and when
    # cancelled delays finishing until the release event is set.
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: Records the call, sets the entered event, waits forever; on
        # cancellation waits for the release event and then re-raises the cancellation.
        await super().__call__(cell_id, info)
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self.release.wait()
            raise


class TestRayWorkerProviderWatchCellsInitialSync:
    # [CLASS DOC] Tests of the synchronous first pass of the cell watch.
    async def test_every_initial_cell_is_reconciled_before_the_watch_is_established(self):
        # [FUNCTION DOC] Two cells are reported by the first poll; immediately after the watch call
        # returns the reconciler has seen cell-a then cell-b.
        """Callers may assume the pool is fully observed once watch_cells returns."""
        handle = _make_watching_handle({"cell-a": _cell_info("cell-a"), "cell-b": _cell_info("cell-b")})
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])
        reconciler = _RecordingReconciler()

        stop = await provider.watch_cells(reconciler)
        try:
            assert [cell_id for cell_id, _ in reconciler.calls] == ["cell-a", "cell-b"]
        finally:
            await stop()

    async def test_a_failing_initial_sync_propagates_instead_of_starting_the_loop(self):
        # [FUNCTION DOC] The first poll raises a runtime error "manager unreachable"; the watch call
        # raises it.
        """A pool we never managed to read must not look like an empty pool."""
        handle = _make_watching_handle(RuntimeError("manager unreachable"))
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])

        with pytest.raises(RuntimeError, match="manager unreachable"):
            await provider.watch_cells(_RecordingReconciler())

    async def test_an_initial_reconcile_failure_prevents_watch_establishment(self):
        # [FUNCTION DOC] The reconciler always fails; the watch call raises "reconcile rejected the
        # cell", and after 20 ms the manager was polled only once (no loop was started).
        """A caller that never learned about the initial cells must not be left with a live watch."""
        handle = _make_watching_handle({"cell-a": _cell_info("cell-a")})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )

        with pytest.raises(RuntimeError, match="reconcile rejected the cell"):
            await provider.watch_cells(_AlwaysFailingReconciler())

        await asyncio.sleep(0.02)

        assert handle.get_cell_infos.calls == [["inference-engine-0-0"]]

    async def test_watching_without_pool_ids_fails_before_contacting_manager(self):
        # [FUNCTION DOC] A provider built without pool ids raises an assertion error mentioning
        # "without the pool_ids" on watch and made no manager call.
        """A provider built for address lookups only must refuse to watch rather than watch nothing."""
        handle = _make_watching_handle({})
        provider = RayWorkerProvider(worker_manager_handle=handle)

        with pytest.raises(AssertionError, match="without the pool_ids"):
            await provider.watch_cells(_RecordingReconciler())

        assert handle.get_cell_infos.calls == []

    async def test_only_the_requested_pools_are_asked_for(self):
        # [FUNCTION DOC] The single poll call carried exactly the configured pool id list.
        """The controller must not observe cells belonging to someone else."""
        handle = _make_watching_handle({})
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])

        stop = await provider.watch_cells(_RecordingReconciler())
        try:
            assert handle.get_cell_infos.calls == [["inference-engine-0-0"]]
        finally:
            await stop()


class TestRayWorkerProviderWatchCellsPolling:
    # [CLASS DOC] Tests of the background poll loop (1 ms interval) and its reconcile decisions.
    async def test_an_unchanged_cell_is_not_reconciled_again(self):
        # [FUNCTION DOC] The same info is returned on repeated polls; after at least 3 polls the
        # reconciler has exactly one call.
        """Re-reconciling every poll would restart cells every interval."""
        info = _cell_info("cell-a")
        handle = _make_watching_handle({"cell-a": info}, {"cell-a": info})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _RecordingReconciler()

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: len(handle.get_cell_infos.calls) >= 3)
            assert reconciler.calls == [("cell-a", info)]
        finally:
            await stop()

    async def test_a_stopped_cell_is_reported_as_gone_exactly_once(self):
        # [FUNCTION DOC] First poll alive, later polls the same cell not alive; after at least 4
        # polls the calls are the alive info then a single report with no info.
        """A suspended cell must look like a disappeared cell, and must not be re-reported."""
        alive = _cell_info("cell-a")
        handle = _make_watching_handle({"cell-a": alive}, {"cell-a": _cell_info("cell-a", alive=False)})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _RecordingReconciler()

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: len(handle.get_cell_infos.calls) >= 4)
            assert reconciler.calls == [("cell-a", alive), ("cell-a", None)]
        finally:
            await stop()

    async def test_a_relaunched_cell_is_reconciled_again_because_its_workers_changed(self):
        # [FUNCTION DOC] The same cell id with workers hash hash-0 then hash-1; the first two
        # reconciles are the two infos in order.
        """A replacement cell keeps its id, so only the workers hash can reveal it."""
        first = _cell_info("cell-a", workers_hash="hash-0")
        second = _cell_info("cell-a", workers_hash="hash-1")
        handle = _make_watching_handle({"cell-a": first}, {"cell-a": second})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _RecordingReconciler()

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: len(reconciler.calls) >= 2)
            assert reconciler.calls[:2] == [("cell-a", first), ("cell-a", second)]
        finally:
            await stop()

    async def test_a_failing_poll_is_retried_instead_of_killing_the_watch(self):
        # [FUNCTION DOC] Polls answer empty, then raise a transient error, then report a cell; the
        # reconciler eventually receives exactly that one call.
        """One unreachable manager call must not silently end pool observation."""
        info = _cell_info("cell-a")
        handle = _make_watching_handle({}, RuntimeError("transient"), {"cell-a": info})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _RecordingReconciler()

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: reconciler.calls == [("cell-a", info)])
        finally:
            await stop()

    async def test_a_cell_added_by_a_partially_failed_poll_can_still_disappear(self):
        # [FUNCTION DOC] Second poll reports cells a and b with b failing once; third poll reports
        # only b. Eventually cell a is reported gone, and its earlier reconcile was recorded.
        """A reconcile that raises mid-poll must not lose the bookkeeping of the cells already delivered."""
        info_a = _cell_info("cell-a")
        info_b = _cell_info("cell-b")
        handle = _make_watching_handle({}, {"cell-a": info_a, "cell-b": info_b}, {"cell-b": info_b})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _FailingOnceReconciler(failing_cell_id="cell-b")

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: ("cell-a", None) in reconciler.calls)
            assert ("cell-a", info_a) in reconciler.calls
        finally:
            await stop()

    async def test_a_failed_reconcile_is_retried_before_the_cell_is_marked_seen(self):
        # [FUNCTION DOC] A cell whose first reconcile fails is delivered again on a later poll;
        # eventually the recorded calls are exactly that one successful call.
        """A cell whose reconcile raised must be delivered again instead of being silently forgotten."""
        info = _cell_info("cell-a")
        handle = _make_watching_handle({}, {"cell-a": info})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _FailingOnceReconciler(failing_cell_id="cell-a")

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: reconciler.calls == [("cell-a", info)])
        finally:
            await stop()


class _RpcDemoWorker:
    # [CLASS DOC] HELPER: Placeholder served worker class with one method.
    def report(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the text "ok".
        return "ok"


_RPC_DEMO_WORKER_PATH = f"{__name__}._RpcDemoWorker"


@dataclass
class _ServingWorkerInfosMethod:
    # [CLASS DOC] HELPER: Fake remote worker-infos method returning scripted info lists per call
    # (the last repeats) and recording the cell ids asked for.
    answers: list[list[WorkerInfo]]
    calls: list[str] = field(default_factory=list)

    def remote(self, cell_id: str) -> Any:
        # [FUNCTION DOC] HELPER: Records the cell id and returns the scripted list for this call
        # (not an awaitable; the tests replace ray's getter with an identity function).
        self.calls.append(cell_id)
        return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]


@dataclass
class _ServingManagerHandle:
    # [CLASS DOC] HELPER: Fake manager handle exposing only the worker-infos method.
    get_worker_infos: _ServingWorkerInfosMethod


def _served_worker_info(*, generation: int, port: int = 15000) -> WorkerInfo:
    # [FUNCTION DOC] HELPER: Builds a worker info for trainer-engine-actor-00000-00000 with the
    # given generation, an rpc address 10.0.0.7 at the given port (default 15000), no gpus and the
    # demo worker class path.
    return WorkerInfo(
        name="trainer-engine-actor-00000-00000",
        generation=generation,
        self_addrs={"rpc": HostAndPort(host="10.0.0.7", port=port)},
        gpu_ids=[],
        worker_class=_RPC_DEMO_WORKER_PATH,
    )


class TestRayWorkerProviderRpcHandles:
    # [CLASS DOC] Tests of handle building for workers that have a worker class (rpc mode).
    def test_a_served_worker_is_called_over_its_own_server(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With ray's getter replaced by identity, the handle for the worker is an rpc
        # handle whose transport url is "http://10.0.0.7:15000".
        """Under rpc the launcher answers with the class, and the caller is what builds the client."""
        handle = _ServingManagerHandle(
            get_worker_infos=_ServingWorkerInfosMethod(answers=[[_served_worker_info(generation=0)]])
        )
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["trainer-engine-actor"])
        monkeypatch.setattr(ray_worker_provider_mod.ray, "get", lambda refs: refs)

        built = provider.get_handle("trainer-engine-actor-00000-00000")

        assert isinstance(built, RpcWorkerHandle)
        assert built._transport._server_url == "http://10.0.0.7:15000"

    def test_the_worker_of_the_named_cell_is_asked_for(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Getting the handle of worker trainer-engine-actor-00000-00000 asks the
        # manager for exactly the cell trainer-engine-actor-00000.
        """A lookup that derived another cell would hand back a handle to somebody else's worker."""
        handle = _ServingManagerHandle(
            get_worker_infos=_ServingWorkerInfosMethod(answers=[[_served_worker_info(generation=0)]])
        )
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["trainer-engine-actor"])
        monkeypatch.setattr(ray_worker_provider_mod.ray, "get", lambda refs: refs)

        provider.get_handle("trainer-engine-actor-00000-00000")

        assert handle.get_worker_infos.calls == ["trainer-engine-actor-00000"]


@dataclass
class _ActorHandleMethod:
    # [CLASS DOC] HELPER: Fake remote actor-handle method that checks the expected generation
    # against the one it was configured with.
    generation_of_worker: int
    requested: list[tuple[str, int]] = field(default_factory=list)

    def remote(self, worker_name: str, *, expected_generation: int) -> Any:
        # [FUNCTION DOC] HELPER: Records (name, expected generation), fails an assertion if the
        # expected generation differs from the configured one, and returns the text
        # "actor-of-<name>".
        self.requested.append((worker_name, expected_generation))
        assert expected_generation == self.generation_of_worker, f"generation {self.generation_of_worker}"
        return f"actor-of-{worker_name}"


@dataclass
class _RayCommManagerHandle:
    # [CLASS DOC] HELPER: Fake manager handle with worker-infos and actor-handle methods.
    get_worker_infos: _ServingWorkerInfosMethod
    get_actor_handle: _ActorHandleMethod


def _actor_worker_info(*, generation: int, name: str = "trainer-engine-actor-00000-00000") -> WorkerInfo:
    # [FUNCTION DOC] HELPER: Builds a worker info (default name trainer-engine-actor-00000-00000)
    # with the given generation, a master address 10.0.0.7:20000, no gpus and no worker class
    # (an actor-communication worker).
    return WorkerInfo(
        name=name,
        generation=generation,
        self_addrs={"master": HostAndPort(host="10.0.0.7", port=20000)},
        gpu_ids=[],
        worker_class=None,
    )


class TestRayWorkerProviderRayHandlesAreOfTheGenerationDescribed:
    # [CLASS DOC] Tests that actor handles are requested with the generation the info described.
    def test_a_handle_of_the_described_generation_is_built(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Infos and actors agree on generation 3; the actor handle method was asked
        # once with the worker name and generation 3.
        """The ordinary path: the cell did not move between the two calls the resolution takes."""
        handle = _RayCommManagerHandle(
            get_worker_infos=_ServingWorkerInfosMethod(answers=[[_actor_worker_info(generation=3)]]),
            get_actor_handle=_ActorHandleMethod(generation_of_worker=3),
        )
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["trainer-engine-actor"])
        monkeypatch.setattr(ray_worker_provider_mod.ray, "get", lambda ref: ref)

        provider.get_handle("trainer-engine-actor-00000-00000")

        assert handle.get_actor_handle.requested == [("trainer-engine-actor-00000-00000", 3)]

    def test_a_cell_restarted_between_the_two_calls_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Infos say generation 3 while the actor is generation 4; getting the handle
        # raises an assertion error.
        """Pairing an old cell's addresses with a new cell's actors sends new ranks to a dead rendezvous."""
        handle = _RayCommManagerHandle(
            get_worker_infos=_ServingWorkerInfosMethod(answers=[[_actor_worker_info(generation=3)]]),
            get_actor_handle=_ActorHandleMethod(generation_of_worker=4),
        )
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["trainer-engine-actor"])
        monkeypatch.setattr(ray_worker_provider_mod.ray, "get", lambda ref: ref)

        with pytest.raises(AssertionError):
            provider.get_handle("trainer-engine-actor-00000-00000")


@dataclass
class _PendingActorHandle:
    # [CLASS DOC] HELPER: Unresolved actor handle request (worker name and expected generation).
    worker_name: str
    expected_generation: int


@dataclass
class _PendingActorHandleMethod:
    # [CLASS DOC] HELPER: Fake remote actor-handle method returning unresolved requests and
    # recording each (name, expected generation).
    requested: list[tuple[str, int]] = field(default_factory=list)

    def remote(self, worker_name: str, *, expected_generation: int) -> _PendingActorHandle:
        # [FUNCTION DOC] HELPER: Records the request and returns an unresolved request object.
        self.requested.append((worker_name, expected_generation))
        return _PendingActorHandle(worker_name=worker_name, expected_generation=expected_generation)


@dataclass
class _PendingManagerHandle:
    # [CLASS DOC] HELPER: Fake manager handle exposing only the pending actor-handle method.
    get_actor_handle: _PendingActorHandleMethod


@dataclass
class _RecordingRayGet:
    # [CLASS DOC] HELPER: Fake ray getter that resolves a batch of pending requests, checking each
    # against a configured current generation per worker, and records every batch.
    generation_of_workers: dict[str, int]
    batches: list[list[_PendingActorHandle]] = field(default_factory=list)

    def __call__(self, refs: list[_PendingActorHandle]) -> list[str]:
        # [FUNCTION DOC] HELPER: Records the batch and returns the resolved actor texts in order.
        self.batches.append(refs)
        return [self._resolve(ref) for ref in refs]

    def _resolve(self, ref: _PendingActorHandle) -> str:
        # [FUNCTION DOC] HELPER: Fails an assertion "<name> is now generation N, not the M it was
        # described as" if the configured generation differs from the expected one; otherwise
        # returns "actor-of-<name>". Raises a key error for an unconfigured worker.
        generation = self.generation_of_workers[ref.worker_name]
        assert (
            generation == ref.expected_generation
        ), f"{ref.worker_name} is now generation {generation}, not the {ref.expected_generation} it was described as"
        return f"actor-of-{ref.worker_name}"


class _BatchingProbe(NamedTuple):
    # [CLASS DOC] HELPER: Bundle of the provider under test, its fake manager handle and the
    # recording ray getter.
    provider: RayWorkerProvider
    manager_handle: _PendingManagerHandle
    ray_get: _RecordingRayGet


def _build_batching_provider(
    *, monkeypatch: pytest.MonkeyPatch, generation_of_workers: dict[str, int]
) -> _BatchingProbe:
    # [FUNCTION DOC] HELPER: Builds a provider over the trainer-engine-actor pool with a pending
    # manager handle, and replaces the provider module's ray getter with a recording one using the
    # given current generations. Returns the bundle.
    handle = _PendingManagerHandle(get_actor_handle=_PendingActorHandleMethod())
    provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["trainer-engine-actor"])
    ray_get = _RecordingRayGet(generation_of_workers=generation_of_workers)
    monkeypatch.setattr(ray_worker_provider_mod.ray, "get", ray_get)
    return _BatchingProbe(provider=provider, manager_handle=handle, ray_get=ray_get)


class TestRayWorkerProviderHandlesAreFetchedInOneBatch:
    # [CLASS DOC] Tests of batched handle resolution for several worker infos.
    def test_many_workers_cost_a_single_round_trip(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] 4 actor workers of generation 3; one ray getter batch and 4 requests with
        # generation 3 in order.
        """Resolving per worker blocks the event loop for one round trip each, so all of them go out together."""
        names = [f"trainer-engine-actor-00000-{index:05d}" for index in range(4)]
        probe = _build_batching_provider(monkeypatch=monkeypatch, generation_of_workers={name: 3 for name in names})

        probe.provider.get_handles_of_worker_infos([_actor_worker_info(name=name, generation=3) for name in names])

        assert len(probe.ray_get.batches) == 1
        assert probe.manager_handle.get_actor_handle.requested == [(name, 3) for name in names]

    def test_every_worker_is_paired_with_its_own_handle(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] 3 workers; each returned handle's inner actor handle (private attribute)
        # is the actor text for its own worker name.
        """A batch that mixes up the order would hand each caller somebody else's actor."""
        names = [f"trainer-engine-actor-00000-{index:05d}" for index in range(3)]
        probe = _build_batching_provider(monkeypatch=monkeypatch, generation_of_workers={name: 3 for name in names})

        handles = probe.provider.get_handles_of_worker_infos(
            [_actor_worker_info(name=name, generation=3) for name in names]
        )

        assert {name: handle._actor_handle for name, handle in handles.items()} == {
            name: f"actor-of-{name}" for name in names
        }

    def test_a_worker_of_another_generation_is_still_refused(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] The second worker is generation 4 while described as 3; the batch raises an
        # assertion error mentioning that worker's name.
        """Batching must not swallow the generation check that keeps stale actors out of a new cell."""
        names = ["trainer-engine-actor-00000-00000", "trainer-engine-actor-00000-00001"]
        probe = _build_batching_provider(monkeypatch=monkeypatch, generation_of_workers={names[0]: 3, names[1]: 4})

        with pytest.raises(AssertionError, match=names[1]):
            probe.provider.get_handles_of_worker_infos([_actor_worker_info(name=name, generation=3) for name in names])

    def test_asking_for_nothing_answers_nothing(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] An empty list returns an empty mapping and requests nothing.
        """An empty batch must answer an empty mapping instead of failing on a round trip with nothing to ask."""
        probe = _build_batching_provider(monkeypatch=monkeypatch, generation_of_workers={})

        assert probe.provider.get_handles_of_worker_infos([]) == {}
        assert probe.manager_handle.get_actor_handle.requested == []

    def test_served_workers_keep_their_locally_built_handles(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] A mixed list of one served worker and one actor worker: the served one's
        # handle is an rpc handle, and only the actor worker was requested from the manager.
        """An rpc worker needs no round trip, and must not be dropped from a batch that also holds actors."""
        actor_name = "trainer-engine-actor-00000-00001"
        probe = _build_batching_provider(monkeypatch=monkeypatch, generation_of_workers={actor_name: 3})

        handles = probe.provider.get_handles_of_worker_infos(
            [_served_worker_info(generation=3), _actor_worker_info(name=actor_name, generation=3)]
        )

        assert isinstance(handles["trainer-engine-actor-00000-00000"], RpcWorkerHandle)
        assert probe.manager_handle.get_actor_handle.requested == [(actor_name, 3)]


class TestRayWorkerProviderWatchCellsStop:
    # [CLASS DOC] Tests of the stop function returned by the cell watch.
    async def test_stopping_cancels_an_in_flight_reconcile(self):
        # [FUNCTION DOC] A reconcile blocks forever; once it has entered, stop completes within 2
        # seconds and the reconcile saw cancellation.
        """Stopping must not hang waiting for a reconcile that is still running."""
        handle = _make_watching_handle({}, {"cell-a": _cell_info("cell-a")})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _BlockingReconciler()

        stop = await provider.watch_cells(reconciler)
        await asyncio.wait_for(reconciler.entered.wait(), timeout=2.0)
        await asyncio.wait_for(stop(), timeout=2.0)

        assert reconciler.cancelled

    async def test_stopping_ends_the_polling(self):
        # [FUNCTION DOC] After at least 2 polls the watch is stopped; 20 ms later the poll count is
        # unchanged.
        """The returned stop function must actually stop the loop, not just detach from it."""
        handle = _make_watching_handle({})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )

        stop = await provider.watch_cells(_RecordingReconciler())
        await _wait_until(lambda: len(handle.get_cell_infos.calls) >= 2)
        await stop()
        settled = len(handle.get_cell_infos.calls)

        await asyncio.sleep(0.02)

        assert len(handle.get_cell_infos.calls) == settled

    async def test_stopping_does_not_swallow_the_callers_cancellation(self):
        # [FUNCTION DOC] A reconcile that delays finishing its own cancellation. A task calling stop
        # is cancelled while the reconcile is stuck; after releasing the reconcile and calling stop
        # again, asserts the stopper task ended cancelled and never reported returning.
        """A stop that eats its caller's cancellation lets a timed-out shutdown run on past its teardown."""
        handle = _make_watching_handle({}, {"cell-a": _cell_info("cell-a")})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _StuckOnCancelReconciler()
        returned: list[str] = []

        stop = await provider.watch_cells(reconciler)
        await asyncio.wait_for(reconciler.entered.wait(), timeout=2.0)

        async def _stopper() -> None:
            # [FUNCTION DOC] HELPER: Awaits the stop function and then appends "returned".
            await stop()
            returned.append("returned")

        stopper = asyncio.create_task(_stopper())
        await asyncio.sleep(0)
        stopper.cancel()
        await asyncio.gather(stopper, return_exceptions=True)
        reconciler.release.set()
        await stop()

        assert stopper.cancelled() and returned == []


class TestRayWorkerProviderPollIsolation:
    # [CLASS DOC] Tests that one cell's reconcile failure does not hide other cells.
    async def test_one_cells_failing_reconcile_does_not_block_the_others(self):
        # [FUNCTION DOC] Cell a always fails its reconcile in the initial sync; the watch call
        # raises "reconcile rejected cell-a" yet the reconciler was still called for a then b.
        """A cell that keeps failing to reconcile must not leave the rest of the pool unobserved."""
        handle = _make_watching_handle({"cell-a": _cell_info("cell-a"), "cell-b": _cell_info("cell-b")})
        provider = RayWorkerProvider(worker_manager_handle=handle, pool_ids=["inference-engine-0-0"])
        reconciler = _FailingForCellReconciler(failing_cell_id="cell-a")

        with pytest.raises(RuntimeError, match="reconcile rejected cell-a"):
            await provider.watch_cells(reconciler)

        assert [cell_id for cell_id, _ in reconciler.calls] == ["cell-a", "cell-b"]

    async def test_a_failed_cell_is_retried_while_the_others_are_not(self):
        # [FUNCTION DOC] Cell a fails once; the loop retries only a. After a succeeds and the watch
        # is stopped, the recorded calls are b then a (b not repeated).
        """Only the cell whose reconcile failed stays unobserved for the next poll."""
        info_a = _cell_info("cell-a")
        info_b = _cell_info("cell-b")
        handle = _make_watching_handle({}, {"cell-a": info_a, "cell-b": info_b})
        provider = RayWorkerProvider(
            worker_manager_handle=handle, pool_ids=["inference-engine-0-0"], poll_interval_seconds=0.001
        )
        reconciler = _FailingOnceReconciler(failing_cell_id="cell-a")

        stop = await provider.watch_cells(reconciler)
        try:
            await _wait_until(lambda: ("cell-a", info_a) in reconciler.calls)
        finally:
            await stop()

        assert reconciler.calls == [("cell-b", info_b), ("cell-a", info_a)]
