# [MODULE DOC]
# Unit tests for the Ray implementation of cell operations. A recording worker-manager handle and
# a recording engine provider stand in for the Ray actors, and a real inference controller is
# built around them. The tests check which operations wait for the controller's context lock
# (disruptive ones on rollout cells: suspend and fault injection) and which go straight to the
# worker manager (trainer cells, reads, resumes), plus the forwarding of arguments.
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.ray import RayCellOperations

_TRAINER_CELL_ID = "trainer-engine-actor-00001"


class _RecordingEngineProvider:
    # [CLASS DOC] HELPER: Fake engine provider that exposes a worker-manager handle and records
    # the cell ids it was asked to stop.
    def __init__(self, *, worker_manager: _RecordingWorkerManagerHandle) -> None:
        # [FUNCTION DOC] HELPER: Keeps the worker-manager handle and starts with an empty stopped
        # list.
        self._worker_manager_handle = worker_manager
        self.stopped: list[str] = []

    async def stop_cells(self, *, cell_ids: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Appends the given cell ids to the stopped list.
        self.stopped.extend(cell_ids)


class _RecordingRemoteMethod:
    # [CLASS DOC] HELPER: Fake remote actor method whose awaitable call records its name and
    # arguments into a shared list and returns a preset result.
    def __init__(self, *, name: str, calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> None:
        # [FUNCTION DOC] HELPER: Stores the method name and the shared call list; the result
        # starts as an empty mapping and tests may overwrite it.
        self._name = name
        self._calls = calls
        self.result: dict[str, Any] = {}

    async def remote(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Records the method name with positional and keyword arguments,
        # then returns the preset result.
        self._calls.append((self._name, args, kwargs))
        return self.result


class _RecordingWorkerManagerHandle:
    # [CLASS DOC] HELPER: Fake worker-manager actor handle with recording remote methods for
    # reading cell infos, starting cells, stopping cells and injecting faults, all sharing one
    # ordered call list.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the shared call list and the four recording methods.
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.get_cell_infos = _RecordingRemoteMethod(name="get_cell_infos", calls=self.calls)
        self.start_cells = _RecordingRemoteMethod(name="start_cells", calls=self.calls)
        self.stop_cells = _RecordingRemoteMethod(name="stop_cells", calls=self.calls)
        self.inject_fault = _RecordingRemoteMethod(name="inject_fault", calls=self.calls)


@dataclass(frozen=True)
class _Fixture:
    # [CLASS DOC] HELPER: Immutable bundle of the fake provider, the real inference controller,
    # the fake worker-manager handle and the Ray cell operations under test.
    provider: _RecordingEngineProvider
    controller: InferenceController
    worker_manager: _RecordingWorkerManagerHandle
    operations: RayCellOperations


def _make_fixture() -> _Fixture:
    # [FUNCTION DOC] HELPER: Wires up the fakes. Builds an inference controller with no router
    # providers, then sets its servers to one actor server holding the single rollout cell
    # "engine-0-2" and an active health-checker tracker. The operations object resolves that
    # controller through a function returning it.
    worker_manager = _RecordingWorkerManagerHandle()
    provider = _RecordingEngineProvider(worker_manager=worker_manager)
    controller = InferenceController(SimpleNamespace(), engine_provider=provider, router_providers=[])
    controller.servers = {
        "actor": SimpleNamespace(
            server_cells={"engine-0-2": SimpleNamespace()},
            health_checker_activeness=ActivenessTracker(active=True),
        )
    }
    return _Fixture(
        provider=provider,
        controller=controller,
        worker_manager=worker_manager,
        operations=RayCellOperations(
            worker_manager_handle=worker_manager, resolve_inference_controller=lambda: controller
        ),
    )


async def _hold_lock(*, lock: ContextLock, acquired: asyncio.Event, release: asyncio.Event) -> None:
    # [FUNCTION DOC] HELPER: Holds the lock as a long-running weight update would: acquires it,
    # sets the acquired event, waits for the release event, then lets go.
    async with lock:
        acquired.set()
        await release.wait()


async def _settle() -> None:
    # [FUNCTION DOC] HELPER: Yields to the event loop five times so pending tasks can progress.
    for _ in range(5):
        await asyncio.sleep(0)


async def test_a_suspend_waits_for_the_controller_lock_instead_of_reaching_the_worker_manager() -> None:
    # [FUNCTION DOC] Holds the controller lock, starts a suspend of a rollout cell, and asserts it
    # stays pending with nothing stopped and no worker-manager call. After the lock is released
    # it asserts the provider stopped that cell and the worker manager was still never called.
    """A suspend arriving mid weight update must not reach the worker manager until the update ends."""
    fixture = _make_fixture()
    acquired, release = asyncio.Event(), asyncio.Event()
    holding = asyncio.create_task(_hold_lock(lock=fixture.controller.context_lock, acquired=acquired, release=release))
    await acquired.wait()

    suspending = asyncio.create_task(fixture.operations.suspend(cell_id="engine-0-2"))
    await _settle()
    assert not suspending.done()
    assert fixture.provider.stopped == []
    assert fixture.worker_manager.calls == []

    release.set()
    await holding
    await suspending
    assert fixture.provider.stopped == ["engine-0-2"]
    assert fixture.worker_manager.calls == []


async def test_inject_fault_waits_for_the_controller_lock() -> None:
    # [FUNCTION DOC] Holds the lock, starts a kill-fault injection on a rollout cell, asserts it
    # is pending with no worker-manager call; after release asserts exactly one fault-injection
    # call for that cell with mode "sigkill" and worker index 0.
    """A fault arriving mid weight update must wait before killing an engine worker."""
    fixture = _make_fixture()
    acquired, release = asyncio.Event(), asyncio.Event()
    holding = asyncio.create_task(_hold_lock(lock=fixture.controller.context_lock, acquired=acquired, release=release))
    await acquired.wait()

    injecting = asyncio.create_task(
        fixture.operations.inject_fault(cell_id="engine-0-2", mode=FailureMode.SIGKILL, sub_index=0)
    )
    await _settle()
    assert not injecting.done()
    assert fixture.worker_manager.calls == []

    release.set()
    await holding
    await injecting
    assert fixture.worker_manager.calls == [
        ("inject_fault", ("engine-0-2",), {"mode": "sigkill", "worker_in_cell_index": 0})
    ]


async def test_a_trainer_cells_fault_reaches_the_worker_manager() -> None:
    # [FUNCTION DOC] While the lock is held, injects a kill fault into a trainer cell with a
    # five-second timeout; asserts it completes (not blocked by the lock) with one fault call to
    # the worker manager (mode "sigkill", worker index 0). Releases the lock at the end.
    """Regression: routing a trainer cell through the rollout controller raised, so the actor never died."""
    fixture = _make_fixture()
    acquired, release = asyncio.Event(), asyncio.Event()
    holding = asyncio.create_task(_hold_lock(lock=fixture.controller.context_lock, acquired=acquired, release=release))
    await acquired.wait()

    await asyncio.wait_for(
        fixture.operations.inject_fault(cell_id=_TRAINER_CELL_ID, mode=FailureMode.SIGKILL, sub_index=0),
        timeout=5.0,
    )

    assert fixture.worker_manager.calls == [
        ("inject_fault", (_TRAINER_CELL_ID,), {"mode": "sigkill", "worker_in_cell_index": 0})
    ]

    release.set()
    await holding


async def test_a_trainer_cells_suspend_reaches_the_worker_manager() -> None:
    # [FUNCTION DOC] While the lock is held, suspends a trainer cell with a five-second timeout;
    # asserts it completes through a stop-cells call on the worker manager and the engine provider
    # stopped nothing. Releases the lock at the end.
    """A trainer cell is none of the controller's business, and its lock would only stall the stop."""
    fixture = _make_fixture()
    acquired, release = asyncio.Event(), asyncio.Event()
    holding = asyncio.create_task(_hold_lock(lock=fixture.controller.context_lock, acquired=acquired, release=release))
    await acquired.wait()

    await asyncio.wait_for(fixture.operations.suspend(cell_id=_TRAINER_CELL_ID), timeout=5.0)

    assert fixture.worker_manager.calls == [("stop_cells", ([_TRAINER_CELL_ID],), {})]
    assert fixture.provider.stopped == []

    release.set()
    await holding


async def test_a_rollout_cell_the_controller_does_not_list_yet_still_goes_through_the_controller() -> None:
    # [FUNCTION DOC] Injects a fault into an unlisted rollout cell id; asserts the controller path
    # raises a key error (the cell is unknown to it) and the worker manager is never called.
    """Routing on live membership would kill an engine being replaced without the weight-update lock."""
    fixture = _make_fixture()

    with pytest.raises(KeyError):
        await asyncio.wait_for(
            fixture.operations.inject_fault(cell_id="engine-0-7", mode=FailureMode.SIGKILL, sub_index=0), timeout=5.0
        )

    assert fixture.worker_manager.calls == []


async def test_non_disruptive_operations_go_straight_through() -> None:
    # [FUNCTION DOC] With the lock held, lists cell infos and resumes a rollout cell, each under a
    # five-second timeout; asserts they complete with worker-manager calls to get cell infos then
    # start cells, and nothing stopped. Releases the lock at the end.
    """Cell reads and resumes do not wait for the weight-update lock."""
    fixture = _make_fixture()
    acquired, release = asyncio.Event(), asyncio.Event()
    holding = asyncio.create_task(_hold_lock(lock=fixture.controller.context_lock, acquired=acquired, release=release))
    await acquired.wait()

    await asyncio.wait_for(fixture.operations.cell_infos(pool_ids=["engine-0"]), timeout=5.0)
    await asyncio.wait_for(fixture.operations.resume(cell_id="engine-0-2"), timeout=5.0)

    assert [name for name, _, _ in fixture.worker_manager.calls] == ["get_cell_infos", "start_cells"]
    assert fixture.provider.stopped == []

    release.set()
    await holding


class TestRayCellOperationsProtocol:
    # [CLASS DOC] Tests how reads and resumes are forwarded to the worker manager.
    async def test_cell_infos_forwards_pool_ids_and_returns_the_actor_result(self) -> None:
        # [FUNCTION DOC] Presets the actor result, lists cell infos for two pool ids; asserts one
        # get-cell-infos call with the pool ids as a keyword argument and that the very same
        # result object is returned.
        """Cell info reads forward every pool ID by keyword and preserve the actor result."""
        fixture = _make_fixture()
        pool_ids = ["engine-0", "rollout-1"]
        actor_result = {"engine-0-2": SimpleNamespace(), "rollout-1-3": SimpleNamespace()}
        fixture.worker_manager.get_cell_infos.result = actor_result

        result = await fixture.operations.cell_infos(pool_ids=pool_ids)

        assert fixture.worker_manager.calls == [("get_cell_infos", (), {"pool_ids": pool_ids})]
        assert result is actor_result

    async def test_resume_starts_exactly_the_requested_cell(self) -> None:
        # [FUNCTION DOC] Resumes one cell; asserts a single start-cells call with a one-element
        # list of that cell id and no keyword arguments.
        """Resuming a cell sends exactly that cell ID in a one-element list."""
        fixture = _make_fixture()

        await fixture.operations.resume(cell_id="engine-0-2")

        assert fixture.worker_manager.calls == [("start_cells", (["engine-0-2"],), {})]


class TestRayCellOperationsInferenceControllerResolution:
    # [CLASS DOC] Tests that the inference controller is resolved lazily.
    async def test_the_inference_controller_is_resolved_only_when_a_disruptive_operation_needs_it(self) -> None:
        # [FUNCTION DOC] Builds operations with a resolver that asserts a readiness flag (initially
        # false) and counts calls. Lists infos and resumes while not ready and asserts zero
        # resolutions; then marks ready, suspends a rollout cell, and asserts exactly one
        # resolution and that the fake controller was asked to stop that cell.
        """Construction, reads, and resumes do not resolve the controller before a suspend needs it."""
        worker_manager = _RecordingWorkerManagerHandle()
        controller = _FakeInferenceController()
        ready = False
        resolution_count = 0

        def resolve_controller() -> _FakeInferenceController:
            # [FUNCTION DOC] HELPER: Fails if called before the readiness flag is set; otherwise
            # counts the call and returns the fake controller.
            nonlocal resolution_count
            assert ready, "the inference controller is not ready"
            resolution_count += 1
            return controller

        operations = RayCellOperations(
            worker_manager_handle=worker_manager,
            resolve_inference_controller=resolve_controller,
        )

        await operations.cell_infos(pool_ids=["engine-0"])
        await operations.resume(cell_id="engine-0-2")
        assert resolution_count == 0

        ready = True
        await operations.suspend(cell_id="engine-0-2")

        assert resolution_count == 1
        assert controller.suspended_cell_ids == ["engine-0-2"]


class _FakeInferenceController:
    # [CLASS DOC] HELPER: Minimal controller stand-in that records the cell ids it was asked to
    # stop.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty list of suspended cell ids.
        self.suspended_cell_ids: list[str] = []

    async def stop_cell_between_weight_updates(self, *, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Appends the cell id to the suspended list.
        self.suspended_cell_ids.append(cell_id)
