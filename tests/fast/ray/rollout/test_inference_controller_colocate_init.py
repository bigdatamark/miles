# [MODULE DOC]
# Unit tests for the controller's private routine that makes all cells ready for a weight update
# window (initializing colocated cells that still wait, and waiting until every cell is pending
# weights or serving). Uses fake cells with a small state machine, stub servers holding a cell
# mapping, and a controller built without its constructor. A fixture shrinks the poll interval to
# zero. The routine under test is always run under the controller's context lock by a helper at
# the bottom of the file.
from __future__ import annotations

import asyncio

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import inference_controller as inference_controller_module
from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.context_lock import ContextLock


class _FakeCell:
    # [CLASS DOC] HELPER: Fake server cell with a string state (uninitialized, initializing or
    # pending_weights). Its init counts the call, signals a started event, optionally waits for a
    # gate event, and moves an uninitialized cell to initializing; a test then calls the
    # become-ready method to simulate the engine finishing loading.
    def __init__(self, *, state: str = "uninitialized", init_gate: asyncio.Event | None = None):
        # [FUNCTION DOC] HELPER: Stores the initial state, an init counter of zero, an init-started
        # event and the optional init gate event.
        self.state = state
        self.init_count = 0
        self.init_started = asyncio.Event()
        self._init_gate = init_gate

    async def init(self) -> None:
        # [FUNCTION DOC] HELPER: Counts the call, sets the started event, waits for the gate event
        # if one exists, and moves the state from uninitialized to initializing (other states are
        # left unchanged).
        self.init_count += 1
        self.init_started.set()
        if self._init_gate is not None:
            await self._init_gate.wait()
        if self.state == "uninitialized":
            self.state = "initializing"

    def become_ready(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the state to pending_weights (engine loaded).
        self.state = "pending_weights"

    @property
    def is_uninitialized(self) -> bool:
        # [FUNCTION DOC] HELPER: Property, true only while the state is uninitialized.
        return self.state == "uninitialized"

    @property
    def is_pending_weights_or_serving(self) -> bool:
        # [FUNCTION DOC] HELPER: Property, true only while the state is pending_weights (the fake
        # has no serving state).
        return self.state == "pending_weights"


class _StubServer:
    # [CLASS DOC] HELPER: Minimal server stand-in exposing only a mutable cell mapping.
    def __init__(self, server_cells: dict):
        # [FUNCTION DOC] HELPER: Stores the cell mapping (shared, not copied, so tests can add cells
        # later).
        self.server_cells = server_cells


def _make_controller(servers: dict, *, colocate: bool) -> InferenceController:
    # [FUNCTION DOC] HELPER: Builds a controller without running its constructor, with test
    # arguments carrying the given colocate flag, the given servers mapping and a fresh context lock.
    controller = InferenceController.__new__(InferenceController)
    controller.args = make_args(colocate=colocate)
    controller.servers = servers
    controller.context_lock = ContextLock("InferenceController")
    return controller


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture that sets the controller module's cells-ready poll
    # interval to zero seconds so tests do not sleep between polls.
    monkeypatch.setattr(inference_controller_module, "CELLS_READY_POLL_INTERVAL_SECONDS", 0.0)


class TestEnsureCellsReady:
    # [CLASS DOC] Tests the controller's ensure-cells-ready routine: in colocate mode it calls init
    # on every uninitialized cell concurrently and re-scans for new cells, then polls (with the
    # lock released between polls) until every cell is pending weights or serving, raising a
    # timeout error past the deadline; in disaggregated mode it only waits.
    async def test_a_disaggregated_cell_is_never_initialized_by_the_window(self):
        # [FUNCTION DOC] Non-colocate, one cell already pending weights: the routine returns
        # within a second and the cell's init was never called.
        """Disaggregated cells initialize on arrival, so the window only has to wait for them."""
        cell = _FakeCell(state="pending_weights")
        controller = _make_controller({"default": _StubServer({"a": cell})}, colocate=False)

        await asyncio.wait_for(_ensure_ready_under_lock(controller), timeout=1)

        assert cell.init_count == 0

    async def test_a_disaggregated_engine_still_loading_is_waited_for(self):
        # [FUNCTION DOC] Non-colocate, one cell initializing: the routine is still running after
        # 20 ms; once the cell becomes ready it finishes, and init was never called on the cell.
        """Weights pushed into an engine that is not up yet would be lost."""
        cell = _FakeCell(state="initializing")
        controller = _make_controller({"default": _StubServer({"a": cell})}, colocate=False)

        task = asyncio.create_task(_ensure_ready_under_lock(controller))
        await asyncio.sleep(0.02)
        assert not task.done()
        cell.become_ready()
        await asyncio.wait_for(task, timeout=1)

        assert cell.init_count == 0

    async def test_it_initializes_every_colocated_cell_that_is_still_waiting(self):
        # [FUNCTION DOC] Colocate, two uninitialized cells in two different servers; after both
        # become ready the routine finishes and each cell's init ran exactly once.
        """This window is the only moment a colocated engine may claim gpu memory."""
        first, second = _FakeCell(), _FakeCell()
        controller = _make_controller(
            {"default": _StubServer({"a": first}), "frozen": _StubServer({"b": second})}, colocate=True
        )

        task = asyncio.create_task(_ensure_ready_under_lock(controller))
        await asyncio.sleep(0)
        first.become_ready()
        second.become_ready()
        await asyncio.wait_for(task, timeout=1)

        assert (first.init_count, second.init_count) == (1, 1)

    async def test_an_already_running_cell_is_not_initialized_again(self):
        # [FUNCTION DOC] Colocate, one cell already pending weights: the routine returns and the
        # cell's init was not called.
        """Re-initializing a live engine would tear down the memory it already holds."""
        running = _FakeCell(state="pending_weights")
        controller = _make_controller({"default": _StubServer({"a": running})}, colocate=True)

        await asyncio.wait_for(_ensure_ready_under_lock(controller), timeout=1)

        assert running.init_count == 0

    async def test_it_waits_until_the_engines_finished_loading(self):
        # [FUNCTION DOC] Colocate, one uninitialized cell: the routine is still running after 20 ms
        # and completes (within a second) only after the cell becomes ready.
        """Returning early would let the trainer push weights into an engine that is not up."""
        cell = _FakeCell()
        controller = _make_controller({"default": _StubServer({"a": cell})}, colocate=True)

        task = asyncio.create_task(_ensure_ready_under_lock(controller))
        await asyncio.sleep(0.02)
        assert not task.done()

        cell.become_ready()
        await asyncio.wait_for(task, timeout=1)

    async def test_a_cell_arriving_while_the_window_waits_is_initialized_on_the_next_pass(self):
        # [FUNCTION DOC] Colocate; waits until the first cell's init started, then adds a second
        # uninitialized cell to the server mapping and waits until its init also started. Makes both
        # ready, expects the routine to finish, and asserts the late cell's init ran exactly once.
        """Reconcile adds cells while the window has the lock released, and the next pass over the
        loop must pick them up rather than wait out the deadline on a list read once."""
        early = _FakeCell()
        srv = _StubServer({"a": early})
        controller = _make_controller({"default": srv}, colocate=True)

        task = asyncio.create_task(_ensure_ready_under_lock(controller))
        await asyncio.wait_for(early.init_started.wait(), timeout=1)
        late = _FakeCell()
        srv.server_cells["late"] = late
        await asyncio.wait_for(late.init_started.wait(), timeout=1)
        early.become_ready()
        late.become_ready()
        await asyncio.wait_for(task, timeout=1)

        assert late.init_count == 1

    async def test_the_lock_is_open_while_it_waits_so_the_tick_sweep_can_make_progress(self):
        # [FUNCTION DOC] Colocate; after 10 ms of waiting asserts the context lock is not held, then
        # takes the lock itself and makes the cell ready inside it, and expects the routine to
        # finish within five seconds.
        """Only the ticker moves a cell out of initializing, and the ticker needs this very lock."""
        cell = _FakeCell()
        controller = _make_controller({"default": _StubServer({"a": cell})}, colocate=True)

        task = asyncio.create_task(_ensure_ready_under_lock(controller))
        await asyncio.sleep(0.01)
        assert not controller.context_lock.locked

        async with controller.context_lock:
            cell.become_ready()
        await asyncio.wait_for(task, timeout=5)

    async def test_a_run_without_any_cells_yet_returns_immediately(self):
        # [FUNCTION DOC] Colocate with an empty cell mapping: the routine returns within a second.
        """The first window can legitimately find nothing to do."""
        controller = _make_controller({"default": _StubServer({})}, colocate=True)

        await asyncio.wait_for(_ensure_ready_under_lock(controller), timeout=1)

    async def test_the_cells_are_initialized_concurrently(self):
        # [FUNCTION DOC] Colocate, two cells whose init blocks on a shared gate. Asserts both inits
        # started before the gate opened (so they run concurrently); then opens the gate, makes both
        # ready, expects the routine to finish and each init to have run exactly once.
        """Activating a gate blocks until the engine reaches it, so serialising would add minutes per cell."""
        gate = asyncio.Event()
        first, second = _FakeCell(init_gate=gate), _FakeCell(init_gate=gate)
        controller = _make_controller({"default": _StubServer({"a": first, "b": second})}, colocate=True)

        task = asyncio.create_task(_ensure_ready_under_lock(controller))
        await asyncio.wait_for(asyncio.gather(first.init_started.wait(), second.init_started.wait()), timeout=1.0)
        gate.set()
        for cell in (first, second):
            cell.become_ready()
        await asyncio.wait_for(task, timeout=1.0)

        assert (first.init_count, second.init_count) == (1, 1)

    async def test_an_engine_that_never_comes_up_eventually_times_out(self, monkeypatch):
        # [FUNCTION DOC] Sets the cells-ready timeout constant to zero with a cell stuck in
        # initializing (non-colocate) and expects the routine to raise a timeout error.
        """A silent hang here would stall training with no explanation."""
        monkeypatch.setattr(inference_controller_module, "CELLS_READY_TIMEOUT_SECONDS", 0.0)
        cell = _FakeCell(state="initializing")
        controller = _make_controller({"default": _StubServer({"a": cell})}, colocate=False)

        with pytest.raises(TimeoutError):
            await _ensure_ready_under_lock(controller)


async def _ensure_ready_under_lock(controller: InferenceController) -> None:
    # [FUNCTION DOC] HELPER: Takes the controller's context lock and runs the private
    # ensure-cells-ready routine inside it. Returns nothing; propagates its errors (e.g. timeout).
    async with controller.context_lock:
        await controller._ensure_cells_ready()
