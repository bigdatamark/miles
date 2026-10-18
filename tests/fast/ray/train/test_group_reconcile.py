# [MODULE DOC] Async tests for the trainer controller's cell reconciliation, its public cell
# inventory/status view, and the wait for the expected number of cells.
# The controller is built without running its constructor: only the attributes the code under test
# reads are filled in (arguments namespace, role, pool id, a provider from the shared conftest, an
# empty cell mapping, an active activeness tracker). Cell observations are fabricated cell-info
# records whose worker names and workers hash are controlled by the test. All tests run under the
# async pytest marker. The wait tests replace the retry utility's clock and sleep with a fake clock.
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest
from tests.fast.ray.train.conftest import make_provider

from miles.ray.specs.train import compute_trainer_pool_id
from miles.ray.train.group import TrainerController
from miles.utils import retry_utils
from miles.utils.ft_utils.api_server.models import CellStatus, TriState
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_provider.base import CellInfo

pytestmark = pytest.mark.asyncio

_POOL_ID = compute_trainer_pool_id("actor")


def _cell_id(cell_index: int) -> str:
    # [FUNCTION DOC] HELPER: Returns the cell id string for the given cell index within the fixed
    # actor trainer pool used by this module.
    return compute_cell_id(pool_id=_POOL_ID, cell_index=cell_index)


def _make_controller(*, num_cells: int = 2, indep_dp: bool = False) -> TrainerController:
    # [FUNCTION DOC] HELPER: Builds a bare trainer controller for the actor role, bypassing its
    # constructor. The arguments namespace requests one node with num_cells GPUs per node, no
    # tensor/pipeline/context parallelism and the megatron backend; indep_dp is passed through.
    # No reference or teacher model, no health checker config, an active activeness tracker, a
    # provider from the conftest factory and an empty cell mapping. Returns the controller.
    group = object.__new__(TrainerController)
    group.args = SimpleNamespace(
        indep_dp=indep_dp,
        actor_num_nodes=1,
        actor_num_gpus_per_node=num_cells,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        train_backend="megatron",
    )
    group._role = "actor"
    group._with_ref = False
    group._with_opd_teacher = False
    group._pool_id = _POOL_ID
    group._health_checker_config = None
    group._health_checker_activeness = ActivenessTracker(active=True)
    group._provider = make_provider()
    group._cells_by_id = {}
    return group


def _make_cell_info(cell_index: int, *, workers_hash: str = "pseudo-hash-1") -> CellInfo:
    # [FUNCTION DOC] HELPER: Builds an alive cell-info record for the cell index, with one worker
    # name, the given workers hash (the generation stamp; default is a fixed pseudo hash) and
    # metadata holding the actor role and the cell index. Returns the record.
    return CellInfo(
        cell_id=_cell_id(cell_index),
        pool_id=_POOL_ID,
        alive=True,
        worker_names=[compute_worker_name(pool_id=_POOL_ID, cell_index=cell_index)],
        workers_hash=workers_hash,
        meta={"role": "actor", "cell_index": cell_index},
    )


class TestReconcile:
    # [CLASS DOC] Tests of the controller's reconcile step, which adds, keeps, replaces or drops a
    # cell given the latest observation (or absence) reported for a cell id.
    async def test_an_observed_cell_is_added(self):
        # [FUNCTION DOC] Reconciles one observed cell into an empty controller and asserts the
        # controller then holds exactly that cell, identified by index zero.
        """The group learns about its cells from the manager instead of creating them."""
        group = _make_controller()

        await group._reconcile(_cell_id(0), _make_cell_info(0))

        assert [cell.cell_index for cell in group._cells] == [0]

    async def test_a_disappeared_cell_is_dropped(self):
        # [FUNCTION DOC] Adds a cell, then reconciles the same id with no observation, and asserts
        # the controller's cell list becomes empty.
        """A cell the manager no longer reports must stop being trained."""
        group = _make_controller()
        await group._reconcile(_cell_id(1), _make_cell_info(1))

        await group._reconcile(_cell_id(1), None)

        assert group._cells == []

    async def test_reobserving_a_known_cell_keeps_the_same_object(self):
        # [FUNCTION DOC] Reconciles the same cell twice with an identical observation and asserts
        # the cell object is identical (not recreated) after the second pass.
        """Recreating the cell would throw away its state machine and health checker."""
        group = _make_controller()
        await group._reconcile(_cell_id(0), _make_cell_info(0))
        first = group._cells[0]

        await group._reconcile(_cell_id(0), _make_cell_info(0))

        assert group._cells[0] is first

    async def test_a_relaunched_cell_is_replaced(self):
        # [FUNCTION DOC] Reconciles a cell, then the same id with a different workers hash, and
        # asserts the stored cell is a new object carrying the new hash.
        """A new generation hands out new actor handles, so keeping the old object would use dead ones."""
        group = _make_controller()
        await group._reconcile(_cell_id(0), _make_cell_info(0))
        first = group._cells[0]

        await group._reconcile(_cell_id(0), _make_cell_info(0, workers_hash="pseudo-hash-2"))

        assert group._cells[0] is not first
        assert group._cells[0].workers_hash == "pseudo-hash-2"

    async def test_a_dropped_cell_has_its_health_checker_stopped(self):
        # [FUNCTION DOC] Adds a cell, keeps its health checker, drops the cell by reconciling with
        # no observation, and asserts the checker was stopped.
        """A leaked health checker keeps heartbeating a dead actor and logs a stacktrace every interval."""
        group = _make_controller()
        await group._reconcile(_cell_id(1), _make_cell_info(1))
        health_checker = group._cells[0].health_checker

        await group._reconcile(_cell_id(1), None)

        assert health_checker.stopped

    async def test_a_replaced_cell_has_its_old_health_checker_stopped(self):
        # [FUNCTION DOC] Replaces a cell via a changed workers hash and asserts the old cell's
        # health checker is stopped while the new cell's checker is still running.
        """The replace path removes before adding, so the superseded checker must be stopped too."""
        group = _make_controller()
        await group._reconcile(_cell_id(0), _make_cell_info(0))
        old_health_checker = group._cells[0].health_checker

        await group._reconcile(_cell_id(0), _make_cell_info(0, workers_hash="pseudo-hash-2"))

        assert old_health_checker.stopped
        assert not group._cells[0].health_checker.stopped

    async def test_cells_are_ordered_by_index_whatever_the_arrival_order(self):
        # [FUNCTION DOC] Reconciles three cells in the order 2, 0, 1 and asserts the controller's
        # cell list is sorted by cell index (0, 1, 2).
        """Independent DP ranks are derived from position, so order must be stable."""
        group = _make_controller(num_cells=3)

        for cell_index in [2, 0, 1]:
            await group._reconcile(_cell_id(cell_index), _make_cell_info(cell_index))

        assert [cell.cell_index for cell in group._cells] == [0, 1, 2]


def _healthy_condition(status: CellStatus) -> tuple[TriState, str | None]:
    # [FUNCTION DOC] HELPER: Extracts the single condition of type "Healthy" from a cell status
    # (unpacking fails if there is not exactly one) and returns its tri-state status and reason.
    [healthy] = [condition for condition in status.conditions if condition.type == "Healthy"]
    return healthy.status, healthy.reason


class TestPublicCellInventory:
    # [CLASS DOC] Tests of the controller's public inventory properties (pool id, expected and
    # actual cell counts, cell ids) and of the per-cell statuses it reports.
    async def test_the_inventory_and_statuses_follow_the_reconciled_cells(self):
        # [FUNCTION DOC] With independent data parallelism and three cells reconciled out of order,
        # marks the cell at position 1 as errored, then asserts the pool id, expected and actual
        # cell counts (3), and the sorted cell ids. Fetches statuses and asserts they are keyed by
        # those ids, all in the Running phase, with the Healthy condition true (no reason) for
        # cells 0 and 2 and false with reason ExecutionErrored for cell 1.
        """The FT controller heals by these names and statuses, so a stale or mis-ordered view heals the wrong cell."""
        group = _make_controller(num_cells=3, indep_dp=True)
        for cell_index in [2, 0, 1]:
            await group._reconcile(_cell_id(cell_index), _make_cell_info(cell_index))
        group._cells[1]._mark_as_errored()

        assert group.pool_id == _POOL_ID
        assert group.expected_num_cells == 3
        assert group.num_cells == 3
        assert group.cell_ids == [_cell_id(cell_index) for cell_index in range(3)]

        statuses = await group.get_cell_statuses()
        assert sorted(statuses) == group.cell_ids
        assert [status.phase for status in statuses.values()] == ["Running", "Running", "Running"]
        assert _healthy_condition(statuses[_cell_id(0)]) == (TriState.TRUE, None)
        assert _healthy_condition(statuses[_cell_id(1)]) == (TriState.FALSE, "ExecutionErrored")
        assert _healthy_condition(statuses[_cell_id(2)]) == (TriState.TRUE, None)

    async def test_each_status_carries_the_generation_it_describes(self):
        # [FUNCTION DOC] Reconciles one cell with a given workers hash and asserts its reported
        # status carries that same hash.
        """The api server joins this with a separately polled cell listing, so an unstamped status can mislead."""
        group = _make_controller(num_cells=1, indep_dp=True)
        await group._reconcile(_cell_id(0), _make_cell_info(0, workers_hash="hash-9"))

        statuses = await group.get_cell_statuses()

        assert statuses[_cell_id(0)].workers_hash == "hash-9"

    async def test_each_cell_is_stamped_with_its_own_generation(self):
        # [FUNCTION DOC] Reconciles two cells with different workers hashes and asserts each status
        # carries its own cell's hash rather than a neighbour's.
        """One cell relaunches without the others, so a stamp taken from a neighbour would clear the wrong verdict."""
        group = _make_controller(num_cells=2, indep_dp=True)
        await group._reconcile(_cell_id(0), _make_cell_info(0, workers_hash="hash-9"))
        await group._reconcile(_cell_id(1), _make_cell_info(1, workers_hash="hash-10"))

        statuses = await group.get_cell_statuses()

        assert (statuses[_cell_id(0)].workers_hash, statuses[_cell_id(1)].workers_hash) == (
            "hash-9",
            "hash-10",
        )

    async def test_a_relaunched_cell_reports_the_new_generation(self):
        # [FUNCTION DOC] Reconciles a cell, then re-reconciles it with a new workers hash, and
        # asserts the reported status carries the new hash.
        """The stamp is what tells the api server the previous process' verdict no longer applies."""
        group = _make_controller(num_cells=1, indep_dp=True)
        await group._reconcile(_cell_id(0), _make_cell_info(0, workers_hash="hash-9"))

        await group._reconcile(_cell_id(0), _make_cell_info(0, workers_hash="hash-10"))

        assert (await group.get_cell_statuses())[_cell_id(0)].workers_hash == "hash-10"


class _AutoAdvancingClock:
    # [CLASS DOC] HELPER: Fake clock for the retry utility. Holds a current time (starts at zero),
    # a list of every sleep duration requested, and an optional async callback invoked after each
    # sleep with the running sleep count. Sleeping advances time instantly by the requested amount.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes time to zero, an empty sleep record, and no callback.
        self.now: float = 0.0
        self.sleeps: list[float] = []
        self.on_sleep: Callable[[int], Awaitable[None]] | None = None

    def monotonic(self) -> float:
        # [FUNCTION DOC] HELPER: Returns the fake current time.
        return self.now

    async def sleep(self, seconds: float) -> None:
        # [FUNCTION DOC] HELPER: Records the requested duration, advances the fake time by it
        # without really sleeping, then awaits the callback (if set) with the total number of
        # sleeps so far, letting a test inject state changes between retries.
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            await self.on_sleep(len(self.sleeps))


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> _AutoAdvancingClock:
    # [FUNCTION DOC] HELPER: Fixture that creates a fake clock and patches the retry utility
    # module's time source and asynchronous sleep to use it (undone by the monkeypatch fixture at
    # teardown). Returns the clock so tests can inspect sleeps or install a callback.
    clock = _AutoAdvancingClock()
    monkeypatch.setattr(retry_utils, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(retry_utils, "asyncio", SimpleNamespace(sleep=clock.sleep))
    return clock


class TestWaitExpectedNumCells:
    # [CLASS DOC] Tests of the controller's wait for the expected number of cells, which retries
    # with sleeps until enough cells are reconciled or a timeout elapses (using the fake clock).
    async def test_waiting_keeps_polling_until_the_late_cells_are_observed(self, fake_clock: _AutoAdvancingClock):
        # [FUNCTION DOC] Expects four cells but reconciles only one. A callback on the fake clock
        # reconciles the remaining three on the second sleep. Waits with a 600 second timeout and
        # asserts exactly two sleeps happened and all four cells are present in index order.
        """Training must not start against half a pool, so the wait retries until the missing cells arrive."""
        group = _make_controller(num_cells=4, indep_dp=True)
        await group._reconcile(_cell_id(0), _make_cell_info(0))

        async def _add_remaining_cells_on_the_second_sleep(sleep_count: int) -> None:
            # [FUNCTION DOC] HELPER: Sleep callback; when the running sleep count is two,
            # reconciles cells 1 through 3 into the controller. Does nothing on other counts.
            if sleep_count == 2:
                for cell_index in range(1, 4):
                    await group._reconcile(_cell_id(cell_index), _make_cell_info(cell_index))

        fake_clock.on_sleep = _add_remaining_cells_on_the_second_sleep

        await group._wait_expected_num_cells(timeout=600.0)

        assert len(fake_clock.sleeps) == 2
        assert [cell.cell_index for cell in group._cells] == [0, 1, 2, 3]

    async def test_waiting_returns_immediately_when_every_cell_is_already_observed(
        self, fake_clock: _AutoAdvancingClock
    ):
        # [FUNCTION DOC] Reconciles all four expected cells first, waits, and asserts the fake
        # clock recorded no sleeps.
        """A complete pool must not cost a single retry sleep."""
        group = _make_controller(num_cells=4, indep_dp=True)
        for cell_index in range(4):
            await group._reconcile(_cell_id(cell_index), _make_cell_info(cell_index))

        await group._wait_expected_num_cells(timeout=600.0)

        assert fake_clock.sleeps == []

    async def test_waiting_gives_up_when_cells_never_appear(self, fake_clock: _AutoAdvancingClock):
        # [FUNCTION DOC] With no cells ever reconciled and a 10 second timeout, asserts the wait
        # raises a timeout error, that it slept at least twice (so it retried), and that the fake
        # time never exceeded the timeout.
        """A silent hang here would look like a stuck first step, so the wait retries and then times out."""
        group = _make_controller(num_cells=4, indep_dp=True)

        with pytest.raises(TimeoutError):
            await group._wait_expected_num_cells(timeout=10.0)

        assert len(fake_clock.sleeps) >= 2
        assert fake_clock.now <= 10.0
