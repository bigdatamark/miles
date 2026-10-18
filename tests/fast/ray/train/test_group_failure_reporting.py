# [MODULE DOC]
# Tests for how the trainer controller reports and classifies failures. Controllers are built
# without running their constructor, directly over hand-made cells (dummy ray actors from the
# shared conftest, or recording stubs). Covered: a lone failing cell makes train, save, export
# and lifecycle calls fail fast with a non-retryable error that keeps the worker error as its
# cause; the recoverability rule counts only alive cells (healing and errored cells do not
# count); save moves on to the next alive cell; the terminal decision log says give up or retry
# consistently with the raised error and names the cells behind it; a controller without cells
# fails fast; and offload and onload bracket health checking by switching its activeness off and on.
import logging
from types import SimpleNamespace

import pytest
import ray
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_alive_cell, make_cell

from miles.backends.training_utils.types import TrainStepOutcome
from miles.ray.train.group import TrainerController
from miles.utils import object_store
from miles.utils.data import RolloutDataPack
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.object_store import _MooncakeStoreObjectRef
from miles.utils.retry_utils import NonRetryableError


async def _noop_run_after_step(**kwargs) -> None:
    # [FUNCTION DOC] HELPER: Stand-in for the fault test action runner after a step; accepts any
    # keyword arguments and does nothing.
    return None


pytestmark = pytest.mark.asyncio

_DUMMY_DATA_PACK = RolloutDataPack(sample_indices=[0], data_ref=_MooncakeStoreObjectRef(payload="data"))


@pytest.fixture(autouse=True)
def _object_store_of_inited_controller() -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture. Initializes the object store singleton with the ray
    # backends (without contributing a storage segment), standing in for what a real controller
    # init would do, because these controllers are built without running init.
    """init() mints the store these controllers are faked past, and the failure path frees refs through it."""
    object_store.init_instance(
        SimpleNamespace(object_store_backend="ray", worker_comm_backend="ray"), contribute_segment=False
    )


def _make_controller(cells: list) -> TrainerController:
    # [FUNCTION DOC] HELPER: Creates a trainer controller without running its constructor and
    # sets only what the tested paths read: the cells-by-id mapping built from the given cells, an
    # argument object with the event analyzer off and no debug event data, no witness allocator,
    # quorum id 0, an active health checker activeness tracker, and a fault test action executor
    # stub whose after-step runner does nothing. Returns the controller.
    group = object.__new__(TrainerController)
    group._cells_by_id = {cell.cell_id: cell for cell in cells}
    group.args = SimpleNamespace(enable_event_analyzer=False, save_debug_event_data=None)
    group._witness_allocator = None
    group._indep_dp_quorum_id = 0
    group._health_checker_activeness = ActivenessTracker(active=True)
    group._test_action_executor = SimpleNamespace(run_after_step=_noop_run_after_step)
    return group


def _make_failing_controller(fn_name: str) -> TrainerController:
    # [FUNCTION DOC] HELPER: Builds a controller over a single alive cell whose dummy actors are
    # set to fail the given worker method.
    cell = make_alive_cell(0, alive_cell_indices=[0])
    for handle in get_raw_actor_handles(cell):
        ray.get(handle.set_fail_methods.remote([fn_name]))
    return _make_controller([cell])


class TestSingleCellFailsFast:
    # [CLASS DOC] Tests that a controller with a single failing cell fails with a non-retryable error.
    async def test_train_does_not_retry_when_no_cell_is_left(self):
        # [FUNCTION DOC] Training on a controller whose only cell fails train raises a
        # non-retryable error (no retry is attempted).
        """A lone dead cell can never be healed, so retrying only delays the crash."""
        group = _make_failing_controller("train")

        with pytest.raises(NonRetryableError):
            await group.train(3, _DUMMY_DATA_PACK)

    async def test_train_keeps_the_original_failure_as_the_cause(self):
        # [FUNCTION DOC] The non-retryable error raised by train has the worker's injected failure
        # message in its cause.
        """Without the cause the driver traceback says nothing about why training died."""
        group = _make_failing_controller("train")

        with pytest.raises(NonRetryableError) as excinfo:
            await group.train(3, _DUMMY_DATA_PACK)

        assert "Injected failure in train" in str(excinfo.value.__cause__)

    async def test_save_model_does_not_retry_when_no_cell_is_left(self):
        # [FUNCTION DOC] Saving the model on a controller whose only cell fails save raises a
        # non-retryable error.
        """The save path shares the retry wrapper and must fail fast too."""
        group = _make_failing_controller("save_model")

        with pytest.raises(NonRetryableError):
            await group.save_model(3)


class TestLifecycleCallsAreNotSilent:
    # [CLASS DOC] Tests that lifecycle calls (onload, offload, clear memory) surface a lost last cell.
    @pytest.mark.parametrize(
        ("method_name", "actor_fn_name"),
        [("onload", "wake_up"), ("offload", "sleep"), ("clear_memory", "clear_memory")],
    )
    async def test_a_lost_last_cell_is_reported(self, method_name, actor_fn_name):
        # [FUNCTION DOC] Parametrized over (onload, wake up), (offload, sleep) and (clear memory,
        # clear memory). When the only cell fails the underlying worker method, the controller
        # method raises a non-retryable error whose cause mentions that injected failure.
        """Swallowing this hides the real error until an unrelated call fails much later."""
        group = _make_failing_controller(actor_fn_name)

        with pytest.raises(NonRetryableError) as excinfo:
            await getattr(group, method_name)()

        assert f"Injected failure in {actor_fn_name}" in str(excinfo.value.__cause__)


class TestOnlyAliveCellsKeepTheControllerRetryable:
    # [CLASS DOC] Tests the retryable versus fatal classification of a failed train attempt, which
    # depends on whether any other cell is still alive.
    async def test_a_failed_attempt_is_retryable_while_another_cell_is_still_alive(self):
        # [FUNCTION DOC] Two alive cells; the first is marked errored and the attempt snapshot
        # holds only it with a value error result. Checking the attempt raises a runtime error that
        # is not the non-retryable kind, and the surviving cell stays alive.
        """The next attempt runs on the surviving cell, so the failure must stay retryable, not fatal."""
        errored_cell = make_alive_cell(0, alive_cell_indices=[0, 1])
        surviving_cell = make_alive_cell(1, alive_cell_indices=[0, 1])
        group = _make_controller([errored_cell, surviving_cell])
        errored_cell._mark_as_errored()

        with pytest.raises(RuntimeError) as excinfo:
            group._check_train_one_attempt(
                snapshot_alive_cells=[errored_cell],
                results=[ValueError("Injected failure in train")],
            )

        assert not isinstance(excinfo.value, NonRetryableError)
        assert surviving_cell.is_alive

    async def test_a_failed_attempt_is_fatal_when_only_a_healing_cell_is_left(self):
        # [FUNCTION DOC] One alive cell (then marked errored) and one uninitialized (healing)
        # cell: checking the failed attempt raises a non-retryable error, and the healing cell is untouched.
        """The next attempt refreshes on alive cells alone, so calling a healing cell recoverable only stalls."""
        alive_cell = make_alive_cell(0, alive_cell_indices=[0])
        uninitialized_cell = make_cell(1)
        group = _make_controller([alive_cell, uninitialized_cell])
        alive_cell._mark_as_errored()

        with pytest.raises(NonRetryableError):
            group._check_train_one_attempt(
                snapshot_alive_cells=[alive_cell],
                results=[ValueError("Injected failure in train")],
            )

        assert uninitialized_cell.is_uninitialized

    async def test_a_failed_attempt_is_fatal_once_no_cell_can_come_back(self):
        # [FUNCTION DOC] A single alive cell marked errored: checking its failed attempt raises a
        # non-retryable error.
        """With every cell errored there is nothing left to heal, so the group must fail fast."""
        alive_cell = make_alive_cell(0, alive_cell_indices=[0])
        group = _make_controller([alive_cell])
        alive_cell._mark_as_errored()

        with pytest.raises(NonRetryableError):
            group._check_train_one_attempt(
                snapshot_alive_cells=[alive_cell],
                results=[ValueError("Injected failure in train")],
            )

    async def test_offload_fails_fast_when_the_last_alive_cell_is_lost(self):
        # [FUNCTION DOC] The only alive cell fails sleep while an uninitialized cell exists.
        # Offload raises a non-retryable error, the formerly alive cell is no longer alive, and the
        # uninitialized cell is still uninitialized (it is not counted as a survivor).
        """_refresh_cells accepts alive cells only, so a healing cell must not be reported as a survivor."""
        alive_cell = make_alive_cell(0, alive_cell_indices=[0])
        for handle in get_raw_actor_handles(alive_cell):
            ray.get(handle.set_fail_methods.remote(["sleep"]))
        uninitialized_cell = make_cell(1)
        group = _make_controller([alive_cell, uninitialized_cell])

        with pytest.raises(NonRetryableError):
            await group.offload()

        assert not alive_cell.is_alive
        assert uninitialized_cell.is_uninitialized


class TestIsRecoverableCountsOnlyAliveCells:
    # [CLASS DOC] Tests the controller's recoverability predicate over combinations of cell states.
    async def test_an_alive_cell_next_to_an_errored_one_keeps_the_group_recoverable(self):
        # [FUNCTION DOC] One alive plus one errored cell: recoverable.
        """One survivor can carry the next attempt, so the group must still be reported as retryable."""
        alive_cell = make_alive_cell(0, alive_cell_indices=[0, 1])
        errored_cell = make_alive_cell(1, alive_cell_indices=[0, 1])
        errored_cell._mark_as_errored()
        group = _make_controller([alive_cell, errored_cell])

        assert group._is_recoverable()

    async def test_a_healing_cell_next_to_an_alive_one_keeps_the_group_recoverable(self):
        # [FUNCTION DOC] One alive plus one uninitialized cell: recoverable.
        """A cell that is still coming back must not make a group with a survivor look doomed."""
        group = _make_controller([make_alive_cell(0, alive_cell_indices=[0]), make_cell(1)])

        assert group._is_recoverable()

    async def test_a_group_of_only_healing_cells_is_not_recoverable(self):
        # [FUNCTION DOC] Two uninitialized cells: not recoverable.
        """The next attempt refreshes on alive cells alone, so healing cells alone cannot rescue the group."""
        group = _make_controller([make_cell(0), make_cell(1)])

        assert not group._is_recoverable()

    async def test_an_errored_cell_next_to_a_healing_one_is_not_recoverable(self):
        # [FUNCTION DOC] One errored plus one uninitialized cell: not recoverable.
        """Neither an errored nor a healing cell can run the next attempt, so the group must be fatal."""
        errored_cell = make_alive_cell(0, alive_cell_indices=[0])
        errored_cell._mark_as_errored()
        group = _make_controller([errored_cell, make_cell(1)])

        assert not group._is_recoverable()

    async def test_a_group_without_any_cell_is_not_recoverable(self):
        # [FUNCTION DOC] A controller with no cells: not recoverable.
        """With no cell at all there is nothing to heal from, so the group must never be called retryable."""
        group = _make_controller([])

        assert not group._is_recoverable()


class TestSaveModelNeedsAnAliveCellToRetryOnto:
    # [CLASS DOC] Tests save model fallback behavior across cells in different states.
    async def test_save_model_reports_the_original_failure_when_only_a_healing_cell_is_left(self):
        # [FUNCTION DOC] An alive cell failing save plus an uninitialized cell: saving raises a
        # non-retryable error whose message says all cells failed during the first-alive save, whose
        # cause is the injected failure, and the healing cell is untouched.
        """A healing cell cannot take the save, so retrying past it would bury the real error behind 'no alive cells'."""
        alive_cell = make_alive_cell(0, alive_cell_indices=[0])
        for handle in get_raw_actor_handles(alive_cell):
            ray.get(handle.set_fail_methods.remote(["save_model"]))
        uninitialized_cell = make_cell(1)
        group = _make_controller([alive_cell, uninitialized_cell])

        with pytest.raises(NonRetryableError) as excinfo:
            await group.save_model(3)

        assert "All cells failed during execute_first_alive#save_model" in str(excinfo.value)
        assert "Injected failure in save_model" in str(excinfo.value.__cause__)
        assert uninitialized_cell.is_uninitialized

    async def test_save_model_moves_on_to_the_next_alive_cell(self):
        # [FUNCTION DOC] Two alive cells where the first fails save: saving succeeds, the failing
        # cell is no longer alive, and the surviving cell's call log is exactly one save call.
        """A second alive cell keeps the save retryable, so the group must not give up when the first cell dies."""
        failing_cell = make_alive_cell(0, alive_cell_indices=[0, 1])
        surviving_cell = make_alive_cell(1, alive_cell_indices=[0, 1])
        for handle in get_raw_actor_handles(failing_cell):
            ray.get(handle.set_fail_methods.remote(["save_model"]))
        group = _make_controller([failing_cell, surviving_cell])

        await group.save_model(3)

        assert not failing_cell.is_alive
        for handle in get_raw_actor_handles(surviving_cell):
            assert [call[0] for call in ray.get(handle.get_calls.remote())] == ["save_model"]


class TestTerminalDecisionLogging:
    # [CLASS DOC] Tests that the decision logged for a failed attempt (give up or retry) agrees
    # with what the attempt check then raises. Log records are captured from the group logger.
    async def test_an_unrecoverable_attempt_is_logged_as_giving_up(self, caplog: pytest.LogCaptureFixture):
        # [FUNCTION DOC] A lone alive cell marked errored: the attempt check raises a non-retryable
        # error and an error-level record contains the give-up decision text.
        """A log that says retry while the raised error is fatal sends every reader looking for a retry that never came."""
        alive_cell = make_alive_cell(0, alive_cell_indices=[0])
        group = _make_controller([alive_cell])
        alive_cell._mark_as_errored()

        with caplog.at_level(logging.ERROR, logger="miles.ray.train.group"), pytest.raises(NonRetryableError):
            group._check_train_one_attempt(
                snapshot_alive_cells=[alive_cell],
                results=[ValueError("Injected failure in train")],
            )

        assert any("decision=give_up" in record.message for record in caplog.records)

    async def test_a_recoverable_attempt_is_logged_as_retrying(self, caplog: pytest.LogCaptureFixture):
        # [FUNCTION DOC] Two alive cells, the first errored: the attempt check raises a runtime
        # error and an error-level record contains the retry decision text.
        """A surviving cell makes the failure retryable, and the log must say so."""
        errored_cell = make_alive_cell(0, alive_cell_indices=[0, 1])
        surviving_cell = make_alive_cell(1, alive_cell_indices=[0, 1])
        group = _make_controller([errored_cell, surviving_cell])
        errored_cell._mark_as_errored()

        with caplog.at_level(logging.ERROR, logger="miles.ray.train.group"), pytest.raises(RuntimeError):
            group._check_train_one_attempt(
                snapshot_alive_cells=[errored_cell],
                results=[ValueError("Injected failure in train")],
            )

        assert any("decision=retry" in record.message for record in caplog.records)

    async def test_a_discarded_attempt_with_no_survivor_is_logged_as_giving_up(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] The attempt outcome computation is replaced to report cell 0 as discarded
        # while the only cell is errored. The attempt check raises a value error and a warning-level
        # record contains the give-up decision text.
        """The retry it asks for cannot happen once refresh finds no alive cell, so the log must not promise one."""
        alive_cell = make_alive_cell(0, alive_cell_indices=[0])
        group = _make_controller([alive_cell])
        alive_cell._mark_as_errored()
        monkeypatch.setattr(
            TrainerController,
            "_compute_attempt_outcomes",
            staticmethod(lambda cells, results: {"normal": [], "discarded": [0], "errored": []}),
        )

        with caplog.at_level(logging.WARNING, logger="miles.ray.train.group"), pytest.raises(ValueError):
            group._check_train_one_attempt(snapshot_alive_cells=[alive_cell], results=[None])

        assert any("decision=give_up" in record.message for record in caplog.records)

    async def test_a_fatal_cause_is_logged_as_giving_up_even_while_another_cell_survives(
        self, caplog: pytest.LogCaptureFixture
    ):
        # [FUNCTION DOC] Two alive cells with the first errored, but its result is itself a
        # non-retryable error. The attempt check raises a non-retryable error, the surviving cell
        # stays alive, an error-level record has the give-up decision, and none has the retry decision.
        """A cause that forbids retrying wins over the surviving cell, so a log reading the pool instead of the raised error lies."""
        errored_cell = make_alive_cell(0, alive_cell_indices=[0, 1])
        surviving_cell = make_alive_cell(1, alive_cell_indices=[0, 1])
        group = _make_controller([errored_cell, surviving_cell])
        errored_cell._mark_as_errored()

        with caplog.at_level(logging.ERROR, logger="miles.ray.train.group"), pytest.raises(NonRetryableError):
            group._check_train_one_attempt(
                snapshot_alive_cells=[errored_cell],
                results=[NonRetryableError("Injected fatal failure in train")],
            )

        assert surviving_cell.is_alive
        assert any("decision=give_up" in record.message for record in caplog.records)
        assert not any("decision=retry" in record.message for record in caplog.records)

    async def test_a_discarded_attempt_with_a_survivor_is_logged_as_retrying(self, caplog: pytest.LogCaptureFixture):
        # [FUNCTION DOC] One alive cell whose result is a discarded-should-retry step outcome: the
        # attempt check raises a value error, and a warning-level record has the retry decision but
        # none has the give-up decision.
        """A discarded step is replayed while a cell is still alive, so the log must announce that retry."""
        alive_cell = make_alive_cell(0, alive_cell_indices=[0])
        group = _make_controller([alive_cell])

        with caplog.at_level(logging.WARNING, logger="miles.ray.train.group"), pytest.raises(ValueError):
            group._check_train_one_attempt(
                snapshot_alive_cells=[alive_cell],
                results=[[SimpleNamespace(outcome=TrainStepOutcome.DISCARDED_SHOULD_RETRY)]],
            )

        assert any("decision=retry" in record.message for record in caplog.records)
        assert not any("decision=give_up" in record.message for record in caplog.records)

    async def test_the_terminal_decision_log_still_names_the_cells_behind_it(self, caplog: pytest.LogCaptureFixture):
        # [FUNCTION DOC] Two cells both errored with failed results: the attempt check raises a
        # non-retryable error, give-up records exist, and every one of them contains the text
        # listing errored cell indices 0 and 1.
        """A decision without the per-cell outcomes leaves an operator unable to tell which cells drove it."""
        errored_cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        group = _make_controller(errored_cells)
        for cell in errored_cells:
            cell._mark_as_errored()

        with caplog.at_level(logging.ERROR, logger="miles.ray.train.group"), pytest.raises(NonRetryableError):
            group._check_train_one_attempt(
                snapshot_alive_cells=errored_cells,
                results=[ValueError("Injected failure in train") for _ in errored_cells],
            )

        decision_records = [record.message for record in caplog.records if "decision=give_up" in record.message]
        assert decision_records
        assert all("errored=0,1" in message for message in decision_records)


class TestTrainWithoutAnyCell:
    # [CLASS DOC] Tests training on a controller whose cell set is empty.
    async def test_train_fails_fast_once_reconcile_has_dropped_every_cell(self):
        # [FUNCTION DOC] Training on a controller with no cells raises a non-retryable error whose
        # message says recovery is impossible when all cells are dead.
        """A pool the manager no longer reports can never come back on its own, so retrying only stalls the driver."""
        group = _make_controller([])

        with pytest.raises(NonRetryableError, match="Cannot recover when all cells are dead"):
            await group.train(3, _DUMMY_DATA_PACK)


class TestExportHf:
    # [CLASS DOC] Tests the hugging face export when workers fail.
    async def test_export_stops_once_no_cell_is_left_to_take_it(self):
        # [FUNCTION DOC] Exporting on a controller whose only cell fails export raises a
        # non-retryable error (no endless retry).
        """Retrying an export forever would hide the crash behind a run that never finishes the checkpoint."""
        group = _make_failing_controller("export_hf")

        with pytest.raises(NonRetryableError):
            await group.export_hf(3, "/ckpt/hf-3")


class _ActivenessRecordingCell:
    # [CLASS DOC] HELPER: Cell stub that is always alive and never uninitialized. Executing a
    # worker method records (method name, whether health checking was active at that moment) using
    # a shared activeness tracker assigned by the test. The cell index is parsed from the numeric
    # suffix of the cell id.
    def __init__(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Stores the cell id, parses the cell index from the text after the
        # last dash (raises a value error if it is not an integer), and starts with no tracker and
        # no recorded calls.
        self.cell_id = cell_id
        self.cell_index = int(cell_id.rsplit("-", 1)[1])
        self.tracker: ActivenessTracker | None = None
        self.calls: list[tuple[str, bool]] = []

    @property
    def is_alive(self) -> bool:
        # [FUNCTION DOC] HELPER: Property; always true.
        return True

    @property
    def is_uninitialized(self) -> bool:
        # [FUNCTION DOC] HELPER: Property; always false.
        return False

    async def execute(self, fn_name: str, **_kwargs) -> None:
        # [FUNCTION DOC] HELPER: Records the method name with the tracker's current active flag;
        # other keyword arguments are ignored. Requires the tracker to be set.
        self.calls.append((fn_name, self.tracker.get().active))


class TestOffloadOnloadBracketsHealthChecking:
    # [CLASS DOC] Tests that offload and onload switch health checking off for the window between them.
    async def test_health_checks_are_off_for_the_whole_sleep_and_wake_up_window(self):
        # [FUNCTION DOC] Two recording cells share the controller's activeness tracker. After
        # offload the tracker is inactive; after onload it is active again. Each cell recorded a
        # sleep then a wake up, both with health checking inactive at call time.
        """A probe that reaches a sleeping or half-woken worker recycles a perfectly healthy cell."""
        cells = [_ActivenessRecordingCell(f"trainer-actor-{cell_index}") for cell_index in range(2)]
        group = _make_controller(cells)
        for cell in cells:
            cell.tracker = group._health_checker_activeness

        await group.offload()
        assert not group._health_checker_activeness.get().active

        await group.onload()

        assert group._health_checker_activeness.get().active
        for cell in cells:
            assert cell.calls == [("sleep", False), ("wake_up", False)]


class TestMultipleCellsStillTolerateFailures:
    # [CLASS DOC] Tests that lifecycle calls tolerate individual cell failures when others survive.
    async def test_one_dead_cell_does_not_stop_the_lifecycle_call(self):
        # [FUNCTION DOC] With two alive cells where the first fails sleep, offload completes
        # without error, the first cell is no longer alive and the second is still alive.
        """Fault tolerance depends on surviving cells carrying on without the dead one."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        ray.get(get_raw_actor_handles(cells[0])[0].set_fail_methods.remote(["sleep"]))
        group = _make_controller(cells)

        await group.offload()

        assert not cells[0].is_alive
        assert cells[1].is_alive
