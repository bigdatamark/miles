# [MODULE DOC] Unit tests for the fault-tolerance test action machinery: building the launch
# argument fragment, loading and validating an action plan (inline JSON or a plan file, with
# filtering per executor kind), the controller-side executor that stops and resumes cells after a
# step, the actor-side executor that crashes a chosen rank, the orchestration-side executor that
# parks the run forever at a chosen step (and the sentinel it leaves behind), plan file read
# caching, and the restrictions on which training loops may be parked. Cell ids have the form
# pool id, dash, integer index. Several tests are marked as an ad hoc hack to be reverted after
# an argument refactor.
import json
import os
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from tests.e2e.ft.conftest_ft.execution import DEFAULT_TRAIN_SCRIPT

from miles.utils.test_utils import ft_test_actions
from miles.utils.test_utils.ft_test_actions import (
    _ACTOR_ACTIONS,
    _CONTROLLER_ACTIONS,
    _ORCHESTRATION_ACTIONS,
    PARKABLE_TRAIN_SCRIPT,
    SLEEP_FOREVER_AT_END_ACTION,
    FTTestAction,
    FTTestActionActorExecutor,
    FTTestActionControllerExecutor,
    FTTestActionOrchestrationExecutor,
    _load_actions,
    compute_ft_test_actions_arg,
    read_frozen_rollout_id,
    read_ft_test_actions,
    write_frozen_sentinel,
    write_ft_test_actions,
)

_POOL_ID = "trainer-engine-actor"


class TestComputeFTTestActionsArg:
    # [CLASS DOC] Tests of the helper that renders an action plan as a command line fragment.
    def test_the_launch_fragment_round_trips_the_complete_action_plan(self) -> None:
        # [FUNCTION DOC] Renders a plan with a stop action and a crash action (with rank and
        # attempt) and asserts the fragment starts with the flag and an opening single quote,
        # ends with a closing quote and a space, and that the text in between parses as JSON equal
        # to the original plan.
        """The launch fragment preserves its flag, quoted action plan, and trailing separator."""
        actions: list[dict[str, object]] = [
            {"at_rollout": 2, "action": "stop_cell_at_end", "cell_id": "trainer-engine-actor-0"},
            {
                "at_rollout": 5,
                "action": "crash_before_allreduce",
                "cell_id": "trainer-engine-actor-1",
                "rank": 3,
                "attempt": 1,
            },
        ]

        fragment = compute_ft_test_actions_arg(actions)

        prefix = "--ci-ft-test-actions '"
        assert fragment.startswith(prefix)
        assert fragment.endswith("' ")
        assert json.loads(fragment[len(prefix) : -2]) == actions


def _args(ci_ft_test_actions: object = None, **overrides: object) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds a fake argument namespace with the inline actions value (default
    # none), no actions path, an update weights interval of 1, and any keyword overrides applied
    # last.
    return SimpleNamespace(
        **{
            "ci_ft_test_actions": ci_ft_test_actions,
            "ci_ft_test_actions_path": None,
            "update_weights_interval": 1,
            **overrides,
        }
    )


def test_load_actions_returns_empty_when_attr_is_none() -> None:
    # [FUNCTION DOC] An unset inline plan loads as an empty list.
    """None ci_ft_test_actions yields an empty action list without parsing."""
    assert _load_actions(_args(None), _CONTROLLER_ACTIONS) == []


def test_load_actions_returns_empty_when_attr_is_empty_string() -> None:
    # [FUNCTION DOC] An empty-string inline plan loads as an empty list.
    """Empty-string ci_ft_test_actions is falsy and yields an empty list."""
    assert _load_actions(_args(""), _ACTOR_ACTIONS) == []


def test_load_actions_refuses_arguments_that_declare_no_plan_at_all() -> None:
    # [FUNCTION DOC] An argument namespace lacking both plan attributes makes loading raise an
    # attribute error.
    """Every run registers both flags, so arguments carrying neither are not a run's arguments."""
    with pytest.raises(AttributeError):
        _load_actions(SimpleNamespace(), _CONTROLLER_ACTIONS)


def test_load_actions_parses_single_crash_action_with_defaults() -> None:
    # [FUNCTION DOC] Loads one crash action JSON with only rollout, action and cell id for the
    # actor filter; asserts one action object of the action model type with those values and
    # defaults of rank 0 and attempt 0.
    """A single crash_before_allreduce action loads with the model's default fields."""
    raw = json.dumps([{"at_rollout": 3, "action": "crash_before_allreduce", "cell_id": "trainer-engine-actor-00002"}])
    actions = _load_actions(_args(raw), _ACTOR_ACTIONS)
    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, FTTestAction)
    assert action.at_rollout == 3
    assert action.action == "crash_before_allreduce"
    assert action.cell_id == "trainer-engine-actor-00002"
    assert action.rank == 0
    assert action.attempt == 0


def test_load_actions_filters_to_only_matching_actions() -> None:
    # [FUNCTION DOC] A plan with stop, crash and start actions: the controller filter returns the
    # stop and start actions in order; the actor filter returns only the crash action.
    """Mixed actions are filtered down to those whose action is in the filter set."""
    raw = json.dumps(
        [
            {"at_rollout": 1, "action": "stop_cell_at_end", "cell_id": "trainer-engine-actor-0"},
            {"at_rollout": 2, "action": "crash_before_allreduce", "cell_id": "trainer-engine-actor-1"},
            {"at_rollout": 3, "action": "start_cell_at_end", "cell_id": "trainer-engine-actor-0"},
        ]
    )
    group_actions = _load_actions(_args(raw), _CONTROLLER_ACTIONS)
    assert [a.action for a in group_actions] == ["stop_cell_at_end", "start_cell_at_end"]
    actor_actions = _load_actions(_args(raw), _ACTOR_ACTIONS)
    assert [a.action for a in actor_actions] == ["crash_before_allreduce"]


def test_load_actions_returns_empty_when_no_action_matches_filter() -> None:
    # [FUNCTION DOC] A crash action loaded with the controller filter yields an empty list.
    """Valid actions that fall outside the filter set produce an empty result."""
    raw = json.dumps([{"at_rollout": 1, "action": "crash_before_allreduce", "cell_id": "trainer-engine-actor-1"}])
    assert _load_actions(_args(raw), _CONTROLLER_ACTIONS) == []


def test_load_actions_rejects_extra_field() -> None:
    # [FUNCTION DOC] An action with an unknown extra field raises a validation error.
    """An unexpected JSON field is rejected because the model forbids extras."""
    raw = json.dumps(
        [{"at_rollout": 1, "action": "stop_cell_at_end", "cell_id": "trainer-engine-actor-0", "bogus": 5}]
    )
    with pytest.raises(ValidationError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


def test_load_actions_rejects_invalid_action_literal() -> None:
    # [FUNCTION DOC] An action name outside the allowed set raises a validation error.
    """An action string outside the allowed Literal set raises a validation error."""
    raw = json.dumps([{"at_rollout": 1, "action": "not_a_real_action", "cell_id": "trainer-engine-actor-0"}])
    with pytest.raises(ValidationError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


def test_load_actions_rejects_missing_cell_id() -> None:
    # [FUNCTION DOC] A stop action without a cell id raises a validation error.
    """cell_id is required, so an action that omits it fails to load instead of guessing a target."""
    raw = json.dumps([{"at_rollout": 1, "action": "stop_cell_at_end"}])
    with pytest.raises(ValidationError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


def test_load_actions_rejects_legacy_cell_index_field() -> None:
    # [FUNCTION DOC] An action using the retired cell index field raises a validation error.
    """The retired cell_index field is an extra field now, so stale JSON fails loudly."""
    raw = json.dumps([{"at_rollout": 1, "action": "stop_cell_at_end", "cell_index": -1}])
    with pytest.raises(ValidationError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


def test_load_actions_rejects_cell_id_without_index_suffix() -> None:
    # [FUNCTION DOC] A cell id with no trailing index raises a value error at load time.
    """A cell_id that carries no trailing index cannot be parsed and is rejected at load time."""
    raw = json.dumps([{"at_rollout": 1, "action": "stop_cell_at_end", "cell_id": "traineractor"}])
    with pytest.raises(ValueError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


def test_load_actions_rejects_cell_id_with_non_numeric_index() -> None:
    # [FUNCTION DOC] A cell id whose suffix after the last dash is not an integer raises a value
    # error at load time.
    """A cell_id whose suffix is not an integer is rejected at load time."""
    raw = json.dumps([{"at_rollout": 1, "action": "stop_cell_at_end", "cell_id": "trainer-engine-actor-last"}])
    with pytest.raises(ValueError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


def test_load_actions_validates_cell_id_of_actions_outside_the_filter() -> None:
    # [FUNCTION DOC] A crash action with a malformed cell id, loaded with the controller filter
    # that would exclude it, still raises a value error, so validation covers all actions.
    """Validation runs over every action, so a typo in another executor's action still fails here."""
    raw = json.dumps([{"at_rollout": 1, "action": "crash_before_allreduce", "cell_id": "bogus"}])
    with pytest.raises(ValueError):
        _load_actions(_args(raw), _CONTROLLER_ACTIONS)


class FakeController:
    # [CLASS DOC] HELPER: Fake pool controller exposing a pool id, the expected number of cells,
    # and a cell ids property that counts its reads and reports no cells until a configured
    # number of reads has happened.
    def __init__(self, num_cells: int, *, pool_id: str = _POOL_ID, observed_after_reads: int = 0) -> None:
        # [FUNCTION DOC] HELPER: Stores the pool id, the expected cell count, a zeroed read
        # counter and the number of initial reads that must see no cells.
        self.pool_id = pool_id
        self.expected_num_cells = num_cells
        self.cell_ids_reads = 0
        self._observed_after_reads = observed_after_reads

    @property
    def cell_ids(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Property. Increments the read counter; while the counter is within
        # the configured number of unobserved reads returns an empty list, afterwards returns the
        # ids pool id dash index for each index below the expected cell count.
        self.cell_ids_reads += 1
        if self.cell_ids_reads <= self._observed_after_reads:
            return []
        return [f"{self.pool_id}-{index}" for index in range(self.expected_num_cells)]


class FakeCellOperations:
    # [CLASS DOC] HELPER: Fake cell operations that record which cell ids were suspended (stopped
    # list) and resumed (started list).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the two empty lists.
        self.stopped: list[str] = []
        self.started: list[str] = []

    async def suspend(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Appends the cell id to the stopped list.
        self.stopped.append(cell_id)

    async def resume(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Appends the cell id to the started list.
        self.started.append(cell_id)


class TestRunAfterStep:
    # [CLASS DOC] Tests of the controller-side executor's after-step hook, which performs the
    # stop and start cell actions whose rollout matches the finished step.
    async def test_start_cell_that_is_never_observed_fails_at_the_bounded_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Async test. A controller that never reports the cell, a resume action,
        # and the observation timeout patched to zero: asserts the hook raises a timeout error
        # saying the cell was resumed but not observed yet, and that the resume was still issued.
        """A resumed cell that stays absent raises TimeoutError when the observation deadline expires."""
        operations = FakeCellOperations()
        controller = FakeController(num_cells=1, observed_after_reads=sys.maxsize)
        action = FTTestAction(at_rollout=3, action="start_cell_at_end", cell_id="trainer-engine-actor-0")
        executor = FTTestActionControllerExecutor(actions=[action], controller=controller, cell_operations=operations)
        monkeypatch.setattr(ft_test_actions, "_CELL_RESUME_OBSERVED_TIMEOUT_SECONDS", 0.0)

        with pytest.raises(TimeoutError, match="was resumed but is not observed yet"):
            await executor.run_after_step(3)

        assert operations.started == ["trainer-engine-actor-0"]

    @pytest.mark.asyncio
    async def test_stop_cell_fires_on_matching_rollout(self):
        # [FUNCTION DOC] A stop action at rollout 5 for cell 1: running the hook for rollout 5
        # suspends exactly that cell and resumes none.
        """stop_cell_at_end suspends the action's cell_id through the backend's operations."""
        operations = FakeCellOperations()
        action = FTTestAction(at_rollout=5, action="stop_cell_at_end", cell_id="trainer-engine-actor-1")
        executor = FTTestActionControllerExecutor(
            actions=[action], controller=FakeController(num_cells=3), cell_operations=operations
        )

        await executor.run_after_step(5)

        assert operations.stopped == ["trainer-engine-actor-1"]
        assert operations.started == []

    @pytest.mark.asyncio
    async def test_no_action_on_non_matching_rollout(self):
        # [FUNCTION DOC] The stop action is armed for rollout 5; running the hook for rollout 4
        # performs no operations.
        """run_after_step does nothing when no action's at_rollout matches the given rollout."""
        operations = FakeCellOperations()
        action = FTTestAction(at_rollout=5, action="stop_cell_at_end", cell_id="trainer-engine-actor-1")
        executor = FTTestActionControllerExecutor(
            actions=[action], controller=FakeController(num_cells=3), cell_operations=operations
        )

        await executor.run_after_step(4)

        assert operations.stopped == []
        assert operations.started == []

    @pytest.mark.asyncio
    async def test_start_cell_targets_the_named_cell(self):
        # [FUNCTION DOC] A start action for cell 2 at rollout 2 resumes exactly that cell id and
        # suspends none.
        """start_cell_at_end resumes exactly the cell_id the action names."""
        operations = FakeCellOperations()
        action = FTTestAction(at_rollout=2, action="start_cell_at_end", cell_id="trainer-engine-actor-2")
        executor = FTTestActionControllerExecutor(
            actions=[action], controller=FakeController(num_cells=3), cell_operations=operations
        )

        await executor.run_after_step(2)

        assert operations.started == ["trainer-engine-actor-2"]
        assert operations.stopped == []

    @pytest.mark.asyncio
    async def test_start_cell_does_not_return_until_the_controller_observes_the_cell(self):
        # [FUNCTION DOC] The controller hides its cells for the first read. After the hook returns
        # for a start action, asserts the cell was resumed and that the controller's cell ids
        # were read more than once, i.e. the hook polled until the cell appeared.
        """The next step reconfigures against what is observed, so returning early races the heal."""
        operations = FakeCellOperations()
        controller = FakeController(num_cells=2, observed_after_reads=1)
        action = FTTestAction(at_rollout=3, action="start_cell_at_end", cell_id="trainer-engine-actor-1")
        executor = FTTestActionControllerExecutor(actions=[action], controller=controller, cell_operations=operations)

        await executor.run_after_step(3)

        assert operations.started == ["trainer-engine-actor-1"]
        assert controller.cell_ids_reads > 1, "the resume returned on the read that still lacked the cell"

    @pytest.mark.asyncio
    async def test_stop_cell_does_not_wait_for_anything_to_be_observed(self):
        # [FUNCTION DOC] A stop action suspends the cell and the controller's cell ids are never
        # read (zero reads).
        """Only the resume has a cell to wait for; making suspend wait would hang on the cell it removed."""
        operations = FakeCellOperations()
        controller = FakeController(num_cells=2)
        action = FTTestAction(at_rollout=3, action="stop_cell_at_end", cell_id="trainer-engine-actor-1")
        executor = FTTestActionControllerExecutor(actions=[action], controller=controller, cell_operations=operations)

        await executor.run_after_step(3)

        assert operations.stopped == ["trainer-engine-actor-1"]
        assert controller.cell_ids_reads == 0

    @pytest.mark.asyncio
    async def test_start_cell_after_that_cell_was_dropped_still_targets_it(self):
        # [FUNCTION DOC] A start action for cell 1 resumes that named cell and suspends none,
        # regardless of the cell having been stopped earlier.
        """A stopped cell no longer being live does not change the cell_id the action names."""
        operations = FakeCellOperations()
        action = FTTestAction(at_rollout=3, action="start_cell_at_end", cell_id="trainer-engine-actor-1")
        executor = FTTestActionControllerExecutor(
            actions=[action], controller=FakeController(num_cells=2), cell_operations=operations
        )

        await executor.run_after_step(3)

        assert operations.started == ["trainer-engine-actor-1"]
        assert operations.stopped == []

    @pytest.mark.asyncio
    async def test_two_actions_same_rollout_both_fire(self):
        # [FUNCTION DOC] A stop action for cell 0 and a start action for cell 2 at the same
        # rollout both run: cell 0 is suspended and cell 2 is resumed.
        """Two actions sharing the same rollout both dispatch to their respective cell operations."""
        operations = FakeCellOperations()
        stop_action = FTTestAction(at_rollout=7, action="stop_cell_at_end", cell_id="trainer-engine-actor-0")
        start_action = FTTestAction(at_rollout=7, action="start_cell_at_end", cell_id="trainer-engine-actor-2")
        executor = FTTestActionControllerExecutor(
            actions=[stop_action, start_action], controller=FakeController(num_cells=3), cell_operations=operations
        )

        await executor.run_after_step(7)

        assert operations.stopped == ["trainer-engine-actor-0"]
        assert operations.started == ["trainer-engine-actor-2"]

    @pytest.mark.asyncio
    async def test_empty_actions_is_noop(self):
        # [FUNCTION DOC] An executor with no actions performs no operations for any step.
        """An executor with no actions performs no cell operations."""
        operations = FakeCellOperations()
        executor = FTTestActionControllerExecutor(
            actions=[], controller=FakeController(num_cells=3), cell_operations=operations
        )

        await executor.run_after_step(5)

        assert operations.stopped == []
        assert operations.started == []

    @pytest.mark.asyncio
    async def test_action_naming_another_spec_raises(self):
        # [FUNCTION DOC] A stop action whose cell id belongs to a different pool than the
        # controller's raises an assertion error and suspends nothing. (The test name says spec
        # but the code concept is the pool id.)
        """An action aimed at a different spec is a misconfiguration and must fail, not silently no-op."""
        operations = FakeCellOperations()
        action = FTTestAction(at_rollout=1, action="stop_cell_at_end", cell_id="rollout-engine-0")
        executor = FTTestActionControllerExecutor(
            actions=[action], controller=FakeController(num_cells=3), cell_operations=operations
        )

        with pytest.raises(AssertionError):
            await executor.run_after_step(1)

        assert operations.stopped == []

    @pytest.mark.asyncio
    async def test_action_index_beyond_expected_num_cells_raises(self):
        # [FUNCTION DOC] A stop action for index 9 of a three-cell pool raises an assertion error
        # and suspends nothing.
        """A cell index the group can never have is a misconfiguration and must fail at dispatch."""
        operations = FakeCellOperations()
        action = FTTestAction(at_rollout=1, action="stop_cell_at_end", cell_id="trainer-engine-actor-9")
        executor = FTTestActionControllerExecutor(
            actions=[action], controller=FakeController(num_cells=3), cell_operations=operations
        )

        with pytest.raises(AssertionError):
            await executor.run_after_step(1)

        assert operations.stopped == []

    @pytest.mark.asyncio
    async def test_the_index_one_past_the_last_cell_is_rejected(self):
        # [FUNCTION DOC] Boundary check: index 3 of a three-cell pool (indices 0 to 2) raises an
        # assertion error and suspends nothing.
        """Cell indices are half-open, so index N of an N-cell pool is the easiest off-by-one to write in CI config."""
        operations = FakeCellOperations()
        action = FTTestAction(at_rollout=1, action="stop_cell_at_end", cell_id="trainer-engine-actor-3")
        executor = FTTestActionControllerExecutor(
            actions=[action], controller=FakeController(num_cells=3), cell_operations=operations
        )

        with pytest.raises(AssertionError):
            await executor.run_after_step(1)

        assert operations.stopped == []

    @pytest.mark.asyncio
    async def test_a_rejected_stop_propagates_and_the_later_action_never_fires(self):
        # [FUNCTION DOC] Uses cell operations whose suspend always raises a runtime error. With a
        # stop action followed by a start action at the same rollout, asserts the error propagates
        # out of the hook and neither a stop nor a start was recorded (the later action did not
        # run).
        """Carrying on after the requested transition failed turns a broken scenario into a green run."""

        class _RejectingOperations(FakeCellOperations):
            # [CLASS DOC] HELPER: Cell operations whose suspend always fails.
            async def suspend(self, cell_id: str) -> None:
                # [FUNCTION DOC] HELPER: Always raises a runtime error saying the worker manager
                # rejected the stop.
                raise RuntimeError("worker manager rejected the stop")

        operations = _RejectingOperations()
        stop_action = FTTestAction(at_rollout=7, action="stop_cell_at_end", cell_id="trainer-engine-actor-0")
        start_action = FTTestAction(at_rollout=7, action="start_cell_at_end", cell_id="trainer-engine-actor-2")
        executor = FTTestActionControllerExecutor(
            actions=[stop_action, start_action], controller=FakeController(num_cells=3), cell_operations=operations
        )

        with pytest.raises(RuntimeError, match="rejected the stop"):
            await executor.run_after_step(7)

        assert operations.stopped == []
        assert operations.started == []


_CRASH_ACTION = FTTestAction(
    at_rollout=4, action="crash_before_allreduce", cell_id="trainer-engine-actor-1", rank=0, attempt=0
)


def _make_actor_executor(*, cell_id: str, rank: int) -> FTTestActionActorExecutor:
    # [FUNCTION DOC] HELPER: Creates an actor-side executor armed with the module's single crash
    # action (rollout 4, cell 1 of the trainer pool, rank 0, attempt 0), for a worker with the
    # given cell id and rank.
    return FTTestActionActorExecutor(actions=[_CRASH_ACTION], cell_id=cell_id, rank=rank)


@pytest.fixture
def recorded_exit_codes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    # [FUNCTION DOC] HELPER: Fixture that replaces the operating system hard exit function with a
    # recorder (reverted at teardown) and returns the list of exit codes it was called with, so
    # the process does not actually exit.
    exit_codes: list[int] = []
    monkeypatch.setattr(os, "_exit", lambda code: exit_codes.append(code))
    return exit_codes


class TestMaybeCrash:
    # [CLASS DOC] Tests of the actor-side executor's crash hook, which hard-exits the process
    # with code 1 only for the exact cell, rank, rollout and attempt of an armed crash action.
    def test_targeted_cell_and_rank_exits(self, recorded_exit_codes: list[int]) -> None:
        # [FUNCTION DOC] The matching worker at rollout 4, attempt 0 records exactly one exit
        # with code 1.
        """The rank named by the action reaches os._exit(1) on the target rollout and attempt."""
        executor = _make_actor_executor(cell_id="trainer-engine-actor-1", rank=0)

        executor.maybe_crash(rollout_id=4, attempt=0)

        assert recorded_exit_codes == [1]

    def test_other_cell_does_not_exit(self, recorded_exit_codes: list[int]) -> None:
        # [FUNCTION DOC] A worker in cell 0 does not exit.
        """A worker in a cell the action does not name keeps running."""
        executor = _make_actor_executor(cell_id="trainer-engine-actor-0", rank=0)

        executor.maybe_crash(rollout_id=4, attempt=0)

        assert recorded_exit_codes == []

    def test_cell_of_another_spec_does_not_exit(self, recorded_exit_codes: list[int]) -> None:
        # [FUNCTION DOC] A worker whose cell id has the same index but belongs to another pool
        # does not exit (exact string match).
        """Matching is exact string equality, so a same-index cell of another spec survives."""
        executor = _make_actor_executor(cell_id="rollout-engine-1", rank=0)

        executor.maybe_crash(rollout_id=4, attempt=0)

        assert recorded_exit_codes == []

    def test_other_rank_in_targeted_cell_does_not_exit(self, recorded_exit_codes: list[int]) -> None:
        # [FUNCTION DOC] Rank 1 of the targeted cell does not exit.
        """Only the named rank of the named cell crashes, not its siblings."""
        executor = _make_actor_executor(cell_id="trainer-engine-actor-1", rank=1)

        executor.maybe_crash(rollout_id=4, attempt=0)

        assert recorded_exit_codes == []

    def test_other_rollout_does_not_exit(self, recorded_exit_codes: list[int]) -> None:
        # [FUNCTION DOC] The matching worker at rollout 3 does not exit.
        """The crash is armed for one rollout only."""
        executor = _make_actor_executor(cell_id="trainer-engine-actor-1", rank=0)

        executor.maybe_crash(rollout_id=3, attempt=0)

        assert recorded_exit_codes == []

    def test_other_attempt_does_not_exit(self, recorded_exit_codes: list[int]) -> None:
        # [FUNCTION DOC] The matching worker at attempt 1 (a retry) does not exit.
        """The retry after the injected crash must not crash again."""
        executor = _make_actor_executor(cell_id="trainer-engine-actor-1", rank=0)

        executor.maybe_crash(rollout_id=4, attempt=1)

        assert recorded_exit_codes == []

    def test_no_actions_never_exits(self, recorded_exit_codes: list[int]) -> None:
        # [FUNCTION DOC] An executor with no actions never exits.
        """An actor executor with no actions never crashes its worker."""
        executor = FTTestActionActorExecutor(actions=[], cell_id="trainer-engine-actor-1", rank=0)

        executor.maybe_crash(rollout_id=4, attempt=0)

        assert recorded_exit_codes == []


_SLEEP_ACTION = FTTestAction(at_rollout=2, action="sleep_forever_at_end")


class _SleptEnough(Exception):
    # [CLASS DOC] HELPER: Exception raised by the fake sleeper once it has been called the
    # configured number of times, to break out of the infinite sleep loop.
    pass


def _sleeper(*, wakes: list[float], limit: int):
    # [FUNCTION DOC] HELPER: Returns a fake async sleep function that appends each requested
    # duration to the wakes list and raises the slept-enough exception once the list has reached
    # the limit length. It never actually sleeps.
    async def sleep(seconds: float) -> None:
        # [FUNCTION DOC] HELPER: Records the duration and raises when the call limit is reached.
        wakes.append(seconds)
        if len(wakes) >= limit:
            raise _SleptEnough

    return sleep


class TestLoadingASleepForeverAction:
    # [CLASS DOC] Tests of how the sleep-forever action is classified and validated when plans
    # are loaded.
    def test_the_action_that_freezes_the_orchestration_script_is_not_offered_to_the_cell_side_executors(self) -> None:
        # [FUNCTION DOC] Asserts the sleep-forever name is in the orchestration action set and
        # that this set shares no member with the controller and actor sets.
        """A cell executor acting on it would suspend a cell where the scenario asked for a frozen script."""
        assert SLEEP_FOREVER_AT_END_ACTION in _ORCHESTRATION_ACTIONS
        assert not _ORCHESTRATION_ACTIONS & (_CONTROLLER_ACTIONS | _ACTOR_ACTIONS)

    def test_a_sleep_forever_action_loads_without_naming_a_cell(self) -> None:
        # [FUNCTION DOC] A sleep-forever action at rollout 2 with no cell id loads with the
        # orchestration filter as exactly the module's sleep action.
        """The script that drives the run is not a cell, so this action has no cell id to carry."""
        raw = json.dumps([{"at_rollout": 2, "action": "sleep_forever_at_end"}])

        [action] = _load_actions(_args(raw), _ORCHESTRATION_ACTIONS)

        assert action == _SLEEP_ACTION

    def test_a_sleep_forever_action_that_names_a_cell_is_refused(self) -> None:
        # [FUNCTION DOC] A sleep-forever action that carries a cell id raises a validation error.
        """A cell id here would read as a cell being frozen, and nothing freezes a cell."""
        raw = json.dumps([{"at_rollout": 2, "action": "sleep_forever_at_end", "cell_id": "trainer-engine-actor-0"}])

        with pytest.raises(ValidationError):
            _load_actions(_args(raw), _ORCHESTRATION_ACTIONS)

    def test_the_orchestration_executor_ignores_the_actions_of_the_other_sides(self) -> None:
        # [FUNCTION DOC] A plan mixing a stop action and a sleep-forever action loads under the
        # orchestration filter as only the sleep action.
        """The orchestration script runs beside the cells, and suspending one from here would drive it twice."""
        raw = json.dumps(
            [
                {"at_rollout": 1, "action": "stop_cell_at_end", "cell_id": "trainer-engine-actor-0"},
                {"at_rollout": 2, "action": "sleep_forever_at_end"},
            ]
        )

        assert _load_actions(_args(raw), _ORCHESTRATION_ACTIONS) == [_SLEEP_ACTION]


class TestSleepForeverAtEnd:
    # [CLASS DOC] Tests of the orchestration executor's after-step hook, which sleeps in a loop
    # forever at the step named by a sleep-forever action. The sleep function and interval are
    # injectable.
    @pytest.mark.asyncio
    async def test_the_step_the_action_names_never_returns(self) -> None:
        # [FUNCTION DOC] With an interval of 0.5 and a sleeper that raises after 3 calls, the
        # hook for rollout 2 keeps sleeping (raises the slept-enough exception) and the recorded
        # durations are three times 0.5.
        """The whole point is that the run stands still, so the loop is never handed the next step."""
        wakes: list[float] = []
        executor = FTTestActionOrchestrationExecutor(
            actions=[_SLEEP_ACTION], sleep=_sleeper(wakes=wakes, limit=3), interval_seconds=0.5
        )

        with pytest.raises(_SleptEnough):
            await executor.run_after_step(rollout_id=2)

        assert wakes == [0.5, 0.5, 0.5]

    @pytest.mark.asyncio
    async def test_every_other_step_returns_at_once(self) -> None:
        # [FUNCTION DOC] Hook calls for rollouts 0, 1, 3 and 4 (not the armed 2) return without
        # any sleeping.
        """A run frozen a step early would resume from another checkpoint than the one pinned."""
        wakes: list[float] = []
        executor = FTTestActionOrchestrationExecutor(actions=[_SLEEP_ACTION], sleep=_sleeper(wakes=wakes, limit=1))

        for rollout_id in (0, 1, 3, 4):
            await executor.run_after_step(rollout_id=rollout_id)

        assert wakes == []

    @pytest.mark.asyncio
    async def test_an_orchestration_executor_with_no_actions_never_sleeps(self) -> None:
        # [FUNCTION DOC] With no actions the hook returns at once and never sleeps.
        """Every run outside this scenario carries no plan and has to train straight through."""
        wakes: list[float] = []
        executor = FTTestActionOrchestrationExecutor(actions=[], sleep=_sleeper(wakes=wakes, limit=1))

        await executor.run_after_step(rollout_id=2)

        assert wakes == []

    @pytest.mark.asyncio
    async def test_a_cell_action_that_reached_the_orchestration_side_is_refused(self) -> None:
        # [FUNCTION DOC] An executor given a stop-cell action raises an assertion error naming
        # that action when the hook runs for its rollout.
        """Sleeping on a cell action would swallow the suspend the scenario armed and freeze the run instead."""
        executor = FTTestActionOrchestrationExecutor(
            actions=[FTTestAction(at_rollout=2, action="stop_cell_at_end", cell_id="trainer-engine-actor-0")],
            sleep=_sleeper(wakes=[], limit=1),
        )

        with pytest.raises(AssertionError, match="stop_cell_at_end"):
            await executor.run_after_step(rollout_id=2)

    @pytest.mark.asyncio
    async def test_a_plan_the_run_never_reaches_leaves_it_training(self) -> None:
        # [FUNCTION DOC] A sleep action armed for rollout 99 does nothing at rollout 5: no sleeps.
        """An armed step past the end of the run is a scenario bug, not a freeze."""
        wakes: list[float] = []
        executor = FTTestActionOrchestrationExecutor(
            actions=[FTTestAction(at_rollout=99, action="sleep_forever_at_end")],
            sleep=_sleeper(wakes=wakes, limit=1),
        )

        await executor.run_after_step(rollout_id=5)

        assert wakes == []


# ============ adhoc file delivery (revert after the args refactor) ============


def _args_of_path(path: object) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds the default fake arguments with the plan file path set to the
    # given value.
    return _args(ci_ft_test_actions_path=path)


# TODO ad hoc hack: revert after the args refactor
class TestActionsDeliveredThroughAFile:
    # [CLASS DOC] Tests of delivering the action plan through a file path argument instead of an
    # inline value.
    def test_a_plan_written_to_the_file_is_the_plan_the_run_performs(self, tmp_path) -> None:
        # [FUNCTION DOC] Writes a sleep-forever plan to a file in the temporary directory and
        # asserts loading from that path with the orchestration filter gives the sleep action.
        """Every worker pod's command carries the arguments, so a changing plan cannot live in one."""
        path = tmp_path / "plan.json"
        write_ft_test_actions(path, [{"at_rollout": 2, "action": SLEEP_FOREVER_AT_END_ACTION}])

        assert _load_actions(_args_of_path(str(path)), _ORCHESTRATION_ACTIONS) == [_SLEEP_ACTION]

    def test_a_file_rewritten_between_two_reads_is_read_again_rather_than_remembered(self, tmp_path) -> None:
        # [FUNCTION DOC] Loads a plan, rewrites the file with an empty plan, and loads again;
        # asserts the first load gives the sleep action and the second gives an empty list.
        """A take-over relaunches the same command, so only a fresh read can arm the next freeze."""
        path = tmp_path / "plan.json"
        write_ft_test_actions(path, [{"at_rollout": 2, "action": SLEEP_FOREVER_AT_END_ACTION}])
        args = _args_of_path(str(path))
        assert _load_actions(args, _ORCHESTRATION_ACTIONS) == [_SLEEP_ACTION]

        write_ft_test_actions(path, [])

        assert _load_actions(args, _ORCHESTRATION_ACTIONS) == []

    def test_a_file_the_run_was_pointed_at_but_nobody_wrote_is_refused(self, tmp_path) -> None:
        # [FUNCTION DOC] A path to a nonexistent file raises an assertion error saying it does
        # not exist.
        """A plan nothing wrote would leave the run performing no action while looking armed."""
        with pytest.raises(AssertionError, match="does not exist"):
            _load_actions(_args_of_path(str(tmp_path / "absent.json")), _ORCHESTRATION_ACTIONS)

    def test_a_run_told_both_the_plan_and_a_file_holding_one_is_refused(self, tmp_path) -> None:
        # [FUNCTION DOC] Arguments with both an inline plan and a plan file path raise an
        # assertion error saying both name the actions.
        """Two plans mean the run silently follows one of them, and nobody can tell which."""
        args = _args(
            json.dumps([{"at_rollout": 2, "action": SLEEP_FOREVER_AT_END_ACTION}]),
            ci_ft_test_actions_path=str(tmp_path / "plan.json"),
        )

        with pytest.raises(AssertionError, match="both name the actions"):
            _load_actions(args, _ORCHESTRATION_ACTIONS)


# TODO ad hoc hack: revert after the args refactor
class TestTheSentinelAFrozenRunLeaves:
    # [CLASS DOC] Tests of the sentinel the orchestration executor writes next to the plan file
    # when it parks, recording the frozen rollout id.
    @pytest.mark.asyncio
    async def test_a_run_that_parked_says_so_where_the_plan_was_read_from(self, tmp_path) -> None:
        # [FUNCTION DOC] Parks the executor at rollout 2 with a sleeper that raises after one
        # call; asserts the sentinel read for the plan path reports frozen rollout 2.
        """Only the run knows it is parked; a reader of its metrics sees the step end, not the sleep."""
        path = tmp_path / "plan.json"
        write_ft_test_actions(path, [{"at_rollout": 2, "action": SLEEP_FOREVER_AT_END_ACTION}])
        executor = FTTestActionOrchestrationExecutor(
            actions=[_SLEEP_ACTION], sleep=_stop_after_sleep, actions_path=path
        )

        with pytest.raises(_SleptOnce):
            await executor.run_after_step(rollout_id=2)

        assert read_frozen_rollout_id(path) == 2

    @pytest.mark.asyncio
    async def test_a_run_that_never_parked_leaves_no_sentinel(self, tmp_path) -> None:
        # [FUNCTION DOC] Running the hook for rollout 1 (not armed) leaves no sentinel; reading it
        # gives none.
        """Reading a stale sentinel as this freeze would take a still-training run over."""
        path = tmp_path / "plan.json"
        write_ft_test_actions(path, [{"at_rollout": 2, "action": SLEEP_FOREVER_AT_END_ACTION}])
        executor = FTTestActionOrchestrationExecutor(actions=[_SLEEP_ACTION], actions_path=path)

        await executor.run_after_step(rollout_id=1)

        assert read_frozen_rollout_id(path) is None

    @pytest.mark.asyncio
    async def test_each_freeze_overwrites_what_the_previous_one_left(self, tmp_path) -> None:
        # [FUNCTION DOC] Starts with a sentinel for rollout 2, then parks at rollout 4; the
        # sentinel now reads 4.
        """A take-over waits for its own step, so the value has to move on with the run."""
        path = tmp_path / "plan.json"
        write_frozen_sentinel(path, rollout_id=2)
        executor = FTTestActionOrchestrationExecutor(
            actions=[FTTestAction(at_rollout=4, action=SLEEP_FOREVER_AT_END_ACTION)],
            sleep=_stop_after_sleep,
            actions_path=path,
        )

        with pytest.raises(_SleptOnce):
            await executor.run_after_step(rollout_id=4)

        assert read_frozen_rollout_id(path) == 4


class _SleptOnce(Exception):
    # [CLASS DOC] HELPER: Exception raised by the one-shot fake sleep function.
    pass


async def _stop_after_sleep(_seconds: float) -> None:
    # [FUNCTION DOC] HELPER: Fake async sleep that ignores its duration and immediately raises
    # the slept-once exception.
    raise _SleptOnce


# TODO ad hoc hack: revert after the args refactor
class TestHowOftenThePlanIsReadOffDisk:
    # [CLASS DOC] Tests of the plan file read cache, keyed on the file's modification stamp.
    def test_a_plan_whose_stamp_has_not_moved_is_not_read_again(self, tmp_path) -> None:
        # [FUNCTION DOC] Reads a written plan, overwrites the file with spaces of equal length
        # and restores the original access and modification times, then reads again; asserts the
        # second read returns the first read's content (served from the cache, not the disk).
        """The run consults this once per step, and the file sits on shared storage."""
        path = tmp_path / "plan.json"
        write_ft_test_actions(path, [{"at_rollout": 2, "action": SLEEP_FOREVER_AT_END_ACTION}])
        first = read_ft_test_actions(path)
        stamp = path.stat()

        path.write_text(" " * len(first))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))

        assert read_ft_test_actions(path) == first

    def test_a_rewritten_plan_is_read_again(self, tmp_path) -> None:
        # [FUNCTION DOC] Reads a non-empty plan, rewrites the file with an empty plan, and reads
        # again; asserts the second read parses to an empty list.
        """A relaunch arms the next freeze by rewriting the file, and the run has to see it."""
        path = tmp_path / "plan.json"
        write_ft_test_actions(path, [{"at_rollout": 2, "action": SLEEP_FOREVER_AT_END_ACTION}])
        assert json.loads(read_ft_test_actions(path)) != []

        write_ft_test_actions(path, [])

        assert json.loads(read_ft_test_actions(path)) == []


class TestWhichLoopsCanBeParked:
    # [CLASS DOC] Tests of the orchestration executor's factory from arguments, which refuses to
    # arm a sleep action in training loops where parking would not stop the run at the named
    # step. The driven_by fixture sets the process command line to simulate the train script.
    def test_a_plan_reaching_a_loop_of_another_train_script_is_refused(self, driven_by) -> None:
        # [FUNCTION DOC] With the async train script as the process, creating the executor from
        # arguments carrying a sleep plan raises an assertion error saying the script has already
        # started the next rollout.
        """train_async.py has already started the next rollout, so the run would not be where the action names."""
        driven_by("train_async.py")
        args = _args(json.dumps([_SLEEP_ACTION.model_dump()]))

        with pytest.raises(AssertionError, match="train_async.py has already started"):
            FTTestActionOrchestrationExecutor.from_args(args)

    def test_a_loop_is_judged_by_the_script_driving_it_and_not_by_a_flag_it_declares(self, driven_by) -> None:
        # [FUNCTION DOC] With the async train script as the process and the fully-async flag
        # false, creation is still refused with the same assertion error (script wins over flag).
        """--fully-async is one way to reach an async loop; the train script is what actually chooses one."""
        driven_by("train_async.py")
        args = _args(json.dumps([_SLEEP_ACTION.model_dump()]), fully_async=False)

        with pytest.raises(AssertionError, match="train_async.py has already started"):
            FTTestActionOrchestrationExecutor.from_args(args)

    def test_a_plan_reaching_a_run_that_syncs_weights_every_other_step_is_refused(self, driven_by) -> None:
        # [FUNCTION DOC] With the parkable script and an update weights interval of 2, creation
        # raises an assertion error mentioning the update-weights-interval option.
        """The action parks the run where it updates weights; a gated update skips that point."""
        driven_by(PARKABLE_TRAIN_SCRIPT)
        args = _args(json.dumps([_SLEEP_ACTION.model_dump()]), update_weights_interval=2)

        with pytest.raises(AssertionError, match="update-weights-interval"):
            FTTestActionOrchestrationExecutor.from_args(args)

    def test_a_plan_reaching_one_policy_of_a_multi_policy_run_is_refused(self, driven_by) -> None:
        # [FUNCTION DOC] With the parkable script and a trainer model id given (a multi-policy
        # run), creation raises an assertion error mentioning several policies.
        """Only that policy's coroutine would park; the others would train past the step."""
        driven_by(PARKABLE_TRAIN_SCRIPT)
        args = _args(json.dumps([_SLEEP_ACTION.model_dump()]))

        with pytest.raises(AssertionError, match="several policies"):
            FTTestActionOrchestrationExecutor.from_args(args, trainer_model_id="solver")

    def test_the_parkable_script_is_the_one_the_harness_launches_by_default(self) -> None:
        # [FUNCTION DOC] Asserts the parkable train script constant equals the default train
        # script constant of the end-to-end fault-tolerance harness.
        """A literal of its own would go on naming train.py after the harness moved to another script."""
        assert PARKABLE_TRAIN_SCRIPT == DEFAULT_TRAIN_SCRIPT

    def test_the_loop_the_action_was_written_for_is_accepted(self, driven_by) -> None:
        # [FUNCTION DOC] With the parkable script, default interval and a sleep plan, creation
        # succeeds and the executor holds exactly the sleep action (checked through its private
        # actions attribute).
        """train.py updates weights once per step, for one policy, after the step is done."""
        driven_by(PARKABLE_TRAIN_SCRIPT)
        args = _args(json.dumps([_SLEEP_ACTION.model_dump()]))

        assert FTTestActionOrchestrationExecutor.from_args(args)._actions == [_SLEEP_ACTION]

    def test_a_run_with_no_plan_is_never_asked_which_loop_it_is(self, driven_by) -> None:
        # [FUNCTION DOC] With the async script and fully-async set but no plan, creation does not
        # raise.
        """Every run passes through here; only the ones carrying the action are constrained."""
        driven_by("train_async.py")

        FTTestActionOrchestrationExecutor.from_args(_args(fully_async=True))


@pytest.fixture
def driven_by(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture returning an installer function; the installer patches
    # (reverted at teardown) the process argument vector so that the first entry is the given
    # train script under a fixed directory, followed by a dummy flag and value.
    def install(train_script: str) -> None:
        # [FUNCTION DOC] HELPER: Replaces the process argument vector as described above.
        monkeypatch.setattr(sys, "argv", [f"/miles/{train_script}", "--some-flag", "some-value"])

    return install
