# [MODULE DOC] HELPER: Scripted fault-tolerance test actions for CI. A test declares a JSON list of
# actions (each with the rollout index at which to fire, an action name, and for cell actions a
# cell id) and passes it to the training run either inline via a command-line flag or through a
# file path flag. Three executors consume the list, each filtering to its own action kinds:
# controller-side (stop or start a cell after a given rollout step), actor-side (crash a specific
# rank of a specific cell on a given attempt before the all-reduce), and orchestration-side (park
# the orchestration script forever after a given step, optionally writing a "frozen" sentinel
# file). Also holds ad-hoc file delivery helpers (atomic write, stat-cached read, sentinel
# paths) marked as temporary until the argument handling is refactored.
import asyncio
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.naming import parse_cell_id

if TYPE_CHECKING:
    from miles.ray.train.group import TrainerController
    from miles.utils.workers.cell_operations.base import BaseCellOperations

logger = logging.getLogger(__name__)

CI_FT_TEST_ACTIONS_FLAG: str = "--ci-ft-test-actions"
SLEEP_FOREVER_AT_END_ACTION: str = "sleep_forever_at_end"
SLEEP_FOREVER_INTERVAL_SECONDS: float = 60.0
PARKABLE_TRAIN_SCRIPT: str = "train.py"


def compute_ft_test_actions_arg(actions: Sequence[dict]) -> str:
    # [FUNCTION DOC] HELPER: Builds the command-line fragment for inline delivery: the flag name,
    # a space, the JSON rendering of the actions wrapped in single quotes (no escaping of single
    # quotes inside the JSON), and a trailing space.
    return f"{CI_FT_TEST_ACTIONS_FLAG} '{render_ft_test_actions(actions)}' "


def render_ft_test_actions(actions: Sequence[dict]) -> str:
    # [FUNCTION DOC] HELPER: Serializes the sequence of action mappings to a JSON array string.
    return json.dumps(list(actions))


_CELL_RESUME_OBSERVED_TIMEOUT_SECONDS = 300.0

_CONTROLLER_ACTIONS = {"stop_cell_at_end", "start_cell_at_end"}
_ACTOR_ACTIONS = {"crash_before_allreduce"}
_ORCHESTRATION_ACTIONS = {SLEEP_FOREVER_AT_END_ACTION}

SleepFn = Callable[[float], Awaitable[None]]


class FTTestAction(FrozenStrictBaseModel):
    # [CLASS DOC] One immutable, strictly validated action record. Fields: the rollout index at
    # which it fires; the action name, one of stop-cell-at-end, start-cell-at-end,
    # crash-before-allreduce, sleep-forever-at-end; the target cell id (required for cell actions,
    # forbidden for the orchestration action); for actor-level actions the rank within the cell
    # (default 0) and the attempt number (default 0, the first try). Note the actor-only fields
    # rank and attempt are not validated against the action kind.
    at_rollout: int
    action: Literal["stop_cell_at_end", "start_cell_at_end", "crash_before_allreduce", "sleep_forever_at_end"]
    cell_id: str | None = None
    rank: int = 0  # for actor-level actions: which rank within the cell
    attempt: int = 0  # for actor-level actions: which attempt (0 = first try)

    @model_validator(mode="after")
    def _check_cell_name_matches_action(self) -> "FTTestAction":
        # [FUNCTION DOC] HELPER: Post-construction validator: asserts that the cell id is absent
        # exactly when the action is the orchestration action (sleep forever); the assertion
        # error message names the action and cell id. Returns the same object.
        assert (self.action in _ORCHESTRATION_ACTIONS) == (self.cell_id is None), (
            f"an orchestration action names no cell and a cell action names one, and {self.action} names "
            f"cell_id={self.cell_id!r}"
        )
        return self


_ACTION_LIST_ADAPTER: TypeAdapter[list[FTTestAction]] = TypeAdapter(list[FTTestAction])


def _load_actions(args: object, action_filter: set[str]) -> list[FTTestAction]:
    # [FUNCTION DOC] HELPER: Reads the declared actions text from the parsed arguments (inline or
    # file, see the reader helper); returns an empty list when nothing is declared. Parses the JSON
    # into validated action records (validation errors propagate), checks that each declared cell
    # id parses as a valid cell identifier (raising a value error naming the cell id and action,
    # chained from the parse error), then keeps only actions whose name is in the filter set,
    # logging an info line if any remain. Returns them in declaration order.
    if not (raw := _read_declared_actions(args)):
        return []
    all_actions = _ACTION_LIST_ADAPTER.validate_json(raw)

    for action in all_actions:
        if (cell_id := action.cell_id) is None:
            continue
        try:
            parse_cell_id(cell_id)
        except ValueError as e:
            raise ValueError(f"FT test action has malformed cell_id {cell_id!r} (action={action})") from e

    actions = [a for a in all_actions if a.action in action_filter]
    if actions:
        logger.info("FT test actions activated: %d actions (%s)", len(actions), action_filter)
    return actions


def _assert_loop_parkable(args: object, *, trainer_model_id: str | None) -> None:
    # [FUNCTION DOC] HELPER: Asserts the run can be correctly parked by the sleep-forever action:
    # the entry script (basename of the first process argument) must be the train script; the
    # weight update interval argument must equal 1; and the trainer model id must be none (a
    # single-policy run). Each failed assertion has an explanatory message.
    assert (script := Path(sys.argv[0]).name) == PARKABLE_TRAIN_SCRIPT, (
        f"{SLEEP_FOREVER_AT_END_ACTION} parks the orchestration script between two steps, and only "
        f"{PARKABLE_TRAIN_SCRIPT} stands still at that point; {script} has already started the next rollout by the "
        f"time it reaches here, so the run would not be standing where the action names"
    )
    assert (interval := args.update_weights_interval) == 1, (
        f"{SLEEP_FOREVER_AT_END_ACTION} parks the run where it updates weights, and --update-weights-interval "
        f"{interval} means the run does not pass through that point after every step"
    )
    assert trainer_model_id is None, (
        f"{SLEEP_FOREVER_AT_END_ACTION} parks one coroutine, and a run training several policies drives one per "
        f"policy ({trainer_model_id!r} reached it here), so every other policy would keep training past the step "
        f"the action names"
    )


class FTTestActionControllerExecutor:
    # [CLASS DOC] Runs controller-side actions (suspend or resume a whole cell) after a rollout
    # step finishes. Holds the action list, the trainer controller (for pool id, observed cell ids
    # and expected cell count) and the cell operations object (suspend and resume).
    def __init__(
        self, *, actions: list[FTTestAction], controller: "TrainerController", cell_operations: "BaseCellOperations"
    ) -> None:
        # [FUNCTION DOC] Stores the actions, controller and cell operations.
        self._actions = actions
        self._controller = controller
        self._cell_operations = cell_operations

    @staticmethod
    def from_args(
        args: object, *, controller: "TrainerController", cell_operations: "BaseCellOperations"
    ) -> "FTTestActionControllerExecutor":
        # [FUNCTION DOC] Factory: loads the stop-cell and start-cell actions from the parsed
        # arguments and builds the executor with the given controller and cell operations.
        return FTTestActionControllerExecutor(
            actions=_load_actions(args, _CONTROLLER_ACTIONS), controller=controller, cell_operations=cell_operations
        )

    async def run_after_step(self, rollout_id: int) -> None:
        # [FUNCTION DOC] For each stored action scheduled at the given rollout (in order): checks
        # its target against the controller (assertion errors), logs it, then for a stop action
        # awaits suspension of the cell; for a start action awaits resumption of the cell and then
        # waits until the controller observes the cell (up to the observed-timeout retry helper).
        # Raises whatever those operations raise, including the timeout error if the cell never
        # shows up.
        for action in self._actions:
            if action.at_rollout == rollout_id:
                self._check_action_target(action)
                logger.info("FT test action: %s cell %s after rollout %d", action.action, action.cell_id, rollout_id)

                operations = self._cell_operations
                if action.action == "stop_cell_at_end":
                    await operations.suspend(cell_id=action.cell_id)
                elif action.action == "start_cell_at_end":
                    await operations.resume(cell_id=action.cell_id)
                    await self._wait_cell_observed(action.cell_id)

    async def _wait_cell_observed(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Polls until the cell id appears among the controller's observed
        # cell ids, using the deadline retry helper: total 300 seconds, retrying on timeout errors,
        # backoff from 1 to at most 5 seconds, structured-log fields tag, operation and cell.
        # Raises the timeout error from the last attempt when the deadline is exhausted.
        async def _check(_remaining: float) -> None:
            # [FUNCTION DOC] HELPER: One poll attempt; raises a timeout error (retryable) if the
            # cell is not yet observed. The remaining-time argument is ignored.
            if cell_id not in self._controller.cell_ids:
                raise TimeoutError(f"{cell_id} was resumed but is not observed yet")

        await retry_until_deadline(
            _check,
            total_seconds=_CELL_RESUME_OBSERVED_TIMEOUT_SECONDS,
            retry_on=TimeoutError,
            initial_delay=1.0,
            max_delay=5.0,
            log_fields=dict(tag="ft", op="wait_cell_observed", cell=cell_id),
        )

    def _check_action_target(self, action: FTTestAction) -> None:
        # [FUNCTION DOC] HELPER: Asserts the action has a cell id, parses it, and asserts that its
        # pool id equals the controller's pool id and its cell index is below the controller's
        # expected number of cells (assertion errors include the action).
        assert (cell_id := action.cell_id) is not None
        parsed = parse_cell_id(cell_id)
        assert parsed.pool_id == self._controller.pool_id, (
            f"FT test action targets pool_id {parsed.pool_id!r} but this controller drives {self._controller.pool_id!r} "
            f"(action={action})"
        )
        assert parsed.cell_index < self._controller.expected_num_cells, (
            f"FT test action targets cell index {parsed.cell_index} but the pool only has "
            f"{self._controller.expected_num_cells} cells (action={action})"
        )


class FTTestActionActorExecutor:
    # [CLASS DOC] Runs actor-side actions inside a training worker process: crashes the worker
    # when an action matches its own cell and rank at a given rollout and attempt.
    def __init__(self, *, actions: list[FTTestAction], cell_id: str, rank: int) -> None:
        # [FUNCTION DOC] Stores the actions and this worker's cell id and rank within the cell.
        self._actions = actions
        self._cell_id = cell_id
        self._rank = rank

    @staticmethod
    def from_args(
        args: object,
        *,
        cell_id: str,
        rank: int,
    ) -> "FTTestActionActorExecutor":
        # [FUNCTION DOC] Factory: loads the crash-before-all-reduce actions from the parsed
        # arguments and builds the executor for the given cell id and rank.
        return FTTestActionActorExecutor(
            actions=_load_actions(args, _ACTOR_ACTIONS),
            cell_id=cell_id,
            rank=rank,
        )

    def maybe_crash(self, *, rollout_id: int, attempt: int) -> None:
        # [FUNCTION DOC] If any stored action matches all of: rollout index equals the given
        # rollout, attempt equals the given attempt, cell id equals this worker's cell, rank
        # equals this worker's rank, then logs a warning, prints the same message to standard
        # output (flushed) and terminates the process immediately with status 1 without cleanup.
        # Otherwise returns without effect.
        for action in self._actions:
            if (
                action.at_rollout == rollout_id
                and action.attempt == attempt
                and action.cell_id == self._cell_id
                and action.rank == self._rank
            ):
                msg = (
                    f"FT test action: crash_before_allreduce at rollout {rollout_id} "
                    f"attempt {attempt} cell {self._cell_id} rank {self._rank} — calling os._exit(1)"
                )
                logger.warning(msg)
                print(msg, flush=True)
                os._exit(1)


class FTTestActionOrchestrationExecutor:
    # [CLASS DOC] Runs orchestration-side actions in the driver script: after the configured
    # rollout step, parks the script forever (sleeping in fixed intervals) so tests can observe
    # a stalled run. The sleep function is injectable for tests.
    def __init__(
        self,
        *,
        actions: list[FTTestAction],
        sleep: SleepFn = asyncio.sleep,
        interval_seconds: float = SLEEP_FOREVER_INTERVAL_SECONDS,
        actions_path: Path | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the actions, the async sleep function (default the real one), the
        # sleep interval in seconds (default 60) and the optional actions file path (used to place
        # the frozen sentinel next to it).
        self._actions = actions
        self._sleep = sleep
        self._interval_seconds = interval_seconds
        self._actions_path = actions_path

    @staticmethod
    def from_args(args: object, *, trainer_model_id: str | None = None) -> "FTTestActionOrchestrationExecutor":
        # [FUNCTION DOC] Factory: loads the sleep-forever actions from the parsed arguments; if
        # there are any, asserts the run is parkable (train script, update interval 1, single
        # policy given the trainer model id). Reads the actions path argument (required attribute,
        # may be none) and builds the executor with it as a path object when set.
        actions = _load_actions(args, _ORCHESTRATION_ACTIONS)
        if actions:
            _assert_loop_parkable(args, trainer_model_id=trainer_model_id)

        path: str | None = args.ci_ft_test_actions_path
        return FTTestActionOrchestrationExecutor(
            actions=actions,
            actions_path=Path(path) if path is not None else None,
        )

    async def run_after_step(self, rollout_id: int) -> None:
        # [FUNCTION DOC] Selects actions scheduled at the given rollout; returns immediately if
        # none. Asserts each is the sleep-forever action. Otherwise logs a warning and prints a
        # message (flushed) that the script will sleep and never start the next rollout; if an
        # actions path is configured, writes the frozen sentinel file recording this rollout id;
        # then sleeps forever (never returns normally; ends only by cancellation).
        actions = [action for action in self._actions if action.at_rollout == rollout_id]
        if not actions:
            return
        for action in actions:
            assert action.action == SLEEP_FOREVER_AT_END_ACTION, (
                f"the orchestration side runs {SLEEP_FOREVER_AT_END_ACTION} and nothing else, and {action.action} "
                f"reached it (action={action})"
            )

        msg = (
            f"FT test action: {SLEEP_FOREVER_AT_END_ACTION} at rollout {rollout_id} — this orchestration script "
            f"sleeps from here on and never starts rollout {rollout_id + 1}"
        )
        logger.warning(msg)
        print(msg, flush=True)
        if self._actions_path is not None:
            write_frozen_sentinel(self._actions_path, rollout_id=rollout_id)
        await self._sleep_forever()

    async def _sleep_forever(self) -> None:
        # [FUNCTION DOC] HELPER: Infinite loop awaiting the injected sleep function for the
        # configured interval each time.
        while True:
            await self._sleep(self._interval_seconds)


# ============ adhoc file delivery (revert after the args refactor) ============


CI_FT_TEST_ACTIONS_PATH_FLAG: str = "--ci-ft-test-actions-path"


# TODO ad hoc hack: revert after the args refactor
def _read_declared_actions(args: object) -> str:
    # [FUNCTION DOC] HELPER: Returns the JSON text of the declared actions from the parsed
    # arguments: reads the inline actions and actions path attributes (both must exist), asserts
    # they are not both set, and returns the file's contents (via the cached reader) if the path
    # is set, else the inline text, else an empty string.
    inline: str | None = args.ci_ft_test_actions
    path: str | None = args.ci_ft_test_actions_path

    assert inline is None or path is None, (
        f"{CI_FT_TEST_ACTIONS_FLAG} and {CI_FT_TEST_ACTIONS_PATH_FLAG} both name the actions a run performs, and a "
        f"run given both silently follows one of them"
    )
    return read_ft_test_actions(Path(path)) if path is not None else (inline or "")


# TODO ad hoc hack: revert after the args refactor
def write_ft_test_actions(path: Path, actions: Sequence[dict]) -> None:
    # [FUNCTION DOC] HELPER: Writes the JSON rendering of the actions to the path atomically:
    # creates parent directories, writes to a sibling temporary file named with the process id,
    # then renames it over the destination.
    path.parent.mkdir(parents=True, exist_ok=True)

    scratch = path.with_name(f"{path.name}.{os.getpid()}.partial")
    scratch.write_text(render_ft_test_actions(actions))
    scratch.replace(path)


# TODO ad hoc hack: revert after the args refactor
def read_ft_test_actions(path: Path) -> str:
    # [FUNCTION DOC] HELPER: Returns the actions file's text with a module-level cache keyed by
    # path and validated by the file's modification time in nanoseconds and size. Asserts the file
    # exists (assertion error explaining that a missing plan would silently do nothing). On a
    # cache hit with an unchanged stamp returns the cached text, else reads the file and stores
    # it. Mutates the global cache. A rewrite with the same size within the same timestamp
    # granularity would not be detected.
    stamp = _stat_or_none(path)
    assert stamp is not None, (
        f"{CI_FT_TEST_ACTIONS_PATH_FLAG} names {path}, which does not exist; a run told to read its plan from a "
        f"file nothing wrote would quietly perform no action at all"
    )

    stamped_at = (stamp.st_mtime_ns, stamp.st_size)
    if (cached := _ACTIONS_OF_STAMP.get(path)) is not None and cached[0] == stamped_at:
        return cached[1]

    text = path.read_text()
    _ACTIONS_OF_STAMP[path] = (stamped_at, text)
    return text


# TODO ad hoc hack: revert after the args refactor
def _stat_or_none(path: Path) -> os.stat_result | None:
    # [FUNCTION DOC] HELPER: Returns the file status of the path, or none if the status call fails
    # with any operating system error (missing file, permissions, and so on).
    try:
        return path.stat()
    except OSError:
        return None


FROZEN_SENTINEL_SUFFIX: str = "_frozen_at.json"
# TODO ad hoc hack: revert after the args refactor
_ACTIONS_OF_STAMP: dict[Path, tuple[tuple[int, int], str]] = {}


# TODO ad hoc hack: revert after the args refactor
def compute_frozen_sentinel_path(actions_path: Path) -> Path:
    # [FUNCTION DOC] HELPER: Returns the sentinel file path: same directory as the actions file,
    # named as the actions file's stem followed by the frozen-at JSON suffix.
    return actions_path.with_name(f"{actions_path.stem}{FROZEN_SENTINEL_SUFFIX}")


# TODO ad hoc hack: revert after the args refactor
def write_frozen_sentinel(actions_path: Path, *, rollout_id: int) -> None:
    # [FUNCTION DOC] HELPER: Atomically writes a JSON object holding the rollout id to the
    # sentinel path derived from the actions path (creating parent directories, writing a
    # process-id-named temporary file, then renaming over the destination).
    path = compute_frozen_sentinel_path(actions_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scratch = path.with_name(f"{path.name}.{os.getpid()}.partial")
    scratch.write_text(json.dumps({"rollout_id": rollout_id}))
    scratch.replace(path)


# TODO ad hoc hack: revert after the args refactor
def read_frozen_rollout_id(actions_path: Path) -> int | None:
    # [FUNCTION DOC] HELPER: Returns the rollout id recorded in the sentinel file next to the
    # actions file as an integer, or none if the sentinel does not exist. Raises on malformed
    # JSON or a missing key.
    path = compute_frozen_sentinel_path(actions_path)
    if not path.is_file():
        return None
    return int(json.loads(path.read_text())["rollout_id"])
