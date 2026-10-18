# [MODULE DOC]
# HELPER: Verdict for the checkpointed hot-restart scenario: after take-overs of a run that had
# already saved checkpoints, only the training steps after the latest checkpoint were redone.
# Works purely from event logs. Each take-over rolls the live event log aside into a directory
# matching a "trash" glob inside the dump directory, and every checkpoint save writes a copy of
# the event log (in a "debug_events" folder beside each iteration directory). A step's log entry
# is the JSON of its training-step metric event; a step trained again produces a different
# entry. The module checks that rolled-aside logs, the surviving log and the checkpoint copies
# are consistent with the schedule of freezes (which iteration was saved, after which step the
# run was frozen), and returns summary data. Constants: the two glob patterns.
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.driver import ScheduledFreeze
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import TRAIN_STEP_METRIC_KEY, HotRestartRecord

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.audit_utils.event_logger.models import MetricEvent

DISCARDED_EVENTS_GLOB: str = ".trash_*"
CHECKPOINT_SNAPSHOT_GLOB: str = "iter_*/debug_events"


# ================================ redone steps ================================


@dataclass(frozen=True)
class RedoneSteps:
    # [CLASS DOC] HELPER: Immutable summary of what the checks found. Fields: for each take-over,
    # in order, the rollout id it resumed from (the last step kept), the rollout id the run had
    # reached when frozen, and a mapping from every rollout id to the number of distinct times it
    # was trained.
    resume_rollout_ids: tuple[int, ...]
    frozen_rollout_ids: tuple[int, ...]
    attempts_of_rollout_id: dict[int, int]


# ================================ the verdict =================================


def assert_only_post_checkpoint_steps_redone(
    *,
    dump_dir: str,
    checkpoint_dir: str,
    records: Sequence[HotRestartRecord],
    num_rollouts: int,
    schedule: Sequence[ScheduledFreeze],
) -> RedoneSteps:
    # [FUNCTION DOC] Main verdict. Steps: (1) assert records is non-empty and that each record was
    # triggered with a saved checkpoint and matches the schedule; (2) read the rolled-aside
    # event directories and assert there is exactly one per restart record and each has at least
    # one finished step; sort those logs by their highest step and append the surviving live log;
    # (3) assert the highest step of each rolled-aside log equals the scheduled frozen step, in
    # order; (4) for each rolled-aside log and its successor, assert the log holds a contiguous run
    # of steps from zero, and that the steps whose log entry is unchanged in the successor form a
    # contiguous prefix; that prefix end is the resume point (minus one if none); (5) assert the
    # resume points equal the scheduled saved iterations; (6) assert the surviving log holds
    # each step 0..num_rollouts-1 exactly once; (7) assert each resume point is the step set of
    # some checkpoint event-log copy; (8) count for every rollout id the distinct entries over all
    # logs, assert equality with the expected attempts of the schedule, and assert the set of
    # attempt counts is exactly one and two. Returns the summary. Raises assertion errors with
    # explanatory messages; reads files.
    assert (
        records
    ), f"{dump_dir} holds a run nothing restarted, so it says nothing about which steps a take-over redoes"
    _assert_take_overs_handed_frozen_run(records, schedule=schedule)

    discarded_dirs = read_discarded_event_dirs(dump_dir)
    assert len(discarded_dirs) == len(records), (
        f"every take-over leaves the log it rolled back behind, and {len(records)} restart(s) left "
        f"{[one.name for one in discarded_dirs]}"
    )
    discarded_logs = [_read_finished_steps(one) for one in discarded_dirs]
    for discarded_dir, log in zip(discarded_dirs, discarded_logs, strict=True):
        assert log, (
            f"{discarded_dir.name} describes no finished step, so the take-over that left it behind rolled back "
            f"the log of a run that had trained nothing"
        )

    discarded_logs.sort(key=max)
    surviving_log = _read_finished_steps(Path(dump_dir) / EVENTS_DIRNAME)
    logs = [*discarded_logs, surviving_log]

    frozen_rollout_ids = [max(one) for one in discarded_logs]
    assert frozen_rollout_ids == [one.frozen_rollout_id for one in schedule], (
        f"the run was put to sleep after the steps {[one.frozen_rollout_id for one in schedule]}, and the logs the "
        f"take-overs replaced had reached {frozen_rollout_ids}: the freezes never pinned where a take-over landed, "
        f"so what these cost was raced"
    )

    resume_rollout_ids: list[int] = []
    for index, (log, later_log) in enumerate(zip(discarded_logs, logs[1:], strict=True)):
        assert sorted(log) == list(range(max(log) + 1)), (
            f"the log replaced by restart {index} describes the steps {sorted(log)}, and a run training from zero "
            f"leaves every step up to the one it reached in it"
        )

        survived = sorted(rollout_id for rollout_id, event in log.items() if later_log.get(rollout_id) == event)
        resume = max(survived, default=-1)
        assert survived == list(range(resume + 1)), (
            f"restart {index} carried the steps {survived} over into the log that followed it; a take-over resumes "
            f"from a checkpoint, so what it keeps is a prefix and never a hole"
        )
        resume_rollout_ids.append(resume)

    assert resume_rollout_ids == [one.saved_iteration for one in schedule], (
        f"the take-overs were pinned to resume from {[one.saved_iteration for one in schedule]}, and the logs say "
        f"they resumed at {resume_rollout_ids}: the run saved at another pace, or resumed from something other "
        f"than its latest save"
    )

    assert sorted(surviving_log) == list(range(num_rollouts)), (
        f"the surviving event log describes {sorted(surviving_log)}, not each of the {num_rollouts} steps exactly "
        f"once: a take-over that trained from scratch repeats the early steps, and one that resumed past its "
        f"checkpoint skips some"
    )

    _assert_resume_points_are_checkpoints(checkpoint_dir=checkpoint_dir, resume_rollout_ids=resume_rollout_ids)

    attempts = {
        rollout_id: len({log[rollout_id] for log in logs if rollout_id in log}) for rollout_id in range(num_rollouts)
    }
    expected = compute_expected_attempts(num_rollouts=num_rollouts, schedule=schedule)
    assert attempts == expected, (
        f"over the (save, frozen step] windows "
        f"{[(one.saved_iteration, one.frozen_rollout_id) for one in schedule]} a take-over redoes exactly the steps "
        f"inside them, so every step is written {expected} time(s); this run left {attempts}"
    )
    assert sorted(set(attempts.values())) == [1, 2], (
        f"a hot restart wastes only the steps its checkpoint does not cover, so every step is trained once or "
        f"twice; these were trained {sorted(set(attempts.values()))} time(s): {attempts}"
    )

    return RedoneSteps(
        resume_rollout_ids=tuple(resume_rollout_ids),
        frozen_rollout_ids=tuple(frozen_rollout_ids),
        attempts_of_rollout_id=attempts,
    )


# ============================= expected attempts ==============================


def compute_expected_attempts(*, num_rollouts: int, schedule: Sequence[ScheduledFreeze]) -> dict[int, int]:
    # [FUNCTION DOC] Return, for each rollout id from 0 to num_rollouts-1, the expected number of
    # times it is trained: one plus the number of scheduled freezes whose redone window contains
    # it (window: after the saved iteration, up to and including the frozen step; a missing saved
    # iteration counts as minus one, so the window starts at step 0).
    return {
        rollout_id: 1 + sum(1 for one in schedule if _is_inside_redone_window(rollout_id, scheduled=one))
        for rollout_id in range(num_rollouts)
    }


def _is_inside_redone_window(rollout_id: int, *, scheduled: ScheduledFreeze) -> bool:
    # [FUNCTION DOC] HELPER: True when the rollout id is strictly greater than the resume point (the
    # saved iteration, or minus one if none) and at most the frozen rollout id.
    resumed_from = -1 if scheduled.saved_iteration is None else scheduled.saved_iteration
    return resumed_from < rollout_id <= scheduled.frozen_rollout_id


# =========================== supporting assertions ============================


def _assert_take_overs_handed_frozen_run(
    records: Sequence[HotRestartRecord], *, schedule: Sequence[ScheduledFreeze]
) -> None:
    # [FUNCTION DOC] HELPER: Assert every restart record was fired against a run holding a
    # checkpoint (saved iteration at trigger not missing), and that the list of (saved iteration at
    # trigger, frozen rollout id) pairs of the records equals the scheduled pairs, in order.
    triggered = [
        (one.saved_iteration_at_trigger, one.frozen_rollout_id)
        for one in records
        if one.saved_iteration_at_trigger is not None
    ]
    assert len(triggered) == len(records), (
        f"this scenario takes over a run that has saved something, and {len(records) - len(triggered)} of its "
        f"restart(s) fired against a run holding no checkpoint: {records}"
    )
    assert triggered == [(one.saved_iteration, one.frozen_rollout_id) for one in schedule], (
        f"the schedule pins the (save, frozen step) pairs "
        f"{[(one.saved_iteration, one.frozen_rollout_id) for one in schedule]}, and the driver was handed "
        f"{triggered}"
    )


def _assert_resume_points_are_checkpoints(*, checkpoint_dir: str, resume_rollout_ids: Sequence[int]) -> None:
    # [FUNCTION DOC] HELPER: Assert that each resume point corresponds to a real checkpoint. Reads
    # the event-log copies saved beside the checkpoints (must be non-empty) and maps each
    # checkpoint iteration directory name to its sorted finished steps. For every resume point
    # (index and value), requires some copy whose steps are exactly 0..resume, otherwise fails
    # saying the take-over resumed from something no save wrote.
    snapshot_dirs = read_checkpoint_snapshot_dirs(checkpoint_dir)
    assert snapshot_dirs, (
        f"{checkpoint_dir} holds no event log snapshot beside any checkpoint, so what the run resumed from was "
        f"not a checkpoint of this run"
    )

    steps_of_snapshot = {one.parent.name: sorted(_read_finished_steps(one)) for one in snapshot_dirs}
    for index, resume in enumerate(resume_rollout_ids):
        matching = sorted(name for name, steps in steps_of_snapshot.items() if steps == list(range(resume + 1)))
        assert matching, (
            f"restart {index} resumed a run that had finished the steps {list(range(resume + 1))}, and the "
            f"snapshots beside this run's checkpoints hold {steps_of_snapshot}: the take-over resumed from "
            f"something no save wrote"
        )


# =========================== reading the event logs ===========================


def _read_finished_steps(events_dir: Path) -> dict[int, str]:
    # [FUNCTION DOC] HELPER: Read the training-step events under the directory and assert no
    # rollout id appears more than once (a take-over rolls the log back before redoing). Returns a
    # mapping from rollout id to the single JSON entry.
    events_of_rollout_id = read_step_events(events_dir)

    repeated = {rollout_id: len(logged) for rollout_id, logged in events_of_rollout_id.items() if len(logged) > 1}
    assert not repeated, (
        f"{events_dir} describes the steps {repeated} more than once each; a take-over rolls the log back before "
        f"redoing anything, so one log covers each step exactly once"
    )
    return {rollout_id: logged[0] for rollout_id, logged in events_of_rollout_id.items()}


def read_step_events(events_dir: Path) -> dict[int, list[str]]:
    # [FUNCTION DOC] Read all events under the directory and keep metric events that have a
    # rollout id and contain the training-step metric key. Returns a mapping from rollout id to
    # the list of those events serialized as JSON strings, in read order, with keys sorted
    # ascending.
    events_of_rollout_id: dict[int, list[str]] = {}
    for event in read_events(events_dir):
        if isinstance(event, MetricEvent) and event.rollout_id is not None and TRAIN_STEP_METRIC_KEY in event.metrics:
            events_of_rollout_id.setdefault(event.rollout_id, []).append(event.model_dump_json())
    return dict(sorted(events_of_rollout_id.items()))


def read_discarded_event_dirs(dump_dir: str) -> list[Path]:
    # [FUNCTION DOC] Return the sorted list of directories directly inside the dump directory that
    # match the "trash" glob, i.e. the event logs moved aside by take-overs.
    return sorted(Path(dump_dir).glob(DISCARDED_EVENTS_GLOB))


def read_checkpoint_snapshot_dirs(checkpoint_dir: str) -> list[Path]:
    # [FUNCTION DOC] Return the sorted list of event-log copy directories saved beside the
    # checkpoints: those matching the pattern "iter_*/debug_events" under the checkpoint directory.
    return sorted(Path(checkpoint_dir).glob(CHECKPOINT_SNAPSHOT_GLOB))
