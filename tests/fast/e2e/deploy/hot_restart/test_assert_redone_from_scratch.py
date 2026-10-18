# [MODULE DOC]
# Tests for the assertion that a run frozen before its first checkpoint save was redone from
# scratch by its single take-over (all steps trained up to the frozen step are trained again, all
# others once, with a save only after the take-over). A small harness simulates runs in a
# temporary dump directory using the real event logger and event-log checkpoint helpers: finished
# steps are metric events, saving snapshots the log and writes the checkpoint tracker, and a
# take-over restores from the checkpoint and, if the tracker holds no iteration, discards the log
# (moving it aside). The default schedule freezes at rollout 1 with no saved iteration, and the
# default record agrees; each test perturbs one aspect and checks the failure message.
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_redone_from_scratch import (
    RedoneFromScratch,
    assert_unsaved_run_redone_from_scratch,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import ScheduledFreeze
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_hot_restart_deterministic import compute_checkpoint_dir

from miles.backends.training_utils.checkpoint.tracker import read_checkpoint_tracker_iteration
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, EventLogger
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

TRACKER_FILENAME: str = "latest_checkpointed_iteration.txt"
SCHEDULE: tuple[ScheduledFreeze, ...] = (ScheduledFreeze(frozen_rollout_id=1, saved_iteration=None),)


def _write_finished_step(events_dir: Path, *, rollout_id: int) -> None:
    # [FUNCTION DOC] HELPER: Appends one metric event (grad norm 1.0 for the given rollout id) to
    # the "main" event log file in the events directory, without printing.
    logger = EventLogger(log_dir=events_dir, file_name="main.jsonl", source=SimpleProcessIdentity(component="main"))
    logger.log(MetricEvent, {"rollout_id": rollout_id, "metrics": {"train/grad_norm": 1.0}}, print_log=False)


@dataclass(frozen=True)
class _Run:
    # [CLASS DOC] HELPER: Simulated run rooted at a dump directory, with the events directory,
    # checkpoint directory, megatron-like arguments and operations to train steps, save, take
    # over (restoring and, with no checkpoint tracker iteration, discarding the log) and run the
    # assertion under test.
    dump_dir: Path

    @property
    def events_dir(self) -> Path:
        # [FUNCTION DOC] HELPER: The event log directory under the dump directory.
        return self.dump_dir / EVENTS_DIRNAME

    @property
    def checkpoint_dir(self) -> Path:
        # [FUNCTION DOC] HELPER: The checkpoint directory computed by the scenario from the dump
        # directory.
        return Path(compute_checkpoint_dir(str(self.dump_dir)))

    @property
    def megatron_args(self) -> Namespace:
        # [FUNCTION DOC] HELPER: Arguments namespace with save, load and requested-load at the
        # checkpoint directory, no megatron config, and the event save path at the events dir.
        return Namespace(
            save=str(self.checkpoint_dir),
            load=str(self.checkpoint_dir),
            requested_load=str(self.checkpoint_dir),
            megatron_config=None,
            save_debug_event_data=str(self.events_dir),
        )

    def train(self, *rollout_ids: int) -> None:
        # [FUNCTION DOC] HELPER: Writes a finished-step event for each given rollout id in order.
        for rollout_id in rollout_ids:
            _write_finished_step(self.events_dir, rollout_id=rollout_id)

    def save(self, iteration: int) -> None:
        # [FUNCTION DOC] HELPER: Snapshots the event log for the iteration and writes the
        # iteration into the checkpoint tracker file.
        event_logger_checkpoint.snapshot(self.megatron_args, iteration)
        (self.checkpoint_dir / TRACKER_FILENAME).write_text(str(iteration))

    def take_over(self) -> None:
        # [FUNCTION DOC] HELPER: Restores the event log from the checkpoint; if the checkpoint
        # tracker holds no iteration (nothing was saved), additionally discards the current log
        # (moves it aside) so the new script starts from scratch.
        event_logger_checkpoint.restore(self.megatron_args)
        if read_checkpoint_tracker_iteration(self.checkpoint_dir) is None:
            event_logger_checkpoint.discard(self.megatron_args)

    def assert_redone_from_scratch(
        self,
        *,
        records: list[HotRestartRecord] | None = None,
        schedule: tuple[ScheduledFreeze, ...] = SCHEDULE,
        num_rollouts: int = 6,
    ) -> RedoneFromScratch:
        # [FUNCTION DOC] HELPER: Runs the assertion under test with the given records (default:
        # one record with no saved iteration and frozen rollout 1), schedule (default one freeze)
        # and rollout count (default 6). Returns the summary (frozen rollout id and attempts per
        # rollout); failures propagate as assertion errors.
        return assert_unsaved_run_redone_from_scratch(
            dump_dir=str(self.dump_dir),
            checkpoint_dir=str(self.checkpoint_dir),
            records=_records() if records is None else records,
            num_rollouts=num_rollouts,
            schedule=schedule,
        )


def _records() -> list[HotRestartRecord]:
    # [FUNCTION DOC] HELPER: Returns the single expected driver record: index 0, no saved
    # iteration at trigger time, frozen at rollout 1.
    return [HotRestartRecord(index=0, saved_iteration_at_trigger=None, frozen_rollout_id=1)]


def _run_restarted_before_save(tmp_path: Path) -> _Run:
    # [FUNCTION DOC] HELPER: Simulates the ideal run: train 0,1; take over (nothing saved, so the
    # log is discarded); train 0..3; save iteration 3; train 4,5. Returns the run.
    run = _Run(dump_dir=tmp_path)
    run.train(0, 1)
    run.take_over()
    run.train(0, 1, 2, 3)
    run.save(3)
    run.train(4, 5)
    return run


class TestAssertARunThatHadSavedNothingWasRedoneFromScratch:
    # [CLASS DOC] Tests of the redone-from-scratch assertion over simulated runs.
    def test_a_take_over_of_the_frozen_run_that_threw_away_its_two_steps_passes(self, tmp_path):
        # [FUNCTION DOC] The ideal run passes; the summary reports frozen rollout 1 and attempts
        # per rollout {0:2, 1:2, 2:1, 3:1, 4:1, 5:1}.
        """The run was frozen after step 1 holding nothing, so exactly steps 0 and 1 are trained twice."""
        run = _run_restarted_before_save(tmp_path)

        redone = run.assert_redone_from_scratch()

        assert redone.frozen_rollout_id == 1
        assert redone.attempts_of_rollout_id == {0: 2, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1}

    def test_a_carried_over_step_fails(self, tmp_path):
        # [FUNCTION DOC] A save at iteration 0 before the freeze lets the take-over restore step 0
        # from the snapshot instead of retraining it: raises matching "carried them over".
        """A save before the freeze leaves a snapshot to restore, and the restored steps are not retrained ones."""
        run = _Run(dump_dir=tmp_path)
        run.train(0)
        run.save(0)
        run.train(1)
        run.take_over()
        run.train(1, 2, 3, 4, 5)

        with pytest.raises(AssertionError, match="carried them over"):
            run.assert_redone_from_scratch()

    def test_a_record_that_had_a_checkpoint_at_trigger_time_fails(self, tmp_path):
        # [FUNCTION DOC] A record saying the run had saved iteration 0 at trigger time: raises
        # matching "had saved iteration".
        """The record is the only thing saying which of the two take-over paths this dump describes."""
        run = _run_restarted_before_save(tmp_path)

        with pytest.raises(AssertionError, match="had saved iteration"):
            run.assert_redone_from_scratch(
                records=[HotRestartRecord(index=0, saved_iteration_at_trigger=0, frozen_rollout_id=1)]
            )

    def test_a_take_over_handed_a_run_frozen_at_another_step_fails(self, tmp_path):
        # [FUNCTION DOC] A record with frozen rollout 2 while the schedule pins 1: raises matching
        # "never pinned where it landed".
        """The freeze is what pins how much work the take-over throws away."""
        run = _run_restarted_before_save(tmp_path)

        with pytest.raises(AssertionError, match="never pinned where it landed"):
            run.assert_redone_from_scratch(
                records=[HotRestartRecord(index=0, saved_iteration_at_trigger=None, frozen_rollout_id=2)]
            )

    def test_more_than_one_restart_fails(self, tmp_path):
        # [FUNCTION DOC] Two identical records (two restarts): raises matching "taken over once".
        """Every take-over after the first resumes from what the one before it made the run save."""
        run = _run_restarted_before_save(tmp_path)

        with pytest.raises(AssertionError, match="taken over once"):
            run.assert_redone_from_scratch(records=_records() * 2)

    def test_a_schedule_freezing_the_run_more_than_once_fails(self, tmp_path):
        # [FUNCTION DOC] A schedule with two freezes (the second at rollout 4 with iteration 3):
        # raises matching "taken over once".
        """The second freeze would find a run holding a checkpoint of its own."""
        run = _run_restarted_before_save(tmp_path)

        with pytest.raises(AssertionError, match="taken over once"):
            run.assert_redone_from_scratch(
                schedule=(*SCHEDULE, ScheduledFreeze(frozen_rollout_id=4, saved_iteration=3))
            )

    def test_a_schedule_freezing_the_run_on_a_checkpoint_fails(self, tmp_path):
        # [FUNCTION DOC] A schedule freezing at rollout 2 with saved iteration 1 (and a matching
        # record with no saved iteration): raises matching "already covers".
        """That take-over resumes from a save, which is exactly what the checkpointed mode measures."""
        run = _run_restarted_before_save(tmp_path)

        with pytest.raises(AssertionError, match="already covers"):
            run.assert_redone_from_scratch(
                schedule=(ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),),
                records=[HotRestartRecord(index=0, saved_iteration_at_trigger=None, frozen_rollout_id=2)],
            )

    def test_a_freeze_later_than_the_schedule_fails(self, tmp_path):
        # [FUNCTION DOC] The frozen run trained steps 0..2 although the schedule freezes at 1:
        # raises matching "every step the frozen run had trained".
        """The log moved aside is what the frozen run had trained, so it ends where the freeze says."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1, 2)
        run.take_over()
        run.train(0, 1, 2, 3)
        run.save(3)
        run.train(4, 5)

        with pytest.raises(AssertionError, match="every step the frozen run had trained"):
            run.assert_redone_from_scratch()

    def test_a_restart_past_step_zero_fails(self, tmp_path):
        # [FUNCTION DOC] After the take-over the new log resumes at step 1 instead of 0: raises
        # matching "not each of the 6 steps exactly once".
        """Resuming at step 1 is a resume from a checkpoint, however it came about."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.take_over()
        run.train(1, 2, 3)
        run.save(3)
        run.train(4, 5)

        with pytest.raises(AssertionError, match="not each of the 6 steps exactly once"):
            run.assert_redone_from_scratch()

    def test_a_second_discarded_log_fails(self, tmp_path):
        # [FUNCTION DOC] Two take-overs without a save in between produce two discarded logs:
        # raises matching "instead of exactly one".
        """One take-over throws away one run's worth of work, and a third attempt is a second take-over."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.take_over()
        run.train(0, 1)
        run.take_over()
        run.train(0, 1, 2, 3)
        run.save(3)
        run.train(4, 5)

        with pytest.raises(AssertionError, match="instead of exactly one"):
            run.assert_redone_from_scratch()

    def test_a_step_logged_twice_fails(self, tmp_path):
        # [FUNCTION DOC] After the take-over, step 0 is written twice into the same log: raises
        # matching "more than once".
        """This is the symptom the product fix removes: one log holding both attempts at a redone step."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.take_over()
        run.train(0, 0, 1, 2, 3)
        run.save(3)
        run.train(4, 5)

        with pytest.raises(AssertionError, match="more than once"):
            run.assert_redone_from_scratch()

    def test_a_run_that_never_finished_every_step_fails(self, tmp_path):
        # [FUNCTION DOC] Asking for 7 rollouts from a run that trained 0..5: raises matching "not
        # each of the 7 steps exactly once".
        """A comparison over fewer steps than the run was asked for would quietly prove less."""
        run = _run_restarted_before_save(tmp_path)

        with pytest.raises(AssertionError, match="not each of the 7 steps exactly once"):
            run.assert_redone_from_scratch(num_rollouts=7)

    def test_a_run_that_never_saved_after_it_was_restarted_fails(self, tmp_path):
        # [FUNCTION DOC] The restarted run trains 0..5 but never saves: raises matching "nothing
        # about saving".
        """The point of restarting before the first save is to watch the save that follows it."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.take_over()
        run.train(0, 1, 2, 3, 4, 5)

        with pytest.raises(AssertionError, match="nothing about saving"):
            run.assert_redone_from_scratch()

    def test_a_checkpoint_predating_the_freeze_fails(self, tmp_path):
        # [FUNCTION DOC] The only save (iteration 1) happens after the restarted run finished all
        # steps, so nothing was saved after the take-over's work: raises matching "nothing here
        # was saved after the take-over".
        # KNOWN ISSUE: the name says the checkpoint predates the freeze, yet the save is simulated
        # after the restarted run's training.
        """A save the take-over threw away says nothing about the run saving once it had been restarted."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.take_over()
        run.train(0, 1, 2, 3, 4, 5)
        run.save(1)

        with pytest.raises(AssertionError, match="nothing here was saved after the take-over"):
            run.assert_redone_from_scratch()
