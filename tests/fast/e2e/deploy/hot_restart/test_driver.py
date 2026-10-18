# [MODULE DOC]
# Tests for the hot-restart driver: validation of scheduled freezes (a freeze must leave steps to
# redo and start at step 0 or later), the freeze plan content, the hot restart argument naming the
# replaced components, the refusal to relaunch a different release, driver start preconditions,
# the guard against work lost outside a take-over (with regressions around exemption latching),
# the conditions under which a run counts as frozen at the pinned step (progress, sentinel file),
# the record a take-over leaves (with save cadence checks), the take-over loop (waiting, arming
# the next freeze on each relaunch, ending after the last take-over, timeouts, retry of failed
# reads), the final verdict (including SIGTERM exits of replaced launches), the context manager
# that drives take-overs around a run, continued sampling after the last take-over, behaviour
# when the run cannot be read, and tolerance of a checkpoint tracker lagging behind the freeze.
# The cluster observer, progress reader and sentinel reader are replaced by scripted fakes and
# the relaunch callable by a recorder; threads and short polling intervals are used where the
# driver loop must run.
import signal
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart import driver as driver_module
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import (
    HOT_RESTART_ARG,
    HotRestartDriver,
    ScheduledFreeze,
    compute_freeze_plan,
    driving_hot_restarts,
    relaunch_with_hot_restart,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord, RunProgress
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.fast.e2e.deploy.hot_restart.cluster_facts import NAMESPACE, RELEASE

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.launcher.entrypoint import RunExitedError
from miles.utils.test_utils.ft_test_actions import SLEEP_FOREVER_AT_END_ACTION, FTTestAction
from miles.utils.workers.types import HotRestartComponent

_CHECKPOINTED: ScheduledFreeze = ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1)
_FROM_SCRATCH: ScheduledFreeze = ScheduledFreeze(frozen_rollout_id=1, saved_iteration=None)
_SCHEDULE: tuple[ScheduledFreeze, ...] = (
    ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),
    ScheduledFreeze(frozen_rollout_id=4, saved_iteration=3),
)
_DRIVE_TIMEOUT_SECONDS: float = 30.0
_DRIVE_POLL_SECONDS: float = 0.01


class TestScheduledFreeze:
    # [CLASS DOC] Validation tests of the scheduled freeze value object.
    def test_a_freeze_past_the_checkpoint_it_resumes_from_is_accepted(self):
        # [FUNCTION DOC] A freeze at rollout 2 with saved iteration 1 is constructed and keeps
        # saved iteration 1.
        """This is the whole window: a checkpoint to resume from, and a step past it to redo."""
        assert ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1).saved_iteration == 1

    def test_a_freeze_before_the_first_save_is_accepted(self):
        # [FUNCTION DOC] A freeze at rollout 1 with no saved iteration is accepted.
        """A run holding no checkpoint at all resumes from the reference weights, which is a scenario of its own."""
        assert ScheduledFreeze(frozen_rollout_id=1, saved_iteration=None).saved_iteration is None

    def test_a_freeze_at_the_step_its_own_checkpoint_covers_is_refused(self):
        # [FUNCTION DOC] Frozen rollout 2 with saved iteration 2 raises an assertion error
        # matching "nothing would be redone".
        """Such a take-over resumes exactly where the script it replaced stood, so it redoes nothing."""
        with pytest.raises(AssertionError, match="nothing would be redone"):
            ScheduledFreeze(frozen_rollout_id=2, saved_iteration=2)

    def test_a_freeze_before_the_run_starts_is_refused(self):
        # [FUNCTION DOC] Frozen rollout -1 raises an assertion error matching "a run starts at
        # step 0".
        """A run frozen before step 0 has trained nothing, so a take-over of it costs nothing."""
        with pytest.raises(AssertionError, match="a run starts at step 0"):
            ScheduledFreeze(frozen_rollout_id=-1, saved_iteration=None)


class TestTheFreezePlan:
    # [CLASS DOC] Tests of the freeze plan computation.
    def test_the_plan_arms_the_sleep_forever_action_at_the_pinned_step(self):
        # [FUNCTION DOC] The plan for rollout 3 is a single fault-test action at rollout 3 with
        # the sleep-forever-at-end action.
        """The run reads this plan, and it is the only thing that decides where the run stands still."""
        [action] = [FTTestAction(**one) for one in compute_freeze_plan(3)]

        assert action == FTTestAction(at_rollout=3, action=SLEEP_FOREVER_AT_END_ACTION)

    def test_a_run_with_no_freeze_left_carries_an_empty_plan(self):
        # [FUNCTION DOC] The plan for no freeze (none) is an empty list.
        """The last relaunch has to train to the end, and a plan repeating the old step would freeze it again."""
        assert compute_freeze_plan(None) == []


class TestHotRestartArg:
    # [CLASS DOC] Test of the hot restart argument value.
    def test_the_flag_names_both_components_a_take_over_replaces(self):
        # [FUNCTION DOC] The comma-separated argument value lists exactly the values of the hot
        # restart component enumeration (compared sorted).
        """A hot restart replaces the orchestration script and the rollout executor together or not at all."""
        assert sorted(HOT_RESTART_ARG.split(",")) == sorted(one.value for one in HotRestartComponent)


class TestRelaunchWithHotRestart:
    # [CLASS DOC] Test of the relaunch entry's release check.
    def test_a_relaunch_that_would_install_another_release_is_refused(self):
        # [FUNCTION DOC] A relaunch whose configuration (run id "demo") yields a different
        # release than the installed release name raises an assertion error matching "already up".
        """A relaunch building its own config gets a run id of its own and leaves the run behind."""
        mode = FTTestMode(
            model_name="demo", model_hf_repo="demo/demo", megatron_model_type="demo", num_cells=1, parallel_args=""
        )

        with pytest.raises(AssertionError, match="already up"):
            relaunch_with_hot_restart(
                train_args="",
                mode=mode,
                config=ExecuteTrainConfig(run_id="demo"),
                installed_release="miles-run-someone-else-all",
            )


def _driver(tmp_path: Path, **overrides: Any) -> HotRestartDriver:
    # [FUNCTION DOC] HELPER: Builds a driver with a no-op relaunch, checkpoint and events
    # directories under the temporary path, the demo release and namespace, trainer id "actor", a
    # freeze plan path under the path, the two-freeze schedule and zero settle and poll
    # intervals. Keyword overrides replace any of these arguments.
    kwargs: dict[str, Any] = dict(
        relaunch=lambda _frozen_rollout_id: None,
        checkpoint_dir=tmp_path / "checkpoints",
        events_dir=tmp_path / "events",
        release=RELEASE,
        namespace=NAMESPACE,
        trainer_id="actor",
        freeze_plan_path=tmp_path / "hot_restart" / "target_freeze_plan.json",
        schedule=_SCHEDULE,
        tracker_settle_interval_seconds=0.0,
        poll_interval_seconds=0.0,
    )
    kwargs.update(overrides)
    return HotRestartDriver(**kwargs)


def _join_relaunches(driver: HotRestartDriver) -> None:
    # [FUNCTION DOC] HELPER: Joins every relaunch thread the driver started, waiting up to 30
    # seconds for each.
    for thread in driver._relaunch_threads:
        thread.join(timeout=30.0)


def _drive_until_take_overs_landed(driver: HotRestartDriver, *, take_overs: int) -> None:
    # [FUNCTION DOC] HELPER: Runs the driver's private drive loop in a thread, polling every 10 ms
    # (up to 30 s) until the driver has the requested number of records or the thread ends, then
    # sets the stop event and joins (up to 30 s). Asserts the thread stopped, with a message about
    # records racing a still-driving driver.
    stop_event = threading.Event()
    thread = threading.Thread(target=driver._drive, args=(stop_event,), name="hot-restart-driver-under-test")
    deadline = time.monotonic() + _DRIVE_TIMEOUT_SECONDS

    thread.start()
    while len(driver.records) < take_overs and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(_DRIVE_POLL_SECONDS)

    stop_event.set()
    thread.join(timeout=_DRIVE_TIMEOUT_SECONDS)

    assert not thread.is_alive(), (
        f"the hot restart driver was still driving {_DRIVE_TIMEOUT_SECONDS}s after it was asked to stop, so the "
        f"records about to be read would race it"
    )


def _install_frozen_at(monkeypatch, rollout_id: int | None) -> None:
    # [FUNCTION DOC] HELPER: Replaces the driver module's frozen-sentinel reader with one that
    # always returns the given rollout id (or none for not frozen).
    monkeypatch.setattr(driver_module, "read_frozen_rollout_id", lambda _path: rollout_id)


def _install_progress(monkeypatch, reported: list[RunProgress]) -> None:
    # [FUNCTION DOC] HELPER: Scripts the run: the progress reader returns the reported values one
    # per call and then keeps repeating the last; the cluster observer's warn-observe is a no-op;
    # the frozen-sentinel reader reports the most recently read progress's last finished rollout
    # id (or none before any read), i.e. the run is always parked where it last reported.
    remaining = list(reported)
    latest: list[RunProgress] = []

    def read(**_kwargs) -> RunProgress:
        # [FUNCTION DOC] HELPER: Fake progress reader; records and returns the next scripted value.
        latest.append(remaining.pop(0) if remaining else reported[-1])
        return latest[-1]

    monkeypatch.setattr(driver_module, "read_run_progress", read)
    monkeypatch.setattr(driver_module.ClusterObserver, "observe_once_or_warn", lambda _self: None)
    monkeypatch.setattr(
        driver_module,
        "read_frozen_rollout_id",
        lambda _path: latest[-1].last_finished_rollout_id if latest else None,
    )


class TestHotRestartDriverStart:
    # [CLASS DOC] Tests of driver construction and start preconditions.
    def test_a_dump_directory_holding_a_previous_run_is_refused(self, tmp_path):
        # [FUNCTION DOC] If the checkpoint directory already exists, starting the driver raises an
        # assertion error matching "before this run was even installed".
        """Resuming from what a previous run left would compare a run that never started from nothing."""
        driver = _driver(tmp_path)
        (tmp_path / "checkpoints").mkdir()

        with pytest.raises(AssertionError, match="before this run was even installed"):
            driver.start()

    def test_a_run_nothing_would_freeze_is_refused(self, tmp_path):
        # [FUNCTION DOC] Constructing a driver with an empty schedule raises an assertion error
        # matching "nothing freezes".
        """The driver waits for a frozen run, and a run with no plan never stands still for it."""
        with pytest.raises(AssertionError, match="nothing freezes"):
            _driver(tmp_path, schedule=())


class TestHotRestartDriverProgressGuard:
    # [CLASS DOC] Tests of the guard that fails when finished steps decrease outside a take-over.
    def test_a_run_whose_event_log_shrank_before_any_restart_fails(self, tmp_path):
        # [FUNCTION DOC] Progress at last finished rollout 2 then 1 with no records: raises an
        # assertion error matching "lost work".
        """Nothing but a take-over rolls the log back, so this is a run reading someone else's dumps."""
        driver = _driver(tmp_path)
        driver._assert_no_step_lost_outside_take_over(RunProgress(last_saved_iteration=1, last_finished_rollout_id=2))

        with pytest.raises(AssertionError, match="lost work"):
            driver._assert_no_step_lost_outside_take_over(
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=1)
            )

    def test_the_rollback_a_take_over_performs_is_not_read_as_lost_work(self, tmp_path):
        # [FUNCTION DOC] After a recorded take-over, progress going from 2 back to 1 does not
        # raise (the rollback to the checkpoint is expected).
        """The log going back to the checkpoint is the very thing a hot restart is supposed to do."""
        driver = _driver(tmp_path)
        driver.records.append(HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2))
        driver._assert_no_step_lost_outside_take_over(RunProgress(last_saved_iteration=1, last_finished_rollout_id=2))

        driver._assert_no_step_lost_outside_take_over(RunProgress(last_saved_iteration=1, last_finished_rollout_id=1))

    def test_a_replacement_that_loses_work_after_it_caught_up_still_fails(self, tmp_path):
        # [FUNCTION DOC] Regression: after a take-over record, progress 1 then 2 (caught up to the
        # frozen step) ends the exemption; a later drop to 1 raises matching "lost work".
        """Regression: the exemption covered the whole take-over, so a later rollback stayed silent."""
        driver = _driver(tmp_path)
        driver._assert_no_step_lost_outside_take_over(RunProgress(last_saved_iteration=1, last_finished_rollout_id=2))
        driver.records.append(HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2))
        for finished in (1, 2):
            driver._assert_no_step_lost_outside_take_over(
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=finished)
            )

        with pytest.raises(AssertionError, match="lost work"):
            driver._assert_no_step_lost_outside_take_over(
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=1)
            )

    def test_a_replacement_that_jumped_straight_past_the_mark_is_no_longer_exempt(self, tmp_path):
        # [FUNCTION DOC] Regression: after a take-over record, progress 1 then 3 (above the frozen
        # step 2) also ends the exemption; a later drop to 1 raises matching "lost work".
        """Regression: reporting above the frozen step left the latch shut, so a later rollback stayed silent."""
        driver = _driver(tmp_path)
        driver._assert_no_step_lost_outside_take_over(RunProgress(last_saved_iteration=1, last_finished_rollout_id=2))
        driver.records.append(HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2))
        for finished in (1, 3):
            driver._assert_no_step_lost_outside_take_over(
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=finished)
            )

        with pytest.raises(AssertionError, match="lost work"):
            driver._assert_no_step_lost_outside_take_over(
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=1)
            )


class TestTheFreezeATakeOverWaitsFor:
    # [CLASS DOC] Tests of the frozen-run predicate: true only when progress is at the pinned step
    # and the run wrote the frozen sentinel for that step; raises if the run trained past it.
    def test_a_run_that_has_finished_nothing_is_not_frozen_yet(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] No finished rollout and no sentinel: not frozen (false).
        """Polling starts before the pods are even up."""
        _install_frozen_at(monkeypatch, None)

        assert not _driver(tmp_path)._stands_frozen_at(
            _CHECKPOINTED, progress=RunProgress(last_saved_iteration=None, last_finished_rollout_id=None)
        )

    def test_a_run_short_of_the_pinned_step_is_not_frozen_yet(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Last finished rollout 1 against the pinned 2, no sentinel: not frozen.
        """Taking a run over one step early would resume it from another checkpoint than the pinned one."""
        _install_frozen_at(monkeypatch, None)

        assert not _driver(tmp_path)._stands_frozen_at(
            _CHECKPOINTED, progress=RunProgress(last_saved_iteration=1, last_finished_rollout_id=1)
        )

    def test_a_run_standing_at_the_pinned_step_that_has_not_parked_is_not_frozen_yet(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Last finished rollout equals the pinned 2 but no sentinel exists: not
        # frozen (the run has not parked yet).
        """The metric event is written before the sleep, so the step being over is not the run being parked."""
        _install_frozen_at(monkeypatch, None)

        assert not _driver(tmp_path)._stands_frozen_at(
            _CHECKPOINTED, progress=RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)
        )

    def test_the_run_that_wrote_the_sentinel_at_the_pinned_step_is_the_frozen_one(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Sentinel value 2 and last finished rollout 2 with the pinned freeze at 2:
        # frozen (true).
        """The run writes it from inside its sleep loop, so it can no longer read a rewritten plan."""
        _install_frozen_at(monkeypatch, 2)

        assert _driver(tmp_path)._stands_frozen_at(
            _CHECKPOINTED, progress=RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)
        )

    def test_a_sentinel_left_by_the_previous_take_over_is_not_read_as_this_one(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Sentinel still holds the previous freeze's 2 while the pinned freeze is at
        # 4 with progress 3: not frozen.
        """Each freeze overwrites it, so the value only matches once the new run has parked."""
        _install_frozen_at(monkeypatch, 2)

        assert not _driver(tmp_path)._stands_frozen_at(
            ScheduledFreeze(frozen_rollout_id=4, saved_iteration=3),
            progress=RunProgress(last_saved_iteration=3, last_finished_rollout_id=3),
        )

    def test_a_run_that_trained_past_the_pinned_step_fails_instead_of_firing(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Last finished rollout 3 beyond the pinned 2 with no sentinel: raises an
        # assertion error matching "never fired".
        """The freeze is what makes the take-over exact, and a run that moved on was never frozen."""
        _install_frozen_at(monkeypatch, None)

        with pytest.raises(AssertionError, match="never fired"):
            _driver(tmp_path)._stands_frozen_at(
                _CHECKPOINTED, progress=RunProgress(last_saved_iteration=3, last_finished_rollout_id=3)
            )


class TestTheRecordATakeOverLeaves:
    # [CLASS DOC] Tests of the take-over record computation (index, scheduled freeze, progress).
    def test_the_record_is_the_pinned_take_over_the_run_was_frozen_for(self, tmp_path):
        # [FUNCTION DOC] Saved iteration 1 and finished rollout 2 for the checkpointed freeze give
        # a record with index 0, saved iteration 1 and frozen rollout 2.
        """The comparison reads the window off this record, and the freeze makes it the scheduled one."""
        record = _driver(tmp_path)._compute_record(
            index=0,
            scheduled=_CHECKPOINTED,
            progress=RunProgress(last_saved_iteration=1, last_finished_rollout_id=2),
        )

        assert record == HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2)

    def test_a_run_holding_no_checkpoint_is_recorded_as_such(self, tmp_path):
        # [FUNCTION DOC] No saved iteration and finished rollout 1 for the from-scratch freeze
        # give a record with saved iteration none and frozen rollout 1.
        """Only the record tells the comparison which of the two take-over paths this dump describes."""
        record = _driver(tmp_path)._compute_record(
            index=0,
            scheduled=_FROM_SCRATCH,
            progress=RunProgress(last_saved_iteration=None, last_finished_rollout_id=1),
        )

        assert record == HotRestartRecord(index=0, saved_iteration_at_trigger=None, frozen_rollout_id=1)

    def test_a_run_saving_at_another_pace_than_the_scenario_pinned_records_nothing(self, tmp_path):
        # [FUNCTION DOC] Saved iteration 2 where the freeze pins 1: raises an assertion error
        # matching "save cadence".
        """The pinned save is what the take-over rolls back to, so a run that wrote another one lands elsewhere."""
        with pytest.raises(AssertionError, match="save cadence"):
            _driver(tmp_path)._compute_record(
                index=0,
                scheduled=_CHECKPOINTED,
                progress=RunProgress(last_saved_iteration=2, last_finished_rollout_id=2),
            )

    def test_a_run_that_saved_before_the_take_over_of_a_run_holding_nothing_records_nothing(self, tmp_path):
        # [FUNCTION DOC] Saved iteration 0 where the from-scratch freeze pins none: raises an
        # assertion error matching "save cadence".
        """A save here means the take-over walks the path the checkpointed mode already covers."""
        with pytest.raises(AssertionError, match="save cadence"):
            _driver(tmp_path)._compute_record(
                index=0,
                scheduled=_FROM_SCRATCH,
                progress=RunProgress(last_saved_iteration=0, last_finished_rollout_id=1),
            )


class TestTheTakeOverLoop:
    # [CLASS DOC] Tests of the driver's private drive loop with scripted progress.
    def test_nothing_is_relaunched_until_the_run_stands_at_the_pinned_step(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] With progress short of the pinned step and the stop event already set,
        # the loop produces no records and never calls the relaunch.
        """A take-over of a run still training would land wherever the poll happened to fall."""
        armed: list[int | None] = []
        driver = _driver(tmp_path, relaunch=armed.append)
        _install_progress(monkeypatch, [RunProgress(last_saved_iteration=1, last_finished_rollout_id=1)])

        stop_event = threading.Event()
        stop_event.set()
        driver._drive(stop_event)

        assert driver.records == []
        assert armed == []

    def test_each_relaunch_arms_the_freeze_the_next_take_over_waits_for(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] With progress for both scheduled freezes, the loop lands two take-overs.
        # Records are index 0 (saved 1, frozen 2) and index 1 (saved 3, frozen 4). The relaunch
        # received the next freeze's rollout 4 first and none (no more freezes) the second time,
        # and the driver reports all restarts happened.
        """A relaunch repeating the old step would freeze the run again where it was already taken over."""
        armed: list[int | None] = []
        driver = _driver(tmp_path, relaunch=armed.append)
        _install_progress(
            monkeypatch,
            [
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=2),
                RunProgress(last_saved_iteration=3, last_finished_rollout_id=4),
            ],
        )

        _drive_until_take_overs_landed(driver, take_overs=2)
        _join_relaunches(driver)

        assert driver.records == [
            HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2),
            HotRestartRecord(index=1, saved_iteration_at_trigger=3, frozen_rollout_id=4),
        ]
        assert armed == [4, None]
        driver.assert_all_restarts_happened()

    def test_the_loop_ends_once_the_last_take_over_was_triggered(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] With a one-freeze schedule and a later progress reading far past the
        # freeze (5, 5), the loop does not fail for training past the last freeze: exactly one
        # record and one relaunch (with no further freeze).
        """A run read again after its final take-over would fail for training past its last freeze."""
        armed: list[int | None] = []
        driver = _driver(
            tmp_path, relaunch=armed.append, schedule=(ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),)
        )
        _install_progress(
            monkeypatch,
            [
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=2),
                RunProgress(last_saved_iteration=5, last_finished_rollout_id=5),
            ],
        )

        _drive_until_take_overs_landed(driver, take_overs=1)
        _join_relaunches(driver)

        assert armed == [None]
        assert len(driver.records) == 1

    def test_a_run_that_never_reaches_the_pinned_step_runs_out_of_time(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] With a zero freeze timeout and progress never reaching the step, the loop
        # returns and the final check raises an assertion error matching "waited 0.0s".
        """Waiting forever would leave the suite hanging on a run whose freeze never fired."""
        driver = _driver(tmp_path, freeze_timeout_seconds=0.0)
        _install_progress(monkeypatch, [RunProgress(last_saved_iteration=1, last_finished_rollout_id=1)])

        driver._drive(threading.Event())

        with pytest.raises(AssertionError, match="waited 0.0s"):
            driver.assert_all_restarts_happened()

    def test_a_read_that_failed_is_retried_rather_than_ending_the_run(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] The first progress read raises an OS error; later reads report the run
        # frozen at the pinned step (the sentinel appears once more than one read happened). The
        # driver retries and still lands exactly one take-over record.
        """A dump directory on shared storage answers late now and then, and that is not a verdict."""
        reads: list[int] = []

        def read(**_kwargs):
            # [FUNCTION DOC] HELPER: Counts the call; raises an OS error on the first call, then
            # returns progress with saved iteration 1 and finished rollout 2.
            reads.append(1)
            if len(reads) == 1:
                raise OSError("shared storage said no")
            return RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)

        driver = _driver(tmp_path, schedule=(ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),))
        monkeypatch.setattr(driver_module, "read_run_progress", read)
        monkeypatch.setattr(driver_module.ClusterObserver, "observe_once_or_warn", lambda _self: None)
        monkeypatch.setattr(driver_module, "read_frozen_rollout_id", lambda _path: 2 if len(reads) > 1 else None)

        _drive_until_take_overs_landed(driver, take_overs=1)
        _join_relaunches(driver)

        assert len(driver.records) == 1


class TestHotRestartDriverVerdict:
    # [CLASS DOC] Tests of the driver's final verdict.
    def test_a_run_that_ended_before_every_take_over_landed_fails(self, tmp_path):
        # [FUNCTION DOC] With 1 record of 2 scheduled, the final check raises an assertion error
        # matching "1 of 2 hot restart".
        """A run replaced once when the scenario pinned two proves half of what it claims."""
        driver = _driver(tmp_path)
        driver.records.append(HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2))

        with pytest.raises(AssertionError, match="1 of 2 hot restart"):
            driver.assert_all_restarts_happened()

    def test_a_relaunch_that_raised_is_reported_rather_than_swallowed(self, tmp_path):
        # [FUNCTION DOC] A relaunch raising a runtime error is captured by the private relaunch
        # wrapper; the final check raises matching "the hot restart driver failed".
        """A take-over nothing installed leaves a run the comparison would read as untouched."""
        driver = _driver(tmp_path, relaunch=_raise_boom)

        driver._relaunch(4)

        with pytest.raises(AssertionError, match="the hot restart driver failed"):
            driver.assert_all_restarts_happened()

    def test_an_intermediate_launch_replaced_by_the_next_take_over_is_not_a_failure(self, tmp_path):
        # [FUNCTION DOC] With two records present, a relaunch (armed for rollout 4) that exits as
        # if killed by a termination signal is not a failure: the final check passes.
        """SIGTERM is the expected local verdict of a launch whose orchestrator the schedule replaces."""
        driver = _driver(tmp_path, relaunch=_raise_sigterm)
        driver.records.extend(
            [
                HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2),
                HotRestartRecord(index=1, saved_iteration_at_trigger=3, frozen_rollout_id=4),
            ]
        )

        driver._relaunch(4)

        driver.assert_all_restarts_happened()

    def test_a_final_launch_killed_by_sigterm_is_still_a_failure(self, tmp_path):
        # [FUNCTION DOC] The final relaunch (no next freeze) exiting by termination signal is a
        # failure: the final check raises matching "the hot restart driver failed".
        """Nothing replaces the final orchestrator, so its SIGTERM remains the target run's real verdict."""
        driver = _driver(tmp_path, relaunch=_raise_sigterm)

        driver._relaunch(None)

        with pytest.raises(AssertionError, match="the hot restart driver failed"):
            driver.assert_all_restarts_happened()


def _raise_boom(_frozen_rollout_id: int | None) -> None:
    # [FUNCTION DOC] HELPER: Fake relaunch that raises a runtime error "boom".
    raise RuntimeError("boom")


def _raise_sigterm(_frozen_rollout_id: int | None) -> None:
    # [FUNCTION DOC] HELPER: Fake relaunch that raises the run-exited error with exit code 128
    # plus the termination signal number (143).
    raise RunExitedError(128 + signal.SIGTERM)


class TestDrivingHotRestarts:
    # [CLASS DOC] Tests of the context manager that drives take-overs around a run body.
    def test_sigterm_before_any_take_over_is_still_the_run_verdict(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] A run-exited error with the termination exit code raised in the body
        # before any record exists propagates unchanged with its exit code.
        """No scheduled replacement explains SIGTERM before the driver has triggered its first take-over."""
        driver = _driver(tmp_path)
        monkeypatch.setattr(driver_module.ClusterObserver, "observe_once_or_warn", lambda _self: None)

        with pytest.raises(RunExitedError) as error:
            with driving_hot_restarts(driver, dump_dir=str(tmp_path)):
                raise RunExitedError(128 + signal.SIGTERM)

        assert error.value.exit_code == 128 + signal.SIGTERM

    def test_a_system_exit_that_is_not_a_run_verdict_is_never_a_replaced_launch(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] With a record present, a plain system exit (not the run-exited error)
        # with the same code raised in the body propagates as a system exit and is not converted.
        """Only a followed run's own exit code says a launch was replaced; any other SystemExit ends the side."""
        driver = _driver(tmp_path)
        monkeypatch.setattr(driver_module.ClusterObserver, "observe_once_or_warn", lambda _self: None)
        driver.records.append(HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2))

        with pytest.raises(SystemExit) as error:
            with driving_hot_restarts(driver, dump_dir=str(tmp_path)):
                raise SystemExit(128 + signal.SIGTERM)

        assert not isinstance(error.value, RunExitedError)

    def test_the_previous_launch_exit_does_not_stop_the_remaining_take_overs(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Inside the context, waits (up to 30 s) for the first take-over record,
        # then raises the termination run-exited error from the body (the replaced launch). The
        # context swallows it as expected: after joining relaunches, both freezes were armed
        # ([4, none]) and the final check passes.
        """A launch-level SIGTERM must not end a target side whose schedule still has a second take-over."""
        armed: list[int | None] = []
        driver = _driver(tmp_path, relaunch=armed.append)
        _install_progress(
            monkeypatch,
            [
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=2),
                RunProgress(last_saved_iteration=3, last_finished_rollout_id=4),
            ],
        )

        with driving_hot_restarts(driver, dump_dir=str(tmp_path)):
            deadline = time.monotonic() + _DRIVE_TIMEOUT_SECONDS
            while not driver.records and time.monotonic() < deadline:
                time.sleep(_DRIVE_POLL_SECONDS)
            raise RunExitedError(128 + signal.SIGTERM)

        _join_relaunches(driver)

        assert armed == [4, None]
        driver.assert_all_restarts_happened()


class TestWatchingPastTheLastTakeOver:
    # [CLASS DOC] Test that cluster sampling continues after the schedule is spent.
    def test_the_run_is_still_sampled_after_the_last_take_over_was_triggered(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] One-freeze schedule with zero poll interval. The cluster observation is
        # replaced by a recorder of the current record count that sets the stop event after 6
        # samples. After driving, exactly one record exists and the observation ran at least twice
        # with one record (still sampled after the last take-over).
        """The last relaunch is when its pods are replaced, and nothing else would ever look at them."""
        stop_event = threading.Event()
        driver = _driver(
            tmp_path, schedule=(ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),), poll_interval_seconds=0.0
        )
        _install_progress(monkeypatch, [RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)])

        observed_after: list[int] = []

        def observe(_self) -> None:
            # [FUNCTION DOC] HELPER: Fake observation; records the current number of take-over
            # records and sets the stop event once six observations have been made.
            observed_after.append(len(driver.records))
            if len(observed_after) >= 6:
                stop_event.set()

        monkeypatch.setattr(driver_module.ClusterObserver, "observe_once_or_warn", observe)

        driver._drive(stop_event)
        _join_relaunches(driver)

        assert len(driver.records) == 1
        assert observed_after.count(1) >= 2, "the run has to keep being sampled once the schedule is spent"


class TestWhenTheDriverCannotReadTheRun:
    # [CLASS DOC] Tests of read failure handling in the drive loop.
    def test_a_read_that_never_answers_still_runs_out_of_time(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] With every progress read raising and a zero freeze timeout, the loop
        # still ends and the final check raises matching "waited 0.0s" (the deadline is checked on
        # failed reads too).
        """The deadline used to be checked only on a successful read, so this hung to the ci limit."""
        driver = _driver(tmp_path, freeze_timeout_seconds=0.0)
        monkeypatch.setattr(driver_module, "read_run_progress", _refuse_to_read)
        monkeypatch.setattr(driver_module.ClusterObserver, "observe_once_or_warn", lambda _self: None)

        driver._drive(threading.Event())

        with pytest.raises(AssertionError, match="waited 0.0s"):
            driver.assert_all_restarts_happened()

    def test_a_read_that_keeps_failing_is_reported_rather_than_retried_forever(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] With a huge freeze timeout and a read failure limit of 3, always failing
        # reads end the loop and the final check raises matching "3 time(s) in a row".
        """A dump directory nobody can read is not a run that has not frozen yet."""
        driver = _driver(tmp_path, freeze_timeout_seconds=1_000_000.0, read_failure_limit=3)
        monkeypatch.setattr(driver_module, "read_run_progress", _refuse_to_read)
        monkeypatch.setattr(driver_module.ClusterObserver, "observe_once_or_warn", lambda _self: None)

        driver._drive(threading.Event())

        with pytest.raises(AssertionError, match="3 time\\(s\\) in a row"):
            driver.assert_all_restarts_happened()


def _refuse_to_read(**_kwargs):
    # [FUNCTION DOC] HELPER: Fake progress reader that ignores its arguments and raises an OS
    # error saying the dump directory did not answer.
    raise OSError("the dump directory did not answer")


class TestWhenTheCheckpointTrackerLagsTheFreeze:
    # [CLASS DOC] Tests of tolerance for a checkpoint tracker that has not caught up when the
    # run freezes (the driver re-reads the saved iteration).
    def test_a_tracker_that_catches_up_within_a_few_reads_is_accepted(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Progress reports no saved iteration but the re-read returns 1; the record
        # computed for the checkpointed freeze has saved iteration 1.
        """The tracker is written by the save itself, a moment after the step it belongs to ends."""
        driver = _driver(tmp_path)
        monkeypatch.setattr(driver_module.HotRestartDriver, "_reread_saved_iteration", lambda _self: 1)

        record = driver._compute_record(
            index=0,
            scheduled=_CHECKPOINTED,
            progress=RunProgress(last_saved_iteration=None, last_finished_rollout_id=2),
        )

        assert record.saved_iteration_at_trigger == 1

    def test_a_reread_that_answered_nothing_does_not_erase_what_the_freeze_reported(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Progress reports saved iteration 0 against the from-scratch freeze and
        # the re-read returns nothing; the stale reported value stays and the record computation
        # raises matching "save cadence".
        """A tracker nobody could read is no evidence, and reading it as none matches a scenario pinning none."""
        driver = _driver(tmp_path)
        monkeypatch.setattr(driver_module.HotRestartDriver, "_reread_saved_iteration", lambda _self: None)

        with pytest.raises(AssertionError, match="save cadence"):
            driver._compute_record(
                index=0,
                scheduled=_FROM_SCRATCH,
                progress=RunProgress(last_saved_iteration=0, last_finished_rollout_id=1),
            )

    def test_a_tracker_that_never_catches_up_still_fails(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Both the reported progress and the re-read give saved iteration 5 against
        # a pinned 1: raises matching "save cadence".
        """Re-reading is for a save in flight, not for a run saving at another pace altogether."""
        driver = _driver(tmp_path)
        monkeypatch.setattr(driver_module.HotRestartDriver, "_reread_saved_iteration", lambda _self: 5)

        with pytest.raises(AssertionError, match="save cadence"):
            driver._compute_record(
                index=0,
                scheduled=_CHECKPOINTED,
                progress=RunProgress(last_saved_iteration=5, last_finished_rollout_id=2),
            )
