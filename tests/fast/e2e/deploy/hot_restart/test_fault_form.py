# [MODULE DOC]
# Tests for the hot-restart fault form used by the fault-injection soak: which workloads a
# take-over restamps and what counts as a landed take-over (every replaced workload's stamp value
# changed, from a successful read), the injection window rule (eligible while finished rollouts
# are below the configured maximum), injecting a draw (relaunching the already-installed release
# with the hot restart argument, recording the step and saved iteration where it fired, waiting
# for restamped workloads, timing out, baseline read retries, non-blocking relaunch, reporting
# relaunches that never landed or were refused), the form's name and harmless-to-cell flag, and
# the closing contract (joining relaunch threads, reporting failures, still-running relaunches and
# replaced-launch exits depending on whether a later take-over started or landed). Cluster stamp
# reads, run progress and the relaunch callable are replaced by fakes.
import threading
from pathlib import Path
from random import Random
from typing import Any

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart import fault_form as fault_form_module
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import compute_hot_restart_workloads
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import (
    HOT_RESTART_ARG,
    REPLACED_LAUNCH_EXIT_CODE,
    compute_release_of_config,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord, RunProgress
from tests.e2e.deploy.conftest_deploy.hot_restart.fault_form import (
    HOT_RESTART_FORM_NAME,
    HotRestartFaultForm,
    restamped_replaced_workloads,
)

from miles.ray.specs.rollout import ROLLOUT_EXECUTOR_POOL_ID
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.launcher.entrypoint import RunExitedError
from miles.utils.external_utils.command_utils.helm_backend.naming import ORCHESTRATOR_COMPONENT
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name

CELL: dict = {"metadata": {"name": "actor-0"}}
CONFIG: ExecuteTrainConfig = ExecuteTrainConfig(run_id="demo", namespace="rl")
RELEASE: str = compute_release_of_config(CONFIG)
STAMPED: frozenset[str] = compute_hot_restart_workloads(RELEASE)
ORCHESTRATOR: str = component_name(RELEASE, ORCHESTRATOR_COMPONENT)
ROLLOUT_EXECUTOR: str = component_name(RELEASE, ROLLOUT_EXECUTOR_POOL_ID)
TRAINER: str = "a-workload-a-take-over-leaves-alone"


def _form(launch, **overrides: Any) -> HotRestartFaultForm:
    # [FUNCTION DOC] HELPER: Builds the fault form around the given launch callable with the demo
    # configuration, checkpoint and events directories under /dumps, a maximum allowed rollout id
    # of 234, zero poll interval and a 5 second timeout; keyword overrides replace any of these.
    kwargs: dict[str, Any] = dict(
        launch=launch,
        config=CONFIG,
        checkpoint_dir=Path("/dumps/checkpoints"),
        events_dir=Path("/dumps/events"),
        max_allowed_rollout_id=234,
        poll_interval_seconds=0.0,
        timeout_seconds=5.0,
    )
    kwargs.update(overrides)
    return HotRestartFaultForm(**kwargs)


def _install_run(
    monkeypatch,
    *,
    finished: int | None = 1,
    saved: int | None = 1,
    stamps: list[dict[str, str | None] | None] | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Scripts the run for the fault form. The restart stamp reader returns
    # the given stamp mappings one per call (default: stamps "t1" then "t2"), repeating the last
    # once only one remains; a none entry simulates a failed read. The run progress reader always
    # reports the given saved iteration (default 1) and finished rollout (default 1; none allowed).
    reads = list(stamps) if stamps is not None else [_stamps("t1"), _stamps("t2")]
    monkeypatch.setattr(
        fault_form_module,
        "read_restart_stamp_of_workload",
        lambda **_kwargs: reads.pop(0) if len(reads) > 1 else reads[0],
    )
    monkeypatch.setattr(
        fault_form_module,
        "read_run_progress",
        lambda **_kwargs: RunProgress(last_saved_iteration=saved, last_finished_rollout_id=finished),
    )


class TestWhatCountsAsATakeOverThatLanded:
    # [CLASS DOC] Tests of the restamp predicate: true only if every watched workload has a
    # changed stamp value between a successful before read and a successful after read.
    def test_a_take_over_replaces_the_orchestrator_and_the_rollout_executor_of_the_release(self):
        # [FUNCTION DOC] The watched workload set is exactly the orchestrator and rollout executor
        # names of the release.
        """These two names are what every stamp assertion below is written against."""
        assert STAMPED == {ORCHESTRATOR, ROLLOUT_EXECUTOR}

    def test_a_run_whose_replaced_workloads_were_all_restamped_counts(self):
        # [FUNCTION DOC] Stamps going from none to "t2" on both workloads count as landed.
        """A take-over rewrites the stamp of each workload it replaces, whatever the run had trained."""
        assert restamped_replaced_workloads(
            before=_stamps(orchestrator=None, rollout_executor=None), after=_stamps(), workloads=STAMPED
        )

    def test_a_second_take_over_rewriting_the_first_ones_stamps_counts(self):
        # [FUNCTION DOC] Stamps going from "t1" to "t2" on both workloads count as landed.
        """One object carries one stamp, rewritten each time, so a landing is a value that changed."""
        assert restamped_replaced_workloads(
            before=_stamps(orchestrator="t1", rollout_executor="t1"), after=_stamps(), workloads=STAMPED
        )

    def test_a_run_whose_workloads_still_carry_the_stamps_they_carried_does_not_count(self):
        # [FUNCTION DOC] Identical before and after stamps do not count.
        """The relaunch is still installing, and the stamps it will rewrite are the ones drawn against."""
        assert not restamped_replaced_workloads(before=_stamps(), after=_stamps(), workloads=STAMPED)

    def test_a_run_only_half_of_whose_workloads_were_restamped_does_not_count(self):
        # [FUNCTION DOC] Only the orchestrator restamped (rollout executor still none) does not
        # count.
        """The two workloads are rolled by one upgrade but observed apart, so one of them is a half-landing."""
        assert not restamped_replaced_workloads(
            before=_stamps(orchestrator=None, rollout_executor=None),
            after=_stamps(rollout_executor=None),
            workloads=STAMPED,
        )

    def test_a_workload_the_read_did_not_return_does_not_count(self):
        # [FUNCTION DOC] An empty before read and an after read with only one workload do not
        # count.
        """A workload absent from the read has not been seen carrying anything, which is not evidence."""
        assert not restamped_replaced_workloads(before={}, after={ORCHESTRATOR: "t2"}, workloads=STAMPED)

    def test_a_stamp_the_run_never_carried_on_another_workload_does_not_count(self):
        # [FUNCTION DOC] A stamp appearing only on an unrelated workload (not watched) while the
        # watched ones stay unstamped does not count.
        """Only the two workloads a take-over replaces are stamped; anything else says nothing about it."""
        assert not restamped_replaced_workloads(
            before=_stamps(orchestrator=None, rollout_executor=None),
            after={**_stamps(orchestrator=None, rollout_executor=None), TRAINER: "t2"},
            workloads=STAMPED,
        )

    def test_a_stamp_read_that_failed_does_not_count(self):
        # [FUNCTION DOC] An after read of none (failed read) does not count.
        """A kubectl call that did not answer must not become a take-over that never landed."""
        assert not restamped_replaced_workloads(before=_stamps(), after=None, workloads=STAMPED)


class TestInject:
    # [CLASS DOC] Tests of the form's injection window and inject action.
    def test_a_draw_before_the_closing_window_is_eligible(self, monkeypatch):
        # [FUNCTION DOC] With 233 finished rollouts (below the maximum 234) the form is within the
        # injection window.
        """The scheduler keeps seeing this form while the run remains before its closing window."""
        _install_run(monkeypatch, finished=233)

        assert _form(lambda _config: None).is_within_injection_window()

    def test_a_draw_at_the_closing_window_is_ineligible(self, monkeypatch):
        # [FUNCTION DOC] With 234 finished rollouts (the maximum) the form is outside the window.
        """Completing the configured boundary prevents the scheduler from starting another take-over."""
        _install_run(monkeypatch, finished=234)

        assert not _form(lambda _config: None).is_within_injection_window()

    def test_every_draw_relaunches_the_release_that_is_already_up(self, monkeypatch):
        # [FUNCTION DOC] One injection launches exactly one configuration with the hot restart
        # argument set and the original run id "demo".
        """A relaunch under another release would leave the trainers this run is watching behind."""
        launched: list[ExecuteTrainConfig] = []
        _install_run(monkeypatch)

        _form(launched.append).inject(CELL, Random(0))

        assert [one.hot_restart for one in launched] == [HOT_RESTART_ARG]
        assert [one.run_id for one in launched] == ["demo"]

    def test_a_draw_before_the_first_save_fires_like_any_other(self, monkeypatch):
        # [FUNCTION DOC] With no saved iteration the injection still launches exactly once.
        """Taking a run over before it saved costs everything it trained, which is a path worth covering."""
        launched: list[ExecuteTrainConfig] = []
        _install_run(monkeypatch, saved=None)

        _form(launched.append).inject(CELL, Random(0))

        assert len(launched) == 1

    def test_a_draw_before_the_run_trained_a_step_lands_on_the_restart_stamps(self, monkeypatch):
        # [FUNCTION DOC] With no finished rollout and no save, and stamps going from none to a new
        # value, injecting records one take-over whose frozen rollout id is -1.
        """A run holding nothing yet is a path this soak covers, and its take-over lands like any other."""
        _install_run(
            monkeypatch,
            finished=None,
            saved=None,
            stamps=[_stamps(orchestrator=None, rollout_executor=None), _stamps()],
        )

        form = _form(lambda _config: None)
        form.inject(CELL, Random(0))

        assert [one.frozen_rollout_id for one in form.records] == [-1]

    def test_the_step_a_record_carries_is_the_one_the_run_stood_at_when_it_was_drawn(self, monkeypatch):
        # [FUNCTION DOC] With finished rollout 4 and saved iteration 3, the single record carries
        # frozen rollout 4 and saved iteration 3.
        """The record is what the artifact reports a take-over cost, read where the draw fired."""
        _install_run(monkeypatch, finished=4, saved=3)

        form = _form(lambda _config: None)
        form.inject(CELL, Random(0))

        assert [(one.frozen_rollout_id, one.saved_iteration_at_trigger) for one in form.records] == [(4, 3)]

    def test_a_second_draw_before_the_run_trained_a_step_lands_on_the_rewritten_stamps(self, monkeypatch):
        # [FUNCTION DOC] Stamps going from "t2" (already stamped) to "t3" count as landed: one
        # record is made.
        """One object carries one stamp: a second take-over adds none, and only the rewrite says it landed."""
        _install_run(monkeypatch, finished=None, saved=None, stamps=[_stamps(), _stamps("t3")])

        form = _form(lambda _config: None)
        form.inject(CELL, Random(0))

        assert len(form.records) == 1

    def test_a_draw_before_the_run_trained_a_step_still_times_out_without_a_restamp(self, monkeypatch):
        # [FUNCTION DOC] Stamps never change and the launch blocks on an event (up to 30 s); with a
        # 0.2 second timeout, injecting raises an assertion error matching "still carry the stamps
        # they carried". The finally block releases the blocked launch.
        """Without a rewritten stamp there is nothing saying the take-over installed, and that must not pass."""
        never_returns = threading.Event()
        _install_run(monkeypatch, finished=None, saved=None, stamps=[_stamps()])

        try:
            with pytest.raises(AssertionError, match="still carry the stamps they carried"):
                _form(lambda _config: never_returns.wait(timeout=30.0), timeout_seconds=0.2).inject(CELL, Random(0))
        finally:
            never_returns.set()

    def test_a_draw_that_cannot_read_the_workloads_it_would_compare_against_is_reported(self, monkeypatch):
        # [FUNCTION DOC] Every baseline stamp read fails (none) and baseline read attempts are
        # limited to 2: injecting raises an assertion error matching "has nothing to compare
        # against".
        """Without a baseline nothing could tell a rewritten stamp from the one that was already there."""
        _install_run(monkeypatch, finished=None, saved=None, stamps=[None])

        with pytest.raises(AssertionError, match="has nothing to compare against"):
            _form(lambda _config: None, baseline_read_attempts=2).inject(CELL, Random(0))

    def test_the_injector_is_not_blocked_by_a_relaunch_that_drives_the_run_to_its_end(self, monkeypatch):
        # [FUNCTION DOC] A launch that blocks (up to 30 s) on an event does not block injection:
        # inject returns, and the event is then set to release the background launch thread.
        """The relaunch installs a script that trains to the end, so a call that waits for it never returns."""
        driving = threading.Event()
        _install_run(monkeypatch)

        _form(lambda _config: driving.wait(timeout=30.0)).inject(CELL, Random(0))

        driving.set()

    def test_a_relaunch_that_never_reached_the_run_is_reported_rather_than_counted(self, monkeypatch):
        # [FUNCTION DOC] A launch that returns immediately while stamps never change makes inject
        # raise an assertion error matching "returned without restamping".
        """An injection counted as landed would let this soak pass on a run nothing ever replaced."""
        _install_run(monkeypatch, stamps=[_stamps()])

        with pytest.raises(AssertionError, match="returned without restamping"):
            _form(lambda _config: None).inject(CELL, Random(0))

    def test_a_relaunch_the_cluster_refused_is_reported_rather_than_counted(self, monkeypatch):
        # [FUNCTION DOC] A launch that exits (system exit) because the upgrade would change too
        # much makes inject raise matching "refused rather than installed".
        """A refused upgrade leaves the run training under the very script this injection meant to replace."""
        _install_run(monkeypatch, stamps=[_stamps()])

        with pytest.raises(AssertionError, match="refused rather than installed"):
            _form(_raise_refused).inject(CELL, Random(0))

    def test_a_later_draw_is_judged_on_its_own_relaunch_and_not_on_an_earlier_one(self, monkeypatch):
        # [FUNCTION DOC] After a refused first injection (raises), the stamps and launch are
        # replaced by working ones (the form's private launch attribute is overwritten) and a
        # second injection succeeds without raising.
        """A refused upgrade may well be transient, and a soak that gives up on the first one proves less."""
        _install_run(monkeypatch, stamps=[_stamps()])
        form = _form(_raise_refused)
        with pytest.raises(AssertionError, match="refused rather than installed"):
            form.inject(CELL, Random(0))

        _install_run(monkeypatch)
        form._launch = lambda _config: None

        form.inject(CELL, Random(0))


def _raise_refused(_config: ExecuteTrainConfig) -> None:
    # [FUNCTION DOC] HELPER: Fake launch that raises a system exit with a message that the
    # relaunch would change more than the size of the run (a refused upgrade).
    raise SystemExit("the relaunch would change more than the size of this run")


class TestForm:
    # [CLASS DOC] Tests of the form's static properties.
    def test_the_form_is_named_after_what_it_does(self):
        # [FUNCTION DOC] The form's name equals the hot restart form name constant.
        """The name is what the injection log carries, and a form nobody can name cannot be read back."""
        assert _form(lambda _config: None).name == HOT_RESTART_FORM_NAME

    def test_the_form_declares_that_it_leaves_the_cell_it_was_drawn_for_running(self):
        # [FUNCTION DOC] The form does not declare that it harms its cell.
        """A cell counted as crashed is dropped from the live set forever, and no later draw would fire."""
        assert not _form(lambda _config: None).harms_cell


class TestTheClosingContract:
    # [CLASS DOC] Tests of the end-of-soak check that all take-overs installed cleanly.
    def test_a_soak_whose_take_overs_all_installed_cleanly_passes(self, monkeypatch):
        # [FUNCTION DOC] After one successful injection, joining relaunch threads (30 s) and the
        # clean-install check pass without raising.
        """Nothing raised and nothing is still running, so what was collected can be read."""
        _install_run(monkeypatch)
        form = _form(lambda _config: None)
        form.inject(CELL, Random(0))

        form.join_relaunches(timeout_seconds=30.0)
        form.assert_take_overs_installed_cleanly()

    def test_the_run_verdict_raised_by_the_last_relaunch_is_not_lost(self, monkeypatch):
        # [FUNCTION DOC] A launch that exits with a run verdict message (metric below the required
        # value) after landing is captured; the closing check raises matching "did not install
        # cleanly".
        """The last relaunch's launcher is what observes the run's own metric checker."""
        _install_run(monkeypatch)
        form = _form(_raise_run_verdict)
        form.inject(CELL, Random(0))

        form.join_relaunches(timeout_seconds=30.0)

        with pytest.raises(AssertionError, match="did not install cleanly"):
            form.assert_take_overs_installed_cleanly()

    def test_a_relaunch_still_running_at_the_end_is_reported(self, monkeypatch):
        # [FUNCTION DOC] A launch blocked on an event outlives a 0.05 second join; the closing
        # check raises matching "still installing a hot restart". The event is released in a
        # finally block.
        """A run still being replaced under the dumps about to be read is not a finished soak."""
        never_returns = threading.Event()
        _install_run(monkeypatch)
        form = _form(lambda _config: never_returns.wait(timeout=30.0))
        form.inject(CELL, Random(0))

        try:
            form.join_relaunches(timeout_seconds=0.05)
            with pytest.raises(AssertionError, match="still installing a hot restart"):
                form.assert_take_overs_installed_cleanly()
        finally:
            never_returns.set()

    def test_a_failure_from_an_earlier_take_over_is_still_reported_at_the_end(self, monkeypatch):
        # [FUNCTION DOC] A refused first injection raises; a successful second injection follows.
        # The closing check still raises, mentioning "take-over 0".
        """Per-draw judgement must not throw away what an earlier draw already proved broken."""
        _install_run(monkeypatch, stamps=[_stamps()])
        form = _form(_raise_refused)
        with pytest.raises(AssertionError, match="refused rather than installed"):
            form.inject(CELL, Random(0))

        _install_run(monkeypatch)
        form._launch = lambda _config: None
        form.inject(CELL, Random(0))
        form.join_relaunches(timeout_seconds=30.0)

        with pytest.raises(AssertionError, match="take-over 0"):
            form.assert_take_overs_installed_cleanly()

    def test_a_replaced_launch_exit_before_the_next_take_over_starts_is_reported_even_if_one_lands_later(self):
        # [FUNCTION DOC] The private relaunch for take-over 0 captures a replaced-launch exit
        # before any later take-over thread started; a record for take-over 1 added afterwards
        # does not excuse it: the check raises matching "take-over 0".
        """A later take-over cannot explain an exit captured before that replacement started."""
        form = _form(_raise_replaced_launch_exit)
        form._relaunch(0)
        form._records.append(_record(1))

        with pytest.raises(AssertionError, match="take-over 0"):
            form.assert_take_overs_installed_cleanly()

    def test_a_replaced_launch_exit_after_the_next_take_over_starts_is_accepted_after_it_lands(self):
        # [FUNCTION DOC] With two take-over threads already started, the replaced-launch exit of
        # take-over 0 is captured and, once a record for take-over 1 lands, the check passes.
        # KNOWN ISSUE: the threads are never started, they are only placeholders in the list.
        """A started and later landed take-over explains a replacement exit captured between those events."""
        form = _form(_raise_replaced_launch_exit)
        form._threads = _started_take_over_threads(2)
        form._relaunch(0)

        form._records.append(_record(1))

        form.assert_take_overs_installed_cleanly()

    def test_a_replaced_launch_exit_after_the_next_take_over_lands_is_accepted(self):
        # [FUNCTION DOC] With two threads started and the record for take-over 1 already landed,
        # capturing the replaced-launch exit of take-over 0 is accepted by the check.
        """A later landed take-over explains an earlier launch's replacement exit when landing wins the race."""
        form = _form(_raise_replaced_launch_exit)
        form._threads = _started_take_over_threads(2)
        form._records.append(_record(1))

        form._relaunch(0)

        form.assert_take_overs_installed_cleanly()

    def test_a_replaced_launch_exit_without_a_later_take_over_is_reported(self):
        # [FUNCTION DOC] Only a record for take-over 0 exists; its replaced-launch exit has no
        # later take-over to explain it: the check raises matching "take-over 0".
        """Exit 143 without a later landed take-over is an unexplained launcher failure."""
        form = _form(_raise_replaced_launch_exit)
        form._records.append(_record(0))

        form._relaunch(0)

        with pytest.raises(AssertionError, match="take-over 0"):
            form.assert_take_overs_installed_cleanly()

    def test_a_replaced_launch_exit_without_the_exact_next_landing_is_reported(self):
        # [FUNCTION DOC] Two threads started but the landed record has index 2 instead of the
        # exact next index 1: the exit of take-over 0 is not explained and the check raises
        # matching "take-over 0".
        """A started replacement needs its exact landing record to explain exit 143."""
        form = _form(_raise_replaced_launch_exit)
        form._threads = _started_take_over_threads(2)
        form._records.append(_record(2))

        form._relaunch(0)

        with pytest.raises(AssertionError, match="take-over 0"):
            form.assert_take_overs_installed_cleanly()

    def test_another_exit_is_reported_despite_a_later_take_over(self):
        # [FUNCTION DOC] A launch exiting with a code one below the replaced-launch code is a real
        # failure even if take-over 1 landed: the check raises matching "take-over 0".
        """A later take-over only explains the replacement exit code, not an unrelated launcher failure."""
        form = _form(_raise_another_launch_exit)
        form._records.append(_record(1))

        form._relaunch(0)

        with pytest.raises(AssertionError, match="take-over 0"):
            form.assert_take_overs_installed_cleanly()


def _raise_run_verdict(_config: ExecuteTrainConfig) -> None:
    # [FUNCTION DOC] HELPER: Fake launch that raises a system exit carrying a metric-check failure
    # message (an evaluation metric below its required value).
    raise SystemExit("eval/gsm8k 0.31 is below the required 0.55")


def _raise_replaced_launch_exit(_config: ExecuteTrainConfig) -> None:
    # [FUNCTION DOC] HELPER: Fake launch raising the run-exited error with the replaced-launch
    # exit code.
    raise RunExitedError(REPLACED_LAUNCH_EXIT_CODE)


def _raise_another_launch_exit(_config: ExecuteTrainConfig) -> None:
    # [FUNCTION DOC] HELPER: Fake launch raising the run-exited error with a code one below the
    # replaced-launch code.
    raise RunExitedError(REPLACED_LAUNCH_EXIT_CODE - 1)


def _started_take_over_threads(count: int) -> list[threading.Thread]:
    # [FUNCTION DOC] HELPER: Returns the given number of unstarted no-op threads named
    # "take-over-<index>", used as stand-ins for take-over threads.
    return [threading.Thread(target=lambda: None, name=f"take-over-{index}") for index in range(count)]


def _record(index: int) -> HotRestartRecord:
    # [FUNCTION DOC] HELPER: Builds a record with the given index, saved iteration 1 and frozen
    # rollout 1.
    return HotRestartRecord(index=index, saved_iteration_at_trigger=1, frozen_rollout_id=1)


def _stamps(
    at: str = "t2", *, orchestrator: str | None = "", rollout_executor: str | None = ""
) -> dict[str, str | None]:
    # [FUNCTION DOC] HELPER: Builds a mapping from the orchestrator and rollout executor names to
    # stamp values. Each workload uses the default stamp "t2" (or the positional value) unless its
    # keyword is given; an empty string keyword means use the default, while none means unstamped.
    return {
        ORCHESTRATOR: at if orchestrator == "" else orchestrator,
        ROLLOUT_EXECUTOR: at if rollout_executor == "" else rollout_executor,
    }
