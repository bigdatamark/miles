# [MODULE DOC]
# Tests for the realistic gsm8k hot-restart scenario (a long soak with randomly drawn take-overs):
# it reuses the fault-tolerance gsm8k recipe and bounds rather than copying them, adds only the
# arguments take-overs require (checkpointing, wandb run id, weight update checker disabled) and
# keeps the shared recipe free of fault-tolerance components; the injection plan (virtual cells
# available only before the closing window, a single hot restart fault form wired to the run's
# launch, config, checkpoint and events directories); consistency of constants with the
# deployment readme; checkpoint arguments; the requirement that no attempt failed and at least
# one restart happened; per-take-over loss bounds from the recorded progress; and the log-based
# check that each take-over resumed within a save interval, using event logs written with the
# real event logger and a helper that moves a log aside as a take-over does.
import shlex
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.e2e.deploy.conftest_deploy.hot_restart import scenario_hot_restart_realistic_gsm8k as scenario
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.fault_form import HotRestartFaultForm
from tests.e2e.ft.conftest_ft import scenario_realistic_gsm8k
from tests.e2e.ft.conftest_ft.fault_injection.state import InjectionEvent

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, EventLogger
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import ArgvManipulator


def _run(dump_dir: str) -> scenario_realistic_gsm8k.Gsm8kRun:
    # [FUNCTION DOC] HELPER: Builds a gsm8k run description with a fixed orchestrator url, a
    # configuration with run id "demo" and namespace "rl", the given dump directory, empty
    # training arguments and a no-op launch callable.
    return scenario_realistic_gsm8k.Gsm8kRun(
        base_url="http://orchestrator:18080",
        config=ExecuteTrainConfig(run_id="demo", namespace="rl"),
        dump_dir=dump_dir,
        train_args="",
        launch=lambda config: None,
    )


class TestTheRecipeIsTheOneFtConverges:
    # [CLASS DOC] Tests that the scenario reuses the fault-tolerance gsm8k recipe.
    def test_the_run_is_the_realistic_gsm8k_run_and_not_a_copy_of_it(self):
        # [FUNCTION DOC] The scenario's run function is the same object as the shared recipe's.
        """A second recipe would drift from the one whose reward bounds this test inherits."""
        assert scenario.run_realistic_gsm8k is scenario_realistic_gsm8k.run_realistic_gsm8k

    def test_the_bounds_the_run_is_graded_against_are_the_ones_ft_declares(self):
        # [FUNCTION DOC] The scenario's default metric threshold and default rollout count are the
        # same objects as the shared recipe's.
        """The reward improvement is asserted by the run itself, off this threshold."""
        assert scenario.DEFAULT_METRIC_THRESHOLD is scenario_realistic_gsm8k.DEFAULT_METRIC_THRESHOLD
        assert scenario.DEFAULT_NUM_ROLLOUT is scenario_realistic_gsm8k.DEFAULT_NUM_ROLLOUT

    def test_this_scenario_only_adds_arguments_its_take_over_requires(self):
        # [FUNCTION DOC] The flags (words starting with two dashes) in the scenario's training
        # arguments for a dump directory and wandb run id, sorted, are exactly: weight update
        # checker disable, load, save, save interval, wandb run id.
        """Hot restart keeps the FT recipe except for checkpointing, tracking identity, and its tensor checker."""
        declared = [
            one
            for one in shlex.split(scenario._build_train_args("/dumps", wandb_run_id="run-one"))
            if one.startswith("--")
        ]

        assert sorted(declared) == [
            "--ci-disable-weight-update-checker",
            "--load",
            "--save",
            "--save-interval",
            "--wandb-run-id",
        ]

    def test_the_run_id_is_the_identity_of_the_deployment_this_test_restarts(self):
        # [FUNCTION DOC] The wandb run id flag in the built arguments has the single value given.
        """Every process receives the outer config ID instead of an identity derived from another subsystem."""
        argv = shlex.split(scenario._build_train_args("/dumps/another-id", wandb_run_id="run-one"))

        assert ArgvManipulator.get(argv, "--wandb-run-id") == ["run-one"]

    def test_the_shared_recipe_can_keep_its_api_without_enabling_training_ft(self):
        # [FUNCTION DOC] The shared recipe's argument builder, called with fault tolerance
        # disabled and the scenario's seed, rollout count, threshold and test name, still defines
        # the api server port flag but not the fault tolerance, ft components or mini controller
        # flags.
        """Hot restart uses the cell API for injection without combining with automatic FT recovery."""
        argv = shlex.split(
            scenario_realistic_gsm8k.get_gsm8k_train_args(
                config=ExecuteTrainConfig(run_id="260101-000000-000", namespace="miles-e2e"),
                seed=scenario.DEFAULT_SEED,
                num_rollout=scenario.DEFAULT_NUM_ROLLOUT,
                metric_threshold=scenario.DEFAULT_METRIC_THRESHOLD,
                fully_async=False,
                test_name=scenario.TEST_NAME,
                enable_fault_tolerance=False,
            )
        )

        assert ArgvManipulator.get(argv, "--api-server-port")
        assert "--use-fault-tolerance" not in argv
        assert "--ft-components" not in argv
        assert "--mini-ft-controller-enable" not in argv


class TestTheInjectionPlan:
    # [CLASS DOC] Tests of the injection plan (virtual cells and forms).
    def test_virtual_cells_remain_available_before_the_closing_window(self):
        # [FUNCTION DOC] With a mocked form reporting it is within the injection window, the
        # virtual cell creator returns 2 cells.
        """A draw before the final fifteen rollouts still reaches the ordinary scheduler."""
        form = MagicMock(spec=HotRestartFaultForm)
        form.is_within_injection_window.return_value = True
        cells = scenario._create_virtual_cells_before(form)

        assert len(cells) == 2

    def test_virtual_cells_disappear_at_the_closing_window(self):
        # [FUNCTION DOC] With a mocked form reporting it is outside the window, the creator
        # returns an empty list.
        """Completing rollout 234 leaves all of 235-249 free of new take-overs."""
        form = MagicMock(spec=HotRestartFaultForm)
        form.is_within_injection_window.return_value = False
        cells = scenario._create_virtual_cells_before(form)

        assert cells == []

    def test_the_plan_supplies_two_healthy_virtual_cells(self):
        # [FUNCTION DOC] The unconditional creator returns cells named as the scenario's virtual
        # cell names list, each labelled with the hot restart cell type.
        """The regular scheduler sees a spare target without borrowing a real FT cell."""
        cells = scenario._create_virtual_cells()

        assert [cell["metadata"]["name"] for cell in cells] == list(scenario._VIRTUAL_CELL_NAMES)
        assert all(
            cell["metadata"]["labels"]["miles.io/cell-type"] == scenario._HOT_RESTART_CELL_TYPE for cell in cells
        )

    def test_the_only_fault_the_plan_may_draw_is_a_hot_restart(self):
        # [FUNCTION DOC] The forms mapping has only the hot restart cell type as a key, whose
        # value is a list with a single hot restart fault form.
        """A pod kill mixed in would make the trainer boot uuid this test pins change for a second reason."""
        forms = scenario.create_hot_restart_forms(_run("/dumps"), max_allowed_rollout_id=234)

        assert list(forms) == [scenario._HOT_RESTART_CELL_TYPE]
        assert [type(one) for one in forms[scenario._HOT_RESTART_CELL_TYPE]] == [HotRestartFaultForm]

    def test_the_plan_relaunches_the_release_the_run_was_installed_under(self):
        # [FUNCTION DOC] The created form's private launch and config attributes are the very
        # objects of the run description.
        """A relaunch of another release would leave the trainers of this run behind."""
        run = _run("/dumps")
        [form] = scenario.create_hot_restart_forms(run, max_allowed_rollout_id=234)[scenario._HOT_RESTART_CELL_TYPE]

        assert form._launch is run.launch
        assert form._config is run.config

    def test_the_form_reads_the_progress_of_the_run_it_restarts(self):
        # [FUNCTION DOC] The form's checkpoint directory equals the scenario's computation from the
        # run's dump directory and its events directory equals the run's events directory.
        """Eligibility is read off this run's checkpoints and events, not off a neighbouring dump directory."""
        run = _run("/dumps/gsm8k")
        [form] = scenario.create_hot_restart_forms(run, max_allowed_rollout_id=234)[scenario._HOT_RESTART_CELL_TYPE]

        assert form._checkpoint_dir == scenario.compute_checkpoint_dir(run.dump_dir)
        assert form._events_dir == run.events_dir


class TestTheSpecTheseConstantsAreDocumentedIn:
    # [CLASS DOC] Tests that the deployment readme quotes the scenario's constants.
    def test_the_spec_names_the_save_interval_the_run_is_installed_with(self):
        # [FUNCTION DOC] The readme text contains the save interval flag with the scenario's
        # save interval.
        """The spec is what a reader reaches for, and a stale number there is worse than none."""
        assert f"--save-interval {scenario.SAVE_INTERVAL}" in _readme()

    def test_the_spec_names_the_interval_the_draws_are_spaced_by(self):
        # [FUNCTION DOC] The readme text contains "interval <N>s" with N the integer part of the
        # default hot restart interval seconds.
        """Both halves of this soak's cost are quoted there, and both drift the same way."""
        assert f"interval {int(scenario.DEFAULT_HOT_RESTART_INTERVAL_SECONDS)}s" in _readme()


def _readme() -> str:
    # [FUNCTION DOC] HELPER: Reads and returns the text of the deployment readme in the e2e
    # deploy directory, located four directory levels above this file.
    return (Path(__file__).resolve().parents[4] / "e2e" / "deploy" / "README.md").read_text()


class TestCheckpointArgs:
    # [CLASS DOC] Tests of the checkpoint argument builder.
    def test_the_run_saves_and_resumes_from_one_directory_of_its_own(self):
        # [FUNCTION DOC] The save and load flags both have the single value of the checkpoint
        # directory computed for the dump directory.
        """A take-over restores the latest checkpoint, which the run has to be both writing and reading."""
        argv = shlex.split(scenario.build_checkpoint_args("/dumps/gsm8k"))
        checkpoint_dir = str(scenario.compute_checkpoint_dir("/dumps/gsm8k"))

        assert ArgvManipulator.get(argv, "--save") == [checkpoint_dir]
        assert ArgvManipulator.get(argv, "--load") == [checkpoint_dir]

    def test_the_run_saves_often_enough_for_a_take_over_to_find_a_checkpoint(self):
        # [FUNCTION DOC] The save interval flag has the scenario's save interval as value, and that
        # interval is below the default rollout count.
        """A run saving once would leave the whole soak ineligible, and no restart would ever fire."""
        argv = shlex.split(scenario.build_checkpoint_args("/dumps/gsm8k"))

        assert ArgvManipulator.get(argv, "--save-interval") == [str(scenario.SAVE_INTERVAL)]
        assert scenario.SAVE_INTERVAL < scenario.DEFAULT_NUM_ROLLOUT


class TestEveryDrawHasToLand:
    # [CLASS DOC] Tests of the all-attempts-landed assertion over injection event lists.
    def test_a_soak_where_no_attempt_failed_passes(self):
        # [FUNCTION DOC] Two successful hot restart injections and an unsuccessful pod crash
        # injection pass: only hot restart attempts count.
        """Every draw of this form fires, so the log should hold successes only."""
        scenario.assert_no_take_over_attempt_failed(
            [_injection(succeeded=True), _injection(succeeded=True), _crash(succeeded=False)]
        )

    def test_a_soak_where_a_take_over_attempt_failed_is_a_failure(self):
        # [FUNCTION DOC] One failed hot restart injection raises an assertion error matching
        # "take-over attempt(s) failed".
        """Without the eligibility gate a failed attempt can only mean a relaunch that did not land."""
        with pytest.raises(AssertionError, match="take-over attempt\\(s\\) failed"):
            scenario.assert_no_take_over_attempt_failed([_injection(succeeded=True), _injection(succeeded=False)])

    def test_a_run_nothing_restarted_is_a_failure_and_not_a_pass(self):
        # [FUNCTION DOC] Asserts the minimum required hot restart count constant is at least 1.
        # KNOWN ISSUE: only a constant is checked, not that a run without restarts is rejected.
        """Every assertion past this one is vacuous on a run whose script was never replaced."""
        assert scenario.MIN_HOT_RESTARTS >= 1


def _injection(*, succeeded: bool) -> InjectionEvent:
    # [FUNCTION DOC] HELPER: Builds a hot restart injection event on a virtual cell, with the
    # given success flag and not harming the cell.
    return InjectionEvent(
        cell_name="hot-restart-virtual-cell-0",
        form_name=scenario.HOT_RESTART_FORM_NAME,
        succeeded=succeeded,
        harmed=False,
    )


def _crash(*, succeeded: bool) -> InjectionEvent:
    # [FUNCTION DOC] HELPER: Builds a pod crash injection event on cell "actor-1" with the given
    # success flag and marked as harming the cell.
    return InjectionEvent(cell_name="actor-1", form_name="crash_pod", succeeded=succeeded, harmed=True)


class TestWhatEachTakeOverCost:
    # [CLASS DOC] Tests of the per-take-over loss bound computed from records.
    def test_a_take_over_resuming_from_the_last_save_is_within_the_bound(self):
        # [FUNCTION DOC] Two records: saved 19 with finished rollout 19 plus the save interval,
        # and saved 29 with finished rollout 29 plus the maximum redone steps; neither raises.
        """Whatever the draw's timing, a checkpointed take-over owes at most one save interval."""
        scenario.assert_take_over_loss_within_save_interval(
            [
                _record(index=0, saved=19, finished=19 + scenario.SAVE_INTERVAL),
                _record(index=1, saved=29, finished=29 + scenario.MAX_REDONE_STEPS_PER_TAKE_OVER),
            ]
        )

    def test_a_take_over_before_the_first_save_is_charged_for_every_step(self):
        # [FUNCTION DOC] A record with no saved iteration and finished rollout of the save
        # interval minus 2 is within the bound (the whole run so far is charged).
        """Starting over from the reference weights costs the whole run so far, which the bound still covers."""
        scenario.assert_take_over_loss_within_save_interval(
            [_record(index=0, saved=None, finished=scenario.SAVE_INTERVAL - 2)]
        )

    def test_a_take_over_that_threw_away_more_than_a_save_interval_fails(self):
        # [FUNCTION DOC] A record where finished rollout exceeds saved iteration by one more than
        # the maximum redone steps raises an assertion error matching "threw away".
        """That means the run resumed from something older than its latest checkpoint."""
        with pytest.raises(AssertionError, match="threw away"):
            scenario.assert_take_over_loss_within_save_interval(
                [_record(index=0, saved=9, finished=9 + scenario.MAX_REDONE_STEPS_PER_TAKE_OVER + 1)]
            )

    def test_the_bound_follows_the_save_interval_the_run_is_installed_with(self):
        # [FUNCTION DOC] The maximum redone steps constant equals the save interval plus one.
        """A bound spelled independently would drift the moment the interval changed."""
        assert scenario.MAX_REDONE_STEPS_PER_TAKE_OVER == scenario.SAVE_INTERVAL + 1


def _record(*, index: int, saved: int | None, finished: int) -> HotRestartRecord:
    # [FUNCTION DOC] HELPER: Builds a record with the index, saved iteration at trigger and the
    # finished rollout used as the frozen rollout id.
    return HotRestartRecord(index=index, saved_iteration_at_trigger=saved, frozen_rollout_id=finished)


def _records(count: int) -> list[HotRestartRecord]:
    # [FUNCTION DOC] HELPER: Builds the given number of records with saved 0 and finished 0.
    return [_record(index=index, saved=0, finished=0) for index in range(count)]


def _write_finished_steps(events_dir: Path, rollout_ids: Iterable[int]) -> None:
    # [FUNCTION DOC] HELPER: For each rollout id, writes one train metric event (grad norm 1.0)
    # into its own file "step-<id>.jsonl" in the events directory, without printing.
    for rollout_id in rollout_ids:
        logger = EventLogger(
            log_dir=events_dir, file_name=f"step-{rollout_id}.jsonl", source=SimpleProcessIdentity(component="main")
        )
        logger.log(MetricEvent, {"rollout_id": rollout_id, "metrics": {"train/grad_norm": 1.0}}, print_log=False)


def _replace_log(dump_dir: Path, *, rolled_aside_at: str, kept: Iterable[int]) -> None:
    # [FUNCTION DOC] HELPER: Simulates a take-over rolling the log aside: moves the events
    # directory to a hidden trash directory named with the given time stamp text and a random
    # 8-character suffix, recreates an empty events directory, and copies back the step files of
    # the kept rollout ids from the moved log.
    events_dir = dump_dir / EVENTS_DIRNAME
    replaced = dump_dir / f".trash_{rolled_aside_at}_{uuid.uuid4().hex[:8]}"
    shutil.move(str(events_dir), str(replaced))
    events_dir.mkdir(parents=True)
    for rollout_id in kept:
        shutil.copy(replaced / f"step-{rollout_id}.jsonl", events_dir / f"step-{rollout_id}.jsonl")


class TestAssertEveryTakeOverResumedWithinASaveInterval:
    # [CLASS DOC] Tests of the log-based resume bound check across take-overs.
    def test_a_take_over_that_redid_only_what_its_checkpoint_missed_passes(self, tmp_path):
        # [FUNCTION DOC] Steps 0..3 trained, log rolled aside keeping steps 0..1, steps 2..5
        # trained: the take-over redid two steps, within the bound, for one record.
        """What a take-over cost is the log it replaced minus the prefix the run that followed kept."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=range(2))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(2, 6))

        scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_a_take_over_that_resumed_further_back_than_a_save_interval_fails(self, tmp_path):
        # [FUNCTION DOC] Steps 0..5 trained, log rolled aside keeping nothing, steps 0..5 trained
        # again: raises matching "redid 6 step".
        """Measuring the resume point off the trigger's tracker would pass a run that reloaded an older save."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(6))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=[])
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(6))

        with pytest.raises(AssertionError, match="redid 6 step"):
            scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_a_take_over_that_left_no_rolled_back_log_fails(self, tmp_path):
        # [FUNCTION DOC] A single log with no rolled-aside log while one record exists: raises
        # matching "rolls the log it replaced aside".
        """A take-over whose log is missing hides the very steps this assertion counts."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))

        with pytest.raises(AssertionError, match="rolls the log it replaced aside"):
            scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_a_take_over_that_carried_a_hole_over_fails(self, tmp_path):
        # [FUNCTION DOC] The surviving log kept steps 0 and 2 (skipping 1) before continuing at 3:
        # raises matching "carried the steps".
        """A run resumes from one checkpoint, so what survives a take-over is a prefix and never a hole."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=[0, 2])
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(3, 6))

        with pytest.raises(AssertionError, match="carried the steps"):
            scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_a_take_over_fired_during_the_catch_up_is_read_off_the_log_it_rolled_aside(self, tmp_path):
        # [FUNCTION DOC] Two take-overs, the second fired while the run caught up (the second
        # rolled-aside log reaches a lower step than the first): the check passes for two records
        # when each take-over's cost is read off its own rolled-aside log.
        """A take-over during the catch-up leaves a log reaching a lower step than the log before it."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(10))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=range(8))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, [8])
        _replace_log(tmp_path, rolled_aside_at="20260902_120100", kept=range(7))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(7, 12))

        scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(2))

    def test_a_take_over_fired_during_the_catch_up_is_measured_against_its_own_log(self, tmp_path):
        # [FUNCTION DOC] The second take-over keeps only 2 steps of a log that had reached step 8
        # (ordered by when rolled aside, not by how far trained): raises matching "take-over 1
        # replaced a log that had reached step 8".
        """Ordering the rolled-aside logs by how far they trained blames the wrong take-over for the redone steps."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(10))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=range(8))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, [8])
        _replace_log(tmp_path, rolled_aside_at="20260902_120100", kept=range(2))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(2, 12))

        with pytest.raises(AssertionError, match="take-over 1 replaced a log that had reached step 8"):
            scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(2))

    def test_two_take_overs_that_rolled_their_logs_aside_in_the_same_second_fail(self, tmp_path):
        # [FUNCTION DOC] Two logs rolled aside with identical time stamps: raises matching "same
        # second" since their order is ambiguous.
        """Which log a take-over replaced is read off when it was rolled aside, so a tie leaves it unpinned."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=range(2))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(2, 5))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=range(3))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(3, 6))

        with pytest.raises(AssertionError, match="same second"):
            scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(2))
