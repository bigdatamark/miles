# [MODULE DOC]
# Tests for the hot-restart freeze plan file and the run arguments that name it (marked in the
# source as an ad hoc hack to be reverted after the argument refactor): the plan file is rewritten
# atomically (no partial files left), arming the first freeze clears the previous run's frozen
# sentinel, plans of different sides live in different files outside the directory the side
# clears and survive that clearing, a directory naming no known side is refused, and the run
# arguments carry the plan path (shell-quoted, stable across relaunches) and never the plan.
import json
import shlex

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_freeze_plan
from tests.e2e.deploy.conftest_deploy.hot_restart.freeze_plan import (
    arm_first_freeze,
    compute_freeze_plan_path,
    with_freeze_plan_of,
    write_freeze_plan,
)
from tests.e2e.ft.conftest_ft import app as ft_app
from tests.e2e.ft.conftest_ft import execution as ft_execution
from tests.e2e.ft.conftest_ft.app import TARGET_SIDE, RunSideRequest
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import ArgvManipulator
from miles.utils.test_utils.ft_test_actions import (
    CI_FT_TEST_ACTIONS_PATH_FLAG,
    SLEEP_FOREVER_AT_END_ACTION,
    FTTestAction,
    read_frozen_rollout_id,
    write_frozen_sentinel,
)


# TODO ad hoc hack: revert after the args refactor
class TestTheFreezePlanFile:
    # [CLASS DOC] Tests of the freeze plan file contents and lifecycle.
    def test_the_plan_a_relaunch_writes_replaces_the_one_the_run_was_installed_with(self, tmp_path):
        # [FUNCTION DOC] Writing a plan for rollout 2 then rollout 4 to the same path leaves a
        # file holding exactly one action at rollout 4 with the sleep-forever action.
        """The run rereads this one path every step, so a second plan beside it would never be seen."""
        path = compute_freeze_plan_path(f"{tmp_path}/target")
        write_freeze_plan(path, frozen_rollout_id=2)
        write_freeze_plan(path, frozen_rollout_id=4)

        assert [FTTestAction(**one) for one in json.loads(path.read_text())] == [
            FTTestAction(at_rollout=4, action=SLEEP_FOREVER_AT_END_ACTION)
        ]

    def test_arming_a_run_clears_what_the_previous_run_froze_at(self, tmp_path):
        # [FUNCTION DOC] After writing a plan and a frozen sentinel for a previous run, arming the
        # first freeze (with an argument string, the side's dump directory and rollout 4) leaves
        # no readable frozen rollout id.
        """The plan directory outlives the run that clears its dumps, so its sentinel outlives it too, and a
        driver reading a stale one lets the take-over go before this run has frozen anything."""
        path = compute_freeze_plan_path(f"{tmp_path}/target")
        write_freeze_plan(path, frozen_rollout_id=2)
        write_frozen_sentinel(path, rollout_id=2)

        arm_first_freeze("--some-flag some-value ", side_dump_dir=f"{tmp_path}/target", frozen_rollout_id=4)

        assert read_frozen_rollout_id(path) is None

    def test_the_plan_of_two_dump_directories_never_lands_in_one_file(self, tmp_path):
        # [FUNCTION DOC] The plan paths for the target and baseline side dump directories differ.
        """The two sides of the comparison run at once, and one shared plan would freeze the baseline too."""
        assert compute_freeze_plan_path(f"{tmp_path}/target") != compute_freeze_plan_path(f"{tmp_path}/baseline")

    def test_a_partial_write_is_never_what_the_run_reads(self, tmp_path):
        # [FUNCTION DOC] After writing a plan for rollout 2 the file parses as the freeze plan
        # computed for 2 and no leftover partial-write files remain in the directory.
        """The run rereads the plan every step, so it may only ever see a whole one."""
        path = compute_freeze_plan_path(f"{tmp_path}/target")
        write_freeze_plan(path, frozen_rollout_id=2)

        assert json.loads(path.read_text()) == compute_freeze_plan(2)
        assert list(path.parent.glob("*.partial")) == []


# TODO ad hoc hack: revert after the args refactor
class TestTheArgumentsThatNameThePlan:
    # [CLASS DOC] Tests of the run argument string that references the plan file.
    def test_the_run_is_told_the_path_and_never_the_plan_itself(self, tmp_path):
        # [FUNCTION DOC] After adding the plan path to an argument string, splitting it like a
        # shell yields the actions-path flag with exactly the plan path as its value, and the
        # original arguments are preserved.
        """Every worker pod's command carries these arguments, and a hot restart may not change one of them."""
        path = compute_freeze_plan_path(f"{tmp_path}/target")
        args = with_freeze_plan_of("--save /ckpt --num-rollout 6 ", plan_path=path)

        assert ArgvManipulator.get(shlex.split(args), CI_FT_TEST_ACTIONS_PATH_FLAG) == [str(path)]
        assert "--save /ckpt" in args

    def test_the_arguments_of_a_relaunch_are_the_ones_the_run_is_already_up_with(self, tmp_path):
        # [FUNCTION DOC] The argument string built before and after two plan rewrites (rollout 2
        # then none) is identical, since only the file changes.
        """A relaunch whose argv differs from the installed one is refused as more than a hot restart."""
        path = compute_freeze_plan_path(f"{tmp_path}/target")
        installed = with_freeze_plan_of("--save /ckpt ", plan_path=path)
        write_freeze_plan(path, frozen_rollout_id=2)
        write_freeze_plan(path, frozen_rollout_id=None)

        assert installed == with_freeze_plan_of("--save /ckpt ", plan_path=path)

    def test_the_path_survives_being_split_back_into_arguments(self, tmp_path):
        # [FUNCTION DOC] A plan path containing a space is quoted so that shell splitting yields
        # exactly one value for the actions-path flag.
        """The path reaches the pods as one argument, so an unquoted one would arrive as several."""
        args = with_freeze_plan_of("--save /ckpt ", plan_path=compute_freeze_plan_path(f"{tmp_path}/a dir/target"))

        assert len(ArgvManipulator.get(shlex.split(args), CI_FT_TEST_ACTIONS_PATH_FLAG)) == 1


# TODO ad hoc hack: revert after the args refactor
class TestWhereTheFreezePlanLives:
    # [CLASS DOC] Tests of the plan file location rules.
    def test_the_plan_sits_outside_the_directory_its_side_clears(self, tmp_path):
        # [FUNCTION DOC] The plan path is neither the side's dump directory nor inside it.
        """Each side deletes its own dump directory as its first act, and would take the plan with it."""
        side_dump = tmp_path / "target"
        path = compute_freeze_plan_path(str(side_dump))

        assert side_dump not in path.parents and path != side_dump

    def test_a_directory_that_names_no_side_is_refused(self, tmp_path):
        # [FUNCTION DOC] A dump directory named "checkpoints" (neither side) raises an assertion
        # error matching "names neither of them".
        """A plan under a path nobody recognises is a plan nothing guarantees the lifetime of."""
        with pytest.raises(AssertionError, match="names neither of them"):
            compute_freeze_plan_path(f"{tmp_path}/checkpoints")

    def test_the_plan_survives_the_side_clearing_its_dumps(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Writes a plan for the target side and a leftover file in its dump
        # directory, replaces config resolution with a fake whose backend does nothing, then runs
        # one release for the target side. Asserts the side cleared its own dump directory (the
        # leftover file is gone) while the plan file still exists.
        """The run's first act is to clear its dump directory; a plan it deleted is a run never frozen."""
        base = tmp_path / "dumps"
        side_dump = base / TARGET_SIDE
        path = compute_freeze_plan_path(str(side_dump))
        write_freeze_plan(path, frozen_rollout_id=2)
        side_dump.mkdir(parents=True, exist_ok=True)
        (side_dump / "leftover.txt").write_text("from a previous run")

        monkeypatch.setattr(ft_execution, "_resolve_config", lambda config: _FakeConfig())
        ft_app.run_one_release(
            RunSideRequest(
                side=TARGET_SIDE,
                mode=_MODE,
                train_args="--some-flag some-value ",
                dump_dir=str(side_dump),
                config=ExecuteTrainConfig(run_id="demo"),
                enable_dumper=False,
            )
        )

        assert not (side_dump / "leftover.txt").exists(), "the side is expected to clear its own dump directory"
        assert path.is_file(), "the freeze plan has to outlive the side clearing its dumps"


class _FakeBackend:
    # [CLASS DOC] HELPER: Fake execution backend whose train execution does nothing.
    def execute_train(self, **_kwargs) -> None:
        # [FUNCTION DOC] HELPER: Ignores all keyword arguments and returns nothing.
        return None


class _FakeConfig:
    # [CLASS DOC] HELPER: Fake resolved configuration that creates the fake backend.
    @staticmethod
    def create_backend() -> _FakeBackend:
        # [FUNCTION DOC] HELPER: Returns a new fake backend.
        return _FakeBackend()


_MODE: FTTestMode = FTTestMode(
    model_name="demo", model_hf_repo="demo/demo", megatron_model_type="demo", num_cells=1, parallel_args=""
)
