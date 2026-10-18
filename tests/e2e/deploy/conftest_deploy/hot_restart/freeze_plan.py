# [MODULE DOC]
# HELPER: Files that tell a deployed training run when to freeze, for the hot-restart deploy
# tests. A freeze plan is a JSON file of fault-injection test actions (see the driver module's
# plan builder); the training command is pointed at it through a command-line flag so the run
# puts itself to sleep forever at the end of the planned rollout. The plan is kept beside the
# baseline and target side directories (not inside a side's own directory, which is cleared when
# that side's run starts), in a folder named "hot_restart", one file per side. The whole module
# is an ad hoc workaround marked for removal after an arguments refactor.
# KNOWN ISSUE: the existing TODO comment below the imports is truncated mid-sentence ("Reverting
# it means"), so the removal instructions are lost.
import logging
import shlex
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_freeze_plan
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE

from miles.utils.test_utils.ft_test_actions import (
    CI_FT_TEST_ACTIONS_PATH_FLAG,
    compute_frozen_sentinel_path,
    write_ft_test_actions,
)

# TODO ad hoc hack: this whole module goes away with the args refactor. Reverting it means
logger = logging.getLogger(__name__)

FREEZE_PLAN_DIRNAME: str = "hot_restart"


# TODO ad hoc hack: revert after the args refactor
def compute_freeze_plan_path(side_dump_dir: str) -> Path:
    # [FUNCTION DOC] Return the freeze plan path for a side: the parent of the side directory, then
    # "hot_restart", then "<side name>_freeze_plan.json". Asserts the side directory's last name
    # component is the baseline side name or the target side name (else the plan would be deleted
    # with the side's directory).
    side = Path(side_dump_dir)

    assert side.name in (BASELINE_SIDE, TARGET_SIDE), (
        f"the freeze plan is kept beside the two sides of one comparison, and {side_dump_dir} names neither of "
        f"them; a plan kept under a side's own dump directory is deleted when that side's run clears it"
    )
    return side.parent / FREEZE_PLAN_DIRNAME / f"{side.name}_freeze_plan.json"


# TODO ad hoc hack: revert after the args refactor
def with_freeze_plan_of(train_args: str, *, plan_path: Path) -> str:
    # [FUNCTION DOC] Return the training argument string with the test-actions path flag and the
    # shell-quoted plan path appended, followed by a trailing space. Assumes the given string
    # already ends with a space.
    return f"{train_args}{CI_FT_TEST_ACTIONS_PATH_FLAG} {shlex.quote(str(plan_path))} "


# TODO ad hoc hack: revert after the args refactor
def write_freeze_plan(plan_path: Path, *, frozen_rollout_id: int | None) -> None:
    # [FUNCTION DOC] Build the plan for the rollout id (empty plan when none) and write it to the
    # plan path through the test-actions writer; logs the result. Side effect: file I/O (the
    # writer defines the file format and directory creation).
    write_ft_test_actions(plan_path, compute_freeze_plan(frozen_rollout_id))
    logger.info(f"{plan_path} now freezes the run after step {frozen_rollout_id}")


# TODO ad hoc hack: revert after the args refactor
def arm_first_freeze(train_args: str, *, side_dump_dir: str, frozen_rollout_id: int) -> str:
    # [FUNCTION DOC] Arm the first freeze of a side. Computes the side's plan path, deletes any
    # stale frozen-sentinel file of that plan (ignoring a missing file), writes a plan freezing
    # after the given rollout id, and returns the training arguments with the plan flag appended.
    plan_path = compute_freeze_plan_path(side_dump_dir)
    compute_frozen_sentinel_path(plan_path).unlink(missing_ok=True)
    write_freeze_plan(plan_path, frozen_rollout_id=frozen_rollout_id)

    return with_freeze_plan_of(train_args, plan_path=plan_path)
