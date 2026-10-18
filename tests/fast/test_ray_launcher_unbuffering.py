# [MODULE DOC] Repository lint-style tests for launcher scripts that submit ray jobs: among the
# git-tracked python and shell files under the examples, scripts, tools and external-utils
# directories, every file mentioning a ray runtime environment marker must set the unbuffered
# python environment variable (otherwise ray buffers worker stdout) and must not use the
# misspelled variant. Also checks that more than 30 such launchers are found, so a discovery bug
# cannot make the checks vacuous. Discovery runs at collection time and needs git and a git
# checkout.
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_DIRS = ("examples", "scripts", "tools", "miles/utils/external_utils")
RAY_RUNTIME_ENV_MARKERS = ("runtime-env-json", "runtime_env=", "runtime_env_json")


def tracked_files() -> list[Path]:
    # [FUNCTION DOC] HELPER: Runs git ls-files with null separators in the repository for the
    # launcher directories (subprocess; raises a called-process error on failure) and returns the
    # absolute paths of tracked files ending in .py or .sh.
    listing = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", *LAUNCHER_DIRS],
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / name for name in listing.stdout.split("\0") if name.endswith((".py", ".sh"))]


def ray_launchers() -> list[Path]:
    # [FUNCTION DOC] HELPER: Returns the tracked files whose text contains any of the ray
    # runtime-environment markers (reads every tracked launcher file).
    return [path for path in tracked_files() if any(marker in path.read_text() for marker in RAY_RUNTIME_ENV_MARKERS)]


def test_the_repo_has_ray_launchers_to_check() -> None:
    # [FUNCTION DOC] Asserts discovery finds more than 30 ray launchers.
    """A discovery bug that finds nothing would make every other check in this file vacuous."""
    assert len(ray_launchers()) > 30


@pytest.mark.parametrize("launcher", ray_launchers(), ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_every_ray_launcher_unbuffers_python(launcher: Path) -> None:
    # [FUNCTION DOC] Parametrized over discovered launchers (ids are repo-relative paths): the
    # file text contains the unbuffered python variable name.
    """Ray buffers worker stdout unless PYTHONUNBUFFERED rides along with the job it submits."""
    assert "PYTHONUNBUFFERED" in launcher.read_text()


@pytest.mark.parametrize("launcher", ray_launchers(), ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_no_ray_launcher_spells_the_variable_wrong(launcher: Path) -> None:
    # [FUNCTION DOC] Parametrized over discovered launchers: after removing every correct
    # unbuffered variable name from the text, the misspelled buffered name no longer appears.
    """PYTHONBUFFERED is not a variable python reads; the typo silently buffers everything."""
    assert "PYTHONBUFFERED" not in launcher.read_text().replace("PYTHONUNBUFFERED", "")
