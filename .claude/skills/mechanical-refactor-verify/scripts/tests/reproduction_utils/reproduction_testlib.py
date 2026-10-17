# [MODULE DOC] HELPER: Shared helpers for the reproduction utilities tests: a function that
# applies a built reproduction recipe to a plain directory, plus git wrappers and a file
# writer used to stage commits in test repositories.
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_reproduction_utils import Repro


def _apply(repro: Repro, root: Path) -> None:
    # [FUNCTION DOC] HELPER: Executes, in recorded order, each operation stored on the given
    # reproduction recipe object, passing the directory root to each. No git is involved;
    # operations mutate files in the directory. Returns nothing.
    """Run a built Repro's recorded operations against a plain directory (no git)."""
    for op in repro.ops:
        op(root)


def _git(repo: Path, *args: str) -> str:
    # [FUNCTION DOC] HELPER: Runs git with the arguments in the repository directory, raising
    # on non-zero exit, and returns stripped standard output text.
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _write(repo: Path, **files: str | None) -> None:
    # [FUNCTION DOC] HELPER: Writes or deletes files under the repository directory. Each
    # keyword name is a relative path where a double underscore means a directory separator.
    # A value of nothing deletes the existing file; a string creates parent directories and
    # writes the text.
    for name, content in files.items():
        path = repo / name.replace("__", "/")
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


def _commit(repo: Path, message: str) -> str:
    # [FUNCTION DOC] HELPER: Stages everything and commits quietly with the message; returns
    # the full SHA of the new HEAD.
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")
