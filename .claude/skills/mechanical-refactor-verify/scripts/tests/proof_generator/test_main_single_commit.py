# [MODULE DOC] Tests for the single-commit mode of the proof generator command line entry
# point. A helper builds a commit extracting a helper method; tests change into the test
# repository and call the entry point with a commit id, checking the returned exit status.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generator_testlib import _commit, _write  # noqa: F401
from mechanical_refactor_proof_generator import _main


def _extract_function_commit(repo: Path) -> str:
    # [FUNCTION DOC] HELPER: Creates two commits in the repository. Base: class C with a dispatch
    # method computing x from an attribute and y from x plus n and returning y, and a keep method.
    # Second ("extract _combine from dispatch"): the two computation lines move into a new
    # keyword-only-parameter method _combine that also returns y, and dispatch assigns y from a
    # call to it. Returns the SHA of the second commit.
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        x = self.a\n"
                "        y = x + n\n"
                "        return y\n"
                "\n"
                "    def keep(self):\n"
                "        return 0\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        y = self._combine(n=n)\n"
                "        return y\n"
                "\n"
                "    def _combine(self, *, n):\n"
                "        x = self.a\n"
                "        y = x + n\n"
                "        return y\n"
                "\n"
                "    def keep(self):\n"
                "        return 0\n"
            )
        },
    )
    return _commit(repo, "extract _combine from dispatch")


def test_single_commit_extract_function_reproduces_instead_of_unsupported(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Builds the extract-helper commit, changes the working directory to the
    # repository (the entry point locates the repository from the current directory) and asserts
    # the entry point, given that commit id, returns exit status 0 (reproduced, not unsupported).
    """A pure intra-file extract_function commit run in single-commit mode reproduces (exit 0),
    not UNSUPPORTED -- the relocates check must count extract_functions like the range path.
    """
    sha = _extract_function_commit(repo)
    monkeypatch.chdir(repo)
    assert _main([sha]) == 0


def test_single_commit_pure_rename_is_unsupported(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Base commit: a module defining foo. Second commit renames it to bar with the
    # same body. Changes into the repository and asserts the entry point returns exit status 1
    # for that commit (no definition relocated, so unsupported).
    """A commit that relocates no definition (a bare rename) stays UNSUPPORTED with exit 1."""
    _write(repo, **{"m.py": "def foo():\n    return 1\n"})
    _commit(repo, "base")
    _write(repo, **{"m.py": "def bar():\n    return 1\n"})
    sha = _commit(repo, "rename foo to bar")
    monkeypatch.chdir(repo)
    assert _main([sha]) == 1
