# [MODULE DOC] Tests for the standalone verification function and the builder's run method.
# Verification tests use the repository fixture with a base commit (one file) and a target
# commit (the file split in two), a transform that recreates the split, and a stubbed formatter
# step (the pre-commit command is intercepted by replacing the command runner on the module) and
# a temporary-directory factory patched to return a fixed non-existing path. They check the
# printed pass message, the failure exit, the branch left behind, and the extra commit when the
# formatter dirties the tree. The run tests check the end-to-end result of a lowering recipe
# (empty diff and pass message) and a bundled unrelated change (non-empty diff and residual
# message).
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mechanical_refactor_reproduction_utils as rr
from mechanical_refactor_reproduction_utils import Repro, verify_mechanical_refactor
from reproduction_testlib import _apply, _commit, _git, _write  # noqa: F401

# --- verify_mechanical_refactor ------------------------------------------------


def _silence_precommit(monkeypatch) -> None:
    # [FUNCTION DOC] HELPER: Replaces the command runner of the utilities module with a wrapper
    # (via the test monkeypatch fixture) that returns an empty string for any command starting
    # with the formatter name and delegates everything else to the real runner, so the formatter
    # is never run.
    real = rr.exec_command

    def fake(cmd: str, cwd=None, check=True):
        # [FUNCTION DOC] HELPER: Stand-in command runner: formatter commands are skipped (empty
        # output); all others run through the real runner with the same arguments.
        if cmd.startswith("pre-commit"):
            return ""
        return real(cmd, cwd=cwd, check=check)

    monkeypatch.setattr(rr, "exec_command", fake)


def test_reproduce_passes_when_transform_matches_target(repo: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    # [FUNCTION DOC] Base commit has one three-line file; the target splits it into two files
    # (two lines and one line) and removes the original. The transform reads the original lines,
    # writes the two files, deletes the original and commits. With the repository as working
    # directory, a fixed worktree path and the formatter silenced, verification prints a message
    # containing the pass marker and does not exit the process.
    """A transform that recreates the target tree reports PASS and does not exit."""
    _write(repo, **{"src.py": "line1\nline2\nline3\n"})
    base = _commit(repo, "base")
    _write(repo, **{"src.py": None, "a.py": "line1\nline2\n", "b.py": "line3\n"})
    target = _commit(repo, "split")

    def transform(root: Path) -> None:
        # [FUNCTION DOC] HELPER: Recreates the split in the worktree root: reads the original file
        # lines, writes the first two lines to one new file and the third to another, deletes the
        # original, and commits everything with the message "split".
        lines = (root / "src.py").read_text().splitlines(keepends=True)
        (root / "a.py").write_text("".join(lines[0:2]))
        (root / "b.py").write_text("".join(lines[2:3]))
        (root / "src.py").unlink()
        rr.git_add_and_commit("split", cwd=str(root))

    monkeypatch.chdir(repo)
    monkeypatch.setattr(rr.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wt"))
    _silence_precommit(monkeypatch)

    verify_mechanical_refactor(base, target, transform)
    assert "PASS" in capsys.readouterr().out


def test_reproduce_exits_when_transform_diverges(repo: Path, tmp_path: Path, monkeypatch) -> None:
    # [FUNCTION DOC] Same base and target as the passing test (file split in two). The transform
    # writes wrong content ("WRONG") into the first new file, so the resulting tree differs from
    # the target. Asserts verification raises a system exit.
    """A transform that produces a different tree fails with a non-zero exit."""
    _write(repo, **{"src.py": "line1\nline2\nline3\n"})
    base = _commit(repo, "base")
    _write(repo, **{"src.py": None, "a.py": "line1\nline2\n", "b.py": "line3\n"})
    target = _commit(repo, "split")

    def wrong_transform(root: Path) -> None:
        # [FUNCTION DOC] HELPER: Writes "WRONG" to the first new file and the correct third line
        # to the second, deletes the original and commits with the message "split".
        (root / "a.py").write_text("WRONG\n")
        (root / "b.py").write_text("line3\n")
        (root / "src.py").unlink()
        rr.git_add_and_commit("split", cwd=str(root))

    monkeypatch.chdir(repo)
    monkeypatch.setattr(rr.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wt"))
    _silence_precommit(monkeypatch)

    with pytest.raises(SystemExit):
        verify_mechanical_refactor(base, target, wrong_transform)


def test_reproduce_creates_verify_branch_on_pass(repo: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    # [FUNCTION DOC] Same setup and transform as the passing test. After a pass, asserts the
    # message has the pass marker and that the repository still lists the branch named with the
    # verify prefix and the first eight characters of the base commit (it is left behind).
    """A PASS run leaves a verify-mechanical-<base[:8]> branch in the repo."""
    _write(repo, **{"src.py": "line1\nline2\nline3\n"})
    base = _commit(repo, "base")
    _write(repo, **{"src.py": None, "a.py": "line1\nline2\n", "b.py": "line3\n"})
    target = _commit(repo, "split")

    def transform(root: Path) -> None:
        # [FUNCTION DOC] HELPER: Same split transform as in the passing test: first two lines to
        # one new file, third line to another, original deleted, committed as "split".
        lines = (root / "src.py").read_text().splitlines(keepends=True)
        (root / "a.py").write_text("".join(lines[0:2]))
        (root / "b.py").write_text("".join(lines[2:3]))
        (root / "src.py").unlink()
        rr.git_add_and_commit("split", cwd=str(root))

    monkeypatch.chdir(repo)
    monkeypatch.setattr(rr.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wt"))
    _silence_precommit(monkeypatch)

    verify_mechanical_refactor(base, target, transform)
    assert "PASS" in capsys.readouterr().out
    branch = f"verify-mechanical-{base[:8]}"
    assert _git(repo, "branch", "--list", branch).endswith(branch)


def _precommit_writes_file(monkeypatch, filename: str, contents: str) -> None:
    # [FUNCTION DOC] HELPER: Replaces the command runner of the utilities module (via the test
    # monkeypatch fixture) with a wrapper that, for any command starting with the formatter name,
    # writes the given contents to the given file name inside the command's working directory and
    # returns an empty string (simulating a formatter that creates or rewrites a file); every
    # other command goes to the real runner.
    real = rr.exec_command

    def fake(cmd: str, cwd=None, check=True):
        # [FUNCTION DOC] HELPER: Stand-in command runner as described for the enclosing helper;
        # a missing working directory for a formatter command would raise a type error.
        if cmd.startswith("pre-commit"):
            (Path(cwd) / filename).write_text(contents)
            return ""
        return real(cmd, cwd=cwd, check=check)

    monkeypatch.setattr(rr, "exec_command", fake)


def test_reproduce_commits_pre_commit_fixes_when_tree_left_dirty(
    repo: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    # [FUNCTION DOC] Base commit has one small file; the target edits it and adds a second file
    # that the stubbed formatter will create. The transform rewrites the first file and commits
    # as "transform". With the formatter stub writing the second file, verification passes (pass
    # marker in output), and the log of the verification branch shows two latest commits: the
    # automatic fixes commit on top of the transform commit.
    """When pre-commit reformats and leaves the tree dirty, a 'pre-commit fixes' commit
    is created on top of the transform commit."""
    _write(repo, **{"src.py": "hello\n"})
    base = _commit(repo, "base")
    _write(repo, **{"src.py": "hello world\n", "formatted.py": "auto\n"})
    target = _commit(repo, "edit")

    def transform(root: Path) -> None:
        # [FUNCTION DOC] HELPER: Rewrites the source file with the new text and commits it as
        # "transform".
        (root / "src.py").write_text("hello world\n")
        rr.git_add_and_commit("transform", cwd=str(root))

    monkeypatch.chdir(repo)
    monkeypatch.setattr(rr.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wt"))
    _precommit_writes_file(monkeypatch, "formatted.py", "auto\n")

    verify_mechanical_refactor(base, target, transform)
    assert "PASS" in capsys.readouterr().out
    branch = f"verify-mechanical-{base[:8]}"
    subjects = _git(repo, "log", "--format=%s", "-2", branch).splitlines()
    assert subjects == ["pre-commit fixes", "transform"]


# --- Repro.run end-to-end ------------------------------------------------------


def test_repro_run_passes_on_a_faithful_call_site_lowering(repo: Path, monkeypatch, capsys) -> None:
    # [FUNCTION DOC] Base: a file with a call of an owner's method with a receiver and a number;
    # target: the call lowered onto the receiver. Runs the builder with the lowering recipe in the
    # repository as working directory and the formatter silenced. Asserts the returned diff is
    # empty and the pass message was printed.
    """End-to-end: a lowering reproduces the commit byte-for-byte (pre-commit stubbed)."""
    _write(repo, **{"c.py": "r = Old.foo(self.n, 5)\n"})
    base = _commit(repo, "base")
    _write(repo, **{"c.py": "r = self.n.foo(5)\n"})
    target = _commit(repo, "lower the call site")
    monkeypatch.chdir(repo)
    _silence_precommit(monkeypatch)

    diff = Repro(base, target).lower_call_sites("foo", "Old", paths=["c.py"]).run()
    assert diff == ""
    assert "PASS" in capsys.readouterr().out


def test_repro_run_reports_residual_when_a_change_is_bundled(repo: Path, monkeypatch, capsys) -> None:
    # [FUNCTION DOC] Same as the previous test, but the target also changes an unrelated
    # constant's value. Asserts the returned diff mentions that constant and the residual message
    # was printed.
    """A bundled non-relocation change surfaces as a non-empty residual diff."""
    _write(repo, **{"c.py": "r = Old.foo(self.n, 5)\nUNRELATED = 1\n"})
    base = _commit(repo, "base")
    _write(repo, **{"c.py": "r = self.n.foo(5)\nUNRELATED = 2\n"})
    target = _commit(repo, "lower the call AND change a constant")
    monkeypatch.chdir(repo)
    _silence_precommit(monkeypatch)

    diff = Repro(base, target).lower_call_sites("foo", "Old", paths=["c.py"]).run()
    assert "UNRELATED" in diff
    assert "RESIDUAL" in capsys.readouterr().out
