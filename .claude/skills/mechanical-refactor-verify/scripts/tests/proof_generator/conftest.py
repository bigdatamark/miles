# [MODULE DOC] HELPER: Pytest configuration for the proof generator tests. Adds the scripts
# directory (two levels above this file) to the front of the import path so the generator
# module can be imported by the tests, and provides the shared git repository fixture.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generator_testlib import _git


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    # [FUNCTION DOC] HELPER: Test fixture. Creates an empty directory named repo inside the
    # per-test temporary path, initializes a quiet git repository there (default branch name
    # left to git's configuration), and sets a fixed test author email and name and disables
    # commit signing so commits work non-interactively. Returns the repository root path.
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    _git(root, "config", "commit.gpgsign", "false")
    return root
