# [MODULE DOC] HELPER: Pytest configuration for the reproduction utilities tests. Puts the
# scripts directory on the import path and provides the shared git repository fixture. The
# trailing section comment is a leftover heading with no code under it.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reproduction_testlib import _git


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    # [FUNCTION DOC] HELPER: Test fixture. Creates a directory named repo in the per-test
    # temporary path, initializes a quiet git repository (default branch per git config), sets
    # a fixed author email and name, and disables commit signing. Returns the root path.
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    _git(root, "config", "commit.gpgsign", "false")
    return root


# --- exec_command --------------------------------------------------------------
