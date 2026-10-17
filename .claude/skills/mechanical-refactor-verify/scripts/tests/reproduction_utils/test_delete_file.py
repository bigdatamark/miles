# [MODULE DOC] Tests for the file-deleting primitive of the reproduction builder: it deletes a
# module holding only scaffolding (imports, optionally a module logger), is a no-op for a
# missing file, and refuses (assertion error, file kept) when a definition or a non-logger
# assignment remains. Each test writes a file in a temporary directory, applies the primitive
# and checks existence.
# KNOWN ISSUE: the file ends with a trailing section heading comment about another primitive
# that has no tests under it.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_reproduction_utils import Repro
from reproduction_testlib import _apply, _commit, _git, _write  # noqa: F401


def test_delete_file_removes_emptied_source(tmp_path: Path) -> None:
    # [FUNCTION DOC] A file with just an import is deleted by the primitive.
    """delete_file removes a source module left empty after its defs relocated."""
    (tmp_path / "gone.py").write_text("import os\n")
    r = Repro("b", "t").delete_file("gone.py")
    _apply(r, tmp_path)
    assert not (tmp_path / "gone.py").exists()


def test_delete_file_refuses_a_file_with_remaining_definitions(tmp_path: Path) -> None:
    # [FUNCTION DOC] A file with a function definition: applying the primitive raises an
    # assertion error and the file still exists.
    """Deleting a module that still contains defs must fail loudly."""
    (tmp_path / "live.py").write_text("def still_used():\n    return 42\n")
    r = Repro("b", "t").delete_file("live.py")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)
    assert (tmp_path / "live.py").exists()


def test_delete_file_on_a_missing_path_is_a_no_op(tmp_path: Path) -> None:
    # [FUNCTION DOC] Deleting a path that does not exist raises nothing and leaves it absent.
    """Deleting an already-absent file does nothing and raises nothing."""
    r = Repro("b", "t").delete_file("nope.py")
    _apply(r, tmp_path)
    assert not (tmp_path / "nope.py").exists()


def test_delete_file_allows_a_bare_module_logger(tmp_path: Path) -> None:
    # [FUNCTION DOC] A file with only a logging import and a module logger assignment is deleted.
    """A leftover module holding only imports and a `logger` is deletable scaffolding."""
    (tmp_path / "gone.py").write_text("import logging\n\nlogger = logging.getLogger(__name__)\n")
    r = Repro("b", "t").delete_file("gone.py")
    _apply(r, tmp_path)
    assert not (tmp_path / "gone.py").exists()


def test_delete_file_still_refuses_a_non_logger_assignment(tmp_path: Path) -> None:
    # [FUNCTION DOC] A file with a dictionary-valued module constant: applying the primitive
    # raises an assertion error and the file still exists.
    """A leftover module-level assignment other than a logger blocks deletion."""
    (tmp_path / "live.py").write_text("CONFIG = {'a': 1}\n")
    r = Repro("b", "t").delete_file("live.py")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)
    assert (tmp_path / "live.py").exists()


# --- adversarial audit: extract_function -----------------------------------------
