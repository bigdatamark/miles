# [MODULE DOC] Tests for the imported-name-removing primitive of the reproduction builder:
# dropping one name from a multi-name import, removing the whole statement when it was the only
# name, plain imports (module nothing), alias matching, refusal when the name is absent, the
# handling of parenthesized multi-line imports (kept exploded with two or more survivors,
# collapsed to one line for a lone survivor, kept exploded for a lone survivor when asked or when
# a comment is present), relative modules, and comment preservation. Each test writes a small
# module in a temporary directory, applies the primitive and compares the exact text.
# KNOWN ISSUE: a stray section heading comment about another primitive precedes the multi-line
# tests.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_reproduction_utils import Repro
from reproduction_testlib import _apply, _commit, _git, _write  # noqa: F401

# --- remove_imported_name ------------------------------------------------------


def test_remove_imported_name_drops_one_name_from_a_multi_name_import(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A single-line from-import of three names; removing the middle one yields the
    # single-line import of the other two, rest of the file unchanged.
    """One name is dropped from a `from m import a, b, c`; the others stay on the line."""
    (tmp_path / "m.py").write_text("from pkg import a, moved, b\n\nx = a + b\n")
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "from pkg import a, b\n\nx = a + b\n"


def test_remove_imported_name_drops_whole_statement_when_sole_name(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A from-import of one name followed by a plain import; removing that name
    # deletes its whole line, leaving the other import, a blank line and the statement.
    """Dropping the only name removes the whole `from` statement."""
    (tmp_path / "m.py").write_text("from pkg import moved\nimport os\n\nx = 1\n")
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "import os\n\nx = 1\n"


def test_remove_imported_name_drops_a_plain_import_with_module_none(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Two plain imports; with the module argument nothing and the first name,
    # that statement is removed and the other remains.
    """With module=None a plain `import name` statement is removed."""
    (tmp_path / "m.py").write_text("import gc\nimport os\n\nx = 1\n")
    r = Repro("b", "t").remove_imported_name("m.py", module=None, name="gc")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "import os\n\nx = 1\n"


def test_remove_imported_name_matches_an_asname(tmp_path: Path) -> None:
    # [FUNCTION DOC] A plain import with an alias is found and removed when both the name and the
    # alias are given.
    """The alias is matched on both the name and the asname, so `import numpy as np` is found."""
    (tmp_path / "m.py").write_text("import numpy as np\nimport os\n\nx = 1\n")
    r = Repro("b", "t").remove_imported_name("m.py", module=None, name="numpy", asname="np")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "import os\n\nx = 1\n"


def test_remove_imported_name_asserts_when_absent(tmp_path: Path) -> None:
    # [FUNCTION DOC] The name is not among those imported from the module; applying the primitive
    # raises an assertion error.
    """Removing a name that is not imported raises, so a wrong recipe fails loudly."""
    (tmp_path / "m.py").write_text("from pkg import a, b\n")
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="missing")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


# --- add_import ----------------------------------------------------------------


def test_remove_imported_name_preserves_the_multiline_form(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A parenthesized three-name import, one name per line with a trailing comma;
    # removing the middle name deletes only its line, keeping the parentheses and trailing comma.
    """Pruning a name from an exploded import deletes only that line, so the parens and the
    magic trailing comma survive and the formatter keeps it multi-line (a flat rebuild would
    collapse an import the target left multi-line)."""
    (tmp_path / "m.py").write_text("from pkg import (\n    a,\n    moved,\n    b,\n)\n\nx = a + b\n")
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "from pkg import (\n    a,\n    b,\n)\n\nx = a + b\n"


def test_remove_imported_name_multiline_down_to_one_collapses(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A parenthesized two-name import; removing the first name leaves one name,
    # and the statement is rebuilt on a single line without parentheses.
    """Pruning an exploded import down to a single surviving name collapses it to one line:
    the formatter does not keep a lone name exploded, so a preserved-multiline form would
    not match the target."""
    (tmp_path / "m.py").write_text("from pkg import (\n    moved,\n    a,\n)\n\nx = a\n")
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "from pkg import a\n\nx = a\n"


def test_remove_imported_name_down_to_one_with_a_comment_stays_exploded(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A parenthesized two-name import where the surviving name has a trailing
    # comment; removing the other name deletes only its line, keeping the parentheses, the comma
    # and the comment.
    """A lone survivor that carries a comment stays exploded (a rebuild would drop the
    comment); only its own line is deleted."""
    (tmp_path / "m.py").write_text("from pkg import (\n    moved,\n    a,  # keep me\n)\n\nx = a\n")
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "from pkg import (\n    a,  # keep me\n)\n\nx = a\n"


def test_remove_imported_name_matches_a_relative_module(tmp_path: Path) -> None:
    # [FUNCTION DOC] A single-line relative from-import (one leading dot) of two names; passing
    # the module as a dot plus the package name removes the second name.
    """A relative `from .pkg import` is matched via its level dots."""
    (tmp_path / "m.py").write_text("from .pkg import a, moved\n\nx = a\n")
    r = Repro("b", "t").remove_imported_name("m.py", module=".pkg", name="moved")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "from .pkg import a\n\nx = a\n"


def test_remove_imported_name_preserves_comments_in_a_multiline_import(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A three-name parenthesized import where a surviving line has a comment;
    # after removing the middle name the comment text is still in the file.
    """Comments on surviving lines of a pruned parenthesized import must not vanish."""
    (tmp_path / "m.py").write_text(
        "from pkg import (\n" "    a,  # used by frobnicator\n" "    moved,\n" "    b,\n" ")\n" "\n" "x = a + b\n"
    )
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved")
    _apply(r, tmp_path)
    assert "# used by frobnicator" in (tmp_path / "m.py").read_text()


def test_remove_imported_name_keep_exploded_holds_a_lone_survivor_multiline(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A parenthesized two-name import; removing the first name with the keep
    # exploded option deletes only its line, so the survivor stays on its own line inside the
    # parentheses with its trailing comma.
    """With keep_exploded, pruning down to a single survivor deletes only the removed line,
    so the survivor keeps its magic trailing comma and the import stays multi-line (the
    author's choice, which the source cannot reveal). A regenerating impl would collapse.
    """
    (tmp_path / "m.py").write_text("from pkg import (\n    moved,\n    a,\n)\n\nx = a\n")
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved", keep_exploded=True)
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "from pkg import (\n    a,\n)\n\nx = a\n"
