# [MODULE DOC] Tests for the import-removing primitive of the reproduction builder: scoping to
# one function, removing every occurrence, removal with the trailing blank line, keeping a code
# line that follows, refusals (no match, missing scope function), semicolon-joined statements,
# and exact token matching (a package import is not matched as a prefix of a submodule import).
# Each test writes a small module in a temporary directory, applies the primitive and compares
# text or checks for substrings.
# KNOWN ISSUE: stray section heading comments (one for another primitive, one for an
# adversarial audit of import primitives) sit between the tests.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_reproduction_utils import Repro
from reproduction_testlib import _apply, _commit, _git, _write  # noqa: F401

# --- remove_import -------------------------------------------------------------


def test_remove_import_scoped_leaves_module_level_same_text(tmp_path: Path) -> None:
    # [FUNCTION DOC] The file has the same import both in a type-checking block at module level
    # and inside a function body (followed by a blank line). Removing it scoped to the function:
    # exactly one occurrence of the import text remains (in the block), and the function now
    # starts directly with its return statement (the blank line after the import was dropped).
    """Scoped to a function, it removes the local import but not a same-text module-level
    one (e.g. a TYPE_CHECKING guard), and drops the import's trailing blank line."""
    (tmp_path / "m.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from pkg.mod import Thing\n"
        "\n"
        "def caller(self):\n"
        "    from pkg.mod import Thing\n"
        "\n"
        "    return Thing.go(self.x)\n"
    )
    r = Repro("b", "t").remove_import("m.py", "from pkg.mod import Thing", in_function="caller")
    _apply(r, tmp_path)
    out = (tmp_path / "m.py").read_text()
    assert out.count("from pkg.mod import Thing") == 1
    assert "if TYPE_CHECKING:\n    from pkg.mod import Thing" in out
    assert "def caller(self):\n    return Thing.go(self.x)\n" in out


def test_remove_import_removes_every_occurrence_in_scope(tmp_path: Path) -> None:
    # [FUNCTION DOC] A function imports the same name twice (once at its top, once in a nested
    # conditional). Scoped removal deletes both; the text no longer appears.
    """All matching local imports in the function are removed, not just the first."""
    (tmp_path / "m.py").write_text(
        "def caller(self):\n"
        "    from pkg import M\n"
        "\n"
        "    M.a(self.x)\n"
        "    if cond:\n"
        "        from pkg import M\n"
        "\n"
        "        M.b(self.y)\n"
    )
    r = Repro("b", "t").remove_import("m.py", "from pkg import M", in_function="caller")
    _apply(r, tmp_path)
    assert "from pkg import M" not in (tmp_path / "m.py").read_text()


# --- remove_imported_name ------------------------------------------------------


# --- adversarial audit: import primitives ----------------------------------------


def test_remove_import_unscoped_removes_module_level_import_and_blank(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Module with a plain import, a from-import, a blank line and a statement.
    # Removing the from-import with no scope deletes it and the following blank line, leaving the
    # plain import directly followed by the statement.
    """Without in_function the matching module-level import and its trailing blank go."""
    (tmp_path / "m.py").write_text("import os\nfrom pkg import Thing\n\nx = Thing\n")
    r = Repro("b", "t").remove_import("m.py", "from pkg import Thing")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "import os\nx = Thing\n"


def test_remove_import_keeps_a_code_line_directly_after_the_import(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] An import directly followed by a code line (no blank line). Removal deletes
    # just the import; the code line is untouched.
    """Only a blank line after the import is absorbed; a code line stays untouched."""
    (tmp_path / "m.py").write_text("import os\nx = 1\n")
    r = Repro("b", "t").remove_import("m.py", "import os")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "x = 1\n"


def test_remove_import_asserts_when_text_absent(tmp_path: Path) -> None:
    # [FUNCTION DOC] The requested import text matches nothing in the file; applying the
    # primitive raises an assertion error.
    """Removing an import text that matches nothing fails loudly."""
    (tmp_path / "m.py").write_text("import os\n")
    r = Repro("b", "t").remove_import("m.py", "from pkg import Q")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_remove_import_asserts_when_scope_function_missing(tmp_path: Path) -> None:
    # [FUNCTION DOC] The scope names a function that does not exist; applying the primitive
    # raises an assertion error.
    """Scoping to a function that does not exist fails loudly."""
    (tmp_path / "m.py").write_text("def f():\n    import os\n")
    r = Repro("b", "t").remove_import("m.py", "import os", in_function="nope")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_remove_import_leaves_other_statements_on_a_semicolon_line(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Two imports joined by a semicolon on one line, followed by a use. Removing
    # the first leaves the second import and the following statement in the file.
    """Removing 'import os' from a semicolon-joined line must keep 'import sys'."""
    (tmp_path / "m.py").write_text("import os; import sys\nprint(sys.path)\n")
    r = Repro("b", "t").remove_import("m.py", "import os")
    _apply(r, tmp_path)
    out = (tmp_path / "m.py").read_text()
    assert "import sys" in out and "print(sys.path)" in out


def test_remove_import_trailing_on_a_semicolon_line_leaves_no_dangling_separator(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Same layout with the import to remove being the second (trailing) one on the
    # line. After removal the file has no semicolon at all, no trace of the removed import, and
    # still has the first import and the following statement.
    """Removing the trailing import on a semicolon-joined line drops the dangling ';' too
    (a trailing space may remain for the formatter to strip, but the separator is gone).
    """
    (tmp_path / "m.py").write_text("import sys; import os\nprint(sys.path)\n")
    r = Repro("b", "t").remove_import("m.py", "import os")
    _apply(r, tmp_path)
    out = (tmp_path / "m.py").read_text()
    assert ";" not in out
    assert "import os" not in out
    assert "import sys" in out and "print(sys.path)" in out


def test_remove_import_does_not_overmatch_a_submodule_import(tmp_path: Path) -> None:
    # [FUNCTION DOC] The file imports a package and, separately, one of its submodules by dotted
    # name. Removing the package import text leaves the submodule import line in place.
    """Removing 'import os' must not also remove 'import os.path'."""
    (tmp_path / "m.py").write_text("import os\nimport os.path\nprint(os.path.sep)\n")
    r = Repro("b", "t").remove_import("m.py", "import os")
    _apply(r, tmp_path)
    assert "import os.path\n" in (tmp_path / "m.py").read_text()
