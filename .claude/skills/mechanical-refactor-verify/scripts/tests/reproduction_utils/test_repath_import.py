# [MODULE DOC] Tests for the import-repathing primitive of the reproduction builder: a nested
# (function-scoped) from-import has its module replaced in place, a module-level import is not
# touched (the primitive then raises because nothing changed), multi-line aliased imports are
# handled on their first line, and relative nested imports are matched by module name and
# rewritten to the absolute new module. Each test writes a small module in a temporary directory
# and applies the primitive.
# KNOWN ISSUE: the section heading comment names the type-checking import primitive as well, but
# no tests of it are in this file.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_reproduction_utils import Repro
from reproduction_testlib import _apply, _commit, _git, _write  # noqa: F401

# --- repath_import / add_typechecking_import -----------------------------------


def test_repath_import_rewrites_nested_import(tmp_path: Path) -> None:
    # [FUNCTION DOC] A method has a local import from the old module followed by a call. After
    # repathing, only the module in the import line changed; the call is untouched.
    """A function-scoped import is repathed in place; the bare call is untouched."""
    (tmp_path / "c.py").write_text(
        "class K:\n" "    def run(self):\n" "        from old.mod import foo\n" "\n" "        return foo(1)\n"
    )
    r = Repro("b", "t").repath_import("c.py", old_module="old.mod", new_module="new.mod", name="foo")
    _apply(r, tmp_path)
    assert (tmp_path / "c.py").read_text() == (
        "class K:\n" "    def run(self):\n" "        from new.mod import foo\n" "\n" "        return foo(1)\n"
    )


def test_repath_import_leaves_a_module_level_import(tmp_path: Path) -> None:
    # [FUNCTION DOC] The only matching import is at module level, so nothing is nested to
    # change; applying the primitive raises an assertion error (that error, not an unchanged
    # file, is what the test checks).
    """Only nested imports are repathed; a module-level import is left to the sorter."""
    (tmp_path / "c.py").write_text("from old.mod import foo\n\n\nx = foo(1)\n")
    r = Repro("b", "t").repath_import("c.py", old_module="old.mod", new_module="new.mod", name="foo")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_repath_import_repaths_a_multiline_aliased_nested_import(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A nested parenthesized import with the name aliased; only the first line's
    # module changes, the name line with its alias and the closing parenthesis are unchanged.
    """A nested multi-line from-import with an alias is repathed on its first line."""
    (tmp_path / "c.py").write_text(
        "def run():\n" "    from old.mod import (\n" "        foo as f,\n" "    )\n" "\n" "    return f(1)\n"
    )
    r = Repro("b", "t").repath_import("c.py", old_module="old.mod", new_module="new.mod", name="foo")
    _apply(r, tmp_path)
    assert (tmp_path / "c.py").read_text() == (
        "def run():\n" "    from new.mod import (\n" "        foo as f,\n" "    )\n" "\n" "    return f(1)\n"
    )


def test_repath_import_rewrites_a_relative_nested_import(tmp_path: Path) -> None:
    # [FUNCTION DOC] A nested relative import (one leading dot, module mod). Repathing from old
    # module mod to new module pkg.mod rewrites it as an absolute import from pkg.mod.
    """A nested `from .mod import` matched by module name must actually be repathed."""
    (tmp_path / "c.py").write_text("def run():\n    from .mod import foo\n\n    return foo(1)\n")
    r = Repro("b", "t").repath_import("c.py", old_module="mod", new_module="pkg.mod", name="foo")
    _apply(r, tmp_path)
    assert "from pkg.mod import foo" in (tmp_path / "c.py").read_text()
