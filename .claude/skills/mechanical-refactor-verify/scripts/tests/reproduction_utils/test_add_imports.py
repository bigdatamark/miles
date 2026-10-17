# [MODULE DOC] Tests for the import-adding primitives of the reproduction builder: adding a
# whole import statement (default position, after a named import, into an empty file, below a
# docstring), adding a name to an existing from-import (with alias, refusals), and adding an
# import inside the type-checking-only block (existing block, qualified flag form, multi-line
# imports, creating the block, placeholder pass handling, refusal without imports). Every test
# writes a small module in a temporary directory, records the primitive, applies the recorded
# operations to that directory and compares the file text.
# KNOWN ISSUE: the section heading comment above the type-checking tests also names a repath
# primitive that is tested in another module, and some tests of the same primitive appear
# outside their section.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_reproduction_utils import Repro
from reproduction_testlib import _apply, _commit, _git, _write  # noqa: F401

# --- add_import ----------------------------------------------------------------


def test_add_import_appends_after_last_top_level_import(tmp_path: Path) -> None:
    # [FUNCTION DOC] File with two imports, a blank line and an assignment. Adding a from-import
    # statement puts it on the line right after the second import, before the blank line.
    """A new import is inserted right after the last module-level import."""
    (tmp_path / "m.py").write_text("import os\nimport sys\n\nx = 1\n")
    r = Repro("b", "t").add_import("m.py", "from pkg import Thing")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "import os\nimport sys\nfrom pkg import Thing\n\nx = 1\n"


# --- add_imported_name ---------------------------------------------------------


def test_add_imported_name_extends_a_single_line_import(tmp_path: Path) -> None:
    # [FUNCTION DOC] File importing one name from a package. Adding a second name from the same
    # package rewrites the statement as a single line listing both names; the rest is unchanged.
    """A new name is appended to an existing from-import on the same statement."""
    (tmp_path / "m.py").write_text("from pkg import a\n\nx = 1\n")
    r = Repro("b", "t").add_imported_name("m.py", module="pkg", name="b")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "from pkg import a, b\n\nx = 1\n"


def test_add_imported_name_carries_an_asname(tmp_path: Path) -> None:
    # [FUNCTION DOC] Adds a name with an alias to an existing from-import; the result lists the
    # old name then the new name followed by "as" and the alias.
    """The added name keeps its `as` alias."""
    (tmp_path / "m.py").write_text("from pkg import a\n")
    r = Repro("b", "t").add_imported_name("m.py", module="pkg", name="b", asname="c")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "from pkg import a, b as c\n"


def test_add_imported_name_refuses_a_commented_import(tmp_path: Path) -> None:
    # [FUNCTION DOC] The existing multi-line from-import contains a comment. Applying the
    # operation raises an assertion error.
    """An import carrying comments is refused, since a rebuild would drop them."""
    (tmp_path / "m.py").write_text("from pkg import (\n    a,  # keep\n)\n")
    r = Repro("b", "t").add_imported_name("m.py", module="pkg", name="b")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_add_imported_name_rejects_a_name_already_present(tmp_path: Path) -> None:
    # [FUNCTION DOC] The from-import already lists the name being added. Applying the operation
    # raises an assertion error.
    """Adding a name the import already has fails loudly."""
    (tmp_path / "m.py").write_text("from pkg import a, b\n")
    r = Repro("b", "t").add_imported_name("m.py", module="pkg", name="b")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_add_imported_name_raises_without_a_matching_import(tmp_path: Path) -> None:
    # [FUNCTION DOC] The file only imports from a different module than the requested one.
    # Applying the operation raises an assertion error.
    """A file lacking a `from module import` for the module fails loudly."""
    (tmp_path / "m.py").write_text("from other import a\n")
    r = Repro("b", "t").add_imported_name("m.py", module="pkg", name="b")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


# --- repath_import / add_typechecking_import -----------------------------------


def test_add_typechecking_import_inserts_in_block(tmp_path: Path) -> None:
    # [FUNCTION DOC] File with an existing type-checking block holding one import, followed by a
    # function. The new import line appears right after the existing one, indented the same, and
    # nothing else changes.
    """The import is appended inside the existing TYPE_CHECKING block."""
    (tmp_path / "m.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from a import X\n"
        "\n"
        "\n"
        "def f():\n"
        "    pass\n"
    )
    r = Repro("b", "t").add_typechecking_import("m.py", "from b import Y")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == (
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from a import X\n"
        "    from b import Y\n"
        "\n"
        "\n"
        "def f():\n"
        "    pass\n"
    )


def test_add_typechecking_import_creates_missing_block(tmp_path: Path) -> None:
    # [FUNCTION DOC] File importing the flag and one module but with no block. The result has a
    # blank line, a new conditional on the flag and the indented import, inserted right after the
    # last import, ahead of the existing blank lines and function.
    """With no TYPE_CHECKING block, one is created after the trailing module import."""
    (tmp_path / "m.py").write_text(
        "from typing import TYPE_CHECKING\n" "\n" "from a import X\n" "\n" "\n" "def f():\n" "    pass\n"
    )
    r = Repro("b", "t").add_typechecking_import("m.py", "from b import Y")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == (
        "from typing import TYPE_CHECKING\n"
        "\n"
        "from a import X\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from b import Y\n"
        "\n"
        "\n"
        "def f():\n"
        "    pass\n"
    )


def test_add_import_into_an_empty_file(tmp_path: Path) -> None:
    # [FUNCTION DOC] Empty file; adding a plain import produces a file holding just that
    # statement and a trailing newline.
    """Adding an import to an empty file writes just the statement."""
    (tmp_path / "m.py").write_text("")
    r = Repro("b", "t").add_import("m.py", "import os")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "import os\n"


def test_add_import_lands_below_a_module_docstring(tmp_path: Path) -> None:
    # [FUNCTION DOC] File with a docstring, a blank line and an assignment but no imports. After
    # adding an import, the file text still starts with the docstring.
    """In a file with only a docstring, the new import must land below the docstring."""
    (tmp_path / "m.py").write_text('"""Module doc."""\n\nx = 1\n')
    r = Repro("b", "t").add_import("m.py", "import os")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text().startswith('"""Module doc."""')


def test_add_typechecking_import_matches_qualified_typing_form(tmp_path: Path) -> None:
    # [FUNCTION DOC] File whose block is conditioned on the flag qualified by the typing module
    # (module imported as a whole). The new import is appended inside that block, with exact
    # expected text compared.
    """A `if typing.TYPE_CHECKING:` block is recognized and receives the import."""
    (tmp_path / "m.py").write_text(
        "import typing\n" "\n" "if typing.TYPE_CHECKING:\n" "    from a import X\n" "\n" "\n" "def f():\n" "    pass\n"
    )
    r = Repro("b", "t").add_typechecking_import("m.py", "from b import Y")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == (
        "import typing\n"
        "\n"
        "if typing.TYPE_CHECKING:\n"
        "    from a import X\n"
        "    from b import Y\n"
        "\n"
        "\n"
        "def f():\n"
        "    pass\n"
    )


def test_add_typechecking_import_after_a_multiline_final_import(tmp_path: Path) -> None:
    # [FUNCTION DOC] The block's only import is a parenthesized multi-line import. The new
    # import line is inserted after the closing parenthesis line, not in the middle.
    """The insert lands after the closing paren of a multi-line final guarded import."""
    (tmp_path / "m.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from a import (\n"
        "        X,\n"
        "    )\n"
        "\n"
        "x = 1\n"
    )
    r = Repro("b", "t").add_typechecking_import("m.py", "from b import Y")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == (
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from a import (\n"
        "        X,\n"
        "    )\n"
        "    from b import Y\n"
        "\n"
        "x = 1\n"
    )


def test_add_typechecking_import_raises_without_imports(tmp_path: Path) -> None:
    # [FUNCTION DOC] File with only an assignment: no block and no import to anchor a new one.
    # Applying the operation raises an assertion error.
    """A file with no imports cannot anchor a new TYPE_CHECKING block and fails loudly."""
    (tmp_path / "m.py").write_text("x = 1\n")
    r = Repro("b", "t").add_typechecking_import("m.py", "from b import Y")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_add_typechecking_import_drops_a_lone_pass_placeholder(tmp_path: Path) -> None:
    # [FUNCTION DOC] The block contains only a pass statement. After the operation the pass line
    # is replaced by the indented import and the rest is unchanged.
    """Populating a `pass`-only TYPE_CHECKING block replaces the placeholder."""
    (tmp_path / "m.py").write_text(
        "from typing import TYPE_CHECKING\n" "\n" "if TYPE_CHECKING:\n" "    pass\n" "\n" "x = 1\n"
    )
    r = Repro("b", "t").add_typechecking_import("m.py", "from b import Y")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == (
        "from typing import TYPE_CHECKING\n" "\n" "if TYPE_CHECKING:\n" "    from b import Y\n" "\n" "x = 1\n"
    )


def test_add_typechecking_import_keeps_a_pass_that_is_not_alone(tmp_path: Path) -> None:
    # [FUNCTION DOC] The block contains an import followed by a pass. The new import is appended
    # after the pass (the last statement) and the pass is kept.
    """A `pass` beside a real import is left untouched; only the new import is appended."""
    (tmp_path / "m.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from a import X\n"
        "    pass\n"
        "\n"
        "x = 1\n"
    )
    r = Repro("b", "t").add_typechecking_import("m.py", "from b import Y")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == (
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from a import X\n"
        "    pass\n"
        "    from b import Y\n"
        "\n"
        "x = 1\n"
    )


# --- add_import(after=...) -----------------------------------------------------


def test_add_import_after_anchors_into_a_split_import_block(tmp_path: Path) -> None:
    # [FUNCTION DOC] File with an import, an assignment that splits the import sections, a second
    # import section and an assignment. Adding an import with the after option naming the first
    # import places it on the line directly after that import (before the splitting assignment),
    # not after the last import.
    """With `after`, the import lands right after the named import -- needed when a
    statement splits the imports into separate blocks and the default (after the last
    import) would land in the wrong block."""
    (tmp_path / "m.py").write_text("import os\n\n_flag = os.getpid()\n\nfrom pkg import a\n\nx = 1\n")
    r = Repro("b", "t").add_import("m.py", "from new import Thing", after="import os")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == (
        "import os\nfrom new import Thing\n\n_flag = os.getpid()\n\nfrom pkg import a\n\nx = 1\n"
    )


def test_add_import_after_raises_when_anchor_absent(tmp_path: Path) -> None:
    # [FUNCTION DOC] The after text matches no top-level import in the file. Applying the
    # operation raises an assertion error.
    """An `after` substring that matches no top-level import raises."""
    (tmp_path / "m.py").write_text("import os\n\nx = 1\n")
    r = Repro("b", "t").add_import("m.py", "from new import Thing", after="import nope")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)
