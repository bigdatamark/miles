# [MODULE DOC] Tests for the module-constant moving primitive of the reproduction builder:
# default placement after the destination's imports, placement above a named top-level
# statement, annotated constants, and the error when the constant is absent. Each test writes a
# source and a destination file in a temporary directory, applies the primitive and compares
# the resulting text.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_reproduction_utils import Repro
from reproduction_testlib import _apply  # noqa: F401


def test_move_assign_relocates_a_module_constant(tmp_path: Path) -> None:
    # [FUNCTION DOC] Source: an import, a constant with a trailing comment, and a function using
    # it; destination: an import and a function. Moves the constant with no anchor. Asserts the
    # constant no longer appears in the part of the source before the function, and the
    # destination text is its import, a blank line, the constant line with its comment, two blank
    # lines and the function.
    """The assignment is cut verbatim from the source and lands after the destination's imports."""
    (tmp_path / "src.py").write_text("import os\n\nLIMIT = 480  # seconds\n\n\ndef stay():\n    return LIMIT\n")
    (tmp_path / "dst.py").write_text("import sys\n\n\ndef keep():\n    return 1\n")
    r = Repro("b", "t").move_assign("LIMIT", src="src.py", dst="dst.py")
    _apply(r, tmp_path)
    assert "LIMIT" not in (tmp_path / "src.py").read_text().split("def stay")[0]
    assert (tmp_path / "dst.py").read_text() == (
        "import sys\n" "\n" "LIMIT = 480  # seconds\n" "\n" "\n" "def keep():\n" "    return 1\n"
    )


def test_move_assign_pastes_above_the_named_sibling(tmp_path: Path) -> None:
    # [FUNCTION DOC] Source is a single constant assignment; destination is a single function.
    # Moving the constant with the before option naming that function gives a destination of the
    # constant line, one blank line, then the function.
    """With before=, the constant lands immediately above the named top-level statement."""
    (tmp_path / "src.py").write_text("RATIO = 3\n")
    (tmp_path / "dst.py").write_text("def first():\n    return 1\n")
    r = Repro("b", "t").move_assign("RATIO", src="src.py", dst="dst.py", before="first")
    _apply(r, tmp_path)
    assert (tmp_path / "dst.py").read_text() == ("RATIO = 3\n\ndef first():\n    return 1\n")


def test_move_assign_relocates_an_annotated_constant(tmp_path: Path) -> None:
    # [FUNCTION DOC] Same layout as the first test but the constant has a type annotation. The
    # destination ends up with the annotated line kept intact after its import, and the constant
    # is gone from the head of the source.
    """An annotated module constant (AnnAssign) is cut verbatim with its annotation intact."""
    (tmp_path / "src.py").write_text("import os\n\nLIMIT: int = 480\n\n\ndef stay():\n    return LIMIT\n")
    (tmp_path / "dst.py").write_text("import sys\n\n\ndef keep():\n    return 1\n")
    r = Repro("b", "t").move_assign("LIMIT", src="src.py", dst="dst.py")
    _apply(r, tmp_path)
    assert "LIMIT" not in (tmp_path / "src.py").read_text().split("def stay")[0]
    assert (tmp_path / "dst.py").read_text() == (
        "import sys\n" "\n" "LIMIT: int = 480\n" "\n" "\n" "def keep():\n" "    return 1\n"
    )


def test_move_assign_missing_source_raises(tmp_path: Path) -> None:
    # [FUNCTION DOC] The requested name is not assigned in the source. Applying the primitive
    # raises an assertion error.
    """A name with no module-level assignment in the source fails loudly."""
    (tmp_path / "src.py").write_text("x = 1\n")
    (tmp_path / "dst.py").write_text("import os\n")
    r = Repro("b", "t").move_assign("MISSING", src="src.py", dst="dst.py")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)
