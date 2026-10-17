# [MODULE DOC] Tests for the tail-extraction primitive of the reproduction builder, which cuts
# the trailing block of a source file (scaffolding plus the named definitions) into a new file:
# the exact resulting texts with the future import, a trailing class with a decorator, a
# requested symbol that is not in the tail, and a trailing main-guard that blocks the cut. Each
# test writes the source in a temporary directory and checks the resulting files.
# KNOWN ISSUE: two stray section heading comments (one naming another primitive and one naming
# an adversarial audit) sit before the later tests.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_reproduction_utils import Repro
from reproduction_testlib import _apply, _commit, _git, _write  # noqa: F401

# --- extract_to_new_module -----------------------------------------------------


def test_extract_to_new_module_cuts_trailing_block(tmp_path: Path) -> None:
    # [FUNCTION DOC] Source: a class, then a logging import, a logger assignment and a function
    # foo. Extracting foo with the future import on. Asserts the source is reduced to the class
    # text plus its two trailing blank lines, and the new file is exactly the future import line,
    # the logging import, a blank line, the logger assignment, two blank lines and foo.
    """Cuts the trailing scaffolding+def block into a new file, prepending the future import."""
    (tmp_path / "src.py").write_text(
        "class M:\n"
        "    def keep(self):\n"
        "        return 1\n"
        "\n"
        "\n"
        "import logging\n"
        "\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "\n"
        "def foo(x):\n"
        "    return x + 1\n"
    )
    r = Repro("b", "t").extract_to_new_module("src.py", "new.py", symbols=["foo"], future_import=True)
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == ("class M:\n    def keep(self):\n        return 1\n\n\n")
    assert (tmp_path / "new.py").read_text() == (
        "from __future__ import annotations\n"
        "import logging\n"
        "\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "\n"
        "def foo(x):\n"
        "    return x + 1\n"
    )


def test_extract_to_new_module_carries_a_trailing_class(tmp_path: Path) -> None:
    # [FUNCTION DOC] Source: a class, then a dataclass import, a decorated dataclass Cfg and a
    # function foo. Extracting both symbols without the future import. Asserts the source becomes
    # just the first class text with two trailing blank lines, and the new file contains both the
    # Cfg class and foo.
    """A class in the staged tail (not just a def) travels with the cut block."""
    (tmp_path / "src.py").write_text(
        "class M:\n"
        "    pass\n"
        "\n"
        "\n"
        "from dataclasses import dataclass\n"
        "\n"
        "\n"
        "@dataclass\n"
        "class Cfg:\n"
        "    x: int\n"
        "\n"
        "\n"
        "def foo():\n"
        "    return Cfg(1)\n"
    )
    r = Repro("b", "t").extract_to_new_module("src.py", "new.py", symbols=["Cfg", "foo"], future_import=False)
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == "class M:\n    pass\n\n\n"
    assert "class Cfg:" in (tmp_path / "new.py").read_text()
    assert "def foo():" in (tmp_path / "new.py").read_text()


# --- extract_symbols_to_new_module ---------------------------------------------


# --- adversarial audit: module extraction ----------------------------------------


def test_extract_to_new_module_asserts_when_symbol_not_in_the_tail(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] The wanted function is followed by a print statement (not scaffolding), so
    # the backward scan stops before reaching it. Applying the primitive raises an assertion error.
    """A wanted symbol above a non-scaffolding statement is not in the tail and raises."""
    (tmp_path / "src.py").write_text("def wanted():\n    return 1\n\n\nprint('side effect')\n")
    r = Repro("b", "t").extract_to_new_module("src.py", "n.py", symbols=["wanted"])
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_extract_to_new_module_refuses_a_trailing_main_guard(tmp_path: Path) -> None:
    # [FUNCTION DOC] The wanted function is followed by a conditional on the module name equal to
    # the main marker that calls it. That conditional is not the type-checking flag, so it stops
    # the scan. Applying the primitive raises an assertion error, and the source file still
    # contains the main-guard text (unchanged).
    """A trailing __main__ guard is executable code, not scaffolding: the tail cut raises."""
    (tmp_path / "src.py").write_text(
        "class Keep:\n"
        "    pass\n"
        "\n"
        "\n"
        "def foo():\n"
        "    return 1\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    foo()\n"
    )
    r = Repro("b", "t").extract_to_new_module("src.py", "new.py", symbols=["foo"], future_import=False)
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)
    assert "__main__" in (tmp_path / "src.py").read_text()
