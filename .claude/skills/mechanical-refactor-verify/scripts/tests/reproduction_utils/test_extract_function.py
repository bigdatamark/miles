# [MODULE DOC] Tests for the inline-block function extraction primitive of the reproduction
# builder: relocating a block verbatim under an authored signature with re-indentation and a
# replacement call, insertion before a named sibling, refusal when the block is not unique or
# only matches mid-line, blank-line and string-literal handling during re-indentation, and
# insertion into a class at method depth. Each test writes source and destination files in a
# temporary directory, applies the primitive and inspects the files.
# KNOWN ISSUE: the file contains two stray, consecutive section heading comments, one of which
# names module extraction, which is not tested here.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_reproduction_utils import Repro
from reproduction_testlib import _apply, _commit, _git, _write  # noqa: F401

# --- extract_function ----------------------------------------------------------


def test_extract_function_relocates_body_and_replaces_with_call(tmp_path: Path) -> None:
    # [FUNCTION DOC] Source: a class method summing squares with a three-line block at indent
    # eight; destination: a file with one function. Extracts the block as a function with a
    # one-parameter signature, a replacement call line and a return line. Asserts: the call line
    # is in the source and the loop text is gone; the destination contains the signature followed
    # by the block re-indented to four spaces and then the return line.
    """An inline block is cut verbatim, re-indented under the new signature, and the call site
    replaced; the body lands at function-body indent."""
    (tmp_path / "src.py").write_text(
        "class Q:\n"
        "    def run(self, n):\n"
        "        total = 0\n"
        "        for i in range(n):\n"
        "            total += i * i\n"
        "        return total\n"
    )
    (tmp_path / "dst.py").write_text("def existing():\n    return 0\n")
    body = "        total = 0\n        for i in range(n):\n            total += i * i\n"
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="sum_squares",
        signature="def sum_squares(n):",
        body=body,
        body_indent=8,
        call="        total = sum_squares(n)\n",
        return_text="    return total\n",
    )
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    assert "        total = sum_squares(n)\n" in src_out
    assert "for i in range(n)" not in src_out
    assert (
        "def sum_squares(n):\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        "        total += i * i\n"
        "    return total\n"
    ) in (tmp_path / "dst.py").read_text()


def test_extract_function_inserts_before_named_sibling(tmp_path: Path) -> None:
    # [FUNCTION DOC] Source is a single assignment line; destination has functions a and c.
    # Extracting with the before option set to c: the new function b appears between a and c in
    # the destination (by text position), and the source file becomes exactly the call line.
    """With before=, the new function lands immediately above that sibling at module level."""
    (tmp_path / "src.py").write_text("x = compute()\n")
    (tmp_path / "dst.py").write_text("def a():\n    return 1\n\n\ndef c():\n    return 3\n")
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="b",
        signature="def b():",
        body="x = compute()\n",
        body_indent=0,
        call="x = b()\n",
        return_text="    return x\n",
        before="c",
    )
    _apply(r, tmp_path)
    dst_out = (tmp_path / "dst.py").read_text()
    assert dst_out.index("def a") < dst_out.index("def b") < dst_out.index("def c")
    assert "x = b()\n" == (tmp_path / "src.py").read_text()


def test_extract_function_asserts_block_not_unique(tmp_path: Path) -> None:
    # [FUNCTION DOC] The source repeats the same line twice, so the block occurs twice. Applying
    # the primitive raises an assertion error.
    """A block that occurs more than once in the source raises, so the cut is unambiguous."""
    (tmp_path / "src.py").write_text("p = f()\np = f()\n")
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="g",
        signature="def g():",
        body="p = f()\n",
        body_indent=0,
        call="p = g()\n",
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


# --- adversarial audit: module extraction ----------------------------------------


# --- adversarial audit: extract_function -----------------------------------------


def test_extract_function_does_not_pad_blank_lines_in_the_body(tmp_path: Path) -> None:
    # [FUNCTION DOC] The source is two statements at indent eight separated by an empty line,
    # extracted to a function in a destination with one function. Asserts the source becomes just
    # the call line, and the destination text equals the old function, a blank line, then the new
    # function whose body is dedented to four spaces with the empty line kept empty (no padding).
    """Interior blank lines of the extracted body stay bare newlines, unpadded."""
    (tmp_path / "src.py").write_text("        a = 1\n\n        b = 2\n")
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="g",
        signature="def g():",
        body="        a = 1\n\n        b = 2\n",
        body_indent=8,
        call="        g()\n",
    )
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == "        g()\n"
    assert (tmp_path / "dst.py").read_text() == ("def z():\n    return 0\n\ndef g():\n    a = 1\n\n    b = 2\n")


def test_extract_function_does_not_reindent_string_literal_interiors(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] The block contains a triple-quoted string spanning lines. After extraction
    # at indent zero (shifted by four), the destination still contains the string's interior
    # line with no added indentation.
    """Triple-quoted string interior lines keep their exact bytes through the extraction."""
    (tmp_path / "src.py").write_text("TEMPLATE = '''\nliteral line\n'''\nx = TEMPLATE\n")
    (tmp_path / "dst.py").write_text("def existing():\n    return 0\n")
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="make",
        signature="def make():",
        body="TEMPLATE = '''\nliteral line\n'''\nx = TEMPLATE\n",
        body_indent=0,
        call="x = make()\n",
        return_text="    return x\n",
    )
    _apply(r, tmp_path)
    assert "\nliteral line\n" in (tmp_path / "dst.py").read_text()


def test_extract_function_rejects_a_mid_line_substring_match(tmp_path: Path) -> None:
    # [FUNCTION DOC] The block text appears only as the tail of a longer line. Applying the
    # primitive raises an assertion error because the match does not start at a line boundary.
    """A body that only matches mid-line must fail loudly instead of splicing the call."""
    (tmp_path / "src.py").write_text("value = prefix_total = 0\n")
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="g",
        signature="def g():",
        body="total = 0\n",
        body_indent=0,
        call="total = g()\n",
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_extract_function_into_class_indents_body_to_method_depth(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Extracts a single unindented statement into class H of the destination using
    # a method signature indented four spaces and a return line indented eight. Asserts the
    # resulting destination compiles and contains the method header followed by the statement
    # indented eight spaces.
    """Extracting into a class must indent the relocated body to method depth."""
    (tmp_path / "src.py").write_text("val = compute_thing()\n")
    (tmp_path / "dst.py").write_text("class H:\n    def last(self):\n        return 0\n")
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="helper",
        signature="    def helper(self):",
        body="val = compute_thing()\n",
        body_indent=0,
        call="val = h.helper()\n",
        return_text="        return val\n",
        into_class="H",
    )
    _apply(r, tmp_path)
    out = (tmp_path / "dst.py").read_text()
    compile(out, "dst.py", "exec")
    assert "    def helper(self):\n        val = compute_thing()\n" in out
