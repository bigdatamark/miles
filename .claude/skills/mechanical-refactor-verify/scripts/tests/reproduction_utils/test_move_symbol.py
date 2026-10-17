# [MODULE DOC] Tests for the symbol-moving primitive of the reproduction builder (without the
# delegate-stub option, which has its own test module): dropping the self annotation, shedding
# static and class method decorators, placement (end of class, module level, before or after a
# named sibling, below an assignment ahead of a type-checking guard), dedent in both directions,
# same-file moves, async definitions, whole classes, byte preservation (comments left behind,
# missing final newlines, string interiors, CRLF), and refusals (missing destination class,
# ambiguous names, conflicting or missing anchors). Each test writes a source and a destination
# file in a temporary directory, records the primitive, applies it and inspects the files.
# KNOWN ISSUE: several stray section heading comments in this file name other primitives
# (import primitives, delegate stubs) and appear out of place.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_reproduction_utils import Repro
from reproduction_testlib import _apply, _commit, _git, _write  # noqa: F401


def test_move_symbol_drops_self_annotation_into_class(tmp_path: Path) -> None:
    # [FUNCTION DOC] A static method whose first parameter is an annotated self moves into the
    # annotation's class with the drop-annotation option. The destination has the method with a
    # plain self parameter and no annotation text.
    """Moving a `def foo(self: Target)` into Target drops the now-redundant annotation."""
    (tmp_path / "src.py").write_text(
        "class M:\n" "    @staticmethod\n" "    def foo(self: Target, x):\n" "        return self.y + x\n"
    )
    (tmp_path / "dst.py").write_text("class Target:\n    def keep(self):\n        return 1\n")
    r = Repro("b", "t").move_symbol(
        "foo",
        src="src.py",
        dst="dst.py",
        into_class="Target",
        dedent=0,
        drop_self_annotation=True,
    )
    _apply(r, tmp_path)
    text = (tmp_path / "dst.py").read_text()
    assert "def foo(self, x):" in text
    assert "self: Target" not in text


# --- adversarial audit: import primitives ----------------------------------------


# --- move_symbol ---------------------------------------------------------------


def test_move_symbol_into_class_drops_decorator_and_appends(tmp_path: Path) -> None:
    # [FUNCTION DOC] A static method moves from one class into another. Asserts the source no
    # longer has it but keeps its other method; the destination has no static decorator, has
    # the new method after the existing one, and has the body line unchanged.
    """The def leaves the source, its @staticmethod is dropped, and it lands at the end of
    the destination class with its body verbatim."""
    (tmp_path / "src.py").write_text(
        "class Old:\n"
        "    @staticmethod\n"
        "    def foo(x):\n"
        "        return x + 1\n"
        "\n"
        "    def keep(self):\n"
        "        return 0\n"
    )
    (tmp_path / "dst.py").write_text("class New:\n    def existing(self):\n        return 1\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    dst_out = (tmp_path / "dst.py").read_text()
    assert "def foo" not in src_out and "def keep" in src_out
    assert "@staticmethod" not in dst_out
    assert dst_out.index("def existing") < dst_out.index("def foo")
    assert "        return x + 1\n" in dst_out


def test_move_symbol_to_module_level_with_dedent(tmp_path: Path) -> None:
    # [FUNCTION DOC] A static method moves to module level with no destination class and a
    # dedent of four. The destination contains the function at column zero with its body at four
    # spaces, and the source no longer defines it.
    """With into_class=None and a dedent, the def lands at module level, dedented."""
    (tmp_path / "src.py").write_text("class Old:\n    @staticmethod\n    def helper(x):\n        return x * 2\n")
    (tmp_path / "dst.py").write_text("import os\n")
    r = Repro("b", "t").move_symbol("helper", src="src.py", dst="dst.py", into_class=None, dedent=4)
    _apply(r, tmp_path)
    assert "def helper(x):\n    return x * 2\n" in (tmp_path / "dst.py").read_text()
    assert "def helper" not in (tmp_path / "src.py").read_text()


def test_move_symbol_before_inserts_above_named_sibling(tmp_path: Path) -> None:
    # [FUNCTION DOC] Destination class has methods first and last; moving a method with the
    # before option naming the last one places it between them (checked by text positions).
    """With before=, the relocated def lands immediately above that sibling, not at the end."""
    (tmp_path / "src.py").write_text("class Old:\n    @staticmethod\n    def moved(self):\n        return 1\n")
    (tmp_path / "dst.py").write_text(
        "class New:\n" "    def first(self):\n        return 0\n" "\n" "    def last(self):\n        return 2\n"
    )
    r = Repro("b", "t").move_symbol("moved", src="src.py", dst="dst.py", into_class="New", before="last")
    _apply(r, tmp_path)
    dst_out = (tmp_path / "dst.py").read_text()
    assert dst_out.index("def first") < dst_out.index("def moved") < dst_out.index("def last")


# --- adversarial audit: move_symbol edge cases -----------------------------------


def test_move_symbol_moves_an_async_def_verbatim(tmp_path: Path) -> None:
    # [FUNCTION DOC] An async method moves into another class. Asserts the exact source text
    # (the class header, a blank line left where the method was, and the remaining method) and
    # the exact destination text (existing method, a blank line, the async method unchanged).
    """An async def relocates with its `async` keyword and body byte-identical."""
    (tmp_path / "src.py").write_text(
        "class Old:\n"
        "    async def foo(self):\n"
        "        return 1\n"
        "\n"
        "    def keep(self):\n"
        "        return 0\n"
    )
    (tmp_path / "dst.py").write_text("class New:\n    def e(self):\n        return 1\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == ("class Old:\n\n    def keep(self):\n        return 0\n")
    assert (tmp_path / "dst.py").read_text() == (
        "class New:\n" "    def e(self):\n" "        return 1\n" "\n" "    async def foo(self):\n" "        return 1\n"
    )


def test_move_symbol_moves_a_def_within_the_same_file(tmp_path: Path) -> None:
    # [FUNCTION DOC] Two functions a and b in one file; moving b above a with the same file as
    # source and destination yields b, one blank line, a, then the remaining blank lines.
    """With src == dst the def is cut and re-inserted above its sibling in one file."""
    (tmp_path / "m.py").write_text("def a():\n    return 1\n\n\ndef b():\n    return 2\n")
    r = Repro("b", "t").move_symbol("b", src="m.py", dst="m.py", into_class=None, before="a")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == ("def b():\n    return 2\n\ndef a():\n    return 1\n\n\n")


def test_move_symbol_prefers_a_module_level_def_over_an_earlier_class_method(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] The source has a class method and a module-level function with the same
    # name, the method first. Moving that name to module level with no source class picks the
    # module-level function: the source keeps the class (with trailing blank lines) and the
    # destination gains the module-level function.
    """When a class method and a module-level def share a name, the module-level def moves."""
    (tmp_path / "src.py").write_text(
        "class C:\n"
        "    def foo(self):\n"
        "        return 'method'\n"
        "\n"
        "\n"
        "def foo():\n"
        "    return 'module'\n"
    )
    (tmp_path / "dst.py").write_text("x = 1\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class=None)
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == ("class C:\n    def foo(self):\n        return 'method'\n\n\n")
    assert (tmp_path / "dst.py").read_text() == ("x = 1\n\ndef foo():\n    return 'module'\n")


def test_move_symbol_keeps_a_real_decorator_while_dropping_classmethod(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A method has the class method decorator and a caching decorator. Moved to
    # module level with dedent four, the destination ends with the function preceded only by the
    # caching decorator (the class method decorator is shed), dedented.
    """@classmethod is shed on the move but any other decorator travels verbatim."""
    (tmp_path / "src.py").write_text(
        "import functools\n"
        "\n"
        "\n"
        "class Old:\n"
        "    @classmethod\n"
        "    @functools.lru_cache(maxsize=None)\n"
        "    def foo(cls, x):\n"
        "        return x + 1\n"
    )
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class=None, dedent=4)
    _apply(r, tmp_path)
    assert (tmp_path / "dst.py").read_text() == (
        "def z():\n"
        "    return 0\n"
        "\n"
        "@functools.lru_cache(maxsize=None)\n"
        "def foo(cls, x):\n"
        "    return x + 1\n"
    )


def test_move_symbol_leaves_a_comment_above_the_def_in_the_source(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A comment line sits directly above the function being moved. After the move
    # the source keeps the comment (followed by the blank lines and the other function), and the
    # destination gets only the function.
    """A comment above the def is not part of its span, so it stays behind in the source."""
    (tmp_path / "src.py").write_text("# explains foo\ndef foo():\n    return 1\n\n\ndef keep():\n    return 2\n")
    (tmp_path / "dst.py").write_text("x = 1\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class=None)
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == ("# explains foo\n\n\ndef keep():\n    return 2\n")
    assert (tmp_path / "dst.py").read_text() == "x = 1\n\ndef foo():\n    return 1\n"


def test_move_symbol_without_trailing_newlines_keeps_moved_bytes(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Neither file ends with a newline; the moved function is last in the source.
    # The source retains the other function and blank lines, and the destination is the old text,
    # a newline, then the function without any added final newline.
    """Files lacking a final newline lose no bytes of the moved def or the remainder."""
    (tmp_path / "src.py").write_text("def keep():\n    return 0\n\n\ndef foo():\n    return 1")
    (tmp_path / "dst.py").write_text("x = 1")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class=None)
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == "def keep():\n    return 0\n\n\n"
    assert (tmp_path / "dst.py").read_text() == "x = 1\ndef foo():\n    return 1"


def test_move_symbol_dedent_leaves_string_literal_interior_lines(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A method nested two class levels deep contains a triple-quoted string whose
    # interior line has only four leading spaces and whose closing line has none. Moved to module
    # level with dedent eight, only lines with at least eight leading spaces are trimmed; the
    # string interior lines stay as written (compared with the exact destination text).
    """Dedent only strips lines with exactly n leading spaces, so string interiors survive."""
    (tmp_path / "src.py").write_text(
        "class Old:\n"
        "    class Deep:\n"
        "        def foo(self):\n"
        "            s = '''raw\n"
        "    partial\n"
        "'''\n"
        "            return s\n"
    )
    (tmp_path / "dst.py").write_text("import os\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class=None, dedent=8)
    _apply(r, tmp_path)
    assert (tmp_path / "dst.py").read_text() == (
        "import os\n" "\n" "def foo(self):\n" "    s = '''raw\n" "    partial\n" "'''\n" "    return s\n"
    )


def test_move_symbol_asserts_when_destination_class_missing(tmp_path: Path) -> None:
    # [FUNCTION DOC] The destination class name does not exist in the destination file; applying
    # the primitive raises an assertion error.
    """Naming an into_class absent from the destination fails loudly."""
    (tmp_path / "src.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "dst.py").write_text("x = 1\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="Nope")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_move_symbol_preserves_staticmethod_inside_moved_body(tmp_path: Path) -> None:
    # [FUNCTION DOC] The moved static method contains a nested class with its own static method.
    # After the move into a class, the destination still has the nested static decorator line
    # (at its original deep indentation) followed by the nested definition.
    """A @staticmethod on a nested def inside the moved body must survive the move."""
    (tmp_path / "src.py").write_text(
        "class Old:\n"
        "    @staticmethod\n"
        "    def foo(x):\n"
        "        class Inner:\n"
        "            @staticmethod\n"
        "            def helper(y):\n"
        "                return y\n"
        "        return Inner.helper(x)\n"
    )
    (tmp_path / "dst.py").write_text("class New:\n    def keep(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    _apply(r, tmp_path)
    dst_out = (tmp_path / "dst.py").read_text()
    assert "            @staticmethod\n            def helper(y):\n" in dst_out


def test_move_symbol_rejects_ambiguous_duplicate_names(tmp_path: Path) -> None:
    # [FUNCTION DOC] Two classes each define a method with the requested name and no source
    # class is given. Applying the primitive raises an assertion error.
    """Two same-named defs at equal depth must raise instead of silently picking one."""
    (tmp_path / "src.py").write_text(
        "class A:\n"
        "    def foo(self):\n"
        "        return 'A'\n"
        "\n"
        "class B:\n"
        "    def foo(self):\n"
        "        return 'B'\n"
    )
    (tmp_path / "dst.py").write_text("class New:\n    def keep(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_move_symbol_after_inserts_below_named_function(tmp_path: Path) -> None:
    # [FUNCTION DOC] Destination has functions first and last; moving a function with the after
    # option naming first places it between them (checked by text positions).
    """With after=, the relocated def lands immediately below that sibling def."""
    (tmp_path / "src.py").write_text("def moved():\n    return 1\n")
    (tmp_path / "dst.py").write_text("def first():\n    return 0\n\n\ndef last():\n    return 2\n")
    r = Repro("b", "t").move_symbol("moved", src="src.py", dst="dst.py", into_class=None, after="first")
    _apply(r, tmp_path)
    dst_out = (tmp_path / "dst.py").read_text()
    assert dst_out.index("def first") < dst_out.index("def moved") < dst_out.index("def last")


def test_move_symbol_after_assign_lands_before_typechecking_guard(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Destination has an import, a module constant assigned from a call, and a
    # type-checking conditional. Moving a function with the after option naming the constant
    # puts it after the constant and before the conditional (checked by text positions).
    """after= anchors on a module-level assignment target, landing the def just below it and
    above a following ``if TYPE_CHECKING:`` guard (which is not a nameable anchor)."""
    (tmp_path / "src.py").write_text("def helper(x):\n    return x + 1\n")
    (tmp_path / "dst.py").write_text(
        "from u import is_hip\n" "\n" "_is_hip = is_hip()\n" "\n" "if TYPE_CHECKING:\n" "    from m import Thing\n"
    )
    r = Repro("b", "t").move_symbol("helper", src="src.py", dst="dst.py", into_class=None, after="_is_hip")
    _apply(r, tmp_path)
    dst_out = (tmp_path / "dst.py").read_text()
    assert dst_out.index("_is_hip = is_hip()") < dst_out.index("def helper") < dst_out.index("if TYPE_CHECKING:")


def test_move_symbol_before_and_after_are_mutually_exclusive(tmp_path: Path) -> None:
    # [FUNCTION DOC] Passing both anchors raises an assertion error immediately when recording
    # the primitive, before any operation is applied.
    """Passing both before= and after= is rejected up front."""
    (tmp_path / "src.py").write_text("def moved():\n    return 1\n")
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    with pytest.raises(AssertionError):
        Repro("b", "t").move_symbol(
            "moved",
            src="src.py",
            dst="dst.py",
            into_class=None,
            before="z",
            after="z",
        )


def test_move_symbol_asserts_when_after_symbol_missing(tmp_path: Path) -> None:
    # [FUNCTION DOC] The after anchor names a symbol absent from the destination; applying the
    # primitive raises an assertion error instead of appending.
    """An after= anchor absent from the destination must raise, not fall back to append."""
    (tmp_path / "src.py").write_text("def moved():\n    return 1\n")
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    r = Repro("b", "t").move_symbol("moved", src="src.py", dst="dst.py", into_class=None, after="NO_SUCH_SYMBOL")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_move_symbol_asserts_when_before_sibling_missing(tmp_path: Path) -> None:
    # [FUNCTION DOC] The before anchor names a definition absent from the destination; applying
    # the primitive raises an assertion error instead of appending.
    """A before= anchor absent from the destination must raise, not fall back to append."""
    (tmp_path / "src.py").write_text("def moved():\n    return 1\n")
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    r = Repro("b", "t").move_symbol("moved", src="src.py", dst="dst.py", into_class=None, before="NO_SUCH_DEF")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_move_symbol_preserves_crlf_line_endings(tmp_path: Path) -> None:
    # [FUNCTION DOC] Both files are written as bytes with carriage-return line endings. After a
    # method move into the other class, the destination's count of newline bytes equals its count
    # of carriage-return-newline pairs, i.e. no bare newline was introduced.
    """Moving a def in a CRLF file must keep every line ending CRLF."""
    (tmp_path / "src.py").write_bytes(b"class Old:\r\n    def foo(self):\r\n        return 1\r\n")
    (tmp_path / "dst.py").write_bytes(b"class New:\r\n    def keep(self):\r\n        return 0\r\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    _apply(r, tmp_path)
    dst_bytes = (tmp_path / "dst.py").read_bytes()
    assert dst_bytes.count(b"\n") == dst_bytes.count(b"\r\n")


def test_move_symbol_negative_dedent_indents_into_the_class(tmp_path: Path) -> None:
    # [FUNCTION DOC] A module-level function moves into a destination class with dedent of minus
    # four. The destination contains the function indented as a method (header at four spaces,
    # body at eight).
    """Moving a module-level def into a class with dedent=-4 must indent it as a method."""
    (tmp_path / "src.py").write_text("def helper(x):\n    return x\n")
    (tmp_path / "dst.py").write_text("class New:\n    def e(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol("helper", src="src.py", dst="dst.py", into_class="New", dedent=-4)
    _apply(r, tmp_path)
    assert "    def helper(x):\n        return x\n" in (tmp_path / "dst.py").read_text()


def test_move_symbol_relocates_a_top_level_class(tmp_path: Path) -> None:
    # [FUNCTION DOC] A whole top-level class with a method moves to module level of another file.
    # The source no longer has the class and the destination has the class text unchanged.
    """move_symbol relocates a whole top-level class (with its methods) verbatim."""
    (tmp_path / "src.py").write_text("x = 1\n\n\nclass Widget:\n    def get(self, rank):\n        return rank\n")
    (tmp_path / "dst.py").write_text("y = 2\n")
    r = Repro("b", "t").move_symbol("Widget", src="src.py", dst="dst.py", into_class=None)
    _apply(r, tmp_path)
    assert "class Widget:" not in (tmp_path / "src.py").read_text()
    assert "class Widget:\n    def get(self, rank):\n        return rank\n" in (tmp_path / "dst.py").read_text()


# --- adversarial audit: leave_delegate stubs -------------------------------------
