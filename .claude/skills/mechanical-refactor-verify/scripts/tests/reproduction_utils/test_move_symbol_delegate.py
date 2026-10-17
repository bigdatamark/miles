# [MODULE DOC] Tests for the forwarding-stub option of the symbol-moving primitive: after a
# method moves into a collaborator class, the source keeps its original header and a one-line
# body forwarding to the collaborator through a named attribute of self. Covers comment handling,
# multi-line headers, every parameter kind, string defaults containing brackets, async methods,
# self annotation handling (dropped only when it names the destination class), a different
# forwarded method name, and the refusal for an unannotated static method. Each test writes a
# source and a destination file in a temporary directory and checks the resulting texts.
# KNOWN ISSUE: the two stray section heading comments in the middle of the file duplicate or
# precede unrelated headings.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_reproduction_utils import Repro
from reproduction_testlib import _apply, _commit, _git, _write  # noqa: F401


def test_move_symbol_leave_delegate_keeps_forwarding_stub(tmp_path: Path) -> None:
    # [FUNCTION DOC] A typed method of a mixin class moves into a config class with the delegate
    # attribute cfg. The source keeps the typed header and gets a body that returns the call of
    # the same method on self.cfg with the argument; the old body line is gone from the source and
    # present in the destination.
    """With leave_delegate, the source keeps a forwarding stub through the named field and the
    destination gets the full method body."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n" "    def compute(self, n: int) -> int:\n" "        return n + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text("class Cfg:\n    def existing(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol(
        "compute",
        src="src.py",
        dst="dst.py",
        into_class="Cfg",
        leave_delegate="cfg",
    )
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    dst_out = (tmp_path / "dst.py").read_text()
    assert "def compute(self, n: int) -> int:" in src_out
    assert "return self.cfg.compute(n)" in src_out
    assert "return n + self.cfg.base" not in src_out
    assert "return n + self.cfg.base" in dst_out


def test_move_symbol_leave_delegate_does_not_absorb_leading_comments(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] The moved method body starts with two comment lines before its return. The
    # source afterwards is exactly the class header, the method header and the forwarding return
    # (no comments); the destination contains the comment text.
    """A leading comment before the first statement is not an AST node, so it must not be
    pulled into the forwarding stub -- the delegate is just the header plus the return.
    """
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    def compute(self, n: int) -> int:\n"
        "        # explain the maths\n"
        "        # second comment line\n"
        "        return n + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text("class Cfg:\n    def existing(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol("compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg")
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    dst_out = (tmp_path / "dst.py").read_text()
    assert "# explain the maths" not in src_out
    assert src_out == "class Mixin:\n" "    def compute(self, n: int) -> int:\n" "        return self.cfg.compute(n)\n"
    assert "# explain the maths" in dst_out


# --- adversarial audit: move_symbol edge cases -----------------------------------


# --- adversarial audit: leave_delegate stubs -------------------------------------


def test_move_symbol_leave_delegate_keeps_a_multiline_signature_verbatim(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] The method has a multi-line header with a keyword-only parameter with a
    # default. The source keeps the header lines exactly and forwards with the positional
    # parameter and the keyword-only parameter passed as a same-named keyword. The destination
    # gets the full original method appended after its existing method.
    """A multi-line header is carried into the stub byte-for-byte via the bracket scan."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    def compute(\n"
        "        self,\n"
        "        n: int,\n"
        "        *,\n"
        "        scale: float = 1.0,\n"
        "    ) -> int:\n"
        "        return int(n * scale) + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text("class Cfg:\n    def e(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol("compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg")
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "class Mixin:\n"
        "    def compute(\n"
        "        self,\n"
        "        n: int,\n"
        "        *,\n"
        "        scale: float = 1.0,\n"
        "    ) -> int:\n"
        "        return self.cfg.compute(n, scale=scale)\n"
    )
    assert (tmp_path / "dst.py").read_text() == (
        "class Cfg:\n"
        "    def e(self):\n"
        "        return 0\n"
        "\n"
        "    def compute(\n"
        "        self,\n"
        "        n: int,\n"
        "        *,\n"
        "        scale: float = 1.0,\n"
        "    ) -> int:\n"
        "        return int(n * scale) + self.cfg.base\n"
    )


def test_move_symbol_leave_delegate_forwards_posonly_vararg_kwonly_kwargs(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] The method has a positional-only parameter, a normal parameter, extra
    # positional arguments, two keyword-only parameters (one defaulted) and extra keyword
    # arguments. The forwarding call passes the first two by position, the extras with a star,
    # each keyword-only one as a same-named keyword, and the extra keywords with a double star.
    """Every parameter kind is forwarded correctly in the delegate's return call."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n" "    def compute(self, a, /, b, *args, c, d=3, **kw):\n" "        return a\n"
    )
    (tmp_path / "dst.py").write_text("class Cfg:\n    def keep(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol("compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg")
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "class Mixin:\n"
        "    def compute(self, a, /, b, *args, c, d=3, **kw):\n"
        "        return self.cfg.compute(a, b, *args, c=c, d=d, **kw)\n"
    )


def test_move_symbol_leave_delegate_survives_paren_in_string_default(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A parameter default is a string consisting of an opening parenthesis in a
    # multi-line header. The resulting source compiles and its forwarding call passes the two
    # parameters by position.
    """A string default containing '(' must not break the delegate's header scan."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    def compute(\n"
        "        self,\n"
        '        sep: str = "(",\n'
        "        n: int = 0,\n"
        "    ) -> int:\n"
        "        return n + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text("class Cfg:\n    def keep(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol("compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg")
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    compile(src_out, "src.py", "exec")
    assert "return self.cfg.compute(sep, n)" in src_out


def test_move_symbol_async_leave_delegate_awaits_the_forwarded_call(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] An async method is moved with a delegate; the source stub returns an await
    # of the forwarded call.
    """An async method's delegate stub must await the forwarded coroutine."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n" "    async def compute(self, n):\n" "        return n + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text("class Cfg:\n    def e(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol("compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg")
    _apply(r, tmp_path)
    assert "return await self.cfg.compute(n)" in (tmp_path / "src.py").read_text()


def test_move_symbol_leave_delegate_on_self_annotated_staticmethod(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A static method whose first parameter is an annotated self naming the
    # destination class moves with a delegate, the source class and the drop-annotation option.
    # The source keeps an ordinary method (no decorator, no self annotation) forwarding to
    # self.comp; the destination gets the method with a plain self and the original body.
    """A de-self'd staticmethod (self: Target) moves into Target; the stub drops the
    decorator and the self annotation."""
    (tmp_path / "src.py").write_text(
        "class Runner:\n"
        "    @staticmethod\n"
        "    def work(self: Comp, n: int) -> int:\n"
        "        return n + self.base\n"
    )
    (tmp_path / "dst.py").write_text("class Comp:\n    def existing(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol(
        "work",
        src="src.py",
        dst="dst.py",
        into_class="Comp",
        from_class="Runner",
        drop_self_annotation=True,
        leave_delegate="comp",
    )
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "class Runner:\n" "    def work(self, n: int) -> int:\n" "        return self.comp.work(n)\n"
    )
    assert (tmp_path / "dst.py").read_text() == (
        "class Comp:\n"
        "    def existing(self):\n"
        "        return 0\n"
        "\n"
        "    def work(self, n: int) -> int:\n"
        "        return n + self.base\n"
    )


def test_move_symbol_leave_delegate_keeps_unrelated_self_annotation(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] The method's self annotation names a class other than the destination. With
    # the delegate and the drop-annotation option, the source stub keeps that annotation in its
    # header and forwards to self.comp.
    """A self annotation naming a class other than the destination survives in the stub."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n" "    def work(self: Runner, n: int) -> int:\n" "        return n + self.base\n"
    )
    (tmp_path / "dst.py").write_text("class Comp:\n    def existing(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol(
        "work",
        src="src.py",
        dst="dst.py",
        into_class="Comp",
        from_class="Mixin",
        drop_self_annotation=True,
        leave_delegate="comp",
    )
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "class Mixin:\n" "    def work(self: Runner, n: int) -> int:\n" "        return self.comp.work(n)\n"
    )


def test_move_symbol_delegate_name_forwards_to_the_renamed_collaborator_method(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] With a delegate name differing from the method name, the source stub calls
    # that other name on self.cfg, while the destination keeps the original method name.
    """delegate_name makes the stub call a differently-named method on the collaborator."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n" "    def compute(self, n: int) -> int:\n" "        return n + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text("class Cfg:\n    def existing(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol(
        "compute",
        src="src.py",
        dst="dst.py",
        into_class="Cfg",
        leave_delegate="cfg",
        delegate_name="compute_impl",
    )
    _apply(r, tmp_path)
    assert "return self.cfg.compute_impl(n)" in (tmp_path / "src.py").read_text()
    assert "def compute(self, n: int) -> int:" in (tmp_path / "dst.py").read_text()


def test_move_symbol_leave_delegate_on_unannotated_staticmethod_raises(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A static method with no self parameter is moved with a delegate; applying
    # the primitive raises an assertion error because there is no receiver to forward through.
    """A staticmethod with no self: Target annotation has no receiver to forward through, so
    leave_delegate refuses rather than author a bogus self.<field>.<name>(...) stub."""
    (tmp_path / "src.py").write_text(
        "class Runner:\n" "    @staticmethod\n" "    def work(x: int) -> int:\n" "        return x + 1\n"
    )
    (tmp_path / "dst.py").write_text("class Comp:\n    def existing(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol(
        "work",
        src="src.py",
        dst="dst.py",
        into_class="Comp",
        leave_delegate="comp",
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)
