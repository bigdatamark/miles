# [MODULE DOC] Tests for the call-site rewriting primitives of the reproduction builder: lowering
# (first argument becomes the receiver), requalifying (dropping the owner qualifier) and routing
# through a field. Covers argument preservation (trailing commas, comments, literal spellings,
# redundant parentheses), nesting, non-ASCII columns, form feed line starts, owner filtering,
# convergence and zero-argument calls. Each test writes a small module in a temporary directory,
# records the primitive, applies it and compares the text.
# KNOWN ISSUE: the section heading comments for requalify and for the adversarial audit are each
# written twice in a row.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_reproduction_utils import Repro
from reproduction_testlib import _apply, _commit, _git, _write  # noqa: F401


def test_lowered_call_text_preserves_magic_trailing_comma(tmp_path: Path) -> None:
    # [FUNCTION DOC] A multi-line call with the receiver first and a trailing comma after the
    # last argument. Lowering yields the receiver call with the remaining arguments still one per
    # line and the trailing comma intact.
    """A magic trailing comma in the original call survives the textual lowering."""
    (tmp_path / "m.py").write_text("x = Old.foo(\n    self.r,\n    a,\n    b,\n)\n")
    r = Repro("b", "t").lower_call_sites("foo", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "x = self.r.foo(\n    a,\n    b,\n)\n"


# --- lower_call_sites ----------------------------------------------------------


def test_lower_call_sites_moves_receiver_out_of_args(tmp_path: Path) -> None:
    # [FUNCTION DOC] Single-line call of the owner with the receiver and two more arguments
    # becomes a call on the receiver with the two arguments.
    """Owner.foo(receiver, rest) becomes receiver.foo(rest)."""
    (tmp_path / "m.py").write_text("x = ModelRunner.foo(self.r, a, b)\n")
    r = Repro("b", "t").lower_call_sites("foo", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "x = self.r.foo(a, b)\n"


def test_lower_call_sites_handles_only_receiver_arg(tmp_path: Path) -> None:
    # [FUNCTION DOC] Call with only the receiver argument becomes a call on the receiver with no
    # arguments (and the result is not lowered again).
    """Owner.foo(receiver) becomes receiver.foo() without re-lowering the result."""
    (tmp_path / "m.py").write_text("ModelRunner.foo(self.r)\n")
    r = Repro("b", "t").lower_call_sites("foo", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "self.r.foo()\n"


def test_lower_call_sites_ignores_a_different_owner(tmp_path: Path) -> None:
    # [FUNCTION DOC] A same-named call on an unrelated object is left byte-for-byte unchanged.
    """A same-named call on another receiver (e.g. the moved body's own call) is untouched."""
    (tmp_path / "m.py").write_text("worker.foo(zmq)\n")
    r = Repro("b", "t").lower_call_sites("foo", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "worker.foo(zmq)\n"


def test_lower_call_sites_preserves_magic_trailing_comma(tmp_path: Path) -> None:
    # [FUNCTION DOC] Multi-line call with receiver, one argument and a trailing comma; the
    # result keeps the multi-line layout with the remaining argument and the trailing comma.
    """A magic trailing comma is kept so the formatter re-explodes the lowered call."""
    (tmp_path / "m.py").write_text("ModelRunner.foo(\n    self.r,\n    a,\n)\n")
    r = Repro("b", "t").lower_call_sites("foo", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "self.r.foo(\n    a,\n)\n"


# --- requalify_call_sites ------------------------------------------------------


# --- requalify_call_sites ------------------------------------------------------


def test_requalify_call_sites_drops_the_qualifier(tmp_path: Path) -> None:
    # [FUNCTION DOC] A call qualified by the owner with two arguments becomes a bare call with
    # the same arguments.
    """Owner.bar(args) becomes bar(args) when bar moves to a free function."""
    (tmp_path / "m.py").write_text("y = ModelRunner.bar(a, b)\n")
    r = Repro("b", "t").requalify_call_sites("bar", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "y = bar(a, b)\n"


def test_route_call_sites_through_field_inserts_the_field(tmp_path: Path) -> None:
    # [FUNCTION DOC] A call on a dotted receiver gets the field name inserted between the
    # receiver and the method name (no owner filter given).
    """recv.bar(a) becomes recv.updater.bar(a) when bar moves onto a collaborator field."""
    (tmp_path / "m.py").write_text("y = self.worker.runner.bar(a)\n")
    r = Repro("b", "t").route_call_sites_through_field("bar", field="updater", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "y = self.worker.runner.updater.bar(a)\n"


def test_route_call_sites_through_field_skips_an_already_routed_call(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A call whose receiver already ends with the field name is left unchanged,
    # so applying the primitive terminates and is idempotent.
    """A call already going through the field is left alone, so the pass converges."""
    (tmp_path / "m.py").write_text("y = self.runner.updater.bar(a)\n")
    r = Repro("b", "t").route_call_sites_through_field("bar", field="updater", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "y = self.runner.updater.bar(a)\n"


def test_route_call_sites_through_field_honors_owner_filter(tmp_path: Path) -> None:
    # [FUNCTION DOC] Two calls of the same method on different receivers; with the owner filter
    # set to the first receiver, only the first call gets the field inserted.
    """With owner set, only calls on that exact receiver are routed through the field."""
    (tmp_path / "m.py").write_text("a = x.bar(1)\nb = y.bar(2)\n")
    r = Repro("b", "t").route_call_sites_through_field("bar", field="updater", paths=["m.py"], owner="x")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "a = x.updater.bar(1)\nb = y.bar(2)\n"


# --- adversarial audit: call-site rewrites ---------------------------------------


# --- adversarial audit: call-site rewrites ---------------------------------------


def test_requalify_call_sites_matches_a_zero_argument_call(tmp_path: Path) -> None:
    # [FUNCTION DOC] A qualified call with no arguments becomes a bare call with empty
    # parentheses.
    """Owner.bar() with no arguments is requalified to bar()."""
    (tmp_path / "m.py").write_text("y = Owner.bar()\n")
    r = Repro("b", "t").requalify_call_sites("bar", "Owner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "y = bar()\n"


def test_lower_call_sites_preserves_comments_inside_a_multiline_call(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A multi-line call has a trailing comment on one of the later arguments.
    # After lowering the comment text is still in the file.
    """A comment between arguments of the rewritten call must survive."""
    (tmp_path / "m.py").write_text("x = Old.foo(\n    self.r,\n    a,  # keep me\n    b,\n)\n")
    r = Repro("b", "t").lower_call_sites("foo", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert "# keep me" in (tmp_path / "m.py").read_text()


def test_lower_call_sites_preserves_arg_literal_spelling(tmp_path: Path) -> None:
    # [FUNCTION DOC] Arguments include a hexadecimal integer literal and a double-quoted string;
    # after lowering they keep their exact spelling (not normalized to decimal or single quotes).
    """Hex literals and quote styles inside the rewritten call must not be normalized."""
    (tmp_path / "m.py").write_text('x = Old.foo(self.r, 0x10, "s")\n')
    r = Repro("b", "t").lower_call_sites("foo", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == 'x = self.r.foo(0x10, "s")\n'


def test_lower_call_sites_lowers_a_nested_matching_call_too(tmp_path: Path) -> None:
    # [FUNCTION DOC] A matching call passed as an argument of another matching call. Both are
    # lowered (the outer first, the inner on a later pass), giving a nested receiver call.
    """A matching call nested inside another matching call is lowered as well."""
    (tmp_path / "m.py").write_text("x = Old.foo(self.r, Old.foo(self.q, 1))\n")
    r = Repro("b", "t").lower_call_sites("foo", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "x = self.r.foo(self.q.foo(1))\n"


def test_lower_call_sites_magic_comma_with_sole_receiver_arg_stays_valid(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A multi-line call whose only argument is the receiver followed by a
    # trailing comma. After lowering, the file text must still compile as valid source (checked
    # by compiling it); no exact text is asserted.
    """Lowering a magic-comma call whose only argument is the receiver stays valid Python."""
    (tmp_path / "m.py").write_text("Owner.foo(\n    self.r,\n)\n")
    r = Repro("b", "t").lower_call_sites("foo", "Owner", paths=["m.py"])
    _apply(r, tmp_path)
    out = (tmp_path / "m.py").read_text()
    compile(out, "m.py", "exec")


def test_call_rewrite_is_column_accurate_on_non_ascii_lines(tmp_path: Path) -> None:
    # [FUNCTION DOC] The call follows a string with Chinese characters on the same line, so byte
    # and character columns differ. The call is still rewritten correctly and the string kept.
    """A call after a non-ASCII string on the same line is rewritten at the right columns."""
    (tmp_path / "m.py").write_text('x = "中文"; y = Owner.foo(self.r, 1)\n')
    r = Repro("b", "t").lower_call_sites("foo", "Owner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == 'x = "中文"; y = self.r.foo(1)\n'


def test_call_rewrite_survives_a_form_feed_line_start(tmp_path: Path) -> None:
    # [FUNCTION DOC] A line starting with a form feed character precedes the call. Only the call
    # line is rewritten and the form feed line is left intact (line numbering must agree with
    # the syntax tree's).
    """A form feed at a line start must not shift the rewrite onto the wrong line."""
    (tmp_path / "m.py").write_text("a = 1\n\x0cb = 2\ny = Owner.foo(self.r, 1)\n")
    r = Repro("b", "t").lower_call_sites("foo", "Owner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "a = 1\n\x0cb = 2\ny = self.r.foo(1)\n"


def test_requalify_call_sites_preserves_redundant_parens_in_kwargs(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] A multi-line qualified call has a keyword argument whose value is wrapped in
    # redundant parentheses. After requalifying, that parenthesized text is still present.
    """Redundant parentheses around a keyword value survive the requalification."""
    (tmp_path / "m.py").write_text("y = Old.bar(\n    a=1,\n    b=(2),\n)\n")
    r = Repro("b", "t").requalify_call_sites("bar", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert "b=(2)" in (tmp_path / "m.py").read_text()
