# [MODULE DOC] Library for certifying that a git commit is a purely mechanical refactor. The
# caller builds a reproduction recipe (the builder class defined here) out of small,
# source-text-preserving primitives: moving a symbol between modules, extracting a function or
# a set of symbols into a new module, rewriting call sites, and adding, removing or re-pathing
# imports. The verifier checks out the base commit into a temporary git worktree, applies the
# recipe, runs the pre-commit formatter on the changed files, commits any formatter fixes, and
# diffs the worktree against the target commit; an empty diff means the recipe reproduces the
# commit exactly (PASS), otherwise the residual diff is printed and the process exits with
# status one. Design principles shared by all primitives: locate code with the Python syntax
# tree, but splice the ORIGINAL source text (never regenerate code from the tree) so literals,
# comments and layout survive byte-for-byte; preserve each file's line-ending style (no newline
# translation on read or write); count columns as UTF-8 byte offsets like the syntax tree does;
# and refuse (raise an assertion error) when a request is ambiguous or cannot be done
# faithfully. Depends only on git, the pre-commit tool and the standard library. Operations are
# recorded by the builder and executed later, in order, against a root directory.
"""Reproduce a whole mechanical refactor and diff it byte-for-byte against a commit.

You write a ``transform()`` that recreates the change; ``verify_mechanical_refactor``
checks out the base commit in a throwaway worktree, runs the transform, runs pre-commit on
the changed files, and diffs the result against the target commit. An empty diff is a PASS.
Use this for a single mechanical PR -- a relocation, a whole-file split, or a rename where a
formatter re-wraps lines, which reproduce-and-byte-diff certifies exactly.

The ``Repro`` builder composes faithful relocation primitives (``move_symbol``,
``extract_to_new_module``, ``extract_symbols_to_new_module``, ``extract_function``,
``lower_call_sites``, ``requalify_call_sites``, ``remove_import``, ``remove_imported_name``,
``add_import``, ``repath_import``, ``add_typechecking_import``) into a transform, so a move
that a formatter re-wrapped can be reproduced and certified. Each primitive does only a relocation-faithful
edit (it never changes logic), so a byte match after the formatter certifies the commit is
exactly that relocation. The primitives are deliberately small -- AST-located, spliced as
original source text; see spec-reproduction-utils.md.

This module is self-contained and needs only git and the standard library.
"""

import ast
import io
import re
import shlex
import subprocess
import sys
import tempfile
import tokenize
from collections.abc import Callable
from pathlib import Path


def exec_command(cmd: str, cwd: str | None = None, check: bool = True) -> str:
    # [FUNCTION DOC] Runs a shell command string (through the shell) in the optional working
    # directory, echoing the command to standard output first (prefixed with a dollar sign,
    # flushed). Captures standard output and standard error as text. When check is true (the
    # default) and the exit code is non-zero, raises a runtime error containing the command and
    # the stripped standard error. Returns the stripped standard output text. Side effects:
    # arbitrary subprocess execution.
    print(f"  $ {cmd}", flush=True)
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\n{result.stderr.strip()}")
    return result.stdout.strip()


def git_add_and_commit(message: str, cwd: str) -> None:
    # [FUNCTION DOC] HELPER: Stages every change (including deletions and untracked files) in
    # the given directory's repository and commits with the shell-quoted message. Raises a
    # runtime error if either git step fails (for example nothing to commit). Returns nothing.
    exec_command(f"git add -A && git commit -m {shlex.quote(message)}", cwd=cwd)


def dedent(text: str, n: int) -> str:
    # [FUNCTION DOC] Removes exactly n leading spaces from every line of the text that begins
    # with at least n spaces; lines with fewer leading spaces (including blank lines and lines
    # starting with a tab) are left unchanged rather than partially trimmed. Lines are split
    # only on the newline character (see the split helper), and line terminators are kept.
    # Returns the new text.
    """Remove exactly n leading spaces from each line."""
    lines = _split_keepends(text)
    return "".join(line[n:] if line[:n] == " " * n else line for line in lines)


def _split_keepends(text: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Splits text into lines, each keeping its terminating newline
    # character; a carriage return before the newline stays inside the line. Only the newline
    # character is a separator. If the text does not end with a newline the last piece (when
    # non-empty) is returned without a terminator; an empty text gives an empty list.
    """Split into lines ending in "\\n" only -- unlike ``str.splitlines``, a form feed or
    other exotic line break stays inside its line, matching ast's line numbering."""
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _read_source(path: Path) -> str:
    # [FUNCTION DOC] HELPER: Reads a text file with the platform default encoding and with
    # newline translation disabled so carriage returns survive. Returns the full contents.
    """Read preserving the file's line endings (no universal-newline translation), so a
    CRLF file round-trips byte-for-byte through the primitives."""
    with path.open("r", newline="") as f:
        return f.read()


def _write_source(path: Path, text: str) -> None:
    # [FUNCTION DOC] HELPER: Overwrites a text file with the given text, with newline
    # translation disabled (the counterpart of the read helper). Returns nothing.
    with path.open("w", newline="") as f:
        f.write(text)


def _newline_style(text: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the two-character carriage-return plus newline sequence if
    # it occurs anywhere in the text, otherwise the single newline character. Used to match the
    # file's line-ending style when synthesizing new lines.
    return "\r\n" if "\r\n" in text else "\n"


def verify_mechanical_refactor(
    base_commit: str,
    target_commit: str,
    transform: "Callable[[Path], None]",
) -> None:
    # [FUNCTION DOC] Verifies that applying the transform to the base commit reproduces the
    # target commit exactly. Inputs: base commit id, target commit id, and a callable taking the
    # worktree root path. Steps, printed as four numbered stages: (1) find the repository root
    # from the current directory, create a temporary directory and a new git worktree there on a
    # new branch named after the first eight characters of the base commit; (2) call the
    # transform with the worktree path; (3) stage everything, list changed files that were added,
    # copied, modified or renamed relative to the base, run pre-commit on just those files
    # (its failure exit status is deliberately ignored since it rewrites files), and if the
    # worktree is then dirty commit it as "pre-commit fixes"; (4) diff the worktree against the
    # target commit. A non-empty diff prints a FAIL message with the diff and exits the process
    # with status one; an empty diff prints a PASS message. In all cases (a finally block) it
    # prints where the worktree and branch were left and the manual cleanup command; nothing is
    # removed automatically. Raises a runtime error if a git step fails.
    # KNOWN ISSUE: the branch name depends only on the base commit, so a second verification of
    # the same base while the first worktree or branch still exists fails at worktree creation.
    repo_root = exec_command("git rev-parse --show-toplevel")
    worktree_dir = tempfile.mkdtemp(prefix="verify-mechanical-")
    branch_name = f"verify-mechanical-{base_commit[:8]}"

    try:
        print(f"[1/4] Creating worktree at {base_commit[:8]}...")
        exec_command(
            f"git worktree add -b {branch_name} {worktree_dir} {base_commit}",
            cwd=repo_root,
        )

        print("[2/4] Running transformation...")
        transform(Path(worktree_dir))

        print("[3/4] Running pre-commit...")
        exec_command("git add -A", cwd=worktree_dir)
        changed = exec_command(
            f"git diff --cached --name-only --diff-filter=ACMR {base_commit}",
            cwd=worktree_dir,
        ).split()
        if changed:
            files = " ".join(shlex.quote(path) for path in changed)
            exec_command(f"pre-commit run --files {files}", cwd=worktree_dir, check=False)
        if exec_command("git status --porcelain", cwd=worktree_dir):
            git_add_and_commit("pre-commit fixes", cwd=worktree_dir)

        print(f"[4/4] Diffing against {target_commit[:8]}...")
        diff = exec_command(
            f"git diff {target_commit} -- .",
            cwd=worktree_dir,
            check=False,
        )

        if diff:
            print(f"\nFAIL: diff is non-empty:\n{diff}")
            sys.exit(1)
        else:
            print("\nPASS: transform reproduces the commit exactly.")

    finally:
        print(f"\nWorktree left at: {worktree_dir}")
        print(f"Branch: {branch_name}")
        print("To clean up manually:")
        print(f"  git worktree remove {worktree_dir} && git branch -D {branch_name}")


# Decorators a method sheds when it becomes a free function; carried on one side of a move.
_MOVE_DECORATORS = {"@staticmethod", "@classmethod"}


def _find_def(tree: ast.AST, name: str) -> ast.AST | None:
    # [FUNCTION DOC] HELPER: Searches the whole syntax tree (breadth-first walk order) for the
    # first function or async function definition with the given name, at any nesting depth,
    # including methods. Returns the node, or nothing if absent. Performs no ambiguity check.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _find_unique_def(tree: ast.AST, name: str, *, from_class: str | None = None, where: str) -> ast.AST:
    # [FUNCTION DOC] HELPER: Finds the single function, async function or class definition
    # called name. If from_class is given the search is restricted to that class's subtree
    # (assertion error if the class is missing). Otherwise, for a module tree, a lone top-level
    # definition of that name wins immediately even if nested definitions share the name.
    # Otherwise all matching definitions in the (sub)tree are collected: none gives an
    # assertion error saying not found in the where description; more than one gives an
    # assertion error telling the caller to pass the class to disambiguate. Returns the node.
    """Resolve ``def name`` (or ``class name``) and refuse ambiguity: with same-named defs in
    scope the first-match lookup could silently cut the wrong body, so the caller must scope
    the search with ``from_class``."""
    definition = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    root: ast.AST = tree
    if from_class is not None:
        cls = _find_class(tree, from_class)
        assert cls is not None, f"class {from_class} not found in {where}"
        root = cls
    elif isinstance(tree, ast.Module):
        top_level = [node for node in tree.body if isinstance(node, definition) and node.name == name]
        if len(top_level) == 1:
            return top_level[0]
    matches = [node for node in ast.walk(root) if isinstance(node, definition) and node.name == name]
    assert matches, f"{name} not found in {where}"
    assert len(matches) == 1, f"{len(matches)} defs named {name} in {where}; pass from_class to disambiguate"
    return matches[0]


def _def_header_end(def_text: str) -> int:
    # [FUNCTION DOC] HELPER: Tokenizes the definition text (which must start at the def line)
    # and tracks bracket nesting depth over operator tokens; returns the 1-based line number of
    # the first colon seen at depth zero, which closes the signature. Raises an assertion error
    # if there is none. Tokenizer errors on malformed text propagate.
    """1-based line (within ``def_text``, which starts at the def line) of the colon that
    opens the body. Tokenize-based, so parentheses inside string defaults do not confuse
    the bracket depth."""
    depth = 0
    for token in tokenize.generate_tokens(io.StringIO(def_text).readline):
        if token.type == tokenize.OP:
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth -= 1
            elif token.string == ":" and depth == 0:
                return token.start[0]
    raise AssertionError("no header-ending colon found")


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    # [FUNCTION DOC] HELPER: Returns the first class definition with the given name found by a
    # breadth-first walk of the whole tree (any nesting depth), or nothing if absent.
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _def_span(node: ast.AST) -> tuple[int, int]:
    # [FUNCTION DOC] HELPER: Returns the 1-based first and last line numbers of a definition
    # node, where the first line is the smallest of the def line and all decorator lines, and
    # the last is the node's end line.
    """(first, last) 1-based line numbers of a def, including its decorators."""
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return start, node.end_lineno


def _symbol_named(node: ast.AST, name: str) -> bool:
    # [FUNCTION DOC] HELPER: Tells whether a statement node defines the symbol name: a function,
    # async function or class by its own name; an annotated assignment whose target is a plain
    # name equal to name; or a plain assignment where any one of its targets is a plain name
    # equal to name. Tuple-unpacking targets, attribute targets and every other statement kind
    # never match.
    """Whether a top-level statement defines the symbol ``name`` -- a def/class by its name,
    or a module-level assignment by one of its target names (so ``_is_hip = is_hip()`` is
    found by ``_is_hip``)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name == name
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == name
    if isinstance(node, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    return False


def _byte_slice(line: str, start: int | None, end: int | None) -> str:
    # [FUNCTION DOC] HELPER: Encodes the line as UTF-8, slices by the given byte offsets (either
    # may be nothing for open-ended) and decodes back. Needed because syntax tree column offsets
    # are byte offsets. Slicing through the middle of a multi-byte character raises a decode
    # error.
    """Slice a line by UTF-8 byte offsets -- ast col_offsets count bytes, not characters."""
    return line.encode("utf-8")[start:end].decode("utf-8")


def _replace_span(text: str, sl: int, sc: int, el: int, ec: int, repl: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the text with the region from (start line, start byte
    # column) up to but excluding (end line, end byte column) replaced by the replacement text.
    # Lines are 1-based, columns 0-based bytes. Everything before the start and after the end,
    # including line terminators, is preserved verbatim.
    """Replace the text from (sl, sc) to (el, ec) -- 1-based lines, 0-based byte columns,
    end exclusive (ast node span semantics) -- with ``repl``."""
    lines = _split_keepends(text)
    before = "".join(lines[: sl - 1]) + _byte_slice(lines[sl - 1], None, sc)
    after = _byte_slice(lines[el - 1], ec, None) + "".join(lines[el:])
    return before + repl + after


def _slice_span(text: str, sl: int, sc: int, el: int, ec: int) -> str:
    # [FUNCTION DOC] HELPER: Returns the substring from (start line, start byte column) up to
    # excluding (end line, end byte column), with the same coordinate conventions as the replace
    # helper. For a single-line span it slices that line; otherwise it joins the tail of the
    # first line, all whole lines between, and the head of the last line.
    lines = _split_keepends(text)
    if sl == el:
        return _byte_slice(lines[sl - 1], sc, ec)
    return _byte_slice(lines[sl - 1], sc, None) + "".join(lines[sl : el - 1]) + _byte_slice(lines[el - 1], None, ec)


def _node_slice(text: str, node: ast.AST) -> str:
    # [FUNCTION DOC] HELPER: Returns the exact original source text covered by a syntax tree
    # node, using its recorded start and end positions.
    return _slice_span(text, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)


def _rewrite_matching_calls(text: str, predicate: "Callable", rewrite: "Callable") -> str:
    # [FUNCTION DOC] HELPER: Repeatedly rewrites call expressions. Loop: parse the current text,
    # take the first call node (in the tree's breadth-first walk order, so outer calls before
    # nested ones) accepted by the predicate; if none, return the text. Otherwise replace that
    # call's exact source span with the string the rewrite callable returns (it receives the
    # current text and the node) and loop again with a fresh parse. Termination requires the
    # rewrite to make the call stop matching; a rewrite that still matches loops forever.
    # Syntax errors from parsing propagate.
    """Rewrite every call ``predicate`` accepts by splicing the original source text
    (never ``ast.unparse``, which would re-spell literals and drop comments). One call is
    rewritten per pass and the text re-parsed, so a matching call nested inside another
    match is rewritten on a later pass instead of being overwritten."""
    while True:
        node = next(
            (n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call) and predicate(n)),
            None,
        )
        if node is None:
            return text
        text = _replace_span(
            text,
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset,
            rewrite(text, node),
        )


def _lowered_call_text(text: str, node: ast.Call) -> str:
    # [FUNCTION DOC] HELPER: Builds the replacement text that turns a call of an attribute on
    # an owner, whose first argument is the receiver, into a method call on the receiver. The
    # receiver's source must be single-line and comment-free, no comment may sit between the
    # opening parenthesis and the receiver or between the receiver and the following comma or
    # closing parenthesis, and only whitespace may separate the receiver from the comma
    # (assertion errors otherwise). Result: the receiver source, a dot, the called attribute
    # name, an opening parenthesis, then the remaining original text after the receiver and its
    # comma with leading spaces and tabs removed (when there were no other arguments, just the
    # closing parenthesis). Keyword arguments, trailing commas, comments and newlines after the
    # first argument are preserved byte-for-byte.
    """Original call text with the leading receiver argument spliced out and made the
    call's new receiver: ``Owner.foo(recv, rest...)`` -> ``recv.foo(rest...)``. All other
    argument bytes (literal spelling, comments, the magic trailing comma) are untouched.
    """
    receiver = node.args[0]
    receiver_src = _node_slice(text, receiver)
    assert (
        "\n" not in receiver_src and "#" not in receiver_src
    ), f"receiver {receiver_src!r} must be single-line and comment-free"
    opener = _slice_span(
        text,
        node.func.end_lineno,
        node.func.end_col_offset,
        receiver.lineno,
        receiver.col_offset,
    )
    assert "#" not in opener, f"comment before the receiver in {opener!r}"
    seg = _slice_span(
        text,
        receiver.end_lineno,
        receiver.end_col_offset,
        node.end_lineno,
        node.end_col_offset,
    )
    head, comma, rest = seg.partition(",")
    assert "#" not in head, f"comment after the receiver in {head!r}"
    if comma:
        assert head.strip() == "", f"unexpected text {head!r} after the receiver"
        rest = rest.lstrip(" \t")
    else:
        assert head.strip() == ")", f"unexpected text {head!r} after the receiver"
        rest = head.lstrip(" \t")
    return f"{receiver_src}.{node.func.attr}({rest}"


def _drop_self_annotation(method_text: str, name: str) -> str:
    # [FUNCTION DOC] HELPER: Removes the type annotation (the colon and annotation text) from
    # the first parameter of the named method when that parameter is called self and is
    # annotated. The text may be indented as inside a class: the indentation of its first line
    # is measured and removed only for parsing, and reported columns are shifted back. Returns
    # the text unchanged if the method is not found, has no positional parameters, or the first
    # parameter is not an annotated self. Only the span from the end of the parameter name to
    # the end of the annotation is deleted.
    # KNOWN ISSUE: the column arithmetic assumes the name is immediately followed by the colon
    # and annotation, so a space between the name and the colon is not handled specially, and
    # the first-line indentation measurement counts only spaces, not tabs.
    """Drop the type annotation from a moved method's ``self`` parameter -- relocating
    ``def foo(self: Target)`` into ``Target`` makes the annotation redundant. The body is
    otherwise untouched, so the relocation stays byte-faithful. ``method_text`` may still be
    class-indented, so it is dedented to parse and the columns are mapped back."""
    first_line = method_text.split("\n", 1)[0]
    base_indent = len(first_line) - len(first_line.lstrip(" "))
    fn = _find_def(ast.parse(dedent(method_text, base_indent)), name)
    if fn is None or not fn.args.args:
        return method_text
    first = fn.args.args[0]
    if first.arg != "self" or first.annotation is None:
        return method_text
    annotation = first.annotation
    return _replace_span(
        method_text,
        first.lineno,
        first.col_offset + base_indent + len("self"),
        annotation.end_lineno,
        annotation.end_col_offset + base_indent,
        "",
    )


def _multiline_string_interior_lines(top_level_text: str) -> set[int]:
    # [FUNCTION DOC] HELPER: Tokenizes the text and returns the set of 1-based line numbers that
    # are inside a string token spanning several lines: every line after the token's first line
    # through its last line (the closing-delimiter line included). Used to avoid re-indenting
    # text that is part of a string literal's value.
    """1-based lines of ``top_level_text`` that lie inside a multi-line string token
    (every line after the token's opening line, including the closing-delimiter line).
    Re-indenting those lines would change the literal's value, not its layout."""
    interior: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(top_level_text).readline):
        if token.type == tokenize.STRING and token.end[0] > token.start[0]:
            interior.update(range(token.start[0] + 1, token.end[0] + 1))
    return interior


def _audit_extract_header(
    header: str,
    removed_assigns: dict[str, str | None],
    where: str,
    rederivable: dict[str, str | None] | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Audits the hand-authored header text of a new module so no
    # unaccounted code rides along. Inputs: the header text, a mapping from names of assignments
    # deleted from the source module to their value source text, a location description for
    # messages, and an optional mapping (default empty) of surviving source constants to their
    # value text. Each top-level header statement must be one of: an import; a bare string
    # expression (docstring); a conditional on the type-checking flag (plain or module
    # qualified name) whose body is only imports; an assignment of a standard logger obtained
    # with the module name; an assignment whose every plain-name target appears in the removed
    # mapping with an identical value text (those names are recorded as reproduced); or one
    # whose every plain-name target appears in the re-derivable mapping with identical value
    # text. Anything else raises an assertion error naming the statement. Finally every removed
    # assignment name must have been reproduced in the header, else an assertion error lists the
    # missing names. Value text is compared via regenerated source from the tree.
    """Refuse header content the extraction cannot vouch for. The header of a scattered
    extraction is authored text reproduced from the target commit, so anything beyond
    imports, a TYPE_CHECKING import block, a logger, a byte-equivalent copy of an
    assignment deleted from the source (``removed_assigns``), or a byte-equivalent copy of
    a module constant that *survives* in the source (``rederivable`` -- re-derived
    boilerplate such as ``_is_hip = is_hip()``, provably not fiction because the same
    statement still exists in the source) would let arbitrary new code ride into the new
    module under a PASS verdict."""
    rederivable = rederivable or {}
    header_assigned: set[str] = set()
    for stmt in ast.parse(header).body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if (
            isinstance(stmt, ast.If)
            and ast.unparse(stmt.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING")
            and all(isinstance(sub, (ast.Import, ast.ImportFrom)) for sub in stmt.body)
        ):
            continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            names = [x.id for x in targets if isinstance(x, ast.Name)]
            value_src = ast.unparse(stmt.value) if stmt.value is not None else None
            if value_src == "logging.getLogger(__name__)":
                continue
            if names and all(n in removed_assigns and removed_assigns[n] == value_src for n in names):
                header_assigned.update(names)
                continue
            if names and all(n in rederivable and rederivable[n] == value_src for n in names):
                continue
        raise AssertionError(
            f"unverifiable header statement in {where}: {ast.unparse(stmt)!r} is "
            f"neither scaffolding nor a relocated source assignment"
        )
    missing = set(removed_assigns) - header_assigned
    assert not missing, (
        f"drop_assigns {sorted(missing)} deleted from the source but not reproduced " f"in the header of {where}"
    )


class Repro:
    # [CLASS DOC] Fluent recipe builder. Each primitive method validates nothing up front: it
    # appends a closure to the ordered list of recorded operations and returns the builder so
    # calls chain. Every recorded operation takes the root directory of a checked-out tree and
    # edits files under it. Holds the base commit, the target commit, an optional repository
    # root, and the list of operations. The run method (further below) executes them through the
    # verifier. Operations use relative file paths from the root.
    """Builds a faithful relocation transform from primitives, then reproduces a commit.

    Operations are recorded and applied in order to a throwaway worktree at ``base``; the
    formatter (pre-commit) runs, and the result is diffed against ``target``. ``run`` prints
    PASS on an empty diff, otherwise the residual -- exactly what the relocation does not
    account for (a bundled change, or a re-derived scaffold a human must confirm)."""

    def __init__(self, base: str, target: str, repo_root: str | None = None) -> None:
        # [FUNCTION DOC] Stores the base commit, the target commit and the optional repository
        # root, and starts with an empty list of recorded operations.
        self.base = base
        self.target = target
        self.repo_root = repo_root
        self.ops: list[Callable[[Path], None]] = []

    def lower_call_sites(self, name: str, owner: str, *, paths: list[str]) -> "Repro":
        # [FUNCTION DOC] Records an operation that, in each listed file, rewrites every call of
        # the attribute name on an expression whose source text equals owner and that has at
        # least one positional argument, so the first argument becomes the receiver of the
        # method call and the rest of the arguments stay as they were. Returns the builder.
        """Rewrite ``owner.name(receiver, rest)`` to ``receiver.name(rest)`` -- a static
        call becoming an instance-method call when ``name`` moves onto a class."""

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation: for each relative path, reads the
            # file, applies the repeated call rewriting with the predicate below and the
            # lowering text builder, and writes the file back (always, even if unchanged).
            for rel in paths:
                path = root / rel

                def predicate(node: ast.Call) -> bool:
                    # [FUNCTION DOC] HELPER: True for a call of an attribute named name whose
                    # object expression has source text equal to owner and which has at least
                    # one positional argument.
                    return (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == name
                        and bool(node.args)
                        and ast.unparse(node.func.value) == owner
                    )

                _write_source(
                    path,
                    _rewrite_matching_calls(_read_source(path), predicate, _lowered_call_text),
                )

        self.ops.append(op)
        return self

    def requalify_call_sites(self, name: str, owner: str, *, paths: list[str]) -> "Repro":
        # [FUNCTION DOC] Records an operation that, in each listed file, rewrites every call of
        # the attribute name on an expression whose source text equals owner into a plain call
        # of the bare name with identical arguments (the qualifier and dot are dropped). Matches
        # calls with any number of arguments. Returns the builder.
        """Rewrite ``owner.name(args)`` to ``name(args)`` -- dropping the qualifier when
        ``name`` moves to a module-level free function."""

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation: for each relative path, reads the
            # file, applies the repeated call rewriting, and writes it back.
            for rel in paths:
                path = root / rel

                def predicate(node: ast.Call) -> bool:
                    # [FUNCTION DOC] HELPER: True for a call of an attribute named name on an
                    # expression whose source text equals owner.
                    return (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == name
                        and ast.unparse(node.func.value) == owner
                    )

                def rewrite(text: str, node: ast.Call) -> str:
                    # [FUNCTION DOC] HELPER: Returns the bare name followed by the original
                    # call text after the callee expression (the parenthesized arguments).
                    call_src = _node_slice(text, node)
                    func_src = _node_slice(text, node.func)
                    return name + call_src[len(func_src) :]

                _write_source(
                    path,
                    _rewrite_matching_calls(_read_source(path), predicate, rewrite),
                )

        self.ops.append(op)
        return self

    def route_call_sites_through_field(
        self, name: str, *, field: str, paths: list[str], owner: str | None = None
    ) -> "Repro":
        # [FUNCTION DOC] Records an operation that, in each listed file, inserts an attribute
        # access named field between the receiver and the called attribute name for every call
        # of that attribute (any number of arguments). Calls whose receiver is already an
        # attribute access ending in field are skipped (so repeated passes converge). When owner
        # is given only calls whose receiver source text equals owner are rewritten. Returns the
        # builder.
        """Rewrite ``<recv>.name(args)`` to ``<recv>.field.name(args)`` -- the method moved
        onto a collaborator reached through ``self.field``, so its callers route through that
        field. With ``owner`` given, only calls whose receiver text equals ``owner`` are
        rewritten. A call already routed through ``field`` is skipped, so the pass converges.
        """

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation: for each relative path, reads the
            # file, applies the repeated call rewriting, and writes it back.
            for rel in paths:
                path = root / rel

                def predicate(node: ast.Call) -> bool:
                    # [FUNCTION DOC] HELPER: True for a call of an attribute named name, whose
                    # receiver is not already an attribute named field, and (if owner is set)
                    # whose receiver source text equals owner.
                    return (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == name
                        and not (isinstance(node.func.value, ast.Attribute) and node.func.value.attr == field)
                        and (owner is None or ast.unparse(node.func.value) == owner)
                    )

                def rewrite(text: str, node: ast.Call) -> str:
                    # [FUNCTION DOC] HELPER: Returns the receiver source, a dot, the field
                    # name, a dot, the method name, then the original text after the callee
                    # expression (the parenthesized arguments).
                    call_src = _node_slice(text, node)
                    func_src = _node_slice(text, node.func)
                    receiver_src = _node_slice(text, node.func.value)
                    return receiver_src + f".{field}.{name}" + call_src[len(func_src) :]

                _write_source(
                    path,
                    _rewrite_matching_calls(_read_source(path), predicate, rewrite),
                )

        self.ops.append(op)
        return self

    def remove_import(self, rel: str, import_text: str, *, in_function: str | None = None) -> "Repro":
        # [FUNCTION DOC] Records an operation that deletes import statements from the file at
        # the relative path rel. A statement (plain or from-import, at any nesting depth) is
        # selected when its source text contains import_text as a whole token, meaning the
        # characters before and after the match are not word characters or dots. If in_function
        # is given it must name exactly one definition (assertion error if missing or
        # ambiguous) and only imports whose first line lies within that definition's line range
        # are considered. At least one match is required (assertion error otherwise). An import
        # that is alone on its lines is deleted whole, along with one immediately following blank
        # line if present. An import sharing its lines with other simple statements (separated
        # by semicolons) is cut out by span and the leftover semicolons and spaces on that line
        # are tidied. Returns the builder.
        # KNOWN ISSUE: the docstring says "contains import_text" but the match is token-bounded,
        # so a partial name inside a longer dotted path does not match.
        """Remove every import statement whose text contains ``import_text`` (and a trailing
        blank), optionally scoped to one function so a same-text module-level import (e.g. a
        ``TYPE_CHECKING`` guard) is left untouched."""

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation; does the work described for the
            # builder method above, reading and writing the file with line endings preserved.
            path = root / rel
            lines = _split_keepends(_read_source(path))
            tree = ast.parse("".join(lines))
            scope: tuple[int, int] | None = None
            if in_function is not None:
                fn = _find_unique_def(tree, in_function, where=rel)
                scope = (fn.lineno, fn.end_lineno)
            compound = (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.With,
                ast.AsyncWith,
                ast.Try,
                ast.Match,
            )
            simple_stmt_lines: dict[int, int] = {}
            for stmt in ast.walk(tree):
                if isinstance(stmt, ast.stmt) and not isinstance(stmt, compound):
                    for lineno in range(stmt.lineno, stmt.end_lineno + 1):
                        simple_stmt_lines[lineno] = simple_stmt_lines.get(lineno, 0) + 1
            pattern = re.compile(rf"(?<![\w.]){re.escape(import_text)}(?![\w.])")
            matched: list[ast.stmt] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if scope is not None and not (scope[0] <= node.lineno <= scope[1]):
                        continue
                    seg = _slice_span(
                        "".join(lines),
                        node.lineno,
                        node.col_offset,
                        node.end_lineno,
                        node.end_col_offset,
                    )
                    if pattern.search(seg):
                        matched.append(node)
            assert matched, f"import {import_text!r} not found in {rel}"
            whole: list[tuple[int, int]] = []
            shared: list[ast.stmt] = []
            for node in matched:
                alone = all(simple_stmt_lines[lineno] == 1 for lineno in range(node.lineno, node.end_lineno + 1))
                if alone:
                    lo, hi = node.lineno, node.end_lineno
                    if hi < len(lines) and lines[hi].strip() == "":
                        hi += 1
                    whole.append((lo, hi))
                else:
                    shared.append(node)
            text = "".join(lines)
            for node in sorted(shared, key=lambda n: (n.lineno, n.col_offset), reverse=True):
                text = _replace_span(
                    text,
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset,
                    "",
                )
                fixed = _split_keepends(text)
                joined_line = fixed[node.lineno - 1]
                cleaned = re.sub(r";\s*;", ";", joined_line)
                cleaned = re.sub(r"^(\s*);\s*", r"\1", cleaned)
                cleaned = re.sub(r"\s*;(\s*)$", r"\1", cleaned)
                fixed[node.lineno - 1] = cleaned
                text = "".join(fixed)
            lines = _split_keepends(text)
            for lo, hi in sorted(whole, reverse=True):
                del lines[lo - 1 : hi]
            _write_source(path, "".join(lines))

        self.ops.append(op)
        return self

    def remove_imported_name(
        self,
        rel: str,
        *,
        module: str | None,
        name: str,
        asname: str | None = None,
        keep_exploded: bool = False,
    ) -> "Repro":
        # [FUNCTION DOC] Records an operation that removes one imported name from top-level
        # import statements of the file at relative path rel. With module given, only top-level
        # from-imports whose module text (leading dots for relative level, then the module
        # name) equals module are considered; with module nothing, only plain imports. Within
        # them the alias with the given name and given alias (asname) is dropped; statements
        # without it are skipped. At least one statement must change (assertion error otherwise).
        # If it was the only name the whole statement lines are deleted. Otherwise, when the
        # dropped alias sits alone on its own line and (two or more names survive, or the
        # statement contains any comment, or keep_exploded is true), just that alias line is
        # deleted, leaving the other lines intact. Else, if the statement contains comments but
        # the alias shares a line with other text, an assertion error is raised. Else the whole
        # statement is rebuilt on a single line from the surviving names, using the file's
        # newline style. All edits are applied from the bottom up. Returns the builder.
        """Drop a single imported ``name`` from a module-level import: from a ``from module
        import a, b`` keep the rest and drop only ``name``; when it was the sole name -- or for
        a plain ``import name`` (``module=None``) -- remove the whole statement. The symbol's
        home changed, so an importer that no longer references it loses exactly that name; the
        import sorter rewrites the surviving line. An import diff is always whitelisted, so this
        realises a lost name directly instead of relying on the formatter to prune it.

        Removing down to a single surviving name collapses the import to one line by default
        (the common case). Pass ``keep_exploded`` when the target kept the sole survivor
        exploded (its magic trailing comma preserved): the alias line is then merely deleted
        so the surviving name keeps its comma and the formatter leaves the import multi-line.
        The choice is the commit author's and cannot be inferred from the source.
        """

        def alias_text(alias: ast.alias) -> str:
            # [FUNCTION DOC] HELPER: Returns the imported name, followed by the word "as" and
            # the alias if one exists.
            return alias.name + (f" as {alias.asname}" if alias.asname else "")

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation; performs the edit described for
            # the builder method above and writes the file back.
            path = root / rel
            lines = _split_keepends(_read_source(path))
            nl = _newline_style("".join(lines))
            edits: list[tuple[int, int, str | None]] = []
            for node in ast.parse("".join(lines)).body:
                if module is None:
                    if not isinstance(node, ast.Import):
                        continue
                else:
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    if "." * node.level + (node.module or "") != module:
                        continue
                dropped_alias = next(
                    (a for a in node.names if a.name == name and a.asname == asname),
                    None,
                )
                if dropped_alias is None:
                    continue
                kept = [a for a in node.names if a is not dropped_alias]
                if not kept:
                    edits.append((node.lineno, node.end_lineno, None))
                    continue
                stmt_lines = lines[node.lineno - 1 : node.end_lineno]
                own = dropped_alias.lineno
                own_line = lines[own - 1]
                on_own_line = own_line.strip().rstrip(",").strip() == alias_text(dropped_alias)
                has_comments = any("#" in ln for ln in stmt_lines)
                # Preserve the exploded form -- delete just this alias's line -- when the
                # import stays multi-line: 2+ surviving names keep the magic trailing comma
                # exploded, and an import carrying comments must not be rebuilt (a rebuild
                # would drop them). Both match how the target was edited (a flat rebuild
                # would drop the magic comma and collapse an import the target left
                # multi-line). A lone surviving name collapses to one line by default, unless
                # keep_exploded says the target preserved the magic comma for it too.
                if on_own_line and (len(kept) >= 2 or has_comments or keep_exploded):
                    edits.append((own, own, None))
                elif has_comments and not on_own_line:
                    raise AssertionError(
                        f"cannot drop {name!r}: it shares a line with other text and "
                        f"the import holds comments that a rebuild would delete"
                    )
                else:
                    keyword = "import " if module is None else f"from {module} import "
                    rebuilt = keyword + ", ".join(alias_text(a) for a in kept) + nl
                    edits.append((node.lineno, node.end_lineno, rebuilt))
            assert edits, f"import of {name!r} from {module!r} not found in {rel}"
            for lo, hi, repl in sorted(edits, reverse=True):
                if repl is None:
                    del lines[lo - 1 : hi]
                else:
                    lines[lo - 1 : hi] = [repl]
            _write_source(path, "".join(lines))

        self.ops.append(op)
        return self

    def add_imported_name(self, rel: str, *, module: str, name: str, asname: str | None = None) -> "Repro":
        # [FUNCTION DOC] Records an operation that appends one name (with optional alias) to the
        # first top-level from-import of the file at rel whose module text (leading dots for
        # relative level plus module name) equals module. The statement is rebuilt on a single
        # line from its existing names plus the new one, using the file's newline style. Raises
        # an assertion error if the statement holds any comment, if the name is already
        # imported there, or if no such statement exists. Only the first matching statement is
        # changed. Returns the builder.
        """Add a single ``name`` to an existing module-level ``from module import a, b`` --
        the dual of ``remove_imported_name``. A relocated symbol gains a new importer that
        already imports other names from the same module, so the target extends that line
        rather than adding a fresh statement (which the sorter would not merge across an
        intervening non-import statement). The import sorter rewrites the surviving line; an
        import carrying comments is refused, since a rebuild would drop them."""

        def alias_text(target_name: str, target_asname: str | None) -> str:
            # [FUNCTION DOC] HELPER: Returns the name, followed by the word "as" and the alias
            # if one is given.
            return target_name + (f" as {target_asname}" if target_asname else "")

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation; performs the edit described above.
            path = root / rel
            lines = _split_keepends(_read_source(path))
            nl = _newline_style("".join(lines))
            for node in ast.parse("".join(lines)).body:
                if not isinstance(node, ast.ImportFrom):
                    continue
                if "." * node.level + (node.module or "") != module:
                    continue
                stmt_lines = lines[node.lineno - 1 : node.end_lineno]
                if any("#" in ln for ln in stmt_lines):
                    raise AssertionError(
                        f"cannot add {name!r} to the import from {module!r} in {rel}: "
                        f"it holds comments that a rebuild would delete"
                    )
                existing = [alias_text(a.name, a.asname) for a in node.names]
                added = alias_text(name, asname)
                assert added not in existing, f"{name!r} already imported from {module!r} in {rel}"
                rebuilt = f"from {module} import " + ", ".join(existing + [added]) + nl
                lines[node.lineno - 1 : node.end_lineno] = [rebuilt]
                _write_source(path, "".join(lines))
                return
            raise AssertionError(f"no `from {module} import` statement in {rel}")

        self.ops.append(op)
        return self

    def add_import(self, rel: str, import_stmt: str, *, after: str | None = None) -> "Repro":
        # [FUNCTION DOC] Records an operation that inserts one import statement line (the given
        # statement text plus the file's newline style) into the file at relative path rel.
        # Without the after option the line goes right after the last top-level import statement
        # (imports nested inside other statements are ignored); if the file starts with a
        # docstring the line goes at least after the docstring; if there is neither, it goes at
        # the very top of the file. With after given, the line goes immediately after the first
        # top-level import statement whose raw source text contains that substring (assertion
        # error if none). No blank line or sort order is handled; the formatter does that.
        # Returns the builder.
        """Append an import after the last top-level import; the formatter's import sorter
        places it (so the exact insertion point does not matter). When ``after`` is given,
        insert immediately after the top-level import statement whose source text contains
        that substring instead -- needed for a file whose imports are split into separate
        isort sections by an intervening statement (e.g. ``_is_hip = is_hip()``), where the
        sorter will not carry the new import across the boundary into the intended block.
        """

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation; inserts the import line as
            # described for the builder method above and writes the file back.
            path = root / rel
            lines = _split_keepends(_read_source(path))
            nl = _newline_style("".join(lines))
            body = ast.parse("".join(lines)).body
            if after is not None:
                anchor = None
                for node in body:
                    if isinstance(node, (ast.Import, ast.ImportFrom)) and after in "".join(
                        lines[node.lineno - 1 : node.end_lineno]
                    ):
                        anchor = node
                        break
                if anchor is None:
                    raise AssertionError(f"no top-level import containing {after!r} in {rel}")
                at = anchor.end_lineno
                _write_source(path, "".join(lines[:at] + [import_stmt + nl] + lines[at:]))
                return
            last = 0
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                last = body[0].end_lineno
            for node in body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    last = max(last, node.end_lineno)
            _write_source(path, "".join(lines[:last] + [import_stmt + nl] + lines[last:]))

        self.ops.append(op)
        return self

    def add_typechecking_import(self, rel: str, import_stmt: str) -> "Repro":
        # [FUNCTION DOC] Records an operation that adds one import statement line to the
        # type-checking-only block of the file at relative path rel. It looks at top-level
        # conditionals whose test is the type-checking flag (plain or qualified by the typing
        # module); in the first one found, the new line is indented to match the block's first
        # statement. If the block's only statement is a placeholder pass, that pass is replaced
        # by the import; otherwise the import is inserted right after the block's last body
        # statement. If no such block exists, a new one is created right after the last top-level
        # import: a blank line, the conditional header, and the import indented by four spaces
        # (the flag name itself is not imported by this operation; an assertion error is raised
        # if the file has no top-level import to anchor to). Uses the file's newline style.
        # Returns the builder.
        """Append ``import_stmt`` inside the file's ``if TYPE_CHECKING:`` block -- a moved
        definition whose annotations reference a type needs that type imported there. The
        import sorter orders the block, so the exact insertion point does not matter. A lone
        ``pass`` placeholder (the block's only statement) is dropped: populating an empty
        ``TYPE_CHECKING`` block makes its placeholder redundant, so the target removes it.
        With no existing block, one is created after the trailing module import -- the
        destination gains the guard together with its first import.
        """

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation; edits the file as described for the
            # builder method above and writes it back.
            path = root / rel
            lines = _split_keepends(_read_source(path))
            nl = _newline_style("".join(lines))
            for node in ast.parse("".join(lines)).body:
                if isinstance(node, ast.If) and ast.unparse(node.test) in (
                    "TYPE_CHECKING",
                    "typing.TYPE_CHECKING",
                ):
                    indent = " " * node.body[0].col_offset
                    lone_pass = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
                    if lone_pass:
                        placeholder = node.body[0]
                        lines[placeholder.lineno - 1 : placeholder.end_lineno] = [indent + import_stmt + nl]
                    else:
                        lines.insert(node.body[-1].end_lineno, indent + import_stmt + nl)
                    _write_source(path, "".join(lines))
                    return
            tree = ast.parse("".join(lines))
            imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
            assert imports, f"no imports to anchor a new `if TYPE_CHECKING:` block in {rel}"
            insert_at = imports[-1].end_lineno
            lines[insert_at:insert_at] = [
                nl,
                "if TYPE_CHECKING:" + nl,
                "    " + import_stmt + nl,
            ]
            _write_source(path, "".join(lines))

        self.ops.append(op)
        return self

    def repath_import(self, rel: str, *, old_module: str, new_module: str, name: str) -> "Repro":
        # [FUNCTION DOC] Records an operation that changes the module of from-imports in the
        # file at relative path rel. It considers only from-import statements that are not
        # direct children of the module body (nested in functions, conditionals and so on) whose
        # module attribute equals old_module (compared without any leading dots of a relative
        # import) and whose imported names include name. On the statement's first line only, the
        # first occurrence of the text "from <original spelling, with relative dots> import" is
        # replaced by "from <new_module> import"; the new module is always written as given
        # (never relative). An assertion error is raised if that text is not found on the line,
        # or if no statement was changed at all. Other names in the same statement move with it.
        # Returns the builder.
        """Repath every function-scoped ``from old_module import ... name ...`` to
        ``from new_module import ...`` in place -- the moved symbol's home changed, so its
        importer adjusts. Only imports nested below module level are touched; a module-level
        import is left to the import sorter via add_import / remove_import."""

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation; rewrites matching nested imports as
            # described for the builder method above and writes the file back.
            path = root / rel
            lines = _split_keepends(_read_source(path))
            tree = ast.parse("".join(lines))
            top_level = {id(node) for node in tree.body}
            changed = False
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and id(node) not in top_level
                    and node.module == old_module
                    and any(alias.name == name for alias in node.names)
                ):
                    spelled = "." * node.level + (node.module or "")
                    replaced = lines[node.lineno - 1].replace(f"from {spelled} import", f"from {new_module} import", 1)
                    assert (
                        replaced != lines[node.lineno - 1]
                    ), f"import spelling {spelled!r} not found on its line in {rel}"
                    lines[node.lineno - 1] = replaced
                    changed = True
            assert changed, f"nested import of {name} from {old_module} not in {rel}"
            _write_source(path, "".join(lines))

        self.ops.append(op)
        return self

    def move_assign(
        self,
        name: str,
        *,
        src: str,
        dst: str,
        before: str | None = None,
    ) -> "Repro":
        # [FUNCTION DOC] Records an operation that moves one module-level assignment from file
        # src to file dst (both relative paths). Source side: among top-level statements, the
        # plain assignment with exactly one plain-name target equal to name, or the annotated
        # assignment to that name; if several match, the last one wins; assertion error if none.
        # Its lines are deleted from src (surrounding blank lines stay). Destination side (the
        # destination is read after the source was written): with before given, the original
        # lines followed by one blank line are inserted immediately above the top-level
        # definition named before (decorators included), found among classes, functions and
        # single-target plain assignments (assertion error if not found); without before, a blank
        # line and then the original lines are inserted right after the last top-level import, or
        # at the very top if there are no imports. Returns the builder.
        # KNOWN ISSUE: an annotated assignment cannot serve as the before anchor, only the
        # definition kinds and single-target plain assignments are recognized there.
        """Cut the module-level assignment binding ``name`` from ``src`` and paste it verbatim
        into ``dst`` at module level -- a module constant relocated together with the code
        that reads it. Pasted immediately above the top-level statement named ``before``
        when given, else after the last top-level import."""

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation; cuts from the source file and pastes
            # into the destination file as described for the builder method above.
            src_path = root / src
            dst_path = root / dst
            src_lines = _split_keepends(_read_source(src_path))
            node = None
            for cand in ast.parse("".join(src_lines)).body:
                if (
                    isinstance(cand, ast.Assign)
                    and len(cand.targets) == 1
                    and isinstance(cand.targets[0], ast.Name)
                    and cand.targets[0].id == name
                ) or (
                    isinstance(cand, ast.AnnAssign) and isinstance(cand.target, ast.Name) and cand.target.id == name
                ):
                    node = cand
            assert node is not None, f"module assignment {name} not found in {src}"
            block = "".join(src_lines[node.lineno - 1 : node.end_lineno])
            _write_source(
                src_path,
                "".join(src_lines[: node.lineno - 1] + src_lines[node.end_lineno :]),
            )

            dst_lines = _split_keepends(_read_source(dst_path))
            dst_nl = _newline_style("".join(dst_lines))
            dst_tree = ast.parse("".join(dst_lines))
            at = None
            if before is not None:
                for cand in dst_tree.body:
                    cand_name = getattr(cand, "name", None) or (
                        cand.targets[0].id
                        if isinstance(cand, ast.Assign)
                        and len(cand.targets) == 1
                        and isinstance(cand.targets[0], ast.Name)
                        else None
                    )
                    if cand_name == before:
                        at = (
                            min(
                                [d.lineno for d in getattr(cand, "decorator_list", [])],
                                default=cand.lineno,
                            )
                            - 1
                        )
                        break
                assert at is not None, f"before={before!r} not found in {dst}"
                dst_lines[at:at] = [block, dst_nl]
            else:
                imports = [n for n in dst_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
                at = imports[-1].end_lineno if imports else 0
                dst_lines[at:at] = [dst_nl, block]
            _write_source(dst_path, "".join(dst_lines))

        self.ops.append(op)
        return self

    def move_symbol(
        self,
        name: str,
        *,
        src: str,
        dst: str,
        into_class: str | None,
        from_class: str | None = None,
        dedent: int = 0,
        drop_self_annotation: bool = False,
        before: str | None = None,
        after: str | None = None,
        leave_delegate: str | None = None,
        delegate_name: str | None = None,
    ) -> "Repro":
        # [FUNCTION DOC] Records an operation that moves one function, async function or class
        # definition (with its decorators) from file src to file dst. Raises an assertion error
        # immediately (at recording time) if both before and after are given. The into_class
        # argument is required (pass nothing for module level). Source lookup: the definition
        # called name, restricted to the class from_class if given, must be unique (see the
        # unique-definition helper). Source edit: with no leave_delegate the definition lines are
        # deleted. With leave_delegate (the name of an attribute of self) they are replaced by a
        # stub that keeps the original header and forwards to that attribute: the stub body is a
        # single return (return await for async definitions) of a call to self.<attribute>.<
        # delegate_name or name> passing every positional parameter except self by position,
        # then star-args, every keyword-only parameter as a same-named keyword, and double-star
        # kwargs. The header text runs from the definition line to the line whose top-level colon
        # closes the signature (found by tokenizing, so comments between header and body are not
        # absorbed). If the method is decorated with the static or class method decorator the stub
        # starts at the definition line, dropping ALL its decorators; leave_delegate on such a
        # method additionally requires the first parameter to be an annotated self (assertion
        # error otherwise). The stub drops the annotation of self only when the annotation (name,
        # string, or last component of a dotted name) equals into_class. Moved text: the
        # definition lines minus any static/class method decorator lines, shifted by dedent
        # (positive removes that many leading spaces where present; negative adds that many to
        # non-blank lines), and with the annotation of self removed when drop_self_annotation is
        # set. Destination: located inside class into_class (assertion error if missing) or at
        # module level. With after, inserted (preceded by a blank line) right after the first
        # container statement defining the symbol after; with before, inserted (followed by a
        # blank line) immediately above the named sibling def/class including its decorators
        # (assertion error if not found); otherwise appended after the last class member, or at
        # the end of the file at module level, preceded by a blank line. The source is written
        # before the destination is read, so src may equal dst. Returns the builder.
        """Cut ``def name`` (with decorators) from ``src`` and paste it into ``dst`` --
        immediately above the sibling def ``before`` when given (so the relocated def lands in
        the chain's order), immediately below the top-level symbol ``after`` when given (a
        sibling def/class or a module-level assignment target -- used to land the def just
        before a following ``if TYPE_CHECKING:`` guard, which is not a nameable anchor), else
        at the end of ``into_class`` (or module level when None) -- dropping a move decorator
        and dedenting by ``dedent``. When ``drop_self_annotation``, the moved method's
        ``self: Target`` annotation is dropped (redundant inside the class). The body is moved
        verbatim; the formatter normalises the surrounding blank lines.
        """
        assert before is None or after is None, "move_symbol: before and after are mutually exclusive"

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation; cuts the definition from the source
            # file (optionally leaving a delegating stub) and pastes the adjusted text into the
            # destination file, as described for the builder method above.
            src_path = root / src
            dst_path = root / dst
            src_lines = _split_keepends(_read_source(src_path))
            src_nl = _newline_style("".join(src_lines))
            node = _find_unique_def(ast.parse("".join(src_lines)), name, from_class=from_class, where=src)
            start, end = _def_span(node)
            block = src_lines[start - 1 : end]
            decorator_lines = node.lineno - start
            if leave_delegate is not None:
                args = node.args
                has_move_decorator = any(ln.strip() in _MOVE_DECORATORS for ln in block[:decorator_lines])
                arg_list = args.posonlyargs + args.args
                self_annotated = bool(arg_list) and arg_list[0].arg == "self" and arg_list[0].annotation is not None
                assert not has_move_decorator or self_annotated, (
                    f"leave_delegate on a {_MOVE_DECORATORS} method has no self to "
                    "forward (a de-self'd staticmethod must annotate its self param)"
                )
                parts = [p.arg for p in args.posonlyargs + args.args if p.arg != "self"]
                if args.vararg is not None:
                    parts.append(f"*{args.vararg.arg}")
                parts += [f"{k.arg}={k.arg}" for k in args.kwonlyargs]
                if args.kwarg is not None:
                    parts.append(f"**{args.kwarg.arg}")
                # The signature spans the def header only (def line through the line whose
                # colon opens the body). node.body[0].lineno would skip over any leading
                # comment/blank lines (not AST nodes), wrongly absorbing them into the
                # delegate, so the header end is found by tokenizing the def.
                header_end = node.lineno - 1 + _def_header_end("".join(src_lines[node.lineno - 1 : end]))
                sig_start = node.lineno - 1 if has_move_decorator else start - 1
                signature_text = "".join(src_lines[sig_start:header_end])
                # The stub drops the self annotation only when it names the class the
                # def moved into (now redundant); an unrelated annotation (a mixin's
                # `self: ModelRunner`) is part of the surviving header and stays.
                ann = arg_list[0].annotation if self_annotated else None
                ann_name = None
                if isinstance(ann, ast.Name):
                    ann_name = ann.id
                elif isinstance(ann, ast.Constant) and isinstance(ann.value, str):
                    ann_name = ann.value.split(".")[-1]
                elif isinstance(ann, ast.Attribute):
                    ann_name = ann.attr
                if self_annotated and ann_name == into_class:
                    sig_indent = len(signature_text) - len(signature_text.lstrip(" "))
                    parsable = signature_text + " " * sig_indent + "    pass" + src_nl
                    stripped = _drop_self_annotation(parsable, name)
                    assert stripped.endswith(" " * sig_indent + "    pass" + src_nl)
                    signature_text = stripped[: -len(" " * sig_indent + "    pass" + src_nl)]
                body_indent = " " * node.body[0].col_offset
                returning = "return await" if isinstance(node, ast.AsyncFunctionDef) else "return"
                forward = (
                    f"{body_indent}{returning} self.{leave_delegate}."
                    f"{delegate_name or name}({', '.join(parts)})" + src_nl
                )
                delegate = signature_text + forward
                _write_source(
                    src_path,
                    "".join(src_lines[: start - 1] + [delegate] + src_lines[end:]),
                )
            else:
                _write_source(src_path, "".join(src_lines[: start - 1] + src_lines[end:]))

            kept = [
                ln
                for index, ln in enumerate(block)
                if not (index < decorator_lines and ln.strip() in _MOVE_DECORATORS)
            ]
            if dedent > 0:
                kept = [ln[dedent:] if ln[:dedent] == " " * dedent else ln for ln in kept]
            elif dedent < 0:
                pad = " " * -dedent
                kept = [pad + ln if ln.strip() else ln for ln in kept]
            method_text = "".join(kept)
            if drop_self_annotation:
                method_text = _drop_self_annotation(method_text, name)

            dst_lines = _split_keepends(_read_source(dst_path))
            dst_nl = _newline_style("".join(dst_lines))
            dst_tree = ast.parse("".join(dst_lines))
            container = dst_tree.body
            if into_class is not None:
                cls = _find_class(dst_tree, into_class)
                assert cls is not None, f"class {into_class} not found in {dst}"
                container = cls.body
            target = None
            if before is not None:
                target = next(
                    (
                        n
                        for n in container
                        if isinstance(
                            n,
                            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                        )
                        and n.name == before
                    ),
                    None,
                )
                assert target is not None, f"before={before!r} not found in {dst}"
            if after is not None:
                anchor = next(
                    (n for n in container if _symbol_named(n, after)),
                    None,
                )
                assert anchor is not None, f"after={after!r} not found in {dst}"
                at = anchor.end_lineno
                _write_source(
                    dst_path,
                    "".join(dst_lines[:at] + [dst_nl, method_text] + dst_lines[at:]),
                )
            elif target is not None:
                at = _def_span(target)[0] - 1
                _write_source(
                    dst_path,
                    "".join(dst_lines[:at] + [method_text, dst_nl] + dst_lines[at:]),
                )
            else:
                at = container[-1].end_lineno if into_class is not None else len(dst_lines)
                _write_source(
                    dst_path,
                    "".join(dst_lines[:at] + [dst_nl, method_text] + dst_lines[at:]),
                )

        self.ops.append(op)
        return self

    def extract_to_new_module(
        self,
        src: str,
        dst: str,
        *,
        symbols: list[str],
        future_import: bool = True,
    ) -> "Repro":
        # [FUNCTION DOC] Records an operation that splits the end of file src into a new module
        # dst (relative paths). It walks the top-level statements backwards from the end,
        # including each one that is either a function or class named in symbols, or scaffolding
        # (any import, a conditional on the type-checking flag, or an assignment whose targets
        # are all plain names), and stops at the first statement that is neither. The included
        # statements form the cut tail; every listed symbol must be a definition inside it
        # (assertion error naming the missing ones). The cut starts at the first tail statement's
        # first line (including its decorators); any comment or blank line directly above stays
        # in src. Everything from there to the end of src is removed from src and written to dst
        # (parent directories created, an existing file overwritten), preceded by a future
        # annotations import line when future_import is true (default). Returns the builder.
        # KNOWN ISSUE: with an empty symbols list and no scaffolding at the file end the tail is
        # empty and the first-element access raises an index error instead of a clear message.
        """Cut the contiguous tail of ``src`` -- the moved ``symbols`` and the module
        scaffolding that leads into them (imports, a ``TYPE_CHECKING`` guard, a logger,
        module constants) -- and write it as the new module ``dst``, prepending
        ``from __future__ import annotations`` when ``future_import``. The body is moved
        verbatim; the formatter sorts the imports and normalises the blank lines."""

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation; computes the tail, truncates the
            # source file and writes the new module, as described for the builder method above.
            src_path = root / src
            dst_path = root / dst
            src_lines = _split_keepends(_read_source(src_path))
            body = ast.parse("".join(src_lines)).body
            wanted = set(symbols)

            def is_scaffolding(node: ast.stmt) -> bool:
                # [FUNCTION DOC] HELPER: True for an import, a conditional whose test is the
                # type-checking flag (plain or typing-qualified; the body is not inspected), a
                # plain assignment whose targets are all simple names, or an annotated
                # assignment to a simple name; false for everything else.
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    return True
                if isinstance(node, ast.If):
                    return ast.unparse(node.test) in (
                        "TYPE_CHECKING",
                        "typing.TYPE_CHECKING",
                    )
                if isinstance(node, ast.Assign):
                    return all(isinstance(x, ast.Name) for x in node.targets)
                if isinstance(node, ast.AnnAssign):
                    return isinstance(node.target, ast.Name)
                return False

            cut = len(body)
            while cut > 0:
                node = body[cut - 1]
                is_symbol = (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in wanted
                )
                if is_symbol or is_scaffolding(node):
                    cut -= 1
                else:
                    break
            tail = body[cut:]
            present = {
                node.name for node in tail if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            assert wanted <= present, f"{wanted - present} not in the cut tail of {src}"

            decorators = getattr(tail[0], "decorator_list", [])
            start = min([tail[0].lineno] + [d.lineno for d in decorators])
            block = "".join(src_lines[start - 1 :])
            _write_source(src_path, "".join(src_lines[: start - 1]))

            nl = _newline_style(block)
            prefix = "from __future__ import annotations" + nl if future_import else ""
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            _write_source(dst_path, prefix + block)

        self.ops.append(op)
        return self

    def extract_symbols_to_new_module(
        self,
        src: str,
        dst: str,
        *,
        symbols: list[str],
        header: str,
        order: list[str],
        drop_assigns: list[str] | None = None,
    ) -> "Repro":
        # [FUNCTION DOC] Records an operation that moves the named top-level functions and
        # classes from anywhere in file src into a new module dst, preceded by an authored header.
        # Preconditions (assertion errors): order must contain exactly the same names as symbols;
        # every symbol must be a top-level function or class of src; every name in drop_assigns
        # (optional) must be assigned at top level of src, and any statement assigning such a name
        # must have only plain-name targets. Steps: (1) each symbol's block is the exact line
        # range including decorators. (2) For top-level assignments (plain or annotated) naming
        # any dropped name, remember each dropped name's value as regenerated source text; if the
        # statement also binds surviving names it is rewritten as a chained assignment of just
        # the surviving names to the original value text, otherwise its lines are deleted. (3)
        # Build the map of constants that survive in src (every other top-level assignment's
        # names to value text; later statements override earlier ones, and statements touching
        # dropped names are excluded). (4) If header has non-blank content or any assignment was
        # dropped, audit the header with the audit helper using those two maps. (5) Delete all
        # symbol blocks and dropped assignments from src, bottom up, and write src. (6) Write
        # dst (parent directories created) as the header with trailing line breaks removed
        # followed by two blank lines (only if the header is non-blank), then the blocks in the
        # given order separated by two blank lines with trailing line breaks stripped, ending in a
        # single newline. Comments directly above moved definitions stay in src. Returns the
        # builder.
        """Relocate the named top-level defs/classes from *scattered* positions in ``src`` into
        a new module ``dst`` whose authored ``header`` (the imports, module constants, a logger,
        a ``TYPE_CHECKING`` block -- harmless or re-derived boilerplate reproduced from the
        target) precedes them. Unlike ``extract_to_new_module``, the symbols need not be a
        contiguous tail: each is cut from ``src`` verbatim, so its body stays a proven
        relocation, and the cut blocks are appended in ``order`` (their order in the target).
        ``drop_assigns`` names module-level assignments (e.g. a ``_is_hip = is_hip()`` constant)
        that moved into the new module's header, so they are deleted from ``src`` too -- their
        relocated copy is reproduced in the authored ``header``. The formatter normalises the
        spacing; the byte diff then certifies the bodies are exactly the source's, while only
        the small header is authored."""

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation; performs the steps described for the
            # builder method above on the source and destination files.
            src_path = root / src
            dst_path = root / dst
            src_lines = _split_keepends(_read_source(src_path))
            src_nl = _newline_style("".join(src_lines))
            wanted = set(symbols)
            dropped = set(drop_assigns or [])
            assert set(order) == wanted, f"order {order} must permute symbols {symbols}"
            tree = ast.parse("".join(src_lines))
            nodes = {
                node.name: node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in wanted
            }
            missing = wanted - set(nodes)
            assert not missing, f"{missing} not top-level defs/classes in {src}"
            spans = {name: _def_span(node) for name, node in nodes.items()}
            blocks = {name: "".join(src_lines[start - 1 : end]) for name, (start, end) in spans.items()}
            assign_spans: list[tuple[int, int]] = []
            assign_rewrites: list[tuple[int, int, str]] = []
            removed_assigns: dict[str, str] = {}
            found_assigns: set[str] = set()
            for node in tree.body:
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target] if isinstance(node, ast.AnnAssign) else []
                )
                names = {t.id for t in targets if isinstance(t, ast.Name)}
                hit = names & dropped
                if not hit:
                    continue
                assert len(names) == len(targets), f"drop_assigns {sorted(hit)}: non-name targets in {src}"
                value_src = ast.unparse(node.value) if node.value is not None else None
                for dropped_name in hit:
                    removed_assigns[dropped_name] = value_src
                surviving = [x.id for x in targets if isinstance(x, ast.Name) and x.id not in dropped]
                if surviving:
                    kept_stmt = (
                        " = ".join(surviving)
                        + " = "
                        + _slice_span(
                            "".join(src_lines),
                            node.value.lineno,
                            node.value.col_offset,
                            node.value.end_lineno,
                            node.value.end_col_offset,
                        )
                        + src_nl
                    )
                    assign_rewrites.append((node.lineno, node.end_lineno, kept_stmt))
                else:
                    assign_spans.append((node.lineno, node.end_lineno))
                found_assigns |= hit
            assert found_assigns == dropped, f"{dropped - found_assigns} not assigned in {src}"
            rederivable: dict[str, str | None] = {}
            for node in tree.body:
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target] if isinstance(node, ast.AnnAssign) else []
                )
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if not names or set(names) & dropped:
                    continue
                value_src = ast.unparse(node.value) if node.value is not None else None
                for kept_name in names:
                    rederivable[kept_name] = value_src
            if header.strip() or removed_assigns:
                _audit_extract_header(header, removed_assigns, where=dst, rederivable=rederivable)
            cuts = [(start, end, None) for start, end in spans.values()]
            cuts += [(start, end, None) for start, end in assign_spans]
            cuts += assign_rewrites
            for start, end, repl in sorted(cuts, key=lambda c: (c[0], c[1]), reverse=True):
                if repl is None:
                    del src_lines[start - 1 : end]
                else:
                    src_lines[start - 1 : end] = [repl]
            _write_source(src_path, "".join(src_lines))

            gap = src_nl * 3
            relocated = gap.join(blocks[name].rstrip("\r\n") for name in order)
            prefix = header.rstrip("\r\n") + gap if header.strip() else ""
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            _write_source(dst_path, prefix + relocated + src_nl)

        self.ops.append(op)
        return self

    def extract_function(
        self,
        src: str,
        dst: str,
        *,
        name: str,
        signature: str,
        body: str,
        body_indent: int,
        call: str,
        return_text: str | None = None,
        before: str | None = None,
        into_class: str | None = None,
    ) -> "Repro":
        # [FUNCTION DOC] Records an operation that turns an inline block of statements in file
        # src into a new function in file dst. Inputs: name (informational only; the function
        # name comes from the authored signature text), signature (full header text, which may be
        # indented and may span lines), body (the exact original block text, which must occur
        # exactly once in src and start at a line start), body_indent (number of spaces of
        # indentation the block has in src), call (replacement text for the block in src,
        # authored), optional return_text (extra lines appended to the new function, authored),
        # optional before (sibling def/class above which to insert) and into_class. Steps: assert
        # the block is unique and begins at a line boundary; replace the first occurrence in src
        # by call and write src; build the function as the signature (trailing line breaks
        # trimmed) then the re-indented block, then return_text if any, ending with exactly one
        # newline. Re-indentation shifts every line by the signature's own indentation plus four
        # minus body_indent; lines inside multi-line string literals are left alone, a positive
        # shift skips blank lines, a negative shift only trims lines that start with enough
        # spaces. Insertion into dst (read after src was written, so they may be the same
        # file), where the container is class into_class (assertion error if missing) or the
        # module: above the sibling named before (including decorators) followed by a blank line,
        # otherwise appended after the last member of the class or at the end of the file,
        # preceded by a blank line. Returns the builder.
        # KNOWN ISSUE: the name parameter is never used, and a before sibling that is not found is
        # silently ignored (the function is appended instead) whereas the symbol-move primitive
        # raises an assertion error in that case.
        """Extract an inline block into a new ``name`` function. The block ``body`` is cut from
        ``src`` *verbatim* (so the byte diff certifies the function body is exactly the source's),
        re-indented from ``body_indent`` to a function-body indent, and wrapped under the
        authored ``signature`` (with ``return_text`` appended when given); the def is inserted
        into ``dst`` (above the sibling ``before`` or at the end of ``into_class`` / module), and
        the block in ``src`` is replaced by the authored ``call``.

        This is the certifiable core of an extract-function: the bulk (the relocated body) is
        machine-checked, and only the small signature/return/call interface is authored. It is
        faithful **only** when the body is moved unchanged -- a de-self (``self.x`` -> a
        parameter), a control-flow restructure, or a bookkeeping consolidation must be done as a
        separate semantic commit first, since those are not relocations (see
        guide-split.md)."""

        def reindent(text: str, shift: int) -> str:
            # [FUNCTION DOC] HELPER: Shifts the indentation of a block of text by shift spaces
            # (relative to the enclosing body_indent for finding string interiors). Zero returns
            # the text unchanged. Lines lying inside multi-line string literals (determined on the
            # block dedented by body_indent) are never touched. Negative: removes that many
            # leading spaces from each other line that has at least that many leading spaces.
            # Positive: prepends that many spaces to each other non-blank line.
            if shift == 0:
                return text
            interior = _multiline_string_interior_lines(dedent(text, body_indent))
            lines = _split_keepends(text)
            if shift < 0:
                return "".join(
                    (line[-shift:] if index + 1 not in interior and line[:-shift] == " " * -shift else line)
                    for index, line in enumerate(lines)
                )
            pad = " " * shift
            return "".join(
                pad + line if line.strip() and index + 1 not in interior else line for index, line in enumerate(lines)
            )

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation; replaces the block in the source file
            # by the call text and inserts the new function into the destination file, as
            # described for the builder method above.
            src_path = root / src
            src_text = _read_source(src_path)
            assert src_text.count(body) == 1, f"block not found uniquely in {src}"
            at = src_text.find(body)
            assert (
                at == 0 or src_text[at - 1] == "\n"
            ), f"block matches mid-line in {src}; it must start at a line boundary"
            _write_source(src_path, src_text.replace(body, call, 1))

            dst_path = root / dst
            dst_lines = _split_keepends(_read_source(dst_path))
            dst_nl = _newline_style("".join(dst_lines))
            sig_first = _split_keepends(signature)[0]
            sig_indent = len(sig_first) - len(sig_first.lstrip(" "))
            function = signature.rstrip("\r\n") + dst_nl + reindent(body, sig_indent + 4 - body_indent)
            if return_text is not None:
                function = function.rstrip("\r\n") + dst_nl + return_text
            function = function.rstrip("\r\n") + dst_nl

            dst_tree = ast.parse("".join(dst_lines))
            container = dst_tree.body
            if into_class is not None:
                cls = _find_class(dst_tree, into_class)
                assert cls is not None, f"class {into_class} not found in {dst}"
                container = cls.body
            anchor = None
            if before is not None:
                anchor = next(
                    (
                        node
                        for node in container
                        if isinstance(
                            node,
                            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                        )
                        and node.name == before
                    ),
                    None,
                )
            if anchor is not None:
                at = _def_span(anchor)[0] - 1
                _write_source(
                    dst_path,
                    "".join(dst_lines[:at] + [function, dst_nl] + dst_lines[at:]),
                )
            else:
                at = container[-1].end_lineno if into_class is not None else len(dst_lines)
                _write_source(
                    dst_path,
                    "".join(dst_lines[:at] + [dst_nl, function] + dst_lines[at:]),
                )

        self.ops.append(op)
        return self

    def delete_file(self, path: str) -> "Repro":
        # [FUNCTION DOC] Records an operation that deletes the file at relative path path if it
        # holds only scaffolding. If the file does not exist nothing happens (no error). Otherwise
        # every top-level statement must be an import, a bare string expression (docstring), a
        # conditional on the type-checking flag (plain or typing-qualified), or an assignment of
        # the standard module logger; anything else makes an assertion error that lists up to
        # three offending statements as regenerated source text. Then the file is removed.
        # Returns the builder.
        # KNOWN ISSUE: the body of a type-checking conditional is not inspected, so live code
        # placed inside such a block would be deleted without complaint.
        """Delete a source module that its symbols' relocation left empty (the chain deletes
        the leftover scaffolding-only file). Run after the moves that empty it. Refuses a
        file that still holds anything beyond a docstring, imports, a TYPE_CHECKING block, or
        a bare module ``logger`` -- deleting live code is not a relocation."""

        def is_module_logger(stmt: ast.stmt) -> bool:
            # [FUNCTION DOC] HELPER: True when the statement is an assignment (any targets) whose
            # value, as regenerated source text, is the standard library logger lookup that
            # takes the module name.
            return (
                isinstance(stmt, ast.Assign)
                and stmt.value is not None
                and ast.unparse(stmt.value) == "logging.getLogger(__name__)"
            )

        def op(root: Path) -> None:
            # [FUNCTION DOC] HELPER: The recorded operation; checks the file and deletes it as
            # described for the builder method above.
            target = root / path
            if not target.exists():
                return
            leftover = [
                ast.unparse(stmt)
                for stmt in ast.parse(_read_source(target)).body
                if not (
                    isinstance(stmt, (ast.Import, ast.ImportFrom))
                    or (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)
                    )
                    or (
                        isinstance(stmt, ast.If)
                        and ast.unparse(stmt.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING")
                    )
                    or is_module_logger(stmt)
                )
            ]
            assert not leftover, f"{path} still holds non-scaffolding code, refusing to delete: " f"{leftover[:3]}"
            target.unlink()

        self.ops.append(op)
        return self

    def run(self) -> str:
        # [FUNCTION DOC] Executes the recorded recipe and reports. Uses the configured repository
        # root, or finds it from the current directory. Creates a temporary directory whose name
        # starts with "repro-" and a git worktree on a new branch of the same name at the base
        # commit; applies every recorded operation in order to the worktree path; stages all,
        # runs pre-commit on changed files (added, copied, modified or renamed relative to base;
        # its failure is ignored); commits any resulting changes with message "repro"; diffs the
        # worktree against the target commit. Prints a residual message with the diff line count
        # and the diff when non-empty, or a PASS message when empty. Returns the diff text (empty
        # string on a clean reproduction). A finally block always force-removes the worktree and
        # deletes the branch (failures ignored), unlike the standalone verify function which
        # leaves them behind. Does not exit the process; exceptions from operations propagate.
        """Apply the operations to a worktree at base, run pre-commit, diff against target.
        Returns the residual diff ("" on a clean reproduction)."""
        repo_root = self.repo_root or exec_command("git rev-parse --show-toplevel")
        worktree = tempfile.mkdtemp(prefix="repro-")
        branch = Path(worktree).name
        try:
            exec_command(f"git worktree add -b {branch} {worktree} {self.base}", cwd=repo_root)
            for op in self.ops:
                op(Path(worktree))
            exec_command("git add -A", cwd=worktree)
            changed = exec_command(
                f"git diff --cached --name-only --diff-filter=ACMR {self.base}",
                cwd=worktree,
            ).split()
            if changed:
                files = " ".join(shlex.quote(path) for path in changed)
                exec_command(f"pre-commit run --files {files}", cwd=worktree, check=False)
            if exec_command("git status --porcelain", cwd=worktree):
                git_add_and_commit("repro", cwd=worktree)
            diff = exec_command(f"git diff {self.target} -- .", cwd=worktree, check=False)
            if diff:
                print(f"\nRESIDUAL ({len(diff.splitlines())} lines):\n{diff}")
            else:
                print("\nPASS: reproduces the commit byte-for-byte.")
            return diff
        finally:
            exec_command(f"git worktree remove --force {worktree}", cwd=repo_root, check=False)
            exec_command(f"git branch -D {branch}", cwd=repo_root, check=False)
