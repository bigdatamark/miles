# [MODULE DOC] Proof generator for the mechanical refactor verifier. For a git commit that
# relocates code, it (1) infers a "recipe": a structured description of the primitives of the
# reproduction utilities module needed to rebuild the commit from its parent, derived from the
# commit's diff plus the syntax trees of the files before and after; (2) renders that recipe as
# a standalone python script (named after the commit id prefix) which builds the recipe object
# from the utilities module, runs it, and exits non-zero unless the byte-for-byte diff against
# the commit is empty; (3) optionally runs that script and records output; and (4) for a range
# of commits, writes a folder of scripts plus a plain text log and an HTML report. Inference
# supports these commit shapes: a method moved onto an existing class (call sites lowered), a
# method moved to a free module-level function (call sites requalified), a free function moved
# to another module (importers repathed), a trailing block of a module cut into a new module,
# scattered symbols gathered into a new module, a module-level assignment moved, and an inline
# block extracted into a new function; renames and statement-level reorders are reported as
# unsupported (recipe flagged unsupported with notes). Imports gained or lost at module level
# are inferred symmetrically from before/after trees. Can be run as a command line program
# with a single commit or a base..tip range. Reads git state through subprocesses and never
# modifies the repository itself (the generated scripts do, in a temporary worktree).
"""Infer a faithful reproduce recipe for a move commit, then emit and run a self-contained
reproduce script. Lets the verifier turn a commit a formatter re-wrapped into an auditable,
runnable reproduce script -- no one hand-writes it.

A recipe is inferred from the commit's diff and its before-state AST: which symbols moved
(src -> dst, into which class, or into a new module), which call sites were adapted, which
imports were repathed, and the symmetric module-level import diff each file gained or lost
(realised directly with add_import / remove_imported_name, since an import diff is always
whitelisted).
``recipe_to_script`` emits a standalone ``repro_scripts/<sha>.py`` (importing only the
reproduce util); running it reproduces the commit, diffs it byte-for-byte, and exits
non-zero unless the diff is empty (PASS).
``generate_range`` writes a whole folder (scripts + output.log + output.html) for a range.

Handles a method moved onto an existing class (call sites lowered), a method moved to a
module-level free function (call sites requalified), a free-function-source move to an
existing module (callers repath their import), a new-file extract -- where the prep
commit staged the whole module body (scaffolding plus def) as a trailing block in the
source, so the move cuts that tail into the new file (extract_to_new_module) -- and an
intra-file inline-block extract-function (a new helper whose verbatim body is a block carved
from a sibling function, that block replaced by a call). A rename or a statement-level
reorder relocates no def and is reported unsupported. Runnable directly:

    python3 mechanical_refactor_proof_generator.py <commit>
    python3 mechanical_refactor_proof_generator.py <base>..<tip> \
        --match '(?<!_)mechanical_provable' --out DIR
"""

import ast
import html
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mechanical_refactor_reproduction_utils as rr


def _git_output(args: list[str], cwd: str) -> str:
    # [FUNCTION DOC] HELPER: Runs git with the arguments in directory cwd. Returns standard
    # output text exactly as produced (not stripped) when the exit status is zero, or an empty
    # string on any failure (errors are swallowed, so a missing file looks like an empty file).
    """Raw stdout of a git command ("" if it fails). Not stripped, so ``ast`` line numbers
    stay aligned with a file's real lines."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def _repo_root() -> str:
    # [FUNCTION DOC] HELPER: Returns the stripped top-level directory of the repository that
    # contains the current directory; raises the called-process error outside a repository.
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _removed_symbol_names(lines: list[str]) -> set[str]:
    # [FUNCTION DOC] HELPER: Scans diff content lines and returns the set of names introduced by
    # lines that start (after optional whitespace) with a function definition (optionally async)
    # or a class definition. Indentation is ignored, so methods and nested definitions count
    # too, despite the docstring saying top-level.
    """Names of top-level defs and classes among removed diff lines (the extract's source
    relinquishes these), so a moved class is found, not just a moved def."""
    return {m.group(2) for ln in lines if (m := re.match(r"\s*(?:async\s+)?(def|class)\s+(\w+)", ln))}


def _def_indent(lines: list[str], name: str) -> int | None:
    # [FUNCTION DOC] HELPER: Returns the number of leading whitespace characters of the first
    # line that defines a function (optionally async) with exactly this name (word boundary after
    # the name), or nothing if no line does.
    for line in lines:
        match = re.match(r"(\s*)(?:async\s+)?def\s+" + re.escape(name) + r"\b", line)
        if match:
            return len(match.group(1))
    return None


def _per_file_diff(commit: str, root: str) -> dict[str, dict]:
    # [FUNCTION DOC] HELPER: Shows the commit (no message, no color, no external diff tool) and
    # parses the unified diff text. Returns a mapping from each file's new path to a mapping
    # with: removed (list of removed content lines without the leading minus), added (list of
    # added lines without the plus), new (true if the file was created) and deleted (true if the
    # file was removed). Only lines inside hunks count as content, so file header lines are not
    # mistaken for content. Renamed files are keyed by their new path. An empty or failing git
    # call yields an empty mapping.
    """Per-file removed/added content lines (whitespace intact) and a new-file flag."""
    out = _git_output(["show", commit, "--format=", "--no-color", "--no-ext-diff"], root)
    files: dict[str, dict] = {}
    path: str | None = None
    in_hunk = False
    for line in out.splitlines():
        header = re.match(r"diff --git a/(.*) b/(.+)$", line)
        if header:
            path = header.group(2)
            files[path] = {"removed": [], "added": [], "new": False, "deleted": False}
            in_hunk = False
        elif line.startswith("new file"):
            files[path]["new"] = True
        elif line.startswith("deleted file"):
            files[path]["deleted"] = True
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            files[path]["added"].append(line[1:])
        elif in_hunk and line.startswith("-"):
            files[path]["removed"].append(line[1:])
    return files


def _enclosing_function(tree: ast.AST, lineno: int) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the name of the innermost function or async function whose
    # line range contains the given 1-based line (innermost meaning the one starting on the
    # latest line), or nothing if the line is outside every function.
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= lineno <= node.end_lineno:
                if best is None or node.lineno > best[0]:
                    best = (node.lineno, node.name)
    return best[1] if best else None


def _enclosing_class_of_def(tree: ast.AST, name: str) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the name of the first class (in tree walk order) that has a
    # function or async function called name as a direct member of its body, or nothing. Does
    # not look at deeper nesting.
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name:
                    return node.name
    return None


def _delegate_stub_attr(tree: ast.AST, name: str) -> tuple[str, str] | None:
    # [FUNCTION DOC] HELPER: Looks only at the FIRST function called name in tree walk order.
    # If its body is exactly one return statement returning a call of a method on an attribute
    # of the self parameter, returns a pair of the attribute name and the called method name;
    # in every other case returns nothing (including when no such function exists).
    # KNOWN ISSUE: a stub of an async method returns an await expression, not a plain call, so
    # it is not recognized, although the move primitive generates such stubs for async methods.
    """The component attribute a forwarding stub ``def name``: ``return self.<attr>.<m>(...)``
    delegates through, with the forwarded method name -- None when no such stub exists.
    """
    for node in ast.walk(tree):
        if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name):
            continue
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Return)
            and isinstance(node.body[0].value, ast.Call)
            and isinstance(node.body[0].value.func, ast.Attribute)
            and isinstance(node.body[0].value.func.value, ast.Attribute)
            and isinstance(node.body[0].value.func.value.value, ast.Name)
            and node.body[0].value.func.value.value.id == "self"
        ):
            return (node.body[0].value.func.value.attr, node.body[0].value.func.attr)
        return None
    return None


def _nested_in_function(tree: ast.AST, name: str) -> bool:
    # [FUNCTION DOC] HELPER: Finds the first function called name; returns false if there is
    # none; otherwise true when some other function's line range contains the found function's
    # first line (that is, it is defined inside another function).
    target = rr._find_def(tree, name)
    if target is None:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node is not target
            and node.lineno <= target.lineno <= node.end_lineno
        ):
            return True
    return False


def _module_of_path(path: str) -> str:
    # [FUNCTION DOC] HELPER: Converts a repository-relative python file path to a dotted module
    # name: drops a leading "python/" directory if present, drops the ".py" suffix and replaces
    # slashes by dots.
    return path.removeprefix("python/").removesuffix(".py").replace("/", ".")


def _import_pairs(text: str) -> dict:
    # [FUNCTION DOC] HELPER: Parses the source and, over its top-level statements only, maps
    # each imported name to the statement text that would import it alone. A plain import gives
    # a text key equal to its own value ("import x" with " as y" appended if aliased). A
    # from-import gives, for each name, a key of the triple (module text including leading dots
    # for relative level, name, alias or nothing) mapped to a one-name from-import text. Raises
    # a syntax error for unparsable text.
    """Module-level imports keyed one entry per imported name, so removing a single name from
    a multi-name ``from x import a, b`` is not mistaken for the whole statement changing. The
    value is the one-name ``add_import`` text for a gained name (the import sorter merges it).
    """
    pairs: dict = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                stmt = "import " + alias.name
                if alias.asname:
                    stmt += f" as {alias.asname}"
                pairs[stmt] = stmt
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                name = alias.name + (f" as {alias.asname}" if alias.asname else "")
                pairs[(module, alias.name, alias.asname)] = f"from {module} import {name}"
    return pairs


def _typechecking_pairs(text: str) -> dict:
    # [FUNCTION DOC] HELPER: Like the import-pairs helper but for imports found directly in the
    # body of each top-level conditional on the type-checking flag (plain or typing-qualified):
    # same key and value shapes for from-imports (triple key, one-name statement) and plain
    # imports (text key equal to its value). Nested conditionals inside the block are ignored.
    """Imports inside the module's ``if TYPE_CHECKING:`` block, keyed per name (same shape as
    ``_import_pairs``), so a type-only import the destination gains for a moved annotation is
    inferable separately from the runtime imports."""
    pairs: dict = {}
    for node in ast.parse(text).body:
        if not (isinstance(node, ast.If) and ast.unparse(node.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING")):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom):
                module = "." * stmt.level + (stmt.module or "")
                for alias in stmt.names:
                    name = alias.name + (f" as {alias.asname}" if alias.asname else "")
                    pairs[(module, alias.name, alias.asname)] = f"from {module} import {name}"
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    stmt_text = "import " + alias.name
                    if alias.asname:
                        stmt_text += f" as {alias.asname}"
                    pairs[stmt_text] = stmt_text
    return pairs


def _local_import_of(tree: ast.AST, fn_name: str, module: str, symbol: str) -> str | None:
    # [FUNCTION DOC] HELPER: Finds the first function called fn_name; returns nothing if none.
    # Searches inside it (any depth) for a from-import whose module attribute equals module
    # and that imports symbol, and returns the canonical text "from <module> import <symbol>"
    # (aliases and other names in the statement are not reflected), or nothing if absent.
    fn = rr._find_def(tree, fn_name)
    if fn is None:
        return None
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == module
            and any(alias.name == symbol for alias in node.names)
        ):
            return f"from {module} import {symbol}"
    return None


def _removal_from_key(path: str, key) -> dict:
    # [FUNCTION DOC] HELPER: Converts a key of the import-pairs mapping into the keyword record
    # for an imported-name removal. Returns a mapping with path, module, name, asname. A triple
    # key gives its three parts. A text key of a plain import has the "import " prefix removed
    # and is split on " as " (first occurrence) into name and alias; the module is nothing.
    """Turn an ``_import_pairs`` key the target dropped into a ``remove_imported_name`` call. A
    ``(module, name, asname)`` key drops one name from a ``from`` import; a bare ``import x``
    statement string drops a plain import."""
    if isinstance(key, tuple):
        module, name, asname = key
        return {"path": path, "module": module, "name": name, "asname": asname}
    rest = key.removeprefix("import ")
    if " as " in rest:
        name, asname = rest.split(" as ", 1)
    else:
        name, asname = rest, None
    return {"path": path, "module": None, "name": name, "asname": asname}


def _module_assign_names(text: str) -> set:
    # [FUNCTION DOC] HELPER: Returns the set of plain names bound at module top level by plain
    # assignments (every plain-name target) and annotated assignments. Tuple unpacking,
    # attribute targets, augmented assignments and imports are ignored. Raises a syntax error
    # on unparsable text.
    """Names bound by module-level assignments (``logger = ...``, ``_is_hip = is_hip()``), so a
    constant relocated into an extracted module can be told apart from one the source keeps.
    """
    names: set = set()
    for node in ast.parse(text).body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        names |= {t.id for t in targets if isinstance(t, ast.Name)}
    return names


def _import_additions(path: str, after: str, before_pairs: dict, after_pairs: dict) -> list:
    # [FUNCTION DOC] HELPER: Computes import additions for one file from the import-pairs
    # mappings before and after plus the after-state source text. Returns a list of records
    # with path and statement text. For each key present after but not before: a from-import
    # key whose module was not imported at all before is deferred; every other key (a plain
    # import, or a name from a module already imported before) yields its one-name statement.
    # Then for the deferred modules, every top-level from-import statement of the after-state
    # whose module text is in that set is added verbatim (original lines, trailing newlines
    # trimmed), preserving multi-line wrapping. Note the verbatim statement may include names
    # that were already imported before from other statements, which cannot happen for a wholly
    # new module.
    """The module-level imports a file gained, as ``add_import`` texts. A name gained from a
    module the file already imported is added per-name (the sorter merges it); a name from a
    wholly new module is added as the target's *verbatim* statement, so a multi-line or
    magic-trailing-comma wrapping the target chose is reproduced (a freshly merged single line
    would otherwise collapse and not match)."""
    before_modules = {key[0] for key in before_pairs if isinstance(key, tuple)}
    additions: list = []
    verbatim_modules: set = set()
    for key in after_pairs:
        if key in before_pairs:
            continue
        if isinstance(key, tuple) and key[0] not in before_modules:
            verbatim_modules.add(key[0])
        else:
            additions.append({"path": path, "text": after_pairs[key]})
    if verbatim_modules:
        after_lines = after.splitlines(keepends=True)
        for node in ast.parse(after).body:
            if isinstance(node, ast.ImportFrom) and "." * node.level + (node.module or "") in verbatim_modules:
                text = "".join(after_lines[node.lineno - 1 : node.end_lineno])
                additions.append({"path": path, "text": text.rstrip("\n")})
    return additions


@dataclass
class Recipe:
    # [CLASS DOC] Mutable data record describing how to reproduce one commit. Fields: base and
    # target commit ids; a supported flag (default true; set false with explanations in notes
    # when the commit shape cannot be inferred); and lists of plain mappings, one list per
    # primitive family, consumed by the script renderer: symbol moves, assignment moves, tail
    # extracts into new modules, inline-block function extracts, scattered extracts, call-site
    # adaptations (lowerings and requalifications), function-scoped import repaths, in-function
    # import removals, module-level imported-name removals, import additions, type-checking
    # import additions, file deletions, and human-readable notes. Converted to a plain mapping
    # with the standard dataclass helper when serialized.
    base: str
    target: str
    supported: bool = True
    moves: list = field(default_factory=list)
    assign_moves: list = field(default_factory=list)
    extracts: list = field(default_factory=list)
    extract_functions: list = field(default_factory=list)
    scatter_extracts: list = field(default_factory=list)
    lowerings: list = field(default_factory=list)
    repaths: list = field(default_factory=list)
    import_removals: list = field(default_factory=list)
    module_import_removals: list = field(default_factory=list)
    import_additions: list = field(default_factory=list)
    typechecking_additions: list = field(default_factory=list)
    deletes: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _infer_call_adaptations(
    recipe: Recipe,
    files: dict[str, dict],
    *,
    name: str,
    src: str,
    src_class: str,
    into_class: str | None,
    commit: str,
    root: str,
) -> None:
    # [FUNCTION DOC] HELPER: Finds call sites that must be adapted because the method name moved
    # out of class src_class, and appends records to the recipe (mutated; returns nothing). The
    # kind is lowering when the method moved onto a class (into_class set) and requalification
    # when it became a free function. For every file touched by the commit (the files mapping
    # from the per-file diff), parses the parent commit's version (skipping files that do not
    # parse or are empty) and collects the enclosing functions of every call of an attribute
    # called name whose object expression source text equals src_class, where for lowering at
    # least one argument is required and for requalification any argument count is accepted.
    # Calls outside any function are ignored. If a file has such callers, appends a lowering
    # record (name, owner, path, kind). Then for each caller function sorted by name, if it
    # contains a local from-import of src_class from the source module and that statement text
    # appears inside any removed diff line of the file, appends an in-function import removal
    # record (path, text, function). The src parameter is the source file path (converted to a
    # dotted module name) and dst is not used.
    """A method-source move adapts its call sites: a move onto a class lowers the receiver
    out of the args; a move to a module-level free function drops the qualifier. A caller is
    a before-state call ``<src_class>.name(...)`` -- matched on ``src_class`` (not a loose
    text search) so the moved body's own same-named calls on a different receiver are
    excluded -- and its orphaned local import of ``src_class`` is removed."""
    kind = "lower" if into_class is not None else "requalify"
    src_module = _module_of_path(src)
    for path, f in files.items():
        before = _git_output(["show", f"{commit}^:{path}"], root)
        try:
            tree = ast.parse(before)
        except SyntaxError:
            continue
        caller_fns: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == name
                and (node.args or kind == "requalify")
                and ast.unparse(node.func.value) == src_class
            ):
                fn = _enclosing_function(tree, node.lineno)
                if fn is not None:
                    caller_fns.add(fn)
        if not caller_fns:
            continue
        recipe.lowerings.append({"name": name, "owner": src_class, "path": path, "kind": kind})
        for fn in sorted(caller_fns):
            imp = _local_import_of(tree, fn, src_module, src_class)
            if imp is not None and any(imp in r for r in f["removed"]):
                recipe.import_removals.append({"path": path, "text": imp, "in_function": fn})


def _infer_function_scoped_repaths(
    recipe: Recipe,
    files: dict[str, dict],
    *,
    name: str,
    src: str,
    dst: str,
    commit: str,
    root: str,
) -> None:
    # [FUNCTION DOC] HELPER: Appends a repath record (path, old module, new module, name) to the
    # recipe (mutated; returns nothing) for each file, other than the source file, in the commit
    # whose parent-state contains a from-import of name from the dotted source module that is not
    # a direct child of the module body (nested in a function or conditional). Files are visited
    # in sorted order; unparsable or empty files are skipped. Only absolute module spellings are
    # matched, never relative imports.
    """A moved free function keeps the same bare call, so a caller only repaths its import.
    Module-level repaths fall out of the symmetric import diff; only function-scoped imports
    (which ``add_import`` cannot place) need an explicit in-place repath."""
    src_module = _module_of_path(src)
    dst_module = _module_of_path(dst)
    for path in sorted(files):
        if path == src:
            continue
        before = _git_output(["show", f"{commit}^:{path}"], root)
        try:
            tree = ast.parse(before)
        except SyntaxError:
            continue
        top_level = {id(node) for node in tree.body}
        nested = any(
            isinstance(node, ast.ImportFrom)
            and id(node) not in top_level
            and node.module == src_module
            and any(alias.name == name for alias in node.names)
            for node in ast.walk(tree)
        )
        if nested:
            recipe.repaths.append(
                {
                    "path": path,
                    "old_module": src_module,
                    "new_module": dst_module,
                    "name": name,
                }
            )


def _self_annotation_dropped(src_def: ast.AST | None, dst_def: ast.AST | None) -> bool:
    # [FUNCTION DOC] HELPER: Compares the source-side and destination-side definition nodes of a
    # moved function. Returns true only when the source's first positional parameter (not
    # counting positional-only ones) is called self and annotated while the destination's is not
    # (or is missing or absent). A missing node counts as having no annotation.
    """Whether the move drops a ``self: Target`` annotation -- the source had it and the
    destination does not. Some class moves keep it (a retyped self that stays annotated), so
    this is inferred from both sides rather than assumed."""

    def has_self_annotation(node: ast.AST | None) -> bool:
        # [FUNCTION DOC] HELPER: True when the node exists, has at least one positional
        # parameter, the first is named self, and it carries a type annotation.
        return bool(
            node is not None
            and node.args.args
            and node.args.args[0].arg == "self"
            and node.args.args[0].annotation is not None
        )

    return has_self_annotation(src_def) and not has_self_annotation(dst_def)


def _wants_future_import(files: dict[str, dict], src: str, dst: str) -> bool:
    # [FUNCTION DOC] HELPER: Decides whether a new module needs the future annotations import
    # prepended. True when some added line of the destination's diff contains that import text
    # and no removed line of the source's diff does (so it was not simply carried over with the
    # cut tail). Raises a key error if src or dst is not in the files mapping.
    future = "from __future__ import annotations"
    gained = any(future in line for line in files[dst]["added"])
    travelled = any(future in line for line in files[src]["removed"])
    return gained and not travelled


def _next_sibling_def_name(dst_tree: ast.AST, name: str, into_class: str | None) -> str | None:
    # [FUNCTION DOC] HELPER: Takes the list of function and class definitions that are direct
    # children of the container (the body of class into_class when given, found anywhere in the
    # tree and treated as empty if missing; otherwise the module body), and returns the name of
    # the definition right after the one called name, or nothing if name is last or not present.
    """The name of the def that immediately follows ``name`` at its scope in the destination
    (module level, or inside ``into_class``), or None when ``name`` is the last def there. Lets
    a move reinsert the relocated def in the chain's order instead of appending at the end.
    """
    container: list = []
    if into_class is not None:
        cls = rr._find_class(dst_tree, into_class)
        container = cls.body if cls is not None else []
    else:
        container = getattr(dst_tree, "body", [])
    defs = [n for n in container if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for i, node in enumerate(defs):
        if node.name == name:
            return defs[i + 1].name if i + 1 < len(defs) else None
    return None


def _next_sibling_assign_or_def(dst_tree: ast.AST, name: str) -> str | None:
    # [FUNCTION DOC] HELPER: Considers only top-level statements that have a name (function,
    # class, single plain-name assignment, annotated assignment to a name) in order, finds the
    # one called name, and returns the name of the next named statement, or nothing if it is
    # last or not found. Unnamed statements between them are skipped.
    """The name of the top-level statement (def/class/single-Name assign) that immediately
    follows the assignment ``name`` in the destination, or None when it is last."""

    def stmt_name(node: ast.AST) -> str | None:
        # [FUNCTION DOC] HELPER: Name defined by a statement: function or class name, the target
        # of a single-target plain-name assignment, or the target of an annotated assignment to
        # a name; nothing for any other statement.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return node.target.id
        return None

    named = [(stmt_name(n), n) for n in getattr(dst_tree, "body", []) if stmt_name(n)]
    for i, (nm, _) in enumerate(named):
        if nm == name:
            return named[i + 1][0] if i + 1 < len(named) else None
    return None


def _stmt_symbol_name(node: ast.AST) -> str | None:
    # [FUNCTION DOC] HELPER: Same rule as the statement-name helper nested in the sibling
    # lookup: function or class name, single-target plain-name assignment target, or annotated
    # assignment name target; nothing otherwise.
    """The name a top-level statement defines -- a def/class name, or a single-Name
    assignment target -- else None (an ``if TYPE_CHECKING:`` guard, a tuple assign, ...).
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _module_move_anchor(dst_tree: ast.AST, name: str, into_class: str | None) -> tuple[str | None, str | None]:
    # [FUNCTION DOC] HELPER: Chooses how to reinsert a moved definition, returning a pair
    # (before, after) of sibling names, at most one of which is set. Starts from the next
    # sibling definition name as the before anchor. Returns it with no after anchor when moving
    # into a class, when the module-level definition is not found, when it is the last statement,
    # when the following statement is itself a function or class, or when the statement
    # immediately above it has no usable symbol name. Otherwise (the definition sits right above
    # some other kind of statement and has a nameable statement above it) returns no before
    # anchor and the preceding statement's symbol name as after anchor.
    # KNOWN ISSUE: the docstring speaks of an unnameable following statement, but the check
    # applies to any following non-definition statement, including a nameable assignment.
    """The ``(before, after)`` anchor for reinserting the moved def ``name``. Normally
    ``before=<next sibling def>``. But when a module-level def lands immediately above an
    unnameable statement (e.g. an ``if TYPE_CHECKING:`` guard) with a nameable statement
    immediately above it, a ``before`` anchor would resolve to the next def *past* that block
    and overshoot, so anchor with ``after=<preceding symbol>`` instead."""
    before = _next_sibling_def_name(dst_tree, name, into_class)
    if into_class is not None:
        return before, None
    body = list(getattr(dst_tree, "body", []))
    idx = next(
        (
            i
            for i, n in enumerate(body)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name
        ),
        None,
    )
    if idx is None:
        return before, None
    following = body[idx + 1] if idx + 1 < len(body) else None
    next_is_named_def = isinstance(following, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    if following is None or next_is_named_def:
        return before, None
    preceding = body[idx - 1] if idx > 0 else None
    prev_name = _stmt_symbol_name(preceding) if preceding is not None else None
    if prev_name is None:
        return before, None
    return None, prev_name


def _symbols_form_tail(src_text: str, symbols: list[str]) -> bool:
    # [FUNCTION DOC] HELPER: Parses the source and applies the same backward scan as the
    # tail-extraction primitive: from the end of the module body, include statements that are
    # either wanted function or class definitions or scaffolding. Returns whether the scan
    # covers all wanted names as definitions within the included tail (see the rest of the body
    # for the final comparison).
    """Whether ``symbols`` sit at the end of the source as a contiguous block of defs/classes
    and the scaffolding leading into them -- the trailing block a prep commit stages for a
    new-module extract. A method still inside a class, or a symbol separated from the tail by
    other code, fails this and is not an extractable tail."""
    body = ast.parse(src_text).body
    wanted = set(symbols)

    def is_scaffolding(node: ast.stmt) -> bool:
        # [FUNCTION DOC] HELPER: True for an import, a conditional on the type-checking flag
        # (plain or typing-qualified), an assignment whose targets are all plain names, or an
        # annotated assignment to a plain name; false otherwise.
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True
        if isinstance(node, ast.If):
            return ast.unparse(node.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING")
        if isinstance(node, ast.Assign):
            return all(isinstance(x, ast.Name) for x in node.targets)
        if isinstance(node, ast.AnnAssign):
            return isinstance(node.target, ast.Name)
        return False

    cut = len(body)
    while cut > 0:
        node = body[cut - 1]
        is_symbol = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in wanted
        if is_symbol or is_scaffolding(node):
            cut -= 1
        else:
            break
    present = {
        node.name for node in body[cut:] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    return wanted <= present


def _scatter_extract_layout(dst_after: str, symbols: list[str]) -> dict | None:
    # [FUNCTION DOC] HELPER: Splits a new module's final source into an authored header and the
    # ordered symbol names. Parses the text; finds top-level functions and classes named in
    # symbols. Returns nothing if their count differs from the number of distinct wanted names
    # (a missing or duplicated symbol), or if any statement from the first symbol onward is not
    # a wanted symbol (something interleaved). Otherwise returns a mapping with header (all
    # lines before the first symbol's first line, decorators included, so any comments or
    # blank lines directly above it belong to the header) and order (the symbol names in
    # their order in the module).
    """For a new module whose relocated ``symbols`` form a contiguous block at the end (after
    an authored header of imports, constants, a logger, a ``TYPE_CHECKING`` block), return the
    ``header`` text and the ``symbols`` in target order. Returns None when a non-symbol
    statement is interleaved among the symbols, so there is no clean header/body split.
    """
    lines = dst_after.splitlines(keepends=True)
    body = ast.parse(dst_after).body
    wanted = set(symbols)

    def is_wanted(node: ast.AST) -> bool:
        # [FUNCTION DOC] HELPER: True for a function, async function or class whose name is in
        # the wanted symbol set.
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in wanted

    sym_nodes = [node for node in body if is_wanted(node)]
    if len(sym_nodes) != len(wanted):
        return None
    first_index = body.index(sym_nodes[0])
    if any(not is_wanted(node) for node in body[first_index:]):
        return None
    header = "".join(lines[: rr._def_span(sym_nodes[0])[0] - 1])
    return {"header": header, "order": [node.name for node in sym_nodes]}


def _iter_defs_with_container(
    tree: ast.AST,
) -> list[tuple[str | None, ast.AST]]:
    # [FUNCTION DOC] HELPER: Lists, in source order, pairs of (container class name or nothing,
    # definition node) for each module-level function or async function (container nothing) and
    # each method or async method directly inside a module-level class. Deeper nesting is not
    # visited.
    """(container_class_name_or_None, def_node) for every module-level function and every
    method one class deep -- the two nesting depths an extract_function helper can land at.
    """
    out: list[tuple[str | None, ast.AST]] = []
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((None, node))
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append((node.name, child))
    return out


def _statements_parse(lines: list[str]) -> bool:
    # [FUNCTION DOC] HELPER: Joins the lines; blank or whitespace-only text counts as parsing
    # (true). Otherwise removes the smallest leading-space indentation among non-blank lines
    # from every line and attempts to parse the result as a module; returns true on success and
    # false on a syntax error.
    """Whether ``lines`` (dedented to their own minimum indent) parse as complete Python
    statements -- used to keep a prefix/suffix split from cutting through the middle of a
    multi-line statement."""
    text = "".join(lines)
    if not text.strip():
        return True
    indents = [len(ln) - len(ln.lstrip(" ")) for ln in lines if ln.strip()]
    dedented = rr.dedent(text, min(indents)) if indents else text
    try:
        ast.parse(dedented)
        return True
    except SyntaxError:
        return False


def _common_prefix_suffix(a: list[str], b: list[str]) -> tuple[int, int]:
    # [FUNCTION DOC] HELPER: Returns a pair (prefix length, suffix length) in lines. First the
    # greedy longest common prefix; then the longest common suffix that does not overlap the
    # prefix in either list. Then, while the differing middle of either list is not a complete
    # set of statements (per the statements-parse helper), the suffix is shortened one line at
    # a time; if the suffix reaches zero and the middles still do not parse, the prefix is
    # shortened one line at a time under the same test (using the final suffix). The result may
    # leave a middle that still does not parse if even a zero prefix fails.
    """Longest common leading and trailing run of identical lines between two line lists,
    kept non-overlapping -- isolates the single contiguous region where they differ. The
    greedy match is then shrunk (suffix first, then prefix) until the differing middle of
    *both* lists parses as complete statements, so a boundary line the removed block and its
    replacement happen to share (e.g. a lone ``)``) is not absorbed mid-statement."""
    prefix = 0
    while prefix < len(a) and prefix < len(b) and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while suffix < len(a) - prefix and suffix < len(b) - prefix and a[-1 - suffix] == b[-1 - suffix]:
        suffix += 1
    while suffix > 0 and not (
        _statements_parse(a[prefix : len(a) - suffix]) and _statements_parse(b[prefix : len(b) - suffix])
    ):
        suffix -= 1
    while prefix > 0 and not (
        _statements_parse(a[prefix : len(a) - suffix]) and _statements_parse(b[prefix : len(b) - suffix])
    ):
        prefix -= 1
    return prefix, suffix


def _call_names_in(node: ast.AST) -> set[str]:
    # [FUNCTION DOC] HELPER: Walks the subtree and returns the set of called names: the
    # attribute name for a call on an attribute of any object (receiver not checked, so the
    # receiver need not be self), and the identifier for a call of a bare name. Other callee
    # shapes are ignored.
    """Names invoked as ``self.<name>(...)`` or ``<name>(...)`` anywhere under ``node``."""
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Attribute):
                names.add(sub.func.attr)
            elif isinstance(sub.func, ast.Name):
                names.add(sub.func.id)
    return names


def _infer_extract_functions(recipe: Recipe, files: dict[str, dict], commit: str, root: str) -> None:
    # [FUNCTION DOC] HELPER: Appends inline-block extraction records to the recipe (mutated;
    # returns nothing). For every changed existing python file (skipping files created or
    # deleted by the commit and files that fail to parse in either state): lists functions and
    # one-level methods before and after; each function present after but not before (same
    # container and name) and with a non-empty body is a candidate new helper. Its header text is
    # the signature lines through the colon, excluding decorators and the body. The origin
    # function is searched among functions existing in both states (same container and name,
    # not the helper) whose line text changed and whose new version calls the helper's name;
    # exactly one such function is required, else the helper is skipped. Takes the origin's
    # old and new line lists, finds the common prefix and suffix, and defines the extracted
    # block as the old middle and the call text as the new middle; both must be non-empty. The
    # block's indentation is its first line's leading spaces. The block, dedented, must parse
    # into statements; the helper body must then have either the same number of statements, or
    # one more where the last is a return (its source text, with surrounding newlines trimmed,
    # becomes the return text); otherwise skipped. Record fields: src and dst (same file), name,
    # signature (header text), body (block text), body_indent, call (new middle text),
    # return_text (or nothing), into_class (container), and before (name of the definition
    # following the helper in the new file, or nothing).
    # KNOWN ISSUE: the statement count of the helper body includes a leading docstring, so a
    # helper that has a docstring is skipped unless its last statement is a return.
    """Infer intra-file extract_function ops: a new helper ``H`` whose body is a verbatim block
    cut from another function ``F`` of the same file, with ``F``'s block replaced by a call to
    ``H`` (optionally an ``lhs = self.H(...)`` assignment mirrored by a ``return lhs`` the helper
    appends). The relocated body is byte-checked; only the header/call/return are authored. A
    body whose reindent does not reconstruct the helper (a bundled edit) yields no op, so the
    residual surfaces it instead of a false pass."""
    for path, f in files.items():
        if f.get("new") or f.get("deleted"):
            continue
        before_text = _git_output(["show", f"{commit}^:{path}"], root)
        after_text = _git_output(["show", f"{commit}:{path}"], root)
        try:
            before_tree = ast.parse(before_text)
            after_tree = ast.parse(after_text)
        except SyntaxError:
            continue
        before_lines = rr._split_keepends(before_text)
        after_lines = rr._split_keepends(after_text)
        before_defs = _iter_defs_with_container(before_tree)
        before_keys = {(c, n.name) for c, n in before_defs}
        for container, helper in _iter_defs_with_container(after_tree):
            if (container, helper.name) in before_keys:
                continue
            if not helper.body:
                continue
            # The signature is the def header only (through its colon), not everything up to
            # the first statement -- a leading comment sits between them and belongs to the
            # extracted body, not the authored signature.
            helper_text = "".join(after_lines[helper.lineno - 1 : helper.end_lineno])
            header_len = rr._def_header_end(helper_text)
            header_text = "".join(after_lines[helper.lineno - 1 : helper.lineno - 1 + header_len])
            # F is the one sibling function that changed and now calls the helper.
            candidates = []
            for cont, node in before_defs:
                after_node = next(
                    (n for c, n in _iter_defs_with_container(after_tree) if c == cont and n.name == node.name),
                    None,
                )
                if after_node is None or node.name == helper.name:
                    continue
                b_lines = before_lines[node.lineno - 1 : node.end_lineno]
                a_lines = after_lines[after_node.lineno - 1 : after_node.end_lineno]
                if b_lines == a_lines:
                    continue
                if helper.name not in _call_names_in(after_node):
                    continue
                candidates.append((b_lines, a_lines))
            if len(candidates) != 1:
                continue
            f_before, f_after = candidates[0]
            prefix, suffix = _common_prefix_suffix(f_before, f_after)
            block = f_before[prefix : len(f_before) - suffix]
            call_lines = f_after[prefix : len(f_after) - suffix]
            if not block or not call_lines:
                continue
            body_indent = len(block[0]) - len(block[0].lstrip(" "))
            body_text = "".join(block)
            # Detect the authored `return <name>` structurally, by statement count -- the
            # formatter reflows lines differently at the helper's shallower indent, so a
            # byte comparison of the reindented body would spuriously fail; the repro's
            # byte-diff (which runs the formatter) is the real arbiter.
            try:
                block_stmts = ast.parse(rr.dedent(body_text, body_indent)).body
            except SyntaxError:
                continue
            helper_stmts = helper.body
            return_text: str | None = None
            if len(helper_stmts) == len(block_stmts) + 1 and isinstance(helper_stmts[-1], ast.Return):
                ret = helper_stmts[-1]
                return_text = "".join(after_lines[ret.lineno - 1 : ret.end_lineno]).strip("\n")
            elif len(helper_stmts) != len(block_stmts):
                continue
            recipe.extract_functions.append(
                {
                    "src": path,
                    "dst": path,
                    "name": helper.name,
                    "signature": header_text,
                    "body": body_text,
                    "body_indent": body_indent,
                    "call": "".join(call_lines),
                    "return_text": return_text,
                    "into_class": container,
                    "before": _next_sibling_def_name(after_tree, helper.name, container),
                }
            )


def infer_recipe(commit: str, root: str) -> Recipe:
    # [FUNCTION DOC] Builds a recipe for one commit in the repository at root; the recipe's base
    # is the commit's first parent (the commit id with a tilde-one suffix) and its target is the
    # commit. Reads file contents of the parent and the commit through git. Returns the recipe
    # (never raises for unsupported shapes; it sets the unsupported flag and notes instead, but
    # unparsable sources in some branches raise a syntax error). Algorithm, in order:
    # (1) Take the per-file diff; keep only python files; each non-python changed file adds a
    #   note that it is left to the residual diff.
    # (2) New-file extracts: for every newly created python file (sorted) that parses, take its
    #   top-level function and class names as the moved symbols (skip if none). The source is
    #   the set of non-new files whose removed diff lines define any of those names (at any
    #   indentation); if that is not exactly one file, mark unsupported with a note and go on.
    #   If the symbols form a trailing block of the source's old version, record a tail extract
    #   (source, destination, symbols, and whether the future annotations import must be
    #   prepended). Otherwise, if not all symbols are top-level in the source, mark unsupported
    #   with a note; else compute the scattered layout of the new file (header plus order), and
    #   if there is none, mark unsupported with a note; else record a scattered extract whose
    #   dropped assignments are the module-level names removed from the source and present in
    #   the new file.
    # (3) Class moves: for each top-level class name (a class line starting at column zero)
    #   both removed and added somewhere, in sorted order: take the first file that removed it
    #   and the first that added it; skip if either is missing, they are the same file, or the
    #   destination is new. Record a symbol move with no enclosing classes, zero dedent, the
    #   class's line number in the destination as ordering key, and the next sibling definition
    #   as the insertion anchor. Remember the name as a moved class.
    # (4) Definition moves: for each function name (at any indentation) both removed and added,
    #   sorted: the source is the first file that removed it; the destination is the first other
    #   file that added it, or the source itself if only the source re-added it (an in-file
    #   reorder). Skip if either is missing or the destination is new. Skip with a note if the
    #   function is nested inside another function in the old source. Determine the enclosing
    #   class on each side from the diff's definition-line indentation (column zero means module
    #   level); skip with a note when the source class itself was moved as a class. Resolve the
    #   unique definition on each side; ambiguity marks unsupported with a note. The dedent is
    #   the indentation difference. If the source file also re-added a same-named definition
    #   that is a forwarding stub, the stub's attribute becomes the delegate (and its forwarded
    #   method name, if different, the delegate name). The insertion anchor comes from the
    #   module-move-anchor helper, and the self-annotation drop flag from the destination
    #   comparison. Record the move, then, if the source was in a class, infer call-site
    #   adaptations; otherwise infer function-scoped import repaths.
    # (5) In-place stub moves: for each function name added but never removed (sorted) whose
    #   first adding file is existing: look in the other changed files for one whose new version
    #   has a forwarding stub of that name and whose old version had a real (non-stub)
    #   definition; the first such file is the source. Record a move with dedent equal to the
    #   source definition's column minus the destination indentation, the stub's attribute as
    #   delegate and, when the forwarded name differs, the delegate name. No call adaptations
    #   are inferred for these. Files failing to parse are skipped.
    # (6) Assignment moves: for each pair of distinct changed existing files, every module-level
    #   assigned name lost by the first and gained by the second yields an assignment-move
    #   record whose anchor is the next named top-level statement in the second file's new
    #   version.
    # (7) Import diffs: for every changed file except new files and tail-extract destinations,
    #   compare module-level imports before and after (per imported name). Gained names become
    #   import additions (via the additions helper); lost names become imported-name removals,
    #   except for tail-extract sources; type-checking-block imports gained become type-checking
    #   additions.
    # (8) If no definition moves were recorded at all, try inferring inline-block function
    #   extractions.
    # (9) Any deleted file that was a move source is scheduled for deletion.
    # (10) If there are no moves, tail extracts, scattered extracts or function extractions, mark
    #   the recipe unsupported and, when there are no notes yet, add the note that no definition
    #   was relocated (a rename or statement-level change that should be reviewed as prep).
    # KNOWN ISSUE: the try-except around the git reads of the before/after texts in step (6) can
    # never trigger because the git helper swallows its own errors. Also, for a deleted move
    # source, step (7) infers removal of every import of the deleted file, and the operation list
    # runs those removals after the file deletion, so they would try to read a missing file.
    """Infer a faithful relocation recipe for a move commit from its diff + before-state.

    A move onto an existing module/class becomes a ``move_symbol``; a move whose destination
    file is new becomes an ``extract_to_new_module`` (the prep commit staged the whole module
    body -- scaffolding plus def -- as a trailing block in the source, so the move cuts that
    tail into the new file). Method-source moves adapt their call sites; free-function-source
    moves keep the bare call and only repath imports. A rename or a statement-level reorder
    relocates no def, so nothing is inferred and the commit is reported unsupported."""
    all_files = _per_file_diff(commit, root)
    files = {path: f for path, f in all_files.items() if path.endswith(".py")}
    recipe = Recipe(base=f"{commit}~1", target=commit)
    for path in sorted(set(all_files) - set(files)):
        recipe.notes.append(f"non-Python file changed: {path} (left to the residual diff)")

    def def_names(lines: list[str]) -> set[str]:
        # [FUNCTION DOC] HELPER: Names of functions (optionally async) defined on any of the
        # given diff lines, at any indentation.
        return {m.group(1) for ln in lines if (m := re.match(r"\s*(?:async\s+)?def\s+(\w+)", ln))}

    def class_names(lines: list[str]) -> set[str]:
        # [FUNCTION DOC] HELPER: Names of classes defined on any of the given diff lines, only
        # when the class keyword starts at the very first column (top-level classes).
        return {m.group(1) for ln in lines if (m := re.match(r"class\s+(\w+)", ln))}

    new_files = {p for p, f in files.items() if f["new"]}

    # A new file is a staged module body cut from one source: its top-level defs and classes
    # are exactly the relocated symbols (the prep commit inlined them, scaffolding included, as
    # a trailing block in the source). Take the symbol list from the new file itself so a
    # moved class -- not just a moved def -- is in the cut tail.
    for dst in sorted(new_files):
        dst_after = _git_output(["show", f"{commit}:{dst}"], root)
        try:
            dst_body = ast.parse(dst_after).body
        except SyntaxError:
            continue
        symbols = [
            node.name for node in dst_body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if not symbols:
            continue
        srcs = {
            p
            for name in symbols
            for p, f in files.items()
            if p not in new_files and name in _removed_symbol_names(f["removed"])
        }
        if len(srcs) != 1:
            recipe.supported = False
            recipe.notes.append(f"{dst}: extract source not a single file ({sorted(srcs)})")
            continue
        src = next(iter(srcs))
        src_before = _git_output(["show", f"{commit}^:{src}"], root)
        if _symbols_form_tail(src_before, symbols):
            recipe.extracts.append(
                {
                    "src": src,
                    "dst": dst,
                    "symbols": symbols,
                    "future_import": _wants_future_import(files, src, dst),
                }
            )
            continue
        src_top_level = {
            node.name
            for node in ast.parse(src_before).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        if not set(symbols) <= src_top_level:
            recipe.supported = False
            recipe.notes.append(
                f"{dst}: relocated symbols are not all top-level in {src} "
                "(a method still inside a class needs prep to lift it out first)"
            )
            continue
        # The symbols are scattered in the source (not a staged trailing block): cut each one
        # verbatim and assemble the new module under its authored header (imports/constants/
        # logger/TYPE_CHECKING reproduced from the target). The defs are the proven relocation;
        # only the small header is authored, so the prep no longer has to gather them at the
        # source tail first.
        layout = _scatter_extract_layout(dst_after, symbols)
        if layout is None:
            recipe.supported = False
            recipe.notes.append(
                f"{dst}: relocated symbols are not a trailing block in the new module "
                "(a non-symbol statement is interleaved with them)"
            )
            continue
        # A module-level constant the source no longer assigns but the new module does (e.g. a
        # ``_is_hip = is_hip()`` flag) relocated into the header, so it is dropped from the
        # source too -- its copy is reproduced in the authored header.
        src_after = _git_output(["show", f"{commit}:{src}"], root)
        drop_assigns = sorted(
            (_module_assign_names(src_before) - _module_assign_names(src_after)) & _module_assign_names(dst_after)
        )
        recipe.scatter_extracts.append(
            {
                "src": src,
                "dst": dst,
                "symbols": symbols,
                "header": layout["header"],
                "order": layout["order"],
                "drop_assigns": drop_assigns,
            }
        )

    all_removed = [ln for f in files.values() for ln in f["removed"]]
    all_added = [ln for f in files.values() for ln in f["added"]]

    # A top-level class relocated between existing files moves as one block: move_symbol
    # cuts the ClassDef; its methods are excluded from the per-def loop below.
    moved_classes: set[str] = set()
    for cname in sorted(class_names(all_removed) & class_names(all_added)):
        csrc = next((p for p, f in files.items() if cname in class_names(f["removed"])), None)
        cdst = next((p for p, f in files.items() if cname in class_names(f["added"])), None)
        if csrc is None or cdst is None or csrc == cdst or cdst in new_files:
            continue
        cdst_tree = ast.parse(_git_output(["show", f"{commit}:{cdst}"], root))
        cdst_def = rr._find_def(cdst_tree, cname) or next(
            (n for n in ast.walk(cdst_tree) if isinstance(n, ast.ClassDef) and n.name == cname),
            None,
        )
        moved_classes.add(cname)
        recipe.moves.append(
            {
                "name": cname,
                "src": csrc,
                "dst": cdst,
                "into_class": None,
                "from_class": None,
                "dedent": 0,
                "dst_order": cdst_def.lineno if cdst_def else 0,
                "before": _next_sibling_def_name(cdst_tree, cname, None),
                "drop_self_annotation": False,
            }
        )

    # A move whose destination already exists becomes a move_symbol (the def relocated in
    # order).
    for name in sorted(def_names(all_removed) & def_names(all_added)):
        src = next((p for p, f in files.items() if name in def_names(f["removed"])), None)
        dst = next(
            (p for p, f in files.items() if name in def_names(f["added"]) and p != src),
            None,
        )
        # A def cut and re-added within the same file (no other file gained it) is an
        # in-file reorder -- a move_symbol whose src and dst are that file. A signature or
        # body edit that happens to touch the def line is not a faithful move, but the
        # reproduction's byte-diff surfaces it as a residual, so this never false-passes.
        if dst is None and src is not None and name in def_names(files[src]["added"]):
            dst = src
        if src is None or dst is None or dst in new_files:
            continue
        src_before = _git_output(["show", f"{commit}^:{src}"], root)
        if _nested_in_function(ast.parse(src_before), name):
            recipe.notes.append(f"skip {name}: nested function (moves with parent)")
            continue
        src_tree = ast.parse(src_before)
        dst_tree = ast.parse(_git_output(["show", f"{commit}:{dst}"], root))
        src_indent = _def_indent(files[src]["removed"], name)
        dst_indent = _def_indent(files[dst]["added"], name)
        # The diff's def-line indentation says which same-named def actually moved: a
        # column-0 cut is the module-level def even when a class method shares its name.
        src_class = None if src_indent == 0 else _enclosing_class_of_def(src_tree, name)
        if src_class in moved_classes:
            recipe.notes.append(f"skip {name}: method of relocated class {src_class}")
            continue
        into_class = None if dst_indent == 0 else _enclosing_class_of_def(dst_tree, name)
        try:
            src_def = rr._find_unique_def(src_tree, name, from_class=src_class, where=src)
            dst_def = rr._find_unique_def(dst_tree, name, from_class=into_class, where=dst)
        except AssertionError as exc:
            recipe.supported = False
            recipe.notes.append(f"{name}: cannot disambiguate moved def ({exc})")
            continue
        src_indent = src_indent or 0
        dst_indent = dst_indent or 0
        # A same-named def re-added to the source is a forwarding delegate the move
        # leaves behind: a body of exactly `return self.<attr>.<name>(...)` names the
        # component attribute move_symbol authors the stub through.
        leave_delegate = None
        delegate_name = None
        if name in def_names(files[src]["added"]):
            src_after_tree = ast.parse(_git_output(["show", f"{commit}:{src}"], root))
            stub = _delegate_stub_attr(src_after_tree, name)
            if stub is not None:
                leave_delegate, forwarded = stub
                if forwarded != name:
                    delegate_name = forwarded
        move_before, move_after = _module_move_anchor(dst_tree, name, into_class)
        recipe.moves.append(
            {
                "name": name,
                "src": src,
                "dst": dst,
                "into_class": into_class,
                "from_class": src_class,
                "dedent": src_indent - dst_indent,
                "dst_order": dst_def.lineno if dst_def else 0,
                "before": move_before,
                "after": move_after,
                "drop_self_annotation": _self_annotation_dropped(src_def, dst_def),
                "leave_delegate": leave_delegate,
                "delegate_name": delegate_name,
            }
        )
        if src_class is not None:
            _infer_call_adaptations(
                recipe,
                files,
                name=name,
                src=src,
                src_class=src_class,
                into_class=into_class,
                commit=commit,
                root=root,
            )
        else:
            _infer_function_scoped_repaths(recipe, files, name=name, src=src, dst=dst, commit=commit, root=root)

    # A method whose signature line never changed leaves no def-line in the removed set:
    # the body was replaced by a forwarding stub in place while the full body landed in
    # another file. Detect it from the destination's added def + the source's after-state
    # delegate stub.
    for name in sorted(def_names(all_added) - def_names(all_removed)):
        dst = next((p for p, f in files.items() if name in def_names(f["added"])), None)
        if dst is None or dst in new_files:
            continue
        src = None
        stub_info = None
        for p in files:
            if p == dst:
                continue
            try:
                after_tree = ast.parse(_git_output(["show", f"{commit}:{p}"], root))
                before_tree = ast.parse(_git_output(["show", f"{commit}^:{p}"], root))
            except Exception:
                continue
            stub = _delegate_stub_attr(after_tree, name)
            if stub is None:
                continue
            full_before = rr._find_def(before_tree, name)
            if full_before is None or _delegate_stub_attr(before_tree, name):
                continue
            src, stub_info = p, stub
            break
        if src is None:
            continue
        src_before = _git_output(["show", f"{commit}^:{src}"], root)
        src_tree = ast.parse(src_before)
        dst_tree = ast.parse(_git_output(["show", f"{commit}:{dst}"], root))
        src_class = _enclosing_class_of_def(src_tree, name)
        dst_indent = _def_indent(files[dst]["added"], name)
        into_class = None if dst_indent == 0 else _enclosing_class_of_def(dst_tree, name)
        try:
            src_def = rr._find_unique_def(src_tree, name, from_class=src_class, where=src)
            dst_def = rr._find_unique_def(dst_tree, name, from_class=into_class, where=dst)
        except AssertionError as exc:
            recipe.supported = False
            recipe.notes.append(f"{name}: cannot disambiguate moved def ({exc})")
            continue
        leave_delegate, forwarded = stub_info
        move_before, move_after = _module_move_anchor(dst_tree, name, into_class)
        recipe.moves.append(
            {
                "name": name,
                "src": src,
                "dst": dst,
                "into_class": into_class,
                "from_class": src_class,
                "dedent": (src_def.col_offset or 0) - (dst_indent or 0),
                "dst_order": dst_def.lineno if dst_def else 0,
                "before": move_before,
                "after": move_after,
                "drop_self_annotation": _self_annotation_dropped(src_def, dst_def),
                "leave_delegate": leave_delegate,
                "delegate_name": forwarded if forwarded != name else None,
            }
        )

    # A module-level constant that vanished from one changed file and appeared in another
    # relocated with the moved code: realise it as a move_assign.
    changed_paths = [p for p in files if p.endswith(".py")]
    texts_before: dict = {}
    texts_after: dict = {}
    for p in changed_paths:
        try:
            texts_before[p] = "" if files[p]["new"] else _git_output(["show", f"{commit}^:{p}"], root)
            texts_after[p] = _git_output(["show", f"{commit}:{p}"], root)
        except Exception:
            continue
    for p_src in changed_paths:
        if p_src not in texts_before:
            continue
        lost = _module_assign_names(texts_before[p_src]) - _module_assign_names(texts_after.get(p_src, ""))
        if not lost:
            continue
        for p_dst in changed_paths:
            if p_dst == p_src or p_dst in new_files or p_dst not in texts_after:
                continue
            gained = _module_assign_names(texts_after[p_dst]) - _module_assign_names(texts_before.get(p_dst, ""))
            for cname in sorted(lost & gained):
                dst_tree = ast.parse(texts_after[p_dst])
                recipe.assign_moves.append(
                    {
                        "name": cname,
                        "src": p_src,
                        "dst": p_dst,
                        "before": _next_sibling_assign_or_def(dst_tree, cname),
                    }
                )

    # Module-level imports a file gained or lost are realised directly from the symmetric
    # base<->target diff: a gained name is added (the destination needs the moved code's
    # imports, or a caller of a moved free function gains one), a lost name is removed. An
    # import diff is always whitelisted, so this is deterministic and does not depend on the
    # formatter pruning (this repo's ruff has no F811, so a still-used symbol repointed to a new
    # module would otherwise leave a duplicate). A file written whole by extract_to_new_module
    # (the new file, or the extract source whose tail the cut removed) is skipped.
    extract_dsts = {ex["dst"] for ex in recipe.extracts}
    extract_srcs = {ex["src"] for ex in recipe.extracts}
    for path in sorted(files):
        if path in new_files or path in extract_dsts:
            continue
        before = _git_output(["show", f"{commit}^:{path}"], root)
        after = _git_output(["show", f"{commit}:{path}"], root)
        before_pairs = _import_pairs(before) if before.strip() else {}
        after_pairs = _import_pairs(after) if after.strip() else {}
        recipe.import_additions.extend(_import_additions(path, after, before_pairs, after_pairs))
        if path not in extract_srcs:
            for key in before_pairs:
                if key not in after_pairs:
                    recipe.module_import_removals.append(_removal_from_key(path, key))
        before_tc = _typechecking_pairs(before) if before.strip() else {}
        after_tc = _typechecking_pairs(after) if after.strip() else {}
        for key, stmt in after_tc.items():
            if key not in before_tc:
                recipe.typechecking_additions.append({"path": path, "text": stmt})

    # An intra-file helper carved out of a sibling function's body (its block replaced by a
    # call) is an extract_function -- inferred only when no cross-file move already explains it.
    if not recipe.moves:
        _infer_extract_functions(recipe, files, commit, root)

    # A move source the commit deletes (its defs all relocated, leaving only scaffolding) is
    # removed after the moves; move_symbol only cuts defs, it does not delete the emptied file.
    move_srcs = {mv["src"] for mv in recipe.moves}
    for path, f in files.items():
        if f.get("deleted") and path in move_srcs:
            recipe.deletes.append(path)

    if not recipe.moves and not recipe.extracts and not recipe.scatter_extracts and not recipe.extract_functions:
        recipe.supported = False
        if not recipe.notes:
            recipe.notes.append("no def relocated (rename or statement-level change): review as prep")
    return recipe


def _recipe_ops(recipe: Recipe) -> list:
    # [FUNCTION DOC] HELPER: Flattens the recipe into the ordered list of operations to replay,
    # each a triple of the builder method name, a tuple of positional arguments and a mapping of
    # keyword arguments. Order: call-site rewrites (requalify when the kind says so, else lower);
    # import repaths; in-function import removals; symbol moves sorted by destination path and
    # then by descending destination line number (so each earlier-in-file anchor already
    # exists); assignment moves; function extractions; tail extracts to new modules (symbols,
    # future import flag); scattered extracts (symbols, header, order, dropped assignments); file
    # deletions; module-level imported-name removals (module, name, alias); import additions
    # (path, text); type-checking import additions (path, text). The optional move fields
    # (from class, before, after, delegate, delegate name) are passed through, being nothing
    # when missing. The builder methods for routing call sites through a field and for adding
    # a name to an existing import are never produced.
    """The ordered relocation operations a recipe replays, as ``(method, args, kwargs)`` --
    shared by ``build_repro`` (which runs them on a Repro) and ``recipe_to_script`` (which
    renders them as ``r.method(...)`` lines), so the emitted script and the in-process run can
    never drift.

    Call sites and import repaths/removals run BEFORE the moves, so a call to a moved method
    from inside another moved method is adapted while still in the source and travels with the
    body. The moves (in destination order) and the new-module extracts relocate next.
    Module-level import additions/removals run LAST, so a consumer import lands after an extract
    has cut the source tail (otherwise it would be swept into the new module). Same-destination
    moves are emitted in reverse destination order so each move's ``before`` anchor (a sibling
    further down) is already present when the move is inserted."""
    ops: list = []
    for lo in recipe.lowerings:
        method = "requalify_call_sites" if lo["kind"] == "requalify" else "lower_call_sites"
        ops.append((method, (lo["name"], lo["owner"]), {"paths": [lo["path"]]}))
    for rp in recipe.repaths:
        ops.append(
            (
                "repath_import",
                (rp["path"],),
                {
                    "old_module": rp["old_module"],
                    "new_module": rp["new_module"],
                    "name": rp["name"],
                },
            )
        )
    for im in recipe.import_removals:
        ops.append(
            (
                "remove_import",
                (im["path"], im["text"]),
                {"in_function": im["in_function"]},
            )
        )
    for mv in sorted(recipe.moves, key=lambda m: (m["dst"], -m["dst_order"])):
        ops.append(
            (
                "move_symbol",
                (mv["name"],),
                {
                    "src": mv["src"],
                    "dst": mv["dst"],
                    "into_class": mv["into_class"],
                    "from_class": mv.get("from_class"),
                    "dedent": mv["dedent"],
                    "drop_self_annotation": mv["drop_self_annotation"],
                    "before": mv.get("before"),
                    "after": mv.get("after"),
                    "leave_delegate": mv.get("leave_delegate"),
                    "delegate_name": mv.get("delegate_name"),
                },
            )
        )
    for am in recipe.assign_moves:
        ops.append(
            (
                "move_assign",
                (am["name"],),
                {"src": am["src"], "dst": am["dst"], "before": am.get("before")},
            )
        )
    for ex in recipe.extract_functions:
        ops.append(
            (
                "extract_function",
                (ex["src"], ex["dst"]),
                {
                    "name": ex["name"],
                    "signature": ex["signature"],
                    "body": ex["body"],
                    "body_indent": ex["body_indent"],
                    "call": ex["call"],
                    "return_text": ex["return_text"],
                    "into_class": ex["into_class"],
                    "before": ex["before"],
                },
            )
        )
    for ex in recipe.extracts:
        ops.append(
            (
                "extract_to_new_module",
                (ex["src"], ex["dst"]),
                {"symbols": ex["symbols"], "future_import": ex["future_import"]},
            )
        )
    for ex in recipe.scatter_extracts:
        ops.append(
            (
                "extract_symbols_to_new_module",
                (ex["src"], ex["dst"]),
                {
                    "symbols": ex["symbols"],
                    "header": ex["header"],
                    "order": ex["order"],
                    "drop_assigns": ex["drop_assigns"],
                },
            )
        )
    for path in recipe.deletes:
        ops.append(("delete_file", (path,), {}))
    for im in recipe.module_import_removals:
        ops.append(
            (
                "remove_imported_name",
                (im["path"],),
                {"module": im["module"], "name": im["name"], "asname": im["asname"]},
            )
        )
    for im in recipe.import_additions:
        ops.append(("add_import", (im["path"], im["text"]), {}))
    for im in recipe.typechecking_additions:
        ops.append(("add_typechecking_import", (im["path"], im["text"]), {}))
    return ops


def build_repro(recipe: Recipe, repo_root: str | None = None) -> rr.Repro:
    # [FUNCTION DOC] Creates a reproduction builder with the recipe's base and target commits
    # and the optional repository root, then calls on it, in order, each operation produced by
    # the operations helper. Returns the builder, ready to run; nothing is executed or touched
    # on disk here.
    """Compose a Repro from the recipe's canonical ordered operations (``_recipe_ops``)."""
    repro = rr.Repro(base=recipe.base, target=recipe.target, repo_root=repo_root)
    for method, args, kwargs in _recipe_ops(recipe):
        getattr(repro, method)(*args, **kwargs)
    return repro


def recipe_to_script(recipe: Recipe, subject: str) -> str:
    # [FUNCTION DOC] Renders the recipe as the text of a standalone python script and returns
    # it (no I/O). The script has a docstring naming the target commit and the subject; imports
    # sys and the path class; inserts the grandparent directory of the script's own file (the
    # folder holding the utilities module, since scripts live one folder below it) at the front
    # of the import path; imports the builder class from the utilities module; constructs it
    # with the literal base and target; then has one line per operation of the operations helper
    # calling the builder method with literal (repr-style) positional and keyword arguments;
    # finally runs it and exits with status one if a residual diff text came back, else zero.
    # The text ends with a newline.
    # KNOWN ISSUE: the subject is placed unescaped inside a triple-quoted docstring, so a subject
    # containing triple quotes or a backslash escape that is invalid would break the script.
    """A standalone, auditable reproduce script (imports only the reproduce util)."""
    lines = [
        '"""Auto-generated reproduce script. Audit each call, then run.',
        "",
        f"commit:  {recipe.target}",
        f"subject: {subject}",
        "",
        "Each call is a faithful relocation primitive. Running this reproduces the commit",
        "in a throwaway worktree and diffs it byte-for-byte; PASS means the commit is",
        "exactly these relocations.",
        '"""',
        "import sys",
        "from pathlib import Path",
        "",
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))",
        "from mechanical_refactor_reproduction_utils import Repro",
        "",
        f"r = Repro(base={recipe.base!r}, target={recipe.target!r})",
    ]
    for method, args, kwargs in _recipe_ops(recipe):
        rendered = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        lines.append(f"r.{method}(" + ", ".join(rendered) + ")")
    lines += ["residual = r.run()", "sys.exit(1 if residual else 0)", ""]
    return "\n".join(lines)


@dataclass
class GenResult:
    # [CLASS DOC] Mutable record of the generation outcome for one commit: full commit id,
    # subject line, whether the recipe was supported and relocates something, whether the
    # reproduction passed with an empty residual diff, the residual diff text (or an error
    # message when generation raised), the rendered script text, and the list of note strings.
    # Serialized to a mapping for the HTML report.
    commit: str
    subject: str
    supported: bool
    passed: bool
    residual: str
    script: str
    notes: list


def generate_range(
    rev_range: str,
    *,
    match: str | None = None,
    out_dir: str,
    repo_root: str | None = None,
) -> list[GenResult]:
    # [FUNCTION DOC] Generates proofs for every commit of a revision range. Inputs: the range
    # text, an optional regular expression (searched, not anchored, in each commit subject; only
    # matching commits are processed; all when absent), the required output folder, and an
    # optional repository root (default: found from the current directory). Lists commits oldest
    # first (an invalid range silently yields none). Creates the output folder and its
    # repro_scripts subfolder, and copies the utilities module source into the output folder so
    # scripts can import it. For each processed commit: infers the recipe, renders the script and
    # writes it as the first nine characters of the commit id plus the python extension (also for
    # unsupported recipes); the commit counts as supported only if the recipe is supported and
    # relocates something; if supported, builds the reproduction and runs it (creating and then
    # removing a temporary worktree), passed meaning an empty residual. Any exception during
    # these steps is caught: the result is unsupported, the residual holds a message with the
    # exception type and text, and that message is appended to the notes. Results are collected in
    # order; finally the plain text log and the HTML report are written into the output folder.
    # Returns the list of results.
    """For each matched commit: infer a recipe, emit repro_scripts/<sha>.py, run it, and
    record PASS / residual. Writes output.log + output.html and copies the reproduce util
    so the folder is self-contained."""
    root = repo_root or _repo_root()
    commits = _git_output(["rev-list", "--reverse", rev_range], root).split()
    pattern = re.compile(match) if match else None

    out = Path(out_dir)
    scripts_dir = out / "repro_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (out / "mechanical_refactor_reproduction_utils.py").write_text(Path(rr.__file__).read_text())

    results: list[GenResult] = []
    for commit in commits:
        subject = _git_output(["log", "-1", "--format=%s", commit], root).strip()
        if pattern is not None and not pattern.search(subject):
            continue
        passed, residual, script, supported, notes = False, "", "", False, []
        try:
            recipe = infer_recipe(commit, root)
            script = recipe_to_script(recipe, subject)
            (scripts_dir / f"{commit[:9]}.py").write_text(script)
            relocates = bool(recipe.moves or recipe.extracts or recipe.scatter_extracts or recipe.extract_functions)
            supported = recipe.supported and relocates
            notes = recipe.notes
            if supported:
                residual = build_repro(recipe, repo_root=root).run()
                passed = residual == ""
        except Exception as exc:
            supported = False
            residual = f"reproduce raised {type(exc).__name__}: {exc}"
            notes = notes + [residual]
        results.append(
            GenResult(
                commit=commit,
                subject=subject,
                supported=supported,
                passed=passed,
                residual=residual,
                script=script,
                notes=notes,
            )
        )

    _write_log(out / "output.log", rev_range, results)
    _write_html(out / "output.html", rev_range, results)
    return results


def _write_log(path: Path, rev_range: str, results: list[GenResult]) -> None:
    # [FUNCTION DOC] HELPER: Writes the plain text summary log. Line one names the range; line
    # two gives the number of commits with the counts reproduced and not; a blank line; then one
    # line per result: first nine characters of the commit id, a verdict, and the subject. The
    # verdict is PASS if passed; otherwise UNSUPPORTED followed by the notes joined with
    # semicolons in parentheses if not supported; otherwise RESIDUAL with the residual's line
    # count. Overwrites the file; ends with a newline.
    n_pass = sum(1 for r in results if r.passed)
    lines = [
        f"reproduce-gen: {rev_range}",
        f"{len(results)} commit(s): {n_pass} reproduced, {len(results) - n_pass} not",
        "",
    ]
    for r in results:
        if r.passed:
            verdict = "PASS"
        elif not r.supported:
            verdict = "UNSUPPORTED (" + "; ".join(r.notes) + ")"
        else:
            verdict = f"RESIDUAL ({len(r.residual.splitlines())} lines)"
        lines.append(f"{r.commit[:9]}  {verdict}  {r.subject}")
    path.write_text("\n".join(lines) + "\n")


def _write_html(path: Path, rev_range: str, results: list[GenResult]) -> None:
    # [FUNCTION DOC] HELPER: Writes the self-contained HTML report. Builds a payload mapping
    # (title = range text, pass count, total, and each result converted to a mapping), serializes
    # it as JSON without ASCII escaping and with every closing-tag opener escaped so it cannot
    # end the embedding script element, and substitutes it for the data placeholder in the HTML
    # template, while the HTML-escaped range text replaces the title placeholder. Overwrites the
    # file.
    payload = {
        "title": rev_range,
        "passed": sum(1 for r in results if r.passed),
        "total": len(results),
        "results": [asdict(r) for r in results],
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(_HTML_TEMPLATE.replace("__TITLE__", html.escape(rev_range)).replace("__DATA_JSON__", data))


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>reproduce-gen __TITLE__</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#fff;color:#1f2328;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #d1d9e0;padding:10px 16px}
header h1{margin:0 0 4px;font-size:15px}
.chip{display:inline-block;padding:1px 8px;border-radius:10px;margin-right:6px;font-size:12px}
.chip.ok{background:#e6ffec;border:1px solid #4ac26b}
.chip.no{background:#ffebe9;border:1px solid #ff8182}
main{padding:8px 16px 40px}
.card{border:1px solid #d1d9e0;border-left-width:4px;border-radius:6px;margin:8px 0}
.card.ok{border-left-color:#4ac26b}.card.no{border-left-color:#ff8182}.card.un{border-left-color:#d4a72c}
.hd{padding:6px 10px;cursor:pointer;font-size:13px;display:flex;align-items:center;gap:8px}
.hd:hover{background:#f6f8fa}
.badge{font-size:11px;font-weight:600;padding:1px 7px;border-radius:10px;flex:none}
.badge.ok{background:#e6ffec;border:1px solid #4ac26b;color:#1a7f37}
.badge.no{background:#ffebe9;border:1px solid #ff8182;color:#cf222e}
.badge.un{background:#fff8c5;border:1px solid #d4a72c;color:#7d4e00}
.hd code{color:#59636e;flex:none}
.bd{display:none;padding:6px 10px;border-top:1px solid #f0f1f3;font:11.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.card.open .bd{display:block}
.k{color:#59636e;margin-top:6px}
pre{white-space:pre-wrap;word-break:break-word;background:#f6f8fa;padding:8px;border-radius:6px;overflow-x:auto}
.res .d{white-space:pre-wrap;word-break:break-word}
.res .add{background:#e6ffec}.res .del{background:#ffebe9}.res .h{background:#eff5ff;color:#0550ae}
</style></head>
<body>
<header><h1>reproduce-gen: <code>__TITLE__</code></h1>
<div><span id="counts"></span></div></header>
<main id="list"></main>
<script id="data" type="application/json">__DATA_JSON__</script>
<script>
'use strict';
const DATA = JSON.parse(document.getElementById('data').textContent);
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function cls(r){ return r.passed?'ok':(r.supported?'no':'un'); }
function label(r){ return r.passed?'PASS':(r.supported?'RESIDUAL':'UNSUPPORTED'); }
let out='';
DATA.results.forEach(r => {
  const c = cls(r);
  out += '<div class="card '+c+'"><div class="hd"><span class="badge '+c+'">'+label(r)+'</span>';
  out += '<code>'+esc(r.commit.slice(0,9))+'</code><span>'+esc(r.subject)+'</span></div><div class="bd">';
  if(r.notes.length) out += '<div class="k">notes:</div><pre>'+esc(r.notes.join('\\n'))+'</pre>';
  out += '<div class="k">repro script:</div><pre>'+esc(r.script)+'</pre>';
  if(r.residual){
    out += '<div class="k">residual diff:</div><div class="res">';
    r.residual.split('\\n').forEach(l => {
      const k = l.startsWith('@@')?'h':(l[0]==='+'?'add':(l[0]==='-'?'del':''));
      out += '<div class="d '+k+'">'+esc(l)+'</div>';
    });
    out += '</div>';
  }
  out += '</div></div>';
});
document.getElementById('counts').innerHTML =
  '<span class="chip ok">'+DATA.passed+' reproduced</span><span class="chip no">'+(DATA.total-DATA.passed)+' not</span>';
document.getElementById('list').innerHTML = out;
document.getElementById('list').addEventListener('click', e => {
  const hd = e.target.closest('.hd'); if(hd) hd.parentNode.classList.toggle('open');
});
</script></body></html>"""


def _main(argv: list[str]) -> int:
    # [FUNCTION DOC] HELPER: Command line entry point taking the argument list without the
    # program name. Manually extracts two optional flags with a following value each: the output
    # folder flag and the subject-regex flag (a flag given last without a value raises an index
    # error). Exactly one positional argument must remain, otherwise prints usage to standard
    # error and returns 2. A positional containing two consecutive dots is a revision range: it
    # requires the output folder (assertion error otherwise), generates the proofs for the
    # range, prints a line with the reproduced count, total and folder, and returns 0
    # regardless of how many commits reproduced. Otherwise it is a single commit: infers the
    # recipe, prints the rendered script to standard output, and if the recipe is unsupported or
    # relocates nothing prints an unsupported message with the notes to standard error and returns
    # 1; else runs the reproduction and returns 0 for an empty residual, 1 otherwise.
    out_dir = None
    if "--out" in argv:
        i = argv.index("--out")
        out_dir = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    match = None
    if "--match" in argv:
        i = argv.index("--match")
        match = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    if len(argv) != 1:
        print(
            "usage: python3 mechanical_refactor_proof_generator.py <commit>\n"
            "       python3 mechanical_refactor_proof_generator.py <base>..<tip> "
            "[--match REGEX] --out DIR",
            file=sys.stderr,
        )
        return 2
    target = argv[0]
    if ".." in target:
        assert out_dir, "--out DIR is required for a range"
        results = generate_range(target, match=match, out_dir=out_dir)
        n = sum(1 for r in results if r.passed)
        print(f"{n}/{len(results)} reproduced; folder: {out_dir}")
        return 0
    root = _repo_root()
    recipe = infer_recipe(target, root)
    print(recipe_to_script(recipe, _git_output(["log", "-1", "--format=%s", target], root)))
    relocates = bool(recipe.moves or recipe.extracts or recipe.scatter_extracts or recipe.extract_functions)
    if not (recipe.supported and relocates):
        print("UNSUPPORTED: " + "; ".join(recipe.notes), file=sys.stderr)
        return 1
    residual = build_repro(recipe, repo_root=root).run()
    return 0 if residual == "" else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
