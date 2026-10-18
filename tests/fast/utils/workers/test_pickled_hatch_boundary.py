# [MODULE DOC] Source scanning test that keeps the pickled escape hatch of the rpc wire types on a
# whitelist. It parses every shipped framework module (excluding the tests directory) with the syntax
# tree library and finds uses of the hatch annotation: method parameters and return annotations of
# methods inside classes, and annotated fields in classes, including unions, subscripts, aliased
# imports, module level aliases, quoted annotations and methods nested in blocks. The set of found
# (file, qualified name, parameter) triples must equal the whitelist, which authorizes only the args
# parameter of trainer init methods. A second group of tests feeds small source snippets to the
# scanner to prove it sees each shape. Also checks the hatch definition carries a greppable marker
# and a todo.
from __future__ import annotations

import ast
import functools
from pathlib import Path

from tests.fast.source_scan import FRAMEWORK_ROOT, REPO_ROOT, shipped_modules

from miles.utils.workers.rpc.common.wire_types import PICKLED_HATCH_MARKER

EXCLUDED_DIRS = (REPO_ROOT / "tests",)

HATCH_NAME = "Pickled"

HATCH_DEFINITION = FRAMEWORK_ROOT / "utils" / "workers" / "rpc" / "common" / "wire_types.py"

RETURN_ANNOTATION = "return"

PICKLED_PARAMETERS = {
    ("miles/ray/train/group.py", "TrainerController.init", "args"),
    ("miles/ray/train_actor.py", "TrainRayActor.init", "args"),
    ("miles/backends/megatron_utils/actor.py", "MegatronTrainRayActor.init", "args"),
    ("miles/backends/fsdp_utils/actor.py", "FSDPTrainRayActor.init", "args"),
    ("miles/backends/torchtitan_utils/actor.py", "TorchtitanTrainRayActor.init", "args"),
}


def _hatch_aliases(tree: ast.Module) -> set[str]:
    # [FUNCTION DOC] HELPER: Returns the set of names under which the hatch type is visible in the
    # module: the hatch's own name, the name (or "as" alias) of every from-import that imports it, and
    # the target names of every plain assignment whose value is itself a hatch annotation by the
    # current alias set. Walks the whole tree in traversal order, so an alias must be assigned after
    # the names it depends on appear in the walk.
    aliases = {HATCH_NAME}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            aliases.update(alias.asname or alias.name for alias in node.names if alias.name == HATCH_NAME)
        elif isinstance(node, ast.Assign) and _is_pickled(node.value, aliases):
            aliases.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return aliases


def _is_pickled(annotation: ast.expr | None, aliases: set[str]) -> bool:
    # [FUNCTION DOC] HELPER: Decides whether an annotation expression mentions the hatch. A missing
    # annotation is false. Otherwise walks all sub-expressions and returns true on: a name in the alias
    # set; an attribute access ending in the hatch name; or a string constant that parses as an
    # expression mentioning the hatch.
    if annotation is None:
        return False
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id in aliases:
            return True
        if isinstance(node, ast.Attribute) and node.attr == HATCH_NAME:
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _is_pickled_text(node.value, aliases):
            return True
    return False


def _is_pickled_text(text: str, aliases: set[str]) -> bool:
    # [FUNCTION DOC] HELPER: Parses a quoted annotation (stripped of whitespace) as an expression and
    # applies the hatch check to it; text that is not valid syntax counts as not pickled.
    try:
        parsed = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return False
    return _is_pickled(parsed.body, aliases)


def _pickled_annotations_in(source: str, *, label: str) -> set[tuple[str, str, str]]:
    # [FUNCTION DOC] HELPER: Scans module source text and returns the set of (label, qualified name,
    # item) triples using the hatch. For every class definition anywhere in the tree (nested classes
    # included), every function or async function found anywhere below it contributes a triple per
    # positional-only, ordinary or keyword-only parameter with a hatch annotation (qualified name is
    # class name dot function name, item is the parameter name) and one with item "return" when its
    # return annotation uses the hatch; every annotated assignment found below the class contributes
    # (label, class name, target text). Because the walk is not scoped, a method of a nested class is
    # also attributed to the outer class and counted again, and functions nested in methods are
    # attributed to the class too.
    tree = ast.parse(source)
    aliases = _hatch_aliases(tree)
    found: set[tuple[str, str, str]] = set()
    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue
        for node in ast.walk(class_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = node.args
                for parameter in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]:
                    if _is_pickled(parameter.annotation, aliases):
                        found.add((label, f"{class_node.name}.{node.name}", parameter.arg))
                if _is_pickled(node.returns, aliases):
                    found.add((label, f"{class_node.name}.{node.name}", RETURN_ANNOTATION))
            elif isinstance(node, ast.AnnAssign) and _is_pickled(node.annotation, aliases):
                found.add((label, class_node.name, ast.unparse(node.target)))
    return found


def _pickled_annotations_of(path: Path) -> set[tuple[str, str, str]]:
    # [FUNCTION DOC] HELPER: Reads the file and scans it, labelling results with the path relative to
    # the repository root as text (platform path separators; the whitelist uses forward slashes).
    return _pickled_annotations_in(path.read_text(), label=str(path.relative_to(REPO_ROOT)))


@functools.cache
def _all_pickled_annotations() -> frozenset[tuple[str, str, str]]:
    # [FUNCTION DOC] HELPER: Cached (computed once per process). Scans every shipped module, excluding
    # the tests directory, and returns the union of all found triples as a frozen set. Reads many
    # files from disk.
    return frozenset(
        entry for path in shipped_modules(exclude_dirs=EXCLUDED_DIRS) for entry in _pickled_annotations_of(path)
    )


class TestThePickledHatchStaysWhitelisted:
    # [CLASS DOC] Tests the whole shipped code base against the whitelist.
    def test_nothing_uses_the_hatch_without_being_listed(self):
        # [FUNCTION DOC] Asserts there is no found usage outside the whitelist; the failure message
        # lists the unlisted triples.
        """The hatch punches a hole in the strict wire typing, so every user of it is reviewed here."""
        unlisted = sorted(_all_pickled_annotations() - PICKLED_PARAMETERS)

        assert unlisted == [], (
            f"{unlisted} use the Pickled escape hatch; it is reserved for the argparse Namespace a trainer "
            f"is built from, so give the parameter a wire type instead"
        )

    def test_the_list_names_no_parameter_that_no_longer_uses_it(self):
        # [FUNCTION DOC] Asserts every whitelist entry is still found in the code (no stale entries).
        """A stale entry is an invitation to reach for pickle where a wire type already exists."""
        stale = sorted(PICKLED_PARAMETERS - _all_pickled_annotations())

        assert stale == []

    def test_every_listed_parameter_is_the_args_namespace(self):
        # [FUNCTION DOC] Asserts the set of parameter names in the whitelist is exactly {args}.
        """Only args was authorized; anything else crossing as pickle is a different decision."""
        assert {parameter for _path, _method, parameter in PICKLED_PARAMETERS} == {"args"}

    def test_the_hatch_carries_a_greppable_marker_and_a_todo(self):
        # [FUNCTION DOC] Reads the hatch definition file and asserts it contains a todo tagged with the
        # hatch marker constant and the phrase "arguments subsystem".
        """It is temporary, and the way it gets reclaimed is by finding it."""
        source = HATCH_DEFINITION.read_text()

        assert f"TODO({PICKLED_HATCH_MARKER})" in source
        assert "arguments subsystem" in source


class TestTheShapesThatUsedToCarryTheHatchThrough:
    # [CLASS DOC] Tests the scanner on small inline sources; each expects exactly one triple from the
    # file labelled "w.py" (or none for the negative case).
    def test_a_union_is_seen(self):
        # [FUNCTION DOC] A parameter annotated as the hatch unioned with none is found.
        """`Pickled | None` is the same hole with an extra bar in it."""
        source = "class W:\n    def go(self, args: Pickled | None) -> None: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", "args")}

    def test_a_subscript_is_seen(self):
        # [FUNCTION DOC] A parameter annotated as a list of the hatch is found.
        """A container of pickled values pickles just as much as one value does."""
        source = "class W:\n    def go(self, args: list[Pickled]) -> None: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", "args")}

    def test_an_aliased_import_is_seen(self):
        # [FUNCTION DOC] A parameter using the hatch imported under another name is found.
        """Renaming the hatch on the way in does not make the parameter wire-typed."""
        source = "from miles.utils.workers.rpc.common.wire_types import Pickled as _P\n"
        source += "class W:\n    def go(self, args: _P) -> None: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", "args")}

    def test_a_module_level_alias_is_seen(self):
        # [FUNCTION DOC] A parameter using a module level alias assignment of the hatch is found.
        """A second name for the same annotation is still the same annotation."""
        source = "Blob = Pickled\nclass W:\n    def go(self, args: Blob) -> None: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", "args")}

    def test_a_string_annotation_is_seen(self):
        # [FUNCTION DOC] A quoted annotation of the hatch unioned with none is found.
        """Quoting an annotation is how a forward reference is written, not how a rule is escaped."""
        source = 'class W:\n    def go(self, args: "Pickled | None") -> None: ...\n'

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", "args")}

    def test_a_method_defined_under_a_type_checking_guard_is_seen(self):
        # [FUNCTION DOC] A method defined inside an if block in the class body is found.
        """A method nested in any block is still a method the served class exposes."""
        source = "class W:\n    if TYPE_CHECKING:\n        def go(self, args: Pickled) -> None: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", "args")}

    def test_a_return_annotation_is_seen(self):
        # [FUNCTION DOC] A method returning the hatch is found with the item "return".
        """A method that answers pickle sends it in the direction the parameter check never looks."""
        source = "class W:\n    def go(self) -> Pickled: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", RETURN_ANNOTATION)}

    def test_a_wire_model_field_is_seen(self):
        # [FUNCTION DOC] An annotated field of a model class is found with the class name and the
        # field name.
        """A hatch inside a model field crosses the wire on every call that carries the model."""
        source = "class M(StrictBaseModel):\n    args: Pickled\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "M", "args")}

    def test_a_wire_typed_parameter_is_not_reported(self):
        # [FUNCTION DOC] A parameter annotated as an integer yields no triples.
        """A check that reports everything would make the ledger meaningless."""
        source = "class W:\n    def go(self, args: int) -> None: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == set()
