# [MODULE DOC] Tests for inferring symbol moves in the proof generator recipe: methods moved onto
# classes or out to free functions (with call-site adaptation), free functions moved between
# modules (with import repaths), anchors for the insertion point, whole class moves, forwarding
# delegates left behind, relocated module constants, in-file reorders, and robustness (nested
# functions, same-named methods, non-python files in the commit). Each test commits a base and a
# second version in the shared repository fixture and checks the inferred recipe's fields.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generator_testlib import (  # noqa: F401
    _commit,
    _free_function_move_with_module_level_caller,
    _git,
    _method_onto_class,
    _write,
)
from mechanical_refactor_proof_generator import infer_recipe, recipe_to_script


def test_infer_recipe_method_onto_class(repo: Path) -> None:
    # [FUNCTION DOC] Uses the canned scenario of a static method moved from class M onto class C
    # with its caller lowered. Asserts: recipe supported; exactly one move whose tuple of (name,
    # source, destination, enclosing class, dedent) is foo, the model file, the comp file, C,
    # zero; exactly one lowering record (name foo, owner M, caller file, kind lower); exactly one
    # in-function import removal (caller file, the local import of M, function run); and no
    # import additions.
    """A method move onto a class infers the move, the call-site lowering, and the orphaned
    local import removal."""
    _method_onto_class(repo)
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported
    assert [(m["name"], m["src"], m["dst"], m["into_class"], m["dedent"]) for m in recipe.moves] == [
        ("foo", "model.py", "comp.py", "C", 0)
    ]
    assert recipe.lowerings == [{"name": "foo", "owner": "M", "path": "caller.py", "kind": "lower"}]
    assert recipe.import_removals == [{"path": "caller.py", "text": "from model import M", "in_function": "run"}]
    assert recipe.import_additions == []


def test_infer_recipe_move_before_typechecking_uses_after_anchor(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: model with keep and helper; util with an import, a constant
    # assigned from a call, and a type-checking conditional. Second commit moves helper into
    # util between the constant and the conditional. Asserts the recipe is supported with exactly
    # one move of helper to util, whose before anchor is nothing and whose after anchor is the
    # constant's name.
    """A module-level def relocated to land just above an ``if TYPE_CHECKING:`` guard cannot
    be anchored with before= (the next def sits past the guard), so the recipe anchors it with
    after=<the preceding assignment>."""
    _write(
        repo,
        **{
            "model.py": ("def keep():\n    return 0\n\n\ndef helper(x):\n    return x + 1\n"),
            "util.py": (
                "from u import is_hip\n"
                "\n"
                "_is_hip = is_hip()\n"
                "\n"
                "if TYPE_CHECKING:\n"
                "    from m import Thing\n"
            ),
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n",
            "util.py": (
                "from u import is_hip\n"
                "\n"
                "_is_hip = is_hip()\n"
                "\n"
                "\n"
                "def helper(x):\n"
                "    return x + 1\n"
                "\n"
                "\n"
                "if TYPE_CHECKING:\n"
                "    from m import Thing\n"
            ),
        },
    )
    _commit(repo, "move helper above the TYPE_CHECKING guard")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported
    assert len(recipe.moves) == 1
    move = recipe.moves[0]
    assert move["name"] == "helper" and move["dst"] == "util.py"
    assert move["before"] is None
    assert move["after"] == "_is_hip"


def test_infer_recipe_free_function_move_uses_requalify(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: class M with a static method foo and a method other; util with
    # an import; caller whose method imports M locally and calls the static method with 9. Second
    # commit moves foo to a module-level function in util and the caller calls it bare. Asserts
    # one move of foo with no destination class and dedent 4, and one lowering record of kind
    # requalify (owner M, caller file).
    """A move to a module-level free function dedents and requalifies the call site
    (drops the qualifier), rather than lowering a receiver."""
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n"
                "    @staticmethod\n"
                "    def foo(x):\n"
                "        return x + 1\n"
                "\n"
                "    def other(self):\n"
                "        return 0\n"
            ),
            "util.py": "import os\n",
            "caller.py": (
                "class K:\n" "    def run(self):\n" "        from model import M\n" "\n" "        return M.foo(9)\n"
            ),
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "class M:\n    def other(self):\n        return 0\n",
            "util.py": "import os\n\n\ndef foo(x):\n    return x + 1\n",
            "caller.py": ("class K:\n    def run(self):\n        return foo(9)\n"),
        },
    )
    _commit(repo, "move foo to util as a free function")
    recipe = infer_recipe("HEAD", str(repo))
    assert [(m["name"], m["into_class"], m["dedent"]) for m in recipe.moves] == [("foo", None, 4)]
    assert recipe.lowerings == [{"name": "foo", "owner": "M", "path": "caller.py", "kind": "requalify"}]


def test_infer_recipe_excludes_the_moved_bodys_own_call(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: class M with a static method foo whose body calls foo on an
    # unrelated object named worker; comp module with class C. Second commit moves foo onto C
    # unchanged. Asserts the recipe records no call-site lowerings, since the only same-named call
    # is on a different receiver than M.
    """A same-named call on a different receiver inside the moved body is not a caller
    lowering (only `M.foo(...)` is, not `worker.foo(...)`)."""
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n"
                "    @staticmethod\n"
                "    def foo(self, x):\n"
                "        worker.foo(x)\n"
                "        return x\n"
                "\n"
                "    def other(self):\n"
                "        return 0\n"
            ),
            "comp.py": "class C:\n    def keep(self):\n        return 1\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "class M:\n    def other(self):\n        return 0\n",
            "comp.py": (
                "class C:\n"
                "    def keep(self):\n"
                "        return 1\n"
                "\n"
                "    def foo(self, x):\n"
                "        worker.foo(x)\n"
                "        return x\n"
            ),
        },
    )
    _commit(repo, "move foo onto C")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.lowerings == []


def test_infer_recipe_skips_nested_functions(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: class M whose wrap method defines an inner function and returns
    # it, plus method other; comp module with class C. Second commit moves wrap onto C. Asserts
    # the only move recorded is wrap, and that some note mentions inner (the skipped nested
    # function).
    """A def nested inside a moved method is not inferred as its own move."""
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n"
                "    def wrap(self):\n"
                "        def inner(z):\n"
                "            return z\n"
                "        return inner\n"
                "\n"
                "    def other(self):\n"
                "        return 0\n"
            ),
            "comp.py": "class C:\n    def keep(self):\n        return 1\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "class M:\n    def other(self):\n        return 0\n",
            "comp.py": (
                "class C:\n"
                "    def keep(self):\n"
                "        return 1\n"
                "\n"
                "    def wrap(self):\n"
                "        def inner(z):\n"
                "            return z\n"
                "        return inner\n"
            ),
        },
    )
    _commit(repo, "move wrap onto C")
    recipe = infer_recipe("HEAD", str(repo))
    names = [m["name"] for m in recipe.moves]
    assert names == ["wrap"]
    assert any("inner" in n for n in recipe.notes)


def test_infer_recipe_free_function_source_move_repaths_caller(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: model with keep and resolve; util with an import; caller whose
    # method imports resolve locally from model and calls it. Second commit moves resolve to util
    # and the caller's local import now names util. Asserts the recipe is supported, has exactly
    # one move (resolve from model to util, no destination class), no lowerings, and exactly one
    # repath record (caller file, old module model, new module util, name resolve).
    """A free function moved to an existing module becomes a move_symbol with the call left
    bare; a caller's function-scoped import is repathed."""
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\ndef resolve(m):\n    return m\n",
            "util.py": "import os\n",
            "caller.py": (
                "class K:\n"
                "    def run(self):\n"
                "        from model import resolve\n"
                "\n"
                "        return resolve(self.m)\n"
            ),
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n",
            "util.py": "import os\n\n\ndef resolve(m):\n    return m\n",
            "caller.py": (
                "class K:\n"
                "    def run(self):\n"
                "        from util import resolve\n"
                "\n"
                "        return resolve(self.m)\n"
            ),
        },
    )
    _commit(repo, "move resolve to util")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported
    assert [(m["name"], m["src"], m["dst"], m["into_class"]) for m in recipe.moves] == [
        ("resolve", "model.py", "util.py", None)
    ]
    assert recipe.lowerings == []
    assert recipe.repaths == [
        {
            "path": "caller.py",
            "old_module": "model",
            "new_module": "util",
            "name": "resolve",
        }
    ]


def test_infer_recipe_survives_a_non_python_file_in_the_commit(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: model with foo and keep, util with a constant, a readme file.
    # Second commit moves foo to util and also edits the readme. Asserts the moves list contains
    # only foo and some note mentions the readme file name.
    """A commit also touching a .md file infers the move and notes the non-Python path."""
    _write(
        repo,
        **{
            "model.py": "def foo():\n    return 1\n\n\ndef keep():\n    return 0\n",
            "util.py": "x = 1\n",
            "README.md": "hello\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n",
            "util.py": "x = 1\n\n\ndef foo():\n    return 1\n",
            "README.md": "hello world, this is plain markdown text\n",
        },
    )
    commit = _commit(repo, "move foo and touch docs")

    recipe = infer_recipe(commit, str(repo))

    assert [mv["name"] for mv in recipe.moves] == ["foo"]
    assert any("README.md" in note for note in recipe.notes)


def test_infer_recipe_records_the_source_class_for_disambiguation(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: model with two classes, M and Other, each having a method foo
    # with different bodies; comp module with class C. Second commit moves M's foo onto C, leaving
    # Other's untouched (M gets a pass body). Asserts the recorded move's source class list is
    # exactly M, and the rendered script contains the source-class argument naming M.
    """A method move carries from_class so the cut cannot hit a same-named other method."""
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n"
                "    def foo(self, x):\n"
                "        return x + 1\n"
                "\n"
                "\n"
                "class Other:\n"
                "    def foo(self, x):\n"
                "        return x + 2\n"
            ),
            "comp.py": "class C:\n    def keep(self):\n        return 1\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n" "    pass\n" "\n" "\n" "class Other:\n" "    def foo(self, x):\n" "        return x + 2\n"
            ),
            "comp.py": (
                "class C:\n"
                "    def keep(self):\n"
                "        return 1\n"
                "\n"
                "    def foo(self, x):\n"
                "        return x + 1\n"
            ),
        },
    )
    commit = _commit(repo, "move M.foo onto C")

    recipe = infer_recipe(commit, str(repo))

    assert [mv["from_class"] for mv in recipe.moves] == ["M"]
    script = recipe_to_script(recipe, "move M.foo onto C")
    assert "from_class='M'" in script


def test_infer_recipe_module_level_def_shadowed_by_method_name(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: model with a module-level keyword-only function foo and a class
    # M whose method foo calls the module-level foo; util with keep. Second commit moves the
    # module-level foo to util and imports it in model, leaving the method. Asserts the recipe is
    # supported, the only move is named foo, and that move has neither a source class nor a
    # destination class (the module-level definition was chosen, not the method).
    """A column-0 cut resolves to the module-level def even when a method shares its name."""
    _write(
        repo,
        **{
            "model.py": (
                "def foo(*, x):\n"
                "    return x + 1\n"
                "\n"
                "\n"
                "class M:\n"
                "    def foo(self):\n"
                "        return foo(x=self.x)\n"
            ),
            "util.py": "def keep():\n    return 1\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": (
                "from util import foo\n" "\n" "\n" "class M:\n" "    def foo(self):\n" "        return foo(x=self.x)\n"
            ),
            "util.py": ("def keep():\n" "    return 1\n" "\n" "\n" "def foo(*, x):\n" "    return x + 1\n"),
        },
    )
    commit = _commit(repo, "move module-level foo to util")

    recipe = infer_recipe(commit, str(repo))

    assert recipe.supported
    assert [mv["name"] for mv in recipe.moves] == ["foo"]
    assert recipe.moves[0]["from_class"] is None
    assert recipe.moves[0]["into_class"] is None


def test_infer_recipe_class_move_between_existing_files(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: model with class Payload (method get) and function stay; comp
    # with function keep. Second commit moves the whole Payload class to the end of comp. Asserts
    # the recipe is supported and the only move is Payload (its method get is not a separate
    # move), with no source class and no destination class.
    """A top-level class relocated to an existing module moves whole; its methods do not."""
    _write(
        repo,
        **{
            "model.py": (
                "class Payload:\n"
                "    def get(self):\n"
                "        return 1\n"
                "\n"
                "\n"
                "def stay():\n"
                "    return 2\n"
            ),
            "comp.py": "def keep():\n    return 3\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def stay():\n    return 2\n",
            "comp.py": (
                "def keep():\n"
                "    return 3\n"
                "\n"
                "\n"
                "class Payload:\n"
                "    def get(self):\n"
                "        return 1\n"
            ),
        },
    )
    commit = _commit(repo, "move Payload to comp")

    recipe = infer_recipe(commit, str(repo))

    assert recipe.supported
    assert [mv["name"] for mv in recipe.moves] == ["Payload"]
    assert recipe.moves[0]["from_class"] is None
    assert recipe.moves[0]["into_class"] is None


def test_infer_recipe_move_leaving_a_forwarding_delegate(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: class M with method work returning its argument plus one; comp
    # with class C. Second commit moves the real body onto C and leaves in M a method work that
    # returns a call of work on the comp attribute of self. Asserts the recipe is supported with
    # one move of work to the comp file, whose delegate attribute is comp and whose delegate name
    # is nothing (same name), and the rendered script contains the delegate argument naming comp.
    """A same-named stub re-added to the source infers leave_delegate on the move."""
    _write(
        repo,
        **{
            "model.py": ("class M:\n" "    def work(self, x):\n" "        return x + 1\n"),
            "comp.py": "class C:\n    def keep(self):\n        return 1\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": ("class M:\n" "    def work(self, x):\n" "        return self.comp.work(x)\n"),
            "comp.py": (
                "class C:\n"
                "    def keep(self):\n"
                "        return 1\n"
                "\n"
                "    def work(self, x):\n"
                "        return x + 1\n"
            ),
        },
    )
    commit = _commit(repo, "move M.work onto C, leaving a delegate")

    recipe = infer_recipe(commit, str(repo))

    assert recipe.supported
    assert [mv["name"] for mv in recipe.moves] == ["work"]
    assert recipe.moves[0]["dst"] == "comp.py"
    assert recipe.moves[0]["leave_delegate"] == "comp"
    assert recipe.moves[0]["delegate_name"] is None
    script = recipe_to_script(recipe, "move with delegate")
    assert "leave_delegate='comp'" in script


def test_infer_recipe_constant_relocated_with_the_move(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: model with a module constant RATIO, function work using it, and
    # function stay; comp with an import and function keep. Second commit moves both the constant
    # (placed after the import) and work to comp. Asserts the recipe is supported, the list of
    # assignment moves has exactly RATIO, and the rendered script mentions the assignment-move
    # call.
    """A module constant that vanished from the source and appeared in the existing
    destination becomes a move_assign."""
    _write(
        repo,
        **{
            "model.py": (
                "RATIO = 3\n"
                "\n"
                "\n"
                "def work(x):\n"
                "    return x * RATIO\n"
                "\n"
                "\n"
                "def stay():\n"
                "    return 1\n"
            ),
            "comp.py": "import os\n\n\ndef keep():\n    return 2\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def stay():\n    return 1\n",
            "comp.py": (
                "import os\n"
                "\n"
                "RATIO = 3\n"
                "\n"
                "\n"
                "def keep():\n"
                "    return 2\n"
                "\n"
                "\n"
                "def work(x):\n"
                "    return x * RATIO\n"
            ),
        },
    )
    commit = _commit(repo, "move work + RATIO to comp")

    recipe = infer_recipe(commit, str(repo))

    assert recipe.supported
    assert [am["name"] for am in recipe.assign_moves] == ["RATIO"]
    script = recipe_to_script(recipe, "move with constant")
    assert "move_assign" in script


def test_infer_recipe_in_file_method_reorder(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: class C with methods a, b, c in that order in one module.
    # Second commit reorders them to c, a, b. Asserts the recipe is supported with exactly one
    # move: method c with the same file as source and destination, enclosing class C, and
    # insertion anchor a (the next sibling).
    """A method cut and re-inserted elsewhere in the same class (no other file gains it) infers
    an in-file move_symbol (src == dst) anchored above its new next sibling."""
    _write(
        repo,
        **{
            "m.py": (
                "class C:\n"
                "    def a(self):\n"
                "        return 1\n"
                "\n"
                "    def b(self):\n"
                "        return 2\n"
                "\n"
                "    def c(self):\n"
                "        return 3\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "m.py": (
                "class C:\n"
                "    def c(self):\n"
                "        return 3\n"
                "\n"
                "    def a(self):\n"
                "        return 1\n"
                "\n"
                "    def b(self):\n"
                "        return 2\n"
            )
        },
    )
    commit = _commit(repo, "move c above a")
    recipe = infer_recipe(commit, str(repo))
    assert recipe.supported
    assert len(recipe.moves) == 1
    mv = recipe.moves[0]
    assert mv["name"] == "c" and mv["src"] == "m.py" and mv["dst"] == "m.py"
    assert mv["into_class"] == "C" and mv["before"] == "a"
