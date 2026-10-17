# [MODULE DOC] Tests for inferring module-level import changes in the proof generator recipe.
# Each test commits a base and a refactoring commit in the shared repository fixture (some
# using the canned scenario builders) and checks the import additions and imported-name
# removals the inferred recipe records for the head commit.
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
from mechanical_refactor_proof_generator import infer_recipe


def test_infer_recipe_infers_added_module_imports(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: model module importing gc, with a static method foo using gc;
    # comp module with class C. Second commit moves foo onto C and adds the gc import to comp.
    # Asserts the recipe's import additions contain a record for the comp file with the plain
    # gc import text.
    """An import the destination module gains (the moved code needs it) is inferred."""
    _write(
        repo,
        **{
            "model.py": (
                "import gc\n"
                "\n"
                "class M:\n"
                "    @staticmethod\n"
                "    def foo(self):\n"
                "        gc.collect()\n"
                "        return 1\n"
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
                "import gc\n"
                "\n"
                "class C:\n"
                "    def keep(self):\n"
                "        return 1\n"
                "\n"
                "    def foo(self):\n"
                "        gc.collect()\n"
                "        return 1\n"
            ),
        },
    )
    _commit(repo, "move foo onto C")
    recipe = infer_recipe("HEAD", str(repo))
    assert {"path": "comp.py", "text": "import gc"} in recipe.import_additions


def test_infer_recipe_module_level_import_repoint_realised_by_diff(repo: Path) -> None:
    # [FUNCTION DOC] Uses the canned scenario where a free function moves from model to util and
    # a caller imports it at module level. Asserts the recipe has no function-scoped repaths,
    # lists a removal of the name from the model module import in the caller (module model, name
    # resolve, no alias), and lists an addition with the statement importing it from util.
    """A module-level consumer whose import is repointed old -> new yields a remove of the old
    name and an add of the new -- not a reliance on the formatter pruning a duplicate.
    """
    _free_function_move_with_module_level_caller(repo)
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.repaths == []
    assert {
        "path": "caller.py",
        "module": "model",
        "name": "resolve",
        "asname": None,
    } in recipe.module_import_removals
    assert {"path": "caller.py", "text": "from util import resolve"} in (recipe.import_additions)


def test_infer_recipe_removes_an_import_the_source_no_longer_uses(repo: Path) -> None:
    # [FUNCTION DOC] Same scenario as the added-imports test: the moved method was the only user
    # of gc in the model module, so that module no longer imports it. Asserts the recipe's
    # imported-name removals contain a record for the model file with no module (plain import),
    # name gc and no alias.
    """When the moved body took the source's only use of an import, the source's lost name is
    realised as a removal (deterministic, not left to the formatter)."""
    _write(
        repo,
        **{
            "model.py": (
                "import gc\n"
                "\n"
                "class M:\n"
                "    @staticmethod\n"
                "    def foo(self):\n"
                "        gc.collect()\n"
                "        return 1\n"
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
                "import gc\n"
                "\n"
                "class C:\n"
                "    def keep(self):\n"
                "        return 1\n"
                "\n"
                "    def foo(self):\n"
                "        gc.collect()\n"
                "        return 1\n"
            ),
        },
    )
    _commit(repo, "move foo onto C")
    recipe = infer_recipe("HEAD", str(repo))
    assert {
        "path": "model.py",
        "module": None,
        "name": "gc",
        "asname": None,
    } in recipe.module_import_removals


def test_infer_recipe_adds_wholly_new_module_import_verbatim(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: model with keep and solve, util importing os, caller importing
    # solve from model. Second commit moves solve to util and rewrites the caller's import as a
    # parenthesized multi-line import with a trailing comma from util. Asserts the caller's
    # import additions include the verbatim multi-line statement text (without trailing newline)
    # and the caller's removals include dropping solve from the model import.
    """An import gained from a module not present in base is captured as the target's verbatim
    statement (so an exploded/magic-comma wrapping is reproduced, not collapsed per-name).
    """
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\ndef solve(x):\n    return x\n",
            "util.py": "import os\n",
            "caller.py": ("from model import solve\n\n\ndef run():\n    return solve(1)\n"),
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n",
            "util.py": "import os\n\n\ndef solve(x):\n    return x\n",
            "caller.py": ("from util import (\n    solve,\n)\n\n\ndef run():\n    return solve(1)\n"),
        },
    )
    _commit(repo, "move solve to util")
    recipe = infer_recipe("HEAD", str(repo))
    caller_adds = [a["text"] for a in recipe.import_additions if a["path"] == "caller.py"]
    assert "from util import (\n    solve,\n)" in caller_adds
    assert {
        "path": "caller.py",
        "module": "model",
        "name": "solve",
        "asname": None,
    } in recipe.module_import_removals
