# [MODULE DOC] Tests for inferring new-module extractions in the proof generator: a method still
# inside a class (unsupported), a staged trailing block cut into a new file (tail extract),
# scattered top-level functions gathered into a new file (scattered extract), and a module-level
# constant that moves into the new module along with the code. Each test commits a base and a
# second version in the shared repository fixture and checks the inferred recipe fields.
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


def test_infer_recipe_new_file_extract_from_class_method_unsupported(
    repo: Path,
) -> None:
    # [FUNCTION DOC] Base commit: class M with a static method foo and a method other. Second
    # commit removes foo from the class and creates a new module with a top-level foo. Asserts
    # the recipe is flagged unsupported and one note contains the phrase saying the symbols are
    # not all top-level in the source.
    """A method still inside the class cut straight into a new module cannot be cut as a
    top-level symbol, so the extract is reported unsupported (prep must lift it out first).
    """
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n"
                "    @staticmethod\n"
                "    def foo(self):\n"
                "        return 1\n"
                "\n"
                "    def other(self):\n"
                "        return 0\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "class M:\n    def other(self):\n        return 0\n",
            "newmod.py": "def foo():\n    return 1\n",
        },
    )
    _commit(repo, "extract foo to a new module")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported is False
    assert any("not all top-level" in note for note in recipe.notes)


def test_infer_recipe_new_file_extract_from_staged_tail(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: a module with class M, then a trailing block of a logging
    # import, a logger assignment and a function foo. Second commit leaves only the class in the
    # module and creates a new module holding the future annotations import and that block.
    # Asserts the recipe is supported, has no symbol moves, and its extracts list equals exactly
    # one record: source, destination, symbols list with foo, and future import flag true.
    """A staged trailing block (scaffolding + def at the source tail) cut into a new file
    infers an extract_to_new_module, prepending the future import."""
    _write(
        repo,
        **{
            "model.py": (
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
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "class M:\n    def keep(self):\n        return 1\n",
            "newmod.py": (
                "from __future__ import annotations\n"
                "\n"
                "import logging\n"
                "\n"
                "logger = logging.getLogger(__name__)\n"
                "\n"
                "\n"
                "def foo(x):\n"
                "    return x + 1\n"
            ),
        },
    )
    _commit(repo, "extract foo to a new module")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported
    assert recipe.moves == []
    assert recipe.extracts == [
        {
            "src": "model.py",
            "dst": "newmod.py",
            "symbols": ["foo"],
            "future_import": True,
        }
    ]


def test_infer_recipe_scattered_new_module_extract(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: a module importing os with functions keep, beta, stay, alpha
    # in that order. Second commit leaves keep and stay and creates a new module with the future
    # import, a logging import, a logger assignment, then alpha and beta. Asserts the recipe is
    # supported with no tail extracts and no moves, and exactly one scattered extract whose source
    # and destination are right, whose sorted symbols are alpha and beta, whose order is alpha
    # then beta, whose header starts with the future import line and contains the logger
    # assignment, and whose dropped assignments are empty; and that the rendered script mentions
    # the scattered-extract call.
    """Scattered top-level defs cut into a new module (no staged trailing block) infer a scatter
    extract with the authored header and target order, not UNSUPPORTED."""
    _write(
        repo,
        **{
            "common.py": (
                "import os\n"
                "\n"
                "\n"
                "def keep():\n"
                "    return 0\n"
                "\n"
                "\n"
                "def beta():\n"
                "    return 2\n"
                "\n"
                "\n"
                "def stay():\n"
                "    return 9\n"
                "\n"
                "\n"
                "def alpha():\n"
                "    return 1\n"
            ),
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "common.py": (
                "import os\n" "\n" "\n" "def keep():\n" "    return 0\n" "\n" "\n" "def stay():\n" "    return 9\n"
            ),
            "alloc.py": (
                "from __future__ import annotations\n"
                "\n"
                "import logging\n"
                "\n"
                "logger = logging.getLogger(__name__)\n"
                "\n"
                "\n"
                "def alpha():\n"
                "    return 1\n"
                "\n"
                "\n"
                "def beta():\n"
                "    return 2\n"
            ),
        },
    )
    _commit(repo, "extract alpha, beta to alloc.py")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported
    assert recipe.extracts == []
    assert recipe.moves == []
    assert len(recipe.scatter_extracts) == 1
    sx = recipe.scatter_extracts[0]
    assert sx["src"] == "common.py" and sx["dst"] == "alloc.py"
    assert sorted(sx["symbols"]) == ["alpha", "beta"]
    assert sx["order"] == ["alpha", "beta"]
    assert sx["header"].startswith("from __future__ import annotations\n")
    assert "logger = logging.getLogger(__name__)" in sx["header"]
    assert sx["drop_assigns"] == []
    script = recipe_to_script(recipe, "extract alpha, beta to alloc.py")
    assert "extract_symbols_to_new_module" in script


def test_infer_recipe_scatter_extract_drops_relocated_constant(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: a module importing is_hip, assigning a flag constant from a
    # call of it, assigning a logger-like constant, and defining moved (which reads the flag) and
    # keep (which reads the other constant). Second commit keeps only the other constant and
    # keep, and creates a new module with the import, the flag assignment and moved. Asserts one
    # scattered extract whose dropped assignments list contains exactly the flag name.
    """A module-level constant relocated into the new module is inferred as a drop_assign so the
    scatter extract removes it from the source too; a constant the source keeps is not.
    """
    _write(
        repo,
        **{
            "common.py": (
                "from u import is_hip\n"
                "\n"
                "_IS_HIP = is_hip()\n"
                "logger = 1\n"
                "\n"
                "\n"
                "def moved():\n"
                "    return _IS_HIP\n"
                "\n"
                "\n"
                "def keep():\n"
                "    return logger\n"
            ),
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "common.py": ("logger = 1\n\n\ndef keep():\n    return logger\n"),
            "alloc.py": (
                "from __future__ import annotations\n"
                "\n"
                "from u import is_hip\n"
                "\n"
                "_IS_HIP = is_hip()\n"
                "\n"
                "\n"
                "def moved():\n"
                "    return _IS_HIP\n"
            ),
        },
    )
    _commit(repo, "extract moved to alloc.py")
    recipe = infer_recipe("HEAD", str(repo))
    assert len(recipe.scatter_extracts) == 1
    assert recipe.scatter_extracts[0]["drop_assigns"] == ["_IS_HIP"]
