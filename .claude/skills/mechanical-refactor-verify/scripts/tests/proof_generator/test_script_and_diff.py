# [MODULE DOC] Tests for the script rendering and diff parsing parts of the proof generator:
# structure and ordering of the emitted script text, running emitted scripts as real subprocesses
# (exit status and verdict text for a faithful commit, a commit with a bundled change, and a move
# placed above a type-checking guard), and the per-file diff parser's handling of content lines
# that start with plus signs. Uses the shared repository fixture.
import subprocess
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


def test_recipe_to_script_is_self_contained_and_ordered(repo: Path) -> None:
    # [FUNCTION DOC] Uses the canned scenario of a method moved onto a class. Renders the script
    # for the head commit and asserts it imports the builder class from the utilities module,
    # that the call-site lowering appears before the symbol move, that it ends with the run call
    # and the exit-with-status line, and that it does not mention an older utilities module name
    # or the proof generator module.
    """The emitted script imports only the reproduce util and lowers before moving."""
    _method_onto_class(repo)
    script = recipe_to_script(infer_recipe("HEAD", str(repo)), "move foo onto C")
    assert "from mechanical_refactor_reproduction_utils import Repro" in script
    assert script.index("lower_call_sites") < script.index("move_symbol")
    assert "residual = r.run()" in script
    assert "sys.exit(1 if residual else 0)" in script
    # importing nothing else from the skill keeps the script auditable in isolation
    assert "mechanical_refactor_verify_utils" not in script
    assert "mechanical_refactor_proof_generator" not in script


def test_recipe_to_script_orders_import_ops_after_moves(repo: Path) -> None:
    # [FUNCTION DOC] Uses the canned scenario of a free function moved with a module-level
    # caller. Asserts that in the rendered script the symbol move comes before both the
    # imported-name removal and the import addition.
    """The emitted script applies module-level import add/remove AFTER the move, matching
    build_repro's run order so the script and the in-process verdict cannot diverge."""
    _free_function_move_with_module_level_caller(repo)
    script = recipe_to_script(infer_recipe("HEAD", str(repo)), "move resolve to util")
    assert script.index("move_symbol") < script.index("remove_imported_name")
    assert script.index("move_symbol") < script.index("add_import")


def _emit_runnable_script(repo: Path, out: Path, commit: str, subject: str) -> Path:
    # [FUNCTION DOC] HELPER: Lays out a proof folder under the output directory: creates a
    # repro_scripts subfolder, copies the utilities module source (taken from the scripts
    # directory two levels above this test) into the output directory, infers the recipe for the
    # commit, renders the script with the given subject, and writes it as the first nine
    # characters of the commit id plus the python extension in the subfolder. Returns the script
    # path.
    """Write the emitted script plus its util dependency into a proof-folder layout."""
    scripts_dir = out / "repro_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    utils_src = Path(__file__).resolve().parents[2] / ("mechanical_refactor_reproduction_utils.py")
    (out / "mechanical_refactor_reproduction_utils.py").write_text(utils_src.read_text())
    script = recipe_to_script(infer_recipe(commit, str(repo)), subject)
    script_path = scripts_dir / f"{commit[:9]}.py"
    script_path.write_text(script)
    return script_path


def test_emitted_script_exits_zero_on_faithful_commit(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Base commit: model with keep and resolve, util importing os, caller
    # importing resolve from model. Second commit moves resolve to util with exactly the
    # primitives' output text (no formatter in the bare repository) and updates the caller
    # import. Emits the runnable script, runs it as a subprocess with the repository as working
    # directory, and asserts exit status 0 and the pass marker in standard output.
    """Running the emitted script on a clean move exits 0 and prints the PASS verdict."""
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\ndef resolve(m):\n    return m\n",
            "util.py": "import os\n",
            "caller.py": ("from model import resolve\n\n\ndef run(m):\n    return resolve(m)\n"),
        },
    )
    _commit(repo, "base")
    # The after-state is the primitives' exact output (this bare repo has no formatter
    # to absorb the cut's leftover blank lines, unlike a pre-commit-clean real repo).
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\n",
            "util.py": "import os\n\ndef resolve(m):\n    return m\n",
            "caller.py": ("from util import resolve\n\n\ndef run(m):\n    return resolve(m)\n"),
        },
    )
    commit = _commit(repo, "move resolve to util")
    script_path = _emit_runnable_script(repo, tmp_path / "out", commit, "move")

    result = subprocess.run([sys.executable, str(script_path)], cwd=repo, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_emitted_script_exits_nonzero_on_bundled_change(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Base commit: model with keep (returning 0) and resolve, util importing os.
    # Second commit moves resolve to util and also changes keep to return 99. Emits and runs the
    # script as a subprocess; asserts exit status 1 and the residual marker in standard output.
    """A commit bundling a non-move change makes the emitted script exit non-zero."""
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\ndef resolve(m):\n    return m\n",
            "util.py": "import os\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 99\n",
            "util.py": "import os\n\n\ndef resolve(m):\n    return m\n",
        },
    )
    commit = _commit(repo, "move resolve AND change keep")
    script_path = _emit_runnable_script(repo, tmp_path / "out", commit, "dirty move")

    result = subprocess.run([sys.executable, str(script_path)], cwd=repo, capture_output=True, text=True)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "RESIDUAL" in result.stdout


def test_emitted_script_passes_on_move_above_typechecking_guard(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Base commit: model with keep and helper; util with an import, a constant
    # assigned from a call, and a type-checking conditional importing a name. Second commit moves
    # helper into util between the constant and the conditional (exact primitive output). Emits
    # and runs the script; asserts exit status 0, the pass marker in output, and that the script
    # text anchors the move after the constant (using the after option with that constant name).
    """A module-level def relocated to just above an ``if TYPE_CHECKING:`` guard reproduces
    via an inferred after= anchor and the emitted script exits 0."""
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
    # After-state = the primitive's exact output (bare repo, no formatter to absorb blanks).
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\n",
            "util.py": (
                "from u import is_hip\n"
                "\n"
                "_is_hip = is_hip()\n"
                "\n"
                "def helper(x):\n"
                "    return x + 1\n"
                "\n"
                "if TYPE_CHECKING:\n"
                "    from m import Thing\n"
            ),
        },
    )
    commit = _commit(repo, "move helper above the TYPE_CHECKING guard")
    script_path = _emit_runnable_script(repo, tmp_path / "out", commit, "after-anchor move")

    result = subprocess.run([sys.executable, str(script_path)], cwd=repo, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
    assert "after='_is_hip'" in script_path.read_text()


def test_per_file_diff_keeps_content_lines_starting_with_plus_signs(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: a notes module with one assignment. Second commit adds a line
    # assigning a string whose value begins with two plus signs. Parses the diff of the head
    # commit with the per-file diff helper and asserts the added lines list for the notes file
    # is exactly that new line, i.e. a content line is not mistaken for a diff header.
    """An added content line beginning with '++' is collected, not mistaken for a header."""
    from mechanical_refactor_proof_generator import _per_file_diff

    _write(repo, **{"notes.py": "a = 1\n"})
    _commit(repo, "base")
    _write(repo, **{"notes.py": 'a = 1\nb = "++x"\n'})
    commit = _commit(repo, "add plus-plus line")

    files = _per_file_diff(commit, str(repo))

    assert files["notes.py"]["added"] == ['b = "++x"']
