# [MODULE DOC] HELPER: Shared helpers for the proof generator tests: thin git wrappers, a
# helper that writes or deletes files from keyword arguments, and two canned scenario builders
# that create a base commit followed by a refactoring commit in a test repository. Importing
# this module also puts the scripts directory on the import path.
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _git(repo: Path, *args: str) -> str:
    # [FUNCTION DOC] HELPER: Runs git with the given arguments inside the repository directory,
    # raising the called-process error on non-zero exit, and returns standard output as text
    # with surrounding whitespace stripped. Standard error is captured and discarded.
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _write(repo: Path, **files: str | None) -> None:
    # [FUNCTION DOC] HELPER: Writes or deletes files in the repository directory. Each keyword
    # argument name is a relative file path in which every double underscore stands for a
    # directory separator (since dots and slashes cannot appear in keyword names; the file
    # extension is therefore part of the name only when passed through an unpacked mapping).
    # A value of nothing deletes that file (which must exist, else a file-not-found error);
    # a string value creates missing parent directories and writes the text.
    for name, content in files.items():
        path = repo / name.replace("__", "/")
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


def _commit(repo: Path, message: str) -> str:
    # [FUNCTION DOC] HELPER: Stages all changes (including deletions) and commits them quietly
    # with the given message. Returns the full SHA of the new HEAD commit.
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _method_onto_class(repo: Path) -> None:
    # [FUNCTION DOC] HELPER: Scenario builder with two commits. Base commit: a model module
    # with class M holding a static method foo (taking a self-named first parameter and x,
    # returning x plus one) and a method other; a comp module with class C holding a method
    # keep; and a caller module with class K whose run method imports M locally and calls
    # M.foo on its c attribute and the number 9. Second commit ("move foo onto C"): foo is
    # removed from M, added to C as an ordinary instance method with the same body, and the
    # caller now calls foo as a method on its c attribute. Returns nothing.
    """Stage a base + a 'move foo from M onto C, lower the caller' commit."""
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n"
                "    @staticmethod\n"
                "    def foo(self, x):\n"
                "        return x + 1\n"
                "\n"
                "    def other(self):\n"
                "        return 0\n"
            ),
            "comp.py": "class C:\n    def keep(self):\n        return 1\n",
            "caller.py": (
                "class K:\n"
                "    def run(self):\n"
                "        from model import M\n"
                "\n"
                "        return M.foo(self.c, 9)\n"
            ),
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
                "        return x + 1\n"
            ),
            "caller.py": ("class K:\n    def run(self):\n        return self.c.foo(9)\n"),
        },
    )
    _commit(repo, "move foo onto C")


def _free_function_move_with_module_level_caller(repo: Path) -> None:
    # [FUNCTION DOC] HELPER: Scenario builder with two commits. Base commit: model module with
    # functions keep and resolve, a util module importing os, and a caller module that imports
    # resolve from model at module level and calls it in a run function. Second commit ("move
    # resolve to util"): resolve is removed from model and appended to util with the same body,
    # and the caller's module-level import now takes it from util. Returns nothing.
    """Stage a free function moved model.py -> util.py whose caller imports it at module
    level (so the repoint shows up in the symmetric module-level import diff)."""
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\ndef resolve(m):\n    return m\n",
            "util.py": "import os\n",
            "caller.py": ("from model import resolve\n\n\ndef run(m):\n    return resolve(m)\n"),
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n",
            "util.py": "import os\n\n\ndef resolve(m):\n    return m\n",
            "caller.py": ("from util import resolve\n\n\ndef run(m):\n    return resolve(m)\n"),
        },
    )
    _commit(repo, "move resolve to util")
