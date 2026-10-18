# [MODULE DOC] Static layering checks for the tinker gateway package (miles/tinker), by parsing
# source files without importing them: the core sub-package may import only standard library
# modules and itself; the server sub-package must not import ray, training-backend or general
# utility packages of the framework; and, across the whole package, only the runtime module may
# import ray or backend modules. Constants: the package directory (three levels above this file's
# directory, under the miles directory) and the set of standard library module names.
"""Machine enforcement of the gateway layering law: core speaks only the
internal language, server never touches miles infrastructure, and runtime.py
is the single miles translator inside the package."""

import ast
import sys
from pathlib import Path

TINKER_DIR = Path(__file__).resolve().parents[3] / "miles" / "tinker"
STDLIB = set(sys.stdlib_module_names)


def _import_roots(path: Path) -> set[str]:
    # [FUNCTION DOC] HELPER: Parses the file and returns the set of imported module names: the
    # full dotted name of each plain import and the module name of each absolute from-import.
    # Despite the function name, names are not reduced to their root package.
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add(node.module)
    return roots


def test_core_imports_only_stdlib_and_itself():
    # [FUNCTION DOC] For every python file in the core sub-package, each imported module must have
    # a standard-library top-level name or start with the core package name; otherwise an
    # assertion error names the file and module.
    for path in sorted((TINKER_DIR / "core").glob("*.py")):
        for module in _import_roots(path):
            allowed = module.split(".")[0] in STDLIB or module.startswith("miles.tinker.core")
            assert allowed, f"{path.name} imports {module}; core must stay stdlib-only"


def test_server_never_imports_miles_infrastructure():
    # [FUNCTION DOC] For every python file in the server sub-package, no imported module may start
    # with the ray, backends or utils package names of the framework.
    for path in sorted((TINKER_DIR / "server").glob("*.py")):
        for module in _import_roots(path):
            assert not module.startswith(
                ("miles.ray", "miles.backends", "miles.utils")
            ), f"{path.name} imports {module}; server may only speak SDK wire and core"


def test_runtime_is_the_only_miles_speaker():
    # [FUNCTION DOC] For every python file anywhere in the package except any named runtime.py, no
    # imported module may start with the ray or backends package names of the framework.
    for path in sorted(TINKER_DIR.rglob("*.py")):
        if path.name == "runtime.py":
            continue
        for module in _import_roots(path):
            assert not module.startswith(
                ("miles.ray", "miles.backends")
            ), f"{path.relative_to(TINKER_DIR)} imports {module}; only runtime.py translates miles"
