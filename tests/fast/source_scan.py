# [MODULE DOC] HELPER: Shared static-analysis helpers for the fast tests that scan the
# repository's Python sources without importing them: enumerating framework and shipped modules
# (skipping bytecode cache directories and excluded directories), converting to repo-relative
# path strings, parsing a module into a syntax tree, and collecting the set of imported module
# names (static imports, absolute only, plus string literals passed to dynamic import calls).
# Module constants: the repository root (two levels above this file), the framework package root
# (the miles directory), and the tuple of shipped package root names (miles, miles_plugins,
# scripts, examples, tools).
from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_ROOT = REPO_ROOT / "miles"

SHIPPED_PACKAGE_ROOTS = ("miles", "miles_plugins", "scripts", "examples", "tools")


def framework_modules(*, exclude_dirs: Iterable[Path] = ()) -> list[Path]:
    # [FUNCTION DOC] HELPER: Returns the sorted python files under the framework package root,
    # excluding any file under the given excluded directories.
    return python_modules(roots=[FRAMEWORK_ROOT], exclude_dirs=exclude_dirs)


def shipped_modules(*, exclude_dirs: Iterable[Path] = ()) -> list[Path]:
    # [FUNCTION DOC] HELPER: Returns a sorted list of all python files under every shipped package
    # root directory (those that exist), plus the python files directly in the repository root,
    # excluding any file under the given excluded directories.
    packaged = python_modules(roots=[REPO_ROOT / name for name in SHIPPED_PACKAGE_ROOTS], exclude_dirs=exclude_dirs)
    excluded = tuple(exclude_dirs)
    scripts = [
        path for path in REPO_ROOT.glob("*.py") if not any(path.is_relative_to(directory) for directory in excluded)
    ]
    return sorted([*packaged, *scripts])


def python_modules(*, roots: Iterable[Path], exclude_dirs: Iterable[Path] = ()) -> list[Path]:
    # [FUNCTION DOC] HELPER: Recursively finds python files under each existing root directory,
    # dropping files with a bytecode-cache path component or located under an excluded directory.
    # Returns them sorted.
    excluded = tuple(exclude_dirs)
    return sorted(
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and not any(path.is_relative_to(directory) for directory in excluded)
    )


def relative_paths(paths: Iterable[Path]) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the sorted list of the given paths as strings relative to the
    # repository root (raises a value error if a path is outside it).
    return sorted(str(path.relative_to(REPO_ROOT)) for path in paths)


def parse_module(path: Path) -> ast.Module:
    # [FUNCTION DOC] HELPER: Reads the file as text and parses it into a syntax tree, using the
    # path as the file name; raises a syntax error for invalid source.
    return ast.parse(path.read_text(), filename=str(path))


def imported_modules(path: Path) -> set[str]:
    # [FUNCTION DOC] HELPER: Returns the set of module names imported by the file at the path;
    # reads the file and delegates to the source-level function.
    return imported_modules_of_source(path.read_text(), filename=str(path))


def imported_modules_of_source(source: str, *, filename: str = "<source>") -> set[str]:
    # [FUNCTION DOC] HELPER: Parses the source text and walks every node of its syntax tree.
    # Collects the full module name of each plain import; for from-imports, the module name only
    # when it is present and the import is absolute (relative imports are ignored); and for each
    # call to a dynamic import function (by bare name or attribute), every positional or keyword
    # argument that is a string literal. Returns the set of names. Raises a syntax error on
    # invalid source.
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
        elif isinstance(node, ast.Call) and _is_dynamic_import(node.func):
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            imported.update(
                argument.value
                for argument in arguments
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
            )
    return imported


def imports_package(modules: Iterable[str], package: str) -> bool:
    # [FUNCTION DOC] HELPER: True when any module name equals the package or is a dotted
    # submodule of it (prefix match requires the dot separator).
    return any(module == package or module.startswith(f"{package}.") for module in modules)


_DYNAMIC_IMPORT_NAMES = ("import_module", "__import__")


def _is_dynamic_import(func: ast.expr) -> bool:
    # [FUNCTION DOC] HELPER: True if the callee expression is a bare name or an attribute whose
    # final name is one of the dynamic import function names (import_module or the built-in
    # import function).
    if isinstance(func, ast.Name):
        return func.id in _DYNAMIC_IMPORT_NAMES
    return isinstance(func, ast.Attribute) and func.attr in _DYNAMIC_IMPORT_NAMES
