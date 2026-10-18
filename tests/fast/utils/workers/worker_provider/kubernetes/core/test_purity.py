# [MODULE DOC] Static purity check of the kubernetes core view modules (cell view and pod view):
# starting from each view module, follows first-party imports transitively (resolving module names
# under the repository root; only names starting with the project prefix whose source file exists)
# and asserts that nothing reachable is a forbidden module (the kubernetes client library, the
# provider module, the reflector and the reconcile loop). Imports guarded by a type-checking-only
# condition are ignored. Pure static analysis of source files; nothing is imported or run.
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from miles.utils.workers.worker_provider.kubernetes import core

FORBIDDEN_IMPORTS = (
    "kubernetes_asyncio",
    "miles.utils.workers.worker_provider.kubernetes.core.provider",
    "miles.utils.workers.reconcile.k8s_reflector",
    "miles.utils.workers.reconcile.loop",
)
VIEW_MODULE_NAMES = ("cell_view.py", "pod_view.py")

_REPO_ROOT = Path(__file__).resolve().parents[7]


def view_module_paths() -> list[Path]:
    # [FUNCTION DOC] HELPER: Returns the paths of the cell view and pod view source files, located
    # next to the core package file.
    return [Path(core.__file__).parent / name for name in VIEW_MODULE_NAMES]


def imported_module_names(source_path: Path) -> list[str]:
    # [FUNCTION DOC] HELPER: Parses the source file and returns the names of all modules it
    # imports: every module of a plain import and the module of each absolute from-import
    # (relative imports are skipped). The bodies of conditional blocks guarded by the type-checking
    # flag name are emptied first, so those imports are not counted. Reads the file; may raise on
    # syntax errors.
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _guards_type_checking(node):
            node.body = []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            names.append(node.module)
    return names


def reachable_module_names(source_path: Path) -> list[str]:
    # [FUNCTION DOC] HELPER: Depth-first walk over imports starting at the file. Every imported
    # name is appended to the result (duplicates kept). A name is followed further only if it
    # starts with "miles.", maps to an existing source file under the repository root, and was not
    # visited. Returns all imported names in visiting order.
    # KNOWN ISSUE: package imports resolve only to a module file, never to a package init file, so
    # imports made inside package init files are not followed.
    reached: list[str] = []
    seen: set[Path] = {source_path}
    pending = [source_path]
    while pending:
        for name in imported_module_names(pending.pop()):
            reached.append(name)
            candidate = _REPO_ROOT / f"{name.replace('.', '/')}.py"
            if name.startswith("miles.") and candidate.exists() and candidate not in seen:
                seen.add(candidate)
                pending.append(candidate)
    return reached


def _guards_type_checking(node: ast.If) -> bool:
    # [FUNCTION DOC] HELPER: True when the conditional's test is the bare name TYPE_CHECKING
    # (attribute forms such as the typing module attribute are not recognised).
    return isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"


class TestViewsPurity:
    # [CLASS DOC] Purity assertions over the view modules.
    @pytest.mark.parametrize("source_path", view_module_paths(), ids=lambda path: path.name)
    def test_a_view_module_reaches_nothing_that_talks_to_a_cluster(self, source_path: Path) -> None:
        # [FUNCTION DOC] Parametrized over the two view modules; collects reachable names equal to
        # or below a forbidden module and asserts none, with a message listing offenders.
        """Views must stay pure projections; a scan of direct imports alone would miss a helper that reaches on."""
        offenders = [
            name
            for name in reachable_module_names(source_path)
            for forbidden in FORBIDDEN_IMPORTS
            if name == forbidden or name.startswith(f"{forbidden}.")
        ]

        assert not offenders, f"{source_path.name} reaches {sorted(set(offenders))}, which is past a pure projection"

    def test_the_scan_actually_follows_first_party_imports(self) -> None:
        # [FUNCTION DOC] The names reachable from the cell view include a provider utils module and
        # the rpc client handle module (a transitive hop), proving the walk goes beyond direct
        # imports.
        """A scan that stopped at the first hop would pass this suite while proving almost nothing."""
        reached = reachable_module_names(Path(core.__file__).parent / "cell_view.py")

        assert "miles.utils.workers.worker_provider.utils" in reached
        assert "miles.utils.workers.rpc.client.handle" in reached

    def test_the_scan_sees_every_view_module(self) -> None:
        # [FUNCTION DOC] Asserts both view module files exist.
        """A renamed view module would make the purity check silently scan nothing."""
        assert all(path.exists() for path in view_module_paths())
