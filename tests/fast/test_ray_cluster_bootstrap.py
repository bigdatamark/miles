# [MODULE DOC] Static guard that only the shared session-scoped fixture (in the tests root
# conftest) starts the ray cluster: scans every python file under the tests directory (skipping
# the snapshots directory of rendered output) for calls to ray's init function, including aliased
# module imports and direct or renamed imports of init, and asserts the conftest is the only
# caller. Also tests the detector itself on synthetic sources (detected: plain, aliased, from-
# import, renamed from-import, submodule import; not flagged: shutdown, an unrelated local or
# other-library init). Constants: the tests root (parent of this directory) and snapshots root.
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOTS_ROOT = _TESTS_ROOT / "snapshots"


def test_only_the_session_fixture_starts_the_ray_cluster():
    # [FUNCTION DOC] Collects the relative paths of all test-tree files that call ray init and
    # asserts the sorted list is exactly the shared conftest, with a message telling to use the
    # session-scoped local-mode fixture instead.
    """A CPU suite runs many files in one pytest process, so whichever file calls ray.init
    first fixes the cluster's resources for every later file. A second initializer there
    silently produces a cluster without the logical GPUs the placement-group tests need."""
    offenders = sorted(str(path.relative_to(_TESTS_ROOT)) for path in _test_files_calling_ray_init())

    assert offenders == ["conftest.py"], (
        f"{offenders} call ray.init directly; depend on the session-scoped ray_local_mode "
        f"fixture in tests/conftest.py instead so the cluster has one owner"
    )


def _test_files_calling_ray_init() -> list[Path]:
    # [FUNCTION DOC] HELPER: Returns the python files under the tests root (regular files only,
    # excluding anything under the snapshots directory) in which the detector finds a ray init
    # call. Parses every candidate, so a file with a syntax error raises a syntax error.
    # Snapshot fixtures include directories named like modules, and the launch scripts
    # recorded under them are rendered output rather than code this suite runs.
    candidates = (path for path in _TESTS_ROOT.rglob("*.py") if path.is_file())
    return [path for path in candidates if _SNAPSHOTS_ROOT not in path.parents and _calls_ray_init(path)]


def _calls_ray_init(path: Path) -> bool:
    # [FUNCTION DOC] HELPER: Parses the file, computes the names bound to the ray module and to
    # ray's init, and returns true if any call is an attribute call named init on a name in the
    # module aliases, or a bare call to a directly imported init name; false otherwise.
    tree = ast.parse(path.read_text(), filename=str(path))
    module_aliases, direct_names = _ray_init_bindings(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "init":
            if isinstance(func.value, ast.Name) and func.value.id in module_aliases:
                return True
        if isinstance(func, ast.Name) and func.id in direct_names:
            return True
    return False


class TestCallsRayInit:
    # [CLASS DOC] Tests of the ray-init detector on small synthetic sources written to a
    # temporary file.
    @pytest.mark.parametrize(
        "source",
        [
            "import ray\nray.init()\n",
            "import ray as r\nr.init()\n",
            "from ray import init\ninit()\n",
            "from ray import init as ray_init\nray_init()\n",
            "import ray.util\nray.init()\n",
        ],
    )
    def test_every_way_of_reaching_ray_init_is_detected(self, source: str, tmp_path: Path) -> None:
        # [FUNCTION DOC] Writes each source to a candidate file in the temporary directory and
        # asserts the detector returns true.
        """A second cluster owner is just as real when it imports ray under another name."""
        path = tmp_path / "candidate.py"
        path.write_text(source)

        assert _calls_ray_init(path) is True

    @pytest.mark.parametrize(
        "source",
        [
            "import ray\nray.shutdown()\n",
            "from ray import shutdown\nshutdown()\n",
            "def init():\n    pass\n\ninit()\n",
            "import numpy as ray_like\nray_like.init()\n",
        ],
    )
    def test_unrelated_init_calls_are_not_flagged(self, source: str, tmp_path: Path) -> None:
        # [FUNCTION DOC] Writes each source to a candidate file in the temporary directory and
        # asserts the detector returns false.
        """Flagging a local init() or another library's would make the guard unusable."""
        path = tmp_path / "candidate.py"
        path.write_text(source)

        assert _calls_ray_init(path) is False


def _ray_init_bindings(tree: ast.Module) -> tuple[set[str], set[str]]:
    # [FUNCTION DOC] HELPER: Walks the syntax tree. The module-alias set starts with the plain
    # name ray and gains, for each plain import of ray or a ray submodule, the alias or the first
    # dotted component. The direct-name set gains, for each from-import of ray that imports init,
    # the alias or init. Returns (module aliases, direct names).
    """Names bound to the ray module and to ray.init itself, so aliased imports still count."""
    module_aliases = {"ray"}
    direct_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ray" or alias.name.startswith("ray."):
                    module_aliases.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module == "ray":
            for alias in node.names:
                if alias.name == "init":
                    direct_names.add(alias.asname or alias.name)
    return module_aliases, direct_names
