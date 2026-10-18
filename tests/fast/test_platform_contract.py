# [MODULE DOC] Static architecture tests (parse sources, no imports) that enforce the layering of
# the platform: only a listed set of files (the exemption table, each with a reason) may import
# upper-layer, backend-specific knowledge (kubernetes clients, ray spec/wiring modules, backend
# capability and worker-provider names); every exemption must still exist and still reach upwards;
# each orchestration script must call the shared driver initializer exactly once and the shared
# driver must call the backend capability launcher exactly once; the framework must never import
# the replaceable deployment package while that package may import the framework; and each launch
# script's train subcommand must not call preparation, download or conversion helpers. Module
# constants define the replaceable package name, the tooling directories excluded from scans, the
# upper-layer module and name lists, and the exemption table mapping repo-relative paths (files or
# directories) to reasons.
import ast
from pathlib import Path

import pytest
from tests.fast.source_scan import (
    FRAMEWORK_ROOT,
    REPO_ROOT,
    framework_modules,
    imported_modules,
    parse_module,
    python_modules,
)

REPLACEABLE_PACKAGE = "miles.utils.external_utils"

_TOOLING_DIRS = (
    FRAMEWORK_ROOT / "utils" / "external_utils",
    FRAMEWORK_ROOT / "utils" / "debug_utils",
    FRAMEWORK_ROOT / "utils" / "test_utils",
)

TRAIN_ONLY_SUBCOMMAND = "train"

ORCHESTRATION_SCRIPTS = ("train.py", "train_async.py")

BACKEND_CAPABILITY_FN = "launch_worker_manager"
ORCHESTRATION_INIT_FN = "init_orchestration_script"
BACKEND_NEUTRAL_WIRING_NAMES = {"shutdown_worker_manager"}

UPPER_LAYER_MODULES = (
    "kubernetes",
    "kubernetes_asyncio",
    "miles.ray.specs.bootstrap",
    "miles.ray.specs.entrypoint",
    "miles.ray.wiring",
)

UPPER_LAYER_NAMES = (
    "ClusterBackend",
    "KubernetesWorkerProvider",
    "KubernetesBackendCapability",
    "compute_helm_backend_capability",
    "RayBackendCapability",
    "RayWorkerManager",
    "compute_ctor_kwargs",
    "compute_specs",
    "launch_worker_manager",
    "get_backend_capability",
    "create_worker_backend_capability",
)

UPPER_LAYER_EXEMPTIONS = {
    "miles/ray/wiring.py": "the glue layer holding the driver process's single fork between the backends",
    "miles/utils/orchestration_utils.py": (
        "the shared driver composition root that launches the selected worker manager"
    ),
    "miles/utils/workers/worker_provider": "the infrastructure that owns every provider implementation",
    "miles/utils/workers/backend_capability": "the package that owns every capability implementation",
    "miles/utils/workers/cell_operations": "the package that owns every cell-operations implementation",
    "miles/ray/placement_group.py": "the driver composition the orchestration scripts delegate their wiring to",
    "miles/utils/ft_utils/mini_ft_controller.py": "kubernetes is the only backend that resumes a cell without being asked",
    "miles/utils/workers/serving/serve_inner.py": "the composition root of a served worker process",
    "miles/utils/workers/ray_worker_manager.py": "the composition root of a worker process an actor wraps",
    "miles/utils/workers/deployment_entrypoint.py": "the composition root of a deployment that carries no orchestration script",
    "miles/utils/workers/backend_capability/factory.py": "the fork itself: it is the switch every composition root asks",
    "miles/utils/workers/reconcile/k8s_api.py": "the kubernetes client the observing provider is written against",
    "miles/utils/arguments.py": "declares the --cluster-backend flag the composition roots read",
    "miles/utils/tracking_utils/base.py": "the prometheus collector is a ray actor and has no kubernetes form",
}


def _framework_modules() -> list[Path]:
    # [FUNCTION DOC] HELPER: Returns the framework's python files excluding the tooling
    # directories (external utils, debug utils, test utils).
    return framework_modules(exclude_dirs=_TOOLING_DIRS)


def _layered_modules() -> list[Path]:
    # [FUNCTION DOC] HELPER: Returns the modules subject to the layering rule: the framework
    # modules plus the orchestration scripts at the repository root, minus any exempted path.
    candidates = [*_framework_modules(), *(REPO_ROOT / name for name in ORCHESTRATION_SCRIPTS)]
    return [path for path in candidates if not _is_exempt(path)]


def _is_exempt(path: Path) -> bool:
    # [FUNCTION DOC] HELPER: True when the path, relative to the repository root in posix form,
    # equals an exemption entry or lies under an exemption directory.
    relative = path.relative_to(REPO_ROOT).as_posix()
    return any(relative == exempt or relative.startswith(f"{exempt}/") for exempt in UPPER_LAYER_EXEMPTIONS)


def _exempted_files(exemption: str) -> list[Path]:
    # [FUNCTION DOC] HELPER: Resolves an exemption entry against the repository root. A directory
    # expands to all python files beneath it; anything else (a file, even a missing one) is
    # returned as a single-element list.
    target = REPO_ROOT / exemption
    if not target.is_dir():
        return [target]
    return python_modules(roots=[target])


def _upper_layer_imports(path: Path) -> list[str]:
    # [FUNCTION DOC] HELPER: Parses the file and collects upper-layer imports. Plain imports
    # count when the module is an upper-layer module or inside one. For from-imports of the
    # wiring module, every imported name counts except the backend-neutral shutdown helper; for
    # other from-imports, the module counts if absolute and upper-layer, and each imported name
    # counts if it is in the upper-layer name list. Returns the list of offending names (with
    # repeats).
    tree = parse_module(path)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names if _is_upper_layer_module(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "miles.ray.wiring":
                found.extend(alias.name for alias in node.names if alias.name not in BACKEND_NEUTRAL_WIRING_NAMES)
            else:
                if node.module is not None and node.level == 0 and _is_upper_layer_module(node.module):
                    found.append(node.module)
                found.extend(alias.name for alias in node.names if alias.name in UPPER_LAYER_NAMES)
    return found


def _is_upper_layer_module(module: str) -> bool:
    # [FUNCTION DOC] HELPER: True if the dotted module name equals an upper-layer module or is a
    # submodule of one.
    return any(module == name or module.startswith(f"{name}.") for name in UPPER_LAYER_MODULES)


def _calls_of(path: Path, name: str) -> list[ast.Call]:
    # [FUNCTION DOC] HELPER: Parses the file and returns every call node whose callee is a bare
    # name equal to the given name (attribute calls are not counted).
    tree = parse_module(path)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


class TestLayering:
    # [CLASS DOC] Layering-rule tests over the framework and the orchestration scripts.
    def test_only_the_listed_files_learn_which_backend_the_workers_come_from(self):
        # [FUNCTION DOC] Collects, for every non-exempt layered module, each upper-layer import and
        # asserts there are none.
        """Every other module is handed the abstractions it needs, so knowing the backend would be a second answer."""
        offenders = [
            f"{path.relative_to(REPO_ROOT)} imports {name}"
            for path in _layered_modules()
            for name in _upper_layer_imports(path)
        ]

        assert offenders == [], offenders

    def test_the_check_sees_the_upper_layer_knowledge_the_glue_layer_holds(self):
        # [FUNCTION DOC] Sanity check of the scanner: the ray wiring module must report at least
        # one upper-layer import.
        """A check that finds nothing anywhere would pass on a codebase that reaches upwards from everywhere."""
        assert _upper_layer_imports(FRAMEWORK_ROOT / "ray" / "wiring.py") != []

    @pytest.mark.parametrize("exemption", sorted(UPPER_LAYER_EXEMPTIONS))
    def test_an_exemption_still_names_a_file_of_this_repo(self, exemption: str):
        # [FUNCTION DOC] Parametrized over the exemption keys: the path exists under the
        # repository root.
        """An exemption nobody removes outlives the file it was written for and quietly widens the rule."""
        assert (REPO_ROOT / exemption).exists()

    @pytest.mark.parametrize("exemption", sorted(UPPER_LAYER_EXEMPTIONS))
    def test_an_exemption_still_covers_a_file_that_reaches_upwards(self, exemption: str):
        # [FUNCTION DOC] Parametrized over the exemption keys: at least one file covered by the
        # exemption (the file, or any python file in the directory) has an upper-layer import;
        # otherwise fails with the exemption's reason.
        """Once the code it was written for stops reaching upwards, the exemption only shelters the next arrival."""
        reaching = [path for path in _exempted_files(exemption) if _upper_layer_imports(path)]

        assert reaching != [], f"{exemption} no longer reaches upwards: {UPPER_LAYER_EXEMPTIONS[exemption]}"

    @pytest.mark.parametrize("script", ORCHESTRATION_SCRIPTS)
    def test_an_orchestration_script_initializes_the_shared_driver_exactly_once(self, script: str):
        # [FUNCTION DOC] Parametrized over the train and async-train scripts: the script contains
        # exactly one bare call to the orchestration init function.
        """Every driver enters the shared composition root once, or it starts no manager or starts two."""
        assert len(_calls_of(REPO_ROOT / script, ORCHESTRATION_INIT_FN)) == 1

    def test_the_shared_driver_forks_the_backend_exactly_once(self):
        # [FUNCTION DOC] The shared orchestration utilities module contains exactly one bare call
        # to the worker-manager launch function.
        """The whole run hangs off one factory, and a second one would observe the same workers twice."""
        path = FRAMEWORK_ROOT / "utils" / "orchestration_utils.py"

        assert len(_calls_of(path, BACKEND_CAPABILITY_FN)) == 1


class TestImportDirection:
    # [CLASS DOC] Tests that dependencies point from the replaceable deployment code to the
    # framework only.
    def test_the_framework_never_imports_the_replaceable_deployment_code(self):
        # [FUNCTION DOC] For every framework module (tooling excluded), collects imports that equal
        # or lie under the external-utils package and asserts there are none.
        """A platform replacing the charts must be able to drop this half; an import would make it load anyway."""
        offenders = [
            f"{path.relative_to(REPO_ROOT)} imports {module}"
            for path in _framework_modules()
            for module in imported_modules(path)
            if module == REPLACEABLE_PACKAGE or module.startswith(f"{REPLACEABLE_PACKAGE}.")
        ]

        assert offenders == [], offenders

    def test_the_replaceable_code_may_import_the_framework(self):
        # [FUNCTION DOC] The helm backend launcher entrypoint under external utils imports at least
        # one module under the framework's workers package.
        """The dependency has to point one way, and this is the way it points."""
        launcher = (
            FRAMEWORK_ROOT
            / "utils"
            / "external_utils"
            / "command_utils"
            / "helm_backend"
            / "launcher"
            / "entrypoint.py"
        )

        assert any(module.startswith("miles.utils.workers") for module in imported_modules(launcher))


class TestLaunchScriptContract:
    # [CLASS DOC] Contract tests over the launch scripts in the scripts directory.
    @pytest.mark.parametrize("script", sorted((REPO_ROOT / "scripts").glob("run_*.py")), ids=lambda path: path.name)
    def test_a_train_subcommand_only_trains(self, script):
        # [FUNCTION DOC] Parametrized over launch scripts named run_ plus anything. Finds the
        # top-level function named train; skips if absent (script stays ray-only). Otherwise
        # collects the names of all functions it calls (bare or attribute) and asserts none
        # contains prepare, download or convert.
        """The Kubernetes backend runs this subcommand in a pod, where preparation has already happened."""
        tree = parse_module(script)
        train = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == TRAIN_ONLY_SUBCOMMAND),
            None,
        )
        if train is None:
            pytest.skip(f"{script.name} has no {TRAIN_ONLY_SUBCOMMAND} subcommand, so it stays Ray only")

        called = {
            node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            for node in ast.walk(train)
            if isinstance(node, ast.Call)
        }
        prepared = {name for name in called if "prepare" in name or "download" in name or "convert" in name}

        assert prepared == set(), f"{script.name} prepares data inside {TRAIN_ONLY_SUBCOMMAND}: {prepared}"
