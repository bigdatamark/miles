# [MODULE DOC] Architecture guard: ray may be imported only by an explicit ledger of modules
# (RAY_USING_MODULES, repository-relative path to reason) that belong to the launcher closure, are
# tooling/examples, or are tracked debt. The test scans all shipped modules (everything except the
# tests directory, including top-level driver scripts) through the shared source-scan helpers and
# checks: no unlisted module imports ray, no ledger entry is stale or lacks a reason, the rpc layer
# and the serve actor never import ray, the scan covers the driver scripts, and the import detector
# recognises submodule, aliased and dynamic imports but not look-alike names. Pure static analysis;
# no ray is started.
from __future__ import annotations

import functools
from pathlib import Path

from tests.fast.source_scan import (
    FRAMEWORK_ROOT,
    REPO_ROOT,
    imported_modules,
    imported_modules_of_source,
    imports_package,
    relative_paths,
    shipped_modules,
)

EXCLUDED_DIRS = (REPO_ROOT / "tests",)

RAY_USING_MODULES = {
    "miles/utils/external_utils/ray_job.py": "launcher closure: submitting, following, and stopping launcher-owned ray jobs",
    "miles/ray/placement_group.py": "launcher closure: placement groups are how ray is asked to schedule",
    "miles/ray/wiring.py": "launcher closure: driver shutdown kills the manager it launched by ActorHandle",
    "miles/utils/ray_utils.py": "launcher closure: node lookup and pinning options for the launcher's own calls",
    "miles/utils/orchestration_utils.py": (
        "launcher closure: the shared driver composition root returns its ray worker manager"
    ),
    "miles/utils/workers/ray_worker_manager.py": "launcher closure: it is the launcher",
    "miles/utils/workers/ray_worker_handle.py": "launcher closure: the handle of the ray communication mode itself",
    "miles/utils/workers/worker_provider/ray.py": "launcher closure: it reads the launcher's own bookkeeping",
    "miles/utils/workers/backend_capability/ray.py": "launcher closure: it builds the ray provider and operations",
    "miles/utils/workers/cell_operations/ray.py": "launcher closure: suspend and resume are launcher verbs",
    "miles/utils/external_utils/command_utils/ray_backend/command.py": "launcher closure: running the ray launch scripts on every node the launcher has",
    "miles/utils/misc.py": "launcher closure: the node probe every launched actor answers with",
    "miles/utils/http_utils.py": "launcher closure: reaching a port on a node the launcher scheduled",
    "miles/utils/object_store.py": "object store exemption: the ray object store is a data plane of its own",
    "miles/utils/object_store_config.py": "object store exemption: config discovers the local node for its own data plane",
    "miles/dashboard/backend.py": "known debt: the dashboard collector is a named actor, tracked outside M23",
    "miles/dashboard/collector.py": "known debt: the dashboard collector is a named actor, tracked outside M23",
    "miles/dashboard/hooks.py": "known debt: the dashboard reads its gpu ids from ray, tracked outside M23",
    "miles/utils/tracking_utils/prometheus_utils.py": "known debt: the prometheus collector is a ray actor, skipped under kubernetes",
    "miles/ray/train_actor.py": "launcher closure: a launched actor reads the gpu ids ray gave it",
    "miles/backends/training_utils/weight_update/protocols/broadcast.py": "node ip lookup for a collective, not a call to another worker",
    "miles/backends/training_utils/weight_update/protocols/p2p_transfer_utils.py": "node ip lookup for a collective, not a call to another worker",
    "miles/utils/debug_utils/replay_reward_fn.py": "tooling: a standalone debugging script",
    "miles/utils/test_utils/mock_sglang_engine.py": "tooling: a test double that stands in for a ray-launched engine",
    "tools/convert_torch_dist_to_hf_ray.py": "tooling: a standalone conversion script that fans out over a ray cluster",
    "examples/experimental/formal_math/single_round/kimina_wrapper.py": "user example: a verifier pool of its own, outside the worker layer",
    "examples/experimental/formal_math/single_round/reward_fn.py": "user example: a verifier pool of its own, outside the worker layer",
}


@functools.cache
def _scanned_modules() -> tuple[Path, ...]:
    # [FUNCTION DOC] HELPER: Cached (computed once per process) tuple of all shipped python module
    # paths, excluding the tests directory.
    return tuple(shipped_modules(exclude_dirs=EXCLUDED_DIRS))


@functools.cache
def _imports_ray(path: Path) -> bool:
    # [FUNCTION DOC] HELPER: Cached per path; true when the module at the path imports the ray
    # package (any import form the scan helper recognises). Reads and parses the file.
    return imports_package(imported_modules(path), "ray")


@functools.cache
def _ray_using_module_paths() -> tuple[str, ...]:
    # [FUNCTION DOC] HELPER: Cached tuple of repository-relative path strings of every scanned
    # module that imports ray.
    return tuple(relative_paths(path for path in _scanned_modules() if _imports_ray(path)))


class TestRayIsOnlyUsedInsideTheLauncherClosure:
    # [CLASS DOC] Checks the ledger of ray-importing modules against the actual scan result.
    def test_no_module_reaches_for_ray_without_being_listed(self):
        # [FUNCTION DOC] Asserts the set of scanned modules importing ray minus the ledger keys is
        # empty; the failure message lists offenders and tells how to fix it.
        """Ray communication outside the launcher closure is what the rpc comm backend exists to remove."""
        unlisted = sorted(set(_ray_using_module_paths()) - set(RAY_USING_MODULES))

        assert unlisted == [], (
            f"{unlisted} import ray; a worker must be reachable over rpc too, so either drop the import or "
            f"add it to RAY_USING_MODULES with the reason it belongs to the launcher closure"
        )

    def test_the_list_names_no_module_that_stopped_using_ray(self):
        # [FUNCTION DOC] Asserts every ledger key is still among the ray-importing scanned modules
        # (no stale exemption).
        """A stale exemption reads as permission to bring ray back into a module that no longer needs it."""
        stale = sorted(set(RAY_USING_MODULES) - set(_ray_using_module_paths()))

        assert stale == []

    def test_every_listed_module_says_why(self):
        # [FUNCTION DOC] Asserts no ledger entry has an empty or whitespace-only reason.
        """The exemptions are a ledger of remaining debt, and a blank reason hides an entry from review."""
        assert sorted(name for name, reason in RAY_USING_MODULES.items() if not reason.strip()) == []

    def test_the_rpc_layer_itself_never_touches_ray(self):
        # [FUNCTION DOC] Selects scanned modules having a path component named rpc that import ray,
        # and asserts the list is empty.
        """The rpc client and server must run in a process that has no ray at all, such as a kubernetes pod."""
        rpc_modules = [path for path in _scanned_modules() if "rpc" in path.parts and _imports_ray(path)]

        assert rpc_modules == []

    def test_a_served_worker_is_started_without_ray(self):
        # [FUNCTION DOC] Asserts the serve actor module (utils/workers/serving/serve_actor.py under
        # the framework root) does not import ray, even though it is hosted as a ray actor, so the
        # same server code can start unchanged in a pod.
        """serve_actor happens to run inside a ray actor, but it starts the same server a pod starts."""
        assert not _imports_ray(FRAMEWORK_ROOT / "utils" / "workers" / "serving" / "serve_actor.py")


class TestTheScanReachesEveryProcessTheDriverIsPartOf:
    # [CLASS DOC] Sanity checks that the scan covers driver scripts and detects a plain import.
    def test_the_orchestration_scripts_are_scanned(self):
        # [FUNCTION DOC] Asserts the relative paths of scanned modules include the top-level train
        # and async-train driver scripts.
        """The driver is where a ray-only exception handler hid, and miles/ alone never covered it."""
        scanned = relative_paths(_scanned_modules())

        assert {"train.py", "train_async.py"} <= set(scanned)

    def test_a_driver_script_that_reaches_for_ray_would_be_reported(self):
        # [FUNCTION DOC] Feeds the source text of a plain ray import to the source-based import
        # collector and asserts the ray detector reports it.
        """A check that only ever looks under miles/ passes on the very file that broke under rpc."""
        assert imports_package(imported_modules_of_source("import ray\n"), "ray")


class TestTheShapesThatUsedToSlipThrough:
    # [CLASS DOC] Unit tests of the import detector on small source snippets.
    def test_a_submodule_import_counts(self):
        # [FUNCTION DOC] A submodule import of ray is detected as using ray.
        """`import ray.exceptions` reaches ray just as much as `import ray` does."""
        assert imports_package(imported_modules_of_source("import ray.exceptions\n"), "ray")

    def test_an_aliased_import_counts(self):
        # [FUNCTION DOC] An import of ray under an alias is detected.
        """Renaming the module on the way in does not rename what it talks to."""
        assert imports_package(imported_modules_of_source("import ray as r\n"), "ray")

    def test_a_dynamic_import_counts(self):
        # [FUNCTION DOC] A dynamic import of the module name through importlib's import-module
        # function is detected.
        """importlib is the shape an import lands in once someone wants it to not look like one."""
        assert imports_package(imported_modules_of_source('import importlib\nimportlib.import_module("ray")\n'), "ray")

    def test_a_module_that_merely_shares_the_prefix_does_not_count(self):
        # [FUNCTION DOC] An import of a package whose name merely starts with the letters of ray is
        # not detected (matching is by whole package name).
        """A false positive costs the ledger its meaning, and `raydium` is not ray."""
        assert not imports_package(imported_modules_of_source("import raydium\n"), "ray")
