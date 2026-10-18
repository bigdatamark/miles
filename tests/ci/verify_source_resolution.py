# [MODULE DOC]
# CI helper script (run directly) that verifies the Python modules used by a CI job are imported
# from the expected source checkouts rather than from stale installed copies. For each module in
# a fixed table it reads the environment variable naming the expected source root (the
# workspace for the miles packages, the sglang and megatron source roots for those), resolves
# the module's location without importing it, and requires the resolved origin file to lie
# inside that root. Prints one line per module with its origin. Raises a runtime error if a
# module cannot be resolved or resolves outside its root; a missing environment variable raises
# a key error. Tests are not defined here.
import os
from importlib.util import find_spec
from pathlib import Path


MODULE_ROOT_ENV = {
    "miles": "GITHUB_WORKSPACE",
    "miles_plugins": "GITHUB_WORKSPACE",
    "sglang": "SGLANG_SOURCE_ROOT",
    "megatron.core": "MEGATRON_SOURCE_ROOT",
    "megatron.training": "MEGATRON_SOURCE_ROOT",
}


def main() -> None:
    # [FUNCTION DOC] For each module name and its root environment variable in the table, in order:
    # resolve the absolute expected root from the environment (missing variable raises a key
    # error), look up the module spec; raise a runtime error "cannot resolve <module>" if there is
    # no spec or no origin file; require the resolved origin to be inside the expected root,
    # otherwise raise a runtime error naming the module, the origin and the expected root
    # (chained from the underlying error); on success print the module name and origin.
    # Stops at the first failure. Returns nothing.
    for module_name, root_env in MODULE_ROOT_ENV.items():
        expected_root = Path(os.environ[root_env]).resolve()
        spec = find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot resolve {module_name}")
        origin = Path(spec.origin).resolve()
        try:
            origin.relative_to(expected_root)
        except ValueError as exc:
            raise RuntimeError(f"{module_name} resolved to {origin}, expected {expected_root}") from exc
        print(f"{module_name}: {origin}")


if __name__ == "__main__":
    main()
