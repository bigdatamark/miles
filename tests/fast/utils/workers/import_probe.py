# [MODULE DOC] HELPER:
# Utilities to report which third-party top-level modules a process has imported, and to flag
# heavy runtimes among them. Used to prove that the light serve entrypoint has not imported
# training or inference frameworks before it replaces itself with the real server. Standard
# library modules and installer shim modules are excluded from the report.
from __future__ import annotations

import sys

IMPORTED_MODULES_SEPARATOR = ","

FORBIDDEN_LIGHT_ENTRYPOINT_IMPORTS = frozenset(
    {"torch", "megatron", "sglang", "vllm", "transformers", "deepspeed", "ray", "uvicorn"}
)

_INSTALLER_SHIM_PREFIX = "__editable__"
_SYSCONFIGDATA_PREFIX = "_sysconfigdata"


def imported_top_level_modules() -> set[str]:
    # [FUNCTION DOC] HELPER: Returns the set of top-level package names (the part before the
    # first dot) of all currently loaded modules, excluding standard library modules and names
    # starting with the editable-install shim prefix or the platform configuration data prefix.
    # Reads the interpreter's global module table; no side effects.
    top_level_names = {name.partition(".")[0] for name in sys.modules}
    return {
        name
        for name in top_level_names
        if name not in sys.stdlib_module_names
        and not name.startswith(_INSTALLER_SHIM_PREFIX)
        and not name.startswith(_SYSCONFIGDATA_PREFIX)
    }


def report_imported_top_level_modules() -> str:
    # [FUNCTION DOC] HELPER: Returns the imported top-level module names, sorted, joined with a
    # comma, suitable for passing through an environment variable.
    return IMPORTED_MODULES_SEPARATOR.join(sorted(imported_top_level_modules()))


def unexpected_light_entrypoint_imports(reported: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Splits a comma-joined report (empty entries ignored) and returns,
    # sorted, the names that appear in the forbidden set of heavy runtimes (torch, megatron,
    # sglang, vllm, transformers, deepspeed, ray, uvicorn). An empty list means the entrypoint
    # stayed light.
    imported = {name for name in reported.split(IMPORTED_MODULES_SEPARATOR) if name}
    return sorted(imported & FORBIDDEN_LIGHT_ENTRYPOINT_IMPORTS)
