# [MODULE DOC]
# Subprocess based import tests. Each test starts a fresh interpreter (so the module table is
# empty), with the repository root prepended to the python path, and imports modules there. They
# guard against circular imports between the inference specs and the router manager, and against
# the inference specs eagerly loading gpu-only dumper dependencies.
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[3]

CYCLE_PRONE_MODULES: tuple[str, str] = (
    "miles.ray.specs.inference",
    "miles.ray.rollout.router_manager",
)


@pytest.mark.parametrize(
    "module_names",
    [
        CYCLE_PRONE_MODULES,
        tuple(reversed(CYCLE_PRONE_MODULES)),
    ],
    ids=["specs_first", "router_manager_first"],
)
def test_cycle_prone_modules_import_in_either_order(module_names: tuple[str, str]) -> None:
    # [FUNCTION DOC] Parametrized with the two cycle-prone modules in both orders. Builds a small
    # program importing the modules one after another, runs it in a fresh interpreter with a
    # five minute timeout, the repository root as working directory and python path, captures
    # output, and asserts a zero exit code; on failure the message includes the standard error text.
    # Side effect: spawns a subprocess; raises a timeout error if the import hangs.
    """A fresh interpreter can import the spec and router-manager modules in either order."""
    program: str = "\n".join(f"import {module_name}" for module_name in module_names)
    env: dict[str, str] = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")]))

    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"importing {module_names} failed:\n{result.stderr}"


def test_inference_specs_import_without_loading_gpu_dumper_dependencies() -> None:
    # [FUNCTION DOC] In a fresh interpreter, imports the inference specs module and then asserts
    # that neither the miles dumper utilities module nor the sglang debug dumper module is in the
    # module table. The subprocess must exit with zero (five minute timeout, repository root as
    # working directory and python path); otherwise the test fails with the standard error text.
    # Side effect: spawns a subprocess.
    """A CPU-only controller can import inference specs without loading GPU dumper dependencies."""
    program = "\n".join(
        [
            "import sys",
            "import miles.ray.specs.inference",
            "assert 'miles.utils.dumper_utils' not in sys.modules",
            "assert 'sglang.srt.debug_utils.dumper' not in sys.modules",
        ]
    )
    env: dict[str, str] = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")]))

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"importing inference specs loaded GPU dumper dependencies:\n{result.stderr}"
