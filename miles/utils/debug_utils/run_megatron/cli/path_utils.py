# [MODULE DOC] Locates external resources for the run_megatron tool: the Megatron-LM checkout
# (explicit argument, then the MEGATRON_PATH environment variable, then /root/Megatron-LM) and the
# model definition script for a model type inside this repository's scripts/models directory.
"""Path resolution utilities for run_megatron CLI."""

import os
from pathlib import Path

import typer

_DEFAULT_MEGATRON_PATH: Path = Path("/root/Megatron-LM")


def resolve_megatron_path(megatron_path: Path | None) -> Path:
    # [FUNCTION DOC] Returns the explicit path if given; else the MEGATRON_PATH environment
    # variable as a path if set and non-empty; else the default /root/Megatron-LM. Does not check
    # that the path exists.
    if megatron_path is not None:
        return megatron_path
    env_path: str | None = os.environ.get("MEGATRON_PATH")
    if env_path:
        return Path(env_path)
    return _DEFAULT_MEGATRON_PATH


def resolve_model_script(model_type: str) -> Path:
    # [FUNCTION DOC] Returns the path of scripts/models/<model_type>.py under the repository root
    # (found by searching upward for the project file). Raises a typer bad-parameter error if the
    # script does not exist, or a runtime error if the repository root cannot be found.
    repo_base: Path = _resolve_repo_base()
    script: Path = repo_base / "scripts" / "models" / f"{model_type}.py"
    if not script.exists():
        raise typer.BadParameter(f"Model script not found: {script}")
    return script


def _resolve_repo_base() -> Path:
    # [FUNCTION DOC] HELPER: Walks upward from this file's resolved location and returns the first
    # directory containing pyproject.toml; raises a runtime error if none is found.
    current: Path = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Cannot find repo root (no pyproject.toml found above {current})")
