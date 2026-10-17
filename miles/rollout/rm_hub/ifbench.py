# [MODULE DOC]
# Instruction-following reward using the external IFBench benchmark (AllenAI). IMPORTANT import
# side effects: importing this module clones the IFBench git repository into the parent of the
# workspace root (network and subprocess), prepends that directory to the import path and to the
# PYTHONPATH environment variable, imports the benchmark's evaluation library (installing the
# local requirements file with pip once, marked by a sentinel file in the clone, if the first
# import fails), and binds module-level names for the library and its input example type.
# Module constants: workspace root (three directory levels above this file), its parent, the
# local requirements file path under the examples directory, and type aliases for json-like
# mappings and per-instruction keyword mappings.
# The public function scores a response 1.0 if it follows all instructions in strict mode else 0.0.
from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_PARENT = _WORKSPACE_ROOT.parent
_LOCAL_IFBENCH_REQUIREMENTS = _WORKSPACE_ROOT / "examples" / "eval_multi_task" / "requirements_ifbench.txt"


def _ensure_ifbench_repo() -> Path:
    # [FUNCTION DOC] HELPER: Make sure the IFBench clone exists next to the workspace root and is
    # importable. If the directory is missing, run a git clone of the upstream repository
    # (captured output); any failure is re-raised as an import error asking for a manual clone.
    # Then prepend the path to the import path if absent and to the PYTHONPATH environment variable
    # (set when unset, prepended when not already a listed entry). Returns the clone path.
    """Clone IFBench repo if needed and ensure it is available on sys.path."""

    repo_path = _WORKSPACE_PARENT / "IFBench"

    if not repo_path.exists():
        clone_cmd = ["git", "clone", "https://github.com/allenai/IFBench.git", str(repo_path)]
        try:
            subprocess.run(clone_cmd, check=True, capture_output=True)
        except Exception as exc:
            raise ImportError(
                "Unable to automatically clone IFBench. Please clone "
                "https://github.com/allenai/IFBench.git into the repo root."
            ) from exc

    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    current_pythonpath = os.environ.get("PYTHONPATH")
    if current_pythonpath is None:
        os.environ["PYTHONPATH"] = repo_str
    elif repo_str not in current_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_str, current_pythonpath])

    return repo_path


def _ensure_ifbench_dependencies(repo_path: Path) -> None:
    # [FUNCTION DOC] HELPER: Install the local IFBench requirements with pip using the current
    # interpreter (subprocess, network). Skips silently (debug log) if the requirements file does
    # not exist, and does nothing if the sentinel file in the clone already exists. Install
    # failure is only logged as a warning; on success the sentinel file is written so the install
    # is not repeated. Returns nothing.
    """Install IFBench requirements the first time the module is imported."""

    requirements_file = _LOCAL_IFBENCH_REQUIREMENTS

    if not requirements_file.exists():
        logger.debug("Local IFBench requirements file not found at %s; skipping install.", requirements_file)
        return

    sentinel = repo_path / ".deps_installed"
    if sentinel.exists():
        return

    install_cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
    try:
        subprocess.run(install_cmd, check=True)
    except Exception as exc:
        logger.warning("Failed to install IFBench dependencies automatically: %s", exc)
    else:
        sentinel.write_text("installed\n")


def _load_evaluation_lib():
    # [FUNCTION DOC] HELPER: Ensure the repo is cloned and on the path, import the benchmark's
    # evaluation library; on an import error install dependencies once and import again (a second
    # failure propagates). Returns the module.
    repo_path = _ensure_ifbench_repo()
    try:
        return importlib.import_module("evaluation_lib")
    except ImportError:
        _ensure_ifbench_dependencies(repo_path)
        return importlib.import_module("evaluation_lib")


evaluation_lib = _load_evaluation_lib()
InputExample = evaluation_lib.InputExample


JsonDict = dict[str, Any]
KwargsDict = dict[str, str | int | float | None]


def _normalize_instruction_ids(raw_ids: Sequence[Any]) -> list[str]:
    # [FUNCTION DOC] HELPER: Convert raw ids to trimmed strings, dropping null entries and entries
    # empty after trimming; null or empty input gives an empty list. Order is preserved.
    """Ensure instruction identifiers are clean strings."""

    normalized: list[str] = []
    for entry in raw_ids or []:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def _coerce_kwargs_list(
    raw_kwargs: Any,
    num_instructions: int,
) -> list[KwargsDict]:
    # [FUNCTION DOC] HELPER: Produce exactly one keyword mapping per instruction. A list input
    # keeps mapping entries (copied) and replaces non-mapping entries with empty mappings; a single
    # mapping is replicated for every instruction; anything else gives empty mappings. A too-short
    # list is padded with copies of its last entry (or empty mappings if it was empty); a too-long
    # list is truncated. Finally entries with null values are removed to match the benchmark's
    # official preprocessing. Inputs are not mutated.
    """Convert stored kwargs into the list structure expected by IFBench."""

    if isinstance(raw_kwargs, list):
        processed: list[KwargsDict] = []
        for entry in raw_kwargs:
            if isinstance(entry, dict):
                processed.append(dict(entry))
            else:
                processed.append({})
    elif isinstance(raw_kwargs, dict):
        processed = [dict(raw_kwargs) for _ in range(num_instructions)]
    else:
        processed = [{} for _ in range(num_instructions)]

    if len(processed) < num_instructions:
        tail = processed[-1] if processed else {}
        processed.extend([dict(tail) for _ in range(num_instructions - len(processed))])
    elif len(processed) > num_instructions:
        processed = processed[:num_instructions]

    # Remove explicit None values to match official preprocessing.
    sanitized: list[KwargsDict] = []
    for entry in processed:
        sanitized.append({k: v for k, v in entry.items() if v is not None})
    return sanitized


def _build_input_example(metadata: JsonDict) -> InputExample | None:
    # [FUNCTION DOC] HELPER: Build the benchmark's input example from sample metadata, or return
    # none (debug log) if there are no usable instruction ids. Uses the metadata entries:
    # instruction id list, prompt text (null becomes empty, otherwise converted to text), keyword
    # data (coerced to one mapping per instruction) and record id (integer; missing or falsy
    # becomes zero; a non-numeric value raises a value error).
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list") or [])
    if not instruction_ids:
        logger.debug("Missing instruction identifiers in metadata: %s", metadata)
        return None

    prompt_text = metadata.get("prompt_text")
    if prompt_text is None:
        prompt_text = ""
    else:
        prompt_text = str(prompt_text)

    raw_kwargs = metadata.get("kwargs")
    kwargs_list = _coerce_kwargs_list(raw_kwargs, len(instruction_ids))

    return InputExample(
        key=int(metadata.get("record_id") or 0),
        instruction_id_list=instruction_ids,
        prompt=prompt_text,
        kwargs=kwargs_list,
    )


def compute_ifbench_reward(response: str, label: Any, metadata: JsonDict | None = None) -> float:
    # [FUNCTION DOC]
    # Return 1.0 if the response follows all instructions under the benchmark's strict test, else
    # 0.0. Returns 0.0 without scoring when metadata is null, the response is null, or no input
    # example could be built. The label argument is unused. The benchmark is called with a mapping
    # from the example's prompt to the response text.
    """Score a model response using the official IFBench rules."""

    if metadata is None:
        logger.debug("No metadata provided for IFBench scoring.")
        return 0.0

    if response is None:
        return 0.0

    inp = _build_input_example(metadata)
    if inp is None:
        return 0.0

    prompt_to_response = {inp.prompt: str(response or "")}
    output = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    return 1.0 if output.follow_all_instructions else 0.0
