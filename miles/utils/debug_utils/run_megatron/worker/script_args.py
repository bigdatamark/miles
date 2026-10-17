# [MODULE DOC] Defines the immutable set of custom options passed from the command-line layer to
# the torchrun worker, and the bridge object that converts it to command-line flags (with a
# "script" prefix) and back from a parsed argument namespace.
"""WorkerScriptArgs: our custom arguments passed from CLI to torchrun worker.

All flags use the ``--script-`` prefix to avoid collision with Megatron's own
argparse namespace.  Adding a new field here is the *only* change needed —
serialization (CLI → worker) and deserialization (worker argparse) are automatic.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from miles.utils.argparse_utils import DataclassArgparseBridge


@dataclasses.dataclass(frozen=True)
class WorkerScriptArgs:
    # [CLASS DOC] Immutable worker options. Required: hf_checkpoint (HuggingFace checkpoint, also
    # used for the tokenizer) and token_ids_file (json file of prompt token ids). Optional:
    # role (default actor), ref_load (Megatron checkpoint to load), run_backward (default false),
    # source_patcher_config (YAML of source patches), routing replay dump and load paths, top_k
    # (default 0, disabled) and logprob_output (directory for logprob files).
    hf_checkpoint: Path
    token_ids_file: Path
    role: str = "actor"
    ref_load: Path | None = None
    run_backward: bool = False
    source_patcher_config: Path | None = None
    routing_replay_dump_path: Path | None = None
    routing_replay_load_path: Path | None = None
    top_k: int = 0
    logprob_output: Path | None = None


WORKER_SCRIPT_ARGS_BRIDGE: DataclassArgparseBridge[WorkerScriptArgs] = DataclassArgparseBridge(
    WorkerScriptArgs,
    prefix="script",
    group_title="run_megatron script args",
)
