# [MODULE DOC]
# HELPER: Small LoRA helpers: the fixed adapter name, target pattern matching, feature gating
# predicates computed from parsed args, adapter config and file writing, target module name
# extraction from weight names, and detection of expert targets.
import json
from argparse import Namespace
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

LORA_ADAPTER_NAME = "miles_lora"


def matches_lora_target(module: str, target: str) -> bool:
    # [FUNCTION DOC] HELPER: Case-sensitive wildcard match. A target containing a dot is matched
    # against the full module name; a dotless target is matched against only the last dotted
    # component of the module name.
    return fnmatchcase(module if "." in target else module.rsplit(".", 1)[-1], target)


def is_lora_weight_name(name: str) -> bool:
    # [FUNCTION DOC] HELPER: True if the name contains the lora A or lora B marker segment.
    """Check if an HF weight name corresponds to a LoRA adapter weight."""
    return ".lora_A." in name or ".lora_B." in name


def is_lora_enabled(args: Namespace) -> bool:
    # [FUNCTION DOC] HELPER: True if the rank attribute is positive (missing counts as zero) or an
    # adapter path is set.
    """Check if LoRA is enabled based on arguments."""
    return getattr(args, "lora_rank", 0) > 0 or getattr(args, "lora_adapter_path", None) is not None


def lora_rollout_enabled(args: Namespace) -> bool:
    # [FUNCTION DOC] HELPER: LoRA enabled and the train-only flag not set (missing counts as unset).
    """LoRA enabled AND the rollout side participates; false under --lora-train-only.

    Gates everything rollout-facing: SGLang's ``enable_lora``, the per-request
    ``lora_path``, and the adapter weight sync. Training-side LoRA is unaffected.
    """
    return is_lora_enabled(args) and not getattr(args, "lora_train_only", False)


def engine_loads_adapter_from_disk(args: Namespace) -> bool:
    # [FUNCTION DOC] HELPER: True if an adapter path is set and either rollout-only debugging or
    # skipping weight updates is on (so no trainer will push the adapter). Attributes must exist.
    """Only when no trainer will push the adapter; otherwise the first weight sync carries it."""
    return args.lora_adapter_path is not None and (args.debug_rollout_only or args.debug_skip_weight_update)


def lora_base_cpu_backup_enabled(args: Namespace) -> bool:
    # [FUNCTION DOC] HELPER: True when LoRA is enabled, colocate is on and the base cpu backup flag
    # is on (missing flags count as off).
    """LoRA + --colocate + --lora-base-cpu-backup all set."""
    return is_lora_enabled(args) and getattr(args, "colocate", False) and getattr(args, "lora_base_cpu_backup", False)


def save_adapter_to_disk(out_dir, config: dict, tensors: dict) -> None:
    # [FUNCTION DOC] HELPER: Creates the directory (with parents), writes the config as indented json
    # to the adapter config file and the tensors mapping to a safetensors file. Overwrites existing
    # files. Imports safetensors lazily.
    """Write a LoRA adapter dir (adapter_config.json + adapter_model.safetensors)."""
    import safetensors.torch  # lazy: this module is imported on paths that never touch weights

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    safetensors.torch.save_file(tensors, str(out / "adapter_model.safetensors"))


def build_lora_config(args, *, target_modules):
    # [FUNCTION DOC] HELPER: Adapter config mapping in peft format: type LORA, rank, alpha, target
    # modules (a string kept as is, otherwise a list copy), dropout, no bias, causal lm task.
    return {
        "peft_type": "LORA",
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules if isinstance(target_modules, str) else list(target_modules),
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def get_adapter_target_modules(weight_names):
    # [FUNCTION DOC] HELPER: Sorted unique module names: each weight name with the peft base model
    # prefix removed and cut at the last lora marker segment.
    return sorted({name.removeprefix("base_model.model.").rsplit(".lora_", 1)[0] for name in weight_names})


@dataclass(frozen=True)
class AdapterSpec:
    # [CLASS DOC] HELPER: Immutable adapter export info: slot index, rank and alpha scaling.
    """The slot and scaling needed to export one adapter."""

    slot: int
    rank: int
    alpha: float


def is_multi_lora_enabled(args: Any) -> bool:
    # [FUNCTION DOC] HELPER: The multi-lora flag on args (set by validation), false if absent.
    return getattr(args, "multi_lora", False)


# Leaf module names that can live inside MoE experts (they also name the dense MLP
# projections); the bulk aliases expand to them during target-module resolution.
_EXPERT_LEAF_NAMES = frozenset({"linear_fc1", "linear_fc2", "gate_proj", "up_proj", "gate_up_proj", "down_proj"})
_ALL_MODULE_ALIASES = frozenset({"all", "all-linear", "all_linear"})


def targets_expert_leaves(target_modules: Any) -> bool:
    # [FUNCTION DOC] HELPER: Normalizes the selector (a single string becomes a list; none becomes
    # empty; entries stripped and lower-cased). True if any entry is a bulk alias (all, all-linear,
    # all_linear) or its last dotted component is a known expert leaf name (fc1, fc2, gate, up,
    # gate-up, down projections).
    """Whether ``target_modules`` can put adapters on MoE expert linears."""
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    entries = [str(tm).strip().lower() for tm in (target_modules or [])]
    if any(entry in _ALL_MODULE_ALIASES for entry in entries):
        return True
    # Map each entry (possibly a dotted or wildcard path) to its leaf module name.
    return any(entry.split(".")[-1] in _EXPERT_LEAF_NAMES for entry in entries)
