# [MODULE DOC]
# Save and load of one LoRA adapter slot in Megatron's distributed (torch_dist) checkpoint format.
# The checkpoint holds two top-level groups: adapter weights and (optionally) adapter optimizer
# state. Storage keys are rewritten from the slot-specific marker to a slot-agnostic one so a
# checkpoint saved from one slot loads into any slot. Writing goes through the checkpoint-directory
# writer (optionally with metadata); loading fills sharded "shell" objects from disk.
"""Per-slot checkpoints in megatron's torch_dist format.

Every tensor carries its global shard coordinates and a slot-agnostic key,
so a checkpoint reloads under any tp/pp/ep/world-size layout and into any
free slot."""

from collections.abc import Sequence
from pathlib import Path

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.utils import unwrap_model

from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir

_WEIGHTS_KEY = "adapter_weights"
_OPTIM_KEY = "adapter_optimizer"


def _slot_weights_sharded_state_dict(model: Sequence[DDP], slot: int) -> dict:
    # [FUNCTION DOC] HELPER: Merges the sharded state dicts of all unwrapped model chunks, keeps
    # entries whose key contains ".adapters.<slot>." and asserts at least one remains (assertion
    # error naming the slot otherwise). Returns the mapping of key to sharded tensor.
    sharded: dict = {}
    for chunk in unwrap_model(model):
        sharded |= chunk.sharded_state_dict()
    marker = f".adapters.{slot}."
    slot_sharded = {key: value for key, value in sharded.items() if marker in key}
    assert slot_sharded, f"slot {slot} exposed no adapter tensors"
    return slot_sharded


def _canonicalize_slot_keys(tree: dict, slot: int) -> dict:
    # [FUNCTION DOC] HELPER: Walks all leaf values of a nested mapping and, for those that have a
    # storage key attribute, replaces the substring ".adapters.<slot>." with ".adapter." in place.
    # Returns the same (mutated) tree. Leaves without a key attribute are untouched.
    """Rewrite storage keys so a checkpoint saved from slot i loads into slot j."""
    marker, canonical = f".adapters.{slot}.", ".adapter."
    for sharded in nested_values(tree):
        if hasattr(sharded, "key"):
            sharded.key = sharded.key.replace(marker, canonical)
    return tree


def save_slot(model: Sequence[DDP], slot_optimizer: SlotOptimizer, path: str, metadata: dict | None = None) -> None:
    # [FUNCTION DOC] Collective save. Builds the slot's sharded weights, builds the optimizer's
    # sharded state for saving (not loading), puts them under the weights and optimizer keys,
    # canonicalizes the keys (note: the optimizer's tensors may share objects with weights, so the
    # rewrite applies to both), and writes the checkpoint directory at the path through the
    # directory writer, passing metadata along. Side effects: disk writes, distributed collectives.
    weights = _slot_weights_sharded_state_dict(model, slot_optimizer.slot)
    sharded = {_WEIGHTS_KEY: weights, _OPTIM_KEY: slot_optimizer.sharded_state(weights, is_loading=False)}
    _canonicalize_slot_keys(sharded, slot_optimizer.slot)
    write_checkpoint_dir(
        path, lambda checkpoint_dir: dist_checkpointing.save(sharded, str(checkpoint_dir)), metadata=metadata
    )


def load_slot(model: Sequence[DDP], slot_optimizer: SlotOptimizer, path: str, load_optimizer: bool) -> None:
    # [FUNCTION DOC] Collective load into the slot. Builds the slot's sharded weight shells (and,
    # when load_optimizer is true, optimizer shells built for loading), canonicalizes their keys,
    # and loads them from the checkpoint directory in place. With load_optimizer, hands the loaded
    # optimizer state to the slot optimizer; otherwise leaves the freshly created optimizer state
    # and re-derives the optimizer's master weights from the loaded model weights. Side effects:
    # disk reads, mutation of model weights and slot optimizer.
    checkpoint_dir = str(Path(path))
    weights = _slot_weights_sharded_state_dict(model, slot_optimizer.slot)
    shells = {_WEIGHTS_KEY: weights}
    if load_optimizer:
        shells[_OPTIM_KEY] = slot_optimizer.sharded_state(weights, is_loading=True)
    _canonicalize_slot_keys(shells, slot_optimizer.slot)

    loaded = dist_checkpointing.load(shells, checkpoint_dir)
    if load_optimizer:
        slot_optimizer.load_sharded_state(loaded[_OPTIM_KEY])
    else:
        # weights-only load keeps the fresh Adam state the slot init just created
        slot_optimizer.reload_masters()
