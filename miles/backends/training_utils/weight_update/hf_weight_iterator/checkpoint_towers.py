# [MODULE DOC]
# Reads multimodal "tower" weights (vision and audio encoders, whose HF parameter names contain a
# path component "visual" or "audio") directly from a Hugging Face safetensors checkpoint on disk,
# for backends that do not train those towers but still must push them to the inference engine.
# Loads once per checkpoint path (process-wide cache of CPU tensors) and yields them to the GPU as
# single-tensor units.
import functools
import glob
import json
import os

import torch
from safetensors import safe_open

_TOWERS = ("visual", "audio")


def is_tower_key(name: str) -> bool:
    # [FUNCTION DOC] True if any dotted component of the name other than the last equals "visual" or
    # "audio" (the final component, e.g. weight, is never tested).
    return any(part in _TOWERS for part in name.split(".")[:-1])


def _tower_keys_by_shard(hf_checkpoint: str) -> dict[str, list[str]]:
    # [FUNCTION DOC] HELPER: Map shard file name to the tower tensor names stored in it. If the
    # checkpoint directory has the safetensors index file, reads its weight map and returns the
    # tower keys per shard. Otherwise scans every safetensors file in sorted order, opens it on CPU
    # and lists keys (without reading tensors); shards with no tower keys are omitted. Reads files.
    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    by_shard: dict[str, list[str]] = {}
    if os.path.isfile(index_path):
        with open(index_path) as f:
            for key, shard in json.load(f)["weight_map"].items():
                if is_tower_key(key):
                    by_shard.setdefault(shard, []).append(key)
        return by_shard
    for path in sorted(glob.glob(os.path.join(hf_checkpoint, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = [k for k in f.keys() if is_tower_key(k)]
        if keys:
            by_shard[os.path.basename(path)] = keys
    return by_shard


@functools.cache
def _load_towers(hf_checkpoint: str) -> tuple[tuple[str, torch.Tensor], ...]:
    # [FUNCTION DOC] HELPER: Memoized per checkpoint path for the process lifetime. Loads every tower
    # tensor onto CPU, visiting shards in sorted name order and keys in sorted order within each, and
    # returns a tuple of (name, tensor) pairs. The cached CPU tensors are never freed.
    towers = []
    for shard, keys in sorted(_tower_keys_by_shard(hf_checkpoint).items()):
        with safe_open(os.path.join(hf_checkpoint, shard), framework="pt", device="cpu") as f:
            towers.extend((key, f.get_tensor(key)) for key in sorted(keys))
    return tuple(towers)


def iter_checkpoint_tower_units(hf_checkpoint: str, *, materialize: bool):
    # [FUNCTION DOC] Generator of single-tensor units: for each cached tower tensor, yields a list
    # with one (name, tensor copied to the current CUDA device) pair. Yields nothing and loads
    # nothing when materialize is false. No collectives.
    if not materialize:
        return
    for name, tensor in _load_towers(hf_checkpoint):
        yield [(name, tensor.to(torch.cuda.current_device()))]
