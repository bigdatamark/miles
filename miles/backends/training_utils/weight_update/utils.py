# [MODULE DOC] HELPER: Small shared helpers for weight sync: computing a rank's position among the
# ranks that hold identical gathered weights, and accumulating sha256 checksums of streamed LoRA
# tensors.
import hashlib

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


def get_data_replica_rank_and_size(parallel_state: ParallelState, placement: WeightUpdatePlacement) -> tuple[int, int]:
    # [FUNCTION DOC] Collective over the default process group when pipeline is not gathered. If
    # pipeline is gathered every rank holds the full set, so returns (global rank, world size).
    # Otherwise each rank's "column id" is the smallest global rank of its pipeline group; column ids
    # are all-gathered as Python objects, and the result is (index of this column id among the sorted
    # distinct ids, world size divided by pipeline size). So ranks in the same pipeline column share
    # a replica rank, and replica rank 0 is the column containing global rank 0.
    """(replica_rank, replica_size): this rank's index among the ranks that hold
    identical data after gathering per ``placement``, and their count. Collective."""
    if placement.gather_pp:
        return dist.get_rank(), dist.get_world_size()

    column_id = min(dist.get_process_group_ranks(parallel_state.pp.group))
    all_column_ids: list = [None] * dist.get_world_size()
    dist.all_gather_object(all_column_ids, column_id)
    return sorted(set(all_column_ids)).index(column_id), dist.get_world_size() // parallel_state.pp.size


def record_lora_checksums(bucket, checksums) -> None:
    # [FUNCTION DOC] For each (name, tensor) in a bucket: skips names without a colon (base
    # weights). Splits at the first colon into adapter name and HF key; computes the sha256 hex digest
    # of the tensor's raw bytes (detached, copied to CPU, contiguous, viewed as unsigned bytes); stores
    # it in the nested mapping under adapter name then key. Mutates the mapping in place; raises a
    # key error if the adapter name has no entry. Device-to-host copies per tensor.
    """Accumulate the sha256 checksums the engines verify at end_weight_update."""
    for name, tensor in bucket:
        if ":" not in name:
            continue
        lora_name, hf_key = name.split(":", 1)
        digest = hashlib.sha256(
            tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        checksums[lora_name][hf_key] = digest
