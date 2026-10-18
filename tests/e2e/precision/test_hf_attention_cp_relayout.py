# [MODULE DOC] Distributed correctness test (multi-process, GPU, NCCL) for the context-parallel
# relayout helpers of the hybrid linear-attention model plugin that convert between the zigzag
# layout (each rank holds two mirrored sub-chunks per sequence) and the packed-shard layout (each
# rank holds a contiguous slice of the packed full sequence), including their gradients. Each rank
# builds synthetic inputs with recognizable values, runs both directions, and compares exactly.
# Registered for CUDA 4-GPU H200 (est. 30 s) and ROCm 4-GPU MI350 nightly (est. 30 s), label
# precision. Run as a script: if not already under torchrun (no rank variable) it replaces the
# process with torchrun using 4 processes on this file; otherwise runs the test on each rank.
# KNOWN ISSUE: the registration calls are followed by an import in the middle of the module,
# unlike the usual import-first layout.
"""Distributed correctness test for zigzag <-> packed-shard hybrid CP relayout.

Run with:
    torchrun --nproc_per_node=2 tests/e2e/precision/test_hf_attention_cp_relayout.py
    torchrun --nproc_per_node=4 tests/e2e/precision/test_hf_attention_cp_relayout.py
"""

import os
import sys

import torch
import torch.distributed as dist

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(est_time=30, suite="stage-c-4-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=30, suite="nightly-stage-c-4-gpu-mi350", labels=["precision"])

from miles_plugins.models.linear_attn import packed_shard_to_zigzag, zigzag_to_packed_shard


def setup_dist():
    # [FUNCTION DOC] HELPER: Reads rank, world size and local rank from the launcher environment
    # variables (raises a key error if missing), selects the local-rank CUDA device, and
    # initializes the default NCCL process group. Returns the tuple of rank, world size and local
    # rank. Side effects: GPU device selection and global process-group state.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def _make_subchunk(sample_id: int, sub_id: int, chunk_len: int, device: torch.device) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Builds one sub-chunk of recognizable float32 values on the given
    # device: consecutive numbers of the given length starting at sample id times 1000 plus
    # sub-chunk id times 100, shaped as length by 1 by 1.
    base = sample_id * 1000 + sub_id * 100
    values = torch.arange(base, base + chunk_len, device=device, dtype=torch.float32)
    return values.view(-1, 1, 1)


def _build_rank_inputs(rank: int, world_size: int, device: torch.device):
    # [FUNCTION DOC] HELPER: Builds this rank's test inputs. Two samples with sub-chunk lengths 3
    # and 5 are each split into twice-world-size sub-chunks. The rank's zigzag piece for a sample is
    # sub-chunk number rank followed by sub-chunk number (twice world size minus one minus rank).
    # The full sequence for a sample is all its sub-chunks in order; cumulative sequence lengths
    # grow by twice world size times the chunk length. A third tail-padding segment is appended:
    # each rank holds 3 values starting at rank times 10000, and its full sequence is those pieces
    # of all ranks concatenated by rank order (length world size times 3). The local zigzag tensor
    # (all rank pieces concatenated, gradient tracking on) has the local length; the expected
    # packed shard is the rank's contiguous slice of the packed full sequences with that same local
    # length. Returns the zigzag tensor, the expected packed shard, and the cumulative sequence
    # lengths as a 32-bit integer tensor on the device.
    chunk_lens = [3, 5]
    tail_pad_local_len = 3
    zigzag_chunks = []
    full_sequences = []
    cu = [0]

    for sample_id, chunk_len in enumerate(chunk_lens):
        subchunks = [_make_subchunk(sample_id, sub_id, chunk_len, device) for sub_id in range(2 * world_size)]
        zigzag_chunks.extend([subchunks[rank], subchunks[2 * world_size - 1 - rank]])
        full_sequences.append(torch.cat(subchunks, dim=0))
        cu.append(cu[-1] + 2 * world_size * chunk_len)

    tail_pad = (rank * 10000 + torch.arange(tail_pad_local_len, device=device, dtype=torch.float32)).view(-1, 1, 1)
    zigzag_chunks.append(tail_pad)
    full_sequences.append(
        torch.cat(
            [
                (r * 10000 + torch.arange(tail_pad_local_len, device=device, dtype=torch.float32)).view(-1, 1, 1)
                for r in range(world_size)
            ],
            dim=0,
        )
    )
    cu.append(cu[-1] + world_size * tail_pad_local_len)

    zigzag = torch.cat(zigzag_chunks, dim=0).requires_grad_(True)
    packed_full = torch.cat(full_sequences, dim=0)
    local_len = zigzag.size(0)
    packed_shard = packed_full[rank * local_len : (rank + 1) * local_len]
    cu_seqlens = torch.tensor(cu, device=device, dtype=torch.int32)
    return zigzag, packed_shard, cu_seqlens


def test_relayout(rank: int, world_size: int):
    # [FUNCTION DOC] The test body, run on every rank with the whole world as the context-parallel
    # group and the device being the CUDA device whose index equals the rank (assumes a single
    # node). Steps: (1) convert zigzag to packed shard and require exact equality with the
    # expected shard; backpropagate the sum of the product with the expected shard and require the
    # zigzag gradient to equal the zigzag values (the inverse mapping). (2) Convert a gradient-
    # tracking copy of the expected packed shard back to zigzag and require exact equality with the
    # original zigzag; backpropagate the sum of the product with the zigzag values and require the
    # packed gradient to equal the expected shard. Rank 0 prints the three pass flags and, if any
    # failed, exits the process with status 1.
    # KNOWN ISSUE: only rank 0 exits on failure, so a failure on another rank alone is not
    # reported; the function takes plain positional parameters and a test-style name, so a test
    # collector would treat them as fixtures (it is meant to be run via the script entry point).
    device = torch.device(f"cuda:{rank}")
    cp_group = dist.group.WORLD

    zigzag, expected_packed_shard, cu_seqlens = _build_rank_inputs(rank, world_size, device)

    packed_shard = zigzag_to_packed_shard(zigzag, cu_seqlens, cp_group, rank, world_size)
    packed_ok = torch.equal(packed_shard, expected_packed_shard)
    (packed_shard * expected_packed_shard).sum().backward()
    grad_ok = torch.equal(zigzag.grad, zigzag.detach())

    packed_leaf = expected_packed_shard.clone().requires_grad_(True)
    roundtrip = packed_shard_to_zigzag(packed_leaf, cu_seqlens, cp_group, rank, world_size)
    roundtrip_ok = torch.equal(roundtrip, zigzag.detach())
    (roundtrip * zigzag.detach()).sum().backward()
    grad_ok = grad_ok and torch.equal(packed_leaf.grad, expected_packed_shard)

    passed = packed_ok and roundtrip_ok and grad_ok
    if rank == 0:
        print(f"\n=== HF Attention Hybrid CP Relayout Test CP={world_size} ===")
        print(f"zigzag->packed PASS: {packed_ok}")
        print(f"roundtrip PASS: {roundtrip_ok}")
        print(f"backward PASS: {grad_ok}")
        if not passed:
            sys.exit(1)


def main():
    # [FUNCTION DOC] Initializes the distributed process group, runs the relayout test on this rank,
    # and always destroys the process group afterwards (even on error or exit).
    rank, world_size, _ = setup_dist()
    try:
        test_relayout(rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Self-bootstrap under torchrun when invoked as `python3 file.py` (the
    # CUDA CI runner's mode). Already inside torchrun => RANK is set.
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=4", __file__])
    main()
