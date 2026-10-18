# [MODULE DOC] Distributed script-style test (torchrun with N GPUs; self-bootstrapping to 2
# processes) of the linear-attention layer wrapper using Kimi delta attention (hidden 256, 8 heads
# of dim 64, bfloat16, two packed sequences of 256 tokens each). Three checks against the sibling
# linear_attn_reference module (put on the import path by this file): sequence-parallel with tensor
# parallel size equal to the world size; context-parallel through the zigzag sequence layout with
# context parallel size equal to the world size; and a tensor-parallel distributed-checkpoint save and
# load round trip into a layer built from different weights. Relative error tolerance 2e-2 for
# outputs and input gradients; the checkpoint mismatch count must be exactly zero. Registered for the
# 2-GPU CUDA suite with the precision label.
"""LinearAttentionLayer's sequence-parallel and context-parallel paths against the replicated KDA reference,
and a TP-sharded checkpoint round trip of its sharded_state_dict.

    torchrun --nproc_per_node=2 tests/fast-gpu/test_linear_attn_layer.py
"""

import os
import sys
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads

sys.path.insert(0, os.path.dirname(__file__))
from linear_attn_reference import ReplicatedKDA, build_layer, gather, packed, rel_err  # noqa: E402

register_cuda_ci(est_time=120, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

HIDDEN = 256
HEADS = LinearAttnHeads(num_k_heads=8, num_v_heads=8, head_k_dim=64, head_v_dim=64)
SEQLENS = [256, 256]
TOLERANCE = 2e-2


def _config(tp: int, cp: int, sequence_parallel: bool) -> TransformerConfig:
    # [FUNCTION DOC] HELPER: Builds a one-layer bfloat16 transformer config (hidden 256, 8 heads) with
    # the given tensor-parallel size, context-parallel size and sequence-parallel flag.
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        params_dtype=torch.bfloat16,
        bf16=True,
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        sequence_parallel=sequence_parallel,
    )


def _setup(tp: int, cp: int):
    # [FUNCTION DOC] HELPER: Destroys any existing Megatron model-parallel state, re-initializes it
    # with the given tensor- and context-parallel sizes (on the already-initialized default group), and
    # seeds the model-parallel RNG with 1234.
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
    model_parallel_cuda_manual_seed(1234)


def _reference_and_input(seed: int):
    # [FUNCTION DOC] HELPER: Seeds the generator, builds a replicated Kimi delta attention reference
    # (bfloat16, GPU), a random input of shape (1, 512, hidden), a random output gradient of the same
    # shape, and integer cumulative sequence lengths [0, 256, 512]. Returns (reference, input, grad,
    # cumulative lengths).
    torch.manual_seed(seed)
    ref = ReplicatedKDA(HIDDEN, HEADS, torch.bfloat16).cuda()
    x = torch.randn(1, sum(SEQLENS), HIDDEN, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    return ref, x, grad, cu_seqlens


def _reference_out_and_dx(ref, x, grad, cu_seqlens):
    # [FUNCTION DOC] HELPER: Runs the reference on a gradient-requiring clone of x, backpropagates
    # grad, and returns the detached output and the input gradient, both with the batch dimension
    # removed (shape tokens by hidden).
    x_ref = x.clone().requires_grad_()
    out = ref(x_ref, cu_seqlens)
    out.backward(grad)
    return out.detach()[0], x_ref.grad[0]


def check_sequence_parallel(world: int) -> dict[str, float]:
    # [FUNCTION DOC] Sets up tensor parallelism equal to the world size, computes the reference output
    # and input gradient, builds the sharded layer with sequence parallelism, gives each rank its chunk
    # of the sequence (sequence-first layout with a singleton batch dimension), runs forward and
    # backward with the matching chunk of the output gradient, gathers the outputs and input gradients
    # along the sequence and returns relative errors under the keys sp_out and sp_dx. Collective.
    """TP = world with sequence parallelism: the layer sees this rank's sequence shard and returns one."""
    _setup(tp=world, cp=1)
    ref, x, grad, cu_seqlens = _reference_and_input(seed=7)
    out_ref, dx_ref = _reference_out_and_dx(ref, x, grad, cu_seqlens)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    layer = build_layer(ref, _config(tp=world, cp=1, sequence_parallel=True))
    shard = lambda t: t[0].chunk(world, dim=0)[tp_group.rank()].unsqueeze(1).contiguous()  # noqa: E731
    x_local = shard(x).requires_grad_()
    out, _ = layer(x_local, packed_seq_params=packed(cu_seqlens))
    out.backward(shard(grad))
    return {
        "sp_out": rel_err(out_ref, gather(out[:, 0], 0, tp_group)),
        "sp_dx": rel_err(dx_ref, gather(x_local.grad[:, 0], 0, tp_group)),
    }


def _zigzag(full: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Selects this context-parallel rank's tokens in the load-balanced zigzag
    # layout. Each packed sequence (lengths from the module's sequence length list) is split into 2 x
    # size equal chunks; the rank takes chunk number rank and chunk number (2 x size - 1 - rank) of
    # each sequence; results are concatenated in sequence order. Sequence lengths must divide evenly.
    parts = []
    for segment in full.split(SEQLENS):
        chunks = segment.chunk(2 * size)
        parts += [chunks[rank], chunks[2 * size - 1 - rank]]
    return torch.cat(parts)


def check_context_parallel(world: int) -> dict[str, float]:
    # [FUNCTION DOC] Sets up context parallelism equal to the world size and builds the sharded layer
    # with the all-gather context-parallel mode off (using the zigzag relayout). Inputs and output
    # gradients are zigzag-sliced for the rank; the layer's local output and input gradient are
    # compared with the zigzag slice of the reference's. Returns relative errors under cp_out and cp_dx.
    """CP = world through the zigzag relayout (allgather_cp=False), each sequence split across ranks."""
    _setup(tp=1, cp=world)
    ref, x, grad, cu_seqlens = _reference_and_input(seed=11)
    out_ref, dx_ref = _reference_out_and_dx(ref, x, grad, cu_seqlens)
    rank = parallel_state.get_context_parallel_rank()
    layer = build_layer(ref, _config(tp=1, cp=world, sequence_parallel=False), allgather_cp=False)
    x_local = _zigzag(x[0], rank, world).unsqueeze(1).contiguous().requires_grad_()
    out, _ = layer(x_local, packed_seq_params=packed(cu_seqlens))
    out.backward(_zigzag(grad[0], rank, world).unsqueeze(1))
    return {
        "cp_out": rel_err(_zigzag(out_ref, rank, world), out[:, 0]),
        "cp_dx": rel_err(_zigzag(dx_ref, rank, world), x_local.grad[:, 0]),
    }


def check_checkpoint_round_trip(world: int, ckpt_dir: str) -> dict[str, float]:
    # [FUNCTION DOC] Sets up tensor parallelism equal to the world size. Builds a layer from one
    # random reference (seed 21) and saves its sharded state dict with distributed checkpointing into
    # the directory; builds a second layer from a different reference (seed 22), loads the checkpoint
    # into it, and counts parameters that are not exactly equal between the two layers. Returns a
    # mapping with the single key ckpt_mismatched_params holding that count as a float. Collective;
    # all ranks must share the directory.
    """Save a TP-sharded layer with dist_checkpointing and load it into one holding other weights."""
    _setup(tp=world, cp=1)
    config = _config(tp=world, cp=1, sequence_parallel=False)
    saved = build_layer(_reference_and_input(seed=21)[0], config)
    dist_checkpointing.save(saved.sharded_state_dict(), ckpt_dir)
    loaded = build_layer(_reference_and_input(seed=22)[0], config)
    loaded.load_state_dict(dist_checkpointing.load(loaded.sharded_state_dict(), ckpt_dir))
    loaded_params = dict(loaded.named_parameters())
    mismatched = [name for name, p in saved.named_parameters() if not torch.equal(p, loaded_params[name])]
    return {"ckpt_mismatched_params": float(len(mismatched))}


def main():
    # [FUNCTION DOC] Per-rank entry point. Reads rank info from the environment, selects the GPU,
    # initializes NCCL (300 second timeout), runs the sequence-parallel and context-parallel checks,
    # creates a temporary checkpoint directory on rank 0 and broadcasts its path to all ranks, runs the
    # checkpoint round trip, destroys model-parallel state, and prints (rank 0) all errors. Failures are
    # any errors above the 2e-2 tolerance, or any checkpoint mismatch (the checkpoint key limit is 0).
    # After destroying the process group, prints the failures and exits with status 1, else rank 0
    # prints a success line.
    # KNOWN ISSUE: the temporary checkpoint directory is never removed.
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    errors = {**check_sequence_parallel(world), **check_context_parallel(world)}
    ckpt_dir = [tempfile.mkdtemp(prefix="linear_attn_ckpt_") if rank == 0 else None]
    dist.broadcast_object_list(ckpt_dir, src=0)
    errors.update(check_checkpoint_round_trip(world, ckpt_dir[0]))
    parallel_state.destroy_model_parallel()
    if rank == 0:
        print("  ".join(f"{k}={v:.2e}" for k, v in errors.items()))
    bad = {k: v for k, v in errors.items() if v > (0 if k.startswith("ckpt") else TOLERANCE)}
    dist.destroy_process_group()
    if bad:
        print(f"FAILED: {bad}")
        sys.exit(1)
    if rank == 0:
        print(f"LinearAttentionLayer SP / CP / checkpoint paths match at world size {world}")


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__])
    main()
