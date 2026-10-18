# [MODULE DOC] Distributed (multi-process, GPU, NCCL) precision test comparing two linear-attention
# layer types (gated delta net and a KDA variant), head-sharded and run under native context
# parallelism with an all-gather scheme, against the replicated reference implementation on the
# whole sequence. The sequence has two packed segments whose boundary (one third of the length)
# falls inside a context-parallel shard so recurrent state must cross ranks. Forward output and
# input gradient must match within relative error 0.01. The reference layers and helpers are
# imported from a sibling fast-gpu test directory, which is added to the import path at import
# time. Registered for CUDA 4-GPU H200 (est. 300 s) and ROCm 4-GPU MI350 nightly (est. 200 s),
# label precision. Module constant: hidden size 256. Run as a script: if no rank variable is set,
# replaces the process with torchrun using 4 processes; otherwise runs main on each rank.
"""Head-sharded GDN and KDA under native fla context parallelism against the same layer on the whole
sequence. The sequence boundary falls inside a CP shard, so the recurrent state is passed mid-sequence.

    torchrun --nproc_per_node=2 tests/e2e/precision/test_qwen3_5_cp_correctness.py   # CP=2
    torchrun --nproc_per_node=4 tests/e2e/precision/test_qwen3_5_cp_correctness.py   # CP=4
"""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "fast-gpu"))
from linear_attn_reference import ReplicatedGDN, ReplicatedKDA, build_layer, packed, rel_err  # noqa: E402

register_cuda_ci(est_time=300, suite="stage-c-4-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=200, suite="nightly-stage-c-4-gpu-mi350", labels=["precision"])

HIDDEN = 256


def check(ref, config, rank, world_size):
    # [FUNCTION DOC] Runs one layer comparison and returns two relative errors: forward output and
    # input gradient. Builds the context-parallel layer from the reference layer and config with
    # all-gather context parallelism. Total length is 128 times world size; random bfloat16 input
    # (fixed seed 123) of batch 1; cumulative lengths split the sequence at one third. Reference:
    # run the layer core on the full input, apply the output projection, backpropagate the sum.
    # Context-parallel: each rank takes its contiguous slice of the sequence in sequence-first
    # layout, runs the layer with packed-sequence parameters, sums the output, sums the loss
    # across ranks, and backpropagates. Outputs and input gradients from all ranks are gathered
    # along the sequence and compared with the reference using the imported relative-error helper.
    # The loss reduction is not part of the autograd graph, so each rank's gradient is local.
    # Side effects: global random seed, collective communication.
    layer = build_layer(ref, config, allgather_cp=True)
    core = layer.linear_attn
    total = 128 * world_size
    torch.manual_seed(123)
    full = torch.randn(1, total, HIDDEN, device="cuda", dtype=torch.bfloat16)
    full_cu = torch.tensor([0, total // 3, total], dtype=torch.int32, device="cuda")

    x_ref = full.clone().requires_grad_()
    ref_out = core.out_proj(core(x_ref, full_cu, None))
    ref_out.sum().backward()

    local = total // world_size
    x_local = full[:, rank * local : (rank + 1) * local].transpose(0, 1).contiguous().requires_grad_()
    cp_out, _ = layer(x_local, packed_seq_params=packed(full_cu))
    loss = cp_out.sum()
    dist.all_reduce(loss)
    loss.backward()

    def gather_seq(t):
        # [FUNCTION DOC] HELPER: All-gathers the given tensor from every rank (made contiguous) and
        # concatenates the pieces along the first (sequence) dimension in rank order. Returns the
        # concatenated tensor.
        parts = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(parts, t.contiguous())
        return torch.cat(parts, dim=0)

    return rel_err(ref_out.detach()[0], gather_seq(cp_out.detach())[:, 0]), rel_err(
        x_ref.grad[0], gather_seq(x_local.grad)[:, 0]
    )


def main():
    # [FUNCTION DOC] Entry point per rank. Reads rank, world size and local rank from the launcher
    # environment, selects the CUDA device, initializes the NCCL group with a 120-second timeout,
    # initializes Megatron model parallelism with tensor size 1 and context size equal to the world
    # size, and seeds the model-parallel random state with 1234. Builds a one-layer bfloat16
    # transformer config (hidden 256, 4 heads, GPU initialization). Seeds with 42 and creates two
    # replicated reference layers on the GPU: a gated delta net in qwen3_5 style (head layout 2, 4,
    # 64, 64) and a KDA layer (head layout 4, 4, 64, 64). For each, calls check and passes if both
    # errors are below 0.01; rank 0 prints the errors and PASS or FAIL. Always tears down model
    # parallelism and the process group afterwards. Exits with status 1 if any layer failed on
    # this rank.
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=world_size)
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        context_parallel_size=world_size,
    )
    torch.manual_seed(42)
    refs = {
        "gdn": ReplicatedGDN("qwen3_5", HIDDEN, LinearAttnHeads(2, 4, 64, 64), torch.bfloat16).cuda(),
        "kda": ReplicatedKDA(HIDDEN, LinearAttnHeads(4, 4, 64, 64), torch.bfloat16).cuda(),
    }
    failed = False
    try:
        for name, ref in refs.items():
            out_diff, grad_diff = check(ref, config, rank, world_size)
            ok = out_diff < 1e-2 and grad_diff < 1e-2
            failed |= not ok
            if rank == 0:
                print(
                    f"[{name} CP={world_size}] rel err fwd {out_diff:.2e}  dx {grad_diff:.2e}  {'PASS' if ok else 'FAIL'}"
                )
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=4", __file__])
    main()
