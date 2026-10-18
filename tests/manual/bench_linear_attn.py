# [MODULE DOC] Manual multi-gpu benchmark for the head-sharded linear-attention layer (gated delta
# net or kimi delta attention) of the Megatron backend. Launched with torchrun using
# tensor-parallel size times context-parallel size processes; command line gives the tensor
# parallel size, the context parallel size and optionally sequence lengths (default 4096 to 131072
# doubling). For each model in the MODELS table (hidden size, head layout, layer kind) and each
# sequence length, rank 0 prints one line with wall-clock milliseconds and peak gpu memory (GiB) of
# forward plus backward for the head-sharded layer. With context parallel size 1 it also times the
# replicated reference layer it replaced and prints speed and memory ratios and the relative
# errors of output and input gradient versus the replicated layer. With context parallel size
# above 1 it compares the context-parallel layer (contiguous shards) with the same layer run
# without context parallelism and prints relative errors per output and input gradient. Reference
# layers and helpers are imported from the fast-gpu test directory, which is added to the import
# path. Out-of-memory cases print "OOM". Requires RANK and LOCAL_RANK environment variables set by
# the launcher; initialises and tears down the distributed groups.
"""Speed, memory and precision of the head-sharded linear-attention layer at the attention configs of the
latest models, across sequence lengths and one TP x CP layout (world size = TP * CP).

    torchrun --nproc_per_node=<TP*CP> tests/manual/bench_linear_attn.py <TP> <CP> [seq_len ...]

Per (model, seq_len): wall ms / peak GiB of fwd+bwd for the head-sharded layer (sequence parallel,
zigzag CP relayout as in training) and, at CP=1, the replicated layer it replaced; relative error of
the output and input gradient against the replicated layer (CP=1) or against the same layer without
CP (CP>1, contiguous shards).
"""

import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fast-gpu"))
from linear_attn_reference import ReplicatedGDN, ReplicatedKDA, build_layer, packed, rel_err  # noqa: E402

MODELS = {
    "qwen3.8-flash GDN": (2560, LinearAttnHeads(16, 48, 128, 128), "gdn"),
    "glm5.3-flash KDA": (4096, LinearAttnHeads(64, 64, 128, 128), "kda"),
    "kimi-k3 KDA": (7168, LinearAttnHeads(96, 96, 128, 128), "kda"),
}


def wall_ms(fn, iters):
    # [FUNCTION DOC] HELPER: Warm-up call, gpu synchronise, reset the peak memory statistics, then
    # time iters calls with the wall clock and synchronise. Returns a pair: average milliseconds
    # per call and the peak allocated gpu memory in GiB since the reset.
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3, torch.cuda.max_memory_allocated() / 2**30


def zigzag_shard(full, cp_rank, cp_size):
    # [FUNCTION DOC] HELPER: Splits the tensor along its first (sequence) dimension into twice
    # cp_size equal chunks and returns the contiguous concatenation of chunk cp_rank and chunk
    # (2 * cp_size - 1 - cp_rank) (the zigzag layout used by context parallel training).
    chunks = full.chunk(2 * cp_size, dim=0)
    return torch.cat([chunks[cp_rank], chunks[2 * cp_size - 1 - cp_rank]]).contiguous()


def contiguous_shard(full, cp_rank, cp_size):
    # [FUNCTION DOC] HELPER: Returns the cp_rank-th of cp_size equal contiguous chunks of the first
    # dimension, as a contiguous tensor.
    return full.chunk(cp_size, dim=0)[cp_rank].contiguous()


def tp_gather(x, tp_group):
    # [FUNCTION DOC] HELPER: Gathers the sequence-parallel shards of x across the tensor-parallel
    # group along the first dimension (without tensor-parallel output gradient handling) when the
    # group has more than one member; otherwise returns x unchanged.
    return (
        gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False, group=tp_group)
        if tp_group.size() > 1
        else x
    )


def sp_shard(x, tp_group):
    # [FUNCTION DOC] HELPER: Returns this rank's contiguous sequence-parallel slice of x along the
    # first dimension (chunk number equal to the tensor-parallel rank out of the group size).
    return x.chunk(tp_group.size(), dim=0)[tp_group.rank()].contiguous()


def run(name, seq_len, tp, cp, log):
    # [FUNCTION DOC] HELPER: Benchmarks one (model, sequence length) case. Builds a transformer
    # config (one layer, bfloat16, sequence parallel when the tensor-parallel size exceeds 1), a
    # seeded replicated reference layer of the model's kind, random full input and output gradient
    # of shape (sequence, 1, hidden) on the gpu, and a cumulative sequence length tensor for a
    # single packed sequence. Builds the head-sharded layer, feeds it the zigzag context shard
    # further split for sequence parallelism, and measures it. For context parallel size 1 also
    # measures the replicated path and compares output and input gradient. Otherwise runs the
    # contiguous-shard layer and a no-context-parallel reference built from the same layer's
    # attention core, and compares the overlapping window of output and input gradient after
    # gathering over tensor parallelism. Collects the output into one text row and passes it to
    # the log callable (printing only on rank 0). On gpu out-of-memory appends "OOM" to the row.
    # Empties the gpu cache at the end. The timed layer uses the zigzag context layout while the
    # context-parallel comparison layer uses contiguous shards.
    hidden, heads, kind = MODELS[name]
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=heads.num_v_heads,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        sequence_parallel=tp > 1,
    )
    torch.manual_seed(7)
    ref = (
        ReplicatedKDA(hidden, heads, torch.bfloat16)
        if kind == "kda"
        else ReplicatedGDN("qwen3_5", hidden, heads, torch.bfloat16)
    ).cuda()
    torch.manual_seed(11)
    full = torch.randn(seq_len, 1, hidden, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(full)
    cu = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
    iters = max(2, min(10, 262144 // seq_len))
    row = f"[{name:18s} TP={tp} CP={cp} seq={seq_len:6d}]"
    try:
        layer = build_layer(ref, config, allgather_cp=False)
        x = sp_shard(zigzag_shard(full, cp_group.rank(), cp), tp_group).requires_grad_()
        new_ms, new_mem = wall_ms(lambda: layer(x, packed_seq_params=packed(cu))[0].float().sum().backward(), iters)
        row += f"  head-sharded {new_ms:8.2f} ms {new_mem:6.2f} GiB"

        if cp == 1:

            def replicated(x=x):
                # [FUNCTION DOC] HELPER: Forward of the replicated reference layer for the given
                # sequence-parallel shard (default: the benchmark input): gathers the full
                # sequence across tensor parallelism when its size exceeds 1, runs the reference
                # layer on the transposed (batch-first) input with the cumulative lengths,
                # transposes back, and scatters the output back to sequence-parallel shards when
                # the tensor-parallel size exceeds 1. Returns the output.
                whole = (
                    gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False, group=tp_group)
                    if tp > 1
                    else x
                )
                out = ref(whole.transpose(0, 1), cu).transpose(0, 1)
                return scatter_to_sequence_parallel_region(out, group=tp_group) if tp > 1 else out

            old_ms, old_mem = wall_ms(lambda: replicated().float().sum().backward(), iters)
            row += f"  replicated {old_ms:8.2f} ms {old_mem:6.2f} GiB  speed x{old_ms / new_ms:4.2f} mem x{old_mem / new_mem:4.2f}"
            x_old, x_new = x.detach().clone().requires_grad_(), x.detach().clone().requires_grad_()
            g = sp_shard(grad, tp_group)
            out_old = replicated(x_old)
            out_old.backward(g)
            out_new = layer(x_new, packed_seq_params=packed(cu))[0]
            out_new.backward(g)
            row += f"  rel err out {rel_err(out_old, out_new):.1e} dx {rel_err(x_old.grad, x_new.grad):.1e}"
        else:
            contig = build_layer(ref, config, allgather_cp=True)
            x_cp = sp_shard(contiguous_shard(full, cp_group.rank(), cp), tp_group).requires_grad_()
            out_cp = contig(x_cp, packed_seq_params=packed(cu))[0]
            out_cp.backward(sp_shard(contiguous_shard(grad, cp_group.rank(), cp), tp_group))
            core = contig.linear_attn
            x_ref = sp_shard(full, tp_group).requires_grad_()
            whole = (
                gather_from_sequence_parallel_region(x_ref, tensor_parallel_output_grad=True, group=tp_group)
                if tp > 1
                else x_ref
            )
            out_ref = core.out_proj(core(whole.transpose(0, 1), cu, None)).transpose(0, 1)
            out_ref = (
                reduce_scatter_to_sequence_parallel_region(out_ref, group=tp_group)
                if tp > 1
                else reduce_from_tensor_model_parallel_region(out_ref, group=tp_group)
            )
            out_ref.backward(sp_shard(grad, tp_group))
            start = cp_group.rank() * (seq_len // cp)
            ref_out, ref_dx = tp_gather(out_ref.detach(), tp_group), tp_gather(x_ref.grad, tp_group)
            cp_out, cp_dx = tp_gather(out_cp.detach(), tp_group), tp_gather(x_cp.grad, tp_group)
            window = slice(start, start + seq_len // cp)
            row += f"  vs no-CP rel err out {rel_err(ref_out[window], cp_out):.1e} dx {rel_err(ref_dx[window], cp_dx):.1e}"
    except torch.cuda.OutOfMemoryError:
        row += "  OOM"
    torch.cuda.empty_cache()
    log(row)


def main():
    # [FUNCTION DOC] Entry point. Reads tensor-parallel size, context-parallel size and optional
    # sequence lengths from the command line (no validation; missing arguments raise an index
    # error), reads RANK and LOCAL_RANK from the environment, selects the gpu, initialises the
    # nccl process group (30 minute timeout), the Megatron model-parallel groups and the model
    # parallel gpu random seed 1234. Defines a log function that prints (flushing) on rank 0 and
    # ignores output elsewhere, runs every model at every sequence length, then synchronises and
    # destroys the parallel state and process group.
    tp, cp = int(sys.argv[1]), int(sys.argv[2])
    seq_lens = [int(s) for s in sys.argv[3:]] or [4096, 8192, 16384, 32768, 65536, 131072]
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
    model_parallel_cuda_manual_seed(1234)
    log = (lambda s: print(s, flush=True)) if rank == 0 else (lambda s: None)
    for name in MODELS:
        for seq_len in seq_lens:
            run(name, seq_len, tp, cp, log)
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
