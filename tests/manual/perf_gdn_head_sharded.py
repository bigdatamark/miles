# [MODULE DOC] Manual GPU benchmark script (run with torchrun, two processes by default; it
# re-executes itself under torchrun when launched without a rank variable). Compares one gated
# delta-net layer's forward plus backward cost between a replicated reference implementation
# and the head-sharded Megatron layer, under tensor parallelism across all ranks and optionally
# sequence parallelism. Command line: mode word (sp or nosp), hidden size, key head count, value
# head count, then one or more sequence lengths. Rank zero prints a table row per length with
# wall milliseconds, GPU milliseconds, peak GiB for each variant and their ratios.
"""One GDN layer, fwd+bwd wall time, GPU time and peak memory: replicated (pre-sharding) vs head-sharded.

    torchrun --nproc_per_node=2 tests/manual/perf_gdn_head_sharded.py <sp|nosp> <hidden> <k_heads> <v_heads> <seq_len>...

With ``sp`` both run as they do under ``--sequence-parallel``: the replicated layer all-gathers its input
and slices its output (what the old HF-attention shim did), the head-sharded layer takes the SP shard.
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
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.profiler import ProfilerActivity, profile

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fast-gpu"))
from linear_attn_reference import ReplicatedGDN, build_layer, packed  # noqa: E402


def bench(fn, iters=10):
    # [FUNCTION DOC] HELPER: Measures a callable. Runs three warm-up calls, synchronizes the GPU,
    # resets peak memory statistics, times the given number of iterations (default 10) by wall
    # clock with synchronization, then runs the iterations again under the profiler collecting
    # device activity. Returns a triple: mean wall time per iteration in milliseconds, mean
    # summed self device time per iteration in milliseconds (profiler microseconds divided by
    # 1000), and peak allocated GPU memory in GiB. Peak memory is read after the profiled runs.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) / iters * 1e3
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    gpu = sum(e.self_device_time_total for e in prof.key_averages()) / iters / 1e3
    return wall, gpu, torch.cuda.max_memory_allocated() / 2**30


def main():
    # [FUNCTION DOC] Entry point for each torchrun worker. Parses the mode word, hidden size,
    # key and value head counts and sequence lengths from the command line; reads rank, world
    # size and local rank from the environment and binds the GPU. Initializes the NCCL process
    # group (120 second timeout) and model parallelism with tensor parallel size equal to the
    # world size, seeds the model-parallel random state with 1234, and builds a one-layer bf16
    # transformer config (16 attention heads). Creates the replicated reference layer and the
    # head-sharded layer from it (head dim 128 for keys and values). For each sequence length
    # (local length is length divided by world size under sequence parallelism): makes a random
    # bf16 input that requires gradients and a single-sample cumulative length tensor, benchmarks
    # the replicated path (all-gather input and scatter output when sequence parallel, batch-first
    # transposes, sum backward) and the sharded path (packed sequence parameters, sum backward),
    # and on rank zero prints both measurements and the old-over-new ratios. Ends with a barrier
    # and tears down model parallelism and the process group. Requires GPUs; no return value.
    sequence_parallel = sys.argv[1] == "sp"
    hidden, k_heads, v_heads = (int(a) for a in sys.argv[2:5])
    seq_lens = [int(a) for a in sys.argv[5:]]
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=16,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        tensor_model_parallel_size=world,
        sequence_parallel=sequence_parallel,
    )
    tp_group = parallel_state.get_tensor_model_parallel_group()
    heads = LinearAttnHeads(k_heads, v_heads, 128, 128)
    ref = ReplicatedGDN("qwen3_5", hidden, heads, torch.bfloat16).cuda()
    layer = build_layer(ref, config)
    if rank == 0:
        print(
            f"GDN layer fwd+bwd, hidden={hidden} heads={k_heads}/{v_heads}, TP={world}, SP={sequence_parallel}; wall ms / GPU ms / peak GiB"
        )
    for seq_len in seq_lens:
        local_len = seq_len // world if sequence_parallel else seq_len
        x_sbh = torch.randn(local_len, 1, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        cu = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

        def replicated(x=x_sbh, cu=cu):
            # [FUNCTION DOC] HELPER: One forward plus backward of the replicated reference layer
            # bound to the current input and cumulative lengths. Gathers the sequence-parallel
            # shards first when enabled, runs the layer batch-first and transposes back, scatters
            # the output when sequence parallel, then backpropagates the float32 sum. Accumulates
            # gradients into the input; returns nothing.
            full = (
                gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False, group=tp_group)
                if sequence_parallel
                else x
            )
            out = ref(full.transpose(0, 1), cu).transpose(0, 1)
            out = scatter_to_sequence_parallel_region(out, group=tp_group) if sequence_parallel else out
            out.float().sum().backward()

        old = bench(replicated)
        new = bench(lambda x=x_sbh, cu=cu: layer(x, packed_seq_params=packed(cu))[0].float().sum().backward())
        if rank == 0:
            print(
                f"  seq {seq_len:6d}  replicated {old[0]:7.2f} / {old[1]:7.2f} / {old[2]:5.2f}"
                f"   head-sharded {new[0]:7.2f} / {new[1]:7.2f} / {new[2]:5.2f}"
                f"   wall x{old[0] / new[0]:.2f}  gpu x{old[1] / new[1]:.2f}  mem x{old[2] / new[2]:.2f}"
            )
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__, *sys.argv[1:]])
    main()
