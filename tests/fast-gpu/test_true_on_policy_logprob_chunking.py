# [MODULE DOC] Two-GPU tests (tensor-parallel size 2 via spawned NCCL worker processes) of the
# chunked log-probability and entropy computation in "true on policy" mode with a vocabulary sharded
# across tensor-parallel ranks and padded (padding columns hold large logits that must be ignored).
# Three tests: (1) exact float64 correctness and gradients against a full-vocabulary reference for
# unchunked and chunked runs, with and without entropy, with a sampling mask; (2) bfloat16/float16
# reduction-error measurements showing chunking is bitwise identical to unchunked in values and
# gradients (and finite, zero gradient on padding); (3) peak GPU memory of the full log-prob wrapper
# for 8192 rows of a 151936-token vocabulary: chunked must save at least half of one full-size bfloat16
# buffer versus unchunked on every rank. Each test raises a runtime error (not skip) if fewer than two
# GPUs are visible. Registered for 2-GPU Hopper CUDA and ROCm nightly suites (megatron label).
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(est_time=180, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper"])
register_rocm_ci(est_time=60, suite="nightly-stage-c-2-gpu-mi350", labels=["megatron"])

import gc
import os
import socket
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.math_utils import calculate_log_probs_and_entropy
from miles.backends.training_utils.parallel import set_parallel_state
from miles.utils.ft_utils.process_group_utils import GroupInfo

_WORLD_SIZE = 2
_CORRECTNESS_ROWS = 5
_CORRECTNESS_CHUNK_SIZE = 2
_LOCAL_PADDED_VOCAB_SIZE = 4
_REAL_VOCAB_SIZE = 7

_MEMORY_ROWS = 8192
_MEMORY_CHUNK_SIZE = 1024
_MEMORY_REAL_VOCAB_SIZE = 151936
_MEMORY_LOCAL_PADDED_VOCAB_SIZE = 76032


def _free_port() -> int:
    # [FUNCTION DOC] HELPER: Returns a currently free local TCP port by binding to port 0 on
    # localhost and reading the assigned number (the socket is closed before returning).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def _init_worker(rank: int, world_size: int, port: int) -> torch.device:
    # [FUNCTION DOC] HELPER: Per-process setup. Sets the rendezvous environment variables to
    # localhost and the port, binds the GPU equal to the rank, initializes the default NCCL group, and
    # returns the CUDA device for the rank.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return torch.device("cuda", rank)


def _correctness_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Returns hand-written float64 logits (5 rows by 8 padded vocabulary
    # columns, the last column a large padding value 80 to 120), a boolean sampling mask of the same
    # shape (padding column always true), and 5 target token ids, all on the given device. The real
    # vocabulary size is 7, the 8th column is padding.
    full_logits = torch.tensor(
        [
            [0.2, -1.1, 2.0, 0.7, -0.3, 1.4, -2.2, 80.0],
            [1.5, 0.1, -0.8, 2.3, -1.7, 0.9, 1.1, 90.0],
            [-0.5, 2.1, 0.4, -1.3, 1.7, -0.2, 0.8, 100.0],
            [2.4, -0.6, 1.2, 0.3, -1.0, 1.8, -2.5, 110.0],
            [-1.4, 0.6, 2.2, -0.1, 1.0, -2.0, 0.5, 120.0],
        ],
        device=device,
        dtype=torch.float64,
    )
    full_sampling_mask = torch.tensor(
        [
            [True, False, False, False, True, False, False, True],
            [False, True, True, False, False, True, False, True],
            [False, False, False, True, False, False, True, True],
            [True, False, False, False, False, True, True, True],
            [False, False, True, False, True, False, True, True],
        ],
        device=device,
        dtype=torch.bool,
    )
    tokens = torch.tensor([4, 1, 6, 5, 2], device=device, dtype=torch.long)
    return full_logits, full_sampling_mask, tokens


def _assert_correctness_case(rank: int, device: torch.device, *, chunk_size: int, objective: str) -> None:
    # [FUNCTION DOC] HELPER: Runs one tensor-parallel (size 2, 4 local vocabulary columns each)
    # correctness case on this rank. Takes the rank's shard of the logits and mask and calls the
    # log-prob/entropy function with the chunk size (-1 means unchunked), true-on-policy mode, real
    # vocabulary 7, temperature 1 and entropy only when the objective is "entropy". The reference is
    # computed in float64 on the unsharded real-vocabulary logits: masked log-softmax gathered at the
    # target tokens for log probs, and unmasked entropy. Asserts log probs (and entropy) match to 1e-10
    # relative, including the last row (which is dominated by the padding-like column in the raw data).
    # For the entropy objective also asserts the masked entropy would differ (so the test is
    # discriminating). Then differentiates a weighted sum of the chosen quantity (log probs or entropy)
    # for both, and asserts the local gradient equals the matching slice of the reference gradient with
    # zeros in padded columns; on the last rank the final padded column's gradient must be zero.
    full_logits, full_sampling_mask, tokens = _correctness_inputs(device)
    shard_start = rank * _LOCAL_PADDED_VOCAB_SIZE
    shard_end = shard_start + _LOCAL_PADDED_VOCAB_SIZE
    local_logits = full_logits[:, shard_start:shard_end].detach().clone().requires_grad_(True)
    local_sampling_mask = full_sampling_mask[:, shard_start:shard_end]

    with_entropy = objective == "entropy"
    actual_log_probs, actual_entropy = calculate_log_probs_and_entropy(
        local_logits,
        tokens,
        dist.group.WORLD,
        with_entropy=with_entropy,
        entropy_requires_grad=True,
        chunk_size=chunk_size,
        true_on_policy=True,
        vocab_size=_REAL_VOCAB_SIZE,
        sampling_mask=local_sampling_mask,
        temperature=1.0,
    )

    reference_logits = full_logits[:, :_REAL_VOCAB_SIZE].detach().clone().requires_grad_(True)
    reference_mask = full_sampling_mask[:, :_REAL_VOCAB_SIZE]
    reference_masked_log_probs = torch.log_softmax(
        reference_logits.masked_fill(~reference_mask, float("-inf")), dim=-1
    )
    reference_log_probs = reference_masked_log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    reference_entropy_log_probs = torch.log_softmax(reference_logits, dim=-1)
    reference_entropy = -(reference_entropy_log_probs.exp() * reference_entropy_log_probs).sum(dim=-1)

    torch.testing.assert_close(actual_log_probs, reference_log_probs, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual_log_probs[-1], reference_log_probs[-1], rtol=1e-10, atol=1e-12)

    weights = torch.tensor([0.5, -1.0, 0.75, 1.25, -0.25], device=device, dtype=torch.float64)
    if objective == "log_prob":
        assert actual_entropy is None
        actual_value = (actual_log_probs * weights).sum()
        reference_value = (reference_log_probs * weights).sum()
    else:
        assert actual_entropy is not None
        torch.testing.assert_close(actual_entropy, reference_entropy, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(actual_entropy[-1], reference_entropy[-1], rtol=1e-10, atol=1e-12)

        finite_masked_log_probs = reference_masked_log_probs.masked_fill(~reference_mask, 0)
        masked_entropy = -(reference_masked_log_probs.exp() * finite_masked_log_probs).sum(dim=-1)
        assert not torch.allclose(reference_entropy, masked_entropy)
        actual_value = (actual_entropy * weights).sum()
        reference_value = (reference_entropy * weights).sum()

    (actual_gradient,) = torch.autograd.grad(actual_value, local_logits)
    (reference_gradient,) = torch.autograd.grad(reference_value, reference_logits)

    expected_local_gradient = torch.zeros_like(local_logits)
    real_shard_end = min(shard_end, _REAL_VOCAB_SIZE)
    if shard_start < real_shard_end:
        expected_local_gradient[:, : real_shard_end - shard_start] = reference_gradient[:, shard_start:real_shard_end]
    torch.testing.assert_close(actual_gradient, expected_local_gradient, rtol=1e-10, atol=1e-12)
    if rank == _WORLD_SIZE - 1:
        torch.testing.assert_close(actual_gradient[:, -1], torch.zeros_like(actual_gradient[:, -1]))


def _correctness_worker(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Spawn entry point. Initializes the process group, runs the correctness
    # case for chunk sizes -1 and 2, each for the log-prob and entropy objectives, then barriers; the
    # process group is always destroyed.
    device = _init_worker(rank, world_size, port)
    try:
        for chunk_size in (-1, _CORRECTNESS_CHUNK_SIZE):
            _assert_correctness_case(rank, device, chunk_size=chunk_size, objective="log_prob")
            _assert_correctness_case(rank, device, chunk_size=chunk_size, objective="entropy")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _values_and_separate_gradients(
    logits: torch.Tensor, log_probs: torch.Tensor, entropy: torch.Tensor, weights: torch.Tensor
) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Differentiates log probs and entropy separately with respect to the
    # logits using the given per-row weights as upstream gradients (the first with the graph retained).
    # Returns a mapping with detached log probs, detached entropy and the two gradients.
    (log_prob_gradient,) = torch.autograd.grad(log_probs, logits, grad_outputs=weights, retain_graph=True)
    (entropy_gradient,) = torch.autograd.grad(entropy, logits, grad_outputs=weights)
    return {
        "log_probs": log_probs.detach(),
        "entropy": entropy.detach(),
        "log_prob_gradient": log_prob_gradient,
        "entropy_gradient": entropy_gradient,
    }


def _reduction_error(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Compares in float64 after flattening each row. Returns a two-element
    # tensor: the maximum absolute difference over all elements, and the maximum over rows of the row's
    # L2 difference divided by the row's reference L2 norm (floored at the smallest positive float64).
    delta = (actual.double() - reference.double()).reshape(actual.shape[0], -1)
    reference = reference.double().reshape(actual.shape[0], -1)
    # Row-wise L2 avoids dividing each near-zero vocabulary gradient by itself.
    relative_l2 = delta.norm(dim=-1) / reference.norm(dim=-1).clamp_min(torch.finfo(torch.float64).tiny)
    return torch.stack((delta.abs().max(), relative_l2.max()))


def _assert_low_precision_reductions(rank: int, device: torch.device, dtype: torch.dtype, masked: bool) -> None:
    # [FUNCTION DOC] HELPER: Measures reduction error of chunking in low precision. Builds 17 rows of
    # random logits (seeded 2825, per-row scale growing from 2 to 6) over the full padded vocabulary
    # (2 x 76032 columns for 2 ranks; real vocabulary 151936; padding columns set to 100) in the
    # given dtype, pseudo-random targets, signed weights, and optionally a sampling mask with the
    # targets forced on. The float64 reference promotes the same quantized inputs. For chunk sizes -1,
    # 1 and 4 runs the function on this rank's shard, computes values and separate gradients, takes the
    # unchunked run as baseline, and for each quantity max-reduces the errors across ranks versus the
    # baseline and versus float64, printed on rank 0. Asserts every quantity is finite and bitwise equal
    # to the baseline (chunking adds no error), and on the last rank that gradient columns in padding
    # are exactly zero.
    rows = 17
    real_vocab_size = _MEMORY_REAL_VOCAB_SIZE
    local_vocab_size = _MEMORY_LOCAL_PADDED_VOCAB_SIZE
    generator = torch.Generator(device=device).manual_seed(2825)
    row_ids = torch.arange(rows, device=device)
    vocab_ids = torch.arange(local_vocab_size * _WORLD_SIZE, device=device)
    scales = torch.linspace(2.0, 6.0, rows, device=device).unsqueeze(-1)
    full_logits = (torch.randn(rows, vocab_ids.numel(), generator=generator, device=device) * scales).to(dtype)
    full_logits[:, real_vocab_size:] = 100
    tokens = (row_ids * 104729 + 17) % real_vocab_size
    weights = ((row_ids % 4 + 1) / 4 * torch.where(row_ids % 2 == 0, 1, -1)).to(dtype)
    full_mask = None
    if masked:
        full_mask = (vocab_ids.unsqueeze(0) + row_ids.unsqueeze(-1)) % 3 != 0
        full_mask[row_ids, tokens] = True

    # Promote the same quantized inputs, so this measures arithmetic, not input quantization.
    reference_logits = full_logits.double().requires_grad_(True)
    real_logits = reference_logits[:, :real_vocab_size]
    scoring_logits = (
        real_logits if full_mask is None else real_logits.masked_fill(~full_mask[:, :real_vocab_size], -torch.inf)
    )
    reference_log_probs = scoring_logits.log_softmax(dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    reference_full_log_probs = real_logits.log_softmax(dim=-1)
    reference_entropy = -(reference_full_log_probs.exp() * reference_full_log_probs).sum(dim=-1)
    reference = _values_and_separate_gradients(
        reference_logits, reference_log_probs, reference_entropy, weights.double()
    )
    shard = slice(rank * local_vocab_size, (rank + 1) * local_vocab_size)
    for name in ("log_prob_gradient", "entropy_gradient"):
        reference[name] = reference[name][:, shard]

    baseline = None
    for chunk_size in (-1, 1, 4):
        local_logits = full_logits[:, shard].clone().requires_grad_(True)
        log_probs, entropy = calculate_log_probs_and_entropy(
            local_logits,
            tokens,
            dist.group.WORLD,
            with_entropy=True,
            entropy_requires_grad=True,
            chunk_size=chunk_size,
            true_on_policy=True,
            vocab_size=real_vocab_size,
            sampling_mask=None if full_mask is None else full_mask[:, shard],
            temperature=1.0,
        )
        actual = _values_and_separate_gradients(local_logits, log_probs, entropy, weights)
        if baseline is None:
            baseline = actual
        for name, value in actual.items():
            errors = torch.stack((_reduction_error(value, baseline[name]), _reduction_error(value, reference[name])))
            dist.all_reduce(errors, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(
                    f"TP2 reduction_error dtype={dtype} masked={masked} chunk_size={chunk_size} quantity={name} "
                    f"vs_unchunked(max_abs={errors[0, 0].item():.8e}, max_row_rel_l2={errors[0, 1].item():.8e}) "
                    f"vs_fp64(max_abs={errors[1, 0].item():.8e}, max_row_rel_l2={errors[1, 1].item():.8e})",
                    flush=True,
                )
            assert torch.isfinite(value).all(), (dtype, masked, chunk_size, name)
            # FP64 errors characterize the existing dtype path; chunking must add none.
            torch.testing.assert_close(value, baseline[name], rtol=0, atol=0)
            if name.endswith("gradient") and rank == _WORLD_SIZE - 1:
                assert torch.count_nonzero(value[:, real_vocab_size - rank * local_vocab_size :]).item() == 0


def _low_precision_worker(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Spawn entry point. Initializes the group and runs the low-precision
    # check for bfloat16 and float16, each unmasked and masked; barriers; always destroys the group.
    device = _init_worker(rank, world_size, port)
    try:
        for dtype in (torch.bfloat16, torch.float16):
            for masked in (False, True):
                _assert_low_precision_reductions(rank, device, dtype, masked)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _memory_args(chunk_size: int, vocab_size: int) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds the minimal training-arguments namespace for the log-prob
    # wrapper: packed (thd) format, rollout temperature 1, true-on-policy on, bfloat16 on and float16
    # off, the given log-prob chunk size and vocabulary size, no context-parallel all-gather, and the
    # fused log-prob debug option off.
    return Namespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=True,
        bf16=True,
        fp16=False,
        log_probs_chunk_size=chunk_size,
        vocab_size=vocab_size,
        allgather_cp=False,
        debug_unified_grad_fused_logprob=False,
    )


def _run_memory_forward(
    device: torch.device,
    *,
    rows: int,
    local_padded_vocab_size: int,
    real_vocab_size: int,
    chunk_size: int,
) -> tuple[int, torch.Tensor, dict[str, list[torch.Tensor]]]:
    # [FUNCTION DOC] HELPER: Measures peak extra GPU memory of the log-prob/entropy wrapper for one
    # sample of rows + 1 tokens (rows response tokens). Collects garbage and empties the CUDA cache,
    # builds zero bfloat16 logits (1, rows + 1, local padded vocabulary) requiring grad and tokens
    # counting up modulo the real vocabulary, resets peak statistics, records baseline allocation, calls
    # the wrapper with entropy and entropy gradients enabled, synchronizes, and returns (peak allocated
    # minus baseline in bytes, the logits tensor, the result mapping).
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    total_length = rows + 1
    logits = torch.zeros(
        (1, total_length, local_padded_vocab_size),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    tokens = torch.arange(total_length, device=device, dtype=torch.long).remainder_(real_vocab_size)
    args = _memory_args(chunk_size, real_vocab_size)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    result = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=[tokens],
        total_lengths=[total_length],
        response_lengths=[rows],
        with_entropy=True,
        entropy_requires_grad=True,
    )
    torch.cuda.synchronize(device)
    peak_delta = torch.cuda.max_memory_allocated(device) - baseline
    return peak_delta, logits, result


def _release_cuda_memory() -> None:
    # [FUNCTION DOC] HELPER: Runs garbage collection, empties the CUDA allocator cache and
    # synchronizes the current device.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _warm_up_memory_path(device: torch.device) -> None:
    # [FUNCTION DOC] HELPER: Runs the memory forward on a tiny problem (17 rows, local vocabulary
    # 128, real vocabulary 255) for chunk sizes -1 and 4, releasing memory after each, so one-time
    # allocations (kernels, workspace) do not pollute the real measurements.
    for chunk_size in (-1, 4):
        _, logits, result = _run_memory_forward(
            device,
            rows=17,
            local_padded_vocab_size=128,
            real_vocab_size=255,
            chunk_size=chunk_size,
        )
        del result
        del logits
        _release_cuda_memory()


def _measure_memory_case(device: torch.device, chunk_size: int) -> int:
    # [FUNCTION DOC] HELPER: Runs the memory forward at full size (8192 rows, local vocabulary
    # 76032, real vocabulary 151936) with the chunk size; asserts the log-prob and entropy outputs
    # have shape (8192,) and are finite; deletes all references, releases CUDA memory, and returns the
    # measured peak delta in bytes.
    peak_delta, logits, result = _run_memory_forward(
        device,
        rows=_MEMORY_ROWS,
        local_padded_vocab_size=_MEMORY_LOCAL_PADDED_VOCAB_SIZE,
        real_vocab_size=_MEMORY_REAL_VOCAB_SIZE,
        chunk_size=chunk_size,
    )
    log_probs = result["log_probs"][0]
    entropy = result["entropy"][0]
    assert log_probs.shape == (_MEMORY_ROWS,)
    assert entropy.shape == (_MEMORY_ROWS,)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()
    del entropy
    del log_probs
    del result
    del logits
    _release_cuda_memory()
    return peak_delta


def _memory_worker(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Spawn entry point for the memory test. Initializes the group and
    # installs a global parallel state with the tensor-parallel group info (rank, size, world group) and
    # a trivial context-parallel info. Warms up, measures the peak for chunk size 1024 then unchunked,
    # all-gathers each rank's (unchunked, chunked) peaks, and asserts on every rank that the saving
    # (unchunked minus chunked) is at least half of one full-size bfloat16 buffer of (8192 - 1024) rows
    # times the real vocabulary size (message lists peaks and savings). Rank 0 prints a summary.
    # Barriers and always destroys the group.
    # KNOWN ISSUE: the global parallel state set here is never restored (harmless in a spawned process).
    device = _init_worker(rank, world_size, port)
    try:
        tp = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD)
        cp = GroupInfo(rank=0, size=1, group=None)
        set_parallel_state(SimpleNamespace(tp=tp, cp=cp))

        _warm_up_memory_path(device)
        chunked_peak = _measure_memory_case(device, _MEMORY_CHUNK_SIZE)
        unchunked_peak = _measure_memory_case(device, -1)

        local_peaks = torch.tensor([unchunked_peak, chunked_peak], device=device, dtype=torch.int64)
        gathered_peaks = [torch.empty_like(local_peaks) for _ in range(world_size)]
        dist.all_gather(gathered_peaks, local_peaks)
        peaks_by_rank = [tuple(int(value) for value in peaks.cpu().tolist()) for peaks in gathered_peaks]

        bf16_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
        one_buffer_saving = (_MEMORY_ROWS - _MEMORY_CHUNK_SIZE) * _MEMORY_REAL_VOCAB_SIZE * bf16_bytes
        minimum_saving = one_buffer_saving // 2
        savings = [unchunked - chunked for unchunked, chunked in peaks_by_rank]
        assert min(savings) >= minimum_saving, (
            f"expected at least {minimum_saving} bytes of true-on-policy peak-memory saving on every TP rank; "
            f"peaks={peaks_by_rank}, savings={savings}"
        )
        if rank == 0:
            print(
                "true-on-policy TP2 peak memory: "
                f"peaks_by_rank={peaks_by_rank}, savings={savings}, minimum_saving={minimum_saving}",
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_true_on_policy_chunking_tp2_correctness_and_gradients() -> None:
    # [FUNCTION DOC] Raises a runtime error if fewer than 2 GPUs are visible; otherwise spawns 2
    # worker processes running the correctness worker on a free port and joins them.
    if torch.cuda.device_count() < _WORLD_SIZE:
        raise RuntimeError(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")
    mp.spawn(_correctness_worker, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)


def test_true_on_policy_chunking_tp2_low_precision_reduction_errors() -> None:
    # [FUNCTION DOC] Same GPU-count check; spawns 2 processes running the low-precision worker.
    if torch.cuda.device_count() < _WORLD_SIZE:
        raise RuntimeError(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")
    mp.spawn(_low_precision_worker, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)


def test_true_on_policy_chunking_reduces_8192_row_peak_memory() -> None:
    # [FUNCTION DOC] Same GPU-count check; spawns 2 processes running the memory worker.
    if torch.cuda.device_count() < _WORLD_SIZE:
        raise RuntimeError(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")
    mp.spawn(_memory_worker, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
