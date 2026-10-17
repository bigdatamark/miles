# [MODULE DOC] Triton backward pass for causal self-attention (multi-head, optional grouped-query
# sharing of key/value heads), computed tile by tile without materializing the sequence-by-sequence
# score matrix. The forward kernel does not save the softmax normalizer, so the backward first
# recomputes the per-query log-sum-exp (LSE) and the per-query delta (row sum of output times output
# gradient), then recomputes attention probabilities exactly as exp(score - LSE) inside the
# gradient kernels. Gradients for keys/values and for queries are produced by two separate kernels
# (one owns key blocks, the other owns query blocks), so no atomic additions are needed.
# The driver, dense_attention_backward, processes each sequence of a batch separately on fixed-size
# sequences and expands/reduces key and value heads for grouped-query attention.
# Imports sglang's HIP detection to choose AMD-specific launch options. All arithmetic on scores is
# float32; matrix products use the input dtype.
# KNOWN ISSUE: the text below mentions per-sequence processing "via indptrs" (variable-length
# batching), but nothing here takes indptrs; the driver only supports a batch of equal-length
# sequences (batch size and sequence length). It also lists three kernels while four are defined.
"""Triton backward kernels for causal self-attention.

Memory-efficient backward pass using tiled computation (no S*S materialization).
Adapted from Tri Dao's flash_attn_triton.py and the Triton tutorial 06-fused-attention,
with the following simplifications for our training use case:
  - No bias, no dropout
  - Causal masking only
  - GQA support (kv_group_num)
  - Per-sequence processing via indptrs (variable-length batching)

Three kernels:
  1. _bwd_preprocess: computes delta = rowsum(O * dO) per query position
  2. _compute_lse: computes LSE (log-sum-exp) for each query position using online
     accumulation across K blocks. This is needed because the forward kernel doesn't
     save LSE, but we need it to correctly recompute softmax in backward.
  3. _bwd_kernel_dk_dv / _bwd_kernel_dq: compute dQ, dK, dV using LSE to recompute
     globally correct attention weights: P_ij = exp(q_i @ k_j * scale - LSE_i)

Two-pass design for dK/dV and dQ (no atomics needed):
  - _bwd_kernel_dk_dv: each program owns one K/V block, iterates over Q blocks
  - _bwd_kernel_dq: each program owns one Q block, iterates over K/V blocks

Reference: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
"""

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hip

_is_hip = is_hip()


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_obs,
    stride_oh,
    stride_dobs,
    stride_doh,
    seqlen,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel computing delta for one sequence. Grid: query tiles of
    # BLOCK_M by heads. Delta, laid out heads by sequence length (float32), receives for every query
    # position the sum over the (padded to BLOCK_HEADDIM, masked to headdim) channels of output times
    # output gradient, both loaded as float32. Strides are given for the sequence axis and head axis
    # of each tensor; the channel axis must be contiguous.
    """Compute delta_i = sum_d(O_i * dO_i) for each query position.

    Grid: (cdiv(seqlen, BLOCK_M), num_heads)
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    mask_m = offs_m < seqlen
    mask_d = offs_d < headdim

    o_ptrs = Out + pid_h * stride_oh + offs_m[:, None] * stride_obs + offs_d[None, :]
    do_ptrs = DO + pid_h * stride_doh + offs_m[:, None] * stride_dobs + offs_d[None, :]

    o = tl.load(o_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + pid_h * seqlen + offs_m, delta, mask=mask_m)


@triton.jit
def _compute_lse(
    Q,
    K,
    LSE,
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    seqlen,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel producing the causal log-sum-exp per query, written to
    # LSE (heads by sequence length, float32). One program per query tile and head. Loads the query
    # tile, then loops over key tiles from position 0 up to the end of the causal range of the tile
    # (min of the sequence length and the tile end), computing scores = q dot k transposed times
    # sm_scale, masking positions where the key index exceeds the query index or is out of range with
    # minus infinity, and updating a running maximum and running sum (online softmax with rescaling
    # of the old sum). The result is maximum + log(sum); a zero sum is replaced by one before the log
    # to avoid log of zero (rows with no valid key then get minus infinity). Key head index equals
    # the program's head index (grouped-query keys are expanded beforehand).
    """Compute LSE (log-sum-exp) for each query position using online accumulation.

    LSE_i = log(sum_{j: j<=i} exp(q_i @ k_j * scale))

    Grid: (cdiv(seqlen, BLOCK_M), num_heads)
    Each program handles BLOCK_M query positions and iterates over all valid K blocks.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    mask_m = offs_m < seqlen
    mask_d = offs_d < headdim

    # Load Q block
    q_ptrs = Q + pid_h * stride_qh + offs_m[:, None] * stride_qbs + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Online accumulation of max and sum for LSE
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Causal: Q at position i can attend to K at positions 0..i
    end_n_causal = tl.minimum(seqlen, (pid_m + 1) * BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, end_n_causal, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < seqlen

        # Load K block
        k_ptrs = K + pid_h * stride_kh + offs_n_curr[:, None] * stride_kbs + offs_d[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # qk = Q @ K^T * scale
        qk = tl.dot(q.to(k.dtype), tl.trans(k))
        qk *= sm_scale

        # Causal mask
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(causal_mask & mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        # Online softmax update
        block_max = tl.max(qk, axis=1)
        new_m = tl.maximum(m_i, block_max)
        # Rescale old sum and add new sum
        l_i = l_i * tl.exp(m_i - new_m) + tl.sum(tl.exp(qk - new_m[:, None]), axis=1)
        m_i = new_m

    # LSE = m + log(l)
    # Handle rows where m_i is -inf (no valid K positions - shouldn't happen for causal with valid Q)
    lse = m_i + tl.log(tl.where(l_i == 0.0, 1.0, l_i))
    tl.store(LSE + pid_h * seqlen + offs_m, lse, mask=mask_m)


@triton.jit
def _bwd_kernel_dk_dv(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    LSE,
    Delta,
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_dobs,
    stride_doh,
    stride_dkbs,
    stride_dkh,
    stride_dvbs,
    stride_dvh,
    seqlen,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel producing key and value gradients for one sequence. One
    # program per key tile and head; holds float32 accumulators for dK and dV of the tile. Loops over
    # query tiles starting at the first query tile that can see the key tile (begin index rounded
    # down to a multiple of BLOCK_M) to the end of the sequence. For each query tile: recompute
    # scores times sm_scale with the causal/range mask, probabilities p = exp(score - LSE of the
    # query row), accumulate dV += p transposed times dO, compute dp = dO times V transposed,
    # ds = p times (dp - delta) times sm_scale, and accumulate dK += ds transposed times q (ds cast to
    # the query dtype, p cast to the dO dtype). Stores dK and dV cast to the key/value dtype, masked
    # to valid rows and channels. LSE and delta are read per head and query (other value 0 for
    # masked rows, which then produce zero probability because their scores are minus infinity).
    """Compute dK and dV for one K/V block.

    Grid: (cdiv(seqlen, BLOCK_N), num_heads)

    Each program owns one K/V block, iterates over Q blocks, accumulates dK/dV in registers.
    Uses LSE for globally correct softmax recomputation: P_ij = exp(s_ij - LSE_i).
    """
    start_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    mask_n = offs_n < seqlen
    mask_d = offs_d < headdim

    # Load K, V for this block (stay in SRAM)
    k_ptrs = K + pid_h * stride_kh + offs_n[:, None] * stride_kbs + offs_d[None, :]
    v_ptrs = V + pid_h * stride_vh + offs_n[:, None] * stride_vbs + offs_d[None, :]
    k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    # Causal: only Q positions >= first K position in this block can attend here
    begin_m = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    offs_m = tl.arange(0, BLOCK_M)

    num_block_m = tl.cdiv(seqlen, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        mask_m = offs_m_curr < seqlen

        # Load Q
        q_ptrs = Q + pid_h * stride_qh + offs_m_curr[:, None] * stride_qbs + offs_d[None, :]
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

        # Recompute qk = Q @ K^T * scale
        qk = tl.dot(q.to(k.dtype), tl.trans(k))
        qk *= sm_scale

        # Causal mask
        causal_mask = offs_m_curr[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask & mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        # Correct softmax using LSE: p = exp(qk - LSE_i)
        lse_i = tl.load(LSE + pid_h * seqlen + offs_m_curr, mask=mask_m, other=0.0)
        p = tl.exp(qk - lse_i[:, None])

        # Load dO
        do_ptrs = DO + pid_h * stride_doh + offs_m_curr[:, None] * stride_dobs + offs_d[None, :]
        do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

        # dV += P^T @ dO
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        # dp = dO @ V^T
        dp = tl.dot(do, tl.trans(v))

        # ds = P * (dp - delta) * scale
        delta_i = tl.load(Delta + pid_h * seqlen + offs_m_curr, mask=mask_m, other=0.0)
        ds = p * (dp - delta_i[:, None]) * sm_scale

        # dK += ds^T @ Q
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    # Store dK, dV
    dk_ptrs = DK + pid_h * stride_dkh + offs_n[:, None] * stride_dkbs + offs_d[None, :]
    dv_ptrs = DV + pid_h * stride_dvh + offs_n[:, None] * stride_dvbs + offs_d[None, :]
    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None] & mask_d[None, :])
    tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_n[:, None] & mask_d[None, :])


@triton.jit
def _bwd_kernel_dq(
    Q,
    K,
    V,
    DO,
    DQ,
    LSE,
    Delta,
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_dobs,
    stride_doh,
    stride_dqbs,
    stride_dqh,
    seqlen,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel producing query gradients for one sequence. One program
    # per query tile and head, keeping the query tile, dO, LSE and delta resident and a float32 dQ
    # accumulator. Loops over key/value tiles from position 0 to the causal end of the query tile;
    # per tile recomputes the masked scaled scores, probabilities exp(score - LSE), dp = dO times V
    # transposed, ds = p times (dp - delta) times sm_scale, and accumulates dQ += ds (cast to the key
    # dtype) times K. Stores dQ cast to the query dtype, masked to valid rows and channels.
    """Compute dQ for one Q block.

    Grid: (cdiv(seqlen, BLOCK_M), num_heads)

    Each program owns one Q block, iterates over K/V blocks, accumulates dQ in registers.
    Uses LSE for globally correct softmax recomputation.
    """
    start_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    mask_m = offs_m < seqlen
    mask_d = offs_d < headdim

    # Load Q, dO for this block (stay in SRAM)
    q_ptrs = Q + pid_h * stride_qh + offs_m[:, None] * stride_qbs + offs_d[None, :]
    do_ptrs = DO + pid_h * stride_doh + offs_m[:, None] * stride_dobs + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    lse_i = tl.load(LSE + pid_h * seqlen + offs_m, mask=mask_m, other=0.0)
    delta_i = tl.load(Delta + pid_h * seqlen + offs_m, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Causal: this Q block can only attend to K positions <= last Q position in block
    end_n_causal = tl.minimum(seqlen, (start_m + 1) * BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, end_n_causal, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < seqlen

        # Load K, V
        k_ptrs = K + pid_h * stride_kh + offs_n_curr[:, None] * stride_kbs + offs_d[None, :]
        v_ptrs = V + pid_h * stride_vh + offs_n_curr[:, None] * stride_vbs + offs_d[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Recompute qk
        qk = tl.dot(q.to(k.dtype), tl.trans(k))
        qk *= sm_scale

        # Causal mask
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(causal_mask & mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        # Correct softmax using LSE: p = exp(qk - LSE_i)
        p = tl.exp(qk - lse_i[:, None])

        # dp = dO @ V^T
        dp = tl.dot(do, tl.trans(v))

        # ds = P * (dp - delta) * scale
        ds = p * (dp - delta_i[:, None]) * sm_scale

        # dQ += ds @ K
        dq += tl.dot(ds.to(k.dtype), k)

    # Store dQ
    dq_ptrs = DQ + pid_h * stride_dqh + offs_m[:, None] * stride_dqbs + offs_d[None, :]
    tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_m[:, None] & mask_d[None, :])


def dense_attention_backward(q, k, v, o, do, B, S, sm_scale=None):
    # [FUNCTION DOC] Computes gradients (dq, dk, dv) of causal attention for B sequences of equal
    # length S packed along the first axis (shape B*S by heads by head dim; query heads must be a
    # multiple of key/value heads). o is the forward output and do the output gradient. The softmax
    # scale defaults to one over the square root of the head dim; note a falsy scale (zero) also
    # falls back to the default. Steps: pick block sizes (64 by 64 on all platforms, head dim
    # rounded up to a power of two); for grouped-query attention expand K and V to the number of
    # query heads by repeating each key/value head consecutively for its group of query heads;
    # allocate outputs; then for each sequence in a Python loop run four kernels: delta, LSE, dK/dV
    # (4 warps, 1 stage) and dQ (4 warps, 1 stage), with an AMD-only launch option of one wave per
    # execution unit; copy the per-sequence results into the full outputs. Finally, for grouped-query
    # attention, sum the expanded key/value gradients over each group and cast to the key/value dtype.
    # Returns (dq, dk, dv) shaped like q, k, v. Allocates temporary expanded tensors and per-
    # sequence buffers; GPU launches only, no other side effects.
    # KNOWN ISSUE: the block-size selection uses the same value for HIP and non-HIP, so the
    # conditional is redundant; the per-sequence dq buffer and the full dq are both zero-initialized
    # although every element is overwritten.
    """Triton backward for causal self-attention (two-pass with LSE).

    Args:
        q: [B*S, num_heads, D], bf16
        k: [B*S, num_kv_heads, D], bf16
        v: [B*S, num_kv_heads, D], bf16
        o: [B*S, num_heads, D], bf16 (forward output)
        do: [B*S, num_heads, D], bf16 (grad of output)
        B: batch size
        S: sequence length
        sm_scale: softmax scale (default 1/sqrt(D))

    Returns:
        dq, dk, dv with same shapes as q, k, v
    """
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    D = q.shape[2]
    kv_group_num = num_heads // num_kv_heads

    sm_scale = sm_scale or 1.0 / (D**0.5)

    BLOCK_HEADDIM = triton.next_power_of_2(D)
    BLOCK_M = 64 if _is_hip else 64
    BLOCK_N = 64 if _is_hip else 64

    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1}

    # For GQA: expand K, V to match num_heads
    if kv_group_num > 1:
        k_exp = (
            k.view(B, S, num_kv_heads, D)
            .unsqueeze(3)
            .expand(B, S, num_kv_heads, kv_group_num, D)
            .reshape(B * S, num_heads, D)
            .contiguous()
        )
        v_exp = (
            v.view(B, S, num_kv_heads, D)
            .unsqueeze(3)
            .expand(B, S, num_kv_heads, kv_group_num, D)
            .reshape(B * S, num_heads, D)
            .contiguous()
        )
    else:
        k_exp = k
        v_exp = v

    dq_all = torch.zeros_like(q)
    dk_exp_all = torch.empty_like(k_exp)
    dv_exp_all = torch.empty_like(v_exp)

    # Process each sequence independently
    for b in range(B):
        s, e = b * S, (b + 1) * S

        q_b = q[s:e]
        k_b = k_exp[s:e]
        v_b = v_exp[s:e]
        o_b = o[s:e]
        do_b = do[s:e]

        dq_b = torch.zeros(S, num_heads, D, device=q.device, dtype=q.dtype)
        dk_b = torch.empty(S, num_heads, D, device=q.device, dtype=q.dtype)
        dv_b = torch.empty(S, num_heads, D, device=q.device, dtype=q.dtype)
        delta = torch.empty(num_heads, S, device=q.device, dtype=torch.float32)
        lse = torch.empty(num_heads, S, device=q.device, dtype=torch.float32)

        # Step 1: preprocess delta
        grid_pre = (triton.cdiv(S, BLOCK_M), num_heads)
        _bwd_preprocess[grid_pre](
            o_b,
            do_b,
            delta,
            o_b.stride(0),
            o_b.stride(1),
            do_b.stride(0),
            do_b.stride(1),
            S,
            D,
            BLOCK_M=BLOCK_M,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            **extra_kargs,
        )

        # Step 2: compute LSE
        grid_lse = (triton.cdiv(S, BLOCK_M), num_heads)
        _compute_lse[grid_lse](
            q_b,
            k_b,
            lse,
            sm_scale,
            q_b.stride(0),
            q_b.stride(1),
            k_b.stride(0),
            k_b.stride(1),
            S,
            D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            num_warps=4,
            num_stages=1,
            **extra_kargs,
        )

        # Step 3: compute dK, dV
        grid_kv = (triton.cdiv(S, BLOCK_N), num_heads)
        _bwd_kernel_dk_dv[grid_kv](
            q_b,
            k_b,
            v_b,
            do_b,
            dk_b,
            dv_b,
            lse,
            delta,
            sm_scale,
            q_b.stride(0),
            q_b.stride(1),
            k_b.stride(0),
            k_b.stride(1),
            v_b.stride(0),
            v_b.stride(1),
            do_b.stride(0),
            do_b.stride(1),
            dk_b.stride(0),
            dk_b.stride(1),
            dv_b.stride(0),
            dv_b.stride(1),
            S,
            D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            num_warps=4,
            num_stages=1,
            **extra_kargs,
        )

        # Step 4: compute dQ
        grid_q = (triton.cdiv(S, BLOCK_M), num_heads)
        _bwd_kernel_dq[grid_q](
            q_b,
            k_b,
            v_b,
            do_b,
            dq_b,
            lse,
            delta,
            sm_scale,
            q_b.stride(0),
            q_b.stride(1),
            k_b.stride(0),
            k_b.stride(1),
            v_b.stride(0),
            v_b.stride(1),
            do_b.stride(0),
            do_b.stride(1),
            dq_b.stride(0),
            dq_b.stride(1),
            S,
            D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            num_warps=4,
            num_stages=1,
            **extra_kargs,
        )

        dq_all[s:e] = dq_b
        dk_exp_all[s:e] = dk_b
        dv_exp_all[s:e] = dv_b

    # GQA: reduce gradients back to kv_heads
    if kv_group_num > 1:
        dk_all = dk_exp_all.view(B * S, num_kv_heads, kv_group_num, D).sum(dim=2).to(k.dtype)
        dv_all = dv_exp_all.view(B * S, num_kv_heads, kv_group_num, D).sum(dim=2).to(v.dtype)
    else:
        dk_all = dk_exp_all
        dv_all = dv_exp_all

    return dq_all, dk_all, dv_all
