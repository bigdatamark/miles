# [MODULE DOC] Fused forward/backward for the per-layer-embedding (PLE) increment computed from a
# multi-stream hidden state. Tokens are rows; the hidden state has n streams of C channels (row width
# W = n times C). Pipeline per token and stream (all in float32, one cast to the output dtype):
# (1) gate chain: RMS-normalize the key row and the hidden-state ("query") row of the stream (each
# with its own weight in "one plus weight" form), take the dot product of the two normalized vectors
# divided by sqrt(C), turn it into a gate in (0,1) with sigmoid of (sign times square root of the
# magnitude, magnitude floored at 1e-6), and scale the per-token value row (C channels, shared by
# all streams) by that gate: "gated"; (2) normalize gated per stream with a grouped RMS norm
# (separate weight); (3) causal depthwise convolution over tokens (K taps, with a dilation, never
# crossing sequence boundaries given by cumulative lengths) of the normalized signal, then SiLU;
# (4) output = gated + SiLU(conv) (residual). Backward recomputes the gate chain and the norm from
# the saved inputs instead of storing them, then runs the conv backward (weight gradient with atomic
# additions), the grouped-norm backward and the gate-chain backward; per-token partial weight
# gradients are summed on the host side with torch. Public entry point: ple_gate_conv.
# Token indices are widened to 64-bit before multiplying by the row width, including the neighbors
# used by the convolution.
"""Triton PLE kernels: fused gate chain and segment-aware causal depthwise conv.

fp32 for every reduction and elementwise step, one cast onto the output dtype.
Token indices use int64 before width multiplication, including convolution neighbors.
"""

import math

import torch
import triton
import triton.language as tl

from miles.kernels.norm.grouped_rmsnorm import block_c, grouped_rmsnorm_bwd_kernel, grouped_rmsnorm_fwd


@triton.jit(do_not_specialize=["T"])
def _ple_gate_fwd_kernel(
    key_ptr,  # [T, n*C] (any float dtype)
    query_ptr,  # [T, n*C] hc_state
    value_ptr,  # [T, C]
    wk_ptr,
    wq_ptr,  # norm weights [n*C]
    gated_ptr,  # fp32 out [T, n*C]
    gate_ptr,  # fp32 out [T, n]
    rstdk_ptr,
    rstdq_ptr,  # fp32 out [T, n]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    SQRTC: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel; one program per (token, stream) pair (grid of T times N;
    # program id divided by N is the token, remainder the stream; surplus programs exit). Loads the
    # stream's key and query rows and the stream's slices of the key and query norm weights, in
    # float32. Computes reciprocal root-mean-square for each (epsilon EPS), normalized-and-weighted
    # vectors (times one plus weight), score = dot product divided by SQRTC (square root of C),
    # magnitude = max of absolute score and 1e-6, gate = sigmoid of (sign of score, where zero counts
    # as positive, times square root of magnitude). Stores: gated = gate times the token's value row
    # (shared across streams) into a float32 token-by-W buffer; the gate and both reciprocal RMS
    # values into float32 token-by-N buffers.
    pid = tl.program_id(0).to(tl.int64)
    t = pid // N
    c = pid % N
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    base = t * (N * C) + c * C

    k = tl.load(key_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(query_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    wk = tl.load(wk_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    wq = tl.load(wq_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)

    rk = 1.0 / tl.sqrt(tl.sum(k * k, axis=0) / C + EPS)
    rq = 1.0 / tl.sqrt(tl.sum(q * q, axis=0) / C + EPS)
    kn = k * rk * (1.0 + wk)
    qn = q * rq * (1.0 + wq)

    score = tl.sum(kn * qn, axis=0) / SQRTC
    mag = tl.maximum(tl.abs(score), 1e-6)
    sgn = tl.where(score >= 0, 1.0, -1.0)
    gate = tl.sigmoid(sgn * tl.sqrt(mag))

    v = tl.load(value_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(gated_ptr + base + offs, gate * v, mask=mask)
    tl.store(gate_ptr + t * N + c, gate)
    tl.store(rstdk_ptr + t * N + c, rk)
    tl.store(rstdq_ptr + t * N + c, rq)


@triton.jit(do_not_specialize=["T"])
def _ple_gate_bwd_kernel(
    dgated_ptr,  # fp32 in [T, n*C]
    key_ptr,
    query_ptr,
    value_ptr,
    wk_ptr,
    wq_ptr,
    gate_ptr,
    rstdk_ptr,
    rstdq_ptr,
    dkey_ptr,  # out, key dtype [T, n*C]
    dquery_ptr,  # out, query dtype [T, n*C]
    dvalue_ptr,  # fp32 out [T, n, C] (summed over n by the host)
    dwk_partial_ptr,  # fp32 out [T, n*C] (dnormed_k * k_hat; host sums over T)
    dwq_partial_ptr,
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    SQRTC: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton backward of the gate chain; same one-program-per-(token, stream)
    # layout. Inputs: float32 gradient of gated, the original key / query / value, norm weights, and
    # the saved gate and reciprocal RMS values. Steps: gradient of the value row per stream =
    # incoming gradient times gate (stored per stream, summed over streams on the host); gradient of
    # the gate = sum of incoming gradient times value; through the sigmoid; through the signed square
    # root (derivative one over two times square root of magnitude, set to zero where the absolute
    # score is not above 1e-6); gradients of the normalized key and query are the other normalized
    # vector times that gradient divided by sqrt(C). Stores per-token partial weight gradients
    # (gradient of normalized vector times its unweighted normalized input; summed over tokens on
    # the host) and finally the RMS-norm backward for key and query (reciprocal RMS times gradient
    # minus input times its cube times dot-product over C), cast to the key / query dtypes.
    pid = tl.program_id(0).to(tl.int64)
    t = pid // N
    c = pid % N
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    base = t * (N * C) + c * C
    sqrtC = SQRTC

    k = tl.load(key_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(query_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(value_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
    wk = tl.load(wk_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    wq = tl.load(wq_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + t * N + c)
    rk = tl.load(rstdk_ptr + t * N + c)
    rq = tl.load(rstdq_ptr + t * N + c)
    dg_out = tl.load(dgated_ptr + base + offs, mask=mask, other=0.0)

    kn = k * rk * (1.0 + wk)
    qn = q * rq * (1.0 + wq)

    dgate = tl.sum(dg_out * v, axis=0)
    tl.store(dvalue_ptr + (t * N + c) * C + offs, dg_out * g, mask=mask)

    score = tl.sum(kn * qn, axis=0) / sqrtC
    mag = tl.maximum(tl.abs(score), 1e-6)
    du = dgate * g * (1.0 - g)
    ds = tl.where(tl.abs(score) > 1e-6, du / (2.0 * tl.sqrt(mag)), 0.0)

    dkn = qn * (ds / sqrtC)
    dqn = kn * (ds / sqrtC)

    tl.store(dwk_partial_ptr + base + offs, dkn * (k * rk), mask=mask)
    tl.store(dwq_partial_ptr + base + offs, dqn * (q * rq), mask=mask)

    gk = dkn * (1.0 + wk)
    dotk = tl.sum(gk * k, axis=0)
    dk = rk * gk - k * (rk * rk * rk) * (dotk / C)
    gq = dqn * (1.0 + wq)
    dotq = tl.sum(gq * q, axis=0)
    dq = rq * gq - q * (rq * rq * rq) * (dotq / C)

    tl.store(dkey_ptr + base + offs, dk.to(dkey_ptr.dtype.element_ty), mask=mask)
    tl.store(dquery_ptr + base + offs, dq.to(dquery_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T", "W"])
def _ple_conv_fwd_kernel(
    normed_ptr,  # fp32 [T, W]
    gated_ptr,  # fp32 [T, W]
    convw_ptr,  # [W, K] (weight[:, 0, :] contiguous)
    segstart_ptr,  # int32 [T]
    out_ptr,  # out dtype [T, W]
    conv_ptr,  # fp32 out [T, W]  pre-SiLU conv result (saved for bwd)
    T,
    W,
    K: tl.constexpr,
    DIL: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton forward of the convolution stage; grid of tokens by channel
    # blocks of BLOCK_W. For token t and each channel computes the causal dilated depthwise
    # convolution over the K taps: tap j reads the normalized signal at token t minus (K-1-j) times
    # DIL, only if that token index is non-negative and not before the token's sequence start
    # (otherwise contributes zero), weighted by the channel's tap weight (weights laid out channel by
    # tap). Stores the pre-activation result (float32, kept for backward), then output = gated +
    # SiLU(result), cast to the output dtype. W is the full row width, K and DIL are compile-time.
    t = tl.program_id(0).to(tl.int64)
    wb = tl.program_id(1)
    if t >= T:
        return
    offs = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = offs < W
    seg_lo = tl.load(segstart_ptr + t)

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    for j in tl.static_range(K):
        src = t - (K - 1 - j) * DIL
        wgt = tl.load(convw_ptr + offs * K + j, mask=mask, other=0.0).to(tl.float32)
        if src >= 0:
            ok = src >= seg_lo
            x = tl.load(normed_ptr + src * W + offs, mask=mask & ok, other=0.0)
            acc += wgt * x
    tl.store(conv_ptr + t * W + offs, acc, mask=mask)
    silu = acc * tl.sigmoid(acc)
    gt = tl.load(gated_ptr + t * W + offs, mask=mask, other=0.0)
    tl.store(out_ptr + t * W + offs, (gt + silu).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T", "W"])
def _ple_conv_bwd_kernel(
    dout_ptr,  # incoming grad [T, W] (any float dtype)
    conv_ptr,  # fp32 [T, W] pre-SiLU
    normed_ptr,  # fp32 [T, W]
    convw_ptr,  # [W, K]
    segstart_ptr,
    segend_ptr,  # int32 [T] (exclusive)
    dnormed_ptr,  # fp32 out [T, W]
    dconvw_ptr,  # fp32 out [W, K] via atomics
    dgated_add_ptr,  # fp32 out [T, W]  (dout passthrough for the residual)
    T,
    W,
    K: tl.constexpr,
    DIL: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton backward of the convolution stage; same grid as the forward.
    # For its token and channel block: copies the incoming gradient (as float32) to the residual
    # gradient output; computes the gradient at the convolution result through SiLU (using the saved
    # pre-activation); for every tap with a valid source token (non-negative and not before the
    # sequence start) atomically adds gradient times source value into the convolution weight
    # gradient (float32, channel by tap, must be zero-initialized); and computes the gradient of the
    # normalized signal at this token as the sum over taps of tap weight times the SiLU-backward
    # gradient at the destination token that read it (destination = t plus (K-1-j) times DIL,
    # included only if inside the token range and before this token's sequence end).
    t = tl.program_id(0).to(tl.int64)
    wb = tl.program_id(1)
    if t >= T:
        return
    offs = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = offs < W
    seg_lo = tl.load(segstart_ptr + t)
    seg_hi = tl.load(segend_ptr + t)

    do = tl.load(dout_ptr + t * W + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(dgated_add_ptr + t * W + offs, do, mask=mask)

    cv = tl.load(conv_ptr + t * W + offs, mask=mask, other=0.0)
    sig = tl.sigmoid(cv)
    dconv = do * sig * (1.0 + cv * (1.0 - sig))

    for j in tl.static_range(K):
        src = t - (K - 1 - j) * DIL
        if src >= 0:
            ok = src >= seg_lo
            x = tl.load(normed_ptr + src * W + offs, mask=mask & ok, other=0.0)
            tl.atomic_add(dconvw_ptr + offs * K + j, dconv * x, mask=mask & ok)

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    for j in tl.static_range(K):
        dst = t + (K - 1 - j) * DIL
        wgt = tl.load(convw_ptr + offs * K + j, mask=mask, other=0.0).to(tl.float32)
        if dst < T:
            ok = dst < seg_hi
            do2 = tl.load(dout_ptr + dst * W + offs, mask=mask & ok, other=0.0).to(tl.float32)
            cv2 = tl.load(conv_ptr + dst * W + offs, mask=mask & ok, other=0.0)
            sig2 = tl.sigmoid(cv2)
            acc += wgt * do2 * sig2 * (1.0 + cv2 * (1.0 - sig2))
    tl.store(dnormed_ptr + t * W + offs, acc, mask=mask)


def _seg_bounds(T: int, cu_seqlens, device):
    # [FUNCTION DOC] HELPER: Builds per-token sequence boundaries as two int32 tensors of length T
    # on the device: the first token index of the token's sequence and one past its last. Without
    # cumulative lengths the whole range is one sequence (0 and T). Otherwise each boundary pair is
    # repeated for the length of its sequence (negative lengths clamped to zero). The lengths are
    # expected to sum to T.
    if cu_seqlens is None:
        lo = torch.zeros(T, dtype=torch.int32, device=device)
        hi = torch.full((T,), T, dtype=torch.int32, device=device)
        return lo, hi
    cu = cu_seqlens.to(torch.long)
    lens = (cu[1:] - cu[:-1]).clamp_min(0)
    lo = torch.repeat_interleave(cu[:-1], lens).to(torch.int32)
    hi = torch.repeat_interleave(cu[1:], lens).to(torch.int32)
    return lo, hi


class _PLEGateConv(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function for the whole PLE increment. Saves only the inputs and
    # small per-token statistics (gate values, reciprocal RMS values, sequence bounds, and the
    # pre-activation convolution result); large intermediates (gated signal, normalized signal) are
    # recomputed in backward.
    @staticmethod
    def forward(ctx, hc_state, key, value, wk, wq, wc, conv_w, n, eps, dilation, cu_seqlens):
        # [FUNCTION DOC] Inputs: hc_state (tokens by W) used as the query stream, key (tokens by W),
        # value (tokens by C = W / n), key and query norm weights wk, wq (length W), conv-norm weight
        # wc (length W), conv_w (W by 1 by K depthwise taps), number of streams n, epsilon, convolution
        # dilation, and optional cumulative sequence lengths. Runs the gate kernel, the grouped RMS
        # norm of the gated signal (using wc), the sequence bounds, and the convolution kernel with
        # 256-channel blocks; empty token counts skip launches. Returns the output (tokens by W) in
        # the dtype of hc_state. Stores the weight dtype and sizes on the context.
        T, W = hc_state.shape
        C = W // n
        Kk = conv_w.shape[-1]
        dev = hc_state.device

        gated = torch.empty(T, W, dtype=torch.float32, device=dev)
        gate = torch.empty(T, n, dtype=torch.float32, device=dev)
        rstdk = torch.empty(T, n, dtype=torch.float32, device=dev)
        rstdq = torch.empty(T, n, dtype=torch.float32, device=dev)
        if T > 0:
            _ple_gate_fwd_kernel[(T * n,)](
                key,
                hc_state,
                value,
                wk,
                wq,
                gated,
                gate,
                rstdk,
                rstdq,
                T,
                N=n,
                C=C,
                EPS=eps,
                SQRTC=math.sqrt(C),
                BLOCK_C=block_c(C),
            )

        normed, rstdc = grouped_rmsnorm_fwd(gated, wc, n, eps)

        seg_lo, seg_hi = _seg_bounds(T, cu_seqlens, dev)
        convw2d = conv_w.reshape(W, Kk).contiguous()
        out = torch.empty(T, W, dtype=hc_state.dtype, device=dev)
        conv_pre = torch.empty(T, W, dtype=torch.float32, device=dev)
        BW = 256
        if T > 0:
            _ple_conv_fwd_kernel[(T, triton.cdiv(W, BW))](
                normed,
                gated,
                convw2d,
                seg_lo,
                out,
                conv_pre,
                T,
                W,
                K=Kk,
                DIL=dilation,
                BLOCK_W=BW,
            )

        ctx.save_for_backward(
            hc_state, key, value, wk, wq, wc, convw2d, gate, rstdk, rstdq, rstdc, seg_lo, seg_hi, conv_pre
        )
        ctx.dims = (n, eps, dilation, Kk, conv_w.dtype)
        return out

    @staticmethod
    def backward(ctx, dout):
        # [FUNCTION DOC] Returns gradients in the order of the forward inputs: for hc_state, key,
        # value, wk, wq, wc, conv_w, then none for n, eps, dilation and the sequence lengths.
        # Steps: recompute the gated signal and the grouped-norm output (discarding the recomputed
        # gate statistics, the saved ones are used later); run the convolution backward kernel to get
        # the gradient of the normalized signal, the convolution weight gradient (float32 buffer with
        # atomics, cast to the weight dtype and reshaped to W by 1 by K) and the residual gradient;
        # the conv-norm weight gradient is the token sum of normalized-gradient times the
        # weight-free normalized signal; run the grouped norm backward and add its result to the
        # gated-signal gradient; run the gate backward kernel for the gradients of key and hc_state;
        # the value gradient sums per-stream contributions over streams; the key / query norm weight
        # gradients sum per-token partials over tokens. All reductions are in float32 and cast to the
        # parameter dtypes.
        (hc_state, key, value, wk, wq, wc, convw2d, gate, rstdk, rstdq, rstdc, seg_lo, seg_hi, conv_pre) = (
            ctx.saved_tensors
        )
        n, eps, dilation, Kk, conv_w_dtype = ctx.dims
        T, W = hc_state.shape
        C = W // n
        dev = hc_state.device
        dout = dout.contiguous()

        gated = torch.empty(T, W, dtype=torch.float32, device=dev)
        _g = torch.empty(T, n, dtype=torch.float32, device=dev)
        _rk = torch.empty(T, n, dtype=torch.float32, device=dev)
        _rq = torch.empty(T, n, dtype=torch.float32, device=dev)
        if T > 0:
            _ple_gate_fwd_kernel[(T * n,)](
                key,
                hc_state,
                value,
                wk,
                wq,
                gated,
                _g,
                _rk,
                _rq,
                T,
                N=n,
                C=C,
                EPS=eps,
                SQRTC=math.sqrt(C),
                BLOCK_C=block_c(C),
            )
        normed, _ = grouped_rmsnorm_fwd(gated, wc, n, eps)

        dnormed = torch.empty(T, W, dtype=torch.float32, device=dev)
        dconvw = torch.zeros(W, Kk, dtype=torch.float32, device=dev)
        dgated = torch.empty(T, W, dtype=torch.float32, device=dev)
        BW = 256
        if T > 0:
            _ple_conv_bwd_kernel[(T, triton.cdiv(W, BW))](
                dout,
                conv_pre,
                normed,
                convw2d,
                seg_lo,
                seg_hi,
                dnormed,
                dconvw,
                dgated,
                T,
                W,
                K=Kk,
                DIL=dilation,
                BLOCK_W=BW,
            )

        x_hat = (gated.view(T, n, C) * rstdc.unsqueeze(-1)).view(T, W)
        dwc = (dnormed * x_hat).sum(dim=0).to(wc.dtype)
        dgated_norm = torch.empty(T, W, dtype=torch.float32, device=dev)
        if T > 0:
            grouped_rmsnorm_bwd_kernel[(T * n,)](
                gated, wc, rstdc, dnormed, dgated_norm, T, N=n, C=C, BLOCK_C=block_c(C)
            )
        dgated += dgated_norm

        dkey = torch.empty_like(key)
        dquery = torch.empty_like(hc_state)
        dvalue_pern = torch.empty(T, n, C, dtype=torch.float32, device=dev)
        dwk_part = torch.empty(T, W, dtype=torch.float32, device=dev)
        dwq_part = torch.empty(T, W, dtype=torch.float32, device=dev)
        if T > 0:
            _ple_gate_bwd_kernel[(T * n,)](
                dgated,
                key,
                hc_state,
                value,
                wk,
                wq,
                gate,
                rstdk,
                rstdq,
                dkey,
                dquery,
                dvalue_pern,
                dwk_part,
                dwq_part,
                T,
                N=n,
                C=C,
                SQRTC=math.sqrt(C),
                BLOCK_C=block_c(C),
            )
        dvalue = dvalue_pern.sum(dim=1).to(value.dtype)
        dwk = dwk_part.sum(dim=0).to(wk.dtype)
        dwq = dwq_part.sum(dim=0).to(wq.dtype)
        dconv_w = dconvw.view(W, 1, Kk).to(conv_w_dtype)

        return (dquery, dkey, dvalue, dwk, dwq, dwc, dconv_w, None, None, None, None)


def ple_gate_conv(
    hc_state,
    key,
    value,
    norm_key_w,
    norm_query_w,
    norm_conv_w,
    conv1d_weight,
    n: int,
    eps: float,
    dilation: int,
    cu_seqlens,
):
    # [FUNCTION DOC] Public entry point: makes hc_state, key and value contiguous and applies the
    # fused autograd function (argument meanings as described for its forward). norm_key_w and
    # norm_query_w are the per-channel gate-chain norm weights, norm_conv_w the weight of the grouped
    # norm before the convolution, conv1d_weight the depthwise filter (W by 1 by K), n the stream
    # count, eps the norm epsilon, dilation the tap spacing in tokens, cu_seqlens cumulative sequence
    # lengths or None for one sequence. Returns the output with the shape and dtype of hc_state.
    """Full PLE increment (gate chain + norm + causal conv + SiLU + residual)."""
    return _PLEGateConv.apply(
        hc_state.contiguous(),
        key.contiguous(),
        value.contiguous(),
        norm_key_w,
        norm_query_w,
        norm_conv_w,
        conv1d_weight,
        n,
        eps,
        dilation,
        cu_seqlens,
    )
