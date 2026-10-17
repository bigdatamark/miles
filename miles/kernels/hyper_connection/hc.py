# [MODULE DOC] Fused hyper-connection operations: the hidden state of each token is kept as n
# parallel streams of C channels (flattened row of width W = n times C). Two public operations:
# (1) hc_mix_inject: grouped RMS-normalize each stream (shared weight of length W), compute a
# per-channel gate from the whole normalized row through a two-layer bottleneck (linear down with
# a 1/n scale, SiLU, linear up, sigmoid), and mix the streams into one C-channel vector as the mean
# over streams of gate times normalized value; optionally also compute a per-stream "injection"
# coefficient h_post = 2 times sigmoid of a linear map of the normalized row divided by n.
# (2) hc_combine: write the block output back to every stream, new stream c = old stream c plus
# h_post for stream c times the block output y.
# Elementwise stages are Triton kernels (one program per token, or per token and stream); the
# matrix products of the gate chain run in torch in float32. Backward recomputes the normalization
# and gate chain instead of saving them. Every reduction/elementwise step is float32 with one final
# cast; token indices are widened to 64-bit before multiplying by the row width.
"""Triton hyper-connection kernels for Qwen3.8-Next (Qwen4Exp).

Every reduction and elementwise step runs in fp32, with one cast onto the output dtype
at the end. Token indices are promoted before multiplying by the stream width:
long sequences can exceed 2**31 elements even when the token count fits in int32.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from miles.kernels.norm.grouped_rmsnorm import block_c, grouped_rmsnorm_bwd_kernel, grouped_rmsnorm_fwd


@triton.jit(do_not_specialize=["T"])
def _gate_mul_mean_fwd_kernel(
    gate_ptr,  # fp32 [T, n*C]
    normed_ptr,  # fp32 [T, n*C]
    out_ptr,  # out dtype [T, C]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel, one program per token (surplus programs exit). For each
    # channel, the output is the mean over the N streams of gate times normalized value (both
    # float32, token rows of width N times C); the result row of C channels is cast to the output
    # dtype and stored. BLOCK_C is the padded channel block.
    t = tl.program_id(0).to(tl.int64)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for c in tl.static_range(N):
        base = t * (N * C) + c * C
        g = tl.load(gate_ptr + base + offs, mask=mask, other=0.0)
        nrm = tl.load(normed_ptr + base + offs, mask=mask, other=0.0)
        acc += g * nrm
    acc = acc / N
    tl.store(out_ptr + t * C + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T"])
def _gate_mul_mean_bwd_kernel(
    dmixed_ptr,  # fp32 [T, C]
    gate_ptr,
    normed_ptr,
    dgate_ptr,  # fp32 out [T, n*C]
    dnormed_ptr,  # fp32 out [T, n*C] (mix contribution only)
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Backward of the gate-times-mean kernel, one program per token. Takes the
    # float32 gradient of the mixed row (C channels) divided by N, and for each stream writes the
    # gate gradient (that value times the normalized input) and the normalized-input gradient
    # contribution of the mix (that value times the gate) into float32 token-by-(N times C) buffers.
    t = tl.program_id(0).to(tl.int64)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    dm = tl.load(dmixed_ptr + t * C + offs, mask=mask, other=0.0) / N
    for c in tl.static_range(N):
        base = t * (N * C) + c * C
        g = tl.load(gate_ptr + base + offs, mask=mask, other=0.0)
        nrm = tl.load(normed_ptr + base + offs, mask=mask, other=0.0)
        tl.store(dgate_ptr + base + offs, dm * nrm, mask=mask)
        tl.store(dnormed_ptr + base + offs, dm * g, mask=mask)


@triton.jit(do_not_specialize=["T"])
def _combine_fwd_kernel(
    res_ptr,  # bf16 [T, n*C]
    y_ptr,  # bf16 [T, C]
    hpost_ptr,  # fp32 [T, n]
    out_ptr,  # res dtype [T, n*C]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel, one program per (token, stream) pair (program id divided
    # by N gives the token, remainder the stream). Output stream row = residual stream row plus the
    # stream's scalar coefficient (float32, from the token-by-N table) times the token's block output
    # row (C channels, shared by all streams), computed in float32 and cast to the output dtype.
    pid = tl.program_id(0).to(tl.int64)
    t = pid // N
    c = pid % N
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    base = t * (N * C) + c * C
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(hpost_ptr + t * N + c)
    tl.store(out_ptr + base + offs, (r + a * y).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T"])
def _combine_bwd_kernel(
    dout_ptr,  # bf16 [T, n*C] (incoming grad, cast on load)
    y_ptr,  # bf16 [T, C]
    hpost_ptr,  # fp32 [T, n]
    dy_ptr,  # out y dtype [T, C]
    dhpost_ptr,  # fp32 out [T, n]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Backward of the combine kernel, one program per token. The gradient of
    # the shared block output is the sum over streams of coefficient times incoming stream gradient
    # (float32, cast to the output dtype); the gradient of each stream's coefficient is the sum over
    # channels of incoming stream gradient times the block output (float32 token-by-N buffer). The
    # gradient of the residual is the incoming gradient itself and is handled by the caller.
    t = tl.program_id(0).to(tl.int64)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    y = tl.load(y_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.zeros([BLOCK_C], dtype=tl.float32)
    for c in tl.static_range(N):
        base = t * (N * C) + c * C
        do = tl.load(dout_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(hpost_ptr + t * N + c)
        dy += a * do
        tl.store(dhpost_ptr + t * N + c, tl.sum(do * y, axis=0))
    tl.store(dy_ptr + t * C + offs, dy.to(dy_ptr.dtype.element_ty), mask=mask)


def _gate_chain_fwd(normed, w_down, w_up, n):
    # [FUNCTION DOC] HELPER: Torch computation of the gate in float32: z1 = linear(normed, w_down)
    # divided by n; gate = sigmoid(linear(SiLU(z1), w_up)), weights converted to float32. Normed has
    # width n times C, w_down maps it to the bottleneck width and w_up maps back to n times C.
    # Returns (gate, z1), both float32.
    """fp32 gate = sigmoid(Wup silu(Wdown N / n)). Returns (gate, z1) fp32."""
    z1 = F.linear(normed, w_down.float()) / n
    gate = torch.sigmoid(F.linear(F.silu(z1), w_up.float()))
    return gate, z1


class _HCMixInject(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function for the normalize, gate, mix and injection-coefficient
    # stage. Saves the input, norm weight, gate-chain weights, optional injection weight and the
    # per-group reciprocal RMS; everything else is recomputed in backward.
    """Fused mix + inject gate."""

    @staticmethod
    def forward(ctx, x2d, weight, w_down, w_up, w_inject, n, eps):
        # [FUNCTION DOC] Inputs: x2d (tokens by W = n times C), norm weight (length W), gate-chain
        # weights w_down and w_up, optional injection weight w_inject (n by W, or none), stream
        # count n and norm epsilon. Normalizes with the grouped RMS norm (float32), computes the gate,
        # launches the gate-times-mean kernel to produce mixed (tokens by C, dtype of x2d), and, when
        # w_inject is given, h_post = 2 times sigmoid(linear(normed, w_inject) / n) in float32
        # (tokens by n); otherwise h_post is an empty float32 tensor. Returns (mixed, h_post).
        T, W = x2d.shape
        C = W // n
        normed, rstd = grouped_rmsnorm_fwd(x2d, weight, n, eps)
        gate, _ = _gate_chain_fwd(normed, w_down, w_up, n)
        mixed = torch.empty(T, C, dtype=x2d.dtype, device=x2d.device)
        if T > 0:
            _gate_mul_mean_fwd_kernel[(T,)](gate, normed, mixed, T, N=n, C=C, BLOCK_C=block_c(C))
        if w_inject is not None:
            h_post = 2.0 * torch.sigmoid(F.linear(normed, w_inject.float()) / n)
        else:
            h_post = x2d.new_empty(0, dtype=torch.float32)
        ctx.save_for_backward(x2d, weight, w_down, w_up, w_inject, rstd)
        ctx.n, ctx.eps = n, eps
        return mixed, h_post

    @staticmethod
    def backward(ctx, dmixed, dh_post):
        # [FUNCTION DOC] Returns gradients for (x2d, weight, w_down, w_up, w_inject) and none for n
        # and epsilon. Recomputes the normalized input, gate and bottleneck activations, runs the
        # gate-times-mean backward kernel, then back-propagates in float32 torch: through the sigmoid
        # of the gate, the up projection (weight gradient = transposed gradient times SiLU
        # activation), SiLU (derivative sigmoid(z) times (1 + z times (1 - sigmoid(z)))), the 1/n
        # scale, and the down projection, adding the contribution to the normalized-input gradient.
        # If an injection weight exists and the h_post gradient is non-empty, adds the injection
        # branch likewise (derivative 2 times sigmoid times (1 - sigmoid), divided by n). The norm
        # weight gradient is the token sum of normalized-input gradient times the weight-free
        # normalized input; the input gradient comes from the grouped-norm backward kernel (dtype of
        # x2d). Weight gradients are cast to the parameter dtypes.
        x2d, weight, w_down, w_up, w_inject, rstd = ctx.saved_tensors
        n, eps = ctx.n, ctx.eps
        T, W = x2d.shape
        C = W // n
        normed, _ = grouped_rmsnorm_fwd(x2d, weight, n, eps)
        gate, z1 = _gate_chain_fwd(normed, w_down, w_up, n)
        s1 = F.silu(z1)

        dmixed = dmixed.float()
        dgate = torch.empty(T, W, dtype=torch.float32, device=x2d.device)
        dnormed = torch.empty(T, W, dtype=torch.float32, device=x2d.device)
        if T > 0:
            _gate_mul_mean_bwd_kernel[(T,)](dmixed, gate, normed, dgate, dnormed, T, N=n, C=C, BLOCK_C=block_c(C))

        dz2 = dgate * gate * (1.0 - gate)
        dw_up = dz2.t() @ s1
        ds1 = dz2 @ w_up.float()
        sig_z1 = torch.sigmoid(z1)
        dz1 = ds1 * sig_z1 * (1.0 + z1 * (1.0 - sig_z1))
        dz1 = dz1 / n  # z1 = (N @ w_down^T) / n
        dw_down = dz1.t() @ normed
        dnormed += dz1 @ w_down.float()

        dw_inject = None
        if w_inject is not None and dh_post is not None and dh_post.numel() > 0:
            u = F.linear(normed, w_inject.float()) / n
            sig_u = torch.sigmoid(u)
            du = dh_post.float() * 2.0 * sig_u * (1.0 - sig_u)
            du = du / n
            dw_inject = (du.t() @ normed).to(w_inject.dtype)
            dnormed += du @ w_inject.float()

        x_hat = (x2d.float().view(T, n, C) * rstd.unsqueeze(-1)).view(T, W)
        dweight = (dnormed * x_hat).sum(dim=0).to(weight.dtype)

        dx = torch.empty_like(x2d)
        if T > 0:
            grouped_rmsnorm_bwd_kernel[(T * n,)](x2d, weight, rstd, dnormed, dx, T, N=n, C=C, BLOCK_C=block_c(C))
        return (
            dx,
            dweight,
            dw_down.to(w_down.dtype),
            dw_up.to(w_up.dtype),
            dw_inject,
            None,
            None,
        )


class _HCCombine(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function for the write-back stage; saves the block output and the
    # float32 coefficients, and remembers the coefficients' original dtype for the gradient.
    """out = residual + h_post[:, c, None] * y  (flattened [T, n*C])."""

    @staticmethod
    def forward(ctx, residual2d, y2d, h_post, n):
        # [FUNCTION DOC] Inputs: residual (tokens by n times C), block output y (tokens by C),
        # h_post (tokens by n; converted to contiguous float32), stream count n. Returns a new
        # tensor shaped and typed like the residual with each stream row plus its coefficient times y.
        T, W = residual2d.shape
        C = W // n
        ctx.h_post_dtype = h_post.dtype
        h_post = h_post.float().contiguous()
        out = torch.empty_like(residual2d)
        if T > 0:
            _combine_fwd_kernel[(T * n,)](residual2d, y2d, h_post, out, T, N=n, C=C, BLOCK_C=block_c(C))
        ctx.save_for_backward(y2d, h_post)
        ctx.n = n
        return out

    @staticmethod
    def backward(ctx, dout):
        # [FUNCTION DOC] Returns (gradient for the residual, which is the incoming gradient made
        # contiguous and passed through unchanged; gradient for y from the backward kernel in y's
        # dtype; gradient for h_post in float32 cast to the original h_post dtype; none for n).
        y2d, h_post = ctx.saved_tensors
        n = ctx.n
        T, W = dout.shape
        C = W // n
        dout = dout.contiguous()
        dy = torch.empty_like(y2d)
        dh_post = torch.empty(T, n, dtype=torch.float32, device=dout.device)
        if T > 0:
            _combine_bwd_kernel[(T,)](dout, y2d, h_post, dy, dh_post, T, N=n, C=C, BLOCK_C=block_c(C))
        return dout, dy, dh_post.to(ctx.h_post_dtype), None


def hc_mix_inject(x, weight, w_down, w_up, w_inject, n: int, eps: float):
    # [FUNCTION DOC] Public entry: flattens all leading dimensions of x (last dimension n times C)
    # to tokens, applies the fused autograd function, and restores the leading shape. Returns
    # (mixed with last dimension C, h_post with last dimension n) when w_inject is given; when it is
    # none, the second value is the empty tensor from the autograd function (not reshaped).
    """Fused (grouped_gemma_rmsnorm -> hc_mix, hc_inject_gate)."""
    lead = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    mixed, h_post = _HCMixInject.apply(x2d, weight, w_down, w_up, w_inject, n, eps)
    mixed = mixed.reshape(*lead, -1)
    if w_inject is not None:
        h_post = h_post.reshape(*lead, n)
    return mixed, h_post


def hc_combine(residual, block_output, h_post, n: int):
    # [FUNCTION DOC] Public entry: flattens leading dimensions of the residual (last dimension n
    # times C) and the block output (last dimension C) to tokens, reshapes h_post to tokens by n,
    # applies the fused autograd function, and returns the result with the residual's shape.
    """``X'_c = X_c + a_c * y`` -- fused elementwise."""
    lead = residual.shape[:-1]
    r2d = residual.reshape(-1, residual.shape[-1]).contiguous()
    y2d = block_output.reshape(-1, block_output.shape[-1]).contiguous()
    hp2d = h_post.reshape(-1, n)
    out = _HCCombine.apply(r2d, y2d, hp2d, n)
    return out.reshape(*lead, -1)
