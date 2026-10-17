# [MODULE DOC] SwiGLU gated activation with float32 arithmetic and an optional per-token scale, as
# a differentiable operation backed by Triton kernels.
# Math: the input has a last dimension of 2N; its first half is the gate g and its second half the
# up projection u. Output (last dimension N) = g times sigmoid(g) times u, optionally multiplied
# per token by a scale. All math is done in float32 and the result is stored in the input dtype.
# Backward gives d_g = grad times scale times u times (s + silu(g) times (1-s)) with s=sigmoid(g),
# d_u = grad times scale times silu(g), and (if the scale needs a gradient) d_scale = sum over N of
# grad times silu(g) times u. Public entry point: swiglu_fp32.
"""fp32 SwiGLU with an optional per-token scale; forward is bit-identical to sglang's silu_and_mul."""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_kernel(
    x_ptr,
    scale_ptr,
    y_ptr,
    M,
    N,
    HAS_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton forward kernel on a grid of BLOCK_M by BLOCK_N tiles over an
    # M by N output. x is M rows of 2N values (gate then up), scale is one value per row (only read
    # when HAS_SCALE is set), y is the M by N output. Computes gate times sigmoid(gate) times up in
    # float32, multiplies by the row scale if present, and stores (implicitly cast to y's dtype).
    # Out-of-range rows and columns are masked.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    g = tl.load(x_ptr + offs_m[:, None] * (2 * N) + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + offs_m[:, None] * (2 * N) + N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    y = g * tl.sigmoid(g) * u
    if HAS_SCALE:
        sc = tl.load(scale_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        y = y * sc[:, None]
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], y, mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    x_ptr,
    scale_ptr,
    go_ptr,
    dx_ptr,
    dscale_ptr,
    M,
    N,
    HAS_SCALE: tl.constexpr,
    NEED_DSCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton backward kernel with one program per row (grid of M). Walks the
    # N columns in chunks of BLOCK_N. Per chunk loads gate, up and output gradient in float32 and
    # writes the gate gradient into the first half and the up gradient into the second half of the
    # row of dx (formulas in the module doc, with the row scale, or 1 if absent). If NEED_DSCALE,
    # accumulates the gradient-times-silu-times-up products (unscaled) and stores their sum over the
    # row as the scale gradient (float32). dx is written in dx's dtype.
    row = tl.program_id(0)
    if row >= M:
        return
    sc = 1.0
    if HAS_SCALE:
        sc = tl.load(scale_ptr + row).to(tl.float32)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n0 in tl.range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        g = tl.load(x_ptr + row * (2 * N) + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + row * (2 * N) + N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        go = tl.load(go_ptr + row * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        s = tl.sigmoid(g)
        silu = g * s
        gos = go * sc
        d_g = gos * u * (s + silu * (1 - s))
        d_u = gos * silu
        tl.store(dx_ptr + row * (2 * N) + offs_n, d_g, mask=mask_n)
        tl.store(dx_ptr + row * (2 * N) + N + offs_n, d_u, mask=mask_n)
        if NEED_DSCALE:
            acc += tl.where(mask_n, go * silu * u, 0.0)
    if NEED_DSCALE:
        tl.store(dscale_ptr + row, tl.sum(acc))


class _SwigluFP32Triton(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function wrapping the SwiGLU kernels. Saves the original input
    # and the scale (possibly none) for backward and recomputes everything there.
    @staticmethod
    def forward(ctx, fc1_out: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
        # [FUNCTION DOC] Flattens the leading dimensions of fc1_out to M rows of 2N values (made
        # contiguous), flattens the optional scale to one float32 value per row, and launches the
        # forward kernel on tiles of 32 rows by 128 columns (skipped when M is zero, leaving an
        # uninitialized output). Returns a tensor with the leading shape of the input and last
        # dimension N, in the input dtype. The last dimension must be even (not checked).
        shp = fc1_out.shape
        x = fc1_out.reshape(-1, shp[-1]).contiguous()
        M, N2 = x.shape
        N = N2 // 2
        sc = scale.reshape(-1).contiguous().float() if scale is not None else x.new_empty(1, dtype=torch.float32)
        y = torch.empty(M, N, device=x.device, dtype=x.dtype)
        if M > 0:
            grid = (triton.cdiv(M, 32), triton.cdiv(N, 128))
            _swiglu_fwd_kernel[grid](x, sc, y, M, N, HAS_SCALE=scale is not None, BLOCK_M=32, BLOCK_N=128)
        ctx.save_for_backward(fc1_out, scale)
        return y.reshape(*shp[:-1], N)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # [FUNCTION DOC] Returns (gradient for the input, gradient for the scale or none). Runs one
        # program per row with column blocks of 256. The scale gradient is computed only when a
        # scale was given and it requires grad; it is reshaped to the scale's shape and cast to the
        # scale dtype. The input gradient has the input's shape and dtype.
        fc1_out, scale = ctx.saved_tensors
        shp = fc1_out.shape
        x = fc1_out.reshape(-1, shp[-1]).contiguous()
        M, N2 = x.shape
        N = N2 // 2
        go = grad_out.reshape(-1, N).contiguous()
        sc = scale.reshape(-1).contiguous().float() if scale is not None else x.new_empty(1, dtype=torch.float32)
        need_dscale = scale is not None and ctx.needs_input_grad[1]
        dx = torch.empty_like(x)
        dsc = (
            torch.empty(M, device=x.device, dtype=torch.float32)
            if need_dscale
            else x.new_empty(1, dtype=torch.float32)
        )
        if M > 0:
            _swiglu_bwd_kernel[(M,)](
                x,
                sc,
                go,
                dx,
                dsc,
                M,
                N,
                HAS_SCALE=scale is not None,
                NEED_DSCALE=need_dscale,
                BLOCK_N=256,
            )
        d_scale = dsc.reshape(scale.shape).to(scale.dtype) if need_dscale else None
        return dx.reshape(shp), d_scale


def swiglu_fp32(fc1_out: torch.Tensor, per_token_scale: torch.Tensor | None = None) -> torch.Tensor:
    # [FUNCTION DOC] Public entry: applies SwiGLU to fc1_out (leading dims, last dim 2N) and
    # optionally multiplies each token's result by one scale value (per_token_scale must contain
    # one element per token, i.e. the product of the leading dimensions). Returns the result with
    # last dimension N; differentiable in both inputs. Runs on GPU.
    """Triton swiglu, forward bit-identical to sglang's silu_and_mul kernel."""
    return _SwigluFP32Triton.apply(fc1_out, per_token_scale)
