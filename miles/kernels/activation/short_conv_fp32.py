# [MODULE DOC] Residual depthwise causal short convolution over packed (concatenated) variable
# length sequences, with float32 accumulation, as a differentiable operation.
# Math: input x has shape tokens by channels, weight has one filter of W taps per channel. For
# token t in a segment starting at token b, output[t, d] = x[t, d] (the residual) plus the sum over
# taps i in 0..W-1 of weight[d, i] times x[t - (W-1) + i, d], where inputs before the segment start
# b count as zero (so the last tap multiplies the current token and the first tap the oldest).
# Segments never see each other. Forward runs a Triton kernel; backward computes the input gradient
# with a second Triton kernel and the weight gradient with dense torch operations in float32.
# Public entry point: short_conv_fp32.
"""Residual depthwise causal short convolution over packed sequences in fp32; forward is
bit-identical to sglang's sconv kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sconv_fwd_kernel(
    x_ptr,
    w_ptr,
    bos_ptr,
    y_ptr,
    T,
    D,
    W: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton forward kernel. Grid: dimension 0 tiles tokens by BLOCK_T,
    # dimension 1 tiles channels by BLOCK_D. Inputs: x (tokens T by channels D, row-major), w (D
    # by W filter taps, contiguous), bos (per-token index of the first token of its segment).
    # Output y (same shape and dtype as x). For each tile, accumulates in float32 over the W taps
    # in tap order: tap i reads the token W-1-i positions earlier, masked to zero when that
    # position precedes the token's segment start (or is out of range); the last tap uses the
    # current token. Finally adds the current token (residual) and stores cast to the output dtype.
    # Out-of-range rows/columns are masked. The earlier-token loads use an evict-last cache hint.
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    bos = tl.load(bos_ptr + t_off, mask=t_mask, other=0).to(tl.int64)
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    x_cur = tl.load(x_ptr + t_off[:, None] * D + d_off[None, :], mask=td_mask, other=0)

    for iw in tl.static_range(W):
        shifted = t_off - (W - 1) + iw
        if iw == W - 1:
            tap = x_cur.to(tl.float32)
        else:
            in_x = shifted >= bos
            x_val = tl.load(
                x_ptr + shifted[:, None] * D + d_off[None, :],
                mask=in_x[:, None] & td_mask,
                other=0,
                eviction_policy="evict_last",
            )
            tap = x_val.to(tl.float32)
        w_val = tl.load(w_ptr + d_off * W + iw, mask=d_mask, other=0).to(tl.float32)
        acc += tap * w_val[None, :]

    acc += x_cur.to(tl.float32)
    tl.store(y_ptr + t_off[:, None] * D + d_off[None, :], acc.to(y_ptr.dtype.element_ty), mask=td_mask)


@triton.jit
def _sconv_bwd_dx_kernel(
    go_ptr,
    w_ptr,
    eos_ptr,
    dx_ptr,
    T,
    D,
    W: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel for the gradient with respect to the input, same tiling
    # as the forward kernel. Inputs: go (output gradient, T by D), w (D by W taps), eos (per-token
    # index one past the last token of its segment). For token t it sums, in tap order, the
    # gradient of the later token t+(W-1)-i times weight tap i, counting only later tokens that
    # lie before the segment end, in float32; then adds the gradient of token t itself (the
    # residual path) and stores cast to dx's dtype.
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    eos = tl.load(eos_ptr + t_off, mask=t_mask, other=0).to(tl.int64)
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    for iw in tl.static_range(W):
        fwd_t = t_off + (W - 1) - iw
        in_seg = fwd_t < eos
        g_val = tl.load(
            go_ptr + fwd_t[:, None] * D + d_off[None, :],
            mask=in_seg[:, None] & td_mask,
            other=0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        w_val = tl.load(w_ptr + d_off * W + iw, mask=d_mask, other=0).to(tl.float32)
        acc += g_val * w_val[None, :]

    g_cur = tl.load(go_ptr + t_off[:, None] * D + d_off[None, :], mask=td_mask, other=0).to(tl.float32)
    acc += g_cur
    tl.store(dx_ptr + t_off[:, None] * D + d_off[None, :], acc.to(dx_ptr.dtype.element_ty), mask=td_mask)


def _bos_eos(seqlens, T, device):
    # [FUNCTION DOC] HELPER: Builds per-token segment boundaries as two int32 tensors of length T
    # on the device: bos (index of the segment's first token) and eos (index one past its last
    # token). With no sequence lengths, or just one, the whole range is a single segment (0 to T);
    # the single length value itself is ignored. Otherwise each token gets the cumulative start/end
    # of its sequence; the lengths are expected to sum to T (a mismatch yields tensors whose
    # length differs from T and later out-of-range reads).
    if seqlens is None or len(seqlens) <= 1:
        bos = torch.zeros(T, device=device, dtype=torch.int32)
        eos = torch.full((T,), T, device=device, dtype=torch.int32)
        return bos, eos
    sl = torch.as_tensor(list(seqlens), device=device, dtype=torch.int32)
    ends = sl.cumsum(0)
    starts = ends - sl
    bos = torch.repeat_interleave(starts, sl)
    eos = torch.repeat_interleave(ends, sl)
    return bos.to(torch.int32), eos.to(torch.int32)


class _SconvFP32Triton(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function pairing the forward Triton kernel with the backward
    # (input gradient by Triton, weight gradient by torch). Saves the contiguous input, the
    # weight reshaped to channels by taps, and the segment boundary tensors, plus the original
    # weight shape.
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, seqlens) -> torch.Tensor:
        # [FUNCTION DOC] Computes the convolution for x (tokens by channels) and weight (any shape
        # reshapeable to channels by taps; the tap count is the last dimension). seqlens is an
        # optional sequence of segment lengths. Returns a new tensor of x's shape and dtype. Launches
        # a grid of 64-token by 128-channel tiles unless there are no tokens. No error checking.
        T, D = x.shape
        W = weight.shape[-1]
        xc = x.contiguous()
        w2 = weight.reshape(D, W).contiguous()
        bos, eos = _bos_eos(seqlens, T, x.device)
        y = torch.empty_like(xc)
        if T > 0:
            grid = (triton.cdiv(T, 64), triton.cdiv(D, 128))
            _sconv_fwd_kernel[grid](xc, w2, bos, y, T, D, W=W, BLOCK_T=64, BLOCK_D=128)
        ctx.save_for_backward(xc, w2, bos, eos)
        ctx.w_shape = weight.shape
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # [FUNCTION DOC] Returns the gradients for (x, weight, seqlens): the input gradient from the
        # Triton kernel (same dtype as x), the weight gradient, and none for the lengths. The weight
        # gradient is built in float32 per tap i with shift = W-1-i: the sum over tokens of the
        # output gradient times the input shifted back by that amount, counting only tokens whose
        # shifted source lies inside their own segment (shift 0 counts all tokens); it is then
        # reshaped to the original weight shape and cast to the working weight dtype. This
        # materializes float32 copies of both the input and the gradient.
        xc, w2, bos, eos = ctx.saved_tensors
        T, D = xc.shape
        W = w2.shape[-1]
        go = grad_out.contiguous()
        dx = torch.empty_like(xc)
        if T > 0:
            grid = (triton.cdiv(T, 64), triton.cdiv(D, 128))
            _sconv_bwd_dx_kernel[grid](go, w2, eos, dx, T, D, W=W, BLOCK_T=64, BLOCK_D=128)
        go32 = go.float()
        x32 = xc.float()
        dw = torch.empty(D, W, device=xc.device, dtype=torch.float32)
        ar = torch.arange(T, device=xc.device, dtype=torch.int32)
        for iw in range(W):
            shift = (W - 1) - iw
            if shift == 0:
                dw[:, iw] = (go32 * x32).sum(0)
            else:
                valid = (ar[shift:] - shift) >= bos[shift:]
                dw[:, iw] = (go32[shift:] * x32[: T - shift] * valid.unsqueeze(1)).sum(0)
        return dx, dw.reshape(ctx.w_shape).to(w2.dtype), None


def short_conv_fp32(x: torch.Tensor, weight: torch.Tensor, seqlens=None) -> torch.Tensor:
    # [FUNCTION DOC] Public entry: applies the residual causal depthwise short convolution to x
    # (tokens by channels) with per-channel taps from weight, optionally restricted to the packed
    # segments given by seqlens (a sequence of lengths summing to the token count; None means one
    # segment). Returns a tensor of x's shape and dtype; differentiable in x and weight. Runs on GPU.
    """Packed-sequence residual sconv, forward bit-identical to sglang's kernel."""
    return _SconvFP32Triton.apply(x, weight, seqlens)
