# [MODULE DOC] Triton FP8 (E4M3) block-wise cast for 2-D tensors with one scale per 2-D tile (default
# 128 by 128, "blockwise" scaling as used for weights). The scale of a tile is its absolute maximum
# (floored at 1e-10) divided by the E4M3 maximum (448); values are divided by the scale (multiplied
# by its reciprocal), clamped to the FP8 range and cast. The returned scales are the dequantization
# multipliers (quantized value times scale recovers the input). Public entry point:
# fp8_blockwise_cast, with the helper ceil_div.
import torch
import triton
import triton.language as tl

fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max


def ceil_div(x: int, y: int) -> int:
    # [FUNCTION DOC] HELPER: Integer ceiling division of x by y (y positive).
    """
    Perform ceiling division of two integers.

    Args:
            x: the dividend.
            y: the divisor.

    Returns:
            The result of the ceiling division.
    """
    return (x + y - 1) // y


@triton.jit
def _fp8_blockwise_cast_kernel(
    X,
    Y,
    S,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    M,
    N,
    eps,
    fp8_min,
    fp8_max,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 128,
):
    # [FUNCTION DOC] HELPER: Triton kernel; one program per BLOCK_M by BLOCK_N tile (2-D grid, program
    # ids widened to 64-bit). Loads the tile (out-of-range elements as zero) in float32, takes the
    # tile-wide absolute maximum, floors it at eps, sets scale = maximum / fp8_max, multiplies the
    # tile by the reciprocal of the scale, clamps to [fp8_min, fp8_max], casts to the output dtype and
    # stores it; the float32 scale is stored at the tile's position in the scale tensor. Strides of
    # input, output and scales are passed explicitly. The block sizes default to 32 by 128 but the
    # host always overrides them.
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = off_m < M
    mask_n = off_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(x)), eps)
    x_s = _absmax / fp8_max
    s_inv = 1.0 / x_s
    y_q = tl.clamp(x * s_inv, fp8_min, fp8_max).to(Y.dtype.element_ty)

    tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y_q, mask=mask)
    tl.store(S + pid_m * stride_sm + pid_n * stride_sn, x_s)


def fp8_blockwise_cast(x: torch.Tensor, block_size=None) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Casts a 2-D tensor x (M by N, any float dtype and any strides) to FP8 E4M3 with
    # one scale per tile. block_size is an optional pair (rows, columns) of tile size, default 128 by
    # 128. Returns (quantized tensor M by N of float8 E4M3 on x's device, scales float32 of shape
    # ceil(M / rows) by ceil(N / columns)). Launch settings depend on layout: contiguous inputs use 8
    # warps and 2 stages, non-contiguous inputs 1 warp and 4 stages. The epsilon for the maximum is
    # 1e-10 and the clamp range is the E4M3 finite range.
    BLOCK_M, BLOCK_N = 128, 128
    if block_size:
        BLOCK_M, BLOCK_N = block_size[0], block_size[1]
    M, N = x.shape
    y = torch.empty(M, N, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N), dtype=torch.float32, device=x.device)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    if x.is_contiguous():
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 8, "num_stages": 2}
    else:
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 1, "num_stages": 4}
    _fp8_blockwise_cast_kernel[grid](
        x, y, s, *x.stride(), *y.stride(), *s.stride(), M, N, 1e-10, fp8_min, fp8_max, **kwargs
    )
    return y, s
