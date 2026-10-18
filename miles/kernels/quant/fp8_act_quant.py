# [MODULE DOC] Tilelang block-wise FP8 (E4M3) activation quantization, ported from the DeepSeek-V4
# reference inference kernels for bit-exact parity. Each row of the last dimension is split into
# groups of block_size (default 128) elements; each (row, group) gets a scale = group absolute
# maximum (floored at 1e-4) divided by 448, optionally rounded up to a power of two (MX-style) via
# integer bit tricks; values are divided by the scale, clamped to [-448, 448] and cast to FP8. A
# fused "inplace" mode quantizes and immediately dequantizes back to the input dtype (fake
# quantization). Contains three small bit-manipulation helpers, the kernel factory act_quant_kernel
# and the host wrapper act_quant. Importing this module sets the tilelang log level to WARNING
# (a global side effect). Requires tilelang and a GPU.
"""Block-wise FP8 activation quantization for DeepSeek-V4.

Ported verbatim from deepseek-ai/DeepSeek-V4-Pro/inference/kernel.py to keep
bit-exact parity with the upstream inference kernel. Keep this file in sync
when DeepSeek updates the reference implementation.

Source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py
"""

import tilelang
import tilelang.language as T
import torch

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
FP4 = "float4_e2m1fn"
FE8M0 = "float8_e8m0fnu"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


def fast_log2_ceil(x):
    # [FUNCTION DOC] HELPER: Tilelang expression builder. Reinterprets a float32 value as bits and
    # returns the integer ceil of log base 2: biased exponent minus 127, plus one when the mantissa
    # bits are non-zero. Valid for positive normal values only.
    """Compute ceil(log2(x)) via IEEE 754 bit manipulation. Avoids slow log/ceil intrinsics."""
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    # [FUNCTION DOC] HELPER: Tilelang expression builder returning two to the integer power x as a
    # float32, by placing (x plus 127) into the exponent field; valid for exponents in the normal range.
    """Compute 2^x for integer x via IEEE 754 bit manipulation."""
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    # [FUNCTION DOC] HELPER: Rounds the scale amax times fp8_max_inv up to the next power of two
    # (two to the integer ceil of its log base 2), for power-of-two (MX) scale formats.
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(
    N, block_size=128, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32, round_scale=False, inplace=False
):
    # [FUNCTION DOC] Kernel factory (jit-compiled, cached per argument values). N is the row width
    # (columns); block_size the group width; in_dtype the input type name (default bfloat16);
    # out_dtype the quantized type name (default float8 E4M3), which is replaced by the input type when
    # inplace; scale_dtype the stored scale type name; round_scale selects power-of-two scales;
    # inplace selects fake-quantization back to the input type. The row count M is symbolic.
    # Returns a program with a grid of (row tiles of 32, column groups), 128 threads: for each tile
    # it loads the 32-by-group block, takes the per-row absolute maximum, floors it at 1e-4, derives
    # the scale (maximum / 448, or power-of-two rounded), then writes either the clamped quotient
    # cast to the output type, or (inplace) the clamped quotient cast to FP8, widened to float32,
    # multiplied by the scale and cast to the input type; the scale (cast to scale_dtype) is
    # always written to the scale tensor of shape rows by groups. Pipeline stages are 0 when
    # round_scale or inplace, else 2. Pass options disable warp specialization and TMA lowering.
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16."""
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale or inplace else 2
    blk_m = 32
    group_size = block_size
    # Internal computation in FP32; scale_dtype controls output storage format.
    compute_dtype = FP32
    out_dtype = in_dtype if inplace else out_dtype

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        # [FUNCTION DOC] HELPER: The device program built by the factory above (see its notes for
        # the algorithm). Inputs: X, the M-by-N input; outputs written in place: Y, the M-by-N
        # quantized (or fake-quantized) result, and S, the per-row per-group scales. One thread block
        # handles a 32-row by group-width tile, using shared and register fragments, a single-iteration
        # pipelined loop, and a parallel per-row scale computation.
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                if inplace:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(
                                compute_dtype, T.Cast(out_dtype, T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max))
                            )
                            * s_local[i],
                        )
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> torch.Tensor:
    # [FUNCTION DOC] Host wrapper. x has any leading shape and last dimension N divisible by
    # block_size (asserted). scale_fmt: any non-none value turns on power-of-two scales; scale_dtype
    # float8 E8M0 selects that scale storage, any other value stores float32 scales (the tilelang
    # scale type is only distinguished for E8M0; other requested dtypes would still be allocated in
    # the requested dtype). Not inplace: returns the tuple (FP8 tensor shaped like x, scales shaped like
    # x's leading shape plus N / block_size). Inplace: runs fake quantization into a temporary, copies
    # it back into x (mutating the caller's tensor) and returns x only; the scales are computed but
    # discarded. The input is made contiguous for the kernel.
    # KNOWN ISSUE: the declared return type is a single tensor, but the non-inplace path returns a
    # pair (quantized, scales).
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16.
    When scale_fmt is set, scales are rounded to power-of-2 (MXFP).
    """
    N = x.size(-1)
    assert N % block_size == 0
    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
    z = x.contiguous()
    y = torch.empty_like(z) if inplace else torch.empty_like(z, dtype=torch.float8_e4m3fn)
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=scale_dtype)
    kernel = act_quant_kernel(
        N,
        block_size,
        scale_dtype=tl_dtype,
        round_scale=scale_fmt is not None,
        inplace=inplace,
    )
    kernel(z.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    if inplace:
        x.copy_(y)
        return x
    return y, s
