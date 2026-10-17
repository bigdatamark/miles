# [MODULE DOC] Fake (quantize then dequantize) block-scaled quantization of activations, forward
# only, with a straight-through gradient. The last dimension is split into groups ("blocks") of 32
# elements (16 for compressed key/value cache); each group gets one scale; values are divided by the
# scale, clamped to the format's range, rounded to the low-precision grid, and multiplied back by the
# scale, returning the same dtype and shape as the input. Formats: FP8 (E4M3, max 448) and FP4 (E2M1,
# magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6) both with a power-of-two scale (smallest power of two not
# below group maximum divided by the format maximum, with a floor on the maximum), and a compressed-
# KV variant (FP4 values, 16-element groups, scale = group maximum divided by 6 clamped to
# [2 to the -9, 448] and rounded to E4M3, divided with round-to-nearest). The core is one Triton
# kernel processing 64 groups per program. Public entry points: fake_quant_fp8, fake_quant_fp4,
# fake_quant_compressed_kv (autograd-safe, identity gradient) and their *_forward variants.
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

FP8_MAX = 448.0
FP4_MAX = 6.0
BLOCK_SIZE = 32
COMPRESSED_KV_BLOCK_SIZE = 16
FP8_AMAX_FLOOR = 1e-4
FP4_AMAX_FLOOR = 6 * 2.0**-126
COMPRESSED_KV_SCALE_MIN = 2.0**-9


@triton.jit
def _ceil_pow2(v):
    # [FUNCTION DOC] HELPER: Triton device function. For a positive normal float32 value (or block of
    # values) returns the smallest power of two that is not less than it, using the bit pattern: take
    # the unbiased exponent, add one if the mantissa is non-zero, and rebuild a float with that
    # exponent. Exact powers of two map to themselves. Zero, negative, denormal or non-finite inputs
    # give meaningless results.
    bits = v.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) - 127
    has_mantissa = (bits & 0x7FFFFF) != 0
    exponent = exponent + has_mantissa.to(tl.int32)
    return ((exponent + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _fake_quant_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    FMAX: tl.constexpr,
    INV_FMAX: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
    SCALE_MAX: tl.constexpr,
    FP4: tl.constexpr,
    E4M3_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel. The input is viewed as n_rows groups of BLOCK contiguous
    # elements; each program handles ROWS groups. Per group: take the maximum absolute value in
    # float32 and compute a scale. If E4M3_SCALE: scale = clamp(max times INV_FMAX, AMAX_FLOOR,
    # SCALE_MAX), rounded through the float8 E4M3 type, and values are divided with round-to-nearest
    # division. Otherwise: max = at least AMAX_FLOOR, scale = smallest power of two not below
    # max times INV_FMAX, plain division. Scaled values are clamped to [-FMAX, FMAX]. If FP4: round the
    # magnitude to the E2M1 grid (step 0.5 below 2, 1 below 4, 2 above, round half to even, sign
    # restored; zero stays zero); else round through float8 E4M3. The result times scale is stored in
    # the output dtype. Rows past n_rows are masked.
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    mask = (rows < n_rows)[:, None]
    offs = rows[:, None] * BLOCK + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    if E4M3_SCALE:
        scale = tl.minimum(tl.maximum(amax * INV_FMAX, AMAX_FLOOR), SCALE_MAX)
        scale = scale.to(tl.float8e4nv).to(tl.float32)
        scaled = tl.div_rn(x, scale[:, None])
    else:
        amax = tl.maximum(amax, AMAX_FLOOR)
        scale = _ceil_pow2(amax * INV_FMAX)
        scaled = x / scale[:, None]
    scaled = tl.minimum(tl.maximum(scaled, -FMAX), FMAX)
    if FP4:
        mag = tl.abs(scaled)
        step = tl.where(mag < 2.0, 0.5, tl.where(mag < 4.0, 1.0, 2.0))
        sign = tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))
        q = libdevice.rint(mag / step) * step * sign
    else:
        q = scaled.to(tl.float8e4nv).to(tl.float32)
    out = q * scale[:, None]
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def _fake_quant(x: torch.Tensor, fp4: bool, compressed_kv: bool = False) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Host launcher. Group size is 16 if compressed_kv else 32; asserts the
    # last dimension is divisible by it. Makes the input contiguous, allocates an output of the same
    # shape and dtype, and launches one program per 64 groups. Format parameters: compressed_kv uses
    # maximum 6 and scale floor 2 to the -9 (whatever fp4 says for the maximum); otherwise FP4 uses
    # maximum 6 with amax floor 6 times 2 to the -126, and FP8 uses maximum 448 with floor 1e-4. The
    # scale cap (448) is only used in the compressed-KV mode, whose scale is E4M3. Returns the output.
    block = COMPRESSED_KV_BLOCK_SIZE if compressed_kv else BLOCK_SIZE
    assert x.shape[-1] % block == 0
    xc = x.contiguous()
    out = torch.empty_like(xc)
    n_rows = xc.numel() // block
    rows_per_prog = 64
    grid = (triton.cdiv(n_rows, rows_per_prog),)
    if compressed_kv:
        fmax, floor = FP4_MAX, COMPRESSED_KV_SCALE_MIN
    else:
        fmax, floor = (FP4_MAX, FP4_AMAX_FLOOR) if fp4 else (FP8_MAX, FP8_AMAX_FLOOR)
    _fake_quant_kernel[grid](
        xc,
        out,
        n_rows,
        FMAX=fmax,
        INV_FMAX=1.0 / fmax,
        AMAX_FLOOR=floor,
        SCALE_MAX=FP8_MAX,
        FP4=fp4,
        E4M3_SCALE=compressed_kv,
        BLOCK=block,
        ROWS=rows_per_prog,
    )
    return out


def fake_quant_compressed_kv_forward(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Fake-quantizes to FP4 values with 16-element groups and E4M3 scales (compressed
    # key/value cache format). No autograd support (use the straight-through wrapper below).
    return _fake_quant(x, True, compressed_kv=True)


def fake_quant_fp4_forward(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Fake-quantizes to FP4 (E2M1) with 32-element groups and power-of-two scales.
    return _fake_quant(x, True)


def fake_quant_fp8_forward(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Fake-quantizes to FP8 (E4M3) with 32-element groups and power-of-two scales.
    return _fake_quant(x, False)


class _StraightThrough(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function implementing the straight-through estimator: the forward
    # applies an arbitrary function to the input, the backward passes the gradient through unchanged.
    @staticmethod
    def forward(ctx, x, fn):
        # [FUNCTION DOC] Returns the result of calling fn on x; nothing is saved.
        return fn(x)

    @staticmethod
    def backward(ctx, grad):
        # [FUNCTION DOC] Returns the incoming gradient for x (identity) and none for fn.
        return grad, None


def fake_quant_fp8(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Differentiable FP8 fake quantization: FP8 forward, identity gradient.
    return _StraightThrough.apply(x, fake_quant_fp8_forward)


def fake_quant_fp4(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Differentiable FP4 fake quantization: FP4 forward, identity gradient.
    return _StraightThrough.apply(x, fake_quant_fp4_forward)


def fake_quant_compressed_kv(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Differentiable compressed key/value fake quantization: forward as the compressed
    # variant, identity gradient.
    return _StraightThrough.apply(x, fake_quant_compressed_kv_forward)
