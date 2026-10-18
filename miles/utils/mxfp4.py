# [MODULE DOC]
# Pure torch MXFP4 quantization: 4-bit E2M1 elements (two per byte, low nibble first; sign in bit 3)
# with one 8-bit exponent-only scale (E8M0, stored with a bias of 127) per block of consecutive
# elements along the last dimension. Provides quantize and dequantize.
"""MXFP4 (E2M1 elements + E8M0 block scales) pack/unpack.

Torch-only, so checkpoint tooling can use it without importing Megatron.
"""

import torch

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_mxfp4(weight, group_size):
    # [FUNCTION DOC] Quantizes a floating tensor whose last dimension is divisible by the group size
    # and by 2 (asserted). Per block (flattened rows of group size): block scale exponent is the
    # ceiling of log2(block absolute max / 6) clamped to [-127, 127] (all-zero blocks clamp to the
    # minimum); values are divided by two to that exponent, and each magnitude is rounded to the
    # nearest of the eight E2M1 magnitudes (0, 0.5, 1, 1.5, 2, 3, 4, 6) by counting how many of the
    # midpoint thresholds (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0) it strictly exceeds (ties go to the
    # lower value); the sign bit (also set for negative zero) goes in bit 3. Adjacent element pairs
    # are packed into a byte (even index in the low nibble). Returns (packed bytes with last
    # dimension halved, unsigned 8-bit scales with biased exponent, last dimension divided by the
    # group size). Values above 6 times two to the exponent cannot occur since the exponent is
    # rounded up. Non-finite inputs are not handled.
    assert weight.shape[-1] % group_size == 0
    assert weight.shape[-1] % 2 == 0

    blocks = weight.reshape(-1, group_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).float()
    scale_exp = torch.ceil(torch.log2(amax / 6.0)).clamp_(-127, 127)
    normalized = blocks.float() * torch.exp2(-scale_exp)

    magnitude = torch.zeros_like(normalized, dtype=torch.uint8)
    for bound in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        magnitude.add_(normalized.abs() > bound)
    encoded = magnitude | (torch.signbit(normalized).to(torch.uint8) << 3)

    packed = encoded[:, 0::2] | (encoded[:, 1::2] << 4)
    packed = packed.reshape(*weight.shape[:-1], weight.shape[-1] // 2).contiguous()
    scale = (scale_exp + 127).to(torch.uint8)
    scale = scale.reshape(*weight.shape[:-1], weight.shape[-1] // group_size).contiguous()
    return packed, scale


def dequantize_mxfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    # [FUNCTION DOC] Inverse of quantize. Inputs: packed bytes (unsigned 8-bit; last dimension holds
    # two elements per byte) and biased block scales (unsigned 8-bit, one per group size consecutive
    # elements in flattened order), plus the positive group size (all asserted, including that the
    # element count divides by the group size and the scale count matches). Unpacks nibbles (low
    # nibble first), decodes sign and E2M1 magnitude via lookup, multiplies each block by two to the
    # power (scale minus 127), and returns a contiguous bfloat16 tensor with the last dimension
    # doubled. Computation is in float32 before the final cast.
    assert weight_packed.dtype == torch.uint8
    assert weight_scale.dtype == torch.uint8
    assert group_size > 0

    unpacked = torch.empty(
        *weight_packed.shape[:-1],
        weight_packed.shape[-1] * 2,
        dtype=torch.uint8,
        device=weight_packed.device,
    )
    unpacked[..., 0::2] = weight_packed & 0x0F
    unpacked[..., 1::2] = (weight_packed >> 4) & 0x0F

    signs = 1.0 - 2.0 * ((unpacked & 0b1000) >> 3).float()
    magnitudes = unpacked & 0b0111
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=weight_packed.device)
    dequantized = signs * values[magnitudes.long()]

    assert dequantized.numel() % group_size == 0
    assert weight_scale.numel() == dequantized.numel() // group_size
    dequantized = dequantized.reshape(-1, group_size)
    scales = torch.exp2(weight_scale.float().reshape(-1, 1) - 127.0)
    return (dequantized * scales).reshape(unpacked.shape).to(torch.bfloat16).contiguous()
