# [MODULE DOC]
# MXFP8 weight quantization through transformer engine: 8-bit E4M3 elements with one block scale per
# 32 consecutive elements along the last dimension. Transformer engine is imported lazily.
import torch


MXFP8_GROUP_SIZE = 32
TE_MXFP8_ROW_ALIGNMENT = 32


def mxfp8_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Quantizes a weight of any rank whose last dimension is divisible by 32 (value
    # error otherwise). Makes it contiguous, flattens to rows, zero-pads the row count up to a
    # multiple of 32 (the quantizer's alignment), quantizes row-wise only with E4M3 element type,
    # then strips the padding. Returns (float8 E4M3 data shaped like the input, scales with the last
    # dimension divided by 32 and the other dimensions as the input; compact, unswizzled, dtype as
    # produced by the engine). Requires a gpu-capable transformer engine.
    """Quantize a tensor to rowwise MXFP8 with compact, unswizzled scales."""
    from transformer_engine.pytorch import MXFP8Quantizer
    from transformer_engine.pytorch.constants import TE_DType

    weight = weight.contiguous()
    k = weight.shape[-1]
    if k % MXFP8_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {k} must be divisible by {MXFP8_GROUP_SIZE} for MXFP8.")

    weight_flat = weight.view(-1, k)
    num_rows = weight_flat.shape[0]
    pad_rows = (-num_rows) % TE_MXFP8_ROW_ALIGNMENT
    if pad_rows:
        padding = torch.zeros((pad_rows, k), device=weight.device, dtype=weight.dtype)
        weight_flat = torch.cat((weight_flat, padding), dim=0)

    quantizer = MXFP8Quantizer(
        fp8_dtype=TE_DType[torch.float8_e4m3fn],
        rowwise=True,
        columnwise=False,
    )
    quantized = quantizer.quantize(weight_flat)
    qweight = quantized._rowwise_data[:num_rows, :k].contiguous()
    qweight = qweight.view(torch.float8_e4m3fn).view_as(weight)
    scale = quantized._rowwise_scale_inv[:num_rows, : k // MXFP8_GROUP_SIZE]
    scale = scale.contiguous().view(*weight.shape[:-1], k // MXFP8_GROUP_SIZE)
    return qweight, scale
