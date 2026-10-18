# [MODULE DOC]
# NVFP4 weight quantization through transformer engine: 4-bit E2M1 elements with an 8-bit E4M3 scale
# per block of 16 elements along the last dimension plus one per-tensor float32 global scale. Behavior
# is influenced by transformer-engine style environment variables (the "4over6" mode and its error
# mode). Includes helpers for the global scale math, row padding, and quantizing two weights jointly
# so they share a global scale. Transformer engine is imported lazily.
import os

import torch

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16
TE_NVFP4_ROW_ALIGNMENT = 16


def nvfp4_weight_e4m3_max() -> int:
    # [FUNCTION DOC] Maximum E4M3 scale value used for weights: 256 when the 4over6 environment
    # variable selects weights or all AND the use-256 variable (default all) also selects weights or
    # all; otherwise 448. Reads the environment at call time.
    if os.getenv("NVTE_NVFP4_4OVER6", "").strip().lower() in ("weights", "all") and os.getenv(
        "NVTE_NVFP4_4OVER6_E4M3_USE_256", "all"
    ).strip().lower() in ("weights", "all"):
        return 256
    return int(FP8_E4M3_MAX)


def nvfp4_global_encode_scale_te(
    global_amax: torch.Tensor,
    nvfp4_e4m3_max: int = int(FP8_E4M3_MAX),
) -> torch.Tensor:
    # [FUNCTION DOC] Global encode scale as a float32 tensor: (e4m3 max times 6) divided by the
    # global amax (converted to float32; works for scalar or batched amax), clamped from above at
    # the largest float32 (so a zero amax gives the maximum rather than infinity), and where the
    # result is exactly zero (infinite amax) replaced by 1.0. Done on device without host sync.
    # Python scalar operands are passed to device kernels without allocating
    # and copying scalar tensors. Both supported numerators (1536 and 2688)
    # are exactly representable in FP32.
    global_encode_scale = torch.div(float(nvfp4_e4m3_max) * FP4_E2M1_MAX, global_amax.to(torch.float32)).clamp_max(
        torch.finfo(torch.float32).max
    )
    # A Python conditional on a CUDA scalar synchronizes once per expert
    # quantization during weight publication. Keep the TE zero-scale rule on
    # device for scalar and batched amax alike.
    return torch.where(global_encode_scale == 0.0, 1.0, global_encode_scale)


def nvfp4_global_decode_scale_te(
    global_amax: torch.Tensor,
    nvfp4_e4m3_max: int = int(FP8_E4M3_MAX),
) -> torch.Tensor:
    # [FUNCTION DOC] Reciprocal of the global encode scale for the same amax and e4m3 max.
    return torch.div(1.0, nvfp4_global_encode_scale_te(global_amax, nvfp4_e4m3_max))


def _nvfp4_4over6_enabled() -> bool:
    # [FUNCTION DOC] HELPER: True when the 4over6 environment variable (stripped, case-insensitive)
    # equals weights or all.
    return os.getenv("NVTE_NVFP4_4OVER6", "").strip().lower() in ("weights", "all")


def _pad_rows_for_te_quantizer(weight: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Appends zero rows so the row count is a multiple of 16; returns the
    # input itself if already aligned. Expects a 2D tensor.
    pad_rows = (-weight.shape[0]) % TE_NVFP4_ROW_ALIGNMENT
    if pad_rows == 0:
        return weight
    padding = torch.zeros((pad_rows, weight.shape[1]), device=weight.device, dtype=weight.dtype)
    return torch.cat((weight, padding), dim=0)


def nvfp4_quantize_1d(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Quantizes a 2D weight with row-wise one-dimensional blocks of 16 (no column-wise
    # data, no random hadamard transform, no stochastic rounding, no row scaling, no amax
    # reduction). The 4over6 mode and its error mode (default mean absolute error; upper-cased) and
    # e4m3 max come from the environment helpers. Pads rows to alignment, quantizes, then strips
    # padding. Returns (packed data with last dimension halved, block scales viewed as float8 E4M3 with
    # last dimension divided by 16, global decode scale from the row-wise amax's first element).
    # KNOWN ISSUE: the number of columns is assumed divisible by 16; no check is made.
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    weight = weight.contiguous()
    num_rows, num_cols = weight.shape
    nvfp4_e4m3_max = nvfp4_weight_e4m3_max()

    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=_nvfp4_4over6_enabled(),
        nvfp4_e4m3_max=nvfp4_e4m3_max,
        nvfp4_4over6_err_mode=os.getenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE").strip().upper(),
        with_random_sign_mask=False,
    )

    quantized = quantizer.quantize(_pad_rows_for_te_quantizer(weight))
    qweight = quantized._rowwise_data[:num_rows, : num_cols // 2].contiguous()
    block_scale = quantized._rowwise_scale_inv[:num_rows, : num_cols // NVFP4_GROUP_SIZE].contiguous()
    amax = quantized._amax_rowwise.reshape(-1)[0]
    return qweight, block_scale.view(torch.float8_e4m3fn), nvfp4_global_decode_scale_te(amax, nvfp4_e4m3_max)


def nvfp4_quantize_1d_pair(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    # [FUNCTION DOC] Quantizes two 2D weights with the same column count together so that they share
    # one global scale. Raises value errors for non-2D inputs or mismatched columns. Rows are
    # combined (zero-copy view when the tensors are adjacent in the same storage, otherwise a
    # concatenation), quantized once, and split back at the first tensor's row count. Returns two
    # result triples (data, block scales, global decode scale), each with its own contiguous copies
    # (the scale is cloned).
    if first.dim() != 2 or second.dim() != 2:
        raise ValueError("nvfp4_quantize_1d_pair expects two 2D tensors.")
    if first.shape[1] != second.shape[1]:
        raise ValueError(
            f"NVFP4 paired quantization requires matching K dimensions, got {first.shape[1]} and {second.shape[1]}."
        )

    first_rows = first.shape[0]
    combined = _contiguous_pair_view(first, second)
    if combined is None:
        combined = torch.cat((first.contiguous(), second.contiguous()), dim=0)
    combined_qweight, combined_block_scale, global_scale = nvfp4_quantize_1d(combined)
    first_result = (
        combined_qweight[:first_rows].contiguous(),
        combined_block_scale[:first_rows].contiguous(),
        global_scale.clone(),
    )
    second_result = (
        combined_qweight[first_rows:].contiguous(),
        combined_block_scale[first_rows:].contiguous(),
        global_scale.clone(),
    )
    return first_result, second_result


def _contiguous_pair_view(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor | None:
    # [FUNCTION DOC] HELPER: If both tensors are contiguous with the same device, dtype and strides,
    # live in the same storage, and the second begins exactly where the first ends, returns one
    # strided view spanning both (rows stacked, no copy); otherwise none. A failure to create the
    # view (runtime error) also gives none.
    if not first.is_contiguous() or not second.is_contiguous():
        return None
    if first.device != second.device or first.dtype != second.dtype or first.stride() != second.stride():
        return None
    if first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr():
        return None
    if first.storage_offset() + first.numel() != second.storage_offset():
        return None

    try:
        return first.as_strided(
            (first.shape[0] + second.shape[0], first.shape[1]),
            first.stride(),
            first.storage_offset(),
        )
    except RuntimeError:
        return None
