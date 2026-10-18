# [MODULE DOC]
# Adapter exposing fused NVFP4 fake quantization-aware training to megatron's grouped linear layers,
# enabled by an environment flag (the module constant names it). Kernel code is imported lazily so
# processes that do not enable it never import it.
"""Megatron-facing adapter for fused NVFP4 fake QAT."""

from __future__ import annotations

import os

import torch

NVFP4_FAKE_QAT_FLAG = "OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG"


def maybe_fake_quantize_nvfp4_weight_tensors(
    weight_tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    # [FUNCTION DOC] If the flag environment variable (default "0") is not exactly "1", returns the
    # input list unchanged. Otherwise fetches the current quantize-dequantize config and returns a new
    # list with each weight passed through the fake quantization op with a straight-through gradient
    # estimator.
    """Apply env-gated fused NVFP4 fake QAT to discrete TE grouped-linear weights."""
    if os.getenv(NVFP4_FAKE_QAT_FLAG, "0") != "1":
        return weight_tensors

    # Keep CuTe DSL optional for every process that does not enable this path.
    from miles.kernels.quant.nvfp4_qdq import current_nvfp4_qdq_config, fake_nvfp4_quantization_ste

    qdq_config = current_nvfp4_qdq_config()
    return [fake_nvfp4_quantization_ste(weight, qdq_config) for weight in weight_tensors]


__all__ = ["NVFP4_FAKE_QAT_FLAG", "maybe_fake_quantize_nvfp4_weight_tensors"]
