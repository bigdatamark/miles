# [MODULE DOC] GPU tests for MXFP8 (8-bit float with 32-element block scales) weight quantization
# used when converting or updating weights: the Megatron-to-HF processor path and the standalone
# HF conversion tool are both compared bit for bit against Transformer Engine's MXFP8 quantizer over
# many shapes and four data patterns, plus tests that the processor skips quantization for
# configured high-precision layers (extra layers, first/last layers kept in bf16) and that the tool's
# should-quantize predicate honors skip substrings. Registered but currently disabled in CI with a
# FIXME reason (no tracking issue cited, unlike the CI registration rule).
# KNOWN ISSUE: the disabled reason is a FIXME without an issue or PR reference.
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="stage-b-2-gpu-h200",
    labels=["precision"],
    hardware=["hopper"],
    disabled="FIXME: re-enable after the MXFP8 H200 reference path is settled.",
)


import pytest
import torch
from tools.convert_hf_to_mxfp8 import quantize_mxfp8 as tool_quantize_mxfp8
from tools.convert_hf_to_mxfp8 import should_quantize as tool_should_quantize_mxfp8
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.constants import TE_DType

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8 import (
    _quantize_param as processor_quantize_mxfp8_param,
)
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8 import quantize_params_mxfp8

MXFP8_GROUP_SIZE = 32
MXFP8_SHAPES = [
    (1, 64),
    (1, 1024),
    (3, 128),
    (16, 64),
    (64, 128),
    (128, 64),
    (256, 128),
    (512, 256),
    (128, 1024),
    (1024, 2048),
    (7168, 2048),
    (2048, 7168),
    (128, 16384),
]


def _make_weight(init_data: str, dtype: torch.dtype, shape: tuple[int, int], device: str) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Builds a test weight of the given (rows, columns) shape, dtype and
    # device by pattern name: "random" is 16 times standard normal; "boundary" is a single row (repeated
    # for all rows) of a linspace from -512 to 512 with n/2 points, each point emitted twice as value
    # minus and plus a tiny epsilon (1e-3, floored at 1e-4) at even/odd columns, to sit near rounding
    # boundaries; "zeros"; "maxes" (every element the dtype's largest finite value). Raises a value
    # error for an unknown pattern. Columns for the boundary pattern must be even.
    m, n = shape
    if init_data == "random":
        return 16 * torch.randn((m, n), dtype=dtype, device=device)
    if init_data == "boundary":
        base = torch.linspace(-512.0, 512.0, steps=n // 2, dtype=torch.float32, device=device)
        eps = torch.full_like(base, 1e-3)
        eps = torch.maximum(eps, 1e-4 * torch.ones_like(base))
        row = torch.empty(n, dtype=torch.float32, device=device)
        row[0::2] = base - eps
        row[1::2] = base + eps
        return row.unsqueeze(0).repeat(m, 1).to(dtype=dtype)
    if init_data == "zeros":
        return torch.zeros((m, n), dtype=dtype, device=device)
    if init_data == "maxes":
        return torch.full((m, n), torch.finfo(dtype).max, dtype=dtype, device=device)
    raise ValueError(f"Unknown init_data: {init_data}")


def _processor_quantize_mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Adapter that runs the processor's per-parameter quantizer on a weight
    # under a fixed expert down-projection parameter name and returns the pair (quantized weight, scale
    # inverse) looked up from the resulting name-to-tensor pairs.
    quantized = dict(processor_quantize_mxfp8_param("model.layers.0.mlp.experts.0.down_proj.weight", weight))
    return (
        quantized["model.layers.0.mlp.experts.0.down_proj.weight"],
        quantized["model.layers.0.mlp.experts.0.down_proj.weight_scale_inv"],
    )


def _te_mxfp8_reference(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Reference quantization using Transformer Engine's MXFP8 quantizer
    # (float8 e4m3, row-wise only). If the row count is not a multiple of 32 the weight is zero-padded
    # on rows up to the next multiple before quantizing (the engine needs it). Returns the row-wise data
    # and the row-wise inverse scales, both sliced back to the original row count, the scales also to
    # columns divided by 32, made contiguous.
    weight = weight.contiguous()
    m, k = weight.shape
    if m % MXFP8_GROUP_SIZE != 0:
        padded_m = ((m + MXFP8_GROUP_SIZE - 1) // MXFP8_GROUP_SIZE) * MXFP8_GROUP_SIZE
        padded_weight = torch.zeros((padded_m, k), dtype=weight.dtype, device=weight.device)
        padded_weight[:m].copy_(weight)
    else:
        padded_weight = weight

    quantizer = MXFP8Quantizer(fp8_dtype=TE_DType[torch.float8_e4m3fn], rowwise=True, columnwise=False)
    quantized = quantizer(padded_weight)
    return (
        quantized._rowwise_data[:m].contiguous(),
        quantized._rowwise_scale_inv[:m, : k // MXFP8_GROUP_SIZE].contiguous(),
    )


def test_mxfp8_quantize_params_respects_extra_high_precision_layers_megatron():
    # [FUNCTION DOC] With a small bfloat16 weight, a single converted parameter pair, and an args
    # object listing the fc2 layer name among extra high-precision layers, asserts the processor's
    # parameter quantizer returns the very same list object unchanged (nothing quantized) for a
    # matching Megatron parameter name.
    weight = torch.randn((4, MXFP8_GROUP_SIZE), dtype=torch.bfloat16)
    converted_named_params = [
        ("model.layers.0.mlp.experts.0.down_proj.weight", weight),
    ]
    args = type("Args", (), {"extra_high_precision_layers_megatron": ("linear_fc2",)})()

    out = quantize_params_mxfp8(
        args=args,
        megatron_name="decoder.layers.0.mlp.experts.linear_fc2.weight0",
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "mxfp8"},
    )

    assert out is converted_named_params


@pytest.mark.parametrize("layer_idx", [0, 3])
def test_mxfp8_quantize_params_respects_first_last_layers_bf16(layer_idx):
    # [FUNCTION DOC] Parametrized over the first (0) and last (3) of 4 layers, with the first/last
    # bf16 option on and one layer kept in bf16 at each end. Asserts the processor returns the original
    # list unchanged for those layers.
    weight = torch.randn((4, MXFP8_GROUP_SIZE), dtype=torch.bfloat16)
    converted_named_params = [
        ("model.layers.0.mlp.experts.0.down_proj.weight", weight),
    ]
    args = type(
        "Args",
        (),
        {
            "first_last_layers_bf16": True,
            "num_layers": 4,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
        },
    )()

    out = quantize_params_mxfp8(
        args=args,
        megatron_name=f"decoder.layers.{layer_idx}.mlp.experts.linear_fc2.weight0",
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "mxfp8"},
    )

    assert out is converted_named_params


def test_mxfp8_hf_should_quantize_respects_extra_high_precision_layers_hf():
    # [FUNCTION DOC] Asserts the conversion tool's should-quantize predicate returns false when the
    # parameter name contains one of the skip substrings, and true when it does not.
    weight = torch.randn((4, MXFP8_GROUP_SIZE), dtype=torch.bfloat16)

    assert not tool_should_quantize_mxfp8(
        "model.layers.0.mlp.experts.0.down_proj.weight",
        weight,
        skip_weight_substrings=("mlp.experts.0",),
    )
    assert tool_should_quantize_mxfp8(
        "model.layers.0.mlp.experts.0.down_proj.weight",
        weight,
        skip_weight_substrings=("mlp.experts.1",),
    )


@pytest.mark.parametrize(
    "quantize_fn",
    [_processor_quantize_mxfp8, tool_quantize_mxfp8],
    ids=["processor", "convert_tool"],
)
@pytest.mark.parametrize("shape", MXFP8_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
def test_mxfp8_quantize_matches_reference(quantize_fn, shape, dtype, init_data):
    # [FUNCTION DOC] Parametrized over the processor adapter and the tool function, 13 shapes,
    # bfloat16, and four data patterns. With seed 42, quantizes a GPU weight and compares to the
    # Transformer Engine reference. Asserts: quantized weight has the weight's shape and float8 e4m3
    # dtype; scale has shape (leading dims, columns / 32) and uint8 dtype; both match the reference
    # bit-for-bit (compared as unsigned bytes, zero tolerance).
    device = "cuda"
    torch.manual_seed(42)

    weight = _make_weight(init_data, dtype, shape, device)
    qweight, scale = quantize_fn(weight)
    qweight_ref, scale_ref = _te_mxfp8_reference(weight)

    assert qweight.shape == weight.shape
    assert qweight.dtype == torch.float8_e4m3fn
    assert scale.shape == (*weight.shape[:-1], weight.shape[-1] // MXFP8_GROUP_SIZE)
    assert scale.dtype == torch.uint8
    torch.testing.assert_close(qweight.view(dtype=torch.uint8), qweight_ref.view(dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
