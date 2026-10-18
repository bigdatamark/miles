# [MODULE DOC] GPU tests for the block fake-quantization Triton kernels (quantize then dequantize
# in the original dtype): fp8, fp4 and compressed-kv variants. A torch reference in this file
# reproduces each variant's scale and rounding rules; the tests require exact equality (zero
# tolerance) for bfloat16 and float32 inputs, and check the backward is a straight-through estimator.
# Registered for a 2-GPU Hopper suite with the precision label; skipped without Triton or CUDA.
"""The block fake-quant Triton kernels against a torch reference: fp8 and fp4 with power-of-two
scales over 32-wide blocks, fp4 with e4m3 scales over 16-wide blocks, and the straight-through
backward."""

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.quant.fake_quant import fake_quant_compressed_kv, fake_quant_fp4, fake_quant_fp8  # noqa: E402

FP8_MAX = 448.0
FP4_MAX = 6.0


def _ceil_pow2(value: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Rounds each positive element up to the next power of two (exact powers of
    # two map to themselves), using a mantissa/exponent decomposition. Returns a tensor of the same
    # shape and dtype.
    mantissa, exponent = torch.frexp(value)
    return torch.ldexp(torch.ones_like(value), torch.where(mantissa == 0.5, exponent - 1, exponent))


def _round_to_fp4_grid(scaled: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Rounds values (already scaled into [-6, 6]) to the 4-bit float grid
    # {0, 0.5, 1, 1.5, 2, 3, 4, 6} with sign: the step is 0.5 below magnitude 2, 1 below 4, else 2;
    # magnitude divided by step is rounded with torch's round-half-to-even and multiplied back.
    magnitude = scaled.abs()
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    return torch.round(magnitude / step) * step * torch.sign(scaled)


def _reference(x: torch.Tensor, kind: str) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Reference fake quantization in float32 over contiguous blocks of the
    # flattened tensor (block 16 for the compressed-kv kind, 32 otherwise; the element count must be
    # divisible by the block). Per block, with amax the max magnitude: compressed-kv uses scale =
    # amax / 6 clamped to [2^-9, 448] and rounded through float8 e4m3, then rounds the scaled values to
    # the fp4 grid; fp4 uses a power-of-two scale ceil(amax / 6) with amax floored at 6 * 2^-126;
    # fp8 (the default branch) uses a power-of-two scale ceil(amax / 448) with amax floored at 1e-4, and
    # casts scaled values clamped to +-448 through float8 e4m3. Result is quantized times scale,
    # reshaped and cast back to the input dtype.
    block = 16 if kind == "compressed_kv" else 32
    blocks = x.float().reshape(-1, block)
    amax = blocks.abs().amax(dim=1, keepdim=True)
    if kind == "compressed_kv":
        scale = (amax / FP4_MAX).clamp(2.0**-9, FP8_MAX).to(torch.float8_e4m3fn).float()
        scaled = (blocks / scale).clamp(-FP4_MAX, FP4_MAX)
        quantized = _round_to_fp4_grid(scaled)
    elif kind == "fp4":
        scale = _ceil_pow2(amax.clamp(min=6 * 2.0**-126) / FP4_MAX)
        quantized = _round_to_fp4_grid((blocks / scale).clamp(-FP4_MAX, FP4_MAX))
    else:
        scale = _ceil_pow2(amax.clamp(min=1e-4) / FP8_MAX)
        quantized = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape(x.shape).to(x.dtype)


KERNELS = {"fp8": fake_quant_fp8, "fp4": fake_quant_fp4, "compressed_kv": fake_quant_compressed_kv}


@pytest.mark.parametrize("kind", list(KERNELS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fake_quant_matches_reference(kind, dtype):
    # [FUNCTION DOC] For each kernel kind and dtype (bfloat16, float32): builds a (3, 37, 512) tensor
    # of normal values scaled per channel by a log-spaced factor from 1e-3 to 1e3 (wide dynamic
    # range), zeroes one 32-wide block (to exercise the amax floor), and asserts the kernel output
    # equals the reference exactly.
    torch.manual_seed(0)
    x = torch.randn(3, 37, 512, device="cuda", dtype=dtype) * torch.logspace(-3, 3, 512, device="cuda").to(dtype)
    x[0, 0, :32] = 0
    torch.testing.assert_close(KERNELS[kind](x), _reference(x, kind), rtol=0, atol=0)


@pytest.mark.parametrize("kind", list(KERNELS))
def test_fake_quant_backward_is_straight_through(kind):
    # [FUNCTION DOC] For each kernel kind: applies it to a bfloat16 (64, 256) leaf, backpropagates a
    # random gradient and asserts the input gradient equals the upstream gradient exactly.
    torch.manual_seed(0)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn_like(x)
    KERNELS[kind](x).backward(grad)
    torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
