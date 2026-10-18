# [MODULE DOC] GPU tests for the head-sharded short causal convolution used by linear attention
# (sharded short conv class in the plugins linear_attn module): that it matches a float64 torch
# reference (with silu activation, per packed sequence) on each available backend, and that its
# chunking of channels, applied when the input exceeds the 32-bit indexing limit of the underlying
# kernels, gives the same forward result bit for bit and nearly identical gradients to a single call.
# Registered for the 2-GPU Hopper suite with the precision label; skipped without fla or CUDA.
"""The head-sharded short conv splits inputs past fla's int32 indexing limit into channel chunks. The
conv is per channel, so the chunked forward matches the single call bit for bit. fla picks its backward
tiling from the channel count, so on Blackwell the gradients differ from the single call by fp32
accumulation order (~1e-6 relative); a wrong channel mapping is off by ~1. The conv, on whichever
backend the layer selects, also matches a float64 torch reference to bf16 rounding (< 5e-3)."""

import sys

import pytest
import torch
import torch.nn.functional as F
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

pytest.importorskip("fla")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles_plugins.models import linear_attn  # noqa: E402


def _conv(channels):
    # [FUNCTION DOC] HELPER: Creates the sharded short convolution with the given channel count,
    # kernel size 4, no tensor-parallel group, on the GPU in bfloat16.
    return linear_attn.ShardedShortConv(channels, 4, tp_group=None, device="cuda", dtype=torch.bfloat16)


def _reference(x, weight, cu_seqlens):
    # [FUNCTION DOC] HELPER: Float64 reference depthwise causal convolution with silu. Input x is
    # (batch, tokens, channels), weight (channels, kernel). Transposes to channels-first, then for each
    # packed segment (boundaries from the cumulative lengths) left-pads by kernel minus one zeros,
    # applies a grouped convolution with one group per channel and silu, concatenates segments along
    # tokens and transposes back to (batch, tokens, channels). Differentiable.
    x = x.double().transpose(1, 2)
    weight = weight.double()
    segments = [
        F.silu(F.conv1d(F.pad(x[..., start:end], (weight.shape[-1] - 1, 0)), weight, groups=weight.shape[0]))
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True)
    ]
    return torch.cat(segments, dim=-1).transpose(1, 2)


@pytest.mark.parametrize("backend", sorted({"triton", linear_attn.short_conv_backend()}))
def test_conv_matches_float64_reference(backend):
    # [FUNCTION DOC] Parametrized over the triton backend and the backend the layer selects by
    # default (deduplicated). Uses 2048 tokens in two segments (700 and 1348), 768 channels. Runs the
    # conv forward/backward with the backend set explicitly and the float64 reference, and asserts the
    # relative error (norm ratio) is below 5e-3 for the output, the input gradient and the weight
    # gradient (the conv's own forward/backward is run once with gradients, and its output is then
    # recomputed without gradient tracking for the output comparison).
    torch.manual_seed(0)
    tokens, channels = 2048, 768
    conv = _conv(channels)
    conv.backend = backend
    x = torch.randn(1, tokens, channels, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, 700, tokens], device="cuda", dtype=torch.int32)
    xi = x.clone().requires_grad_()
    conv(xi, cu_seqlens=cu_seqlens).backward(grad)
    xr = x.double().requires_grad_()
    wr = conv.weight.detach().double().requires_grad_()
    out_ref = _reference(xr, wr, cu_seqlens)
    out_ref.backward(grad.double())
    out = conv(x, cu_seqlens=cu_seqlens)
    for actual, expected in ((out, out_ref), (xi.grad, xr.grad), (conv.weight.grad, wr.grad)):
        assert ((actual.double() - expected).norm() / expected.norm()).item() < 5e-3


@pytest.mark.parametrize("channels", [768, 1000])
def test_chunked_conv_matches_single_call(monkeypatch, channels):
    # [FUNCTION DOC] Parametrized over 768 and 1000 channels (the latter does not divide evenly into
    # chunks), 2048 tokens in two segments (700 and 1348). A nested helper runs forward and backward
    # (clearing the weight gradient first) and returns the detached output, input gradient and weight
    # gradient copy. First runs it unchunked, then patches the module's 32-bit element limit and chunk
    # element size to 2048 x 300 elements so the channels are split into chunks, and runs again.
    # Asserts the outputs are exactly equal and that the input and weight gradients differ by less than
    # 1e-4 relative norm.
    torch.manual_seed(0)
    tokens = 2048
    conv = _conv(channels)
    x = torch.randn(1, tokens, channels, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, 700, tokens], device="cuda", dtype=torch.int32)

    def run():
        # [FUNCTION DOC] HELPER: Resets the weight gradient, runs the conv on a gradient-requiring
        # clone of the input and backpropagates the fixed random gradient. Returns a triple: detached
        # output, input gradient and a copy of the weight gradient.
        conv.weight.grad = None
        xi = x.clone().requires_grad_()
        out = conv(xi, cu_seqlens=cu_seqlens)
        out.backward(grad)
        return out.detach(), xi.grad, conv.weight.grad.clone()

    whole = run()
    monkeypatch.setattr(linear_attn, "INT32_ELEMENTS", tokens * 300)
    monkeypatch.setattr(linear_attn, "_CHUNK_ELEMENTS", tokens * 300)
    chunked = run()
    assert torch.equal(whole[0], chunked[0])
    for a, b in zip(whole[1:], chunked[1:], strict=True):
        assert ((a.float() - b.float()).norm() / b.float().norm()).item() < 1e-4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
