# [MODULE DOC] GPU tests for the hyper-connection mix and aggregate Triton ops, compared forward and
# backward with float64 torch references. Uses 4 residual streams, sequence 37, batch 2, hidden sizes
# 1000 (not a power of two) and 2048, bfloat16 data and float32 mixing coefficients. Registered for a
# 2-GPU Hopper suite with the precision label; skipped without Triton or CUDA.
"""The DeepSeek-V4.1 hyper-connection residual mix and stream aggregation Triton kernels against
float64 torch references, forward and backward."""

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.hyper_connection.mhc import mhc_aggregate, mhc_mix  # noqa: E402

NUM_STREAMS = 4


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    # [FUNCTION DOC] HELPER: Returns the Frobenius norm of the difference (actual converted to
    # float64) divided by the norm of expected, as a Python float.
    return ((actual.double() - expected).norm() / expected.norm()).item()


def _mix_reference(x, residual, h_res, h_post, n):
    # [FUNCTION DOC] HELPER: Reference residual mix. The residual (seq, batch, n*hidden) is viewed as
    # n streams; output stream j is the sum over input streams i of the residual mixing weight
    # h_res[i, j] times stream i, plus h_post[j] times the layer output x (seq, batch, hidden).
    # Returns a tensor shaped (seq, batch, n*hidden). Differentiable torch ops in the input dtype.
    s, b, _ = residual.shape
    streams = residual.view(s, b, n, -1)
    mixed = torch.einsum("sbij,sbid->sbjd", h_res, streams)
    return (h_post.unsqueeze(-1) * x.unsqueeze(2) + mixed).view(s, b, -1)


def _aggregate_reference(x, pre, n):
    # [FUNCTION DOC] HELPER: Reference stream aggregation. x (seq, batch, n*hidden) is viewed as n
    # streams, each scaled by its pre weight (seq, batch, n) and summed over streams, giving
    # (seq, batch, hidden).
    s, b, nc = x.shape
    return (pre.unsqueeze(-1) * x.view(s, b, n, nc // n)).sum(dim=2)


def _leaves(*tensors):
    # [FUNCTION DOC] HELPER: Given tensors, returns two lists of fresh gradient-requiring leaves: clones
    # in the original dtype for the kernel, and float64 copies for the reference.
    kernel = [t.clone().requires_grad_() for t in tensors]
    reference = [t.double().requires_grad_() for t in tensors]
    return kernel, reference


@pytest.mark.parametrize("hidden", [1000, 2048])
def test_mix_matches_float64_reference(hidden):
    # [FUNCTION DOC] Runs the mix kernel on random inputs and the float64 reference. Asserts output
    # dtype equals the residual dtype and relative error below 5e-3; then backpropagates the same random
    # gradient through both and asserts every input gradient keeps its input dtype (kernel side) and has
    # relative error below 5e-3 versus the reference.
    torch.manual_seed(0)
    seq, batch, n = 37, 2, NUM_STREAMS
    x = torch.randn(seq, batch, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(seq, batch, n * hidden, device="cuda", dtype=torch.bfloat16)
    h_res = torch.rand(seq, batch, n, n, device="cuda", dtype=torch.float32)
    h_post = torch.rand(seq, batch, n, device="cuda", dtype=torch.float32)
    grad = torch.randn(seq, batch, n * hidden, device="cuda", dtype=torch.bfloat16)

    kernel, reference = _leaves(x, residual, h_res, h_post)
    out = mhc_mix(*kernel, n)
    expected = _mix_reference(*reference, n)
    assert out.dtype == residual.dtype
    assert _rel_err(out, expected) < 5e-3

    out.backward(grad)
    expected.backward(grad.double())
    for actual, ref in zip(kernel, reference, strict=True):
        assert actual.grad.dtype == actual.dtype
        assert _rel_err(actual.grad, ref.grad) < 5e-3


@pytest.mark.parametrize("hidden", [1000, 2048])
def test_aggregate_matches_float64_reference(hidden):
    # [FUNCTION DOC] Same structure as the mix test for the aggregate kernel: asserts output dtype
    # equals input dtype, forward relative error below 5e-3, and gradient relative errors below 5e-3
    # for the stream tensor and the pre weights.
    torch.manual_seed(0)
    seq, batch, n = 37, 2, NUM_STREAMS
    x = torch.randn(seq, batch, n * hidden, device="cuda", dtype=torch.bfloat16)
    pre = torch.rand(seq, batch, n, device="cuda", dtype=torch.float32)
    grad = torch.randn(seq, batch, hidden, device="cuda", dtype=torch.bfloat16)

    kernel, reference = _leaves(x, pre)
    out = mhc_aggregate(*kernel, n)
    expected = _aggregate_reference(*reference, n)
    assert out.dtype == x.dtype
    assert _rel_err(out, expected) < 5e-3

    out.backward(grad)
    expected.backward(grad.double())
    for actual, ref in zip(kernel, reference, strict=True):
        assert _rel_err(actual.grad, ref.grad) < 5e-3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
