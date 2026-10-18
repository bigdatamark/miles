# [MODULE DOC]
# GPU-gated tests (skipped when CUDA is unavailable) for the FSDP2 helper that materializes a
# full parameter from a possibly sharded distributed tensor. Only the plain-tensor contract is
# tested here: the result is a non-distributed CUDA tensor equal to the input, and requesting an
# asynchronous operation returns a plain result (no wait handle) for plain tensors.
"""Unit tests for the FSDP2 DTensor materialization primitive.

``gather_full_param`` always moves to CUDA first (a CPU DTensor picks the wrong collective
backend), so these are GPU-gated. The sharded-DTensor all-gather path is exercised end to
end by the per-model weight-export (lp_diff) runs; here we pin the non-DTensor contract.
"""

import pytest
import torch

from miles.backends.fsdp_utils.dtensor import gather_full_param


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gather_full_param moves to CUDA")
def test_non_dtensor_passthrough_to_cuda():
    # [FUNCTION DOC] A 3-by-4 float32 CPU tensor is gathered; asserts the result is not a
    # distributed tensor, lives on CUDA, and equals the input when moved back to CPU.
    p = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = gather_full_param(p)
    assert not isinstance(out, torch.distributed.tensor.DTensor)
    assert out.is_cuda
    torch.testing.assert_close(out.cpu(), p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gather_full_param moves to CUDA")
def test_async_passthrough_is_noop_for_non_dtensor():
    # [FUNCTION DOC] Gathering a 2-by-2 ones CPU tensor with the async option on returns a result
    # without a wait method, whose CPU copy equals the input.
    # A plain tensor has no async handle to drain; async_op must not change the result.
    p = torch.ones(2, 2)
    out = gather_full_param(p, async_op=True)
    assert not hasattr(out, "wait")
    torch.testing.assert_close(out.cpu(), p)
