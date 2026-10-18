# [MODULE DOC] CPU tests for chunked log-prob calculation numerics: only each chunk (not the whole
# logits) is upcast to fp32 before the kernel, the rollout temperature is applied in fp32 so
# half-precision and fp32 inputs agree within 1e-6, and the zero-valued graph placeholder sums in
# fp32 to avoid an fp16 overflow producing NaN. Runs for bf16 and fp16 model dtypes.
import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils.loss.hub import math_utils
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy

MODEL_DTYPES = [torch.bfloat16, torch.float16]


@pytest.mark.parametrize("model_dtype", MODEL_DTYPES)
def test_chunked_log_probs_upcast_only_each_chunk(monkeypatch, model_dtype):
    # [FUNCTION DOC] Test (parametrized over bf16 and fp16): replaces the kernel with a recorder;
    # computing log-probs and entropy for 5 rows of vocab 8 with chunk size 2 results in three
    # kernel calls with shapes (2,8), (2,8), (1,8), every one in fp32 (each chunk is upcast
    # separately).
    kernel_inputs = []

    def fake_compute_log_probs(
        logits, tokens, _tp_group, *, sampling_mask=None, debug_unified_grad_fused_logprob=False
    ):
        # [FUNCTION DOC] HELPER: Fake kernel; asserts no sampling mask and the debug switch off,
        # records the chunk's shape and dtype, returns zeros of shape (rows, 1) in the logits dtype.
        assert sampling_mask is None
        assert debug_unified_grad_fused_logprob is False
        kernel_inputs.append((logits.shape, logits.dtype))
        return torch.zeros((tokens.size(0), 1), dtype=logits.dtype)

    monkeypatch.setattr(math_utils, "compute_log_probs", fake_compute_log_probs)
    logits = torch.zeros((5, 8), dtype=model_dtype)
    tokens = torch.zeros(5, dtype=torch.long)

    math_utils.calculate_log_probs_and_entropy(logits, tokens, None, chunk_size=2)

    assert kernel_inputs == [
        (torch.Size([2, 8]), torch.float32),
        (torch.Size([2, 8]), torch.float32),
        (torch.Size([1, 8]), torch.float32),
    ]


@pytest.mark.parametrize("model_dtype", MODEL_DTYPES)
def test_temperature_is_applied_in_fp32(monkeypatch, model_dtype):
    # [FUNCTION DOC] Test (parametrized over bf16 and fp16): with the kernel replaced by a naive
    # log-softmax gather and temperature 0.7, chunk size 4, logits (1 by 12 by 32, normal times 10,
    # cast to the model dtype) give log-probs equal within 1e-6 absolute to those from the same
    # logits already cast to fp32 -- so no arithmetic (temperature scaling) happens before the fp32
    # upcast.
    def naive_compute_log_probs(
        logits, tokens, _tp_group, *, sampling_mask=None, debug_unified_grad_fused_logprob=False
    ):
        # [FUNCTION DOC] HELPER: Fake kernel; asserts no sampling mask and the debug switch off,
        # returns log-softmax over the vocab axis gathered at the tokens (trailing axis of size 1).
        assert sampling_mask is None
        assert debug_unified_grad_fused_logprob is False
        return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))

    monkeypatch.setattr(math_utils, "compute_log_probs", naive_compute_log_probs)
    make_parallel_state()
    args = make_args(true_on_policy_mode=False, rollout_temperature=0.7, log_probs_chunk_size=4)

    g = torch.Generator().manual_seed(0)
    logits = (torch.randn((1, 12, 32), generator=g) * 10).to(model_dtype)
    tokens = [torch.randint(0, 32, (12,), generator=g)]

    kwargs = dict(args=args, unconcat_tokens=tokens, total_lengths=[12], response_lengths=[6])
    out_model = get_log_probs_and_entropy(logits, **kwargs)
    out_fp32 = get_log_probs_and_entropy(logits.float(), **kwargs)

    # Model-precision and fp32 inputs carry identical information, so any
    # difference comes from arithmetic done before the fp32 upcast.
    torch.testing.assert_close(out_model["log_probs"][0], out_fp32["log_probs"][0], rtol=0, atol=1e-6)


def test_graph_placeholder_sums_in_fp32():
    # [FUNCTION DOC] Test: a half-precision vector of 70000 twos overflows to infinity when summed
    # in fp16, while zero times the fp32 sum is exactly zero (finite); documents why the graph
    # placeholder must sum in fp32. It does not call project code, only checks the arithmetic.
    # An fp16 sum overflows to inf and 0 * inf is nan; the fp32 sum stays finite.
    logits = torch.full((70_000,), 2.0, dtype=torch.float16)

    assert torch.isinf(logits.sum())
    placeholder = 0 * logits.sum(dtype=torch.float32)
    assert placeholder == 0
