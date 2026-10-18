# [MODULE DOC] CPU tests for the log-prob computation function's fused cross entropy execution
# contract: without the debug unified-grad switch stored (no-grad) log-probs run the kernel in the
# inference mode (no grad mode, logits not requiring grad); with the switch on they run the
# training contract but still return tensors without autograd graph; real training log-probs keep
# their graph and the training contract regardless of the switch and can be backpropagated. A local
# helper records kernel calls (this file does not use the shared fixture).
from __future__ import annotations

import pytest
import torch
from megatron.core.fusions import fused_cross_entropy

from miles.backends.training_utils.loss.hub.math_utils import compute_log_probs


def _record_fused_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[bool, bool]]:
    # [FUNCTION DOC] HELPER: Replaces the Megatron fused vocab-parallel cross entropy with a
    # recording fake (same behaviour as the shared fixture: records (gradient mode enabled, logits
    # require grad), returns the vocab-axis sum times zero). Returns the live list of records.
    calls: list[tuple[bool, bool]] = []

    def fake_fused_cross_entropy(
        logits: torch.Tensor,
        _tokens: torch.Tensor,
        _process_group: object,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: The fake kernel; appends the recording pair and returns the
        # logits' vocab-axis sum multiplied by zero.
        calls.append((torch.is_grad_enabled(), logits.requires_grad))
        return logits.sum(dim=-1) * 0

    monkeypatch.setattr(
        fused_cross_entropy,
        "fused_vocab_parallel_cross_entropy",
        fake_fused_cross_entropy,
    )
    return calls


def test_stored_log_probs_keep_the_inference_contract_while_the_switch_is_off(monkeypatch) -> None:
    # [FUNCTION DOC] Test: under no-grad with the switch off, one kernel call is recorded with
    # gradient mode off and logits not requiring grad; the result does not require grad.
    """Without the switch, stored log probs run the fused kernel exactly as inference always did."""
    calls = _record_fused_calls(monkeypatch)
    logits = torch.randn(3, 8)
    tokens = torch.tensor([1, 2, 3])

    with torch.no_grad():
        log_probs = compute_log_probs(logits, tokens, process_group=None)

    assert calls == [(False, False)]
    assert not log_probs.requires_grad


def test_stored_log_probs_use_the_training_fused_execution_contract_under_the_switch(monkeypatch) -> None:
    # [FUNCTION DOC] Test: under no-grad with the switch on, one kernel call is recorded with
    # gradient mode on and logits requiring grad (training contract), while the returned log-probs
    # do not require grad.
    """With the switch on, stored log probs take the training fused contract but return no autograd graph."""
    calls = _record_fused_calls(monkeypatch)
    logits = torch.randn(3, 8)
    tokens = torch.tensor([1, 2, 3])

    with torch.no_grad():
        log_probs = compute_log_probs(logits, tokens, process_group=None, debug_unified_grad_fused_logprob=True)

    assert calls == [(True, True)]
    assert not log_probs.requires_grad


@pytest.mark.parametrize("switched_on", [False, True])
def test_training_log_probs_keep_the_existing_autograd_contract(monkeypatch, switched_on: bool) -> None:
    # [FUNCTION DOC] Test (parametrized over the switch): with grad mode on and logits requiring
    # grad, one kernel call is recorded with gradient mode on and logits requiring grad; the result
    # requires grad, and backpropagating its sum leaves a gradient on the logits.
    """Training log probs retain their input graph and fused execution contract on either side of the switch."""
    calls = _record_fused_calls(monkeypatch)
    logits = torch.randn(3, 8, requires_grad=True)
    tokens = torch.tensor([1, 2, 3])

    log_probs = compute_log_probs(logits, tokens, process_group=None, debug_unified_grad_fused_logprob=switched_on)

    assert calls == [(True, True)]
    assert log_probs.requires_grad
    log_probs.sum().backward()
    assert logits.grad is not None
