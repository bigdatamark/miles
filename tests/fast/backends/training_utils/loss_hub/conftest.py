# [MODULE DOC] Pytest fixtures shared by the loss hub tests: a fixture that replaces the Megatron
# fused vocab-parallel cross entropy with a recording fake.
from collections.abc import Callable

import pytest
import torch
from megatron.core.fusions import fused_cross_entropy


@pytest.fixture
def fused_cross_entropy_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[bool, bool]]:
    # [FUNCTION DOC] HELPER: Fixture. Replaces the fused vocab-parallel cross entropy in the
    # Megatron module with a fake that records, per call, a pair (is gradient mode globally enabled,
    # does the logits tensor require grad) and returns zeros shaped like logits summed over the
    # vocab axis (keeps the autograd link to logits if it requires grad). Returns the live list of
    # recorded pairs; the patch is undone at test teardown.
    calls: list[tuple[bool, bool]] = []

    def fake_fused_cross_entropy(
        logits: torch.Tensor,
        tokens: torch.Tensor,
        process_group: object,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: The fake kernel; discards tokens and process group, appends the
        # recording pair and returns the logits' vocab-axis sum multiplied by zero.
        del tokens, process_group
        calls.append((torch.is_grad_enabled(), logits.requires_grad))
        return logits.sum(dim=-1) * 0

    fake: Callable[[torch.Tensor, torch.Tensor, object], torch.Tensor] = fake_fused_cross_entropy
    monkeypatch.setattr(fused_cross_entropy, "fused_vocab_parallel_cross_entropy", fake)
    return calls
