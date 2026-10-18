# [MODULE DOC] CPU test that the logit processor's log-prob/entropy extraction forwards the debug
# unified-grad fused log-prob switch from training args to the fused cross entropy kernel (using
# the recording fake fixture and a fake single-rank parallel state).
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.loss.hub import logit_processors


class TestGetLogProbsAndEntropy:
    # [CLASS DOC] Tests for the log-prob and entropy extraction function.
    def test_stored_log_probs_forward_the_debug_grad_switch_from_training_args(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fused_cross_entropy_calls: list[tuple[bool, bool]],
    ) -> None:
        # [FUNCTION DOC] Test: with the parallel state patched to one tensor-parallel and one
        # context-parallel rank (no group), packed format, temperature 1, not true-on-policy, no
        # chunking, no allgather, and the debug switch on, extraction runs under no-grad on random
        # logits of shape 1 by 4 by 8 with a sequence of 4 tokens and response length 3. The fake
        # kernel is called exactly once with gradient mode enabled and logits requiring grad (the
        # training contract), yet the returned stored log-probs do not require grad.
        """Stored log probabilities forward the debug grad contract from training arguments."""
        parallel_state = SimpleNamespace(
            tp=SimpleNamespace(rank=0, group=None),
            cp=SimpleNamespace(rank=0, size=1),
        )
        monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
        args = Namespace(
            qkv_format="thd",
            rollout_temperature=1.0,
            true_on_policy_mode=False,
            log_probs_chunk_size=-1,
            allgather_cp=False,
            debug_unified_grad_fused_logprob=True,
        )

        with torch.no_grad():
            result = logit_processors.get_log_probs_and_entropy(
                torch.randn(1, 4, 8),
                args=args,
                unconcat_tokens=[torch.tensor([0, 1, 2, 3])],
                total_lengths=[4],
                response_lengths=[3],
            )

        assert fused_cross_entropy_calls == [(True, True)]
        assert not result["log_probs"][0].requires_grad
