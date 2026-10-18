# [MODULE DOC] CPU test that the combined log-prob and entropy calculation propagates the debug
# unified-grad contract to the fused cross entropy kernel, both without chunking and with chunking
# (one kernel call per chunk), using the recording fake fixture.
import pytest
import torch

from miles.backends.training_utils.loss.hub.math_utils import calculate_log_probs_and_entropy


class TestCalculateLogProbsAndEntropy:
    # [CLASS DOC] Tests for the combined log-prob and entropy calculation.
    @pytest.mark.parametrize(("chunk_size", "expected_calls"), [(-1, 1), (2, 2)])
    def test_calculate_log_probs_propagates_the_debug_grad_contract_with_and_without_chunking(
        self,
        chunk_size: int,
        expected_calls: int,
        fused_cross_entropy_calls: list[tuple[bool, bool]],
    ) -> None:
        # [FUNCTION DOC] Test (parametrized: chunk size -1 expects 1 kernel call, chunk size 2 over
        # 3 rows expects 2): under no-grad, with random logits of 3 rows by 8, tokens 1,2,3, no
        # process group and the debug switch on, the kernel is called the expected number of times,
        # each with gradient mode enabled and logits requiring grad; the returned log-probs do not
        # require grad and the entropy is none (not requested).
        """Debug stored log probabilities use the grad contract with and without chunking."""
        with torch.no_grad():
            log_probs, entropy = calculate_log_probs_and_entropy(
                torch.randn(3, 8),
                torch.tensor([1, 2, 3]),
                None,
                chunk_size=chunk_size,
                debug_unified_grad_fused_logprob=True,
            )

        assert fused_cross_entropy_calls == [(True, True)] * expected_calls
        assert not log_probs.requires_grad
        assert entropy is None
