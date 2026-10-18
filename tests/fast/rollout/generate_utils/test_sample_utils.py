# [MODULE DOC] Tests of merging consecutive turn samples of one trajectory into a single sample
# (pair merge and list merge). A mock tokenizer whose decode returns a marker string containing the
# token list is used so that the text inserted for observation tokens is predictable. Samples are
# built with a local factory having default prompt, label, reward, index and group index. In a
# valid pair, the later sample's tokens start with the earlier sample's tokens; the extra tokens
# before the later response are observation tokens (loss mask 0, log probability 0).
from unittest.mock import MagicMock

import numpy
import pytest

from miles.rollout.generate_utils.sample_utils import _merge_sample_pair, merge_samples
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


@pytest.fixture
def mock_tokenizer():
    # [FUNCTION DOC] HELPER: Fixture returning a mock tokenizer whose decode turns a token list
    # into a marker string of the form "<decoded:" followed by the list and ">".
    tokenizer = MagicMock()
    tokenizer.decode = lambda tokens: f"<decoded:{tokens}>"
    return tokenizer


def make_sample(
    prompt="test_prompt",
    tokens=None,
    response="",
    response_length=0,
    loss_mask=None,
    rollout_log_probs=None,
    status=Sample.Status.COMPLETED,
    label="test_label",
    reward=1.0,
    index=0,
    group_index=0,
):
    # [FUNCTION DOC] HELPER: Builds a sample from the given fields with defaults: prompt
    # "test_prompt", empty tokens (a missing token list becomes empty), empty response, zero
    # response length, completed status, label "test_label", reward 1.0 and zero indices.
    return Sample(
        prompt=prompt,
        tokens=tokens or [],
        response=response,
        response_length=response_length,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        status=status,
        label=label,
        reward=reward,
        index=index,
        group_index=group_index,
    )


class TestMergeSamples:
    # [CLASS DOC] Tests of the pair merge and list merge of turn samples, covering the merged
    # fields and the validation errors for inconsistent inputs.
    def test_basic_merge(self, mock_tokenizer):
        # [FUNCTION DOC] Merges a first sample (6 tokens, response of 3) with a second sample that
        # extends it by two observation tokens and three new response tokens (truncated status).
        # Asserts the merged tokens are the second's, the response length is 3+2+3, the loss mask
        # is 1,1,1,0,0,1,1,1, the log probabilities have zeros for the observation tokens, prompt,
        # label, index and group index come from the first, status comes from the second, and the
        # merged response text contains both response texts and the decoded observation tokens.
        a = make_sample(
            tokens=[1, 2, 3, 10, 11, 12],
            response="response1",
            response_length=3,
            loss_mask=[1, 1, 1],
            rollout_log_probs=[-0.1, -0.2, -0.3],
        )
        b = make_sample(
            tokens=[1, 2, 3, 10, 11, 12, 20, 21, 30, 31, 32],
            response="response2",
            response_length=3,
            loss_mask=[1, 1, 1],
            rollout_log_probs=[-0.4, -0.5, -0.6],
            status=Sample.Status.TRUNCATED,
        )

        merged = _merge_sample_pair(a, b, mock_tokenizer)

        assert merged.tokens == b.tokens
        assert merged.response_length == 3 + 2 + 3
        assert merged.loss_mask == [1, 1, 1, 0, 0, 1, 1, 1]
        assert merged.rollout_log_probs == [-0.1, -0.2, -0.3, 0.0, 0.0, -0.4, -0.5, -0.6]
        assert merged.prompt == a.prompt
        assert merged.status == b.status
        assert merged.label == a.label
        assert merged.index == a.index
        assert merged.group_index == a.group_index
        assert "response1" in merged.response
        assert "response2" in merged.response
        assert "<decoded:[20, 21]>" in merged.response

    def test_merge_concatenates_weight_versions_per_call(self, mock_tokenizer):
        # [FUNCTION DOC] Gives each sample one per-call weight version record with one span (v1
        # over positions 3 to 6, v2 over 8 to 11). Asserts the merged sample lists both records in
        # order with spans unchanged, and that the merged sample passes its own validation.
        """Merging concatenates per-call entries; spans are unchanged since the token space is shared."""
        a = make_sample(
            tokens=[1, 2, 3, 10, 11, 12],
            response="response1",
            response_length=3,
            loss_mask=[1, 1, 1],
            rollout_log_probs=[-0.1, -0.2, -0.3],
        )
        a.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 3, 6)])]
        b = make_sample(
            tokens=[1, 2, 3, 10, 11, 12, 20, 21, 30, 31, 32],
            response="response2",
            response_length=3,
            loss_mask=[1, 1, 1],
            rollout_log_probs=[-0.4, -0.5, -0.6],
        )
        b.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 8, 11)])]

        merged = _merge_sample_pair(a, b, mock_tokenizer)

        assert merged.weight_versions == [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 3, 6)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 8, 11)]),
        ]
        merged.validate()

    def test_merge_preserves_the_shared_trainer_model_id(self, mock_tokenizer: MagicMock) -> None:
        # [FUNCTION DOC] Two samples with the same trainer model id merge into a sample with that
        # id.
        """Merging turns from one trainer preserves their shared policy identity."""
        a = make_sample(tokens=[1, 2, 10], response_length=1, loss_mask=[1])
        b = make_sample(tokens=[1, 2, 10, 20, 30], response_length=1, loss_mask=[1])
        a.trainer_model_id = "policy-a"
        b.trainer_model_id = "policy-a"

        merged = _merge_sample_pair(a=a, b=b, tokenizer=mock_tokenizer)

        assert merged.trainer_model_id == "policy-a"

    def test_merge_rejects_turns_from_different_trainer_models(self, mock_tokenizer: MagicMock) -> None:
        # [FUNCTION DOC] Two samples with different trainer model ids make the pair merge raise an
        # assertion error mentioning a trainer model id mismatch.
        """Merging turns from different trainers rejects the cross-policy trajectory."""
        a = make_sample(tokens=[1, 2, 10], response_length=1, loss_mask=[1])
        b = make_sample(tokens=[1, 2, 10, 20, 30], response_length=1, loss_mask=[1])
        a.trainer_model_id = "policy-a"
        b.trainer_model_id = "policy-b"

        with pytest.raises(AssertionError, match="trainer_model_id mismatch"):
            _merge_sample_pair(a=a, b=b, tokenizer=mock_tokenizer)

    def test_merge_preserves_indexer_topk_from_final_sample(self, mock_tokenizer):
        # [FUNCTION DOC] Gives the first sample a zero indexer top-k array of 2 positions and the
        # second a ones array of 4 positions (covering all tokens). Asserts the merged sample's
        # array equals the second's (the final sample supersedes earlier ones).
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.1],
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.2],
        )
        a.rollout_indexer_topk = numpy.zeros((2, 2, 3), dtype=numpy.int32)
        b.rollout_indexer_topk = numpy.ones((4, 2, 3), dtype=numpy.int32)

        merged = _merge_sample_pair(a, b, mock_tokenizer)

        numpy.testing.assert_array_equal(merged.rollout_indexer_topk, b.rollout_indexer_topk)

    def test_loss_mask_none_defaults_to_all_ones(self, mock_tokenizer):
        # [FUNCTION DOC] Both samples have no loss mask and no log probabilities, with one
        # response token each and one observation token in between. Asserts the merged loss mask
        # is 1,0,1 (missing masks treated as all ones) and the merged log probabilities are three
        # zeros.
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=None,
            rollout_log_probs=None,
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=None,
            rollout_log_probs=None,
        )

        merged = _merge_sample_pair(a, b, mock_tokenizer)

        assert merged.loss_mask == [1, 0, 1]
        assert merged.rollout_log_probs == [0.0, 0.0, 0.0]

    def test_tokens_prefix_mismatch_raises(self, mock_tokenizer):
        # [FUNCTION DOC] The second sample's tokens do not start with the first's (a differing
        # token); asserts an assertion error saying the later tokens must start with the earlier.
        a = make_sample(
            tokens=[1, 2, 3],
            response_length=1,
            loss_mask=[1],
        )
        b = make_sample(
            tokens=[1, 2, 99, 20, 30],
            response_length=1,
            loss_mask=[1],
        )

        with pytest.raises(AssertionError, match="b.tokens must start with a.tokens"):
            _merge_sample_pair(a, b, mock_tokenizer)

    def test_field_mismatch_raises(self, mock_tokenizer):
        # [FUNCTION DOC] The samples have different index values; asserts an assertion error
        # mentioning an index mismatch.
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            index=0,
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            index=1,
        )

        with pytest.raises(AssertionError, match="index mismatch"):
            _merge_sample_pair(a, b, mock_tokenizer)

    def test_obs_len_invalid_raises(self, mock_tokenizer):
        # [FUNCTION DOC] The second sample has one new token, which is its whole response, leaving
        # no observation tokens between the turns; asserts an assertion error saying the
        # observation length must be positive.
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
        )
        b = make_sample(
            tokens=[1, 2, 10, 30],
            response_length=1,
            loss_mask=[1],
        )

        with pytest.raises(AssertionError, match="obs_len must be > 0"):
            _merge_sample_pair(a, b, mock_tokenizer)

    def test_sample_validate_fails_raises(self, mock_tokenizer):
        # [FUNCTION DOC] The first sample has response length 2 but a loss mask of length 1;
        # asserts the merge raises an assertion error about the loss mask length, i.e. the inputs
        # are validated.
        a = make_sample(
            tokens=[1, 2, 10, 11],
            response_length=2,
            loss_mask=[1],
        )
        b = make_sample(
            tokens=[1, 2, 10, 11, 20, 30],
            response_length=1,
            loss_mask=[1],
        )

        with pytest.raises(AssertionError, match="loss_mask length"):
            _merge_sample_pair(a, b, mock_tokenizer)

    def test_merge_sample_pair_routed_experts_gap_raises_clear_error(self, mock_tokenizer):
        # [FUNCTION DOC] The first sample has routed experts and the second has none; asserts an
        # assertion error saying the first has routed experts but the second does not (rather
        # than an attribute error on none).
        # An aborted turn omits routed_experts; merging a routed turn into it must
        # raise a clear assertion rather than an AttributeError on None.shape.
        a = make_sample(tokens=[1, 2, 10], response_length=1, loss_mask=[1])
        b = make_sample(tokens=[1, 2, 10, 20, 30], response_length=1, loss_mask=[1])
        a.rollout_routed_experts = numpy.zeros((2, 2, 3), dtype=numpy.int32)
        b.rollout_routed_experts = None

        with pytest.raises(AssertionError, match="a has rollout_routed_experts but b does not"):
            _merge_sample_pair(a, b, mock_tokenizer)

    def test_merge_samples_stops_at_routing_gap(self, mock_tokenizer):
        # [FUNCTION DOC] List merge of two turns where the first has routed experts and the
        # second (truncated) lacks them. Asserts the result is the first sample object itself,
        # with its tokens and routed experts intact: merging stops before the turn with the gap.
        # merge_samples should keep the last fully-captured turn when a later turn
        # lacks the replay payload, instead of crashing while extending into it.
        t0 = make_sample(tokens=[1, 2, 10], response_length=1, loss_mask=[1])
        t1 = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            status=Sample.Status.TRUNCATED,
        )
        t0.rollout_routed_experts = numpy.zeros((2, 2, 3), dtype=numpy.int32)
        t1.rollout_routed_experts = None

        merged = merge_samples([t0, t1], mock_tokenizer)

        assert merged is t0
        assert merged.tokens == t0.tokens
        assert merged.rollout_routed_experts is not None
