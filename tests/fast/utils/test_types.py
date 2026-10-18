# [MODULE DOC] Unit tests for the Sample data type and its weight-version tracking helpers:
# stripping trailing output tokens (and the parallel per-token arrays), recording weight version
# spans per generate call from engine meta information, validating spans, serialization round
# trips including legacy dumps, the oldest numeric weight version, and speculative decoding
# statistics. Spans are absolute token index ranges tagged with a version label. A mock tokenizer
# fixture decodes ids to letters.
"""Unit tests for Sample helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy
import pytest

from miles.utils.types import LEGACY_WEIGHT_VERSIONS_KEY, Sample, WeightVersionSpan, WeightVersionsPerCall


def _make_sample(
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    log_probs: bool = False,
    loss_mask: bool = False,
    routed_experts: bool = False,
    indexer_topk: bool = False,
) -> Sample:
    # [FUNCTION DOC] HELPER: Builds a sample whose tokens are the prompt ids followed by the
    # response ids, with response length set to the response count and a dummy response text.
    # Optional flags add per-response-token rollout log-probs (-0.1 each), a loss mask of ones,
    # a zero-filled routed experts array and a zero-filled indexer top-k array, the latter two
    # having one row per token minus one.
    """Create a Sample with the given prompt + response token IDs."""
    tokens = prompt_ids + response_ids
    s = Sample(
        tokens=tokens,
        response_length=len(response_ids),
        response="dummy",
    )
    if log_probs:
        s.rollout_log_probs = [-0.1] * len(response_ids)
    if loss_mask:
        s.loss_mask = [1] * len(response_ids)
    if routed_experts:
        # shape: (num_tokens - 1, ...)
        s.rollout_routed_experts = numpy.zeros((len(tokens) - 1, 2, 2), dtype=numpy.int32)
    if indexer_topk:
        # shape: (num_tokens - 1, ...)
        s.rollout_indexer_topk = numpy.zeros((len(tokens) - 1, 2, 3), dtype=numpy.int32)
    return s


@pytest.fixture
def tokenizer():
    # [FUNCTION DOC] HELPER: Fixture returning a mock tokenizer whose decode maps each id to the
    # uppercase letter at that offset from A, joined into a string.
    tok = MagicMock()
    tok.decode = lambda ids: "".join(chr(65 + i) for i in ids)
    return tok


class TestStripLastOutputTokens:
    # [CLASS DOC] Tests of removing the last N response tokens from a sample.
    def test_strip_zero_is_noop(self, tokenizer):
        # [FUNCTION DOC] Stripping zero tokens leaves tokens and response length (3) unchanged.
        s = _make_sample([1, 2], [3, 4, 5])
        original_tokens = list(s.tokens)
        s.strip_last_output_tokens(0, tokenizer)
        assert s.tokens == original_tokens
        assert s.response_length == 3

    def test_strip_basic(self, tokenizer):
        # [FUNCTION DOC] Stripping 2 of 3 response tokens leaves the prompt plus the first response
        # token and a response length of 1.
        s = _make_sample([1, 2], [3, 4, 5])
        s.strip_last_output_tokens(2, tokenizer)
        assert s.tokens == [1, 2, 3]
        assert s.response_length == 1

    def test_strip_all_response(self, tokenizer):
        # [FUNCTION DOC] Stripping all 3 response tokens leaves only the prompt, response length 0
        # and an empty response text.
        s = _make_sample([1, 2], [3, 4, 5])
        s.strip_last_output_tokens(3, tokenizer)
        assert s.tokens == [1, 2]
        assert s.response_length == 0
        assert s.response == ""

    def test_strip_too_many_raises(self, tokenizer):
        # [FUNCTION DOC] Stripping 3 tokens from a 2-token response raises an assertion error
        # saying it cannot strip 3 tokens.
        s = _make_sample([1, 2], [3, 4])
        with pytest.raises(AssertionError, match="cannot strip 3 tokens"):
            s.strip_last_output_tokens(3, tokenizer)

    def test_strip_truncates_log_probs(self, tokenizer):
        # [FUNCTION DOC] Stripping 2 shrinks the rollout log-probs from 3 entries to 1.
        s = _make_sample([1, 2], [3, 4, 5], log_probs=True)
        assert len(s.rollout_log_probs) == 3
        s.strip_last_output_tokens(2, tokenizer)
        assert len(s.rollout_log_probs) == 1

    def test_strip_truncates_loss_mask(self, tokenizer):
        # [FUNCTION DOC] Stripping 1 shrinks the loss mask from 3 entries to 2.
        s = _make_sample([1, 2], [3, 4, 5], loss_mask=True)
        assert len(s.loss_mask) == 3
        s.strip_last_output_tokens(1, tokenizer)
        assert len(s.loss_mask) == 2

    def test_strip_truncates_routed_experts(self, tokenizer):
        # [FUNCTION DOC] Stripping 2 removes 2 rows from the routed experts array.
        s = _make_sample([1, 2], [3, 4, 5], routed_experts=True)
        original_len = len(s.rollout_routed_experts)
        s.strip_last_output_tokens(2, tokenizer)
        assert len(s.rollout_routed_experts) == original_len - 2

    def test_strip_truncates_indexer_topk(self, tokenizer):
        # [FUNCTION DOC] Stripping 2 removes 2 rows from the indexer top-k array.
        s = _make_sample([1, 2], [3, 4, 5], indexer_topk=True)
        original_len = len(s.rollout_indexer_topk)
        s.strip_last_output_tokens(2, tokenizer)
        assert len(s.rollout_indexer_topk) == original_len - 2

    def test_strip_updates_response_text(self, tokenizer):
        # [FUNCTION DOC] After stripping 1, the response text equals the tokenizer decoding of the
        # remaining response tokens.
        s = _make_sample([1, 2], [3, 4, 5])
        s.strip_last_output_tokens(1, tokenizer)
        # response should be re-decoded from the remaining response tokens
        assert s.response == tokenizer.decode(s.tokens[-s.response_length :])

    def test_strip_negative_is_noop(self, tokenizer):
        # [FUNCTION DOC] Stripping a negative count leaves tokens unchanged.
        s = _make_sample([1, 2], [3, 4])
        original_tokens = list(s.tokens)
        s.strip_last_output_tokens(-1, tokenizer)
        assert s.tokens == original_tokens

    def test_strip_clips_weight_version_spans(self, tokenizer):
        # [FUNCTION DOC] Builds a sample with two calls whose spans cover tokens 2-4 and 4-6 and
        # strips 3 tokens (new end 3). Asserts the flattened spans become one span 2-3 (first
        # truncated, second dropped) while the per-call list still has two entries.
        """Stripping output tokens truncates overlapping spans and drops fully-stripped ones."""
        s = _make_sample([1, 2], [3, 4, 5, 6])
        s.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 4, 6)]),
        ]
        s.strip_last_output_tokens(3, tokenizer)
        assert s.all_weight_version_spans == [WeightVersionSpan("v1", 2, 3)]
        assert len(s.weight_versions) == 2


def _make_args() -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Returns a fake argument namespace with speculative decoding disabled.
    return SimpleNamespace(sglang_speculative_algorithm=None)


def _make_meta_info(output_ids: list[int], **extra) -> dict:
    # [FUNCTION DOC] HELPER: Builds an engine meta information mapping for a finished (stop)
    # generation: completion token count equal to the number of ids, per-token log-prob pairs of
    # -0.1 and each id, plus any extra keys such as weight versions.
    return {
        "finish_reason": {"type": "stop"},
        "completion_tokens": len(output_ids),
        "output_token_logprobs": [(-0.1, token_id) for token_id in output_ids],
        **extra,
    }


class TestWeightVersions:
    # [CLASS DOC] Tests of weight version span bookkeeping on samples and per-call records.
    def test_update_from_meta_info_parses_per_token_weight_versions(self):
        # [FUNCTION DOC] Meta information with two relative spans (0-2 and 2-3) on a sample with a
        # 2-token prompt must yield one call record with absolute spans 2-4 and 4-5.
        """Per-token weight_versions from meta_info are shifted to absolute token indices."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.update_from_meta_info(
            _make_args(),
            _make_meta_info(
                [3, 4, 5],
                weight_versions=[{"version": "v1", "start": 0, "end": 2}, {"version": "v2", "start": 2, "end": 3}],
            ),
        )
        assert s.weight_versions == [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4), WeightVersionSpan("v2", 4, 5)])
        ]

    def test_update_from_meta_info_synthesizes_span_from_scalar_weight_version(self):
        # [FUNCTION DOC] A scalar weight version with three output tokens after a 2-token prompt
        # yields one call with one span 2-5 labelled with that version.
        """Without per-token data, the scalar weight_version becomes one span over the new tokens."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.update_from_meta_info(_make_args(), _make_meta_info([3, 4, 5], weight_version="v7"))
        assert s.weight_versions == [WeightVersionsPerCall(spans=[WeightVersionSpan("v7", 2, 5)])]

    def test_output_end_anchors_the_span_when_the_caller_appends_its_own_tokens(self):
        # [FUNCTION DOC] With 4 logged output tokens, a scalar version and an explicit output end
        # of 14, the span starts at 10 (end minus token count) and ends at 14.
        """output_end marks the end of the generated tokens, so filler stored after them is not covered."""
        meta = {"output_token_logprobs": [(-0.1, i) for i in range(4)], "weight_version": "v1"}
        call = WeightVersionsPerCall.from_meta_info(meta, output_end=14)
        assert call.spans == [WeightVersionSpan("v1", 10, 14)]

    def test_zero_length_spans_are_dropped(self):
        # [FUNCTION DOC] An aborted call with no output tokens and one zero-length reported span
        # produces a call record with no spans.
        """An aborted call with no output tokens reports a zero-length span that covers nothing."""
        call = WeightVersionsPerCall.from_meta_info(
            {"output_token_logprobs": [], "weight_versions": [{"version": "v1", "start": 0, "end": 0}]}, output_end=7
        )
        assert call.spans == []

    def test_a_scalar_weight_version_without_logprobs_fails_instead_of_dropping_the_version(self):
        # [FUNCTION DOC] A scalar version with a positive completion count but no log-prob list
        # raises an assertion error saying log-probs are required.
        """Silently dropping the version would disable staleness checks for return_logprob=False callers."""
        with pytest.raises(AssertionError, match="requires return_logprob=True"):
            WeightVersionsPerCall.from_meta_info(
                {"completion_tokens": 3, "weight_version": "v1"},
                output_end=3,
            )

    def test_a_scalar_weight_version_on_an_empty_output_is_dropped_without_failing(self):
        # [FUNCTION DOC] A scalar version with zero completion tokens and an empty log-prob list
        # yields a record with no spans and does not raise.
        """An aborted call that generated nothing has no output tokens to anchor the version to."""
        call = WeightVersionsPerCall.from_meta_info(
            {"completion_tokens": 0, "output_token_logprobs": [], "weight_version": "v1"},
            output_end=7,
        )
        assert call.spans == []

    def test_from_meta_info_rejects_span_beyond_reported_output_tokens(self):
        # [FUNCTION DOC] Two logged tokens but a reported span ending at 3 raises an assertion
        # error saying the span extends past the 2 output tokens.
        """A span reaching past the reported output tokens means the logprobs are incomplete, so it raises."""
        with pytest.raises(AssertionError, match="extend past the 2 output tokens"):
            WeightVersionsPerCall.from_meta_info(
                {
                    "output_token_logprobs": [(-0.1, 3), (-0.1, 4)],
                    "weight_versions": [{"version": "v1", "start": 0, "end": 3}],
                },
                output_end=5,
            )

    def test_per_token_weight_versions_take_precedence_over_scalar_weight_version(self):
        # [FUNCTION DOC] When meta information has both a per-token list and a scalar version, the
        # result contains only the per-token span (version v1, 0-1), not the scalar one.
        """When the engine reports both, the per-token spans win over the scalar fallback."""
        call = WeightVersionsPerCall.from_meta_info(
            {
                "output_token_logprobs": [(-0.1, 3), (-0.1, 4)],
                "weight_versions": [{"version": "v1", "start": 0, "end": 1}],
                "weight_version": "v9",
            },
            output_end=2,
        )
        assert call.spans == [WeightVersionSpan("v1", 0, 1)]

    def test_update_from_meta_info_records_a_call_without_weight_version(self):
        # [FUNCTION DOC] Meta information without any version still appends one empty call
        # record; the flattened span list stays empty.
        """A call the engine did not stamp still counts as one call, with no spans."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.update_from_meta_info(_make_args(), _make_meta_info([3, 4, 5]))
        assert s.weight_versions == [WeightVersionsPerCall(spans=[])]
        assert s.all_weight_version_spans == []

    def test_turn_count_includes_calls_without_weight_versions(self):
        # [FUNCTION DOC] A first call without a version then a second call (after extending tokens
        # and response length by 2) with version v2: asserts two call records and a single flat
        # span 4-6 for v2.
        """The per-call nesting counts every generate call, stamped or not."""
        s = _make_sample([1, 2], [3, 4])
        s.update_from_meta_info(_make_args(), _make_meta_info([3, 4]))
        s.tokens += [5, 6]
        s.response_length += 2
        s.update_from_meta_info(_make_args(), _make_meta_info([5, 6], weight_version="v2"))
        assert len(s.weight_versions) == 2
        assert s.all_weight_version_spans == [WeightVersionSpan("v2", 4, 6)]

    def test_update_from_meta_info_appends_one_entry_per_call(self):
        # [FUNCTION DOC] Two successive calls (versions v1 then v2, with the sample extended
        # between them) produce two call records with absolute spans 2-4 and 4-7, and the flat
        # list holds both spans in order.
        """Each generate call appends its own entry with correct absolute offsets."""
        s = _make_sample([1, 2], [3, 4])
        s.update_from_meta_info(_make_args(), _make_meta_info([3, 4], weight_version="v1"))
        s.tokens += [5, 6, 7]
        s.response_length += 3
        s.update_from_meta_info(_make_args(), _make_meta_info([5, 6, 7], weight_version="v2"))
        assert s.weight_versions == [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 4, 7)]),
        ]
        assert s.all_weight_version_spans == [WeightVersionSpan("v1", 2, 4), WeightVersionSpan("v2", 4, 7)]

    def test_reset_for_retry_clears_weight_versions(self):
        # [FUNCTION DOC] Resetting a sample for retry empties its call records.
        """reset_for_retry clears weight_versions along with other outputs."""
        s = _make_sample([1, 2], [3, 4])
        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)])]
        s.reset_for_retry()
        assert s.weight_versions == []

    def test_reset_for_retry_clears_the_policy_the_last_generation_stamped(self):
        # [FUNCTION DOC] Resetting a sample that carries a trainer model id sets that id back to
        # none.
        """The policy id marks what a generation produced, so a retry must earn it again instead of inheriting it."""
        s = _make_sample([1, 2], [3, 4])
        s.trainer_model_id = "solver"

        s.reset_for_retry()

        assert s.trainer_model_id is None

    def test_validate_accepts_contiguous_spans(self):
        # [FUNCTION DOC] Ordered adjacent spans 2-4 and 4-5 within a 5-token sample validate
        # without raising.
        """validate passes for ordered non-overlapping spans within the token range."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 4, 5)]),
        ]
        s.validate()

    def test_validate_rejects_overlapping_spans_across_calls(self):
        # [FUNCTION DOC] Spans 2-4 and 3-5 in successive calls raise an assertion error about an
        # invalid weight version span.
        """validate fails when spans from successive calls overlap."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 3, 5)]),
        ]
        with pytest.raises(AssertionError, match="invalid weight version span"):
            s.validate()

    def test_validate_rejects_span_beyond_tokens(self):
        # [FUNCTION DOC] A span ending at 6 in a 5-token sample raises an assertion error about an
        # invalid weight version span.
        """validate fails when a span extends past the token list."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 6)])]
        with pytest.raises(AssertionError, match="invalid weight version span"):
            s.validate()

    def test_validate_rejects_empty_or_reversed_weight_version_span(self):
        # [FUNCTION DOC] Both a zero-length span (2-2) and a reversed span (4-3) raise an
        # assertion error about an invalid weight version span.
        """validate fails for a zero-length span and for one whose end precedes its start."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 2)])]
        with pytest.raises(AssertionError, match="invalid weight version span"):
            s.validate()

        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 4, 3)])]
        with pytest.raises(AssertionError, match="invalid weight version span"):
            s.validate()

    def test_validate_rejects_empty_version(self):
        # [FUNCTION DOC] A span with an empty version label raises an assertion error mentioning
        # an empty version.
        """validate fails when a span carries an empty version string."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("", 2, 4)])]
        with pytest.raises(AssertionError, match="empty version"):
            s.validate()

    def test_to_dict_from_dict_roundtrip(self):
        # [FUNCTION DOC] Serializing a sample with two call records to a dictionary and loading it
        # back gives equal call records, and every flattened span is a typed span object.
        """Per-call weight versions survive a to_dict/from_dict roundtrip as typed objects."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("v2", 4, 5)]),
        ]
        restored = Sample.from_dict(s.to_dict())
        assert restored.weight_versions == s.weight_versions
        assert all(isinstance(span, WeightVersionSpan) for span in restored.all_weight_version_spans)

    def test_from_dict_keeps_pre_span_dumps_loadable(self):
        # [FUNCTION DOC] Loading a legacy dump whose weight versions are a flat list of strings
        # gives empty call records and stores the legacy list under the legacy key attribute.
        """Dumps predating per-call spans load without misparsing their flat version strings."""
        restored = Sample.from_dict({"status": "completed", "weight_versions": ["v1", "v2"], "tokens": [1, 2]})
        assert restored.weight_versions == []
        assert getattr(restored, LEGACY_WEIGHT_VERSIONS_KEY) == ["v1", "v2"]

    def test_from_dict_reads_current_dumps_unchanged(self):
        # [FUNCTION DOC] Loading a dump with nested per-call span dictionaries (absolute start and
        # end keys) gives the typed call records and does not set the legacy key attribute.
        """A dump written with per-call spans still round-trips into the typed structure."""
        restored = Sample.from_dict(
            {
                "status": "completed",
                "weight_versions": [[{"version": "v1", "abs_start": 0, "abs_end": 2}]],
                "tokens": [1, 2],
            }
        )
        assert restored.weight_versions == [WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 0, 2)])]
        assert not hasattr(restored, LEGACY_WEIGHT_VERSIONS_KEY)

    def test_oldest_weight_version_reads_all_spans(self):
        # [FUNCTION DOC] With numeric labels 7 and 5 in one call, the oldest version is the
        # integer 5.
        """oldest_weight_version takes the minimum numeric version across every span."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("7", 2, 4), WeightVersionSpan("5", 4, 5)])]
        assert s.oldest_weight_version == 5

    def test_oldest_weight_version_ignores_nonnumeric_spans(self):
        # [FUNCTION DOC] Labels v1 and 9 give oldest 9 (the nonnumeric one is skipped); a sample
        # with only the nonnumeric label reports none.
        """Nonnumeric version labels are skipped, and a sample carrying only those reports no version."""
        s = _make_sample([1, 2], [3, 4, 5])
        s.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4), WeightVersionSpan("9", 4, 5)])
        ]
        assert s.oldest_weight_version == 9

        s.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 2, 4)])]
        assert s.oldest_weight_version is None


class TestSpecInfo:
    # [CLASS DOC] Tests of the speculative decoding statistics record nested in the sample type.
    def test_ratio_properties(self):
        # [FUNCTION DOC] With 3 correct of 4 proposed drafts, 2 verifications and 5 completion
        # tokens, the accept rate is 0.75 and the accept length is 2.5; an empty record reports
        # 0.0 for both (no division error).
        info = Sample.SpecInfo(
            spec_num_correct_drafts=3,
            spec_num_proposed_drafts=4,
            spec_verify_ct=2,
            completion_tokens=5,
        )

        assert info.spec_accept_rate == 0.75
        assert info.spec_accept_length == 2.5
        assert Sample.SpecInfo().spec_accept_rate == 0.0
        assert Sample.SpecInfo().spec_accept_length == 0.0

    def test_add_ignores_completion_without_speculative_verification(self):
        # [FUNCTION DOC] Adding meta information that has only a completion token count leaves
        # the record equal to a fresh empty one.
        info = Sample.SpecInfo()

        info.add({"completion_tokens": 1})

        assert info == Sample.SpecInfo()

    def test_from_dict_reads_legacy_miles_fields(self):
        # [FUNCTION DOC] Loading a sample dictionary whose speculative info uses the old field
        # names (accept token, draft token, verify count, completion token counts) maps them to
        # the new correct drafts, proposed drafts, verify count and completion tokens fields.
        sample = Sample.from_dict(
            {
                "status": "completed",
                "spec_info": {
                    "spec_accept_token_num": 3,
                    "spec_draft_token_num": 5,
                    "spec_verify_ct": 2,
                    "completion_token_num": 7,
                },
            }
        )

        assert sample.spec_info == Sample.SpecInfo(
            spec_num_correct_drafts=3,
            spec_num_proposed_drafts=5,
            spec_verify_ct=2,
            completion_tokens=7,
        )
