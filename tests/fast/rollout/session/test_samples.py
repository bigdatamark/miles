# [MODULE DOC] Tests of assembling training samples from session records (one record per chat
# completion request/response made through the session server) and merging them across turns, the
# core of the token-in-token-out pipeline. Records are built by a local helper mimicking the
# inference engine's response format (output token ids with log probabilities in meta info, optional
# cache statistics, weight versions, routed experts payloads, sampling masks). A mock tokenizer
# decodes ids to a bracketed text form. Test groups: per-record sample construction (tokens, log
# probabilities, status from finish reason, weight version spans, native sampling support, abort
# handling, conversation snapshot recording); multi-turn merge prefix chains and propagation of
# teacher log probabilities and student top log probabilities; trailing stop token trimming driven
# by the accumulated token list; additional-mode routed experts (R3) patch assembly; the known
# thinking-token prefix break; and prefix cache statistics. Module constants: argument namespaces
# (plain, recording debug trajectories, and with two layers for R3), the stop token id 99, and the
# R3 top-k width 2.
"""Tests for compute_samples_from_openai_records and TITO multi-turn merge workflow.

Validates the contract between session records, sample construction,
and merge_samples — the core of the TITO (Token In Token Out) pipeline.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pybase64
import pytest

from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.samples.merge import compute_samples_from_openai_records, merge_samples_with_addition_r3
from miles.rollout.session.types import SessionRecord
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

# ── helpers ──────────────────────────────────────────────────────────

_ARGS = SimpleNamespace(save_debug_trajectory_data=None, sglang_speculative_algorithm=None)
_ARGS_RECORDING = SimpleNamespace(
    save_debug_trajectory_data="/unused/{rollout_id}.jsonl", sglang_speculative_algorithm=None
)


def _mock_tokenizer():
    # [FUNCTION DOC] HELPER: Returns a mock tokenizer whose decode joins the ids as bracketed
    # numbers (for example "[10][11]").
    tok = MagicMock()
    tok.decode = lambda ids: "".join(f"[{i}]" for i in ids)
    return tok


def _make_record(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    output_log_probs: list[float] | None = None,
    finish_reason: str = "stop",
    cached_tokens: int | None = None,
    prompt_tokens: int | None = None,
    weight_version: str | None = None,
    routed_experts: str | None = None,
    routed_experts_start_len: int | None = None,
    weight_versions: list[dict[str, str | int]] | None = None,
    sampling_masks: list[list[int]] | None = None,
    sampling_log_probs: list[float] | None = None,
) -> SessionRecord:
    # [FUNCTION DOC] HELPER: Builds a successful POST chat completion session record. The request
    # holds one user message, the prompt token ids as input ids, and optionally the routed experts
    # start length, and (when sampling masks are given) the sampling mask request flag. The
    # response has one choice with an assistant message, the finish reason (default stop), a
    # log probability content list and meta info with: output token log probabilities as
    # (log probability, token id) pairs (default log probabilities are minus 0.1 times the
    # 1-based position), the completion token count, and optional cached tokens, prompt tokens,
    # scalar weight version, routed experts payload, per-token weight versions, sampling masks and
    # sampling log probabilities. Timestamp 0.
    """Build a minimal session record mimicking SGLang's response format.

    Token IDs and logprobs are stored in meta_info.output_token_logprobs
    as (logprob, token_id) tuples, matching the real SGLang response.
    `routed_experts` is the base64 int32 buffer exactly as SGLang returns it;
    with `routed_experts_start_len` it covers only rows [start, end), matching
    an addition-R3 request that persisted the offset.
    """
    if output_log_probs is None:
        output_log_probs = [-0.1 * (i + 1) for i in range(len(output_token_ids))]

    output_token_logprobs = [(lp, tid) for tid, lp in zip(output_token_ids, output_log_probs, strict=True)]
    logprobs_content = [
        {"logprob": lp, "token": f"t{tid}"} for tid, lp in zip(output_token_ids, output_log_probs, strict=True)
    ]
    meta_info = {
        "output_token_logprobs": output_token_logprobs,
        "completion_tokens": len(output_token_ids),
    }
    if cached_tokens is not None:
        meta_info["cached_tokens"] = cached_tokens
    if prompt_tokens is not None:
        meta_info["prompt_tokens"] = prompt_tokens
    if weight_version is not None:
        meta_info["weight_version"] = weight_version
    if routed_experts is not None:
        meta_info["routed_experts"] = routed_experts
    request = {"messages": [{"role": "user", "content": "hello"}], "input_ids": prompt_token_ids}
    if routed_experts_start_len is not None:
        request["routed_experts_start_len"] = routed_experts_start_len
    if weight_versions is not None:
        meta_info["weight_versions"] = weight_versions
    if sampling_masks is not None:
        request["return_sampling_mask"] = True
        meta_info["output_token_sampling_mask"] = sampling_masks
        meta_info["output_token_sampling_logprobs"] = sampling_log_probs
    return SessionRecord(
        timestamp=0.0,
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        request=request,
        response={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "response"},
                    "finish_reason": finish_reason,
                    "logprobs": {"content": logprobs_content},
                    "meta_info": meta_info,
                }
            ]
        },
    )


# ── test: compute_samples_from_openai_records ────────────────────────


class TestComputeSamplesFromRecords:
    # [CLASS DOC] Tests of building one sample per record.
    def test_single_record_builds_correct_sample(self):
        # [FUNCTION DOC] One record with prompt [1,2,3] and output [10,11] with log probabilities
        # -0.5 and -0.6 yields one sample with tokens prompt plus output, those log probabilities,
        # response length 2, loss mask [1,1] and completed status.
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            output_log_probs=[-0.5, -0.6],
        )

        samples = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert len(samples) == 1
        s = samples[0]
        assert s.tokens == [1, 2, 3, 10, 11]
        assert s.rollout_log_probs == [-0.5, -0.6]
        assert s.response_length == 2
        assert s.loss_mask == [1, 1]
        assert s.status == Sample.Status.COMPLETED

    def test_single_record_scalar_weight_version_becomes_span(self):
        # [FUNCTION DOC] A record with a scalar weight version "v3" gives one call record with a
        # single span v3 covering the output token positions 3 to 5 (absolute, after the prompt).
        """A scalar weight_version in the record's meta_info synthesizes one span over the output tokens."""
        tok = _mock_tokenizer()
        record = _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11], weight_version="v3")

        samples = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert samples[0].weight_versions == [WeightVersionsPerCall(spans=[WeightVersionSpan("v3", 3, 5)])]

    def test_single_record_per_token_weight_versions_become_spans(self):
        # [FUNCTION DOC] A record with per-token weight versions (v1 for output positions 0 to 2,
        # v2 for 2 to 3) gives one call record with spans v1 at 3 to 5 and v2 at 5 to 6: the
        # output-relative ranges are shifted by the prompt length 3.
        """Per-token weight_versions in meta_info are shifted by the prompt length."""
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11, 12],
            weight_versions=[{"version": "v1", "start": 0, "end": 2}, {"version": "v2", "start": 2, "end": 3}],
        )

        samples = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert samples[0].weight_versions == [
            WeightVersionsPerCall(spans=[WeightVersionSpan("v1", 3, 5), WeightVersionSpan("v2", 5, 6)])
        ]

    def test_trimmed_trailing_tokens_clip_the_weight_version_span(self):
        # [FUNCTION DOC] Two records where the first output ends with the stop token that the
        # second prompt does not contain, with scalar versions v3 and v4 and one trailing token
        # allowed to be trimmed. Asserts the first sample's tokens drop the stop token and its span
        # is clipped to end at 4 (3 to 4), and the second sample's span covers 5 to 6.
        """Spans are built over the untrimmed output, so trimming must clip the last span's end."""
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, STOP], weight_version="v3"),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 4], output_token_ids=[20], weight_version="v4"),
        ]

        samples = compute_samples_from_openai_records(
            _ARGS,
            records,
            tok,
            accumulated_token_ids=[1, 2, 3, 10, 4, 20],
            max_trim_tokens=1,
        )

        assert samples[0].tokens == [1, 2, 3, 10]
        assert samples[0].weight_versions == [WeightVersionsPerCall(spans=[WeightVersionSpan("v3", 3, 4)])]
        assert samples[1].weight_versions == [WeightVersionsPerCall(spans=[WeightVersionSpan("v4", 5, 6)])]

    def test_weight_version_span_past_output_tokens_is_rejected(self):
        # [FUNCTION DOC] A per-token span ending at 3 with only 2 output tokens makes assembly
        # raise an assertion error saying spans extend past the 2 output tokens.
        """A span reaching beyond the reported output tokens cannot be anchored, so no Sample is built."""
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            weight_versions=[{"version": "v1", "start": 0, "end": 3}],
        )

        with pytest.raises(AssertionError, match="extend past the 2 output tokens"):
            compute_samples_from_openai_records(_ARGS, [record], tok)

    def test_record_without_any_weight_version_still_contributes_a_call(self):
        # [FUNCTION DOC] A record without any weight version yields a sample whose weight
        # versions list has one call record with no spans (the number of calls is the list
        # length).
        """turns is len(weight_versions), so an unstamped call must appear as an empty call."""
        tok = _mock_tokenizer()
        record = _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, 11])

        samples = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert samples[0].weight_versions == [WeightVersionsPerCall(spans=[])]

    def test_single_record_uses_native_sampling_support_and_log_probs(self):
        # [FUNCTION DOC] A record has output log probabilities -2.5 and -2.6 but also native
        # sampling log probabilities -0.5 and -0.6 with supports [10,4,7] and [11,3]. Asserts the
        # sample's rollout log probabilities are the sampling ones, its mask has ids
        # 10,4,7,11,3 with offsets 0,3,5, and the sample passes validation.
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            output_log_probs=[-2.5, -2.6],
            sampling_masks=[[10, 4, 7], [11, 3]],
            sampling_log_probs=[-0.5, -0.6],
        )

        (sample,) = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert sample.rollout_log_probs == [-0.5, -0.6]
        ids, offsets = sample.rollout_sampling_mask._as_tensors()
        assert ids.tolist() == [10, 4, 7, 11, 3]
        assert offsets.tolist() == [0, 3, 5]
        sample.validate()

    def test_abort_without_sampling_metadata_is_non_trainable(self):
        # [FUNCTION DOC] An aborted record with no output tokens that requested the sampling mask
        # but returned none yields an aborted sample with no reward and an empty sampling mask.
        tok = _mock_tokenizer()
        record = _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[], finish_reason="abort")
        record.request["return_sampling_mask"] = True

        (sample,) = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert sample.status == Sample.Status.ABORTED
        assert sample.reward is None
        assert len(sample.rollout_sampling_mask) == 0

    def test_successful_turn_still_requires_sampling_metadata(self):
        # [FUNCTION DOC] A successful (stop) record with one output token that requested the
        # sampling mask but returned none makes assembly raise a value error saying the output
        # token sampling mask is missing.
        tok = _mock_tokenizer()
        record = _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10])
        record.request["return_sampling_mask"] = True

        with pytest.raises(ValueError, match="missing output_token_sampling_mask"):
            compute_samples_from_openai_records(_ARGS, [record], tok)

    def test_later_abort_preserves_terminal_status(self):
        # [FUNCTION DOC] Two records: a successful one with a sampling mask and an aborted second
        # one (no output tokens, mask requested but absent) whose prompt extends the first by one
        # token. After assembly and merge, asserts the merged status is aborted, the merged tokens
        # are the second prompt, and the sampling mask length equals the response length (2).
        tok = _mock_tokenizer()
        records = [
            _make_record(
                prompt_token_ids=[1, 2],
                output_token_ids=[10],
                sampling_masks=[[10, 11]],
                sampling_log_probs=[-0.2],
            ),
            _make_record(prompt_token_ids=[1, 2, 10, 20], output_token_ids=[], finish_reason="abort"),
        ]
        records[1].request["return_sampling_mask"] = True

        samples = compute_samples_from_openai_records(_ARGS, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.status == Sample.Status.ABORTED
        assert merged.tokens == [1, 2, 10, 20]
        assert len(merged.rollout_sampling_mask) == merged.response_length == 2

    def test_multiple_records_produce_multiple_samples(self):
        # [FUNCTION DOC] Two records give two samples whose tokens are each record's prompt plus
        # output (no merging at this stage).
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10]),
            _make_record(prompt_token_ids=[1, 2, 10, 20], output_token_ids=[30]),
        ]

        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        assert len(samples) == 2
        assert samples[0].tokens == [1, 2, 10]
        assert samples[1].tokens == [1, 2, 10, 20, 30]

    def test_last_sample_carries_raw_conversation(self):
        # [FUNCTION DOC] With debug trajectory recording enabled in the arguments, two records
        # give two samples; asserts only the last sample's metadata has the messages entry, equal
        # to the last request's messages plus the response's assistant message.
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10]),
            _make_record(prompt_token_ids=[1, 2, 10, 20], output_token_ids=[30]),
        ]

        samples = compute_samples_from_openai_records(_ARGS_RECORDING, records, tok)

        assert "messages" not in (samples[0].metadata or {})
        assert samples[1].metadata["messages"] == records[1].request["messages"] + [
            records[1].response["choices"][0]["message"]
        ]

    def test_merge_keeps_last_conversation_snapshot(self):
        # [FUNCTION DOC] With recording enabled, merging the two assembled samples gives a merged
        # sample whose messages metadata equals the last sample's messages.
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 11, 20, 21], output_token_ids=[30, 31]),
        ]

        samples = compute_samples_from_openai_records(_ARGS_RECORDING, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.metadata["messages"] == samples[-1].metadata["messages"]

    def test_finish_reason_length_gives_truncated(self):
        # [FUNCTION DOC] A record with finish reason length yields a sample with truncated status.
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2],
            output_token_ids=[10],
            finish_reason="length",
        )

        samples = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert samples[0].status == Sample.Status.TRUNCATED


# ── test: multi-turn prefix chain (merge_samples integration) ────────


class TestMultiTurnPrefixChain:
    # [CLASS DOC] Tests of merging the samples of a multi-turn conversation, which requires each
    # record's prompt to start with the previous record's prompt plus output.
    """Validate that session records from a well-behaved multi-turn
    conversation satisfy the prefix chain required by merge_samples.

    The contract: for consecutive records r[i] and r[i+1],
    r[i+1].prompt_token_ids must start with r[i].prompt_token_ids + r[i].output_token_ids.
    This is because the agent includes the previous response in the next prompt.
    """

    def test_two_turn_merge_succeeds(self):
        # [FUNCTION DOC] Two records where the second prompt is the first prompt, the first output
        # and two observation tokens. Asserts the merged tokens are prompt, output 1, observation,
        # output 2; the response length is 6; the loss mask is 1,1,0,0,1,1; and the status is
        # completed.
        """Normal two-turn conversation: samples merge without error."""
        tok = _mock_tokenizer()

        # Turn 1: prompt=[1,2,3], model outputs [10,11]
        # Turn 2: prompt=[1,2,3, 10,11, 20,21], model outputs [30,31]
        #   (tokens 20,21 are the tool/observation tokens added by the environment)
        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=[10, 11],
                output_log_probs=[-0.1, -0.2],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
                output_log_probs=[-0.3, -0.4],
            ),
        ]

        samples = compute_samples_from_openai_records(_ARGS, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.tokens == [1, 2, 3, 10, 11, 20, 21, 30, 31]
        assert merged.response_length == 2 + 2 + 2  # resp1 + obs + resp2
        assert merged.loss_mask == [1, 1, 0, 0, 1, 1]
        assert merged.status == Sample.Status.COMPLETED

    def test_three_turn_merge_succeeds(self):
        # [FUNCTION DOC] Three records each extending the previous by one observation token.
        # Asserts the merged tokens are the full sequence ending in the third output and the
        # response length is 5 (three outputs plus two observation tokens).
        """Three-turn conversation: prefix chain holds across all turns."""
        tok = _mock_tokenizer()

        records = [
            _make_record(
                prompt_token_ids=[1, 2],
                output_token_ids=[10],
                output_log_probs=[-0.1],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 10, 20],
                output_token_ids=[30],
                output_log_probs=[-0.2],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 10, 20, 30, 40],
                output_token_ids=[50],
                output_log_probs=[-0.3],
            ),
        ]

        samples = compute_samples_from_openai_records(_ARGS, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.tokens == [1, 2, 10, 20, 30, 40, 50]
        assert merged.response_length == 1 + 1 + 1 + 1 + 1  # 3 responses + 2 obs

    def test_prefix_mismatch_raises(self):
        # [FUNCTION DOC] The second record's prompt contains tokens (99, 99) that differ from the
        # first output (10, 11); assembly succeeds but merging raises an assertion error saying
        # the later tokens must start with the earlier tokens.
        """When the prefix chain is broken, merge_samples must fail."""
        tok = _mock_tokenizer()

        # Turn 2's prompt does NOT start with turn 1's full tokens
        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=[10, 11],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 3, 99, 99, 20, 21],  # 99,99 != 10,11
                output_token_ids=[30, 31],
            ),
        ]

        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        with pytest.raises(AssertionError, match="b.tokens must start with a.tokens"):
            merge_samples(samples, tok)

    def test_two_turn_merge_propagates_teacher_log_probs(self):
        # [FUNCTION DOC] Attaches teacher log probabilities to each turn's sample (-1.0,-1.1 and
        # -1.2,-1.3) after assembly and merges. Asserts the merged teacher log probabilities are
        # turn 1 values, two zeros for the observation tokens, then turn 2 values; their length
        # equals the response length; and the merged sample passes validation.
        """OPD teacher_log_probs merge like rollout_log_probs: per-turn values
        concatenated with zeros over the injected observation span."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11], output_log_probs=[-0.1, -0.2]),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
                output_log_probs=[-0.3, -0.4],
            ),
        ]
        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        # OPD attaches per-response-token teacher log-probs to each turn's sample.
        samples[0].teacher_log_probs = [-1.0, -1.1]
        samples[1].teacher_log_probs = [-1.2, -1.3]

        merged = merge_samples(samples, tok)

        # resp1 (2) + obs (2 zeros) + resp2 (2)
        assert merged.teacher_log_probs == [-1.0, -1.1, 0.0, 0.0, -1.2, -1.3]
        assert len(merged.teacher_log_probs) == merged.response_length
        merged.validate()  # the new teacher_log_probs length assertion must hold

    def test_two_turn_merge_propagates_opd_student_top_logprobs_metadata(self):
        # [FUNCTION DOC] Each turn's sample metadata holds a per-token student top log
        # probabilities list (two entries per turn) and an equal shared entry "same". After merge,
        # asserts the shared entry is kept, and the student list is turn 1 entries, two empty
        # entries for the observation tokens, then turn 2 entries, with total length equal to the
        # response length.
        """Top-k OPD student top-logprobs are per-token metadata, not equal metadata."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11], output_log_probs=[-0.1, -0.2]),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
                output_log_probs=[-0.3, -0.4],
            ),
        ]
        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        turn_0_top_logprobs = [[[-0.1, 101]], [[-0.2, 102]]]
        turn_1_top_logprobs = [[[-0.3, 103]], [[-0.4, 104]]]
        samples[0].metadata = {
            "opd_student_top_logprobs": turn_0_top_logprobs,
            "shared_metadata": "same",
        }
        samples[1].metadata = {
            "opd_student_top_logprobs": turn_1_top_logprobs,
            "shared_metadata": "same",
        }

        merged = merge_samples(samples, tok)

        assert merged.metadata["shared_metadata"] == "same"
        assert merged.metadata["opd_student_top_logprobs"] == [
            *turn_0_top_logprobs,
            [],
            [],
            *turn_1_top_logprobs,
        ]
        assert len(merged.metadata["opd_student_top_logprobs"]) == merged.response_length

    def test_two_turn_merge_teacher_log_probs_none_stays_none(self):
        # [FUNCTION DOC] Merging two turns that have no teacher log probabilities leaves the merged
        # teacher log probabilities none.
        """Non-OPD runs leave teacher_log_probs unset; merge must keep it None."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 11, 20, 21], output_token_ids=[30, 31]),
        ]
        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        merged = merge_samples(samples, tok)

        assert merged.teacher_log_probs is None

    def test_merge_raises_on_teacher_log_probs_length_mismatch(self):
        # [FUNCTION DOC] Gives the first turn one teacher log probability for a response of two
        # tokens; merging raises an assertion error about the teacher log probabilities length.
        """validate() guards teacher_log_probs length (surfaced via merge_samples)."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 11, 20, 21], output_token_ids=[30, 31]),
        ]
        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        samples[0].teacher_log_probs = [-1.0]  # length 1 != response_length 2

        with pytest.raises(AssertionError, match="teacher_log_probs length"):
            merge_samples(samples, tok)


# ── test: TITO trailing token trimming ────────────────────────────────

STOP = 99  # stands for <|observation|> stop token


class TestTITOTrailingTokenTrim:
    # [CLASS DOC] Tests of trailing stop-token trimming. Assembly walks the accumulated token
    # list with a cursor: for each record, the output tokens are matched against the accumulated
    # tokens from the cursor; a mismatching tail is trimmed (up to the allowed maximum) because the
    # chat template consumed those stop tokens when rendering the next prompt; the last record never
    # trims; at the end the cursor must reach the end of the accumulated list. Without an
    # accumulated list nothing is trimmed.
    """Validate trailing-token trimming via ``accumulated_token_ids``.

    Worked example — agentic tool-call retries
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    An agent makes three turns.  The model's tool call fails to parse on
    turns 1 and 2, so the agent feeds back an error and retries.

    The session server sees three request/response pairs (records).  Each
    record's response is an independent inference re-stitched via
    pretokenized prefix reuse::

        record 0  prompt_token_ids:  [<|sys|>, aaa, <|user|>, bbb, <|asst|>]
                  output_token_ids:  [ccc, <|obs|>]      ← model stopped with <|obs|>

        record 1  prompt_token_ids:  [<|sys|>, aaa, <|user|>, bbb, <|asst|>, ccc, <|sys|>, ddd, <|asst|>]
                  output_token_ids:  [eee, <|obs|>]

        record 2  prompt_token_ids:  [..., eee, <|sys|>, fff, <|asst|>]
                  output_token_ids:  [ggg, <|obs|>]

    ``accumulated_token_ids`` = record 2's prompt + output::

        [<|sys|>, aaa, <|user|>, bbb, <|asst|>, ccc, <|sys|>, ddd,
         <|asst|>, eee, <|sys|>, fff, <|asst|>, ggg, <|obs|>]

    Note: there is NO ``<|obs|>`` between ``ccc`` and ``<|sys|>`` in the
    accumulated sequence — the stop token the model emitted at turn 1 was
    consumed by the chat template when rendering turn 2's prompt.

    The algorithm walks ``accumulated_token_ids`` with a cursor::

        Record 0:  cursor = len(prompt_0) → points to "ccc"
                   Match output [ccc, <|obs|>] against accumulated[cursor:]:
                     ccc OK, <|obs|> MISMATCH (accumulated has <|sys|> here)
                   → trim_count=1, strip <|obs|>; cursor advances past "ccc"

        Record 1:  cursor = len(prompt_1) → points to "eee"
                   Match [eee, <|obs|>]: eee OK, <|obs|> MISMATCH
                   → trim_count=1; cursor advances past "eee"

        Record 2:  cursor = len(prompt_2) → points to "ggg"
                   Match [ggg, <|obs|>]: ggg OK, <|obs|> OK (last turn)
                   → trim_count=0; cursor reaches end

    Result: three Samples with output tokens [ccc], [eee], [ggg, <|obs|>],
    each carrying original per-turn logprobs.

    The tests below encode this example (and variants) with concrete
    token IDs.  We use ``STOP = 99`` to represent ``<|observation|>``.
    """

    def test_three_turn_trim_trailing_stop_tokens(self):
        # [FUNCTION DOC] Three records each ending in the stop token, with an accumulated list in
        # which stop tokens appear only after the last output, and at most one token trimmed.
        # Asserts the first two samples lose their stop token (response length 1 each, tokens
        # end at the real output) and the last sample keeps it (response length 2).
        """Three-turn retry: non-final turns have 1 trailing stop token trimmed."""
        tok = _mock_tokenizer()

        #   prompt: [1, 2, 3]  output: [10, STOP]
        #   prompt: [1, 2, 3, 10, 4, 5, 6]  output: [20, STOP]
        #   prompt: [1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9]  output: [30, STOP]
        # accumulated (no intermediate STOPs):
        #   [1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30, STOP]
        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, STOP]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 4, 5, 6], output_token_ids=[20, STOP]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9], output_token_ids=[30, STOP]),
        ]
        accumulated = [1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30, STOP]

        samples = compute_samples_from_openai_records(
            _ARGS,
            records,
            tok,
            accumulated_token_ids=accumulated,
            max_trim_tokens=1,
        )

        assert len(samples) == 3
        # Turn 0: [10, STOP] → trim 1 → response_length=1
        assert samples[0].tokens == [1, 2, 3, 10]
        assert samples[0].response_length == 1
        # Turn 1: [20, STOP] → trim 1 → response_length=1
        assert samples[1].tokens == [1, 2, 3, 10, 4, 5, 6, 20]
        assert samples[1].response_length == 1
        # Turn 2 (last): [30, STOP] → trim 0 → response_length=2
        assert samples[2].tokens == [1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30, STOP]
        assert samples[2].response_length == 2

    def test_no_trim_when_no_trailing_stop(self):
        # [FUNCTION DOC] Two records whose outputs fully match the accumulated list: asserts no
        # token is trimmed in either sample and both response lengths are 2.
        """When output tokens fully match accumulated, trim_count=0 for all turns."""
        tok = _mock_tokenizer()

        # Two turns, no trailing stop tokens — output aligns perfectly
        #   prompt: [1, 2]  output: [10, 11]
        #   prompt: [1, 2, 10, 11, 3, 4]  output: [20, 21]
        # accumulated: [1, 2, 10, 11, 3, 4, 20, 21]
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, 11]),
            _make_record(prompt_token_ids=[1, 2, 10, 11, 3, 4], output_token_ids=[20, 21]),
        ]
        accumulated = [1, 2, 10, 11, 3, 4, 20, 21]

        samples = compute_samples_from_openai_records(
            _ARGS,
            records,
            tok,
            accumulated_token_ids=accumulated,
            max_trim_tokens=1,
        )

        assert len(samples) == 2
        assert samples[0].tokens == [1, 2, 10, 11]
        assert samples[0].response_length == 2
        assert samples[1].tokens == [1, 2, 10, 11, 3, 4, 20, 21]
        assert samples[1].response_length == 2

    def test_single_turn_no_trim(self):
        # [FUNCTION DOC] A single record ending in the stop token with a matching accumulated
        # list keeps all tokens (response length 3) because the last turn never trims.
        """Single turn: last turn never trims, even with accumulated_token_ids."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11, STOP]),
        ]
        accumulated = [1, 2, 3, 10, 11, STOP]

        samples = compute_samples_from_openai_records(
            _ARGS,
            records,
            tok,
            accumulated_token_ids=accumulated,
            max_trim_tokens=1,
        )

        assert len(samples) == 1
        assert samples[0].tokens == [1, 2, 3, 10, 11, STOP]
        assert samples[0].response_length == 3

    def test_no_accumulated_skips_trimming(self):
        # [FUNCTION DOC] Two records both ending in the stop token, assembled without an
        # accumulated list: asserts both samples keep their stop token (response length 2 each).
        """Without accumulated_token_ids, no trimming is performed at all."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, STOP]),
            _make_record(prompt_token_ids=[1, 2, 10, STOP, 3, 4], output_token_ids=[20, STOP]),
        ]

        samples = compute_samples_from_openai_records(
            _ARGS,
            records,
            tok,
            accumulated_token_ids=None,
        )

        assert len(samples) == 2
        # No trimming — STOP is kept for both turns
        assert samples[0].tokens == [1, 2, 10, STOP]
        assert samples[0].response_length == 2
        assert samples[1].tokens == [1, 2, 10, STOP, 3, 4, 20, STOP]
        assert samples[1].response_length == 2

    def test_trim_exceeding_max_raises(self):
        # [FUNCTION DOC] The first record's output has two trailing stop tokens that do not match
        # the accumulated list while at most one trim is allowed; asserts an assertion error saying
        # trim count 2 exceeds the allowed 1.
        """If trailing tokens exceed max_trim_tokens, assert fires."""
        tok = _mock_tokenizer()

        # Output has 2 trailing tokens that don't match, but max_trim_tokens=1
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, STOP, STOP]),
            _make_record(prompt_token_ids=[1, 2, 10, 3, 4], output_token_ids=[20]),
        ]
        accumulated = [1, 2, 10, 3, 4, 20]

        with pytest.raises(AssertionError, match="trim_count 2 exceeds allowed=1"):
            compute_samples_from_openai_records(
                _ARGS,
                records,
                tok,
                accumulated_token_ids=accumulated,
                max_trim_tokens=1,
            )

    def test_cursor_covers_entire_accumulated(self):
        # [FUNCTION DOC] The accumulated list has one extra trailing token beyond what the records
        # imply; asserts an assertion error saying the cursor differs from the accumulated length.
        """After processing all records, cursor must equal len(accumulated)."""
        tok = _mock_tokenizer()

        # accumulated is shorter than what records imply — cursor won't reach end
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, STOP]),
            _make_record(prompt_token_ids=[1, 2, 10, 3], output_token_ids=[20]),
        ]
        # Missing the last token — accumulated should be [1,2,10,3,20] but we give [1,2,10,3,20,99]
        accumulated = [1, 2, 10, 3, 20, 99]

        with pytest.raises(AssertionError, match="cursor .* != len\\(accumulated_token_ids\\)"):
            compute_samples_from_openai_records(
                _ARGS,
                records,
                tok,
                accumulated_token_ids=accumulated,
                max_trim_tokens=1,
            )


# ── test: additional-R3 patch assembly (in-place weight updates) ──────

_ARGS_R3 = SimpleNamespace(save_debug_trajectory_data=None, sglang_speculative_algorithm=None, num_layers=2)
_R3_TOPK = 2


def _r3_rows(num_rows: int, seed: int) -> np.ndarray:
    # [FUNCTION DOC] HELPER: Returns a 32-bit integer array shaped (rows, layers, top-k) with
    # consecutive values starting at the seed, using the module's layer count (2) and top-k (2).
    """`num_rows` distinct rows of shape (num_layers, topk), values seed, seed+1, …"""
    size = num_rows * _ARGS_R3.num_layers * _R3_TOPK
    return np.arange(seed, seed + size, dtype=np.int32).reshape(num_rows, _ARGS_R3.num_layers, _R3_TOPK)


def _r3_patch(rows: np.ndarray) -> str:
    # [FUNCTION DOC] HELPER: Returns the base64 ASCII text of the array's raw bytes (made
    # contiguous), the format of a routed experts payload.
    return pybase64.b64encode(np.ascontiguousarray(rows).tobytes()).decode("ascii")


def _merge_addition(records, accumulated, max_trim_tokens=0):
    # [FUNCTION DOC] HELPER: Assembles samples from the records with the R3 arguments, the
    # accumulated token list, the maximum trim (default 0) and additional-mode R3 enabled, then
    # applies the additional-R3 merge to samples and records and returns the merged sample.
    tok = _mock_tokenizer()
    samples = compute_samples_from_openai_records(
        _ARGS_R3,
        records,
        tok,
        accumulated_token_ids=accumulated,
        max_trim_tokens=max_trim_tokens,
        use_addition_r3=True,
    )
    return merge_samples_with_addition_r3(_ARGS_R3, samples, records, tok)


class TestAdditionR3Assembly:
    # [CLASS DOC] Tests of additional-mode routed experts assembly: each record carries only a
    # patch of rows (from its recorded start row), which must chain without gaps or overlaps and
    # are concatenated into one tensor; errors are value errors naming the record.
    """Addition-mode records carry an R3 patch of rows [routed_experts_start_len,
    len(input_ids) + len(output) - 1) instead of a full per-turn tensor;
    `merge_samples_with_addition_r3` concatenates the append-only stream into
    one (len(tokens) - 1, num_layers, topk) tensor after the ordinary merge.
    """

    def test_two_patches_reconstruct_full_reference(self):
        # [FUNCTION DOC] Builds a reference tensor of 7 rows. Additional-mode records carry rows
        # 0 to 4 (start 0) and rows 4 to 7 (start 4); full-mode records carry rows 0 to 4 and the
        # whole tensor. Merges both ways and asserts equal tokens (the accumulated list), loss
        # masks and log probabilities, a 32-bit routed experts array, and that both merged routed
        # experts tensors equal each other and the reference.
        """Patches must rebuild byte-for-byte the tensor full-R3 responses give
        for the same tokens under the same in-place lifecycle."""
        full = _r3_rows(7, seed=0)
        accumulated = [1, 2, 3, 10, 11, 4, 20, 21]
        # Turn 1: 5-token checkpoint -> rows [0, 4). Turn 2 extends it by one
        # env token -> start = checkpoint rows = 4, rows [4, 7).
        addition_records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(full[:4]), routed_experts_start_len=0),
            _make_record(
                [1, 2, 3, 10, 11, 4], [20, 21], routed_experts=_r3_patch(full[4:]), routed_experts_start_len=4
            ),
        ]
        full_records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(full[:4])),
            _make_record([1, 2, 3, 10, 11, 4], [20, 21], routed_experts=_r3_patch(full)),
        ]

        merged_addition = _merge_addition(addition_records, accumulated)
        tok = _mock_tokenizer()
        full_samples = compute_samples_from_openai_records(
            _ARGS_R3, full_records, tok, accumulated_token_ids=accumulated
        )
        merged_full = merge_samples(full_samples, tok)

        assert merged_addition.tokens == merged_full.tokens == accumulated
        assert merged_addition.loss_mask == merged_full.loss_mask
        assert merged_addition.rollout_log_probs == merged_full.rollout_log_probs
        assert merged_addition.rollout_routed_experts.dtype == np.int32
        assert np.array_equal(merged_addition.rollout_routed_experts, merged_full.rollout_routed_experts)
        assert np.array_equal(merged_addition.rollout_routed_experts, full)

    def test_overlapping_start_raises(self):
        # [FUNCTION DOC] The first record's patch has 5 rows (start 0) while the second starts at
        # row 4 (overlapping); with up to 2 trimmed tokens, asserts a value error saying record 1
        # starts at row 4 but 5 was expected.
        """Every patch must start exactly after the preceding raw patch."""
        turn1_rows = _r3_rows(5, seed=0)
        turn2_rows = _r3_rows(2, seed=500)
        records = [
            _make_record([1, 2, 3], [10, 98, 99], routed_experts=_r3_patch(turn1_rows), routed_experts_start_len=0),
            _make_record(
                [1, 2, 3, 10, 55], [20, 21], routed_experts=_r3_patch(turn2_rows), routed_experts_start_len=4
            ),
        ]
        accumulated = [1, 2, 3, 10, 55, 20, 21]

        with pytest.raises(ValueError, match="record 1 starts at row 4; expected 5"):
            _merge_addition(records, accumulated, max_trim_tokens=2)

    def test_intermediate_truncated_turn_bounds_materialization(self):
        # [FUNCTION DOC] Three records where the second finishes with length (truncated) and the
        # third has no routed experts payload at all. Asserts the merged sample is truncated, its
        # tokens stop after the second turn (the first 8 accumulated tokens), and its routed
        # experts are the first two patches concatenated (the third is not needed).
        """A non-COMPLETED turn bounds required rows before later patches."""
        records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(_r3_rows(4, 0)), routed_experts_start_len=0),
            _make_record(
                [1, 2, 3, 10, 11, 4],
                [20, 21],
                finish_reason="length",
                routed_experts=_r3_patch(_r3_rows(3, 16)),
                routed_experts_start_len=4,
            ),
            _make_record([1, 2, 3, 10, 11, 4, 20, 21, 5], [30, 31]),
        ]
        accumulated = [1, 2, 3, 10, 11, 4, 20, 21, 5, 30, 31]

        merged = _merge_addition(records, accumulated)

        assert merged.status == Sample.Status.TRUNCATED
        assert merged.tokens == accumulated[:8]
        expected = np.concatenate([_r3_rows(4, 0), _r3_rows(3, 16)])
        assert np.array_equal(merged.rollout_routed_experts, expected)

    def test_missing_required_patch_raises(self):
        # [FUNCTION DOC] Like the truncated case but the second turn is completed, so the third
        # record's missing payload is required; asserts a value error saying record 2 has no
        # routed experts payload.
        """A missing patch cannot silently shorten a merged trajectory."""
        records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(_r3_rows(4, 0)), routed_experts_start_len=0),
            _make_record(
                [1, 2, 3, 10, 11, 4],
                [20, 21],
                routed_experts=_r3_patch(_r3_rows(3, 16)),
                routed_experts_start_len=4,
            ),
            _make_record([1, 2, 3, 10, 11, 4, 20, 21, 5], [30, 31]),
        ]
        accumulated = [1, 2, 3, 10, 11, 4, 20, 21, 5, 30, 31]

        with pytest.raises(ValueError, match="record 2 has no routed_experts payload"):
            _merge_addition(records, accumulated)

    def test_replay_disabled_records_unaffected(self):
        # [FUNCTION DOC] Records with no routed experts payload merged in additional mode give a
        # sample with no routed experts, and tokens and loss mask equal to an ordinary merge of
        # the same records.
        """use_addition_r3=True is dormant when no record carries R3."""
        records = [
            _make_record([1, 2, 3], [10, 11]),
            _make_record([1, 2, 3, 10, 11, 4], [20, 21]),
        ]
        accumulated = [1, 2, 3, 10, 11, 4, 20, 21]

        merged = _merge_addition(records, accumulated)
        tok = _mock_tokenizer()
        reference = merge_samples(
            compute_samples_from_openai_records(_ARGS_R3, records, tok, accumulated_token_ids=accumulated), tok
        )

        assert merged.rollout_routed_experts is None
        assert merged.tokens == reference.tokens
        assert merged.loss_mask == reference.loss_mask

    def test_empty_delta_keeps_empty_buffer_shape(self):
        # [FUNCTION DOC] A single record with a one-token prompt, no output and an empty payload
        # at start row 0 gives routed experts of shape (0, layers, 0) and 32-bit integer type.
        """A trajectory whose only patch is empty keeps the existing
        empty-buffer decode shape instead of inventing a top-k source."""
        records = [_make_record([1], [], routed_experts="", routed_experts_start_len=0)]

        merged = _merge_addition(records, [1])

        assert merged.rollout_routed_experts.shape == (0, _ARGS_R3.num_layers, 0)
        assert merged.rollout_routed_experts.dtype == np.int32

    def test_missing_start_len_raises(self):
        # [FUNCTION DOC] A record with a payload but no recorded start length raises a value error
        # saying it carries no routed experts start length.
        records = [_make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(_r3_rows(4, 0)))]

        with pytest.raises(ValueError, match="carries no routed_experts_start_len"):
            _merge_addition(records, [1, 2, 3, 10, 11])

    def test_empty_payload_for_nonempty_delta_raises(self):
        # [FUNCTION DOC] An empty payload for a record that needs 4 rows raises a value error
        # saying payload presence does not match 4 rows.
        records = [_make_record([1, 2, 3], [10, 11], routed_experts="", routed_experts_start_len=0)]

        with pytest.raises(ValueError, match="payload presence does not match 4 rows"):
            _merge_addition(records, [1, 2, 3, 10, 11])

    def test_gapped_start_raises(self):
        # [FUNCTION DOC] A first record whose patch starts at row 1 leaves a gap; raises a value
        # error saying it starts at row 1 but 0 was expected.
        records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(_r3_rows(3, 0)), routed_experts_start_len=1)
        ]

        with pytest.raises(ValueError, match="starts at row 1; expected 0"):
            _merge_addition(records, [1, 2, 3, 10, 11])

    def test_wrong_value_count_raises(self):
        # [FUNCTION DOC] Four rows are implied by the record (start 0, end 4) but only three rows
        # of values are supplied; assembly raises a value error.
        # 4 rows announced (start=0, end=4) but only 3 rows of values supplied.
        records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(_r3_rows(3, 0)), routed_experts_start_len=0)
        ]

        with pytest.raises(ValueError):
            _merge_addition(records, [1, 2, 3, 10, 11])

    def test_inconsistent_topk_raises(self):
        # [FUNCTION DOC] The first turn implies a top-k width of 2 while the second supplies three
        # rows sized for width 3; assembly raises a value error.
        # Turn 1 infers topk=2; turn 2 supplies 3 rows sized for topk=3.
        bad_rows = np.arange(3 * _ARGS_R3.num_layers * 3, dtype=np.int32).reshape(3, _ARGS_R3.num_layers, 3)
        records = [
            _make_record([1, 2, 3], [10, 11], routed_experts=_r3_patch(_r3_rows(4, 0)), routed_experts_start_len=0),
            _make_record(
                [1, 2, 3, 10, 11, 4], [20, 21], routed_experts=_r3_patch(bad_rows), routed_experts_start_len=4
            ),
        ]
        accumulated = [1, 2, 3, 10, 11, 4, 20, 21]

        with pytest.raises(ValueError):
            _merge_addition(records, accumulated)


# ── test: thinking token issue (documents known failure mode) ────────


class TestThinkingTokenPrefixBreak:
    # [CLASS DOC] Documents (as tests) the known failure mode where the model's thinking tokens
    # are stripped from the agent's conversation history, breaking the prefix chain needed for
    # merging. Class constants are the Qwen3 token ids for the think start, think end and newline.
    """Documents the known issue where model-generated <think>...</think>
    tokens break the prefix chain.

    When a model (e.g. Qwen3) generates <think>reasoning</think> before
    the actual response, agents strip the thinking content from conversation
    history. This causes the next turn's prompt to not include the thinking
    tokens, breaking the prefix assumption in merge_samples.

    This is a MODEL-LEVEL issue — the fix should be at the model/serving
    config level (disable thinking mode), not in the merge logic.
    """

    THINK_TOKEN = 151667  # <think> in Qwen3
    END_THINK_TOKEN = 151668  # </think> in Qwen3
    NEWLINE_TOKEN = 198  # \n

    def test_thinking_tokens_break_prefix_chain(self):
        # [FUNCTION DOC] The first record's output has seven thinking tokens before the two answer
        # tokens, but the second record's prompt contains only the answer tokens plus observation.
        # Assembly succeeds; merging raises an assertion error saying the later tokens must start
        # with the earlier tokens.
        """Demonstrates the failure: model outputs <think>..., but the agent
        strips it from history, so the next prompt doesn't include those tokens."""
        tok = _mock_tokenizer()

        # Turn 1: model generates <think>\nreasoning\n</think>\n then actual response
        thinking_tokens = [
            self.THINK_TOKEN,
            self.NEWLINE_TOKEN,
            42,
            43,
            self.NEWLINE_TOKEN,
            self.END_THINK_TOKEN,
            self.NEWLINE_TOKEN,
        ]
        response_tokens = [10, 11]
        all_output = thinking_tokens + response_tokens

        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=all_output,
            ),
            # Turn 2: agent only included the actual response [10, 11] in history
            # (stripped thinking tokens), plus observation [20, 21]
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
            ),
        ]

        samples = compute_samples_from_openai_records(_ARGS, records, tok)

        # sample[0].tokens = [1,2,3] + thinking + [10,11] = [1,2,3, <think>,\n,42,43,\n,</think>,\n, 10,11]
        # sample[1].tokens = [1,2,3, 10,11, 20,21, 30,31]
        # sample[1] does NOT start with sample[0] — prefix chain broken
        with pytest.raises(AssertionError, match="b.tokens must start with a.tokens"):
            merge_samples(samples, tok)

    def test_no_thinking_tokens_prefix_chain_holds(self):
        # [FUNCTION DOC] The same conversation without thinking tokens merges successfully into
        # the full token sequence.
        """When thinking is disabled, the same conversation merges fine."""
        tok = _mock_tokenizer()

        # Same conversation but model output has no thinking prefix
        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=[10, 11],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
            ),
        ]

        samples = compute_samples_from_openai_records(_ARGS, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.tokens == [1, 2, 3, 10, 11, 20, 21, 30, 31]


# ── test: prefix cache info population ────────────────────────────────


class TestPrefixCacheInfo:
    # [CLASS DOC] Tests of collecting prefix cache statistics from record meta info.
    """Validate that prefix cache statistics from meta_info are collected."""

    def test_single_record_with_cache_stats(self):
        # [FUNCTION DOC] A record with 2 cached tokens and 3 prompt tokens gives a sample whose
        # prefix cache info has those two counts.
        """cached_tokens and prompt_tokens from meta_info populate prefix_cache_info."""
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            cached_tokens=2,
            prompt_tokens=3,
        )
        samples = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert samples[0].prefix_cache_info.cached_tokens == 2
        assert samples[0].prefix_cache_info.total_prompt_tokens == 3

    def test_multi_turn_cache_stats_accumulate_after_merge(self):
        # [FUNCTION DOC] Two records with cached tokens 0 and 5 and prompt tokens 3 and 7 merge
        # into a sample with cached tokens 5, total prompt tokens 10 and hit rate 0.5.
        """After merge_samples, prefix_cache_info sums across turns."""
        tok = _mock_tokenizer()
        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=[10, 11],
                output_log_probs=[-0.1, -0.2],
                cached_tokens=0,
                prompt_tokens=3,
            ),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
                output_log_probs=[-0.3, -0.4],
                cached_tokens=5,
                prompt_tokens=7,
            ),
        ]
        samples = compute_samples_from_openai_records(_ARGS, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.prefix_cache_info.cached_tokens == 0 + 5
        assert merged.prefix_cache_info.total_prompt_tokens == 3 + 7
        assert merged.prefix_cache_info.prefix_cache_hit_rate == 5 / 10

    def test_missing_cache_fields_default_to_zero(self):
        # [FUNCTION DOC] A record without cached or prompt token counts gives a sample whose
        # prefix cache info has zero for both counts.
        """Records without cached_tokens/prompt_tokens give zero prefix_cache_info (regression)."""
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
        )
        samples = compute_samples_from_openai_records(_ARGS, [record], tok)

        assert samples[0].prefix_cache_info.cached_tokens == 0
        assert samples[0].prefix_cache_info.total_prompt_tokens == 0
