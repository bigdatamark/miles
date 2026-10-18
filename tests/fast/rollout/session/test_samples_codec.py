# [MODULE DOC] Tests of the samples wire codec used between the session server worker and the
# driver. The worker encodes samples plus session metadata and an optional empty reason into a
# safetensors container (a one-dimensional unsigned 8-bit metadata tensor holding JSON, plus
# tensors named by field and sample index); the driver decodes it and overlays the computed fields
# onto copies of the input sample, which keeps the input's template fields. Field sets: the base
# set, the version-2 set, and the base set with the sampling mask fields. Helpers build sample
# fixtures, build weight version records, and re-pack a valid payload after a mutation to produce
# malformed payloads. Covered: round trips with exact types, spec info, multi-sample replies with
# independent metadata copies, empty replies, the input-sample defaults guard, non-contiguous
# replay tensors, default field sets omitting the sampling mask, zero-size tensors versus none,
# null markers restoring fresh lists, dtype and value validation on encode, many malformed-payload
# rejections on decode, and weight version round trips and shape validation.
"""Tests for the samples wire codec: encode on the worker, overlay on the driver.

Covers safetensors-tensor round-trips, malformed-payload rejection, and the
input-sample defaults guard (`assert_input_sample_defaults`).
"""

import json

import numpy as np
import pytest
import safetensors.numpy
from safetensors import SafetensorError

from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS,
    COMPUTED_FIELDS_V2,
    ROLLOUT_SAMPLING_MASK_FIELDS,
    decode_samples_and_merge_input_sample,
    encode_samples,
)
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


def _per_call(*spans: tuple[str, int, int]) -> WeightVersionsPerCall:
    # [FUNCTION DOC] HELPER: Builds one call's weight version record from zero or more
    # (version, start, end) triples, one span each.
    """One generation call's weight-version spans, given as `(version, abs_start, abs_end)` triples."""
    return WeightVersionsPerCall(
        spans=[WeightVersionSpan(version=version, abs_start=start, abs_end=end) for version, start, end in spans]
    )


_FIELDS_WITH_SAMPLING_MASK = COMPUTED_FIELDS + ROLLOUT_SAMPLING_MASK_FIELDS


def _versions(version: str) -> WeightVersionsPerCall:
    # [FUNCTION DOC] HELPER: Builds a one-span weight version record for the version covering
    # absolute positions 0 to 1.
    return WeightVersionsPerCall(spans=[WeightVersionSpan(version=version, abs_start=0, abs_end=1)])


def _computed_sample(**overrides) -> Sample:
    # [FUNCTION DOC] HELPER: Builds a sample with all computed fields set like the worker would:
    # tokens [1,2,3,10,11], response "[10][11]" of length 2, loss mask [1,1], two log
    # probabilities (the second with many decimals to test exact float round trip), a sampling mask
    # (ids 10,4,11 with offsets 0,2,3), a 32-bit routed experts array shaped (4,3,2), no indexer
    # top-k, completed status, two weight version records (w1, w2), prefix cache info (2 cached of
    # 3 prompt tokens) and metadata with lifecycle and messages entries. Keyword overrides replace
    # any attribute afterwards.
    """A blank-template sample carrying every computed field, as the worker produces."""
    s = Sample()
    s.tokens = [1, 2, 3, 10, 11]
    s.response = "[10][11]"
    s.response_length = 2
    s.loss_mask = [1, 1]
    s.rollout_log_probs = [-0.5, -0.1234567891234567]
    s.rollout_sampling_mask = RolloutSamplingMask(ids=[10, 4, 11], offsets=[0, 2, 3])
    s.rollout_routed_experts = np.arange(24, dtype=np.int32).reshape(4, 3, 2)
    s.rollout_indexer_topk = None
    s.status = Sample.Status.COMPLETED
    s.weight_versions = [_versions("w1"), _versions("w2")]
    s.prefix_cache_info = Sample.PrefixCacheInfo.from_dict({"cached_tokens": 2, "total_prompt_tokens": 3})
    s.metadata = {
        "lifecycle": [{"t0": 1.0, "t1": 2.0, "turn": 1}],
        "messages": [{"role": "assistant", "content": "done"}],
    }
    for name, value in overrides.items():
        setattr(s, name, value)
    return s


def _mutated_payload(payload: bytes, mutate) -> bytes:
    # [FUNCTION DOC] HELPER: Loads the safetensors payload, parses and removes the JSON metadata
    # tensor, calls the mutate function with the metadata mapping and the tensor mapping (it may
    # edit both in place), then re-encodes the metadata as compact JSON in a uint8 tensor, saves
    # the tensors again and returns the new bytes. Used to craft malformed payloads.
    """Re-pack a valid payload after `mutate(meta, tensors)` edits, for malformed-wire cases."""
    tensors = safetensors.numpy.load(payload)
    meta = json.loads(tensors.pop("_samples_meta").tobytes().decode("utf-8"))
    mutate(meta, tensors)
    meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tensors["_samples_meta"] = np.frombuffer(meta_bytes, dtype=np.uint8)
    return safetensors.numpy.save(tensors)


class TestSamplesWireCodec:
    # [CLASS DOC] Tests of encode and decode of samples on the wire.
    def test_round_trip_overlays_computed_and_keeps_template(self):
        # [FUNCTION DOC] Encodes one computed sample with the sampling mask field set and decodes
        # it against a template sample with group index, index, prompt, label, reward, metadata
        # (including a stale lifecycle entry), routing key and train metadata. Asserts: no empty
        # reason and the session metadata restored; computed fields overlaid with exact types and
        # values (token list type, float precision, sampling mask ids and offsets, loss mask,
        # response, status, weight versions, prefix cache info, routed experts dtype and values,
        # absent indexer top-k); template fields carried over; metadata merged with the computed
        # lifecycle and messages replacing the stale entry; and the template itself not mutated.
        template = Sample(
            group_index=7,
            index=3,
            prompt=[{"role": "user", "content": "hi"}],
            label="lbl",
            reward=1.5,
            metadata={"task": "t", "lifecycle": "stale"},
            routing_key="rk",
            train_metadata={"loss": "ppo"},
        )
        payload = encode_samples([_computed_sample()], {"max_trim_tokens": 1}, None, fields=_FIELDS_WITH_SAMPLING_MASK)
        reply = decode_samples_and_merge_input_sample(payload, template, fields=_FIELDS_WITH_SAMPLING_MASK)

        assert reply.empty_reason is None
        assert reply.session_metadata == {"max_trim_tokens": 1}
        (out,) = reply.samples
        # computed fields overlaid, with exact types/values
        assert out.tokens == [1, 2, 3, 10, 11] and type(out.tokens) is list
        assert out.rollout_log_probs == [-0.5, -0.1234567891234567]
        sampling_mask_ids, sampling_mask_offsets = out.rollout_sampling_mask._as_tensors()
        assert sampling_mask_ids.tolist() == [10, 4, 11]
        assert sampling_mask_offsets.tolist() == [0, 2, 3]
        assert out.loss_mask == [1, 1]
        assert out.response == "[10][11]" and out.response_length == 2
        assert out.status == Sample.Status.COMPLETED
        assert out.weight_versions == [_versions("w1"), _versions("w2")]
        assert out.prefix_cache_info.to_dict() == {"cached_tokens": 2, "total_prompt_tokens": 3}
        assert out.rollout_routed_experts.dtype == np.int32
        assert np.array_equal(out.rollout_routed_experts, np.arange(24, dtype=np.int32).reshape(4, 3, 2))
        assert out.rollout_indexer_topk is None
        # template fields carried from the input sample, untouched
        assert out.group_index == 7 and out.index == 3
        assert out.prompt == [{"role": "user", "content": "hi"}]
        assert out.label == "lbl" and out.reward == 1.5
        assert out.metadata == {
            "task": "t",
            "lifecycle": [{"t0": 1.0, "t1": 2.0, "turn": 1}],
            "messages": [{"role": "assistant", "content": "done"}],
        }
        assert out.routing_key == "rk"
        assert out.train_metadata == {"loss": "ppo"}
        # the input template itself is never mutated
        assert template.tokens == [] and template.metadata == {"task": "t", "lifecycle": "stale"}

    @pytest.mark.parametrize("fields", [COMPUTED_FIELDS, COMPUTED_FIELDS_V2])
    def test_spec_info_round_trips(self, fields):
        # [FUNCTION DOC] Parametrized over the base and version-2 field sets. A sample with
        # speculative decoding info (3 correct, 5 proposed, 2 verifications, 7 completion tokens)
        # is encoded and decoded; asserts the decoded info equals the original as a mapping.
        spec_info = Sample.SpecInfo(
            spec_num_correct_drafts=3,
            spec_num_proposed_drafts=5,
            spec_verify_ct=2,
            completion_tokens=7,
        )

        payload = encode_samples([_computed_sample(spec_info=spec_info)], {}, fields=fields)
        (out,) = decode_samples_and_merge_input_sample(payload, Sample(), fields=fields).samples

        assert out.spec_info.to_dict() == spec_info.to_dict()

    def test_multi_sample_reply_keeps_per_sample_tensors(self):
        # [FUNCTION DOC] Encodes two different samples (different tokens, routed experts shapes,
        # log probabilities and server-only metadata) and decodes them against a template with
        # input-only metadata. Asserts each decoded sample has its own tokens, routed experts and
        # log probabilities; each metadata merges the input's and its own server entry; and the
        # decoded metadata objects are independent copies (changing one sample's nested input
        # value affects neither the other sample nor the template).
        a = _computed_sample(metadata={"server_only": {"sample": "a"}})
        b = _computed_sample(
            tokens=[1, 2, 3, 10, 11, 20, 21, 30],
            rollout_routed_experts=np.arange(100, 100 + 42, dtype=np.int32).reshape(7, 3, 2),
            rollout_log_probs=[-1.0, -2.0],
            metadata={"server_only": {"sample": "b"}},
        )
        template = Sample(metadata={"input_only": {"value": 1}})
        reply = decode_samples_and_merge_input_sample(encode_samples([a, b], {}, None), template)
        out_a, out_b = reply.samples
        assert out_a.tokens == a.tokens and out_b.tokens == b.tokens
        assert np.array_equal(out_a.rollout_routed_experts, a.rollout_routed_experts)
        assert np.array_equal(out_b.rollout_routed_experts, b.rollout_routed_experts)
        assert out_b.rollout_log_probs == [-1.0, -2.0]
        assert out_a.metadata == {"input_only": {"value": 1}, "server_only": {"sample": "a"}}
        assert out_b.metadata == {"input_only": {"value": 1}, "server_only": {"sample": "b"}}
        out_a.metadata["input_only"]["value"] = 2
        assert out_b.metadata["input_only"] == {"value": 1}
        assert template.metadata == {"input_only": {"value": 1}}

    def test_empty_reply_round_trips_reason_and_skips_defaults_guard(self):
        # [FUNCTION DOC] Encodes a reply with no samples and the reason "no_records", and decodes
        # it against an input sample that already has weight versions (which would violate the
        # defaults guard for non-empty replies). Asserts no samples and the reason preserved, i.e.
        # the guard is skipped for empty replies.
        # The empty reply is decoded before the driver takes its ABORTED path, so
        # the overlay-defaults guard must not fire on it — even for an input
        # sample that would violate the guard.
        evolved = Sample(weight_versions=[_versions("stale")])
        reply = decode_samples_and_merge_input_sample(
            encode_samples([], {"max_trim_tokens": 0}, "no_records"), evolved
        )
        assert reply.samples == [] and reply.empty_reason == "no_records"

    def test_defaults_guard_rejects_evolved_template(self):
        # [FUNCTION DOC] Decoding a non-empty reply against an input sample that already has
        # weight versions, teacher log probabilities, or an on-policy-distillation student top
        # log probabilities metadata entry raises an assertion error naming that field (the input
        # sample must be at its defaults for fields the overlay will set).
        payload = encode_samples([_computed_sample()], {}, None)
        with pytest.raises(AssertionError, match="weight_versions"):
            decode_samples_and_merge_input_sample(payload, Sample(weight_versions=[_versions("stale")]))
        with pytest.raises(AssertionError, match="teacher_log_probs"):
            decode_samples_and_merge_input_sample(payload, Sample(teacher_log_probs=[-1.0]))
        with pytest.raises(AssertionError, match="opd_student_top_logprobs"):
            decode_samples_and_merge_input_sample(
                payload, Sample(metadata={"opd_student_top_logprobs": [[[-0.1, 1]]]})
            )

    def test_safetensors_container_round_trips_non_contiguous_replay_tensors(self):
        # [FUNCTION DOC] Builds non-contiguous routed experts (a transposed view) and indexer
        # top-k (a strided slice) arrays and encodes a sample with them using the sampling mask
        # field set. Opens the payload as a plain safetensors buffer and asserts the tensor names
        # are exactly the metadata tensor plus tokens, loss mask, log probabilities, sampling
        # mask ids and offsets, routed experts and indexer top-k for sample 0; the metadata tensor
        # is one-dimensional unsigned 8-bit and the loss mask is unsigned 8-bit. Decoding gives
        # equal tokens (a list), log probabilities, 32-bit routed experts with the transposed
        # shape (4,3,2) and values, and equal 32-bit indexer top-k.
        routed = np.arange(24, dtype=np.int32).reshape(3, 4, 2).transpose(1, 0, 2)
        indexer = np.arange(48, dtype=np.int32).reshape(8, 3, 2)[::2]
        assert not routed.flags["C_CONTIGUOUS"] and not indexer.flags["C_CONTIGUOUS"]
        sample = _computed_sample(rollout_routed_experts=routed, rollout_indexer_topk=indexer)

        payload = encode_samples([sample], {}, None, fields=_FIELDS_WITH_SAMPLING_MASK)
        # the reply is a plain safetensors buffer: no Miles framing needed to open it
        tensors = safetensors.numpy.load(payload)
        assert set(tensors) == {
            "_samples_meta",
            "tokens.0",
            "loss_mask.0",
            "rollout_log_probs.0",
            "rollout_sampling_mask.ids.0",
            "rollout_sampling_mask.offsets.0",
            "rollout_routed_experts.0",
            "rollout_indexer_topk.0",
        }
        assert tensors["_samples_meta"].dtype == np.uint8 and tensors["_samples_meta"].ndim == 1
        assert tensors["loss_mask.0"].dtype == np.uint8

        (out,) = decode_samples_and_merge_input_sample(payload, Sample(), fields=_FIELDS_WITH_SAMPLING_MASK).samples
        assert out.tokens == sample.tokens and type(out.tokens) is list
        assert out.rollout_log_probs == sample.rollout_log_probs
        assert out.rollout_routed_experts.dtype == np.int32 and out.rollout_routed_experts.shape == (4, 3, 2)
        assert np.array_equal(out.rollout_routed_experts, routed)
        assert out.rollout_indexer_topk.dtype == np.int32 and np.array_equal(out.rollout_indexer_topk, indexer)

    def test_default_fields_do_not_emit_sampling_mask(self):
        # [FUNCTION DOC] Encodes with the default field set and asserts the metadata's null list
        # for sample 0 does not list the sampling mask, no tensor name starts with the sampling
        # mask prefix, and the decoded sample's sampling mask is none.
        payload = encode_samples([_computed_sample()], {}, None)
        tensors = safetensors.numpy.load(payload)
        meta = json.loads(tensors["_samples_meta"].tobytes().decode("utf-8"))

        assert "rollout_sampling_mask" not in meta["samples"][0]["nulls"]
        assert not any(name.startswith("rollout_sampling_mask.") for name in tensors)
        (out,) = decode_samples_and_merge_input_sample(payload, Sample()).samples
        assert out.rollout_sampling_mask is None

    def test_zero_size_tensor_is_distinct_from_none(self):
        # [FUNCTION DOC] A sample with an empty routed experts array of shape (0,3,2) and no
        # indexer top-k round trips to an array of that shape for routed experts and none for
        # the indexer top-k.
        sample = _computed_sample(
            rollout_routed_experts=np.empty((0, 3, 2), dtype=np.int32), rollout_indexer_topk=None
        )
        (out,) = decode_samples_and_merge_input_sample(encode_samples([sample], {}, None), Sample()).samples
        assert isinstance(out.rollout_routed_experts, np.ndarray) and out.rollout_routed_experts.shape == (0, 3, 2)
        assert out.rollout_indexer_topk is None

    def test_null_tokens_restore_fresh_empty_lists(self):
        # [FUNCTION DOC] Encodes two samples, then rewrites the payload so each sample marks its
        # tokens as null and drops the token tensors. Asserts both decoded token lists are empty
        # and are different list objects.
        # A null marker restores per-sample fresh instances, never one shared list.
        def null_out_tokens(meta, tensors):
            # [FUNCTION DOC] HELPER: Mutator for the payload re-packer: for every sample adds the
            # tokens marker to its null list and deletes that sample's token tensor.
            for index, sample_meta in enumerate(meta["samples"]):
                sample_meta["nulls"].append("tokens")
                del tensors[f"tokens.{index}"]

        payload = _mutated_payload(encode_samples([_computed_sample(), _computed_sample()], {}, None), null_out_tokens)
        out_a, out_b = decode_samples_and_merge_input_sample(payload, Sample()).samples
        assert out_a.tokens == [] and out_b.tokens == []
        assert out_a.tokens is not out_b.tokens

    def test_encode_rejects_non_int32_replay_dtype(self):
        # [FUNCTION DOC] Encoding a sample whose routed experts array is 64-bit integers raises a
        # value error saying it must have dtype int32.
        sample = _computed_sample(rollout_routed_experts=np.arange(24, dtype=np.int64).reshape(4, 3, 2))
        with pytest.raises(ValueError, match="rollout_routed_experts must have dtype int32"):
            encode_samples([sample], {}, None)

    def test_encode_rejects_loss_mask_values_that_wrap_in_uint8(self):
        # [FUNCTION DOC] Encoding a sample whose loss mask holds -1 raises a value error saying
        # the loss mask values do not fit the unsigned 8-bit wire type (rather than wrapping).
        with pytest.raises(ValueError, match="loss_mask values do not fit wire dtype uint8"):
            encode_samples([_computed_sample(loss_mask=[1, -1])], {}, None)

    @pytest.mark.parametrize(
        ("build_payload", "expected_error", "match"),
        [
            pytest.param(lambda p: p[: len(p) - 100], SafetensorError, None, id="truncated-container"),
            pytest.param(lambda p: b"", SafetensorError, None, id="empty-container"),
            pytest.param(
                lambda p: safetensors.numpy.save({"tokens.0": np.arange(3, dtype=np.int64)}),
                KeyError,
                "_samples_meta",
                id="missing-samples-meta",
            ),
            pytest.param(
                lambda p: safetensors.numpy.save({"_samples_meta": np.zeros(4, dtype=np.int32)}),
                ValueError,
                "rank-one uint8",
                id="meta-wrong-dtype",
            ),
            pytest.param(
                lambda p: safetensors.numpy.save({"_samples_meta": np.zeros((2, 2), dtype=np.uint8)}),
                ValueError,
                "rank-one uint8",
                id="meta-wrong-rank",
            ),
            pytest.param(
                lambda p: _mutated_payload(p, lambda meta, tensors: tensors.pop("tokens.0")),
                KeyError,
                "tokens.0",
                id="promised-tensor-missing",
            ),
            pytest.param(
                lambda p: _mutated_payload(
                    p,
                    lambda meta, tensors: tensors.update({"tokens.0": tensors["tokens.0"].astype(np.int32)}),
                ),
                ValueError,
                "tokens must have dtype int64",
                id="wire-dtype-contract-violated",
            ),
            pytest.param(
                lambda p: _mutated_payload(p, lambda meta, tensors: meta["samples"][0]["nulls"].append("response")),
                ValueError,
                "non-tensor fields",
                id="null-marker-for-non-tensor-field",
            ),
            pytest.param(
                lambda p: _mutated_payload(
                    p, lambda meta, tensors: meta["samples"][0].update(metadata=["not", "an", "object"])
                ),
                ValueError,
                "metadata must be a JSON object",
                id="metadata-not-object",
            ),
            pytest.param(
                lambda p: _mutated_payload(
                    p, lambda meta, tensors: tensors.update(orphan=np.zeros(1, dtype=np.uint8))
                ),
                ValueError,
                "unreferenced tensors",
                id="unreferenced-leftover-tensor",
            ),
        ],
    )
    def test_malformed_safetensors_reply_fails_loudly(self, build_payload, expected_error, match):
        # [FUNCTION DOC] Parametrized over payload builders that start from a valid encoded sample:
        # a truncated container or empty bytes (safetensors error); a container without the
        # metadata tensor (key error naming it); a metadata tensor of the wrong dtype or rank
        # (value error: must be rank-one uint8); a promised tensor missing (key error naming
        # tokens.0); a token tensor of the wrong dtype (value error: tokens must be int64); a
        # null marker for a non-tensor field (value error); metadata that is not a JSON object
        # (value error); and an extra unreferenced tensor (value error). Asserts decoding raises
        # the expected error type with a message matching the pattern when one is given.
        valid = encode_samples([_computed_sample()], {}, None)
        with pytest.raises(expected_error, match=match):
            decode_samples_and_merge_input_sample(build_payload(valid), Sample())


class TestWeightVersionsOnTheSamplesWire:
    # [CLASS DOC] Tests of how per-call weight version records travel on the wire.
    def test_weight_versions_round_trip_preserves_multiple_spans_per_call(self):
        # [FUNCTION DOC] Encodes a sample with two call records (the first with two spans, the
        # second with one) and asserts the decoded sample has equal records, spans nested under
        # their call.
        """A call whose weights changed mid-decode keeps all of its spans nested under that call."""
        weight_versions = [_per_call(("w1", 0, 2), ("w2", 2, 3)), _per_call(("w3", 3, 5))]
        payload = encode_samples([_computed_sample(weight_versions=weight_versions)], {}, None)

        (out,) = decode_samples_and_merge_input_sample(payload, Sample()).samples

        assert out.weight_versions == weight_versions

    def test_weight_versions_round_trip_preserves_empty_calls(self):
        # [FUNCTION DOC] Encodes records for three calls where the first and last have no spans,
        # and asserts the decoded records equal them, preserving the call count.
        """Unstamped generation calls survive as empty entries so the call count is preserved."""
        weight_versions = [_per_call(), _per_call(("w1", 3, 5)), _per_call()]
        payload = encode_samples([_computed_sample(weight_versions=weight_versions)], {}, None)

        (out,) = decode_samples_and_merge_input_sample(payload, Sample()).samples

        assert out.weight_versions == weight_versions

    @pytest.mark.parametrize(
        ("weight_versions", "match"),
        [
            pytest.param(["w1", "w2"], "must be a mapping", id="legacy-scalar-versions"),
            pytest.param(
                [{"version": "w1", "abs_start": 3, "abs_end": 5}],
                "must be a mapping",
                id="spans-not-nested-per-call",
            ),
            pytest.param([[{"version": "w1", "abs_start": 3}]], "abs_end", id="span-missing-field"),
        ],
    )
    def test_malformed_weight_versions_shape_fails_loudly(self, weight_versions, match):
        # [FUNCTION DOC] Parametrized over a legacy list of version strings, a flat list of span
        # mappings (not nested per call) and a span missing its end field. Encodes a valid sample,
        # replaces the weight versions entry in the metadata with the bad shape, and asserts decode
        # raises a type error whose message matches (must be a mapping, or naming the missing
        # field).
        """A weight_versions payload that is not a per-call list of span mappings raises instead of decoding."""
        valid = encode_samples([_computed_sample(weight_versions=[_per_call(("w1", 3, 5))])], {}, None)
        payload = _mutated_payload(
            valid, lambda meta, tensors: meta["samples"][0].update(weight_versions=weight_versions)
        )

        with pytest.raises(TypeError, match=match):
            decode_samples_and_merge_input_sample(payload, Sample())
