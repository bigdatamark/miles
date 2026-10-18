# [MODULE DOC] Tests of the score-centering contracts: candidate (top-k log probability)
# bookkeeping on samples, its request configuration for native and OpenAI-style requests, its
# wire encoding in the session sample codec, conversion to training data, and validation of the
# run arguments. Uses the shared score centering fixtures for argument namespaces (_args, accepting
# overrides) and a single-turn sample builder (_turn, taking prompt tokens, response tokens and
# probabilities). A candidate row holds token ids padded with -1 and their log probabilities.
"""Validate score-centering sampling and candidate transport contracts."""

from copy import deepcopy

import numpy as np
import pytest
from tests.fast.fixtures.score_centering_fixtures import _args, _turn

from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
    pad_rollout_topk_logprobs,
    validate_rollout_topk_logprobs_sample,
)
from miles.rollout.session.samples.codec import COMPUTED_FIELDS, decode_samples_and_merge_input_sample, encode_samples
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.score_centering import validate_score_centering_args
from miles.utils.types import Sample


@pytest.mark.parametrize("field", ["rollout_topk_token_ids", "rollout_topk_log_probs", "rollout_log_probs"])
@pytest.mark.parametrize("ci_test", [False, True])
def test_missing_custom_producer_probabilities_fail_at_conversion(field: str, ci_test: bool) -> None:
    # [FUNCTION DOC] Parametrized over which field is cleared (top-k ids, top-k log probabilities
    # or rollout log probabilities) and the CI test flag. Converts a complete sample and a deep
    # copy with index 1 and that field set to none, and asserts conversion to training data raises
    # a value error naming the field and the sample index 1.
    complete = _turn([0, 1], [2, 3], [0.5, 0.25])
    incomplete = deepcopy(complete)
    incomplete.index = 1
    setattr(incomplete, field, None)
    with pytest.raises(ValueError, match=rf"{field}.*sample_index=1"):
        convert_samples_to_train_data(_args(ci_test=ci_test), [complete, incomplete], {}, None, None)


def test_observation_padding_retry_and_disabled_wire() -> None:
    # [FUNCTION DOC] Pads the candidate arrays for two observation tokens (also extending tokens,
    # response length, log probabilities and a zero loss mask) and asserts validation passes for
    # 3 candidates. After resetting the sample for a retry, asserts both candidate fields are
    # none. Then checks the wire format: encoding a default sample with all fields equals
    # encoding it without the candidate fields (they are omitted when empty), and decoding the
    # old-format payload yields a sample whose candidate ids are none.
    sample = _turn([0], [2, 3], [0.5, 0.25])
    pad_rollout_topk_logprobs(sample, 2)
    sample.tokens.extend([6, 6])
    sample.response_length += 2
    sample.rollout_log_probs.extend([0.0, 0.0])
    sample.loss_mask.extend([0, 0])
    validate_rollout_topk_logprobs_sample(sample, 3)
    sample.reset_for_retry()
    assert sample.rollout_topk_token_ids is None and sample.rollout_topk_log_probs is None
    old_fields = tuple(field for field in COMPUTED_FIELDS if not field.startswith("rollout_topk_"))
    assert encode_samples([Sample()], {}) == encode_samples([Sample()], {}, fields=old_fields)
    assert (
        decode_samples_and_merge_input_sample(encode_samples([Sample()], {}, fields=old_fields), Sample())
        .samples[0]
        .rollout_topk_token_ids
        is None
    )


@pytest.mark.parametrize("openai", [False, True])
def test_request_candidates_and_sampler_contract(openai: bool) -> None:
    # [FUNCTION DOC] Parametrized over native versus OpenAI-style requests. Configuring a request
    # for 128 candidates sets the candidates count field (top logprobs for OpenAI, top logprobs
    # number for native, the latter inside the request) and the sampling temperature becomes 0.7
    # from the arguments. After the test sets top-p 0.9 and top-k 64 and reconfigures with the
    # support log probability mode, asserts the request records that mode and drops the candidate
    # count fields. Reconfiguring with zero candidates in support mode leaves the request
    # unchanged.
    request = {} if openai else {"sampling_params": {}}
    configure_rollout_topk_logprobs_request(_args(rollout_top_logprobs_num=128), request, openai=openai)
    assert request["top_logprobs" if openai else "top_logprobs_num"] == 128
    sampling = request if openai else request["sampling_params"]
    assert sampling["temperature"] == 0.7
    sampling["top_p"] = 0.9
    sampling["top_k"] = 64
    support = _args(rollout_top_logprobs_num=128, rollout_sampling_logprobs_mode="support")
    configure_rollout_topk_logprobs_request(support, request, openai=openai)
    assert request["sampling_logprobs_mode"] == "support"
    assert "top_logprobs" not in request and "top_logprobs_num" not in request
    original = deepcopy(request)
    configure_rollout_topk_logprobs_request(
        _args(rollout_top_logprobs_num=0, rollout_sampling_logprobs_mode="support"), request, openai=openai
    )
    assert request == original


@pytest.mark.parametrize("openai", [False, True])
@pytest.mark.parametrize("mode", ["selected", "support"])
def test_rollout_args_override_client_candidate_fields(openai: bool, mode: str) -> None:
    # [FUNCTION DOC] Parametrized over request style and the sampling log probability mode. The
    # client request carries a candidate count of 5 and the opposite mode; for support mode it
    # also carries top-p 0.9 and top-k 2. After configuring with 3 candidates and the run's mode,
    # asserts: in support mode the request mode is support and no candidate count remains; in
    # selected mode the candidate count is 3 and no mode field remains. The run's values override
    # the client's.
    field = "top_logprobs" if openai else "top_logprobs_num"
    request = {field: 5, "sampling_logprobs_mode": "selected" if mode == "support" else "support"}
    sampling = {"top_p": 0.9, "top_k": 2} if mode == "support" else {}
    if openai:
        request.update(sampling)
    else:
        request["sampling_params"] = sampling
    configure_rollout_topk_logprobs_request(
        _args(rollout_top_logprobs_num=3, rollout_sampling_logprobs_mode=mode), request, openai=openai
    )
    if mode == "support":
        assert request["sampling_logprobs_mode"] == "support"
        assert "top_logprobs" not in request and "top_logprobs_num" not in request
    else:
        assert request[field] == 3
        assert "sampling_logprobs_mode" not in request


@pytest.mark.parametrize(
    "constraint",
    [
        {"tool_choice": "required"},
        {"tool_choice": {"type": "function", "function": {"name": "test"}}},
        {"tools": [{"type": "function", "function": {"name": "test", "strict": True}}]},
        {"response_format": {"type": "json_object"}},
    ],
)
def test_implicit_openai_grammar_constraints_are_rejected(constraint: dict) -> None:
    # [FUNCTION DOC] Parametrized over OpenAI request fragments that imply constrained decoding
    # (required tool choice, a specific function tool choice, a strict tool, a JSON object
    # response format). Asserts configuring the request raises a value error, since grammar
    # constraints change the sampling distribution.
    with pytest.raises(ValueError):
        configure_rollout_topk_logprobs_request(_args(), constraint, openai=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rollout_temperature", 0),
        ("rollout_top_k", 129),
        ("rollout_top_logprobs_num", 0),
        ("score_centering_tis_clip", float("inf")),
        ("score_centering_mis_low", 6),
        ("use_tis", True),
        ("advantage_estimator", "gspo"),
        ("recompute_logprobs_via_prefill", True),
        ("sglang_speculative_algorithm", "EAGLE"),
        ("custom_convert_samples_to_train_data_path", "pkg.convert"),
    ],
)
def test_invalid_options_fail_early(field: str, value: object) -> None:
    # [FUNCTION DOC] Parametrized over invalid argument overrides: zero rollout temperature,
    # top-k 129, zero candidates, an infinite clip, a low bound of 6, importance sampling on, the
    # gspo advantage estimator, prefill recomputation, a speculative algorithm, and a custom
    # sample-to-training-data conversion path. Each makes argument validation raise a value error.
    with pytest.raises(ValueError):
        validate_score_centering_args(_args(**{field: value}))


@pytest.mark.parametrize("session", ["v1", "v2"])
@pytest.mark.parametrize("top_k", [21, 128])
def test_large_session_heads_can_use_sglang_router(session: str, top_k: int) -> None:
    # [FUNCTION DOC] Parametrized over session server versions v1 and v2 and candidate counts 21
    # and 128, with the project's own router disabled. Asserts argument validation passes (no
    # exception).
    validate_score_centering_args(
        _args(use_session_server=session, rollout_top_logprobs_num=top_k, use_miles_router=False)
    )


@pytest.mark.parametrize("session", ["v1", "v2"])
def test_standard_openai_head_size_can_use_sglang_router(session: str) -> None:
    # [FUNCTION DOC] Parametrized over session versions v1 and v2 with 20 candidates (the
    # standard OpenAI limit) and the project's router disabled; validation must pass.
    validate_score_centering_args(
        _args(use_session_server=session, rollout_top_logprobs_num=20, use_miles_router=False)
    )


def test_large_native_heads_can_use_sglang_router() -> None:
    # [FUNCTION DOC] With no session server, 128 candidates and the project's router disabled,
    # validation must pass.
    validate_score_centering_args(_args(use_session_server=None, rollout_top_logprobs_num=128, use_miles_router=False))


@pytest.mark.parametrize("session", ["v1", "v2"])
def test_filtered_session_support_does_not_use_top_logprobs_router_cap(session: str) -> None:
    # [FUNCTION DOC] Parametrized over session versions v1 and v2: with 128 candidates, top-p
    # 0.9, top-k 64, support log probability mode, support replay on and the project's router
    # disabled, validation must pass (the support mode is not subject to the router's top log
    # probabilities cap).
    validate_score_centering_args(
        _args(
            use_session_server=session,
            use_miles_router=False,
            rollout_top_logprobs_num=128,
            rollout_top_p=0.9,
            rollout_top_k=64,
            rollout_sampling_logprobs_mode="support",
            use_sampling_support_replay=True,
        )
    )


def test_other_losses_do_not_require_score_centering_router() -> None:
    # [FUNCTION DOC] With the policy loss type, session v2, 128 candidates and the project's
    # router disabled, validation must pass.
    validate_score_centering_args(
        _args(loss_type="policy_loss", use_session_server="v2", rollout_top_logprobs_num=128, use_miles_router=False)
    )


def test_missing_or_mismatched_probabilities_fail_before_training() -> None:
    # [FUNCTION DOC] Takes a valid single-turn sample, perturbs the first rollout log probability
    # by -0.1 and asserts validation (for 3 candidates) raises a value error saying the values
    # must come from the same sampler. Then sets a candidate id to -1 and asserts a value error
    # saying every trained token needs candidates. Finally asserts that appending candidates from
    # a response lacking the output top log probabilities entry raises a value error naming it.
    sample = _turn([0], [2, 3], [0.5, 0.25])
    sample.rollout_log_probs[0] -= 0.1
    with pytest.raises(ValueError, match="same sampler"):
        validate_rollout_topk_logprobs_sample(sample, 3)
    sample.rollout_topk_token_ids[1] = -1
    with pytest.raises(ValueError, match="Every trained token"):
        validate_rollout_topk_logprobs_sample(sample, 3)
    with pytest.raises(ValueError, match="output_top_logprobs"):
        append_rollout_topk_logprobs(Sample(response_length=1), {"output_token_logprobs": [(-1.0, 2, None)]}, 3)


@pytest.mark.parametrize("mode", ["selected", "support"])
def test_zero_token_completion_accepts_missing_candidate_fields(mode: str) -> None:
    # [FUNCTION DOC] Parametrized over the selected and support modes. Appending candidates for a
    # sample with zero response tokens and an empty response payload yields candidate id and log
    # probability arrays of shape (0, 3).
    sample = Sample(response_length=0)
    append_rollout_topk_logprobs(sample, {}, 3, sampling_logprobs_mode=mode)
    assert sample.rollout_topk_token_ids.shape == (0, 3)
    assert sample.rollout_topk_log_probs.shape == (0, 3)


@pytest.mark.parametrize("support", [False, True])
@pytest.mark.parametrize("candidates", [[2, 2, -1], [-1, 3, 3], [2, 3, 2]])
def test_duplicate_candidates_are_rejected(support: bool, candidates: list[int]) -> None:
    # [FUNCTION DOC] Parametrized over whether a sampling support mask is also attached and over
    # candidate rows with repeated token ids (including a repeated non-padding id and repeated
    # padding combined with a real id). Overwrites every candidate row of a valid sample with the
    # row and asserts validation raises a value error about duplicate candidate token ids.
    sample = _turn([0], [2, 3], [0.5, 0.25])
    sample.rollout_topk_token_ids[:] = candidates
    if support:
        sample.rollout_sampling_mask = RolloutSamplingMask.from_mask_list([[2, 3], [2, 3]])
    with pytest.raises(ValueError, match="Duplicate.*candidate token IDs"):
        validate_rollout_topk_logprobs_sample(sample, 3)


@pytest.mark.parametrize("candidates", [[2, 3, -1], [-1, 3, 2], [-1, 2, -1], [-1, -1, -1]])
def test_candidate_validation_preserves_order_and_allows_repeated_padding(candidates: list[int]) -> None:
    # [FUNCTION DOC] Parametrized over valid candidate rows (ordering variations, padding in
    # different places, all padding). Overwrites each row of a valid sample with the ids and
    # matching log probabilities (log of 0.5 for token 2, log of 0.25 otherwise, negative infinity
    # for padding), zeroes the loss mask when all candidates are padding, and asserts validation
    # passes and neither array is modified or reordered.
    sample = _turn([0], [2, 3], [0.5, 0.25])
    sample.rollout_topk_token_ids[:] = candidates
    sample.rollout_topk_log_probs[:] = [
        -np.inf if token == -1 else np.log(0.5 if token == 2 else 0.25) for token in candidates
    ]
    if all(token == -1 for token in candidates):
        sample.loss_mask = [0, 0]
    original_ids = sample.rollout_topk_token_ids.copy()
    original_logps = sample.rollout_topk_log_probs.copy()
    validate_rollout_topk_logprobs_sample(sample, 3)
    np.testing.assert_array_equal(sample.rollout_topk_token_ids, original_ids)
    np.testing.assert_array_equal(sample.rollout_topk_log_probs, original_logps)
