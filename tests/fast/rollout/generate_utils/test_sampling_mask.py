# [MODULE DOC] Tests of the sampling-mask (sampling support replay) helpers of rollout
# generation: deciding whether a generation request must ask the server for the sampling mask
# (validating the request's top-p, top-k, temperature and logit transforms against the run
# configuration), and appending/merging the per-token support sets (ragged lists of token ids kept
# by the sampler) and their log probabilities on samples. Arguments are bare namespaces carrying
# only the fields the code reads. A mask is stored as flat ids plus offsets (sample i's support is
# the ids between offset i and i+1), read through a private tensor accessor.
from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload
from miles.rollout.generate_utils.sampling_mask import (
    append_forced_sampling_tokens,
    append_sampling_metadata,
    merge_sampling_masks,
    should_return_sampling_mask,
)
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample


@pytest.mark.parametrize(
    ("rollout_top_p", "rollout_top_k", "request_top_p", "request_top_k", "expected"),
    [
        (0.95, 32, 0.95, 32, True),
        (1.0, 32, 1.0, 32, True),
        (1.0, -1, 1.0, -1, False),
    ],
)
def test_generate_payload_automatically_requests_sampling_mask(
    rollout_top_p,
    rollout_top_k,
    request_top_p,
    request_top_k,
    expected,
):
    # [FUNCTION DOC] Parametrized over (top-p, top-k) combinations: (0.95, 32) and (1.0, 32)
    # expect a mask request; (1.0, -1), i.e. unbounded sampling, expects none. The support replay
    # flag in the arguments is set to the expectation. Builds a request payload for a two-token
    # prompt with matching request sampling parameters and temperature 1.0, and asserts there is
    # no halt status and that the payload requests the sampling mask exactly when expected.
    args = SimpleNamespace(
        use_sampling_support_replay=expected,
        rollout_top_p=rollout_top_p,
        rollout_top_k=rollout_top_k,
        rollout_temperature=1.0,
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
    )

    payload, halt_status = compute_request_payload(
        args,
        input_ids=[1, 2],
        sampling_params={
            "max_new_tokens": 4,
            "top_p": request_top_p,
            "top_k": request_top_k,
            "temperature": 1.0,
        },
    )

    assert halt_status is None
    assert payload.get("return_sampling_mask", False) is expected


def test_enabled_replay_accepts_request_top_k_above_default():
    # [FUNCTION DOC] With support replay on, a request with top-p 1.0, top-k 64 and temperature
    # 1.0 makes the mask decision return true.
    args = SimpleNamespace(use_sampling_support_replay=True, rollout_temperature=1.0)

    assert should_return_sampling_mask(args, {"top_p": 1.0, "top_k": 64, "temperature": 1.0}) is True


def test_unbounded_run_rejects_bounded_training_request():
    # [FUNCTION DOC] With support replay off (an unbounded run), a request bounded by top-k 32
    # makes the decision raise a value error saying bounded training-request sampling requires
    # bounded rollout sampling.
    args = SimpleNamespace(use_sampling_support_replay=False, rollout_temperature=1.0)

    with pytest.raises(ValueError, match="bounded training-request sampling requires bounded rollout sampling"):
        should_return_sampling_mask(args, {"top_p": 1.0, "top_k": 32, "temperature": 1.0})


def test_unbounded_run_treats_an_unset_request_top_k_as_unbounded():
    # [FUNCTION DOC] With replay off, a request with top-p 1.0 and a top-k of none (and no
    # temperature) yields false (no mask needed).
    args = SimpleNamespace(use_sampling_support_replay=False, rollout_temperature=1.0)

    assert should_return_sampling_mask(args, {"top_p": 1.0, "top_k": None}) is False


def test_evaluation_does_not_request_or_validate_training_sampling_support():
    # [FUNCTION DOC] With replay on and a run configured for top-p 0.95 and top-k 32, builds an
    # evaluation payload whose sampling parameters are unbounded with temperature 0.5 (which
    # would be invalid for training). Asserts no halt status and no sampling mask request.
    args = SimpleNamespace(
        use_sampling_support_replay=True,
        rollout_top_p=0.95,
        rollout_top_k=32,
        rollout_temperature=1.0,
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
    )

    payload, halt_status = compute_request_payload(
        args,
        input_ids=[1, 2],
        sampling_params={"max_new_tokens": 4, "top_p": 1.0, "top_k": -1, "temperature": 0.5},
        evaluation=True,
    )

    assert halt_status is None
    assert "return_sampling_mask" not in payload


def test_training_request_temperature_must_match_actor_scoring_temperature():
    # [FUNCTION DOC] With replay on and run temperature 1.0, a request temperature of 0.5 makes
    # the decision raise a value error mentioning the request temperature 0.5.
    args = SimpleNamespace(use_sampling_support_replay=True, rollout_temperature=1.0)

    with pytest.raises(ValueError, match="request temperature 0.5"):
        should_return_sampling_mask(args, {"top_p": 0.95, "top_k": 32, "temperature": 0.5})


@pytest.mark.parametrize("missing_param", ["top_p", "top_k", "temperature"])
def test_sampling_support_replay_requires_explicit_request_parameters(missing_param):
    # [FUNCTION DOC] Parametrized over top-p, top-k and temperature. With replay on, removes that
    # one parameter from an otherwise complete request and asserts a value error listing explicit
    # request parameters including the missing name.
    args = SimpleNamespace(use_sampling_support_replay=True, rollout_temperature=1.0)
    params = {"top_p": 0.95, "top_k": 32, "temperature": 1.0}
    del params[missing_param]

    with pytest.raises(ValueError, match=rf"explicit request parameters:.*{missing_param}"):
        should_return_sampling_mask(args, params)


@pytest.mark.parametrize("request_top_k", [-1, 0])
def test_training_request_top_k_must_be_positive(request_top_k):
    # [FUNCTION DOC] Parametrized over request top-k -1 and 0. With replay on, asserts a value
    # error saying the request top-k must be positive.
    args = SimpleNamespace(use_sampling_support_replay=True, rollout_temperature=1.0)

    with pytest.raises(ValueError, match="request top_k must be positive"):
        should_return_sampling_mask(
            args,
            {"top_p": 0.95, "top_k": request_top_k, "temperature": 1.0},
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("frequency_penalty", 0.1),
        ("presence_penalty", 0.1),
        ("repetition_penalty", 1.1),
        ("logit_bias", {"1": 0.1}),
        ("custom_logit_processor", "serialized-processor"),
    ],
)
def test_training_request_rejects_unreplayed_logit_transform(name, value):
    # [FUNCTION DOC] Parametrized over frequency penalty, presence penalty, repetition penalty,
    # logit bias and a custom logit processor. With replay on and an otherwise valid request,
    # adding that parameter makes the decision raise a value error saying it is not supported
    # (such transforms cannot be replayed by the trainer).
    args = SimpleNamespace(use_sampling_support_replay=True, rollout_temperature=1.0)

    with pytest.raises(ValueError, match=rf"{name} is not supported"):
        should_return_sampling_mask(
            args,
            {"top_p": 0.95, "top_k": 32, "temperature": 1.0, name: value},
        )


def test_append_sampling_metadata_preserves_ragged_support_and_native_logprobs():
    # [FUNCTION DOC] Appends metadata for two sampled tokens whose supports have different sizes
    # (3 and 2) with flat sampling log probabilities. Asserts the returned log probabilities are
    # the given flat values, then (after the caller extends tokens and response length) the mask
    # holds the flattened ids with offsets 0,3,5, and the sample passes validation.
    sample = Sample(tokens=[1, 2])
    meta_info = {
        "output_token_sampling_mask": [[10, 4, 7], [11, 3]],
        "output_token_sampling_logprobs": [-0.25, -0.5],
    }

    log_probs = append_sampling_metadata(sample, [10, 11], meta_info)
    sample.tokens.extend([10, 11])
    sample.response_length = 2

    assert log_probs == [-0.25, -0.5]
    ids, offsets = sample.rollout_sampling_mask._as_tensors()
    assert ids.tolist() == [10, 4, 7, 11, 3]
    assert offsets.tolist() == [0, 3, 5]
    sample.validate()


def test_append_sampling_metadata_selects_sampled_probability_from_support_mode():
    # [FUNCTION DOC] In support log probability mode each position carries log probabilities for
    # its whole support. Appends for sampled tokens 4 and 11 and asserts the returned values are
    # the entries at those tokens' positions in their supports (-0.6 and -0.4), with the mask
    # ids and offsets as in the ragged case.
    sample = Sample(tokens=[1])
    meta_info = {
        "output_token_sampling_mask": [[10, 4, 7], [11, 3]],
        "output_token_sampling_logprobs": [[-1.2, -0.6, -2.0], [-0.4, -1.1]],
    }

    log_probs = append_sampling_metadata(sample, [4, 11], meta_info, sampling_logprobs_mode="support")

    assert log_probs == [-0.6, -0.4]
    ids, offsets = sample.rollout_sampling_mask._as_tensors()
    assert ids.tolist() == [10, 4, 7, 11, 3]
    assert offsets.tolist() == [0, 3, 5]


def test_append_sampling_metadata_rejects_misaligned_support_logprobs():
    # [FUNCTION DOC] In support mode, a support of two ids paired with only one log probability
    # makes the append raise a value error mentioning alignment.
    sample = Sample(tokens=[1])
    meta_info = {
        "output_token_sampling_mask": [[10, 4]],
        "output_token_sampling_logprobs": [[-0.5]],
    }

    with pytest.raises(ValueError, match="align"):
        append_sampling_metadata(sample, [4], meta_info, sampling_logprobs_mode="support")


def test_forced_tokens_append_singleton_support_and_strip_cleanly():
    # [FUNCTION DOC] Starts from a sample with one sampled token (support of two ids) and
    # appends two forced tokens (such as tool output), which each get a singleton support. After
    # extending tokens and response length, asserts validation passes and the mask is ids
    # 10,4,20,21 with offsets 0,2,3,4. Then strips the last two output tokens using a stub
    # tokenizer (decode returns empty text) and asserts the mask returns to ids 10,4 with offsets
    # 0,2.
    sample = Sample(
        tokens=[1, 10],
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask(ids=[10, 4], offsets=[0, 2]),
    )

    append_forced_sampling_tokens(sample, [20, 21])
    sample.tokens.extend([20, 21])
    sample.response_length += 2
    sample.validate()

    ids, offsets = sample.rollout_sampling_mask._as_tensors()
    assert ids.tolist() == [10, 4, 20, 21]
    assert offsets.tolist() == [0, 2, 3, 4]

    tokenizer = type("Tokenizer", (), {"decode": staticmethod(lambda _: "")})()
    sample.strip_last_output_tokens(2, tokenizer)
    ids, offsets = sample.rollout_sampling_mask._as_tensors()
    assert ids.tolist() == [10, 4]
    assert offsets.tolist() == [0, 2]


def test_merge_sampling_masks_inserts_singleton_observation_supports():
    # [FUNCTION DOC] Merges the mask of a first turn (one position, support 10,4), two
    # observation tokens (20, 21) and the mask of a second turn (one position, support 30,7,8).
    # Asserts the merged mask has ids 10,4,20,21,30,7,8 and offsets 0,2,3,4,7, i.e. each
    # observation token becomes a one-element support between the two turns.
    first = Sample(
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask(ids=[10, 4], offsets=[0, 2]),
    )
    second = Sample(
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask(ids=[30, 7, 8], offsets=[0, 3]),
    )

    sampling_mask = merge_sampling_masks(first, [20, 21], second)
    ids, offsets = sampling_mask._as_tensors()

    assert ids.tolist() == [10, 4, 20, 21, 30, 7, 8]
    assert offsets.tolist() == [0, 2, 3, 4, 7]


def test_append_sampling_metadata_rejects_support_without_sampled_token():
    # [FUNCTION DOC] Appends for sampled token 10 whose support (4 and 7) does not contain it and
    # asserts a value error saying the sampled token 10 is absent.
    with pytest.raises(ValueError, match="sampled token 10 is absent"):
        append_sampling_metadata(
            Sample(),
            [10],
            {
                "output_token_sampling_mask": [[4, 7]],
                "output_token_sampling_logprobs": [-0.25],
            },
        )


def test_abort_before_sampling_does_not_require_sampling_metadata():
    # [FUNCTION DOC] Appending with no sampled tokens and meta info that only has an abort finish
    # reason returns an empty list and leaves the sample's mask empty.
    sample = Sample()

    assert append_sampling_metadata(sample, [], {"finish_reason": {"type": "abort"}}) == []
    assert len(sample.rollout_sampling_mask) == 0
