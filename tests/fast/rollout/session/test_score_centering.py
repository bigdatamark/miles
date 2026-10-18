# [MODULE DOC] Tests of score centering (top log probability candidates and sampling support
# replay) in the session server layer, for both session implementations (v1 and v2 registries and
# cores). Covered: sessions record whether they are evaluation sessions and prepared chat requests
# differ accordingly (training requests carry the configured candidate count and temperature and
# reject a conflicting temperature; evaluation requests keep the client's values); candidate
# trimming when assembling samples from recorded requests with token-in-token-out token lists;
# evaluation sessions not collecting training candidates; filtered (top-p/top-k) requests asking
# for the sampling mask and support-mode log probabilities; the session owning the sampling log
# probability mode over client-supplied values; validation of the final sampling rules; and
# explicit defaults for nullable sampling fields. Uses the shared score centering and session
# fixtures and fake tokenizers.
"""Session configuration must distinguish training calls from evaluation."""

import asyncio
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from tests.fast.fixtures.score_centering_fixtures import _args, _meta, _Tokenizer
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.core import SessionCore
from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.request_args import prepare_chat_request
from miles.rollout.session.samples.merge import compute_samples_from_openai_records
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import SessionRegistryV2


@pytest.mark.parametrize(
    "registry_type,core_type", [(SessionRegistry, SessionCore), (SessionRegistryV2, SessionCoreV2)]
)
def test_score_centering_training_and_evaluation_sessions(registry_type: type, core_type: type) -> None:
    # [FUNCTION DOC] Parametrized over the v1 and v2 registry and core types. Config asks for
    # 128 candidates and temperature 0.7; the tokenizer is a namespace with a no-op comparator
    # creator, empty chat template kwargs and an identity request resolver. For a training and an
    # evaluation session (created through the core, each in its own event loop run), looks the
    # session up from the response's session id and asserts its evaluation flag. For evaluation:
    # a prepared request with temperature 0 keeps temperature 0 and has no top logprobs. For
    # training: an empty request gets 128 top logprobs and temperature 0.7, and a request with
    # temperature 0 raises a message validation error mentioning temperature.
    config = make_session_server_config(rollout_top_logprobs_num=128, rollout_temperature=0.7)
    tokenizer = SimpleNamespace(
        create_comparator=lambda: None,
        chat_template_kwargs={},
        resolve_request_args=lambda request, **kwargs: request,
    )
    registry = registry_type(tokenizer=None, tito_tokenizer=tokenizer)
    core = core_type(None, registry, config)
    for evaluation in (False, True):
        response = asyncio.run(core.create_session(evaluation=evaluation))
        session = registry.get_session(json.loads(response.body)["session_id"])
        assert session.evaluation is evaluation
        if evaluation:
            request = prepare_chat_request(
                {"temperature": 0}, tokenizer, config=config, turn_args=None, evaluation=session.evaluation
            ).body
            assert request["temperature"] == 0 and "top_logprobs" not in request
        else:
            request = prepare_chat_request(
                {}, tokenizer, config=config, turn_args=None, evaluation=session.evaluation
            ).body
            assert request["top_logprobs"] == 128 and request["temperature"] == 0.7
            with pytest.raises(MessageValidationError, match="temperature"):
                prepare_chat_request(
                    {"temperature": 0}, tokenizer, config=config, turn_args=None, evaluation=session.evaluation
                )


def test_session_producer_trims_candidates_with_tito_tokens() -> None:
    # [FUNCTION DOC] Builds two recorded chat requests (prompts [0,1] and [0,1,2,6] with outputs
    # of two tokens each and candidate metadata) and assembles samples from them with the
    # accumulated token list [0,1,2,6,4,5] and a maximum trim of 1 token. Asserts the first sample
    # has response length 1 (one token trimmed) with candidate ids of shape (1, 3); after merging
    # the samples the loss mask is 1,0,1,1 and the merged sample passes candidate validation for 3
    # candidates.
    records = []
    for prompt, output, probabilities in (([0, 1], [2, 3], [0.5, 0.25]), ([0, 1, 2, 6], [4, 5], [0.55, 0.2])):
        records.append(
            SessionRecord(
                timestamp=2.0,
                request_timestamp=1.0,
                method="POST",
                path="v1/chat/completions",
                status_code=200,
                request={"input_ids": prompt, "top_logprobs": 3},
                response={"choices": [{"meta_info": _meta(output, probabilities), "finish_reason": "stop"}]},
            )
        )
    samples = compute_samples_from_openai_records(
        _args(save_debug_trajectory_data=None, sglang_speculative_algorithm=None),
        records,
        _Tokenizer(),
        accumulated_token_ids=[0, 1, 2, 6, 4, 5],
        max_trim_tokens=1,
    )
    assert samples[0].response_length == 1
    assert samples[0].rollout_topk_token_ids.shape == (1, 3)
    merged = merge_samples(samples, _Tokenizer())
    assert merged.loss_mask == [1, 0, 1, 1]
    validate_rollout_topk_logprobs_sample(merged, 3)


@pytest.mark.parametrize("evaluation", [False, True])
def test_evaluation_client_top_logprobs_do_not_collect_training_candidates(evaluation: bool) -> None:
    # [FUNCTION DOC] Parametrized over the evaluation flag. A recorded request that includes a
    # client-requested top logprobs value is assembled into samples. Asserts exactly one sample,
    # and that its candidate id and log probability arrays are none exactly when the flag is
    # evaluation (training collects them, evaluation does not).
    record = SessionRecord(
        timestamp=2.0,
        request_timestamp=1.0,
        method="POST",
        path="v1/chat/completions",
        status_code=200,
        request={"input_ids": [0, 1], "top_logprobs": 3},
        response={"choices": [{"meta_info": _meta([2], [0.5]), "finish_reason": "stop"}]},
    )
    samples = compute_samples_from_openai_records(
        _args(save_debug_trajectory_data=None, sglang_speculative_algorithm=None),
        [record],
        _Tokenizer(),
        evaluation=evaluation,
    )
    assert len(samples) == 1
    assert (samples[0].rollout_topk_token_ids is None) is evaluation
    assert (samples[0].rollout_topk_log_probs is None) is evaluation


def test_filtered_session_request_and_producer() -> None:
    # [FUNCTION DOC] With 3 candidates, the support log probability mode, temperature 0.7 and
    # support replay on, prepares a chat request with sampling defaults (temperature 0.7, top-p
    # 0.6, top-k 3). Asserts the body asks for the sampling mask, sets the support mode and has no
    # top logprobs. Then builds a recorded response with one token (probability 0.4), a support of
    # two token ids and post-filter log probabilities (4/7 and 3/7), assembles samples with the
    # token list [0,1,2] and no trimming, and asserts one sample that passes candidate validation,
    # has candidate ids [2, 3, -1] and a rollout log probability of log(4/7) (the post-filter
    # value of the sampled token).
    config = make_session_server_config(
        rollout_top_logprobs_num=3,
        rollout_sampling_logprobs_mode="support",
        rollout_temperature=0.7,
        use_sampling_support_replay=True,
    )
    tokenizer = SimpleNamespace(resolve_request_args=lambda request, **kwargs: request)
    prepared = prepare_chat_request(
        {"messages": [{"role": "user", "content": "Solve."}]},
        tokenizer,
        config=config,
        turn_args=None,
        sampling_defaults={"temperature": 0.7, "top_p": 0.6, "top_k": 3},
        sampling_support_replay=True,
    )
    assert prepared.body["return_sampling_mask"] is True
    assert prepared.body["sampling_logprobs_mode"] == "support"
    assert "top_logprobs" not in prepared.body
    meta = {
        "output_token_logprobs": [(math.log(0.4), 2, None)],
        "output_token_sampling_mask": [[2, 3]],
        "output_token_sampling_logprobs": [[math.log(4 / 7), math.log(3 / 7)]],
    }
    record = SessionRecord(
        timestamp=2.0,
        request_timestamp=1.0,
        method="POST",
        path="v1/chat/completions",
        status_code=200,
        request={"input_ids": [0, 1], **prepared.body},
        response={"choices": [{"meta_info": meta, "finish_reason": "stop"}]},
    )
    samples = compute_samples_from_openai_records(
        _args(save_debug_trajectory_data=None, sglang_speculative_algorithm=None),
        [record],
        _Tokenizer(),
        accumulated_token_ids=[0, 1, 2],
        max_trim_tokens=0,
    )
    assert len(samples) == 1
    validate_rollout_topk_logprobs_sample(samples[0], 3)
    np.testing.assert_array_equal(samples[0].rollout_topk_token_ids, [[2, 3, -1]])
    np.testing.assert_allclose(samples[0].rollout_log_probs, [math.log(4 / 7)])


@pytest.mark.parametrize("candidate_count", [0, 3])
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("evaluation", [False, True])
def test_session_request_owns_sampling_logprobs_mode(candidate_count: int, filtered: bool, evaluation: bool) -> None:
    # [FUNCTION DOC] Parametrized over candidate count (0, 3), filtered sampling and evaluation.
    # The client request carries the opposite log probability mode and 5 top logprobs, and the
    # request resolver merges the client values back in. Asserts the prepared body requests the
    # sampling mask exactly when filtered and not evaluation. For training with candidates:
    # filtered gives support mode and removes top logprobs; unfiltered removes the mode field and
    # sets top logprobs to the candidate count. Otherwise (no candidates or evaluation) the mode
    # field is absent and the client's 5 top logprobs remain.
    mode = "support" if filtered else "selected"
    client_args = {"sampling_logprobs_mode": "selected" if filtered else "support", "top_logprobs": 5}
    tokenizer = SimpleNamespace(resolve_request_args=lambda request, **kwargs: request | client_args)
    prepared = prepare_chat_request(
        client_args,
        tokenizer,
        config=make_session_server_config(
            rollout_top_logprobs_num=candidate_count,
            rollout_sampling_logprobs_mode=mode,
            rollout_temperature=0.7,
            use_sampling_support_replay=filtered,
        ),
        turn_args=None,
        evaluation=evaluation,
        sampling_defaults={"temperature": 0.7, "top_p": 0.6 if filtered else 1.0, "top_k": 3 if filtered else -1},
        sampling_support_replay=filtered,
    )
    assert prepared.body.get("return_sampling_mask", False) is (filtered and not evaluation)
    if candidate_count and not evaluation:
        if filtered:
            assert prepared.body["sampling_logprobs_mode"] == "support"
            assert "top_logprobs" not in prepared.body
        else:
            assert "sampling_logprobs_mode" not in prepared.body
            assert prepared.body["top_logprobs"] == candidate_count
    else:
        assert "sampling_logprobs_mode" not in prepared.body
        assert prepared.body["top_logprobs"] == 5


@pytest.mark.parametrize(
    "override,match",
    [
        ({"temperature": 0.8}, "Rollout top-k logprobs"),
        ({"top_p": 0.9}, "bounded training-request sampling"),
        ({"min_p": 0.1}, "Rollout top-k logprobs"),
        ({"logit_bias": {"1": 2}}, "Rollout top-k logprobs"),
    ],
)
def test_final_model_sampling_rules_are_validated(override: dict, match: str) -> None:
    # [FUNCTION DOC] Parametrized over a changed temperature, a top-p below 1, a min-p and a
    # logit bias. The request resolver applies the override to the request after client values are
    # merged, simulating what the final model request would hold. Asserts preparing the request
    # raises a message validation error whose text matches (top-k log probability rules, or
    # bounded training-request sampling for top-p), and that the client's argument mapping was not
    # mutated.
    config = make_session_server_config(rollout_top_logprobs_num=128, rollout_temperature=0.7)
    tokenizer = SimpleNamespace(resolve_request_args=lambda request, **kwargs: request | override)
    client_args = {"messages": [{"role": "user", "content": "Solve the problem."}]}
    with pytest.raises(MessageValidationError, match=match):
        prepare_chat_request(client_args, tokenizer, config=config, turn_args=None)
    assert client_args == {"messages": [{"role": "user", "content": "Solve the problem."}]}


def test_nullable_sampling_fields_receive_explicit_defaults() -> None:
    # [FUNCTION DOC] A client request with temperature, top-p, top-k and min-p all set to none is
    # prepared with 128 candidates and temperature 0.7. Asserts the body holds explicit defaults:
    # temperature 0.7, top-p 1.0, top-k -1 and min-p 0.0.
    config = make_session_server_config(rollout_top_logprobs_num=128, rollout_temperature=0.7)
    tokenizer = SimpleNamespace(resolve_request_args=lambda request, **kwargs: request)
    prepared = prepare_chat_request(
        {"temperature": None, "top_p": None, "top_k": None, "min_p": None},
        tokenizer,
        config=config,
        turn_args=None,
    )
    assert {key: prepared.body[key] for key in ("temperature", "top_p", "top_k", "min_p")} == {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
    }
