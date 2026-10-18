# [MODULE DOC] Tests of score-centering (top log probability candidates and sampling support
# information) recorded by the native generate endpoint helpers: building the request payload from
# arguments and updating a sample from a server response. Fake responses come from the shared score
# centering fixtures (argument and meta-info builders); the asynchronous update is driven with a
# fresh event loop per call.
"""Exercise score-centering metadata through native generation."""

import asyncio
import math

import numpy as np
import pytest
from tests.fast.fixtures.score_centering_fixtures import _args, _meta

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, update_sample_from_response
from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample
from miles.utils.types import Sample


def test_native_generate_producer_appends_each_call() -> None:
    # [FUNCTION DOC] Builds the request payload for a two-token prompt (max response 20, no
    # replay features, no speculative decoding) and asserts it has no error status, requests 3 top
    # log probabilities and enables log probabilities. Then applies two successive responses (two
    # tokens each, with different probabilities) to one sample and asserts the sample tokens are
    # the prompt plus all four generated tokens, the top-k token ids' first column is [2, 2, 4, 4]
    # (appended per call across calls), and the top-k sample validator accepts it for 3 candidates.
    args = _args(
        rollout_max_response_len=20,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
    )
    payload, status = compute_request_payload(args, [0, 1], {})
    assert status is None and payload["top_logprobs_num"] == 3 and payload["return_logprob"]
    sample = Sample()
    for output, probabilities in (([2, 3], [0.5, 0.25]), ([4, 5], [0.55, 0.2])):
        asyncio.run(
            update_sample_from_response(
                args, sample, payload, {"text": str(output), "meta_info": _meta(output, probabilities)}, True
            )
        )
    assert sample.tokens == [0, 1, 2, 3, 4, 5]
    np.testing.assert_array_equal(sample.rollout_topk_token_ids[:, 0], [2, 2, 4, 4])
    validate_rollout_topk_logprobs_sample(sample, 3)


@pytest.mark.parametrize("candidate_count", [0, 3])
def test_filtered_native_generation_records_post_filter_distribution(candidate_count: int) -> None:
    # [FUNCTION DOC] Parametrized by the candidate count (0 or 3). Arguments request top-p 0.6,
    # top-k 3 and the support sampling log probability mode with support replay on. Asserts the
    # payload has no error status, requests the sampling mask, includes the support log
    # probability mode only when candidates are requested, and carries no top log probabilities
    # number. A response with one token of probability 0.4, a sampling mask of two token ids and
    # the post-filter log probabilities (3/7 and 4/7; a flat list for the zero-candidate case) is
    # applied. With candidates, asserts the sample passes top-k validation, its top-k ids are the
    # mask ids padded with -1, and the exponentiated first two top-k log probabilities are 3/7 and
    # 4/7. Without candidates, asserts both top-k fields stay none. In both cases the sample's
    # rollout log probability is the log of 4/7 (the post-filter value, not the raw 0.4).
    args = _args(
        rollout_top_logprobs_num=candidate_count,
        rollout_top_p=0.6,
        rollout_top_k=3,
        rollout_sampling_logprobs_mode="support",
        use_sampling_support_replay=True,
        rollout_max_response_len=20,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
    )
    payload, status = compute_request_payload(args, [0, 1], {"temperature": 0.7, "top_p": 0.6, "top_k": 3})
    assert status is None
    assert payload["return_sampling_mask"]
    assert payload.get("sampling_logprobs_mode") == ("support" if candidate_count else None)
    assert "top_logprobs_num" not in payload
    meta = {
        "output_token_logprobs": [(math.log(0.4), 2, None)],
        "output_token_sampling_mask": [[3, 2]],
        "output_token_sampling_logprobs": (
            [[math.log(3 / 7), math.log(4 / 7)]] if candidate_count else [math.log(4 / 7)]
        ),
        "finish_reason": {"type": "stop"},
    }
    sample = Sample()
    asyncio.run(update_sample_from_response(args, sample, payload, {"text": "2", "meta_info": meta}, True))
    if candidate_count:
        validate_rollout_topk_logprobs_sample(sample, 3)
        np.testing.assert_array_equal(sample.rollout_topk_token_ids, [[3, 2, -1]])
        np.testing.assert_allclose(np.exp(sample.rollout_topk_log_probs[0, :2]), [3 / 7, 4 / 7])
    else:
        assert sample.rollout_topk_token_ids is None and sample.rollout_topk_log_probs is None
    np.testing.assert_allclose(sample.rollout_log_probs, [math.log(4 / 7)])


def test_greedy_evaluation_does_not_collect_training_candidates() -> None:
    # [FUNCTION DOC] Builds an evaluation payload with temperature 0 and asserts the temperature
    # is kept and no top log probabilities number is requested. Applies a response whose meta info
    # has the output top log probabilities entry removed (the shared builder normally includes
    # it), and asserts the sample's top-k token ids stay none.
    args = _args(
        rollout_max_response_len=20,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
    )
    payload, _ = compute_request_payload(args, [0, 1], {"temperature": 0.0}, evaluation=True)
    assert payload["sampling_params"]["temperature"] == 0.0 and "top_logprobs_num" not in payload
    meta = _meta([2, 3], [0.5, 0.25])
    del meta["output_top_logprobs"]
    sample = Sample()
    asyncio.run(update_sample_from_response(args, sample, payload, {"text": "eval", "meta_info": meta}))
    assert sample.rollout_topk_token_ids is None
