# [MODULE DOC] Async tests of recomputing rollout log probabilities with a prefill-only request to
# the inference server (the response tail of a sample is scored again by the engine with new token
# count zero and temperature zero). The module's HTTP post function is replaced by fakes that
# record calls and return canned input-token log probability lists (the first entry has no
# probability; the rest carry a probability and token id). Covered: single-sample recompute using
# the response tail, token alignment validation, batched recompute with a cache flush before each
# batch, and grouping of samples into batches by the log probability start position.
from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils import prefill_logprobs
from miles.utils.types import Sample


@pytest.mark.asyncio
async def test_recompute_rollout_logprobs_via_prefill_uses_response_tail(monkeypatch):
    # [FUNCTION DOC] A completed sample has six tokens, response length 3 and placeholder log
    # probabilities. A fake post records its URL, payload and headers and returns four input token
    # log probabilities (the first without a value). After recompute with a URL, sampling
    # parameters and headers, asserts the sample's log probabilities become the last three values,
    # its metadata records the source as sglang prefill recompute, the call used the given URL and
    # headers, the payload has all sample tokens as input ids, requests log probabilities, starts
    # them at position 2 (one token before the response start: 6 tokens minus 3 response tokens
    # minus 1), and overrides the sampling parameters to zero new tokens and zero temperature.
    sample = Sample(
        tokens=[10, 11, 12, 20, 21, 22],
        response_length=3,
        rollout_log_probs=[-9.0, -9.0, -9.0],
        status=Sample.Status.COMPLETED,
    )
    args = SimpleNamespace(recompute_logprobs_via_prefill=True, sglang_enable_lora=False)
    seen = {}

    async def fake_post(url, payload, headers=None):
        # [FUNCTION DOC] HELPER: Fake post; stores the URL, payload and headers in the enclosing
        # record and returns a response whose meta info has four input token log probabilities:
        # a first entry without a value for token 12, then -0.1, -0.2 and -0.3 for tokens 20, 21
        # and 22.
        seen["url"] = url
        seen["payload"] = payload
        seen["headers"] = headers
        return {
            "meta_info": {
                "input_token_logprobs": [
                    (None, 12),
                    (-0.1, 20),
                    (-0.2, 21),
                    (-0.3, 22),
                ]
            }
        }

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    await prefill_logprobs.recompute_rollout_logprobs_via_prefill(
        args,
        sample,
        url="http://localhost/generate",
        sampling_params={"temperature": 1, "max_new_tokens": 128},
        headers={"X-Test": "1"},
    )

    assert sample.rollout_log_probs == [-0.1, -0.2, -0.3]
    assert sample.metadata["rollout_log_probs_source"] == "sglang_prefill_recompute"
    assert seen["url"] == "http://localhost/generate"
    assert seen["headers"] == {"X-Test": "1"}
    assert seen["payload"]["input_ids"] == sample.tokens
    assert seen["payload"]["return_logprob"] is True
    assert seen["payload"]["logprob_start_len"] == 2
    assert seen["payload"]["sampling_params"]["max_new_tokens"] == 0
    assert seen["payload"]["sampling_params"]["temperature"] == 0


@pytest.mark.asyncio
async def test_recompute_rollout_logprobs_via_prefill_checks_token_alignment(monkeypatch):
    # [FUNCTION DOC] The fake server echoes a wrong token id (999 instead of the sample's 20) for
    # the response position. Asserts recompute raises a value error mentioning a token alignment
    # mismatch.
    sample = Sample(tokens=[10, 11, 20], response_length=1, status=Sample.Status.COMPLETED)
    args = SimpleNamespace(recompute_logprobs_via_prefill=True, sglang_enable_lora=False)

    async def fake_post(url, payload, headers=None):
        # [FUNCTION DOC] HELPER: Fake post returning two input token log probabilities whose
        # second token id (999) does not match the sample's response token.
        return {"meta_info": {"input_token_logprobs": [(None, 11), (-0.1, 999)]}}

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    with pytest.raises(ValueError, match="token alignment mismatch"):
        await prefill_logprobs.recompute_rollout_logprobs_via_prefill(
            args,
            sample,
            url="http://localhost/generate",
            sampling_params={},
        )


@pytest.mark.asyncio
async def test_recompute_samples_flushes_each_batch_and_batches_prefill_score(monkeypatch):
    # [FUNCTION DOC] Two completed samples with equal prompt length and a one-token response are
    # recomputed together with a round-robin router policy. Asserts the new log probabilities are
    # minus the response token ids (-20 and -21); that exactly two posts happened, first to the
    # cache flush endpoint and then to the generate endpoint, both with the default post action;
    # and that the generate payload carries both token lists as one batch with log probability
    # start position 1.
    samples = [
        Sample(tokens=[10, 11, 20], response_length=1, status=Sample.Status.COMPLETED),
        Sample(tokens=[10, 11, 21], response_length=1, status=Sample.Status.COMPLETED),
    ]
    args = SimpleNamespace(
        recompute_logprobs_via_prefill=True,
        sglang_enable_lora=False,
        sglang_router_policy="round_robin",
    )
    calls = []

    async def fake_post(url, payload, action="post", headers=None):
        # [FUNCTION DOC] HELPER: Fake post; records the call. For URLs ending in the cache flush
        # path returns an empty mapping; otherwise returns, for every token list in the payload's
        # input ids, a response whose input token log probabilities are an unscored second-to-last
        # entry and the last token with a value of minus its id.
        calls.append((url, payload, action, headers))
        if url.endswith("/flush_cache"):
            return {}
        return [
            {"meta_info": {"input_token_logprobs": [(None, 11), (-float(tokens[-1]), tokens[-1])]}}
            for tokens in payload["input_ids"]
        ]

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    await prefill_logprobs.recompute_samples_rollout_logprobs_via_prefill(
        args,
        samples,
        url="http://localhost/generate",
        sampling_params={"max_new_tokens": 32},
    )

    assert [sample.rollout_log_probs for sample in samples] == [[-20.0], [-21.0]]
    assert [call[0] for call in calls] == [
        "http://localhost/flush_cache",
        "http://localhost/generate",
    ]
    assert [call[2] for call in calls] == ["post", "post"]
    assert calls[1][1]["input_ids"] == [[10, 11, 20], [10, 11, 21]]
    assert calls[1][1]["logprob_start_len"] == 1


@pytest.mark.asyncio
async def test_recompute_samples_batches_by_logprob_start_len(monkeypatch):
    # [FUNCTION DOC] Three samples, the middle one with a longer prompt (so a different log
    # probability start position). Asserts each gets its minus-id log probability; that the posts
    # are a cache flush then generate for the first batch and again a flush then generate for the
    # second; that the first batch holds samples one and three with start position 1; and that the
    # second batch holds only the longer sample with start position 2 (batching groups samples by
    # start position, ordered by first appearance).
    samples = [
        Sample(tokens=[10, 11, 20], response_length=1, status=Sample.Status.COMPLETED),
        Sample(tokens=[10, 11, 12, 21], response_length=1, status=Sample.Status.COMPLETED),
        Sample(tokens=[10, 11, 22], response_length=1, status=Sample.Status.COMPLETED),
    ]
    args = SimpleNamespace(
        recompute_logprobs_via_prefill=True,
        sglang_enable_lora=False,
        sglang_router_policy="round_robin",
    )
    calls = []

    async def fake_post(url, payload, action="post", headers=None):
        # [FUNCTION DOC] HELPER: Fake post; records the call. For cache flush URLs returns an
        # empty mapping; otherwise returns, per token list in the payload, a response whose input
        # token log probabilities are an unscored entry for the second-to-last token and the last
        # token scored as minus its id.
        calls.append((url, payload, action, headers))
        if url.endswith("/flush_cache"):
            return {}
        return [
            {
                "meta_info": {
                    "input_token_logprobs": [
                        (None, tokens[-2]),
                        (-float(tokens[-1]), tokens[-1]),
                    ]
                }
            }
            for tokens in payload["input_ids"]
        ]

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    await prefill_logprobs.recompute_samples_rollout_logprobs_via_prefill(
        args,
        samples,
        url="http://localhost/generate",
        sampling_params={"max_new_tokens": 32},
    )

    assert [sample.rollout_log_probs for sample in samples] == [
        [-20.0],
        [-21.0],
        [-22.0],
    ]
    assert [call[0] for call in calls] == [
        "http://localhost/flush_cache",
        "http://localhost/generate",
        "http://localhost/flush_cache",
        "http://localhost/generate",
    ]
    assert calls[1][1]["logprob_start_len"] == 1
    assert calls[1][1]["input_ids"] == [[10, 11, 20], [10, 11, 22]]
    assert calls[3][1]["logprob_start_len"] == 2
    assert calls[3][1]["input_ids"] == [[10, 11, 12, 21]]
