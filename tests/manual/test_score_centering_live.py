# [MODULE DOC] Opt-in live tests (skipped unless environment variables name a server endpoint
# and a local model path) for the rollout top-k log-probability contract. A module-scoped
# fixture queries a running SGLang server with a fixed math prompt at three temperatures through
# the native generate route and once through the OpenAI-compatible chat route, then the tests
# validate the returned metadata with the production helpers and check that top-k log
# probabilities scale consistently with temperature.
"""Exercise the production score-centering contract against a live SGLang server.

Set MILES_LIVE_SCORE_CENTERING_ENDPOINT and MILES_LIVE_SCORE_CENTERING_MODEL.
MILES_LIVE_SCORE_CENTERING_TOP_K optionally selects the candidate count (default 128).
This probe generates short responses; it does not train or modify the server.
"""

import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import requests
from transformers import AutoTokenizer

from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
    validate_rollout_topk_logprobs_sample,
)
from miles.utils.types import Sample


@pytest.fixture(scope="module")
def live_responses(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Module-scoped fixture that collects all server responses once.
    # Reads the endpoint and model path from the environment (skips the module if either is
    # missing), strips trailing slashes, reads the candidate count (default 128, must be
    # positive) and an artifact directory (environment value, else a fresh temp directory),
    # which is created. Loads the tokenizer from the model path and tokenizes a chat prompt (one
    # user message asking for 17 times 23) with the generation prompt, thinking enabled and
    # preserved. Sends a one-token warm-up generate request at temperature 1.0 so cached and
    # uncached prefill paths agree, then for temperatures 0.7, 1.0, 1.3 sends a native generate
    # request (32 new tokens, log probabilities returned, top-k request fields added by the
    # production configure helper) with a 300 second timeout and stores the JSON under the
    # temperature string. Then one chat-completions request (served model name from an optional
    # environment variable, default "default"; 32 tokens, logprobs, meta info, thinking flags)
    # configured in the OpenAI style at temperature 1.0 is stored under "openai". Writes all
    # outputs as JSON to a responses file in the artifact directory. Returns the mapping that
    # also holds the prompt token ids and the candidate count. Raises on any HTTP error status.
    endpoint = os.environ.get("MILES_LIVE_SCORE_CENTERING_ENDPOINT")
    model = os.environ.get("MILES_LIVE_SCORE_CENTERING_MODEL")
    if not endpoint or not model:
        pytest.skip("set the live score-centering endpoint and local model path to opt in")
    endpoint = endpoint.rstrip("/")
    top_k = int(os.environ.get("MILES_LIVE_SCORE_CENTERING_TOP_K", "128"))
    assert top_k > 0, "The live probe requires a positive candidate count"
    artifact_dir = Path(
        os.environ.get("MILES_LIVE_SCORE_CENTERING_ARTIFACT_DIR") or tmp_path_factory.mktemp("score-centering-live")
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    messages = [{"role": "user", "content": "Compute 17 times 23. Show your reasoning and put the answer in a box."}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=True,
        preserve_thinking=True,
    )
    # Warm the same prefix before comparing temperatures: GDN cached and
    # uncached prefills can use different kernels and produce different logits.
    warmup = requests.post(
        f"{endpoint}/generate",
        json={"input_ids": prompt, "sampling_params": {"max_new_tokens": 1, "temperature": 1.0}},
        timeout=300,
    )
    warmup.raise_for_status()
    outputs: dict[str, Any] = {"prompt": prompt, "top_k": top_k}
    for temperature in (0.7, 1.0, 1.3):
        args = Namespace(
            rollout_top_logprobs_num=top_k, rollout_sampling_logprobs_mode="selected", rollout_temperature=temperature
        )
        payload = {
            "input_ids": prompt,
            "return_logprob": True,
            "sampling_params": {"max_new_tokens": 32, "temperature": temperature},
        }
        configure_rollout_topk_logprobs_request(args, payload)
        response = requests.post(f"{endpoint}/generate", json=payload, timeout=300)
        response.raise_for_status()
        outputs[str(temperature)] = response.json()
    args = Namespace(
        rollout_top_logprobs_num=top_k, rollout_sampling_logprobs_mode="selected", rollout_temperature=1.0
    )
    payload = {
        "model": os.environ.get("MILES_LIVE_SCORE_CENTERING_SERVED_MODEL", "default"),
        "messages": messages,
        "max_tokens": 32,
        "logprobs": True,
        "return_meta_info": True,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
    }
    configure_rollout_topk_logprobs_request(args, payload, openai=True)
    response = requests.post(f"{endpoint}/v1/chat/completions", json=payload, timeout=300)
    response.raise_for_status()
    outputs["openai"] = response.json()
    (artifact_dir / "responses.json").write_text(json.dumps(outputs))
    return outputs


def _validate_metadata(meta: dict[str, Any], prompt: list[int], top_k: int) -> None:
    # [FUNCTION DOC] HELPER: Checks one response's meta info. Asserts at least one generated
    # token, builds a sample (tokens = prompt plus generated ids, response length, rollout log
    # probabilities from the pairs), appends the top-k log probabilities from the meta info using
    # the production helper, runs the production sample validator, then asserts every top-k token
    # id is non-negative (no padding) and every top-k log probability is finite.
    generated = meta["output_token_logprobs"]
    assert generated, "The server must generate at least one token"
    sample = Sample(
        tokens=prompt + [item[1] for item in generated],
        response_length=len(generated),
        rollout_log_probs=[item[0] for item in generated],
    )
    append_rollout_topk_logprobs(sample, meta, top_k)
    validate_rollout_topk_logprobs_sample(sample, top_k)
    assert np.all(sample.rollout_topk_token_ids >= 0), "This model should return the requested candidates"
    assert np.all(np.isfinite(sample.rollout_topk_log_probs))


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.3])
def test_native_sampler_contract(live_responses: dict[str, Any], temperature: float) -> None:
    # [FUNCTION DOC] Validates the native route response metadata for each temperature (0.7,
    # 1.0, 1.3) with the shared metadata checker.
    _validate_metadata(
        live_responses[str(temperature)]["meta_info"], live_responses["prompt"], live_responses["top_k"]
    )


def test_openai_sampler_contract(live_responses: dict[str, Any]) -> None:
    # [FUNCTION DOC] Validates the meta info of the first choice of the chat-completions response
    # with the shared metadata checker.
    choice = live_responses["openai"]["choices"][0]
    _validate_metadata(choice["meta_info"], live_responses["prompt"], live_responses["top_k"])


@pytest.mark.parametrize("temperature", [0.7, 1.3])
def test_candidate_probabilities_include_temperature(live_responses: dict[str, Any], temperature: float) -> None:
    # [FUNCTION DOC] For temperatures 0.7 and 1.3 compared with the 1.0 run: builds maps from
    # token id to log probability for the first generated position's top-k list; asserts both
    # runs report the same cached-token count; takes the sorted common token ids and asserts
    # there are at least min(100, max(1, three quarters of top-k)) of them; computes for each
    # common token temperature times the scaled log probability minus the base one, and asserts
    # the spread (max minus min) is under 0.005, since that difference is constant when the
    # logits are equal and only the unknown normalizer differs.
    # For the same first-token logits, T*log(q_T) - log(q_1) is constant.
    # Testing differences cancels the unknown full-vocabulary partition sums.
    base = {entry[1]: entry[0] for entry in live_responses["1.0"]["meta_info"]["output_top_logprobs"][0]}
    scaled = {entry[1]: entry[0] for entry in live_responses[str(temperature)]["meta_info"]["output_top_logprobs"][0]}
    base_meta = live_responses["1.0"]["meta_info"]
    scaled_meta = live_responses[str(temperature)]["meta_info"]
    assert base_meta.get("cached_tokens") == scaled_meta.get(
        "cached_tokens"
    ), "Temperature comparison requires the same prefill cache layout"
    common = sorted(base.keys() & scaled.keys())
    assert len(common) >= min(100, max(1, 3 * live_responses["top_k"] // 4))
    differences = np.asarray([temperature * scaled[token] - base[token] for token in common])
    assert (
        np.ptp(differences) < 5e-3
    ), f"Top-k logprobs do not match the temperature-adjusted sampler: spread={np.ptp(differences)}"
