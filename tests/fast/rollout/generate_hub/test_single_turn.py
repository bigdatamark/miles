# [MODULE DOC] Tests of single-turn generation, run against three variants of the generate
# function: the old sglang rollout function, the single-turn function and the multi-turn function
# used for a single turn. The shared generation fixtures provide a mock inference server and
# arguments; a runner returns the requests the mock server saw and the produced sample(s).
# Module constants describe a fixed prompt ("What is 1+7?", 7 tokens) answered by a fixed response
# (a boxed 8, 5 tokens) with a predictable log probability ramp; helper builders produce the
# expected request payload and expected sample for each variant (variants differ in whether
# replay flags appear in the request and whether a loss mask is set). Registered as a CPU CI test
# in the batch-b CPU suite. Some tests download tokenizers or a vision-language processor.
# Covered: basic generation, explicit endpoint routing, resuming an aborted sample, finish reasons,
# routed experts, meta info (cached tokens, weight versions, speculative info), allowed input
# statuses, sampling parameter pass-through, context length boundaries, empty responses and
# multimodal inputs.
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=220, suite="stage-b-cpu", labels=[])

import numpy as np
import pybase64
import pytest
import torch
from PIL import Image
from tests.fast.fixtures.generation_fixtures import GenerateEnv, generation_env, listify, make_sample, run_generate
from transformers import AutoProcessor

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub import single_turn
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.test_utils.mock_sglang_server import ProcessResult, ProcessResultMetaInfo
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

_ = generation_env

# ------------------------------------ fixtures and consts ----------------------------------------


MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPT = "What is 1+7?"
PROMPT_TOKENS = [3838, 374, 220, 16, 10, 22, 30]
PROMPT_TOKEN_LEN = len(PROMPT_TOKENS)
RESPONSE_TOKENS = [59, 79075, 90, 23, 92]
RESPONSE_TEXT = "\\boxed{8}"
RESPONSE_LOG_PROBS = [-0.0, -0.0078125, -0.015625, -0.0234375, -0.03125]
SAMPLING_PARAMS = {"max_new_tokens": 16, "temperature": 0.7}
DEFAULT_MAX_NEW_TOKENS = SAMPLING_PARAMS["max_new_tokens"]


@pytest.fixture(params=["old_sglang_rollout", "single_turn", "multi_turn"])
def variant(request):
    # [FUNCTION DOC] HELPER: Fixture parametrized over the three generate variants (old sglang
    # rollout, single turn, multi turn); returns the current variant name.
    return request.param


def expected_request(
    variant: str,
    *,
    input_ids: list[int] | None = None,
    sampling_params: dict | None = None,
    return_routed_experts: bool = False,
    return_indexer_topk: bool = False,
    image_data: list[str] | None = None,
) -> dict:
    # [FUNCTION DOC] HELPER: Builds the request payload the mock server is expected to receive:
    # input ids (default: the fixed prompt tokens), sampling parameters (default: module values)
    # and log probability return on. The routed experts and indexer top-k flags are included for
    # the single-turn and multi-turn variants always, and for the old variant only when requested
    # true. Image data is included only when given.
    result = {
        "input_ids": input_ids or PROMPT_TOKENS,
        "sampling_params": sampling_params or SAMPLING_PARAMS,
        "return_logprob": True,
    }
    if variant in ("single_turn", "multi_turn") or return_routed_experts:
        result["return_routed_experts"] = return_routed_experts
    if variant in ("single_turn", "multi_turn") or return_indexer_topk:
        result["return_indexer_topk"] = return_indexer_topk
    if image_data is not None:
        result["image_data"] = image_data
    return result


class _Unset:
    # [CLASS DOC] HELPER: Marker type for the sentinel meaning "argument not provided", needed so
    # that none can be passed as a real value to the expected sample builder.
    pass


_UNSET = _Unset()


def expected_sample(
    variant: str,
    *,
    prompt: str = PROMPT,
    response: str = RESPONSE_TEXT,
    response_length: int = 5,
    tokens: list[int] | None | _Unset = _UNSET,
    rollout_log_probs: list[float] | None | _Unset = _UNSET,
    status: Sample.Status = Sample.Status.COMPLETED,
    cached_tokens: int = 0,
    prompt_tokens: int = 7,
    weight_versions: list[str | None] | None = None,
    rollout_routed_experts: np.ndarray | None = None,
    spec_info: Sample.SpecInfo | None = None,
    multimodal_inputs: dict | None = None,
    multimodal_train_inputs: dict | None = None,
    loss_mask: list[int] | None | _Unset = _UNSET,
) -> Sample:
    # [FUNCTION DOC] HELPER: Builds the full expected output sample. Defaults: the fixed prompt,
    # response text and log probabilities, response length 5, completed status, zero cached tokens,
    # 7 prompt tokens, one call with no weight version, no routed experts, default speculative
    # info, no multimodal data. Unset tokens default to prompt plus response tokens. An unset loss
    # mask defaults to all ones of the response length for the multi-turn variant and none for
    # others. Each entry of the weight versions list becomes one per-call record: none means no
    # spans, a version means one span covering the response part of the tokens (from tokens length
    # minus response length to tokens length). Group index, index, label, reward are none; the
    # metadata is empty, the removal flag false, non-generation time zero, and prefix cache info
    # holds the cached and total prompt token counts.
    actual_response_length = response_length if response_length is not None else len(RESPONSE_TOKENS)
    if isinstance(loss_mask, _Unset):
        loss_mask = [1] * actual_response_length if variant == "multi_turn" else None
    actual_tokens = PROMPT_TOKENS + RESPONSE_TOKENS if isinstance(tokens, _Unset) else tokens
    expected_weight_versions = [
        WeightVersionsPerCall(
            spans=(
                []
                if version is None
                else [WeightVersionSpan(version, len(actual_tokens) - actual_response_length, len(actual_tokens))]
            )
        )
        for version in (weight_versions if weight_versions is not None else [None])
    ]

    return Sample(
        group_index=None,
        index=None,
        prompt=prompt,
        tokens=actual_tokens,
        multimodal_inputs=multimodal_inputs,
        multimodal_train_inputs=multimodal_train_inputs,
        response=response,
        response_length=response_length,
        label=None,
        reward=None,
        loss_mask=loss_mask,
        weight_versions=expected_weight_versions,
        rollout_log_probs=RESPONSE_LOG_PROBS if isinstance(rollout_log_probs, _Unset) else rollout_log_probs,
        rollout_routed_experts=rollout_routed_experts,
        remove_sample=False,
        status=status,
        metadata={},
        train_metadata=None,
        non_generation_time=0.0,
        spec_info=spec_info or Sample.SpecInfo(),
        prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=cached_tokens, total_prompt_tokens=prompt_tokens),
    )


def _make_sample(tokens=None, response="", response_length=0, status=Sample.Status.PENDING, multimodal_inputs=None):
    # [FUNCTION DOC] HELPER: Builds an input sample with the fixed prompt via the shared sample
    # factory; defaults give a fresh pending sample with no tokens and an empty response.
    return make_sample(
        prompt=PROMPT,
        tokens=tokens,
        response=response,
        response_length=response_length,
        status=status,
        multimodal_inputs=multimodal_inputs,
    )


def _run_generate(variant: str, env: GenerateEnv, sample: Sample | None = None, sampling_params: dict | None = None):
    # [FUNCTION DOC] HELPER: Runs the shared generate runner for the variant with the given
    # sample (default: a fresh pending sample) and sampling parameters (default: module values).
    # Returns the runner's result with the recorded requests and sample(s).
    return run_generate(env, sample or _make_sample(), sampling_params or SAMPLING_PARAMS, variant=variant)


# ------------------------------------ tests ----------------------------------------


class TestBasicGeneration:
    # [CLASS DOC] The baseline generation test.
    def test_basic_generation(self, variant, generation_env):
        # [FUNCTION DOC] Runs one generation with defaults and asserts exactly one expected request
        # reached the server and the output equals the default expected sample.
        result = _run_generate(variant, generation_env)
        assert result.requests == [expected_request(variant)]
        assert listify(result.sample) == [expected_sample(variant)]


class TestEndpointRouting:
    # [CLASS DOC] Tests of the single-turn function's optional explicit endpoint argument.
    @pytest.mark.parametrize("variant", ["single_turn"])
    async def test_an_explicit_url_routes_generation_to_that_endpoint(
        self, monkeypatch: pytest.MonkeyPatch, variant: str, generation_env: GenerateEnv
    ) -> None:
        # [FUNCTION DOC] Single-turn variant only. Replaces the module's HTTP post function with a
        # fake recording the URL and returning an empty stop response. Builds a generate input from
        # a fresh generation state and calls generate with an explicit router URL. Asserts the only
        # URL posted to is that explicit URL, not the router address configured in the arguments.
        """An explicit endpoint overrides the router address configured in the arguments."""
        requested_urls: list[str] = []

        async def fake_post(
            url: str, payload: dict[str, object], headers: dict[str, str] | None = None
        ) -> dict[str, object]:
            # [FUNCTION DOC] HELPER: Replacement post function; records the URL and returns an
            # empty-text response with a stop finish reason. Payload and headers are ignored.
            requested_urls.append(url)
            return {"text": "", "meta_info": {"finish_reason": {"type": "stop"}}}

        monkeypatch.setattr(single_turn, "post", fake_post)
        state = GenerateState(generation_env.args)
        generate_input = GenerateFnInput(
            state=state,
            sample=_make_sample(),
            sampling_params=SAMPLING_PARAMS.copy(),
            evaluation=False,
        )
        explicit_url = "http://policy-router:4321/generate"

        await single_turn.generate(generate_input, url=explicit_url)

        assert requested_urls == [explicit_url]


class TestResumedSingleTurn:
    # [CLASS DOC] Tests of resuming generation on a sample that was previously aborted.
    def test_two_consecutive_calls_on_same_sample(self, variant, generation_env):
        # [FUNCTION DOC] Skipped for the multi-turn variant. First call: the server returns the
        # first part of the response (two tokens) with an abort finish reason; asserts the normal
        # request and an aborted sample holding the partial text, tokens and log probabilities.
        # Second call on that returned sample: the server returns the remaining three tokens with
        # stop; asserts the request now uses the tokens so far as input ids and a maximum new
        # tokens reduced from 16 to 14 (16 minus the 2 already generated), and the final completed
        # sample has the concatenated text, tokens and log probabilities, response length 5, two
        # weight version records without spans, and a total prompt token count of prompt length
        # plus the length of the tokens after turn one.
        if variant == "multi_turn":
            pytest.skip("not tested yet")
        partial_text = "\\boxed"
        partial_tokens = [59, 79075]
        partial_log_probs = [-0.0, -0.0078125]

        remaining_text = "{8}"
        remaining_tokens = [90, 23, 92]
        remaining_log_probs = [-0.0, -0.0078125, -0.015625]

        generation_env.mock_server.process_fn = lambda _: ProcessResult(text=partial_text, finish_reason="abort")
        sample = _make_sample()
        result1 = _run_generate(variant, generation_env, sample)
        assert result1.requests == [expected_request(variant)]
        assert result1.sample == expected_sample(
            variant,
            response=partial_text,
            response_length=2,
            tokens=PROMPT_TOKENS + partial_tokens,
            rollout_log_probs=partial_log_probs,
            status=Sample.Status.ABORTED,
        )

        generation_env.mock_server.process_fn = lambda _: ProcessResult(text=remaining_text, finish_reason="stop")
        result2 = _run_generate(variant, generation_env, result1.sample)
        tokens_after_turn1 = PROMPT_TOKENS + partial_tokens
        assert result2.requests == [
            expected_request(
                variant,
                input_ids=tokens_after_turn1,
                sampling_params={"max_new_tokens": 14, "temperature": 0.7},
            )
        ]
        assert result2.sample == expected_sample(
            variant,
            response=partial_text + remaining_text,
            response_length=2 + 3,
            tokens=tokens_after_turn1 + remaining_tokens,
            rollout_log_probs=partial_log_probs + remaining_log_probs,
            prompt_tokens=len(PROMPT_TOKENS) + len(tokens_after_turn1),
            status=Sample.Status.COMPLETED,
            weight_versions=[None, None],
        )


class TestFinishReason:
    # [CLASS DOC] Tests of the mapping from the server's finish reason to the sample status.
    @pytest.mark.parametrize(
        "generation_env,expected_status",
        [
            ({"process_fn_kwargs": {"finish_reason": "stop"}}, Sample.Status.COMPLETED),
            ({"process_fn_kwargs": {"finish_reason": "length"}}, Sample.Status.TRUNCATED),
            ({"process_fn_kwargs": {"finish_reason": "abort"}}, Sample.Status.ABORTED),
        ],
        indirect=["generation_env"],
    )
    def test_finish_reason_sets_status(self, variant, generation_env, expected_status):
        # [FUNCTION DOC] Parametrized: stop gives completed, length gives truncated, abort gives
        # aborted. Asserts the normal request and the default expected sample with that status.
        result = _run_generate(variant, generation_env)
        assert result.requests == [expected_request(variant)]
        assert listify(result.sample) == [expected_sample(variant, status=expected_status)]


class TestRoutedExperts:
    # [CLASS DOC] Tests of routed experts (routing replay) handling.
    @pytest.mark.parametrize(
        "generation_env",
        [
            {
                "args_kwargs": {"use_rollout_routing_replay": True},
                "process_fn_kwargs": {"routed_experts": "placeholder"},
            }
        ],
        indirect=True,
    )
    def test_routed_experts_enabled_and_parsed(self, variant, generation_env):
        # [FUNCTION DOC] Enables rollout routing replay with 2 layers and router top-k 4. The mock
        # server returns a base64 encoded 32-bit integer ramp array shaped (total tokens minus one,
        # layers, top-k). Asserts the request has the routed experts flag on, and that the sample's
        # routed experts array has that shape and equals the sent array.
        num_layers, moe_router_topk = 2, 4
        num_tokens = len(PROMPT_TOKENS) + len(RESPONSE_TOKENS)
        routed_experts_array = np.arange((num_tokens - 1) * num_layers * moe_router_topk, dtype=np.int32).reshape(
            num_tokens - 1, num_layers, moe_router_topk
        )

        generation_env.args.num_layers = num_layers
        generation_env.args.moe_router_topk = moe_router_topk
        routed_experts_str = pybase64.b64encode(routed_experts_array.tobytes()).decode("ascii")
        generation_env.mock_server.process_fn = lambda _: ProcessResult(
            text=RESPONSE_TEXT,
            finish_reason="stop",
            meta_info=ProcessResultMetaInfo(routed_experts=routed_experts_str),
        )

        result = _run_generate(variant, generation_env)
        assert result.requests == [expected_request(variant, return_routed_experts=True)]
        sample = result.sample[0] if isinstance(result.sample, list) else result.sample
        assert sample.rollout_routed_experts is not None
        assert sample.rollout_routed_experts.shape == (num_tokens - 1, num_layers, moe_router_topk)
        np.testing.assert_array_equal(sample.rollout_routed_experts, routed_experts_array)


class TestMetaInfo:
    # [CLASS DOC] Tests that meta information returned by the server is copied into the sample.
    @pytest.mark.parametrize(
        "generation_env", [{"process_fn_kwargs": {"cached_tokens": 3, "weight_version": "v1.0"}}], indirect=True
    )
    def test_meta_info_fields_updated(self, variant, generation_env):
        # [FUNCTION DOC] The server reports 3 cached tokens and weight version "v1.0". Asserts the
        # sample carries those in its prefix cache info and a single weight version span.
        result = _run_generate(variant, generation_env)
        assert result.requests == [expected_request(variant)]
        assert listify(result.sample) == [expected_sample(variant, cached_tokens=3, weight_versions=["v1.0"])]

    @pytest.mark.parametrize(
        "generation_env",
        [
            {
                "args_kwargs": {"sglang_speculative_algorithm": "EAGLE"},
                "process_fn_kwargs": {
                    "spec_num_correct_drafts": 10,
                    "spec_num_proposed_drafts": 15,
                    "spec_verify_ct": 3,
                },
            }
        ],
        indirect=True,
    )
    def test_spec_info_updated(self, variant, generation_env):
        # [FUNCTION DOC] With the EAGLE speculative algorithm configured, the server reports 10
        # correct drafts, 15 proposed drafts and 3 verifications. Asserts the sample's speculative
        # info holds those values and a completion token count of 5.
        result = _run_generate(variant, generation_env)
        assert result.requests == [expected_request(variant)]
        assert listify(result.sample) == [
            expected_sample(
                variant,
                spec_info=Sample.SpecInfo(
                    spec_num_correct_drafts=10, spec_num_proposed_drafts=15, spec_verify_ct=3, completion_tokens=5
                ),
            )
        ]


class TestInputStatusValidation:
    # [CLASS DOC] Tests of which input sample statuses generation accepts.
    @pytest.mark.parametrize("status", [Sample.Status.PENDING, Sample.Status.ABORTED])
    def test_allowed_statuses(self, variant, generation_env, status):
        # [FUNCTION DOC] Input samples with pending or aborted status (no tokens) generate
        # normally: asserts the normal request and default expected sample.
        result = _run_generate(variant, generation_env, _make_sample(status=status))
        assert result.requests == [expected_request(variant)]
        assert listify(result.sample) == [expected_sample(variant)]

    @pytest.mark.parametrize("status", [Sample.Status.COMPLETED, Sample.Status.TRUNCATED])
    def test_rejected_statuses(self, variant, generation_env, status):
        # [FUNCTION DOC] Skipped for the multi-turn variant. Input samples with completed or
        # truncated status make generation raise an assertion error.
        if variant == "multi_turn":
            pytest.skip("not tested yet")
        with pytest.raises(AssertionError):
            _run_generate(variant, generation_env, _make_sample(status=status))


class TestPayloadStructure:
    # [CLASS DOC] Tests of the request payload sent to the server.
    def test_sampling_params_passed_through(self, variant, generation_env):
        # [FUNCTION DOC] Generates with custom sampling parameters (16 new tokens, temperature
        # 0.5, top-p 1.0) and asserts the request carries them unchanged and the sample is the
        # default expected one.
        result = _run_generate(
            variant, generation_env, sampling_params={"max_new_tokens": 16, "temperature": 0.5, "top_p": 1.0}
        )
        assert result.requests == [
            expected_request(variant, sampling_params={"max_new_tokens": 16, "temperature": 0.5, "top_p": 1.0})
        ]
        assert listify(result.sample) == [expected_sample(variant)]


class TestBoundaryConditions:
    # [CLASS DOC] Tests of limits: exhausted new-token budget and the rollout maximum context
    # length (not supported by the old variant, whose cases are skipped).
    def test_max_new_tokens_zero_returns_truncated(self, variant, generation_env):
        # [FUNCTION DOC] Skipped for the multi-turn variant. The input sample already has a
        # 10-token response (17 tokens in total) and the maximum new tokens is 10, so nothing is
        # left to generate. Asserts no request is sent and the sample is returned unchanged with
        # status truncated, no log probabilities, zero total prompt tokens and no weight version
        # records.
        if variant == "multi_turn":
            pytest.skip("not tested yet")
        existing_tokens = [1, 2, 3, 4, 5, 6, 7] + list(range(100, 110))
        sample = _make_sample(tokens=existing_tokens, response="x" * 10, response_length=10)

        result = _run_generate(variant, generation_env, sample, {"max_new_tokens": 10, "temperature": 0.7})
        assert result.requests == []
        assert result.sample == expected_sample(
            variant,
            response="x" * 10,
            response_length=10,
            tokens=existing_tokens,
            rollout_log_probs=None,
            status=Sample.Status.TRUNCATED,
            prompt_tokens=0,
            weight_versions=[],
        )

    @pytest.mark.parametrize("generation_env", [{"args_kwargs": {"rollout_max_context_len": 5}}], indirect=True)
    def test_prompt_exceeds_max_context_len_returns_truncated(self, variant, generation_env):
        # [FUNCTION DOC] Skipped for the old variant. With the maximum context length 5 (shorter
        # than the 7-token prompt), asserts no request is sent and the sample is truncated with an
        # empty response, no log probabilities, no weight version records and tokens equal to the
        # prompt tokens for the multi-turn variant (with no loss mask) or empty for the single-turn
        # variant.
        if variant == "old_sglang_rollout":
            pytest.skip("old_sglang_rollout does not support rollout_max_context_len")
        result = _run_generate(variant, generation_env)
        assert result.requests == []
        tokens = PROMPT_TOKENS if variant == "multi_turn" else []
        assert listify(result.sample) == [
            expected_sample(
                variant,
                response="",
                response_length=0,
                tokens=tokens,
                rollout_log_probs=None,
                status=Sample.Status.TRUNCATED,
                prompt_tokens=0,
                loss_mask=None if variant == "multi_turn" else _UNSET,
                weight_versions=[],
            )
        ]

    @pytest.mark.parametrize(
        "generation_env,expected_max_new_tokens",
        [
            ({"args_kwargs": {"rollout_max_context_len": 10}}, 10 - PROMPT_TOKEN_LEN),
            ({"args_kwargs": {"rollout_max_context_len": 8}}, 8 - PROMPT_TOKEN_LEN),
            ({"args_kwargs": {"rollout_max_context_len": 100}}, DEFAULT_MAX_NEW_TOKENS),
        ],
        indirect=["generation_env"],
    )
    def test_moderate_length_input_adjusts_max_new_tokens(self, variant, generation_env, expected_max_new_tokens):
        # [FUNCTION DOC] Skipped for the old variant. Parametrized by maximum context length: 10
        # and 8 cap the maximum new tokens at that length minus the prompt length; 100 leaves the
        # default of 16. Asserts one request with the expected maximum new tokens, the unchanged
        # temperature, and the default expected sample.
        if variant == "old_sglang_rollout":
            pytest.skip("old_sglang_rollout does not support rollout_max_context_len")
        result = _run_generate(variant, generation_env)
        assert len(result.requests) == 1
        assert result.requests[0]["sampling_params"]["max_new_tokens"] == expected_max_new_tokens
        assert result.requests[0]["sampling_params"]["temperature"] == SAMPLING_PARAMS["temperature"]
        assert listify(result.sample) == [expected_sample(variant)]

    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"rollout_max_context_len": PROMPT_TOKEN_LEN}}],
        indirect=True,
    )
    def test_adjusted_max_new_tokens_zero_returns_truncated(self, variant, generation_env):
        # [FUNCTION DOC] Skipped for the old variant. With the maximum context length equal to the
        # prompt length, the adjusted new-token budget is zero. Asserts no request and the same
        # truncated empty-response sample as the prompt-too-long case (prompt tokens kept only for
        # the multi-turn variant).
        if variant == "old_sglang_rollout":
            pytest.skip("old_sglang_rollout does not support rollout_max_context_len")
        result = _run_generate(variant, generation_env)
        assert result.requests == []
        tokens = PROMPT_TOKENS if variant == "multi_turn" else []
        assert listify(result.sample) == [
            expected_sample(
                variant,
                response="",
                response_length=0,
                tokens=tokens,
                rollout_log_probs=None,
                status=Sample.Status.TRUNCATED,
                prompt_tokens=0,
                loss_mask=None if variant == "multi_turn" else _UNSET,
                weight_versions=[],
            )
        ]


class TestEmptyResponse:
    # [CLASS DOC] Tests of a server response with no text.
    @pytest.mark.parametrize("generation_env", [{"process_fn_kwargs": {"response_text": ""}}], indirect=True)
    def test_empty_response(self, variant, generation_env):
        # [FUNCTION DOC] The server returns an empty response. Asserts the normal request and a
        # completed sample with an empty response, zero length, tokens equal to the prompt tokens
        # and an empty log probability list.
        result = _run_generate(variant, generation_env)
        assert result.requests == [expected_request(variant)]
        assert listify(result.sample) == [
            expected_sample(variant, response="", response_length=0, tokens=PROMPT_TOKENS, rollout_log_probs=[])
        ]


VLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


class TestMultimodal:
    # [CLASS DOC] Tests of image inputs with a vision-language model (processor loaded from the
    # Hugging Face hub or cache).
    @pytest.mark.parametrize("generation_env", [{"args_kwargs": {"model_name": VLM_MODEL_NAME}}], indirect=True)
    def test_multimodal_inputs_processed(self, variant, generation_env):
        # [FUNCTION DOC] Skipped for the multi-turn variant. Uses a 64 by 64 red image as the input
        # sample's multimodal data and computes the expected training inputs with the model's
        # processor, dropping input ids, attention mask and token type ids. Asserts the request
        # carries the prompt tokens and the image encoded for the rollout engine; that the sample's
        # multimodal training inputs exist, exclude the token type ids and have the same keys as
        # expected, with equal pixel values and image grid values; and that the whole sample equals
        # the expected sample with those multimodal fields.
        if variant == "multi_turn":
            pytest.skip("not tested yet")
        test_image = Image.new("RGB", (64, 64), color="red")
        multimodal_inputs = {"images": [test_image]}
        processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME, trust_remote_code=True)
        expected_mti = {
            k: v
            for k, v in processor(text=PROMPT, **multimodal_inputs).items()
            if k not in ["input_ids", "attention_mask", "mm_token_type_ids"]
        }

        result = _run_generate(variant, generation_env, _make_sample(multimodal_inputs=multimodal_inputs))

        assert result.requests == [
            expected_request(
                variant,
                input_ids=PROMPT_TOKENS,
                image_data=[encode_image_for_rollout_engine(test_image)],
            )
        ]
        actual_mti = result.sample.multimodal_train_inputs
        assert actual_mti is not None
        assert "mm_token_type_ids" not in actual_mti
        assert set(actual_mti.keys()) == set(expected_mti.keys())
        assert torch.all(actual_mti["pixel_values"] == expected_mti["pixel_values"])
        assert torch.all(actual_mti["image_grid_thw"] == expected_mti["image_grid_thw"])
        assert result.sample == expected_sample(
            variant,
            tokens=PROMPT_TOKENS + RESPONSE_TOKENS,
            multimodal_inputs=multimodal_inputs,
            multimodal_train_inputs=actual_mti,
        )
