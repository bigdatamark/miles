# [MODULE DOC] Tests of multi-turn generation, run for two variants of the generate function:
# the engine-level multi-turn function (talks to a mock inference server directly with token
# ids) and the agentic tool-call function (an agent talks to a session server in front of the mock
# server using chat messages). The shared generation fixtures supply the environment (arguments,
# mock server, optional session server); the mock tools module supplies canned two-turn and
# three-turn conversations with tool calls. The Qwen3 0.6B tokenizer is loaded at import time.
# Registered as a CPU CI test in the batch-b CPU suite.
# Verification works by splitting a produced sample's response tokens into consecutive chunks of
# equal loss mask value (model-generated text has mask 1 and a log probability series, tool
# response text has mask 0 and zero log probabilities), decoding each chunk and comparing against
# the expected text, then comparing the rest of the sample with a partial expected sample after
# clearing fields that vary with the mock setup. Covered: single and multi-turn with tool calls,
# conversation metadata, weight version spans, partial rollout rejection, abort, length and
# max-turn exits, maximum context length handling, three turns, routed experts replay, agent
# metadata merging, session identity forwarding, collection failures, and no-record agents.
import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import groupby

import httpx
import numpy as np
import pybase64
import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.fast.fixtures.generation_fixtures import GenerateEnv, generation_env, listify, make_sample, run_generate

from miles.rollout.session.types import SessionServerInstance
from miles.utils.chat_template_utils import TITOTokenizerType, get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.mock_sglang_server import ProcessResult, ProcessResultMetaInfo
from miles.utils.test_utils.mock_tools import SAMPLE_TOOLS, ThreeTurnStub, TwoTurnStub
from miles.utils.types import Sample

register_cpu_ci(est_time=130, suite="stage-b-cpu", labels=[])

_ = generation_env, SAMPLE_TOOLS, TwoTurnStub, ThreeTurnStub


def is_agentic_variant(variant: str) -> bool:
    # [FUNCTION DOC] HELPER: Returns true when the variant name is the agentic tool-call variant.
    return variant == "agentic_tool_call"


# ------------------------------------ fixtures and consts ----------------------------------------


MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 0.7}
TOKENIZER = load_tokenizer(MODEL_NAME, trust_remote_code=True)


@pytest.fixture(params=["multi_turn", "agentic_tool_call"])
def variant(request):
    # [FUNCTION DOC] HELPER: Fixture parametrized over the two generate variants (engine
    # multi-turn and agentic tool call); returns the current variant name.
    return request.param


@dataclass(frozen=True)
class SampleParsedChunk:
    # [CLASS DOC] HELPER: Immutable record of a run of response tokens sharing one loss mask value:
    # the decoded text, the mask value (1 for model-generated, 0 for tool/environment text), and
    # the rollout log probabilities of those tokens.
    tokens_decoded_str: str
    loss_mask_value: int
    rollout_log_probs: list[float]


@dataclass
class ExpectedSampleInfo:
    # [CLASS DOC] HELPER: Expected outcome for one output sample: the ordered list of expected
    # response chunks and a partial sample holding the expected values of all non-token fields.
    chunks: list[SampleParsedChunk]
    partial_sample: Sample


def token_len(text: str) -> int:
    # [FUNCTION DOC] HELPER: Returns the number of tokens the module tokenizer produces for the
    # text without adding special tokens.
    return len(TOKENIZER(text, add_special_tokens=False)["input_ids"])


def expected_chunk(text: str, loss_mask: int) -> SampleParsedChunk:
    # [FUNCTION DOC] HELPER: Builds the expected chunk for text and a loss mask value. For a
    # non-zero mask the log probabilities follow the mock server's pattern: for token position i
    # (from zero) the value is minus i divided by 128; for a zero mask they are all 0.0. The count
    # equals the text's token length.
    n = token_len(text)
    log_probs = [-1 / 128 * i for i in range(n)] if loss_mask else [0.0] * n
    return SampleParsedChunk(text, loss_mask, log_probs)


def parse_sample_into_chunks(sample: Sample, tokenizer) -> list[SampleParsedChunk]:
    # [FUNCTION DOC] HELPER: Splits a sample's response into chunks. The response tokens are the
    # sample tokens after the prompt part (total length minus response length). The loss mask and
    # rollout log probabilities (empty when absent) are walked in groups of consecutive equal mask
    # values; each group yields a chunk with the decoded tokens of that slice, the mask value and
    # the matching log probability slice. Returns the chunks in order (empty if no mask).
    prompt_len = len(sample.tokens) - sample.response_length
    response_tokens = sample.tokens[prompt_len:]
    loss_mask = sample.loss_mask or []
    log_probs = sample.rollout_log_probs or []

    chunks = []
    idx = 0
    for mask_val, group in groupby(loss_mask):
        group_len = len(list(group))
        sli = slice(idx, idx + group_len)
        chunks.append(
            SampleParsedChunk(
                tokens_decoded_str=tokenizer.decode(response_tokens[sli]),
                loss_mask_value=mask_val,
                rollout_log_probs=log_probs[sli],
            )
        )
        idx += group_len
    return chunks


def expected_partial_sample(
    *,
    prompt: list[dict],
    response: str,
    response_length: int,
    status: Sample.Status = Sample.Status.COMPLETED,
) -> Sample:
    # [FUNCTION DOC] HELPER: Builds the expected "partial" sample used for structure comparison:
    # the given prompt, response, response length and status (default completed), with empty
    # tokens, loss mask, log probabilities and weight versions, and default speculative-decoding
    # and prefix-cache info.
    return Sample(
        prompt=prompt,
        response=response,
        response_length=response_length,
        status=status,
        tokens=[],
        loss_mask=[],
        rollout_log_probs=[],
        weight_versions=[],
        spec_info=Sample.SpecInfo(),
        prefix_cache_info=Sample.PrefixCacheInfo(),
    )


def verify_samples(actual: Sample | list[Sample], expected: list[ExpectedSampleInfo]):
    # [FUNCTION DOC] HELPER: Normalizes the actual output to a list and asserts it has the same
    # length as the expected list. For each pair: asserts the parsed chunks equal the expected
    # chunks; then deep-copies the actual sample, clears index, rollout id, tokens, loss mask, log
    # probabilities, weight versions and prefix-cache info, removes diagnostic metadata keys that
    # vary with the mock setup (session mismatch analysis, accumulated token ids, maximum trim
    # tokens, lifecycle timing, leaf, turn arguments) and asserts equality with the expected
    # partial sample. The original sample is not mutated.
    actual = listify(actual)
    assert len(actual) == len(expected)

    for actual_item, expected_item in zip(actual, expected, strict=True):
        actual_chunks = parse_sample_into_chunks(actual_item, TOKENIZER)
        assert actual_chunks == expected_item.chunks

        actual_partial = replace(
            deepcopy(actual_item),
            index=None,
            rollout_id=None,
            tokens=[],
            loss_mask=[],
            rollout_log_probs=[],
            weight_versions=[],
            prefix_cache_info=Sample.PrefixCacheInfo(),
        )
        # Session server populates diagnostic metadata (token IDs,
        # trim config, mismatch analysis, dashboard lifecycle timing, the
        # template kwargs the last turn recorded) that varies with mock setup.
        # Strip these before comparing structure.
        for key in (
            "tito_session_mismatch",
            "accumulated_token_ids",
            "max_trim_tokens",
            "lifecycle",
            "leaf",
            "turn_args",
        ):
            actual_partial.metadata.pop(key, None)
        assert actual_partial == expected_item.partial_sample


def _run_generate(variant: str, env: GenerateEnv, sample: Sample, sampling_params: dict | None = None):
    # [FUNCTION DOC] HELPER: Runs the shared generate runner for the variant. For the agentic
    # variant, a sample without an index gets index zero first (mutating the sample), because that
    # variant requires an identity. Returns the runner's result (requests seen by the mock server
    # and the output sample or samples).
    if is_agentic_variant(variant) and sample.index is None:
        sample.index = 0
    return run_generate(env, sample, sampling_params, variant=variant)


def expected_request(
    input_ids: list[int],
    sampling_params: dict | None = None,
    *,
    return_routed_experts: bool = False,
    return_indexer_topk: bool = False,
) -> dict:
    # [FUNCTION DOC] HELPER: Builds the request mapping the engine multi-turn variant is expected
    # to send to the mock server: the prompt token ids, sampling parameters (default: the module
    # defaults), log probability return on, and the routed experts and indexer top-k return flags
    # (both default off).
    return {
        "input_ids": input_ids,
        "sampling_params": sampling_params or DEFAULT_SAMPLING_PARAMS,
        "return_logprob": True,
        "return_routed_experts": return_routed_experts,
        "return_indexer_topk": return_indexer_topk,
    }


def expected_openai_request(messages: list[dict], **extra) -> dict:
    # [FUNCTION DOC] HELPER: Builds the chat request mapping the agentic variant is expected to
    # send through the session server: the messages, model "default", the sample tools, the
    # session-injected flags (log probabilities and meta info on, no stop trimming off), replay
    # flags off, the default temperature, and a chat template option disabling clearing of
    # thinking. Extra keyword entries are merged last and may override.
    return {
        "messages": messages,
        "model": "default",
        "tools": SAMPLE_TOOLS,
        # Injected by the session route for TITO token tracking
        "logprobs": True,
        "return_meta_info": True,
        "no_stop_trim": False,
        # The R3 replay flags follow the launch flags and are always present.
        "return_routed_experts": False,
        "return_indexer_topk": False,
        # The mock agent drops request_kwargs; the session fills the temperature the
        # generator registered at creation.
        "temperature": DEFAULT_SAMPLING_PARAMS["temperature"],
        "chat_template_kwargs": {"clear_thinking": False},
        **extra,
    }


_SESSION_PRETOKENIZED_KEYS = {"input_ids"}


def _strip_pretokenized(requests: list[dict]) -> list[dict]:
    # [FUNCTION DOC] HELPER: Returns copies of the request mappings without the session-injected
    # input ids entry so they can be compared deterministically.
    """Strip session-injected pretokenized fields for deterministic comparison."""
    return [{k: v for k, v in r.items() if k not in _SESSION_PRETOKENIZED_KEYS} for r in requests]


SINGLE_TURN_PROMPT = [{"role": "user", "content": "What is 1+1?"}]
SINGLE_TURN_RESPONSE = "The answer is 2."
_SINGLE_TURN_PROMPT_TEXT = TOKENIZER.apply_chat_template(
    SINGLE_TURN_PROMPT, tokenize=False, add_generation_prompt=True, tools=SAMPLE_TOOLS
)
SINGLE_TURN_PROMPT_TOKEN_IDS = TOKENIZER(_SINGLE_TURN_PROMPT_TEXT, add_special_tokens=False)["input_ids"]
SINGLE_TURN_PROMPT_TOKEN_LEN = len(SINGLE_TURN_PROMPT_TOKEN_IDS)


# ------------------------------------ tests ----------------------------------------


class TestBasicMultiTurn:
    # [CLASS DOC] Tests of ordinary conversations: one turn without tools and two turns with a
    # tool call between them.
    def test_single_turn_no_tool_call(self, variant, generation_env):
        # [FUNCTION DOC] The mock server answers with a fixed text and a stop finish reason. Asserts
        # the request(s) seen by the server match the variant's expected form, and that the single
        # output sample has one loss-mask-1 chunk with the response text and 6 log probabilities,
        # and the expected completed partial sample with response length 6.
        generation_env.mock_server.process_fn = lambda _: ProcessResult(
            text=SINGLE_TURN_RESPONSE, finish_reason="stop"
        )

        result = _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))

        if is_agentic_variant(variant):
            assert _strip_pretokenized(result.requests) == [expected_openai_request(SINGLE_TURN_PROMPT)]
        else:
            assert result.requests == [expected_request(SINGLE_TURN_PROMPT_TOKEN_IDS)]
        verify_samples(
            result.sample,
            [
                ExpectedSampleInfo(
                    chunks=[
                        SampleParsedChunk(
                            tokens_decoded_str=SINGLE_TURN_RESPONSE,
                            loss_mask_value=1,
                            rollout_log_probs=[-1 / 128 * i for i in range(6)],
                        ),
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=SINGLE_TURN_PROMPT, response=SINGLE_TURN_RESPONSE, response_length=6
                    ),
                ),
            ],
        )

    def test_two_turns_with_tool_call(self, variant, generation_env):
        # [FUNCTION DOC] Uses the two-turn stub conversation (first turn requests tools, second
        # turn answers). Asserts two requests with the expected messages (agentic) or prompt token
        # ids (engine), and one sample whose response chunks are: first response (mask 1), tool
        # response text (mask 0), second response (mask 1), with the response length equal to the
        # token length of the concatenation.
        generation_env.mock_server.process_fn = TwoTurnStub.process_fn

        S = TwoTurnStub
        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        if is_agentic_variant(variant):
            assert _strip_pretokenized(result.requests) == [
                expected_openai_request(S.OPENAI_MESSAGES_FIRST_TURN),
                expected_openai_request(S.OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT),
            ]
        else:
            assert result.requests == [
                expected_request(S.FIRST_PROMPT_TOKEN_IDS),
                expected_request(S.SECOND_PROMPT_TOKEN_IDS),
            ]
        full_response = S.FIRST_RESPONSE + S.FIRST_TOOL_RESPONSE + S.SECOND_RESPONSE
        expected = [
            ExpectedSampleInfo(
                chunks=[
                    expected_chunk(S.FIRST_RESPONSE, 1),
                    expected_chunk(S.FIRST_TOOL_RESPONSE, 0),
                    expected_chunk(S.SECOND_RESPONSE, 1),
                ],
                partial_sample=expected_partial_sample(
                    prompt=S.PROMPT,
                    response=full_response,
                    response_length=token_len(full_response),
                ),
            ),
        ]
        verify_samples(result.sample, expected)


class TestExitConditions:
    # [CLASS DOC] Tests of how generation ends or is bounded: debug conversation recording, weight
    # version spans, partial rollout rejection, abort, length truncation, the turn limit, and
    # interplay with these across both variants (some are skipped for the agentic variant).
    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": ["--save-debug-trajectory-data", "/unused/{rollout_id}.jsonl"]}}],
        indirect=True,
    )
    def test_conversation_metadata_records_turns(self, variant, generation_env):
        # [FUNCTION DOC] Engine variant only (skipped otherwise). Enables debug trajectory saving
        # through extra command-line arguments (the path is never written to) and runs the two-turn
        # stub. Asserts the sample metadata's recorded messages have roles user, assistant, tool,
        # tool, assistant; the first assistant content is the first response; the two tool messages
        # carry the tool names for year and temperature (as a set); and the last assistant content
        # is the second response.
        if variant != "multi_turn":
            pytest.skip("conversation recording asserted once, on the engine multi-turn path")
        generation_env.mock_server.process_fn = TwoTurnStub.process_fn

        S = TwoTurnStub
        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        messages = result.sample.metadata["messages"]
        # prompt message + turn-1 assistant + its two tool results + turn-2 assistant
        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool", "assistant"]
        assert messages[1]["content"] == S.FIRST_RESPONSE
        assert {m["name"] for m in messages[2:4]} == {"get_year", "get_temperature"}
        assert messages[4]["content"] == S.SECOND_RESPONSE

    def test_weight_version_spans_from_the_engine_are_anchored_per_turn(self, variant, generation_env):
        # [FUNCTION DOC] Engine variant only (skipped otherwise). The mock server reports, for each
        # turn, one weight version span covering the whole generated text of that turn (version "7"
        # for the first response text, "8" for the second). Asserts the sample has two weight
        # version calls with versions 7 then 8; that each span's absolute length equals the
        # response's token length; that the first span starts right after the first prompt tokens;
        # that the first ends before the second starts (the tool text lies between); and that the
        # second ends at the end of the sample tokens.
        """Spans returned by the engine land as absolute token positions, one call per turn."""
        if variant != "multi_turn":
            pytest.skip("span anchoring asserted once, on the engine multi-turn path")

        S = TwoTurnStub

        def process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Mock server callback. Takes the stub's result for the prompt
            # and returns a copy whose meta info carries a single weight version span from zero to
            # the response's token count, with version "7" for the stub's first response text and
            # "8" for anything else.
            base_result = S.process_fn(prompt)
            num_tokens = len(TOKENIZER.encode(base_result.text, add_special_tokens=False))
            version = "7" if base_result.text == S.FIRST_RESPONSE else "8"
            return replace(
                base_result,
                meta_info=ProcessResultMetaInfo(weight_versions=[{"version": version, "start": 0, "end": num_tokens}]),
            )

        generation_env.mock_server.process_fn = process_fn
        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        sample = result.sample
        assert [[span.version for span in call.spans] for call in sample.weight_versions] == [["7"], ["8"]]
        first, second = sample.weight_versions[0].spans[0], sample.weight_versions[1].spans[0]
        n1 = len(TOKENIZER.encode(S.FIRST_RESPONSE, add_special_tokens=False))
        n2 = len(TOKENIZER.encode(S.SECOND_RESPONSE, add_special_tokens=False))
        assert (first.abs_end - first.abs_start, second.abs_end - second.abs_start) == (n1, n2)
        assert first.abs_start == len(S.FIRST_PROMPT_TOKEN_IDS)
        assert first.abs_end < second.abs_start
        assert second.abs_end == len(sample.tokens)

    def test_partial_rollout_not_supported(self, variant, generation_env):
        # [FUNCTION DOC] Turns on the partial rollout argument and asserts generation raises an
        # assertion error saying partial rollout is not supported.
        generation_env.args.partial_rollout = True

        with pytest.raises(AssertionError, match="Partial rollout is not supported"):
            _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))

    def test_abort_preserves_content(self, variant, generation_env):
        # [FUNCTION DOC] Engine variant only (agentic skipped because it does not handle abort).
        # The mock server returns text with an abort finish reason. Asserts the single request and
        # that the sample keeps the generated text and log probabilities (mask 1, 6 tokens) with
        # status aborted.
        if is_agentic_variant(variant):
            pytest.skip("agentic_tool_call does not handle abort finish_reason")
        generation_env.mock_server.process_fn = lambda _: ProcessResult(
            text=SINGLE_TURN_RESPONSE, finish_reason="abort"
        )

        result = _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))

        assert result.requests == [expected_request(SINGLE_TURN_PROMPT_TOKEN_IDS)]
        verify_samples(
            result.sample,
            [
                ExpectedSampleInfo(
                    chunks=[
                        SampleParsedChunk(
                            tokens_decoded_str=SINGLE_TURN_RESPONSE,
                            loss_mask_value=1,
                            rollout_log_probs=[-1 / 128 * i for i in range(6)],
                        ),
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=SINGLE_TURN_PROMPT,
                        response=SINGLE_TURN_RESPONSE,
                        response_length=6,
                        status=Sample.Status.ABORTED,
                    ),
                ),
            ],
        )

    def test_finish_reason_length_exits_and_preserves_content(self, variant, generation_env):
        # [FUNCTION DOC] The mock server answers the first turn with a length finish reason. Asserts
        # only the first request is made (no second turn) and the sample holds just the first
        # response chunk with status truncated.
        S = TwoTurnStub
        generation_env.mock_server.process_fn = lambda _: ProcessResult(text=S.FIRST_RESPONSE, finish_reason="length")

        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        if is_agentic_variant(variant):
            assert _strip_pretokenized(result.requests) == [expected_openai_request(S.OPENAI_MESSAGES_FIRST_TURN)]
        else:
            assert result.requests == [expected_request(S.FIRST_PROMPT_TOKEN_IDS)]
        verify_samples(
            result.sample,
            [
                ExpectedSampleInfo(
                    chunks=[expected_chunk(S.FIRST_RESPONSE, 1)],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=S.FIRST_RESPONSE,
                        response_length=token_len(S.FIRST_RESPONSE),
                        status=Sample.Status.TRUNCATED,
                    ),
                ),
            ],
        )

    @pytest.mark.parametrize("generation_env", [{"args_kwargs": {"generate_max_turns": 1}}], indirect=True)
    def test_max_turns_reached(self, variant, generation_env):
        # [FUNCTION DOC] Sets the maximum number of turns to one; the mock server always answers
        # with the first response and a stop reason. Asserts only one request is made. For the
        # engine variant the sample contains the first response plus the tool response text (mask 0)
        # appended after it, completed; for the agentic variant only the first response chunk.
        S = TwoTurnStub
        generation_env.mock_server.process_fn = lambda _: ProcessResult(text=S.FIRST_RESPONSE, finish_reason="stop")

        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        if is_agentic_variant(variant):
            assert _strip_pretokenized(result.requests) == [expected_openai_request(S.OPENAI_MESSAGES_FIRST_TURN)]
        else:
            assert result.requests == [expected_request(S.FIRST_PROMPT_TOKEN_IDS)]
        if variant == "multi_turn":
            expected = [
                ExpectedSampleInfo(
                    chunks=[
                        expected_chunk(S.FIRST_RESPONSE, 1),
                        expected_chunk(S.FIRST_TOOL_RESPONSE, 0),
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=S.FIRST_RESPONSE + S.FIRST_TOOL_RESPONSE,
                        response_length=token_len(S.FIRST_RESPONSE + S.FIRST_TOOL_RESPONSE),
                    ),
                ),
            ]
        else:
            expected = [
                ExpectedSampleInfo(
                    chunks=[expected_chunk(S.FIRST_RESPONSE, 1)],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=S.FIRST_RESPONSE,
                        response_length=token_len(S.FIRST_RESPONSE),
                    ),
                ),
            ]
        verify_samples(result.sample, expected)


class TestRespectMaxContextLen:
    # [CLASS DOC] Tests of the rollout maximum context length limit (engine variant only; the
    # agentic variant is skipped with a to-do note).
    @pytest.mark.parametrize(
        "generation_env", [{"args_kwargs": {"rollout_max_context_len": SINGLE_TURN_PROMPT_TOKEN_LEN}}], indirect=True
    )
    def test_prompt_exceeds_max_context_len_returns_truncated(self, variant, generation_env):
        # [FUNCTION DOC] Sets the maximum context length equal to the prompt length. Asserts no
        # request reaches the mock server and the sample is empty (no chunks, empty response) with
        # status truncated.
        if is_agentic_variant(variant):
            pytest.skip("TODO: implement")
        result = _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))
        assert result.requests == []
        expected = [
            ExpectedSampleInfo(
                chunks=[],
                partial_sample=expected_partial_sample(
                    prompt=SINGLE_TURN_PROMPT, response="", response_length=0, status=Sample.Status.TRUNCATED
                ),
            )
        ]
        verify_samples(result.sample, expected)

    @pytest.mark.parametrize(
        "generation_env",
        [
            {
                "args_kwargs": {
                    "rollout_max_context_len": len(TwoTurnStub.FIRST_PROMPT_TOKEN_IDS)
                    + token_len(TwoTurnStub.FIRST_RESPONSE)
                    + token_len(TwoTurnStub.FIRST_TOOL_RESPONSE)
                }
            }
        ],
        indirect=True,
    )
    def test_second_turn_exceeds_max_context_len_returns_truncated(self, variant, generation_env):
        # [FUNCTION DOC] The maximum context length equals the first prompt plus first response plus
        # tool response lengths, leaving no room for a second turn. Asserts only the first request
        # is sent and the sample holds the first response and tool response chunks with status
        # truncated.
        if is_agentic_variant(variant):
            pytest.skip("TODO: implement")
        S = TwoTurnStub
        generation_env.mock_server.process_fn = S.process_fn

        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        assert result.requests == [expected_request(S.FIRST_PROMPT_TOKEN_IDS)]
        partial_response = S.FIRST_RESPONSE + S.FIRST_TOOL_RESPONSE
        expected = [
            ExpectedSampleInfo(
                chunks=[
                    expected_chunk(S.FIRST_RESPONSE, 1),
                    expected_chunk(S.FIRST_TOOL_RESPONSE, 0),
                ],
                partial_sample=expected_partial_sample(
                    prompt=S.PROMPT,
                    response=partial_response,
                    response_length=token_len(partial_response),
                    status=Sample.Status.TRUNCATED,
                ),
            ),
        ]
        verify_samples(result.sample, expected)

    @pytest.mark.parametrize(
        "generation_env,expected_max_new_tokens",
        [
            (
                {"args_kwargs": {"rollout_max_context_len": len(TwoTurnStub.SECOND_PROMPT_TOKEN_IDS) + 10}},
                10,
            ),
            (
                {"args_kwargs": {"rollout_max_context_len": len(TwoTurnStub.SECOND_PROMPT_TOKEN_IDS) + 100}},
                64,
            ),
        ],
        indirect=["generation_env"],
    )
    def test_second_turn_adjusts_max_new_tokens(self, variant, generation_env, expected_max_new_tokens):
        # [FUNCTION DOC] Parametrized by the maximum context length: second prompt length plus 10
        # expects the second request's maximum new tokens to be capped at 10; plus 100 expects the
        # default of 64 to be kept. Asserts at least two requests were made, the second request's
        # maximum new tokens equals the expectation, and its temperature is unchanged.
        if is_agentic_variant(variant):
            pytest.skip("TODO: implement")
        S = TwoTurnStub
        generation_env.mock_server.process_fn = S.process_fn

        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        assert len(result.requests) >= 2
        assert result.requests[1]["sampling_params"]["max_new_tokens"] == expected_max_new_tokens
        assert result.requests[1]["sampling_params"]["temperature"] == DEFAULT_SAMPLING_PARAMS["temperature"]


class TestThreeTurn:
    """Need to test 3-turn case besides 2-turn, because e.g. merge_samples may behave differently."""

    def test_three_turns_with_sequential_tool_calls(self, variant, generation_env):
        # [FUNCTION DOC] Uses the three-turn stub. Asserts three requests with the expected
        # messages or prompt token ids, and one sample whose chunks alternate model text (mask 1)
        # and tool text (mask 0): first response, first tool, second response, second tool, third
        # response, with response length equal to the token length of the concatenation.
        generation_env.mock_server.process_fn = ThreeTurnStub.process_fn

        S = ThreeTurnStub
        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        if is_agentic_variant(variant):
            assert _strip_pretokenized(result.requests) == [
                expected_openai_request(S.OPENAI_MESSAGES_FIRST_TURN),
                expected_openai_request(S.OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT),
                expected_openai_request(S.OPENAI_MESSAGES_THIRD_TURN_FROM_CLIENT),
            ]
        else:
            assert result.requests == [
                expected_request(S.FIRST_PROMPT_TOKEN_IDS),
                expected_request(S.SECOND_PROMPT_TOKEN_IDS),
                expected_request(S.THIRD_PROMPT_TOKEN_IDS),
            ]
        full_response = (
            S.FIRST_RESPONSE + S.FIRST_TOOL_RESPONSE + S.SECOND_RESPONSE + S.SECOND_TOOL_RESPONSE + S.THIRD_RESPONSE
        )
        expected = [
            ExpectedSampleInfo(
                chunks=[
                    expected_chunk(S.FIRST_RESPONSE, 1),
                    expected_chunk(S.FIRST_TOOL_RESPONSE, 0),
                    expected_chunk(S.SECOND_RESPONSE, 1),
                    expected_chunk(S.SECOND_TOOL_RESPONSE, 0),
                    expected_chunk(S.THIRD_RESPONSE, 1),
                ],
                partial_sample=expected_partial_sample(
                    prompt=S.PROMPT,
                    response=full_response,
                    response_length=token_len(full_response),
                ),
            ),
        ]
        verify_samples(result.sample, expected)


class TestRoutedExpertsMultiTurn:
    # [CLASS DOC] Tests of routing replay (routed experts) in multi-turn generation.
    @pytest.mark.parametrize(
        "generation_env",
        [
            {
                "args_kwargs": {
                    "use_rollout_routing_replay": True,
                    # Must be in args BEFORE the session server starts: the R3
                    # decode now runs inside the worker during sample assembly.
                    "num_layers": 2,
                    "moe_router_topk": 4,
                }
            }
        ],
        indirect=True,
    )
    def test_two_turns_routed_experts(self, variant, generation_env):
        # [FUNCTION DOC] Enables rollout routing replay with 2 layers and top-k 4 (set before the
        # session server starts since decoding happens in its worker). Computes the first and
        # second prompt token ids (for the agentic variant via the Qwen3 token-in-token-out
        # tokenizer applying the chat template with the sample tools; for the engine variant from
        # the stub). The mock server returns, per turn, base64 encoded routed experts: an integer
        # ramp array shaped (prompt+response tokens minus one) by layers by top-k. Asserts the
        # request shapes (agentic: messages equal, logprobs/meta info on, no stop trim off, routed
        # experts return on, input ids present; engine: requests with routed experts return on),
        # then that the last sample's tail of log probabilities matches the second response's
        # pattern, its routed experts array has the second turn's shape and values, and its token
        # count minus one equals the routed experts row count.
        S = TwoTurnStub
        num_layers, moe_router_topk = generation_env.args.num_layers, generation_env.args.moe_router_topk
        if is_agentic_variant(variant):
            tito = get_tito_tokenizer(
                TOKENIZER,
                # Note: tokenizer_type needs to match MODEL_NAME
                tokenizer_type=TITOTokenizerType.QWEN3,
            )
            first_prompt_token_ids = tito.apply_chat_template(
                S.OPENAI_MESSAGES_FIRST_TURN,
                add_generation_prompt=True,
                tokenize=True,
                template_args=tito.default_template_args(SAMPLE_TOOLS),
            )
            second_prompt_token_ids = tito.apply_chat_template(
                S.OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT,
                add_generation_prompt=True,
                tokenize=True,
                template_args=tito.default_template_args(SAMPLE_TOOLS),
            )
        else:
            first_prompt_token_ids = S.FIRST_PROMPT_TOKEN_IDS
            second_prompt_token_ids = S.SECOND_PROMPT_TOKEN_IDS

        def make_routed_experts(prompt_token_ids, response_text):
            # [FUNCTION DOC] HELPER: Builds a 32-bit integer array of consecutive values with shape
            # (prompt tokens plus response tokens minus one, number of layers, router top-k).
            total_tokens = len(prompt_token_ids) + token_len(response_text)
            routed_experts_len = total_tokens - 1
            return np.arange(routed_experts_len * num_layers * moe_router_topk, dtype=np.int32).reshape(
                routed_experts_len, num_layers, moe_router_topk
            )

        first_routed_experts = make_routed_experts(first_prompt_token_ids, S.FIRST_RESPONSE)
        second_routed_experts = make_routed_experts(second_prompt_token_ids, S.SECOND_RESPONSE)

        def process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Mock server callback. Picks the first response and routed
            # experts for the first prompt, the second for the second prompt, and raises a value
            # error for any other prompt. Returns a result with a stop finish reason and the
            # routed experts bytes base64 encoded as ASCII text in the meta info.
            if prompt == S.FIRST_PROMPT:
                text, routed_experts = S.FIRST_RESPONSE, first_routed_experts
            elif prompt == S.SECOND_PROMPT:
                text, routed_experts = S.SECOND_RESPONSE, second_routed_experts
            else:
                raise ValueError(f"Unexpected prompt: {prompt}")
            return ProcessResult(
                text=text,
                finish_reason="stop",
                meta_info=ProcessResultMetaInfo(
                    routed_experts=pybase64.b64encode(routed_experts.tobytes()).decode("ascii")
                ),
            )

        generation_env.mock_server.process_fn = process_fn
        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT), DEFAULT_SAMPLING_PARAMS)

        if is_agentic_variant(variant):
            assert len(result.requests) == 2
            assert result.requests[0]["messages"] == S.OPENAI_MESSAGES_FIRST_TURN
            assert result.requests[1]["messages"] == S.OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT
            for req in result.requests:
                assert req["logprobs"] is True
                assert req["return_meta_info"] is True
                assert req["no_stop_trim"] is False
                assert req["return_routed_experts"] is True
                assert "input_ids" in req
        else:
            assert result.requests == [
                expected_request(S.FIRST_PROMPT_TOKEN_IDS, return_routed_experts=True),
                expected_request(S.SECOND_PROMPT_TOKEN_IDS, return_routed_experts=True),
            ]

        sample = result.sample[-1] if isinstance(result.sample, list) else result.sample
        expected_second_turn_log_probs = [-1 / 128 * i for i in range(token_len(S.SECOND_RESPONSE))]
        assert sample.rollout_log_probs is not None
        assert sample.rollout_log_probs[-len(expected_second_turn_log_probs) :] == expected_second_turn_log_probs
        assert sample.rollout_routed_experts is not None
        assert sample.rollout_routed_experts.shape == second_routed_experts.shape
        np.testing.assert_array_equal(sample.rollout_routed_experts, second_routed_experts)
        assert len(sample.tokens) - 1 == second_routed_experts.shape[0]


_AGENTIC_VARIANTS = ["agentic_tool_call"]
_AGENT_METADATA = {"reward": 1.0, "exit_status": "Submitted", "eval_report": {"passed": True}}


class TestAgentMetadata:
    # [CLASS DOC] Tests specific to the agentic variant: merging of the agent function's returned
    # metadata into the output samples, session mismatch diagnostics, and session identity
    # forwarding. Overrides the variant fixture to run only the agentic variant.
    """Tests specific to agentic_tool_call: agent function returning dict | None → metadata merge."""

    @pytest.fixture(params=_AGENTIC_VARIANTS)
    def variant(self, request):
        # [FUNCTION DOC] HELPER: Fixture overriding the module-level one; parametrized over the
        # agentic variants only and returns the current name.
        return request.param

    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"agentic_return_metadata": _AGENT_METADATA}}],
        indirect=True,
    )
    def test_agent_metadata_merged_into_samples(self, variant, generation_env):
        # [FUNCTION DOC] Configures the mock agent to return a fixed metadata mapping (reward,
        # exit status, evaluation report) and runs the two-turn stub. Asserts every output sample's
        # metadata contains each key with the same value.
        generation_env.mock_server.process_fn = TwoTurnStub.process_fn

        result = _run_generate(variant, generation_env, make_sample(prompt=TwoTurnStub.PROMPT))

        samples = listify(result.sample)
        for s in samples:
            for key, value in _AGENT_METADATA.items():
                assert key in s.metadata, f"metadata should contain key '{key}'"
                assert s.metadata[key] == value, f"metadata['{key}'] should be {value}, got {s.metadata[key]}"

    def test_tito_session_mismatch_present_in_metadata(self, variant, generation_env):
        # [FUNCTION DOC] Runs the two-turn stub and asserts the last sample's metadata has the
        # session mismatch entry, a list, in which every item has exactly the keys type, segment
        # index, expected text, actual text and detail (vacuously true if the list is empty).
        generation_env.mock_server.process_fn = TwoTurnStub.process_fn

        result = _run_generate(variant, generation_env, make_sample(prompt=TwoTurnStub.PROMPT))

        sample = result.sample if not isinstance(result.sample, list) else result.sample[-1]
        assert "tito_session_mismatch" in sample.metadata, "tito_session_mismatch should be present in sample metadata"
        mismatches = sample.metadata["tito_session_mismatch"]
        assert isinstance(mismatches, list)
        for m in mismatches:
            assert {"type", "segment_index", "expected_text", "actual_text", "detail"} == set(m.keys())

    def test_agent_returns_none_metadata_unchanged(self, variant, generation_env):
        # [FUNCTION DOC] The input sample has metadata with an instance id and the agent returns no
        # metadata. Asserts each output sample keeps the instance id and gains no reward key.
        generation_env.mock_server.process_fn = TwoTurnStub.process_fn
        sample = make_sample(prompt=TwoTurnStub.PROMPT)
        sample.metadata = {"instance_id": "test-123"}

        result = _run_generate(variant, generation_env, sample)

        samples = listify(result.sample)
        for s in samples:
            assert s.metadata.get("instance_id") == "test-123"
            assert "reward" not in s.metadata

    def test_session_server_identity_forwarded_to_agent_metadata(self, variant, generation_env):
        # [FUNCTION DOC] Temporarily sets the mock tools module's global agent return metadata
        # hook (reset to none in a finally block) to a function that echoes the session server id
        # and instance id keys from the metadata it receives. Runs the two-turn stub and asserts
        # every output sample's metadata has the server id equal to the local address with the
        # session server's port, and the instance id equal to the run uuid followed by "-0".
        from miles.utils.test_utils import mock_tools

        generation_env.mock_server.process_fn = TwoTurnStub.process_fn

        _SESSION_KEYS = ("session_server_id", "session_server_instance_id")

        def _echo_session(metadata=None):
            # [FUNCTION DOC] HELPER: Returns a mapping with only the session server id and
            # instance id entries that are present in the given metadata (none treated as empty).
            metadata = metadata or {}
            return {k: metadata[k] for k in _SESSION_KEYS if k in metadata}

        mock_tools.AGENTIC_RETURN_METADATA = _echo_session
        try:
            result = _run_generate(variant, generation_env, make_sample(prompt=TwoTurnStub.PROMPT))
        finally:
            mock_tools.AGENTIC_RETURN_METADATA = None

        samples = listify(result.sample)
        (session_server_instance,) = generation_env.args.session_server_instances
        session_server_port = int(session_server_instance.addr.rsplit(":", 1)[1])
        expected_session_server_id = f"127.0.0.1:{session_server_port}"
        for s in samples:
            assert s.metadata["session_server_id"] == expected_session_server_id
            assert s.metadata["session_server_instance_id"] == f"{generation_env.args.run_uuid}-0"


class TestAgentCollectionFailure:
    # [CLASS DOC] Tests of failures while the agentic variant collects samples from the session
    # server. Overrides the variant fixture to run only the agentic variant.
    @pytest.fixture(params=_AGENTIC_VARIANTS)
    def variant(self, request):
        # [FUNCTION DOC] HELPER: Fixture overriding the module-level one; parametrized over the
        # agentic variants only and returns the current name.
        return request.param

    @pytest.mark.parametrize(
        "collect_error",
        [
            pytest.param(asyncio.TimeoutError(), id="timeout"),
            # A broken connection to the session server is one sample's problem: in-place
            # weight updates pause generation under in-flight requests, so letting it
            # propagate takes the whole run down over a routine event.
            pytest.param(httpx.ReadError("connection closed"), id="transport"),
        ],
    )
    def test_collect_transient_failure_aborts_sample_but_other_errors_propagate(
        self, variant, generation_env, monkeypatch, caplog, collect_error
    ):
        # [FUNCTION DOC] Parametrized over a timeout and a transport read error. Replaces the
        # tracer's sample collection with a function raising the current error, runs generation
        # with warning-level log capture, and asserts the single output sample is aborted, the
        # input sample stays pending (not mutated), and a "Failed collecting samples" message was
        # logged. Then rebinds the error variable to a runtime error (which the replacement
        # function reads at call time) and asserts a second run propagates it as a runtime error.
        async def fail_collect(_tracer, _input_sample, *, max_seq_len, agent_metadata=None):
            # [FUNCTION DOC] HELPER: Replacement for the tracer's collect method; ignores its
            # arguments and raises whatever the enclosing test's error variable currently holds.
            raise collect_error

        monkeypatch.setattr(
            "miles.rollout.generate_utils.openai_endpoint_utils.OpenAIEndpointTracer.collect_samples",
            fail_collect,
        )
        input_sample = make_sample(prompt=TwoTurnStub.PROMPT)
        with caplog.at_level(logging.WARNING):
            result = _run_generate(variant, generation_env, input_sample)

        [sample] = listify(result.sample)
        assert sample.status == Sample.Status.ABORTED
        assert input_sample.status == Sample.Status.PENDING
        assert "Failed collecting samples" in caplog.text

        collect_error = RuntimeError("assembly failed")
        with pytest.raises(RuntimeError, match="assembly failed"):
            _run_generate(variant, generation_env, make_sample(prompt=TwoTurnStub.PROMPT))


class TestAgentNoRecords:
    # [CLASS DOC] Tests the agentic variant when the agent makes no model calls.
    """When agent makes no model calls, generate should return an ABORTED sample."""

    @pytest.mark.parametrize("agentic_variant", _AGENTIC_VARIANTS)
    def test_no_records_returns_aborted(self, agentic_variant):
        # [FUNCTION DOC] Builds its own environment instead of the shared fixture: clears the
        # singleton registry, starts a mock server, picks a free port from 31000, creates arguments
        # whose agent function is a no-op agent, starts a session server in front of the mock
        # server, registers it as the only session server instance, and runs generation. Clears the
        # singleton registry again afterwards (not in a finally block, so a failure inside the
        # servers leaves it set). Asserts the one output sample is aborted.
        from tests.fast.fixtures.generation_fixtures import (
            GenerateEnv,
            extra_argv_for_variant,
            make_args,
            with_session_server,
        )
        from miles.utils.http_utils import find_available_port
        from miles.utils.misc import SingletonMeta
        from miles.utils.test_utils.mock_sglang_server import with_mock_server

        SingletonMeta.clear_all_instances()

        with with_mock_server(
            model_name=MODEL_NAME,
            process_fn=lambda _: ProcessResult(text="unused", finish_reason="stop"),
        ) as mock_server:
            session_port = find_available_port(31000)
            noop_argv = extra_argv_for_variant(
                agentic_variant,
                custom_agent_function_path="miles.utils.test_utils.mock_tools.run_agentic_noop",
            )
            args = make_args(
                variant=agentic_variant,
                router_port=session_port,
                extra_argv=noop_argv,
            )
            with with_session_server(mock_server.url, args, port=session_port):
                args.session_server_instances = [SessionServerInstance(addr=f"127.0.0.1:{session_port}")]
                env = GenerateEnv(args=args, mock_server=mock_server)
                result = _run_generate(agentic_variant, env, make_sample(prompt=TwoTurnStub.PROMPT))

        SingletonMeta.clear_all_instances()

        [sample] = listify(result.sample)
        assert sample.status == Sample.Status.ABORTED
