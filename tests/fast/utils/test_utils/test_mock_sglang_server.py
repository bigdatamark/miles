# [MODULE DOC] Tests for the mock sglang inference server used by rollout tests: the metadata
# result record, the default process function (answers simple addition questions in a boxed
# format), the concurrency counter, the basic server lifecycle (request log, latency, maximum
# concurrency, health and abort endpoints), the generate endpoint (token log-probs are synthetic:
# minus index over 128 per token), the chat completions endpoint (tool call parsing from model
# text, weight version spans in metadata), multi-turn tool call stubs, and host resolution. Most
# tests talk to the server over real HTTP with the requests library. A module scoped fixture
# provides one shared server. Expected-value helpers compute log-probs and prompt ids with the
# server's tokenizer.
import asyncio
import concurrent.futures
import time

import pytest
import requests

from miles.utils.http_utils import MILES_HOST_IP_ENV
from miles.utils.misc import NodeProbeMixin
from miles.utils.test_utils.mock_sglang_server import (
    Counter,
    MockSGLangServer,
    ProcessResult,
    ProcessResultMetaInfo,
    default_process_fn,
    with_mock_server,
)
from miles.utils.test_utils.mock_tools import SAMPLE_TOOLS, TwoTurnStub


def expected_logprobs(tokenizer, text: str) -> list[dict]:
    # [FUNCTION DOC] HELPER: Encodes the text without special tokens and returns, per token id, a
    # mapping with the token string, the token id and the synthetic log-prob equal to minus the
    # position divided by 128.
    output_ids = tokenizer.encode(text, add_special_tokens=False)
    return [
        {"token": tokenizer.convert_ids_to_tokens(tid), "token_id": tid, "logprob": -i / 128}
        for i, tid in enumerate(output_ids)
    ]


def expected_prompt_token_ids(tokenizer, messages: list[dict], tools: list[dict] | None) -> list[int]:
    # [FUNCTION DOC] HELPER: Renders the messages (and optional tools) with the tokenizer's chat
    # template as text with a generation prompt appended, then encodes it without special tokens
    # and returns the ids.
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)
    return tokenizer.encode(prompt_str, add_special_tokens=False)


@pytest.fixture(scope="module")
def mock_server():
    # [FUNCTION DOC] HELPER: Module scoped fixture that starts one default mock server (as a
    # context manager) and yields it to all tests of the module; stops it at module end.
    with with_mock_server() as server:
        yield server


class TestProcessResultMetaInfo:
    # [CLASS DOC] Tests of the optional metadata record's conversion to a dictionary, which
    # includes only the fields that were set.
    def test_to_dict_empty(self):
        # [FUNCTION DOC] An empty record converts to an empty dictionary.
        assert ProcessResultMetaInfo().to_dict() == {}

    def test_to_dict_single_field(self):
        # [FUNCTION DOC] A record with only a weight version converts to a one-entry dictionary.
        assert ProcessResultMetaInfo(weight_version="v1").to_dict() == {"weight_version": "v1"}

    def test_to_dict_partial_fields(self):
        # [FUNCTION DOC] A record with a weight version and a correct drafts count converts to a
        # two-entry dictionary.
        assert ProcessResultMetaInfo(weight_version="v1", spec_num_correct_drafts=10).to_dict() == {
            "weight_version": "v1",
            "spec_num_correct_drafts": 10,
        }

    def test_to_dict_all_fields(self):
        # [FUNCTION DOC] A record with weight version, routed experts, correct and proposed
        # drafts and verify count converts to a five-entry dictionary. (The weight versions span
        # list field is not covered here.)
        assert ProcessResultMetaInfo(
            weight_version="v1",
            routed_experts="abc",
            spec_num_correct_drafts=10,
            spec_num_proposed_drafts=15,
            spec_verify_ct=3,
        ).to_dict() == {
            "weight_version": "v1",
            "routed_experts": "abc",
            "spec_num_correct_drafts": 10,
            "spec_num_proposed_drafts": 15,
            "spec_verify_ct": 3,
        }


class TestDefaultProcessFn:
    # [CLASS DOC] Tests of the default text-in text-out function of the mock server.
    def test_math_question(self):
        # [FUNCTION DOC] Prompts asking for the sum of 1 and 5, and of 1 and 10, give results
        # with the sum in a boxed latex form and stop finish reason.
        assert default_process_fn("What is 1+5?") == ProcessResult(text="\\boxed{6}", finish_reason="stop")
        assert default_process_fn("What is 1+10?") == ProcessResult(text="\\boxed{11}", finish_reason="stop")

    def test_unknown_question(self):
        # [FUNCTION DOC] A prompt that is not a sum question gets the fixed text saying it does
        # not understand, with stop finish reason.
        assert default_process_fn("Hello") == ProcessResult(text="I don't understand.", finish_reason="stop")


class TestCounter:
    # [CLASS DOC] Tests of the concurrency counter, which tracks the maximum number of
    # simultaneously active tracked sections.
    def test_tracks_max(self):
        # [FUNCTION DOC] Nested tracking sections raise the max to 1 then 2; a reset brings it
        # back to 0.
        counter = Counter()
        assert counter.max_value == 0

        with counter.track():
            assert counter.max_value == 1
            with counter.track():
                assert counter.max_value == 2

        counter.reset()
        assert counter.max_value == 0

    def test_concurrent_tasks(self):
        # [FUNCTION DOC] Runs three asynchronous tasks concurrently, each tracked for 0.1
        # seconds, in a fresh event loop; the counter's max is 3.
        counter = Counter()

        async def task():
            # [FUNCTION DOC] HELPER: Holds a tracking section open for 0.1 seconds.
            with counter.track():
                await asyncio.sleep(0.1)

        async def run_all():
            # [FUNCTION DOC] HELPER: Runs three copies of the task concurrently and waits for all.
            await asyncio.gather(task(), task(), task())

        asyncio.run(run_all())
        assert counter.max_value == 3


class TestMockServerBasic:
    # [CLASS DOC] Tests of server basics using the shared server or short-lived ones.
    def test_start_stop(self, mock_server):
        # [FUNCTION DOC] The shared server has a positive port and its url is http plus host and
        # port.
        assert mock_server.port > 0
        assert f"http://{mock_server.host}:{mock_server.port}" == mock_server.url

    def test_request_log_and_reset_stats(self, mock_server):
        # [FUNCTION DOC] Resets stats (log empty), posts one generate request and asserts the log
        # holds exactly that payload; resetting again empties the log and zeroes the maximum
        # concurrency.
        mock_server.reset_stats()
        assert len(mock_server.request_log) == 0

        payload = {"input_ids": [1, 2, 3], "sampling_params": {"temperature": 0.5}, "return_logprob": True}
        requests.post(f"{mock_server.url}/generate", json=payload, timeout=5.0)
        assert len(mock_server.request_log) == 1
        assert mock_server.request_log[0] == payload

        mock_server.reset_stats()
        assert len(mock_server.request_log) == 0
        assert mock_server.max_concurrent == 0

    @pytest.mark.parametrize("latency,min_time,max_time", [(0.0, 0.0, 0.3), (0.5, 0.5, 1.0)])
    def test_latency(self, latency, min_time, max_time):
        # [FUNCTION DOC] Parametrized over latency 0.0 (elapsed under 0.3 seconds) and 0.5
        # (elapsed between 0.5 and 1.0 seconds). Starts a server with that latency, times one
        # generate request and asserts the elapsed wall time lies in the range. Timing based, so
        # it may be sensitive to a loaded machine.
        with with_mock_server(latency=latency) as server:
            start = time.time()
            requests.post(f"{server.url}/generate", json={"input_ids": [1], "sampling_params": {}}, timeout=5.0)
            elapsed = time.time() - start
            assert min_time <= elapsed < max_time

    def test_max_concurrent_with_latency(self):
        # [FUNCTION DOC] With latency 0.1, sends three generate requests from a pool of three
        # threads and waits for all; the server's maximum concurrency is 3.
        with with_mock_server(latency=0.1) as server:

            def send_request():
                # [FUNCTION DOC] HELPER: Posts one minimal generate request with a 5 second
                # timeout.
                requests.post(f"{server.url}/generate", json={"input_ids": [1], "sampling_params": {}}, timeout=5.0)

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(send_request) for _ in range(3)]
                concurrent.futures.wait(futures)

            assert server.max_concurrent == 3

    def test_health_endpoint(self, mock_server):
        # [FUNCTION DOC] A GET on the health path returns 200 and an ok status body.
        response = requests.get(f"{mock_server.url}/health", timeout=5.0)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_abort_request_endpoint(self, mock_server):
        # [FUNCTION DOC] A POST with an empty body on the abort path returns 200 and an ok status
        # body.
        response = requests.post(f"{mock_server.url}/abort_request", json={}, timeout=5.0)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestGenerateEndpoint:
    # [CLASS DOC] Tests of the token-in token-out generate endpoint.
    def test_basic(self, mock_server):
        # [FUNCTION DOC] Encodes a sum question prompt (asserting the expected seven token ids),
        # posts it with sampling parameters and the log-prob flag, and asserts status 200 and the
        # exact response: boxed answer text 8, stop finish reason, prompt token count, zero cached
        # tokens, five completion tokens, and five (log-prob, token id) pairs with log-probs 0 to
        # minus 4/128 in steps of 1/128.
        prompt = "What is 1+7?"
        input_ids = mock_server.tokenizer.encode(prompt, add_special_tokens=False)
        assert input_ids == [3838, 374, 220, 16, 10, 22, 30]

        response = requests.post(
            f"{mock_server.url}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0.7, "max_new_tokens": 10},
                "return_logprob": True,
            },
            timeout=5.0,
        )
        assert response.status_code == 200
        assert response.json() == {
            "text": "\\boxed{8}",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": len(input_ids),
                "cached_tokens": 0,
                "completion_tokens": 5,
                "output_token_logprobs": [
                    [-0.0, 59],
                    [-0.0078125, 79075],
                    [-0.015625, 90],
                    [-0.0234375, 23],
                    [-0.03125, 92],
                ],
            },
        }

    def test_with_meta_info(self):
        # [FUNCTION DOC] A server whose process function returns text ok with 5 cached tokens and
        # full metadata (weight version, routed experts, speculative draft counts). Asserts the
        # generate response meta information includes the standard fields plus all the metadata
        # fields verbatim, with a single completion token.
        def process_fn(_: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Ignores the prompt and returns the fixed result with
            # metadata.
            return ProcessResult(
                text="ok",
                finish_reason="stop",
                cached_tokens=5,
                meta_info=ProcessResultMetaInfo(
                    weight_version="v2.0",
                    routed_experts="encoded_data",
                    spec_num_correct_drafts=10,
                    spec_num_proposed_drafts=15,
                    spec_verify_ct=3,
                ),
            )

        with with_mock_server(process_fn=process_fn) as server:
            response = requests.post(
                f"{server.url}/generate",
                json={"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True},
                timeout=5.0,
            )

            assert response.json() == {
                "text": "ok",
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "prompt_tokens": 3,
                    "cached_tokens": 5,
                    "completion_tokens": 1,
                    "output_token_logprobs": [[-0.0, 562]],
                    "weight_version": "v2.0",
                    "routed_experts": "encoded_data",
                    "spec_num_correct_drafts": 10,
                    "spec_num_proposed_drafts": 15,
                    "spec_verify_ct": 3,
                },
            }

    def test_finish_reason_length(self):
        # [FUNCTION DOC] A process function finishing with the length reason makes the response's
        # finish reason type length with a length field equal to the completion token count.
        def process_fn(_: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Ignores the prompt and returns a truncated output with the
            # length finish reason.
            return ProcessResult(text="truncated output", finish_reason="length")

        with with_mock_server(process_fn=process_fn) as server:
            response = requests.post(
                f"{server.url}/generate",
                json={"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True},
                timeout=5.0,
            )
            data = response.json()

        finish_reason = data["meta_info"]["finish_reason"]
        assert finish_reason["type"] == "length"
        assert finish_reason["length"] == data["meta_info"]["completion_tokens"]


class TestChatCompletionsEndpoint:
    # [CLASS DOC] Tests of the OpenAI-style chat completions endpoint of the mock.
    def test_basic(self, mock_server):
        # [FUNCTION DOC] Posts a one-message sum question requesting prompt token ids. Asserts
        # status 200, an id starting with the chat completion prefix, an integer created time, a
        # meta information entry on the choice (then removes it), and that the whole body equals
        # the expected structure: chat completion object, fixed mock model name, one choice with
        # the boxed answer message (no tool calls), synthetic log-prob content, the templated
        # prompt token ids, stop finish reason, and usage counts that add up.
        messages = [{"role": "user", "content": "What is 1+5?"}]
        response = requests.post(
            f"{mock_server.url}/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": messages,
                "return_prompt_token_ids": True,
            },
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()

        assert data["id"].startswith("chatcmpl-")
        assert isinstance(data["created"], int)
        choice = data["choices"][0]
        assert "meta_info" in choice
        choice.pop("meta_info")
        prompt_token_ids = expected_prompt_token_ids(mock_server.tokenizer, messages, None)
        completion_logprobs = expected_logprobs(mock_server.tokenizer, "\\boxed{6}")
        assert data == {
            "id": data["id"],
            "object": "chat.completion",
            "created": data["created"],
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "\\boxed{6}", "tool_calls": None},
                    "logprobs": {"content": completion_logprobs},
                    "prompt_token_ids": prompt_token_ids,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(completion_logprobs),
                "total_tokens": len(prompt_token_ids) + len(completion_logprobs),
            },
        }

    def test_chat_completions_forwards_weight_version_spans_in_meta_info(self):
        # [FUNCTION DOC] A process function returning metadata with two weight version spans.
        # Asserts the choice's meta information carries the spans verbatim, has no scalar weight
        # version key, and its completion token count equals the number of output log-prob
        # entries.
        """Per-token weight version spans set on the process result reach the chat choice's meta_info verbatim."""
        spans = [{"version": "v1", "start": 0, "end": 1}, {"version": "v2", "start": 1, "end": 3}]

        def process_fn(_: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Ignores the prompt and returns a three-word text with the
            # weight version spans metadata.
            return ProcessResult(text="one two three", meta_info=ProcessResultMetaInfo(weight_versions=spans))

        with with_mock_server(process_fn=process_fn) as server:
            response = requests.post(
                f"{server.url}/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "count"}]},
                timeout=5.0,
            )
            meta_info = response.json()["choices"][0]["meta_info"]

        assert meta_info["weight_versions"] == spans
        assert "weight_version" not in meta_info
        assert meta_info["completion_tokens"] == len(meta_info["output_token_logprobs"])

    def test_with_tool_calls(self):
        # [FUNCTION DOC] The model text contains a leading sentence and one tool call block for a
        # get-year tool with empty arguments. With tools supplied, asserts the choice (minus
        # meta information) has the sentence as content, one tool call with id call00000, index 0,
        # function type and the arguments as the string for an empty object, synthetic log-probs
        # over the whole text, templated prompt ids including the tools, and the tool calls finish
        # reason.
        tool_call_response = 'Let me check for you.\n<tool_call>\n{"name": "get_year", "arguments": {}}\n</tool_call>'

        def process_fn(_: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Ignores the prompt and returns the tool call text with stop
            # finish reason.
            return ProcessResult(text=tool_call_response, finish_reason="stop")

        with with_mock_server(process_fn=process_fn) as server:
            response = requests.post(
                f"{server.url}/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "What year is it?"}],
                    "tools": SAMPLE_TOOLS,
                    "return_prompt_token_ids": True,
                },
                timeout=5.0,
            )
            data = response.json()

            choice = data["choices"][0]
            choice.pop("meta_info", None)
            assert choice == {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check for you.",
                    "tool_calls": [
                        {
                            "id": "call00000",
                            "index": 0,
                            "type": "function",
                            "function": {"name": "get_year", "arguments": "{}"},
                        }
                    ],
                },
                "logprobs": {"content": expected_logprobs(server.tokenizer, tool_call_response)},
                "prompt_token_ids": expected_prompt_token_ids(
                    server.tokenizer,
                    [{"role": "user", "content": "What year is it?"}],
                    SAMPLE_TOOLS,
                ),
                "finish_reason": "tool_calls",
            }

    def test_with_tools_but_no_tool_call(self):
        # [FUNCTION DOC] Tools are supplied but the model text has no tool call. Asserts plain
        # content, no tool calls, stop finish reason, and prompt ids templated with the tools.
        response_text = "The weather is sunny today."

        def process_fn(_: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Ignores the prompt and returns the plain text with stop
            # finish reason.
            return ProcessResult(text=response_text, finish_reason="stop")

        with with_mock_server(process_fn=process_fn) as server:
            response = requests.post(
                f"{server.url}/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "What's the weather?"}],
                    "tools": SAMPLE_TOOLS,
                    "return_prompt_token_ids": True,
                },
                timeout=5.0,
            )
            data = response.json()

            choice = data["choices"][0]
            choice.pop("meta_info", None)
            assert choice == {
                "index": 0,
                "message": {"role": "assistant", "content": response_text, "tool_calls": None},
                "logprobs": {"content": expected_logprobs(server.tokenizer, response_text)},
                "prompt_token_ids": expected_prompt_token_ids(
                    server.tokenizer,
                    [{"role": "user", "content": "What's the weather?"}],
                    SAMPLE_TOOLS,
                ),
                "finish_reason": "stop",
            }

    def test_with_multiple_tool_calls(self):
        # [FUNCTION DOC] The model text has a leading sentence and two tool call blocks (get-year
        # with empty arguments, get-temperature for Shanghai). Asserts the content is the sentence,
        # two tool calls with ids call00000 and call00001 and indices 0 and 1 with serialized
        # arguments, and the tool calls finish reason.
        multi_tool_response = (
            "I will get year and temperature.\n"
            '<tool_call>\n{"name": "get_year", "arguments": {}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_temperature", "arguments": {"location": "Shanghai"}}\n</tool_call>'
        )

        def process_fn(_: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Ignores the prompt and returns the two-tool-call text with
            # stop finish reason.
            return ProcessResult(text=multi_tool_response, finish_reason="stop")

        with with_mock_server(process_fn=process_fn) as server:
            response = requests.post(
                f"{server.url}/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "What year and temperature?"}],
                    "tools": SAMPLE_TOOLS,
                    "return_prompt_token_ids": True,
                },
                timeout=5.0,
            )
            data = response.json()

            choice = data["choices"][0]
            choice.pop("meta_info", None)
            assert choice == {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I will get year and temperature.",
                    "tool_calls": [
                        {
                            "id": "call00000",
                            "index": 0,
                            "type": "function",
                            "function": {"name": "get_year", "arguments": "{}"},
                        },
                        {
                            "id": "call00001",
                            "index": 1,
                            "type": "function",
                            "function": {"name": "get_temperature", "arguments": '{"location": "Shanghai"}'},
                        },
                    ],
                },
                "logprobs": {"content": expected_logprobs(server.tokenizer, multi_tool_response)},
                "prompt_token_ids": expected_prompt_token_ids(
                    server.tokenizer,
                    [{"role": "user", "content": "What year and temperature?"}],
                    SAMPLE_TOOLS,
                ),
                "finish_reason": "tool_calls",
            }


class TestMultiTurnToolCallProcessFn:
    # [CLASS DOC] Tests of the two-turn tool call stub process function through both endpoints.
    @pytest.mark.parametrize(
        "prompt,expected_response",
        [
            pytest.param(TwoTurnStub.FIRST_PROMPT, TwoTurnStub.FIRST_RESPONSE, id="first_turn"),
            pytest.param(TwoTurnStub.SECOND_PROMPT, TwoTurnStub.SECOND_RESPONSE, id="second_turn"),
        ],
    )
    def test_generate_endpoint(self, prompt, expected_response):
        # [FUNCTION DOC] Parametrized over the first and second turn prompts. Encodes the prompt,
        # posts it to the generate endpoint of a server using the stub, and asserts status 200,
        # the stub's response text for that turn, and a stop finish reason.
        with with_mock_server(process_fn=TwoTurnStub.process_fn) as server:
            input_ids = server.tokenizer.encode(prompt, add_special_tokens=False)
            response = requests.post(
                f"{server.url}/generate",
                json={"input_ids": input_ids, "sampling_params": {}, "return_logprob": True},
                timeout=5.0,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["text"] == expected_response
            assert data["meta_info"]["finish_reason"] == {"type": "stop"}

    @pytest.mark.parametrize(
        "messages,expected_content,expected_tool_calls,expected_finish_reason",
        [
            pytest.param(
                TwoTurnStub.OPENAI_MESSAGES_FIRST_TURN,
                TwoTurnStub.FIRST_RESPONSE_CONTENT,
                TwoTurnStub.FIRST_TOOL_CALLS_SGLANG_WIRE,
                "tool_calls",
                id="first_turn",
            ),
            pytest.param(
                TwoTurnStub.OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT,
                TwoTurnStub.SECOND_RESPONSE,
                None,
                "stop",
                id="second_turn",
            ),
        ],
    )
    def test_chat_completions_endpoint(self, messages, expected_content, expected_tool_calls, expected_finish_reason):
        # [FUNCTION DOC] Parametrized over the first turn (expects the tool call wire format and
        # tool calls finish reason) and the second turn from the client (expects the final answer,
        # no tool calls, stop). Posts the messages with the sample tools and asserts status 200
        # and the message content, tool calls and finish reason.
        with with_mock_server(process_fn=TwoTurnStub.process_fn) as server:
            response = requests.post(
                f"{server.url}/v1/chat/completions",
                json={"model": "test", "messages": messages, "tools": SAMPLE_TOOLS},
                timeout=5.0,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["content"] == expected_content
            assert data["choices"][0]["message"]["tool_calls"] == expected_tool_calls
            assert data["choices"][0]["finish_reason"] == expected_finish_reason


class TestNodeAddress:
    # [CLASS DOC] Tests of the default advertised host of the helper that starts a server.
    def test_the_default_host_is_the_node_ip_instead_of_loopback(self):
        # [FUNCTION DOC] A default server's host equals the node ip, is not loopback, and its url
        # is built from the node ip and port.
        """``with_mock_server`` must default to an address other nodes can reach."""
        with with_mock_server() as server:
            assert server.host == NodeProbeMixin._get_node_ip()
            assert server.host != "127.0.0.1"
            assert server.url == f"http://{NodeProbeMixin._get_node_ip()}:{server.port}"


class TestHostResolution:
    # [CLASS DOC] Tests of how the server resolves its advertised host from an explicit value,
    # the node ip or the host ip environment override.
    def test_an_explicit_host_is_kept_instead_of_the_node_ip(self):
        # [FUNCTION DOC] A server object constructed directly with host 127.0.0.1 and port 0
        # keeps that host and builds its url from it. (The server is not started or closed here.)
        """A caller that pins a host keeps it, so a test can still ask for a loopback-only mock."""
        server = MockSGLangServer(
            model_name="Qwen/Qwen3-0.6B", process_fn=default_process_fn, host="127.0.0.1", port=0
        )

        assert server.host == "127.0.0.1"
        assert server.url == f"http://127.0.0.1:{server.port}"

    def test_a_missing_host_falls_back_to_the_node_ip(self):
        # [FUNCTION DOC] A server constructed with no host advertises the node ip, not loopback.
        """Without an explicit host the mock advertises the node ip so peers on other nodes can dial it."""
        server = MockSGLangServer(model_name="Qwen/Qwen3-0.6B", process_fn=default_process_fn, host=None, port=0)

        assert server.host == NodeProbeMixin._get_node_ip()
        assert server.host != "127.0.0.1"

    def test_a_server_advertising_the_node_ip_still_answers_on_loopback(self, mock_server):
        # [FUNCTION DOC] A health request over loopback to the shared server's port returns 200.
        """The mock binds every interface, so advertising the node ip must not cut off local clients."""
        response = requests.get(f"http://127.0.0.1:{mock_server.port}/health", timeout=5)

        assert response.status_code == 200

    def test_the_default_host_honours_the_node_ip_override(self, monkeypatch):
        # [FUNCTION DOC] With the host ip override set to loopback, a default server's url uses
        # loopback and its health endpoint answers 200 (10 second timeout).
        """A deployment publishes a reachable address through MILES_HOST_IP; the mock must serve on that one."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "127.0.0.1")

        with with_mock_server() as server:
            assert server.url == f"http://127.0.0.1:{server.port}"
            assert requests.get(f"{server.url}/health", timeout=10).status_code == 200

    def test_an_explicit_host_wins_over_the_node_ip_override(self, monkeypatch):
        # [FUNCTION DOC] With the override set to an unreachable address but an explicit loopback
        # host passed to the starter, the server's host is loopback and its health endpoint
        # answers 200.
        """A caller that pins a host asked for that exact address; an env override must not silently replace it."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "10.9.9.9")

        with with_mock_server(host="127.0.0.1") as server:
            assert server.host == "127.0.0.1"
            assert requests.get(f"{server.url}/health", timeout=10).status_code == 200
