# [MODULE DOC] Integration tests for the Anthropic Messages route of the session server, which must
# reuse the OpenAI chat session path end to end. A real session server (version 1 and version 2
# variants via a module-scoped parametrized fixture) runs on a background thread in front of a mock
# SGLang backend whose scripted replies depend on the prompt text (a tool-call reply when the prompt
# asks to use the weather tool, a final answer when the prompt contains the tool result "sunny",
# otherwise an echo). Tests cover: the unavailable-adapter error, the health capability flag, record
# shape, thinking history handling, intermediate system messages, fake streaming events, parity with
# equivalent OpenAI requests, error envelopes, fixed feature policy rejections and their priority,
# policy traversal boundaries, post-commit failures, the multi-turn tool flow under strict versus
# loose matchers, the delete-during-request race, and the matcher gate itself. Opens local ports
# starting at 31000 and loads a real tokenizer in one test.
"""Integration tests for ``POST /sessions/{session_id}/v1/messages``.

The Anthropic route must reuse the OpenAI session path end to end: canonical
OpenAI ``SessionRecord``s, TITO ``input_ids``, matcher semantics, and the
same commit/skip decisions — clients only ever see Anthropic wire shapes.
"""

import json
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from fastapi.responses import JSONResponse
from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import convert_to_chat_completion_request
from tests.fast.fixtures.session_fixtures import make_session_server_config

import miles.rollout.session.core as core_module
import miles.rollout.session.v2.core as v2_core_module
from miles.rollout.session import sessions as sessions_module
from miles.rollout.session.server import SessionServer
from miles.utils.chat_template_utils.message_matcher_hub import resolve_session_message_matcher, strict_message_matches
from miles.utils.http_utils import find_available_port
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

# Two-key arguments: the qwen25 parser re-serializes them in this key order,
# so a replay whose input object uses the REVERSED key order re-serializes to
# a different spelling — the matcher-gate scenario from the design doc.
_TOOL_CALL_TEXT = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}\n</tool_call>'
_TOOLS = [{"name": "get_weather", "description": "weather", "input_schema": {"type": "object", "properties": {}}}]


def _process_fn(prompt: str) -> ProcessResult:
    # [FUNCTION DOC] HELPER: Mock backend reply policy keyed on the prompt text. If the prompt
    # contains "sunny", returns the text "final-answer" with finish reason stop; else if it contains
    # "use the weather tool", returns the scripted weather tool-call text; otherwise returns
    # "anthropic-echo". All finish with stop.
    if "sunny" in prompt:
        return ProcessResult(text="final-answer", finish_reason="stop")
    if "use the weather tool" in prompt:
        return ProcessResult(text=_TOOL_CALL_TEXT, finish_reason="stop")
    return ProcessResult(text="anthropic-echo", finish_reason="stop")


@contextmanager
def _anthropic_env(extra_args: dict | None = None, *, latency: float = 0.0):
    # [FUNCTION DOC] HELPER: Context manager. Starts a mock backend with the reply policy and the
    # given latency (seconds), builds a session server config pointing at it (30 s timeout, Qwen3-0.6B
    # checkpoint, template kwargs with thinking disabled, default token-in-token-out model, a random
    # instance id, retract pause mode merged with and overridden by the extra args mapping), serves
    # the server on a free port from 31000 on a background thread, and yields a namespace with the
    # server URL and the backend. The server thread is stopped on exit; the backend is stopped by
    # its own context manager.
    # The mock backend already emits choice.meta_info with
    # output_token_logprobs/completion_tokens in the session-server format.
    with with_mock_server(process_fn=_process_fn, latency=latency) as backend:
        config = make_session_server_config(
            backend_url=backend.url,
            timeout=30,
            hf_checkpoint="Qwen/Qwen3-0.6B",
            apply_chat_template_kwargs={"enable_thinking": False},
            tito_model="default",
            instance_id=uuid.uuid4().hex,
            **({"pause_generation_mode": "retract"} | (extra_args or {})),
        )
        server_obj = SessionServer(config)
        port = find_available_port(31000)
        server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
        server.start()
        try:
            yield SimpleNamespace(url=f"http://127.0.0.1:{port}", backend=backend)
        finally:
            server.stop()


_V2_ARGS = {
    "use_session_server": "v2",
    "session_sample_picker_path": "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
    "session_sample_postprocessor_path": "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
}


@pytest.fixture(scope="module", params=["v1", "v2"])
def anthropic_env(request):
    # [FUNCTION DOC] HELPER: Module-scoped fixture parametrized over session server versions "v1"
    # and "v2". Starts the environment with the version 2 argument set for v2 (none for v1) and
    # yields a namespace of the version name plus the environment's URL and backend.
    with _anthropic_env(_V2_ARGS if request.param == "v2" else None) as env:
        yield SimpleNamespace(version=request.param, **vars(env))


@pytest.fixture(scope="module", params=["v1", "v2"])
def anthropic_env_loose(request):
    # [FUNCTION DOC] HELPER: Like the main environment fixture but with the session message matcher
    # set to the loose tool-call matcher in addition (for both versions).
    extra = {"session_message_matcher": "loose_tool_call", **(_V2_ARGS if request.param == "v2" else {})}
    with _anthropic_env(extra) as env:
        yield SimpleNamespace(version=request.param, **vars(env))


def _create_session(url: str) -> str:
    # [FUNCTION DOC] HELPER: Posts to the sessions path of the server (5 s timeout) and returns the
    # new session id from the JSON body.
    return requests.post(f"{url}/sessions", timeout=5.0).json()["session_id"]


def _post_messages(url: str, session_id: str, payload: dict) -> requests.Response:
    # [FUNCTION DOC] HELPER: Posts the JSON payload to the session's Anthropic messages path with a
    # 30 s timeout and returns the raw response (no status check).
    return requests.post(f"{url}/sessions/{session_id}/v1/messages", json=payload, timeout=30.0)


def _records(url: str, session_id: str) -> list[dict]:
    # [FUNCTION DOC] HELPER: Fetches the session over HTTP (5 s timeout) and returns the list of
    # records from its JSON body.
    return requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()["records"]


def _payload(messages, **extra) -> dict:
    # [FUNCTION DOC] HELPER: Builds an Anthropic messages request body with model "claude-test",
    # max tokens 64, the given messages, and any extra top-level fields (which may override).
    return {"model": "claude-test", "max_tokens": 64, "messages": messages, **extra}


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    # [FUNCTION DOC] HELPER: Parses a server-sent-events body: splits on blank lines, skips empty
    # blocks, and for each block asserts it has exactly an event line (prefix "event: ") followed by
    # a data line (prefix "data: "), returning a list of (event name, decoded JSON data) pairs.
    events = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event_line, data_line = block.split("\n", 1)
        assert event_line.startswith("event: ") and data_line.startswith("data: ")
        events.append((event_line[len("event: ") :], json.loads(data_line[len("data: ") :])))
    return events


@pytest.mark.parametrize(
    "missing_module",
    [
        "sglang.srt.entrypoints.anthropic",
        "sglang.srt.entrypoints.anthropic.utils",
        "sglang.srt.entrypoints.anthropic.serving",
        "sglang.srt.entrypoints.anthropic.protocol",
    ],
)
def test_sessions_module_imports_without_optional_sglang_anthropic_helpers(missing_module: str) -> None:
    # [FUNCTION DOC] Parametrized over four SGLang anthropic modules (package, utils, serving,
    # protocol). Runs a fresh interpreter subprocess that marks that module as unimportable, imports
    # the adapter and sessions modules, and asserts: when utils (or the package) is missing the
    # sessions module's utils reference is none; when serving (or the package) is missing the
    # response converter and request converter references are none; when protocol (or the package)
    # is missing the adapter reports itself unavailable. The test passes when the subprocess exits
    # with code 0 (error output is the failure message).
    script = (
        "import sys\n"
        f"missing = {missing_module!r}\n"
        "sys.modules[missing] = None\n"
        "from miles.rollout.session import anthropic_adapter, sessions\n"
        "if missing.endswith(('.utils', 'anthropic')):\n"
        "    assert sessions.anthropic_utils is None\n"
        "if missing.endswith(('.serving', 'anthropic')):\n"
        "    assert sessions.convert_response is None\n"
        "    assert sessions.convert_to_chat_completion_request is None\n"
        "if missing.endswith(('.protocol', 'anthropic')):\n"
        "    assert anthropic_adapter.anthropic_adapter_available() is False\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


class TestAnthropicRoute:
    # [CLASS DOC] Tests of the Anthropic messages route, run against both session server versions
    # through the parametrized environment fixture.
    def test_unavailable_sglang_adapter_returns_501_without_record(self, anthropic_env):
        # [FUNCTION DOC] With the sessions module's anthropic utils reference patched to none, a
        # messages request returns status 501 with an Anthropic error envelope (type api_error,
        # message that the installed SGLang does not support the adapter), and the session has no
        # records.
        session_id = _create_session(anthropic_env.url)
        with patch.object(sessions_module, "anthropic_utils", None):
            resp = _post_messages(anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}]))

        assert resp.status_code == 501
        assert resp.json() == {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "The installed SGLang does not support the Anthropic Messages adapter",
            },
        }
        assert _records(anthropic_env.url, session_id) == []

    def test_health_reports_live_intermediate_system_capability(self, anthropic_env):
        # [FUNCTION DOC] The server's health endpoint body has the intermediate-system-supported
        # capability flag set to true.
        body = requests.get(f"{anthropic_env.url}/health", timeout=5.0).json()
        assert body["anthropic_intermediate_system_supported"] is True

    def test_non_stream_text_creates_canonical_openai_record(self, anthropic_env):
        # [FUNCTION DOC] A non-streaming request with a system string and one user message returns
        # 200 JSON: an assistant message whose model is the backend's "mock-model" (frozen behavior),
        # content one text block "anthropic-echo", stop reason end_turn, and positive input and
        # output token usage. The session snapshot has exactly one record whose response id equals
        # the returned id (for v2 also the first tree node's response id), path is the OpenAI chat
        # completions path, request messages are [system "sys", user "hello"], request input ids is a
        # non-empty list, and the response object is a chat completion.
        session_id = _create_session(anthropic_env.url)
        resp = _post_messages(
            anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}], system="sys")
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert body["type"] == "message" and body["role"] == "assistant"
        # Non-stream JSON keeps the backend response model (frozen behavior).
        assert body["model"] == "mock-model"
        assert body["content"] == [{"type": "text", "text": "anthropic-echo"}]
        assert body["stop_reason"] == "end_turn"
        assert body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] > 0

        snapshot = requests.get(f"{anthropic_env.url}/sessions/{session_id}", timeout=5.0).json()
        records = snapshot["records"]
        assert len(records) == 1
        record = records[0]
        assert body["id"] == record["response"]["id"]
        if anthropic_env.version == "v2":
            assert body["id"] == snapshot["metadata"]["tree"]["nodes"][0]["response_id"]
        assert record["path"] == "/v1/chat/completions"
        assert record["request"]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        assert isinstance(record["request"]["input_ids"], list) and record["request"]["input_ids"]
        assert record["response"]["object"] == "chat.completion"

    def test_assistant_thinking_history_becomes_canonical_reasoning_content(self, anthropic_env):
        # [FUNCTION DOC] A request whose assistant history message has a thinking block ("ponder\n")
        # and a text block ("hello") succeeds, and the stored request's assistant message has content
        # "hello" and a reasoning content field "ponder\n".
        session_id = _create_session(anthropic_env.url)
        resp = _post_messages(
            anthropic_env.url,
            session_id,
            _payload(
                [
                    {"role": "user", "content": "first"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "ponder\n"},
                            {"type": "text", "text": "hello"},
                        ],
                    },
                    {"role": "user", "content": "again"},
                ]
            ),
        )
        assert resp.status_code == 200

        [record] = _records(anthropic_env.url, session_id)
        assistant = record["request"]["messages"][1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "hello"
        assert assistant["reasoning_content"] == "ponder\n"

    @pytest.mark.parametrize(
        ("remaining_block", "expected_content"),
        [
            (None, ""),
            (
                {"type": "tool_use", "id": "call1", "name": "get_weather", "input": {"city": "Paris"}},
                None,
            ),
        ],
    )
    def test_assistant_thinking_strip_preserves_the_rest_of_the_message(self, remaining_block, expected_content):
        # [FUNCTION DOC] No server needed. Parametrized: an assistant message with two thinking
        # blocks ("first", "second") and either nothing else (expected content empty string) or a
        # tool-use block (expected content unchecked, tool call preserved). The request is validated
        # as an Anthropic request, the reasoning history is stripped, converted to an OpenAI chat
        # request (merging inline system), dumped to JSON (excluding unset and none values, by
        # alias), and the reasoning history is restored. The single assistant message must have
        # reasoning content "first\nsecond", and either the expected empty content or the tool call
        # named get_weather.
        content = [{"type": "thinking", "thinking": "first"}, {"type": "thinking", "thinking": "second"}]
        if remaining_block is not None:
            content.append(remaining_block)
        request = AnthropicMessagesRequest.model_validate(_payload([{"role": "assistant", "content": content}]))

        conversion_request, reasoning_history = sessions_module._strip_anthropic_reasoning_history(request)
        converted = convert_to_chat_completion_request(conversion_request, merge_inline_system=True)
        dumped = converted.model_dump(mode="json", exclude_none=True, exclude_unset=True, by_alias=True)
        sessions_module._restore_anthropic_reasoning_history(dumped, reasoning_history)

        [assistant] = dumped["messages"]
        assert assistant["reasoning_content"] == "first\nsecond"
        if expected_content is not None:
            assert assistant["content"] == expected_content
        else:
            assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_assistant_thinking_response_replays_as_the_same_canonical_history(self, anthropic_env, monkeypatch):
        # [FUNCTION DOC] The backend's chat completion computation is wrapped so the reply carries
        # reasoning content "ponder\n". The first request returns content whose first block is a
        # thinking block with that text. The second request replays the first reply's content as the
        # assistant history plus a new user message and succeeds. Two records exist; the stored
        # response message of the first matches (under the strict matcher) the replayed assistant
        # message of the second request, which has reasoning content "ponder\n".
        original = anthropic_env.backend._compute_chat_completions_response

        def with_reasoning(payload):
            # [FUNCTION DOC] HELPER: Calls the original computation and sets the first choice's
            # message reasoning content to "ponder\n"; returns the modified response.
            response = original(payload)
            response["choices"][0]["message"]["reasoning_content"] = "ponder\n"
            return response

        monkeypatch.setattr(anthropic_env.backend, "_compute_chat_completions_response", with_reasoning)
        session_id = _create_session(anthropic_env.url)
        first = _post_messages(
            anthropic_env.url,
            session_id,
            _payload([{"role": "user", "content": "first"}]),
        )
        assert first.status_code == 200
        assert first.json()["content"][0] == {"type": "thinking", "thinking": "ponder\n"}

        second = _post_messages(
            anthropic_env.url,
            session_id,
            _payload(
                [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": first.json()["content"]},
                    {"role": "user", "content": "again"},
                ]
            ),
        )
        assert second.status_code == 200

        records = _records(anthropic_env.url, session_id)
        assert len(records) == 2
        stored = records[0]["response"]["choices"][0]["message"]
        replayed = records[1]["request"]["messages"][1]
        assert strict_message_matches(stored, replayed)
        assert replayed["reasoning_content"] == "ponder\n"

    def test_intermediate_system_stays_in_place(self, anthropic_env):
        # [FUNCTION DOC] Turn 1 sends a user message with a leading system "initial policy". Turn 2
        # replays the first reply and adds an intermediate system message "answer briefly" before a
        # new user message, keeping the leading system. Asserts: the second record's request roles
        # are system, user, assistant, system, user, with the two system messages being the leading
        # and intermediate ones in order. The second request's input ids agree with the first
        # request's input ids plus the first completion's token ids up to the length excluding the
        # session's maximum trim tokens; the decoded remaining suffix (real tokenizer) contains the
        # intermediate system text before the text "again". For v2 the tree nodes have parents none
        # then 0.
        leading_system = "initial policy"
        intermediate_system = "answer briefly"
        session_id = _create_session(anthropic_env.url)
        first = _post_messages(
            anthropic_env.url,
            session_id,
            _payload([{"role": "user", "content": "first"}], system=leading_system),
        )
        assert first.status_code == 200

        second = _post_messages(
            anthropic_env.url,
            session_id,
            _payload(
                [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": first.json()["content"]},
                    {"role": "system", "content": intermediate_system},
                    {"role": "user", "content": "again"},
                ],
                system=leading_system,
            ),
        )
        assert second.status_code == 200

        snapshot = requests.get(f"{anthropic_env.url}/sessions/{session_id}", timeout=5.0).json()
        first_record, second_record = snapshot["records"]
        messages = second_record["request"]["messages"]
        assert [message["role"] for message in messages] == [
            "system",
            "user",
            "assistant",
            "system",
            "user",
        ]
        assert [message for message in messages if message["role"] == "system"] == [
            {"role": "system", "content": leading_system},
            {"role": "system", "content": intermediate_system},
        ]

        completion_ids = [
            item[1] for item in first_record["response"]["choices"][0]["meta_info"]["output_token_logprobs"]
        ]
        expected_prefix = first_record["request"]["input_ids"] + completion_ids
        check_len = max(0, len(expected_prefix) - snapshot["metadata"]["max_trim_tokens"])
        assert second_record["request"]["input_ids"][:check_len] == expected_prefix[:check_len]
        decoded_suffix = load_tokenizer("Qwen/Qwen3-0.6B", trust_remote_code=True).decode(
            second_record["request"]["input_ids"][check_len:]
        )
        assert decoded_suffix.index(intermediate_system) < decoded_suffix.index("again")
        if anthropic_env.version == "v2":
            assert [node["parent"] for node in snapshot["metadata"]["tree"]["nodes"]] == [None, 0]

    def test_stream_returns_eager_fake_sse(self, anthropic_env):
        # [FUNCTION DOC] A streaming request returns 200 with an event-stream content type whose
        # events, in order, are message start, content block start, content block delta, content
        # block stop, message delta and message stop (the whole reply is produced eagerly after the
        # backend finishes). The start event's model is the request model "claude-test", with output
        # tokens 0 and positive input tokens; the delta is a text delta "anthropic-echo"; the message
        # delta has stop reason end_turn and positive output tokens. The single stored record's
        # response id equals the start message id (for v2 also the first tree node's response id).
        session_id = _create_session(anthropic_env.url)
        resp = _post_messages(
            anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}], stream=True)
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.text)
        assert [name for name, _ in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        message_start = events[0][1]["message"]
        # The stream's message model is the original Anthropic request model,
        # never the backend's served-model alias.
        assert message_start["model"] == "claude-test"
        assert message_start["usage"]["output_tokens"] == 0 and message_start["usage"]["input_tokens"] > 0
        assert events[2][1]["delta"] == {"type": "text_delta", "text": "anthropic-echo"}
        assert events[4][1]["delta"]["stop_reason"] == "end_turn"
        assert events[4][1]["usage"]["output_tokens"] > 0
        snapshot = requests.get(f"{anthropic_env.url}/sessions/{session_id}", timeout=5.0).json()
        [record] = snapshot["records"]
        assert message_start["id"] == record["response"]["id"]
        if anthropic_env.version == "v2":
            assert message_start["id"] == snapshot["metadata"]["tree"]["nodes"][0]["response_id"]

    def test_parity_with_equivalent_openai_request(self, anthropic_env):
        # [FUNCTION DOC] Sends an Anthropic request (system "sys", user "hello") in one session and
        # the equivalent OpenAI chat request in another. Both succeed, and the first record of each
        # has equal request messages, equal request input ids, and equal first-choice response
        # messages.
        anthropic_session = _create_session(anthropic_env.url)
        openai_session = _create_session(anthropic_env.url)

        anthropic_resp = _post_messages(
            anthropic_env.url, anthropic_session, _payload([{"role": "user", "content": "hello"}], system="sys")
        )
        assert anthropic_resp.status_code == 200

        openai_resp = requests.post(
            f"{anthropic_env.url}/sessions/{openai_session}/v1/chat/completions",
            json={
                "model": "claude-test",
                "max_tokens": 64,
                "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
            },
            timeout=30.0,
        )
        assert openai_resp.status_code == 200

        anthropic_record = _records(anthropic_env.url, anthropic_session)[0]
        openai_record = _records(anthropic_env.url, openai_session)[0]
        assert anthropic_record["request"]["messages"] == openai_record["request"]["messages"]
        assert anthropic_record["request"]["input_ids"] == openai_record["request"]["input_ids"]
        assert (
            anthropic_record["response"]["choices"][0]["message"] == openai_record["response"]["choices"][0]["message"]
        )

    def test_parity_with_equivalent_openai_request_with_tools(self, anthropic_env):
        # [FUNCTION DOC] Like the plain parity test but with one weather tool. The OpenAI request
        # spells the tool with the description field before the name and no extra defaults (auto tool
        # choice), which the chat template needs to render byte-identically. Asserts equal request
        # messages, tools and input ids between the two sessions' first records.
        """Tool-bearing parity pins the canonical tools spelling: the chat
        template renders tools in dict key order, so byte-identical TITO
        input_ids require the OpenAI client to write the codec's field order
        (function.description before function.name, no extra defaults)."""
        anthropic_session = _create_session(anthropic_env.url)
        openai_session = _create_session(anthropic_env.url)

        anthropic_resp = _post_messages(
            anthropic_env.url, anthropic_session, _payload([{"role": "user", "content": "hi"}], tools=_TOOLS)
        )
        assert anthropic_resp.status_code == 200

        openai_resp = requests.post(
            f"{anthropic_env.url}/sessions/{openai_session}/v1/chat/completions",
            json={
                "model": "claude-test",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "description": "weather",
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "auto",
            },
            timeout=30.0,
        )
        assert openai_resp.status_code == 200

        anthropic_record = _records(anthropic_env.url, anthropic_session)[0]
        openai_record = _records(anthropic_env.url, openai_session)[0]
        assert anthropic_record["request"]["messages"] == openai_record["request"]["messages"]
        assert anthropic_record["request"]["tools"] == openai_record["request"]["tools"]
        assert anthropic_record["request"]["input_ids"] == openai_record["request"]["input_ids"]

    def test_unexpected_core_exception_wears_anthropic_envelope(self, anthropic_env):
        # [FUNCTION DOC] With the completion extraction function patched (in both version 1 and
        # version 2 core modules, by attribute on pre-imported modules) to raise a runtime error, a
        # messages request returns 500 with a JSON content type and the generic envelope (type
        # error, api_error, message "Internal server error"), and no record is stored.
        """Unexpected (non-SessionError) processing failures keep the frozen
        wire behavior: a scrubbed generic Anthropic api_error envelope, not
        the framework's text/plain 500 page."""
        session_id = _create_session(anthropic_env.url)
        # patch.object on pre-imported modules: a string-target patch would
        # import v2.core lazily INSIDE the first patch's window and bake the
        # first mock into its from-import binding, leaking it after restore.
        with (
            patch.object(core_module, "extract_completion", side_effect=RuntimeError("boom")),
            patch.object(v2_core_module, "extract_completion", side_effect=RuntimeError("boom")),
        ):
            resp = _post_messages(anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}]))
        assert resp.status_code == 500
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json() == {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}
        assert _records(anthropic_env.url, session_id) == []

    def test_nan_sampling_param_maps_to_invalid_request_error(self, anthropic_env):
        # [FUNCTION DOC] A raw JSON body with the non-standard NaN literal as temperature is posted
        # with a JSON content type. The response is status 400 with error type
        # invalid_request_error, and no record is stored.
        """json.loads admits the non-standard NaN literal and the wire models
        accept it, but ``_render_json`` (allow_nan=False) rejects it — that
        failure must stay a 400 request error, never a plain-text 500."""
        session_id = _create_session(anthropic_env.url)
        raw = (
            b'{"model": "claude-test", "max_tokens": 64, "temperature": NaN,'
            b' "messages": [{"role": "user", "content": "x"}]}'
        )
        resp = requests.post(
            f"{anthropic_env.url}/sessions/{session_id}/v1/messages",
            data=raw,
            headers={"content-type": "application/json"},
            timeout=10.0,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"
        assert _records(anthropic_env.url, session_id) == []

    def test_error_envelopes_and_no_record_on_validation_failure(self, anthropic_env):
        # [FUNCTION DOC] Posting to a nonexistent session returns 404 with a not_found_error envelope
        # whose message is "session not found: session_id=nonexistent". For a real session, two bad
        # bodies (malformed JSON with an exact message about invalid JSON; a body missing max tokens
        # whose message mentions max_tokens) each return 400 invalid_request_error. During those
        # calls both cores' chat completion methods are patched with auto-specced mocks: they are
        # never awaited, the backend request count is unchanged, and the session's records are
        # unchanged.
        payload = _payload([{"role": "user", "content": "hello"}])
        resp = _post_messages(anthropic_env.url, "nonexistent", payload)
        assert resp.status_code == 404
        assert resp.json() == {
            "type": "error",
            "error": {"type": "not_found_error", "message": "session not found: session_id=nonexistent"},
        }

        session_id = _create_session(anthropic_env.url)
        cases = [
            (
                b"{not json",
                "invalid JSON body: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
            ),
            (
                json.dumps({"model": "claude-test", "messages": [{"role": "user", "content": "x"}]}).encode(),
                None,
            ),
        ]
        backend_requests = len(anthropic_env.backend.request_log)
        records = _records(anthropic_env.url, session_id)
        with (
            patch.object(core_module.SessionCore, "chat_completions", autospec=True) as v1_chat,
            patch.object(v2_core_module.SessionCoreV2, "chat_completions", autospec=True) as v2_chat,
        ):
            for raw, exact_message in cases:
                resp = requests.post(
                    f"{anthropic_env.url}/sessions/{session_id}/v1/messages",
                    data=raw,
                    headers={"content-type": "application/json"},
                    timeout=10.0,
                )
                assert resp.status_code == 400, raw
                error = resp.json()["error"]
                assert error["type"] == "invalid_request_error"
                if exact_message is not None:
                    assert error["message"] == exact_message
                else:
                    assert "max_tokens" in error["message"]
        assert v1_chat.await_count == 0
        assert v2_chat.await_count == 0
        assert len(anthropic_env.backend.request_log) == backend_requests
        assert _records(anthropic_env.url, session_id) == records

    def test_fixed_feature_policy_rejects_before_core(self, anthropic_env):
        # [FUNCTION DOC] A table of unsupported-feature payloads, each with its exact error message:
        # thinking enabled/disabled setting; output config; betas; server tools such as web search;
        # thinking blocks in user, system, system-field and tool-result positions (allowed only in
        # assistant history); redacted thinking; image blocks; tool reference and search result
        # blocks; tool results flagged as errors; tool-reference blocks inside tool results; image
        # blocks in the system field. Each is posted to a real session with both cores' chat
        # completion methods mocked: every response is 400 with an invalid_request_error envelope of
        # exactly that message, the mocks are never awaited, the backend saw no new requests, and the
        # session records are unchanged.
        messages = [{"role": "user", "content": "q"}]
        cases = [
            (
                _payload(messages, thinking={"type": "disabled"}),
                "thinking is not supported by this endpoint",
            ),
            (
                _payload(messages, output_config={"effort": "high"}),
                "output_config is not enabled for this deployment",
            ),
            (_payload(messages, betas=["b-1"]), "betas is not enabled for this deployment"),
            (
                _payload(messages, tools=[{"type": "web_search_20250305", "name": "web_search"}]),
                "server tool 'web_search' (type='web_search_20250305') is not enabled for this deployment",
            ),
            (
                _payload([{"role": "user", "content": [{"type": "thinking", "thinking": "t"}]}]),
                "thinking content blocks are only supported in assistant history",
            ),
            (
                _payload([{"role": "system", "content": [{"type": "thinking", "thinking": "t"}]}]),
                "thinking content blocks are only supported in assistant history",
            ),
            (
                _payload(messages, system=[{"type": "thinking", "thinking": "t"}]),
                "thinking content blocks are only supported in assistant history",
            ),
            (
                _payload(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "t",
                                    "content": [{"type": "thinking", "thinking": "t"}],
                                }
                            ],
                        }
                    ]
                ),
                "thinking content blocks are only supported in assistant history",
            ),
            (
                _payload([{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]}]),
                "redacted_thinking content blocks are not supported by this endpoint",
            ),
            (
                _payload(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "data": "eA=="},
                                }
                            ],
                        }
                    ]
                ),
                "image content blocks are not enabled for this deployment",
            ),
            (
                _payload([{"role": "user", "content": [{"type": "tool_reference", "tool_name": "f"}]}]),
                "tool_reference content blocks are not enabled for this deployment",
            ),
            (
                _payload([{"role": "user", "content": [{"type": "search_result", "title": "t"}]}]),
                "search_result content blocks are not enabled for this deployment",
            ),
            (
                _payload(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "t",
                                    "content": "failed",
                                    "is_error": True,
                                }
                            ],
                        }
                    ]
                ),
                "tool_result is_error=true is not supported by this endpoint",
            ),
            (
                _payload(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "t",
                                    "content": [{"type": "tool_reference", "tool_name": "f"}],
                                }
                            ],
                        }
                    ]
                ),
                "tool_reference content blocks are not enabled for this deployment",
            ),
            (
                _payload(messages, system=[{"type": "image", "source": {"url": "https://x"}}]),
                "image content blocks are not enabled for this deployment",
            ),
        ]
        session_id = _create_session(anthropic_env.url)
        backend_requests = len(anthropic_env.backend.request_log)
        records = _records(anthropic_env.url, session_id)
        with (
            patch.object(core_module.SessionCore, "chat_completions", autospec=True) as v1_chat,
            patch.object(v2_core_module.SessionCoreV2, "chat_completions", autospec=True) as v2_chat,
        ):
            for payload, message in cases:
                resp = _post_messages(anthropic_env.url, session_id, payload)
                assert resp.status_code == 400, payload
                assert resp.json() == {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": message},
                }
        assert v1_chat.await_count == 0
        assert v2_chat.await_count == 0
        assert len(anthropic_env.backend.request_log) == backend_requests
        assert _records(anthropic_env.url, session_id) == records

    def test_converter_exception_is_400_before_core(self, anthropic_env):
        # [FUNCTION DOC] With the request converter in the sessions module patched to raise a runtime
        # error "conversion boom" and the cores' chat completion methods mocked, a valid request
        # returns 400 with an invalid_request_error envelope carrying that message; the mocks are
        # never awaited, the backend saw no new requests and the records are unchanged.
        session_id = _create_session(anthropic_env.url)
        backend_requests = len(anthropic_env.backend.request_log)
        records = _records(anthropic_env.url, session_id)
        with (
            patch.object(
                sessions_module,
                "convert_to_chat_completion_request",
                side_effect=RuntimeError("conversion boom"),
            ),
            patch.object(core_module.SessionCore, "chat_completions", autospec=True) as v1_chat,
            patch.object(v2_core_module.SessionCoreV2, "chat_completions", autospec=True) as v2_chat,
        ):
            resp = _post_messages(anthropic_env.url, session_id, _payload([{"role": "user", "content": "q"}]))
        assert resp.status_code == 400
        assert resp.json() == {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "conversion boom"},
        }
        assert v1_chat.await_count == 0
        assert v2_chat.await_count == 0
        assert len(anthropic_env.backend.request_log) == backend_requests
        assert _records(anthropic_env.url, session_id) == records

    def test_feature_priority_is_stable(self):
        # [FUNCTION DOC] No server needed. Builds five payloads that each carry the offending
        # features from one level downward in the priority order: thinking, output config, betas,
        # server tool, system image block (all over a user message containing a tool reference). For
        # each, the session sessions module's private feature validator raises a value error whose
        # text is the message for the highest-priority feature present: thinking, then output config,
        # betas, server tool, finally image blocks.
        messages = [
            {"role": "user", "content": [{"type": "tool_reference", "tool_name": "f"}]},
        ]
        payloads = [
            (
                _payload(
                    messages,
                    thinking={"type": "disabled"},
                    output_config={"effort": "high"},
                    betas=["b-1"],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    system=[{"type": "image", "source": {"url": "https://x"}}],
                ),
                "thinking is not supported by this endpoint",
            ),
            (
                _payload(
                    messages,
                    output_config={"effort": "high"},
                    betas=["b-1"],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    system=[{"type": "image", "source": {"url": "https://x"}}],
                ),
                "output_config is not enabled for this deployment",
            ),
            (
                _payload(
                    messages,
                    betas=["b-1"],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    system=[{"type": "image", "source": {"url": "https://x"}}],
                ),
                "betas is not enabled for this deployment",
            ),
            (
                _payload(
                    messages,
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    system=[{"type": "image", "source": {"url": "https://x"}}],
                ),
                "server tool 'web_search' (type='web_search_20250305') is not enabled for this deployment",
            ),
            (
                _payload(messages, system=[{"type": "image", "source": {"url": "https://x"}}]),
                "image content blocks are not enabled for this deployment",
            ),
        ]
        for payload, message in payloads:
            request = AnthropicMessagesRequest.model_validate(payload)
            with pytest.raises(ValueError) as exc_info:
                sessions_module._validate_anthropic_features(request)
            assert str(exc_info.value) == message

    def test_policy_traversal_boundaries_match_converter(self):
        # [FUNCTION DOC] No server needed. Checks that the feature validator only inspects the
        # content positions the SGLang converter actually uses. Case 1: a tool-result block placed in
        # the system field holding an image block passes validation, and the converted request has
        # only the user message. Case 2: an image block nested two levels deep inside tool results
        # passes validation and is dropped by the converter (the tool message has content "depth1" and
        # tool call id "outer", and no image URL appears in the dump). Case 3: image/thinking/search
        # result type names appearing inside arbitrary JSON (a tool-use input, request metadata, a
        # tool input schema) pass validation and are preserved verbatim: the converted tool call
        # arguments decode to the original object and the tool parameters equal the original schema.
        system_nested = _payload(
            [{"role": "user", "content": "q"}],
            betas=[],
            system=[
                {
                    "type": "tool_result",
                    "tool_use_id": "system-tool",
                    "content": [{"type": "image", "source": {"url": "https://x"}}],
                }
            ],
        )
        request = AnthropicMessagesRequest.model_validate(system_nested)
        sessions_module._validate_anthropic_features(request)
        converted = convert_to_chat_completion_request(request, merge_inline_system=True)
        assert converted.model_dump(mode="json", exclude_none=True)["messages"] == [{"role": "user", "content": "q"}]

        depth_two = _payload(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "outer",
                            "content": [
                                {"type": "text", "text": "depth1"},
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "inner",
                                    "content": [{"type": "image", "source": {"type": "base64", "data": "eA=="}}],
                                },
                            ],
                        }
                    ],
                }
            ]
        )
        request = AnthropicMessagesRequest.model_validate(depth_two)
        sessions_module._validate_anthropic_features(request)
        converted = convert_to_chat_completion_request(request, merge_inline_system=True)
        dump = converted.model_dump(mode="json", exclude_none=True)
        assert dump["messages"] == [{"role": "tool", "content": "depth1", "tool_call_id": "outer"}]
        assert "image_url" not in json.dumps(dump)

        arbitrary_json = _payload(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call1",
                            "name": "f",
                            "input": {"type": "image", "nested": {"type": "thinking"}},
                        }
                    ],
                }
            ],
            metadata={"type": "search_result"},
            tools=[
                {
                    "name": "f",
                    "input_schema": {"type": "object", "properties": {"x": {"type": "image"}}},
                }
            ],
        )
        request = AnthropicMessagesRequest.model_validate(arbitrary_json)
        sessions_module._validate_anthropic_features(request)
        converted = convert_to_chat_completion_request(request, merge_inline_system=True)
        dump = converted.model_dump(mode="json", exclude_none=True)
        assert json.loads(dump["messages"][0]["tool_calls"][0]["function"]["arguments"]) == {
            "type": "image",
            "nested": {"type": "thinking"},
        }
        assert dump["tools"][0]["function"]["parameters"] == {
            "type": "object",
            "properties": {"x": {"type": "image"}},
        }

    def test_backend_failure_maps_to_anthropic_error_without_record(self, anthropic_env):
        # [FUNCTION DOC] With the mock backend's generate-like request handler replaced so every call
        # returns a 500 JSON response, a messages request returns 500 with an api_error envelope with
        # message "Internal server error" (backend detail scrubbed), and no record is stored.
        session_id = _create_session(anthropic_env.url)

        async def reject(self, request, compute_fn):
            # [FUNCTION DOC] HELPER: Replacement handler; ignores its inputs and returns a JSON
            # response with an error entry "mock backend failure" and status 500.
            return JSONResponse(content={"error": "mock backend failure"}, status_code=500)

        with patch.object(MockSGLangServer, "_handle_generate_like_request", new=reject):
            resp = _post_messages(anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}]))
        assert resp.status_code == 500
        body = resp.json()
        assert body["type"] == "error" and body["error"]["type"] == "api_error"
        assert body["error"]["message"] == "Internal server error"
        assert _records(anthropic_env.url, session_id) == []

    def test_post_commit_conversion_failure_returns_500_and_keeps_record(self, anthropic_env):
        # [FUNCTION DOC] With the response converter patched to raise after the core committed, a
        # request returns 500 with the generic api_error envelope, yet the record stays stored
        # (accepted first-version boundary): one record exists.
        session_id = _create_session(anthropic_env.url)
        with patch.object(sessions_module, "convert_response", side_effect=RuntimeError("boom")):
            resp = _post_messages(anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}]))
        assert resp.status_code == 500
        assert resp.json() == {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}
        # The accepted first-version boundary: core already committed.
        assert len(_records(anthropic_env.url, session_id)) == 1

    def test_sse_build_failure_returns_json_500_not_partial_stream(self, anthropic_env):
        # [FUNCTION DOC] With the fake streaming event builder patched to raise, a streaming request
        # returns a JSON (not event-stream) 500 whose error type is api_error; the record stays
        # stored (same post-commit boundary as the non-streaming case).
        session_id = _create_session(anthropic_env.url)
        with patch.object(
            sessions_module.anthropic_utils, "to_anthropic_fake_sse_events", side_effect=RuntimeError("boom")
        ):
            resp = _post_messages(
                anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}], stream=True)
            )
        assert resp.status_code == 500
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["error"]["type"] == "api_error"
        # Same post-commit boundary as the non-stream twin: the record stays.
        assert len(_records(anthropic_env.url, session_id)) == 1


def _tool_turn1(url: str, session_id: str) -> dict:
    # [FUNCTION DOC] HELPER: Sends the tool-eliciting first turn (user asks to use the weather tool,
    # weather tool declared), asserts status 200, stop reason tool_use, and a tool-use block named
    # get_weather with input city Paris and unit C. Returns that block.
    """Run the tool-eliciting first turn; returns the tool_use block."""
    turn1 = _post_messages(
        url, session_id, _payload([{"role": "user", "content": "please use the weather tool"}], tools=_TOOLS)
    )
    assert turn1.status_code == 200
    body = turn1.json()
    assert body["stop_reason"] == "tool_use"
    tool_use = next(block for block in body["content"] if block["type"] == "tool_use")
    assert tool_use["name"] == "get_weather" and tool_use["input"] == {"city": "Paris", "unit": "C"}
    return tool_use


def _tool_turn2_payload(tool_use: dict) -> dict:
    # [FUNCTION DOC] HELPER: Builds the second-turn request: the original user message, an assistant
    # message holding the tool-use block with its input keys in reversed order (so the re-serialized
    # arguments differ in spelling), and a user message with a tool result "sunny" for the tool-use
    # id; the weather tool is declared. Returns the payload.
    """Replay with the tool_use input keys REVERSED: json.dumps preserves the
    object's key order, so the re-serialized arguments spelling differs from
    the stored one — accepted only by ``loose_tool_call``."""
    respelled_input = dict(reversed(list(tool_use["input"].items())))
    return _payload(
        [
            {"role": "user", "content": "please use the weather tool"},
            {"role": "assistant", "content": [{**tool_use, "input": respelled_input}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "sunny"}]},
        ],
        tools=_TOOLS,
    )


class TestAnthropicToolFlow:
    # [CLASS DOC] End-to-end tool-calling flow tests across two turns.
    def test_multi_turn_tool_flow_reuses_stored_prefix(self, anthropic_env_loose):
        # [FUNCTION DOC] Under the loose tool-call matcher, turn 1 elicits a tool call and turn 2
        # replays it with reversed input key order and the tool result. Turn 2 returns 200 with
        # content one text block "final-answer". Two records exist; the second request's input ids
        # start with the first request's input ids (stored prefix reused); the replayed assistant
        # message has a tool call named get_weather whose arguments decode to city Paris and unit C.
        url = anthropic_env_loose.url
        session_id = _create_session(url)
        tool_use = _tool_turn1(url, session_id)

        turn2 = _post_messages(url, session_id, _tool_turn2_payload(tool_use))
        assert turn2.status_code == 200
        assert turn2.json()["content"] == [{"type": "text", "text": "final-answer"}]

        records = _records(url, session_id)
        assert len(records) == 2
        first_ids = records[0]["request"]["input_ids"]
        second_ids = records[1]["request"]["input_ids"]
        # The stored TITO prefix is reused: turn 2's prompt extends turn 1's.
        assert second_ids[: len(first_ids)] == first_ids
        replayed_assistant = next(m for m in records[1]["request"]["messages"] if m["role"] == "assistant")
        assert replayed_assistant["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(replayed_assistant["tool_calls"][0]["function"]["arguments"]) == {
            "city": "Paris",
            "unit": "C",
        }

    def test_strict_matcher_rejects_respelled_tool_arguments(self, anthropic_env):
        # [FUNCTION DOC] Skipped for version 2 (it branches into a new lineage instead of rolling
        # back). For version 1 under the default strict matcher: turn 1 elicits a tool call; turn 2
        # replays it with reversed input key order and succeeds with 200, but the re-serialized
        # arguments differ from the stored message, so the session rolls back to empty, discarding
        # turn 1's record. Exactly one record remains with request roles user, assistant, tool, and
        # its input ids do not start with turn 1's input ids (re-rendered from scratch).
        """Design matcher gate, end to end: under the default ``strict``
        matcher the re-serialized tool arguments diverge from the stored
        assistant message, so v1 rolls back to the empty checkpoint —
        discarding turn 1's record — and re-renders from scratch instead of
        reusing the stored TITO prefix (contrast with the loose test above)."""
        if anthropic_env.version != "v1":
            pytest.skip("v2 branches to a new lineage instead of rolling back; the loose fixture covers v2")
        url = anthropic_env.url
        session_id = _create_session(url)
        tool_use = _tool_turn1(url, session_id)
        first_ids = _records(url, session_id)[0]["request"]["input_ids"]

        turn2 = _post_messages(url, session_id, _tool_turn2_payload(tool_use))
        assert turn2.status_code == 200

        records = _records(url, session_id)
        assert len(records) == 1
        assert [m["role"] for m in records[0]["request"]["messages"]] == ["user", "assistant", "tool"]
        second_ids = records[0]["request"]["input_ids"]
        assert second_ids[: len(first_ids)] != first_ids


class TestAnthropicCloseRace:
    # [CLASS DOC] Tests of closing a session while a request is being served.
    def test_delete_during_inflight_chat_skips_update_gracefully(self):
        # [FUNCTION DOC] Starts a dedicated server with a 0.35 s backend latency. In a thread pool a
        # messages request is submitted; the test polls (up to 5 s, every 10 ms) until the backend has
        # logged the request (failing with an assertion error otherwise), then deletes the session.
        # Asserts: delete returns 204; the in-flight request still returns 200 with a message-typed
        # body whose content is one text block "anthropic-echo" (the commit is skipped because the
        # session was closing, but the client gets a well-formed reply); a later request to the same
        # session returns 404 with a not_found_error.
        """Split-lock close race through the Anthropic route: DELETE lands
        while the chat is mid-proxy; Phase 3 sees closing=True and skips the
        commit, but the client still gets a well-formed Anthropic response —
        the same 200-with-skip outcome as the OpenAI route."""
        with _anthropic_env(latency=0.35) as env:
            session_id = _create_session(env.url)
            payload = _payload([{"role": "user", "content": "hello"}])

            with ThreadPoolExecutor(max_workers=2) as pool:
                inflight = pool.submit(_post_messages, env.url, session_id, payload)

                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if env.backend.request_log:
                        break
                    time.sleep(0.01)
                else:
                    raise AssertionError("in-flight request did not reach backend in time")

                delete_resp = requests.delete(f"{env.url}/sessions/{session_id}", timeout=30.0)
                inflight_resp = inflight.result(timeout=30.0)

            assert delete_resp.status_code == 204
            assert inflight_resp.status_code == 200
            body = inflight_resp.json()
            assert body["type"] == "message"
            assert body["content"] == [{"type": "text", "text": "anthropic-echo"}]

            post_delete = _post_messages(env.url, session_id, payload)
            assert post_delete.status_code == 404
            assert post_delete.json()["error"]["type"] == "not_found_error"


class TestMatcherGate:
    # [CLASS DOC] Unit tests (no server) of the strict and loose message matchers applied to tool
    # call arguments re-serialized from an object by the Anthropic-to-OpenAI conversion.
    """Design matcher gate: object → string re-serialization may change tool
    argument spelling; ``strict`` must reject it, ``loose_tool_call`` must
    accept it, so tool launch profiles default to ``loose_tool_call``."""

    def _replayed_assistant(self, arguments_object: dict) -> dict:
        # [FUNCTION DOC] HELPER: Builds an Anthropic request with a user message and an assistant
        # message holding one tool-use block (id call1, name f, the given input object), converts it
        # to an OpenAI chat request (merging inline system) and returns the last message of the dump
        # (excluding none values, by alias), i.e. the replayed assistant message.
        request = AnthropicMessagesRequest.model_validate(
            _payload(
                [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "call1", "name": "f", "input": arguments_object}],
                    },
                ]
            )
        )
        openai_request = convert_to_chat_completion_request(request, merge_inline_system=True)
        return openai_request.model_dump(mode="json", exclude_none=True, by_alias=True)["messages"][-1]

    def test_strict_rejects_and_loose_accepts_respelled_arguments(self):
        # [FUNCTION DOC] A stored assistant tool call with compact arguments '{"a":1}' is compared
        # with the replayed message whose arguments are the standard-library spelling '{"a": 1}':
        # the strict matcher returns false and the loose tool-call matcher (resolved by name) returns
        # true.
        stored = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call1", "index": 0, "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}
            ],
        }
        replayed = self._replayed_assistant({"a": 1})
        assert replayed["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'  # json.dumps spelling
        assert strict_message_matches(stored, replayed) is False
        loose = resolve_session_message_matcher("loose_tool_call")
        assert loose(stored, replayed) is True

    def test_strict_accepts_identical_spelling(self):
        # [FUNCTION DOC] When the stored arguments already use the standard-library spelling '{"a":
        # 1}', the strict matcher accepts the replayed message (returns true).
        stored = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call1", "index": 0, "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
            ],
        }
        replayed = self._replayed_assistant({"a": 1})
        assert strict_message_matches(stored, replayed) is True
