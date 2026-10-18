# [MODULE DOC] Integration tests for the session server's HTTP routes, run against a real
# uvicorn-hosted session server in a thread in front of a mock SGLang backend: health, session
# create/get/delete, chat-completion proxying and record keeping, header hygiene, error mapping
# (400/502), chat-template kwarg overrides, response filtering (meta info and logprobs hidden from
# the client but kept in the record), fake streaming (single SSE chunk), and routed-experts
# request offsets for the non-retract pause modes. The module constant is the fixed instance id
# the server reports.
"""Integration tests for session HTTP routes (create / get / delete / proxy)."""

import asyncio
import json
import socket
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
import requests
from fastapi.responses import JSONResponse
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.server import SessionServer
from miles.utils.chat_template_utils import strict_message_matches
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.openai_stream_client import stream_chat_completions
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


_INSTANCE_ID = "0123456789abcdef-0"


def _create_session(url: str) -> str:
    # [FUNCTION DOC] HELPER: Posts to the server's sessions route (5 second timeout) and returns
    # the new session id from the JSON body.
    return requests.post(f"{url}/sessions", timeout=5.0).json()["session_id"]


def _post_chat(url: str, session_id: str, payload: dict) -> requests.Response:
    # [FUNCTION DOC] HELPER: Posts the JSON payload to the session's chat-completions route (10
    # second timeout) and returns the raw HTTP response.
    return requests.post(f"{url}/sessions/{session_id}/v1/chat/completions", json=payload, timeout=10.0)


def _parse_sse(body: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Splits a server-sent-events body on blank lines and returns, in
    # order, the payload text (without the data prefix) of each block that starts with data.
    """Return the data payload of each SSE event, in order."""
    return [block[len("data: ") :] for block in body.split("\n\n") if block.startswith("data: ")]


@pytest.fixture(scope="class")
def router_env():
    # [FUNCTION DOC] HELPER: Class-scoped fixture. Patches the mock backend's chat-completion
    # computation so each choice carries a meta-info block (output token logprobs pairs built from
    # the logprob content, the completion token count, fake routed-experts and indexer-top-k
    # replay payloads), starts a mock backend whose generation echoes the prompt, builds a
    # session server config (Qwen3 0.6B checkpoint, thinking disabled in the chat-template kwargs,
    # default TITO model, fixed instance id, retract pause mode), serves it with uvicorn in a
    # thread on a free port starting at 31000, and yields a namespace with the server url and the
    # backend. Stops the server on teardown; the patch is active during the whole fixture.
    """Create a standalone SessionServer with session routes and a mock backend."""

    def process_fn(prompt: str) -> ProcessResult:
        # [FUNCTION DOC] HELPER: Mock generation callback returning the prompt prefixed by echo
        # with a stop finish reason.
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    original_chat_response = MockSGLangServer._compute_chat_completions_response

    def patched_chat_response(self, payload: dict) -> dict:
        # [FUNCTION DOC] HELPER: Calls the original computation, then sets the first choice's meta
        # info (logprob/token-id pairs from the logprobs content, completion token count, and fixed
        # replay payloads). Returns the modified response.
        response = original_chat_response(self, payload)
        choice = response["choices"][0]
        logprobs_content = choice["logprobs"]["content"]
        output_token_logprobs = [
            (item["logprob"], self.tokenizer.convert_tokens_to_ids(item["token"])) for item in logprobs_content
        ]
        choice["meta_info"] = {
            "output_token_logprobs": output_token_logprobs,
            "completion_tokens": len(output_token_logprobs),
            # Replay payloads stay in the record along with the rest of meta_info.
            "routed_experts": [[0, 1], [2, 3]],
            "indexer_topk": [[4], [5]],
        }
        return response

    with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response):
        with with_mock_server(process_fn=process_fn) as backend:
            config = make_session_server_config(
                backend_url=backend.url,
                hf_checkpoint="Qwen/Qwen3-0.6B",
                apply_chat_template_kwargs={"enable_thinking": False},
                tito_model="default",
                instance_id=_INSTANCE_ID,
                pause_generation_mode="retract",
            )
            server_obj = SessionServer(config)

            port = find_available_port(31000)
            server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
            server.start()

            url = f"http://127.0.0.1:{port}"

            try:
                yield SimpleNamespace(url=url, backend=backend)
            finally:
                server.stop()


class TestSessionRoutes:
    # [CLASS DOC] Tests for the health and session CRUD routes of the running server.
    def test_health_reports_stable_instance_id(self, router_env):
        # [FUNCTION DOC] Two health requests both return 200 with status ok and the same
        # configured instance id.
        first = requests.get(f"{router_env.url}/health", timeout=5.0)
        second = requests.get(f"{router_env.url}/health", timeout=5.0)

        assert first.status_code == 200
        assert second.status_code == 200
        first_body = first.json()
        second_body = second.json()
        assert first_body["status"] == "ok"
        assert second_body["status"] == "ok"
        assert first_body["session_server_instance_id"] == _INSTANCE_ID
        assert second_body["session_server_instance_id"] == first_body["session_server_instance_id"]

    def test_create_session(self, router_env):
        # [FUNCTION DOC] Creating a session returns 200 with a 32-character session id.
        response = requests.post(f"{router_env.url}/sessions", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 32

    def test_get_session_initial_state(self, router_env):
        # [FUNCTION DOC] Fetching a new session returns 200 with its id and an empty records list.
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        get_resp = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["session_id"] == session_id
        assert data["records"] == []

    def test_get_session_not_found(self, router_env):
        # [FUNCTION DOC] Fetching an unknown session returns 404 with the error text "session not
        # found: session_id=nonexistent".
        response = requests.get(f"{router_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found: session_id=nonexistent"

    def test_delete_session(self, router_env):
        # [FUNCTION DOC] Deleting an existing session returns 204 with an empty body; deleting it
        # again returns 404.
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        delete_resp = requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        assert delete_resp.status_code == 204
        assert delete_resp.text == ""

        assert requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0).status_code == 404

    def test_delete_session_not_found(self, router_env):
        # [FUNCTION DOC] Deleting an unknown session returns 404 with the same "session not
        # found" error text.
        response = requests.delete(f"{router_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found: session_id=nonexistent"


class TestSessionProxy:
    # [CLASS DOC] Tests for the chat-completions proxy route: record keeping, response shaping,
    # header hygiene, error mapping and template kwarg overrides.
    def test_proxy_chat_appends_record(self, router_env):
        # [FUNCTION DOC] A chat request with return-logprob returns 200 with non-empty choices;
        # fetching the session then shows exactly one record with the chat path and status 200.
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        payload = {
            "messages": [{"role": "user", "content": "What is 1+2?"}],
            "return_logprob": True,
        }
        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "choices" in body
        assert body["choices"]

        get_resp = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        records = get_resp.json()["records"]

        assert isinstance(records, list)
        assert len(records) == 1
        record = records[0]
        assert record["path"] == "/v1/chat/completions"
        assert record["status_code"] == 200

    def test_proxy_chat_postprocesses_completion_once(self, router_env, monkeypatch):
        # [FUNCTION DOC] Wraps the TITO tokenizer's completion post-processing hook with a
        # recorder (restored by monkeypatch), sends one chat request, and asserts a 200 response,
        # exactly one hook call, that the choice's message is the same object as the assistant
        # message passed, and that completion token ids are non-empty.
        calls = []
        original = TITOTokenizer.postprocess_completion

        def tracked_postprocess(self, *, choice, assistant_message, completion_token_ids):
            # [FUNCTION DOC] HELPER: Records the three keyword arguments in the shared call list
            # and delegates to the original hook, returning its result.
            calls.append((choice, assistant_message, completion_token_ids))
            return original(
                self,
                choice=choice,
                assistant_message=assistant_message,
                completion_token_ids=completion_token_ids,
            )

        monkeypatch.setattr(TITOTokenizer, "postprocess_completion", tracked_postprocess)
        session_id = _create_session(router_env.url)

        response = _post_chat(
            router_env.url,
            session_id,
            {"messages": [{"role": "user", "content": "hook once"}]},
        )

        assert response.status_code == 200
        assert len(calls) == 1
        choice, assistant_message, completion_token_ids = calls[0]
        assert choice["message"] is assistant_message
        assert completion_token_ids

    def test_proxy_chat_response_has_no_duplicate_server_or_date_header(self, router_env):
        # [FUNCTION DOC] Sends a hand-built HTTP/1.1 chat request over a raw socket (close
        # connection, 10 second timeout), reads until the end of the response headers, and asserts
        # a 200 status line and that the server and date headers each appear exactly once
        # (upstream copies must not be echoed). Raises an assertion if the connection closes early.
        # Both the backend and this server run under uvicorn, so each emits its own
        # server/date. Echoing upstream's copy puts two of each on the wire, and
        # aiohttp (the transport litellm uses) then refuses to read the body at all.
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]
        host, port = router_env.url.removeprefix("http://").split(":")
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "return_logprob": True})
        request = (
            f"POST /sessions/{session_id}/v1/chat/completions HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
            f"{body}"
        )

        with socket.create_connection((host, int(port)), timeout=10.0) as sock:
            sock.sendall(request.encode())
            raw = b""
            while b"\r\n\r\n" not in raw:
                chunk = sock.recv(4096)
                assert chunk, "connection closed before response headers were complete"
                raw += chunk

        head = raw.split(b"\r\n\r\n", 1)[0].decode()
        assert head.splitlines()[0].endswith("200 OK")
        names = [line.split(":", 1)[0].lower() for line in head.splitlines()[1:]]
        assert names.count("server") == 1
        assert names.count("date") == 1

    def test_chat_malformed_json_body_returns_400(self, router_env):
        # [FUNCTION DOC] A non-JSON body returns 400 with an error starting "invalid JSON body:".
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
            timeout=10.0,
        )
        assert resp.status_code == 400
        assert resp.json()["error"].startswith("invalid JSON body:")

    def test_chat_template_kwargs_override_reaches_render_and_backend(self, router_env):
        # [FUNCTION DOC] One request with no kwargs shows the backend receives the launch default
        # (thinking disabled); a second with thinking enabled shows the backend receives the
        # override and its input ids differ from the first.
        """A request response-mode kwarg wins over the launch default in both
        the locally rendered input_ids and the outbound backend request."""
        default_session = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]
        resp = requests.post(
            f"{router_env.url}/sessions/{default_session}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            timeout=10.0,
        )
        assert resp.status_code == 200
        default_payload = router_env.backend.request_log[-1]
        assert default_payload["chat_template_kwargs"] == {"enable_thinking": False}

        override_session = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]
        resp = requests.post(
            f"{router_env.url}/sessions/{override_session}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": True},
            },
            timeout=10.0,
        )
        assert resp.status_code == 200
        override_payload = router_env.backend.request_log[-1]
        assert override_payload["chat_template_kwargs"] == {"enable_thinking": True}
        assert override_payload["input_ids"] != default_payload["input_ids"]

    def test_chat_upstream_null_message_returns_502(self, router_env):
        # [FUNCTION DOC] With the backend patched to return a null message in the first choice, a
        # chat request returns 502 whose error contains "assistant message content is None".
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        fixture_response = MockSGLangServer._compute_chat_completions_response

        def null_message_response(self, payload: dict) -> dict:
            # [FUNCTION DOC] HELPER: Computes the normal response, replaces the first choice's
            # message with null and returns it.
            response = fixture_response(self, payload)
            response["choices"][0]["message"] = None
            return response

        with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=null_message_response):
            resp = requests.post(
                f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                timeout=10.0,
            )
        assert resp.status_code == 502
        assert "assistant message content is None" in resp.json()["error"]

    def test_chat_response_omits_meta_info_but_record_keeps_it(self, router_env):
        # [FUNCTION DOC] With logprobs and return-meta-info requested, the client choice has no
        # meta info, while the stored record's choice keeps the routed-experts, indexer-top-k and
        # output-logprob payloads with a consistent completion token count. The client message,
        # logprobs and usage equal those in the record.
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        payload = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "logprobs": True,
            "return_meta_info": True,
        }
        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200
        client_response = resp.json()
        client_choice = client_response["choices"][0]
        assert "meta_info" not in client_choice

        record = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"][0]
        record_meta = record["response"]["choices"][0]["meta_info"]
        assert record_meta["routed_experts"] == [[0, 1], [2, 3]]
        assert record_meta["indexer_topk"] == [[4], [5]]
        assert record_meta["output_token_logprobs"]
        assert record_meta["completion_tokens"] == len(record_meta["output_token_logprobs"])
        assert client_choice["message"] == record["response"]["choices"][0]["message"]
        assert client_choice["logprobs"] == record["response"]["choices"][0]["logprobs"]
        assert client_response["usage"] == record["response"]["usage"]

    @pytest.mark.parametrize(
        "client_flags, expect_logprobs",
        [({}, False), ({"logprobs": True}, True), ({"logprobs": False}, False)],
        ids=["omitted", "asked", "explicit-false"],
    )
    def test_chat_response_carries_logprobs_only_when_the_client_asked(
        self, router_env, client_flags, expect_logprobs
    ):
        # [FUNCTION DOC] Parametrized over omitted, true and false logprobs flags: the client
        # response carries logprobs only when the flag was true, while the stored record always
        # keeps non-empty logprobs content.
        session_id = _create_session(router_env.url)

        resp = _post_chat(
            router_env.url, session_id, {"messages": [{"role": "user", "content": "What is 3+3?"}], **client_flags}
        )

        assert resp.status_code == 200
        assert ("logprobs" in resp.json()["choices"][0]) is expect_logprobs
        # The record keeps logprobs for TITO and training, whatever the client asked for.
        record = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"][0]
        assert record["response"]["choices"][0]["logprobs"]["content"]


class TestChatFakeStreaming:
    # [CLASS DOC] Tests for the fake-streaming mode. The class constant MESSAGES is a single user
    # question shared by the tests.
    """``stream: true`` is fake streaming: non-streaming backend call, single SSE chunk."""

    MESSAGES = [{"role": "user", "content": "What is 1+2?"}]

    def test_stream_single_chunk_matches_non_stream(self, router_env):
        # [FUNCTION DOC] Sends the same request once non-streaming and once streaming with usage
        # included. Asserts a 200 event-stream response of exactly two events (one chunk then the
        # done marker); the chunk is a chat completion chunk from the mock model whose single
        # choice has an assistant delta with the same content and a stop finish reason as the
        # non-streamed answer, with no meta info or replay payload text anywhere. The backend
        # request carried no stream flags, and the stored records agree on request, message and
        # meta info.
        non_stream_sid = _create_session(router_env.url)
        stream_sid = _create_session(router_env.url)

        non_stream = _post_chat(router_env.url, non_stream_sid, {"messages": self.MESSAGES}).json()

        resp = _post_chat(
            router_env.url,
            stream_sid,
            {"messages": self.MESSAGES, "stream": True, "stream_options": {"include_usage": True}},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.text)
        assert len(events) == 2
        assert events[1] == "[DONE]"

        chunk = json.loads(events[0])
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "mock-model"
        [chunk_choice] = chunk["choices"]
        expected_choice = non_stream["choices"][0]
        assert chunk_choice["delta"]["role"] == "assistant"
        assert chunk_choice["delta"]["content"] == expected_choice["message"]["content"]
        assert chunk_choice["finish_reason"] == expected_choice["finish_reason"] == "stop"
        # Training payloads never reach the chunk; they live on the session record.
        assert "meta_info" not in chunk_choice
        assert "routed_experts" not in resp.text
        assert "indexer_topk" not in resp.text

        # Neither the backend nor the record sees the stream flags.
        backend_payload = router_env.backend.request_log[-1]
        assert "stream" not in backend_payload
        assert "stream_options" not in backend_payload

        stream_record = requests.get(f"{router_env.url}/sessions/{stream_sid}", timeout=5.0).json()["records"][0]
        non_stream_record = requests.get(f"{router_env.url}/sessions/{non_stream_sid}", timeout=5.0).json()["records"][
            0
        ]
        assert stream_record["request"] == non_stream_record["request"]
        assert (
            stream_record["response"]["choices"][0]["message"]
            == non_stream_record["response"]["choices"][0]["message"]
        )
        assert (
            stream_record["response"]["choices"][0]["meta_info"]
            == non_stream_record["response"]["choices"][0]["meta_info"]
        )

    def test_stream_tool_calls_single_chunk_with_index(self, router_env):
        # [FUNCTION DOC] With the backend patched to emit two tool calls (weather and time) and two
        # matching tools declared, a streaming request yields one chunk with finish reason tool
        # calls whose delta lists two tool calls indexed 0 and 1 in order, with arguments that are
        # valid JSON.
        session_id = _create_session(router_env.url)

        def tool_call_process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Ignores the prompt and returns generated text containing two
            # tool-call blocks.
            return ProcessResult(
                text=(
                    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>\n<tool_call>\n{"name": "get_time", "arguments": {"timezone": "UTC"}}\n</tool_call>'
                )
            )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}},
                },
            },
        ]

        with patch.object(router_env.backend, "process_fn", new=tool_call_process_fn):
            resp = _post_chat(router_env.url, session_id, {"messages": self.MESSAGES, "tools": tools, "stream": True})

        assert resp.status_code == 200
        chunk = json.loads(_parse_sse(resp.text)[0])
        [chunk_choice] = chunk["choices"]
        assert chunk_choice["finish_reason"] == "tool_calls"
        tool_calls = chunk_choice["delta"]["tool_calls"]
        assert [tool_call["index"] for tool_call in tool_calls] == [0, 1]
        assert [tool_call["function"]["name"] for tool_call in tool_calls] == ["get_weather", "get_time"]
        for tool_call in tool_calls:
            json.loads(tool_call["function"]["arguments"])

    def test_stream_passes_through_usage_and_length_finish_reason(self, router_env):
        # [FUNCTION DOC] With the backend patched to report a length finish reason and fixed usage
        # counts (5, 7, 12), the streamed chunk carries that finish reason and exactly that usage.
        session_id = _create_session(router_env.url)
        fixture_response = MockSGLangServer._compute_chat_completions_response

        def length_with_usage(self, payload: dict) -> dict:
            # [FUNCTION DOC] HELPER: Computes the normal response, sets the first choice's finish
            # reason to length and the usage to fixed counts, returns it.
            response = fixture_response(self, payload)
            response["choices"][0]["finish_reason"] = "length"
            response["usage"] = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
            return response

        with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=length_with_usage):
            resp = _post_chat(router_env.url, session_id, {"messages": self.MESSAGES, "stream": True})

        assert resp.status_code == 200
        chunk = json.loads(_parse_sse(resp.text)[0])
        assert chunk["choices"][0]["finish_reason"] == "length"
        assert chunk["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}

    def test_stream_backend_error_passes_through_json(self, router_env):
        # [FUNCTION DOC] With the backend patched to answer HTTP 400 "context too long", a
        # streaming request gets 400, a JSON content type (not event-stream), the same error body,
        # and no record is stored.
        session_id = _create_session(router_env.url)

        async def reject(self, request, compute_fn):
            # [FUNCTION DOC] HELPER: Async stand-in for the mock backend's request handler; ignores
            # its inputs and returns a 400 JSON response with the error "context too long".
            return JSONResponse(content={"error": "context too long"}, status_code=400)

        with patch.object(MockSGLangServer, "_handle_generate_like_request", new=reject):
            resp = _post_chat(router_env.url, session_id, {"messages": self.MESSAGES, "stream": True})

        assert resp.status_code == 400
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json() == {"error": "context too long"}
        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert records == []

    def test_stream_false_keeps_json_response(self, router_env):
        # [FUNCTION DOC] An explicit stream false returns 200 with a JSON content type and
        # non-empty message content.
        session_id = _create_session(router_env.url)
        resp = _post_chat(router_env.url, session_id, {"messages": self.MESSAGES, "stream": False})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["choices"][0]["message"]["content"]

    def test_streaming_client_multi_turn_passes_prefix_check(self, router_env):
        # [FUNCTION DOC] Uses the streaming test client for two turns: the second turn sends the
        # first turn's rebuilt assistant message plus a tool result. Asserts both answers have
        # content and the session stores two records (the second turn extended, not rolled back).
        """The rebuilt assistant message must survive the next turn's TITO prefix check."""
        session_id = _create_session(router_env.url)
        url = f"{router_env.url}/sessions/{session_id}/v1/chat/completions"

        async def run():
            # [FUNCTION DOC] HELPER: Async: with one HTTP client (10 second timeout) streams turn
            # one, builds turn two's messages (original, the rebuilt assistant, a tool message with
            # content ok), streams turn two, and returns both responses.
            async with httpx.AsyncClient(timeout=10) as client:
                first = await stream_chat_completions(client, url, {"messages": self.MESSAGES}, label="turn 1")
                messages = [
                    *self.MESSAGES,
                    first["choices"][0]["message"],
                    {"role": "tool", "content": "ok", "tool_call_id": "t0"},
                ]
                second = await stream_chat_completions(client, url, {"messages": messages}, label="turn 2")
                return first, second

        first, second = asyncio.run(run())
        assert first["choices"][0]["message"]["content"]
        assert second["choices"][0]["message"]["content"]
        # Turn 2 extended (not rolled back) the session: both records are kept.
        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert len(records) == 2

    def test_streaming_client_rebuilt_tool_calls_match_stored(self, router_env):
        # [FUNCTION DOC] With the backend emitting two tool calls and two tools declared, streams a
        # tool turn and compares the client-rebuilt message with the stored one: stored tool calls
        # all carry an index, rebuilt ones do not, the strict message matcher still treats them
        # equal, and the tool-call function parts and ids are identical.
        """Rebuilt tool_calls must match the stored assistant under strict_message_matches.

        The stored message keeps SGLang's wire shape (tool_calls carry
        ``index``); a protocol-faithful streaming client drops that
        streaming-only key when rebuilding.  The server's own comparison
        (``strict_message_matches``) must treat the two as equal — dict equality is
        deliberately NOT the contract."""
        session_id = _create_session(router_env.url)
        url = f"{router_env.url}/sessions/{session_id}/v1/chat/completions"

        def tool_call_process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Ignores the prompt and returns text with two tool-call blocks.
            return ProcessResult(
                text=(
                    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>\n'
                    '<tool_call>\n{"name": "get_time", "arguments": {"timezone": "UTC"}}\n</tool_call>'
                )
            )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}},
                },
            },
        ]

        async def run():
            # [FUNCTION DOC] HELPER: Async: streams one tool-declaring chat turn with a single HTTP
            # client (10 second timeout) and returns the rebuilt response.
            async with httpx.AsyncClient(timeout=10) as client:
                return await stream_chat_completions(
                    client, url, {"messages": self.MESSAGES, "tools": tools}, label="tool turn"
                )

        with patch.object(router_env.backend, "process_fn", new=tool_call_process_fn):
            response = asyncio.run(run())

        rebuilt_message = response["choices"][0]["message"]
        record = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"][0]
        stored_message = record["response"]["choices"][0]["message"]
        # Mock fidelity guard: the stored wire shape carries index, the rebuilt drops it.
        assert all("index" in tool_call for tool_call in stored_message["tool_calls"])
        assert all("index" not in tool_call for tool_call in rebuilt_message["tool_calls"])
        assert strict_message_matches(stored_message, rebuilt_message)
        # Template-relevant substance survives the round-trip exactly.
        assert [tool_call["function"] for tool_call in rebuilt_message["tool_calls"]] == [
            tool_call["function"] for tool_call in stored_message["tool_calls"]
        ]
        assert [tool_call["id"] for tool_call in rebuilt_message["tool_calls"]] == [
            tool_call["id"] for tool_call in stored_message["tool_calls"]
        ]

    def test_streaming_client_tool_call_turn_then_tool_result_continues(self, router_env):
        # [FUNCTION DOC] Streams a tool-call turn, then a second turn with the rebuilt assistant
        # and a tool result containing a sentinel. The backend answers with a tool call unless the
        # sentinel is in the prompt, then with a final text. Asserts the first finish reason is
        # tool calls, the second content is the final text, and two records are stored.
        """Regression: a streamed tool-call turn must not roll back the next turn.

        This is the CPU incarnation of the stage-c multi-role e2e failure: the
        stored assistant carries SGLang-shaped tool_calls (with ``index``), the
        client replays the SSE-rebuilt message (without it), and the follow-up
        tool_result request must extend the session instead of dying with
        HTTP 400 ``rollback failed``."""
        session_id = _create_session(router_env.url)
        url = f"{router_env.url}/sessions/{session_id}/v1/chat/completions"

        def tool_then_text_process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Returns the final answer text when the prompt contains the
            # tool-result sentinel, otherwise a single weather tool call for Beijing.
            if "TOOL_SENTINEL_42" in prompt:
                return ProcessResult(text="Final answer after tool.")
            return ProcessResult(
                text='<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>'
            )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            },
        ]

        async def run():
            # [FUNCTION DOC] HELPER: Async: with one HTTP client streams the tool turn, builds the
            # follow-up (original messages, rebuilt assistant, tool message with the sentinel and
            # the first call id), streams it, and returns both responses.
            async with httpx.AsyncClient(timeout=10) as client:
                first = await stream_chat_completions(
                    client, url, {"messages": self.MESSAGES, "tools": tools}, label="tool turn"
                )
                assistant = first["choices"][0]["message"]
                messages = [
                    *self.MESSAGES,
                    assistant,
                    {
                        "role": "tool",
                        "content": "TOOL_SENTINEL_42",
                        "tool_call_id": assistant["tool_calls"][0]["id"],
                    },
                ]
                second = await stream_chat_completions(
                    client, url, {"messages": messages, "tools": tools}, label="tool_result turn"
                )
                return first, second

        with patch.object(router_env.backend, "process_fn", new=tool_then_text_process_fn):
            first, second = asyncio.run(run())

        assert first["choices"][0]["finish_reason"] == "tool_calls"
        assert second["choices"][0]["message"]["content"] == "Final answer after tool."
        # Turn 2 extended (not rolled back) the session: both records are kept.
        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert len(records) == 2

    def test_openai_sdk_accumulates_fake_stream(self, router_env):
        # [FUNCTION DOC] Skipped if the openai package is missing. Gets the expected content from a
        # non-streaming call, then uses the SDK client (no retries) in streaming mode against the
        # session base url, accumulating delta contents and the last finish reason; asserts the
        # content equals the expected one and the finish reason is stop.
        openai = pytest.importorskip("openai")

        non_stream_sid = _create_session(router_env.url)
        stream_sid = _create_session(router_env.url)

        expected = _post_chat(router_env.url, non_stream_sid, {"messages": self.MESSAGES}).json()
        expected_content = expected["choices"][0]["message"]["content"]

        client = openai.OpenAI(
            base_url=f"{router_env.url}/sessions/{stream_sid}/v1", api_key="not-used", max_retries=0
        )
        stream = client.chat.completions.create(model="mock-model", messages=self.MESSAGES, stream=True)
        content = ""
        finish_reason = None
        for chunk in stream:
            [choice] = chunk.choices
            if choice.delta.content:
                content += choice.delta.content
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        assert content == expected_content
        assert finish_reason == "stop"


# ── additional R3 (non-retract pause modes): derivation and request offsets ──


@contextmanager
def _serve_router(extra_args: dict | None = None):
    # [FUNCTION DOC] HELPER: Context manager that starts a mock backend (generation returns "ok"
    # with stop) and a dedicated v1 session server over it: defaults are the Qwen3 0.6B
    # checkpoint, thinking disabled, default TITO model and retract pause mode, overridden by the
    # optional extra_args mapping; 30 second timeout and a random instance id. Serves it with
    # uvicorn in a thread on a free port from 31000, yields a namespace with url and backend, and
    # stops the server on exit.
    """A dedicated v1 SessionServer for tests that flip args router_env's
    class-scoped fixture pins (routing replay, pause_generation_mode)."""

    def process_fn(prompt: str) -> ProcessResult:
        # [FUNCTION DOC] HELPER: Mock generation callback returning "ok" with a stop finish reason.
        return ProcessResult(text="ok", finish_reason="stop")

    with with_mock_server(process_fn=process_fn) as backend:
        defaults = {
            "hf_checkpoint": "Qwen/Qwen3-0.6B",
            "apply_chat_template_kwargs": {"enable_thinking": False},
            "tito_model": "default",
            "pause_generation_mode": "retract",
        }
        config = make_session_server_config(
            backend_url=backend.url,
            timeout=30,
            instance_id=uuid.uuid4().hex,
            **(defaults | (extra_args or {})),
        )
        server_obj = SessionServer(config)
        port = find_available_port(31000)
        server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
        server.start()
        try:
            yield SimpleNamespace(url=f"http://127.0.0.1:{port}", backend=backend)
        finally:
            server.stop()


class TestUseAdditionR3Derivation:
    # [CLASS DOC] Tests the derivation of the additional routing-replay flag from the pause mode.
    """use_addition_r3 is derived once at server bootstrap from
    pause_generation_mode; it is not independently configurable."""

    @pytest.mark.parametrize(("mode", "expected"), [("abort", True), ("in_place", True), ("retract", False)])
    def test_mode_mapping(self, mode, expected):
        # [FUNCTION DOC] Builds a server config with no checkpoint and the given pause mode; the
        # server's addition-r3 flag is true for abort and in-place and false for retract.
        config = make_session_server_config(hf_checkpoint=None, pause_generation_mode=mode)
        assert SessionServer(config).use_addition_r3 is expected


class TestAdditionR3RequestOffset:
    # [CLASS DOC] Tests the routed-experts start-length offset the server sends to the backend in
    # non-retract pause modes. The class constant MESSAGES is a one-message user prompt.
    MESSAGES = [{"role": "user", "content": "hi"}]

    def _accumulated(self, url: str, session_id: str) -> list[int]:
        # [FUNCTION DOC] HELPER: Fetches the session over HTTP and returns the accumulated token
        # ids list from its metadata.
        data = requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()
        return data["metadata"]["accumulated_token_ids"]

    def _turn2_messages(self, first_response: dict, tool_content: str) -> list[dict]:
        # [FUNCTION DOC] HELPER: Builds the second-turn message list: the original messages, the
        # first response's assistant message, and a tool message with the given content and the
        # call id t0.
        return [
            *self.MESSAGES,
            first_response["choices"][0]["message"],
            {"role": "tool", "content": tool_content, "tool_call_id": "t0"},
        ]

    @pytest.mark.parametrize("mode", ["abort", "in_place"])
    def test_incremental_offsets_across_turns_and_rollback(self, mode):
        # [FUNCTION DOC] For abort and in-place modes with routing replay on: turn one's backend
        # request asks for routed experts with start length 0; turn two's start length is the
        # first checkpoint's accumulated length minus one, and the accumulated ids grow; a retry of
        # turn two with a different tool result rolls back so the start length returns to the
        # turn-one value. The stored records carry exactly the offsets used, zero then that value.
        with _serve_router({"use_rollout_routing_replay": True, "pause_generation_mode": mode}) as env:
            session_id = _create_session(env.url)

            first = _post_chat(env.url, session_id, {"messages": self.MESSAGES})
            assert first.status_code == 200
            turn1_body = env.backend.request_log[-1]
            assert turn1_body["return_routed_experts"] is True
            assert turn1_body["routed_experts_start_len"] == 0
            checkpoint1 = self._accumulated(env.url, session_id)

            second = _post_chat(env.url, session_id, {"messages": self._turn2_messages(first.json(), "ok")})
            assert second.status_code == 200
            turn2_body = env.backend.request_log[-1]
            # The turn-1 checkpoint's prefix is stable, so only its last row is new.
            assert turn2_body["routed_experts_start_len"] == len(checkpoint1) - 1
            checkpoint2 = self._accumulated(env.url, session_id)
            assert len(checkpoint2) > len(checkpoint1)

            # Retrying turn 2 rolls back to checkpoint 1: the offset moves backward.
            retry = _post_chat(env.url, session_id, {"messages": self._turn2_messages(first.json(), "different")})
            assert retry.status_code == 200
            retry_body = env.backend.request_log[-1]
            assert retry_body["routed_experts_start_len"] == len(checkpoint1) - 1
            assert len(checkpoint1) - 1 < len(checkpoint2) - 1

            # The exact offset each successful turn used is persisted on its record.
            records = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
            assert [r["request"]["routed_experts_start_len"] for r in records] == [0, len(checkpoint1) - 1]

    def test_retract_request_has_no_start_len(self):
        # [FUNCTION DOC] In retract mode with replay on, the backend request asks for routed
        # experts but includes no start-length field.
        with _serve_router({"use_rollout_routing_replay": True, "pause_generation_mode": "retract"}) as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": self.MESSAGES}).status_code == 200
            body = env.backend.request_log[-1]
            assert body["return_routed_experts"] is True
            assert "routed_experts_start_len" not in body

    def test_in_place_without_replay_sends_false_flag_and_no_start_len(self):
        # [FUNCTION DOC] In in-place mode without replay, the backend request has the
        # return-routed-experts flag explicitly false and no start-length field.
        with _serve_router({"pause_generation_mode": "in_place"}) as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": self.MESSAGES}).status_code == 200
            body = env.backend.request_log[-1]
            # The launch flag is authoritative: off means an explicit False, never a client-provided value.
            assert body["return_routed_experts"] is False
            assert "routed_experts_start_len" not in body
