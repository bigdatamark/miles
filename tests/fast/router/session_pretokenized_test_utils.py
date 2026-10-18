# [MODULE DOC] HELPER: Shared helpers for session-server pretokenized end-to-end tests. Provides a
# scripted fake chat-completion backend (served by a real HTTP server on a background thread) that
# replays predetermined assistant turns while rendering prompts with a real chat template, a function
# that starts a session server pointed at such a backend and returns a handle with its URL, a
# teardown function, a function to fetch a session payload over HTTP, and helpers that compare a
# session's accumulated token ids against a locally rendered chat template and pick out forbidden
# mismatch kinds. Side effects: opens local network ports (searching from 31000 for the session
# server and 32000 for the backend), starts threads, loads tokenizers.
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.server import SessionServer
from miles.utils.chat_template_utils import MismatchType, apply_chat_template, get_tito_tokenizer
from miles.utils.http_utils import find_available_port
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

FORBIDDEN_MISMATCH_TYPES: frozenset[str] = frozenset(
    {
        MismatchType.SPECIAL_TOKEN_COUNT.value,
        MismatchType.SPECIAL_TOKEN_TYPE.value,
        MismatchType.NON_ASSISTANT_TEXT.value,
    }
)


@dataclass(frozen=True)
class ScriptedBackendTurn:
    # [CLASS DOC] HELPER: Immutable record of one scripted backend turn: the assistant message
    # returned to the client in the completion response, and the message used to render the
    # expected response text (it may differ from the returned message, for example in how tool
    # calls are shaped for the chat template).
    response_message: dict[str, Any]
    render_message: dict[str, Any]


def load_test_tokenizer(hf_checkpoint: str, chat_template_path: str | None):
    # [FUNCTION DOC] HELPER: Loads the tokenizer for the checkpoint, optionally overriding the chat
    # template from the given file path, always allowing remote code. Returns the tokenizer; may read
    # from disk or download from the model hub.
    return load_tokenizer(
        hf_checkpoint,
        chat_template_path=chat_template_path,
        trust_remote_code=True,
    )


def make_router_env(
    backend,
    *,
    hf_checkpoint: str,
    chat_template_path: str | None,
    tito_model: str,
):
    # [FUNCTION DOC] HELPER: Builds a session server configuration pointing at the backend's URL,
    # with the given checkpoint, chat template path and token-in-token-out model name; version 2 of
    # the session server, no routing replay, no speculative decoding, retract pause mode, the
    # drop-same-prompt-retries sample picker and the default postprocessor. Creates a session server
    # from it, finds a free port starting at 31000, and starts the server's application on a
    # background thread bound to the loopback address. Returns a namespace with the server base URL,
    # the backend and the server thread handle. The backend is expected to be started by the caller.
    config = make_session_server_config(
        backend_url=backend.url,
        hf_checkpoint=hf_checkpoint,
        chat_template_path=chat_template_path,
        tito_model=tito_model,
        use_session_server="v2",
        use_rollout_routing_replay=False,
        sglang_speculative_algorithm=None,
        pause_generation_mode="retract",
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    session_server = SessionServer(config)

    port = find_available_port(31000)
    server = UvicornThreadServer(session_server.app, host="127.0.0.1", port=port)
    server.start()

    return SimpleNamespace(
        url=f"http://127.0.0.1:{port}",
        backend=backend,
        server=server,
    )


def teardown_router_env(env) -> None:
    # [FUNCTION DOC] HELPER: Stops the session server thread, then the backend, of an environment
    # made by the router environment builder.
    env.server.stop()
    env.backend.stop()


def fetch_session_payload(base_url: str, session_id: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Issues an HTTP GET to the sessions path for the session id under the
    # base URL with a 5 second timeout, raises on a non-success status, and returns the decoded JSON
    # body as a mapping.
    response = requests.get(f"{base_url}/sessions/{session_id}", timeout=5.0)
    response.raise_for_status()
    return response.json()


def compute_local_session_mismatch(
    tokenizer,
    *,
    tito_model: str,
    messages: list[dict[str, Any]],
    accumulated_token_ids: list[int],
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Recomputes, locally, what the session's accumulated token ids should be.
    # Builds the token-in-token-out tokenizer for the model name and its comparator, renders the
    # messages (with the tools, without a generation prompt) into token ids with the chat template,
    # compares those expected ids against the accumulated ids, and returns each mismatch converted to
    # a plain mapping (empty list when identical).
    comparator = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=tito_model,
    ).create_comparator()
    expected_ids = apply_chat_template(
        messages,
        tokenizer=tokenizer,
        tools=tools,
        add_generation_prompt=False,
        tokenize=True,
    )
    return [m.to_dict() for m in comparator.compare_sequences(expected_ids, accumulated_token_ids)]


def forbidden_mismatches(mismatch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Filters the mismatch mappings to those whose type is one of the
    # forbidden kinds (special token count, special token type, non-assistant text), preserving
    # order. A mismatch without a type is never forbidden.
    return [m for m in mismatch if m.get("type") in FORBIDDEN_MISMATCH_TYPES]


class ScriptedChatBackend:
    # [CLASS DOC] HELPER: Fake chat-completion backend served over HTTP from a background thread.
    # It exposes a health endpoint and a chat completions endpoint. The Nth completion request
    # (counting from 0) is answered with the Nth scripted turn: the response text is derived by
    # rendering the request messages with and without the scripted assistant message through the
    # real chat template, and its token ids and fake log-probabilities are included in the response.
    # More requests than scripted turns fail an assertion inside the handler. All requests are
    # logged for later inspection.
    def __init__(self, tokenizer, scripted_turns: list[ScriptedBackendTurn]):
        # [FUNCTION DOC] HELPER: Stores the tokenizer and scripted turns, zeroes the call counter,
        # empties the request log, picks the loopback host and a free port starting at 32000, creates
        # the web application and the background server (not started), and registers the routes.
        self.tokenizer = tokenizer
        self._scripted_turns = scripted_turns
        self._call_count = 0
        self.request_log: list[dict[str, Any]] = []
        self.host = "127.0.0.1"
        self.port = find_available_port(32000)
        self.app = FastAPI()
        self._server = UvicornThreadServer(self.app, host=self.host, port=self.port)
        self._setup_routes()

    @property
    def url(self) -> str:
        # [FUNCTION DOC] HELPER: Property: the backend's base HTTP URL (host and port).
        return f"http://{self.host}:{self.port}"

    def start(self):
        # [FUNCTION DOC] HELPER: Starts the background HTTP server thread.
        self._server.start()

    def stop(self):
        # [FUNCTION DOC] HELPER: Stops the background HTTP server thread.
        self._server.stop()

    def reset_stats(self):
        # [FUNCTION DOC] HELPER: Clears the request log and resets the call counter to zero, so the
        # scripted turns replay from the first.
        self.request_log.clear()
        self._call_count = 0

    def _setup_routes(self):
        # [FUNCTION DOC] HELPER: Registers the two endpoints on the application (see the class
        # description). Each handler is defined inline as a nested function.
        @self.app.get("/health")
        async def health():
            # [FUNCTION DOC] HELPER: Returns a JSON body with status "ok".
            return JSONResponse(content={"status": "ok"})

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            # [FUNCTION DOC] HELPER: Handles one completion request. Steps: parse the JSON body and
            # append it to the request log; take the next scripted turn (assertion failure "unexpected
            # extra request" if none left) and advance the counter; render the prompt text of the
            # request messages (with tools, generation prompt added); render the text of the messages
            # plus the turn's render message (no generation prompt) and assert it extends the prompt
            # text; the response text is the extension, and the output token ids are its encoding
            # without special tokens added; the prompt token ids are the request's explicit input ids
            # when present, otherwise the tokenized prompt rendering. Returns a chat completion JSON
            # with id scripted-N, the turn's response message, the prompt token ids, a finish reason
            # of tool calls when the response message has tool calls and stop otherwise, the
            # completion token count, and output token log-probabilities of minus index over 128
            # paired with each token id.
            payload = await request.json()
            self.request_log.append(payload)

            idx = self._call_count
            assert idx < len(self._scripted_turns), f"Unexpected extra request #{idx + 1}"
            self._call_count += 1

            turn = self._scripted_turns[idx]
            messages = payload["messages"]
            tools = payload.get("tools")

            prompt_text = apply_chat_template(
                messages,
                tokenizer=self.tokenizer,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
            with_assistant_text = apply_chat_template(
                messages + [turn.render_message],
                tokenizer=self.tokenizer,
                tools=tools,
                add_generation_prompt=False,
                tokenize=False,
            )
            assert with_assistant_text.startswith(prompt_text), "Scripted assistant must extend prompt text"

            response_text = with_assistant_text[len(prompt_text) :]
            output_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
            input_ids = payload.get("input_ids")
            prompt_ids = (
                list(input_ids)
                if input_ids is not None
                else apply_chat_template(
                    messages,
                    tokenizer=self.tokenizer,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )

            return JSONResponse(
                content={
                    "id": f"scripted-{idx}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "scripted-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": turn.response_message,
                            "prompt_token_ids": prompt_ids,
                            "finish_reason": "tool_calls" if turn.response_message.get("tool_calls") else "stop",
                            "meta_info": {
                                "completion_tokens": len(output_ids),
                                "output_token_logprobs": [[-i / 128, tid] for i, tid in enumerate(output_ids)],
                            },
                        }
                    ],
                }
            )
