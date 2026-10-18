# [MODULE DOC] HELPER: Custom agent function for the session-server tool-call end-to-end test. The
# agent drives a multi-turn conversation with a single mock weather tool through the session
# proxy's chat completions route, feeding back canned tool results, and stops when a final answer
# appears in answer tags, after 8 turns, or after more than 2 consecutive turns without a valid
# tool call. Module constants: the maximum turns and retries, the retry hint text sent as a tool
# message, the tool schema (one function taking a required city string), and four canned JSON
# weather results used cyclically. Correctness of token handling is enforced server side (a prefix
# mismatch causes an HTTP 500 that fails the request here); this module only generates traffic.
"""Custom agent function for the session-server tool-call e2e test.

Runs a multi-turn tool-calling conversation through the session proxy
and validates that the full TITO pretokenization + rollback pipeline
works end-to-end with real model inference.

Token-level correctness is guaranteed by the server-side prefix check
in ``LinearTrajectory.update_pretokenized_state`` —
if TITO produces wrong tokens, the server raises ``ValueError`` and
the request fails with HTTP 500, which surfaces here as an assertion
failure on the response status code.

When tool-call parsing fails, the agent randomly (50/50) chooses one
of two retry strategies:

1. **Rollback** — discard the failed assistant message and re-send the
   previous messages, triggering ``_detect_and_rollback`` on the server.
2. **Tool-message retry** — keep the failed assistant message and
   append a tool message asking the model to try again.

If no prior assistant checkpoint exists (``total_tool_calls == 0``),
strategy 2 is used unconditionally since there is nothing to roll back
to.  This exercises both retry code paths under real inference.

The agent is loaded at runtime by ``agentic_tool_call.generate`` via
``--custom-agent-function-path tests.e2e.sglang.utils.session_tool_agent.run_agent``.
"""

import logging
import random

import httpx

from miles.utils.test_utils.openai_stream_client import stream_chat_completions

logger = logging.getLogger(__name__)

MAX_TOOL_TURNS = 8
MAX_RETRIES = 2

RETRY_TOOL_MESSAGE = (
    "Your previous response did not include a valid tool call or a final answer. "
    "Please either call a tool or provide your answer in "
    "<final_answer>...</final_answer> tags."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. Beijing",
                    },
                },
                "required": ["location"],
            },
        },
    },
]

MOCK_TOOL_RESULTS = [
    '{"temperature_celsius": 22, "condition": "sunny", "humidity": 45}',
    '{"temperature_celsius": 15, "condition": "cloudy", "humidity": 70}',
    '{"temperature_celsius": 30, "condition": "rainy", "humidity": 90}',
    '{"temperature_celsius": 8, "condition": "snowy", "humidity": 85}',
]


def _is_task_complete(assistant_msg: dict) -> bool:
    # [FUNCTION DOC] HELPER: Returns true when the assistant message's content (empty if missing or
    # null) contains both the opening and closing final-answer tags.
    """Check if the assistant has produced a final answer."""
    content = assistant_msg.get("content") or ""
    return "<final_answer>" in content and "</final_answer>" in content


def _extract_tool_calls(assistant_msg: dict) -> list[dict] | None:
    # [FUNCTION DOC] HELPER: Returns the message's structured tool-call list if it is present and
    # non-empty, otherwise none. No text-based fallback parsing is attempted.
    """Return structured tool_calls from the assistant message, or None.

    Only trusts the structured ``tool_calls`` field populated by sglang's
    tool-call parser.  No fallback parsing — if the parser didn't produce
    structured output, the caller should treat it as a failed parse and
    retry (via rollback or tool message).
    """
    if assistant_msg.get("tool_calls"):
        return assistant_msg["tool_calls"]
    return None


async def _chat(
    client: httpx.AsyncClient,
    url: str,
    messages,
    rk,
    label="",
    tool_choice=None,
):
    # [FUNCTION DOC] HELPER: Sends one chat completions request to the base URL's chat route using
    # the given HTTP client. The payload is the messages, the module's tool schema, and the
    # request-kwargs mapping merged in (so a "tools" entry in it would override the schema), plus
    # the tool-choice if given. A "stream" entry in the request kwargs is removed from the payload
    # and read (default true): when true, delegates to the shared streaming client (which consumes
    # server-sent events and returns the assembled response); when false, posts normally, asserts
    # HTTP status 200 with the label in the message, and returns the parsed JSON.
    """Send a chat completions request and return the response JSON."""
    payload = {
        "messages": messages,
        "tools": TOOLS,
        **rk,
    }
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    # Streaming is the e2e default: black-box agent harnesses mostly consume
    # chat completions as SSE, so exercise the session server's fake-streaming
    # path unless the caller opts out with stream=False in request_kwargs.
    if payload.pop("stream", True):
        return await stream_chat_completions(client, f"{url}/v1/chat/completions", payload, label=label)
    resp = await client.post(f"{url}/v1/chat/completions", json=payload)
    assert resp.status_code == 200, f"{label} failed ({resp.status_code}): {resp.text}"
    return resp.json()


async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    # [FUNCTION DOC] Runs the agent loop and returns a mapping with counts: turns completed, total
    # tool calls, total rollbacks and total tool-message retries. Copies the prompt messages and
    # the request kwargs minus any tools entry. Uses one async HTTP client with a 180 second
    # timeout. For turn 1 through 8: send the conversation with automatic tool choice; append the
    # assistant reply; stop if it holds a final answer. If it has structured tool calls: append a
    # tool message per call with the canned result chosen cyclically by running call count and the
    # call's id, add to the count and reset the consecutive-failure counter. Otherwise increment
    # the consecutive-failure counter and stop if it exceeds 2; remove the failed assistant reply;
    # with probability one half, and only if at least one tool call has already happened, leave it
    # removed (a rollback: the next request resends earlier messages so the server detects and
    # rolls back); else restore the reply and append a tool message carrying the retry hint with a
    # deliberately invalid tool call id (some chat templates require an id field). Failed HTTP
    # statuses surface as assertion or client errors. Uses the global random generator, so runs are
    # not reproducible. Side effects: network calls, logging.
    # KNOWN ISSUE: the metadata and extra keyword parameters are accepted but unused.
    """Multi-turn tool-call agent for e2e session pretokenization testing.

    Exercises the full TITO pipeline including both retry strategies:
    on tool-call parse failure, randomly chooses rollback (discard
    assistant, trigger ``_detect_and_rollback``) or tool-message
    retry (append a retry prompt).  Falls back to tool-message retry
    when no prior checkpoint exists.

    Token-level correctness is validated server-side by the prefix check
    in ``update_pretokenized_state`` — any TITO mismatch causes HTTP 500.
    """
    messages = list(prompt)

    rk = {k: v for k, v in request_kwargs.items() if k not in ("tools",)}

    turns_completed = 0
    total_tool_calls = 0
    total_rollbacks = 0
    total_tool_retries = 0
    consecutive_retries = 0

    async with httpx.AsyncClient(timeout=180) as client:
        for turn in range(1, MAX_TOOL_TURNS + 1):
            label = f"Session Turn {turn}"
            resp_data = await _chat(
                client,
                base_url,
                messages,
                rk,
                label=label,
                tool_choice="auto",
            )

            turns_completed = turn

            assistant_msg = resp_data["choices"][0]["message"]
            logger.info(
                "Turn %d: content=%r, tool_calls=%s",
                turn,
                (assistant_msg.get("content") or "")[:80],
                "present" if assistant_msg.get("tool_calls") else "absent",
            )

            messages.append(assistant_msg)

            if _is_task_complete(assistant_msg):
                logger.info("Turn %d: task complete, ending loop", turn)
                break

            tool_calls = _extract_tool_calls(assistant_msg)
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    mock_idx = (total_tool_calls + i) % len(MOCK_TOOL_RESULTS)
                    messages.append(
                        {
                            "role": "tool",
                            "content": MOCK_TOOL_RESULTS[mock_idx],
                            "tool_call_id": tc["id"],
                        }
                    )
                total_tool_calls += len(tool_calls)
                logger.info(
                    "Turn %d: appended %d tool result(s), total tool calls so far: %d",
                    turn,
                    len(tool_calls),
                    total_tool_calls,
                )
                consecutive_retries = 0
            else:
                consecutive_retries += 1
                if consecutive_retries > MAX_RETRIES:
                    logger.warning("Turn %d: exceeded %d retries, ending loop", turn, MAX_RETRIES)
                    break

                use_rollback = random.random() < 0.5 and total_tool_calls > 0

                messages.pop()

                if use_rollback:
                    total_rollbacks += 1
                    logger.info(
                        "Turn %d: discarded failed assistant, " "will trigger session rollback (%d/%d)",
                        turn,
                        consecutive_retries,
                        MAX_RETRIES,
                    )
                else:
                    messages.append(assistant_msg)
                    # Add invalid tool call id field here. Otherwise template like kimi-k2 which explicitly
                    # insert tool call id might failed. This won't affect most of chat templates.
                    messages.append(
                        {"role": "tool", "content": RETRY_TOOL_MESSAGE, "tool_call_id": "invalid_tool_call"}
                    )
                    total_tool_retries += 1
                    logger.info(
                        "Turn %d: kept assistant + appended tool retry message (%d/%d)",
                        turn,
                        consecutive_retries,
                        MAX_RETRIES,
                    )

    logger.info(
        "Agent done: %d turns, %d tool_calls, %d rollbacks, %d tool_retries",
        turns_completed,
        total_tool_calls,
        total_rollbacks,
        total_tool_retries,
    )

    return {
        "turns_completed": turns_completed,
        "total_tool_calls": total_tool_calls,
        "total_rollbacks": total_rollbacks,
        "total_tool_retries": total_tool_retries,
    }
