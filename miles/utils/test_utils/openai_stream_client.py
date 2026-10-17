# [MODULE DOC] HELPER: Client-side helpers to consume an OpenAI-style streaming chat completions
# response (server-sent events over HTTP) and rebuild the equivalent non-streaming response. Two
# pieces: a pure accumulator that folds chunk objects into one response, and an async function that
# posts a request with streaming enabled through an httpx async client, validates the event stream
# framing, and returns the accumulated response.
"""Black-box OpenAI streaming client for e2e agents.

Consumes a ``stream: true`` chat completions response as SSE and rebuilds the
non-streaming response shape, so agent code can stay wire-format agnostic.
The accumulator follows the OpenAI delta semantics (fragments concatenate,
tool_calls merge by ``index``) and therefore works for both the session
server's single-chunk fake streaming and a real multi-chunk stream.

The rebuilt assistant message must survive the session server's
``strict_message_matches`` prefix check on the next turn, so tool_call fragments
keep every upstream key except the streaming-only ``index``.
"""

import json

import httpx


def accumulate_chat_chunks(chunks: list[dict], *, label: str = "") -> dict:
    # [FUNCTION DOC] Merges a list of chunk mappings into one response mapping. Asserts the list is
    # non-empty and each chunk's object type is the chunk type; each chunk must have exactly one
    # choice (unpacking error otherwise). Message: starts with role assistant and null content;
    # a delta role overrides the role; content and reasoning-content fragments (when not null) are
    # concatenated in order; tool-call fragments are merged per their index: all keys except the
    # index and function (and null values) overwrite the entry's keys, the function name overwrites
    # when present, and the function arguments fragments are concatenated. The last non-empty finish
    # reason wins, and the last non-null usage wins. If any tool calls were seen, they are put on
    # the message ordered by index. The result has id, creation time and model taken from the first
    # chunk, object type chat completion, a single choice at index 0 with the message and finish
    # reason (none if never seen), and usage only when seen. The label prefixes assertion messages.
    """Fold ``chat.completion.chunk`` dicts into a non-streaming response."""
    assert chunks, f"{label}: SSE stream contained no chunks before [DONE]"
    message: dict = {"role": "assistant", "content": None}
    tool_calls_by_index: dict[int, dict] = {}
    finish_reason = None
    usage = None
    for chunk in chunks:
        assert chunk.get("object") == "chat.completion.chunk", f"{label}: unexpected SSE payload {chunk!r}"
        [choice] = chunk["choices"]
        delta = choice.get("delta") or {}
        if delta.get("role"):
            message["role"] = delta["role"]
        if delta.get("content") is not None:
            message["content"] = (message["content"] or "") + delta["content"]
        if delta.get("reasoning_content") is not None:
            message["reasoning_content"] = (message.get("reasoning_content") or "") + delta["reasoning_content"]
        for fragment in delta.get("tool_calls") or []:
            entry = tool_calls_by_index.setdefault(fragment["index"], {"function": {}})
            for key, value in fragment.items():
                if key in ("index", "function") or value is None:
                    continue
                entry[key] = value
            function = fragment.get("function") or {}
            if function.get("name") is not None:
                entry["function"]["name"] = function["name"]
            if function.get("arguments") is not None:
                entry["function"]["arguments"] = entry["function"].get("arguments", "") + function["arguments"]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
    if tool_calls_by_index:
        message["tool_calls"] = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    first = chunks[0]
    response = {
        "id": first.get("id"),
        "object": "chat.completion",
        "created": first.get("created"),
        "model": first.get("model"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage is not None:
        response["usage"] = usage
    return response


async def stream_chat_completions(client: httpx.AsyncClient, url: str, payload: dict, *, label: str = "") -> dict:
    # [FUNCTION DOC] Sends a streaming POST of the payload with the stream flag forced on (the
    # caller's mapping is not mutated). Raises an assertion error with status and decoded body if
    # the status is not 200, and asserts the content type starts with the event-stream type.
    # Reads lines: skips empty lines and comment lines (starting with a colon); every other line
    # must start with "data: " (assertion error otherwise) and must not come after the done marker;
    # the done marker "[DONE]" ends the stream, other data lines are parsed as JSON into chunks.
    # Asserts the done marker was seen, then returns the accumulated response (which asserts at
    # least one chunk). The label is used in messages.
    """POST *payload* with ``stream: true`` via httpx and rebuild the response."""
    chunks: list[dict] = []
    done = False
    async with client.stream("POST", url, json={**payload, "stream": True}) as resp:
        if resp.status_code != 200:
            body = await resp.aread()
            raise AssertionError(f"{label} failed ({resp.status_code}): {body.decode(errors='replace')}")
        content_type = resp.headers.get("content-type", "")
        assert content_type.startswith(
            "text/event-stream"
        ), f"{label}: expected SSE, got content-type {content_type!r}"
        async for line in resp.aiter_lines():
            if not line or line.startswith(":"):
                continue
            assert line.startswith("data: "), f"{label}: unexpected SSE line {line!r}"
            assert not done, f"{label}: SSE event after [DONE]"
            data = line[len("data: ") :]
            if data == "[DONE]":
                done = True
                continue
            chunks.append(json.loads(data))
    assert done, f"{label}: stream ended without data: [DONE]"
    return accumulate_chat_chunks(chunks, label=label)
