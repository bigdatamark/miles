# [MODULE DOC]
# Helpers that let the session server speak the Anthropic messages protocol on top of SGLang's
# Anthropic compatibility utilities. The SGLang imports are optional: when unavailable, the
# module-level names are set to none and the availability check returns false. Provides request
# parsing and feature gating (rejects thinking mode, server tools, images, etc.), wire-format
# JSON/SSE rendering, Anthropic-style error responses that forward only selected headers, and
# a pair of functions that carry assistant reasoning (thinking blocks) across the conversion to an
# OpenAI-style body. Module constants: an allow-list of exact error header names to keep
# (authenticate, retry-after, request id) and prefixes of rate-limit headers to keep.
"""Anthropic protocol helpers for the session HTTP adapter."""

import json

from pydantic import ValidationError
from starlette.responses import Response

try:
    from sglang.srt.entrypoints.anthropic import utils as anthropic_utils
    from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest, is_server_tool
except ImportError:
    anthropic_utils = None
    AnthropicMessagesRequest = None
    is_server_tool = None

from miles.rollout.session.core import JSON_MEDIA_TYPE, _render_json


def anthropic_adapter_available() -> bool:
    # [FUNCTION DOC] True only if both the SGLang Anthropic utilities module and the request
    # model were imported successfully. Pure check (the server-tool predicate is not checked).
    """Whether the optional SGLang Anthropic helpers this adapter needs exist."""
    return anthropic_utils is not None and AnthropicMessagesRequest is not None


# Preserve end-to-end error metadata; drop headers tied to the replaced body.
_ANTHROPIC_ERROR_HEADER_ALLOWLIST = ("www-authenticate", "retry-after", "x-request-id")
_ANTHROPIC_ERROR_HEADER_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-")


def _anthropic_wire_json(model) -> bytes:
    # [FUNCTION DOC] HELPER: Serialize a pydantic model to JSON bytes in wire form: JSON-mode
    # values, null fields omitted, field aliases used, rendered by the core JSON renderer.
    return _render_json(model.model_dump(mode="json", exclude_none=True, by_alias=True))


def _anthropic_error_response(status_code: int, body: bytes, headers: dict | None = None) -> Response:
    # [FUNCTION DOC] HELPER: Convert an upstream error (status and raw body bytes) into an
    # Anthropic-format error HTTP response with the same status. Header names are matched case-
    # insensitively; only allow-listed names and rate-limit prefixed headers are forwarded (others,
    # such as content length, are dropped since the body is replaced). Media type is JSON.
    envelope = anthropic_utils.to_anthropic_error(status_code, body)
    kept_headers = {
        k: v
        for k, v in (headers or {}).items()
        if k.lower() in _ANTHROPIC_ERROR_HEADER_ALLOWLIST or k.lower().startswith(_ANTHROPIC_ERROR_HEADER_PREFIXES)
    }
    return Response(
        content=_anthropic_wire_json(envelope),
        status_code=status_code,
        headers=kept_headers,
        media_type=JSON_MEDIA_TYPE,
    )


def _anthropic_sse_body(events) -> bytes:
    # [FUNCTION DOC] HELPER: Render an iterable of stream event models into one server-sent-event
    # byte string: for each event, an event line with its type, a data line with its wire JSON, and
    # a blank line, concatenated in order.
    return b"".join(
        f"event: {event.type}\ndata: ".encode() + _anthropic_wire_json(event) + b"\n\n" for event in events
    )


def _parse_anthropic_request(body: bytes) -> AnthropicMessagesRequest:
    # [FUNCTION DOC] HELPER: Decode JSON bytes and validate them into the SGLang Anthropic request
    # model. Invalid JSON/encoding and model validation failures are both raised as a value error
    # (with the cause chained) carrying a descriptive message.
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON body: {exc}") from exc
    try:
        return AnthropicMessagesRequest.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def _validate_anthropic_content_block(block, *, allow_thinking: bool = False) -> None:
    # [FUNCTION DOC] HELPER: Reject unsupported content block kinds with a value error. Thinking
    # blocks pass only when allowed (assistant history); redacted thinking, image, tool reference
    # and search result blocks always fail; a tool result flagged as an error fails. All other
    # blocks pass. Returns nothing.
    if block.type == "thinking":
        if allow_thinking:
            return
        raise ValueError("thinking content blocks are only supported in assistant history")
    if block.type == "redacted_thinking":
        raise ValueError("redacted_thinking content blocks are not supported by this endpoint")
    if block.type == "image":
        raise ValueError("image content blocks are not enabled for this deployment")
    if block.type == "tool_reference":
        raise ValueError("tool_reference content blocks are not enabled for this deployment")
    if block.type == "search_result":
        raise ValueError("search_result content blocks are not enabled for this deployment")
    if block.type == "tool_result" and block.is_error is True:
        raise ValueError("tool_result is_error=true is not supported by this endpoint")


def _validate_anthropic_features(request: AnthropicMessagesRequest) -> None:
    # [FUNCTION DOC] HELPER: Gate a parsed request to the supported feature subset, raising a
    # value error naming the first violation: thinking mode set, output config set, any beta
    # flags, any server-side tool. Then validate content blocks: system blocks (if the system
    # prompt is a block list), every message block (thinking allowed only in assistant messages),
    # and blocks nested inside list-valued tool results (thinking not allowed there). String
    # contents are accepted as is.
    if request.thinking is not None:
        raise ValueError("thinking is not supported by this endpoint")
    if request.output_config is not None:
        raise ValueError("output_config is not enabled for this deployment")
    if request.betas:
        raise ValueError("betas is not enabled for this deployment")
    if request.tools:
        for tool in request.tools:
            if is_server_tool(tool):
                raise ValueError(f"server tool {tool.name!r} (type={tool.type!r}) is not enabled for this deployment")

    if request.system is not None and not isinstance(request.system, str):
        for block in request.system:
            _validate_anthropic_content_block(block)
    for message in request.messages:
        if isinstance(message.content, str):
            continue
        for block in message.content:
            _validate_anthropic_content_block(block, allow_thinking=message.role == "assistant")
            if block.type == "tool_result" and isinstance(block.content, list):
                for nested_block in block.content:
                    _validate_anthropic_content_block(nested_block)


def _strip_anthropic_reasoning_history(
    anthropic_request: AnthropicMessagesRequest,
) -> tuple[AnthropicMessagesRequest, list[str | None]]:
    # [FUNCTION DOC] HELPER: Split thinking out of the history before conversion to OpenAI form.
    # Returns a copy of the request whose assistant messages no longer contain thinking blocks, and
    # a list with one entry per assistant message in order: the non-empty thinking texts joined by
    # newlines, or none when there is no thinking text (including string-content messages). The
    # input request is not mutated; non-assistant messages are passed through unchanged.
    """Return a conversion copy plus canonical assistant reasoning history."""
    conversion_messages = []
    reasoning_history: list[str | None] = []
    for message in anthropic_request.messages:
        if message.role != "assistant":
            conversion_messages.append(message)
            continue
        if isinstance(message.content, str):
            conversion_messages.append(message)
            reasoning_history.append(None)
            continue
        thinking_blocks = [block for block in message.content if block.type == "thinking"]
        thinking_parts = [block.thinking for block in thinking_blocks if block.thinking]
        reasoning_history.append("\n".join(thinking_parts) or None)
        if thinking_blocks:
            message = message.model_copy(
                update={"content": [block for block in message.content if block.type != "thinking"]}
            )
        conversion_messages.append(message)
    return anthropic_request.model_copy(update={"messages": conversion_messages}), reasoning_history


def _restore_anthropic_reasoning_history(openai_body: dict, reasoning_history: list[str | None]) -> None:
    # [FUNCTION DOC] HELPER: Mutates the converted OpenAI-style body in place: pairs the assistant
    # messages with the saved reasoning list in order and sets the reasoning content field on each
    # message whose saved reasoning is not none. Raises a value error if the number of assistant
    # messages differs from the saved list length (conversion merged or split turns).
    """Map replayed assistant thinking blocks back to canonical reasoning content."""
    assistants = [message for message in openai_body["messages"] if message["role"] == "assistant"]
    if len(assistants) != len(reasoning_history):
        raise ValueError(
            f"assistant history count changed during Anthropic conversion: "
            f"{len(reasoning_history)} before, {len(assistants)} after"
        )
    for message, reasoning_content in zip(assistants, reasoning_history, strict=True):
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
