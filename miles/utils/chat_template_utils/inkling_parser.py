# [MODULE DOC] Parser for the token id stream of an Inkling assistant completion. It is a small
# state machine over control tokens (message start, content kinds thinking/text/JSON tool
# invocation/text tool invocation, end of message, end of sampling) that yields a client-facing
# OpenAI-style message and a richer stored message with ordered content blocks. Parsing is
# lenient: the first problem is recorded as a parse error string and parsing stops or continues
# as noted per case, but a result is always returned. Pure apart from tokenizer calls.

"""Token-aware parsing of raw Inkling assistant completions.

SGLang returns the generated token IDs in ``meta_info.output_token_logprobs``.
The session server uses those IDs as the source of truth and exposes ordinary
OpenAI ``reasoning_content``/``content``/``tool_calls`` fields to the agent.
Parsing is stateless and linear in the completion length so every session
server process can use its own tokenizer instance without coordination.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedAssistantCompletion:
    # [CLASS DOC] Immutable parse result: the message exposed to the client, the message stored
    # (client message plus a content blocks list), the parser name, and an optional error code
    # string (None when parsing was clean).
    """Optional model-family projection of one raw assistant completion."""

    client_message: dict[str, Any]
    stored_message: dict[str, Any]
    parser_name: str
    parse_error: str | None = None


class InklingResponseParser:
    # [CLASS DOC] Stateless (after construction) parser bound to a tokenizer. Holds a mapping from
    # control token names (message start, thinking, text, JSON tool, text tool, end message, end
    # sampling) to their single token ids, and the set of all ids treated as framing (those plus
    # the tokenizer's special ids).
    """Parse Inkling control-token IDs without interpreting marker-like text."""

    _TOKEN_TEXT = {
        "message_model": "<|message_model|>",
        "content_thinking": "<|content_thinking|>",
        "content_text": "<|content_text|>",
        "content_invoke_tool_json": "<|content_invoke_tool_json|>",
        "content_invoke_tool_text": "<|content_invoke_tool_text|>",
        "end_message": "<|end_message|>",
        "end_sampling": "<|content_model_end_sampling|>",
    }

    def __init__(self, tokenizer: Any):
        # [FUNCTION DOC] Stores the tokenizer, resolves the id of every control token literal
        # (value error if a literal does not map to exactly one id) and builds the framing id set
        # as the control ids plus the tokenizer's special ids (empty if it has none).
        self.tokenizer = tokenizer
        self.ids = {name: self._single_token_id(literal) for name, literal in self._TOKEN_TEXT.items()}
        # Treat every tokenizer-native special ID as framing.  This rejects a
        # real control token in payload while preserving marker-looking text
        # that the model spelled with ordinary BPE tokens.
        self._control_ids = frozenset({*self.ids.values(), *getattr(tokenizer, "all_special_ids", [])})

    def _single_token_id(self, literal: str) -> int:
        # [FUNCTION DOC] HELPER: Looks the literal up as a vocabulary token. If missing or equal
        # to the unknown token id, encodes the literal without special tokens and requires exactly
        # one resulting id (else a value error showing the encoding). Returns the id as an
        # integer.
        token_id = self.tokenizer.convert_tokens_to_ids(literal)
        if token_id is None or token_id == getattr(self.tokenizer, "unk_token_id", None):
            encoded = self.tokenizer.encode(literal, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Inkling control token {literal!r} did not encode to one ID: {encoded}")
            token_id = encoded[0]
        return int(token_id)

    def _decode_text(self, token_ids: list[int]) -> str:
        # [FUNCTION DOC] HELPER: Decodes ids to text keeping special tokens in the output.
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    @staticmethod
    def _tool_call_id(completion_token_ids: list[int], index: int) -> str:
        # [FUNCTION DOC] HELPER: Deterministic tool call id. Hashes with SHA-256 the call index as
        # 4 big-endian bytes followed by every completion token id as a 4-byte signed big-endian
        # integer. Returns "call_" plus the first 24 hex characters of the digest.
        digest = hashlib.sha256()
        digest.update(index.to_bytes(4, "big"))
        for token_id in completion_token_ids:
            digest.update(int(token_id).to_bytes(4, "big", signed=True))
        return f"call_{digest.hexdigest()[:24]}"

    def parse(
        self,
        completion_token_ids: list[int],
        *,
        finish_reason: str | None,
    ) -> ParsedAssistantCompletion:
        # [FUNCTION DOC] Parses one completion's generated token ids (keyword-only finish reason
        # is accepted but ignored). The generation prompt already supplied the first message-start
        # token, so parsing begins in the header state. States: header (collects ordinary ids as
        # the recipient until a content-kind token opens a block; a leading message-start token
        # with an empty header is skipped; end-sampling with an empty header ends the stream; any
        # other control token is an error), payload (collects ordinary ids until end-message
        # closes the block; any other control token is an error), between (after end-message:
        # expects message-start to begin another header or end-sampling to finish; anything else
        # is an error). Errors record only the first reason and stop the scan, except that errors
        # from closing a block do not stop it. Tokens after end-sampling record an error and stop.
        # Closing a block decodes payload and recipient: thinking and text blocks must have an
        # empty recipient (else error) and append to the reasoning or visible text; a JSON tool
        # block must decode to a JSON object with a string name and an arguments object, yielding
        # a function-type tool call with a deterministic id and compact JSON arguments plus a
        # tool-call block keeping the raw JSON (otherwise errors for malformed JSON, non-object
        # payload or bad shape); a text-tool block is always unsupported. After the scan, adds an
        # error for an unfinished payload block, an unfinished header with ids, or a missing
        # end-sampling. Outputs: the client message has the assistant role, the concatenated
        # visible text, reasoning text only if any exists, and tool calls only if there are some
        # and no error occurred; the stored message adds the ordered content blocks (thinking,
        # text, tool call) regardless of errors. The parser name is "inkling".
        # KNOWN ISSUE: the finish reason parameter is deliberately discarded, so truncation due to
        # length is only detected indirectly through the missing end-sampling error.
        del finish_reason
        blocks: list[dict[str, Any]] = []
        visible_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        error: str | None = None
        ended = False

        # The generation prompt already supplied the first MESSAGE_MODEL token.
        state = "header"
        header_ids: list[int] = []
        payload_ids: list[int] = []
        block_kind: str | None = None

        def fail(reason: str) -> None:
            # [FUNCTION DOC] HELPER: Records the reason as the parse error unless one was already
            # recorded (first error wins).
            nonlocal error
            if error is None:
                error = reason

        def close_block() -> None:
            # [FUNCTION DOC] HELPER: Finishes the open block per the rules in the parse
            # description, appending to the shared blocks, visible text, reasoning and tool call
            # lists, and recording errors via the failure helper. If no block kind was set (an
            # end-message with no content-kind token), records the error "end_message_without_
            # content" and returns without resetting. Otherwise resets the block kind and both id
            # buffers at the end.
            nonlocal block_kind, header_ids, payload_ids
            if block_kind is None:
                fail("end_message_without_content")
                return
            text = self._decode_text(payload_ids)
            recipient = self._decode_text(header_ids)
            if block_kind == "thinking":
                if recipient:
                    fail("unexpected_thinking_recipient")
                blocks.append({"type": "thinking", "text": text})
                reasoning_parts.append(text)
            elif block_kind == "text":
                if recipient:
                    fail("unexpected_text_recipient")
                blocks.append({"type": "text", "text": text})
                visible_parts.append(text)
            elif block_kind == "tool_json":
                try:
                    payload = json.loads(text)
                except (TypeError, json.JSONDecodeError):
                    fail("malformed_tool_json")
                    payload = None
                if not isinstance(payload, dict):
                    fail("tool_payload_not_object")
                else:
                    name = payload.get("name")
                    arguments = payload.get("args")
                    if not isinstance(name, str) or not isinstance(arguments, dict):
                        fail("invalid_tool_payload_shape")
                    else:
                        call_index = len(tool_calls)
                        tool_calls.append(
                            {
                                "id": self._tool_call_id(completion_token_ids, call_index),
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                                },
                            }
                        )
                        blocks.append(
                            {
                                "type": "tool_call",
                                "recipient": recipient,
                                "name": name,
                                "arguments": arguments,
                                "raw_json": text,
                            }
                        )
            else:
                fail("unsupported_text_tool_call")
            block_kind = None
            header_ids = []
            payload_ids = []

        for token_id in completion_token_ids:
            token_id = int(token_id)
            if ended:
                fail("tokens_after_end_sampling")
                break

            if state == "payload":
                if token_id == self.ids["end_message"]:
                    close_block()
                    state = "between"
                elif token_id in self._control_ids:
                    fail("control_token_inside_open_block")
                    break
                else:
                    payload_ids.append(token_id)
                continue

            if state == "between":
                if token_id == self.ids["message_model"]:
                    state = "header"
                elif token_id == self.ids["end_sampling"]:
                    ended = True
                else:
                    fail("missing_message_model_boundary")
                    break
                continue

            # header: ordinary IDs are the model-emitted invocation recipient.
            if token_id == self.ids["message_model"] and not header_ids:
                continue
            if token_id == self.ids["content_thinking"]:
                block_kind = "thinking"
            elif token_id == self.ids["content_text"]:
                block_kind = "text"
            elif token_id == self.ids["content_invoke_tool_json"]:
                block_kind = "tool_json"
            elif token_id == self.ids["content_invoke_tool_text"]:
                block_kind = "tool_text"
            elif token_id == self.ids["end_sampling"] and not header_ids:
                ended = True
                state = "between"
                continue
            elif token_id in self._control_ids:
                fail("unexpected_control_token_in_header")
                break
            else:
                header_ids.append(token_id)
                continue
            payload_ids = []
            state = "payload"

        if state == "payload":
            fail("unterminated_content_block")
        elif state == "header" and header_ids:
            fail("unterminated_message_header")
        elif not ended:
            fail("missing_end_sampling")

        parse_error = error
        client_message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(visible_parts),
        }
        if reasoning_parts:
            client_message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls and parse_error is None:
            client_message["tool_calls"] = tool_calls
        stored_message = {
            **client_message,
            "content_blocks": blocks,
        }
        return ParsedAssistantCompletion(
            client_message=client_message,
            stored_message=stored_message,
            parser_name="inkling",
            parse_error=parse_error,
        )
