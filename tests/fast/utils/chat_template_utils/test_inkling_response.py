# [MODULE DOC] Tests for the Inkling response parser (InklingResponseParser) and the Inkling TITO
# tokenizer hooks that use it (preserving server-side message state, post-processing a completion
# into a stored message). A fake tokenizer with a small special-token table maps text characters
# to 1000 plus their code point and special markers to fixed ids. Covers ordered block parsing
# (thinking, text, tool call with recipient kept separate from the tool name), literal marker text
# staying payload, parse errors requesting retry (control token inside block, malformed tool JSON,
# unterminated block on length cut), independence of parser instances, and the post-processing
# effects on the choice dictionary (finish reason, meta info).
import json

from miles.utils.chat_template_utils.inkling_parser import InklingResponseParser
from miles.utils.chat_template_utils.tito_tokenizer import InklingTITOTokenizer


class FakeInklingTokenizer:
    # [CLASS DOC] HELPER: Fake tokenizer with a fixed special token table (end of text, model
    # message start, thinking/text/tool-json/tool-text content starts, end message, end of sampling).
    # Ordinary characters encode as 1000 plus the code point; unknown tokens map to -1.
    SPECIAL = {
        "<|endoftext|>": 99,
        "<|message_model|>": 101,
        "<|content_thinking|>": 108,
        "<|content_text|>": 104,
        "<|content_invoke_tool_json|>": 149,
        "<|content_invoke_tool_text|>": 157,
        "<|end_message|>": 110,
        "<|content_model_end_sampling|>": 106,
    }

    all_special_ids = list(SPECIAL.values())
    unk_token_id = -1

    def convert_tokens_to_ids(self, text):
        # [FUNCTION DOC] HELPER: Return the special token id for the text, or the unknown id (-1)
        # when the text is not a special token.
        return self.SPECIAL.get(text, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        # [FUNCTION DOC] HELPER: Encode each character as 1000 plus its code point; the special
        # tokens flag is ignored. Special marker text is NOT recognized (encoded char by char).
        del add_special_tokens
        return [1000 + ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=False):
        # [FUNCTION DOC] HELPER: Inverse of the encoding: special ids become their marker text,
        # other ids become the character with code point id minus 1000. The skip flag is ignored.
        del skip_special_tokens
        inverse = {value: key for key, value in self.SPECIAL.items()}
        return "".join(inverse[token_id] if token_id in inverse else chr(token_id - 1000) for token_id in token_ids)


def _text(tokenizer, value):
    # [FUNCTION DOC] HELPER: Encode a string with the given tokenizer without special tokens and
    # return the id list.
    return tokenizer.encode(value, add_special_tokens=False)


def test_parses_ordered_blocks_and_keeps_recipient_separate_from_tool_name():
    # [FUNCTION DOC] Builds a completion of three messages: a thinking block "first", a model
    # message with a text block "visible", and a model message addressed to recipient "bash" with a
    # tool-json block holding a compact JSON payload (name bash_command, args command pwd), ending
    # with the end-of-sampling token. Parses with finish reason stop and asserts: no parse error;
    # client reasoning is "first" and content "visible"; stored content blocks are thinking, text,
    # tool call in order with the last block's recipient "bash"; the client tool call has function
    # name bash_command and arguments JSON equal to the command pwd mapping; and the stored message
    # carries no raw completion token ids entry.
    tokenizer = FakeInklingTokenizer()
    parser = InklingResponseParser(tokenizer)
    ids = tokenizer.SPECIAL
    payload = json.dumps(
        {"name": "bash_command", "args": {"command": "pwd"}},
        separators=(",", ":"),
    )
    completion = [
        ids["<|content_thinking|>"],
        *_text(tokenizer, "first"),
        ids["<|end_message|>"],
        ids["<|message_model|>"],
        ids["<|content_text|>"],
        *_text(tokenizer, "visible"),
        ids["<|end_message|>"],
        ids["<|message_model|>"],
        *_text(tokenizer, "bash"),
        ids["<|content_invoke_tool_json|>"],
        *_text(tokenizer, payload),
        ids["<|end_message|>"],
        ids["<|content_model_end_sampling|>"],
    ]

    parsed = parser.parse(completion, finish_reason="stop")

    assert parsed.parse_error is None
    assert parsed.client_message["reasoning_content"] == "first"
    assert parsed.client_message["content"] == "visible"
    assert [block["type"] for block in parsed.stored_message["content_blocks"]] == [
        "thinking",
        "text",
        "tool_call",
    ]
    assert parsed.stored_message["content_blocks"][-1]["recipient"] == "bash"
    tool_call = parsed.client_message["tool_calls"][0]
    assert tool_call["function"]["name"] == "bash_command"
    assert json.loads(tool_call["function"]["arguments"]) == {"command": "pwd"}
    assert "_miles_raw_completion_token_ids" not in parsed.stored_message


def test_preserves_ordered_blocks_after_flattened_client_replay():
    # [FUNCTION DOC] The stored message has content blocks (thinking, text) while the client's
    # replayed request message is flattened (content and reasoning only). Asserts the tokenizer's
    # state preservation returns the request message with the stored content blocks attached, and
    # does not mutate the request message (no content blocks added to it).
    tokenizer = FakeInklingTokenizer()
    content_blocks = [
        {"type": "thinking", "text": "first"},
        {"type": "text", "text": "visible"},
    ]
    request_messages = [
        {
            "role": "assistant",
            "content": "visible",
            "reasoning_content": "first",
        }
    ]
    stored_messages = [{**request_messages[0], "content_blocks": content_blocks}]

    preserved = InklingTITOTokenizer(tokenizer).preserve_server_message_state(
        stored_messages,
        request_messages,
    )

    assert preserved == [{**request_messages[0], "content_blocks": content_blocks}]
    assert "content_blocks" not in request_messages[0]


def test_marker_spelled_with_ordinary_tokens_remains_payload():
    # [FUNCTION DOC] A thinking block whose text spells an end-of-text marker with ordinary
    # character tokens; asserts no parse error and the reasoning equals that literal text.
    tokenizer = FakeInklingTokenizer()
    ids = tokenizer.SPECIAL
    completion = [
        ids["<|content_thinking|>"],
        *_text(tokenizer, "literal <|endoftext|> text"),
        ids["<|end_message|>"],
        ids["<|content_model_end_sampling|>"],
    ]

    parsed = InklingResponseParser(tokenizer).parse(completion, finish_reason="stop")

    assert parsed.parse_error is None
    assert parsed.client_message["reasoning_content"] == "literal <|endoftext|> text"


def test_native_special_token_inside_payload_requests_retry():
    # [FUNCTION DOC] A text block containing the real end-of-text special token id; asserts parse
    # error "control_token_inside_open_block" and no tool calls on the client message.
    tokenizer = FakeInklingTokenizer()
    ids = tokenizer.SPECIAL
    completion = [
        ids["<|content_text|>"],
        ids["<|endoftext|>"],
        ids["<|end_message|>"],
        ids["<|content_model_end_sampling|>"],
    ]

    parsed = InklingResponseParser(tokenizer).parse(completion, finish_reason="stop")

    assert parsed.parse_error == "control_token_inside_open_block"
    assert "tool_calls" not in parsed.client_message


def test_malformed_and_length_cut_completions_request_retry():
    # [FUNCTION DOC] Parses a tool-json block with truncated JSON (finish reason stop) and a
    # thinking block never closed (finish reason length). Asserts parse error "malformed_tool_json"
    # with no tool calls for the first, and "unterminated_content_block" for the second.
    tokenizer = FakeInklingTokenizer()
    ids = tokenizer.SPECIAL
    malformed = [
        *_text(tokenizer, "bash"),
        ids["<|content_invoke_tool_json|>"],
        *_text(tokenizer, '{"name":"bash_command","args":'),
        ids["<|end_message|>"],
        ids["<|content_model_end_sampling|>"],
    ]
    truncated = [
        ids["<|content_thinking|>"],
        *_text(tokenizer, "unfinished"),
    ]

    malformed_result = InklingResponseParser(tokenizer).parse(malformed, finish_reason="stop")
    truncated_result = InklingResponseParser(tokenizer).parse(truncated, finish_reason="length")

    assert malformed_result.parse_error == "malformed_tool_json"
    assert "tool_calls" not in malformed_result.client_message
    assert truncated_result.parse_error == "unterminated_content_block"


def test_parser_instances_have_no_shared_state():
    # [FUNCTION DOC] Two parsers over two separate tokenizers parse the same simple text completion
    # and both yield client content "ok" (completion built with the first tokenizer's ids).
    first_tokenizer = FakeInklingTokenizer()
    second_tokenizer = FakeInklingTokenizer()
    first = InklingResponseParser(first_tokenizer)
    second = InklingResponseParser(second_tokenizer)
    completion = [
        first_tokenizer.SPECIAL["<|content_text|>"],
        *_text(first_tokenizer, "ok"),
        first_tokenizer.SPECIAL["<|end_message|>"],
        first_tokenizer.SPECIAL["<|content_model_end_sampling|>"],
    ]

    assert first.parse(completion, finish_reason="stop").client_message["content"] == "ok"
    assert second.parse(completion, finish_reason="stop").client_message["content"] == "ok"


def test_postprocesses_tool_call_response_and_returns_stored_message():
    # [FUNCTION DOC] Post-processes a tool-call completion into a choice dictionary (finish reason
    # stop, meta info with one existing entry). Asserts the choice's finish reason becomes
    # "tool_calls", meta info keeps the existing entry and adds the response parser name "inkling",
    # the choice message's tool calls equal the returned stored message's tool calls, the stored
    # message has no raw completion token ids entry, and its first content block is a tool call.
    tokenizer = FakeInklingTokenizer()
    ids = tokenizer.SPECIAL
    payload = json.dumps(
        {"name": "bash_command", "args": {"command": "pwd"}},
        separators=(",", ":"),
    )
    completion = [
        *_text(tokenizer, "bash"),
        ids["<|content_invoke_tool_json|>"],
        *_text(tokenizer, payload),
        ids["<|end_message|>"],
        ids["<|content_model_end_sampling|>"],
    ]
    assistant_message = {"role": "assistant", "content": "upstream"}
    choice = {
        "message": assistant_message,
        "finish_reason": "stop",
        "meta_info": {"existing": True},
    }

    stored_message = InklingTITOTokenizer(tokenizer).postprocess_completion(
        choice=choice,
        assistant_message=assistant_message,
        completion_token_ids=completion,
    )

    assert choice["finish_reason"] == "tool_calls"
    assert choice["meta_info"] == {
        "existing": True,
        "miles_response_parser": "inkling",
    }
    assert choice["message"]["tool_calls"] == stored_message["tool_calls"]
    assert "_miles_raw_completion_token_ids" not in stored_message
    assert stored_message["content_blocks"][0]["type"] == "tool_call"


def test_postprocesses_parse_error_without_changing_finish_reason():
    # [FUNCTION DOC] Post-processes a completion with a control token inside a text block. Asserts
    # the finish reason stays "stop", meta info gains the parser name and the parse error
    # "control_token_inside_open_block" besides the existing entry, the choice message gets no tool
    # calls, and the stored message has no raw completion token ids entry.
    tokenizer = FakeInklingTokenizer()
    ids = tokenizer.SPECIAL
    completion = [
        ids["<|content_text|>"],
        ids["<|endoftext|>"],
        ids["<|end_message|>"],
        ids["<|content_model_end_sampling|>"],
    ]
    assistant_message = {"role": "assistant", "content": "upstream"}
    choice = {
        "message": assistant_message,
        "finish_reason": "stop",
        "meta_info": {"existing": True},
    }

    stored_message = InklingTITOTokenizer(tokenizer).postprocess_completion(
        choice=choice,
        assistant_message=assistant_message,
        completion_token_ids=completion,
    )

    assert choice["finish_reason"] == "stop"
    assert choice["meta_info"] == {
        "existing": True,
        "miles_response_parser": "inkling",
        "miles_response_parse_error": "control_token_inside_open_block",
    }
    assert "tool_calls" not in choice["message"]
    assert "_miles_raw_completion_token_ids" not in stored_message
