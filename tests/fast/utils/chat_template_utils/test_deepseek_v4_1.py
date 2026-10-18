# [MODULE DOC] Tests for the DeepSeek V4.1 chat template bridge. Registers as a CPU test for the
# stage-a suite with estimated time 25 seconds. The bridge must render exactly what the sglang
# serving chat V4.1 branch renders: the V4.1 encoder, the server's default reasoning effort
# (taken from an environment variable, default "high"), tool payloads in the client-only form, and
# no empty system message unless tools need a host. Covers detection by config model type (V4 is
# not V4.1), render parity across several scenarios in chat and thinking modes, V4.1 differing from
# V4, the generation prompt being an exact suffix, environment default and per-request override of
# reasoning effort (unsupported tier falls back to the default), tool payload shape, dispatch through
# the generic entry point, and binding of the V4.1 token-in-token-out tokenizer class.
"""The DeepSeek V4.1 bridge must render exactly what sglang's ``serving_chat``
dsv41 branch renders: the ``encoding_dsv41`` encoder, the server's default
``reasoning_effort``, client-only tool payloads, and no empty system message
unless tools need a host.
"""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="stage-a-cpu", labels=[])

import json

import pytest
from sglang.srt.entrypoints.openai import chat_encoding, encoding_dsv4, encoding_dsv41
from sglang.srt.entrypoints.openai.protocol import Tool

from miles.utils.chat_template_utils import apply_chat_template, deepseek
from miles.utils.chat_template_utils.tito_tokenizer import (
    DeepSeekV4TITOTokenizer,
    DeepSeekV41TITOTokenizer,
    TITOTokenizerType,
)

_MSGS = {
    "no_system": [{"role": "user", "content": "Hello"}],
    "system": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}],
    "mid_system": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "Now answer in French."},
    ],
    "tool_calls_and_result": [
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
            ],
        },
        {"role": "tool", "content": "sunny", "tool_call_id": "call_0"},
    ],
}

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]


class _FakeTokenizer:
    # [CLASS DOC] HELPER: Minimal tokenizer stand-in: name-or-path attribute, a character-code
    # encoder and a token-to-id lookup.
    def __init__(self, name_or_path: str):
        # [FUNCTION DOC] HELPER: Store the name-or-path string (directory holding a config file).
        self.name_or_path = name_or_path

    def encode(self, text, add_special_tokens=False):
        # [FUNCTION DOC] HELPER: Return the Unicode code points of the text as integers; asserts
        # special tokens were not requested.
        assert add_special_tokens is False
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, token):
        # [FUNCTION DOC] HELPER: Return a pseudo token id: the string hash modulo 1000.
        # KNOWN ISSUE: Python string hashing is randomized per process, so the id is not stable
        # across runs (fine here since only the call succeeding matters).
        return hash(token) % 1000


def _tok_with_model_type(tmp_path, model_type: str) -> _FakeTokenizer:
    # [FUNCTION DOC] HELPER: Write a UTF-8 JSON config file with the model type into the directory
    # and return a fake tokenizer pointing at it.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return _FakeTokenizer(str(tmp_path))


def _server_default_effort(monkeypatch=None):
    # [FUNCTION DOC] HELPER: Return the server's default V4.1 reasoning effort as computed by
    # sglang from its environment variable with fallback "high". KNOWN ISSUE: the monkeypatch
    # parameter is unused.
    return chat_encoding.default_dsv41_reasoning_effort_from_env("high")


def test_detect_by_config(tmp_path):
    # [FUNCTION DOC] A config with model type deepseek_v41 is detected as "deepseek_v41", and the
    # family registry maps that name to the V4.1 family object.
    assert deepseek.model_type(_tok_with_model_type(tmp_path, "deepseek_v41")) == "deepseek_v41"
    assert deepseek._FAMILIES["deepseek_v41"] is deepseek.V41


def test_v4_is_not_v41(tmp_path):
    # [FUNCTION DOC] A deepseek_v4 config maps through the registry to the V4 family object, not V4.1.
    assert deepseek._FAMILIES[deepseek.model_type(_tok_with_model_type(tmp_path, "deepseek_v4"))] is deepseek.V4


@pytest.mark.parametrize("scenario", list(_MSGS), ids=list(_MSGS))
@pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
def test_render_matches_server_encode(monkeypatch, scenario, thinking):
    # [FUNCTION DOC] Parametrized over four scenarios (no system, system, mid-conversation system,
    # tool call and result) and chat/thinking. Clears the reasoning effort environment variable,
    # computes the expected text with the V4.1 encoder and the server default effort, and asserts
    # the V4.1 family render is identical.
    monkeypatch.delenv("SGLANG_DSV41_REASONING_EFFORT", raising=False)
    thinking_mode = "thinking" if thinking else "chat"
    expected = encoding_dsv41.encode_messages(
        _MSGS[scenario], thinking_mode=thinking_mode, reasoning_effort=_server_default_effort()
    )
    assert deepseek.V41.render_messages(_MSGS[scenario], thinking_mode=thinking_mode) == expected


def test_render_differs_from_v4():
    # [FUNCTION DOC] For the system scenario in thinking mode, V4.1 and V4 renders differ.
    assert deepseek.V41.render_messages(_MSGS["system"], thinking_mode="thinking") != deepseek.V4.render_messages(
        _MSGS["system"], thinking_mode="thinking"
    )


@pytest.mark.parametrize("scenario", list(_MSGS), ids=list(_MSGS))
def test_generation_prompt_strip_is_exact_prefix(scenario):
    # [FUNCTION DOC] Parametrized over the four scenarios in thinking mode: the render with the
    # prompt switch off is a strict prefix of the default render (starts it and differs from it).
    without = deepseek.V41.render_messages(_MSGS[scenario], thinking_mode="thinking", add_generation_prompt=False)
    with_prompt = deepseek.V41.render_messages(_MSGS[scenario], thinking_mode="thinking")
    assert with_prompt.startswith(without)
    assert with_prompt != without


def test_reasoning_effort_env_default_and_request_override(monkeypatch):
    # [FUNCTION DOC] Sets the reasoning effort environment variable to "max" and asserts: no
    # per-request value renders like the encoder with "max"; a numeric request value 0.3 renders
    # like the encoder with the integer 30; an unsupported value "ultra" falls back to the
    # environment default "max" (as the serving layer does).
    monkeypatch.setenv("SGLANG_DSV41_REASONING_EFFORT", "max")
    assert deepseek.V41.render_messages(
        _MSGS["no_system"], thinking_mode="thinking"
    ) == encoding_dsv41.encode_messages(_MSGS["no_system"], thinking_mode="thinking", reasoning_effort="max")
    assert deepseek.V41.render_messages(
        _MSGS["no_system"], thinking_mode="thinking", reasoning_effort=0.3
    ) == encoding_dsv41.encode_messages(_MSGS["no_system"], thinking_mode="thinking", reasoning_effort=30)
    # an unsupported tier falls back to the server default, as serving_chat does
    assert deepseek.V41.render_messages(
        _MSGS["no_system"], thinking_mode="thinking", reasoning_effort="ultra"
    ) == encoding_dsv41.encode_messages(_MSGS["no_system"], thinking_mode="thinking", reasoning_effort="max")


def test_tools_use_client_only_payload(monkeypatch):
    # [FUNCTION DOC] Clears the effort environment variable. Builds the client-only tool payload
    # with sglang's helper and asserts it differs from the plain tool model dump. Expected text is
    # the encoder output for an explicit empty system message carrying the payload followed by the
    # basic message, in chat mode with the default effort; asserts the family render with tools
    # equals it and that it contains the tool name.
    monkeypatch.delenv("SGLANG_DSV41_REASONING_EFFORT", raising=False)
    payload = chat_encoding.dsv41_tool_payload(Tool.model_validate(_TOOLS[0]))
    assert payload != Tool.model_validate(_TOOLS[0]).model_dump()
    expected = encoding_dsv41.encode_messages(
        [{"role": "system", "content": "", "tools": [payload]}, *_MSGS["no_system"]],
        thinking_mode="chat",
        reasoning_effort=_server_default_effort(),
    )
    assert deepseek.V41.render_messages(_MSGS["no_system"], tools=_TOOLS, thinking_mode="chat") == expected
    assert "get_weather" in expected


def test_apply_chat_template_dispatch(tmp_path):
    # [FUNCTION DOC] The generic entry point in tokenize mode with chat thinking mode on a
    # deepseek_v41 tokenizer returns the character codes of the V4.1 family render of the system scenario.
    tok = _tok_with_model_type(tmp_path, "deepseek_v41")
    ids = apply_chat_template(_MSGS["system"], tokenizer=tok, tokenize=True, thinking_mode="chat")
    assert ids == [ord(c) for c in deepseek.V41.render_messages(_MSGS["system"], thinking_mode="chat")]


def test_tito_binding(tmp_path):
    # [FUNCTION DOC] Asserts: the DeepSeek V4.1 tokenizer type resolves to the V4.1 TITO tokenizer
    # class, which subclasses the V4 one, with reasoning parser "deepseek-v41" and tool call parser
    # "deepseekv41". Constructs it with enable-thinking false in the template kwargs and asserts the
    # stored kwargs have thinking false (alias translated). Also asserts the V4 and V4.1 encoder
    # modules are different objects.
    assert TITOTokenizerType.get_tokenizer_class(TITOTokenizerType.DEEPSEEKV41) is DeepSeekV41TITOTokenizer
    assert issubclass(DeepSeekV41TITOTokenizer, DeepSeekV4TITOTokenizer)
    assert (DeepSeekV41TITOTokenizer.reasoning_parser, DeepSeekV41TITOTokenizer.tool_call_parser) == (
        "deepseek-v41",
        "deepseekv41",
    )
    tok = _tok_with_model_type(tmp_path, "deepseek_v41")
    tito = DeepSeekV41TITOTokenizer(tok, chat_template_kwargs={"enable_thinking": False})
    assert tito.chat_template_kwargs["thinking"] is False
    assert encoding_dsv4 is not encoding_dsv41
