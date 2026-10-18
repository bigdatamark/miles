# [MODULE DOC] Tests for the DeepSeek V4 chat template bridge; mirrors the V3.2 test module.
# Registers as a CPU test for the stage-a suite with estimated time 25 seconds. Covers detection by
# config model_type only (architectures hint ignored, bad configs fall back), rendering parity with
# the installed sglang V4 encoder, tokenize dispatch, inherited keyword passthrough, the V4-only
# reasoning-effort option (omitted equals null, "max" changes output, "high" accepted, invalid
# values rejected by the encoder), tool schema injection (system block, empty list ignored, bare
# function dictionaries rejected), input immutability, unknown keyword rejection, the
# enable-thinking alias, add-generation-prompt handling for several tails, routing (V3.2 checkpoint
# not routed to V4, non-DeepSeek falls back to the tokenizer's own template), and append-only
# behavior of injected messages. Uses a tiny fake tokenizer and temporary config files.
"""Tests for the DeepSeek V4 chat-template bridge and its dispatch through
``apply_chat_template``.

Mirrors ``test_deepseek_v32.py``: the bridge renders via sglang's
``encoding_dsv4.encode_messages`` (a pure string operation, no tokenizer
needed), so most cases use plain message lists. Detection and the
``tokenize=True`` dispatch use a tiny tokenizer stub backed by a temporary
``config.json`` -- no real DeepSeek V4 checkpoint is required. The only
behavioral delta versus V3.2 is the optional ``reasoning_effort`` kwarg.
"""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="stage-a-cpu", labels=[])

import copy
import json

import pytest
from pydantic import ValidationError

from miles.utils.chat_template_utils import apply_chat_template, deepseek

_MSGS_BASIC = [{"role": "user", "content": "Hello"}]


class _FakeTokenizer:
    # [CLASS DOC] HELPER: Minimal tokenizer stand-in with a name-or-path attribute (directory
    # holding a config file) and a character-code encode method.
    """Minimal tokenizer stub: ``name_or_path`` drives detection, ``encode`` is a
    deterministic char->id map that asserts ``add_special_tokens=False``."""

    def __init__(self, name_or_path: str):
        # [FUNCTION DOC] HELPER: Store the name-or-path string used for model type detection.
        self.name_or_path = name_or_path

    def encode(self, text, add_special_tokens=False):
        # [FUNCTION DOC] HELPER: Return the Unicode code point of each character as a list of
        # integers; asserts special tokens were not requested.
        assert add_special_tokens is False
        return [ord(c) for c in text]


class _FakeHFTokenizer(_FakeTokenizer):
    # [CLASS DOC] HELPER: Fake tokenizer for non-DeepSeek checkpoints. It has a dummy chat template
    # and a template-application method returning a fixed sentinel string, so tests can tell the
    # fallback path was taken.
    """Non-DeepSeek tokenizer stub that records an ``apply_chat_template`` call,
    so we can assert the HF fallback path is taken for unrelated checkpoints."""

    def __init__(self, name_or_path: str):
        # [FUNCTION DOC] HELPER: Initialize as the parent and set a dummy chat template string.
        super().__init__(name_or_path)
        self.chat_template = "dummy"

    def apply_chat_template(self, messages, **kwargs):
        # [FUNCTION DOC] HELPER: Ignore all inputs and return the sentinel string
        # "HF_FALLBACK_SENTINEL". KNOWN ISSUE: despite the class docstring it does not record the call.
        return "HF_FALLBACK_SENTINEL"


def _tok_with_model_type(tmp_path, model_type: str, **extra) -> _FakeTokenizer:
    # [FUNCTION DOC] HELPER: Write a config file (UTF-8 JSON) into the directory holding the model
    # type plus any extra keyword entries, and return a fake tokenizer pointing at that directory.
    config = {"model_type": model_type, **extra}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return _FakeTokenizer(str(tmp_path))


def _reference_encode(messages, *, thinking: bool = False, drop_thinking: bool = True, **extra) -> str:
    # [FUNCTION DOC] HELPER: Reference rendering: import the installed sglang V4 encoder lazily and
    # call its message encoder with thinking mode "thinking" or "chat", the drop-thinking flag
    # (default true) and any extra keyword options passed through. Returns the rendered string.
    """The canonical V4 rendering: a direct ``encode_messages`` call. Locks
    ``render_messages`` to this thin-bridge contract (no preprocessing of its own)."""
    from sglang.srt.entrypoints.openai import encoding_dsv4

    return encoding_dsv4.encode_messages(
        messages, thinking_mode="thinking" if thinking else "chat", drop_thinking=drop_thinking, **extra
    )


# ---------------------------------------------------------------------------
# Detection (by config.json model_type only)
# ---------------------------------------------------------------------------


def test_detect_dsv4_by_config(tmp_path):
    # [FUNCTION DOC] A config with model type deepseek_v4 is detected as exactly "deepseek_v4".
    assert deepseek.model_type(_tok_with_model_type(tmp_path, "deepseek_v4")) == "deepseek_v4"


def test_detect_non_dsv4(tmp_path):
    # [FUNCTION DOC] A qwen3 config is not detected as deepseek_v4.
    assert deepseek.model_type(_tok_with_model_type(tmp_path, "qwen3")) != "deepseek_v4"


def test_detect_dsv32_is_not_dsv4(tmp_path):
    # [FUNCTION DOC] A deepseek_v32 config is not detected as deepseek_v4.
    # Cross-version exactness: a V3.2 checkpoint is not detected as V4.
    assert deepseek.model_type(_tok_with_model_type(tmp_path, "deepseek_v32")) != "deepseek_v4"


def test_detect_ignores_architectures(tmp_path):
    # [FUNCTION DOC] A config with model type deepseek_v3 but an architectures hint naming the V4
    # causal LM class is still not detected as deepseek_v4.
    # Detection is by model_type ONLY: an architectures hint must not flip it.
    tok = _tok_with_model_type(tmp_path, "deepseek_v3", architectures=["DeepseekV4ForCausalLM"])
    assert deepseek.model_type(tok) != "deepseek_v4"


def test_detect_missing_config_falls_back(tmp_path):
    # [FUNCTION DOC] No config file: detection does not raise and does not report deepseek_v4.
    # No config.json -> empty model_type -> not dsv4, no exception.
    assert deepseek.model_type(_FakeTokenizer(str(tmp_path))) != "deepseek_v4"


def test_detect_invalid_config_falls_back(tmp_path):
    # [FUNCTION DOC] Malformed JSON config: detection does not raise and does not report deepseek_v4.
    # Malformed JSON must fall back to HF, not raise.
    (tmp_path / "config.json").write_text("{ not valid json", encoding="utf-8")
    assert deepseek.model_type(_FakeTokenizer(str(tmp_path))) != "deepseek_v4"


def test_detect_non_object_config_falls_back(tmp_path):
    # [FUNCTION DOC] Config holding a JSON list: detection does not raise and does not report deepseek_v4.
    # Valid JSON that is not an object (e.g. a list) must fall back to HF, not raise.
    (tmp_path / "config.json").write_text("[]", encoding="utf-8")
    assert deepseek.model_type(_FakeTokenizer(str(tmp_path))) != "deepseek_v4"


def test_detect_empty_name_or_path():
    # [FUNCTION DOC] Empty name-or-path: not detected as deepseek_v4, no exception.
    assert deepseek.model_type(_FakeTokenizer("")) != "deepseek_v4"


# ---------------------------------------------------------------------------
# Rendering parity with encode_messages (full scenario matrix)
# ---------------------------------------------------------------------------

_PARITY_SCENARIOS = {
    "no_system": [{"role": "user", "content": "Hello"}],
    "system": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}],
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


@pytest.mark.parametrize("scenario", list(_PARITY_SCENARIOS), ids=list(_PARITY_SCENARIOS))
@pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
def test_render_matches_direct_encode_messages(scenario, thinking):
    # [FUNCTION DOC] Parametrized over three scenarios and chat/thinking modes; asserts the V4
    # family render equals the direct reference encoder output.
    messages = _PARITY_SCENARIOS[scenario]
    thinking_mode = "thinking" if thinking else "chat"
    assert deepseek.V4.render_messages(messages, thinking_mode=thinking_mode) == _reference_encode(
        messages, thinking=thinking
    )


@pytest.mark.parametrize("scenario", list(_PARITY_SCENARIOS), ids=list(_PARITY_SCENARIOS))
@pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
def test_apply_chat_template_tokenize_matches_render(tmp_path, scenario, thinking):
    # [FUNCTION DOC] With the generic entry point in tokenize mode on a deepseek_v4 tokenizer,
    # asserts the ids equal the character codes of the V4 rendered string.
    # tokenize=True path encodes the rendered string with add_special_tokens=False.
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
    messages = _PARITY_SCENARIOS[scenario]
    thinking_mode = "thinking" if thinking else "chat"
    ids = apply_chat_template(messages, tokenizer=tok, tokenize=True, thinking_mode=thinking_mode)
    assert ids == [ord(c) for c in deepseek.V4.render_messages(messages, thinking_mode=thinking_mode)]


def test_inherited_kwargs_pass_through():
    # [FUNCTION DOC] Renders the system scenario with thinking mode, drop-thinking off and the
    # default beginning-of-sequence token disabled; asserts equality with the reference encoder
    # called with the same options.
    # The V3.2 known kwargs are inherited verbatim and forwarded to encode_messages.
    messages = _PARITY_SCENARIOS["system"]
    assert deepseek.V4.render_messages(
        messages, thinking_mode="thinking", drop_thinking=False, add_default_bos_token=False
    ) == _reference_encode(messages, thinking=True, drop_thinking=False, add_default_bos_token=False)


def test_thinking_mode_changes_output():
    # [FUNCTION DOC] Thinking and chat modes render the basic message differently.
    assert deepseek.V4.render_messages(_MSGS_BASIC, thinking_mode="thinking") != deepseek.V4.render_messages(
        _MSGS_BASIC, thinking_mode="chat"
    )


# ---------------------------------------------------------------------------
# reasoning_effort (the single V4 delta)
# ---------------------------------------------------------------------------


def test_reasoning_effort_omitted_equals_none():
    # [FUNCTION DOC] Omitting reasoning effort renders identically to passing null in thinking mode.
    assert deepseek.V4.render_messages(_MSGS_BASIC, thinking_mode="thinking") == deepseek.V4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking", reasoning_effort=None
    )


def test_reasoning_effort_max_differs_from_none():
    # [FUNCTION DOC] Reasoning effort "max" in thinking mode changes the output (a max-effort
    # prefix is emitted) compared with no effort.
    # "max" + thinking emits the max-effort prefix (sglang behavior), so output differs.
    assert deepseek.V4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking", reasoning_effort="max"
    ) != deepseek.V4.render_messages(_MSGS_BASIC, thinking_mode="thinking")


def test_reasoning_effort_max_matches_direct_encode():
    # [FUNCTION DOC] Reasoning effort "max" through the family render equals the reference encoder
    # called with the same option in thinking mode.
    assert deepseek.V4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking", reasoning_effort="max"
    ) == _reference_encode(_MSGS_BASIC, thinking=True, reasoning_effort="max")


def test_reasoning_effort_high_accepted_and_matches_direct():
    # [FUNCTION DOC] Reasoning effort "high" is accepted and equals the reference encoder output;
    # deliberately not asserted to differ from no effort (currently a no-op upstream).
    # "high" is accepted (no exception) and equals a direct call; it is currently a
    # no-op in sglang, so we do NOT assert it differs from None.
    assert deepseek.V4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking", reasoning_effort="high"
    ) == _reference_encode(_MSGS_BASIC, thinking=True, reasoning_effort="high")


def test_reasoning_effort_invalid_value_raises():
    # [FUNCTION DOC] Reasoning effort "ultra" must raise an assertion error or a value error whose
    # message says "Invalid reasoning effort" (validation is done by the upstream encoder).
    # Value-level validation is delegated to sglang; depending on the sglang
    # version it asserts the allowed set or raises ValueError. Both carry the
    # same "Invalid reasoning effort" message.
    with pytest.raises((AssertionError, ValueError), match="Invalid reasoning effort"):
        deepseek.V4.render_messages(_MSGS_BASIC, thinking_mode="thinking", reasoning_effort="ultra")


# ---------------------------------------------------------------------------
# Tool injection (sglang-aligned: tools go into the system <functions> block)
# ---------------------------------------------------------------------------

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


def test_render_with_tools_injects_tool_schemas():
    # [FUNCTION DOC] Rendering with a tool list in chat mode yields text containing the
    # "Available Tool Schemas" heading and the tool name.
    out = deepseek.V4.render_messages([{"role": "user", "content": "hi"}], tools=_TOOLS, thinking_mode="chat")
    assert "### Available Tool Schemas" in out
    assert "get_weather" in out


def test_render_with_tools_matches_manual_system_injection():
    # [FUNCTION DOC] Canonicalizes the tools via sglang's tool model dump, renders a conversation
    # with an explicit empty system message carrying them, and asserts passing the tools directly
    # gives the identical output.
    # Passing tools= must equal pre-canonicalizing them into the system message and
    # passing tools=None -- i.e. the injection is exactly the Tool-pydantic model_dump
    # that sglang's serving path applies.
    from sglang.srt.entrypoints.openai.protocol import Tool

    canonical = [Tool.model_validate(t).model_dump() for t in _TOOLS]
    msgs = [{"role": "user", "content": "weather?"}]
    expected = deepseek.V4.render_messages(
        [{"role": "system", "content": "", "tools": canonical}, *msgs], thinking_mode="chat"
    )
    assert deepseek.V4.render_messages(msgs, tools=_TOOLS, thinking_mode="chat") == expected


def test_render_with_tools_reuses_existing_system_message():
    # [FUNCTION DOC] With an existing system message and tools, the output contains the system
    # text and the tool schemas heading.
    msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]
    out = deepseek.V4.render_messages(msgs, tools=_TOOLS, thinking_mode="chat")
    assert "You are helpful." in out
    assert "### Available Tool Schemas" in out


def test_empty_tools_list_not_injected():
    # [FUNCTION DOC] An empty tool list and a null tool list both render identically to omitting tools.
    # tools=[] is falsy (like None): neither is injected, so output matches no-tools.
    msgs = [{"role": "user", "content": "hi"}]
    assert deepseek.V4.render_messages(msgs, tools=[], thinking_mode="chat") == deepseek.V4.render_messages(
        msgs, thinking_mode="chat"
    )
    assert deepseek.V4.render_messages(msgs, tools=None, thinking_mode="chat") == deepseek.V4.render_messages(
        msgs, thinking_mode="chat"
    )


def test_bare_function_dict_rejected():
    # [FUNCTION DOC] A tool given as a bare function description (without the outer type/function
    # wrapper) must raise a model validation error; the lenient handling of the tokenizer-library
    # path is not inherited.
    # The DeepSeek path requires a full {"type":"function","function":{...}} dict
    # (Tool.model_validate); the HF-path leniency for bare function dicts is NOT inherited.
    bare = [{"name": "get_weather", "parameters": {"type": "object", "properties": {}}}]
    with pytest.raises(ValidationError):
        deepseek.V4.render_messages([{"role": "user", "content": "hi"}], tools=bare, thinking_mode="chat")


def test_render_with_tools_does_not_mutate_input():
    # [FUNCTION DOC] Rendering with tools leaves the input messages equal to a prior deep copy.
    msgs = [{"role": "user", "content": "hi"}]
    snapshot = copy.deepcopy(msgs)
    deepseek.V4.render_messages(msgs, tools=_TOOLS, thinking_mode="chat")
    assert msgs == snapshot


def test_apply_chat_template_with_tools_dispatches_to_bridge(tmp_path):
    # [FUNCTION DOC] The generic entry point with tools on a deepseek_v4 tokenizer (no
    # tokenization) equals the V4 family render with the same tools.
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
    msgs = [{"role": "user", "content": "hi"}]
    via_apply = apply_chat_template(msgs, tokenizer=tok, tools=_TOOLS, tokenize=False)
    assert via_apply == deepseek.V4.render_messages(msgs, tools=_TOOLS)


# ---------------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------------


def test_does_not_mutate_input(tmp_path):
    # [FUNCTION DOC] The generic entry point on a conversation with dictionary tool arguments leaves
    # the original messages equal to a prior deep copy.
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
    original = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"x": 2}}}],
        },
    ]
    snapshot = copy.deepcopy(original)
    apply_chat_template(original, tokenizer=tok, tokenize=False)
    assert original == snapshot


def test_dict_arguments_equal_string_arguments(tmp_path):
    # [FUNCTION DOC] A conversation with dictionary tool arguments and the same one with JSON string
    # arguments render to identical text through the generic entry point.
    # The dsv4 dispatch normalizes dict tool arguments to JSON strings, so dict-form
    # and string-form arguments render identically.
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
    base = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"a": 1, "b": "x"}}}],
        },
        {"role": "tool", "content": "r", "tool_call_id": "c0"},
    ]
    as_string = copy.deepcopy(base)
    as_string[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"a": 1, "b": "x"}, ensure_ascii=False)
    assert apply_chat_template(base, tokenizer=tok, tokenize=False) == apply_chat_template(
        as_string, tokenizer=tok, tokenize=False
    )


# ---------------------------------------------------------------------------
# Rejection of unknown kwargs
# ---------------------------------------------------------------------------


def test_reject_unknown_kwargs():
    # [FUNCTION DOC] An unrecognized keyword raises a type error about an unexpected keyword argument.
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        deepseek.V4.render_messages(_MSGS_BASIC, some_unknown_kwarg=1)


def test_accept_none_tools_and_known_kwargs():
    # [FUNCTION DOC] Null tools, thinking mode, drop-thinking off and null reasoning effort are all
    # accepted without raising.
    deepseek.V4.render_messages(
        _MSGS_BASIC, tools=None, thinking_mode="thinking", drop_thinking=False, reasoning_effort=None
    )


# ---------------------------------------------------------------------------
# enable_thinking -> thinking_mode translation (miles alias for the encoder knob)
# ---------------------------------------------------------------------------


def test_enable_thinking_true_maps_to_thinking():
    # [FUNCTION DOC] The enable-thinking alias true renders as thinking mode.
    assert deepseek.V4.render_messages(_MSGS_BASIC, enable_thinking=True) == deepseek.V4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking"
    )


def test_enable_thinking_false_maps_to_chat():
    # [FUNCTION DOC] The enable-thinking alias false renders as chat mode.
    assert deepseek.V4.render_messages(_MSGS_BASIC, enable_thinking=False) == deepseek.V4.render_messages(
        _MSGS_BASIC, thinking_mode="chat"
    )


def test_enable_thinking_absent_defaults_to_thinking():
    # [FUNCTION DOC] Without the alias or a mode, the default is thinking mode.
    # No enable_thinking and no thinking_mode -> the cfg default ("thinking").
    assert deepseek.V4.render_messages(_MSGS_BASIC) == deepseek.V4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking"
    )


def test_enable_thinking_none_defaults_to_thinking():
    # [FUNCTION DOC] A null alias is treated as absent: default thinking mode.
    # Explicit None is treated as absent: falls through to the "thinking" default.
    assert deepseek.V4.render_messages(_MSGS_BASIC, enable_thinking=None) == deepseek.V4.render_messages(
        _MSGS_BASIC, thinking_mode="thinking"
    )


def test_explicit_thinking_mode_wins_over_enable_thinking():
    # [FUNCTION DOC] Alias false with an explicit thinking mode "thinking" renders as thinking mode.
    assert deepseek.V4.render_messages(
        _MSGS_BASIC, enable_thinking=False, thinking_mode="thinking"
    ) == deepseek.V4.render_messages(_MSGS_BASIC, thinking_mode="thinking")


def test_enable_thinking_is_consumed_not_rejected():
    # [FUNCTION DOC] Rendering with the alias set does not raise.
    # enable_thinking is translated away, so it is not rejected as an unknown kwarg.
    deepseek.V4.render_messages(_MSGS_BASIC, enable_thinking=True)


def test_build_config_does_not_mutate_input_kwargs():
    # [FUNCTION DOC] The V4 family's private config builder leaves the passed keyword mapping unchanged.
    kwargs = {"enable_thinking": True}
    deepseek.V4._build_encode_config(kwargs)
    assert kwargs == {"enable_thinking": True}


# ---------------------------------------------------------------------------
# Generation-prompt behavior: the encoder's auto opener honors the knob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["user", "developer"])
def test_add_generation_prompt_false_strips_the_auto_opener(role):
    # [FUNCTION DOC] Parametrized over user and developer roles. In both modes the default render
    # equals the no-prompt render plus the assistant opener (assistant marker with an open think
    # tag in thinking mode, with a close think tag in chat mode).
    for mode, opener in (("thinking", "<｜Assistant｜><think>"), ("chat", "<｜Assistant｜></think>")):
        messages = [{"role": role, "content": "Hello"}]
        with_opener = deepseek.V4.render_messages(messages, thinking_mode=mode)
        without = deepseek.V4.render_messages(messages, thinking_mode=mode, add_generation_prompt=False)
        assert with_opener == without + opener


def test_add_generation_prompt_false_strips_the_tool_tail_opener():
    # [FUNCTION DOC] For the tool call plus result conversation, the default render equals the
    # no-prompt render plus the same assistant opener in each mode.
    messages = _PARITY_SCENARIOS["tool_calls_and_result"]
    for mode, opener in (("thinking", "<｜Assistant｜><think>"), ("chat", "<｜Assistant｜></think>")):
        with_opener = deepseek.V4.render_messages(messages, thinking_mode=mode)
        without = deepseek.V4.render_messages(messages, thinking_mode=mode, add_generation_prompt=False)
        assert with_opener == without + opener


def test_add_generation_prompt_false_strips_one_opener_after_tool_and_user_tail():
    # [FUNCTION DOC] For a tool call, a tool result and then a user question, the default render
    # equals the no-prompt render plus exactly one assistant opener in each mode.
    messages = _PARITY_SCENARIOS["tool_calls_and_result"] + [{"role": "user", "content": "and tomorrow?"}]
    for mode, opener in (("thinking", "<｜Assistant｜><think>"), ("chat", "<｜Assistant｜></think>")):
        with_opener = deepseek.V4.render_messages(messages, thinking_mode=mode)
        without = deepseek.V4.render_messages(messages, thinking_mode=mode, add_generation_prompt=False)
        assert with_opener == without + opener


def test_add_generation_prompt_false_noop_on_empty_wo_eos_assistant_tail():
    # [FUNCTION DOC] A conversation ending in an empty assistant message flagged as without
    # end-of-sequence marker (with reasoning): switching the prompt off changes nothing in either mode.
    messages = _MSGS_BASIC + [{"role": "assistant", "content": "", "reasoning_content": "r", "wo_eos": True}]
    for mode in ("thinking", "chat"):
        assert deepseek.V4.render_messages(
            messages, thinking_mode=mode, add_generation_prompt=False
        ) == deepseek.V4.render_messages(messages, thinking_mode=mode)


@pytest.mark.parametrize("task", ["query", "action"])
def test_add_generation_prompt_false_noop_on_task_tail(task):
    # [FUNCTION DOC] Parametrized over task kinds "query" and "action": a user message tagged with
    # a task renders the same with the prompt switch off or on, in both modes.
    for mode in ("thinking", "chat"):
        messages = [{"role": "user", "content": "q", "task": task}]
        assert deepseek.V4.render_messages(
            messages, thinking_mode=mode, add_generation_prompt=False
        ) == deepseek.V4.render_messages(messages, thinking_mode=mode)


def test_apply_chat_template_forwards_add_generation_prompt(tmp_path):
    # [FUNCTION DOC] The generic entry point with the switch off equals the V4 render with the
    # switch off, and differs from the default output.
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
    without_prompt = apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False, add_generation_prompt=False)
    assert without_prompt == deepseek.V4.render_messages(_MSGS_BASIC, add_generation_prompt=False)
    assert apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False) != without_prompt


# ---------------------------------------------------------------------------
# Dispatch integration through apply_chat_template
# ---------------------------------------------------------------------------


def test_apply_chat_template_dispatches_to_bridge(tmp_path):
    # [FUNCTION DOC] The generic entry point on a deepseek_v4 tokenizer equals the V4 family render.
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
    assert apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False) == deepseek.V4.render_messages(_MSGS_BASIC)


def test_apply_chat_template_is_generation_ready(tmp_path):
    # [FUNCTION DOC] The generic entry point output for a basic message contains both the user and
    # the assistant markers.
    tok = _tok_with_model_type(tmp_path, "deepseek_v4")
    out = apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False)
    assert "<｜User｜>" in out
    assert "<｜Assistant｜>" in out


def test_dsv32_tokenizer_not_routed_to_dsv4(tmp_path):
    # [FUNCTION DOC] With tools on a deepseek_v32 tokenizer, asserts the output has the V3.2 style
    # functions block and lacks the V4 tool schemas heading, proving V3.2 routing.
    # A V3.2 checkpoint routes to the V3.2 bridge, not the V4 one. With tools the two
    # encoders emit different tool blocks (<functions> vs "## Tools"), making the
    # routing observable (a bare message renders identically under both encoders).
    tok = _tok_with_model_type(tmp_path, "deepseek_v32")
    out = apply_chat_template(_MSGS_BASIC, tokenizer=tok, tools=_TOOLS, tokenize=False)
    assert "<functions>" in out  # V3.2 tool block
    assert "### Available Tool Schemas" not in out  # not the V4 tool block


def test_non_deepseek_uses_hf_fallback(tmp_path):
    # [FUNCTION DOC] A qwen3 config with the fake fallback tokenizer: the generic entry point
    # returns the tokenizer's own sentinel result, showing both DeepSeek bridges were bypassed.
    # A non-DeepSeek checkpoint bypasses both bridges and uses the HF chat template.
    tok = _FakeHFTokenizer(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    assert apply_chat_template(_MSGS_BASIC, tokenizer=tok, tokenize=False) == "HF_FALLBACK_SENTINEL"


# ---------------------------------------------------------------------------
# Injected system / assistant appends (role-independent template resolution)
# ---------------------------------------------------------------------------

_INJECT_HISTORY = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
]

_INJECT_SHAPES = {
    "system": [{"role": "system", "content": "mid-session system"}],
    "assistant": [{"role": "assistant", "content": "injected"}],
    "assistant_then_user": [
        {"role": "assistant", "content": "injected"},
        {"role": "user", "content": "next question"},
    ],
    "consecutive_assistants_then_user": [
        {"role": "assistant", "content": "first injected"},
        {"role": "assistant", "content": "second injected"},
        {"role": "user", "content": "next question"},
    ],
}


@pytest.mark.parametrize("shape", list(_INJECT_SHAPES), ids=list(_INJECT_SHAPES))
def test_drop_thinking_false_injected_appends_are_append_only(shape):
    # [FUNCTION DOC] Parametrized over four injected shapes appended to a one-round history with
    # reasoning. With dropping off in thinking mode, asserts the longer reference rendering starts
    # with the history's own rendering.
    before = _reference_encode(_INJECT_HISTORY, thinking=True, drop_thinking=False)
    after = _reference_encode(_INJECT_HISTORY + _INJECT_SHAPES[shape], thinking=True, drop_thinking=False)
    assert after.startswith(before)
