# [MODULE DOC] Alignment tests between miles' apply_chat_template and SGLang's chat serving
# pipeline, plus dataset routing and message-matching checks. Reference helpers build a minimal
# SGLang serving-chat object (created without running its initializer, with mocked tokenizer and
# template managers and the attributes the message processing reads) and return its prompt token
# ids for plain Jinja models, for DeepSeek V3.2 and for DeepSeek V4 encoder paths. Tests compare
# those ids with the ids produced by miles for a per-model table (model id, thinking support, fixed
# template family or none, allowed append roles) across mock trajectories and thinking variants;
# verify JSON-string versus dictionary tool arguments, no-tools prompts, input immutability, data
# path routing versus the raw tokenizer, dataset prompt rendering with tools, DeepSeek V3.2 / V4
# TITO prompt alignment (skipped when the local cluster-storage checkpoints are absent), and the
# strict message matcher's treatment of wire index and id fields. Tokenizers are downloaded and
# cached in a module-level cache. Network or model cache access is required.
"""Test that apply_chat_template aligns with SGLang's _apply_jinja_template.

The reference function ``sglang_prompt_ids`` calls
``OpenAIServingChat._process_messages`` directly — the *actual* SGLang code
path, not a re-implementation.  A lightweight ``OpenAIServingChat`` instance
is constructed via ``object.__new__`` (bypassing ``__init__``) with only the
attributes that ``_process_messages`` / ``_apply_jinja_template`` read:

- ``tokenizer_manager.tokenizer`` — the HF tokenizer under test
- ``template_manager.chat_template_name = None`` → selects the Jinja path
- ``template_manager.jinja_template_content_format = "string"`` → text-only
- ``use_dpsk_v32_encoding = False`` / ``chat_encoding_spec = None``
- ``is_gpt_oss = False`` / ``is_gemma4 = False``

Each test asserts that our ``apply_chat_template`` produces identical token IDs.
"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sglang.srt.entrypoints.openai.chat_encoding import resolve_dsv4_reasoning_effort_profile
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from transformers import AutoTokenizer

from miles.utils.chat_template_utils import (
    TITOTokenizerType,
    get_tito_tokenizer,
    resolve_fixed_chat_template,
    strict_message_matches,
)
from miles.utils.chat_template_utils.template import apply_chat_template
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.chat_template_verify import (
    CaseSpec,
    enable_thinking_variants,
    format_case_id,
    select_cases,
)
from miles.utils.test_utils.mock_trajectories import (
    MultiRoleSequenceTrajectory,
    SimpleNoToolTrajectory,
    SingleToolThinkingTrajectory,
    SingleToolTrajectory,
)

# ---------------------------------------------------------------------------
# SGLang reference: calls OpenAIServingChat._process_messages directly
# ---------------------------------------------------------------------------


def _make_serving(tokenizer) -> OpenAIServingChat:
    # [FUNCTION DOC] HELPER: Create a bare SGLang serving-chat object (skipping its initializer)
    # and set only the state its message processing reads: mocked tokenizer manager exposing the
    # given tokenizer; mocked template manager with no named template and "string" content format;
    # DeepSeek V3.2 encoding off, no chat encoding spec, not gpt-oss, not gemma4, no tool call or
    # reasoning parser, no V4 reasoning profile, empty default chat-template kwargs, a flag telling
    # whether the tokenizer auto-adds special tokens (encoding the empty string yields tokens), the
    # result of the object's own prompt-text round-trip probe, and an empty ordered template cache.
    # Returns it. Mirrors attributes added by specific SGLang versions.
    """Create a minimal ``OpenAIServingChat`` that can run ``_process_messages``."""
    serving = object.__new__(OpenAIServingChat)
    serving.tokenizer_manager = MagicMock()
    serving.tokenizer_manager.tokenizer = tokenizer
    serving.template_manager = MagicMock()
    serving.template_manager.chat_template_name = None
    serving.template_manager.jinja_template_content_format = "string"
    serving.use_dpsk_v32_encoding = False
    serving.chat_encoding_spec = None
    serving.is_gpt_oss = False
    serving.is_gemma4 = False
    serving.tool_call_parser = None
    serving.reasoning_parser = None
    serving._dsv4_reasoning_effort_profile = None
    # sglang v0.5.16 added server-level default chat-template kwargs, merged into
    # the request's chat_template_kwargs at the top of _process_messages.
    # __init__ always sets it (to `... or {}`); mirror the empty default so the
    # request's own kwargs are the only ones applied.
    serving.default_chat_template_kwargs = {}
    # sglang v0.5.13 probes whether the tokenizer auto-adds special tokens
    # (encode("") non-empty) to decide add_special_tokens at the chat-template
    # encode site. __init__ always sets this; mirror it here so _process_messages
    # takes the real production path instead of hitting AttributeError.
    serving._tokenizer_auto_adds_specials = len(tokenizer.encode("")) > 0
    serving._prompt_text_round_trip_is_lossy = serving._probe_prompt_text_round_trip()
    serving._chat_template_cache = OrderedDict()
    return serving


def sglang_prompt_ids(
    tokenizer,
    messages: list[dict],
    tools: list[dict] | None = None,
    **kwargs,
) -> list[int]:
    # [FUNCTION DOC] HELPER: Reference prompt ids. Builds a chat completion request from deep
    # copies of the messages (and tools when non-empty; extra keyword arguments become the request's
    # chat-template kwargs, model name "test"), runs the minimal serving object's message processing
    # in text-only mode and returns the resulting prompt token id list.
    """Get prompt_ids by calling SGLang's ``_process_messages`` directly."""
    request_data: dict = {"messages": copy.deepcopy(messages), "model": "test"}
    if tools:
        request_data["tools"] = copy.deepcopy(tools)
    if kwargs:
        request_data["chat_template_kwargs"] = kwargs
    request = ChatCompletionRequest(**request_data)

    serving = _make_serving(tokenizer)
    result = serving._process_messages(request, is_multimodal=False)
    return result.prompt_ids


def sglang_dsv32_prompt_ids(
    tokenizer,
    messages: list[dict],
    tools: list[dict] | None = None,
    **kwargs,
) -> list[int]:
    # [FUNCTION DOC] HELPER: Same as the plain reference helper but forces the serving object's
    # chat encoding spec to "dsv32", the reasoning parser to "deepseek-v3" and the tool call parser
    # to "deepseekv32" before processing, so the DeepSeek V3.2 encoder is used. Returns prompt ids.
    """Get prompt_ids through sglang's DeepSeek V3.2 encoding path.

    sglang selects the V3.2 encoder in ``_apply_jinja_template`` when
    ``chat_encoding_spec == "dsv32"`` (it auto-detects this from architecture +
    a missing jinja template).  We set it directly so the test does not depend
    on architecture introspection.  Thinking is requested via the ``thinking``
    chat-template kwarg, matching the runtime knob.
    """
    request_data: dict = {"messages": copy.deepcopy(messages), "model": "test"}
    if tools:
        request_data["tools"] = copy.deepcopy(tools)
    if kwargs:
        request_data["chat_template_kwargs"] = kwargs
    request = ChatCompletionRequest(**request_data)

    serving = _make_serving(tokenizer)
    serving.chat_encoding_spec = "dsv32"
    serving.reasoning_parser = "deepseek-v3"
    serving.tool_call_parser = "deepseekv32"
    result = serving._process_messages(request, is_multimodal=False)
    return result.prompt_ids


def sglang_dsv4_prompt_ids(
    tokenizer,
    messages: list[dict],
    tools: list[dict] | None = None,
    **kwargs,
) -> list[int]:
    # [FUNCTION DOC] HELPER: Same as the plain reference helper but forces the chat encoding spec
    # to "dsv4", reasoning parser "deepseek-v4", tool call parser "deepseekv4", and sets the V4
    # reasoning effort profile resolved for the local DeepSeek V4 model path constant (read lazily
    # at call time from the module-level path). Returns prompt ids.
    """Get DeepSeek-V4 prompt_ids through SGLang's DSv4 encoder path."""
    request_data: dict = {"messages": copy.deepcopy(messages), "model": "test"}
    if tools:
        request_data["tools"] = copy.deepcopy(tools)
    if kwargs:
        request_data["chat_template_kwargs"] = kwargs
    request = ChatCompletionRequest(**request_data)

    serving = _make_serving(tokenizer)
    serving.chat_encoding_spec = "dsv4"
    serving.reasoning_parser = "deepseek-v4"
    serving.tool_call_parser = "deepseekv4"
    serving._dsv4_reasoning_effort_profile = resolve_dsv4_reasoning_effort_profile(model_path=_DEEPSEEK_V4_MODEL)
    result = serving._process_messages(request, is_multimodal=False)
    return result.prompt_ids


# ---------------------------------------------------------------------------
# Tokenizer cache & fixed-template loader
# ---------------------------------------------------------------------------

_TOK_CACHE: dict[str, AutoTokenizer] = {}

# DeepSeek V4/V3.2 HF repos are huge and not pulled in CI; tests use the
# cluster-mounted copy when present and skip otherwise (Stage 2 only needs the
# tokenizer).
_DEEPSEEK_V4_MODEL = "/cluster-storage/models/deepseek-ai/DeepSeek-V4-Flash"
_DEEPSEEK_V32_MODEL = "/cluster-storage/models/deepseek-ai/DeepSeek-V3.2"


def _get_tokenizer(model_id: str) -> AutoTokenizer:
    # [FUNCTION DOC] HELPER: Return a tokenizer for the model id from the module-level cache,
    # loading it on first use through the project's tokenizer loader with remote code trusted
    # (may download). The cache is never cleared.
    if model_id not in _TOK_CACHE:
        _TOK_CACHE[model_id] = load_tokenizer(model_id, trust_remote_code=True)
    return _TOK_CACHE[model_id]


def _get_deepseek_v32_tokenizer() -> AutoTokenizer:
    # [FUNCTION DOC] HELPER: Skip the calling test if the local DeepSeek V3.2 checkpoint directory
    # does not exist; otherwise return its cached tokenizer.
    if not Path(_DEEPSEEK_V32_MODEL).exists():
        pytest.skip(f"DeepSeek V3.2 tokenizer not found: {_DEEPSEEK_V32_MODEL}")
    return _get_tokenizer(_DEEPSEEK_V32_MODEL)


def _get_deepseek_v4_tokenizer() -> AutoTokenizer:
    # [FUNCTION DOC] HELPER: Skip the calling test if the local DeepSeek V4 Flash checkpoint
    # directory does not exist; otherwise return its cached tokenizer.
    if not Path(_DEEPSEEK_V4_MODEL).exists():
        pytest.skip(f"DeepSeek V4 tokenizer not found: {_DEEPSEEK_V4_MODEL}")
    return _get_tokenizer(_DEEPSEEK_V4_MODEL)


def _load_fixed_or_none(tito_model: TITOTokenizerType | None) -> str | None:
    # [FUNCTION DOC] HELPER: Return the text of the family's bundled fixed chat template file, or
    # none when the family is none or has no template file. Reads the file.
    """Return the bundled fixed chat-template content for *tito_model*, or ``None``."""
    if tito_model is None:
        return None
    path, _kwargs = resolve_fixed_chat_template(tito_model)
    if path is None:
        return None
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Per-model declarations
# ---------------------------------------------------------------------------
#
# (model_id, supports_thinking, fixed_template_tito_model, allowed_append_roles)
#
# ``fixed_template_tito_model`` is ``None`` when the model uses its native HF
# template; set to a ``TITOTokenizerType`` when the test should swap in the
# bundled fixed template registered for that family.  ``allowed_append_roles``
# reflects the set of append-role combinations the model's template can render
# without raising — test asserts that the sglang path and our path produce
# identical tokens on all such cases. Qwen3.5/3.6 use the bundled fixed
# template which raises on intermediate system post-revert, so the role set
# is narrowed to {tool} only.

_MODELS: list[tuple[str, bool, TITOTokenizerType | None, frozenset[str]]] = [
    ("Qwen/Qwen3-4B", True, None, frozenset({"tool", "user", "system"})),
    ("zai-org/GLM-4.7-Flash", True, None, frozenset({"tool", "user", "system"})),
    ("Qwen/Qwen3.5-4B", True, TITOTokenizerType.QWEN35, frozenset({"tool"})),
    ("Qwen/Qwen3.6-35B-A3B", True, TITOTokenizerType.QWEN36, frozenset({"tool"})),
    ("Qwen/Qwen3.8-27B", True, TITOTokenizerType.QWEN38_SMALL, frozenset({"tool"})),
    ("Qwen/Qwen3.8-Flash-Next", True, TITOTokenizerType.QWEN4_EXP, frozenset({"tool"})),
    ("Qwen/Qwen3-Coder-Next", False, None, frozenset({"tool", "user", "system"})),
]


def _build_align_params():
    # [FUNCTION DOC] HELPER: Build parameters (model id, fixed family, case, thinking variant) for
    # the alignment test. For each model, selects cases by allowed append roles (non-thinking
    # trajectories only when the model has no thinking), crosses with both thinking variants (or
    # off only), and skips trajectories ending in a plain assistant message (SimpleNoTool,
    # SingleToolThinking, MultiRoleSequence) because SGLang treats those as continue-final-message
    # and drops the trailing assistant header. Ids are the short model name plus the case id.
    # Thinking templates: every selected trajectory × {enable_thinking=True, False}.
    # Non-thinking templates: only non-thinking trajectories, no enable_thinking kwarg.
    params = []
    for model_id, supports_thinking, fixed_tito, allowed_roles in _MODELS:
        short = model_id.split("/")[-1]
        cases = select_cases(
            allowed_append_roles=allowed_roles,
            is_thinking=None if supports_thinking else False,
        )
        variants = enable_thinking_variants("both" if supports_thinking else "off")
        for case in cases:
            # Trajectories ending with a plain assistant message (no tool_calls):
            # sglang's _process_messages treats that as continue_final_message and
            # drops the trailing <|im_start|>assistant header, which diverges from
            # apply_chat_template(add_generation_prompt=True). SGLang-alignment-
            # specific quirk — kept inline rather than living on mock_trajectories.
            if case.traj_cls in (
                SimpleNoToolTrajectory,
                SingleToolThinkingTrajectory,
                MultiRoleSequenceTrajectory,
            ):
                continue
            for variant in variants:
                ident = f"{short}-{format_case_id(case, variant)}"
                params.append(pytest.param(model_id, fixed_tito, case, variant, id=ident))
    return params


def _per_model_params():
    # [FUNCTION DOC] HELPER: One parameter per table model: (model id, fixed family or none) with
    # the short model name as id.
    return [pytest.param(model_id, fixed_tito, id=model_id.split("/")[-1]) for model_id, _, fixed_tito, _ in _MODELS]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _assert_aligned(tokenizer, case: CaseSpec, kwargs: dict, chat_template: str | None):
    # [FUNCTION DOC] HELPER: Compute the SGLang reference prompt ids and the miles tokenized
    # apply-chat-template ids for the case's messages and tools with the same template kwargs (and
    # an override chat template when given), and assert they are equal lists.
    # ``kwargs`` are template kwargs (e.g. ``enable_thinking``, ``clear_thinking``)
    # — both ``sglang_prompt_ids`` (via ``chat_template_kwargs``) and
    # ``apply_chat_template`` (via ``**template_kwargs``) route them into jinja.
    # Don't put non-template kwargs in here.
    extra = {"chat_template": chat_template} if chat_template else {}
    expected = sglang_prompt_ids(tokenizer, case.traj_cls.MESSAGES, case.traj_cls.TOOLS, **kwargs, **extra)
    actual = apply_chat_template(
        case.traj_cls.MESSAGES,
        tokenizer=tokenizer,
        tools=case.traj_cls.TOOLS,
        tokenize=True,
        **kwargs,
        **extra,
    )
    assert actual == expected


class TestAlignWithSGLang:
    # [CLASS DOC] Alignment of miles' apply-chat-template with the SGLang reference for the
    # per-model table.
    """apply_chat_template must produce identical prompt_ids to SGLang's pipeline."""

    @pytest.mark.parametrize("model_id, fixed_tito, case, kwargs", _build_align_params())
    def test_align(self, model_id, fixed_tito, case, kwargs):
        # [FUNCTION DOC] Parametrized over the generated matrix. Loads the model tokenizer and the
        # optional fixed template and asserts aligned prompt ids.
        tokenizer = _get_tokenizer(model_id)
        chat_template = _load_fixed_or_none(fixed_tito)
        _assert_aligned(tokenizer, case, kwargs, chat_template)

    @pytest.mark.parametrize("model_id, fixed_tito", _per_model_params())
    def test_json_string_arguments(self, model_id, fixed_tito):
        # [FUNCTION DOC] Parametrized per model. Uses a weather conversation (assistant tool call
        # whose arguments are a JSON string, then a tool result) with one tool definition, and
        # asserts the miles ids equal the SGLang reference ids (with the fixed template if any).
        """JSON-string tool_call arguments should produce same IDs as dict arguments."""
        tokenizer = _get_tokenizer(model_id)
        chat_template = _load_fixed_or_none(fixed_tito)
        extra = {"chat_template": chat_template} if chat_template else {}
        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                    }
                ],
            },
            {"role": "tool", "content": "sunny", "tool_call_id": "call_1", "name": "get_weather"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        expected = sglang_prompt_ids(tokenizer, messages, tools, **extra)
        actual = apply_chat_template(messages, tokenizer=tokenizer, tools=tools, tokenize=True, **extra)
        assert actual == expected

    @pytest.mark.parametrize("model_id, fixed_tito", _per_model_params())
    def test_no_tools(self, model_id, fixed_tito):
        # [FUNCTION DOC] Parametrized per model. A system plus user conversation without tools gives
        # identical ids from miles and the SGLang reference.
        """Plain conversation without tools."""
        tokenizer = _get_tokenizer(model_id)
        chat_template = _load_fixed_or_none(fixed_tito)
        extra = {"chat_template": chat_template} if chat_template else {}
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        expected = sglang_prompt_ids(tokenizer, messages, **extra)
        actual = apply_chat_template(messages, tokenizer=tokenizer, tokenize=True, **extra)
        assert actual == expected

    @pytest.mark.parametrize("model_id, fixed_tito", _per_model_params())
    def test_does_not_mutate_input(self, model_id, fixed_tito):
        # [FUNCTION DOC] Parametrized per model (the fixed family parameter is unused). Applies the
        # template to deep copies of the single-tool trajectory messages and tools and asserts both
        # equal their snapshots afterwards.
        tokenizer = _get_tokenizer(model_id)
        messages = copy.deepcopy(SingleToolTrajectory.MESSAGES)
        tools = copy.deepcopy(SingleToolTrajectory.TOOLS)
        saved_msgs = copy.deepcopy(messages)
        saved_tools = copy.deepcopy(tools)
        apply_chat_template(messages, tokenizer=tokenizer, tools=tools, tokenize=True)
        assert messages == saved_msgs
        assert tools == saved_tools

    @pytest.mark.parametrize("model_id, fixed_tito", _per_model_params())
    def test_data_path_routing_matches_raw_tokenizer(self, model_id, fixed_tito):
        # [FUNCTION DOC] Parametrized per model (the fixed family parameter is unused). For one user
        # message without tools, asserts the rendered string (not tokenized, with generation prompt)
        # from miles equals the tokenizer's own chat template output.
        """Dataset rendering routes non-DeepSeek models through
        ``apply_chat_template`` instead of the old raw ``tokenizer.apply_chat_template``.
        For the no-tools dataset case the rendered string must be byte-identical."""
        tokenizer = _get_tokenizer(model_id)
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        new_path = apply_chat_template(messages, tokenizer=tokenizer, tokenize=False, add_generation_prompt=True)
        old_raw = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert new_path == old_raw


def _build_dataset(tmp_path, tokenizer, row, **dataset_kwargs):
    # [FUNCTION DOC] HELPER: Write the row as one JSON line into a rows file in the given
    # directory and construct the project's Dataset from it with the tokenizer, no processor, no
    # extra options and chat template application enabled, plus any keyword options. Returns the
    # dataset. Needs no GPU or Ray.
    """Instantiate ``Dataset`` from a one-row temp jsonl (CPU-only, no ray/GPU)."""
    import json as _json
    import os

    from miles.utils.data import Dataset

    path = os.path.join(tmp_path, "rows.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_json.dumps(row) + "\n")
    return Dataset(path, tokenizer, None, None, apply_chat_template=True, **dataset_kwargs)


class TestDatasetRouting:
    # [CLASS DOC] Tests of dataset prompt rendering through the unified template wiring.
    """End-to-end check that ``miles.utils.data.Dataset`` renders prompts through
    the unified ``apply_chat_template`` wiring (tool_key parsing, per-sample
    rendering, ``Sample.prompt`` output)."""

    @pytest.mark.parametrize("model_id, fixed_tito", _per_model_params())
    def test_no_tool_prompt_byte_identical_to_raw(self, tmp_path, model_id, fixed_tito):
        # [FUNCTION DOC] Parametrized per model (family unused). Builds a one-row dataset with a
        # user message and asserts the first sample's prompt string equals the tokenizer's own
        # chat template output with generation prompt.
        tokenizer = _get_tokenizer(model_id)
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        dataset = _build_dataset(str(tmp_path), tokenizer, {"text": messages})
        expected = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert dataset.samples[0].prompt == expected

    def test_tool_prompt_matches_sglang_ids(self, tmp_path):
        # [FUNCTION DOC] Builds a one-row dataset for Qwen3-4B with a user message and a weather
        # tool read from the tools column, and asserts that encoding the first sample's prompt
        # string without special tokens equals the SGLang reference prompt ids.
        tokenizer = _get_tokenizer("Qwen/Qwen3-4B")
        messages = [{"role": "user", "content": "What's the weather in Paris?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        dataset = _build_dataset(str(tmp_path), tokenizer, {"text": messages, "tools": tools}, tool_key="tools")
        expected_ids = sglang_prompt_ids(tokenizer, messages, tools)
        assert tokenizer.encode(dataset.samples[0].prompt, add_special_tokens=False) == expected_ids


# ---------------------------------------------------------------------------
# DeepSeek V3.2 TITO alignment
# ---------------------------------------------------------------------------


class TestDeepSeekV32TITOAlignWithSGLang:
    # [CLASS DOC] Alignment tests of the DeepSeek V3.2 TITO tokenizer with the SGLang V3.2 encoder;
    # tests skip when the local checkpoint is missing. Holds a class-level weather tool list.
    """V3.2 TITO prompt tokenization must match sglang's dsv32 encoder.

    V3.2 ships no jinja chat_template; the ``DEEPSEEKV32`` TITO family rides
    miles' ``apply_chat_template`` -> ``chat_template_utils.deepseek`` bridge,
    which mirrors sglang's ``chat_encoding_spec == "dsv32"`` branch.  These pin
    that ``get_tito_tokenizer(...).apply_chat_template`` produces identical
    prompt_ids to sglang for the tool surface actually used in training, with
    thinking on/off.  ``thinking`` is the sglang request knob; ``enable_thinking``
    is the equivalent miles chat-template kwarg — both map to the encoder's
    ``thinking_mode``.
    """

    _TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    @pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
    def test_prompt_ids_match_sglang_dsv32_with_tools(self, thinking):
        # [FUNCTION DOC] Parametrized chat/thinking. For a single user question with the weather
        # tool, the SGLang reference ids (thinking request knob) equal the TITO tokenizer's ids
        # (enable-thinking template kwarg, generation prompt, tokenized, default template args
        # built from the tools).
        tokenizer = _get_deepseek_v32_tokenizer()
        messages = [{"role": "user", "content": "What is the weather in Paris?"}]

        expected = sglang_dsv32_prompt_ids(tokenizer, messages, self._TOOLS, thinking=thinking)
        tito = get_tito_tokenizer(
            tokenizer,
            tokenizer_type=TITOTokenizerType.DEEPSEEKV32,
            chat_template_kwargs={"enable_thinking": thinking},
        )
        actual = tito.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, template_args=tito.default_template_args(self._TOOLS)
        )
        assert actual == expected

    def test_absent_enable_thinking_defaults_to_thinking(self):
        # [FUNCTION DOC] With no enable-thinking kwarg in the TITO tokenizer, the ids equal the
        # SGLang reference ids with thinking requested true (the encoder default).
        # With the translation now living in the encoder, an absent enable_thinking
        # renders in thinking mode (the encoder default), matching sglang thinking=True.
        tokenizer = _get_deepseek_v32_tokenizer()
        messages = [{"role": "user", "content": "What is the weather in Paris?"}]

        expected = sglang_dsv32_prompt_ids(tokenizer, messages, self._TOOLS, thinking=True)
        tito = get_tito_tokenizer(
            tokenizer,
            tokenizer_type=TITOTokenizerType.DEEPSEEKV32,
            chat_template_kwargs={},
        )
        actual = tito.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, template_args=tito.default_template_args(self._TOOLS)
        )
        assert actual == expected

    @pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
    def test_prompt_ids_match_sglang_dsv32_with_tool_history(self, thinking):
        # [FUNCTION DOC] Parametrized chat/thinking. A conversation with a user question, an
        # assistant tool call and a tool result: SGLang reference ids equal the TITO tokenizer ids.
        tokenizer = _get_deepseek_v32_tokenizer()
        messages = [
            {"role": "user", "content": "Weather in Beijing?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                    }
                ],
            },
            {"role": "tool", "content": "sunny, 22C", "tool_call_id": "call_1", "name": "get_weather"},
        ]

        expected = sglang_dsv32_prompt_ids(tokenizer, messages, self._TOOLS, thinking=thinking)
        tito = get_tito_tokenizer(
            tokenizer,
            tokenizer_type=TITOTokenizerType.DEEPSEEKV32,
            chat_template_kwargs={"enable_thinking": thinking},
        )
        actual = tito.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, template_args=tito.default_template_args(self._TOOLS)
        )
        assert actual == expected


# ---------------------------------------------------------------------------
# DeepSeek V4 TITO alignment
# ---------------------------------------------------------------------------


class TestDeepSeekV4TITOAlignWithSGLang:
    # [CLASS DOC] Alignment tests of the DeepSeek V4 TITO tokenizer with the SGLang V4 encoder;
    # tests skip when the local checkpoint is missing.
    """DeepSeek V4 TITO prompt tokenization must match SGLang's DSv4 encoder."""

    @pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
    def test_prompt_ids_match_sglang_dsv4_with_tools(self, thinking):
        # [FUNCTION DOC] Parametrized chat/thinking. A single user question with a no-parameter
        # year tool: SGLang V4 reference ids equal the V4 TITO tokenizer ids.
        tokenizer = _get_deepseek_v4_tokenizer()
        messages = [{"role": "user", "content": "What is 1+1?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_year",
                    "description": "Get current year",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]

        expected = sglang_dsv4_prompt_ids(tokenizer, messages, tools, thinking=thinking)
        tito = get_tito_tokenizer(
            tokenizer,
            tokenizer_type=TITOTokenizerType.DEEPSEEKV4,
            chat_template_kwargs={"enable_thinking": thinking},
        )

        actual = tito.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, template_args=tito.default_template_args(tools)
        )
        assert actual == expected

    @pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
    def test_prompt_ids_match_sglang_dsv4_with_tool_history(self, thinking):
        # [FUNCTION DOC] Parametrized chat/thinking. A system message, user question, assistant
        # tool call (arguments JSON text with non-ASCII kept) and tool result, with a weather tool:
        # SGLang V4 reference ids equal the V4 TITO tokenizer ids.
        tokenizer = _get_deepseek_v4_tokenizer()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather in Beijing?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Beijing"}, ensure_ascii=False),
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temperature": 25}', "tool_call_id": "call_1"},
        ]

        expected = sglang_dsv4_prompt_ids(tokenizer, messages, tools, thinking=thinking)
        tito = get_tito_tokenizer(
            tokenizer,
            tokenizer_type=TITOTokenizerType.DEEPSEEKV4,
            chat_template_kwargs={"enable_thinking": thinking},
        )

        actual = tito.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, template_args=tito.default_template_args(tools)
        )
        assert actual == expected


class TestMessageMatches:
    # [CLASS DOC] Tests of the strict message matcher against a stored SGLang wire-format
    # assistant tool-call message (index 0, null reasoning) and a rebuilt replay message.
    """strict_message_matches compares template-relevant content, not wire idiosyncrasies."""

    STORED_SGLANG_WIRE = {
        "role": "assistant",
        "content": "",
        "reasoning_content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "index": 0,
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    }

    def _rebuilt(self, **tool_call_overrides):
        # [FUNCTION DOC] HELPER: Build the replayed assistant message: empty content and one tool
        # call with id "call_abc", type function, weather function with Paris arguments, no index,
        # then the keyword overrides merged over the call (an override can add an index or change
        # the id).
        tool_call = {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            **tool_call_overrides,
        }
        return {"role": "assistant", "content": "", "tool_calls": [tool_call]}

    @pytest.mark.parametrize("index_value", [0, 7, None], ids=["int", "other-int", "null"])
    def test_stream_rebuilt_replay_matches_sglang_wire(self, index_value):
        # [FUNCTION DOC] Parametrized over stored index 0, 7 and null; the strict matcher accepts
        # the stored message against the rebuilt one (no index) regardless of the stored index.
        stored = copy.deepcopy(self.STORED_SGLANG_WIRE)
        stored["tool_calls"][0]["index"] = index_value
        assert strict_message_matches(stored, self._rebuilt())

    def test_null_index_matches_absent_index(self):
        # [FUNCTION DOC] A rebuilt message with a null index matches the stored message.
        assert strict_message_matches(self.STORED_SGLANG_WIRE, self._rebuilt(index=None))

    def test_null_tool_call_id_does_not_match_absent_id(self):
        # [FUNCTION DOC] Stored id null versus a rebuilt call with the id key removed: the strict
        # matcher rejects (null id is not equal to absent id).
        stored = copy.deepcopy(self.STORED_SGLANG_WIRE)
        stored["tool_calls"][0]["id"] = None
        rebuilt = self._rebuilt()
        del rebuilt["tool_calls"][0]["id"]
        assert not strict_message_matches(stored, rebuilt)

    def test_different_arguments_still_mismatch(self):
        # [FUNCTION DOC] Rebuilt arguments naming Rome instead of Paris are rejected.
        rebuilt = self._rebuilt()
        rebuilt["tool_calls"][0]["function"] = {"name": "get_weather", "arguments": '{"city": "Rome"}'}
        assert not strict_message_matches(self.STORED_SGLANG_WIRE, rebuilt)

    def test_different_tool_call_id_still_mismatches(self):
        # [FUNCTION DOC] A rebuilt call with a different id is rejected.
        rebuilt = self._rebuilt(id="call_zzz")
        assert not strict_message_matches(self.STORED_SGLANG_WIRE, rebuilt)

    def test_content_mismatch_still_detected(self):
        # [FUNCTION DOC] A rebuilt message with non-empty different content is rejected.
        rebuilt = self._rebuilt()
        rebuilt["content"] = "different"
        assert not strict_message_matches(self.STORED_SGLANG_WIRE, rebuilt)
