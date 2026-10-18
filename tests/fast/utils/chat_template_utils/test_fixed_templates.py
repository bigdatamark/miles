# [MODULE DOC] Tests for resolve_fixed_chat_template and the FixedTemplate registration of the
# token-in-token-out (TITO) tokenizer families. A table maps each non-default tokenizer type to the
# expected fixed template file name (or none when the native template is used) and the extra
# template kwargs that keep rendering append-only (preserve/clear thinking switches and so on).
# Checks: the table covers every family, resolution returns existing template files and kwargs,
# defaults of FixedTemplate, rejection of unknown append roles, restricted role sets of some
# families, string family names accepted, returned kwargs being copies, registered kwargs overriding
# conflicting launch values, and tool-argument serialization of the Qwen 3.5 and 3.6 templates.
"""Unit tests for ``resolve_fixed_chat_template`` — one template per family.

Each ``TITOTokenizer`` subclass registers a single ``FIXED_TEMPLATE``:
template path, required kwargs, and the role surface that renderer supports.
"""

import os
from unittest.mock import MagicMock

import pytest
from miles.utils.chat_template_utils import TEMPLATE_DIR, TITOTokenizerType, resolve_fixed_chat_template
from miles.utils.chat_template_utils.template import apply_chat_template_from_str
from miles.utils.chat_template_utils.tito_tokenizer import (
    ALL_APPEND_ROLES,
    DeepSeekV4TITOTokenizer,
    FixedTemplate,
    MinimaxM25TITOTokenizer,
    MinimaxM27TITOTokenizer,
    Qwen3TITOTokenizer,
    Qwen35TITOTokenizer,
    Qwen36TITOTokenizer,
    Qwen38SmallTITOTokenizer,
    TITOTokenizer,
)

_EXPECTED_FIXED_TEMPLATES = {
    TITOTokenizerType.QWEN3: ("qwen3_fixed.jinja", {"clear_thinking": False}),
    TITOTokenizerType.QWEN35: ("qwen3.5_fixed.jinja", {"preserve_thinking": True}),
    TITOTokenizerType.QWEN36: ("qwen3.6_fixed.jinja", {"preserve_thinking": True}),
    TITOTokenizerType.QWEN38_SMALL: (
        "qwen3.8_small_and_flash_next_fixed.jinja",
        {"preserve_thinking": True},
    ),
    TITOTokenizerType.QWEN4_EXP: (
        "qwen3.8_small_and_flash_next_fixed.jinja",
        {"preserve_thinking": True},
    ),
    TITOTokenizerType.QWENNEXT: ("qwen3_thinking_2507_and_next_fixed.jinja", {"clear_thinking": False}),
    TITOTokenizerType.GLM47: (None, {"clear_thinking": False}),
    TITOTokenizerType.GLM53: (None, {"clear_thinking": False, "enable_thinking": True}),
    TITOTokenizerType.NEMOTRON3: (None, {"truncate_history_thinking": False}),
    TITOTokenizerType.KIMI25: ("kimi_k25_fixed.jinja", {"preserve_thinking": True}),
    TITOTokenizerType.KIMI26: (None, {"preserve_thinking": True}),
    TITOTokenizerType.MINIMAX_M25: ("minimax_m25_fixed.jinja", {"clear_thinking": False}),
    TITOTokenizerType.MINIMAX_M27: ("minimax_m27_fixed.jinja", {"clear_thinking": False}),
    TITOTokenizerType.DEEPSEEKV32: (None, {"drop_thinking": False}),
    TITOTokenizerType.DEEPSEEKV4: (None, {"drop_thinking": False}),
    TITOTokenizerType.DEEPSEEKV41: (None, {"drop_thinking": False}),
    TITOTokenizerType.INKLING: ("inkling_fixed.jinja", {}),
}


def test_every_non_default_family_is_covered():
    # [FUNCTION DOC] Asserts the expectation table's keys equal the set of all tokenizer types
    # except the default one, so a new family forces a new row.
    # New families must register a FIXED_TEMPLATE and take a row here.
    assert set(_EXPECTED_FIXED_TEMPLATES) == set(TITOTokenizerType) - {TITOTokenizerType.DEFAULT}


@pytest.mark.parametrize(
    "tito_model", list(_EXPECTED_FIXED_TEMPLATES), ids=[t.value for t in _EXPECTED_FIXED_TEMPLATES]
)
def test_family_resolves_template_and_preserve_think_kwargs(tito_model):
    # [FUNCTION DOC] Parametrized over every family in the table. Resolves the fixed template; if
    # the table expects no template file asserts the path is none, otherwise asserts the path equals
    # the template directory joined with the expected file name and that the file exists. Asserts the
    # returned kwargs equal the expected kwargs exactly.
    # Every family pins its preserve-think kwargs unconditionally, so renders
    # stay append-only regardless of which roles the harness appends.
    expected_template, expected_kwargs = _EXPECTED_FIXED_TEMPLATES[tito_model]
    path, kwargs = resolve_fixed_chat_template(tito_model)
    if expected_template is None:
        assert path is None
    else:
        assert path == str(TEMPLATE_DIR / expected_template)
        assert os.path.isfile(path)
    assert kwargs == expected_kwargs


def test_default_family_uses_native_template():
    # [FUNCTION DOC] The default family resolves to no template path and empty kwargs.
    assert resolve_fixed_chat_template(TITOTokenizerType.DEFAULT) == (None, {})


def test_fixed_template_defaults_to_all_roles():
    # [FUNCTION DOC] A default-constructed fixed template allows all append roles as an immutable
    # set, and equals the base tokenizer class's registered fixed template.
    fixed = FixedTemplate()
    assert fixed.allowed_append_roles == ALL_APPEND_ROLES
    assert isinstance(fixed.allowed_append_roles, frozenset)
    assert TITOTokenizer.FIXED_TEMPLATE == fixed


def test_fixed_template_rejects_unknown_role():
    # [FUNCTION DOC] Constructing a fixed template whose allowed roles include "developer" raises a
    # value error about unknown allowed append roles.
    with pytest.raises(ValueError, match="Unknown FixedTemplate allowed_append_roles"):
        FixedTemplate(allowed_append_roles=frozenset({"developer"}))


@pytest.mark.parametrize(
    "tokenizer_cls",
    [
        DeepSeekV4TITOTokenizer,
        Qwen35TITOTokenizer,
        Qwen36TITOTokenizer,
        Qwen38SmallTITOTokenizer,
        MinimaxM25TITOTokenizer,
        MinimaxM27TITOTokenizer,
    ],
)
def test_restricted_fixed_template_excludes_mid_session_system(tokenizer_cls):
    # [FUNCTION DOC] Parametrized over the DeepSeek V4, Qwen 3.5, 3.6, 3.8-small and MiniMax M2.5 and
    # M2.7 tokenizer classes; asserts each allows exactly the tool, user and assistant append roles.
    assert tokenizer_cls.FIXED_TEMPLATE.allowed_append_roles == frozenset({"tool", "user", "assistant"})


def test_string_tito_model_accepted():
    # [FUNCTION DOC] Resolving with the string "qwen3" equals resolving with the Qwen3 enum member.
    assert resolve_fixed_chat_template("qwen3") == resolve_fixed_chat_template(TITOTokenizerType.QWEN3)


def test_kwargs_are_copied_not_shared(monkeypatch):
    # [FUNCTION DOC] Temporarily replaces the Qwen3 class registration with a template-less one
    # carrying clear-thinking false, resolves it, mutates the returned kwargs, and asserts the
    # registered extra kwargs are unchanged.
    # Mutating the returned kwargs must not leak into the registration.
    monkeypatch.setattr(
        Qwen3TITOTokenizer,
        "FIXED_TEMPLATE",
        FixedTemplate(template=None, extra_kwargs={"clear_thinking": False}),
    )
    _path, kwargs = resolve_fixed_chat_template(TITOTokenizerType.QWEN3)
    kwargs["clear_thinking"] = True
    assert Qwen3TITOTokenizer.FIXED_TEMPLATE.extra_kwargs == {"clear_thinking": False}


@pytest.mark.parametrize(
    ("tokenizer_cls", "chat_template_kwargs"),
    [
        (Qwen3TITOTokenizer, {"clear_thinking": True}),
        (Qwen38SmallTITOTokenizer, {"preserve_thinking": False}),
    ],
)
def test_registered_kwargs_override_conflicting_launch_values(tokenizer_cls, chat_template_kwargs):
    # [FUNCTION DOC] Parametrized over Qwen3 with clear-thinking true and Qwen 3.8-small with
    # preserve-thinking false. Builds the tokenizer wrapper around a mock tokenizer whose encode
    # returns [1] and passes the conflicting kwargs; asserts the stored kwargs equal the class's
    # registered extra kwargs (registration wins).
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1]
    tito = tokenizer_cls(tokenizer, chat_template_kwargs=chat_template_kwargs)
    assert tito.chat_template_kwargs == tokenizer_cls.FIXED_TEMPLATE.extra_kwargs


@pytest.mark.parametrize(
    ("family", "expected_boolean", "expected_nothing"),
    [
        (TITOTokenizerType.QWEN35, "False", "None"),
        (TITOTokenizerType.QWEN36, "false", "null"),
    ],
)
def test_qwen35_and_qwen36_preserve_family_tool_argument_serialization(family, expected_boolean, expected_nothing):
    # [FUNCTION DOC] Parametrized over Qwen 3.5 (boolean rendered "False", null rendered "None")
    # and Qwen 3.6 ("false", "null"). Reads the resolved template file, renders a user message plus
    # an assistant tool call with arguments of string, boolean, number, null, array and object
    # types (no generation prompt, family kwargs applied), and asserts each parameter block: string
    # as plain text, boolean and null per the family's spelling, number as 3, array as "[1, 2]" and
    # object as compact JSON.
    template_path, kwargs = resolve_fixed_chat_template(family)
    assert template_path is not None
    with open(template_path, encoding="utf-8") as template_file:
        chat_template = template_file.read()
    rendered = apply_chat_template_from_str(
        chat_template,
        [
            {"role": "user", "content": "call"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "arguments": {
                                "string": "value",
                                "boolean": False,
                                "number": 3,
                                "nothing": None,
                                "array": [1, 2],
                                "object": {"a": 1},
                            },
                        },
                    }
                ],
            },
        ],
        add_generation_prompt=False,
        **kwargs,
    )
    assert "<parameter=string>\nvalue\n</parameter>" in rendered
    assert f"<parameter=boolean>\n{expected_boolean}\n</parameter>" in rendered
    assert "<parameter=number>\n3\n</parameter>" in rendered
    assert f"<parameter=nothing>\n{expected_nothing}\n</parameter>" in rendered
    assert "<parameter=array>\n[1, 2]\n</parameter>" in rendered
    assert '<parameter=object>\n{"a": 1}\n</parameter>' in rendered
