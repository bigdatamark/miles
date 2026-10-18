# [MODULE DOC] Tests that rendering with a pretokenized prefix (token ids of the first N messages
# supplied separately) yields the same token ids as the standard apply-chat-template path
# (the append-only / prefix invariant), for many chat templates. A module-level table lists
# templates (fixed templates resolved from the TITO families and HF-native templates downloaded at
# import time, so importing needs network or cache access), each with: name, template text, whether it
# supports thinking, the set of roles that may be appended, and extra template kwargs. Parameters
# are generated as template x selected trajectory case x enable-thinking variant. Also contains
# negative tests (original unfixed Qwen templates must show a prefix mismatch; cutting before the
# last user turn of a multi-user-turn thinking trajectory must mismatch for templates that compress
# older reasoning) and a matrix of injected system/assistant append shapes per TITO family.
# KNOWN ISSUE: the table rows "qwen3_thinking_2507_fixed" and "qwen3_next_thinking_fixed" use the
# identical template, role set and kwargs, so one is redundant.
"""
Unit tests for the pretokenized chat completion path.

Tests that using pretokenized_token_ids + pretokenized_num_message produces
identical token IDs as the standard apply_chat_template path.

Ported from sglang test/unit/test_pretokenized_chat.py.
"""

from copy import deepcopy

import pytest

from miles.utils.chat_template_utils import TITOTokenizerType, resolve_fixed_chat_template
from miles.utils.chat_template_utils.template import apply_chat_template_from_str, load_hf_chat_template
from miles.utils.test_utils.chat_template_verify import (
    CaseSpec,
    assert_pretokenized_equals_standard,
    enable_thinking_variants,
    format_case_id,
    select_cases,
    simulate_pretokenized_path,
)
from miles.utils.test_utils.mock_trajectories import (
    MultiTurnTrajectory,
    MultiUserTurnThinkingTrajectory,
    SingleToolTrajectory,
    last_user_index,
)


def _load_fixed(tito_model: TITOTokenizerType) -> str:
    # [FUNCTION DOC] HELPER: Resolve the fixed chat template of a TITO family and return its file
    # text. Asserts that the family has a template file (a message names the family otherwise);
    # kwargs from the resolution are discarded. Reads the file.
    path, _kwargs = resolve_fixed_chat_template(tito_model)
    assert path is not None, f"resolve_fixed_chat_template should resolve {tito_model.value}"
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Per-template capability declarations
# ---------------------------------------------------------------------------
#
# Each entry: (name, content, supports_thinking, allowed_append_roles, extra_template_kwargs)
#
# Role sets only filter this test's shared non-assistant trajectories; template
# resolution is role-independent. Rows without preserve-think kwargs are comparison
# baselines, and system trajectories run only for rows that include ``system``.

_TEMPLATES: list[tuple[str, str, bool, frozenset[str], dict]] = [
    # Qwen fixed-template comparison rows
    ("qwen3_fixed", _load_fixed(TITOTokenizerType.QWEN3), True, frozenset({"tool"}), {}),
    (
        "qwen3_fixed_clear_thinking_off",
        _load_fixed(TITOTokenizerType.QWEN3),
        True,
        frozenset({"tool", "user"}),
        {"clear_thinking": False},
    ),
    ("qwen3.5_3.6_fixed", _load_fixed(TITOTokenizerType.QWEN35), True, frozenset({"tool"}), {}),
    (
        "qwen3.5_3.6_fixed_preserve_thinking",
        _load_fixed(TITOTokenizerType.QWEN35),
        True,
        frozenset({"tool", "user"}),
        {"preserve_thinking": True},
    ),
    (
        "qwen3.8_small_fixed_preserve_thinking",
        _load_fixed(TITOTokenizerType.QWEN38_SMALL),
        True,
        frozenset({"tool", "user"}),
        {"preserve_thinking": True, "reasoning_effort": "xhigh"},
    ),
    ("qwen3_thinking_2507_fixed", _load_fixed(TITOTokenizerType.QWENNEXT), True, frozenset({"tool"}), {}),
    ("qwen3_next_thinking_fixed", _load_fixed(TITOTokenizerType.QWENNEXT), True, frozenset({"tool"}), {}),
    (
        "qwen3_next_thinking_fixed_clear_thinking_off",
        _load_fixed(TITOTokenizerType.QWENNEXT),
        True,
        frozenset({"tool", "user"}),
        {"clear_thinking": False},
    ),
    # GLM thinking: default (tool only) + user-append variant with clear_thinking=False
    ("glm5", load_hf_chat_template("zai-org/GLM-5"), True, frozenset({"tool"}), {}),
    (
        "glm5_clear_thinking_off",
        load_hf_chat_template("zai-org/GLM-5"),
        True,
        frozenset({"tool", "user"}),
        {"clear_thinking": False},
    ),
    ("glm47_flash", load_hf_chat_template("zai-org/GLM-4.7-Flash"), True, frozenset({"tool"}), {}),
    (
        "glm47_flash_clear_thinking_off",
        load_hf_chat_template("zai-org/GLM-4.7-Flash"),
        True,
        frozenset({"tool", "user"}),
        {"clear_thinking": False},
    ),
    (
        "glm47_flash_clear_thinking_off_with_system",
        load_hf_chat_template("zai-org/GLM-4.7-Flash"),
        True,
        frozenset({"tool", "user", "system"}),
        {"clear_thinking": False},
    ),
    # Kimi K2: K2.5 needs patched jinja gating the "drop reasoning of prior
    # assistants once a non-tool-call assistant arrives" loop on
    # preserve_thinking=True; K2.6 already exposes that gate natively.
    # This matrix covers their tool/user trajectories.
    (
        "kimi_k25_fixed_preserve_thinking",
        _load_fixed(TITOTokenizerType.KIMI25),
        True,
        frozenset({"tool", "user"}),
        {"preserve_thinking": True},
    ),
    (
        "kimi_k26_preserve_thinking",
        load_hf_chat_template("moonshotai/Kimi-K2.6"),
        True,
        frozenset({"tool", "user"}),
        {"preserve_thinking": True},
    ),
    # MiniMax reasoning is gated by last_user_index, so user appends can strip
    # prior thinking. The matrix keeps HF-native tool-only baselines for comparison;
    # family auto-resolution uses FIXED_TEMPLATE with clear_thinking=False.
    ("minimax_m25", load_hf_chat_template("MiniMaxAI/MiniMax-M2.5"), True, frozenset({"tool"}), {}),
    (
        "minimax_m25_fixed_clear_thinking_off",
        _load_fixed(TITOTokenizerType.MINIMAX_M25),
        True,
        frozenset({"tool", "user"}),
        {"clear_thinking": False},
    ),
    ("minimax_m27", load_hf_chat_template("MiniMaxAI/MiniMax-M2.7"), True, frozenset({"tool"}), {}),
    (
        "minimax_m27_fixed_clear_thinking_off",
        _load_fixed(TITOTokenizerType.MINIMAX_M27),
        True,
        frozenset({"tool", "user"}),
        {"clear_thinking": False},
    ),
    # Nemotron uses its HF-native template with fixed kwargs. Family auto-resolution
    # pins truncate_history_thinking=False; the matrix keeps a native tool baseline.
    (
        "nemotron3",
        load_hf_chat_template("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"),
        True,
        frozenset({"tool"}),
        {},
    ),
    (
        "nemotron3_truncate_history_thinking_off",
        load_hf_chat_template("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"),
        True,
        frozenset({"tool", "user"}),
        {"truncate_history_thinking": False},
    ),
    (
        "nemotron3_truncate_history_thinking_off_with_system",
        load_hf_chat_template("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"),
        True,
        frozenset({"tool", "user", "system"}),
        {"truncate_history_thinking": False},
    ),
    # Other HF-native comparison rows: tool-only trajectories
    ("qwen3_instruct_2507", load_hf_chat_template("Qwen/Qwen3-4B-Instruct-2507"), False, frozenset({"tool"}), {}),
    ("qwen3_next_instruct", load_hf_chat_template("Qwen/Qwen3-Next-80B-A3B-Instruct"), False, frozenset({"tool"}), {}),
    ("qwen3_coder_next", load_hf_chat_template("Qwen/Qwen3-Coder-Next"), False, frozenset({"tool"}), {}),
    ("glm4", load_hf_chat_template("THUDM/glm-4-9b-chat"), False, frozenset({"tool"}), {}),
]

# Original (unfixed) HF templates referenced by negative tests
_ORIGINAL_TEMPLATES = {
    "qwen3_original": load_hf_chat_template("Qwen/Qwen3-0.6B"),
    "qwen3_thinking_2507": load_hf_chat_template("Qwen/Qwen3-4B-Thinking-2507"),
    "qwen3_next_thinking": load_hf_chat_template("Qwen/Qwen3-Next-80B-A3B-Thinking"),
}


def _build_pretokenized_params():
    # [FUNCTION DOC] HELPER: Build the pytest parameter list for the core test. For each table row,
    # select trajectory cases filtered by the row's allowed append roles (and restricted to
    # non-thinking trajectories when the template does not support thinking); thinking templates are
    # crossed with both enable-thinking values, non-thinking ones with the off variant only. Each
    # parameter is (template text, case, merged kwargs of the variant and the row's extra kwargs)
    # with an id made of the row name and the formatted case id.
    # Thinking templates: every selected trajectory × {enable_thinking=True, False}.
    # Non-thinking templates: only non-thinking trajectories, no enable_thinking kwarg.
    params = []
    for name, content, supports_thinking, allowed_roles, extra_kwargs in _TEMPLATES:
        cases = select_cases(
            allowed_append_roles=allowed_roles,
            is_thinking=None if supports_thinking else False,
        )
        variants = enable_thinking_variants("both" if supports_thinking else "off")
        for case in cases:
            for variant in variants:
                kwargs = {**variant, **extra_kwargs}
                ident = f"{name}-{format_case_id(case, kwargs)}"
                params.append(pytest.param(content, case, kwargs, id=ident))
    return params


# ===========================================================================
# Core tests: every (template, case, kwargs) tuple satisfies append-only
# ===========================================================================


@pytest.mark.parametrize("chat_template, case, kwargs", _build_pretokenized_params())
def test_pretokenized(chat_template: str, case: CaseSpec, kwargs: dict):
    # [FUNCTION DOC] Core parametrized test: runs the shared verification helper on a deep copy of
    # the case's messages, pretokenizing the first N messages of the case, with the case's tools and
    # the kwargs; the helper raises if the pretokenized path differs from the standard path.
    assert_pretokenized_equals_standard(
        chat_template=chat_template,
        messages=deepcopy(case.traj_cls.MESSAGES),
        pretokenized_num_message=case.pretokenize_n,
        tools=case.tools,
        **kwargs,
    )


# ===========================================================================
# Negative tests: original (unfixed) templates fail prefix invariant
# ===========================================================================

# (chat_template, trajectory_cls, pretokenize_n)
_MISMATCH_CASES = [
    pytest.param(_ORIGINAL_TEMPLATES["qwen3_original"], SingleToolTrajectory, 3, id="qwen3_original-single_tool"),
    pytest.param(_ORIGINAL_TEMPLATES["qwen3_original"], MultiTurnTrajectory, 3, id="qwen3_original-multi_turn"),
    pytest.param(
        _ORIGINAL_TEMPLATES["qwen3_thinking_2507"], SingleToolTrajectory, 3, id="qwen3_thinking_2507-single_tool"
    ),
    pytest.param(
        _ORIGINAL_TEMPLATES["qwen3_next_thinking"], SingleToolTrajectory, 3, id="qwen3_next_thinking-single_tool"
    ),
    pytest.param(
        _ORIGINAL_TEMPLATES["qwen3_next_thinking"], MultiTurnTrajectory, 3, id="qwen3_next_thinking-multi_turn"
    ),
]


@pytest.mark.parametrize("chat_template,trajectory_cls,pretokenize_n", _MISMATCH_CASES)
def test_original_template_prefix_mismatch(chat_template, trajectory_cls, pretokenize_n):
    # [FUNCTION DOC] Parametrized over original (unfixed) Qwen templates and trajectories, with
    # the first 3 messages pretokenized. Asserts the simulation raises a value error whose message
    # says "Prefix mismatch".
    """Original templates with loop.last cause prefix mismatch (our fix resolves this)."""
    with pytest.raises(ValueError, match="Prefix mismatch"):
        simulate_pretokenized_path(
            chat_template,
            deepcopy(trajectory_cls.MESSAGES),
            pretokenize_n,
            tools=trajectory_cls.TOOLS,
        )


# ===========================================================================
# Negative test: cross-user-turn thinking compression breaks prefix invariant
# ===========================================================================

# Pretokenizing BEFORE the last user turn in a multi-user-turn thinking
# trajectory fails because templates compress reasoning_content from earlier
# turns.  This is a known template limitation, not a bug in the fixed templates.
_CROSS_USER_THINKING_N = last_user_index(MultiUserTurnThinkingTrajectory.MESSAGES)


def _unique_thinking_templates():
    # [FUNCTION DOC] HELPER: Build parameters for the cross-user-turn negative test: the distinct
    # template texts (by content) of table rows that support thinking, skipping rows whose name
    # starts with the Qwen 3.8 prefix (they preserve reasoning by default) or the Kimi prefix
    # (they compress at a different boundary). Each parameter carries the template text and the
    # row name as id (first row wins for duplicate content).
    seen: set[str] = set()
    out = []
    for name, content, supports_thinking, _, _ in _TEMPLATES:
        if not supports_thinking:
            continue
        # Qwen3.8 preserves reasoning by default, so it has no cross-user
        # compression mismatch under the registered contract.
        if name.startswith("qwen3.8_"):
            continue
        # Kimi K2.5/K2.6 compress reasoning at the "first non-tool-call assistant"
        # boundary (single_tool_thinking trajectories), not at the "last user
        # message" boundary like qwen3/glm — so MultiUserTurnThinking's
        # cross-user pretokenize does not trip their compression.  This negative
        # test only applies to templates whose compression keys off user index.
        if name.startswith("kimi_"):
            continue
        if content in seen:
            continue
        seen.add(content)
        out.append(pytest.param(content, id=name))
    return out


@pytest.mark.parametrize("chat_template", _unique_thinking_templates())
@pytest.mark.parametrize("enable_thinking", [True, False], ids=["thinking_on", "thinking_off"])
def test_cross_user_turn_thinking_prefix_mismatch(chat_template, enable_thinking):
    # [FUNCTION DOC] Parametrized over the distinct thinking templates and thinking on/off.
    # Pretokenizes the multi-user-turn thinking trajectory up to the last user index and asserts the
    # simulation raises a value error saying "Prefix mismatch".
    """Thinking templates compress reasoning_content from earlier user turns, breaking prefix invariant."""
    with pytest.raises(ValueError, match="Prefix mismatch"):
        simulate_pretokenized_path(
            chat_template,
            deepcopy(MultiUserTurnThinkingTrajectory.MESSAGES),
            _CROSS_USER_THINKING_N,
            tools=MultiUserTurnThinkingTrajectory.TOOLS,
            enable_thinking=enable_thinking,
        )


# ---------------------------------------------------------------------------
# Intermediate system / injected-assistant appends — role-independent surface
# ---------------------------------------------------------------------------
#
# allowed_append_roles no longer participates in template resolution. Each
# family/shape cell must either remain append-only or match its fixed-template
# rejection contract.
# DeepSeek V3.2/V4 ride the encoder bridge and are covered in their own files.

_APPEND_ROLE_FAMILIES = [
    (TITOTokenizerType.QWEN3, None),
    (TITOTokenizerType.QWEN35, None),
    (TITOTokenizerType.QWEN36, None),
    (TITOTokenizerType.QWEN38_SMALL, None),
    (TITOTokenizerType.QWEN4_EXP, None),
    (TITOTokenizerType.QWENNEXT, None),
    (TITOTokenizerType.GLM47, "zai-org/GLM-4.7-Flash"),
    (TITOTokenizerType.GLM53, "zai-org/GLM-5.3"),
    (TITOTokenizerType.GLM53, "zai-org/GLM-5.3-Flash"),
    (TITOTokenizerType.KIMI25, None),
    (TITOTokenizerType.KIMI26, "moonshotai/Kimi-K2.6"),
    (TITOTokenizerType.MINIMAX_M25, None),
    (TITOTokenizerType.MINIMAX_M27, None),
    (TITOTokenizerType.NEMOTRON3, "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"),
]

_APPEND_SHAPES = {
    "system": [{"role": "system", "content": "mid-session system"}],
    "assistant": [{"role": "assistant", "content": "injected", "reasoning_content": ""}],
    "assistant_then_user": [
        {"role": "assistant", "content": "injected", "reasoning_content": ""},
        {"role": "user", "content": "next question"},
    ],
    "consecutive_assistants_then_user": [
        {"role": "assistant", "content": "first injected"},
        {"role": "assistant", "content": "second injected"},
        {"role": "user", "content": "next question"},
    ],
}


@pytest.mark.parametrize("shape", list(_APPEND_SHAPES), ids=list(_APPEND_SHAPES))
@pytest.mark.parametrize("family, hf_model_id", _APPEND_ROLE_FAMILIES, ids=[f.value for f, _ in _APPEND_ROLE_FAMILIES])
def test_appends_are_append_only_on_family_template(family, hf_model_id, shape):
    # [FUNCTION DOC] Parametrized over TITO families (with an HF model id when the family has no
    # fixed template file) and four append shapes (system, injected assistant, assistant then user,
    # two assistants then user). Resolves the family's fixed template and kwargs (or downloads the
    # HF template), renders a one-round history without generation prompt as the base, then:
    # for the Qwen 3.5, 3.6, 3.8-small and Qwen4-exp families with the system shape, asserts the
    # template raises a value error "System message must be at the beginning" and returns;
    # otherwise renders history plus the shape and asserts the result starts with the base text.
    path, kwargs = resolve_fixed_chat_template(family)
    if path is not None:
        with open(path, encoding="utf-8") as f:
            template = f.read()
    else:
        template = load_hf_chat_template(hf_model_id)
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
    ]

    base = apply_chat_template_from_str(template, history, add_generation_prompt=False, **kwargs)
    if (
        family
        in (
            TITOTokenizerType.QWEN35,
            TITOTokenizerType.QWEN36,
            TITOTokenizerType.QWEN38_SMALL,
            TITOTokenizerType.QWEN4_EXP,
        )
        and shape == "system"
    ):
        with pytest.raises(ValueError, match="System message must be at the beginning"):
            apply_chat_template_from_str(
                template,
                history + _APPEND_SHAPES[shape],
                add_generation_prompt=False,
                **kwargs,
            )
        return
    full = apply_chat_template_from_str(
        template, history + _APPEND_SHAPES[shape], add_generation_prompt=False, **kwargs
    )
    assert full.startswith(base)
