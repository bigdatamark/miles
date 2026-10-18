# [MODULE DOC] Tests for the token-in-token-out (TITO) tokenizer family
# (miles.utils.chat_template_utils.tito_tokenizer): per-family configuration, request argument
# resolution (chat-template kwargs and tools merging over launch values and recorded turn
# arguments), the merge of a pretokenized prefix with incrementally tokenized appended messages
# (boundary fix-ups per family), incremental tokenization of appended messages, completion
# post-processing hooks, append-only validation, and the factory. Tokenizers are real Hugging Face
# tokenizers (downloaded, cached in a module-level cache keyed by model and fixed template path),
# so most tests need network or model cache access; some use mock tokenizers. The module docstring
# below describes the intended test classes. Trajectory-based tests derive split positions
# automatically from message structure of the mock trajectories listed in a module-level list.
"""Tests for TITOTokenizer: merge_tokens boundary logic, incremental tokenization, and factory.

## Test structure

TestConfig
    Smoke-checks that each subclass stores the correct model-specific config
    (assistant_start_str, trailing_token_ids, max_trim_tokens) and propagates
    them to the comparator.  These are NOT behavioral tests — they guard
    against accidental config regressions when modifying __init__.

TestCompletionPostprocess
    Verifies that the default completion hook is an identity operation.

TestMergeTokensBoundary
    Unit tests for the core merge_tokens boundary logic, using *synthetic*
    prefix IDs ([100, 200, ...]) so the assertions are purely about prefix
    manipulation — not about template rendering.

    Why synthetic IDs?  merge_tokens is: ``prefix + [boundary fix] + incremental``.
    The incremental part comes from tokenize_additional_messages (tested
    separately); boundary logic depends only on the last token of the prefix.
    Synthetic IDs isolate this and make failures trivially diagnosable.

    Covers three subclass behaviors:
    - Qwen3: inserts ``\\n`` when prefix ends with ``<|im_end|>`` (model stops
      at im_end without the trailing newline the template expects).
    - GLM47: strips trailing ``<|observation|>`` or ``<|user|>`` (model emits
      the stop token, but the template also emits it as the next turn's opener).
    - Default: plain concatenation (no boundary handling).

TestTokenizeAdditional
    Behavioral tests for tokenize_additional_messages — the single synthetic-prefix diff that computes incremental token IDs for the complete appended suffix.

    ``test_produces_nonempty_incremental`` is parametrized over:
      _TOOL_TRAJECTORIES (trajectory classes) × _TITO_MODELS (qwen3, glm47)
    Split points are auto-detected by _find_tito_splits from message structure,
    so adding a trajectory to _TOOL_TRAJECTORIES automatically extends coverage.

    Remaining tests cover complete-appendix rendering, generation-prompt
    timing, stored tool-call preservation, merge structure preservation, and
    append-only validation (reject prefix mutation, fewer messages, or
    forbidden roles).

TestFactory
    get_tito_tokenizer factory: string/enum dispatch, invalid input handling.
"""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from transformers import AutoTokenizer

from miles.utils.chat_template_utils import (
    TEMPLATE_DIR,
    MismatchType,
    apply_chat_template,
    apply_chat_template_from_str,
    resolve_fixed_chat_template,
)
from miles.utils.chat_template_utils.tito_tokenizer import (
    ALL_APPEND_ROLES,
    DeepSeekV4TITOTokenizer,
    DeepSeekV32TITOTokenizer,
    FixedTemplate,
    GLM47TITOTokenizer,
    GLM53TITOTokenizer,
    InklingTITOTokenizer,
    Kimi25TITOTokenizer,
    Kimi26TITOTokenizer,
    Nemotron3TITOTokenizer,
    Qwen3TITOTokenizer,
    Qwen35TITOTokenizer,
    Qwen36TITOTokenizer,
    Qwen38SmallTITOTokenizer,
    QwenNextTITOTokenizer,
    TITOTokenizer,
    TITOTokenizerType,
    _build_dummy_assistant,
    extract_template_args,
    get_tito_tokenizer,
)
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.mock_trajectories import (
    IntermediateSystemTrajectory,
    LongChainTrajectory,
    MultiToolSingleTurnTrajectory,
    MultiTurnTrajectory,
    ParallelToolsTrajectory,
    RetrySystemTrajectory,
    SingleToolThinkingTrajectory,
    SingleToolTrajectory,
)

# ---------------------------------------------------------------------------
# Tokenizer cache
# ---------------------------------------------------------------------------

_TOK_CACHE: dict[tuple[str, str | None], AutoTokenizer] = {}


def _get_tokenizer(model_id: str, tito_type: TITOTokenizerType | None = None) -> AutoTokenizer:
    # [FUNCTION DOC] HELPER: Return a cached tokenizer for the model, optionally loaded with the
    # fixed chat template file of a TITO family. The cache key is the pair of model id and resolved
    # template path (none when no family given). On a miss, loads through the project's tokenizer
    # loader with remote code trusted (may download). The cache is never cleared.
    chat_template_path = resolve_fixed_chat_template(tito_type)[0] if tito_type is not None else None
    cache_key = (model_id, chat_template_path)
    if cache_key not in _TOK_CACHE:
        _TOK_CACHE[cache_key] = load_tokenizer(
            model_id,
            chat_template_path=chat_template_path,
            trust_remote_code=True,
        )
    return _TOK_CACHE[cache_key]


# ---------------------------------------------------------------------------
# Fixtures — model-specific TITO tokenizers
#
# `tito` is parametrized over all supported models; use it for tests that
# should run against every model.  Named fixtures (qwen3_tito, etc.) are
# for tests specific to one subclass's boundary logic.
# ---------------------------------------------------------------------------

_TITO_MODELS: dict[str, tuple[str, type[TITOTokenizer], TITOTokenizerType]] = {
    "qwen3": ("Qwen/Qwen3-4B", Qwen3TITOTokenizer, TITOTokenizerType.QWEN3),
    "glm47": ("zai-org/GLM-4.7-Flash", GLM47TITOTokenizer, TITOTokenizerType.GLM47),
}


@pytest.fixture(params=list(_TITO_MODELS.keys()))
def tito(request) -> TITOTokenizer:
    # [FUNCTION DOC] HELPER: Fixture parametrized over the models table (Qwen3-4B and GLM-4.7-Flash).
    # Returns the family's TITO tokenizer class built on the cached tokenizer (with the family's
    # fixed template) and chat template kwargs clear_thinking false.
    model_id, cls, tito_type = _TITO_MODELS[request.param]
    return cls(
        _get_tokenizer(model_id, tito_type),
        chat_template_kwargs={"clear_thinking": False},
    )


@pytest.fixture
def qwen3_tito() -> Qwen3TITOTokenizer:
    # [FUNCTION DOC] HELPER: Fixture returning a Qwen3 TITO tokenizer over the Qwen3-4B tokenizer
    # with the Qwen3 fixed template and clear_thinking false.
    return Qwen3TITOTokenizer(
        _get_tokenizer("Qwen/Qwen3-4B", TITOTokenizerType.QWEN3),
        chat_template_kwargs={"clear_thinking": False},
    )


@pytest.fixture
def glm47_tito() -> GLM47TITOTokenizer:
    # [FUNCTION DOC] HELPER: Fixture returning a GLM 4.7 TITO tokenizer over the GLM-4.7-Flash
    # tokenizer with clear_thinking false.
    return GLM47TITOTokenizer(
        _get_tokenizer("zai-org/GLM-4.7-Flash", TITOTokenizerType.GLM47),
        chat_template_kwargs={"clear_thinking": False},
    )


@pytest.fixture
def default_tito() -> TITOTokenizer:
    # [FUNCTION DOC] HELPER: Fixture returning the base TITO tokenizer (no family boundary logic)
    # over the plain Qwen3-4B tokenizer.
    return TITOTokenizer(_get_tokenizer("Qwen/Qwen3-4B"))


# ---------------------------------------------------------------------------
# Trajectory parametrization
#
# Instead of relying on PRETOKENIZE_POSITIONS (which serves the pretokenized
# *chat* tests), we derive TITO split points directly from message structure:
# every assistant(tool_calls) followed by a tool/system message is a valid
# split.  This way new trajectories get coverage automatically.
#
# To extend: add a trajectory class to _TOOL_TRAJECTORIES.
# To add a model: add an entry to _TITO_MODELS above.
# ---------------------------------------------------------------------------


def _find_tito_splits(traj_cls) -> list[int]:
    # [FUNCTION DOC] HELPER: For a mock trajectory class return the indices i+1 at which message i
    # is an assistant message with tool calls and message i+1 exists and has role tool or system
    # (the index of the first appended message). Empty list if none.
    """Find TITO split positions from message structure.

    A valid split is at index ``i+1`` whenever ``messages[i]`` is an assistant
    message with tool_calls and ``messages[i+1]`` is a tool or system message.
    Returns a list of such positions (the index of the first appended message).
    """
    msgs = traj_cls.MESSAGES
    splits = []
    for i, msg in enumerate(msgs):
        if (
            msg.get("role") == "assistant"
            and msg.get("tool_calls")
            and i + 1 < len(msgs)
            and msgs[i + 1].get("role") in ("tool", "system")
        ):
            splits.append(i + 1)
    return splits


def _split_at(traj_cls, pos: int):
    # [FUNCTION DOC] HELPER: Split a trajectory at the position. Returns (old messages = those
    # before pos, new messages = those before the next assistant message at or after pos, i.e. old
    # plus the following non-assistant run, and the trajectory's tools).
    """Split trajectory at *pos* into ``(old_msgs, new_msgs, tools)``.

    ``old_msgs = messages[:pos]`` — the pretokenized prefix (ends with assistant turn).
    ``new_msgs`` extends through all subsequent non-assistant messages
    (tool/user/system), stopping before the next assistant turn.
    """
    msgs = traj_cls.MESSAGES
    end = pos
    while end < len(msgs) and msgs[end].get("role") != "assistant":
        end += 1
    return msgs[:pos], msgs[:end], traj_cls.TOOLS


_TOOL_TRAJECTORIES = [
    SingleToolTrajectory,  # 1 tool call, 1 response
    MultiTurnTrajectory,  # 2 sequential tool turns
    MultiToolSingleTurnTrajectory,  # 2 parallel tool calls (weather + date)
    ParallelToolsTrajectory,  # 3 parallel tool calls
    LongChainTrajectory,  # 3 sequential turns (weather → date → weather)
    RetrySystemTrajectory,  # tool + system retry injection mid-conversation
    IntermediateSystemTrajectory,  # system messages interleaved with tool turns
    SingleToolThinkingTrajectory,  # tool call with reasoning_content
]

_TRAJ_CASES = [
    pytest.param(traj_cls, pos, id=f"{traj_cls.__name__}-N{pos}")
    for traj_cls in _TOOL_TRAJECTORIES
    for pos in _find_tito_splits(traj_cls)
]

# ---------------------------------------------------------------------------
# TestConfig — subclass configuration smoke-checks
# ---------------------------------------------------------------------------


class TestConfig:
    # [CLASS DOC] Smoke checks of per-family configuration stored at construction.
    """Each subclass stores the correct model-specific configuration at init."""

    def test_qwen3(self, qwen3_tito: Qwen3TITOTokenizer):
        # [FUNCTION DOC] The Qwen3 tokenizer's assistant start string is "<|im_start|>assistant" and
        # its newline token id is among the trailing token ids.
        assert qwen3_tito._assistant_start_str == "<|im_start|>assistant"
        assert qwen3_tito._newline_id in qwen3_tito.trailing_token_ids

    def test_glm47(self, glm47_tito: GLM47TITOTokenizer):
        # [FUNCTION DOC] The GLM 4.7 tokenizer's assistant start string is "<|assistant|>", the
        # observation and user token ids are trailing ids, and the maximum trim tokens is 1.
        assert glm47_tito._assistant_start_str == "<|assistant|>"
        assert glm47_tito._observation_id in glm47_tito.trailing_token_ids
        assert glm47_tito._user_id in glm47_tito.trailing_token_ids
        assert glm47_tito.max_trim_tokens == 1

    def test_default(self, default_tito: TITOTokenizer):
        # [FUNCTION DOC] The base tokenizer has no assistant start string and an empty trailing id set.
        assert default_tito._assistant_start_str is None
        assert default_tito.trailing_token_ids == frozenset()

    @pytest.mark.parametrize(
        "chat_template_kwargs, expected",
        [
            pytest.param({}, True, id="default-thinking"),
            pytest.param({"enable_thinking": False}, False, id="disable-via-miles-kwarg"),
            pytest.param({"thinking": False}, False, id="disable-via-sglang-kwarg"),
            pytest.param(
                {"enable_thinking": False, "thinking": True},
                False,
                id="miles-kwarg-precedes-sglang-kwarg",
            ),
            pytest.param(
                {"thinking_mode": "thinking", "thinking": False},
                True,
                id="explicit-mode-precedes-sglang-kwarg",
            ),
            pytest.param({"thinking_mode": "chat"}, False, id="explicit-chat-mode"),
        ],
    )
    def test_deepseek_v32_forwards_effective_thinking_mode(self, chat_template_kwargs, expected):
        # [FUNCTION DOC] Parametrized over launch kwargs: default (thinking on), enable-thinking
        # false, thinking false, enable-thinking taking precedence over thinking, explicit thinking
        # mode taking precedence over thinking, explicit chat mode. Builds the V3.2 TITO tokenizer on
        # a mock tokenizer (two token id lookups returning 1 and 2) and asserts the stored kwargs'
        # thinking value equals the expected boolean.
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.side_effect = [1, 2]

        tito = DeepSeekV32TITOTokenizer(tokenizer, chat_template_kwargs=chat_template_kwargs)

        assert tito.chat_template_kwargs["thinking"] is expected

    @pytest.mark.parametrize("tito_cls", [DeepSeekV32TITOTokenizer, DeepSeekV4TITOTokenizer])
    @pytest.mark.parametrize(
        "startup_thinking, request_kwargs, expected",
        [
            pytest.param(False, {"thinking": True}, True, id="enable-via-thinking"),
            pytest.param(True, {"enable_thinking": False}, False, id="disable-via-enable-thinking"),
            pytest.param(True, {"thinking_mode": "chat"}, False, id="disable-via-mode"),
            pytest.param(False, {}, False, id="omitted-alias-inherits"),
            pytest.param(False, {"enable_thinking": None}, True, id="null-enable-thinking-replaces-base"),
            pytest.param(False, {"thinking": None}, True, id="null-thinking-replaces-base"),
            pytest.param(True, {"thinking_mode": None}, False, id="null-mode-replaces-base"),
            pytest.param(
                True,
                {"enable_thinking": False, "thinking": True},
                False,
                id="request-enable-thinking-precedes-thinking",
            ),
            pytest.param(
                False, {"thinking_mode": "thinking", "enable_thinking": False}, True, id="request-mode-precedes-toggle"
            ),
            pytest.param(
                True, {"enable_thinking": None, "thinking": False}, False, id="null-toggle-falls-through-to-thinking"
            ),
        ],
    )
    def test_deepseek_request_aliases_override_startup_mode(
        self, tito_cls, startup_thinking, request_kwargs, expected
    ):
        # [FUNCTION DOC] Parametrized over the V3.2 and V4 classes and ten (startup thinking,
        # request kwargs, expected thinking) combinations covering the aliases thinking,
        # enable-thinking and thinking mode, omitted aliases inheriting the startup value, explicit
        # nulls replacing the base, and precedence rules (mode over toggles, enable-thinking over
        # thinking, null toggle falling through). Builds the tokenizer with a startup
        # enable-thinking value and a custom option, resolves request args for a request carrying the
        # kwargs, and asserts: the result has kwargs with drop-thinking false, the expected thinking
        # value and the custom option, and tools null; the tokenizer's own stored kwargs keep the
        # startup values; and the request kwargs mapping is not mutated.
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 1
        startup_tito = tito_cls(
            tokenizer, chat_template_kwargs={"enable_thinking": startup_thinking, "custom_option": "launch"}
        )
        original_kwargs = dict(request_kwargs)

        request_args = startup_tito.resolve_request_args({"chat_template_kwargs": request_kwargs}, turn_args=None)

        assert request_args == {
            "chat_template_kwargs": {"drop_thinking": False, "thinking": expected, "custom_option": "launch"},
            "tools": None,
        }
        assert startup_tito.chat_template_kwargs == {
            "drop_thinking": False,
            "thinking": startup_thinking,
            "custom_option": "launch",
        }
        assert request_kwargs == original_kwargs

    def test_comparator_inherits_trailing_ids(self, qwen3_tito: Qwen3TITOTokenizer):
        # [FUNCTION DOC] The comparator created by the tokenizer has a trim set of trailing ids
        # equal to the tokenizer's trailing token ids.
        """create_comparator propagates trailing_token_ids to the comparator's trim set."""
        comp = qwen3_tito.create_comparator()
        assert comp._trim_trailing_ids == set(qwen3_tito.trailing_token_ids)


class TestResolveRequestArgs:
    # [CLASS DOC] Tests of resolve_request_args: how a request's chat-template kwargs and tools are
    # combined with the launch kwargs and with arguments recorded for earlier turns of a session.
    # Class constants: launch kwargs (enable-thinking false), a weather tool and another tool.
    LAUNCH = {"enable_thinking": False}
    TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    OTHER_TOOLS = [{"type": "function", "function": {"name": "get_time"}}]

    def test_new_root_merges_request_kwargs_over_the_launch_and_takes_the_request_tools(self):
        # [FUNCTION DOC] With no recorded turn, request kwargs (enable-thinking true) override the
        # launch value and the request tools are taken as given.
        launch = TITOTokenizer(MagicMock(), chat_template_kwargs=self.LAUNCH)

        args = launch.resolve_request_args(
            {"chat_template_kwargs": {"enable_thinking": True}, "tools": self.TOOLS}, turn_args=None
        )

        assert args == {"chat_template_kwargs": {"enable_thinking": True}, "tools": self.TOOLS}

    def test_request_without_kwargs_or_tools_renders_like_the_launch(self):
        # [FUNCTION DOC] A request without kwargs or tools resolves to the launch kwargs and null
        # tools (other request fields are kept); null kwargs and an empty tools list behave the same.
        launch = TITOTokenizer(MagicMock(), chat_template_kwargs=self.LAUNCH)
        assert launch.resolve_request_args({"messages": []}, turn_args=None) == {
            "messages": [],
            "chat_template_kwargs": self.LAUNCH,
            "tools": None,
        }
        assert launch.resolve_request_args({"chat_template_kwargs": None, "tools": []}, turn_args=None) == {
            "chat_template_kwargs": self.LAUNCH,
            "tools": None,
        }

    def test_continued_turn_inherits_omitted_kwargs_and_request_overrides_history(self):
        # [FUNCTION DOC] With recorded turn args holding kwargs: an omitting request inherits them
        # (tools null); a request repeating them gives the same; a request with a different value
        # overrides; and the recorded mapping is never mutated.
        launch = TITOTokenizer(MagicMock(), chat_template_kwargs=self.LAUNCH)
        recorded = {"chat_template_kwargs": {"enable_thinking": True}}

        assert launch.resolve_request_args({}, turn_args=recorded) == {**recorded, "tools": None}
        same = launch.resolve_request_args({"chat_template_kwargs": {"enable_thinking": True}}, turn_args=recorded)
        assert same == {**recorded, "tools": None}
        assert launch.resolve_request_args(
            {"chat_template_kwargs": {"enable_thinking": False}}, turn_args=recorded
        ) == {"chat_template_kwargs": {"enable_thinking": False}, "tools": None}
        assert recorded == {"chat_template_kwargs": {"enable_thinking": True}}

    def test_empty_history_keeps_defaults_but_does_not_allow_new_tools(self):
        # [FUNCTION DOC] With an empty recorded mapping the launch kwargs and null tools are kept;
        # tools supplied with no recorded turn at all are accepted; tools supplied against an empty
        # (but present) recorded mapping raise a value error saying tools changed.
        launch = TITOTokenizer(MagicMock(), chat_template_kwargs=self.LAUNCH)
        assert launch.resolve_request_args({}, turn_args={}) == {"chat_template_kwargs": self.LAUNCH, "tools": None}
        assert launch.resolve_request_args({"tools": self.TOOLS}, turn_args=None)["tools"] == self.TOOLS
        with pytest.raises(ValueError, match="tools changed"):
            launch.resolve_request_args({"tools": self.TOOLS}, turn_args={})

    def test_continued_turn_tools_inherit_pass_or_are_refused(self):
        # [FUNCTION DOC] Recorded tools are inherited by an omitting request; the same tools in the
        # bare-function spelling are accepted (kept as given); different tools, or tools against a
        # recorded turn that had none, raise a value error "tools changed on a continued turn".
        launch = TITOTokenizer(MagicMock())
        recorded = {"tools": self.TOOLS}

        assert launch.resolve_request_args({}, turn_args=recorded) == {"chat_template_kwargs": {}, "tools": self.TOOLS}
        bare = [tool["function"] for tool in self.TOOLS]  # the same tools in the other accepted spelling
        assert launch.resolve_request_args({"tools": bare}, turn_args=recorded) == {
            "chat_template_kwargs": {},
            "tools": bare,
        }
        with pytest.raises(ValueError, match="tools changed on a continued turn"):
            launch.resolve_request_args({"tools": self.OTHER_TOOLS}, turn_args=recorded)
        with pytest.raises(ValueError, match="tools changed on a continued turn"):
            launch.resolve_request_args({"tools": self.TOOLS}, turn_args={})  # the turn had none

    def test_a_family_may_allow_tools_to_change_mid_session(self):
        # [FUNCTION DOC] Defines a local tokenizer subclass whose tools hook takes the request's
        # tools if present, else the recorded tools, and sets a deep copy on the resolved args. With
        # different request and recorded tools, asserts the resolved tools are the request's (no
        # refusal), with empty kwargs.
        class _ToolsAtTheTailTITOTokenizer(TITOTokenizer):
            # [CLASS DOC] HELPER: Test-only subclass allowing tools to change between turns.
            def resolve_tools(self, request_args, *, request_source, turn_args):
                # [FUNCTION DOC] HELPER: Pick the tools from the request source, else from the
                # recorded turn args; if any, store a deep copy in the resolved request args.
                # Mutates request_args; returns nothing.
                tools = request_source.get("tools") or (turn_args or {}).get("tools")
                if tools:
                    request_args["tools"] = deepcopy(tools)

        family = _ToolsAtTheTailTITOTokenizer(MagicMock())
        args = family.resolve_request_args({"tools": self.OTHER_TOOLS}, turn_args={"tools": self.TOOLS})
        assert args == {"chat_template_kwargs": {}, "tools": self.OTHER_TOOLS}

    def test_family_constants_override_requested_values(self, qwen3_tito: Qwen3TITOTokenizer):
        # [FUNCTION DOC] Qwen3's family constant clear_thinking false is merged with a request's
        # other kwargs, and wins over a request value of clear_thinking true.
        args = qwen3_tito.resolve_request_args({"chat_template_kwargs": {"enable_thinking": True}}, turn_args=None)
        assert args == {"chat_template_kwargs": {"clear_thinking": False, "enable_thinking": True}, "tools": None}
        assert qwen3_tito.resolve_request_args({"chat_template_kwargs": {"clear_thinking": True}}, turn_args=None) == {
            "chat_template_kwargs": {"clear_thinking": False},
            "tools": None,
        }

    @pytest.mark.parametrize("tito_cls", [DeepSeekV32TITOTokenizer, DeepSeekV4TITOTokenizer])
    def test_deepseek_thinking_rule_uses_resolved_kwargs(self, tito_cls):
        # [FUNCTION DOC] Parametrized over V3.2 and V4 classes. Calls the thinking resolution hook
        # on request args whose kwargs hold thinking mode "thinking" and a custom option, with a
        # request source and recorded turn args that say thinking is off. Asserts the hook rewrites
        # the args' kwargs to thinking true (from the resolved mode) plus the custom option, leaving
        # other fields (temperature) intact; the aliases and mode key are removed.
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 1
        model = tito_cls(tokenizer, chat_template_kwargs={"thinking": False})
        request_args = {
            "chat_template_kwargs": {"thinking_mode": "thinking", "custom_option": "resolved"},
            "temperature": 0.7,
        }

        model.resolve_thinking(
            request_args,
            request_source={"chat_template_kwargs": {"enable_thinking": False}},
            turn_args={"chat_template_kwargs": {"thinking": False}},
        )

        assert request_args == {
            "chat_template_kwargs": {"thinking": True, "custom_option": "resolved"},
            "temperature": 0.7,
        }

    @pytest.mark.parametrize("tito_cls", [DeepSeekV32TITOTokenizer, DeepSeekV4TITOTokenizer])
    def test_deepseek_recorded_mode_overrides_request_aliases(self, tito_cls):
        # [FUNCTION DOC] Parametrized over V3.2 and V4 classes. Launch has enable-thinking false, so
        # stored kwargs are drop-thinking false and thinking false. A first request with thinking
        # true records thinking true; on later turns requests with enable-thinking true or thinking
        # mode "chat" resolve to exactly the recorded args (recorded mode wins over aliases).
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 1
        launch = tito_cls(tokenizer, chat_template_kwargs={"enable_thinking": False})
        assert launch.chat_template_kwargs == {"drop_thinking": False, "thinking": False}

        recorded = launch.resolve_request_args({"chat_template_kwargs": {"thinking": True}}, turn_args=None)
        assert recorded == {"chat_template_kwargs": {"drop_thinking": False, "thinking": True}, "tools": None}
        same = launch.resolve_request_args({"chat_template_kwargs": {"enable_thinking": True}}, turn_args=recorded)
        assert same == recorded
        assert (
            launch.resolve_request_args({"chat_template_kwargs": {"thinking_mode": "chat"}}, turn_args=recorded)
            == recorded
        )

    def test_returns_the_same_full_request_without_inheriting_sampling_fields(self):
        # [FUNCTION DOC] Resolving with recorded args that include sampling fields and a model key;
        # asserts the returned object is the very request mapping passed in (mutated in place).
        launch = TITOTokenizer(MagicMock())
        turn_args = {"temperature": 0.2, "seed": 42, "chat_template_kwargs": {"enable_thinking": True}}
        request_args = {"temperature": 0.8, "model": "m"}
        result = launch.resolve_request_args(request_args, turn_args=turn_args)
        assert result is request_args
        assert result == {
            "temperature": 0.8,
            "model": "m",
            "chat_template_kwargs": {"enable_thinking": True},
            "tools": None,
        }
        assert turn_args["temperature"] == 0.2

    def test_inherited_nested_values_are_owned_by_the_working_request(self):
        # [FUNCTION DOC] Launch kwargs hold a nested options mapping with a list. Resolving with no
        # recorded turn and mutating the nested list of the result must leave the launch kwargs
        # untouched; then resolving with recorded args (nested kwargs and tools) and mutating the
        # result's nested kwargs and tool name must leave the recorded args untouched (values are
        # deep-copied into the working request).
        launch = TITOTokenizer(MagicMock(), chat_template_kwargs={"options": {"labels": ["launch"]}})
        request_args = launch.resolve_request_args({}, turn_args=None)
        request_args["chat_template_kwargs"]["options"]["labels"].append("request")
        assert launch.chat_template_kwargs == {"options": {"labels": ["launch"]}}

        turn_args = {"chat_template_kwargs": {"options": {"labels": ["turn"]}}, "tools": self.TOOLS}
        request_args = launch.resolve_request_args({}, turn_args=turn_args)
        request_args["chat_template_kwargs"]["options"]["labels"].append("next")
        request_args["tools"][0]["function"]["name"] = "changed"
        assert turn_args["chat_template_kwargs"] == {"options": {"labels": ["turn"]}}
        assert turn_args["tools"][0]["function"]["name"] == "get_weather"

    def test_appended_rules_receive_original_sources_and_can_change_full_request(self):
        # [FUNCTION DOC] Defines a local subclass that appends one custom rule to the base rule
        # list. The rule runs only when the request has a temperature; it asserts the original request
        # source still has its own kwargs and tools, and that the working args already contain the
        # request kwargs merged with the launch ones; it then sets temperature 0.25, a tool choice of
        # "auto", renames the working tool and overwrites a kwarg. The test resolves a request and
        # asserts the very same request object is returned with those changes, the launch mapping is
        # unchanged, and the base tokenizer has 4 rules while the subclass has 5.
        class Model(TITOTokenizer):
            # [CLASS DOC] HELPER: Test-only subclass appending a custom request-argument rule.
            def request_arg_rules(self):
                # [FUNCTION DOC] HELPER: Return the parent's rule list with the custom rule appended.
                rules = super().request_arg_rules()
                rules.append(self.resolve_custom_args)
                return rules

            def resolve_custom_args(self, request_args, *, request_source, turn_args):
                # [FUNCTION DOC] HELPER: Custom rule: does nothing when the original request has no
                # temperature; otherwise performs the assertions and mutations described by the
                # enclosing test on the working request args. Returns nothing.
                if "temperature" not in request_source:
                    return
                assert request_source["chat_template_kwargs"] == {"choice": "request"}
                assert request_args["chat_template_kwargs"] == {"choice": "request", "from_launch": True}
                request_args["temperature"] = 0.25
                request_args["tool_choice"] = "auto"
                request_args["tools"][0]["function"]["name"] = "rewritten"
                request_args["chat_template_kwargs"]["choice"] = "model"
                assert request_source["tools"][0]["function"]["name"] == "get_weather"

        launch = {"choice": "launch", "from_launch": True}
        model = Model(MagicMock(), chat_template_kwargs=launch)
        original = {"temperature": 0.8, "tools": self.TOOLS, "chat_template_kwargs": {"choice": "request"}}
        request_args = deepcopy(original)
        assert model.resolve_request_args(request_args, turn_args=None) is request_args
        assert request_args["temperature"] == 0.25
        assert request_args["tool_choice"] == "auto"
        assert request_args["tools"][0]["function"]["name"] == "rewritten"
        assert request_args["chat_template_kwargs"]["choice"] == "model"
        assert launch == {"choice": "launch", "from_launch": True}
        assert len(TITOTokenizer(MagicMock()).request_arg_rules()) == 4
        assert len(model.request_arg_rules()) == 5

    @pytest.mark.parametrize("recorded_present", [True, False], ids=["recorded-value", "recorded-absence"])
    @pytest.mark.parametrize(
        "tito_cls, key, recorded_value, requested_value",
        [
            (Qwen35TITOTokenizer, "add_vision_id", True, False),
            (Qwen36TITOTokenizer, "add_vision_id", True, False),
            (Qwen38SmallTITOTokenizer, "add_vision_id", True, False),
            (Qwen38SmallTITOTokenizer, "enable_thinking", True, False),
            (Qwen38SmallTITOTokenizer, "reasoning_effort", "low", "medium"),
            (GLM53TITOTokenizer, "reasoning_effort", "max", "low"),
            (Nemotron3TITOTokenizer, "low_effort", True, False),
            (Kimi25TITOTokenizer, "thinking", True, False),
            (Kimi25TITOTokenizer, "tools_ts_str", "old declaration", "new declaration"),
            (Kimi26TITOTokenizer, "tools_ts_str", "old declaration", "new declaration"),
            (DeepSeekV32TITOTokenizer, "add_default_bos_token", True, False),
            (DeepSeekV32TITOTokenizer, "context", [{"role": "system", "content": "old"}], []),
            (DeepSeekV4TITOTokenizer, "reasoning_effort", "max", "high"),
            (DeepSeekV4TITOTokenizer, "add_default_bos_token", True, False),
            (DeepSeekV4TITOTokenizer, "context", [{"role": "system", "content": "old"}], []),
            (InklingTITOTokenizer, "reasoning_effort", "high", "low"),
        ],
    )
    def test_model_rules_keep_fields_that_affect_the_reused_prefix(
        self, tito_cls, key, recorded_value, requested_value, recorded_present
    ):
        # [FUNCTION DOC] Parametrized over many families and a template kwarg that affects the
        # already rendered prefix (vision id switch, thinking switch, reasoning effort, low effort,
        # Kimi tool declaration, bos token, context messages, and so on) with a recorded value and a
        # different requested value, and whether the recorded turn has the key at all. Builds the
        # tokenizer on a mock tokenizer with the requested value as launch kwarg, builds recorded
        # turn args from the launch kwargs with the key removed or set to the recorded value, resolves
        # a request that asks for the other value plus a custom option. Asserts: the very request
        # object is returned; when recorded, the result keeps the recorded value (and for the context
        # key the result is a deep copy, which the test mutates); when not recorded, the key is
        # absent; the custom option passes through; the recorded turn args are unchanged; and the
        # tokenizer's own kwargs equal the launch kwargs copy.
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1]
        tokenizer.convert_tokens_to_ids.return_value = 1
        model = tito_cls(tokenizer, chat_template_kwargs={key: requested_value})
        launch = deepcopy(model.chat_template_kwargs)
        turn_kwargs = deepcopy(launch)
        turn_kwargs.pop(key, None)
        if recorded_present:
            turn_kwargs[key] = deepcopy(recorded_value)
        turn_args = {"chat_template_kwargs": turn_kwargs}
        original_turn = deepcopy(turn_args)
        request_args = {"chat_template_kwargs": {key: requested_value, "custom_option": "new"}}

        result = model.resolve_request_args(request_args, turn_args=turn_args)

        assert result is request_args
        if recorded_present:
            assert result["chat_template_kwargs"][key] == recorded_value
            if key == "context":
                result["chat_template_kwargs"][key][0]["content"] = "changed"
        else:
            assert key not in result["chat_template_kwargs"]
        assert result["chat_template_kwargs"]["custom_option"] == "new"
        assert turn_args == original_turn
        assert model.chat_template_kwargs == launch

    @pytest.mark.parametrize(
        "tito_cls, key, old_value, new_value, content",
        [
            (InklingTITOTokenizer, "reasoning_effort", "high", "low", "hello"),
            (Qwen38SmallTITOTokenizer, "enable_thinking", True, False, "hello"),
            (Qwen38SmallTITOTokenizer, "reasoning_effort", "low", "medium", "hello"),
            (
                Qwen35TITOTokenizer,
                "add_vision_id",
                True,
                False,
                [{"type": "image", "image": "image"}, {"type": "text", "text": "describe"}],
            ),
        ],
    )
    def test_model_rules_preserve_rendered_prefix_when_request_changes_a_locked_field(
        self, tito_cls, key, old_value, new_value, content
    ):
        # [FUNCTION DOC] Parametrized over Inkling reasoning effort, Qwen 3.8-small thinking switch
        # and effort, and Qwen 3.5 vision ids (with image content). Resolves a first request
        # recording the old value, then a request asking for the new value on the continued turn.
        # Renders a one-message conversation with the family's fixed template using (a) the recorded
        # args, (b) the old args with the key changed, and (c) the newly resolved args. Asserts that
        # (b) differs from (a) (the field really matters) and that (c) equals (a) (the locked value
        # kept the rendered prefix stable).
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1]
        tokenizer.convert_tokens_to_ids.return_value = 1
        model = tito_cls(tokenizer)
        turn_args = model.resolve_request_args({"chat_template_kwargs": {key: old_value}}, turn_args=None)
        requested = {"chat_template_kwargs": {key: new_value}}
        resolved = model.resolve_request_args(requested, turn_args=turn_args)
        template_text = (TEMPLATE_DIR / model.FIXED_TEMPLATE.template).read_text()
        messages = [{"role": "user", "content": content}]
        old_kwargs = extract_template_args(turn_args)
        old_render = apply_chat_template_from_str(template_text, messages, add_generation_prompt=False, **old_kwargs)
        changed_render = apply_chat_template_from_str(
            template_text, messages, add_generation_prompt=False, **{**old_kwargs, key: new_value}
        )
        assert changed_render != old_render
        assert (
            apply_chat_template_from_str(
                template_text, messages, add_generation_prompt=False, **extract_template_args(resolved)
            )
            == old_render
        )

    @pytest.mark.parametrize(
        "launch_kwargs, request_kwargs, expected_effort",
        [
            ({}, {}, None),
            ({"reasoning_effort": "low"}, {}, "low"),
            ({"reasoning_effort": "low"}, {"reasoning_effort": "medium"}, "medium"),
            ({"reasoning_effort": "medium"}, {"reasoning_effort": "low"}, "low"),
            ({"reasoning_effort": "low"}, {"reasoning_effort": "xhigh"}, "xhigh"),
        ],
    )
    def test_qwen38_new_root_selects_effort(self, launch_kwargs, request_kwargs, expected_effort):
        # [FUNCTION DOC] Parametrized over launch/request reasoning effort combinations. With no
        # recorded turn, the Qwen 3.8-small tokenizer resolves kwargs to preserve-thinking true (the
        # family constant beats a request value of false) plus a reasoning effort equal to the
        # request's value, else the launch value, else absent when neither is given.
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1]
        model = Qwen38SmallTITOTokenizer(tokenizer, chat_template_kwargs=launch_kwargs)
        result = model.resolve_request_args(
            {"chat_template_kwargs": {**request_kwargs, "preserve_thinking": False}}, turn_args=None
        )
        expected = {"preserve_thinking": True}
        if expected_effort is not None:
            expected["reasoning_effort"] = expected_effort
        assert result["chat_template_kwargs"] == expected

    @pytest.mark.parametrize("turn_args", [None, {}, {"chat_template_kwargs": {"reasoning_effort": "low"}}])
    def test_qwen38_effort_follows_history(self, turn_args):
        # [FUNCTION DOC] Parametrized over recorded turn args: none, empty mapping, and one recording
        # effort "low". Launch effort is xhigh and the request asks for medium. Asserts the very
        # request object is returned and the resolved effort is: the request's medium for a new root
        # (no turn), absent for an empty recorded mapping, the recorded low for a recorded effort; and
        # the recorded args are unchanged.
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1]
        tokenizer.convert_tokens_to_ids.return_value = 1
        model = Qwen38SmallTITOTokenizer(tokenizer, chat_template_kwargs={"reasoning_effort": "xhigh"})
        original_turn = deepcopy(turn_args)
        request_args = {"chat_template_kwargs": {"reasoning_effort": "medium"}}

        assert model.resolve_request_args(request_args, turn_args=turn_args) is request_args

        kwargs = request_args["chat_template_kwargs"]
        if turn_args is None:
            assert kwargs["reasoning_effort"] == "medium"
        elif turn_args:
            assert kwargs["reasoning_effort"] == "low"
        else:
            assert "reasoning_effort" not in kwargs
        assert turn_args == original_turn

    def test_qwen38_omitted_effort_uses_template_default(self):
        # [FUNCTION DOC] With nothing specified, the resolved template args have no reasoning effort.
        # Renders a hello message with generation prompt using the fixed template with those args and
        # asserts it equals the render with effort explicitly xhigh (the template default is xhigh).
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1]
        model = Qwen38SmallTITOTokenizer(tokenizer)
        result = model.resolve_request_args({}, turn_args=None)
        kwargs = extract_template_args(result)
        assert "reasoning_effort" not in kwargs
        template_text = (TEMPLATE_DIR / model.FIXED_TEMPLATE.template).read_text()
        messages = [{"role": "user", "content": "hello"}]
        rendered = apply_chat_template_from_str(template_text, messages, add_generation_prompt=True, **kwargs)
        assert rendered == apply_chat_template_from_str(
            template_text, messages, add_generation_prompt=True, **{**kwargs, "reasoning_effort": "xhigh"}
        )


class TestCompletionPostprocess:
    # [CLASS DOC] Tests of the completion post-processing hook.
    def test_default_returns_upstream_message_unchanged(self):
        # [FUNCTION DOC] The base tokenizer's hook returns the very assistant message object it was
        # given and leaves the choice dictionary (message, finish reason, meta info) unchanged.
        tito = TITOTokenizer(MagicMock())
        assistant_message = {"role": "assistant", "content": "upstream"}
        choice = {
            "message": assistant_message,
            "finish_reason": "stop",
            "meta_info": {"existing": True},
        }

        stored_message = tito.postprocess_completion(
            choice=choice,
            assistant_message=assistant_message,
            completion_token_ids=[1, 2, 3],
        )

        assert stored_message is assistant_message
        assert choice == {
            "message": assistant_message,
            "finish_reason": "stop",
            "meta_info": {"existing": True},
        }


class TestInklingComparatorBoundaries:
    # [CLASS DOC] Tests of the sequence comparator created by the Inkling TITO tokenizer, using a
    # mock tokenizer with four message-role special tokens (ids 200000 to 200003).
    @staticmethod
    def _build_comparator():
        # [FUNCTION DOC] HELPER: Build a mock tokenizer whose token lookup returns the role marker
        # ids (user, model, system, tool) and raises a key error for others, whose encode maps each
        # character to its code point, and whose decode maps role ids back to marker text and other
        # ids to characters. Creates the Inkling tokenizer's comparator from it and returns
        # (marker-to-id mapping, mock tokenizer, comparator).
        token_ids = {
            "<|message_user|>": 200000,
            "<|message_model|>": 200001,
            "<|message_system|>": 200002,
            "<|message_tool|>": 200003,
        }
        token_text = {token_id: token for token, token_id in token_ids.items()}
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.side_effect = token_ids.__getitem__
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [ord(char) for char in text]
        tokenizer.decode.side_effect = lambda ids, skip_special_tokens=False: "".join(
            token_text.get(token_id, chr(token_id)) for token_id in ids
        )
        comparator = InklingTITOTokenizer(tokenizer).create_comparator()
        return token_ids, tokenizer, comparator

    def test_uses_exact_message_role_boundaries(self):
        # [FUNCTION DOC] The comparator's special id set equals the four role marker ids.
        token_ids, _, comparator = self._build_comparator()

        assert comparator._special_ids == set(token_ids.values())

    @pytest.mark.parametrize("appended_role", ["user", "system", "tool"])
    def test_appended_non_assistant_content_is_not_assistant_text(self, appended_role):
        # [FUNCTION DOC] Parametrized over appended roles user, system, tool. Two sequences share a
        # model-role prefix "assistant" and then a role marker plus differing text ("expected" versus
        # "changed"). Asserts comparing yields exactly one mismatch of the non-assistant-text type.
        token_ids, tokenizer, comparator = self._build_comparator()
        assistant = [token_ids["<|message_model|>"]] + tokenizer.encode("assistant")
        role_token = token_ids[f"<|message_{appended_role}|>"]
        expected = assistant + [role_token] + tokenizer.encode("expected")
        actual = assistant + [role_token] + tokenizer.encode("changed")

        mismatches = comparator.compare_sequences(expected, actual)

        assert [mismatch.type for mismatch in mismatches] == [MismatchType.NON_ASSISTANT_TEXT]

    def test_assistant_content_remains_soft_mismatch(self):
        # [FUNCTION DOC] Two sequences after a model-role marker with different text; asserts
        # exactly one mismatch of the assistant-text type.
        token_ids, tokenizer, comparator = self._build_comparator()
        assistant_role = [token_ids["<|message_model|>"]]
        expected = assistant_role + tokenizer.encode("expected")
        actual = assistant_role + tokenizer.encode("changed")

        mismatches = comparator.compare_sequences(expected, actual)

        assert [mismatch.type for mismatch in mismatches] == [MismatchType.ASSISTANT_TEXT]


class TestInklingFixedTemplate:
    # [CLASS DOC] Tests of the Inkling fixed chat template's rendering of assistant turns.
    @staticmethod
    def _render(messages):
        # [FUNCTION DOC] HELPER: Read the Inkling fixed template file and render the messages with
        # it without a generation prompt. Returns the string.
        template = (TEMPLATE_DIR / "inkling_fixed.jinja").read_text()
        return apply_chat_template_from_str(template, messages, add_generation_prompt=False)

    def test_tool_call_only_turn_skips_empty_text_block(self):
        # [FUNCTION DOC] An assistant turn with empty content, reasoning "check" and one tool call
        # renders no empty text block; it renders a thinking message followed by a model message
        # addressed to the tool name holding compact JSON of the call (name and args) and then the
        # end-of-sampling marker.
        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "check",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"location": "Beijing"}},
                    }
                ],
            },
        ]

        rendered = self._render(messages)

        assert "<|message_model|><|content_text|><|end_message|>" not in rendered
        assert (
            "<|message_model|><|content_thinking|>check<|end_message|>"
            '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather",'
            '"args":{"location":"Beijing"}}<|end_message|><|content_model_end_sampling|>'
        ) in rendered

    def test_empty_assistant_turn_skips_bare_terminator(self):
        # [FUNCTION DOC] A completely empty assistant turn renders neither a model message marker
        # nor the end-of-sampling marker.
        rendered = self._render(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": ""},
            ]
        )

        assert "<|message_model|>" not in rendered
        assert "<|content_model_end_sampling|>" not in rendered

    def test_nonempty_reasoning_and_final_text_are_preserved(self):
        # [FUNCTION DOC] An assistant turn with reasoning "think" and content "done" renders a
        # thinking message, then a text message, then the end-of-sampling marker, in that order.
        rendered = self._render(
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "done",
                    "reasoning_content": "think",
                },
            ]
        )

        assert (
            "<|message_model|><|content_thinking|>think<|end_message|>"
            "<|message_model|><|content_text|>done<|end_message|>"
            "<|content_model_end_sampling|>"
        ) in rendered

    def test_ordered_blocks_preserve_thinking_tool_call_and_text_order(self):
        # [FUNCTION DOC] An assistant turn carrying both flattened fields and ordered content blocks
        # (thinking, empty text, tool call with recipient and raw JSON text, thinking, text). The
        # render must follow the ordered blocks exactly (including the empty text block and the raw
        # JSON text verbatim), ignore the flattened text and reasoning, and contain exactly one
        # tool-json marker.
        raw_json = '{ "args": {"command": "pwd"}, "name": "bash_command" }'
        rendered = self._render(
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "flattened-text",
                    "reasoning_content": "flattened-thinking",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash_command", "arguments": {"command": "pwd"}},
                        }
                    ],
                    "content_blocks": [
                        {"type": "thinking", "text": "think-1"},
                        {"type": "text", "text": ""},
                        {
                            "type": "tool_call",
                            "recipient": "bash",
                            "name": "bash_command",
                            "arguments": {"command": "pwd"},
                            "raw_json": raw_json,
                        },
                        {"type": "thinking", "text": "think-2"},
                        {"type": "text", "text": "done"},
                    ],
                },
            ]
        )

        assert (
            "<|message_model|><|content_thinking|>think-1<|end_message|>"
            "<|message_model|><|content_text|><|end_message|>"
            f"<|message_model|>bash<|content_invoke_tool_json|>{raw_json}<|end_message|>"
            "<|message_model|><|content_thinking|>think-2<|end_message|>"
            "<|message_model|><|content_text|>done<|end_message|>"
            "<|content_model_end_sampling|>"
        ) in rendered
        assert "flattened-text" not in rendered
        assert "flattened-thinking" not in rendered
        assert rendered.count("<|content_invoke_tool_json|>") == 1

    def test_empty_ordered_block_sequence_keeps_sampling_terminator(self):
        # [FUNCTION DOC] An assistant turn with empty content and an empty list of ordered blocks
        # renders ending with the end-of-sampling marker.
        rendered = self._render(
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "",
                    "content_blocks": [],
                },
            ]
        )

        assert rendered.endswith("<|content_model_end_sampling|>")


class TestDeepSeekV32IncrementalAppend:
    # [CLASS DOC] Tests that for the DeepSeek V3.2 family the synthetic-prefix incremental render
    # equals the suffix of the real-history render, for both tool and user appends.
    """V3.2 rides the default synthetic-prefix suffix diff; with the family's
    pinned ``drop_thinking=False`` the vendored encoder renders every turn
    position-independently, so the synthetic-prefix incremental must equal the
    real-history render suffix for both tool and user appends."""

    class _CharTokenizer:
        # [CLASS DOC] HELPER: Minimal fake tokenizer for the V3.2 incremental tests: name-or-path
        # attribute, character-code encoder, and a lookup of just the user and assistant markers.
        def __init__(self, name_or_path: str):
            # [FUNCTION DOC] HELPER: Store the name-or-path string.
            self.name_or_path = name_or_path

        def encode(self, text, add_special_tokens=False):
            # [FUNCTION DOC] HELPER: Return the code point of each character; asserts special
            # tokens were not requested.
            assert add_special_tokens is False
            return [ord(c) for c in text]

        def convert_tokens_to_ids(self, token):
            # [FUNCTION DOC] HELPER: Map the user marker to 1 and the assistant marker to 2; any
            # other token raises a key error.
            return {"<｜User｜>": 1, "<｜Assistant｜>": 2}[token]

    @pytest.mark.parametrize(
        "appended",
        [
            [{"role": "user", "content": "next question"}],
            [{"role": "tool", "content": "out", "tool_call_id": "c0"}],
            [{"role": "assistant", "content": "injected"}, {"role": "user", "content": "next"}],
            [
                {"role": "assistant", "content": "first injected"},
                {"role": "assistant", "content": "second injected"},
                {"role": "user", "content": "next"},
            ],
        ],
        ids=["user", "tool", "assistant_then_user", "consecutive_assistants_then_user"],
    )
    def test_incremental_equals_real_history_suffix(self, tmp_path, appended):
        # [FUNCTION DOC] Parametrized over four appended message shapes (user, tool, injected
        # assistant then user, two injected assistants then user). Writes a deepseek_v32 config in the
        # temp directory, builds the V3.2 TITO tokenizer (drop-thinking false) over the character
        # tokenizer, and uses a history of a user question and an assistant tool call with reasoning.
        # Computes the incremental tokens for the appended messages, renders the old history without
        # a generation prompt and the new history with one, and asserts the new text starts with the
        # old text and the incremental ids equal the character ids of the new text's remaining suffix.
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v32"}), encoding="utf-8")
        tokenizer = self._CharTokenizer(str(tmp_path))
        tito = DeepSeekV32TITOTokenizer(
            tokenizer,
            chat_template_kwargs={"drop_thinking": False},
        )
        old = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "r",
                "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}],
            },
        ]
        new = old + appended

        incremental = tito.tokenize_additional_messages(old, new)

        text_old = tito.apply_chat_template(old, add_generation_prompt=False)
        text_new = tito.apply_chat_template(new, add_generation_prompt=True)
        assert text_new.startswith(text_old)
        assert incremental == tokenizer.encode(text_new[len(text_old) :])


# ---------------------------------------------------------------------------
# TestMergeTokensBoundary — prefix manipulation with synthetic IDs
#
# All tests use the same trajectory (SingleToolTrajectory split at pos=3)
# to compute incremental tokens, then verify prefix manipulation with
# synthetic IDs like [100, 200, <boundary_token>].
# ---------------------------------------------------------------------------

_BND_OLD, _BND_NEW, _BND_TOOLS = _split_at(SingleToolTrajectory, 3)


class TestMergeTokensBoundary:
    # [CLASS DOC] Tests of the prefix manipulation done by merge_tokens, using the single-tool
    # trajectory split at position 3 (module-level old messages, new messages and tools) and
    # synthetic prefix token ids such as 100 and 200 followed by a boundary token.
    """merge_tokens correctly manipulates the prefix before concatenating incremental tokens."""

    # -- Qwen3: insert \n after <|im_end|> --

    def test_qwen3_inserts_newline_after_im_end(self, qwen3_tito: Qwen3TITOTokenizer):
        # [FUNCTION DOC] For a prefix ending with the Qwen3 end-of-message token, asserts the merge
        # result is the prefix, then the newline token, then the incrementally tokenized appended
        # messages.
        """Model stops at <|im_end|> without trailing \\n; merge_tokens inserts it."""
        incremental = qwen3_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=qwen3_tito.default_template_args(_BND_TOOLS)
        )
        im_end = qwen3_tito._im_end_id
        nl = qwen3_tito._newline_id

        result = qwen3_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [100, 200, im_end], template_args=qwen3_tito.default_template_args(_BND_TOOLS)
        )
        assert result == [100, 200, im_end, nl] + incremental

    def test_qwen3_no_newline_otherwise(self, qwen3_tito: Qwen3TITOTokenizer):
        # [FUNCTION DOC] For a prefix ending with an ordinary token, asserts the merge result is just
        # prefix plus incremental tokens.
        """No insertion when prefix does not end with <|im_end|>."""
        incremental = qwen3_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=qwen3_tito.default_template_args(_BND_TOOLS)
        )
        result = qwen3_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [100, 200, 300], template_args=qwen3_tito.default_template_args(_BND_TOOLS)
        )
        assert result == [100, 200, 300] + incremental

    # -- GLM47: strip ambiguous boundary tokens --

    def test_glm47_strips_observation(self, glm47_tito: GLM47TITOTokenizer):
        # [FUNCTION DOC] A prefix ending with the GLM observation token has that last token removed
        # before appending incremental tokens.
        """Model emits <|observation|> as stop token; merge_tokens strips the duplicate."""
        incremental = glm47_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=glm47_tito.default_template_args(_BND_TOOLS)
        )
        result = glm47_tito.merge_tokens(
            _BND_OLD,
            _BND_NEW,
            [100, 200, glm47_tito._observation_id],
            template_args=glm47_tito.default_template_args(_BND_TOOLS),
        )
        assert result == [100, 200] + incremental

    def test_glm47_strips_user(self, glm47_tito: GLM47TITOTokenizer):
        # [FUNCTION DOC] A prefix ending with the GLM user token has that last token removed before
        # appending incremental tokens.
        """<|user|> is also an ambiguous boundary — stripped the same way."""
        incremental = glm47_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=glm47_tito.default_template_args(_BND_TOOLS)
        )
        result = glm47_tito.merge_tokens(
            _BND_OLD,
            _BND_NEW,
            [100, 200, glm47_tito._user_id],
            template_args=glm47_tito.default_template_args(_BND_TOOLS),
        )
        assert result == [100, 200] + incremental

    def test_glm47_no_strip_otherwise(self, glm47_tito: GLM47TITOTokenizer):
        # [FUNCTION DOC] A prefix ending with an ordinary token is kept whole before the incremental tokens.
        """Non-boundary trailing token is preserved."""
        incremental = glm47_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=glm47_tito.default_template_args(_BND_TOOLS)
        )
        result = glm47_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [100, 200, 300], template_args=glm47_tito.default_template_args(_BND_TOOLS)
        )
        assert result == [100, 200, 300] + incremental

    # -- Default: no boundary handling --

    def test_default_concatenates(self, default_tito: TITOTokenizer):
        # [FUNCTION DOC] The base tokenizer's merge is plain concatenation of prefix and incremental tokens.
        """Base class does plain concatenation without any prefix modification."""
        incremental = default_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=default_tito.default_template_args(_BND_TOOLS)
        )
        result = default_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [100, 200, 300], template_args=default_tito.default_template_args(_BND_TOOLS)
        )
        assert result == [100, 200, 300] + incremental

    # -- Edge case --

    def test_empty_prefix(self, qwen3_tito: Qwen3TITOTokenizer):
        # [FUNCTION DOC] With an empty prefix the Qwen3 merge result equals just the incremental tokens.
        """Empty prefix → no boundary handling, result is just incremental."""
        incremental = qwen3_tito.tokenize_additional_messages(
            _BND_OLD, _BND_NEW, template_args=qwen3_tito.default_template_args(_BND_TOOLS)
        )
        result = qwen3_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [], template_args=qwen3_tito.default_template_args(_BND_TOOLS)
        )
        assert result == incremental


# ---------------------------------------------------------------------------
# TestTokenizeAdditional — one synthetic diff for the complete appended suffix
#
# test_produces_nonempty_incremental is the scalable core: parametrized over
# _TRAJ_CASES (trajectories × split points) × tito fixture (models).
# 8 trajectories × ~14 splits × 2 models = 28 test cases currently.
#
# Validation tests use a single trajectory since the validation logic
# (assert_messages_append_only_with_allowed_role) is model/trajectory-independent.
# ---------------------------------------------------------------------------


class TestTokenizeAdditional:
    # [CLASS DOC] Tests of tokenize_additional_messages (incremental tokens of the appended
    # suffix, computed through a synthetic prefix) and its append-only validation.
    """tokenize_additional_messages produces valid incremental tokens."""

    @pytest.mark.parametrize("traj_cls, pos", _TRAJ_CASES)
    def test_produces_nonempty_incremental(self, tito: TITOTokenizer, traj_cls, pos):
        # [FUNCTION DOC] Parametrized over every trajectory split position and every model tokenizer
        # fixture: splits the trajectory, tokenizes the appended messages with default template args
        # for its tools, and asserts the incremental token list is non-empty.
        """Every valid TITO split yields non-empty incremental tokens.

        This is the primary scalability test — it runs every trajectory's
        TITO splits against every model tokenizer.
        """
        old_msgs, new_msgs, tools = _split_at(traj_cls, pos)
        incremental = tito.tokenize_additional_messages(
            old_msgs, new_msgs, template_args=tito.default_template_args(tools)
        )
        assert len(incremental) > 0

    def test_complete_appendix_reaches_renderer_and_its_error_propagates(
        self, qwen3_tito: Qwen3TITOTokenizer, monkeypatch
    ):
        # [FUNCTION DOC] Old messages are the first three of the single-tool trajectory; the appended
        # suffix is its fourth message, a user message and another tool message. Replaces the
        # tokenizer's private rendered-suffix tokenizer with a recording fake that raises a value
        # error "invalid tool ordering" when the appended roles are tool, user, tool (else returns
        # [1]). Asserts: the value error propagates; the fake was called exactly once with a base of
        # a system and an assistant message whose tool calls equal the old last message's, the whole
        # appended list as one block, the same template args, and the generation prompt flag true.
        # The nested fake is called with base messages, appended messages, template args and the flag.
        old_msgs = list(SingleToolTrajectory.MESSAGES[:3])
        appended = [
            SingleToolTrajectory.MESSAGES[3],
            {"role": "user", "content": "Try another route."},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": false}'},
        ]
        calls = []

        def reject_invalid_order(base_messages, appended_messages, *, template_args=None, add_generation_prompt=False):
            # [FUNCTION DOC] HELPER: Fake suffix tokenizer. Records its four arguments as a tuple in
            # the enclosing list; raises a value error "invalid tool ordering" if the appended roles
            # are exactly tool, user, tool; otherwise returns the one-element list [1].
            calls.append((base_messages, appended_messages, template_args, add_generation_prompt))
            if [message["role"] for message in appended_messages] == ["tool", "user", "tool"]:
                raise ValueError("invalid tool ordering")
            return [1]

        monkeypatch.setattr(qwen3_tito, "_tokenize_rendered_suffix", reject_invalid_order)

        with pytest.raises(ValueError, match="invalid tool ordering"):
            qwen3_tito.tokenize_additional_messages(
                old_msgs,
                old_msgs + appended,
                template_args=qwen3_tito.default_template_args(SingleToolTrajectory.TOOLS),
            )

        assert len(calls) == 1
        base_messages, rendered_appendix, template_args, add_generation_prompt = calls[0]
        assert [message["role"] for message in base_messages] == ["system", "assistant"]
        assert base_messages[-1]["tool_calls"] == old_msgs[-1]["tool_calls"]
        assert rendered_appendix == appended
        assert template_args == qwen3_tito.default_template_args(SingleToolTrajectory.TOOLS)
        assert add_generation_prompt is True

    def test_generation_prompt_is_appended_once_for_full_suffix(self, qwen3_tito: Qwen3TITOTokenizer):
        # [FUNCTION DOC] Appends a tool result and a user message to the thinking single-tool
        # trajectory. Decodes the incremental tokens and asserts the assistant start string occurs
        # exactly once and the decoded text ends with the decoded tokens of the full new conversation
        # rendered with a generation prompt and no appended block (so the prompt is added once).
        old_msgs = list(SingleToolThinkingTrajectory.MESSAGES[:3])
        new_msgs = old_msgs + [
            SingleToolThinkingTrajectory.MESSAGES[3],
            {"role": "user", "content": "Now check Shanghai too."},
        ]
        tools = SingleToolThinkingTrajectory.TOOLS

        incremental = qwen3_tito.tokenize_additional_messages(
            old_msgs, new_msgs, template_args=qwen3_tito.default_template_args(tools)
        )
        decoded = qwen3_tito.tokenizer.decode(incremental)
        assert decoded.count(qwen3_tito._assistant_start_str) == 1
        assert decoded.endswith(
            qwen3_tito.tokenizer.decode(
                qwen3_tito._tokenize_rendered_suffix(
                    new_msgs, [], template_args=qwen3_tito.default_template_args(tools), add_generation_prompt=True
                )
            )
        )

    def test_dummy_assistant_preserves_parallel_tool_calls(self):
        # [FUNCTION DOC] Builds the dummy assistant from the stored assistant message of the
        # multi-tool single-turn trajectory; asserts empty content, reasoning content of a single
        # space, tool calls equal to the stored ones, and tool call ids call_1 and call_2 in order.
        stored_assistant = MultiToolSingleTurnTrajectory.MESSAGES[2]
        dummy_assistant = _build_dummy_assistant(stored_assistant)

        assert dummy_assistant["content"] == ""
        assert dummy_assistant["reasoning_content"] == " "
        assert dummy_assistant["tool_calls"] == stored_assistant["tool_calls"]
        assert [call["id"] for call in dummy_assistant["tool_calls"]] == ["call_1", "call_2"]

    @pytest.mark.parametrize(
        "traj_cls, pos",
        [
            pytest.param(SingleToolTrajectory, 3, id="single-tool"),
            pytest.param(RetrySystemTrajectory, 3, id="tool-plus-system"),
            pytest.param(IntermediateSystemTrajectory, 3, id="intermediate-system"),
        ],
    )
    def test_qwen3_merge_preserves_non_assistant_structure(self, qwen3_tito: Qwen3TITOTokenizer, traj_cls, pos):
        # [FUNCTION DOC] Parametrized over the single-tool, tool-plus-system and intermediate-system
        # trajectories at position 3. Tokenizes the old messages with the tokenizer's template (no
        # generation prompt), merges with the appended messages, tokenizes the full new messages with
        # a generation prompt as the expected sequence, compares expected and merged with the
        # comparator and asserts every mismatch is of the assistant-text type.
        """Merged tokens may differ in assistant text, but not in tool/system structure."""
        old_msgs, new_msgs, tools = _split_at(traj_cls, pos)
        pretokenized = apply_chat_template(
            old_msgs,
            tokenizer=qwen3_tito.tokenizer,
            tokenize=True,
            add_generation_prompt=False,
            tools=tools,
        )
        merged = qwen3_tito.merge_tokens(
            old_msgs, new_msgs, pretokenized, template_args=qwen3_tito.default_template_args(tools)
        )
        expected = apply_chat_template(
            new_msgs,
            tokenizer=qwen3_tito.tokenizer,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
        )
        mismatches = qwen3_tito.create_comparator().compare_sequences(expected, merged)
        assert all(m.type == MismatchType.ASSISTANT_TEXT for m in mismatches)

    # -- Append-only validation (assert_messages_append_only_with_allowed_role is called internally) --

    def test_rejects_prefix_mutation(self, qwen3_tito: Qwen3TITOTokenizer):
        # [FUNCTION DOC] Changes the first old message's content in the new list and asserts that
        # tokenizing the additional messages raises a value error mentioning a mismatch.
        """Modifying an existing message in new_messages raises ValueError."""
        old_msgs, new_msgs, _ = _split_at(SingleToolTrajectory, 3)
        mutated_old = [{"role": "user", "content": "CHANGED"}] + list(old_msgs[1:])
        mutated_new = mutated_old + list(new_msgs[len(old_msgs) :])
        with pytest.raises(ValueError, match="mismatch"):
            qwen3_tito.tokenize_additional_messages(old_msgs, mutated_new)

    def test_rejects_fewer_messages(self, qwen3_tito: Qwen3TITOTokenizer):
        # [FUNCTION DOC] A new list shorter than the old one raises a value error mentioning "fewer".
        """new_messages shorter than old_messages raises ValueError."""
        old_msgs = SingleToolTrajectory.MESSAGES[:3]
        with pytest.raises(ValueError, match="fewer"):
            qwen3_tito.tokenize_additional_messages(old_msgs, old_msgs[:1])

    def test_restricted_template_rejects_unsupported_role(self, qwen3_tito: Qwen3TITOTokenizer):
        # [FUNCTION DOC] Defines a Qwen3 subclass whose fixed template only allows appended tool
        # messages, builds it on the fixture's tokenizer, and asserts that appending an assistant
        # message raises a value error mentioning the role.
        """A template with an explicit narrow capability rejects other roles."""

        class _ToolOnlyQwen3TITOTokenizer(Qwen3TITOTokenizer):
            # [CLASS DOC] HELPER: Test-only Qwen3 variant whose registered fixed template (same file
            # and kwargs as the parent) permits only the tool role to be appended.
            FIXED_TEMPLATE = FixedTemplate(
                template=Qwen3TITOTokenizer.FIXED_TEMPLATE.template,
                extra_kwargs=dict(Qwen3TITOTokenizer.FIXED_TEMPLATE.extra_kwargs),
                allowed_append_roles=frozenset({"tool"}),
            )

        restricted = _ToolOnlyQwen3TITOTokenizer(
            qwen3_tito.tokenizer,
            chat_template_kwargs={"clear_thinking": False},
        )
        old_msgs = SingleToolTrajectory.MESSAGES[:3]
        bad_new = list(old_msgs) + [{"role": "assistant", "content": "hi"}]
        with pytest.raises(ValueError, match="role"):
            restricted.tokenize_additional_messages(old_msgs, bad_new)


# ---------------------------------------------------------------------------
# TestFactory — get_tito_tokenizer dispatch
# ---------------------------------------------------------------------------


class TestFactory:
    # [CLASS DOC] Tests of the get_tito_tokenizer factory.
    """get_tito_tokenizer creates the correct subclass from string or enum type."""

    @pytest.mark.parametrize(
        "type_str, model_id, cls",
        [
            ("qwen3", "Qwen/Qwen3-4B", Qwen3TITOTokenizer),
            ("qwen35", "Qwen/Qwen3-4B", Qwen35TITOTokenizer),
            ("qwen36", "Qwen/Qwen3-4B", Qwen36TITOTokenizer),
            ("qwen38small", "Qwen/Qwen3-4B", Qwen38SmallTITOTokenizer),
            ("qwen4exp", "Qwen/Qwen3-4B", Qwen38SmallTITOTokenizer),
            ("qwennext", "Qwen/Qwen3-4B", QwenNextTITOTokenizer),
            ("glm47", "zai-org/GLM-4.7-Flash", GLM47TITOTokenizer),
            ("default", "Qwen/Qwen3-4B", TITOTokenizer),
        ],
    )
    def test_creates_correct_type(self, type_str, model_id, cls):
        # [FUNCTION DOC] Parametrized over string type names ("qwen3", "qwen35", "qwen36",
        # "qwen38small", "qwen4exp", "qwennext", "glm47", "default"); the factory returns an instance of
        # the expected class (qwen4exp maps to the Qwen 3.8-small class).
        tito = get_tito_tokenizer(_get_tokenizer(model_id), tokenizer_type=type_str)
        assert isinstance(tito, cls)

    def test_enum_input(self):
        # [FUNCTION DOC] Passing the Qwen3 enum member gives a Qwen3 tokenizer whose allowed append
        # roles equal the full role set.
        """Enum values work the same as string values."""
        tito = get_tito_tokenizer(_get_tokenizer("Qwen/Qwen3-4B"), tokenizer_type=TITOTokenizerType.QWEN3)
        assert isinstance(tito, Qwen3TITOTokenizer)
        assert tito.allowed_append_roles == ALL_APPEND_ROLES

    @pytest.mark.parametrize(
        "type_str, cls",
        [
            ("qwen35", Qwen35TITOTokenizer),
            ("qwen36", Qwen36TITOTokenizer),
            ("qwen38small", Qwen38SmallTITOTokenizer),
            ("qwen4exp", Qwen38SmallTITOTokenizer),
            ("qwennext", QwenNextTITOTokenizer),
        ],
    )
    def test_qwen_variant_inherits_qwen3_boundary_logic(self, type_str, cls):
        # [FUNCTION DOC] Parametrized over Qwen 3.5, 3.6, 3.8-small, qwen4exp and Qwen-Next type names;
        # the factory result is an instance of the named class and also of the Qwen3 class.
        """Qwen3.5 / Qwen3-Next reuse Qwen3's boundary handling via inheritance.
        The named subclass owns its FixedTemplate contract, while token-level
        merge behavior remains identical to Qwen3."""
        tito = get_tito_tokenizer(_get_tokenizer("Qwen/Qwen3-4B"), tokenizer_type=type_str)
        assert isinstance(tito, cls)
        assert isinstance(tito, Qwen3TITOTokenizer)

    def test_invalid_type_raises(self):
        # [FUNCTION DOC] An unknown type name raises a value error.
        with pytest.raises(ValueError):
            get_tito_tokenizer(_get_tokenizer("Qwen/Qwen3-4B"), tokenizer_type="nonexistent")

    def test_none_tokenizer_raises(self):
        # [FUNCTION DOC] A null tokenizer raises a value error saying it must not be none.
        with pytest.raises(ValueError, match="must not be None"):
            get_tito_tokenizer(None)


class TestParserBinding:
    # [CLASS DOC] Tests of per-family reasoning-parser and tool-call-parser bindings and of
    # resolve_reasoning_and_tool_call_parser, which checks user-supplied parser names against them.
    """Each TITO subclass binds sglang ``--reasoning-parser`` and
    ``--tool-call-parser`` values; ``resolve_reasoning_and_tool_call_parser``
    enforces user-supplied values agree with the bindings (or returns the
    bound values when the user didn't pass one).  The two parsers are
    resolved independently — a missing binding on one doesn't suppress the
    assert on the other."""

    @pytest.mark.parametrize(
        "tito_model, expected_reasoning, expected_tool_call",
        [
            (TITOTokenizerType.QWEN3, "qwen3", "qwen25"),
            (TITOTokenizerType.QWEN35, "qwen3", "qwen3_coder"),
            (TITOTokenizerType.QWEN36, "qwen3", "qwen3_coder"),
            (TITOTokenizerType.QWEN38_SMALL, "qwen3", "qwen3_coder"),
            (TITOTokenizerType.QWEN4_EXP, "qwen3", "qwen3_coder"),
            (TITOTokenizerType.QWENNEXT, "qwen3", "qwen25"),
            (TITOTokenizerType.GLM47, "glm45", "glm47"),
            (TITOTokenizerType.NEMOTRON3, "nemotron_3", "qwen3_coder"),
            (TITOTokenizerType.KIMI25, None, None),
            (TITOTokenizerType.KIMI26, "kimi_k2", "kimi_k2_raw_id"),
            (TITOTokenizerType.MINIMAX_M25, "minimax-append-think", "minimax-m2"),
            (TITOTokenizerType.MINIMAX_M27, "minimax-append-think", "minimax-m2"),
            (TITOTokenizerType.DEEPSEEKV32, "deepseek-v3", "deepseekv32"),
            (TITOTokenizerType.DEEPSEEKV4, "deepseek-v4", "deepseekv4"),
            (TITOTokenizerType.INKLING, None, None),
            (TITOTokenizerType.DEFAULT, None, None),
        ],
    )
    def test_subclass_binding(self, tito_model, expected_reasoning, expected_tool_call):
        # [FUNCTION DOC] Parametrized over families and their expected reasoning parser and tool
        # call parser names (some families have none). Looks up the family's class and asserts both
        # class attributes equal the expected values.
        cls = TITOTokenizerType.get_tokenizer_class(tito_model)
        assert cls.reasoning_parser == expected_reasoning
        assert cls.tool_call_parser == expected_tool_call

    def test_resolve_returns_binding_when_user_omits(self):
        # [FUNCTION DOC] With no user values, resolution returns each family's bound pair (Qwen,
        # GLM, DeepSeek V4 families) and returns (none, none) for the default family.
        from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN3) == ("qwen3", "qwen25")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN35) == ("qwen3", "qwen3_coder")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN36) == ("qwen3", "qwen3_coder")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN38_SMALL) == ("qwen3", "qwen3_coder")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN4_EXP) == ("qwen3", "qwen3_coder")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.GLM47) == ("glm45", "glm47")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.GLM53) == ("glm45", "glm47")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.DEEPSEEKV4) == ("deepseek-v4", "deepseekv4")
        # DEFAULT family has no binding for either parser; both come back None.
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.DEFAULT) == (None, None)

    def test_resolve_accepts_matching_user_value(self):
        # [FUNCTION DOC] User values equal to the binding are accepted (family given as string or
        # enum) and returned.
        from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

        assert resolve_reasoning_and_tool_call_parser("qwen3", "qwen3", "qwen25") == ("qwen3", "qwen25")
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN35, "qwen3", "qwen3_coder") == (
            "qwen3",
            "qwen3_coder",
        )

    def test_resolve_raises_on_reasoning_mismatch(self):
        # [FUNCTION DOC] A user reasoning parser "glm45" for the Qwen3 family raises a value error
        # whose message says the reasoning parser option disagrees.
        from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

        with pytest.raises(ValueError, match="--reasoning-parser='glm45' disagrees"):
            resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN3, user_reasoning_parser="glm45")

    def test_resolve_raises_on_tool_call_mismatch(self):
        # [FUNCTION DOC] A user tool call parser "glm47" for the Qwen3 family raises a value error
        # whose message says the tool call parser option disagrees.
        from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

        with pytest.raises(ValueError, match="--tool-call-parser='glm47' disagrees"):
            resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN3, user_tool_call_parser="glm47")

    def test_resolve_accepts_user_value_when_family_unbound(self):
        # [FUNCTION DOC] For the default family (no bindings), user-provided parser names are
        # returned unchanged.
        # DEFAULT family has no binding for either parser; user-provided wins
        # (for families that haven't been wired up to a sglang parser yet).
        from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

        assert resolve_reasoning_and_tool_call_parser(
            TITOTokenizerType.DEFAULT, "custom_reasoning", "custom_tool_call"
        ) == ("custom_reasoning", "custom_tool_call")

    def test_resolve_partial_user_input(self):
        # [FUNCTION DOC] Supplying only one parser name: the other is filled in independently from the
        # family binding (reasoning only for Qwen3, tool call only for GLM 4.7).
        # User can pass only one of the two; the other auto-resolves from
        # the family binding independently.
        from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

        # User passes reasoning only — tool_call comes from binding.
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.QWEN3, user_reasoning_parser="qwen3") == (
            "qwen3",
            "qwen25",
        )
        # User passes tool_call only — reasoning comes from binding.
        assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.GLM47, user_tool_call_parser="glm47") == (
            "glm45",
            "glm47",
        )
