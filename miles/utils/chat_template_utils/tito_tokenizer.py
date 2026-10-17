# [MODULE DOC] Token-in-token-out (TITO) tokenizers: one tokenizer class per model family, used to
# extend an already tokenized conversation prefix (the assistant's generated token ids) with the
# tokens of newly appended messages, without re-tokenizing the prefix.
# Contents:
#   - FixedTemplate: per-family record of the bundled fixed chat template file, the template kwargs
#     the family forces, which roles may be appended, and which kwargs must stay as recorded.
#   - TITOTokenizer: default implementation (request-argument resolution pipeline, chat template
#     rendering, suffix-diff based incremental tokenization, plain concatenation merge) plus one
#     subclass per family (Qwen3 / 3.5 / 3.6 / 3.8 / Next, GLM 4.7 / 5.3, Nemotron 3, Kimi 2.5 /
#     2.6, MiniMax M2.5 / M2.7, DeepSeek V3.2 / V4 / V4.1, Inkling) that overrides boundary-token
#     handling in the merge and/or the request-argument rules.
#   - TITOTokenizerType: string enum naming the families, mapping each to its class.
#   - Factory and resolver helpers: build a tokenizer by type, look up a family's fixed template path
#     and forced kwargs, and resolve the sglang reasoning-parser / tool-call-parser names bound to a
#     family (checking them against user-supplied values).
# Class-level attributes on each family: the sglang parser names, the fixed template record, the
# default assistant-start string and the set of trailing token ids the comparator trims.
"""TITO tokenizer — incremental tokenization for pretokenized prefix reuse.

``TITOTokenizer`` computes incremental token IDs for messages appended after the assistant's generated token sequence, then merges them with the pretokenized prefix — handling model-specific boundary tokens at the junction.

The default implementation renders the complete appended suffix and the next generation prompt once under a synthetic ``[dummy_system, dummy_assistant]`` prefix.  Model-specific subclasses customize request rules and token-boundary handling.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from pathlib import Path
from typing import Any

from miles.utils.chat_template_utils import deepseek, template
from miles.utils.chat_template_utils.inkling_parser import InklingResponseParser
from miles.utils.chat_template_utils.message_matcher_hub import (
    assert_messages_append_only_with_allowed_role,
    strict_message_matches,
)
from miles.utils.chat_template_utils.token_seq_comparator import TokenSeqComparator

logger = logging.getLogger(__name__)

# Bundled fixed-template files live under this directory; ``FixedTemplate.template``
# values are filenames relative to it.
TEMPLATE_DIR = Path(__file__).parent / "templates"

# Roles that a fixed template may support after the pretokenized assistant
# prefix.  A family narrows this set only when its registered renderer cannot
# preserve every role append-only.
VALID_APPEND_ROLES: tuple[str, ...] = ("tool", "user", "system", "assistant")
ALL_APPEND_ROLES: frozenset[str] = frozenset(VALID_APPEND_ROLES)

_DUMMY_SYSTEM: dict[str, Any] = {"role": "system", "content": "dummy system"}


@dataclass(frozen=True)
class FixedTemplate:
    # [CLASS DOC] Immutable per-family record describing how chat templates are rendered so that
    # prompts stay append-only. Fields: template (file name relative to the bundled template
    # directory, or nothing to keep the model's native template); extra_kwargs (mapping of template
    # keyword arguments the family forces, default empty); allowed_append_roles (set of roles that
    # may follow the pretokenized assistant prefix, default all four of tool, user, system,
    # assistant); consistant_kwargs (list of kwarg names whose recorded value or absence from the
    # earlier turn is kept, default empty). The field name is spelled "consistant" in the code.
    """A family's fixed chat template, required kwargs, and append surface.

    ``template`` is a path relative to ``TEMPLATE_DIR`` for a bundled fixed
    template, or ``None`` to keep the HF-native template (kwargs-only fix).
    ``extra_kwargs`` carry the family's preserve-think constants, so renders
    stay append-only.  ``allowed_append_roles`` defaults to the maximal
    four-role surface; a known restricted template must narrow it explicitly.
    ``consistant_kwargs`` lists fields that must retain their recorded value
    or absence when continuing a turn.
    """

    template: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    allowed_append_roles: frozenset[str] = ALL_APPEND_ROLES
    consistant_kwargs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validates and normalizes the allowed append roles after construction.
        # Converts the roles to an immutable set (written back despite the record being frozen);
        # raises a value error listing the sorted unknown roles and the supported roles if any role
        # is outside the four known ones.
        roles = frozenset(self.allowed_append_roles)
        invalid = roles - ALL_APPEND_ROLES
        if invalid:
            raise ValueError(
                f"Unknown FixedTemplate allowed_append_roles: {sorted(invalid)}; "
                f"supported roles are {sorted(ALL_APPEND_ROLES)}"
            )
        object.__setattr__(self, "allowed_append_roles", roles)


def extract_template_args(request_args: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] Builds the complete chat-template keyword arguments from a resolved request.
    # Input: a request mapping. Output: a new mapping holding a shallow copy of the request's
    # template kwargs (empty if absent or null), plus the request's tools under the tools key when
    # the request has a non-empty tools list. The input is not modified.
    """Select renderer kwargs from an already resolved request, including tools."""
    args = dict(request_args.get("chat_template_kwargs") or {})
    if request_args.get("tools"):
        args["tools"] = request_args["tools"]
    return args


def _build_dummy_assistant(stored_assistant: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Makes a placeholder assistant message used as the rendering base.
    # Returns a message with role assistant, empty content, a single-space reasoning content (so
    # templates that render a reasoning block do so), and the stored message's tool calls (empty
    # list if none), so the rendered base shape matches the real turn. No side effects.
    """Build a dummy assistant that preserves the stored turn's tool calls."""
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": " ",
        "tool_calls": stored_assistant.get("tool_calls") or [],
    }


# ---------------------------------------------------------------------------
# Base / default tokenizer
# ---------------------------------------------------------------------------
# TODO: split different model's TITO tokenizer into different files


class TITOTokenizer:
    # [CLASS DOC] Default tokenizer for incremental tokenization of appended messages and merging
    # them onto a pretokenized prefix. Class attributes (overridden by families): max_trim_tokens
    # (number of tail tokens a merge may drop, default 0; not read within this module),
    # trailing_token_ids (token ids the comparator strips from sequence tails, default empty),
    # FIXED_TEMPLATE (the family's fixed template record, default native template with all roles),
    # reasoning_parser and tool_call_parser (sglang parser names bound to the family, default
    # none). Instance state: the wrapped tokenizer, launch-time chat template kwargs, optional
    # assistant start string, allowed append roles (from the fixed template record) and the
    # special token ids used for segmentation. The default merge is plain concatenation.
    """Incremental tokenization and prefix merging for appended messages."""

    max_trim_tokens: int = 0
    trailing_token_ids: frozenset[int] = frozenset()

    # The family's fixed renderer contract. DEFAULT uses the model's native
    # template with the maximal best-effort append surface.
    FIXED_TEMPLATE: FixedTemplate = FixedTemplate()

    # sglang ``--reasoning-parser`` and ``--tool-call-parser`` values bound to
    # this family.
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        special_token_ids: set[int] | None = None,
    ):
        # [FUNCTION DOC] Stores the tokenizer, a copy of the launch template kwargs (empty if
        # none), the assistant start string, the allowed append roles taken from the family's
        # fixed template record, and the special token ids (kept as given, may be none, in which
        # case the comparator auto-detects them). Then resolves launch-time request arguments with
        # the base class rules only (deliberately bypassing subclass overrides, with no turn
        # history) and replaces the stored template kwargs with the template arguments extracted
        # from that result, so the launch defaults include the family's forced kwargs.
        self.tokenizer = tokenizer
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._assistant_start_str = assistant_start_str
        self.allowed_append_roles = self.FIXED_TEMPLATE.allowed_append_roles
        self.special_token_ids: set[int] = special_token_ids
        launch_args = TITOTokenizer.resolve_request_args(self, {}, turn_args=None)
        self.chat_template_kwargs = extract_template_args(launch_args)

    def request_arg_rules(self) -> list[Callable[..., None]]:
        # [FUNCTION DOC] Returns the ordered list of rule callables applied to a request: resolve
        # template kwargs, resolve tools, resolve history args, then apply the fixed template
        # kwargs (so forced values win over everything before). Each rule takes the request
        # mapping plus keyword arguments request_source and turn_args and mutates the request.
        # Subclasses extend by appending rules. Pure; builds a new list each call.
        """Return this model's rules in execution order; subclasses may extend the list."""
        return [
            self.resolve_template_kwargs,
            self.resolve_tools,
            self.resolve_history_args,
            self.apply_fixed_template_kwargs,
        ]

    def resolve_request_args(
        self, request_args: dict[str, Any], *, turn_args: dict[str, Any] | None
    ) -> dict[str, Any]:
        # [FUNCTION DOC] Runs every rule from the rule list, in order, against the request.
        # Inputs: the request mapping (mutated in place), and turn_args (keyword only), the recorded
        # arguments of the turn being continued or none for a fresh request. A shallow snapshot of
        # the request taken before any rule runs is passed to each rule as the unmodified source.
        # Returns the same request mapping. Rules may raise (for example on changed tools).
        """Apply model rules in place and return the same full request.

        Rules read request_source, turn_args and launch kwargs without modifying
        them. Copy inherited mutable values before writing them into request_args.
        """
        request_source = dict(request_args)
        for rule in self.request_arg_rules():
            rule(request_args, request_source=request_source, turn_args=turn_args)
        return request_args

    def resolve_template_kwargs(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        # [FUNCTION DOC] Rule: sets the request's template kwargs to a deep copy of the merge of
        # three layers, lowest to highest priority: launch defaults, the recorded turn's template
        # kwargs (if any), then the original request's template kwargs. The tools key is dropped
        # from the merged result because requests carry tools at the top level. Mutates only the
        # request mapping's template kwargs entry.
        """Merge request template fields over history and launch defaults."""
        kwargs = {
            **self.chat_template_kwargs,
            **((turn_args or {}).get("chat_template_kwargs") or {}),
            **(request_source.get("chat_template_kwargs") or {}),
        }
        # Launch kwargs may contain tools; requests carry them at the top level.
        kwargs.pop("tools", None)
        request_args["chat_template_kwargs"] = deepcopy(kwargs)

    def resolve_tools(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        # [FUNCTION DOC] Rule: decides the request's tools and stores a deep copy under the tools
        # key. Take the original request's tools (empty list treated as absent). With no recorded
        # turn: fall back to the tools in the launch template kwargs. With a recorded turn: use the
        # recorded tools if the request has none; otherwise compare the tool definitions of both
        # (via the template helper that extracts tool dictionaries) and raise a value error if they
        # differ. The stored value may be none.
        """Inherit omitted tools and reject changes to tools in a reused prefix."""
        tools = request_source.get("tools") or None
        if turn_args is None:
            tools = tools or self.chat_template_kwargs.get("tools")
        else:
            recorded = turn_args.get("tools")
            if tools is None:
                tools = recorded
            elif template.extract_tool_dicts(tools) != template.extract_tool_dicts(recorded):
                raise ValueError(
                    "tools changed on a continued turn: the turn being continued was rendered with different tools, "
                    "and this model family renders tools in the prompt prefix"
                )
        request_args["tools"] = deepcopy(tools)

    def apply_fixed_template_kwargs(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        # [FUNCTION DOC] Rule: if the family's fixed template record has forced kwargs, writes deep
        # copies of them over the request's template kwargs (creating the entry if missing), so
        # forced values override anything the caller or history supplied. Does nothing otherwise.
        # The source and turn arguments are unused.
        """Override requested template values with the family's required settings."""
        if self.FIXED_TEMPLATE.extra_kwargs:
            request_args.setdefault("chat_template_kwargs", {}).update(deepcopy(self.FIXED_TEMPLATE.extra_kwargs))

    def resolve_history_args(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        # [FUNCTION DOC] Rule: for a continued turn, makes the family's "consistent" kwargs equal the
        # recorded turn's. Does nothing when there is no recorded turn. Otherwise removes each
        # consistent kwarg from the request's template kwargs, then re-adds only those that the
        # recorded turn actually had (deep copied), so a kwarg absent in the record stays absent.
        """Keep selected historical fields, including their absence, in a continued turn."""
        if turn_args is None:
            return
        recorded = turn_args.get("chat_template_kwargs") or {}
        kwargs = request_args.setdefault("chat_template_kwargs", {})
        for key in self.FIXED_TEMPLATE.consistant_kwargs:
            kwargs.pop(key, None)
        kwargs.update(
            deepcopy({key: recorded[key] for key in self.FIXED_TEMPLATE.consistant_kwargs if key in recorded})
        )

    def create_comparator(self) -> TokenSeqComparator:
        # [FUNCTION DOC] Builds a sequence comparator using this tokenizer, its assistant start
        # string, its special token ids, and its trailing token ids (passed as none when empty).
        """Create a :class:`TokenSeqComparator` configured with this
        tokenizer's model-specific settings."""
        return TokenSeqComparator(
            self.tokenizer,
            assistant_start_str=self._assistant_start_str,
            special_token_ids=self.special_token_ids,
            trim_trailing_ids=self.trailing_token_ids or None,
        )

    def default_template_args(self, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        # [FUNCTION DOC] Returns a copy of the launch template kwargs, with the given tools added
        # under the tools key when non-empty (overriding any launch tools). Does not mutate state.
        """Return launch template kwargs with optional tools for direct rendering."""
        args = dict(self.chat_template_kwargs)
        if tools:
            args["tools"] = tools
        return args

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool = False,
        template_args: dict[str, Any] | None = None,
    ) -> str | list[int]:
        # [FUNCTION DOC] Renders messages through the shared chat template helper using this
        # tokenizer. Inputs: messages; add_generation_prompt flag; tokenize flag (default off, so
        # text is returned); template_args, a complete argument set that replaces (is not merged
        # with) the launch defaults, or none to use the launch defaults. Returns rendered text, or
        # token ids when tokenize is on.
        """Render messages using resolved template kwargs and tools.

        Pass `template_args` as the complete argument set; it is not merged with
        launch defaults. `None` uses this tokenizer's launch defaults.
        """
        # TODO: Use the unified kwargs resolver for launch and request arguments once
        # available, then check whether callers still need this default fallback.
        args = self.chat_template_kwargs if template_args is None else template_args
        return template.apply_chat_template(
            messages,
            tokenizer=self.tokenizer,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **args,
        )

    def postprocess_completion(
        self,
        *,
        choice: dict[str, Any],
        assistant_message: dict[str, Any],
        completion_token_ids: list[int],
    ) -> dict[str, Any]:
        # [FUNCTION DOC] Hook run on an upstream completion before the assistant message is
        # stored. Inputs (keyword only): the completion choice mapping (may be mutated by
        # overrides), the assistant message already parsed by the engine, and the completion token
        # ids. Returns the message to store. The default returns the engine-parsed message
        # unchanged and ignores the choice and token ids.
        """Postprocess an upstream completion and return the message to store.

        The default path trusts SGLang's parsed message. Model families that
        need token-aware response handling override this hook and may update
        ``choice`` before returning their server-side message representation.
        """
        return assistant_message

    def preserve_server_message_state(
        self,
        stored_messages: list[dict[str, Any]],
        request_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC] Hook that re-applies server-owned extra fields from stored messages onto
        # the messages a client replayed. Returns a new list. The default returns a plain copy of
        # the request messages and ignores the stored ones.
        """Restore model-specific server-owned sidecars after client replay."""
        return list(request_messages)

    def _encode_text(self, text: str) -> list[int]:
        # [FUNCTION DOC] HELPER: Encodes text to token ids with the wrapped tokenizer, without
        # adding special tokens (no beginning-of-sequence marker).
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _tokenize_rendered_suffix(
        self,
        base_messages: list[dict[str, Any]],
        appended_messages: list[dict[str, Any]],
        *,
        template_args: dict[str, Any] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        # [FUNCTION DOC] HELPER: Computes token ids of the text a template adds when messages are
        # appended to a base. Renders the base alone (no generation prompt) and the base plus the
        # appended messages (generation prompt per the flag, default off); the second text must
        # start with the first, otherwise raises a value error naming the appended roles (or a
        # generation-prompt placeholder when nothing was appended). Returns the encoding of the
        # extra text, without special tokens. template_args is passed through to rendering.
        """Render *base_messages* and *base_messages + appended_messages*, return
        tokens for the suffix.

        When *add_generation_prompt* is True and *appended_messages* is empty,
        this computes the generation-prompt suffix (the assistant opener tokens).
        """
        text_without = self.apply_chat_template(
            base_messages, add_generation_prompt=False, template_args=template_args
        )
        text_with = self.apply_chat_template(
            base_messages + appended_messages,
            add_generation_prompt=add_generation_prompt,
            template_args=template_args,
        )
        if not text_with.startswith(text_without):
            roles = [msg["role"] for msg in appended_messages] if appended_messages else ["generation_prompt"]
            raise ValueError(f"rendered suffix diff failed for {roles}")
        return self._encode_text(text_with[len(text_without) :])

    def tokenize_additional_messages(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] Verifies the new message list extends the old one append-only with only
        # allowed roles (the shared checker raises on violation), then takes the messages beyond
        # the old length. Renders them under a synthetic base of a dummy system message plus a
        # dummy assistant message built from the last old message (keeping its tool calls) and
        # returns the suffix tokens including the generation prompt. Assumes old messages are
        # non-empty and end with the assistant turn.
        """Compute incremental token IDs for messages appended after the
        pretokenized prefix.

        Appended roles must be listed in ``self.allowed_append_roles``.  The
        method validates that *new_messages* is an append-only extension of
        *old_messages* via ``assert_messages_append_only_with_allowed_role``.

        Args:
            old_messages: Previously stored messages (prefix).
            new_messages: Full new message list (must be a superset of
                *old_messages* with only allowed-role messages appended).
            template_args: Resolved template kwargs and tools, or `None` for launch defaults.

        Returns:
            Incremental token IDs (including the generation prompt) that,
            when merged with pretokenized prefix via ``merge_tokens``,
            form the full prompt token IDs.
        """
        assert_messages_append_only_with_allowed_role(old_messages, new_messages, self.allowed_append_roles)
        appended_messages = new_messages[len(old_messages) :]
        return self._tokenize_rendered_suffix(
            [_DUMMY_SYSTEM, _build_dummy_assistant(old_messages[-1])],
            appended_messages,
            template_args=template_args,
            add_generation_prompt=True,
        )

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] Returns the full prompt token ids: a copy of the pretokenized prefix
        # followed by the incremental tokens for the appended messages. The default does no
        # boundary repair at the junction.
        """Merge *pretokenized_token_ids* with incremental tokens to produce
        the complete prompt token IDs (including generation prompt).

        The default implementation is simple concatenation.  Subclasses
        override this to handle model-specific boundary token logic.
        """
        incremental = self.tokenize_additional_messages(old_messages, new_messages, template_args=template_args)
        return list(pretokenized_token_ids) + incremental


# ---------------------------------------------------------------------------
# Qwen3 implementation
# ---------------------------------------------------------------------------


class Qwen3TITOTokenizer(TITOTokenizer):
    # [CLASS DOC] Qwen3 family base. Binds the qwen3 reasoning parser and qwen25 tool-call parser,
    # the qwen3 fixed template with thinking-clearing disabled, and the default assistant start
    # marker. Records the newline token as the trailing token the comparator ignores. The merge
    # appends the missing newline after a trailing end-of-message token.
    """Qwen3 variant: handles missing newline at the boundary.

    The Qwen3 chat template emits ``<|im_end|>\\n`` after every message, but
    the model stops at ``<|im_end|>`` without generating the trailing ``\\n``.
    ``merge_tokens`` inserts the missing newline so that the pretokenized
    prefix matches the canonical template output.
    """

    reasoning_parser = "qwen3"
    tool_call_parser = "qwen25"

    FIXED_TEMPLATE = FixedTemplate(
        template="qwen3_fixed.jinja",
        extra_kwargs={"clear_thinking": False},
    )

    _default_assistant_start_str: str = "<|im_start|>assistant"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        # [FUNCTION DOC] Initializes the base with the given assistant start string or the class
        # default. Requires the tokenizer to encode a newline as exactly one token (assertion
        # failure otherwise); stores that id, looks up the id of the end-of-message special token,
        # and marks the newline as the only trailing token id. Does not pass special token ids, so
        # they are auto-detected by the comparator.
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
        )
        nl_ids = tokenizer.encode("\n", add_special_tokens=False)
        assert len(nl_ids) == 1, f"Expected single newline token, got {nl_ids}"
        self._newline_id: int = nl_ids[0]
        self._im_end_id: int = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.trailing_token_ids = frozenset({self._newline_id})

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] Computes incremental tokens, then, when the prefix ends with the
        # end-of-message token (the model stops there without the newline the template emits),
        # appends a newline token to a copy of the prefix before concatenating. A prefix ending
        # with the pretraining end-of-text token is deliberately not repaired, so the strict
        # comparator still reports it.
        incremental = self.tokenize_additional_messages(old_messages, new_messages, template_args=template_args)
        prefix = list(pretokenized_token_ids)
        # Weakly post-trained Qwen3 models, notably 0.6B, may emit the pretraining/padding token `<|endoftext|>`.
        # Seen in TITO's March 2026 bring-up; rare in larger models. See https://github.com/radixark/miles/issues/3113.
        # This is model degeneration, not a valid `<|im_end|>` alias; keep the strict checker reporting it.
        if prefix and prefix[-1] == self._im_end_id:
            prefix.append(self._newline_id)
        return prefix + incremental


# Qwen3.5/3.6 and Qwen3-Next-Thinking share the ``<|im_end|>`` boundary
# handling with Qwen3. Their subclasses only own the distinct fixed-template
# contracts layered on that token boundary.


class Qwen35TITOTokenizer(Qwen3TITOTokenizer):
    # [CLASS DOC] Qwen3.5 family: qwen3_coder tool-call parser, the qwen3.5 fixed template with
    # thinking preserved, appendable roles tool, user and assistant (no system), and the
    # add-vision-id kwarg kept as recorded. Boundary handling is inherited from Qwen3.
    """Qwen3.5 template and Qwen3 token boundary."""

    tool_call_parser = "qwen3_coder"

    FIXED_TEMPLATE = FixedTemplate(
        template="qwen3.5_fixed.jinja",
        extra_kwargs={"preserve_thinking": True},
        allowed_append_roles=frozenset({"tool", "user", "assistant"}),
        consistant_kwargs=["add_vision_id"],
    )


class Qwen36TITOTokenizer(Qwen3TITOTokenizer):
    # [CLASS DOC] Qwen3.6 family: same configuration as Qwen3.5 but with the qwen3.6 fixed
    # template file. Boundary handling inherited from Qwen3.
    """Qwen3.6 template and Qwen3 token boundary."""

    tool_call_parser = "qwen3_coder"

    FIXED_TEMPLATE = FixedTemplate(
        template="qwen3.6_fixed.jinja",
        extra_kwargs={"preserve_thinking": True},
        allowed_append_roles=frozenset({"tool", "user", "assistant"}),
        consistant_kwargs=["add_vision_id"],
    )


class Qwen38SmallTITOTokenizer(Qwen3TITOTokenizer):
    # [CLASS DOC] Qwen3.8 small/flash-next family (also used for the qwen4 experimental type):
    # qwen3_coder tool-call parser, its own fixed template, thinking preserved, appendable roles
    # tool, user and assistant, and the add-vision-id, enable-thinking and reasoning-effort kwargs
    # kept as recorded. Boundary handling inherited from Qwen3.
    """Qwen3.8 reasoning-effort template with the Qwen3 token boundary."""

    tool_call_parser = "qwen3_coder"

    FIXED_TEMPLATE = FixedTemplate(
        template="qwen3.8_small_and_flash_next_fixed.jinja",
        extra_kwargs={"preserve_thinking": True},
        allowed_append_roles=frozenset({"tool", "user", "assistant"}),
        consistant_kwargs=["add_vision_id", "enable_thinking", "reasoning_effort"],
    )


class QwenNextTITOTokenizer(Qwen3TITOTokenizer):
    # [CLASS DOC] Qwen3-Thinking-2507 / Qwen3-Next family: only overrides the fixed template
    # (the shared thinking-2507-and-next file, thinking-clearing disabled). Parsers and boundary
    # handling are inherited from Qwen3.
    """Qwen3-Thinking-2507 / Qwen3-Next-Thinking — same boundary behavior as
    Qwen3, distinct (shared) fixed template."""

    FIXED_TEMPLATE = FixedTemplate(
        template="qwen3_thinking_2507_and_next_fixed.jinja",
        extra_kwargs={"clear_thinking": False},
    )


# ---------------------------------------------------------------------------
# GLM family implementation
# ---------------------------------------------------------------------------


class GLM47TITOTokenizer(TITOTokenizer):
    # [CLASS DOC] GLM 4.7 family: glm45 reasoning parser, glm47 tool-call parser, the native
    # template with thinking-clearing disabled, default assistant marker "<|assistant|>". The
    # observation and user marker tokens are the ambiguous boundary tokens and also the trailing
    # token ids ignored by the comparator. Merge drops a trailing boundary token from the prefix.
    # KNOWN ISSUE: the max_trim_tokens attribute (set to 1) is not read anywhere in this module.
    """GLM 4.7 variant: handles ambiguous boundary tokens in ``merge_tokens``.

    ``<|user|>`` and ``<|observation|>`` are both assistant stop tokens *and*
    next-message start tokens in the chat template.  In ``merge_tokens``,
    the last token of the pretokenized prefix is always stripped when it is
    one of these boundary tokens — whether it matches the first incremental
    token (overlap) or differs (e.g. model stopped with ``<|observation|>`` but
    next turn is ``<|user|>`` because the tool call failed and a system message
    is injected instead).
    """

    reasoning_parser = "glm45"
    tool_call_parser = "glm47"

    # GLM's HF-native chat template already exposes a ``clear_thinking`` kwarg,
    # so no fixed-jinja patch is needed for either append surface.
    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"clear_thinking": False},
    )

    max_trim_tokens: int = 1
    _default_assistant_start_str: str = "<|assistant|>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        # [FUNCTION DOC] Initializes the base with the given or default assistant start string,
        # looks up the ids of the observation and user marker tokens, stores them as the ambiguous
        # boundary id set, and uses the same set as the trailing token ids.
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
        )
        self._observation_id: int = tokenizer.convert_tokens_to_ids("<|observation|>")
        self._user_id: int = tokenizer.convert_tokens_to_ids("<|user|>")
        self._ambiguous_boundary_ids: set[int] = {self._observation_id, self._user_id}
        self.trailing_token_ids = frozenset(self._ambiguous_boundary_ids)

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] Computes incremental tokens (which begin with their own boundary marker),
        # then removes the last prefix token if it is an observation or user marker, regardless of
        # whether it equals the first incremental token, and concatenates.
        incremental = self.tokenize_additional_messages(old_messages, new_messages, template_args=template_args)
        prefix = list(pretokenized_token_ids)
        if prefix and prefix[-1] in self._ambiguous_boundary_ids:
            prefix = prefix[:-1]
        return prefix + incremental


class GLM53TITOTokenizer(GLM47TITOTokenizer):
    # [CLASS DOC] GLM-5.3 family: only overrides the fixed template record (native template,
    # thinking-clearing disabled, thinking forced on) and keeps the reasoning-effort kwarg as
    # recorded. Everything else inherited from GLM 4.7.
    """GLM-5.3 native text renderer with the shared GLM token boundary.

    The GLM-5.3 and GLM-5.3-Flash templates start generation with ``<think>`` even when ``enable_thinking=False``, so this family pins ``enable_thinking=True``. Flash support covers tokenizer text inputs, not multimodal processor inputs.
    """

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"clear_thinking": False, "enable_thinking": True},
        consistant_kwargs=["reasoning_effort"],
    )


# ---------------------------------------------------------------------------
# Nemotron 3 implementation
# ---------------------------------------------------------------------------


class Nemotron3TITOTokenizer(Qwen3TITOTokenizer):
    # [CLASS DOC] Nemotron 3 family: Qwen3 boundary handling with the nemotron_3 reasoning parser,
    # qwen3_coder tool-call parser, native template with history-thinking truncation forced off,
    # the low-effort kwarg kept as recorded, and default assistant marker that includes a trailing
    # newline.
    """NVIDIA Nemotron 3 family: ``<|im_end|>\\n`` message boundaries.

    Inherits Qwen3's boundary handling — Nemotron 3 emits the same
    ``<|im_end|>\\n`` after every message and the model stops at
    ``<|im_end|>`` without the trailing newline.

    No fixed jinja is shipped — HF native template is append-only when
    ``truncate_history_thinking=False``.  Multi-user-turn surfaces
    auto-merge that kwarg via ``extra_kwargs`` below; ``{tool}``-only does
    not need it (no user-turn boundary to truncate across).

    The plain-text assistant turn does not roundtrip cleanly under
    sglang's upstream ``nemotron_3`` reasoning parser (it keeps a trailing
    ``\\n`` in ``reasoning_content``), so step-4 ``assistant_text`` soft
    assertion is expected to fail until the parser is patched upstream —
    out of scope for this family registration.
    """

    reasoning_parser = "nemotron_3"
    tool_call_parser = "qwen3_coder"

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"truncate_history_thinking": False},
        consistant_kwargs=["low_effort"],
    )

    _default_assistant_start_str: str = "<|im_start|>assistant\n"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        # [FUNCTION DOC] Delegates to the Qwen3 initializer with the given assistant start string
        # or this class's default (which ends in a newline). Adds nothing else.
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
        )


# ---------------------------------------------------------------------------
# Kimi K2 implementation
# ---------------------------------------------------------------------------


def _kimi_segment_special_token_ids(tokenizer: Any) -> set[int]:
    # [FUNCTION DOC] HELPER: Returns the set of special token ids the comparator treats as segment
    # boundaries for Kimi: all auto-detected special ids minus the id of the middle separator
    # token, which sits inside a role turn.
    """Kimi specials minus ``<|im_middle|>`` (intra-turn role-name/body
    separator, not a role boundary; must not be a segment boundary)."""
    return TokenSeqComparator.collect_special_ids(tokenizer) - {tokenizer.convert_tokens_to_ids("<|im_middle|>")}


class Kimi25TITOTokenizer(TITOTokenizer):
    # [CLASS DOC] Kimi K2.5 family: bundled fixed template, thinking preserved, thinking and
    # tools-timestamp-string kwargs kept as recorded, default assistant marker "<|im_assistant|>".
    # Uses the default merge (plain concatenation) and the Kimi-specific segment special ids. No
    # parsers are bound.
    # KNOWN ISSUE: the docstring says only the tool and user surface is registered, but the fixed
    # template record does not narrow the roles, so all four append roles are allowed.
    """Moonshot Kimi K2.5: ``<|im_end|>`` boundary (no trailing newline).

    K2.5 has no kwarg escape hatch for the "drop reasoning of prior assistants
    once a new non-tool-call assistant arrives" behavior.  Ships a
    bundled fixed jinja that wraps the ``last_non_tool_call_assistant_msg``
    loop in ``{%- if not preserve_thinking -%}`` so multi-user-turn rollout
    can pass ``preserve_thinking=True`` to keep history append-only.  Only the
    ``{tool, user}`` surface is registered (per current onboarding scope).
    """

    FIXED_TEMPLATE = FixedTemplate(
        template="kimi_k25_fixed.jinja",
        extra_kwargs={"preserve_thinking": True},
        consistant_kwargs=["thinking", "tools_ts_str"],
    )

    _default_assistant_start_str: str = "<|im_assistant|>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        # [FUNCTION DOC] Initializes the base with the given or default assistant start string and
        # the Kimi segment special ids computed from the tokenizer.
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
            special_token_ids=_kimi_segment_special_token_ids(tokenizer),
        )


class Kimi26TITOTokenizer(TITOTokenizer):
    # [CLASS DOC] Kimi K2.6 family: native template, thinking preserved, tools-timestamp-string
    # kwarg kept as recorded, kimi_k2 reasoning parser, kimi_k2_raw_id tool-call parser (keeps
    # model-emitted tool call ids), default merge, Kimi segment special ids.
    """Moonshot Kimi K2.6: same boundary as K2.5 + native ``preserve_thinking`` kwarg.

    K2.6's HF-native template already carries the ``preserve_thinking`` gate
    that K2.5 needs patched in.  No bundled fixed
    template required; ``{tool, user}`` row registers ``template=None`` and
    auto-merges ``preserve_thinking=True`` for multi-user-turn rollout.

    Tool-call parser is bound to ``kimi_k2_raw_id`` rather than ``kimi_k2``:
    RL trajectories need the model-emitted ``tool_call_id`` to round-trip
    verbatim across turns (no ``history_tool_calls_cnt`` renumbering), and
    miles is the primary consumer of this TITO family.
    """

    reasoning_parser = "kimi_k2"
    tool_call_parser = "kimi_k2_raw_id"

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"preserve_thinking": True},
        consistant_kwargs=["tools_ts_str"],
    )

    _default_assistant_start_str: str = "<|im_assistant|>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        # [FUNCTION DOC] Same as the K2.5 initializer: given or default assistant start string and
        # the Kimi segment special ids computed from the tokenizer.
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
            special_token_ids=_kimi_segment_special_token_ids(tokenizer),
        )


# ---------------------------------------------------------------------------
# MiniMax M2 family implementation (M2.5 and M2.7 share tokenizer/arch and
# stop-token semantics; only their default system identity strings differ).
# ---------------------------------------------------------------------------


class MinimaxM25TITOTokenizer(TITOTokenizer):
    # [CLASS DOC] MiniMax M2.5 family: minimax-append-think reasoning parser, minimax-m2 tool-call
    # parser, bundled M2.5 fixed template with thinking-clearing disabled, appendable roles tool,
    # user and assistant, default assistant marker "]~b]ai". The newline token is the trailing
    # token id; the merge appends a newline after a trailing end-of-sequence marker.
    """MiniMax-M2.5 family: bespoke ``]~!b[`` / ``[e~[`` / ``]~b]`` tag set.

    Shares tokenizer.json (sha256) and architecture (MiniMaxM2ForCausalLM)
    with M2.7 — only the chat template's default system identity string
    differs (``MiniMax-M2.5`` vs ``MiniMax-M2.7``).  Stop-token handling
    (``[e~[`` / trailing newline) is identical to M2.7.

    Reasoning is gated by a per-message ``last_user_index`` check:
    ``reasoning_content`` is only rendered for assistant turns *after* the
    last ``user`` — appending a new ``user`` therefore strips prior assistant
    ``<think>`` blocks and breaks append-only.  Only ``{tool}`` surface is
    registered on HF-native template for that reason; multi-user-turn
    requires the fixed jinja with ``clear_thinking=False`` to always
    preserve history reasoning.  The fixed Jinja renders tool, user, and
    assistant appends, but ignores mid-session system messages, so that role
    is excluded from its capability.
    """

    reasoning_parser = "minimax-append-think"
    tool_call_parser = "minimax-m2"

    FIXED_TEMPLATE = FixedTemplate(
        template="minimax_m25_fixed.jinja",
        extra_kwargs={"clear_thinking": False},
        allowed_append_roles=frozenset({"tool", "user", "assistant"}),
    )

    _default_assistant_start_str: str = "]~b]ai"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        # [FUNCTION DOC] Initializes the base with the given or default assistant start string.
        # Requires the newline to encode as exactly one token (assertion failure otherwise); stores
        # its id, looks up the end-of-sequence marker id, and sets the trailing token ids to the
        # newline only.
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
        )
        nl_ids = tokenizer.encode("\n", add_special_tokens=False)
        assert len(nl_ids) == 1, f"Expected single newline token, got {nl_ids}"
        self._newline_id: int = nl_ids[0]
        self._eos_id: int = tokenizer.convert_tokens_to_ids("[e~[")
        self.trailing_token_ids = frozenset({self._newline_id})

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] Computes incremental tokens; when the prefix ends with the
        # end-of-sequence marker, appends a newline token to a copy of the prefix (the model stops
        # at the marker without the newline the template emits); then concatenates.
        incremental = self.tokenize_additional_messages(old_messages, new_messages, template_args=template_args)
        prefix = list(pretokenized_token_ids)
        if prefix and prefix[-1] == self._eos_id:
            prefix.append(self._newline_id)
        return prefix + incremental


class MinimaxM27TITOTokenizer(MinimaxM25TITOTokenizer):
    # [CLASS DOC] MiniMax M2.7 family: identical to M2.5 except the fixed template file is the
    # M2.7 one.
    """MiniMax-M2.7 family: tokenizer / arch / stop-token semantics identical
    to M2.5; the chat template only differs by default system identity string.

    Inherits parsers, ``__init__``, ``merge_tokens``, and
    ``_default_assistant_start_str`` from M2.5; only ``FIXED_TEMPLATE``
    is rebound to ``minimax_m27_fixed.jinja`` so the fixed-template lookup
    points at the M2.7-derived jinja.
    """

    FIXED_TEMPLATE = FixedTemplate(
        template="minimax_m27_fixed.jinja",
        extra_kwargs={"clear_thinking": False},
        allowed_append_roles=frozenset({"tool", "user", "assistant"}),
    )


# ---------------------------------------------------------------------------
# DeepSeek V3.2 implementation
# ---------------------------------------------------------------------------

_DEEPSEEK_MODE_KWARG_ALIASES = frozenset({"thinking_mode", "enable_thinking", "thinking"})


class DeepSeekV32TITOTokenizer(TITOTokenizer):
    # [CLASS DOC] DeepSeek V3.2 family: no jinja template, rendering goes through the vendored
    # encoder bridge. Binds the deepseek-v3 reasoning parser and deepseekv32 tool-call parser;
    # native template with history-thinking dropping disabled; default-bos-token and context
    # kwargs kept as recorded; appendable roles are the default four. Adds a rule that normalizes
    # the thinking-mode aliases into one canonical thinking flag. Uses the default merge.
    """DeepSeek V3.2 — miles' vendored copy of the official ``encoding_dsv32``.

    V3.2 ships no jinja chat_template; prompts render through
    ``templates.encoding_dsv32.encode_messages``, and miles'
    ``apply_chat_template`` routes any V3.2 tokenizer to the thin
    ``chat_template_utils.deepseek`` bridge.  TITO incremental tokenization
    rides that same bridge.

    Upstream ``encoding_dsv32`` gates every thinking block on
    ``last_user_idx``: appending a *user* turn re-classifies every prior
    assistant as "before last user" and strips its thinking block, which is
    not append-only.  The vendored copy honors ``drop_thinking=False`` at the
    render level (like ``encoding_dsv4``), so every surface pins it and the
    ``{tool, user}`` surface becomes legal.
    """

    reasoning_parser = "deepseek-v3"
    tool_call_parser = "deepseekv32"

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"drop_thinking": False},
        consistant_kwargs=["add_default_bos_token", "context"],
    )

    _DEFAULT_ASSISTANT_START = "<｜Assistant｜>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        # [FUNCTION DOC] Initializes the base with the given or default assistant start string
        # (the assistant sentinel) and, as segment special ids, exactly the user and assistant
        # sentinel token ids (not the auto-detected set).
        # V3.2 has no jinja template, so assistant_start_str can't be sniffed
        # from one; pin it explicitly.  The comparator keys off the User /
        # Assistant sentinels to find assistant-content boundaries.
        super().__init__(
            tokenizer,
            chat_template_kwargs=chat_template_kwargs,
            assistant_start_str=assistant_start_str or self._DEFAULT_ASSISTANT_START,
            special_token_ids={
                tokenizer.convert_tokens_to_ids("<｜User｜>"),
                tokenizer.convert_tokens_to_ids("<｜Assistant｜>"),
            },
        )

    def request_arg_rules(self) -> list[Callable[..., None]]:
        # [FUNCTION DOC] Returns the base rule list with the thinking-resolution rule appended as
        # the last rule.
        rules = super().request_arg_rules()
        rules.append(self.resolve_thinking)
        return rules

    def resolve_template_kwargs(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        # [FUNCTION DOC] Rule: runs the base merge, then makes the three thinking-mode aliases
        # (thinking_mode, enable_thinking, thinking) act as one setting. Looks at three sources in
        # priority order: recorded turn kwargs, request kwargs, launch kwargs; the first source
        # containing any alias supplies its alias entries (all aliases it has). All aliases are
        # removed from the merged kwargs and the chosen entries are put back, so a lower-priority
        # alias cannot linger next to a higher-priority one. If no source has an alias, nothing is
        # re-added.
        super().resolve_template_kwargs(request_args, request_source=request_source, turn_args=turn_args)
        # Thinking aliases are one setting: history overrides the request and launch defaults.
        for source in (
            (turn_args or {}).get("chat_template_kwargs") or {},
            request_source.get("chat_template_kwargs") or {},
            self.chat_template_kwargs,
        ):
            mode = {key: source[key] for key in _DEEPSEEK_MODE_KWARG_ALIASES if key in source}
            if mode:
                break
        kwargs = request_args["chat_template_kwargs"]
        for alias in _DEEPSEEK_MODE_KWARG_ALIASES:
            kwargs.pop(alias, None)
        kwargs.update(mode)

    def resolve_thinking(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        # [FUNCTION DOC] Rule: asks the V3.2 encoder bridge whether thinking is enabled for the
        # merged template kwargs, removes all thinking-mode aliases, and stores the boolean result
        # under the canonical thinking key (the key the engine's reasoning parser reads). Requires
        # the template kwargs entry to exist already. Mutates the request.
        kwargs = request_args["chat_template_kwargs"]
        thinking = deepseek.V32.render_thinking_enabled(kwargs)
        for alias in _DEEPSEEK_MODE_KWARG_ALIASES:
            kwargs.pop(alias, None)
        # SGLang's reasoning parser reads the canonical thinking flag.
        kwargs["thinking"] = thinking


# ---------------------------------------------------------------------------
# DeepSeek V4 implementation
# ---------------------------------------------------------------------------


class DeepSeekV4TITOTokenizer(TITOTokenizer):
    # [CLASS DOC] DeepSeek V4 family: like V3.2 (no jinja template, encoder bridge, same alias
    # normalization and canonical thinking rule using the V4 encoder) with deepseek-v4 reasoning
    # parser and deepseekv4 tool-call parser, appendable roles tool, user and assistant, and the
    # reasoning-effort kwarg also kept as recorded. Trailing token ids are the assistant sentinel
    # and the think open/close tags. Its incremental tokenization diffs full renders of old and
    # new messages instead of using a dummy base.
    """DeepSeek V4 — official encoder via sglang's ``encoding_dsv4``.

    Like V3.2, V4 ships no jinja chat_template; miles' ``apply_chat_template``
    routes any V4 tokenizer to the ``chat_template_utils.deepseek`` bridge, and
    TITO incremental tokenization rides that same bridge to stay byte-aligned
    with what the runtime serves.
    """

    reasoning_parser = "deepseek-v4"
    tool_call_parser = "deepseekv4"

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"drop_thinking": False},
        allowed_append_roles=frozenset({"tool", "user", "assistant"}),
        consistant_kwargs=["add_default_bos_token", "context", "reasoning_effort"],
    )

    _DEFAULT_ASSISTANT_START = "<｜Assistant｜>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        # [FUNCTION DOC] Initializes the base with the given or default assistant start string and
        # the user and assistant sentinel ids as segment special ids. Then stores the assistant
        # sentinel id and the ids of the think open and close tags, and sets the trailing token
        # ids to the union of those three.
        super().__init__(
            tokenizer,
            chat_template_kwargs=chat_template_kwargs,
            assistant_start_str=assistant_start_str or self._DEFAULT_ASSISTANT_START,
            special_token_ids={
                tokenizer.convert_tokens_to_ids("<｜User｜>"),
                tokenizer.convert_tokens_to_ids("<｜Assistant｜>"),
            },
        )
        self._assistant_id: int = tokenizer.convert_tokens_to_ids("<｜Assistant｜>")
        self._think_bracket_ids: set[int] = {
            tokenizer.convert_tokens_to_ids("<think>"),
            tokenizer.convert_tokens_to_ids("</think>"),
        }
        self.trailing_token_ids = frozenset({self._assistant_id} | self._think_bracket_ids)

    def request_arg_rules(self) -> list[Callable[..., None]]:
        # [FUNCTION DOC] Returns the base rule list with the thinking-resolution rule appended as
        # the last rule.
        rules = super().request_arg_rules()
        rules.append(self.resolve_thinking)
        return rules

    def resolve_template_kwargs(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        # [FUNCTION DOC] Rule: identical to the V3.2 version. Runs the base merge, then collapses
        # the thinking-mode aliases (thinking_mode, enable_thinking, thinking) to the alias entries
        # of the highest-priority source that has any (recorded turn, then request, then launch).
        # KNOWN ISSUE: duplicated verbatim from the V3.2 class instead of shared.
        super().resolve_template_kwargs(request_args, request_source=request_source, turn_args=turn_args)
        # Thinking aliases are one setting: history overrides the request and launch defaults.
        for source in (
            (turn_args or {}).get("chat_template_kwargs") or {},
            request_source.get("chat_template_kwargs") or {},
            self.chat_template_kwargs,
        ):
            mode = {key: source[key] for key in _DEEPSEEK_MODE_KWARG_ALIASES if key in source}
            if mode:
                break
        kwargs = request_args["chat_template_kwargs"]
        for alias in _DEEPSEEK_MODE_KWARG_ALIASES:
            kwargs.pop(alias, None)
        kwargs.update(mode)

    def resolve_thinking(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        # [FUNCTION DOC] Rule: same as the V3.2 version but asks the V4 encoder bridge whether
        # thinking is enabled; removes all aliases and stores the boolean under the canonical
        # thinking key.
        kwargs = request_args["chat_template_kwargs"]
        thinking = deepseek.V4.render_thinking_enabled(kwargs)
        for alias in _DEEPSEEK_MODE_KWARG_ALIASES:
            kwargs.pop(alias, None)
        # SGLang's reasoning parser reads the canonical thinking flag.
        kwargs["thinking"] = thinking

    def tokenize_additional_messages(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] Verifies append-only extension with allowed roles (shared checker raises
        # otherwise). Renders the old messages without a generation prompt and the new messages
        # with one; raises a value error if the second text does not start with the first.
        # Returns the encoding of the remaining text. Unlike the base, no dummy prefix is used,
        # because V4 merges adjacent tool and user turns.
        """Diff real-history renders because V4 folds adjacent ``tool``/``user`` turns."""
        assert_messages_append_only_with_allowed_role(old_messages, new_messages, self.allowed_append_roles)
        text_old = self.apply_chat_template(old_messages, add_generation_prompt=False, template_args=template_args)
        text_new = self.apply_chat_template(new_messages, add_generation_prompt=True, template_args=template_args)
        if not text_new.startswith(text_old):
            raise ValueError(
                "deepseek_v4 render is not append-only for the appended messages "
                "(prefix render changed; check drop_thinking and tool-result ordering)"
            )
        return self._encode_text(text_new[len(text_old) :])


class DeepSeekV41TITOTokenizer(DeepSeekV4TITOTokenizer):
    # [CLASS DOC] DeepSeek V4.1 family: V4 behavior with deepseek-v41 reasoning parser and
    # deepseekv41 tool-call parser, and a launch default that pins the canonical thinking flag.
    """DeepSeek V4.1 — official encoder via sglang's ``encoding_dsv41``; the
    V4 incremental-render machinery applies unchanged."""

    reasoning_parser = "deepseek-v41"
    tool_call_parser = "deepseekv41"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        # [FUNCTION DOC] Runs the V4 initializer, then replaces the launch template kwargs with a
        # copy that has the thinking key set to what the V4.1 encoder bridge reports for the
        # current launch kwargs.
        super().__init__(tokenizer, chat_template_kwargs=chat_template_kwargs, assistant_start_str=assistant_start_str)
        self.chat_template_kwargs = {
            **self.chat_template_kwargs,
            "thinking": deepseek.V41.render_thinking_enabled(self.chat_template_kwargs),
        }


class InklingTITOTokenizer(TITOTokenizer):
    # [CLASS DOC] Inkling family: bundled inkling fixed template (default role surface), the
    # reasoning-effort kwarg kept as recorded, default assistant marker "<|message_model|>", and
    # all four message-role sentinels (user, model, system, tool) as segment special ids. Parses
    # completions with a lazily created response parser and preserves content blocks of stored
    # assistant messages. Default merge.
    """Inkling family (Inkling / Inkling-Small).

    The runtime serves Inkling through sglang's token-level renderer
    (``chat_encoding_spec == "inkling"``).  The fixed template matches its
    empty scalar-content behavior: no empty text block, and no bare assistant
    terminator when the turn contains no rendered blocks.  All four message-role
    sentinels remain comparator boundaries so non-assistant mismatches are hard
    failures after an assistant turn.
    """

    FIXED_TEMPLATE = FixedTemplate(template="inkling_fixed.jinja", consistant_kwargs=["reasoning_effort"])

    _DEFAULT_ASSISTANT_START = "<|message_model|>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        # [FUNCTION DOC] Initializes the base with the given or default assistant start string and
        # the four role sentinel ids as segment special ids, and leaves the response parser unset
        # (created on first use).
        super().__init__(
            tokenizer,
            chat_template_kwargs=chat_template_kwargs,
            assistant_start_str=assistant_start_str or self._DEFAULT_ASSISTANT_START,
            special_token_ids={
                tokenizer.convert_tokens_to_ids("<|message_user|>"),
                tokenizer.convert_tokens_to_ids("<|message_model|>"),
                tokenizer.convert_tokens_to_ids("<|message_system|>"),
                tokenizer.convert_tokens_to_ids("<|message_tool|>"),
            },
        )
        self._response_parser = None

    def postprocess_completion(
        self,
        *,
        choice: dict[str, Any],
        assistant_message: dict[str, Any],
        completion_token_ids: list[int],
    ) -> dict[str, Any]:
        # [FUNCTION DOC] Parses the completion token ids with the Inkling response parser (created
        # lazily on first call from the tokenizer), passing the choice's finish reason. Side
        # effects on the choice: replaces its message with the parsed client-facing message; in
        # its meta info (created if missing) records the parser name and, if parsing failed, the
        # parse error; if there is no error, the client message has tool calls and the finish
        # reason is stop, changes the finish reason to tool calls. Ignores the engine-parsed
        # assistant message. Returns the parsed stored message.
        if self._response_parser is None:
            self._response_parser = InklingResponseParser(self.tokenizer)
        parsed = self._response_parser.parse(
            completion_token_ids,
            finish_reason=choice.get("finish_reason"),
        )
        choice["message"] = parsed.client_message
        meta_info = choice.setdefault("meta_info", {})
        meta_info["miles_response_parser"] = parsed.parser_name
        if parsed.parse_error is not None:
            meta_info["miles_response_parse_error"] = parsed.parse_error
        elif parsed.client_message.get("tool_calls") and choice.get("finish_reason") == "stop":
            choice["finish_reason"] = "tool_calls"
        return parsed.stored_message

    def preserve_server_message_state(
        self,
        stored_messages: list[dict[str, Any]],
        request_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC] Returns a copy of the request messages in which, for each position that
        # exists in the stored messages, a stored assistant message that strictly matches the
        # request message at that position and carries content blocks passes those blocks into a
        # new copy of the request message. Non-assistant, non-matching or block-less positions are
        # left as the request had them. Stops at the end of the stored list.
        preserved = list(request_messages)
        for index, request_message in enumerate(preserved):
            if index >= len(stored_messages):
                break
            stored_message = stored_messages[index]
            if stored_message.get("role") != "assistant" or not strict_message_matches(
                stored_message, request_message
            ):
                continue
            if "content_blocks" in stored_message:
                preserved[index] = {**request_message, "content_blocks": stored_message["content_blocks"]}
        return preserved


# ---------------------------------------------------------------------------
# Enum + Factory
# ---------------------------------------------------------------------------


class TITOTokenizerType(StrEnum):
    # [CLASS DOC] String enumeration of supported tokenizer families; the values are the strings
    # accepted by the command-line model option (default, qwen3, qwen35, qwen36, qwen38small,
    # qwen4exp, qwennext, glm47, glm53, nemotron3, kimi25, kimi26, minimax_m25, minimax_m27,
    # deepseekv32, deepseekv4, deepseekv41, inkling). qwen4exp is an alias of the qwen3.8 small
    # class.
    DEFAULT = "default"
    QWEN3 = "qwen3"
    QWEN35 = "qwen35"
    QWEN36 = "qwen36"
    QWEN38_SMALL = "qwen38small"
    QWEN4_EXP = "qwen4exp"
    QWENNEXT = "qwennext"
    GLM47 = "glm47"
    GLM53 = "glm53"
    NEMOTRON3 = "nemotron3"
    KIMI25 = "kimi25"
    KIMI26 = "kimi26"
    MINIMAX_M25 = "minimax_m25"
    MINIMAX_M27 = "minimax_m27"
    DEEPSEEKV32 = "deepseekv32"
    DEEPSEEKV4 = "deepseekv4"
    DEEPSEEKV41 = "deepseekv41"
    INKLING = "inkling"

    @classmethod
    def get_tokenizer_class(cls, t: TITOTokenizerType) -> type[TITOTokenizer]:
        # [FUNCTION DOC] Class method mapping a family enum member to its tokenizer class (the base
        # class for the default member; the qwen3.8 small class for both qwen38small and qwen4exp).
        # Raises a value error for an unrecognized member.
        """Resolve the concrete ``TITOTokenizer`` subclass for *t*."""
        match t:
            case cls.DEFAULT:
                return TITOTokenizer
            case cls.QWEN3:
                return Qwen3TITOTokenizer
            case cls.QWEN35:
                return Qwen35TITOTokenizer
            case cls.QWEN36:
                return Qwen36TITOTokenizer
            case cls.QWEN38_SMALL | cls.QWEN4_EXP:
                return Qwen38SmallTITOTokenizer
            case cls.QWENNEXT:
                return QwenNextTITOTokenizer
            case cls.GLM47:
                return GLM47TITOTokenizer
            case cls.GLM53:
                return GLM53TITOTokenizer
            case cls.NEMOTRON3:
                return Nemotron3TITOTokenizer
            case cls.KIMI25:
                return Kimi25TITOTokenizer
            case cls.KIMI26:
                return Kimi26TITOTokenizer
            case cls.MINIMAX_M25:
                return MinimaxM25TITOTokenizer
            case cls.MINIMAX_M27:
                return MinimaxM27TITOTokenizer
            case cls.DEEPSEEKV32:
                return DeepSeekV32TITOTokenizer
            case cls.DEEPSEEKV4:
                return DeepSeekV4TITOTokenizer
            case cls.DEEPSEEKV41:
                return DeepSeekV41TITOTokenizer
            case cls.INKLING:
                return InklingTITOTokenizer
            case _:
                raise ValueError(f"Unknown TITOTokenizerType: {t!r}")


def get_tito_tokenizer(
    tokenizer: Any,
    tokenizer_type: TITOTokenizerType | str = TITOTokenizerType.DEFAULT,
    chat_template_kwargs: dict[str, Any] | None = None,
    assistant_start_str: str | None = None,
) -> TITOTokenizer:
    # [FUNCTION DOC] Factory. Raises a value error if the tokenizer is none. Converts a string
    # type to the enum (an invalid string raises a value error). Instantiates the family class
    # with the tokenizer, the template kwargs, and the assistant start string only when one is
    # given (otherwise the class default applies). Returns the instance.
    """Create a ``TITOTokenizer`` instance.

    Args:
        tokenizer: HuggingFace tokenizer object.
        tokenizer_type: Explicit type (string or enum).  Corresponds to the
            ``--tito-model`` CLI argument.
        chat_template_kwargs: Extra kwargs forwarded to ``template.apply_chat_template``.
        assistant_start_str: Decoded text prefix identifying assistant content
            segments (e.g. ``"<|im_start|>assistant"``).  Auto-detected from
            the chat template by default; pass explicitly to override.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must not be None")
    if isinstance(tokenizer_type, str):
        tokenizer_type = TITOTokenizerType(tokenizer_type)
    cls = TITOTokenizerType.get_tokenizer_class(tokenizer_type)
    kwargs: dict[str, Any] = {"chat_template_kwargs": chat_template_kwargs}
    if assistant_start_str is not None:
        kwargs["assistant_start_str"] = assistant_start_str
    return cls(tokenizer, **kwargs)


# ---------------------------------------------------------------------------
# Fixed-template resolution (one template per family)
# ---------------------------------------------------------------------------


def resolve_fixed_chat_template(
    tito_model: TITOTokenizerType | str,
) -> tuple[str | None, dict[str, Any]]:
    # [FUNCTION DOC] Looks up the family's fixed template record (string types are converted to
    # the enum first) and returns the absolute path of its bundled template file as a string, or
    # none for native templates, together with a copy of the forced kwargs. Logs the family, path,
    # kwargs and sorted allowed roles at info level.
    """The family's fixed chat template and required kwargs.

    Returns ``(template_path, extra_kwargs)``:

    - ``template_path``: absolute path to a bundled ``.jinja`` file, or ``None``
      when the family registers HF-native (kwargs-only fix).
    - ``extra_kwargs``: kwargs owned by the registration and merged into
      ``template.apply_chat_template``.  Conflicting caller values are invalid.

    Template resolution depends only on ``tito_model``.  The DEFAULT family
    resolves to its native template (``None``) with no fixed kwargs.
    """
    if isinstance(tito_model, str):
        tito_model = TITOTokenizerType(tito_model)

    cls = TITOTokenizerType.get_tokenizer_class(tito_model)
    fixed = cls.FIXED_TEMPLATE

    path = str(TEMPLATE_DIR / fixed.template) if fixed.template else None
    logger.info(
        "tito_model=%s -> template=%s kwargs=%s allowed_append_roles=%s",
        tito_model.value,
        path,
        fixed.extra_kwargs,
        sorted(fixed.allowed_append_roles),
    )
    return path, dict(fixed.extra_kwargs)


# ---------------------------------------------------------------------------
# sglang parser resolution (per-family binding + assert-equal on user input)
# ---------------------------------------------------------------------------


def resolve_reasoning_and_tool_call_parser(
    tito_model: TITOTokenizerType | str,
    user_reasoning_parser: str | None = None,
    user_tool_call_parser: str | None = None,
) -> tuple[str | None, str | None]:
    # [FUNCTION DOC] Resolves both parser names for a family (string types converted to the enum).
    # Each is resolved independently by the nested helper; returns the pair (reasoning parser,
    # tool-call parser). Raises a value error from the helper on a user/family disagreement.
    """Resolve sglang ``--reasoning-parser`` and ``--tool-call-parser`` for the
    given TITO family.

    Both parsers are bound on the TITO subclass as class attributes because
    the model's reasoning / tool-call emission shapes are per-family facts.
    For each parser independently:

    * If the user didn't pass a value, return the family's bound value
      (which may itself be ``None`` for ``DEFAULT`` or unbound subclasses
      — the caller is then responsible for supplying one downstream).
    * If the user passed a value and the family is bound, assert equality;
      a mismatch is a configuration bug and raises ``ValueError`` rather
      than silently overriding.
    * If the user passed a value and the family is unbound, accept it.

    Returns ``(reasoning_parser, tool_call_parser)``.
    """
    if isinstance(tito_model, str):
        tito_model = TITOTokenizerType(tito_model)
    cls = TITOTokenizerType.get_tokenizer_class(tito_model)

    def _resolve_one(field: str, bound: str | None, user: str | None) -> str | None:
        # [FUNCTION DOC] HELPER: Picks one parser name. No user value: the family's bound value
        # (possibly none). Family unbound: the user value. Both set and equal: the user value.
        # Both set and different: raises a value error whose message names the command-line flag
        # (field name with underscores turned into hyphens), both values and the family.
        if user is None:
            return bound
        if bound is None:
            return user
        if user != bound:
            raise ValueError(
                f"--{field.replace('_', '-')}={user!r} disagrees with the parser "
                f"registered for tito_model={tito_model.value!r}: {bound!r}. The "
                f"parser is bound on the TITO subclass; either pass {bound!r} or "
                f"omit the flag to auto-resolve."
            )
        return user

    return (
        _resolve_one("reasoning_parser", cls.reasoning_parser, user_reasoning_parser),
        _resolve_one("tool_call_parser", cls.tool_call_parser, user_tool_call_parser),
    )
