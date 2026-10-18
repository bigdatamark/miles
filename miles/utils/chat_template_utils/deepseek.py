# [MODULE DOC] Rendering of chat messages for DeepSeek checkpoints that ship no Jinja template.
# Family objects (V3.2, V4, V4.1) wrap an encoder module and share the rendering logic;
# differences are which roles open an assistant turn (and so get a generation suffix), how tools
# are serialized, and V4.1's reasoning effort handling. A tokenizer is routed to a family by the
# model type string in its checkpoint's config file. Module-level instances V32, V4, V41 and a
# registry keyed by model type (deepseek_v32, deepseek_v4, deepseek_v41) are defined here.
# Requires sglang's encoders at import time. Environment read: SGLANG_DSV41_REASONING_EFFORT.

"""Shared bridge for the DeepSeek official-encoder families (V3.2, V4, V4.1).

None of the families ships a jinja chat_template: V4 and V4.1 render through
sglang's ``encoding_dsv4`` / ``encoding_dsv41``, while V3.2 uses miles' vendored
``templates.encoding_dsv32``.  Both modules share one calling convention,
and miles' ``apply_chat_template`` routes any matching tokenizer here.  Each
family is one ``DeepSeekFamily`` instance wrapping its encoder module;
everything else is shared.
"""

from __future__ import annotations

import copy
import functools
import json
import os
from collections.abc import Callable
from typing import Any

from sglang.srt.entrypoints.openai import chat_encoding, encoding_dsv4, encoding_dsv41
from sglang.srt.entrypoints.openai.protocol import Tool

from miles.utils.chat_template_utils.templates import encoding_dsv32

_ASSISTANT_SP_TOKEN = "<｜Assistant｜>"


@functools.cache
def _read_model_type(name_or_path: str) -> str:
    # [FUNCTION DOC] HELPER: Cached per path. Returns the model type string from the config file
    # inside the directory, or an empty string for an empty path, a missing/unreadable/undecodable
    # or invalid-JSON file, a non-object JSON value, or an absent/empty/null model type. Reads the
    # filesystem on first call per path. Same logic as the helper in the inkling module.
    """Read ``model_type`` from a checkpoint's ``config.json`` (cached per path)."""
    if not name_or_path:
        return ""
    config_path = os.path.join(name_or_path, "config.json")
    if not os.path.isfile(config_path):
        return ""
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(config, dict):
        return ""
    return config.get("model_type", "") or ""


def _inject_tools_into_system(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_payload: Callable[[Tool], dict[str, Any]] = Tool.model_dump,
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Deep-copies the messages (input untouched). If the list is empty or
    # the first message is not a system message, inserts an empty-content system message at the
    # front. Sets the first message's tools entry to the list produced by validating each tool
    # dictionary against the server's tool model and converting it with the payload function
    # (default: full model dump with defaults filled). Returns the new list. Validation errors
    # propagate.
    """Put *tools* in the system message, where ``encode_messages`` reads them.

    The encoder serializes each tool dict verbatim into ``<functions>``, so they
    must round-trip through the same pydantic payload sglang's ``serving_chat``
    builds (``Tool.model_dump()`` fills defaults / orders fields; V4.1 keeps
    only the fields the client sent) or the token ids drift from what sglang serves.
    """
    out = copy.deepcopy(messages)
    if not out or out[0].get("role") != "system":
        out.insert(0, {"role": "system", "content": ""})
    out[0]["tools"] = [tool_payload(Tool.model_validate(t)) for t in tools]
    return out


class DeepSeekFamily:
    # [CLASS DOC] Base for the families. Subclasses set the template attribute (an encoder module
    # exposing an encode function and thinking start/end token constants) and implement the
    # generation-suffix rule; they may override config building and tool injection.
    """Shared behavior for the DeepSeek official-encoder families."""

    template: Any

    def _build_encode_config(self, kwargs: dict) -> dict:
        # [FUNCTION DOC] HELPER: Copies the keyword options (input untouched). Removes the
        # enable-thinking option and, if it was given (not None), sets the thinking mode to
        # thinking when truthy else chat, unless a thinking mode was already supplied. Does the
        # same for the thinking option (also removed). Finally defaults the thinking mode to
        # thinking. So an explicit thinking mode wins, then enable-thinking, then thinking.
        # Returns the options for the encoder.
        kwargs = dict(kwargs)
        if (enable_thinking := kwargs.pop("enable_thinking", None)) is not None:
            kwargs.setdefault("thinking_mode", "thinking" if enable_thinking else "chat")
        # sglang can accept thinking as a kwarg to set thinking_mode, like dsv3.1
        if (thinking := kwargs.pop("thinking", None)) is not None:
            kwargs.setdefault("thinking_mode", "thinking" if thinking else "chat")
        # encode_messages requires thinking_mode; thinking-on matches the server default
        kwargs.setdefault("thinking_mode", "thinking")
        return kwargs

    def render_thinking_enabled(self, chat_template_kwargs: dict[str, Any]) -> bool:
        # [FUNCTION DOC] True when the options resolve (through the family's encode config
        # building, including subclass overrides) to the thinking mode.
        """Whether *chat_template_kwargs* resolve to thinking mode, through the
        same resolution path ``render_messages`` uses."""
        return self._build_encode_config(chat_template_kwargs)["thinking_mode"] == "thinking"

    def _generation_prompt_suffix(self, tail_role: str | None, thinking_token: str) -> str | None:
        # [FUNCTION DOC] HELPER: Abstract hook. Subclasses return the text the encoder appends to
        # open the next assistant turn after a message with the given last role (given the
        # thinking start/end token), or None if no suffix applies. The base raises a
        # not-implemented error.
        raise NotImplementedError

    def _inject_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Default tool placement: puts tools into the system message with the
        # default (full model dump) payload.
        return _inject_tools_into_system(messages, tools)

    def render_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        # [FUNCTION DOC] Builds the encode options from the extra keyword options, injects the
        # tools (if any) into the system message through the family hook, and calls the
        # encoder's message encoder. When the generation prompt is wanted, or the message list is
        # empty, returns the encoder output as is. Otherwise picks the thinking start token if the
        # mode is thinking, else the thinking end token, asks the family for the suffix for the
        # last message's role, and strips that suffix from the end of the output only if there is
        # a suffix and the output ends with it; else returns the output unchanged. Arguments in
        # tool calls must already be JSON strings.
        """Render *messages* into this family's prompt via sglang ``encode_messages``.

        Tool_call ``arguments`` must already be JSON strings; *tools*, if given, are
        injected into the system message (see ``_inject_tools_into_system``).  The
        encoder appends a family- and role-specific next-assistant suffix after
        generation-ready tails; ``add_generation_prompt=False`` strips it.
        """
        encode_config = self._build_encode_config(kwargs)
        if tools:
            messages = self._inject_tools(messages, tools)
        rendered = self.template.encode_messages(messages, **encode_config)
        if add_generation_prompt or not messages:
            return rendered

        thinking_token = (
            self.template.thinking_start_token
            if encode_config["thinking_mode"] == "thinking"
            else self.template.thinking_end_token
        )
        suffix = self._generation_prompt_suffix(messages[-1].get("role"), thinking_token)
        if suffix is None or not rendered.endswith(suffix):
            return rendered
        return rendered[: -len(suffix)]


class DeepSeekV32Family(DeepSeekFamily):
    # [CLASS DOC] V3.2 family using the vendored encoder module.
    template = encoding_dsv32

    def _generation_prompt_suffix(self, tail_role: str | None, thinking_token: str) -> str | None:
        # [FUNCTION DOC] HELPER: After a user or developer message the suffix is the assistant
        # special token followed by the thinking token; after a tool message it is two newlines
        # followed by the thinking token; otherwise none.
        if tail_role in {"user", "developer"}:
            return _ASSISTANT_SP_TOKEN + thinking_token
        if tail_role == "tool":
            return "\n\n" + thinking_token
        return None


class DeepSeekV4Family(DeepSeekFamily):
    # [CLASS DOC] V4 family using sglang's V4 encoder module.
    template = encoding_dsv4

    def _generation_prompt_suffix(self, tail_role: str | None, thinking_token: str) -> str | None:
        # [FUNCTION DOC] HELPER: After a user, developer or tool message the suffix is the
        # assistant special token followed by the thinking token; otherwise none.
        if tail_role in {"user", "developer", "tool"}:
            return _ASSISTANT_SP_TOKEN + thinking_token
        return None


class DeepSeekV41Family(DeepSeekV4Family):
    # [CLASS DOC] V4.1 family using sglang's V4.1 encoder module. Differs from V4 in three ways:
    # the encode config always carries a reasoning effort (caller value, else environment-derived
    # default), tools are injected using the V4.1 payload that keeps only client-sent fields, and
    # the generation prompt suffix rule is inherited from V4 unchanged.
    """V4.1 mirrors sglang's ``serving_chat`` dsv41 branch: tools keep only the
    fields the client sent, ``reasoning_effort`` always reaches the encoder (the
    server default comes from ``SGLANG_DSV41_REASONING_EFFORT``), and a
    mid-conversation system message also opens an assistant turn."""

    template = encoding_dsv41

    def _build_encode_config(self, kwargs: dict) -> dict:
        # [FUNCTION DOC] HELPER: Starts from the base options, then always sets the reasoning
        # effort: parses the supplied reasoning effort option with sglang's parser; if that gives
        # nothing, uses the default derived from the environment variable (default value "high")
        # via sglang's helper. So a valid caller value wins over the environment.
        kwargs = super()._build_encode_config(kwargs)
        default = chat_encoding.default_dsv41_reasoning_effort_from_env(
            os.environ.get("SGLANG_DSV41_REASONING_EFFORT", "high")
        )
        effort = chat_encoding.parse_dsv41_reasoning_effort(kwargs.get("reasoning_effort"))
        kwargs["reasoning_effort"] = default if effort is None else effort
        return kwargs

    def _inject_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Puts tools into the system message using sglang's V4.1 payload
        # function (keeps only the fields the client sent).
        return _inject_tools_into_system(messages, tools, chat_encoding.dsv41_tool_payload)

    def _generation_prompt_suffix(self, tail_role: str | None, thinking_token: str) -> str | None:
        # [FUNCTION DOC] HELPER: After a user, developer, tool or system message the suffix is the
        # assistant special token followed by the thinking token; otherwise none.
        if tail_role in {"user", "developer", "tool", "system"}:
            return _ASSISTANT_SP_TOKEN + thinking_token
        return None


V32 = DeepSeekV32Family()
V4 = DeepSeekV4Family()
V41 = DeepSeekV41Family()

_FAMILIES = {
    "deepseek_v32": V32,
    "deepseek_v4": V4,
    "deepseek_v41": V41,
}


def model_type(tokenizer: Any) -> str | None:
    # [FUNCTION DOC] Reads the model type of the tokenizer's name-or-path checkpoint directory
    # (cached) and returns it if it is one of the registered families, else None. The tokenizer
    # must have a name-or-path attribute.
    """The DeepSeek family ``model_type`` for *tokenizer*, or ``None`` when the
    checkpoint is not a DeepSeek official-encoder family."""
    mt = _read_model_type(tokenizer.name_or_path)
    return mt if mt in _FAMILIES else None


def apply_chat_template(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    tools: list[dict] | None = None,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
    **kwargs: Any,
) -> str | list[int]:
    # [FUNCTION DOC] Resolves the tokenizer's family (value error naming the path if it is not a
    # DeepSeek official-encoder checkpoint), renders the messages with the family (tools, the
    # generation-prompt flag default true, and extra options forwarded), and returns the text, or
    # when the tokenize flag is set the tokenizer's token ids of that text without adding special
    # tokens (the text already contains them). Arguments in tool calls must be JSON strings.
    """Render *messages* for *tokenizer*'s DeepSeek family, optionally encoding.

    Tool_call ``arguments`` must already be JSON strings (the caller
    normalizes).  Raises ``ValueError`` for a non-DeepSeek tokenizer — guard
    call sites with ``model_type``.
    """
    mt = model_type(tokenizer)
    if mt is None:
        raise ValueError(f"not a DeepSeek official-encoder checkpoint: {tokenizer.name_or_path!r}")
    rendered = _FAMILIES[mt].render_messages(
        messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs
    )
    return tokenizer.encode(rendered, add_special_tokens=False) if tokenize else rendered
