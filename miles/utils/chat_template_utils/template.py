# [MODULE DOC] Chat template loading and rendering in the inference server's style. Offers: a
# loader for the original template of a Hugging Face model (cached download), a normalizer of
# tool-call argument representation, a tool definition canonicalizer, a renderer from a template
# string (no tokenizer), and a tokenizer-based renderer that routes DeepSeek and Inkling
# checkpoints to their own paths. Rendering retries once with bare function dictionaries when the
# template rejects the full tool format. Also re-exports two matcher functions for backward
# compatibility. Network access happens in the loader (first call only).

"""Core chat template operations: load from HuggingFace and render from string.

``load_hf_chat_template`` fetches original (unmodified) chat templates via
``hf_hub_download``.  Files are cached locally after the first download —
subsequent calls read from disk without network access.

``apply_chat_template_from_str`` renders a Jinja2 chat template string
without depending on a HuggingFace tokenizer, equivalent to
``tokenizer.apply_chat_template(..., tokenize=False)``.

``apply_chat_template`` applies via an HF tokenizer object (returns
``str`` or ``list[int]``).  Both functions normalize tool arguments,
canonicalize tool definitions, and fall back between tool dict formats.
"""

from __future__ import annotations

import copy
import json
from typing import Literal

from huggingface_hub import hf_hub_download
from jinja2 import TemplateError
from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from transformers.utils.chat_template_utils import render_jinja_template

from miles.utils.chat_template_utils import deepseek, inkling

# Message matching moved to message_matcher_hub; direct aliases keep the
# established import surface working without a second implementation.
from miles.utils.chat_template_utils.message_matcher_hub import (  # noqa: F401
    assert_messages_append_only_with_allowed_role,
    strict_message_matches,
)


def load_hf_chat_template(model_id: str) -> str:
    # [FUNCTION DOC] Downloads (or reads from the local cache) the tokenizer config of the model
    # and reads its chat template field. If the field is a non-empty string, returns it. If it is
    # a list of named templates, returns the one named default or having no name, else the first.
    # If the field is empty or absent, downloads the separate template file and returns its
    # contents. Network access on a cold cache; errors from the download (missing file etc.)
    # propagate.
    # KNOWN ISSUE: files are opened without an explicit encoding, so decoding depends on the
    # platform default.
    """Load an original chat template from HuggingFace (cached locally).

    Handles two layouts:
    - ``chat_template`` field in ``tokenizer_config.json`` (most models)
    - Separate ``chat_template.jinja`` file (e.g. GLM-5)
    """
    config_path = hf_hub_download(model_id, "tokenizer_config.json")
    with open(config_path) as f:
        config = json.load(f)
    template = config.get("chat_template", "")
    if template:
        if isinstance(template, list):
            for t in template:
                if t.get("name") == "default" or not t.get("name"):
                    return t["template"]
            return template[0]["template"]
        return template

    jinja_path = hf_hub_download(model_id, "chat_template.jinja")
    with open(jinja_path) as f:
        return f.read()


def normalize_tool_arguments(messages: list[dict], format: Literal["dict", "json"]) -> list[dict]:
    # [FUNCTION DOC] Returns a deep copy of the messages. For assistant messages only: a null
    # content becomes an empty string, and for each tool call that has a non-empty function entry,
    # the arguments are converted according to the format: for dict, a string is parsed as JSON
    # (raises a JSON decode error if invalid); for json, a dictionary is serialized without
    # escaping non-ASCII characters. Values already in the target form are left alone. Tool calls
    # must be dictionaries. The input is not modified.
    """Deep-copy *messages*, normalize assistant ``content: None`` -> "", and coerce
    tool_call ``arguments`` to the form the downstream renderer needs (``format`` picks
    the direction; never mutates the input):
    - ``"dict"``: JSON string -> dict, for HF-Jinja templates (they index args as objects).
    - ``"json"``: dict -> JSON string, for the DeepSeek DSML encoders (they ``json.loads`` them).
    """
    normalized = copy.deepcopy(messages)
    for msg in normalized:
        if msg.get("role") == "assistant":
            if msg.get("content") is None:
                msg["content"] = ""
            if isinstance(msg.get("tool_calls"), list):
                for item in msg["tool_calls"]:
                    func = item.get("function")
                    if not func:
                        continue
                    args = func.get("arguments")
                    if format == "dict" and isinstance(args, str):
                        func["arguments"] = json.loads(args)
                    elif format == "json" and isinstance(args, dict):
                        func["arguments"] = json.dumps(args, ensure_ascii=False)
    return normalized


def extract_tool_dicts(tools: list[dict] | None) -> list[dict] | None:
    # [FUNCTION DOC] Returns None for None or an empty list. Each entry that is not already a
    # dictionary with a function key is wrapped as a function-type tool around it. The list is
    # validated against the server's tool model (validation errors propagate) and each tool is
    # dumped back to a dictionary with defaults filled in.
    """Canonicalize tools via Pydantic, returning full Tool model dumps.

    Matches SGLang's ``_process_messages`` (``serving_chat.py`` lines 343-344):
    ``tools = [item.model_dump() for item in request.tools]`` — each tool is
    a full ``Tool`` model dump (``{"type": "function", "function": {...}}``).
    """
    if not tools:
        return None

    wrapped = [t if isinstance(t, dict) and "function" in t else {"type": "function", "function": t} for t in tools]
    validated = TypeAdapter(list[Tool]).validate_python(wrapped)
    return [tool.model_dump() for tool in validated]


def apply_chat_template_from_str(
    chat_template: str,
    messages: list[dict],
    add_generation_prompt: bool = True,
    tools: list[dict] | None = None,
    **kwargs,
) -> str:
    # [FUNCTION DOC] Renders the template string for one conversation and returns the text. Steps:
    # normalize tool arguments to dictionaries; canonicalize tools; render via the Hugging Face
    # Jinja helper with the generation-prompt flag (default true), tools and any extra keyword
    # arguments passed to the template. On a template error with tools present, retries once with
    # each tool replaced by its bare function dictionary; if that also fails raises a value error
    # (fallback wording, chained); with no tools, raises a value error directly. No tokenizer is
    # involved.
    """Render a Jinja2 chat template string (tokenize=False, no tokenizer needed).

    Calls HF transformers' ``render_jinja_template`` directly — the same
    function that ``tokenizer.apply_chat_template`` uses internally.  Both
    SGLang and our ``apply_chat_template`` go through that same HF code path.

    Applies SGLang-style normalizations (tool argument parsing, tool dict
    canonicalization, tool format fallback).
    """

    def _render(tool_defs):
        # [FUNCTION DOC] HELPER: Renders the enclosing messages and template with the given tool
        # definitions and returns the single rendered string (first of the batch of one).
        rendered, _ = render_jinja_template(
            conversations=[messages],
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tools=tool_defs,
            **kwargs,
        )
        return rendered[0]

    messages = normalize_tool_arguments(messages, "dict")
    tool_defs = extract_tool_dicts(tools)
    try:
        return _render(tool_defs)
    except TemplateError as e:
        if tool_defs is not None:
            try:
                return _render([t["function"] if "function" in t else t for t in tool_defs])
            except TemplateError as te:
                raise ValueError(f"Chat template rendering failed (tool format fallback): {te}") from te
        raise ValueError(f"Chat template rendering failed: {e}") from e


def apply_chat_template(
    messages: list[dict],
    *,
    tokenizer,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    tokenize: bool = False,
    **kwargs,
) -> str | list[int]:
    # [FUNCTION DOC] Keyword-only tokenizer and options. Routing: (1) if the tokenizer is a
    # DeepSeek official-encoder checkpoint, normalizes arguments to JSON strings and delegates to
    # the DeepSeek renderer. (2) If it is an Inkling checkpoint, calls the tokenizer's own
    # templating with the bundled fixed Inkling template, dictionary-form arguments and
    # canonicalized tools. (3) Otherwise normalizes arguments to dictionaries, canonicalizes
    # tools, and calls the tokenizer's templating (dict output disabled), returning a string or
    # token id list per the tokenize flag; on a template error with tools present retries once
    # with bare function dictionaries, then raises a value error; without tools raises a value
    # error directly. Extra keyword arguments are forwarded to the templating.
    """Apply chat template via HF tokenizer in SGLang style.

    Passes ``return_dict=False`` to match SGLang's ``serving_chat.py``,
    ensuring the result is ``str`` (tokenize=False) or ``list[int]``
    (tokenize=True), not a ``BatchEncoding`` or ``dict``.
    """
    if deepseek.model_type(tokenizer) is not None:
        return deepseek.apply_chat_template(
            normalize_tool_arguments(messages, "json"),
            tokenizer,
            tools=tools,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    if inkling.is_inkling(tokenizer):
        # the fixed template needs parsed tool-call arguments and handles the
        # thinking-effort line, tool_calls, and the end-sampling token itself
        return tokenizer.apply_chat_template(
            normalize_tool_arguments(messages, "dict"),
            chat_template=inkling.fixed_chat_template(),
            tokenize=tokenize,
            tools=extract_tool_dicts(tools),
            return_dict=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    messages = normalize_tool_arguments(messages, "dict")
    tool_defs = extract_tool_dicts(tools)
    render_kwargs = dict(add_generation_prompt=add_generation_prompt, **kwargs)

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, tools=tool_defs, return_dict=False, **render_kwargs
        )
    except TemplateError as e:
        if tool_defs is not None:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=tokenize,
                    tools=[t["function"] if "function" in t else t for t in tool_defs],
                    return_dict=False,
                    **render_kwargs,
                )
            except TemplateError as te:
                raise ValueError(f"Chat template rendering failed (tool format fallback): {te}") from te
        raise ValueError(f"Chat template rendering failed: {e}") from e
