# [MODULE DOC] HELPER:
# Utilities for agentic rollouts that call tools. Covers: building a parser for model-emitted tool
# calls, executing parsed tool calls through a caller-supplied async executor and producing
# tool-role chat messages, tokenizing those tool messages into the token ids that follow an
# assistant turn (by diffing chat-template output with and without them), and appending the
# resulting observation tokens to a sample with the loss mask, log-probs and optional sampling
# mask / top-k log-prob bookkeeping kept aligned. Module constant: a dummy user message used when
# building template prefixes.
"""
Utils to handle tool calls.
"""

import json
import uuid
from collections.abc import Callable
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from miles.rollout.generate_utils.rollout_topk_logprobs import pad_rollout_topk_logprobs
from miles.rollout.generate_utils.sampling_mask import append_forced_sampling_tokens
from miles.utils.types import Sample

_DUMMY_USER = {"role": "user", "content": "dummy"}


def create_tool_call_parser(tool_specs, tool_call_parser):
    # [FUNCTION DOC]
    # Build an SGLang function-call parser. The tool specs (a list of OpenAI-style tool definition
    # mappings) are validated into SGLang tool models (validation errors propagate); the parser
    # format name selects which model-specific tool-call syntax is parsed. Returns the parser.
    return FunctionCallParser(
        tools=TypeAdapter(list[Tool]).validate_python(tool_specs),
        tool_call_parser=tool_call_parser,
    )


async def execute_tool_calls(
    tool_calls: list[ToolCallItem | ChatCompletionMessageToolCall],
    execute_one: Callable,
) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # Execute the given tool calls strictly sequentially (in order, awaiting each) with the async
    # executor callable and return the list of tool messages in the same order. Calls may be either
    # SGLang parsed items or OpenAI message tool calls; errors from the single-call helper
    # (unsupported type, bad JSON, non-string result) propagate and abort the remaining calls.
    tool_messages = []
    for call in tool_calls:
        tool_messages.append(await _execute_tool_call(call, execute_one))
    return tool_messages


async def _execute_tool_call(
    call: ToolCallItem | ChatCompletionMessageToolCall, execute_one: Callable
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Execute one tool call and build its tool-role message.
    # For an OpenAI message tool call: name and id come from the call, arguments are a JSON string
    # parsed into a mapping (empty mapping when the string is empty). For an SGLang parsed item:
    # name from the item, parameters JSON string parsed likewise, and a fresh id "call_" followed
    # by 24 random hex characters is generated. Any other type raises a type error.
    # The executor is awaited with (name, params) and must return a string (asserted).
    # Returns a mapping with role "tool", the tool call id, the content string and the name.
    if isinstance(call, ChatCompletionMessageToolCall):
        name = call.function.name
        params = json.loads(call.function.arguments) if call.function.arguments else {}
        tool_call_id = call.id
    elif isinstance(call, ToolCallItem):
        name = call.name
        params = json.loads(call.parameters) if call.parameters else {}
        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
    else:
        raise TypeError(f"Unsupported tool call type: {type(call)}")

    result = await execute_one(name, params)
    assert isinstance(result, str)

    return {"role": "tool", "tool_call_id": tool_call_id, "content": result, "name": name}


def update_sample_with_tool_responses(sample: Sample, tool_messages: list[dict[str, Any]], tokenizer):
    # [FUNCTION DOC]
    # Append tool outputs to a sample as observation tokens, mutating it in place. Steps: tokenize
    # the tool messages; if the sample carries a sampling mask, append singleton supports for the
    # new tokens; pad the rollout top-k log-prob record for the new tokens; then extend the
    # response text with the decoded tokens, increase the response length, extend the token list,
    # extend the loss mask with zeros (observations are not trained on) and extend the rollout
    # log-probs with 0.0 for each new token. Returns nothing.
    next_obs_tokens_ids: list[int] = tokenize_tool_responses(tool_messages, tokenizer=tokenizer)
    if sample.rollout_sampling_mask is not None:
        append_forced_sampling_tokens(sample, next_obs_tokens_ids)
    pad_rollout_topk_logprobs(sample, len(next_obs_tokens_ids))
    sample.response += tokenizer.decode(next_obs_tokens_ids)
    sample.response_length += len(next_obs_tokens_ids)
    sample.tokens += next_obs_tokens_ids
    sample.loss_mask += [0] * len(next_obs_tokens_ids)
    sample.rollout_log_probs += [0.0] * len(next_obs_tokens_ids)


# TODO: very naive implementation, need the to-be-implemented e2e test to validate.
def tokenize_tool_responses(
    tool_messages: list[dict[str, Any]],
    tokenizer,
) -> list[int]:
    # [FUNCTION DOC]
    # Return the token ids that the chat template emits for the given tool messages when they
    # follow an assistant turn, including the trailing generation prompt for the next assistant
    # turn. Thin wrapper over the postfix tokenizer; its assertion failure (template prefix
    # mismatch) propagates.
    return _tokenize_postfix_messages(tool_messages, tokenizer)


def _tokenize_postfix_messages(
    postfix_messages: list[dict[str, Any]],
    tokenizer,
) -> list[int]:
    # [FUNCTION DOC] HELPER: Tokenize messages that follow an assistant turn by differencing.
    # Builds a conversation of a dummy user message plus a dummy assistant message that makes one
    # empty-argument tool call per postfix message. Applies the chat template twice: to the base
    # (user, assistant) pair without a generation prompt, and to base plus postfix messages with a
    # generation prompt. Asserts the first token list is an exact prefix of the second (fails for
    # thinking models or templates that rewrite earlier turns, since the token list must be
    # append-only) and returns the remaining suffix tokens.
    dummy_assistant = _build_dummy_assistant(postfix_messages)
    base_messages = [_DUMMY_USER, dummy_assistant]

    messages_without = base_messages
    messages_with = base_messages + postfix_messages

    tokens_with = tokenizer.apply_chat_template(
        messages_with, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    tokens_without = tokenizer.apply_chat_template(
        messages_without, tokenize=True, add_generation_prompt=False, return_dict=False
    )

    assert tokens_with[: len(tokens_without)] == tokens_without, (
        f"Fail to tokenize_tool_responses caused by token prefix mismatch. "
        f"This can happen for thinking model or models with special chat template, "
        f"and this simple example does not support it yet, "
        f"since this means we cannot have a append-only token id list. "
        f"{tokens_with=} {tokens_without=} "
        f"{tokenizer.decode(tokens_with)=} {tokenizer.decode(tokens_without)=} "
    )
    return tokens_with[len(tokens_without) :]


def _build_dummy_assistant(tool_responses: list[dict[str, Any]]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Build an assistant message with empty content, a single-space
    # reasoning content, and one function tool call per tool response. Each call reuses the
    # response's tool call id (default "call0000" plus index) and name (default "dummy_func"),
    # with an empty mapping of arguments. Pure function.
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": " ",
        "tool_calls": [
            {
                "id": resp.get("tool_call_id", f"call0000{i}"),
                "type": "function",
                "function": {
                    "name": resp.get("name", "dummy_func"),
                    "arguments": {},
                },
            }
            for i, resp in enumerate(tool_responses)
        ],
    }
