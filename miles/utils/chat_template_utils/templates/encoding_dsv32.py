# [MODULE DOC] Vendored encoder/decoder for the DeepSeek V3.2 chat format (no Jinja template
# exists for it). Encoding: renders a list of OpenAI-style messages (system, developer, user,
# assistant, tool) into a single prompt string, handling thinking mode, tool definitions
# (rendered into the system or developer text), tool calls in the DSML markup and tool results.
# Decoding: parses a completion text back into an assistant message (reasoning, content and tool
# calls). Module-level constants hold the special tokens and the string templates. Miles changes
# relative to upstream: a drop-thinking switch whose false setting keeps every turn's thinking
# block so renders are append-only (needed for token-in-token-out tokenization); see the notes at
# each change. Pure string processing, no I/O. Errors are raised as the encoding error type
# (or a not-implemented error for unknown roles, or a value error for bad tool arguments).
# Vendored from sglang 0.5.14.dev37+gf8cfad3 srt/entrypoints/openai/encoding_dsv32.py so miles can modify the render behavior locally.
# Adapted from https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py
import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union


class DS32EncodingError(Exception):
    # [CLASS DOC] Exception type for malformed messages or malformed completion text; carries
    # only a message.
    pass


TOOLS_SYSTEM_TEMPLATE = """## Tools
You have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a "<{dsml_token}function_calls>" block like the following as part of your reply to the user:
<{dsml_token}function_calls>
<{dsml_token}invoke name="$FUNCTION_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$FUNCTION_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}function_calls>
String and scalar parameters should be specified as is without any escaping or quotes, while lists and objects should use JSON format. The "string" attribute should be set to "true" for string type parameters and "false" for other types (numbers, booleans, arrays, objects).
If the thinking_mode is enabled, then after function results you should strongly consider outputting a thinking block. Here is an example:
<{dsml_token}function_calls>
...
</{dsml_token}function_calls>
<function_results>
...
</function_results>
{thinking_start_token}...thinking about results{thinking_end_token}
Here are the functions available in JSONSchema format:
<functions>
{tool_schemas}
</functions>
"""

bos_token: str = "<｜begin▁of▁sentence｜>"
eos_token: str = "<｜end▁of▁sentence｜>"
thinking_start_token: str = "<think>"
thinking_end_token: str = "</think>"
dsml_token: str = "｜DSML｜"
system_msg_template: str = "{content}"
user_msg_template: str = "<｜User｜>{content}<｜Assistant｜>"
assistant_msg_template: str = "{reasoning}{content}{tool_calls}<｜end▁of▁sentence｜>"
thinking_template = "{reasoning_content}"

response_format_template: str = (
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
)
tool_call_template: str = '<{dsml_token}invoke name="{name}">\n{arguments}\n</{dsml_token}invoke>'
tool_calls_template = "<{dsml_token}function_calls>\n{tool_calls}\n</{dsml_token}function_calls>"

tool_output_template: str = "\n<result>{content}</result>"


def to_json(value: Any) -> str:
    # [FUNCTION DOC] HELPER: Serializes to JSON keeping non-ASCII characters as is; if that fails
    # for any reason, retries with non-ASCII escaped (which may raise the same error again for
    # values that are not serializable at all).
    # KNOWN ISSUE: the fallback uses a bare catch-all, which also swallows interrupts and exit
    # signals.
    try:
        return json.dumps(value, ensure_ascii=False)
    except:
        return json.dumps(value, ensure_ascii=True)


def tools_from_openai_format(tools):
    # [FUNCTION DOC] HELPER: Unwraps OpenAI tool entries to their function dictionaries (keeps
    # order). A missing function key raises a key error.
    return [tool["function"] for tool in tools]


def tool_calls_from_openai_format(tool_calls):
    # [FUNCTION DOC] HELPER: Converts OpenAI tool calls to flat dictionaries holding only the
    # function name and its arguments (call ids and types are dropped). Order is kept; missing
    # keys raise a key error.
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


def tool_calls_to_openai_format(tool_calls):
    # [FUNCTION DOC] HELPER: Inverse of the import conversion: wraps each flat name and arguments
    # dictionary as a function-type OpenAI tool call. No call ids are generated.
    return [
        {
            "type": "function",
            "function": {
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
            },
        }
        for tool_call in tool_calls
    ]


def encode_arguments_to_dsml(tool_call: Dict[str, str]) -> str:
    # [FUNCTION DOC] Renders the tool call's arguments as DSML parameter lines. The arguments may
    # be a JSON string (parsed; invalid JSON raises a JSON decode error) or already a dictionary;
    # anything that is not a JSON object raises a value error. Each key, in insertion order,
    # becomes one parameter element carrying the key name, a flag attribute "true" for string
    # values and "false" otherwise, and the value: strings verbatim (no escaping) and all other
    # values as JSON text. Returns the lines joined by newlines (empty string for no arguments).
    # Keys and values are inserted through string formatting without escaping.
    p_dsml_template = """<{dsml_token}parameter name="{key}" string="{is_str}">{value}</{dsml_token}parameter>"""
    P_dsml_strs = []

    raw_arguments = tool_call["arguments"]
    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    if not isinstance(arguments, dict):
        raise ValueError("Assistant tool call function.arguments must be a JSON object.")

    for k, v in arguments.items():
        p_dsml_str = p_dsml_template.format(
            dsml_token=dsml_token,
            key=k,
            is_str="true" if isinstance(v, str) else "false",
            value=v if isinstance(v, str) else to_json(v),
        )

        P_dsml_strs.append(p_dsml_str)

    return "\n".join(P_dsml_strs)


def decode_dsml_to_arguments(tool_name: str, tool_args: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
    # [FUNCTION DOC] Inverse of the argument encoder. Input: tool name and an ordered mapping from
    # parameter name to a pair of raw value text and a flag string ("true" if it was a string).
    # Builds a JSON object text by hand: each key is JSON-quoted; a string-flagged value is
    # JSON-quoted (escaped) and any other value is spliced in verbatim (assumed to already be
    # JSON), separated by comma and space. Returns a dictionary with the name and the arguments
    # JSON text (the arguments stay a string). Nothing validates that the result is valid JSON.
    def _decode_value(key: str, value: str, string: str):
        # [FUNCTION DOC] HELPER: Formats one key-value pair as JSON-quoted key, colon, and the
        # value (JSON-quoted if the flag says string, verbatim otherwise).
        if string == "true":
            value = to_json(value)
        return f"{to_json(key)}: {value}"

    tool_args_json = (
        "{" + ", ".join([_decode_value(k, v, string=is_str) for k, (v, is_str) in tool_args.items()]) + "}"
    )
    return dict(name=tool_name, arguments=tool_args_json)


def render_tools(tools: List[Dict[str, Union[str, Dict[str, Any]]]]) -> str:
    # [FUNCTION DOC] Renders the tools section: serializes each function dictionary to one JSON
    # line, joins them by newlines, and fills the tools system template together with the DSML
    # and thinking tokens. Returns the text (ends with a newline).
    tools_json = [to_json(t) for t in tools]

    return TOOLS_SYSTEM_TEMPLATE.format(
        tool_schemas="\n".join(tools_json),
        dsml_token=dsml_token,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def find_last_user_index(messages: List[Dict[str, Any]]) -> int:
    # [FUNCTION DOC] Returns the highest index of a message whose role is user or developer, or
    # minus one if there is none.
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in ["user", "developer"]:
            last_user_index = idx
            break
    return last_user_index


def render_message(index: int, messages: List[Dict[str, Any]], thinking_mode: str, drop_thinking: bool = True) -> str:
    # [FUNCTION DOC] Renders the single message at the given index (the whole list is needed for
    # context: last user position and neighbouring assistant). Raises the encoding error for an
    # out-of-range index or a thinking mode other than "chat" or "thinking". Tools (OpenAI form)
    # and tool calls are converted to flat form first. By role:
    # system: content (empty if none), then the tools section and the response-format section,
    # each preceded by a blank line, when present.
    # developer: requires non-empty content (else encoding error); builds a user turn whose text
    # is the optional tools and response-format sections followed by a blank line and the label
    # "The user's message is:" with the content; the user template adds the assistant marker.
    # user: user template around the content (the assistant marker is part of it).
    # For user and developer the turn ends with the thinking start token if thinking mode is
    # "thinking" and either drop_thinking is off or this is the last user/developer message;
    # otherwise with the thinking end token (so chat mode and earlier turns get the end token).
    # tool: finds the nearest preceding non-tool message; it must be an assistant (or the index
    # must be 0), else encoding error. The tool's order among its assistant's calls is its
    # distance from that assistant; the assistant must have at least that many tool calls (else
    # encoding error). The first result is preceded by a blank line and the results-open tag;
    # each result is wrapped in a result element; the last one (order equal to the number of
    # calls) closes the results tag and then adds two newlines plus the thinking start token if
    # thinking mode is on and (drop_thinking is off or the index is at or after the last user
    # index), else the thinking end token.
    # assistant: tool calls rendered into DSML invocations inside a function-calls block preceded
    # by a blank line. The thinking part (reasoning followed by the thinking end token) appears
    # only in thinking mode when drop_thinking is off or the message is after the last user turn;
    # with drop_thinking on, thinking mode, and the message after the last user turn, a message
    # lacking both reasoning and tool calls raises the encoding error. Content, tool calls and
    # the end-of-sentence token follow via the assistant template.
    # Any other role raises a not-implemented error.
    # Returns the rendered text for that message only.
    # KNOWN ISSUE: in the tool branch, walking back from index zero (or through all-tool prefixes)
    # reads the message at position minus one, i.e. the last message in the list, via negative
    # indexing; the index-zero acceptance then relies on that unrelated message's tool calls.
    if not (0 <= index < len(messages)):
        raise DS32EncodingError(f"Index {index} out of range for messages list of length {len(messages)}")
    if thinking_mode not in ["chat", "thinking"]:
        raise DS32EncodingError(f"Invalid thinking_mode `{thinking_mode}`")

    prompt = ""
    msg = messages[index]
    last_user_idx = find_last_user_index(messages)

    role = msg.get("role")
    content = msg.get("content")
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    reasoning_content = msg.get("reasoning_content")

    if tools:
        tools = tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = tool_calls_from_openai_format(tool_calls)

    if role == "system":
        prompt += system_msg_template.format(content=content or "")
        if tools:
            prompt += "\n\n" + render_tools(tools)

        if response_format:
            prompt += "\n\n" + response_format_template.format(schema=to_json(response_format))

    elif role == "developer":
        if not content:
            raise DS32EncodingError(f"Invalid message for role `{role}`: {msg}")
        content_developer = ""
        if tools:
            content_developer += "\n\n" + render_tools(tools)

        if response_format:
            content_developer += "\n\n" + response_format_template.format(schema=to_json(response_format))

        content_developer += "\n\n# The user's message is: {}".format(content)

        prompt += user_msg_template.format(content=content_developer)
        # Miles modification: with drop_thinking=False every turn keeps its
        # thinking block, so the assistant opener is <think> regardless of
        # position (mirrors encoding_dsv4's generation-suffix gate).
        if not drop_thinking and thinking_mode == "thinking":
            prompt += thinking_start_token
        elif drop_thinking and index == last_user_idx and thinking_mode == "thinking":
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token

    elif role == "user":
        prompt += user_msg_template.format(content=content)

        if not drop_thinking and thinking_mode == "thinking":
            prompt += thinking_start_token
        elif drop_thinking and index == last_user_idx and thinking_mode == "thinking":
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token

    elif role == "tool":
        prev_assistant_idx = index - 1
        assistant_msg = messages[prev_assistant_idx]
        while prev_assistant_idx >= 0 and assistant_msg.get("role") == "tool":
            prev_assistant_idx -= 1
            assistant_msg = messages[prev_assistant_idx]

        if not (index == 0 or (prev_assistant_idx >= 0 and assistant_msg.get("role") == "assistant")):
            raise DS32EncodingError(f"Invalid messages at {index}:\n{assistant_msg}")

        tool_call_order = index - prev_assistant_idx
        assistant_tool_calls = assistant_msg.get("tool_calls")
        if not (assistant_tool_calls and len(assistant_tool_calls) >= tool_call_order):
            raise DS32EncodingError("No tool calls but found tool output")

        if tool_call_order == 1:
            prompt += "\n\n<function_results>"

        prompt += tool_output_template.format(content=content)

        if tool_call_order == len(assistant_tool_calls):
            prompt += "\n</function_results>"

            if not drop_thinking and thinking_mode == "thinking":
                prompt += "\n\n" + thinking_start_token
            elif drop_thinking and index >= last_user_idx and thinking_mode == "thinking":
                prompt += "\n\n" + thinking_start_token
            else:
                prompt += "\n\n" + thinking_end_token

    elif role == "assistant":
        prev_assistant_idx = index
        thinking_part = ""

        tool_calls_content = ""
        if tool_calls:
            tool_calls = [
                tool_call_template.format(
                    dsml_token=dsml_token,
                    name=tool_call.get("name"),
                    arguments=encode_arguments_to_dsml(tool_call),
                )
                for tool_call in tool_calls
            ]
            tool_calls_content += "\n\n" + tool_calls_template.format(
                dsml_token=dsml_token, tool_calls="\n".join(tool_calls)
            )

        summary_content = content or ""

        # Miles modification: TiTO disallows drop_thinking=True.
        if drop_thinking and thinking_mode == "thinking" and index > last_user_idx:
            if not (reasoning_content or tool_calls):
                raise DS32EncodingError(
                    f"ThinkingMode: {thinking_mode}, invalid message without reasoning_content/tool_calls `{msg}` after last user message"
                )
        # Miles modification: drop_thinking=False keeps every assistant's
        # thinking block in the render (upstream only renders it after the
        # last user turn), which makes the render append-only across user
        # appends; mirrors encoding_dsv4's assistant gate.
        if thinking_mode == "thinking" and (not drop_thinking or index > last_user_idx):
            thinking_part = thinking_template.format(reasoning_content=reasoning_content or "") + thinking_end_token

        prompt += assistant_msg_template.format(
            reasoning=thinking_part,
            content=summary_content,
            tool_calls=tool_calls_content,
        )
    else:
        raise NotImplementedError(f"Unknown role: {role}")

    return prompt


def drop_thinking_messages(
    messages: List[Dict[str, Any]], last_user_idx: Optional[int] = None
) -> List[Dict[str, Any]]:
    # [FUNCTION DOC] Returns a new list in which assistant messages before the last user index
    # lose their reasoning content (shallow copies; input messages untouched). The index defaults
    # to the last user/developer message. User, system and tool messages, and every message at or
    # after that index, are kept as they are. Order is kept.
    # KNOWN ISSUE: any message before that index whose role is neither user, system, tool nor
    # assistant (notably developer) is silently dropped from the result, which also shifts the
    # indices seen by the renderer; the upstream behavior is presumably the same.
    messages_wo_thinking: List[Dict[str, Any]] = []
    last_user_idx = find_last_user_index(messages) if last_user_idx is None else last_user_idx
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in ["user", "system", "tool"] or idx >= last_user_idx:
            messages_wo_thinking.append(msg)
            continue

        elif role == "assistant":
            msg_wo_thinking = copy.copy(msg)
            msg_wo_thinking.pop("reasoning_content", None)
            messages_wo_thinking.append(msg_wo_thinking)

    return messages_wo_thinking


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
) -> str:
    # [FUNCTION DOC] Main encoder. Inputs: the messages to render, the thinking mode ("chat" or
    # "thinking"), an optional preceding context message list (already rendered earlier, used
    # only for state), a drop-thinking flag (default true) and a flag (default true) for emitting
    # the beginning-of-sentence token. The prompt starts with that token only when the flag is set
    # and there is no context. The full list is context followed by messages; in thinking mode
    # with drop-thinking on, earlier assistants' reasoning is stripped first. Then each of the
    # new messages (not the context) is rendered in order and concatenated. Returns the prompt.
    # Raises as the per-message renderer does. Miles runs it with drop-thinking off so renders are
    # append-only.
    context = context if context else []
    full_messages = context + messages

    prompt = bos_token if add_default_bos_token and len(context) == 0 else ""

    if thinking_mode == "thinking" and drop_thinking:
        full_messages = drop_thinking_messages(full_messages)

    for idx in range(len(messages)):
        prompt += render_message(
            idx + len(context),
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
        )

    return prompt


def _read_until_stop(index: int, text: str, stop: List[str]) -> Tuple[int, str, Optional[str]]:
    # [FUNCTION DOC] HELPER: Scans forward from the index for the earliest occurrence of any stop
    # string (ties go to the one listed first). If found, returns (position just after the stop
    # string, the text between the index and the stop, the stop string). If none is found,
    # returns (text length, the rest of the text, None).
    min_pos = len(text)
    matched_stop = None

    for s in stop:
        pos = text.find(s, index)
        if pos != -1 and pos < min_pos:
            min_pos = pos
            matched_stop = s

    if matched_stop:
        content = text[index:min_pos]
        return min_pos + len(matched_stop), content, matched_stop
    else:
        content = text[index:]
        return len(text), content, None


def parse_tool_calls(index: int, text: str):
    # [FUNCTION DOC] Parses the DSML function-calls block starting right after the opening
    # function-calls tag name (so the text at the index should begin with the tag's closing
    # bracket and newline). Loop: read until the next invoke opener or the block closing tag; the
    # text skipped must be exactly the closing bracket plus newline (else encoding error "Tool
    # call format error"); the closing tag ends the loop; reaching the end of text without either
    # raises "Missing special token". For an invoke: read the name attribute segment up to the
    # first parameter opener or invoke closer; it must match the name attribute pattern followed
    # by a newline (else "Tool name format error"). While parameters follow: read up to the
    # parameter closer, require the name and string-flag attribute pattern with the value (else
    # "Parameter format error"), reject duplicate names, record (value, flag); then read up to the
    # next parameter opener or invoke closer and require the skipped text to be exactly closing
    # bracket plus newline. Each invoke is converted to a flat dictionary with the name and JSON
    # arguments text. Returns (index after parsing, last stop token seen, list of tool calls).
    # Note a loop variable named with a lone underscore is reused for the skipped text.
    tool_calls: List[Dict[str, Any]] = []
    stop_token = None
    tool_calls_end_token = f"</{dsml_token}function_calls>"

    while index < len(text):
        index, _, stop_token = _read_until_stop(index, text, [f"<{dsml_token}invoke", tool_calls_end_token])
        if _ != ">\n":
            raise DS32EncodingError("Tool call format error")

        if stop_token == tool_calls_end_token:
            break

        if stop_token is None:
            raise DS32EncodingError("Missing special token")

        index, tool_name_content, stop_token = _read_until_stop(
            index, text, [f"<{dsml_token}parameter", f"</{dsml_token}invoke"]
        )

        p_tool_name = re.findall(r'^\s*name="(.*?)">\n$', tool_name_content, flags=re.DOTALL)
        if len(p_tool_name) != 1:
            raise DS32EncodingError("Tool name format error")
        tool_name = p_tool_name[0]

        tool_args: Dict[str, Tuple[str, str]] = {}
        while stop_token == f"<{dsml_token}parameter":
            index, param_content, stop_token = _read_until_stop(index, text, [f"/{dsml_token}parameter"])

            param_kv = re.findall(
                r'^ name="(.*?)" string="(true|false)">(.*?)<$',
                param_content,
                flags=re.DOTALL,
            )
            if len(param_kv) != 1:
                raise DS32EncodingError("Parameter format error")
            param_name, string, param_value = param_kv[0]

            if param_name in tool_args:
                raise DS32EncodingError("Duplicate parameter name")
            tool_args[param_name] = (param_value, string)

            index, content, stop_token = _read_until_stop(
                index, text, [f"<{dsml_token}parameter", f"</{dsml_token}invoke"]
            )
            if content != ">\n":
                raise DS32EncodingError("Parameter format error")

        tool_call = decode_dsml_to_arguments(tool_name=tool_name, tool_args=tool_args)
        tool_calls.append(tool_call)

    return index, stop_token, tool_calls


# NOTE: This function is designed to parse only correctly formatted string and will not attempt to correct malformed output that may be generated by the model.
def parse_message_from_completion_text(text: str, thinking_mode: str):
    # [FUNCTION DOC] Parses a model completion into an assistant message. In thinking mode, the
    # text up to the thinking end token is the reasoning (missing token raises encoding error
    # "Invalid thinking format"; a tool-call start before it also fails). Then the summary
    # content runs up to the end-of-sentence token or the tool-calls start marker (a blank line
    # plus the function-calls opener); if neither, raises "Invalid summary format". If the marker
    # was hit, the tool calls are parsed, and anything between them and the end-of-sentence
    # token raises "Unexpected content after tool calls". The whole text must be consumed
    # (else "Unexpected content at end"). Reasoning and content must not contain any of the
    # special tokens (begin, end, thinking start and end, DSML token), else "Unexpected special
    # token in content". Returns an assistant dictionary with content, reasoning content (empty
    # in chat mode) and tool calls in OpenAI format (possibly empty list, without call ids).
    # Only correctly formatted text is handled; no repair is attempted.
    summary_content, reasoning_content, tool_calls = "", "", []
    index, stop_token = 0, None
    tool_calls_start_token = f"\n\n<{dsml_token}function_calls"

    is_thinking, is_tool_calling = thinking_mode == "thinking", False

    if is_thinking:
        index, content_delta, stop_token = _read_until_stop(index, text, [thinking_end_token, tool_calls_start_token])
        reasoning_content = content_delta
        if stop_token != thinking_end_token:
            raise DS32EncodingError("Invalid thinking format")

    index, content_delta, stop_token = _read_until_stop(index, text, [eos_token, tool_calls_start_token])
    summary_content = content_delta
    if stop_token == tool_calls_start_token:
        is_tool_calling = True
    else:
        if stop_token != eos_token:
            raise DS32EncodingError("Invalid summary format")

    if is_tool_calling:
        index, stop_token, tool_calls = parse_tool_calls(index, text)

        index, tool_ends_text, stop_token = _read_until_stop(index, text, [eos_token])
        if tool_ends_text:
            raise DS32EncodingError("Unexpected content after tool calls")

    if not (len(text) == index and stop_token in [eos_token, None]):
        raise DS32EncodingError("Unexpected content at end")

    for sp_token in [
        bos_token,
        eos_token,
        thinking_start_token,
        thinking_end_token,
        dsml_token,
    ]:
        if sp_token in summary_content or sp_token in reasoning_content:
            raise DS32EncodingError("Unexpected special token in content")

    return {
        "role": "assistant",
        "content": summary_content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls_to_openai_format(tool_calls),
    }
