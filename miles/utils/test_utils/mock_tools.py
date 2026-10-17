# [MODULE DOC] HELPER: Mock tool-calling agent support for tests. Provides two sample tool
# definitions (get current year, get temperature for a location) with deterministic executors, an
# agent loop that talks to a chat completions endpoint and executes tool calls until the model stops
# calling tools, tiny agent functions for other test paths, and two scripted conversation stubs
# (two-turn and three-turn) that pair exact rendered prompts with canned model replies, together
# with the token ids and message lists the tests expect on the wire. Importing this module loads
# a real tokenizer (Qwen3-0.6B, remote code trusted) at import time to precompute prompt token
# ids. Two module-level knobs (maximum turns override and canned agent return metadata) are
# globals that tests may set.
import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.mock_sglang_server import ProcessResult

AGENTIC_MAX_TURNS: int | None = None
from miles.utils.http_utils import post

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_year",
            "description": "Get current year",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Get temperature for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
]


def _get_year(params: dict) -> str:
    # [FUNCTION DOC] HELPER: Tool executor returning a JSON string with the fixed year 2026.
    # Asserts the parameters mapping is empty.
    assert len(params) == 0
    return json.dumps({"year": 2026})


def _get_temperature(params: dict) -> str:
    # [FUNCTION DOC] HELPER: Tool executor returning a JSON string with a temperature for the
    # location in the parameters: minus 60 for "Mars", 15 for "Earth". Asserts the location is one
    # of those two (assertion error naming it otherwise, including when missing).
    temps = {"Mars": -60, "Earth": 15}
    location = params.get("location")
    assert location in temps, f"Unknown location: {location}"
    return json.dumps({"temperature": temps[location]})


TOOL_EXECUTORS = {
    "get_year": _get_year,
    "get_temperature": _get_temperature,
}


async def execute_tool_call(name: str, params: dict) -> str:
    # [FUNCTION DOC] HELPER: Runs the named tool's executor on the parameters and returns its JSON
    # string result. Raises a key error for an unknown tool name, and assertion errors from the
    # executors. Declared async but runs synchronously.
    return TOOL_EXECUTORS[name](params)


AGENTIC_RETURN_METADATA: dict[str, Any] | Callable | None = None


async def run_agentic_tool_call(
    base_url: str,
    prompt: list[dict[str, Any]] | str,
    request_kwargs: dict[str, Any] | None = None,
    max_turns: int = 8,
    **kwargs,
) -> dict[str, Any] | None:
    # [FUNCTION DOC] HELPER: Agent loop against a chat completions server. Inputs: the server base
    # URL; a prompt that is either a message list (deep-copied) or a string (wrapped as a single
    # user message); optional request keyword mapping (model, default "default"; tools, default the
    # two sample tools); a maximum number of turns (default 8, overridden by the module-level
    # override when set); extra keywords (only the "metadata" one is used at the end). Each turn
    # posts the messages and tools to the base URL plus the chat completions path using the shared
    # HTTP post helper, takes the first choice's message, and stops if it has no tool calls.
    # Otherwise appends an assistant message (content, refusal, role defaulting to assistant,
    # annotations, audio, function call and the tool calls) and, for each tool call, parses the
    # arguments (JSON string, or a mapping passed through; an empty or missing value means an empty
    # object; malformed JSON becomes an empty mapping), executes the tool and appends a tool
    # message with the call id, result and tool name. After the loop (including when the turn limit
    # is exhausted without a final answer) returns the module-level canned return metadata: if it
    # is callable, the result of calling it with the metadata keyword (taken from the extra
    # keywords, possibly none); otherwise the value itself (default none).
    if AGENTIC_MAX_TURNS is not None:
        max_turns = AGENTIC_MAX_TURNS
    messages = deepcopy(prompt) if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    request_kwargs = request_kwargs or {}
    model = request_kwargs.get("model", "default")
    tools = request_kwargs.get("tools", SAMPLE_TOOLS)

    for _ in range(max_turns):
        payload = {"model": model, "messages": messages, "tools": tools}
        response = await post(base_url + "/v1/chat/completions", payload)
        choice = response["choices"][0]["message"]
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            break

        assistant_msg = {
            "content": choice.get("content"),
            "refusal": choice.get("refusal"),
            "role": choice.get("role", "assistant"),
            "annotations": choice.get("annotations"),
            "audio": choice.get("audio"),
            "function_call": choice.get("function_call"),
            "tool_calls": tool_calls,
        }
        messages.append(assistant_msg)

        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            raw_args = tool_call["function"].get("arguments") or "{}"
            try:
                params = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                params = {}
            result = await execute_tool_call(name, params)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": result,
                    "name": name,
                }
            )

    if callable(AGENTIC_RETURN_METADATA):
        return AGENTIC_RETURN_METADATA(metadata=kwargs.get("metadata"))
    return AGENTIC_RETURN_METADATA


async def run_agentic_tool_call_returning_reward(**kwargs) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Runs the agent loop with the given keywords (discarding its return
    # value) and returns a mapping with an agent reward of 42.0, independent of module state so it
    # works in a fresh process where the globals are unset.
    """``run_agentic_tool_call`` returning a fixed reward without module state, which a fresh process lacks."""
    await run_agentic_tool_call(**kwargs)
    return {"agent_reward": 42.0}


async def run_agentic_noop(**kwargs) -> None:
    # [FUNCTION DOC] HELPER: Ignores all keywords, makes no requests and returns nothing.
    """Agent function that makes no model calls — for testing the no-records path."""
    return None


_SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "# Tools\n"
    "\n"
    "You may call one or more functions to assist with the user query.\n"
    "\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "get_year", "description": "Get current year", "parameters": {"type": "object", "properties": {}, "required": []}}}\n'
    '{"type": "function", "function": {"name": "get_temperature", "description": "Get temperature for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}\n'
    "</tools>\n"
    "\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call><|im_end|>\n"
)


_TOKENIZER = load_tokenizer("Qwen/Qwen3-0.6B", trust_remote_code=True)


class TwoTurnStub:
    # [CLASS DOC] HELPER: Scripted two-request conversation. Constants: the user question; the first
    # model reply (a sentence plus two tool calls, to get the year and the temperature on Mars, in
    # the Qwen tool-call tag format, ending with the end-of-turn token); the rendered tool response
    # block that follows; the final answer text; the exact rendered first prompt (shared system
    # prompt with tool descriptions, the user turn, the assistant start) and second prompt (first
    # prompt plus first reply plus tool responses); the prompt message list; precomputed token ids
    # of both prompts (without special tokens added); the tool calls in OpenAI shape with ids
    # call00000 and call00001 and in the inference server's wire shape (adding the tool's index in
    # the tools list: 0 and 1); the expected message lists the OpenAI client sends for the first
    # turn and, replaying the assistant message and tool results, for the second turn. The static
    # process function returns the first reply for the first prompt and the final answer for the
    # second prompt, with finish reason "stop".
    """Stub for 2-turn: get_year + get_temperature(Mars) -> final answer"""

    USER_QUESTION = "What is 42 + year + temperature?"

    FIRST_RESPONSE = (
        "Let me get the year and temperature first.\n"
        "<tool_call>\n"
        '{"name": "get_year", "arguments": {}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "get_temperature", "arguments": {"location": "Mars"}}\n'
        "</tool_call><|im_end|>\n"
    )

    FIRST_TOOL_RESPONSE = (
        "<|im_start|>user\n"
        "<tool_response>\n"
        '{"year": 2026}\n'
        "</tool_response>\n"
        "<tool_response>\n"
        '{"temperature": -60}\n'
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    SECOND_RESPONSE = "The answer is: 42 + 2026 + -60 = 2008."

    FIRST_PROMPT = _SYSTEM_PROMPT + "<|im_start|>user\n" + USER_QUESTION + "<|im_end|>\n" + "<|im_start|>assistant\n"
    SECOND_PROMPT = FIRST_PROMPT + FIRST_RESPONSE + FIRST_TOOL_RESPONSE

    PROMPT = [{"role": "user", "content": USER_QUESTION}]

    FIRST_PROMPT_TOKEN_IDS = _TOKENIZER(FIRST_PROMPT, add_special_tokens=False)["input_ids"]
    SECOND_PROMPT_TOKEN_IDS = _TOKENIZER(SECOND_PROMPT, add_special_tokens=False)["input_ids"]

    FIRST_RESPONSE_CONTENT = "Let me get the year and temperature first."
    FIRST_TOOL_CALLS_OPENAI_FORMAT = [
        {"id": "call00000", "function": {"arguments": "{}", "name": "get_year"}, "type": "function"},
        {
            "id": "call00001",
            "function": {"arguments": '{"location": "Mars"}', "name": "get_temperature"},
            "type": "function",
        },
    ]
    # What the mock backend serves on the wire: SGLang's non-streaming ToolCall
    # additionally carries ``index`` (the tool's position in the tools list).
    # Clients normalized to the OpenAI shape (above) legitimately drop it.
    FIRST_TOOL_CALLS_SGLANG_WIRE = [
        {**tool_call, "index": index} for index, tool_call in zip((0, 1), FIRST_TOOL_CALLS_OPENAI_FORMAT, strict=True)
    ]

    OPENAI_MESSAGES_FIRST_TURN = [{"role": "user", "content": USER_QUESTION}]

    # The openai SDK preserves unknown keys (extra="allow"), so a client replay
    # echoes the wire shape — index included.
    OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT = OPENAI_MESSAGES_FIRST_TURN + [
        {
            "content": FIRST_RESPONSE_CONTENT,
            "refusal": None,
            "role": "assistant",
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": FIRST_TOOL_CALLS_SGLANG_WIRE,
        },
        {"role": "tool", "tool_call_id": "call00000", "content": '{"year": 2026}', "name": "get_year"},
        {"role": "tool", "tool_call_id": "call00001", "content": '{"temperature": -60}', "name": "get_temperature"},
    ]

    @staticmethod
    def process_fn(prompt: str) -> ProcessResult:
        # [FUNCTION DOC] HELPER: Process function for the mock server: returns the canned reply
        # (finish reason "stop") whose expected rendered prompt equals the given prompt exactly;
        # raises a value error showing the prompt if it matches neither.
        prompt_response_pairs = {
            TwoTurnStub.FIRST_PROMPT: TwoTurnStub.FIRST_RESPONSE,
            TwoTurnStub.SECOND_PROMPT: TwoTurnStub.SECOND_RESPONSE,
        }

        for expect_prompt, response in prompt_response_pairs.items():
            if prompt == expect_prompt:
                return ProcessResult(text=response, finish_reason="stop")

        raise ValueError(f"Unexpected {prompt=}")


class ThreeTurnStub:
    # [CLASS DOC] HELPER: Scripted three-request conversation (year plus Mars temperature, then
    # Earth temperature, then the final answer). Same kinds of constants as the two-turn stub, with
    # an additional second reply (a sentence plus one tool call for Earth's temperature), second
    # tool response block, third prompt (second prompt plus second reply plus second tool
    # response), third prompt token ids, second-turn tool calls (id call00000, wire index 1 being
    # the position of the temperature tool in the tools list), and the expected client message
    # lists for the first, second and third turns. The static process function maps each of the
    # three exact prompts to the matching reply (finish reason "stop").
    """Stub for 3-turn: get_year + get_temperature(Mars) -> get_temperature(Earth) -> final answer"""

    USER_QUESTION = "What is 42 + year + Mars temperature + Earth temperature?"

    FIRST_RESPONSE = (
        "Let me get the year and Mars temperature first.\n"
        "<tool_call>\n"
        '{"name": "get_year", "arguments": {}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "get_temperature", "arguments": {"location": "Mars"}}\n'
        "</tool_call><|im_end|>\n"
    )

    SECOND_RESPONSE = (
        "Now let me get Earth temperature.\n"
        "<tool_call>\n"
        '{"name": "get_temperature", "arguments": {"location": "Earth"}}\n'
        "</tool_call><|im_end|>\n"
    )

    FIRST_TOOL_RESPONSE = (
        "<|im_start|>user\n"
        "<tool_response>\n"
        '{"year": 2026}\n'
        "</tool_response>\n"
        "<tool_response>\n"
        '{"temperature": -60}\n'
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    SECOND_TOOL_RESPONSE = (
        "<|im_start|>user\n"
        "<tool_response>\n"
        '{"temperature": 15}\n'
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    THIRD_RESPONSE = "The answer is: 42 + 2026 + -60 + 15 = 2023."

    FIRST_PROMPT = _SYSTEM_PROMPT + "<|im_start|>user\n" + USER_QUESTION + "<|im_end|>\n" + "<|im_start|>assistant\n"
    SECOND_PROMPT = FIRST_PROMPT + FIRST_RESPONSE + FIRST_TOOL_RESPONSE
    THIRD_PROMPT = SECOND_PROMPT + SECOND_RESPONSE + SECOND_TOOL_RESPONSE

    PROMPT = [{"role": "user", "content": USER_QUESTION}]

    FIRST_PROMPT_TOKEN_IDS = _TOKENIZER(FIRST_PROMPT, add_special_tokens=False)["input_ids"]
    SECOND_PROMPT_TOKEN_IDS = _TOKENIZER(SECOND_PROMPT, add_special_tokens=False)["input_ids"]
    THIRD_PROMPT_TOKEN_IDS = _TOKENIZER(THIRD_PROMPT, add_special_tokens=False)["input_ids"]

    FIRST_RESPONSE_CONTENT = "Let me get the year and Mars temperature first."
    FIRST_TOOL_CALLS_OPENAI_FORMAT = [
        {"id": "call00000", "function": {"arguments": "{}", "name": "get_year"}, "type": "function"},
        {
            "id": "call00001",
            "function": {"arguments": '{"location": "Mars"}', "name": "get_temperature"},
            "type": "function",
        },
    ]

    # SGLang wire shape: index is the tool's position in the tools list
    # (get_year=0, get_temperature=1), not the call position.
    FIRST_TOOL_CALLS_SGLANG_WIRE = [
        {**tool_call, "index": index} for index, tool_call in zip((0, 1), FIRST_TOOL_CALLS_OPENAI_FORMAT, strict=True)
    ]

    SECOND_RESPONSE_CONTENT = "Now let me get Earth temperature."
    SECOND_TOOL_CALLS_OPENAI_FORMAT = [
        {
            "id": "call00000",
            "function": {"arguments": '{"location": "Earth"}', "name": "get_temperature"},
            "type": "function",
        },
    ]
    SECOND_TOOL_CALLS_SGLANG_WIRE = [{**SECOND_TOOL_CALLS_OPENAI_FORMAT[0], "index": 1}]

    OPENAI_MESSAGES_FIRST_TURN = [{"role": "user", "content": USER_QUESTION}]

    # The openai SDK preserves unknown keys (extra="allow"), so a client replay
    # echoes the wire shape — index included.
    OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT = OPENAI_MESSAGES_FIRST_TURN + [
        {
            "content": FIRST_RESPONSE_CONTENT,
            "refusal": None,
            "role": "assistant",
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": FIRST_TOOL_CALLS_SGLANG_WIRE,
        },
        {"role": "tool", "tool_call_id": "call00000", "content": '{"year": 2026}', "name": "get_year"},
        {"role": "tool", "tool_call_id": "call00001", "content": '{"temperature": -60}', "name": "get_temperature"},
    ]

    OPENAI_MESSAGES_THIRD_TURN_FROM_CLIENT = OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT + [
        {
            "content": SECOND_RESPONSE_CONTENT,
            "refusal": None,
            "role": "assistant",
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": SECOND_TOOL_CALLS_SGLANG_WIRE,
        },
        {"role": "tool", "tool_call_id": "call00000", "content": '{"temperature": 15}', "name": "get_temperature"},
    ]

    @staticmethod
    def process_fn(prompt: str) -> ProcessResult:
        # [FUNCTION DOC] HELPER: Process function for the mock server: returns the canned reply
        # (finish reason "stop") for the first, second or third exact prompt; raises a value error
        # showing the prompt otherwise.
        prompt_response_pairs = {
            ThreeTurnStub.FIRST_PROMPT: ThreeTurnStub.FIRST_RESPONSE,
            ThreeTurnStub.SECOND_PROMPT: ThreeTurnStub.SECOND_RESPONSE,
            ThreeTurnStub.THIRD_PROMPT: ThreeTurnStub.THIRD_RESPONSE,
        }

        for expect_prompt, response in prompt_response_pairs.items():
            if prompt == expect_prompt:
                return ProcessResult(text=response, finish_reason="stop")

        raise ValueError(f"Unexpected {prompt=}")
