# [MODULE DOC] HELPER module: a local (in-process) version of the SGLang server's tool-call
# parsing, exposing one function that splits a model response into plain text and structured tool
# calls using SGLang's own parser classes.
from typing import Any

from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Function, Tool


def parse_tools(response: str, tools: list[dict[str, Any]], parser: str = "qwen25"):
    # [FUNCTION DOC] Parses model text into plain text and tool calls. Each input tool description
    # (needs a type entry and a function entry with name, description and parameters) is converted
    # into SGLang's tool structure; then a parser of the named kind (default qwen25) parses the
    # complete (non-streaming) response. Returns a mapping with "normal_text" (the text outside
    # tool calls) and "calls" (list of dictionaries dumped from the parser's call objects, each with
    # at least a name and a JSON text of parameters). Missing keys raise key errors; an unknown
    # parser name raises whatever the SGLang class raises.
    """
    This function mimics the function call parser API from
    https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py#L952
    But running locally
    """
    tools_list = [
        Tool(
            function=Function(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
            ),
            type=tool["type"],
        )
        for tool in tools
    ]
    parser = FunctionCallParser(tools=tools_list, tool_call_parser=parser)

    normal_text, calls = parser.parse_non_stream(response)

    return {
        "normal_text": normal_text,
        "calls": [call.model_dump() for call in calls],  # Convert pydantic objects to dictionaries
    }
