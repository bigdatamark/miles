# [MODULE DOC] Adapter that converts the output of the local SGLang tool-call parser into
# OpenAI-style assistant messages and tool definitions, and into tau-bench Action objects. Two
# small dataclasses model the OpenAI tool call and assistant message; the adapter class holds the
# tool descriptions and the parser name; a factory function builds an adapter.
import json
import logging
from dataclasses import dataclass
from typing import Any

from sglang_tool_parser import parse_tools
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME
from tau_bench.types import Action

# Set up logger for this module
logger = logging.getLogger(__name__)


@dataclass
class OpenAIToolCall:
    # [CLASS DOC] Data holder for one tool call: id, type (default "function") and a function
    # mapping with the tool name and its arguments text (default none).
    """OpenAI format tool call structure"""

    id: str
    type: str = "function"
    function: dict[str, Any] = None


@dataclass
class OpenAIAssistantMessage:
    # [CLASS DOC] Data holder for an assistant message: role (default assistant), optional text
    # content and optional list of tool calls (both default none).
    """OpenAI format assistant message structure"""

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAICompatibleToolCallAdapter:
    # [CLASS DOC] Holds the list of tool descriptions (each with a type and a function entry having
    # name, description and parameters) and the SGLang parser name (default qwen25) and offers
    # conversions of model output text into OpenAI-style messages and tau-bench actions.
    """
    Adapter class that converts sglang tool call parsing results to OpenAI compatible format

    This class encapsulates existing tool call parsing and action conversion logic,
    and provides OpenAI format output interface.
    """

    def __init__(self, tools_info: list[dict[str, Any]], parser_type: str = "qwen25"):
        # [FUNCTION DOC] Stores the tool descriptions and parser name; no other work.
        """
        Initialize adapter

        Args:
            tools_info: List of tool information
            parser_type: Parser type, defaults to "qwen25"
        """
        self.tools_info = tools_info
        self.parser_type = parser_type

    def parse_response_to_openai_format(self, response: str) -> dict[str, Any]:
        # [FUNCTION DOC] Parses raw model text with the SGLang tool parser and returns a mapping.
        # Success: "openai_message" (assistant message), "parsed_result" (the parser's mapping with
        # normal text and call dictionaries) and "success" true. Any exception is caught, logged as
        # a warning, and the result is "openai_message" none, "parsed_result" none, "success" false
        # and "error" with the message text. Never raises.
        # KNOWN ISSUE: the docstring says exceptions are thrown on parse failure, but they are
        # swallowed and reported in the result.
        """
        Parse sglang response to OpenAI compatible format

        Args:
            response: Raw response text from sglang

        Returns:
            Dictionary containing OpenAI format message and parsing results

        Raises:
            Exception: Thrown when parsing fails
        """
        try:
            # Use existing parser to parse tool calls
            parsed = parse_tools(response, self.tools_info, self.parser_type)

            # Extract parsing results
            normal_text = parsed["normal_text"]
            calls = parsed["calls"]

            # Convert to OpenAI format
            openai_message = self._convert_to_openai_message(normal_text, calls)

            return {"openai_message": openai_message, "parsed_result": parsed, "success": True}

        except Exception as e:
            logger.warning(f"Parsing failed with error: {str(e)}")
            return {"openai_message": None, "parsed_result": None, "success": False, "error": str(e)}

    def _convert_to_openai_message(self, normal_text: str, calls: list[dict[str, Any]]) -> OpenAIAssistantMessage:
        # [FUNCTION DOC] HELPER: Builds the assistant message. No calls: content is the normal text
        # and no tool calls. Otherwise one tool call per parsed call, in order, with id
        # "call_<index>_<name>" (name "unknown" if missing), type function, and a function mapping
        # of the name (empty if missing) and the arguments text from the call's parameters entry
        # (the text "{}" if missing); the message content is the normal text, or none if that is
        # blank.
        """
        Convert parsing results to OpenAI format assistant message

        Args:
            normal_text: Normal text content
            calls: List of tool calls

        Returns:
            OpenAI format assistant message
        """
        if not calls:
            # No tool calls, return plain text response
            return OpenAIAssistantMessage(role="assistant", content=normal_text, tool_calls=None)

        # Convert tool calls to OpenAI format
        openai_tool_calls = []
        for i, call in enumerate(calls):
            openai_tool_call = OpenAIToolCall(
                id=f"call_{i}_{call.get('name', 'unknown')}",
                type="function",
                function={"name": call.get("name", ""), "arguments": call.get("parameters", "{}")},
            )
            openai_tool_calls.append(openai_tool_call)

        result = OpenAIAssistantMessage(
            role="assistant", content=normal_text if normal_text.strip() else None, tool_calls=openai_tool_calls
        )
        return result

    def _call_to_action_sglang(self, calls: list[Any], text_response: str) -> Action:
        # [FUNCTION DOC] HELPER: Converts parsed calls to one tau-bench action. Default action is the
        # respond action with the text response as content. If there are calls, only the first is
        # used (a debug log notes extras); its parameters text is decoded as JSON; if the result is a
        # mapping the action becomes that tool name with the mapping as keyword arguments; a
        # non-mapping logs a warning and a decoding failure logs a warning, both keeping the default
        # respond action. A missing name or parameters key raises a key error.
        # KNOWN ISSUE: not called anywhere in this module.
        """
        Convert sglang tool calls to Action object

        This method replicates the original call_to_action_sglang logic,
        ensuring compatibility with existing code.
        """
        # Default action if no action found
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": text_response})

        if calls:
            if len(calls) > 1:
                logger.debug("Multiple tool calls identified, only taking first.")

            tool_call = calls[0]

            try:
                params = json.loads(tool_call["parameters"])

                if not isinstance(params, dict):
                    logger.warning(f"{params} does not follow dict structure for action")
                else:
                    action = Action(name=tool_call["name"], kwargs=params)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse parameters as JSON: {e}")

        return action

    def get_openai_tools_format(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC] Returns the tool list in OpenAI format: for each stored tool a mapping of
        # type "function" and a function mapping with only the name, description and parameters
        # copied from the tool's function entry (missing keys raise a key error; other keys dropped).
        """
        Get OpenAI format tool definitions

        Returns:
            List of OpenAI format tools
        """
        openai_tools = []
        for tool in self.tools_info:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "parameters": tool["function"]["parameters"],
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools


# Usage examples and factory functions
def create_openai_adapter(
    tools_info: list[dict[str, Any]], parser_type: str = "qwen25"
) -> OpenAICompatibleToolCallAdapter:
    # [FUNCTION DOC] Factory: returns a new adapter for the given tool descriptions and parser name.
    """
    Factory function to create OpenAI compatible tool call adapter

    Args:
        tools_info: List of tool information
        parser_type: Parser type

    Returns:
        Configured adapter instance
    """
    return OpenAICompatibleToolCallAdapter(tools_info, parser_type)
