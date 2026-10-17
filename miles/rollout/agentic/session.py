# [MODULE DOC] URL helper for agentic rollouts: derives the OpenAI-compatible API root of a policy
# session from the session's base URL, to be handed to the agent.
"""The session-scoped policy URL an agent function hands to its agent."""


def openai_session_url(base_url: str) -> str:
    # [FUNCTION DOC] Append the version suffix to the session base URL and return the text. Pure string
    # concatenation: no validation, and a base URL ending in a slash would produce a double slash.
    """The OpenAI-compatible API root of a session: its base URL plus the ``/v1`` suffix."""
    return f"{base_url}/v1"
