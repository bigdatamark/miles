# [MODULE DOC]
# Defines the agent used for HUD computer-use episodes: a thin subclass of HUD's OpenAI chat
# completions agent restricted to a single tool, the screen-driving computer tool from the sibling
# module, plus a default system prompt template (with a placeholder for the turn budget) that tells
# the model turns are expensive and keystrokes can be batched. A class factory bakes the screenshot
# width into the tool class used by the agent. The system prompt template is exported for the rollout
# module, which formats and passes it into the agent configuration.
"""The agent Miles drives HUD environments with: HUD's own OpenAI-compatible
harness, plus the one thing it lacks for computer use -- a screen
(:mod:`.computer_tool` explains that gap). The loop, tool plumbing, trace
recording and token-level ``Sample`` collection are all inherited unchanged.
"""

from __future__ import annotations

from examples.experimental.hud.computer_tool import ChatComputerTool, computer_tool_class
from hud.agents.openai_compatible.agent import OpenAIChatAgent


SYSTEM_PROMPT = """You are operating a computer through screenshots.

You have at most {max_steps} turns, and a turn is the expensive thing: each one \
re-reads the whole screen history. Actions are not -- a single computer call \
can carry several keystrokes, played in order. So plan a few steps ahead and \
send them together rather than one per turn.

Keys go to whatever the screen has focused, so click what you mean to type \
into before typing. Look at each screenshot to check that your last action did \
what you expected, and change approach if it did not."""


class ComputerChatAgent(OpenAIChatAgent):
    # [CLASS DOC]
    # Chat-completions agent whose tool catalog contains only the computer tool class. Everything
    # else (loop, tool plumbing, traces, token sample collection) is inherited.
    # KNOWN ISSUE: the class documentation below says the class adds the system prompt, but the
    # class only defines the tool catalog; the prompt template is a separate module constant that
    # the rollout module applies through the agent configuration.
    """OpenAIChatAgent whose only tool is the screen.

    Beyond the tool, the one thing this adds is a system prompt, because the
    task template's prompt says *what* to do and something has to say *how* to
    act here: chiefly that turns are scarce and keystrokes are not. Without it
    a model spends one turn per keystroke and runs out of episode long before
    it runs out of task.
    """

    tool_catalog = (ChatComputerTool,)

    @classmethod
    def for_shot_width(cls, shot_width: int) -> type[ComputerChatAgent]:
        # [FUNCTION DOC] Class method returning an agent class whose tool uses the given screenshot
        # width. If the tool factory returns the default tool class (default width), returns this
        # class itself; otherwise dynamically creates and returns a subclass with the same name whose
        # tool catalog holds the width-specific tool. A new class is created on every call.
        """The agent with a specific screenshot width baked into its tool."""
        tool = computer_tool_class(shot_width)
        if tool is ChatComputerTool:
            return cls
        return type(cls.__name__, (cls,), {"tool_catalog": (tool,)})
