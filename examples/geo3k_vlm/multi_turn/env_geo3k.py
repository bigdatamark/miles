# [MODULE DOC] Interaction environment for multi-turn geo3k. The model must call a scoring tool by
# emitting a tool-call tag containing a JSON payload with the tool name and an answer argument; the
# environment grades that answer against the ground truth with the math answer checker and returns
# short textual feedback as the next observation. Also provides the factory function used by the
# rollout module to construct the environment from a sample and the run args (the turn limit comes
# from the args, which are populated from the custom config file).
from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    orjson = None
from examples.geo3k_vlm.multi_turn.base_env import BaseInteractionEnv

from miles.rollout.rm_hub import grade_answer_verl
from miles.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# Matches the JSON payload emitted between <tool_call> ... </tool_call> tags.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Accept either name; verl uses `calc_geo3k_reward` while the instruction refers to `calc_score`.
SUPPORTED_TOOL_NAMES = {"calc_score", "calc_geo3k_reward"}


class Geo3kEnv(BaseInteractionEnv):
    # [CLASS DOC] Environment for one geo3k problem. State: the ground truth answer text (or none),
    # the list of executed tool calls (name, answer, score), the last tool score, the turn counter
    # and the optional turn limit. An episode ends when no tool call is detected in a response or the
    # turn limit is reached.
    # KNOWN ISSUE: the docstring below says the episode ends immediately after each step, but the
    # code only ends it on a missing tool call or at the turn limit.
    """
    Minimal interaction environment for multi-turn geo3k with a scoring tool.

    The model is expected to emit a <tool_call>{...}</tool_call> payload that includes
    an `answer` argument. We run the math reward checker against the ground truth and
    return the score as the next observation. The episode ends immediately after each
    step; responses are provided but no further turns are taken.
    """

    def __init__(self, *, ground_truth: str | None = None, max_turns: int | None = None):
        # [FUNCTION DOC] Keyword-only constructor: stores the ground truth as text (none stays
        # none), an empty tool call list, no last score, turn counter zero and the turn limit.
        self.ground_truth = str(ground_truth) if ground_truth is not None else None
        self.tool_calls: list[dict[str, Any]] = []
        self.last_tool_score: float | None = None
        self.turn = 0
        self.max_turns = max_turns

    def reset(self):
        # [FUNCTION DOC] Clears the tool call list, last score and turn counter. Returns a pair of an
        # empty observation mapping (the question is already in the prompt) and an info mapping with
        # a flag telling whether a ground truth is available.
        self.tool_calls.clear()
        self.last_tool_score = None
        self.turn = 0
        # No initial observation is needed; the question lives in the prompt.
        observation: dict[str, Any] = {}
        reset_info = {"ground_truth_available": self.ground_truth is not None}
        return observation, reset_info

    def close(self):
        # [FUNCTION DOC] No-op; there are no resources to release.
        """No resources to release."""
        return

    def _extract_tool_call(self, text: str) -> dict[str, Any] | None:
        # [FUNCTION DOC] HELPER: Finds the LAST tool-call tag pair in the text (multi-line, shortest
        # match, with a braced payload between the tags) and returns a mapping with the tool name and
        # arguments, or none. None if there is no tag, the payload is not valid JSON (decoded with the
        # fast JSON library when installed, else the standard one), the name is missing/empty, or the
        # arguments are a text that is not valid JSON (warning logged). The name comes from the
        # payload's name entry, else from a nested function entry; arguments likewise (default empty
        # mapping), and a text value is decoded as JSON.
        # KNOWN ISSUE: decoded arguments that are not a mapping (for example a JSON list) are
        # returned as is, and the step method then fails when it reads the answer from them.
        """
        Parse the latest tool call payload from the assistant response.
        Supports the <tool_call>{...}</tool_call> convention used in the
        SGLang multi-turn templates. Tool tags are mandatory.
        """
        matches = list(TOOL_CALL_RE.finditer(text))
        raw_json = None
        if matches:
            raw_json = matches[-1].group(1).strip()

        if raw_json is None:
            return None

        payload = self._parse_tool_payload(raw_json)
        if payload is None:
            return None

        name = payload.get("name") or payload.get("function", {}).get("name")
        arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("Tool call arguments are not valid JSON; rejecting tool call.")
                return None

        if not name:
            return None
        return {"name": name, "arguments": arguments}

    def _score_answer(self, answer: str) -> float:
        # [FUNCTION DOC] HELPER: Grades an answer against the ground truth. Returns 0.0 if there is
        # no ground truth (or it is empty). Otherwise trims the answer and tries it as given, and, if
        # it has no boxed marker, also wrapped in a boxed marker; returns 1.0 for the first candidate
        # accepted by the math answer checker, 0.0 if none (a checker exception is logged at debug
        # level and the next candidate is tried).
        """
        Use the same logic as the single-turn math reward model.
        We accept either boxed or raw numeric strings by retrying with a boxed wrapper.
        """
        if not self.ground_truth:
            return 0.0

        answer = answer.strip()
        candidates = [answer]
        if "\\boxed" not in answer:
            candidates.append(f"\\boxed{{{answer}}}")

        for candidate in candidates:
            try:
                if grade_answer_verl(candidate, self.ground_truth):
                    return 1.0
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("grade_answer_verl failed on %s: %s", candidate, exc)
                continue
        return 0.0

    def _extract_answer_from_text(self, text: str) -> str | None:
        # [FUNCTION DOC] HELPER: Picks a short answer from free text: the last boxed content if any;
        # else the last non-empty line trimmed to 512 characters; else the trimmed whole text cut to
        # 512 characters; none if the text is blank.
        # KNOWN ISSUE: not called anywhere in this module (dead code).
        """
        Prefer a concise answer by pulling the last \\boxed{} chunk; fall back to the last
        non-empty line (capped) to avoid echoing the whole response body.
        """
        boxed = extract_boxed_answer(text)
        if boxed:
            return str(boxed).strip()
        for line in reversed(text.splitlines()):
            cleaned = line.strip()
            if cleaned:
                return cleaned[:512]
        trimmed = text.strip()
        return trimmed[:512] if trimmed else None

    def _extract_balanced_json(self, text: str, start: int) -> str | None:
        # [FUNCTION DOC] HELPER: Scans from the given position (an opening brace) and returns the
        # substring up to and including the matching closing brace, tracking nesting depth and
        # ignoring braces inside double-quoted strings (a backslash escapes the next character).
        # Returns none if the braces never balance.
        # KNOWN ISSUE: not called anywhere in this module (dead code).
        """
        Best-effort balanced brace extraction starting at `start` (index of an opening '{').
        Keeps string-awareness to avoid terminating inside quoted braces.
        """
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "\\" and not escaped:
                escaped = True
                continue
            if ch == '"' and not escaped:
                in_string = not in_string
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
            escaped = False
        return None

    def _build_tool_feedback(self, score: float, parsed_answer: str) -> str:
        # [FUNCTION DOC] HELPER: Builds the feedback text for a graded answer. The turn just played is
        # the counter minus one (zero based). The warning turn is turn limit minus 2 (minus 1 if the
        # limit is below 2); the current turn is "final" when a limit exists and the turn index is at
        # or after that warning turn (one turn earlier than the real last turn so the model sees it
        # in time). Score 1.0: says the parsed answer matches and asks for the final solution in a
        # boxed form. Score 0.0: says the answer is wrong and not to repeat it; on the final turn
        # additionally tells the model not to call the tool again and to give the final boxed answer.
        # Any other score returns none (implicit).
        # KNOWN ISSUE: the final-turn message contains a typo ("fiinal"); scores other than exactly
        # 1.0 or 0.0 give no text, though the grader only returns those two.
        """
        Provide concise feedback for the model to continue reasoning.
        """
        turn_idx = self.turn - 1  # zero-based
        # Send the final reminder one turn before the true last turn so the model sees it in time.
        last_warning_turn = None
        if self.max_turns is not None:
            if self.max_turns >= 2:
                last_warning_turn = self.max_turns - 2
            else:
                last_warning_turn = self.max_turns - 1
        is_final_turn = last_warning_turn is not None and turn_idx >= last_warning_turn

        if score == 1.0:
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' matches the reference. "
                "You can now stop reasoning and provide the final solution in \\boxed{}."
            )
        if score == 0.0:
            if is_final_turn:
                return (
                    f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
                    "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary. "
                    "Since you only have one chance to answer, don't call tool again. You should provide your final answer in the form below Answer: \\boxed{$Answer} where $Answer is your fiinal answer to this problem."
                )
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
                "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary."
            )

    # Called during rollout after receiving a model response
    def step(self, response_text: str):
        # [FUNCTION DOC] Processes one model response and returns a triple of observation mapping
        # (text under "obs_str", role "tool", plus the tool score when graded), done flag, and an info
        # mapping (copy of the parsed tool call, whether it was executed, and for graded calls the
        # tool name, answer and score). Increments the turn counter; the turn is final when a limit is
        # set and the counter reached it. Cases in order: (1) no valid tool call: observation says no
        # tool call was detected, done true, executed false. (2) tool name not supported (calc_score
        # or calc_geo3k_reward): observation explains the expected call format, done = final turn,
        # executed false. (3) the answer argument is missing/blank: observation asks for an answer,
        # done = final turn, executed false and an answer-missing flag. (4) otherwise grades the answer
        # text, stores the score and appends a record to the tool call list, observation is the
        # feedback text, done = final turn, executed true.
        self.turn += 1
        is_final_turn = self.max_turns is not None and self.turn >= self.max_turns
        tool_call = self._extract_tool_call(response_text)
        info: dict[str, Any] = {"tool_call": deepcopy(tool_call)}

        if not tool_call:
            info["tool_executed"] = False
            obs = {
                "obs_str": "No tool call detected; ending the episode.",
                "role": "tool",
            }
            return obs, True, info

        name = (tool_call.get("name") or "").strip()
        arguments = tool_call.get("arguments") or {}
        if name not in SUPPORTED_TOOL_NAMES:
            obs = {
                "obs_str": (
                    f"Tool `{name}` is not supported. "
                    'Call `calc_score` (or `calc_geo3k_reward`) via <tool_call>{"name": "calc_score", "arguments": {"answer": "<digits>"}}</tool_call> (format must be <tool_call>(JSON)</tool_call>)'
                    "to check your solution."
                ),
                "role": "tool",
            }
            info["tool_executed"] = False
            return obs, is_final_turn, info

        raw_answer = arguments.get("answer", None)
        parsed_answer = "" if raw_answer is None else str(raw_answer)
        if not parsed_answer.strip():
            obs = {
                "obs_str": (
                    "Tool call detected but no `answer` was provided. "
                    'Call `calc_score` (or `calc_geo3k_reward`) via <tool_call>{"name": "calc_score", "arguments": {"answer": "<digits>"}}</tool_call> '
                    "to check your solution."
                ),
                "role": "tool",
            }
            info["tool_executed"] = False
            info["answer_missing"] = True
            return obs, is_final_turn, info

        score = self._score_answer(parsed_answer)
        self.last_tool_score = score
        tool_record = {"name": name, "answer": parsed_answer, "score": score}
        self.tool_calls.append(tool_record)
        info.update(tool_record)
        info["tool_executed"] = True

        obs = {
            "obs_str": self._build_tool_feedback(score, parsed_answer),
            "role": "tool",
            "tool_score": score,
        }

        return obs, is_final_turn, info

    def _parse_tool_payload(self, raw_json: str) -> dict[str, Any] | None:
        # [FUNCTION DOC] HELPER: Decodes the JSON text with the fast JSON library when installed, else
        # the standard one; on any failure logs a warning and returns none.
        """Parse tool payload strictly as JSON. Malformed payloads are rejected."""
        loader = orjson.loads if orjson is not None else json.loads
        try:
            return loader(raw_json)
        except Exception as exc:
            logger.warning("Failed to decode tool call payload: %s", exc)
            return None


def _extract_ground_truth(sample: Sample | None) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the sample's label as text, or none if the sample or its label
    # is missing.
    # KNOWN ISSUE: the docstring says metadata is also consulted, but that fallback is commented
    # out, so only the label is used.
    """Resolve the ground-truth answer from label or metadata."""
    if sample is None:
        return None
    if sample.label is not None:
        return str(sample.label)
    # metadata = sample.metadata
    # for key in ("answer", "ground_truth", "label"):
    #     if key in metadata and metadata[key] is not None:
    #         return str(metadata[key])
    return None


def build_env(sample: Sample | None = None, args: Any | None = None, **_: Any) -> Geo3kEnv:
    # [FUNCTION DOC] Factory called by the rollout module. Reads the turn limit from the args
    # (attribute access; none for args raises an attribute error); if the limit is none raises a
    # value error telling that it must come from the custom config file. Gets the ground truth from
    # the sample label; if missing logs a warning (the tool will always score 0). Returns an
    # environment with that ground truth and turn limit. Extra keyword arguments are ignored.
    # KNOWN ISSUE: the docstring says the ground truth may come from metadata, but only the label
    # is used.
    """
    Construct a Geo3kEnv. Ground truth is pulled from sample.label or metadata.
    """
    ground_truth = _extract_ground_truth(sample)
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    if ground_truth is None:
        logger.warning("Ground truth answer missing; calc_score tool will always return 0.")
    return Geo3kEnv(ground_truth=ground_truth, max_turns=max_turns)
