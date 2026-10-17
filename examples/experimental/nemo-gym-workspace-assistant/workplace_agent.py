# [MODULE DOC] Agent function for the NeMo Gym Workplace Assistant example. For one rollout it opens
# an episode on the Workplace resource server (URL from the environment variable
# WORKPLACE_RESOURCE_URL), then loops: send the transcript to the policy through the Miles
# session-scoped chat completions endpoint (so token ids and log probs are recorded by the
# session), forward every tool call the policy emits to the resource server, append the tool
# outputs as tool messages, and stop on a final answer, truncation, context/response budget
# exhaustion or the turn limit. Finally it asks the resource server to verify the episode and
# returns metrics that the generate function and reward function in the sibling modules consume.
# Entry point for the rollout framework is the run function at the bottom.
"""Workplace tool loop: all policy calls pass through the Miles TITO session."""

import json
import logging
import os
import re
import time
from copy import deepcopy
from typing import Any

import httpx

from miles.rollout.agentic.session import openai_session_url

logger = logging.getLogger(__name__)


def chat_tools(params: dict) -> list[dict]:
    # [FUNCTION DOC] HELPER: Converts the tool declarations of a Responses-API style request
    # (the "tools" list inside params) into chat-completions tool declarations. Each output item
    # has type "function" and a nested function object holding every field of the original tool
    # except its own "type" key. Returns the new list; does not mutate the input.
    return [
        {"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}} for tool in params["tools"]
    ]


def assistant_message(message: dict) -> dict:
    # [FUNCTION DOC] HELPER: Normalizes a chat-completions response message into the assistant
    # message appended to the transcript. Role is assistant; content is the original content or an
    # empty string when missing/null. The reasoning content and the tool calls are copied only when
    # present and non-empty. Returns a new dict; other fields of the response message are dropped.
    result = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("reasoning_content"):
        result["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        result["tool_calls"] = message["tool_calls"]
    return result


async def post_json(client: httpx.AsyncClient, url: str, body: dict) -> dict:
    # [FUNCTION DOC] HELPER: Sends an HTTP POST with the body encoded as JSON and returns the
    # decoded JSON response. Side effect: on an error status it logs the status code and the first
    # 2000 characters of the response text at error level. Raises the HTTP client's status error
    # for any 4xx/5xx response (after logging).
    response = await client.post(url, json=body)
    if response.is_error:
        logger.error("Workplace HTTP %s: %s", response.status_code, response.text[:2000])
    response.raise_for_status()
    return response.json()


class ContextBudgetExceeded(ValueError):
    # [CLASS DOC] Value-error subclass raised by the completion helper when the transcript is so
    # long that the policy has no room left to generate even one more token. The policy loop
    # catches it and ends the episode with the stop reason "context_limit".
    """The accumulated transcript leaves no space for another policy token."""


async def completion(client: httpx.AsyncClient, url: str, body: dict, context: int) -> dict:
    # [FUNCTION DOC] HELPER: Posts one chat-completions request and returns the decoded response,
    # retrying once with a smaller max_tokens when the server rejects the request for exceeding
    # the context window. Inputs: client, endpoint url, request body (must contain max_tokens),
    # and context, the total context length in tokens. Behavior on an HTTP status error:
    #  - status other than 400: re-raised.
    #  - status 400 whose text contains "<N> tokens from the input messages": the available room is
    #    the smaller of the requested max_tokens and (context - N - 8). If that is not positive,
    #    raises the context-budget exception; if it is not smaller than the requested max_tokens,
    #    the original error is re-raised; otherwise the request is repeated once with max_tokens
    #    replaced by the available room (the second call is not guarded, so its errors propagate).
    #  - status 400 without that phrase: if the text mentions input, "longer than" and "context
    #    length" the context-budget exception is raised, else the original error is re-raised.
    # The retry is safe because the session server does not record non-200 responses.
    try:
        return await post_json(client, url, body)
    except httpx.HTTPStatusError as error:
        text = error.response.text
        if error.response.status_code != 400:
            raise
        match = re.search(r"(\d+) tokens from the input messages", text)
        if match is None:
            if "input" in text and "longer than" in text and "context length" in text:
                raise ContextBudgetExceeded(text) from error
            raise
        available = min(body["max_tokens"], context - int(match.group(1)) - 8)
        if available <= 0:
            raise ContextBudgetExceeded(text) from error
        if available >= body["max_tokens"]:
            raise
        # Rejected requests contain no generated tokens. The session server
        # leaves non-200 responses unrecorded, so this transcript can be retried.
        return await post_json(client, url, {**body, "max_tokens": available})


async def policy_loop(
    client: httpx.AsyncClient,
    base_url: str,
    resource_url: str,
    params: dict,
    request_kwargs: dict,
    context: int,
    max_turns: int,
) -> dict:
    # [FUNCTION DOC] Runs the multi-turn tool loop for one episode and returns its metrics.
    # Inputs: client; base_url of the Miles rollout/session server (the session chat endpoint is
    # derived from it); resource_url, the per-episode URL on the Workplace server; params, holding
    # the initial messages ("input") and tool declarations ("tools"); request_kwargs, sampling
    # arguments copied into every request (must contain max_tokens, which is the total response
    # budget across all turns); context, the total context length; max_turns, the turn cap.
    # Per turn: compute remaining budget = total budget minus tokens generated so far (stop with
    # "response_budget" when it is not positive); post a chat request with the fixed served model
    # name, the transcript, the tools, parallel tool calls disabled, thinking enabled with history
    # thinking kept, and the remaining budget as max_tokens. A context-budget failure stops with
    # "context_limit". If the finish reason is "abort", returns immediately with only an invalid
    # flag and stop "abort". Otherwise accumulates completion tokens, appends the normalized
    # assistant message, executes each tool call on the resource server (timing it) and appends a
    # tool message with the sorted-key JSON output. Then: finish reason "length" stops with
    # "truncated"; no tool calls stops with "final_answer". Falling out of the loop leaves the
    # default stop reason "turn_limit".
    # After the loop it calls the verify endpoint (timed). Raises a runtime error if verification
    # reports invalid or state-replay inconsistency. Returns a dict with the validity flag, reward,
    # stop reason, turns, generated completion tokens, number of tool calls recorded by the server,
    # replay-consistency flag, and agent_metrics holding total tool time in seconds (tool calls plus
    # verification).
    messages = deepcopy(params["input"])
    tools = chat_tools(params)
    budget = request_kwargs["max_tokens"]
    generated = 0
    tool_time = 0.0
    stop = "turn_limit"
    turns = 0
    for _ in range(max_turns):
        # The inference server owns tokenization. If the remaining response
        # budget exceeds the actual available context, completion() retries
        # with the exact input count from the server's rejected request.
        available = budget - generated
        if available <= 0:
            stop = "response_budget"
            break
        body = {
            **request_kwargs,
            "model": "nemotron35-lightning",
            "messages": messages,
            "tools": tools,
            "parallel_tool_calls": False,
            "max_tokens": available,
            "chat_template_kwargs": {"enable_thinking": True, "truncate_history_thinking": False},
        }
        try:
            response = await completion(client, openai_session_url(base_url) + "/chat/completions", body, context)
        except ContextBudgetExceeded:
            stop = "context_limit"
            break
        choice = response["choices"][0]
        if choice["finish_reason"] == "abort":
            return {"workplace_episode_valid": False, "workplace_stop": "abort"}
        usage = response["usage"]
        generated += usage["completion_tokens"]
        turns += 1
        message = assistant_message(choice["message"])
        messages.append(message)
        for call in message.get("tool_calls", []):
            started = time.monotonic()
            output = await post_json(client, resource_url + "/tool", call["function"])
            tool_time += time.monotonic() - started
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(output, sort_keys=True, ensure_ascii=False),
                }
            )
        if choice["finish_reason"] == "length":
            stop = "truncated"
            break
        if not message.get("tool_calls"):
            stop = "final_answer"
            break
    started = time.monotonic()
    verification = await post_json(client, resource_url + "/verify", {})
    tool_time += time.monotonic() - started
    if not verification.get("valid") or not verification.get("state_replay_consistent"):
        raise RuntimeError("Native Workplace verification failed")
    return {
        "workplace_episode_valid": True,
        "workplace_reward": verification["reward"],
        "workplace_stop": stop,
        "workplace_turns": turns,
        "workplace_completion_tokens": generated,
        "workplace_tool_calls": verification["tool_calls"],
        "workplace_state_replay_consistent": True,
        "agent_metrics": {"total_tool_time": tool_time},
    }


async def run(base_url: str, prompt: Any, request_kwargs: dict, metadata: dict, **kwargs: Any) -> dict:
    # [FUNCTION DOC] Agent entry point called by the rollout framework for one sample. Reads the
    # policy payload from metadata (key workplace_policy), and requires the framework prompt to be
    # equal to its input messages, else raises a value error. Reads the resource server URL from the
    # required environment variable WORKPLACE_RESOURCE_URL (missing -> key error), creates an
    # episode for the task id in metadata (HTTP side effect), then delegates to the policy loop with
    # the sample's max_seq_len as context and the optional workplace_max_turns (default 24) as the
    # turn cap. Uses a client with a one-hour overall timeout, 30 s connect/pool timeouts and
    # environment proxy settings ignored. In all cases the episode is deleted afterwards; a failed
    # cleanup is only logged (the server expires abandoned sessions after three hours). Extra
    # keyword arguments are ignored. Returns the loop's metrics dict.
    # Neither ground truth nor synthesis provenance is part of this payload.
    params = metadata["workplace_policy"]
    if prompt != params["input"]:
        raise ValueError("Miles prompt and Workplace policy input disagree")
    resource = os.environ["WORKPLACE_RESOURCE_URL"].rstrip("/")
    timeout = httpx.Timeout(3600, connect=30, pool=30)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        response = await post_json(client, f"{resource}/sessions/{metadata['workplace_task_id']}", {})
        session_url = f"{resource}/sessions/{response['session_id']}"
        try:
            return await policy_loop(
                client,
                base_url,
                session_url,
                params,
                request_kwargs,
                metadata["max_seq_len"],
                metadata.get("workplace_max_turns", 24),
            )
        finally:
            try:
                cleanup = await client.delete(session_url)
                cleanup.raise_for_status()
            except (httpx.HTTPError, TimeoutError):
                # The server also expires abandoned sessions after three hours.
                logger.exception("Could not release Workplace episode %s", response["session_id"])
