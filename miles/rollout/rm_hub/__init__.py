# [PACKAGE DOC]
# Reward-model hub. The package init exposes the async reward entry points used by rollouts:
# a single-sample reward dispatcher keyed by reward type (remote service, rule-based math, F1,
# GPQA multiple choice, instruction-following, random baselines, or a user-supplied function) and
# a batched variant. The sibling modules implement the individual rule-based scorers.
import asyncio
import hashlib
import random

import aiohttp

from miles.utils.function_registry import load_function
from miles.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward, get_gemma_math_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl


async def remote_rm(args, sample: Sample):
    # [FUNCTION DOC]
    # Score a sample with a remote reward service (network I/O). Posts a JSON body with the
    # sample's prompt, response and label to the reward-model url launch arg using a fresh HTTP
    # session, raises on an HTTP error status, and returns the decoded JSON response (the reward).
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def _resolve_reward_config(args, sample: Sample) -> tuple[str | None, str]:
    # [FUNCTION DOC] HELPER: Return a pair (custom reward function path or none, reward type
    # string). The path is the sample's reward spec path if set and non-empty, else the launch arg
    # (may be absent). The type is the first non-empty of: the reward spec type, the "rm_type"
    # entry of the sample metadata (only if metadata is a mapping), the launch arg type, else an
    # empty string; the result is stripped of whitespace.
    # Spec fields win when set; unset/empty fields fall back to sample metadata and process-wide args.
    spec = sample.reward_spec
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    custom_rm_path = (spec.custom_rm_path if spec is not None else None) or getattr(args, "custom_rm_path", None)
    rm_type = (
        (spec.rm_type if spec is not None else None) or metadata.get("rm_type") or getattr(args, "rm_type", None) or ""
    ).strip()
    return custom_rm_path, rm_type


async def async_rm(args, sample: Sample, **kwargs):
    # [FUNCTION DOC]
    # Compute the reward for one sample. Extra keyword arguments are forwarded only to a custom
    # reward function. If a custom function path resolves, load it and return the awaited result
    # of calling it with args and the sample. Otherwise dispatch on the reward type:
    # a "boxed_" prefix first replaces the response with the extracted boxed answer (empty string
    # if none) and is removed from the type. Types: remote_rm (remote service); deepscaler and
    # gemma_math (rule-based boxed-answer graders, 0/1); dapo (the DAPO scorer's result); math
    # (1 if the verl grader accepts else 0); f1 (F1 value only, first of the triple); gpqa
    # (multiple choice scorer given the metadata); ifbench (instruction-following scorer, imported
    # lazily when needed); random (random 0 or 1); deterministic_random (0 or 1 from the first
    # byte of a SHA-256 hash of the token list text plus response, stable across runs). An unknown
    # non-empty type, or an empty type, raises a not-implemented error. Despite being async, the
    # rule-based branches run synchronously.
    custom_rm_path, rm_type = _resolve_reward_config(args, sample)

    if custom_rm_path is not None:
        rm_function = load_function(custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    response = sample.response
    label = sample.label
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    # This function is intended for remote or time-consuming reward model evaluation.
    # Implement the actual logic as needed.
    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    elif rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "gemma_math":
        return get_gemma_math_reward(response, label)
    elif rm_type == "dapo":
        return compute_score_dapo(response, label)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    elif rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    elif rm_type == "ifbench":
        from .ifbench import compute_ifbench_reward

        return compute_ifbench_reward(response, label, metadata=metadata)
    elif rm_type == "random":
        return random.randint(0, 1)
    elif rm_type == "deterministic_random":
        content = str(sample.tokens) + response
        content_hash = hashlib.sha256(content.encode()).digest()
        return int(content_hash[0]) % 2
    elif rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")


async def batched_async_rm(
    args,
    samples: list[Sample],
    inplace_set_reward_field: bool = False,
    **kwargs,
) -> list[int | float] | None:
    # [FUNCTION DOC]
    # Compute rewards for a list of samples. If the in-place flag is set: compute rewards (same
    # call without the flag), assert each sample's reward is still unset (assertion error
    # otherwise), store the rewards on the samples and return None. Otherwise: if the custom
    # reward path launch arg is set, load that function and return its awaited result for the
    # whole list (it must handle batches); else run the per-sample reward concurrently and return
    # the rewards in order. Extra keyword arguments are forwarded.
    # KNOWN ISSUE: the batch path reads only the launch arg for the custom path (accessed without
    # a default), ignoring per-sample reward spec paths that the single-sample path honors.
    if inplace_set_reward_field:
        rewards = await batched_async_rm(args, samples, **kwargs)
        for sample, reward in zip(samples, rewards, strict=True):
            assert (
                sample.reward is None
            ), f"Overriding sample.reward from {sample.reward} to {reward}, is this intended?"
            sample.reward = reward
        return None

    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
