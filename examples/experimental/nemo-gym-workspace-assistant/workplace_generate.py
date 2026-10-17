# [MODULE DOC] Custom generate function and reward function for the Workplace Assistant example.
# The generate function wraps the generic agentic tool-call generator and then validates every
# produced sample, aborting invalid episodes and raising on structurally broken ones so that bad
# data never reaches the optimizer. The reward function simply returns the native Workplace
# verifier reward that the agent function stored in sample metadata.
"""Validate Workplace episodes before they can enter an optimizer batch."""

import math
from typing import Any

import numpy as np

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.agentic_tool_call import generate as agentic_generate
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    # [FUNCTION DOC] Runs the generic agentic generator on the input, then post-validates each
    # returned sample (the output may hold a single sample or a list; both are handled) and returns
    # the original output object. Per sample, in order:
    #  1. metadata lacks a truthy "workplace_episode_valid": mark the sample aborted and skip it.
    #  2. status already aborted: skip it.
    #  3. run the sample's own validation; raise a value error if the loss mask is missing or all
    #     zero; raise if there is more than one turn and the loss mask is all ones (tool
    #     observations must be masked out); raise if rollout log probs are missing or contain any
    #     non-finite value.
    #  4. if rollout routing replay is enabled in args: routed experts must exist with shape
    #     (number of tokens minus one, num_layers, moe_router_topk), else raise; the dtype must be
    #     32-bit integer and every index within [0, num_experts), else raise.
    #  5. if the recorded stop reason is truncated, response_budget, context_limit or turn_limit,
    #     set the sample status to truncated.
    # Side effect: mutates sample status in place. Raises value errors as described.
    output = await agentic_generate(input)
    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    for sample in samples:
        if not sample.metadata.get("workplace_episode_valid"):
            sample.status = Sample.Status.ABORTED
            continue
        if sample.status == Sample.Status.ABORTED:
            continue
        sample.validate()
        if sample.loss_mask is None or not any(sample.loss_mask):
            raise ValueError("Workplace episode has no policy tokens to train")
        if sample.metadata["workplace_turns"] > 1 and all(sample.loss_mask):
            raise ValueError("Multi-turn tool observations must be masked out")
        if sample.rollout_log_probs is None or not all(math.isfinite(p) for p in sample.rollout_log_probs):
            raise ValueError("Missing or nonfinite rollout log probabilities")
        routes = sample.rollout_routed_experts
        if input.args.use_rollout_routing_replay:
            if routes is None or routes.shape != (
                len(sample.tokens) - 1,
                input.args.num_layers,
                input.args.moe_router_topk,
            ):
                raise ValueError("Missing or misaligned routing replay")
            if routes.dtype != np.int32 or routes.min() < 0 or routes.max() >= input.args.num_experts:
                raise ValueError("Invalid routing replay expert indices")
        if sample.metadata["workplace_stop"] in {"truncated", "response_budget", "context_limit", "turn_limit"}:
            sample.status = Sample.Status.TRUNCATED
    return output


generate.add_arguments = agentic_generate.add_arguments


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    # [FUNCTION DOC] Custom reward hook. Reads the native reward from sample metadata (key
    # workplace_reward) and returns it as a float. Raises a value error if the episode was not
    # marked valid (invalid episodes are aborted by the generate function and must not reach here)
    # or if the reward is anything other than exactly 0.0 or 1.0. args and extra keyword arguments
    # are unused.
    if not sample.metadata.get("workplace_episode_valid"):
        raise ValueError("An invalid episode reached the reward hook")
    value = sample.metadata["workplace_reward"]
    if value not in (0.0, 1.0):
        raise ValueError(f"Unexpected native Workplace reward: {value!r}")
    return float(value)
