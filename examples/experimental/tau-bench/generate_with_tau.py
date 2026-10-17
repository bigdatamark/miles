# [MODULE DOC] Custom generate function for training on tau-bench (retail by default). The task
# index is carried in the sample prompt; the function builds the tau-bench environment with an
# LLM-driven user simulator, creates a trainable agent, runs one full interaction and converts the
# result to a Miles sample. Import-time side effects: reads the environment variables
# TAU_USER_MODEL_PROVIDER (default gemini) and TAU_USER_MODEL (default gemini-2.5-flash-lite),
# raises a runtime error if the provider's api key variable (GEMINI_API_KEY, DEEPSEEK_API_KEY, or
# <PROVIDER>_API_KEY for others) is unset, and builds the module-level run configuration (env
# retail, tool-calling agent, train split, llm user strategy; model fields are placeholders).
"""
Tau-Bench Integration for miles Training

This module provides the main interface for training agents in tau-bench environments
using the miles framework. It handles agent-environment interactions and converts
results to the format expected by miles's training pipeline.
"""

import logging
import os
from typing import Any

from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from trainable_agents import InteractionResult, Status, agent_factory

from miles.utils.types import Sample

# Set up logger for this module
logger = logging.getLogger(__name__)

# User simulator configuration.
# tau-bench drives the user simulator through litellm, so any litellm-supported
# provider works. Pick the provider/model with the env vars below and supply the
# matching key through that provider's own *_API_KEY env var:
#   Gemini  : TAU_USER_MODEL_PROVIDER=gemini   TAU_USER_MODEL=gemini-2.5-flash-lite  GEMINI_API_KEY=...
#   DeepSeek: TAU_USER_MODEL_PROVIDER=deepseek TAU_USER_MODEL=deepseek-chat          DEEPSEEK_API_KEY=...
# Defaults preserve the original Gemini behavior.
USER_MODEL_PROVIDER = os.environ.get("TAU_USER_MODEL_PROVIDER", "gemini")
USER_MODEL = os.environ.get("TAU_USER_MODEL", "gemini-2.5-flash-lite")

# litellm reads the provider's own *_API_KEY env var; just confirm it is set so
# the run fails fast with a clear message instead of deep inside a rollout.
_PROVIDER_KEY_ENV = {"gemini": "GEMINI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
_key_env = _PROVIDER_KEY_ENV.get(USER_MODEL_PROVIDER, f"{USER_MODEL_PROVIDER.upper()}_API_KEY")
if not os.environ.get(_key_env):
    raise RuntimeError(
        f"tau-bench user simulator needs {_key_env} set for provider "
        f"'{USER_MODEL_PROVIDER}'. Export it before launching the run."
    )

# Tau-bench configuration
TAU_CONFIGS = {
    "env": "retail",  # Select between ["retail", "airline"]
    "agent": "tool-calling",  # Select between ["tool-calling", "act", "react", "few-shot"]
    "user_model": USER_MODEL,  # Cheap model for the user simulator
    "task_split": "train",  # Select between ["train", "test", "dev"] for retail
    "user_strategy": "llm",  # Select between ["llm", "react", "verify", "reflection"]
    "model_provider": "auto_router",  # Unused, required
    "model": "qwen3-4b",  # Unused, required
    "user_model_provider": USER_MODEL_PROVIDER,
}
tau_config = RunConfig(**TAU_CONFIGS)


def res_to_sample(res: InteractionResult, task_index: int) -> Sample:
    # [FUNCTION DOC] HELPER: Builds a Miles sample from an interaction result. Status mapped from
    # the result status to the lowercase strings completed, truncated or aborted (none for any
    # other value). The sample takes index (the task index), prompt, tokens, response, reward, loss
    # mask and the result's info mapping as metadata. Response length: the result's own attribute if
    # present; else the loss mask length if non-empty; else the token count if tokens exist; else 0.
    # Debug logs are written. Returns the sample.
    # KNOWN ISSUE: the comment says the loss mask covers only the response part, which the fallback
    # relies on; the final fallback uses total token count which would include the prompt.
    """
    Convert InteractionResult to Sample format for miles training.

    This function transforms the tau-bench interaction result into the format
    expected by miles's training pipeline, handling status mapping and response
    length calculation.

    Args:
        res: InteractionResult from tau-bench agent
        task_index: Index of the task being processed

    Returns:
        Sample object for miles training
    """
    # Map tau-bench status to miles status
    status_mapping = {
        Status.COMPLETED: "completed",
        Status.TRUNCATED: "truncated",
        Status.ABORTED: "aborted",
    }
    status = status_mapping.get(res.status)

    # Debug logging for response tracking
    logger.debug(
        f"res_to_sample: response_length="
        f"{res.response_length if hasattr(res, 'response_length') else 'None'}, "
        f"loss_mask_len={len(res.loss_mask) if res.loss_mask else 'None'}, "
        f"tokens_len={len(res.tokens) if res.tokens else 'None'}"
    )

    # Create sample with basic information
    sample = Sample(
        index=task_index,
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status,
        metadata=res.info,
    )

    # Ensure response_length is set correctly
    if hasattr(res, "response_length"):
        sample.response_length = res.response_length
    else:
        # Fallback: calculate from loss_mask if available
        if res.loss_mask:
            # loss_mask only contains response part, so length equals response_length
            sample.response_length = len(res.loss_mask)
        elif res.tokens:
            # If no loss_mask available, use total tokens as fallback
            sample.response_length = len(res.tokens)
        else:
            sample.response_length = 0
            logger.debug(f"res_to_sample: Set response_length={sample.response_length}")

    return sample


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    # [FUNCTION DOC] Custom generate function; returns a NEW sample (the input sample is not
    # modified). Asserts partial rollout is off. Takes the task index as the integer value of the
    # sample prompt text. Builds the tau-bench environment from the module config (environment
    # name, user strategy, user model and provider, task split, that task index), creates the
    # agent through the agent factory with the env's tools info and wiki, the config, the rollout
    # args and sampling params, runs the agent's async solve on the environment with the agent's
    # own stored rollout args and sampling params and the task index, converts the result to a
    # sample and returns it. Logs start and finish.
    # KNOWN ISSUE: the args annotation says mapping but the code uses attribute access on a
    # namespace object.
    """
    Generate a complete agent-environment interaction trajectory for tau-bench.

    This is the main entry point for miles training. It creates a tau-bench
    environment, initializes a trainable agent, and executes a full interaction
    trajectory. The result is converted to miles's Sample format for training.

    Args:
        args: Rollout arguments from miles training pipeline
        sample: Sample containing task index in prompt field
        sampling_params: LLM sampling parameters

    Returns:
        Sample object containing the complete interaction trajectory

    Raises:
        AssertionError: If partial rollout is requested (not supported)
    """
    # Validate arguments
    assert not args.partial_rollout, "Partial rollout is not supported for tau-bench interactions."

    # Extract task index from sample prompt
    task_index = int(sample.prompt)
    logger.info(f"Starting agent-environment interaction for task {task_index}")

    # Initialize tau-bench environment
    env = get_env(
        env_name=tau_config.env,
        user_strategy=tau_config.user_strategy,
        user_model=tau_config.user_model,
        user_provider=tau_config.user_model_provider,
        task_split=tau_config.task_split,
        task_index=task_index,
    )

    # Create trainable agent
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=tau_config,
        rollout_args=args,
        sampling_params=sampling_params,
    )

    # Execute agent-environment interaction
    # Note: The sample.prompt field contains the task index for repeatability
    interaction_result = await agent.asolve(env, agent.rollout_args, agent.sampling_params, task_index)

    # Convert to miles Sample format
    result_sample = res_to_sample(interaction_result, task_index)

    logger.info(f"Finished agent-environment interaction for task {task_index}")
    return result_sample
