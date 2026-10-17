# [MODULE DOC] Reward hook for the OpenEnv Terminal-Bench-2 run: reads back the reward that the
# agent function stored in sample metadata. Supports both a single sample and a list of samples.
"""Reward function for the OpenEnv Terminal-Bench-2 (tbench2) run.

Task-agnostic: the agent function (``openenv_agent_function.run``) stores the
env-computed binary pytest reward in ``sample.metadata["reward"]``; this just
reads it back. Wired via ``--custom-rm-path openenv_generate.reward_func`` and
mirrors ``swe-agent-harbor-docker/generate.py:reward_func`` so it works for both the
single-sample (``async_rm``) and batched (``--custom-rm-path``) call paths.
"""

from miles.utils.types import Sample


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    # [FUNCTION DOC] Custom reward hook. For a list of samples returns the list of metadata values
    # under "reward" (default 0.0 when absent) in order; for one sample returns its value. No type
    # validation. args and extra keyword arguments are unused.
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)
