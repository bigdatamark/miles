# [MODULE DOC] Plug-ins for the SWE-agent example: a custom reward function that reads the reward
# the agent environment already computed (stored in each sample's metadata under the reward key,
# default 0), helpers that aggregate per-sample agent statistics (turns, tool calls, timings and
# time ratios) into batch-level logged metrics, and a rollout function class that appends those
# metrics to the output of a normal inference rollout. Agents that omit a metric are skipped
# rather than counted as zero. The launcher scripts refer to this module by name for the reward
# function and the rollout function class.
"""
SWE-Agent: reward, metrics, and rollout class.

The generate function is provided by:
    miles.rollout.generate_hub.agentic_tool_call.generate
with --custom-agent-function-path pointing to swe_agent_function.run

Task-type agnostic — reward is pre-computed by the Harbor environment
and stored in sample.metadata["reward"] regardless of task type.

Dynamic filter uses the general-purpose ``check_no_aborted`` from
``miles.rollout.filter_hub.dynamic_sampling_filters``.

Components:
  - reward_func: reads pre-computed reward from sample metadata
  - aggregate_agent_metrics: aggregates agent timing/count metrics
  - RolloutFn: InferenceRolloutFn subclass that logs agent metrics
"""

import logging

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


# -- Reward --


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    # [FUNCTION DOC] Custom reward function. Given one sample returns its metadata reward value
    # (0.0 if absent); given a list returns the list of such values in order. Args and extra
    # keyword arguments are ignored. Raises if a sample has no metadata mapping.
    """Reward is pre-computed by the agent environment during generate().

    Handles both single-sample calls (from ``async_rm``) and batched calls
    (from ``batched_async_rm`` when ``--custom-rm-path`` is set).
    """
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)


# -- Agent Metrics Aggregation --


def _collect_values(all_metrics: list[dict], key: str) -> list[float]:
    # [FUNCTION DOC] HELPER: From a list of per-sample metric mappings, collect the values under
    # the key, skipping mappings where it is absent or none (zeros are kept). Returns a list.
    """Values agents reported for ``key``; agents that omit it are skipped.

    Defaulting to 0 would log a fake measurement and drag the batch mean down.
    Callers skip empty lists, so an unreported key stays out of the log.
    """
    return [value for metrics in all_metrics if (value := metrics.get(key)) is not None]


def _agg_mean(metrics: dict, all_metrics: list[dict], keys: list[str], prefix: str = "agent/", suffix: str = "_mean"):
    # [FUNCTION DOC] HELPER: For each key, compute the mean of the reported values and store it in
    # the metrics mapping (mutated in place) under the prefix, key and suffix joined together;
    # keys nobody reported are left out. No return.
    for key in keys:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"{prefix}{key}{suffix}"] = sum(values) / len(values)


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    # [FUNCTION DOC] Aggregate the agent metrics mappings found in sample metadata (samples
    # without metadata or with an empty metrics mapping are skipped) into one flat mapping with
    # keys under the agent prefix. Returns an empty mapping if no sample has metrics. Outputs: mean
    # and sum of turns and of tool calls; means (with a mean suffix) of model query time sum,
    # environment execution time sum, eval time and agent run time; means without a suffix of time
    # per turn, average model query time, average environment time, and the model, environment
    # and eval time ratios; and mean, max and min of total time. Only keys reported by at least
    # one sample appear.
    """Aggregate agent metrics across samples for logging."""
    all_metrics = [
        s.metadata.get("agent_metrics", {})
        for s in samples
        if hasattr(s, "metadata") and s.metadata and s.metadata.get("agent_metrics")
    ]
    if not all_metrics:
        return {}

    metrics = {}

    for key in ["turns", "tool_calls"]:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)

    _agg_mean(metrics, all_metrics, ["model_query_time_sum", "env_execution_time_sum", "eval_time", "agent_run_time"])
    _agg_mean(metrics, all_metrics, ["time_per_turn", "model_query_time_avg", "env_execution_time_avg"], suffix="")
    _agg_mean(metrics, all_metrics, ["model_time_ratio", "env_time_ratio", "eval_time_ratio"], suffix="")

    values = _collect_values(all_metrics, "total_time")
    if values:
        metrics["agent/total_time_mean"] = sum(values) / len(values)
        metrics["agent/total_time_max"] = max(values)
        metrics["agent/total_time_min"] = min(values)

    return metrics


# -- Rollout Function --


class RolloutFn(InferenceRolloutFn):
    # [CLASS DOC] Inference rollout function that behaves exactly like its parent for generation
    # and only post-processes training output to add agent metrics. Selected in launchers through
    # the rollout function path flag.
    """Rollout function with agent metrics aggregation."""

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        # [FUNCTION DOC] Run the parent's training rollout, flatten its samples (the output may
        # hold groups as lists or single samples) and aggregate their agent metrics. If any metrics
        # result, merge them into the output's metrics mapping (creating one if absent, overwriting
        # same-named keys) and log them at info level with the rollout id. Returns the output.
        output = await super()._call_train(input)

        all_samples = []
        for group in output.samples:
            if isinstance(group, list):
                all_samples.extend(group)
            else:
                all_samples.append(group)

        agent_metrics = aggregate_agent_metrics(all_samples)
        if agent_metrics:
            metrics = output.metrics or {}
            metrics.update(agent_metrics)
            output.metrics = metrics
            logger.info(f"Agent metrics for rollout {input.rollout_id}: {agent_metrics}")

        return output
