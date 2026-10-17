# [MODULE DOC] Logging of rollout and evaluation metrics to the tracking backend. Public entry
# points: log evaluation results per dataset, log a skipped evaluation point, and log training rollout
# metrics (reward, response length, performance throughput, zero-std groups, speculative decoding,
# prefix cache, reward categories, weight-version mixing, TITO session mismatch rate, pass rate).
# Both main entry points first give an optional user-supplied custom log function (loaded by path
# from args) the chance to handle logging and skip the default if it returns a truthy value.
import logging
from numbers import Number
from typing import Any

import numpy as np

from miles.rollout.session.v2.metrics import SESSION_ROLLOUT_METRICS_KEY
from miles.utils.function_registry import load_function
from miles.utils.iter_utils import group_by
from miles.utils.metric_utils import (
    compute_pass_rate,
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
    has_repetition,
    namespace_metrics,
)
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    # [FUNCTION DOC] Log evaluation metrics. Input data: a mapping from dataset key to a mapping with
    # rewards (list, entries may be none), optionally samples and truncated flags. If a custom eval
    # log function path is configured, call it with (rollout id, args, data, extra metrics); a truthy
    # return ends logging and this function returns nothing. Otherwise build the log mapping starting
    # from the extra metrics mapping itself (KNOWN ISSUE: the caller's mapping is mutated in place
    # when provided). Per dataset key: record none-reward ratio (0.0 when no rewards); if any reward
    # is none, warn and treat them as 0.0; record mean reward under eval/<key> (0.0 when empty); if
    # samples exist, merge sample-derived metrics under eval/<key>/; if truncated flags exist,
    # record their mean as the truncated ratio (divides by zero if the list is empty); if pass-rate
    # logging is on, merge pass@k metrics computed over the (none-replaced) rewards with the eval
    # group size, prefixed eval/<key>-. Then log the mapping at info level, add the eval step
    # (computed from rollout id) and send to the tracking backend keyed on the eval step. Returns the
    # logged mapping.
    if (x := args.custom_eval_rollout_log_function_path) is not None:
        custom_log_func = load_function(x)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        num_none = sum(1 for r in rewards if r is None)
        log_dict[f"eval/{key}-none_reward_ratio"] = num_none / len(rewards) if len(rewards) > 0 else 0.0
        if num_none:
            logger.warning(
                f"eval/{key}: {num_none}/{len(rewards)} samples have reward=None (likely errored/aborted trials); treating as 0.0 for metrics."
            )
            rewards = [0.0 if r is None else r for r in rewards]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards) if len(rewards) > 0 else 0.0
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(_compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    tracking.log(args, log_dict, step_key="eval/step")

    return log_dict


def log_eval_skip(rollout_id, args, reason: str):
    # [FUNCTION DOC] Record that an eval point was skipped: logs a warning and sends a mapping with
    # counter eval/skipped_<reason> equal to 1 and the eval step (from rollout id) to the tracking
    # backend keyed on the eval step. Returns nothing.
    """Log a skipped eval point at ``rollout_id`` so curve gaps are attributable."""
    log_dict = {
        f"eval/skipped_{reason}": 1,
        "eval/step": compute_rollout_step(args, rollout_id),
    }
    logger.warning(f"eval {rollout_id} skipped: {reason}")
    tracking.log(args, log_dict, step_key="eval/step")


def log_rollout_data(
    rollout_id, args, samples, rollout_extra_metrics, rollout_time, trainer_model_id: str | None = None
):
    # [FUNCTION DOC] Log training rollout metrics. A configured custom log function (called with
    # rollout id, args, samples, extra metrics and rollout time, without the trainer model id) can
    # take over by returning truthy. If debug rollout data is being loaded, returns without logging
    # (nothing was generated). Otherwise builds a mapping from a copy of the extra metrics plus
    # sample metrics prefixed rollout/, performance metrics prefixed perf/ (using the rollout wall
    # time in seconds), and, if pass-rate logging is on, pass@k prefixed passrate/. Logs it at info
    # level, namespaces the keys by trainer model id when given (also yielding the step key, with
    # step name rollout/step and the computed rollout step) and sends it to the tracking backend.
    if (x := args.custom_rollout_log_function_path) is not None:
        custom_log_func = load_function(x)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(_compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(_compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    if args.log_passrate:
        log_dict |= dict_add_prefix(
            _compute_passrate_from_samples(args, samples),
            "passrate/",
        )
    logger.info(f"perf {rollout_id}: {log_dict}")
    log_dict, step_key = namespace_metrics(
        log_dict,
        trainer_model_id=trainer_model_id,
        step_name="rollout/step",
        step=compute_rollout_step(args, rollout_id),
    )
    tracking.log(args, log_dict, step_key=step_key)


def _compute_metrics_from_samples(args, samples):
    # [FUNCTION DOC] HELPER: Assemble the per-batch sample metrics mapping (unprefixed keys): training
    # sample metrics; episode response length metrics; statistics of effective response lengths under
    # response_len/; zero-std, speculative decoding, prefix cache and reward-category metrics;
    # repetition fraction (mean of a per-response repetition test); truncated ratio (fraction with
    # truncated status). If any samples carry an oldest weight version: statistics of those under
    # weight_version/ plus the ratio of samples whose weight-version spans contain more than one
    # distinct version (denominator is all samples). TITO: collects non-null per-sample session
    # mismatch lists from sample metadata; if any, the session server version is v1 when the flag is
    # exactly true, else the flag's value, which must be v1 or v2 (assertion error otherwise); logs
    # the rate of samples with any mismatch under tito_session_mismatch_rate/<version> and the rate
    # with a mismatch of each of four types (special token count, special token type, non-assistant
    # text, assistant text) as sub-keys. In CI test mode asserts the rates for special token count
    # (against a configured threshold), special token type and non-assistant text (against zero) do
    # not exceed their thresholds; assistant text mismatches are tolerated. Returns the mapping.
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= _compute_training_sample_metrics(args, samples)
    log_dict |= _compute_episode_response_length_metrics(samples)
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()

    oldest_versions = [s.oldest_weight_version for s in samples if s.oldest_weight_version is not None]
    if oldest_versions:
        log_dict |= dict_add_prefix(compute_statistics(oldest_versions), "weight_version/")
        mixed = sum(1 for s in samples if len({span.version for span in s.all_weight_version_spans}) > 1)
        log_dict["weight_version/mixed_version_ratio"] = mixed / len(samples)

    tito_vals = [s.metadata.get("tito_session_mismatch") for s in samples]
    tito_vals = [v for v in tito_vals if v is not None]
    if tito_vals:
        session_server_version = "v1" if args.use_session_server is True else args.use_session_server
        assert session_server_version in ("v1", "v2"), "TITO metrics require session server v1 or v2"
        metric_prefix = f"tito_session_mismatch_rate/{session_server_version}"
        log_dict[metric_prefix] = np.mean([len(v) > 0 for v in tito_vals]).item()
        for mtype in ("special_token_count", "special_token_type", "non_assistant_text", "assistant_text"):
            log_dict[f"{metric_prefix}/{mtype}"] = np.mean(
                [any(m.get("type") == mtype for m in v) for v in tito_vals]
            ).item()
        if args.ci_test:
            strict_thresholds = {
                "special_token_count": args.ci_tito_special_token_count_threshold,
                "special_token_type": 0,
                "non_assistant_text": 0,
            }
            for strict_type, threshold in strict_thresholds.items():
                rate = log_dict.get(f"{metric_prefix}/{strict_type}", 0)
                assert (
                    rate <= threshold
                ), f"{metric_prefix}/{strict_type}={rate:.4f} exceeds {threshold} — this indicates a bug in the TITO algorithm or chat template. Please check your tito model and chat template."
            # assistant_text mismatch is non-critical: assistant tokens are inherited
            # from the pretokenized prefix and may differ from canonical tokenization.

    return log_dict


def _get_rollout_key(sample: Sample, position: int) -> tuple[str, int | None, int]:
    # [FUNCTION DOC] HELPER: Identity key of the original rollout a sample belongs to, as a triple
    # (kind, group index, id). Priority: sample rollout id (kind "rollout"), else sample index (kind
    # "sample"), else the sample's position in the batch (kind "position"). Samples sharing a key are
    # siblings from one rollout (e.g. after session compaction).
    if sample.rollout_id is not None:
        return ("rollout", sample.group_index, sample.rollout_id)
    if sample.index is not None:
        return ("sample", sample.group_index, sample.index)
    return ("position", sample.group_index, position)


def _compute_episode_response_length_metrics(samples: list[Sample]) -> dict[str, float]:
    # [FUNCTION DOC] HELPER: Per rollout key, sum the effective response length (zero for samples
    # flagged for removal) and separately the total response length (all samples). Returns an empty
    # mapping for no samples; otherwise statistics of the summed effective lengths prefixed
    # episode_response_length/ and the mean total length under episode_total_response_length/mean.
    """Aggregate trainable and total response tokens per original rollout.

    Session compaction can split one rollout into several training samples.
    Sibling samples share a rollout ID, so their lengths are summed before
    computing batch-level statistics. Effective lengths count only trainable
    tokens; total lengths count both masked and unmasked tokens in every sample.
    """
    effective_lengths_by_rollout: dict[tuple[str, int | None, int], int] = {}
    total_lengths_by_rollout: dict[tuple[str, int | None, int], int] = {}
    for position, sample in enumerate(samples):
        rollout_key = _get_rollout_key(sample, position)
        effective_response_length = 0 if sample.remove_sample else sample.effective_response_length
        effective_lengths_by_rollout[rollout_key] = (
            effective_lengths_by_rollout.get(rollout_key, 0) + effective_response_length
        )
        total_lengths_by_rollout[rollout_key] = total_lengths_by_rollout.get(rollout_key, 0) + sample.response_length

    if not effective_lengths_by_rollout:
        return {}

    log_dict = dict_add_prefix(
        compute_statistics(list(effective_lengths_by_rollout.values())),
        "episode_response_length/",
    )
    log_dict["episode_total_response_length/mean"] = np.mean(list(total_lengths_by_rollout.values())).item()
    return log_dict


def _compute_training_sample_metrics(args: Any, samples: list[Sample]) -> dict[str, float | int]:
    # [FUNCTION DOC] HELPER: Returns the sample count and the episode raw reward. Raw reward per
    # sample comes from the raw_reward entry in metadata if the first sample has one (decided once
    # from the first sample only), else from the sample's reward value under args. Only numeric
    # rewards count; they are grouped by rollout key, averaged within each rollout, then averaged
    # across rollouts (0.0 when none).
    """Count training rows and average raw reward with equal weight per rollout.

    Session compaction can turn one rollout into several training samples. The
    sample count includes every resulting row, while the reward first averages
    sibling rows that share a rollout ID so long rollouts do not receive more
    metric weight merely because they produced more samples.
    """
    rewards_by_rollout: dict[tuple[str, int | None, int], list[float]] = {}
    use_metadata_reward = bool(samples and samples[0].metadata and "raw_reward" in samples[0].metadata)
    for position, sample in enumerate(samples):
        rollout_key = _get_rollout_key(sample, position)

        raw_reward = sample.metadata["raw_reward"] if use_metadata_reward else sample.get_reward_value(args)
        if isinstance(raw_reward, Number):
            rewards_by_rollout.setdefault(rollout_key, []).append(raw_reward)

    rollout_rewards = [sum(rewards) / len(rewards) for rewards in rewards_by_rollout.values()]
    return {
        "num_training_samples": len(samples),
        "episode_raw_reward": sum(rollout_rewards) / len(rollout_rewards) if rollout_rewards else 0.0,
    }


def _compute_perf_metrics_from_samples(args, samples, rollout_time):
    # [FUNCTION DOC] HELPER: Throughput metrics from the rollout wall time (seconds). Always records
    # rollout_time. If any sample has positive non-generation time, adds its statistics under
    # non_generation_time/. Then computes token throughput twice: over raw response lengths (no key
    # prefix) and over effective response lengths (key prefix "effective_"). Each pass records, when
    # rollout GPUs are configured, total tokens per GPU per second; the longest sample's tokens per
    # second; and, if any non-generation time is positive, the mean non-generation time of the
    # longest samples (ties averaged) and the longest sample's tokens per second after subtracting it
    # from the rollout time. Division by rollout time or by the remainder can fail if they are zero.
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        # [FUNCTION DOC] HELPER: Writes the throughput entries for one choice of lengths into the
        # enclosing mapping (mutation of the enclosing log mapping), as described by the enclosing
        # function. Returns early (after the first two entries) when all non-generation times are
        # zero.
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # [FUNCTION DOC] HELPER: Statistics of prompt groups whose responses all received the same reward.
    # Returns empty for the ppo advantage estimator. Groups samples by group index; a group is zero
    # std when it is empty or all reward values equal the first. For each such group takes its common
    # reward, rounds to one decimal and formats as text, then counts groups per text value under
    # zero_std/count_<reward>. If there are any groups, also adds the fraction of all groups whose
    # uniform reward is exactly 0 (all_zero_percentage) or exactly 1 (all_one_percentage); these are
    # fractions in [0, 1] despite the name "percentage".
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        # [FUNCTION DOC] HELPER: True if the group is empty or all reward values equal the first.
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    uniform_rewards = [g[0].get_reward_value(args) for g in interesting_sample_groups]
    interesting_rewards = [str(round(reward, 1)) for reward in uniform_rewards]

    counts = {reward: len(items) for reward, items in group_by(interesting_rewards).items()}
    log_dict = {f"zero_std/count_{reward}": count for reward, count in counts.items()}

    # Percentages over total groups, so "too hard" (all-0) and "too easy"
    # (all-1) rates are comparable across runs without needing to know the
    # rollout batch size.
    total_groups = len(all_sample_groups)
    if total_groups > 0:
        log_dict["zero_std/all_zero_percentage"] = sum(1 for reward in uniform_rewards if reward == 0) / total_groups
        log_dict["zero_std/all_one_percentage"] = sum(1 for reward in uniform_rewards if reward == 1) / total_groups

    return log_dict


def _compute_spec_metrics(args, all_samples: list[Sample]):
    # [FUNCTION DOC] HELPER: Speculative decoding metrics; empty when no speculative algorithm is
    # configured. Collects speculative info per sample; samples whose metadata holds a session
    # rollout-metrics carrier contribute instead once per distinct session id (the last carrier seen
    # per session wins), using the spec info embedded in the carrier. Sums correct drafts, proposed
    # drafts, verify count and completion tokens. Returns accept rate (correct / proposed, 0.0 if
    # none proposed) and accept length (completion tokens / verify count, 0.0 if none).
    if args.sglang_speculative_algorithm is None:
        return {}
    carriers = {}
    spec_infos = []
    for sample in all_samples:
        carrier = sample.metadata.get(SESSION_ROLLOUT_METRICS_KEY)
        if carrier is None:
            spec_infos.append(sample.spec_info)
        else:
            carriers[carrier["session_id"]] = carrier
    spec_infos.extend(Sample.SpecInfo.from_dict(carrier["metrics"]["spec_info"]) for carrier in carriers.values())
    num_correct_drafts = sum(info.spec_num_correct_drafts for info in spec_infos)
    num_proposed_drafts = sum(info.spec_num_proposed_drafts for info in spec_infos)
    spec_verify_ct = sum(info.spec_verify_ct for info in spec_infos)
    completion_tokens = sum(info.completion_tokens for info in spec_infos)
    return {
        "spec_accept_rate": num_correct_drafts / num_proposed_drafts if num_proposed_drafts > 0 else 0.0,
        "spec_accept_length": completion_tokens / spec_verify_ct if spec_verify_ct > 0 else 0.0,
    }


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    # [FUNCTION DOC] HELPER: Returns prefix cache hit rate (total cached tokens / total prompt
    # tokens, 0.0 if no prompt tokens) and average cached tokens per sample (divides by zero for an
    # empty batch). The args parameter is unused.
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    # [FUNCTION DOC] HELPER: If a reward category key is configured, groups samples by the category
    # found under that key in each sample's reward mapping and returns the fraction of samples per
    # category under error_cat/<category>; otherwise empty. A reward that is not a mapping raises.
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}


def _compute_passrate_from_samples(args, all_samples: list[Sample]) -> dict[str, float]:
    # [FUNCTION DOC] HELPER: Pass@k metrics over complete groups. Empty if samples per prompt is 1 or
    # less. Groups by group index, keeps only groups with exactly that many samples (warning with the
    # count of excluded groups), returns empty if none remain, otherwise flattens the reward values
    # (group by group, in order) and delegates to the shared pass rate computation with that group
    # size.
    """Compute pass@k metrics from samples using group_index for correct grouping.

    Unlike the trainer-side log_passrate (which assumed a flat reward array with
    contiguous groups of n_samples_per_prompt), this groups samples by their
    group_index field and computes pass@k over complete groups only. This is
    robust to filtering that may remove individual samples from a group —
    incomplete groups are excluded from the estimate rather than skewing it
    or crashing the reshape.

    Called on the rollout side (before convert_samples_to_train_data), so
    normally all samples are present and every group is complete.
    """
    group_size = args.n_samples_per_prompt
    if group_size <= 1:
        return {}

    groups = group_by(all_samples, lambda s: s.group_index)
    completed_groups = [g for g in groups.values() if len(g) == group_size]
    if len(completed_groups) < len(groups):
        logger.warning(
            f"pass@k: excluding {len(groups) - len(completed_groups)}/{len(groups)} incomplete groups (fewer than n_samples_per_prompt={group_size} samples)."
        )
    if not completed_groups:
        return {}

    flat_rewards = [sample.get_reward_value(args) for group in completed_groups for sample in group]

    return compute_pass_rate(
        flat_rewards=flat_rewards,
        group_size=group_size,
    )
