# [MODULE DOC] Built-in filters over sample groups and weight-version statistics of groups. Filters
# return a filter output (keep flag and drop reason): reject groups with aborted samples, groups with
# missing rewards, groups with zero reward standard deviation; a composed pre-put filter chain; and
# helpers computing per-group weight-version statistics and staleness relative to a current version.
from argparse import Namespace
from dataclasses import dataclass

import torch

from miles.rollout.filter_hub.base_types import FilterOutput, call_dynamic_filter, iter_samples
from miles.utils.types import Sample

Group = list[Sample | list[Sample]]


@dataclass(frozen=True)
class GroupWeightVersionStats:
    # [CLASS DOC] Frozen summary of the weight versions behind a group's generated tokens: number of
    # samples; number of samples with at least one numeric-versioned span; number of tokens covered by
    # numeric-versioned spans; sum over those tokens of their version; and the smallest and largest
    # version seen (none when no numeric version). Non-numeric version labels are ignored entirely.
    sample_count: int
    versioned_sample_count: int
    versioned_token_count: int
    token_version_sum: int
    oldest_version: int | None
    newest_version: int | None

    def oldest_lag(self, current_version: int | None) -> int | None:
        # [FUNCTION DOC] Current version minus the oldest version in the group; none if either is
        # unknown.
        if self.oldest_version is None or current_version is None:
            return None
        return current_version - self.oldest_version

    def newest_lag(self, current_version: int | None) -> int | None:
        # [FUNCTION DOC] Current version minus the newest version in the group; none if either is
        # unknown.
        if self.newest_version is None or current_version is None:
            return None
        return current_version - self.newest_version

    def token_weighted_lag(self, current_version: int | None) -> float | None:
        # [FUNCTION DOC] Current version minus the token-weighted mean version (version sum divided by
        # versioned token count); none if the current version is unknown or no tokens are versioned.
        if current_version is None or self.versioned_token_count == 0:
            return None
        return current_version - self.token_version_sum / self.versioned_token_count


def group_weight_version_stats(group: Group) -> GroupWeightVersionStats:
    # [FUNCTION DOC] Scan all samples of the group (nesting flattened) and their weight-version spans.
    # Spans whose version text is not all digits are skipped. For each numeric span: tokens = span end
    # minus span start (absolute positions); accumulate token count and version times tokens; track
    # min and max version; mark the sample versioned. Returns the statistics record.
    sample_count = 0
    versioned_sample_count = 0
    versioned_token_count = 0
    token_version_sum = 0
    oldest_version = None
    newest_version = None

    for sample in iter_samples(group):
        sample_count += 1
        sample_has_version = False
        for span in sample.all_weight_version_spans:
            if not str(span.version).isdigit():
                continue
            version = int(span.version)
            token_count = span.abs_end - span.abs_start
            versioned_token_count += token_count
            token_version_sum += version * token_count
            oldest_version = version if oldest_version is None else min(oldest_version, version)
            newest_version = version if newest_version is None else max(newest_version, version)
            sample_has_version = True
        versioned_sample_count += int(sample_has_version)

    return GroupWeightVersionStats(
        sample_count=sample_count,
        versioned_sample_count=versioned_sample_count,
        versioned_token_count=versioned_token_count,
        token_version_sum=token_version_sum,
        oldest_version=oldest_version,
        newest_version=newest_version,
    )


def apply_preput_filters(args: Namespace, dynamic_filter, samples: Group, **kwargs) -> FilterOutput:
    # [FUNCTION DOC] Chain of filters before a group is stored: first the aborted filter, then the
    # missing reward filter, then the optional dynamic filter (via the uniform caller); the first
    # verdict that drops is returned, else the dynamic filter's verdict. Extra keyword arguments are
    # forwarded to all three.
    output = apply_aborted_filter(args, samples, **kwargs)
    if not output.keep:
        return output

    output = apply_missing_reward_filter(args, samples, **kwargs)
    if not output.keep:
        return output

    return call_dynamic_filter(dynamic_filter, args, samples, **kwargs)


def apply_aborted_filter(args: Namespace, samples: Group, **kwargs) -> FilterOutput:
    # [FUNCTION DOC] Drop the whole group (reason "group_has_aborted") if any sample has the aborted
    # status; else keep. Args and extra keywords are unused.
    """Reject entire group if any sample was aborted (e.g. env timeout, Docker crash)."""
    if any(sample.status == Sample.Status.ABORTED for sample in iter_samples(samples)):
        return FilterOutput(keep=False, reason="group_has_aborted")
    return FilterOutput(keep=True)


def apply_missing_reward_filter(args: Namespace, samples: Group, **kwargs) -> FilterOutput:
    # [FUNCTION DOC] Drop the group (reason "group_has_missing_reward") if any sample has no reward or
    # its reward value resolved under args (reward key selection) is missing; else keep.
    if any(sample.reward is None or sample.get_reward_value(args) is None for sample in iter_samples(samples)):
        return FilterOutput(keep=False, reason="group_has_missing_reward")
    return FilterOutput(keep=True)


def apply_reward_nonzero_std_filter(args, samples: list[Sample | list[Sample]], **kwargs):
    # [FUNCTION DOC] Keep the group only if the sample standard deviation of its reward values (64-bit
    # float) exceeds 1e-8. When dropping, the reason is "zero_std_<first reward rounded to one
    # decimal>". Rewards must all be numeric. A single-sample group has an undefined (not a number)
    # deviation, so the comparison is false and it is dropped. KNOWN ISSUE: the keep field is a
    # one-element tensor rather than a plain boolean.
    rewards = [sample.get_reward_value(args) for sample in iter_samples(samples)]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-8
    return FilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def group_staleness(group: Group, current_version: int | None) -> int | None:
    # [FUNCTION DOC] Staleness of a group: current version minus the oldest numeric weight version
    # among its tokens; none if unknown.
    return group_weight_version_stats(group).oldest_lag(current_version)
