# [MODULE DOC] Shared building blocks for sample group filters: the filter verdict record, a helper to
# iterate samples of possibly nested groups, a uniform caller for dynamic filters that tolerates legacy
# boolean results, and a metric gatherer that counts drop reasons and averages the raw reward of
# groups seen before dynamic filtering.
import argparse
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from miles.utils.types import Sample


@dataclass
class FilterOutput:
    # [CLASS DOC] Verdict of a group filter: whether to keep the group and an optional short reason
    # string (set when dropping; used as a metric key suffix).
    keep: bool
    reason: str | None = None


DynamicFilterOutput = FilterOutput


def iter_samples(group: list[Sample | list[Sample]]) -> Iterator[Sample]:
    # [FUNCTION DOC] Yield every sample of a group in order, flattening one level of nesting (an element
    # that is a list contributes its elements; deeper lists are not flattened).
    for sample in group:
        if isinstance(sample, list):
            yield from sample
        else:
            yield sample


def call_dynamic_filter(fn, args, samples: list[Sample | list[Sample]], **kwargs):
    # [FUNCTION DOC] Apply an optional dynamic filter. No filter (none) keeps the group. Otherwise calls
    # it with (args, samples, extra keyword arguments); a result that is not already a filter output
    # (legacy filters return a plain truthy/falsy value) is wrapped as the keep flag with no reason.
    # Returns a filter output. Exceptions from the filter propagate.
    if fn is None:
        return FilterOutput(keep=True)

    output = fn(args, samples, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, FilterOutput):
        output = FilterOutput(keep=output)

    return output


class MetricGatherer:
    # [CLASS DOC] Accumulates, between collections: a count of dropped groups per reason, and the sum
    # and count of raw rewards of all groups seen before dynamic filtering. Not thread safe; one
    # instance per buffer; the owner replaces the instance to reset it.

    def __init__(self):
        # [FUNCTION DOC] Start with an empty reason counter (default zero) and zero reward sum and count.
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self._unfiltered_reward_sum = 0.0
        self._unfiltered_reward_count = 0

    def on_group_before_dynamic_filter(self, args: argparse.Namespace, group: list) -> None:
        # [FUNCTION DOC] Add the reward of every sample in the group to the running sum and count,
        # skipping samples whose reward is missing, whose reward is a mapping while no reward key is
        # configured (ambiguous), or whose resolved reward value is missing. Values are converted to
        # floating point (may raise if non-numeric).
        for sample in _iter_group_samples(group):
            if sample.reward is None:
                continue
            if not args.reward_key and isinstance(sample.reward, dict):
                continue
            if (value := sample.get_reward_value(args)) is None:
                continue
            self._unfiltered_reward_sum += float(value)
            self._unfiltered_reward_count += 1

    def on_dynamic_filter_drop(self, reason: str | None):
        # [FUNCTION DOC] Count one dropped group under the reason; a missing or empty reason is not
        # counted.
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        # [FUNCTION DOC] Return a mapping of metrics: for each reason, rollout/dynamic_filter/drop_<reason>
        # with its count; and, if any reward was seen, rollout/raw_reward_unfiltered as the mean reward.
        # Does not reset the counters.
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
        if self._unfiltered_reward_count:
            metrics["rollout/raw_reward_unfiltered"] = self._unfiltered_reward_sum / self._unfiltered_reward_count
        return metrics


def _iter_group_samples(group: list) -> Iterator[Sample]:
    # [FUNCTION DOC] HELPER: Same behavior as the public sample iterator (flatten one level of nesting).
    # KNOWN ISSUE: duplicate of the public function above; one could replace the other.
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item
