# [MODULE DOC] Tests of the metric gatherer from the rollout filter hub, specifically the
# unfiltered raw reward metric: a mean of rewards over every sample offered before the dynamic
# filter, whether or not the group is later kept or dropped. Samples are minimal sample records
# built with a given reward, and the arguments are a bare namespace holding only the reward key.
from argparse import Namespace

from miles.rollout.filter_hub.base_types import MetricGatherer
from miles.utils.types import Sample


class TestMetricGathererUnfilteredRawReward:
    # [CLASS DOC] Tests of how the gatherer computes the unfiltered raw reward metric from groups
    # offered before filtering.
    def test_the_mean_covers_kept_and_dropped_groups_alike(self):
        # [FUNCTION DOC] Offers one group of rewards 1.0 and one of rewards 0.0, reports a dynamic
        # filter drop (reason zero_std_0), and asserts the collected unfiltered raw reward mean is
        # 0.5, so the drop does not change the mean.
        """The gatherer sees every group before the filter, so the mean must not depend on the keep decision."""
        gatherer = MetricGatherer()

        gatherer.on_group_before_dynamic_filter(_args(), [_sample(reward=1.0), _sample(reward=1.0)])
        gatherer.on_group_before_dynamic_filter(_args(), [_sample(reward=0.0), _sample(reward=0.0)])
        gatherer.on_dynamic_filter_drop(reason="zero_std_0")

        assert gatherer.collect()["rollout/raw_reward_unfiltered"] == 0.5

    def test_a_nested_group_contributes_every_inner_sample(self):
        # [FUNCTION DOC] Offers a group given as a list of lists (rewards 1.0 | 0.0 and 0.5) and
        # asserts the mean is taken over the three inner samples (0.5), flattening the nesting.
        """Multi-turn groups arrive as lists of lists; counting outer items would weight turns unevenly."""
        gatherer = MetricGatherer()

        gatherer.on_group_before_dynamic_filter(
            _args(), [[_sample(reward=1.0)], [_sample(reward=0.0), _sample(reward=0.5)]]
        )

        assert gatherer.collect()["rollout/raw_reward_unfiltered"] == 0.5

    def test_a_dict_reward_is_read_through_the_reward_key(self):
        # [FUNCTION DOC] Offers a sample whose reward is a mapping, with the reward key argument
        # naming one of its entries, and asserts the metric equals that entry's value (0.25).
        """Custom generate functions store dict rewards; the metric must follow --reward-key like raw_reward does."""
        gatherer = MetricGatherer()

        gatherer.on_group_before_dynamic_filter(
            _args(reward_key="reward_value"), [_sample(reward={"reward_value": 0.25, "outcome": "x"})]
        )

        assert gatherer.collect()["rollout/raw_reward_unfiltered"] == 0.25

    def test_an_unkeyed_structured_reward_does_not_report_a_raw_reward(self):
        # [FUNCTION DOC] Offers a sample with a nested mapping reward while no reward key is set
        # and asserts the collected metrics omit the unfiltered raw reward entry.
        """Structured custom-RM payloads have no unambiguous scalar value to average."""
        gatherer = MetricGatherer()

        gatherer.on_group_before_dynamic_filter(
            _args(), [_sample(reward={"teacher": {"reward": 0.25}, "student": {"reward": 0.5}})]
        )

        assert "rollout/raw_reward_unfiltered" not in gatherer.collect()

    def test_no_offered_group_reports_no_metric(self):
        # [FUNCTION DOC] Collects from a fresh gatherer and asserts there is no unfiltered raw
        # reward entry.
        """A window without generation must yield no point rather than a fabricated zero."""
        gatherer = MetricGatherer()

        assert "rollout/raw_reward_unfiltered" not in gatherer.collect()


def _args(reward_key: str | None = None) -> Namespace:
    # [FUNCTION DOC] HELPER: Returns an arguments namespace holding only the reward key (default
    # none).
    return Namespace(reward_key=reward_key)


def _sample(reward: float | dict | None) -> Sample:
    # [FUNCTION DOC] HELPER: Returns a minimal sample (group and sample index zero, prompt "p",
    # label "l") carrying the given reward, which may be a number, a mapping or none.
    return Sample(group_index=0, index=0, prompt="p", label="l", reward=reward)


class TestMetricGathererUnscoredSamples:
    # [CLASS DOC] Tests of the gatherer's handling of samples that have no reward yet.
    def test_an_unscored_sample_is_left_out_of_the_mean_instead_of_crashing_it(self):
        # [FUNCTION DOC] Offers one sample with reward 1.0 and one with no reward and asserts the
        # metric is 1.0 (the unscored sample is excluded rather than raising).
        """A late-aborted group-RM group arrives COMPLETED with reward None; it carries no score to average."""
        gatherer = MetricGatherer()

        gatherer.on_group_before_dynamic_filter(_args(), [_sample(reward=1.0), _sample(reward=None)])

        assert gatherer.collect()["rollout/raw_reward_unfiltered"] == 1.0
