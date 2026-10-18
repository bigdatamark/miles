# [MODULE DOC]
# Unit tests for the rollout metrics module: per-episode response length statistics, training
# sample counts and episode rewards, zero-std group metrics, speculative decoding metrics (plain
# and session-server deduplicated), tito session mismatch rates (including ci-test strictness and
# thresholds), pass-at-k rates, weight version statistics, and the logging entry points for rollout
# and eval data. Samples and arguments are built with the shared conftest helpers; logging is
# observed by monkeypatching the tracking log function. No external services are used.
from __future__ import annotations

import numpy as np
import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sample, make_samples_grouped

from miles.ray.rollout.metrics import (
    _compute_episode_response_length_metrics,
    _compute_metrics_from_samples,
    _compute_passrate_from_samples,
    _compute_spec_metrics,
    _compute_training_sample_metrics,
    _compute_zero_std_metrics,
    log_eval_rollout_data,
    log_rollout_data,
)
from miles.rollout.session.v2.metrics import SESSION_ROLLOUT_METRICS_KEY
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


class TestEpisodeResponseLengthMetrics:
    # [CLASS DOC] Tests the episode response length metrics, where samples sharing a prompt group
    # and rollout id (compacted siblings) form one episode.
    def test_compacted_siblings_are_summed_before_computing_statistics(self):
        # [FUNCTION DOC] Four samples in three episodes (two siblings in group 0 rollout 10, one in
        # group 0 rollout 11, one in group 1 rollout 10). Asserts the exact metrics: effective
        # length (sum of loss mask ones per episode: 5, 4, 6) gives mean 5, median 5, max 6, min 4,
        # and total length (sum of response lengths: 12, 4, 8) gives mean 8.
        samples = [
            make_sample(group_index=0, index=0, rollout_id=10, response_length=5, loss_mask=[1, 1, 0, 0, 0]),
            make_sample(group_index=0, index=0, rollout_id=10, response_length=7, loss_mask=[1, 1, 1, 0, 0, 0, 0]),
            make_sample(group_index=0, index=1, rollout_id=11, response_length=4, loss_mask=[1, 1, 1, 1]),
            make_sample(group_index=1, index=2, rollout_id=10, response_length=8, loss_mask=[1, 1, 1, 1, 1, 1, 0, 0]),
        ]

        out = _compute_episode_response_length_metrics(samples)

        assert out == {
            "episode_response_length/mean": pytest.approx(5.0),
            "episode_response_length/median": pytest.approx(5.0),
            "episode_response_length/max": pytest.approx(6.0),
            "episode_response_length/min": pytest.approx(4.0),
            "episode_total_response_length/mean": pytest.approx(8.0),
        }

    def test_single_sample_rollouts_match_sample_level_statistics(self):
        # [FUNCTION DOC] Three samples each in its own episode; the general metrics computation
        # (with ppo) must give episode length mean, median, max and min equal to the per-sample
        # response length statistics of the same names.
        samples = [
            make_sample(index=0, rollout_id=10, response_length=5, loss_mask=[1, 1, 0, 0, 0]),
            make_sample(index=1, rollout_id=11, response_length=7, loss_mask=[1, 1, 1, 0, 0, 0, 0]),
            make_sample(index=2, rollout_id=12, response_length=4, loss_mask=[1, 1, 1, 1]),
        ]

        out = _compute_metrics_from_samples(make_args(advantage_estimator="ppo"), samples)

        for statistic in ("mean", "median", "max", "min"):
            assert out[f"episode_response_length/{statistic}"] == out[f"response_len/{statistic}"]

    def test_empty_samples_emit_no_episode_length_metrics(self):
        # [FUNCTION DOC] An empty sample list yields an empty metrics mapping.
        assert _compute_episode_response_length_metrics([]) == {}

    def test_total_length_counts_masked_and_unmasked_tokens_in_every_sample(self):
        # [FUNCTION DOC] Two siblings (lengths 5 and 7, masks with 2 and 3 ones) and one single
        # episode (length 4, 4 ones). Asserts episode effective length mean 4.5 (5 and 4) and total
        # length mean 8 (12 and 4).
        samples = [
            make_sample(index=0, rollout_id=10, response_length=5, loss_mask=[1, 1, 0, 0, 0]),
            make_sample(index=0, rollout_id=10, response_length=7, loss_mask=[1, 1, 1, 0, 0, 0, 0]),
            make_sample(index=1, rollout_id=11, response_length=4, loss_mask=[1, 1, 1, 1]),
        ]

        out = _compute_episode_response_length_metrics(samples)

        assert out["episode_response_length/mean"] == pytest.approx(4.5)
        assert out["episode_total_response_length/mean"] == pytest.approx(8.0)

    def test_removed_sample_has_zero_effective_length_but_keeps_total_length(self):
        # [FUNCTION DOC] One sample flagged removed with length 5 and an all-ones mask: effective
        # episode length mean is 0 while total length mean is 5.
        sample = make_sample(
            index=0,
            rollout_id=10,
            response_length=5,
            loss_mask=[1, 1, 1, 1, 1],
            remove_sample=True,
        )

        out = _compute_episode_response_length_metrics([sample])

        assert out["episode_response_length/mean"] == pytest.approx(0.0)
        assert out["episode_total_response_length/mean"] == pytest.approx(5.0)


class TestTrainingSampleMetrics:
    # [CLASS DOC] Tests the training sample metrics: the number of samples and the mean episode
    # raw reward, where each episode's reward is the mean over its sibling samples.
    def test_compacted_rollouts_are_counted_as_samples_but_rewarded_as_episodes(self):
        # [FUNCTION DOC] Three siblings with reward 1 in one episode plus a second episode with
        # reward 0: sample count is 4 and the episode reward mean is 0.5.
        args = make_args(reward_key=None)
        samples = [
            make_sample(index=0, rollout_id=10, reward=1.0),
            make_sample(index=0, rollout_id=10, reward=1.0),
            make_sample(index=0, rollout_id=10, reward=1.0),
            make_sample(index=1, rollout_id=11, reward=0.0),
        ]

        out = _compute_training_sample_metrics(args, samples)

        assert out["num_training_samples"] == 4
        assert out["episode_raw_reward"] == pytest.approx(0.5)

    def test_different_sibling_rewards_are_averaged_within_rollout_first(self):
        # [FUNCTION DOC] Siblings with rewards 0 and 1 form an episode of mean 0.5, another episode
        # has reward 1; the episode reward mean is 0.75.
        args = make_args(reward_key=None)
        samples = [
            make_sample(index=0, rollout_id=10, reward=0.0),
            make_sample(index=0, rollout_id=10, reward=1.0),
            make_sample(index=1, rollout_id=11, reward=1.0),
        ]

        out = _compute_training_sample_metrics(args, samples)

        assert out["episode_raw_reward"] == pytest.approx(0.75)

    def test_rollout_ids_are_scoped_by_prompt_group(self):
        # [FUNCTION DOC] The same rollout id 10 appears in group 0 (two samples, reward 1) and in
        # group 1 (reward 0); they count as two different episodes, so the reward mean is 0.5.
        args = make_args(reward_key=None)
        samples = [
            make_sample(group_index=0, index=0, rollout_id=10, reward=1.0),
            make_sample(group_index=0, index=0, rollout_id=10, reward=1.0),
            make_sample(group_index=1, index=1, rollout_id=10, reward=0.0),
        ]

        out = _compute_training_sample_metrics(args, samples)

        assert out["episode_raw_reward"] == pytest.approx(0.5)

    def test_metadata_raw_reward_and_fallback_identities(self):
        # [FUNCTION DOC] Two samples without a rollout id (one with index 5, one with no index) and
        # reward 0 but metadata raw rewards 1.0 and 0.0. The result must be exactly a sample count
        # of 2 and an episode reward mean of 0.5: the metadata raw reward is preferred over the
        # reward field, and samples without a rollout id are each their own episode.
        args = make_args(reward_key=None)
        samples = [
            make_sample(index=5, reward=0.0),
            make_sample(index=None, reward=0.0),
        ]
        samples[0].metadata = {"raw_reward": 1.0}
        samples[1].metadata = {"raw_reward": 0.0}

        out = _compute_training_sample_metrics(args, samples)

        assert out == {"num_training_samples": 2, "episode_raw_reward": pytest.approx(0.5)}

    def test_empty_samples(self):
        # [FUNCTION DOC] An empty sample list gives a sample count of 0 and an episode reward of 0.0.
        assert _compute_training_sample_metrics(make_args(), []) == {
            "num_training_samples": 0,
            "episode_raw_reward": 0.0,
        }


class TestComputeZeroStdMetrics:
    # [CLASS DOC] Tests the zero-std metrics for group-relative advantage estimators: groups whose
    # rewards are all identical are counted per rounded reward bucket, and the share of all-zero
    # and all-one groups is reported over all groups. Returns nothing for other estimators.
    @pytest.mark.parametrize("reward_type", [int, float, np.int32, np.int64, np.float32, np.float64])
    def test_binary_percentages_do_not_depend_on_reward_type(self, reward_type):
        # [FUNCTION DOC] Parametrized over Python and numpy integer and float reward types. Three
        # groups of two with rewards (0,0), (1,1), (0,1): asserts one zero-std group in the zero
        # bucket and one in the one bucket, and one third each for the all-zero and all-one
        # percentages.
        args = make_args(advantage_estimator="grpo", reward_key=None)
        zero, one = reward_type(0), reward_type(1)
        samples = make_samples_grouped(3, 2, rewards=[zero, zero, one, one, zero, one])

        out = _compute_zero_std_metrics(args, samples)

        assert out[f"zero_std/count_{round(zero, 1)}"] == 1
        assert out[f"zero_std/count_{round(one, 1)}"] == 1
        assert out["zero_std/all_zero_percentage"] == pytest.approx(1 / 3)
        assert out["zero_std/all_one_percentage"] == pytest.approx(1 / 3)

    def test_mixed_numeric_types_and_unequal_group_sizes_count_groups_equally(self):
        # [FUNCTION DOC] Groups of sizes 1 (reward 0), 3 (rewards 1 in int, float and numpy int
        # forms) and 2 (0.0 and 1.0): the first two are uniform, the last is mixed, so the
        # all-zero and all-one percentages are one third each (groups, not samples, are counted).
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = [
            make_sample(group_index=0, reward=0),
            make_sample(group_index=1, reward=1),
            make_sample(group_index=1, reward=1.0),
            make_sample(group_index=1, reward=np.int64(1)),
            make_sample(group_index=2, reward=0.0),
            make_sample(group_index=2, reward=1.0),
        ]

        out = _compute_zero_std_metrics(args, samples)

        assert out["zero_std/all_zero_percentage"] == pytest.approx(1 / 3)
        assert out["zero_std/all_one_percentage"] == pytest.approx(1 / 3)

    @pytest.mark.parametrize("reward", [-0.04, 0.04, 0.96, 1.04])
    def test_rounded_fractional_rewards_do_not_count_as_binary(self, reward):
        # [FUNCTION DOC] Parametrized over rewards -0.04, 0.04, 0.96 and 1.04, one group of two
        # identical rewards. The group is counted in the bucket of the reward rounded to one
        # decimal, but neither the all-zero nor the all-one percentage counts it (they stay 0).
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(1, 2, rewards=[reward, reward])

        out = _compute_zero_std_metrics(args, samples)

        assert out[f"zero_std/count_{round(reward, 1)}"] == 1
        assert out["zero_std/all_zero_percentage"] == 0.0
        assert out["zero_std/all_one_percentage"] == 0.0

    def test_negative_zero_counts_as_zero(self):
        # [FUNCTION DOC] One group with rewards -0.0 and 0.0: bucket key is count_-0.0 with a count
        # of 1, all-zero percentage 1.0 and all-one percentage 0.0.
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(1, 2, rewards=[-0.0, 0.0])

        out = _compute_zero_std_metrics(args, samples)

        assert out["zero_std/count_-0.0"] == 1
        assert out["zero_std/all_zero_percentage"] == 1.0
        assert out["zero_std/all_one_percentage"] == 0.0

    def test_returns_empty_for_ppo_regardless_of_reward_distribution(self):
        # [FUNCTION DOC] With the ppo estimator and uniform rewards the result is an empty mapping.
        args = make_args(advantage_estimator="ppo")
        out = _compute_zero_std_metrics(args, make_samples_grouped(2, 4, rewards=[1.0] * 8))
        assert out == {}

    def test_grpo_mixed_rewards_yield_zero_percentages_and_no_buckets(self):
        # [FUNCTION DOC] Two groups of four with varied rewards: the result is exactly the two
        # percentage keys, both 0.0, and no bucket count keys.
        """Happy path: every group has reward variation → no group is zero-std →
        no bucket counts; the all_zero/all_one percentages are 0."""
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[0.0, 0.5, 1.0, 0.7, 0.2, 0.8, 0.3, 0.6])
        out = _compute_zero_std_metrics(args, samples)
        assert out == {"zero_std/all_zero_percentage": 0.0, "zero_std/all_one_percentage": 0.0}

    def test_grpo_zero_std_groups_produce_bucket_counts_and_percentages(self):
        # [FUNCTION DOC] Three groups of four: all ones, all zeros, mixed. Asserts one count in the
        # 1.0 bucket, one in the 0.0 bucket, and one third for each percentage.
        """1 group all-1, 1 group all-0, 1 group mixed → bucket counts plus the
        all_zero/all_one percentages over total groups."""
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(3, 4, rewards=[1.0] * 4 + [0.0] * 4 + [0.0, 1.0, 0.0, 1.0])
        out = _compute_zero_std_metrics(args, samples)
        assert out["zero_std/count_1.0"] == 1
        assert out["zero_std/count_0.0"] == 1
        assert out["zero_std/all_zero_percentage"] == pytest.approx(1 / 3)
        assert out["zero_std/all_one_percentage"] == pytest.approx(1 / 3)

    def test_grpo_uniform_non_binary_reward_gets_its_own_bucket(self):
        # [FUNCTION DOC] Two groups all at reward 0.5: the 0.5 bucket count is 2 and both
        # percentages are 0.0 (they only count rewards of exactly zero or one).
        """Every group zero-std at reward=0.5 → bucket count_0.5=2, but
        all_zero/all_one percentages stay 0 because they only count 0.0 and 1.0."""
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[0.5] * 8)
        out = _compute_zero_std_metrics(args, samples)
        assert out["zero_std/count_0.5"] == 2
        assert out["zero_std/all_zero_percentage"] == 0.0
        assert out["zero_std/all_one_percentage"] == 0.0

    def test_empty_samples_does_not_crash(self):
        # [FUNCTION DOC] An empty sample list with the grpo estimator must not raise and must emit
        # neither percentage key (the computation only emits them when there is at least one group).
        args = make_args(advantage_estimator="grpo", reward_key=None)
        out = _compute_zero_std_metrics(args, [])
        # No groups → no all_zero/all_one keys (the function guards on total_groups>0).
        assert "zero_std/all_zero_percentage" not in out
        assert "zero_std/all_one_percentage" not in out


class TestComputeSpecMetrics:
    # [CLASS DOC] Tests the speculative decoding metrics (accept rate and accept length), computed
    # from counters summed over samples, with session-server carriers deduplicated by session id.
    def test_aggregates_sglang_counters_before_computing_ratios(self):
        # [FUNCTION DOC] Two samples with spec info (correct 1 and 9, proposed 2 and 10, verify
        # count 1 and 9, completion tokens 2 and 27). Asserts the accept rate is the summed correct
        # over summed proposed (10/12) and the accept length is the summed completion tokens over
        # the summed verify count (29/10), i.e. ratios of sums, not means of ratios.
        args = make_args(sglang_speculative_algorithm="EAGLE")
        samples = make_samples_grouped(1, 2)
        samples[0].spec_info = Sample.SpecInfo(
            spec_num_correct_drafts=1,
            spec_num_proposed_drafts=2,
            spec_verify_ct=1,
            completion_tokens=2,
        )
        samples[1].spec_info = Sample.SpecInfo(
            spec_num_correct_drafts=9,
            spec_num_proposed_drafts=10,
            spec_verify_ct=9,
            completion_tokens=27,
        )

        out = _compute_spec_metrics(args, samples)

        assert out == {
            "spec_accept_rate": pytest.approx(10 / 12),
            "spec_accept_length": pytest.approx(29 / 10),
        }

    @staticmethod
    def _member(session_id, metrics, *, group_index=0, rollout_id=0):
        # [FUNCTION DOC] HELPER: Builds a sample for the given group index, with index and rollout
        # id both set to the given rollout id, whose metadata holds the session rollout metrics
        # entry (a session id and a metrics mapping). Returns the sample.
        sample = Sample(group_index=group_index, index=rollout_id, rollout_id=rollout_id)
        sample.metadata[SESSION_ROLLOUT_METRICS_KEY] = {
            "session_id": session_id,
            "metrics": metrics,
        }
        return sample

    @staticmethod
    def _spec_info(correct, proposed, verify, completion):
        # [FUNCTION DOC] HELPER: Builds a session metrics mapping with a spec info entry holding the
        # correct drafts, proposed drafts, verify count and completion token count.
        return {
            "spec_info": {
                "spec_num_correct_drafts": correct,
                "spec_num_proposed_drafts": proposed,
                "spec_verify_ct": verify,
                "completion_tokens": completion,
            }
        }

    def test_v2_deduplicates_session_carriers_and_includes_ordinary_samples(self):
        # [FUNCTION DOC] With session server v2: two carrier samples share session sid-1 (metrics:
        # correct 2, proposed 4, verify 2, completion 6) and have misleading own spec info of 100s,
        # plus one ordinary sample (3, 5, 2, 5). Asserts the session counts once, from its metrics
        # entry and not the sample field, giving accept rate 5/9 and accept length 11/4; and that
        # the ordinary sample alone gives 3/5 and 5/2.
        args = make_args(sglang_speculative_algorithm="EAGLE", use_session_server="v2")
        session_1_metrics = self._spec_info(2, 4, 2, 6)
        ordinary_sample = Sample(
            spec_info=Sample.SpecInfo(
                spec_num_correct_drafts=3,
                spec_num_proposed_drafts=5,
                spec_verify_ct=2,
                completion_tokens=5,
            )
        )
        samples = [
            self._member("sid-1", session_1_metrics, rollout_id=10),
            self._member("sid-1", session_1_metrics, rollout_id=10),
            ordinary_sample,
        ]
        for sample in samples[:2]:
            sample.spec_info = Sample.SpecInfo(
                spec_num_correct_drafts=100,
                spec_num_proposed_drafts=100,
                spec_verify_ct=1,
                completion_tokens=100,
            )

        out = _compute_spec_metrics(args, samples)

        assert out == {
            "spec_accept_rate": pytest.approx(5 / 9),
            "spec_accept_length": pytest.approx(11 / 4),
        }
        assert _compute_spec_metrics(args, [ordinary_sample]) == {
            "spec_accept_rate": pytest.approx(3 / 5),
            "spec_accept_length": pytest.approx(5 / 2),
        }


class TestTitoMismatchMetrics:
    # [CLASS DOC] Tests the tito session mismatch rate metrics derived from per-sample metadata,
    # keyed by session server version, and the ci-test strictness: some mismatch types fail an
    # assertion when their rate is above zero (special token count has a configurable threshold).
    def test_no_tito_metadata_emits_no_tito_keys(self):
        # [FUNCTION DOC] Samples without mismatch metadata produce no key starting with the tito
        # session mismatch rate prefix.
        args = make_args(advantage_estimator="ppo", ci_test=False, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        out = _compute_metrics_from_samples(args, samples)
        assert not any(key.startswith("tito_session_mismatch_rate") for key in out)

    @pytest.mark.parametrize(
        ("configured_version", "metric_version"),
        [(True, "v1"), ("v1", "v1"), ("v2", "v2")],
    )
    def test_clean_tito_metadata_yields_zero_rates_per_mismatch_type(self, configured_version, metric_version):
        # [FUNCTION DOC] Parametrized over (configured session server value, metric version):
        # true maps to v1, v1 to v1, v2 to v2. Every sample has an empty mismatch list. Asserts the
        # set of tito keys is exactly the overall rate plus four per-type rates (special token
        # count, special token type, non assistant text, assistant text) under the versioned prefix,
        # and all are 0.0.
        args = make_args(
            advantage_estimator="ppo",
            ci_test=False,
            log_passrate=False,
            use_session_server=configured_version,
        )
        samples = make_samples_grouped(1, 4)
        for s in samples:
            s.metadata = {"tito_session_mismatch": []}
        out = _compute_metrics_from_samples(args, samples)
        metric_prefix = f"tito_session_mismatch_rate/{metric_version}"
        tito_keys = {
            metric_prefix,
            f"{metric_prefix}/special_token_count",
            f"{metric_prefix}/special_token_type",
            f"{metric_prefix}/non_assistant_text",
            f"{metric_prefix}/assistant_text",
        }
        assert {key for key in out if key.startswith("tito_session_mismatch_rate")} == tito_keys
        assert all(out[key] == 0.0 for key in tito_keys)

    def test_strict_mismatch_raises_under_ci_test(self):
        # [FUNCTION DOC] With ci test on and session server v1, one of four samples reports a
        # special token count mismatch. The metrics computation must raise an assertion error whose
        # message contains the v1 special token count rate of 0.2500.
        """Under ci_test=True, a non-zero rate on the strict mismatch types
        (special_token_count / special_token_type / non_assistant_text) must
        hard-fail — these signal a TITO algorithm or chat-template bug."""
        args = make_args(
            advantage_estimator="ppo",
            ci_test=True,
            log_passrate=False,
            use_session_server="v1",
        )
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {"tito_session_mismatch": [{"type": "special_token_count"}]}
        for s in samples[1:]:
            s.metadata = {"tito_session_mismatch": []}
        with pytest.raises(
            AssertionError,
            match=r"tito_session_mismatch_rate/v1/special_token_count=0\.2500",
        ):
            _compute_metrics_from_samples(args, samples)

    @pytest.mark.parametrize(
        ("mismatch_type", "threshold", "raises"),
        [
            ("special_token_count", 0.25, False),
            ("special_token_count", 0.2, True),
            ("special_token_type", 0.25, True),
        ],
    )
    def test_special_token_count_threshold_under_ci_test(self, mismatch_type, threshold, raises):
        # [FUNCTION DOC] Parametrized over (type, threshold, raises): special token count at 0.25
        # does not raise, at 0.2 raises, and special token type at 0.25 raises. With ci test on, v2,
        # and one mismatch among four samples (rate 0.25), checks an assertion error naming the v2
        # rate when raises is set; otherwise checks the logged rate equals 0.25. Shows the
        # threshold relaxes only the special token count type.
        """The special_token_count threshold only relaxes that type; the
        other strict types stay at 0."""
        args = make_args(
            advantage_estimator="ppo",
            ci_test=True,
            ci_tito_special_token_count_threshold=threshold,
            log_passrate=False,
            use_session_server="v2",
        )
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {"tito_session_mismatch": [{"type": mismatch_type}]}
        for s in samples[1:]:
            s.metadata = {"tito_session_mismatch": []}
        if raises:
            with pytest.raises(AssertionError, match=rf"tito_session_mismatch_rate/v2/{mismatch_type}=0\.2500"):
                _compute_metrics_from_samples(args, samples)
        else:
            out = _compute_metrics_from_samples(args, samples)
            assert out[f"tito_session_mismatch_rate/v2/{mismatch_type}"] == 0.25

    def test_assistant_text_mismatch_does_not_raise_under_ci_test(self):
        # [FUNCTION DOC] With ci test on and v2, one assistant text mismatch among four samples
        # does not raise; the v2 assistant text rate is 0.25 and no unversioned assistant text key
        # exists.
        """assistant_text mismatch is non-critical (tokens inherited from the
        pretokenized prefix) — even under ci_test, must not raise."""
        args = make_args(
            advantage_estimator="ppo",
            ci_test=True,
            log_passrate=False,
            use_session_server="v2",
        )
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {"tito_session_mismatch": [{"type": "assistant_text"}]}
        for s in samples[1:]:
            s.metadata = {"tito_session_mismatch": []}
        out = _compute_metrics_from_samples(args, samples)
        assert out["tito_session_mismatch_rate/v2/assistant_text"] == 0.25
        assert "tito_session_mismatch_rate/assistant_text" not in out

    def test_tito_metadata_requires_session_server_version(self):
        # [FUNCTION DOC] Samples carry mismatch metadata but the session server setting is unset:
        # the computation raises an assertion error saying a session server v1 or v2 is required.
        args = make_args(advantage_estimator="ppo", ci_test=False, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        for sample in samples:
            sample.metadata = {"tito_session_mismatch": []}

        with pytest.raises(AssertionError, match="session server v1 or v2"):
            _compute_metrics_from_samples(args, samples)

    def test_rollout_log_fans_out_versioned_tito_keys(self, monkeypatch):
        # [FUNCTION DOC] Replaces the tracking log function with one collecting all metrics into a
        # mapping, then logs rollout data for rollout 0 with four samples (one assistant text
        # mismatch, v2). Asserts the logged keys carry the rollout prefix: sample count 4, episode
        # reward 1.5, episode response length mean 4.0, total length mean 4.0, the v2 assistant text
        # rate 0.25, and no unversioned assistant text rate key.
        args = make_args(
            advantage_estimator="ppo",
            ci_test=False,
            log_passrate=False,
            use_session_server="v2",
        )
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {"tito_session_mismatch": [{"type": "assistant_text"}]}
        for sample in samples[1:]:
            sample.metadata = {"tito_session_mismatch": []}
        logged = {}
        monkeypatch.setattr(
            "miles.ray.rollout.metrics.tracking.log",
            lambda _args, metrics, **_kwargs: logged.update(metrics),
        )

        log_rollout_data(0, args, samples, None, 1.0)

        assert logged["rollout/num_training_samples"] == 4
        assert logged["rollout/episode_raw_reward"] == pytest.approx(1.5)
        assert logged["rollout/episode_response_length/mean"] == pytest.approx(4.0)
        assert logged["rollout/episode_total_response_length/mean"] == pytest.approx(4.0)
        assert logged["rollout/tito_session_mismatch_rate/v2/assistant_text"] == 0.25
        assert "rollout/tito_session_mismatch_rate/assistant_text" not in logged


class TestComputePassrateFromSamples:
    # [CLASS DOC] Tests the pass-at-k computation over complete prompt groups, reporting pass at 1,
    # 2 and 4 for groups of four samples.
    def test_returns_empty_when_group_size_is_one(self):
        # [FUNCTION DOC] With one sample per prompt the result is an empty mapping.
        args = make_args(n_samples_per_prompt=1)
        samples = make_samples_grouped(4, 1, rewards=[1.0, 0.0, 1.0, 0.0])

        assert _compute_passrate_from_samples(args, samples) == {}

    @pytest.mark.parametrize("reward, expected", [(1.0, 1.0), (0.0, 0.0)])
    def test_uniform_rewards(self, reward, expected):
        # [FUNCTION DOC] Parametrized with all-one rewards (expected 1.0) and all-zero rewards
        # (expected 0.0) over two groups of four: pass at 1, 2 and 4 all equal the expected value.
        args = make_args(n_samples_per_prompt=4, reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[reward] * 8)

        out = _compute_passrate_from_samples(args, samples)

        assert out == {
            "pass@1": pytest.approx(expected),
            "pass@2": pytest.approx(expected),
            "pass@4": pytest.approx(expected),
        }

    def test_mixed_rewards_pass_at_k_increases_with_k(self):
        # [FUNCTION DOC] Two groups of four, each with exactly one success: asserts strictly
        # increasing pass at 1 < pass at 2 < pass at 4.
        args = make_args(n_samples_per_prompt=4, reward_key=None)
        rewards = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        samples = make_samples_grouped(2, 4, rewards=rewards)

        out = _compute_passrate_from_samples(args, samples)

        assert out["pass@1"] < out["pass@2"] < out["pass@4"]

    def test_excludes_incomplete_groups(self):
        # [FUNCTION DOC] First group all ones, second group all zeros but with its last sample
        # removed (incomplete). Asserts all three pass-at-k values are 1.0, i.e. only the complete
        # group counted.
        args = make_args(n_samples_per_prompt=4, reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[1.0] * 4 + [0.0] * 4)
        samples.pop()

        out = _compute_passrate_from_samples(args, samples)

        assert out == {
            "pass@1": pytest.approx(1.0),
            "pass@2": pytest.approx(1.0),
            "pass@4": pytest.approx(1.0),
        }


class TestWeightVersionMetrics:
    # [CLASS DOC] Tests the weight version metrics derived from per-call weight version spans on
    # samples: statistics of each sample's oldest version and the ratio of mixed-version samples.
    def test_reports_oldest_version_statistics_and_mixed_ratio(self):
        # [FUNCTION DOC] Sample one saw version 4; sample two saw 5 then 6. Asserts the min of the
        # oldest versions is 4, the max is 5, and the mixed version ratio is 0.5.
        """weight_version/* summarises each sample's oldest version; mixed counts samples spanning an update."""
        samples = [
            _make_versioned_sample(["4"], index=0),
            _make_versioned_sample(["5", "6"], index=1),
        ]

        out = _compute_metrics_from_samples(make_args(), samples)

        assert out["weight_version/min"] == 4
        assert out["weight_version/max"] == 5
        assert out["weight_version/mixed_version_ratio"] == 0.5

    def test_a_call_spanning_no_update_is_not_mixed(self):
        # [FUNCTION DOC] One sample whose two calls both saw version 7: the mixed ratio is 0.0.
        """Two calls that both saw the same version must not count as mixed."""
        samples = [_make_versioned_sample(["7", "7"], index=0)]

        out = _compute_metrics_from_samples(make_args(), samples)

        assert out["weight_version/mixed_version_ratio"] == 0.0

    def test_a_single_call_spanning_two_versions_counts_as_mixed(self):
        # [FUNCTION DOC] One sample with a single call whose spans cover versions 3 and 4: the
        # mixed ratio is 1.0.
        """A weight update landing mid-call makes that single call mixed, just like two calls seeing two versions."""
        sample = make_sample(index=0, group_index=0)
        sample.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan("3", 0, 2), WeightVersionSpan("4", 2, 4)])
        ]

        out = _compute_metrics_from_samples(make_args(), [sample])

        assert out["weight_version/mixed_version_ratio"] == 1.0

    def test_no_version_metrics_when_nothing_was_stamped(self):
        # [FUNCTION DOC] A sample with no weight versions yields no key starting with the weight
        # version prefix.
        """SFT-style batches carry no versions and must not synthesise the series."""
        out = _compute_metrics_from_samples(make_args(), [make_sample(index=0, group_index=0)])

        assert not any(key.startswith("weight_version/") for key in out)


def _make_versioned_sample(versions: list[str], *, index: int) -> Sample:
    # [FUNCTION DOC] HELPER: Builds a sample (group 0, given index) with one weight-version call
    # record per version string, each call holding one span of that version over a one-token range
    # at its position. Returns the sample.
    sample = make_sample(index=index, group_index=0)
    sample.weight_versions = [
        WeightVersionsPerCall(spans=[WeightVersionSpan(version, i, i + 1)]) for i, version in enumerate(versions)
    ]
    return sample


class TestLogRolloutData:
    # [CLASS DOC] Tests the rollout data logging entry point's metric naming.
    def test_the_model_id_comes_from_the_caller_not_from_the_args(self, monkeypatch):
        # [FUNCTION DOC] Captures the payload and step key passed to the tracking log function.
        # Logs rollout data with a trainer model id argument of alpha while the arguments' own
        # trainer model id is unset. Asserts exactly one log call whose step key is
        # alpha/rollout/step and whose every metric key starts with alpha/.
        """One rollout executor serves every policy, so the id must travel with the call, not with the run."""
        calls: list[tuple[dict, str]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.metrics.tracking.log",
            lambda _args, payload, step_key: calls.append((payload, step_key)),
        )
        args = make_args(advantage_estimator="ppo", ci_test=False, log_passrate=False, trainer_model_id=None)

        log_rollout_data(0, args, make_samples_grouped(1, 4), None, 1.0, trainer_model_id="alpha")

        [(payload, step_key)] = calls
        assert step_key == "alpha/rollout/step"
        assert all(key.startswith("alpha/") for key in payload)


class TestEvalMetrics:
    # [CLASS DOC] Tests the eval data logging entry point's metric naming.
    def test_eval_metrics_are_not_namespaced_by_policy(self, monkeypatch):
        # [FUNCTION DOC] Captures the tracking log call while logging eval data for a gsm8k dataset
        # with rewards 1.0 and 0.0 and arguments naming trainer model alpha. Asserts one call with
        # step key eval/step, the metric eval/gsm8k equal to 0.5, and no key prefixed alpha/.
        """Pinning the status quo: a run training several policies is refused an eval, so eval keeps one step axis."""
        calls: list[tuple[dict, str]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.metrics.tracking.log",
            lambda _args, payload, step_key: calls.append((payload, step_key)),
        )
        args = make_args(log_passrate=False, trainer_model_id="alpha")

        log_eval_rollout_data(0, args, {"gsm8k": {"rewards": [1.0, 0.0]}})

        [(payload, step_key)] = calls
        assert step_key == "eval/step"
        assert payload["eval/gsm8k"] == 0.5
        assert not any(key.startswith("alpha/") for key in payload)
