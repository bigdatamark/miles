# [MODULE DOC]
# Unit tests for converting rollout samples into training data on the rollout side: building the
# training data mapping (required and optional keys, loss mask handling, rollout ids, sampling mask,
# weight version serialization, custom conversion hook, dynamic global batch size metadata),
# post-processing of rewards (centering and optional std normalization per prompt group for group
# relative estimators, passthrough for ppo, grouping by explicit group index or prompt group sizes
# or legacy boundaries, shared reward per rollout, custom hook), property-based invariants of the
# group normalization, splitting data across data parallel ranks (strided or balanced, optional and
# shared keys), the raw split with witness ids and other per-sample tensors, eligibility for
# scheduling on the rollout side and the scheduled split (static, compact, dynamic token cap and
# dynamic global batch size), and shard packaging of loss vectors. A module-scoped autouse fixture
# starts a local ray cluster and the split tests initialize the object store singleton.
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import ray
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.fast.ray.rollout.conftest import make_args, make_sample, make_samples_grouped

from miles.ray.rollout.train_data_conversion import (
    _post_process_rewards,
    can_schedule_on_rollout_side,
    convert_samples_to_train_data,
    split_train_data_by_dp,
    split_train_data_by_dp_raw,
    split_train_data_by_dp_scheduled_raw,
)
from miles.utils import object_store
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


@pytest.fixture(scope="module", autouse=True)
def _ray_minicluster(ray_local_mode):
    # [FUNCTION DOC] HELPER: Module-scoped autouse fixture that depends on the shared local-mode ray
    # fixture (so ray is initialized for the module) and yields once; no teardown of its own.
    """split_train_data_by_dp uses ray.put(...) so we need Ray."""
    yield


# ----------------------------- convert_samples_to_train_data -----------------------------


class TestConvertSamplesToTrainData:
    # [CLASS DOC] Tests the conversion of samples to the training data mapping. Every call passes
    # empty metadata and no custom hooks unless stated; reward normalization is mostly disabled.
    def test_default_path_produces_required_keys(self):
        # [FUNCTION DOC] Two groups of four grpo samples: the result has the keys tokens, response
        # lengths, rewards, raw reward, truncated, sample indices and loss masks, and one token list
        # per sample.
        args = make_args(advantage_estimator="grpo", rewards_normalization=False)
        samples = make_samples_grouped(n_groups=2, group_size=4)
        out = convert_samples_to_train_data(
            args,
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        for key in (
            "tokens",
            "response_lengths",
            "rewards",
            "raw_reward",
            "truncated",
            "sample_indices",
            "loss_masks",
        ):
            assert key in out, f"missing required key {key}"
        assert len(out["tokens"]) == len(samples)

    def test_loss_mask_none_filled_with_ones(self):
        # [FUNCTION DOC] A sample with response length 5 and no loss mask gets a mask of five ones.
        args = make_args(rewards_normalization=False)
        s = make_sample(response_length=5)
        s.loss_mask = None
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["loss_masks"][0] == [1] * 5

    def test_remove_sample_zeroes_loss_mask(self):
        # [FUNCTION DOC] A sample flagged removed with an all-ones mask of length 4 yields a mask of
        # four zeros.
        args = make_args(rewards_normalization=False)
        s = make_sample(response_length=4)
        s.loss_mask = [1, 1, 1, 1]
        s.remove_sample = True
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["loss_masks"][0] == [0, 0, 0, 0]

    def test_loss_mask_length_mismatch_asserts(self):
        # [FUNCTION DOC] A loss mask of length 2 for a response of length 4 raises an assertion error.
        args = make_args(rewards_normalization=False)
        s = make_sample(response_length=4)
        s.loss_mask = [1, 1]  # wrong length
        with pytest.raises(AssertionError):
            convert_samples_to_train_data(
                args,
                [s],
                metadata={},
                custom_convert_samples_to_train_data_func=None,
                custom_reward_post_process_func=None,
            )

    def test_truncated_status_marked(self):
        # [FUNCTION DOC] A sample with truncated status gives a truncated flag of 1.
        args = make_args(rewards_normalization=False)
        s = make_sample(status=Sample.Status.TRUNCATED)
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["truncated"][0] == 1

    def test_optional_field_rollout_log_probs_passed_through(self):
        # [FUNCTION DOC] Rollout log probabilities set on a sample appear unchanged in the output.
        args = make_args(rewards_normalization=False)
        s = make_sample()
        s.rollout_log_probs = [-0.1, -0.2, -0.3, -0.4]
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["rollout_log_probs"][0] == [-0.1, -0.2, -0.3, -0.4]

    def test_sampling_mask_passed_through(self):
        # [FUNCTION DOC] A rollout sampling mask (eight ids, offsets 0, 2, 4, 6, 8) appears in the
        # output as array-like ids and offsets whose list forms equal the originals.
        args = make_args(rewards_normalization=False)
        s = make_sample()
        s.rollout_sampling_mask = RolloutSamplingMask(
            ids=[0, 7, 1, 8, 2, 9, 3, 10],
            offsets=[0, 2, 4, 6, 8],
        )
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["rollout_sampling_mask_ids"][0].tolist() == [0, 7, 1, 8, 2, 9, 3, 10]
        assert out["rollout_sampling_mask_offsets"][0].tolist() == [0, 2, 4, 6, 8]

    def test_sampling_mask_requires_complete_batch(self):
        # [FUNCTION DOC] A batch where only one of two samples has a sampling mask raises a value
        # error saying it must be present for every training sample and naming sample index 9.
        args = make_args(rewards_normalization=False)
        captured = make_sample(index=8)
        captured.rollout_sampling_mask = RolloutSamplingMask(
            ids=[0, 7, 1, 8, 2, 9, 3, 10],
            offsets=[0, 2, 4, 6, 8],
        )

        with pytest.raises(
            ValueError,
            match=r"must be present for every training sample.*sample_index=9",
        ):
            convert_samples_to_train_data(
                args,
                [captured, make_sample(index=9)],
                metadata={},
                custom_convert_samples_to_train_data_func=None,
                custom_reward_post_process_func=None,
            )

    def test_optional_field_round_number_from_metadata(self):
        # [FUNCTION DOC] A sample metadata round number 7 appears as round number 7 in the output.
        args = make_args(rewards_normalization=False)
        s = make_sample()
        s.metadata = {"round_number": 7}
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["round_number"][0] == 7

    def test_optional_field_raw_reward_overridden_from_metadata(self):
        # [FUNCTION DOC] A sample with reward 1.0 but metadata raw reward 9.0 outputs raw reward 9.0.
        args = make_args(rewards_normalization=False)
        s = make_sample(reward=1.0)
        s.metadata = {"raw_reward": 9.0}
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["raw_reward"][0] == 9.0

    def test_rollout_ids_default_to_sample_index(self):
        # [FUNCTION DOC] Samples with indices 0, 1, 2 and no rollout ids give rollout ids 0, 1, 2.
        args = make_args(rewards_normalization=False)
        samples = [make_sample(index=i) for i in range(3)]
        out = convert_samples_to_train_data(
            args,
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["rollout_ids"] == [0, 1, 2]

    def test_rollout_ids_use_explicit_rollout_id_when_set(self):
        # [FUNCTION DOC] Four samples with indices 0 to 3 where samples 1 and 2 share an explicit
        # rollout id 1: the rollout ids output is 0, 1, 1, 3.
        args = make_args(rewards_normalization=False)
        samples = [make_sample(index=i) for i in range(4)]
        # two compact siblings sharing one rollout execution
        samples[1].rollout_id = samples[2].rollout_id = 1
        out = convert_samples_to_train_data(
            args,
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["rollout_ids"] == [0, 1, 1, 3]

    def test_weight_versions_are_converted_to_serializable_dicts(self):
        # [FUNCTION DOC] One sample with one call holding a span (version v1, absolute start 2, end
        # 4): the output weight versions is a list per sample of a list per call of span dictionaries
        # with keys version, abs_start, abs_end.
        """Weight version spans cross the object-store boundary as plain msgpack values."""
        args = make_args(rewards_normalization=False)
        sample = make_sample()
        sample.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan(version="v1", abs_start=2, abs_end=4)])
        ]

        out = convert_samples_to_train_data(
            args,
            [sample],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )

        assert out["weight_versions"] == [[[{"version": "v1", "abs_start": 2, "abs_end": 4}]]]

    def test_weight_version_serialization_preserves_empty_samples_and_calls(self):
        # [FUNCTION DOC] A first sample with two calls (the first with no spans, the second with one
        # span) and a second sample with no calls: the output keeps an empty list for the span-less
        # call and an empty list for the call-less sample, so slots stay aligned.
        """A sample without calls and a call without spans keep their slots, so rows and turn counts stay aligned."""
        args = make_args(rewards_normalization=False)
        stamped = make_sample(index=0)
        stamped.weight_versions = [
            WeightVersionsPerCall(spans=[]),
            WeightVersionsPerCall(spans=[WeightVersionSpan(version="v1", abs_start=2, abs_end=4)]),
        ]
        unstamped = make_sample(index=1)
        unstamped.weight_versions = []

        out = convert_samples_to_train_data(
            args,
            [stamped, unstamped],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )

        assert out["weight_versions"] == [[[], [{"version": "v1", "abs_start": 2, "abs_end": 4}]], []]

    def test_custom_convert_func_short_circuits(self):
        # [FUNCTION DOC] A custom conversion function returning a sentinel mapping: the result is
        # that identical object.
        args = make_args()
        sentinel = {"foo": "bar"}
        out = convert_samples_to_train_data(
            args,
            [make_sample()],
            metadata={},
            custom_convert_samples_to_train_data_func=lambda a, s: sentinel,
            custom_reward_post_process_func=None,
        )
        assert out is sentinel

    def test_dynamic_global_batch_size_metadata_must_match(self):
        # [FUNCTION DOC] With the dynamic global batch size option on but no such entry in the
        # metadata, conversion raises an assertion error.
        args = make_args(use_dynamic_global_batch_size=True, rewards_normalization=False)
        with pytest.raises(AssertionError):
            convert_samples_to_train_data(
                args,
                [make_sample()],
                metadata={},
                custom_convert_samples_to_train_data_func=None,
                custom_reward_post_process_func=None,
            )

    def test_dynamic_global_batch_size_metadata_passed_through(self):
        # [FUNCTION DOC] With the option on and metadata dynamic global batch size 16, the output
        # carries 16.
        args = make_args(use_dynamic_global_batch_size=True, rewards_normalization=False)
        out = convert_samples_to_train_data(
            args,
            [make_sample()],
            metadata={"dynamic_global_batch_size": 16},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["dynamic_global_batch_size"] == 16


# ----------------------------- _post_process_rewards -----------------------------


class TestPostProcessRewards:
    # [CLASS DOC] Tests the private reward post-processing, which returns a pair of the raw rewards
    # and the processed rewards (group centered, optionally std normalized, per prompt group).
    def test_ppo_path_returns_raw_rewards_unchanged(self):
        # [FUNCTION DOC] With ppo and normalization requested, the processed rewards equal the raw ones.
        args = make_args(advantage_estimator="ppo", rewards_normalization=True)
        samples = make_samples_grouped(2, 4, rewards=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        raw, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        assert raw == processed

    def test_grpo_normalizes_each_group_to_zero_mean(self):
        # [FUNCTION DOC] Two groups of four with means 2.5 and 25: after grpo centering (no std
        # normalization) each group's mean is within 1e-5 of zero.
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=4,
            rollout_batch_size=2,
        )
        samples = make_samples_grouped(2, 4, rewards=[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        # group means: 2.5 and 25.0 → centered values
        g1 = processed[:4]
        g2 = processed[4:]
        assert abs(sum(g1) / 4) < 1e-5
        assert abs(sum(g2) / 4) < 1e-5

    def test_grpo_with_std_normalization_unit_variance(self):
        # [FUNCTION DOC] One group of rewards 1 to 4 with std normalization: the sample standard
        # deviation (one degree of freedom less) of the processed rewards is within 1e-4 of 1.
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=True,
            n_samples_per_prompt=4,
            rollout_batch_size=1,
        )
        samples = make_samples_grouped(1, 4, rewards=[1.0, 2.0, 3.0, 4.0])
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        # Unit std with epsilon — torch.std uses N-1 by default; centered
        # values have std exactly 1.0 after dividing by their own std.
        import numpy as np

        assert abs(np.std(processed, ddof=1) - 1.0) < 1e-4

    def test_gspo_uses_grpo_normalization_path(self):
        # [FUNCTION DOC] With the gspo estimator and one group of four, the processed mean is zero.
        args = make_args(
            advantage_estimator="gspo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=4,
            rollout_batch_size=1,
        )
        samples = make_samples_grouped(1, 4, rewards=[1.0, 2.0, 3.0, 4.0])
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        assert abs(sum(processed) / 4) < 1e-5

    def test_reinforce_plus_plus_baseline_only_zero_mean_no_std(self):
        # [FUNCTION DOC] With the reinforce-plus-plus-baseline estimator and the std normalization
        # flag on (which must be ignored), rewards 1 to 4 become centered values -1.5, -0.5, 0.5,
        # 1.5: the mean is zero and the population std equals that of the centered values.
        args = make_args(
            advantage_estimator="reinforce_plus_plus_baseline",
            rewards_normalization=True,
            grpo_std_normalization=True,  # should be IGNORED on this path
            n_samples_per_prompt=4,
            rollout_batch_size=1,
        )
        samples = make_samples_grouped(1, 4, rewards=[1.0, 2.0, 3.0, 4.0])
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        # Mean is zero (centering happens) and std is the std of the centered
        # input (no normalization applied) — pin this exactly so a regression
        # that re-enabled the std division is caught.
        assert abs(sum(processed) / 4) < 1e-5
        import numpy as np

        expected_std = float(np.std([-1.5, -0.5, 0.5, 1.5]))
        assert abs(np.std(processed) - expected_std) < 1e-5

    def test_irregular_group_size_uses_explicit_group_index(self):
        # [FUNCTION DOC] Configured for groups of 8 and batch size 2, but only one group of four
        # samples (rewards 2, 4, 6, 8) with an explicit group index is given: the processed rewards
        # sum to zero (centered over the actual group).
        """Explicit group identity keeps an irregularly sized group together."""
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=8,
            rollout_batch_size=2,
        )
        # rewards length 4 — does not match 8 * 2 = 16; trigger view branch
        samples = make_samples_grouped(1, 4, rewards=[2.0, 4.0, 6.0, 8.0])
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        # mean is 5.0, after centering: -3, -1, 1, 3
        assert abs(sum(processed)) < 1e-5

    def test_grpo_normalizes_unique_rollouts_with_unequal_fanout(self):
        # [FUNCTION DOC] Seven samples in two groups where some rollouts emit several sibling rows.
        # Raw rewards are returned per row as given; the processed rewards center per group over
        # unique rollouts (group 0 rollouts 0 and 1, group 1 rollouts 2 and 4) and broadcast to
        # each row: -0.5, 0.5, 0.5, 0.5, -1.0, -1.0, 1.0.
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=2,
            rollout_batch_size=2,
        )
        samples = [
            make_sample(group_index=0, index=0, rollout_id=10, reward=0.0),
            make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
            make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
            make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
            make_sample(group_index=1, index=2, rollout_id=20, reward=2.0),
            make_sample(group_index=1, index=2, rollout_id=20, reward=2.0),
            make_sample(group_index=1, index=3, rollout_id=21, reward=4.0),
        ]

        raw, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert raw == [0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0]
        assert processed == pytest.approx([-0.5, 0.5, 0.5, 0.5, -1.0, -1.0, 1.0])

    def test_grpo_broadcasts_std_normalized_rollout_advantage(self):
        # [FUNCTION DOC] One group with two unique rollouts (rewards 0 and 1, the second emitting two
        # rows) and std normalization: the processed rewards are the negative and positive inverse
        # square root of two, with the positive value repeated for both sibling rows (tolerance 1e-5).
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=True,
            n_samples_per_prompt=2,
            rollout_batch_size=1,
        )
        samples = [
            make_sample(group_index=0, index=0, rollout_id=10, reward=0.0),
            make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
            make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
        ]

        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert processed == pytest.approx([-(2**-0.5), 2**-0.5, 2**-0.5], abs=1e-5)

    def test_grpo_rejects_different_sibling_rewards(self):
        # [FUNCTION DOC] Two sibling rows of rollout 11 with rewards 2.0 and 6.0 raise a value error
        # saying all samples in rollout 11 must share one reward and listing rows 1 and 2 with
        # their rewards.
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=2,
            rollout_batch_size=1,
        )
        samples = [
            make_sample(group_index=0, index=0, rollout_id=10, reward=0.0, loss_mask=[1, 1, 1, 1]),
            make_sample(group_index=0, index=1, rollout_id=11, reward=2.0, loss_mask=[1, 0, 0, 0]),
            make_sample(group_index=0, index=1, rollout_id=11, reward=6.0, loss_mask=[1, 1, 1, 0]),
        ]

        with pytest.raises(
            ValueError,
            match=r"all samples in rollout 11 must share one reward; rows \[1, 2\] have rewards \[2.0, 6.0\]",
        ):
            _post_process_rewards(args, samples, custom_reward_post_process_func=None)

    def test_grpo_shared_reward_ignores_final_training_mask(self):
        # [FUNCTION DOC] Three rows: a removed row and a normal row sharing rollout 10 (reward 2.0)
        # and a row of rollout 11 (reward 6.0, loss mask 1, 1, 0, 0). Full conversion yields raw
        # rewards 2, 2, 6; processed rewards -2, -2, 2 (the removed row still counts); and loss masks
        # eight zeros, four ones and the given mask.
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=2,
            rollout_batch_size=1,
        )
        samples = [
            make_sample(
                group_index=0,
                index=0,
                rollout_id=10,
                reward=2.0,
                response_length=8,
                remove_sample=True,
            ),
            make_sample(group_index=0, index=0, rollout_id=10, reward=2.0, response_length=4),
            make_sample(group_index=0, index=1, rollout_id=11, reward=6.0, loss_mask=[1, 1, 0, 0]),
        ]

        train_data = convert_samples_to_train_data(
            args,
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )

        assert train_data["raw_reward"] == [2.0, 2.0, 6.0]
        assert train_data["rewards"] == pytest.approx([-2.0, -2.0, 2.0])
        assert train_data["loss_masks"] == [[0] * 8, [1] * 4, [1, 1, 0, 0]]

    def test_grpo_shared_reward_uses_selected_reward_key(self):
        # [FUNCTION DOC] Rewards are mappings and the reward key is score; siblings share score 2.0
        # despite different details. Raw rewards are the scores 2, 2, 6 and processed rewards are
        # -2, -2, 2.
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            reward_key="score",
            n_samples_per_prompt=2,
            rollout_batch_size=1,
        )
        samples = [
            make_sample(group_index=0, index=0, rollout_id=10, reward={"score": 2.0, "detail": "first"}),
            make_sample(group_index=0, index=0, rollout_id=10, reward={"score": 2.0, "detail": "second"}),
            make_sample(group_index=0, index=1, rollout_id=11, reward={"score": 6.0}),
        ]

        raw, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert raw == [2.0, 2.0, 6.0]
        assert processed == pytest.approx([-2.0, -2.0, 2.0])

    def test_prompt_group_sizes_override_reused_group_index(self):
        # [FUNCTION DOC] Four samples all with group index 0 but an explicit prompt group sizes list
        # of 2 and 2: the groups are the first two and last two rows, giving -1, 1, -2, 2.
        args = make_args(advantage_estimator="grpo", rewards_normalization=True)
        samples = [
            make_sample(group_index=0, rollout_id=10, reward=0.0),
            make_sample(group_index=0, rollout_id=11, reward=2.0),
            make_sample(group_index=0, rollout_id=20, reward=10.0),
            make_sample(group_index=0, rollout_id=21, reward=14.0),
        ]

        _, processed = _post_process_rewards(
            args,
            samples,
            custom_reward_post_process_func=None,
            prompt_group_sizes=[2, 2],
        )

        assert processed == pytest.approx([-1.0, 1.0, -2.0, 2.0])

    def test_noncontiguous_group_indices_share_reward_group(self):
        # [FUNCTION DOC] Rows alternate between group 0 and group 1; each group is centered
        # separately with results kept in row order: -1, -2, 1, 2.
        args = make_args(advantage_estimator="grpo", rewards_normalization=True)
        samples = [
            make_sample(group_index=0, rollout_id=10, reward=0.0),
            make_sample(group_index=1, rollout_id=20, reward=10.0),
            make_sample(group_index=0, rollout_id=11, reward=2.0),
            make_sample(group_index=1, rollout_id=21, reward=14.0),
        ]

        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert processed == pytest.approx([-1.0, -2.0, 1.0, 2.0])

    @pytest.mark.parametrize(
        ("rewards", "expected"),
        [
            ([0.0, 2.0, 10.0, 14.0], [-1.0, 1.0, -2.0, 2.0]),
            ([0.0, 2.0, 4.0], [-2.0, 0.0, 2.0]),
        ],
    )
    def test_missing_group_indices_use_legacy_boundaries(self, rewards, expected):
        # [FUNCTION DOC] Parametrized over four rewards (two groups of two) and three rewards (an
        # incomplete last group). With no group index on samples, groups are formed by the
        # configured group size of 2 in row order; asserts the expected centered values
        # (-1, 1, -2, 2 and -2, 0, 2 respectively); in the second case the three rows therefore
        # form a single group rather than a group of two plus a leftover.
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            n_samples_per_prompt=2,
            rollout_batch_size=2,
        )
        samples = [
            make_sample(group_index=None, rollout_id=rollout_id, reward=reward)
            for rollout_id, reward in enumerate(rewards)
        ]

        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert processed == pytest.approx(expected)

    def test_rows_without_rollout_identity_stay_distinct(self):
        # [FUNCTION DOC] Two rows in group 0 with no index and no rollout id (rewards 0 and 2): each
        # row is its own rollout, so the processed rewards are -1 and 1.
        args = make_args(advantage_estimator="grpo", rewards_normalization=True)
        samples = [
            make_sample(group_index=0, index=None, rollout_id=None, reward=0.0),
            make_sample(group_index=0, index=None, rollout_id=None, reward=2.0),
        ]

        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert processed == pytest.approx([-1.0, 1.0])

    def test_empty_rewards_stay_empty(self):
        # [FUNCTION DOC] No samples gives empty raw and processed lists.
        args = make_args(advantage_estimator="grpo", rewards_normalization=True)

        raw, processed = _post_process_rewards(args, [], custom_reward_post_process_func=None)

        assert raw == processed == []

    def test_custom_reward_post_process_short_circuits(self):
        # [FUNCTION DOC] A custom post-processing function returning a sentinel pair: the result is
        # exactly that pair.
        args = make_args(advantage_estimator="grpo", rewards_normalization=True)
        sentinel = ([0.0], [1.0])
        raw, processed = _post_process_rewards(
            args, [make_sample()], custom_reward_post_process_func=lambda a, s: sentinel
        )
        assert (raw, processed) == sentinel


class TestPostProcessRewardsProperties:
    # [CLASS DOC] Property-based tests (hypothesis, no deadline, function-scoped fixture health
    # check suppressed) of the group normalization invariants.
    """Hypothesis-driven invariants for the GRPO normalization path.

    The point-tests above pin specific shapes; these guarantee the math holds
    across arbitrary group counts, group sizes, and reward distributions —
    catching bugs (e.g. wrong reshape axis, sign flip) that a fixed example
    might miss."""

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
        max_examples=40,
    )
    @given(
        n_groups=st.integers(min_value=1, max_value=6),
        group_size=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_grpo_zero_mean_invariant(self, n_groups, group_size, seed):
        # [FUNCTION DOC] For up to 40 generated cases (1 to 6 groups of 2 to 8 rows, rewards drawn
        # uniformly in -1000 to 1000 from a seeded random generator), group centering without std
        # normalization yields one value per row and each group's mean is below 1e-3 times the
        # largest absolute reward (or 1).
        """After GRPO centering, every group's mean must be ≈ 0 regardless of
        input distribution."""
        import random

        rng = random.Random(seed)
        rewards_list = [rng.uniform(-1000, 1000) for _ in range(n_groups * group_size)]
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=group_size,
            rollout_batch_size=n_groups,
        )
        samples = make_samples_grouped(n_groups, group_size, rewards=rewards_list)
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert len(processed) == n_groups * group_size
        for g in range(n_groups):
            chunk = processed[g * group_size : (g + 1) * group_size]
            # Tolerance scales with magnitude; 1e-3 of mean(|input|) covers fp32 drift.
            scale = max(abs(min(rewards_list)), abs(max(rewards_list)), 1.0)
            assert abs(sum(chunk) / group_size) < 1e-3 * scale, f"group {g} mean is not zero: chunk={chunk}"

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
        max_examples=40,
    )
    @given(
        n_groups=st.integers(min_value=1, max_value=6),
        group_size=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_grpo_unit_variance_invariant(self, n_groups, group_size, seed):
        # [FUNCTION DOC] For up to 40 generated cases, builds each group as an arithmetic progression
        # with a random base (-100 to 100) and spread (0.5 to 50) so its std is well above the
        # stabilizing epsilon. With std normalization, each group's sample std (one degree of
        # freedom less) of the processed rewards is within 1e-3 of 1.
        """With grpo_std_normalization=True, each group's processed std → 1.

        We construct rewards whose per-group std is well above the 1e-6 epsilon
        floor; otherwise the epsilon-stabilized division produces ≈0 not ≈1."""
        import random

        import numpy as np

        rng = random.Random(seed)
        rewards_list: list[float] = []
        for _g in range(n_groups):
            base = rng.uniform(-100, 100)
            spread = rng.uniform(0.5, 50.0)  # >> epsilon
            for k in range(group_size):
                rewards_list.append(base + spread * (k - (group_size - 1) / 2))

        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=True,
            n_samples_per_prompt=group_size,
            rollout_batch_size=n_groups,
        )
        samples = make_samples_grouped(n_groups, group_size, rewards=rewards_list)
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        for g in range(n_groups):
            chunk = processed[g * group_size : (g + 1) * group_size]
            std_val = float(np.std(chunk, ddof=1))
            assert abs(std_val - 1.0) < 1e-3, f"group {g} std={std_val}, chunk={chunk}"

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
        max_examples=30,
    )
    @given(
        n=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_ppo_path_is_identity(self, n, seed):
        # [FUNCTION DOC] For up to 30 generated cases (1 to 32 rewards uniform in -50 to 50), the ppo
        # estimator returns raw rewards equal to the processed rewards and to the input list.
        """PPO never normalizes rewards regardless of any other flag."""
        import random

        rng = random.Random(seed)
        rewards_list = [rng.uniform(-50, 50) for _ in range(n)]
        args = make_args(
            advantage_estimator="ppo",
            rewards_normalization=True,
            grpo_std_normalization=True,
            n_samples_per_prompt=n,
            rollout_batch_size=1,
        )
        samples = make_samples_grouped(1, n, rewards=rewards_list)
        raw, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        assert raw == processed == rewards_list


# ----------------------------- split_train_data_by_dp -----------------------------


class TestSplitTrainDataByDp:
    # [CLASS DOC] Tests splitting the training data mapping across data parallel ranks, returning
    # one object store reference per rank whose payload is fetched with ray.
    @pytest.fixture(autouse=True)
    def _init_object_store(self):
        # [FUNCTION DOC] HELPER: Autouse fixture initializing the object store singleton from default
        # test arguments. Side effect: sets global state (never reset here).
        """split_train_data_by_dp puts through the object store singleton."""
        object_store.init_instance(make_args())

    def test_strided_partition_when_balance_data_off(self):
        # [FUNCTION DOC] Four samples, two ranks, balancing off: rank 0 gets partition indices 0 and
        # 2, rank 1 gets 1 and 3 (strided), and rank 0's tokens are those of samples 0 and 2.
        args = make_args(balance_data=False)
        data = {
            "tokens": [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10]],
            "response_lengths": [2, 3, 1, 4],
            "rewards": [0.1, 0.2, 0.3, 0.4],
            "truncated": [0, 0, 0, 1],
            "loss_masks": [[1, 1]] * 4,
            "sample_indices": [0, 1, 2, 3],
        }
        refs = split_train_data_by_dp(args, data, {"dp_size": 2})
        parts = [ray.get(r.payload) for r in refs]
        # stride: dp=0 takes [0, 2], dp=1 takes [1, 3]
        assert list(parts[0]["partition"]) == [0, 2]
        assert list(parts[1]["partition"]) == [1, 3]
        assert parts[0]["tokens"] == [[1, 2], [6]]

    def test_balanced_partition_when_balance_data_on(self):
        # [FUNCTION DOC] Four samples of lengths 1 to 4, two ranks, balancing on: the two ranks'
        # sample counts differ by at most one.
        args = make_args(balance_data=True)
        # lengths chosen to force grouping: 1 + 4 vs 2 + 3 are balanced
        data = {
            "tokens": [[1], [2, 3], [4, 5, 6], [7, 8, 9, 10]],
            "response_lengths": [1, 2, 3, 4],
            "rewards": [0, 0, 0, 0],
            "truncated": [0, 0, 0, 0],
            "loss_masks": [[1] * n for n in (1, 2, 3, 4)],
            "sample_indices": [0, 1, 2, 3],
        }
        refs = split_train_data_by_dp(args, data, {"dp_size": 2})
        parts = [ray.get(r.payload) for r in refs]
        sizes = [len(p["tokens"]) for p in parts]
        assert max(sizes) - min(sizes) <= 1

    def test_optional_keys_propagated_when_present(self):
        # [FUNCTION DOC] Rollout log probabilities and round numbers in the input appear in rank 0's
        # payload.
        args = make_args(balance_data=False)
        data = {
            "tokens": [[1], [2]],
            "response_lengths": [1, 1],
            "rewards": [0, 0],
            "truncated": [0, 0],
            "loss_masks": [[1], [1]],
            "sample_indices": [0, 1],
            "rollout_log_probs": [[-0.1], [-0.2]],
            "round_number": [1, 2],
        }
        refs = split_train_data_by_dp(args, data, {"dp_size": 2})
        parts = [ray.get(r.payload) for r in refs]
        assert "rollout_log_probs" in parts[0]
        assert "round_number" in parts[0]

    def test_shared_keys_not_split(self):
        # [FUNCTION DOC] The raw reward list and the dynamic global batch size are copied whole to
        # every rank rather than partitioned.
        """raw_reward, total_lengths, dynamic_global_batch_size are shared, not split."""
        args = make_args(balance_data=False)
        data = {
            "tokens": [[1], [2], [3], [4]],
            "response_lengths": [1, 1, 1, 1],
            "rewards": [0, 0, 0, 0],
            "truncated": [0, 0, 0, 0],
            "loss_masks": [[1]] * 4,
            "sample_indices": [0, 1, 2, 3],
            "raw_reward": [9.0, 8.0, 7.0, 6.0],
            "dynamic_global_batch_size": 4,
        }
        refs = split_train_data_by_dp(args, data, {"dp_size": 2})
        parts = [ray.get(r.payload) for r in refs]
        for p in parts:
            assert p["raw_reward"] == [9.0, 8.0, 7.0, 6.0]
            assert p["dynamic_global_batch_size"] == 4

    def test_partition_indices_form_a_partition(self):
        # [FUNCTION DOC] Twelve samples across four ranks: all ranks' partition indices together,
        # sorted, equal 0 to 11 (each sample on exactly one rank).
        """All partition indices together cover [0, N) exactly once."""
        args = make_args(balance_data=False)
        n = 12
        data = {
            "tokens": [[i] for i in range(n)],
            "response_lengths": [1] * n,
            "rewards": [0] * n,
            "truncated": [0] * n,
            "loss_masks": [[1]] * n,
            "sample_indices": list(range(n)),
        }
        refs = split_train_data_by_dp(args, data, {"dp_size": 4})
        parts = [ray.get(r.payload) for r in refs]
        all_indices = sorted(i for p in parts for i in p["partition"])
        assert all_indices == list(range(n))


class TestSplitTrainDataRaw:
    # [CLASS DOC] Tests the raw split (no object store), called with a mock arguments object that
    # has balancing off, returning one data mapping per rank.
    def test_witness_ids_split_across_dp(self) -> None:
        # [FUNCTION DOC] Four samples each with a tensor of witness ids, two ranks: two shards are
        # returned, each with a witness id entry holding two items.
        tokens = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10, 11]]
        witness_ids = [
            torch.tensor([0, 0, 0]),
            torch.tensor([1, 1]),
            torch.tensor([2, 2, 2, 2]),
            torch.tensor([3, 3]),
        ]

        data = {
            "tokens": tokens,
            "seq_witness_ids": witness_ids,
            "response_lengths": [1, 1, 1, 1],
            "loss_masks": [[0, 0, 1], [0, 1], [0, 0, 0, 1], [0, 1]],
        }

        args = MagicMock()
        args.balance_data = False

        result = split_train_data_by_dp_raw(args, data, dp_size=2)

        assert len(result) == 2
        assert "seq_witness_ids" in result[0]
        assert "seq_witness_ids" in result[1]
        assert len(result[0]["seq_witness_ids"]) == 2
        assert len(result[1]["seq_witness_ids"]) == 2

    def test_indexer_topk_and_opd_reverse_kl_split_across_dp(self) -> None:
        # [FUNCTION DOC] Four samples with per-sample rollout indexer top-k tensors and
        # opd-reverse-kl lists, two ranks: each shard holds two items of each key.
        """Keys from the rollout-side split (rollout_indexer_topk, opd_reverse_kl) partition per sample."""
        data = {
            "tokens": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "response_lengths": [1, 1, 1, 1],
            "loss_masks": [[0, 1], [0, 1], [0, 1], [0, 1]],
            "rollout_indexer_topk": [torch.tensor([i]) for i in range(4)],
            "opd_reverse_kl": [[float(i)] for i in range(4)],
        }

        args = MagicMock()
        args.balance_data = False

        result = split_train_data_by_dp_raw(args, data, dp_size=2)

        assert len(result) == 2
        for part in result:
            assert len(part["rollout_indexer_topk"]) == 2
            assert len(part["opd_reverse_kl"]) == 2

    def test_no_witness_ids_when_absent(self) -> None:
        # [FUNCTION DOC] Without witness ids in the input and with one rank, the shard has no
        # witness id key.
        tokens = [[1, 2], [3, 4]]
        data = {
            "tokens": tokens,
            "response_lengths": [1, 1],
            "loss_masks": [[0, 1], [0, 1]],
        }

        args = MagicMock()
        args.balance_data = False

        result = split_train_data_by_dp_raw(args, data, dp_size=1)
        assert "seq_witness_ids" not in result[0]


FULL_SCHEDULE_CONFIG = {
    "dp_size": 2,
    "cp_size": 1,
    "vpp_size": 1,
    "microbatch_group_size_per_vp_stage": None,
}


def _make_split_data(n: int, *, lengths: list[int] | None = None, rollout_ids: list[int] | None = None) -> dict:
    # [FUNCTION DOC] HELPER: Builds a training data mapping of n samples. Sample lengths default to
    # 2 each (a length list must have n entries, else an assertion error). Tokens are ranges of the
    # given lengths, response lengths are 1, rewards 0.0, truncated 0, loss masks all ones with the
    # token length, sample indices 0 to n-1, and rollout ids default to the sample indices.
    lengths = lengths or [2] * n
    assert len(lengths) == n
    return {
        "tokens": [list(range(length)) for length in lengths],
        "response_lengths": [1] * n,
        "rewards": [0.0] * n,
        "truncated": [0] * n,
        "loss_masks": [[1] * length for length in lengths],
        "sample_indices": list(range(n)),
        "rollout_ids": rollout_ids if rollout_ids is not None else list(range(n)),
    }


class TestCanScheduleOnRolloutSide:
    # [CLASS DOC] Tests the eligibility check for scheduling micro batches on the rollout side. The
    # default test arguments have global batch size 8.
    def test_eligible_with_full_megatron_config(self):
        # [FUNCTION DOC] Eight samples with a full schedule config (dp 2, cp 1, vpp 1) are eligible.
        args = make_args(balance_data=False, micro_batch_size=1, use_dynamic_batch_size=False, multi_lora=False)
        assert can_schedule_on_rollout_side(args, _make_split_data(8), FULL_SCHEDULE_CONFIG)

    def test_rejects_partial_config(self):
        # [FUNCTION DOC] A config with only the data parallel size, an empty config and no config
        # are all rejected.
        """fsdp / torchtitan advertise only dp_size; indep_dp advertises {}."""
        args = make_args(balance_data=False, micro_batch_size=1, multi_lora=False)
        assert not can_schedule_on_rollout_side(args, _make_split_data(8), {"dp_size": 2})
        assert not can_schedule_on_rollout_side(args, _make_split_data(8), {})
        assert not can_schedule_on_rollout_side(args, _make_split_data(8), None)

    def test_rejects_multi_lora(self):
        # [FUNCTION DOC] With multi lora on, a full config is still rejected.
        args = make_args(balance_data=False, micro_batch_size=1, multi_lora=True)
        assert not can_schedule_on_rollout_side(args, _make_split_data(8), FULL_SCHEDULE_CONFIG)

    def test_rejects_multimodal(self):
        # [FUNCTION DOC] Data containing multimodal training inputs is rejected.
        args = make_args(balance_data=False, micro_batch_size=1, multi_lora=False)
        data = _make_split_data(8)
        data["multimodal_train_inputs"] = [None] * 8
        assert not can_schedule_on_rollout_side(args, data, FULL_SCHEDULE_CONFIG)

    def test_rejects_fewer_rollouts_than_gbs(self):
        # [FUNCTION DOC] Six samples with global batch size 8 are rejected.
        args = make_args(balance_data=False, micro_batch_size=1, multi_lora=False)  # global_batch_size=8
        assert not can_schedule_on_rollout_side(args, _make_split_data(6), FULL_SCHEDULE_CONFIG)

    def test_accepts_trailing_partial_step(self):
        # [FUNCTION DOC] Ten samples with global batch size 8 are accepted (the schedule drops the
        # extra ones).
        """Extra rollouts beyond a full step are fine — the schedule drops them."""
        args = make_args(balance_data=False, micro_batch_size=1, multi_lora=False)  # global_batch_size=8
        assert can_schedule_on_rollout_side(args, _make_split_data(10), FULL_SCHEDULE_CONFIG)

    def test_dynamic_gbs_overrides_args_gbs(self):
        # [FUNCTION DOC] Six samples with a dynamic global batch size of 6 in the data are accepted
        # even though the arguments say 8.
        args = make_args(balance_data=False, micro_batch_size=1, multi_lora=False)  # global_batch_size=8
        data = _make_split_data(6)
        data["dynamic_global_batch_size"] = 6
        assert can_schedule_on_rollout_side(args, data, FULL_SCHEDULE_CONFIG)


class TestSplitTrainDataByDpScheduled:
    # [CLASS DOC] Tests the scheduled raw split, which also computes micro batch indices, micro
    # batch counts and rollout counts per step for each rank.
    def test_static_shards_cover_all_samples(self):
        # [FUNCTION DOC] Eight samples, micro batch size 2, static batching, data parallel 2: two
        # shards; each shard's tokens match its partition, it has one step with 2 micro batches and
        # 8 rollouts, its micro batch indices tile its rows exactly, and the partitions together
        # cover all eight samples once.
        """Static path: every sample lands in exactly one shard row, the schedule
        tiles each shard's rows exactly, and shard rows match their partition."""        """Static path: every sample lands in exactly one shard row, the schedule
        tiles each shard's rows exactly, and shard rows match their partition."""
        args = make_args(balance_data=False, micro_batch_size=2, use_dynamic_batch_size=False)
        data = _make_split_data(8)
        scheduled = split_train_data_by_dp_scheduled_raw(args, dict(data), train_parallel_config=FULL_SCHEDULE_CONFIG)

        assert len(scheduled) == 2
        seen = []
        for new in scheduled:
            partition = list(new["partition"])
            seen.extend(partition)
            assert new["tokens"] == [data["tokens"][j] for j in partition]
            # global_batch_size=8, dp=2, mbs=2 -> 4 mbs total, 2 per rank, 1 step
            assert new["num_microbatches"] == [2]
            assert new["num_rollouts"] == [8]
            flat = [i for mbs in new["micro_batch_indices"] for i in mbs]
            assert flat == list(range(len(new["tokens"])))
        assert sorted(seen) == list(range(8))

    def test_compact_rollout_gbs_counts_rollouts_not_samples(self):
        # [FUNCTION DOC] Six samples forming four rollouts (rollout 1 emits three samples), dynamic
        # batching with a token cap of 8 and a dynamic global batch size of 2: rank 0 has two steps
        # of 2 rollouts each, two micro batch counts, and all six samples are covered exactly once
        # across the shards.
        """gbs counts rollouts: 4 rollouts over 6 samples with gbs=2 -> 2 steps,
        and every sample is still covered exactly once across shards."""
        args = make_args(balance_data=False, micro_batch_size=1, use_dynamic_batch_size=True, max_tokens_per_gpu=8)
        rollout_ids = [0, 1, 1, 1, 2, 3]  # rollout 1 emits 3 samples
        data = _make_split_data(6, rollout_ids=rollout_ids)
        data["dynamic_global_batch_size"] = 2
        shards = split_train_data_by_dp_scheduled_raw(args, dict(data), train_parallel_config=FULL_SCHEDULE_CONFIG)

        assert shards[0]["num_rollouts"] == [2, 2]
        assert len(shards[0]["num_microbatches"]) == 2
        seen = sorted(j for shard in shards for j in shard["partition"])
        assert seen == list(range(6))

    def test_dynamic_schedule_respects_token_cap(self):
        # [FUNCTION DOC] Eight samples of varying lengths with a token cap of 6 per micro batch: all
        # ranks have identical micro batch counts, each rank's number of micro batches equals the sum
        # of the counts, and every micro batch's total length is within the cap unless it holds a
        # single sample.
        args = make_args(
            balance_data=False,
            micro_batch_size=1,
            use_dynamic_batch_size=True,
            max_tokens_per_gpu=6,
        )
        lengths = [5, 1, 4, 2, 3, 3, 2, 4]
        data = _make_split_data(8, lengths=lengths)
        shards = split_train_data_by_dp_scheduled_raw(args, data, train_parallel_config=FULL_SCHEDULE_CONFIG)

        nmb = shards[0]["num_microbatches"]
        for shard in shards:
            assert shard["num_microbatches"] == nmb, "num_microbatches must be identical on every rank"
            assert len(shard["micro_batch_indices"]) == sum(nmb)
            partition = list(shard["partition"])
            for mbs in shard["micro_batch_indices"]:
                total = sum(lengths[partition[i]] for i in mbs)
                assert total <= 6 or len(mbs) == 1

    def test_dynamic_gbs_multi_step(self):
        # [FUNCTION DOC] Sixteen samples, micro batch size 2, dynamic global batch size 8: rank 0
        # has micro batch counts 2 and 2, carries the dynamic global batch size 8, and has rollout
        # counts 8 and 8.
        """16 samples with dynamic_global_batch_size=8 -> 2 steps."""
        args = make_args(balance_data=False, micro_batch_size=2, use_dynamic_batch_size=False)
        data = _make_split_data(16)
        data["dynamic_global_batch_size"] = 8
        shards = split_train_data_by_dp_scheduled_raw(args, data, train_parallel_config=FULL_SCHEDULE_CONFIG)

        assert shards[0]["num_microbatches"] == [2, 2]
        assert shards[0]["dynamic_global_batch_size"] == 8
        assert shards[0]["num_rollouts"] == [8, 8]


def test_delayed_dp_split_preserves_the_tinker_loss_vectors():
    # [FUNCTION DOC] Packages four samples that carry loss weights, advantages and rollout log
    # probabilities into two shards (partitions 0, 2 and 1, 3) with no arguments object given.
    # Asserts every shard keeps those three keys with non-empty values and shard 0's advantages
    # hold two items.
    """The tinker losses zip loss_weights / advantages / rollout_log_probs per
    datum; a key missing from the shard whitelist disappears silently and only
    fails inside the trainer."""
    from miles.ray.rollout.train_data_conversion import _package_shards

    data = {
        "tokens": [[1], [2], [3], [4]],
        "loss_weights": [[1.0]] * 4,
        "advantages": [[0.5]] * 4,
        "rollout_log_probs": [[-0.1]] * 4,
    }
    shards = _package_shards(None, data, [[0, 2], [1, 3]])
    for shard in shards:
        assert shard["loss_weights"] and shard["advantages"] and shard["rollout_log_probs"]
    assert shards[0]["advantages"] == [[0.5], [0.5]]
