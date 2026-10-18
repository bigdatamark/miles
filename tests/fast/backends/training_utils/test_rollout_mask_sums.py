# [MODULE DOC] CPU tests for per-rollout loss denominators: the rollout-side precomputation of
# mask sums when converting samples to train data (per-sample sums by default, siblings sharing a
# rollout id share the whole rollout total, masked-out tokens excluded), the sum-of-sample-mean
# reducer with and without precomputed denominators (legacy per-sample mean, token-weighted
# per-rollout mean, strict equivalence when one rollout is one sample), and the training loss
# aggregation's number-of-rollouts divisor. An autouse fixture installs a trivial parallel state.
"""CPU tests for per-rollout loss denominators (``rollout_mask_sums``).

Covers:
- the rollout-side precomputation in ``convert_samples_to_train_data``;
- ``get_sum_of_sample_mean(denominators=...)`` — legacy per-sample mean when
  ``None``, per-rollout token-weighted mean with precomputed denominators, and
  the strict no-op equivalence between the two when 1 rollout = 1 sample;
- ``aggregate_train_losses(num_rollouts=...)`` per-rollout-mean reduction.
"""

from __future__ import annotations

import math

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.metrics import log_utils
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data


@pytest.fixture(autouse=True)
def _parallel_state():
    # [FUNCTION DOC] HELPER: Autouse fixture; installs the trivial single-process parallel state.
    make_parallel_state()


def _convert(samples):
    # [FUNCTION DOC] HELPER: Converts samples to train data with shared test args (reward
    # normalization off), empty metadata and no custom conversion or reward post-processing hooks.
    # Returns the train data dictionary.
    return convert_samples_to_train_data(
        make_args(rewards_normalization=False),
        samples,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )


class TestRolloutMaskSumsPrecompute:
    # [CLASS DOC] Tests for the rollout mask sums produced by train-data conversion.
    def test_defaults_to_per_sample_mask_sums(self):
        # [FUNCTION DOC] Test: three samples with response lengths 2, 3, 4 (default all-ones masks)
        # give mask sums [2, 3, 4].
        samples = [make_sample(index=i, response_length=n) for i, n in enumerate((2, 3, 4))]
        out = _convert(samples)
        assert out["rollout_mask_sums"] == [2, 3, 4]

    def test_compact_siblings_share_whole_rollout_total(self):
        # [FUNCTION DOC] Test: samples 0 and 1 (lengths 2 and 3) share rollout id 0 (same rollout
        # execution), so both get total 5; the third keeps its own 4: [5, 5, 4].
        samples = [make_sample(index=i, response_length=n) for i, n in enumerate((2, 3, 4))]
        # samples 0 and 1 come from the same rollout execution
        samples[0].rollout_id = samples[1].rollout_id = 0
        out = _convert(samples)
        assert out["rollout_mask_sums"] == [5, 5, 4]

    def test_masked_out_tokens_excluded(self):
        # [FUNCTION DOC] Test: one sample of length 4 with loss mask [1,0,1,0] gives mask sum [2].
        s = make_sample(index=0, response_length=4)
        s.loss_mask = [1, 0, 1, 0]
        out = _convert([s])
        assert out["rollout_mask_sums"] == [2]


class TestDenominators:
    # [CLASS DOC] Tests for the sum-of-sample-mean reducer with optional denominators.
    def _masks(self, lens):
        # [FUNCTION DOC] HELPER: Returns all-ones int mask tensors, one per length.
        return [torch.ones(n, dtype=torch.int) for n in lens]

    def test_none_matches_legacy_per_sample_mean(self):
        # [FUNCTION DOC] Test: with lengths [2,3] and values [1,3,2,2,5] and no denominators, the
        # reducer returns the sum of per-sample means: 2 + 3 = 5.
        lens = [2, 3]
        x = torch.tensor([1.0, 3.0, 2.0, 2.0, 5.0])
        som = get_sum_of_sample_mean(list(lens), list(lens), self._masks(lens))
        # sample means: (1+3)/2 = 2, (2+2+5)/3 = 3
        assert math.isclose(som(x).item(), 5.0)

    def test_rollout_denoms_give_token_weighted_rollout_mean(self):
        # [FUNCTION DOC] Test: same data with denominators [5,5] (both samples in one rollout of 5
        # mask tokens) gives (1+3)/5 + (2+2+5)/5 = 13/5, a token-weighted rollout mean.
        lens = [2, 3]
        x = torch.tensor([1.0, 3.0, 2.0, 2.0, 5.0])
        # both samples belong to one rollout with 5 mask tokens total
        denoms = torch.tensor([5.0, 5.0])
        som = get_sum_of_sample_mean(list(lens), list(lens), self._masks(lens), denominators=denoms)
        # (1+3)/5 + (2+2+5)/5 = 13/5 — one token-weighted mean for the rollout
        assert math.isclose(som(x).item(), 13.0 / 5, rel_tol=1e-6)

    def test_per_sample_denominators_equal_legacy(self):
        # [FUNCTION DOC] Test: with random values and denominators equal to each sample's own mask
        # sum (lengths 2,3,4), the reducer's result is close to the legacy no-denominator result.
        """1 rollout = 1 sample: rollout_mask_sums equals each mask's own sum -> strict no-op."""
        lens = [2, 3, 4]
        x = torch.randn(sum(lens))
        masks = self._masks(lens)
        legacy = get_sum_of_sample_mean(list(lens), list(lens), masks)
        denoms = torch.tensor([float(n) for n in lens])
        new = get_sum_of_sample_mean(list(lens), list(lens), masks, denominators=denoms)
        assert torch.allclose(legacy(x), new(x))


class TestAggregateTrainLossesNumRollouts:
    # [CLASS DOC] Tests for the training loss aggregation's divisor.
    @pytest.fixture(autouse=True)
    def _no_allreduce(self, monkeypatch):
        # [FUNCTION DOC] HELPER: Autouse fixture; replaces the cross-rank all-reduce utility with a
        # no-op so aggregation runs in a single process.
        monkeypatch.setattr(log_utils.MultiPGUtil, "all_reduce", staticmethod(lambda *a, **k: None))

    def _mb(self, num_samples, values):
        # [FUNCTION DOC] HELPER: Builds a micro-batch log dictionary with the metric names as keys
        # and a values tensor whose first entry is the sample count followed by the metric values.
        return {"keys": list(values.keys()), "values": torch.tensor([num_samples, *values.values()])}

    def test_num_rollouts_overrides_sample_count(self):
        # [FUNCTION DOC] Test: two micro-batches (2 samples each, loss 2.0 and 6.0) with number of
        # rollouts 4 aggregate to loss 2.0 (total 8 divided by the given rollout count).
        losses = [self._mb(2, {"loss": 2.0}), self._mb(2, {"loss": 6.0})]
        out = log_utils.aggregate_train_losses(losses, num_rollouts=4)
        assert math.isclose(out["loss"], 2.0)

    def test_legacy_reduction_without_divisor(self):
        # [FUNCTION DOC] Test: the same micro-batches without a rollout count aggregate to 2.0
        # (divided by the total sample count, 4).
        # KNOWN ISSUE: both this test and the rollout-count test above use numbers that give the
        # same result (2.0), so the rollout-count test cannot distinguish the two divisors.
        losses = [self._mb(2, {"loss": 2.0}), self._mb(2, {"loss": 6.0})]
        out = log_utils.aggregate_train_losses(losses)
        assert math.isclose(out["loss"], 2.0)
