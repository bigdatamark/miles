# [MODULE DOC] CPU tests for reducing per-rank metric dictionaries into one: keys named in a
# reduction map use max or min across ranks (so extrema are not averaged), keys default to the
# mean, sum/count tuple values ignore the map and reduce to a total ratio, differing key sets raise
# a value error, unknown reduction names raise a value error, an empty gather yields an empty
# dictionary, and the multi-turn logging function passes its explicit extrema reduction map to the
# gather.
"""Tests for the per-key metric reduction in log_utils.gather_log_data.

Rank-local minima and maxima must be reduced as extrema across ranks.
Averaging them (the old behavior) systematically under-reports the global
maximum and over-reports the global minimum.
"""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.metrics import log_utils


def test_min_max_keys_reduce_to_global_extrema():
    # [FUNCTION DOC] Test: two ranks with response length mean/max/min (120/400/8 and 200/900/32);
    # with a map marking the max key as max and the min key as min, the mean key is averaged
    # (160), the max key is 900 and the min key is 8.
    gathered = [
        {
            "raw_response_length/response_length_mean": 120.0,
            "raw_response_length/response_length_max": 400.0,
            "raw_response_length/response_length_min": 8.0,
        },
        {
            "raw_response_length/response_length_mean": 200.0,
            "raw_response_length/response_length_max": 900.0,
            "raw_response_length/response_length_min": 32.0,
        },
    ]
    reduced = log_utils.reduce_gathered_log_dict(
        gathered,
        dp_size=2,
        reduction_by_key={
            "raw_response_length/response_length_max": "max",
            "raw_response_length/response_length_min": "min",
        },
    )

    assert reduced["raw_response_length/response_length_mean"] == pytest.approx(160.0)
    # The old mean reduction would report max=650.0 and min=20.0.
    assert reduced["raw_response_length/response_length_max"] == 900.0
    assert reduced["raw_response_length/response_length_min"] == 8.0


def test_keys_default_to_mean_without_reduction_map():
    # [FUNCTION DOC] Test: without a reduction map, a key whose name ends in max is still averaged
    # (2 and 4 give 3); names alone do not select the reduction.
    gathered = [
        {"custom_metric_max": 2.0},
        {"custom_metric_max": 4.0},
    ]

    reduced = log_utils.reduce_gathered_log_dict(gathered, dp_size=2)

    assert reduced == {"custom_metric_max": pytest.approx(3.0)}


def test_explicit_reduction_selects_extrema():
    # [FUNCTION DOC] Test: with an explicit max entry for a multi-turn round-number key, values 2
    # and 4 reduce to 4.
    gathered = [
        {"multi_turn_metric/round_number_max": 2.0},
        {"multi_turn_metric/round_number_max": 4.0},
    ]

    reduced = log_utils.reduce_gathered_log_dict(
        gathered,
        dp_size=2,
        reduction_by_key={"multi_turn_metric/round_number_max": "max"},
    )

    assert reduced == {"multi_turn_metric/round_number_max": 4.0}


def test_sum_count_tuples_ignore_reduction_map():
    # [FUNCTION DOC] Test: loss entries are (sum, count) tuples (6,2) and (2,2) giving a total
    # ratio of 2.0 (8 over 4), while a score key named in the map as max reduces to 5.0.
    gathered = [
        {"loss": (6.0, 2.0), "score_max": 1.0},
        {"loss": (2.0, 2.0), "score_max": 5.0},
    ]

    reduced = log_utils.reduce_gathered_log_dict(
        gathered,
        dp_size=2,
        reduction_by_key={"score_max": "max"},
    )

    assert reduced == {"loss": pytest.approx(2.0), "score_max": 5.0}


def test_mismatched_keys_across_ranks_raise():
    # [FUNCTION DOC] Test: when one rank has an extra key, reduction raises a value error saying
    # metric keys differ across ranks.
    gathered = [
        {"multi_turn_metric/round_number_max": 3.0},
        {"multi_turn_metric/round_number_max": 3.0, "extra": 1.0},
    ]

    with pytest.raises(ValueError, match="Metric keys differ across ranks"):
        log_utils.reduce_gathered_log_dict(gathered, dp_size=2, reduction_by_key={})


def test_unknown_reduction_name_raises():
    # [FUNCTION DOC] Test: a reduction named median raises a value error saying the metric
    # reduction is unsupported.
    gathered = [{"multi_turn_metric/round_number_max": 3.0}]

    with pytest.raises(ValueError, match="Unsupported metric reduction"):
        log_utils.reduce_gathered_log_dict(
            gathered,
            dp_size=1,
            reduction_by_key={"multi_turn_metric/round_number_max": "median"},
        )


def test_empty_gather_returns_empty_dict():
    # [FUNCTION DOC] Test: an empty list of gathered dictionaries reduces to an empty dictionary.
    assert log_utils.reduce_gathered_log_dict([], dp_size=2, reduction_by_key={}) == {}


def test_log_multi_turn_data_passes_explicit_extrema_reductions(monkeypatch):
    # [FUNCTION DOC] Test: with a fake parallel state (tensor-parallel rank 0, last pipeline stage)
    # and the gather function replaced by a recorder, logging multi-turn data for rollout 7 with
    # two samples (loss masks, round numbers 1 and 3) results in a gather call with metric name
    # "multi_turn", rollout id 7 and the module's constant multi-turn reduction map.
    captured = {}
    parallel_state = SimpleNamespace(tp=SimpleNamespace(rank=0), is_pp_last_stage=True)
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)

    def capture(metric_name, args, rollout_id, log_dict, reduction_by_key=None):
        # [FUNCTION DOC] HELPER: Recorder replacing the gather; stores metric name, rollout id, log
        # dictionary and reduction map in the captured dictionary.
        captured.update(
            metric_name=metric_name,
            rollout_id=rollout_id,
            log_dict=log_dict,
            reduction_by_key=reduction_by_key,
        )

    monkeypatch.setattr(log_utils, "gather_log_data", capture)

    log_utils.log_multi_turn_data(
        rollout_id=7,
        args=SimpleNamespace(rollout_max_response_len=8),
        rollout_data={
            "loss_masks": [torch.tensor([1, 1, 0]), torch.tensor([1, 1])],
            "round_number": [1, 3],
        },
    )

    assert captured["metric_name"] == "multi_turn"
    assert captured["rollout_id"] == 7
    assert captured["reduction_by_key"] == log_utils._MULTI_TURN_REDUCTION_BY_KEY
