# [MODULE DOC]
# Tests for data-parallel-attention engine series in the metric store: samples labelled with a
# data-parallel rank are folded into one engine-level series (summed for extensive gauges and
# counters, averaged for intensive gauges, count-weighted for histogram means), or kept
# separate per rank on request. Also covers legacy dumps with unlabelled same-timestamp
# duplicates. All data is written to a temporary store directory and reloaded.
"""dp-attention engine series: per-rank samples fold into engine-level values
by default (sum, or mean for intensive metrics) and stay apart under
``per_dp_rank=True``."""

from pathlib import Path

from miles.dashboard.store import EngineSample, Meta, MetricStore

ADDR = "http://e:1"


def make_store(tmp_path: Path, samples: list[EngineSample]) -> MetricStore:
    # [FUNCTION DOC] HELPER: Creates a store under the temporary path, writes metadata (run name
    # "t", start 0, no args), appends all samples, flushes, and returns a freshly loaded store.
    store = MetricStore(tmp_path / "dashboard")
    store.write_meta(Meta(run_name="t", start_ts=0.0, args={}))
    for sample in samples:
        store.append(sample)
    store.flush()
    return MetricStore.load(tmp_path / "dashboard")


def dp_samples(metric, points_by_rank):
    # [FUNCTION DOC] HELPER: Expands a mapping from data-parallel rank string to (timestamp,
    # value) pairs into engine samples for the shared test engine address, with the rank stored
    # as the dp_rank label. Returns the flat list in rank then point order.
    return [
        EngineSample(ts=ts, addr=ADDR, metric=metric, labels={"dp_rank": rank}, value=v)
        for rank, points in points_by_rank.items()
        for ts, v in points
    ]


def test_gauge_sums_across_dp_ranks(tmp_path):
    # [FUNCTION DOC] Running-request gauges for ranks 0 and 1 give one engine series with the
    # engine address, empty labels, timestamps [1, 2] and values summed per timestamp [12, 3].
    store = make_store(
        tmp_path,
        dp_samples("sglang_num_running_reqs", {"0": [(1.0, 11.0), (2.0, 3.0)], "1": [(1.0, 1.0), (2.0, 0.0)]}),
    )
    (series,) = store.engine_series("sglang_num_running_reqs")
    assert series["addr"] == ADDR
    assert series["labels"] == {}
    assert series["ts"] == [1.0, 2.0]
    assert series["value"] == [12.0, 3.0]


def test_intensive_gauge_averages_across_dp_ranks(tmp_path):
    # [FUNCTION DOC] Token usage (an intensive gauge) of 0.9 and 0.1 on two ranks folds to their
    # mean, 0.5.
    store = make_store(tmp_path, dp_samples("sglang_token_usage", {"0": [(1.0, 0.9)], "1": [(1.0, 0.1)]}))
    (series,) = store.engine_series("sglang_token_usage")
    assert series["value"] == [0.5]


def test_per_dp_rank_splits_series(tmp_path):
    # [FUNCTION DOC] With the per-rank option on, the query returns one series per rank, keyed by
    # the dp_rank label values "0" and "1", each with the engine address and its own value.
    store = make_store(
        tmp_path,
        dp_samples("sglang_num_running_reqs", {"0": [(1.0, 11.0)], "1": [(1.0, 1.0)]}),
    )
    series = store.engine_series("sglang_num_running_reqs", per_dp_rank=True)
    by_rank = {s["labels"]["dp_rank"]: s for s in series}
    assert set(by_rank) == {"0", "1"}
    assert all(s["addr"] == ADDR for s in series)
    assert by_rank["0"]["value"] == [11.0]
    assert by_rank["1"]["value"] == [1.0]


def test_legacy_folded_duplicates_aggregate(tmp_path):
    # [FUNCTION DOC] Legacy dumps hold several same-timestamp samples with identical (empty)
    # labels: values 11 and 1 at t=1, 3 and 0 at t=2. The query sums them per timestamp:
    # timestamps [1, 2], values [12, 3].
    # dumps written before dp_rank was kept: N same-ts samples, identical labels
    samples = [
        EngineSample(ts=ts, addr=ADDR, metric="sglang_num_running_reqs", labels={}, value=v)
        for ts, v in [(1.0, 11.0), (1.0, 1.0), (2.0, 3.0), (2.0, 0.0)]
    ]
    store = make_store(tmp_path, samples)
    (series,) = store.engine_series("sglang_num_running_reqs")
    assert series["ts"] == [1.0, 2.0]
    assert series["value"] == [12.0, 3.0]


def test_counter_rate_sums_ranks_before_diff(tmp_path):
    # [FUNCTION DOC] Per-rank token counters rise 0 to 100 and 0 to 60 over 2 seconds. The
    # engine-level rate series sums ranks before differencing: one point at t=2 with value 80.
    # Per-rank mode yields rates 50 and 30 keyed by rank label.
    # rank counters: 0 -> 100 and 0 -> 60 over 2s; engine rate must be the
    # total 80/s, not a cross-rank interleave (which would leave gaps/garbage)
    store = make_store(
        tmp_path,
        dp_samples(
            "sglang_generation_tokens_total",
            {"0": [(0.0, 0.0), (2.0, 100.0)], "1": [(0.0, 0.0), (2.0, 60.0)]},
        ),
    )
    (series,) = store.engine_series("sglang_generation_tokens_per_s")
    assert series["ts"] == [2.0]
    assert series["value"] == [80.0]

    per_rank = store.engine_series("sglang_generation_tokens_per_s", per_dp_rank=True)
    assert {s["labels"]["dp_rank"]: s["value"] for s in per_rank} == {"0": [50.0], "1": [30.0]}


def test_histogram_mean_weights_ranks_by_count(tmp_path):
    # [FUNCTION DOC] Rank 0 has 4 completions totalling 8 seconds, rank 1 has 1 completion of 3
    # seconds. The engine-level time-to-first-token mean is 11/5 (total sum over total count),
    # not the mean of the per-rank means, at the single timestamp 2.
    # rank 0: 4 completions totalling 8s; rank 1: 1 completion of 3s
    # engine-level ttft mean = 11s / 5 completions, not mean-of-means
    samples = dp_samples(
        "sglang_time_to_first_token_seconds_sum", {"0": [(0.0, 0.0), (2.0, 8.0)], "1": [(0.0, 0.0), (2.0, 3.0)]}
    ) + dp_samples(
        "sglang_time_to_first_token_seconds_count", {"0": [(0.0, 0.0), (2.0, 4.0)], "1": [(0.0, 0.0), (2.0, 1.0)]}
    )
    store = make_store(tmp_path, samples)
    (series,) = store.engine_series("sglang_ttft_mean_s")
    assert series["ts"] == [2.0]
    assert series["value"] == [11.0 / 5.0]
