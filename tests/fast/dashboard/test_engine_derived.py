# [MODULE DOC]
# Tests for derived engine series in the metric store: per-second rates computed from monotonic
# counters (gaps at counter resets), means computed from histogram sum and count families (gaps
# for idle intervals), and thread safety / failure behaviour of the full-window cache build.
"""Derived engine series: counter rates and histogram means."""

import threading
import time
from pathlib import Path

import pytest

from miles.dashboard.store import EngineSample, Meta, MetricStore, Stream


def make_store(tmp_path: Path, samples: list[EngineSample]) -> MetricStore:
    # [FUNCTION DOC] HELPER: Creates a store under the temporary path, writes metadata (run name
    # "t", start 0, no args), appends all samples, flushes, and returns a freshly loaded store
    # from disk.
    store = MetricStore(tmp_path / "dashboard")
    store.write_meta(Meta(run_name="t", start_ts=0.0, args={}))
    for sample in samples:
        store.append(sample)
    store.flush()
    return MetricStore.load(tmp_path / "dashboard")


def counter(addr, points, metric="sglang_generation_tokens_total"):
    # [FUNCTION DOC] HELPER: Converts (timestamp, value) pairs into engine samples for the given
    # engine address and metric name (default generation tokens total) with empty labels.
    return [EngineSample(ts=ts, addr=addr, metric=metric, labels={}, value=v) for ts, v in points]


def test_counter_rate_and_catalog(tmp_path):
    # [FUNCTION DOC] With counter points (0,0), (2,100), (4,300): the metric catalog lists the
    # derived per-second name and hides the raw total; the derived series has timestamps [2, 4]
    # and rates [50, 100].
    store = make_store(tmp_path, counter("http://e:1", [(0.0, 0.0), (2.0, 100.0), (4.0, 300.0)]))
    names = store.engine_metric_names()
    assert "sglang_generation_tokens_per_s" in names
    assert "sglang_generation_tokens_total" not in names

    (series,) = store.engine_series("sglang_generation_tokens_per_s")
    assert series["ts"] == [2.0, 4.0]
    assert series["value"] == [50.0, 100.0]


def test_counter_reset_leaves_gap(tmp_path):
    # [FUNCTION DOC] Counter values 0, 100, 20, 40 at times 0, 2, 4, 6: the interval ending at 4
    # (a reset) is dropped, giving timestamps [2, 6] and rates [50, 10].
    store = make_store(tmp_path, counter("http://e:1", [(0.0, 0.0), (2.0, 100.0), (4.0, 20.0), (6.0, 40.0)]))
    (series,) = store.engine_series("sglang_generation_tokens_per_s")
    assert series["ts"] == [2.0, 6.0]  # the reset interval is a gap, not a negative rate
    assert series["value"] == [50.0, 10.0]


def test_histogram_mean_skips_idle_intervals(tmp_path):
    # [FUNCTION DOC] Time-to-first-token sum goes 0, 10, 10 and count 0, 5, 5 at times 0, 2, 4.
    # The mean metric is in the catalog; its series has one point at time 2 with value 2.0 (sum
    # delta over count delta); the idle interval ending at 4 is a gap.
    samples = counter(
        "http://e:1", [(0.0, 0.0), (2.0, 10.0), (4.0, 10.0)], metric="sglang_time_to_first_token_seconds_sum"
    ) + counter("http://e:1", [(0.0, 0.0), (2.0, 5.0), (4.0, 5.0)], metric="sglang_time_to_first_token_seconds_count")
    store = make_store(tmp_path, samples)
    assert "sglang_ttft_mean_s" in store.engine_metric_names()

    (series,) = store.engine_series("sglang_ttft_mean_s")
    assert series["ts"] == [2.0]  # no completions in [2, 4): gap
    assert series["value"] == [2.0]


def test_mean_requires_both_families(tmp_path):
    # [FUNCTION DOC] With only the sum family stored, the derived mean metric is absent from the
    # catalog and its series query returns an empty list.
    store = make_store(
        tmp_path, counter("http://e:1", [(0.0, 0.0), (2.0, 10.0)], metric="sglang_time_to_first_token_seconds_sum")
    )
    assert "sglang_ttft_mean_s" not in store.engine_metric_names()
    assert store.engine_series("sglang_ttft_mean_s") == []


def test_full_window_cache_survives_concurrent_cold_readers(tmp_path):
    # [FUNCTION DOC] Wraps the engine series partition reader's window method to sleep 0.2 s
    # before delegating, then starts 8 threads that concurrently query the derived series on a
    # cold cache. Asserts no thread raised and all results are identical with a single rate
    # value 50.0.
    store = make_store(tmp_path, counter("http://e:1", [(0.0, 0.0), (2.0, 100.0)]))
    reader = store._readers[Stream.ENGINE_SERIES]
    real_window = reader.window
    reader.window = lambda t0, t1: (time.sleep(0.2), real_window(t0, t1))[1]

    results, errors = [], []
    threads = [threading.Thread(target=lambda: _collect(store, results, errors), name=f"reader-{i}") for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert all(r == results[0] and r[0]["value"] == [50.0] for r in results)


def _collect(store, results, errors):
    # [FUNCTION DOC] HELPER: Thread body. Queries the derived tokens-per-second series and appends
    # the result to the shared results list; any exception is appended to the errors list instead.
    try:
        results.append(store.engine_series("sglang_generation_tokens_per_s"))
    except Exception as e:  # noqa: BLE001
        errors.append(e)


def test_failed_rebuild_does_not_look_cached(tmp_path):
    # [FUNCTION DOC] Makes the partition reader's window method raise; the first series query
    # raises an OS error. After restoring the real method, a second query succeeds with value
    # [50.0], proving the failed build was not cached.
    store = make_store(tmp_path, counter("http://e:1", [(0.0, 0.0), (2.0, 100.0)]))
    reader = store._readers[Stream.ENGINE_SERIES]
    real_window = reader.window
    reader.window = _raise

    with pytest.raises(OSError):
        store.engine_series("sglang_generation_tokens_per_s")
    reader.window = real_window
    (series,) = store.engine_series("sglang_generation_tokens_per_s")
    assert series["value"] == [50.0]


def _raise(t0, t1):
    # [FUNCTION DOC] HELPER: Stand-in for the window method; ignores its range and raises an OS
    # error saying the partition read failed.
    raise OSError("partition read failed")
