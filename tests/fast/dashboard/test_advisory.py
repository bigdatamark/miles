# [MODULE DOC]
# CPU tests of the dashboard advisory engine (compute_advisories), which reads a loaded metric store
# (and optionally a dump reader) and returns advisories with a level (info, warning, critical) and a
# message. Stores are built in temporary directories from engine samples, phase events, metric
# records and metadata arguments. Covered rules: nothing without engine series; low concurrency
# versus the configured max running requests (info); low cache hit rate (info, not for colocated
# runs); high token usage (warning); window narrowing by time range; data-parallel imbalance (warning,
# message names the applicable router knob; balanced or idle engines are not flagged); stall
# detection from an open phase marker far older than the median closed duration (critical, which
# suppresses tuning advice; not a stall when a closing twin exists, when the marker belongs to a
# previous attempt, or when there is no closed baseline); abort storms from the abort ratio over
# counter deltas that survive counter resets (warning; only while current; ignored at low volume);
# KV-pool-bound concurrency advice; reward-based rules from a stub reader (zero-std groups, all-zero
# rewards, null rewards, truncation); and the sustained low model-flops-utilization rule with a
# configurable threshold (first step excluded, minimum steady steps, disabled by zero). The last test
# checks a blank newest step does not break the rollout advisories (uses the dummy dump generator).
import polars as pl
from tests.fast.dashboard.dummy_dump import blank_samples, dump_dummy_run

from miles.dashboard.advisory import compute_advisories
from miles.dashboard.dump_reader import DumpReader, RolloutIds
from miles.dashboard.store import EngineSample, Meta, MetricsRecord, MetricStore, PhaseEvent, Stream


def _store(tmp_path, *, args: dict, engine_samples: list[EngineSample]) -> MetricStore:
    # [FUNCTION DOC] HELPER: Writes metadata (run name "advisory-test", start time zero, the given
    # argument mapping) and the engine samples to a metric store in the directory, flushes, and
    # returns the store freshly loaded from disk.
    writer = MetricStore(tmp_path)
    writer.write_meta(Meta(run_name="advisory-test", start_ts=0.0, args=args))
    for sample in engine_samples:
        writer.append(sample)
    writer.flush()
    return MetricStore.load(tmp_path)


def _engine(metric: str, value: float, ts: float = 1.0, addr: str = "http://n:1") -> EngineSample:
    # [FUNCTION DOC] HELPER: Builds an engine sample for the metric and value at time ts (default 1.0)
    # and address (default "http://n:1") with no labels.
    return EngineSample(ts=ts, addr=addr, metric=metric, labels={}, value=value)


def test_no_engine_series_returns_empty(tmp_path):
    # [FUNCTION DOC] A store with no engine samples yields no advisories.
    store = _store(tmp_path, args={}, engine_samples=[])
    assert compute_advisories(store) == []


def test_low_concurrency_flagged(tmp_path):
    # [FUNCTION DOC] With max running requests 100 and observed running requests 20, exactly one
    # info advisory is produced and its message mentions the max-running-requests option.
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 100},
        engine_samples=[_engine("sglang_num_running_reqs", 20.0)],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "info"
    assert "max-running-requests" in advisory.message


def test_high_concurrency_not_flagged(tmp_path):
    # [FUNCTION DOC] With max running requests 100 and observed 80, no advisory.
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 100},
        engine_samples=[_engine("sglang_num_running_reqs", 80.0)],
    )
    assert compute_advisories(store) == []


def test_low_cache_hit_flagged_when_not_colocate(tmp_path):
    # [FUNCTION DOC] With colocate off, memory fraction 0.7 and cache hit rate 0.02, one info
    # advisory mentioning mem-fraction-static is produced.
    store = _store(
        tmp_path,
        args={"colocate": False, "sglang_mem_fraction_static": 0.7},
        engine_samples=[_engine("sglang_cache_hit_rate", 0.02)],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "info"
    assert "mem-fraction-static" in advisory.message


def test_low_cache_hit_not_flagged_when_colocate(tmp_path):
    # [FUNCTION DOC] The same low cache hit rate with colocate on produces no advisory.
    # colocate runs deliberately trade cache size for training memory — a low
    # hit rate there is an expected cost, not a misconfiguration to flag
    store = _store(
        tmp_path,
        args={"colocate": True, "sglang_mem_fraction_static": 0.5},
        engine_samples=[_engine("sglang_cache_hit_rate", 0.02)],
    )
    assert compute_advisories(store) == []


def test_high_token_usage_flagged(tmp_path):
    # [FUNCTION DOC] Token usage 0.99 yields one warning advisory whose message mentions throughput.
    store = _store(
        tmp_path,
        args={},
        engine_samples=[_engine("sglang_token_usage", 0.99)],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "warning"
    assert "throughput" in advisory.message


def test_healthy_run_has_no_advisories(tmp_path):
    # [FUNCTION DOC] Colocate off, running requests 80 of 100, cache hit rate 0.6 and token usage 0.5
    # together yield no advisories.
    store = _store(
        tmp_path,
        args={"colocate": False, "sglang_max_running_requests": 100, "sglang_mem_fraction_static": 0.7},
        engine_samples=[
            _engine("sglang_num_running_reqs", 80.0),
            _engine("sglang_cache_hit_rate", 0.6),
            _engine("sglang_token_usage", 0.5),
        ],
    )
    assert compute_advisories(store) == []


def test_window_narrows_to_requested_range(tmp_path):
    # [FUNCTION DOC] Samples: running requests 20 at time 1 and 90 at time 100 (max 100). The window
    # 0 to 10 sees only the low sample and yields advisories; the window 50 to 150 sees only the high
    # sample and yields none.
    # a stale spike outside [t0, t1] must not leak into the windowed view
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 100},
        engine_samples=[
            _engine("sglang_num_running_reqs", 20.0, ts=1.0),
            _engine("sglang_num_running_reqs", 90.0, ts=100.0),
        ],
    )
    assert compute_advisories(store, t0=0.0, t1=10.0) != []
    assert compute_advisories(store, t0=50.0, t1=150.0) == []


def _dp_engine(rank: str, value: float, ts: float = 1.0) -> EngineSample:
    # [FUNCTION DOC] HELPER: Builds a running-requests sample for engine "http://n:1" labeled with the
    # data-parallel rank string.
    return EngineSample(
        ts=ts, addr="http://n:1", metric="sglang_num_running_reqs", labels={"dp_rank": rank}, value=value
    )


def _imbalanced(tmp_path, args: dict) -> str:
    # [FUNCTION DOC] HELPER: Builds a store with four data-parallel ranks running 11, 0.5, 0, and 1
    # requests (heavily skewed) and the given arguments, asserts exactly one warning advisory saying
    # the data-parallel ranks are imbalanced, and returns its message.
    store = _store(
        tmp_path,
        args=args,
        engine_samples=[_dp_engine("0", 11.0), _dp_engine("1", 0.5), _dp_engine("2", 0.0), _dp_engine("3", 1.0)],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "warning"
    assert "dp ranks imbalanced" in advisory.message
    return advisory.message


def test_dp_imbalance_names_the_knob_that_applies(tmp_path):
    # [FUNCTION DOC] The imbalance message depends on the arguments: with none it names the router
    # data-parallel-aware option; with the miles router in use it names the SGLang load balance
    # method option; with a data-parallel-aware router and policy cache_aware it names the router
    # policy; when an SGLang router policy is also set, miles overrides the router policy, so the
    # message names the router assignment mode (random) instead.
    assert "--router-dp-aware" in _imbalanced(tmp_path / "a", {})
    assert "--sglang-load-balance-method" in _imbalanced(tmp_path / "b", {"use_miles_router": True})
    assert "--router-policy (cache_aware)" in _imbalanced(
        tmp_path / "c", {"router_dp_aware": True, "router_policy": "cache_aware"}
    )
    # miles overrides router_policy with sglang_router_policy when both are set
    assert "--router-assignment-mode (random)" in _imbalanced(
        tmp_path / "d",
        {
            "router_dp_aware": True,
            "router_policy": "cache_aware",
            "sglang_router_policy": "manual",
            "router_assignment_mode": "random",
        },
    )


def test_dp_balanced_not_flagged(tmp_path):
    # [FUNCTION DOC] Four ranks with 5, 4, 6 and 5 running requests yield no advisories.
    store = _store(
        tmp_path,
        args={},
        engine_samples=[_dp_engine("0", 5.0), _dp_engine("1", 4.0), _dp_engine("2", 6.0), _dp_engine("3", 5.0)],
    )
    assert compute_advisories(store) == []


def test_dp_idle_engine_not_flagged(tmp_path):
    # [FUNCTION DOC] Two ranks with 0.5 and 0 running requests yield no advisories.
    # everything near zero (drained engine): no load, no imbalance signal
    store = _store(
        tmp_path,
        args={},
        engine_samples=[_dp_engine("0", 0.5), _dp_engine("1", 0.0)],
    )
    assert compute_advisories(store) == []


# ------------------------- v2: health alarms + gating -------------------------


def _phase(name, t0, t1, rank=0):
    # [FUNCTION DOC] HELPER: Builds a training-role phase event on node "n1", GPU 0, with the given
    # name, start, end and rank (default 0).
    return PhaseEvent(name=name, t0=t0, t1=t1, node="n1", gpus=[0], rank=rank, role="train")


def _closed_steps(n=3, duration=10.0, start=0.0):
    # [FUNCTION DOC] HELPER: Returns n (default 3) closed train-wait phase events, each lasting
    # duration seconds (default 10), spaced 100 seconds apart beginning at start (default 0).
    return [_phase("train_wait", start + i * 100.0, start + i * 100.0 + duration) for i in range(n)]


def test_stalled_open_phase_is_critical_and_suppresses_tuning(tmp_path):
    # [FUNCTION DOC] Engine samples at times 1 and 4200 show low concurrency (5 running of max 100);
    # three closed 10-second train-wait phases plus an open train-wait marker starting at 300 are then
    # appended to the same store directory, which is reloaded. Expects exactly one critical advisory
    # mentioning train_wait and stalled (tuning advice suppressed). KNOWN ISSUE: the first store built
    # by the helper is overwritten by the reload before use (dead code); the metadata written by the
    # helper is what the reload reads.
    # the open train_wait is 3900s old vs a 10s median: a stall — and the
    # low-concurrency observation must NOT surface as tuning advice with it
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 100},
        engine_samples=[
            _engine("sglang_num_running_reqs", 5.0, ts=1.0),
            _engine("sglang_num_running_reqs", 5.0, ts=4200.0),
        ],
    )
    writer = MetricStore(tmp_path)
    for event in [*_closed_steps(), _phase("train_wait", 300.0, PhaseEvent.OPEN_T1)]:
        writer.append(event)
    writer.flush()
    store = MetricStore.load(tmp_path)

    [advisory] = compute_advisories(store)
    assert advisory.level == "critical"
    assert "train_wait" in advisory.message
    assert "stalled" in advisory.message


def test_open_phase_with_closing_twin_is_not_a_stall(tmp_path):
    # [FUNCTION DOC] Three closed steps, an open train-wait marker at 300 and a closed twin for the
    # same phase (300 to 310), plus one healthy engine sample at 4200 and default metadata: no
    # advisories.
    writer = MetricStore(tmp_path)
    open_marker = _phase("train_wait", 300.0, PhaseEvent.OPEN_T1)
    for event in [*_closed_steps(), open_marker, _phase("train_wait", 300.0, 310.0)]:
        writer.append(event)
    writer.append(_engine("sglang_num_running_reqs", 80.0, ts=4200.0))
    writer.write_meta(Meta(run_name="advisory-test", start_ts=0.0, args={}))
    writer.flush()
    assert compute_advisories(MetricStore.load(tmp_path)) == []


def test_previous_attempt_open_marker_is_not_a_stall(tmp_path):
    # [FUNCTION DOC] An earlier attempt's open marker (never closed), then a new attempt (metadata
    # start time 10000) with its own closed steps and a healthy engine sample at 20000: the old marker
    # is ignored and there are no advisories.
    # resuming into the same dump dir appends: the crashed attempt's open
    # markers are still in the stream and never close, and their age against
    # this attempt's clock would latch a critical for the rest of the run
    writer = MetricStore(tmp_path)
    for event in [*_closed_steps(), _phase("train_wait", 300.0, PhaseEvent.OPEN_T1)]:
        writer.append(event)
    for event in _closed_steps(start=10000.0):
        writer.append(event)
    writer.append(_engine("sglang_num_running_reqs", 80.0, ts=20000.0))
    writer.write_meta(Meta(run_name="advisory-test", start_ts=10000.0, args={}))
    writer.flush()
    assert compute_advisories(MetricStore.load(tmp_path)) == []


def test_forever_open_phase_without_baseline_is_not_a_stall(tmp_path):
    # [FUNCTION DOC] A single open rollout phase from time 0 with no closed instances, and an engine
    # sample at time 90000: no baseline, so no stall advisory.
    # fully-async rollout keeps one manager phase open for the whole run by
    # design; with no closed instances there is no baseline and no claim
    writer = MetricStore(tmp_path)
    writer.append(_phase("rollout", 0.0, PhaseEvent.OPEN_T1))
    writer.append(_engine("sglang_num_running_reqs", 80.0, ts=90000.0))
    writer.write_meta(Meta(run_name="advisory-test", start_ts=0.0, args={}))
    writer.flush()
    assert compute_advisories(MetricStore.load(tmp_path)) == []


def test_abort_storm_is_warning_and_suppresses_tuning(tmp_path):
    # [FUNCTION DOC] Request counter rising 0 to 100 and abort counter 0 to 40 (40 percent), with low
    # concurrency and 8 samples per prompt: exactly one warning advisory mentioning 40% and the
    # 8-sample group (the low-concurrency tuning advice is suppressed).
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 100, "n_samples_per_prompt": 8},
        engine_samples=[
            _engine("sglang_num_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_requests_total", 100.0, ts=2.0),
            _engine("sglang_num_aborted_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_aborted_requests_total", 40.0, ts=2.0),
            _engine("sglang_num_running_reqs", 5.0, ts=1.0),
        ],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "warning"
    assert "40%" in advisory.message
    assert "8-sample group" in advisory.message


def test_abort_storm_survives_an_engine_restart(tmp_path):
    # [FUNCTION DOC] Counters reset mid-series: requests 0, 1000, 0, 500 and aborts 0, 900, 0, 0.
    # Summing positive deltas per segment gives 900 aborts of 1500 requests: one warning advisory
    # mentioning 60% and "(900/1500".
    # an engine coming back on the same address resets its counters mid-series;
    # last-minus-first would see only the healthy post-restart segment and drop
    # the storm entirely
    store = _store(
        tmp_path,
        args={},
        engine_samples=[
            _engine("sglang_num_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_requests_total", 1000.0, ts=2.0),
            _engine("sglang_num_requests_total", 0.0, ts=3.0),
            _engine("sglang_num_requests_total", 500.0, ts=4.0),
            _engine("sglang_num_aborted_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_aborted_requests_total", 900.0, ts=2.0),
            _engine("sglang_num_aborted_requests_total", 0.0, ts=3.0),
            _engine("sglang_num_aborted_requests_total", 0.0, ts=4.0),
        ],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "warning"
    assert "60%" in advisory.message
    assert "(900/1500" in advisory.message


def test_cleared_abort_storm_stops_alarming(tmp_path):
    # [FUNCTION DOC] Aborts reached 900 early (of 1000 requests) and then stayed flat while requests
    # grew to 9000 over hours, with low concurrency at the end: the storm no longer alarms and the
    # single advisory is the info-level tuning advice.
    # 900/9000 over the run stays above the threshold forever, but the engine
    # has been healthy for hours: the alarm — and the tuning tier it suppresses
    # — must follow the current state
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 100},
        engine_samples=[
            _engine("sglang_num_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_requests_total", 1000.0, ts=100.0),
            _engine("sglang_num_requests_total", 5000.0, ts=9000.0),
            _engine("sglang_num_requests_total", 9000.0, ts=10000.0),
            _engine("sglang_num_aborted_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_aborted_requests_total", 900.0, ts=100.0),
            _engine("sglang_num_aborted_requests_total", 900.0, ts=9000.0),
            _engine("sglang_num_aborted_requests_total", 900.0, ts=10000.0),
            _engine("sglang_num_running_reqs", 5.0, ts=10000.0),
        ],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "info"  # the storm is history; tuning advice is back


def test_low_volume_window_is_not_an_abort_storm(tmp_path):
    # [FUNCTION DOC] 6 requests with 1 abort (about 17 percent) is too little volume: no advisories.
    # an idle window: 6 requests, 1 aborted is 17% but says nothing
    store = _store(
        tmp_path,
        args={},
        engine_samples=[
            _engine("sglang_num_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_requests_total", 6.0, ts=2.0),
            _engine("sglang_num_aborted_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_aborted_requests_total", 1.0, ts=2.0),
        ],
    )
    assert compute_advisories(store) == []


def test_kv_bound_concurrency_names_the_real_bottleneck(tmp_path):
    # [FUNCTION DOC] Token usage 0.96 with only 14 running requests against a cap of 256: one warning
    # advisory saying the run is KV-pool-bound and telling the user not to lower the cap.
    # token usage pegged while concurrency sits far under the cap: the pool is
    # the limit — the advice must be "raise mem-fraction", never "lower the cap"
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 256, "sglang_mem_fraction_static": 0.75},
        engine_samples=[
            _engine("sglang_token_usage", 0.96, ts=1.0),
            _engine("sglang_num_running_reqs", 14.0, ts=1.0),
        ],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "warning"
    assert "KV-pool-bound" in advisory.message
    assert "do NOT lower" in advisory.message


class _StubReader:
    # [CLASS DOC] HELPER: Minimal stand-in for the dump reader returning fixed tables regardless of
    # the rollout id: a groups table and a per-sample summary table, with train rollout ids [3] and no
    # evaluation ids.
    def __init__(self, groups_df, summary_df):
        # [FUNCTION DOC] HELPER: Stores the groups and summary tables.
        self._groups = groups_df
        self._summary = summary_df

    def rollout_ids(self):
        # [FUNCTION DOC] HELPER: Returns rollout ids with a single train id 3 and no eval ids.
        return RolloutIds(train=[3], eval=[])

    def groups(self, rollout_id, *, evaluation=False):
        # [FUNCTION DOC] HELPER: Returns the stored groups table, ignoring the arguments.
        return self._groups

    def summary(self, rollout_id, *, evaluation=False):
        # [FUNCTION DOC] HELPER: Returns the stored summary table, ignoring the arguments.
        return self._summary


def _summary_df(truncated):
    # [FUNCTION DOC] HELPER: Builds a one-column table named truncated from the list of booleans.
    return pl.DataFrame({"truncated": truncated})


def test_zero_std_groups_are_warning(tmp_path):
    # [FUNCTION DOC] Two of three groups have zero reward standard deviation (mean rewards 0, 0, 0.5)
    # and no sample is truncated: one warning advisory mentioning zero reward std.
    store = _store(tmp_path, args={}, engine_samples=[])
    groups = pl.DataFrame({"reward_mean": [0.0, 0.0, 0.5], "zero_std": [True, True, False]})
    [advisory] = compute_advisories(store, _StubReader(groups, _summary_df([False] * 8)))
    assert advisory.level == "warning"
    assert "zero reward std" in advisory.message


def test_all_zero_rewards_note_systematic_failure(tmp_path):
    # [FUNCTION DOC] When every group has zero std and mean reward zero, the advisory message notes a
    # systematic failure.
    store = _store(tmp_path, args={}, engine_samples=[])
    groups = pl.DataFrame({"reward_mean": [0.0, 0.0], "zero_std": [True, True]})
    [advisory] = compute_advisories(store, _StubReader(groups, _summary_df([False] * 8)))
    assert "systematic failure" in advisory.message


def test_null_rewards_are_missing_data_not_zero_std(tmp_path):
    # [FUNCTION DOC] Groups whose mean reward is null (float column) and not flagged zero-std produce
    # no advisories.
    store = _store(tmp_path, args={}, engine_samples=[])
    groups = pl.DataFrame(
        {"reward_mean": [None, None], "zero_std": [False, False]},
        schema={"reward_mean": pl.Float64, "zero_std": pl.Boolean},
    )
    assert compute_advisories(store, _StubReader(groups, _summary_df([False] * 8))) == []


def test_truncation_warning_cites_per_turn_cap(tmp_path):
    # [FUNCTION DOC] With 3 of 8 samples truncated and a rollout max response length of 8192, one
    # warning advisory mentioning the max response length option.
    store = _store(tmp_path, args={"rollout_max_response_len": 8192}, engine_samples=[])
    groups = pl.DataFrame({"reward_mean": [0.5], "zero_std": [False]})
    [advisory] = compute_advisories(store, _StubReader(groups, _summary_df([True] * 3 + [False] * 5)))
    assert advisory.level == "warning"
    assert "--rollout-max-response-len" in advisory.message


def _mfu_store(tmp_path, values: list[float]) -> MetricStore:
    # [FUNCTION DOC] HELPER: Writes metadata and one metrics record per value (step = list position,
    # time = step) with actor-train utilization equal to the value and peak teraflops 989; flushes and
    # returns the store loaded from disk.
    writer = MetricStore(tmp_path)
    writer.write_meta(Meta(run_name="advisory-test", start_ts=0.0, args={}))
    for step, value in enumerate(values):
        metrics = {"perf/actor_train_mfu": value, "perf/mfu_peak_tflops": 989.0}
        writer.append(MetricsRecord(ts=float(step), step_key="rollout/step", step=step, metrics=metrics))
    writer.flush()
    return MetricStore.load(tmp_path)


def test_sustained_low_mfu_is_a_warning(tmp_path):
    # [FUNCTION DOC] Utilization values 0.02, 0.10, 0.11, 0.09, 0.10 (no engine series in the store):
    # one warning advisory with the steady mean 10.0%, the peak "989 TFLOP/s" and the note that
    # rollout stalls cannot depress it.
    store = _mfu_store(tmp_path, [0.02, 0.10, 0.11, 0.09, 0.10])
    assert store.has_stream(Stream.ENGINE_SERIES) is False
    [advisory] = compute_advisories(store)
    assert advisory.level == "warning"
    assert "10.0%" in advisory.message
    assert "989 TFLOP/s" in advisory.message
    assert "rollout stalls cannot depress it" in advisory.message


def test_healthy_mfu_is_quiet(tmp_path):
    # [FUNCTION DOC] Steady utilization around 0.37 yields no advisories.
    assert compute_advisories(_mfu_store(tmp_path, [0.02, 0.38, 0.36, 0.37, 0.39])) == []


def test_threshold_is_configurable(tmp_path):
    # [FUNCTION DOC] Utilization about 0.175: no advisory with the default threshold, one warning with
    # a low-utilization threshold of 0.25.
    store = _mfu_store(tmp_path, [0.02, 0.18, 0.17, 0.18, 0.17])
    assert compute_advisories(store) == []
    [advisory] = compute_advisories(store, low_mfu=0.25)
    assert advisory.level == "warning"


def test_zero_threshold_disables_the_rule(tmp_path):
    # [FUNCTION DOC] Utilization 0.01 in steady steps gives one advisory by default and none when the
    # threshold is 0.0.
    store = _mfu_store(tmp_path, [0.02, 0.01, 0.01, 0.01, 0.01])
    assert len(compute_advisories(store)) == 1
    assert compute_advisories(store, low_mfu=0.0) == []


def test_first_step_is_excluded_from_the_mean(tmp_path):
    # [FUNCTION DOC] A first step with utilization 0.0 followed by healthy 0.35 values yields no
    # advisories (the first step is warm-up).
    assert compute_advisories(_mfu_store(tmp_path, [0.0, 0.35, 0.35, 0.35, 0.35])) == []


def test_too_few_steady_steps_to_judge(tmp_path):
    # [FUNCTION DOC] Only two steps after the first (all low) are not enough to judge: no advisories.
    assert compute_advisories(_mfu_store(tmp_path, [0.01, 0.01, 0.01])) == []


def test_no_mfu_metric_no_claim(tmp_path):
    # [FUNCTION DOC] Five metric records carrying only a teraflops metric and no utilization metric
    # yield no advisories.
    writer = MetricStore(tmp_path)
    writer.write_meta(Meta(run_name="advisory-test", start_ts=0.0, args={}))
    for step in range(5):
        writer.append(
            MetricsRecord(
                ts=float(step), step_key="rollout/step", step=step, metrics={"perf/actor_train_tflops": 10.0}
            )
        )
    writer.flush()
    assert compute_advisories(MetricStore.load(tmp_path)) == []


def test_blank_newest_step_does_not_break_rollout_advisories(tmp_path):
    # [FUNCTION DOC] Generates a two-step dummy dump without evaluation, blanks step 1 (aborted before
    # generation), builds an empty store, and asserts computing advisories with a reader over the dump
    # directory returns none and does not raise.
    dump_dir = tmp_path / "dump"
    dump_dummy_run(dump_dir, steps=2, with_eval=False)
    blank_samples(dump_dir, 1)
    store = _store(tmp_path, args={}, engine_samples=[])
    assert compute_advisories(store, DumpReader(dump_dir)) == []
