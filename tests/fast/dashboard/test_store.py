# [MODULE DOC]
# Unit tests for the dashboard metric store: write/read round trip of one record of every stream,
# repeated flush behaviour, latest data buffer length, gpu process window and lane filtering,
# incremental follow, partial trailing line handling, corrupt record detection, an empty or
# missing directory, metric series filtering, metric / step key catalogs, per-metric step keys,
# bubble extraction for namespaced policies, time range across streams, and the two downsampling
# helpers. All data lives in temporary directories.
import numpy as np
import pytest

from miles.dashboard.store import (
    DataBufferSample,
    EngineInfo,
    EngineSample,
    GpuProcessSample,
    GpuSample,
    Meta,
    MetricsRecord,
    MetricStore,
    PhaseEvent,
    Stream,
    TopologySnapshot,
    TrajectoryEvent,
    TrajectoryEventKind,
    minmax_downsample,
    stride_downsample,
)


def _one_of_each() -> list:
    # [FUNCTION DOC] HELPER: Returns a fresh list with one record of each of the eight stream
    # types (metrics, phase event, topology snapshot with one engine, gpu sample, trajectory
    # event of the generation-start kind, engine sample, gpu process sample, data buffer sample)
    # with fixed example values.
    return [
        MetricsRecord(ts=10.0, step_key="rollout/step", step=3, metrics={"rollout/rewards_mean": 0.5}),
        PhaseEvent(name="actor_train", t0=9.0, t1=11.0, node="10.0.0.2", gpus=[0, 1], rank=5, role="train"),
        TopologySnapshot(
            ts=8.0,
            engines=[
                EngineInfo(
                    addr="http://10.0.0.2:15000",
                    worker_type="regular",
                    engine_rank=0,
                    gpus=[["10.0.0.2", 0]],
                    gpu_uuids=[None],
                )
            ],
        ),
        GpuSample(ts=10.5, node="10.0.0.2", gpu=0, util=87, mem_mb=61234, power_w=612),
        TrajectoryEvent(
            ts=10.7,
            kind=TrajectoryEventKind.GEN_START,
            sample_index=3,
            group_index=0,
            turn=1,
            weight_version="1",
            detail="",
        ),
        EngineSample(ts=10.6, addr="http://10.0.0.2:15000", metric="sglang_num_running_reqs", labels={}, value=42.0),
        GpuProcessSample(ts=10.8, node="10.0.0.2", gpu=0, pid=4321, name="sglang", mem_mb=40960),
        DataBufferSample(ts=10.9, length=5),
    ]


def test_roundtrip_all_streams(tmp_path):
    # [FUNCTION DOC] Writes metadata and one record per stream, flushes, loads, and asserts the
    # metadata equals the written one (schema version 1) and each stream returns exactly the
    # record (partitioned streams via the record iterator, the others via the in-memory lists).
    writer = MetricStore(tmp_path)
    writer.write_meta(Meta(run_name="test-run", start_ts=1.0, args={"colocate": True}))
    records = _one_of_each()
    for record in records:
        writer.append(record)
    writer.flush()

    reader = MetricStore.load(tmp_path)
    assert reader.meta == Meta(run_name="test-run", start_ts=1.0, args={"colocate": True}, schema_version=1)
    for record in records:
        got = (
            reader.iter_records(record.stream)
            if record.stream in MetricStore.PARTITIONED_STREAMS
            else reader.records[record.stream]
        )
        assert got == [record]


def test_flush_clears_buffers_and_appends(tmp_path):
    # [FUNCTION DOC] Writes all streams, flushes twice (the second with empty buffers must add
    # nothing), writes all again and flushes. Every stream then holds exactly 2 records.
    writer = MetricStore(tmp_path)
    for record in _one_of_each():
        writer.append(record)
    writer.flush()
    writer.flush()  # empty buffers: no duplicate lines
    for record in _one_of_each():
        writer.append(record)
    writer.flush()

    reader = MetricStore.load(tmp_path)
    for stream in Stream:
        got = reader.iter_records(stream) if stream in MetricStore.PARTITIONED_STREAMS else reader.records[stream]
        assert len(got) == 2, stream


def test_latest_data_buffer_length(tmp_path):
    # [FUNCTION DOC] Before any report the latest length is none. After samples of 3 then 7 it is
    # 7 (the most recent by order, not the first or the maximum).
    writer = MetricStore(tmp_path)
    assert MetricStore.load(tmp_path).latest_data_buffer_length() is None  # never reported

    writer.append(DataBufferSample(ts=1.0, length=3))
    writer.append(DataBufferSample(ts=2.0, length=7))
    writer.flush()
    assert MetricStore.load(tmp_path).latest_data_buffer_length() == 7  # most recent, not first/max


def test_gpu_processes_window_and_lane_filter(tmp_path):
    # [FUNCTION DOC] Three process samples (two at t=1 on node n1, one at t=20 on n2). The
    # unfiltered query returns all three as mappings in write order; a window 10..30 returns
    # only the last; a lane filter of (n1, gpu 1) returns only the second.
    writer = MetricStore(tmp_path)
    writer.append(GpuProcessSample(ts=1.0, node="n1", gpu=0, pid=1, name="sglang", mem_mb=100))
    writer.append(GpuProcessSample(ts=1.0, node="n1", gpu=1, pid=2, name="train", mem_mb=200))
    writer.append(GpuProcessSample(ts=20.0, node="n2", gpu=0, pid=3, name="sglang", mem_mb=300))
    writer.flush()

    reader = MetricStore.load(tmp_path)
    assert reader.gpu_processes() == [
        dict(ts=1.0, node="n1", gpu=0, pid=1, name="sglang", mem_mb=100),
        dict(ts=1.0, node="n1", gpu=1, pid=2, name="train", mem_mb=200),
        dict(ts=20.0, node="n2", gpu=0, pid=3, name="sglang", mem_mb=300),
    ]
    assert reader.gpu_processes(t0=10.0, t1=30.0) == [
        dict(ts=20.0, node="n2", gpu=0, pid=3, name="sglang", mem_mb=300),
    ]
    assert reader.gpu_processes(lanes={("n1", 1)}) == [
        dict(ts=1.0, node="n1", gpu=1, pid=2, name="train", mem_mb=200),
    ]


def test_follow_incremental(tmp_path):
    # [FUNCTION DOC] A loaded reader sees 1 metrics record and follow returns 0 changes. After
    # priming the empty gpu cache, the writer adds a metrics record and a gpu sample in a new
    # partition. Follow returns 1 (only metrics are tailed eagerly; the new gpu partition stays
    # lazy), metrics steps are [0, 1], the gpu series query then shows lane "n:0", and a further
    # follow returns 0.
    writer = MetricStore(tmp_path)
    writer.append(MetricsRecord(ts=1.0, step_key="rollout/step", step=0, metrics={"a": 1}))
    writer.flush()

    reader = MetricStore.load(tmp_path)
    assert len(reader.records[Stream.METRICS]) == 1
    assert reader.follow() == 0

    assert reader.gpu_series() == {}  # prime the (empty) gpu partition cache

    writer.append(MetricsRecord(ts=2.0, step_key="rollout/step", step=1, metrics={"a": 2}))
    writer.append(GpuSample(ts=2.5, node="n", gpu=0, util=10, mem_mb=1, power_w=1))
    writer.flush()
    # metrics tail eagerly; the gpu partition is new (never cached) so it
    # stays lazy and surfaces through the query below
    assert reader.follow() == 1
    assert [r.step for r in reader.records[Stream.METRICS]] == [0, 1]
    assert list(reader.gpu_series()) == ["n:0"]
    assert reader.follow() == 0


def test_partial_trailing_line_left_for_later(tmp_path):
    # [FUNCTION DOC] After one flushed record, the first 20 characters of a second record line
    # are appended without newline (a crash mid-write). A loaded reader sees only 1 record. After
    # the rest of the line is appended, follow returns 1 and the second record has step 1.
    writer = MetricStore(tmp_path)
    writer.append(MetricsRecord(ts=1.0, step_key="rollout/step", step=0, metrics={"a": 1}))
    writer.flush()

    path = tmp_path / Stream.METRICS.filename
    full_line = '{"ts": 2.0, "step_key": "rollout/step", "step": 1, "metrics": {"a": 2}}\n'
    with open(path, "a") as f:
        f.write(full_line[:20])  # crash mid-write: no trailing newline

    reader = MetricStore.load(tmp_path)
    assert len(reader.records[Stream.METRICS]) == 1

    with open(path, "a") as f:
        f.write(full_line[20:])  # writer completes the record
    assert reader.follow() == 1
    assert reader.records[Stream.METRICS][1].step == 1


def test_corrupt_complete_line_raises(tmp_path):
    # [FUNCTION DOC] A complete (newline-terminated) non-json line in the metrics file makes load
    # raise a value error matching "corrupt record".
    writer = MetricStore(tmp_path)
    writer.append(MetricsRecord(ts=1.0, step_key="rollout/step", step=0, metrics={"a": 1}))
    writer.flush()
    with open(tmp_path / Stream.METRICS.filename, "a") as f:
        f.write("not json\n")

    with pytest.raises(ValueError, match="corrupt record"):
        MetricStore.load(tmp_path)


def test_wrong_field_complete_line_raises(tmp_path):
    # [FUNCTION DOC] A legacy flat gpu file with a record having an unexpected field loads
    # cleanly (partitions parse lazily), but querying the gpu series raises a value error
    # matching "corrupt record".
    with open(tmp_path / Stream.GPU_UTIL.filename, "w") as f:
        f.write('{"ts": 1.0, "unexpected_field": 1}\n')
    store = MetricStore.load(tmp_path)  # partitions parse lazily: load is clean
    with pytest.raises(ValueError, match="corrupt record"):
        store.gpu_series()


def test_empty_dir(tmp_path):
    # [FUNCTION DOC] Loading a non-existent directory gives: no metadata, follow 0, no time
    # range, empty metric key list, and a metric series for key "a" with empty x, y and ts lists.
    reader = MetricStore.load(tmp_path / "does_not_exist")
    assert reader.meta is None
    assert reader.follow() == 0
    assert reader.time_range() is None
    assert reader.metric_keys() == []
    assert reader.metric_series(["a"], x_key="rollout/step") == {"a": {"x": [], "y": [], "ts": []}}


def test_metric_series_filters_by_step_key_and_time(tmp_path):
    # [FUNCTION DOC] Three records on two step axes. Querying a and b on the rollout axis returns
    # only points logged against that axis (a: x [0, 1]; b: x [0]); querying a on the train axis
    # returns x [100]; a time window 2.5..3.5 keeps only the point at t=3.
    writer = MetricStore(tmp_path)
    writer.append(MetricsRecord(ts=1.0, step_key="rollout/step", step=0, metrics={"a": 1.0, "b": 10.0}))
    writer.append(MetricsRecord(ts=2.0, step_key="train/step", step=100, metrics={"a": -1.0}))
    writer.append(MetricsRecord(ts=3.0, step_key="rollout/step", step=1, metrics={"a": 2.0}))
    writer.flush()
    reader = MetricStore.load(tmp_path)

    series = reader.metric_series(["a", "b"], x_key="rollout/step")
    assert series["a"] == {"x": [0, 1], "y": [1.0, 2.0], "ts": [1.0, 3.0]}
    assert series["b"] == {"x": [0], "y": [10.0], "ts": [1.0]}

    series = reader.metric_series(["a"], x_key="train/step")
    assert series["a"] == {"x": [100], "y": [-1.0], "ts": [2.0]}

    series = reader.metric_series(["a"], x_key="rollout/step", t0=2.5, t1=3.5)
    assert series["a"] == {"x": [1], "y": [2.0], "ts": [3.0]}


def test_metric_keys_and_step_keys(tmp_path):
    # [FUNCTION DOC] Metric keys are returned sorted and unique (a, b, c) and step keys sorted
    # (eval step, rollout step).
    writer = MetricStore(tmp_path)
    writer.append(MetricsRecord(ts=1.0, step_key="rollout/step", step=0, metrics={"b": 1, "a": 1}))
    writer.append(MetricsRecord(ts=2.0, step_key="eval/step", step=0, metrics={"c": 1}))
    writer.flush()
    reader = MetricStore.load(tmp_path)
    assert reader.metric_keys() == ["a", "b", "c"]
    assert reader.step_keys() == ["eval/step", "rollout/step"]


def test_step_key_of_metric_key_answers_the_axis_each_namespaced_metric_was_logged_against(tmp_path):
    # [FUNCTION DOC] The mapping from metric key to step key reports, for two namespaced
    # policies, the step key each metric was recorded with (loss with its train step; step time
    # with its rollout step).
    """A multi policy run logs alpha/train/loss against alpha/train/step, which no key prefix rule can guess."""
    writer = MetricStore(tmp_path)
    writer.append(MetricsRecord(ts=1.0, step_key="alpha/train/step", step=0, metrics={"alpha/train/loss": 1.0}))
    writer.append(MetricsRecord(ts=2.0, step_key="beta/rollout/step", step=0, metrics={"beta/perf/step_time": 2.0}))
    writer.flush()
    reader = MetricStore.load(tmp_path)

    assert reader.step_key_of_metric_key() == {
        "alpha/train/loss": "alpha/train/step",
        "beta/perf/step_time": "beta/rollout/step",
    }


def test_bubbles_reads_the_perf_points_of_a_namespaced_policy(tmp_path):
    # [FUNCTION DOC] A record on a namespaced rollout step key with step time 12 and wait ratio
    # 0.25 yields exactly one bubble entry with step 4, timestamp 1, step_time 12 and wait_ratio
    # 0.25.
    """The bubble strip keys off rollout/step, so a namespaced run must not silently fall out of it."""
    writer = MetricStore(tmp_path)
    writer.append(
        MetricsRecord(
            ts=1.0,
            step_key="alpha/rollout/step",
            step=4,
            metrics={"alpha/perf/step_time": 12.0, "alpha/perf/wait_time_ratio": 0.25},
        )
    )
    writer.flush()
    reader = MetricStore.load(tmp_path)

    assert reader.bubbles() == [dict(step=4, ts=1.0, step_time=12.0, wait_ratio=0.25)]


def test_time_range_spans_all_streams(tmp_path):
    # [FUNCTION DOC] With a phase event from 0.5 to 7 and a gpu sample at 9, the time range is
    # (0.5, 9.0).
    writer = MetricStore(tmp_path)
    writer.append(PhaseEvent(name="rollout", t0=0.5, t1=7.0, node="n", gpus=[], rank=-1, role="rollout_manager"))
    writer.append(GpuSample(ts=9.0, node="n", gpu=0, util=1, mem_mb=1, power_w=1))
    writer.flush()
    reader = MetricStore.load(tmp_path)
    assert reader.time_range() == (0.5, 9.0)


def test_stride_downsample():
    # [FUNCTION DOC] Downsampling 1000 points to 100 returns 100 points with first and last x
    # preserved (0 and 999); a target of 2000 leaves all 1000 points unchanged.
    xs, ys = np.arange(1000), np.arange(1000) * 2.0
    dx, dy = stride_downsample(xs, ys, 100)
    assert len(dx) == len(dy) == 100
    assert dx[0] == 0 and dx[-1] == 999  # endpoints preserved

    dx, dy = stride_downsample(xs, ys, 2000)  # already small enough
    assert len(dx) == 1000


def test_minmax_downsample_preserves_extremes():
    # [FUNCTION DOC] 5000 random points (seeded) with an injected spike (+100) and dip (-100).
    # Min-max downsampling to 200 returns at most 200 points, retains the global max and min,
    # and keeps x strictly increasing. A 50-point input with target 200 is returned unchanged.
    rng = np.random.default_rng(0)
    ys = rng.normal(size=5000)
    ys[1234] = 100.0  # spike
    ys[4321] = -100.0  # dip
    xs = np.arange(len(ys))

    dx, dy = minmax_downsample(xs, ys, 200)
    assert len(dx) <= 200
    assert dy.max() == 100.0
    assert dy.min() == -100.0
    assert (np.diff(dx) > 0).all()  # x stays sorted

    dx, dy = minmax_downsample(xs[:50], ys[:50], 200)
    assert len(dx) == 50  # small input unchanged
