# [MODULE DOC]
# Tests for the timeline queries of the metric store against the dummy telemetry fixture (a
# single GPU node, two engines where engine B restarts mid-run, rollout / train / update-weights
# phases per step, a 1 Hz gpu series with a known spike, drain dip and per-step perf metrics):
# lanes, topology windows, rollout expansion over engine lanes (including clipping at a
# restart), train phase skew, time filters, gpu downsampling, engine series split on restart,
# synthesized initialize bands, bubbles, the fleet overview, the HTTP timeline endpoints, and
# open-interval handling.
import pytest
from fastapi.testclient import TestClient
from tests.fast.dashboard.dummy_telemetry import (
    BASE_TS,
    DRAIN_B,
    ENGINE_A,
    ENGINE_B_NEW,
    ENGINE_B_OLD,
    GPU_NODE,
    ROLLOUT_SECONDS,
    dump_dummy_telemetry,
)

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import MetricStore, Role


@pytest.fixture
def loaded(tmp_path):
    # [FUNCTION DOC] HELPER: Fixture. Writes dummy telemetry into the temporary path and returns
    # a pair of the loaded store and the ground-truth description.
    truth = dump_dummy_telemetry(tmp_path)
    return MetricStore.load(tmp_path / "dashboard"), truth


def test_lanes(loaded):
    # [FUNCTION DOC] The store lists one lane per gpu of the single node, each with node, gpu id
    # and index equal to the gpu id.
    store, truth = loaded
    assert store.lanes() == [dict(node=GPU_NODE, gpu=g, index=g) for g in range(truth.gpus)]


def test_topology_windows(loaded):
    # [FUNCTION DOC] Two topology windows: the first from the base time to the restart time with
    # engines [A, old B], the second from the restart to open end (none) with [A, new B].
    store, truth = loaded
    windows = store.topology_windows()
    assert len(windows) == 2
    assert windows[0]["t0"] == BASE_TS and windows[0]["t1"] == truth.restart_ts
    assert windows[1]["t0"] == truth.restart_ts and windows[1]["t1"] is None
    assert [e["addr"] for e in windows[0]["engines"]] == [ENGINE_A, ENGINE_B_OLD]
    assert [e["addr"] for e in windows[1]["engines"]] == [ENGINE_A, ENGINE_B_NEW]


def test_rollout_expansion_covers_engine_lanes(loaded):
    # [FUNCTION DOC] The rollout-manager phase of step 0 (wholly inside the first topology
    # window) is expanded into one interval per gpu lane, each on the node, starting at the step
    # start and ending after the rollout duration.
    store, truth = loaded
    phases = store.phases_by_lane()
    rollout0 = [p for p in phases if p["role"] == Role.ROLLOUT_MANAGER and p["t0"] == truth.step_start(0)]
    # step 0 lies entirely inside the first topology window: one interval per GPU
    assert len(rollout0) == truth.gpus
    assert {p["gpu"] for p in rollout0} == set(range(truth.gpus))
    assert all(p["node"] == GPU_NODE and p["t1"] == truth.step_start(0) + ROLLOUT_SECONDS for p in rollout0)


def test_rollout_expansion_clips_at_topology_restart(loaded):
    # [FUNCTION DOC] For step 1's rollout, during which engine B restarts, lane 2's expanded
    # intervals are two: from the rollout start to the restart time, and from the restart time to
    # the rollout end.
    store, truth = loaded
    t0, t1 = truth.rollout_interval(1)  # engine B restarts mid-way through this rollout
    on_lane_2 = [
        p for p in store.phases_by_lane() if p["role"] == Role.ROLLOUT_MANAGER and p["gpu"] == 2 and t0 <= p["t0"] < t1
    ]
    assert [(p["t0"], p["t1"]) for p in on_lane_2] == [(t0, truth.restart_ts), (truth.restart_ts, t1)]


def test_train_phases_keep_rank_and_skew(loaded):
    # [FUNCTION DOC] Weight update phases before step 1 number one per gpu; the highest rank
    # finishes later than rank 0 (cross-rank skew preserved); each phase's gpu equals its rank.
    store, truth = loaded
    updates = [p for p in store.phases_by_lane() if p["name"] == "update_weights" and p["t0"] < truth.step_start(1)]
    assert len(updates) == truth.gpus
    by_rank = {p["rank"]: p for p in updates}
    assert by_rank[truth.gpus - 1]["t1"] > by_rank[0]["t1"]  # cross-rank skew survives
    assert all(p["gpu"] == p["rank"] for p in updates)


def test_phases_time_filter(loaded):
    # [FUNCTION DOC] A window from step 0's rollout start to one second before its end returns a
    # non-empty list where every phase overlaps the window, and only rollout phases (train events
    # of step 0 start at the rollout's end).
    store, truth = loaded
    t0, t1 = truth.rollout_interval(0)
    phases = store.phases_by_lane(t0=t0, t1=t1 - 1)
    assert phases and all(p["t0"] < t1 - 1 and p["t1"] > t0 for p in phases)
    assert {p["name"] for p in phases} == {"rollout"}  # train events of step 0 start at rollout end


def test_gpu_series_downsampling_preserves_spike(loaded):
    # [FUNCTION DOC] With a cap of 40 points, lane 0 has at most 40 sorted timestamps, still
    # holds the fixture's spike (util 100) and drain dip (util 5), and the four value lists have
    # equal length. Without a cap, lane 0 has one point per second: 30 initialize seconds plus 100
    # per step.
    store, truth = loaded
    lanes = store.gpu_series(max_points=40)
    lane0 = lanes[f"{GPU_NODE}:0"]
    assert len(lane0["ts"]) <= 40
    assert max(lane0["util"]) == 100  # the fixture's known spike survives
    assert min(lane0["util"]) == 5  # so does the drain dip
    assert len(lane0["ts"]) == len(lane0["util"]) == len(lane0["mem_mb"]) == len(lane0["power_w"])
    assert lane0["ts"] == sorted(lane0["ts"])

    full = store.gpu_series()
    # 1 Hz over initialize + steps, no downsampling needed
    assert len(full[f"{GPU_NODE}:0"]["ts"]) == 30 + truth.steps * 100


def test_gpu_series_time_filter(loaded):
    # [FUNCTION DOC] With a window equal to step 0's rollout interval, every lane's timestamps lie
    # within the (inclusive) window.
    store, truth = loaded
    t0, t1 = truth.rollout_interval(0)
    lanes = store.gpu_series(t0=t0, t1=t1)
    for series in lanes.values():
        assert all(t0 <= ts <= t1 for ts in series["ts"])


def test_engine_series_split_on_restart(loaded):
    # [FUNCTION DOC] The running-requests series is split by engine address into three series (A,
    # old B, new B). Old B ends before the restart and new B starts at or after it; A starts at
    # the first step start. In a late rollout window, after B's drain time new B reports only
    # zero running requests. An unknown metric returns an empty list.
    store, truth = loaded
    series = store.engine_series("sglang_num_running_reqs")
    by_addr = {s["addr"]: s for s in series}
    assert set(by_addr) == {ENGINE_A, ENGINE_B_OLD, ENGINE_B_NEW}
    assert max(by_addr[ENGINE_B_OLD]["ts"]) < truth.restart_ts <= min(by_addr[ENGINE_B_NEW]["ts"])
    # engine A spans all steps (engines come up after initialize)
    assert min(by_addr[ENGINE_A]["ts"]) == truth.step_start(0)

    # long tail: B is drained (0 running) before the rollout window ends
    b_new = by_addr[ENGINE_B_NEW]
    t0, t1 = truth.rollout_interval(2)
    in_window = [v for ts, v in zip(b_new["ts"], b_new["value"], strict=True) if t0 + DRAIN_B <= ts < t1]
    assert in_window and all(v == 0 for v in in_window)

    assert store.engine_series("sglang_no_such_metric") == []


def test_initialize_phase_synthesized_per_lane(loaded):
    # [FUNCTION DOC] Initialize bands: one per lane, role derived, rank -1, spanning from the
    # collector start time (metadata start) to the first observed event. A query whose window
    # starts at step 0's rollout returns no initialize bands.
    store, truth = loaded
    init = [p for p in store.phases_by_lane() if p["name"] == "initialize"]
    assert len(init) == truth.gpus  # one band per lane
    for band in init:
        assert band["role"] == Role.DERIVED and band["rank"] == -1
        assert band["t0"] == BASE_TS  # collector start (meta.start_ts)
        assert band["t1"] == truth.step_start(0)  # first observed event
    # time-filtering away the init window drops the synthesized bands too
    t0, _ = truth.rollout_interval(0)
    assert not [p for p in store.phases_by_lane(t0=t0) if p["name"] == "initialize"]


def test_bubbles(loaded):
    # [FUNCTION DOC] Bubble entries are one per step in order; the first has wait ratio 0.4 and
    # step time 95.
    store, truth = loaded
    bubbles = store.bubbles()
    assert [b["step"] for b in bubbles] == list(range(truth.steps))
    assert bubbles[0]["wait_ratio"] == pytest.approx(0.4)
    assert bubbles[0]["step_time"] == pytest.approx(95.0)


def test_fleet_overview(loaded):
    # [FUNCTION DOC] The fleet overview with 32 buckets reports the lane count; in every bucket
    # the phase fractions (including "none") sum to 1 within 1e-6; at least one non-"none" phase
    # exists; wherever the median band is sampled, the bands are ordered min <= p10 <= p50 <= p90
    # and bounded to 0..100.
    store, truth = loaded
    fleet = store.fleet(x_buckets=32)
    assert fleet["lanes"] == truth.gpus
    # every bucket's phase fractions (incl. "none") sum to 1 across the palette
    for x in range(32):
        total = sum(fractions[x] for fractions in fleet["composition"].values())
        assert abs(total - 1.0) < 1e-6, f"bucket {x} sums to {total}"
    assert any(name for name in fleet["composition"] if name != "none")
    band = fleet["band"]
    sampled = [x for x in range(32) if band["p50"][x] is not None]
    assert sampled, "expected util samples in the window"
    for x in sampled:
        assert 0 <= band["min"][x] <= band["p10"][x] <= band["p50"][x] <= band["p90"][x] <= 100


def test_fleet_endpoint(tmp_path):
    # [FUNCTION DOC] Meta reports the utilisation-overview capability false. The fleet endpoint
    # with 16 buckets returns that bucket count and a positive lane count; 1 bucket returns 400.
    dump_dummy_telemetry(tmp_path)
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))
    meta = client.get("/api/meta").json()
    assert meta["capabilities"]["use_utilization_overview"] is False
    fleet = client.get("/api/timeline/fleet", params={"x_buckets": 16}).json()
    assert fleet["x_buckets"] == 16 and fleet["lanes"] > 0
    assert client.get("/api/timeline/fleet", params={"x_buckets": 1}).status_code == 400


def test_timeline_endpoints(tmp_path):
    # [FUNCTION DOC] Through the test client: meta shows timeline and engine-series capabilities;
    # topology returns one lane per gpu and 2 windows; phases for step 0's window include rollout,
    # actor_train and train_wait; gpu with a 50 point cap returns every lane (cap of 1 gives
    # 400); engine series for generation throughput covers the three engine addresses; bubbles
    # count equals steps; gpu processes returns an empty list (the fixture emits none), and a
    # reversed window returns 400.
    truth = dump_dummy_telemetry(tmp_path)
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))

    meta = client.get("/api/meta").json()
    assert meta["capabilities"]["has_timeline"] is True
    assert meta["capabilities"]["has_engine_series"] is True

    topology = client.get("/api/timeline/topology").json()
    assert len(topology["lanes"]) == truth.gpus
    assert len(topology["windows"]) == 2

    phases = client.get("/api/timeline/phases", params={"t0": truth.step_start(0), "t1": truth.step_start(1)}).json()
    assert {p["name"] for p in phases["phases"]} >= {"rollout", "actor_train", "train_wait"}

    gpu = client.get("/api/timeline/gpu", params={"max_points": 50}).json()
    assert set(gpu["lanes"]) == {f"{GPU_NODE}:{g}" for g in range(truth.gpus)}
    assert client.get("/api/timeline/gpu", params={"max_points": 1}).status_code == 400

    engines = client.get("/api/timeline/engine_series", params={"metric": "sglang_gen_throughput"}).json()
    assert {s["addr"] for s in engines["series"]} == {ENGINE_A, ENGINE_B_OLD, ENGINE_B_NEW}

    bubbles = client.get("/api/timeline/bubbles").json()
    assert len(bubbles["bubbles"]) == truth.steps

    # dummy_telemetry doesn't emit per-process samples: smoke-test the wiring
    processes = client.get("/api/timeline/gpu_processes").json()
    assert processes["processes"] == []
    assert client.get("/api/timeline/gpu_processes", params={"t0": 5, "t1": 1}).status_code == 400


def test_open_interval_clips_to_data_edge_and_closed_twin_wins(tmp_path):
    # [FUNCTION DOC] Store with a gpu sample at t=500, an open "rollout" marker from t=100, and a
    # "warmup" that has both an open marker and a closed event 10..40. Phases: the open rollout
    # ends at 500 (clipped to the newest data), warmup ends at 40 and appears exactly once (the
    # closed twin supersedes the open marker).
    # KNOWN ISSUE: re-imports names already imported at module level.
    from miles.dashboard.store import GpuSample, Meta, PhaseEvent, Role

    writer = MetricStore(tmp_path)
    writer.write_meta(Meta(run_name="open", start_ts=0.0, args={}))
    # newest data: a gpu sample at t=500 — the open band must grow to it
    writer.append(GpuSample(ts=500.0, node="n", gpu=0, util=10, mem_mb=1, power_w=1))
    writer.append(
        PhaseEvent(name="rollout", t0=100.0, t1=PhaseEvent.OPEN_T1, node="n", gpus=[0], rank=0, role=Role.TRAIN)
    )
    # a second phase that already closed: its open marker must be superseded
    writer.append(
        PhaseEvent(name="warmup", t0=10.0, t1=PhaseEvent.OPEN_T1, node="n", gpus=[0], rank=0, role=Role.TRAIN)
    )
    writer.append(PhaseEvent(name="warmup", t0=10.0, t1=40.0, node="n", gpus=[0], rank=0, role=Role.TRAIN))
    writer.flush()

    phases = {p["name"]: p for p in MetricStore.load(tmp_path).phases_by_lane() if p["name"] != "initialize"}
    assert phases["rollout"]["t1"] == 500.0  # clipped to the data edge, not -1
    assert phases["warmup"]["t1"] == 40.0  # exactly one warmup, the closed one
    assert sum(1 for p in MetricStore.load(tmp_path).phases_by_lane() if p["name"] == "warmup") == 1
