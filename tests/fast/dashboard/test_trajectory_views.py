# [MODULE DOC]
# Tests for trajectory (per-sample lifecycle) views of the metric store and server: lane assembly
# from raw events (pairing generation and tool spans, attempts, versions, status, aborted
# spans closed by attempt end, single-turn probes, coarse spans superseded by per-turn spans),
# the lifecycle heatmap (palette, submit-ordered rows, row cap), and the per-rollout trajectories
# endpoint (filtering by sample and empty result without probes). Uses the combined dummy dump
# plus telemetry directory (the same composition as the demo mode) and hand-written events.
import numpy as np
import pytest
from fastapi.testclient import TestClient
from tests.fast.dashboard.dummy_dump import dump_dummy_run
from tests.fast.dashboard.dummy_telemetry import SAMPLES_PER_STEP, dump_dummy_telemetry

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import MetricStore, TrajectoryEvent, TrajectoryEventKind


@pytest.fixture
def combined(tmp_path):
    # [FUNCTION DOC] HELPER: Fixture. Writes a dummy dump and dummy telemetry into the same
    # temporary directory; returns a triple of the loaded store, a dump reader and the truth.
    """dump + telemetry in one dir — the same composition serve --demo uses."""
    dump_dummy_run(tmp_path)
    truth = dump_dummy_telemetry(tmp_path)
    return MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path), truth


# ------------------------------ lane assembly --------------------------------


def _event(ts, kind, index=1, turn=-1, version="2", detail=""):
    # [FUNCTION DOC] HELPER: Builds a trajectory event at the given time and kind for the given
    # sample index (default 1), group 0, turn (default -1 meaning attempt level), version
    # (default "2") and detail (default empty).
    return TrajectoryEvent(
        ts=ts, kind=kind, sample_index=index, group_index=0, turn=turn, weight_version=version, detail=detail
    )


def test_trajectory_lanes_pairs_spans_and_attempts(tmp_path):
    # [FUNCTION DOC] Writes the events of one sample with two attempts (the first aborted, with a
    # turn-2 generation never ended; the second completed). Asserts a single lane with sample
    # index 1, status completed (from the last attempt), versions [2, 3, 4], attempts
    # [10..19, 30..36], and segments: gen 11-15 turn 1, tool 15-18 turn 1, gen 18-19 turn 2
    # (closed by the attempt end), gen 31-35 turn 1.
    writer = MetricStore(tmp_path)
    kinds = TrajectoryEventKind
    for event in (
        _event(10.0, kinds.ATTEMPT_START),
        _event(11.0, kinds.GEN_START, turn=1),
        _event(15.0, kinds.GEN_END, turn=1),
        _event(15.0, kinds.TOOL_START, turn=1),
        _event(18.0, kinds.TOOL_END, turn=1),
        _event(18.0, kinds.GEN_START, turn=2, version="3"),
        # turn-2 gen never ends: aborted mid-generation, then a second attempt
        _event(19.0, kinds.ATTEMPT_END, detail="aborted"),
        _event(30.0, kinds.ATTEMPT_START),
        _event(31.0, kinds.GEN_START, turn=1, version="4"),
        _event(35.0, kinds.GEN_END, turn=1, version="4"),
        _event(36.0, kinds.ATTEMPT_END, detail="completed"),
    ):
        writer.append(event)
    writer.flush()

    [lane] = MetricStore.load(tmp_path).trajectory_lanes()
    assert lane["sample_index"] == 1
    assert lane["status"] == "completed"
    assert lane["versions"] == [2, 3, 4]
    assert lane["attempts"] == [dict(t0=10.0, t1=19.0), dict(t0=30.0, t1=36.0)]
    by_kind = [(s["kind"], s["t0"], s["t1"], s["turn"]) for s in lane["segments"]]
    assert by_kind == [
        ("gen", 11.0, 15.0, 1),
        ("tool", 15.0, 18.0, 1),
        ("gen", 18.0, 19.0, 2),  # aborted mid-generation: closed by attempt_end
        ("gen", 31.0, 35.0, 1),
    ]


def test_lifecycle_heatmap_row_cap(combined, monkeypatch):
    # [FUNCTION DOC] With the store's lifecycle row cap lowered to 5, the lifecycle heatmap has
    # 5 rows, reports the total row count (steps times samples per step), and keeps the earliest
    # submitted samples (indices 0..4).
    store, _, truth = combined
    monkeypatch.setattr(MetricStore, "LIFECYCLE_MAX_ROWS", 5)
    result = store.heatmap("lifecycle", x_buckets=50)
    assert len(result["rows"]) == 5
    assert result["rows_total"] == truth.steps * SAMPLES_PER_STEP
    # earliest submits kept (rows are submit-ordered)
    assert [r["sample_index"] for r in result["rows"]] == [0, 1, 2, 3, 4]


def test_lifecycle_heatmap_palette_and_rows(combined):
    # [FUNCTION DOC] The lifecycle heatmap (200 buckets) has palette ["", queue, generating,
    # tool_wait]; one row per sample of all steps; the cell matrix contains generating and
    # tool_wait cells; the first rows' sample indices are in ascending (submit) order.
    store, _, truth = combined
    result = store.heatmap("lifecycle", x_buckets=200)
    assert result["palette"] == ["", "queue", "generating", "tool_wait"]
    assert len(result["rows"]) == truth.steps * SAMPLES_PER_STEP
    matrix = np.frombuffer(result["values"], dtype=np.uint8).reshape(len(result["rows"]), 200)
    assert (matrix == result["palette"].index("generating")).sum() > 0
    assert (matrix == result["palette"].index("tool_wait")).sum() > 0
    # rows are submit-ordered
    firsts = [r["sample_index"] for r in result["rows"][: SAMPLES_PER_STEP + 1]]
    assert firsts == sorted(firsts)


def test_single_turn_gen_closes_at_attempt_end(tmp_path):
    # [FUNCTION DOC] Events: attempt start 10, attempt-level generation start 12 (no end), attempt
    # end 50. The lane has one closed gen segment from 12 to 50 (a dangling span would be painted
    # to the consume line).
    # the generate_and_rm probe alone (single-turn path: no gen_end) must
    # yield a CLOSED gen span; dangling spans painted every sample's end at
    # the consume line (report 2026-07-13)
    writer = MetricStore(tmp_path)
    kinds = TrajectoryEventKind
    for event in (
        _event(10.0, kinds.ATTEMPT_START),
        _event(12.0, kinds.GEN_START),  # turn=-1: attempt-level probe
        _event(50.0, kinds.ATTEMPT_END, detail="completed"),
    ):
        writer.append(event)
    writer.flush()
    [lane] = MetricStore.load(tmp_path).trajectory_lanes()
    assert [(s["kind"], s["t0"], s["t1"]) for s in lane["segments"]] == [("gen", 12.0, 50.0)]


def test_coarse_gen_span_superseded_by_turn_spans(tmp_path):
    # [FUNCTION DOC] Events include a coarse attempt-level generation start plus a per-turn
    # generation (turn 1). The lane's segments contain only the turn-1 gen: the coarse span is
    # dropped when per-turn spans exist.
    writer = MetricStore(tmp_path)
    kinds = TrajectoryEventKind
    for event in (
        _event(10.0, kinds.ATTEMPT_START),
        _event(11.0, kinds.GEN_START),  # coarse attempt-level span
        _event(12.0, kinds.GEN_START, turn=1),
        _event(20.0, kinds.GEN_END, turn=1),
        _event(30.0, kinds.ATTEMPT_END, detail="completed"),
    ):
        writer.append(event)
    writer.flush()
    [lane] = MetricStore.load(tmp_path).trajectory_lanes()
    assert [(s["kind"], s["turn"]) for s in lane["segments"]] == [("gen", 1)]


# --------------------------------- endpoint ----------------------------------


def test_rollout_trajectories_endpoint(combined):
    # [FUNCTION DOC] The trajectories endpoint for step 1 returns lanes exactly for that step's
    # sample indices. An agentic lane (index divisible by 3) has 2 versions and segments gen, tool,
    # gen; a plain lane (index mod 3 equal 1) has 1 version and a single gen; all lanes are
    # completed; if a consume timestamp is reported, every lane's last event is not after it.
    store, reader, truth = combined
    client = TestClient(make_app(store, reader))

    payload = client.get("/api/rollout/1/trajectories").json()
    lanes = payload["lanes"]
    assert {lane["sample_index"] for lane in lanes} == set(range(SAMPLES_PER_STEP, 2 * SAMPLES_PER_STEP))
    agentic = next(lane for lane in lanes if lane["sample_index"] % 3 == 0)
    plain = next(lane for lane in lanes if lane["sample_index"] % 3 == 1)
    assert len(agentic["versions"]) == 2 and [s["kind"] for s in agentic["segments"]] == ["gen", "tool", "gen"]
    assert len(plain["versions"]) == 1 and [s["kind"] for s in plain["segments"]] == ["gen"]
    assert all(lane["status"] == "completed" for lane in lanes)
    # events land within the step's rollout window, before the consume anchor
    if payload["consume_ts"] is not None:
        assert all(lane["last_ts"] <= payload["consume_ts"] + 1e-6 for lane in lanes)


def test_rollout_trajectories_single_sample_filter(combined):
    # [FUNCTION DOC] Passing a sample index filter returns exactly that one lane.
    store, reader, _ = combined
    client = TestClient(make_app(store, reader))
    payload = client.get("/api/rollout/1/trajectories", params={"sample_index": SAMPLES_PER_STEP + 1}).json()
    [lane] = payload["lanes"]
    assert lane["sample_index"] == SAMPLES_PER_STEP + 1


def test_rollout_trajectories_empty_for_probe_less_runs(tmp_path):
    # [FUNCTION DOC] With a dump only (no telemetry or trajectory events), the endpoint returns an
    # empty lanes list.
    dump_dummy_run(tmp_path)  # dump only: no telemetry, no events
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))
    payload = client.get("/api/rollout/0/trajectories").json()
    assert payload["lanes"] == []
