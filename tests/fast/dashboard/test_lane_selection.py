# [MODULE DOC]
# Tests for timeline lane selection and the queries built on it, using the dummy telemetry
# fixture (single GPU node, train rank r on gpu r, engine B on gpus 2-3, every lane colocated)
# and a tiny hand-made store: the lane selector grammar and its error cases, lane-filtered
# phases and gpu series, the heatmap (utilisation bucket maximum, memory scaled to peak, phase
# carpet with dominant phase and initialize band, lane filtering), outlier ranking, and the
# HTTP endpoints for these including 400 responses for invalid parameters.
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from tests.fast.dashboard.dummy_telemetry import DRAIN_A, DRAIN_B, ENGINE_B_NEW, GPU_NODE, dump_dummy_telemetry

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import GpuSample, Meta, MetricStore


@pytest.fixture
def loaded(tmp_path):
    # [FUNCTION DOC] HELPER: Fixture. Writes dummy telemetry into the temporary path and returns
    # a pair of the loaded store (from its dashboard subdirectory) and the ground truth.
    truth = dump_dummy_telemetry(tmp_path)
    return MetricStore.load(tmp_path / "dashboard"), truth


# ------------------------------ lane grammar ---------------------------------


def test_resolve_lanes_grammar(loaded):
    # [FUNCTION DOC] Asserts the selector grammar on the dummy store: none or "all" mean no
    # filter; "rank:N" and "rank:A-B" select the lane of the train rank; "gpu:node:id" selects one
    # gpu; "g:N" and "g:A-B" use global lane numbers (equal to gpu ids on one node); the lane
    # index has one entry per gpu in order; "node:ip" selects all gpus of the node (unknown node
    # yields an empty set, not an error); "engine:port" selects the gpus of that engine
    # (matching either address); comma-separated selectors union; "role:train" equals
    # "role:rollout" in colocated data; "every:N" selects every Nth lane starting at 0.
    store, truth = loaded
    assert store.resolve_lanes(None) is None
    assert store.resolve_lanes("all") is None
    # dummy fixture: train rank r runs on gpu r
    assert store.resolve_lanes("rank:1") == {(GPU_NODE, 1)}
    assert store.resolve_lanes("rank:0-2") == {(GPU_NODE, 0), (GPU_NODE, 1), (GPU_NODE, 2)}
    assert store.resolve_lanes(f"gpu:{GPU_NODE}:3") == {(GPU_NODE, 3)}
    # global lane numbers: single-node fixture makes them equal the gpu ids
    assert store.resolve_lanes("g:2") == {(GPU_NODE, 2)}
    assert store.resolve_lanes("g:1-3") == {(GPU_NODE, g) for g in (1, 2, 3)}
    assert [e["index"] for e in store.lane_index()] == list(range(truth.gpus))
    assert store.resolve_lanes(f"node:{GPU_NODE}") == {(GPU_NODE, g) for g in range(truth.gpus)}
    assert store.resolve_lanes("node:10.9.9.9") == set()  # valid selector, no match
    # engine B (either addr) covers gpus 2-3
    assert store.resolve_lanes("engine:15004") == {(GPU_NODE, 2), (GPU_NODE, 3)}
    assert store.resolve_lanes("rank:0, engine:15000") == {(GPU_NODE, 0), (GPU_NODE, 1)}
    # dummy telemetry is colocate-shaped: every lane carries both roles
    assert store.resolve_lanes("role:train") == store.resolve_lanes("role:rollout")
    assert store.resolve_lanes("every:2") == {(GPU_NODE, g) for g in range(0, truth.gpus, 2)}


def test_resolve_lanes_rejects_bad_grammar(loaded):
    # [FUNCTION DOC] Each of the malformed selectors (unknown word, missing argument, unknown
    # role, gpu without node, non-numeric global lane, zero stride) raises a value error.
    store, _ = loaded
    for bad in ("bananas", "rank", "role:banana", "gpu:3", "g:x", "every:0"):
        with pytest.raises(ValueError):
            store.resolve_lanes(bad)


def test_queries_respect_lane_filter(loaded):
    # [FUNCTION DOC] With a filter of one lane (gpu 2): the phase query returns a non-empty list
    # all on that lane containing rollout, actor_train and initialize phases; the gpu series
    # query returns exactly that lane's key.
    store, truth = loaded
    only = {(GPU_NODE, 2)}
    phases = store.phases_by_lane(lanes=only)
    assert phases and {(p["node"], p["gpu"]) for p in phases} == only
    assert {p["name"] for p in phases} >= {"rollout", "actor_train", "initialize"}

    series = store.gpu_series(lanes=only)
    assert set(series) == {f"{GPU_NODE}:2"}


# -------------------------------- heatmap ------------------------------------


def _tiny_store(tmp_path):
    # [FUNCTION DOC] HELPER: Builds a store with two gpus on node "n" sampled at t=0..9: gpu 0 at
    # util 10 and 1000 MB, gpu 1 at util 50 and 4000 MB, power 100, plus a spike for gpu 0 at
    # t=4 (util 90, 8000 MB). Flushes and returns the reloaded store.
    writer = MetricStore(tmp_path / "dashboard")
    writer.write_meta(Meta(run_name="tiny", start_ts=0.0, args={}))
    for ts in range(10):
        writer.append(GpuSample(ts=float(ts), node="n", gpu=0, util=10, mem_mb=1000, power_w=100))
        writer.append(GpuSample(ts=float(ts), node="n", gpu=1, util=50, mem_mb=4000, power_w=100))
    writer.append(GpuSample(ts=4.0, node="n", gpu=0, util=90, mem_mb=8000, power_w=100))  # spike
    writer.flush()
    return MetricStore.load(tmp_path / "dashboard")


def test_heatmap_util_bucket_max(tmp_path):
    # [FUNCTION DOC] Utilisation heatmap over 0-10 s in 5 buckets: values are one byte per cell
    # in a 2 by 5 matrix; rows are gpus [0, 1]; scale max is 100. Each cell is the bucket maximum
    # scaled to 0-255 (truncated): gpu 0 is 10% except the spike bucket (t=4, bucket 2) at 90%;
    # gpu 1 is 50% everywhere.
    store = _tiny_store(tmp_path)
    result = store.heatmap("util", t0=0.0, t1=10.0, x_buckets=5)
    matrix = np.frombuffer(result["values"], dtype=np.uint8).reshape(2, 5)
    assert [r["gpu"] for r in result["rows"]] == [0, 1]
    assert result["scale"] == {"max": 100.0}
    # lane 0: util 10 everywhere except the spike bucket (t=4 -> bucket 2)
    expected_low = int(10 / 100 * 255)
    assert matrix[0, 2] == int(90 / 100 * 255)
    assert matrix[0, 0] == matrix[0, 4] == expected_low
    assert (matrix[1] == int(50 / 100 * 255)).all()


def test_heatmap_mem_scales_to_peak(tmp_path):
    # [FUNCTION DOC] Memory heatmap scales to the observed peak: scale max is 8000, the spike
    # cell is 255, and gpu 1's 4000 MB cell is half scale (truncated).
    store = _tiny_store(tmp_path)
    result = store.heatmap("mem_mb", t0=0.0, t1=10.0, x_buckets=5)
    matrix = np.frombuffer(result["values"], dtype=np.uint8).reshape(2, 5)
    assert result["scale"] == {"max": 8000.0}
    assert matrix[0, 2] == 255  # the 8000 MB spike
    assert matrix[1, 0] == int(4000 / 8000 * 255)


def test_heatmap_phase_paints_dominant_and_initialize(loaded):
    # [FUNCTION DOC] Phase heatmap with 400 buckets: the palette's index 0 is the empty string and
    # includes initialize, rollout, actor_train and train_wait; the first row's roles are rollout
    # and train (for the divider); lane 0 starts with the initialize band, contains rollout cells,
    # and actor_train wins over idle in shared buckets.
    store, truth = loaded
    result = store.heatmap("phase", x_buckets=400)
    palette = result["palette"]
    matrix = np.frombuffer(result["values"], dtype=np.uint8).reshape(len(result["rows"]), 400)
    assert palette[0] == ""
    assert {"initialize", "rollout", "actor_train", "train_wait"} <= set(palette)

    # rows carry roles so the carpet can draw the train/rollout divider
    assert result["rows"][0]["roles"] == ["rollout", "train"]

    lane0 = matrix[0]
    # the run starts with the initialize band, then rollout
    assert palette[lane0[0]] == "initialize"
    rollout_id = palette.index("rollout")
    assert (lane0 == rollout_id).sum() > 0
    # active phases win over idle in shared buckets: actor_train present
    assert (lane0 == palette.index("actor_train")).sum() > 0


def test_heatmap_respects_lane_selection(loaded):
    # [FUNCTION DOC] A utilisation heatmap restricted to one lane has exactly that row and 50
    # value bytes.
    store, _ = loaded
    result = store.heatmap("util", lanes={(GPU_NODE, 1)}, x_buckets=50)
    assert [(r["node"], r["gpu"]) for r in result["rows"]] == [(GPU_NODE, 1)]
    assert len(result["values"]) == 50


# -------------------------------- outliers -----------------------------------


def test_outliers_lowest_util(loaded):
    # [FUNCTION DOC] Ranking by lowest mean utilisation (top 4): since engine B drains earlier
    # than engine A, the two lowest-ranked lanes are gpus 2 and 3, and scores ascend from first
    # to last.
    store, truth = loaded
    ranked = store.outliers("lowest_util", top_k=4)
    # engine-B lanes (gpus 2-3) drain earlier each rollout -> lower mean util
    assert DRAIN_B < DRAIN_A
    assert {entry["gpu"] for entry in ranked[:2]} == {2, 3}
    assert ranked[0]["score"] < ranked[-1]["score"]


def test_outliers_slowest_phase(loaded):
    # [FUNCTION DOC] Ranking by slowest update_weights phase (top 4): the fixture's duration
    # grows with rank (4 + 0.3 * rank) so the order of gpus is 3, 2, 1, 0. An unknown criterion
    # raises a value error.
    store, truth = loaded
    ranked = store.outliers("slowest_phase:update_weights", top_k=4)
    # fixture skew: update_weights duration grows with rank (4 + 0.3*rank)
    assert [entry["gpu"] for entry in ranked] == [3, 2, 1, 0]

    with pytest.raises(ValueError):
        store.outliers("bananas")


# --------------------------------- server ------------------------------------


def test_server_endpoints(tmp_path):
    # [FUNCTION DOC] Builds the web app over dummy telemetry and exercises endpoints through a
    # test client. Heatmap: status 200, body is a 4-byte little-endian header length, a json
    # header (metric, bucket count, rows) and then one byte per cell (rows times 100). Phases
    # with a rank:0 filter only return lane gpu 0. Gpu endpoint filtered by the port of engine B
    # returns the lanes of gpus 2 and 3. Outliers with top 2 returns 2 entries. Invalid lane
    # selector, unknown heatmap metric, unknown outlier criterion, and a bucket count of 1 each
    # return status 400.
    dump_dummy_telemetry(tmp_path)
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))

    response = client.get("/api/timeline/heatmap", params={"metric": "util", "x_buckets": 100})
    assert response.status_code == 200
    body = response.content
    header_len = int.from_bytes(body[:4], "little")
    header = json.loads(body[4 : 4 + header_len])
    matrix = np.frombuffer(body[4 + header_len :], dtype=np.uint8)
    assert header["metric"] == "util" and header["x_buckets"] == 100
    assert len(matrix) == len(header["rows"]) * 100

    phases = client.get("/api/timeline/phases", params={"lanes": "rank:0"}).json()["phases"]
    assert {(p["node"], p["gpu"]) for p in phases} == {(GPU_NODE, 0)}

    gpu = client.get("/api/timeline/gpu", params={"lanes": f"engine:{ENGINE_B_NEW.rsplit(':', 1)[1]}"}).json()
    assert set(gpu["lanes"]) == {f"{GPU_NODE}:2", f"{GPU_NODE}:3"}

    outliers = client.get("/api/timeline/outliers", params={"criterion": "lowest_util", "top_k": 2}).json()
    assert len(outliers["outliers"]) == 2

    assert client.get("/api/timeline/phases", params={"lanes": "bananas"}).status_code == 400
    assert client.get("/api/timeline/heatmap", params={"metric": "nope"}).status_code == 400
    assert client.get("/api/timeline/outliers", params={"criterion": "nope"}).status_code == 400
    assert client.get("/api/timeline/heatmap", params={"x_buckets": 1}).status_code == 400
