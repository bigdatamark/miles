# [MODULE DOC]
# Tests for the dashboard web application through a test client: meta endpoint (mode, run name,
# rollout ids, metric keys, step keys, capabilities, wandb url, data buffer length, per-metric
# step keys, engine metric catalog), advisory endpoint, metric series (store-backed and
# dump-derived keys), per-rollout summary / groups / tokens / messages endpoints including error
# status codes (400, 404, 503), duplicate-leaf occurrences, NaN serialisation, static index,
# demo directory generation, and the MFU summary. Data comes from the dummy dump and telemetry
# writers plus hand-appended metric records.
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from tests.fast.dashboard.dummy_dump import dump_dummy_run

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import _wandb_url, make_app
from miles.dashboard.store import Meta, MetricsRecord, MetricStore


@pytest.fixture
def dump_dir(tmp_path):
    # [FUNCTION DOC] HELPER: Fixture. Writes a 2-step dummy dump with eval into the temporary
    # path, then appends hand-made records to its dashboard store: run "dummy-run" starting at
    # 100; two rollout-step metric records (steps 0, 1; rewards 0.5, 0.6; MFU 0.4, 0.3; peak
    # tflops 989) and one train-step record (step 7, loss 1.5). Returns the path.
    dump_dummy_run(tmp_path, steps=2, with_eval=True)
    writer = MetricStore(tmp_path / "dashboard")
    writer.write_meta(Meta(run_name="dummy-run", start_ts=100.0, args={}))
    writer.append(
        MetricsRecord(
            ts=101.0,
            step_key="rollout/step",
            step=0,
            metrics={"rollout/rewards_mean": 0.5, "perf/actor_train_mfu": 0.4, "perf/mfu_peak_tflops": 989.0},
        )
    )
    writer.append(
        MetricsRecord(
            ts=102.0,
            step_key="rollout/step",
            step=1,
            metrics={"rollout/rewards_mean": 0.6, "perf/actor_train_mfu": 0.3, "perf/mfu_peak_tflops": 989.0},
        )
    )
    writer.append(MetricsRecord(ts=103.0, step_key="train/step", step=7, metrics={"train/loss": 1.5}))
    writer.flush()
    return tmp_path


@pytest.fixture
def client(dump_dir):
    # [FUNCTION DOC] HELPER: Fixture. Returns a test client over the web app built from the
    # loaded store and a dump reader on the fixture directory.
    store = MetricStore.load(dump_dir / "dashboard")
    return TestClient(make_app(store, DumpReader(dump_dir)))


def test_meta(client):
    # [FUNCTION DOC] The meta endpoint reports: static mode, run name "dummy-run", rollout ids
    # train [0, 1] and eval [0], the rollout reward key among metric keys but not the dump-derived
    # reward key (not advertised when telemetry exists), step keys sorted, capabilities with
    # metrics and tokenizer true but timeline false, no wandb url (empty arguments) and no data
    # buffer length.
    meta = client.get("/api/meta").json()
    assert meta["mode"] == "static"
    assert meta["run_name"] == "dummy-run"
    assert meta["rollout_ids"] == {"train": [0, 1], "eval": [0]}
    assert "rollout/rewards_mean" in meta["metric_keys"]
    # telemetry present: the catalog stays wandb-shaped, dump aggregates
    # are not advertised (they remain servable on explicit request)
    assert "dump/reward_mean" not in meta["metric_keys"]
    assert meta["step_keys"] == ["rollout/step", "train/step"]
    assert meta["capabilities"]["has_metrics"] is True
    assert meta["capabilities"]["has_tokenizer"] is True
    assert meta["capabilities"]["has_timeline"] is False
    assert meta["wandb_url"] is None  # empty args snapshot in this fixture
    assert meta["data_buffer_length"] is None  # never reported in this fixture


class TestMeta:
    # [CLASS DOC] Additional meta endpoint tests.
    def test_meta_reports_the_step_key_for_each_namespaced_metric(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two namespaced policies log loss against their own train step key and
        # reward against their own rollout step key. The meta endpoint's mapping from metric key
        # to step key reports exactly those pairings.
        """Each namespaced metric must report the exact axis recorded with its point."""
        writer = MetricStore(tmp_path / "dashboard")
        writer.append(
            MetricsRecord(
                ts=1.0,
                step_key="alpha/train/step",
                step=3,
                metrics={"alpha/train/loss": 1.5},
            )
        )
        writer.append(
            MetricsRecord(
                ts=2.0,
                step_key="beta/rollout/step",
                step=4,
                metrics={"beta/rollout/reward": 0.5},
            )
        )
        writer.flush()
        client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))

        assert client.get("/api/meta").json()["step_key_of_metric_key"] == {
            "alpha/train/loss": "alpha/train/step",
            "beta/rollout/reward": "beta/rollout/step",
        }


def test_meta_reports_latest_data_buffer_length(dump_dir):
    # [FUNCTION DOC] After appending buffer samples of length 3 then 6, a freshly built app's
    # meta reports data buffer length 6.
    from miles.dashboard.store import DataBufferSample

    writer = MetricStore(dump_dir / "dashboard")
    writer.append(DataBufferSample(ts=104.0, length=3))
    writer.append(DataBufferSample(ts=105.0, length=6))
    writer.flush()

    client = TestClient(make_app(MetricStore.load(dump_dir / "dashboard"), DumpReader(dump_dir)))
    assert client.get("/api/meta").json()["data_buffer_length"] == 6


def test_advisory_endpoint(client):
    # [FUNCTION DOC] The advisory endpoint returns 200 with exactly one advisory: a warning whose
    # message mentions truncation (a quarter of the newest rollout's samples truncate in the
    # dummy dump; no engine series exist). A reversed time window (t0 greater than t1) returns 400.
    resp = client.get("/api/advisory")
    assert resp.status_code == 200
    # no engine series in the fixture, but the reader-side rules see the
    # dummy dump's newest rollout, where a quarter of the samples truncate
    [advisory] = resp.json()["advisories"]
    assert advisory["level"] == "warning"
    assert "truncated" in advisory["message"]
    assert client.get("/api/advisory", params={"t0": 5, "t1": 1}).status_code == 400


def test_wandb_url():
    # [FUNCTION DOC] The url builder returns "https://wandb.ai/team/project/runs/id" for complete
    # arguments; a custom host (with trailing slash) replaces the base; if any of team, project
    # or run id is missing the result is none.
    full = dict(wandb_team="radixark", wandb_project="miles", wandb_run_id="abc123")
    assert _wandb_url(full) == "https://wandb.ai/radixark/miles/runs/abc123"
    assert (
        _wandb_url({**full, "wandb_host": "https://wandb.internal/"})
        == "https://wandb.internal/radixark/miles/runs/abc123"
    )
    for missing in ("wandb_team", "wandb_project", "wandb_run_id"):
        assert _wandb_url({k: v for k, v in full.items() if k != missing}) is None


def test_metrics_from_store(client):
    # [FUNCTION DOC] The metrics endpoint returns, for the rollout reward key, x steps [0, 1], y
    # [0.5, 0.6] and timestamps [101, 102]; with the train step axis selected, loss has x [7];
    # an unknown key returns empty lists.
    series = client.get("/api/metrics", params={"keys": "rollout/rewards_mean"}).json()
    assert series["rollout/rewards_mean"] == {"x": [0, 1], "y": [0.5, 0.6], "ts": [101.0, 102.0]}

    series = client.get("/api/metrics", params={"keys": "train/loss", "x": "train/step"}).json()
    assert series["train/loss"]["x"] == [7]

    series = client.get("/api/metrics", params={"keys": "no/such_key"}).json()
    assert series["no/such_key"] == {"x": [], "y": [], "ts": []}


def test_mfu_and_its_denominator_are_served_together(client):
    # [FUNCTION DOC] Both the MFU key and the peak tflops key appear in the meta metric keys and
    # are served by the metrics endpoint with y values [0.4, 0.3] and [989, 989].
    meta = client.get("/api/meta").json()
    assert "perf/actor_train_mfu" in meta["metric_keys"]
    assert "perf/mfu_peak_tflops" in meta["metric_keys"]

    series = client.get("/api/metrics", params={"keys": "perf/actor_train_mfu,perf/mfu_peak_tflops"}).json()
    assert series["perf/actor_train_mfu"]["y"] == [0.4, 0.3]
    assert series["perf/mfu_peak_tflops"]["y"] == [989.0, 989.0]


def test_metrics_dump_derived(client, dump_dir):
    # [FUNCTION DOC] Requesting the dump-derived mean reward key alongside the store key returns
    # x [0, 1] and y equal to the dump reader's step aggregate means for the former, and the
    # store values for the latter. An unknown dump key and an empty key list return 400.
    series = client.get("/api/metrics", params={"keys": "dump/reward_mean,rollout/rewards_mean"}).json()
    aggregates = DumpReader(dump_dir).step_aggregates()
    assert series["dump/reward_mean"]["x"] == [0, 1]
    assert series["dump/reward_mean"]["y"] == pytest.approx(aggregates["reward_mean"].to_list())
    assert series["rollout/rewards_mean"]["y"] == [0.5, 0.6]

    assert client.get("/api/metrics", params={"keys": "dump/nope"}).status_code == 400
    assert client.get("/api/metrics", params={"keys": ""}).status_code == 400


def test_summary_and_groups_endpoints(client):
    # [FUNCTION DOC] Summary for step 0 reports the rollout id, not-eval flag, 8 rows and columns
    # including sample index, occurrence and log-prob difference mean. With the eval flag it
    # reports eval true and 4 rows. Groups has 4 rows and a zero-std column. Unknown step 42
    # returns 404.
    body = client.get("/api/rollout/0/summary").json()
    assert body["rollout_id"] == 0 and body["evaluation"] is False
    assert len(body["rows"]) == 8
    assert "sample_index" in body["columns"] and "sample_occurrence" in body["columns"]
    assert "mean_abs_lp_diff" in body["columns"]

    eval_body = client.get("/api/rollout/0/summary", params={"eval": "true"}).json()
    assert eval_body["evaluation"] is True
    assert len(eval_body["rows"]) == 4

    groups = client.get("/api/rollout/0/groups").json()
    assert len(groups["rows"]) == 4
    assert "zero_std" in groups["columns"]

    assert client.get("/api/rollout/42/summary").status_code == 404


def test_tokens_endpoint(client):
    # [FUNCTION DOC] Full tokens payload has text and the id count equal to end minus start and
    # to total length; a window 1..4 gives 3 ids; the eval flag gives no train log probs but
    # rollout log probs. Unknown sample returns 404, empty range 400, unknown rollout 404.
    payload = client.get("/api/rollout/0/sample/0/tokens").json()
    assert payload["token_text"] is not None
    assert len(payload["token_ids"]) == payload["end"] - payload["start"] == payload["total_len"]

    window = client.get("/api/rollout/0/sample/0/tokens", params={"start": 1, "end": 4}).json()
    assert len(window["token_ids"]) == 3

    eval_payload = client.get("/api/rollout/0/sample/0/tokens", params={"eval": "true"}).json()
    assert eval_payload["train_log_probs"] is None
    assert eval_payload["rollout_log_probs"] is not None

    assert client.get("/api/rollout/0/sample/999999/tokens").status_code == 404
    assert client.get("/api/rollout/0/sample/0/tokens", params={"start": 5, "end": 5}).status_code == 400
    assert client.get("/api/rollout/42/sample/0/tokens").status_code == 404


def test_duplicate_tito_leaves_are_independently_viewable(tmp_path):
    # [FUNCTION DOC] Dump with a duplicated first sample index. The summary lists occurrences
    # [0, 1] for index 0; for each occurrence the tokens endpoint echoes the occurrence and the
    # row's total length. The messages endpoint accepts occurrence 1 (200); tokens for occurrence
    # 2 returns 404.
    dump_dummy_run(tmp_path, steps=1, duplicate_first_sample_index=True)
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))

    response = client.get("/api/rollout/0/summary")
    assert response.status_code == 200
    rows = [row for row in response.json()["rows"] if row["sample_index"] == 0]
    assert [row["sample_occurrence"] for row in rows] == [0, 1]

    for row in rows:
        payload = client.get("/api/rollout/0/sample/0/tokens", params={"occurrence": row["sample_occurrence"]}).json()
        assert payload["sample_occurrence"] == row["sample_occurrence"]
        assert payload["total_len"] == row["total_length"]

    assert client.get("/api/rollout/0/sample/0/messages", params={"occurrence": 1}).status_code == 200
    assert client.get("/api/rollout/0/sample/0/tokens", params={"occurrence": 2}).status_code == 404


def test_still_writing_maps_to_503(client, dump_dir):
    # [FUNCTION DOC] A fresh garbage rollout file for id 9 makes its summary endpoint return 503
    # (the dump is still being written).
    (dump_dir / "rollout_data" / "9.pt").write_bytes(b"garbage")  # fresh mtime
    assert client.get("/api/rollout/9/summary").status_code == 503


def test_nan_values_serialize_as_null(dump_dir):
    # [FUNCTION DOC] Overwrites one sample's train log probs in rank 0's file with NaN, then ages
    # the file 100 s. The summary response must be valid json; the poisoned sample's log prob
    # difference and importance ratio means are null.
    path = dump_dir / "train_data" / "0_0.pt"
    pack = torch.load(path, weights_only=False)
    pack["rollout_data"]["log_probs"][0][:] = float("nan")
    torch.save(pack, path)
    stamp = time.time() - 100
    import os

    os.utime(path, (stamp, stamp))

    client = TestClient(make_app(MetricStore.load(dump_dir / "dashboard"), DumpReader(dump_dir)))
    body = client.get("/api/rollout/0/summary").json()  # json.loads would fail on bare NaN
    poisoned = pack["rollout_data"]["sample_indices"][0]
    row = next(r for r in body["rows"] if r["sample_index"] == poisoned)
    assert row["mean_abs_lp_diff"] is None
    assert row["mean_imp_ratio"] is None


def test_static_index(client):
    # [FUNCTION DOC] The root path returns 200 with a page containing the text "miles dashboard".
    response = client.get("/")
    assert response.status_code == 200
    assert "miles dashboard" in response.text


def test_dump_keys_advertised_only_without_telemetry(tmp_path):
    # [FUNCTION DOC] With a dump but no telemetry directory, meta reports no metrics capability
    # and advertises the dump-derived reward key.
    from tests.fast.dashboard.dummy_dump import dump_dummy_run

    dump_dummy_run(tmp_path)  # dumps only: no dashboard/ telemetry at all
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))
    meta = client.get("/api/meta").json()
    assert meta["capabilities"]["has_metrics"] is False
    assert "dump/reward_mean" in meta["metric_keys"]


def test_engine_metric_catalog(tmp_path):
    # [FUNCTION DOC] With dummy telemetry and a dummy dump in the same directory, meta's engine
    # metric keys include running requests.
    from tests.fast.dashboard.dummy_telemetry import dump_dummy_telemetry

    dump_dummy_telemetry(tmp_path)
    from tests.fast.dashboard.dummy_dump import dump_dummy_run

    dump_dummy_run(tmp_path)
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))
    meta = client.get("/api/meta").json()
    assert "sglang_num_running_reqs" in meta["engine_metric_keys"]


def test_sample_messages_from_trajectory_sidecar(tmp_path):
    # [FUNCTION DOC] Messages for sample 0 have roles system, user, assistant, tool, assistant
    # with the third message's first tool call named "lookup". Sample 2 (no conversation) and
    # any eval request return 404.
    from tests.fast.dashboard.dummy_dump import dump_dummy_run

    dump_dummy_run(tmp_path)
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))

    row = client.get("/api/rollout/0/sample/0/messages").json()
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert row["messages"][2]["tool_calls"][0]["function"]["name"] == "lookup"
    # sample 2 recorded no conversation; eval dumps carry none at all
    assert client.get("/api/rollout/0/sample/2/messages").status_code == 404
    assert client.get("/api/rollout/0/sample/0/messages?eval=1").status_code == 404


def test_make_demo_dir(tmp_path):
    # [FUNCTION DOC] Generating a demo directory yields a dump with train ids [0, 1, 2]; the app
    # over it reports a timeline capability and the same rollout ids.
    from miles.dashboard.serve import make_demo_dir

    make_demo_dir(tmp_path)
    reader = DumpReader(tmp_path)
    assert reader.rollout_ids().train == [0, 1, 2]
    store = MetricStore.load(tmp_path / "dashboard")
    client = TestClient(make_app(store, reader))
    meta = client.get("/api/meta").json()
    assert meta["capabilities"]["has_timeline"] is True
    assert meta["rollout_ids"]["train"] == [0, 1, 2]


def test_advisory_endpoint_serves_the_mfu_summary_the_rule_used(client):
    # [FUNCTION DOC] The advisory body carries an MFU summary of latest 0.3, mean 0.3, 1 step and
    # peak 989 (only the most recent step counts).
    body = client.get("/api/advisory").json()
    assert body["mfu"] == {"latest": 0.3, "mean": 0.3, "steps": 1, "peak": 989.0}
