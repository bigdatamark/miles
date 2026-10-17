# [MODULE DOC]
# Tests for the CI metric-history backend (CiHistoryBackend, writing JSONL records of selected
# training/rollout metrics into a directory named by an environment variable) and for the CI
# harness helpers that give each test attempt its own record directory and merge the passing
# attempt's records. Registers itself as a CPU test in suite stage-a-cpu. Every backend test points
# the record directory environment variable at a temporary directory, initializes a non-primary
# backend, logs metrics, and inspects the written JSONL files.
"""Tests for the CI metric-history collection backend and harness handoff.

Covers the record structure produced by `CiHistoryBackend` (target keys present,
JSONL parseable, raw unreduced series) and the harness per-attempt record-dir
handoff in `tests.ci.ci_utils`.
"""

import json
import logging
import os
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.ci.ci_utils import TestFile, _attempt_record_dir, _merge_attempt_records, run_unittest_files

from miles.utils.tracking_utils.ci_history import RECORD_DIR_ENV, TARGET_METRIC_KEYS, CiHistoryBackend

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


def test_all_backends_registered():
    # [FUNCTION DOC] Imports the tracking backend registry and asserts its keys are exactly wandb,
    # tensorboard, mlflow, prometheus, ci_history and miles_dashboard, and that the ci_history entry
    # pairs the CiHistoryBackend class with the enabling flag name ci_enable_metrics_capture. Guards
    # against a backend silently vanishing from the registry.
    # The full registry lives in tracking_utils/tracking.py (the entry point),
    # not base.py (base must not back-import a backend -> circular). Guard
    # against the silent-drop failure mode: if a registry entry is removed or an
    # import is pruned, the backend vanishes with no error. Assert the full set.
    from miles.utils.tracking_utils.tracking import BACKEND_REGISTRY

    assert set(BACKEND_REGISTRY) == {"wandb", "tensorboard", "mlflow", "prometheus", "ci_history", "miles_dashboard"}
    cls, flag = BACKEND_REGISTRY["ci_history"]
    assert cls is CiHistoryBackend
    assert flag == "ci_enable_metrics_capture"


def _read_jsonl(path):
    # [FUNCTION DOC] HELPER: Read a UTF-8 file and return the list of JSON values parsed from its
    # non-blank lines.
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_record_has_target_keys_and_is_parseable(tmp_path, monkeypatch):
    # [FUNCTION DOC] Logs two train steps (grad norm, ppo kl) and one rollout step (raw reward and a
    # tito mismatch rate), then finishes. Asserts exactly one JSONL file exists, its records are keyed
    # by metric name with exactly the four target metrics, and each series holds the raw unreduced
    # (step, value) pairs in logging order.
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    backend = CiHistoryBackend()
    backend.init(object(), primary=False)

    backend.log({"train/grad_norm": 1.5, "train/ppo_kl": 0.0, "train/step": 0}, step=0)
    backend.log({"train/grad_norm": 2.5, "train/ppo_kl": 0.1, "train/step": 1}, step=1)
    backend.log(
        {
            "rollout/raw_reward": 0.3,
            "rollout/tito_session_mismatch_rate/v2/assistant_text": 0.125,
            "rollout/step": 0,
        },
        step=0,
    )
    backend.finish()

    files = [f for f in os.listdir(tmp_path) if f.endswith(".jsonl")]
    assert len(files) == 1, f"expected one record file, got {files}"

    records = _read_jsonl(os.path.join(tmp_path, files[0]))
    by_metric = {r["metric"]: r["series"] for r in records}

    assert set(by_metric) == {
        "train/grad_norm",
        "train/ppo_kl",
        "rollout/raw_reward",
        "rollout/tito_session_mismatch_rate/v2/assistant_text",
    }
    # Raw series are preserved unreduced: both grad_norm points are present.
    assert by_metric["train/grad_norm"] == [[0, 1.5], [1, 2.5]]
    assert by_metric["train/ppo_kl"] == [[0, 0.0], [1, 0.1]]
    assert by_metric["rollout/raw_reward"] == [[0, 0.3]]
    assert by_metric["rollout/tito_session_mismatch_rate/v2/assistant_text"] == [[0, 0.125]]


def test_v1_v2_tito_metrics_remain_separate_after_attempt_merge(tmp_path, monkeypatch):
    # [FUNCTION DOC] Creates two backend instances in turn, each logging the tito session mismatch
    # rate under a different version segment (v1 with 0.125, v2 with 0.25) at step 0, then merges the
    # directory's attempt records into a sibling merged file. Asserts the merged file keeps the two
    # versioned metric names as separate series.
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    metric_prefix = "rollout/tito_session_mismatch_rate"
    for version, value in (("v1", 0.125), ("v2", 0.25)):
        backend = CiHistoryBackend()
        backend.init(object(), primary=False)
        metric_key = f"{metric_prefix}/{version}/assistant_text"
        backend.log({metric_key: value}, step=0)

    merged_path = tmp_path.with_suffix(".merged.jsonl")
    _merge_attempt_records(str(tmp_path), str(merged_path))

    by_metric = {record["metric"]: record["series"] for record in _read_jsonl(merged_path)}
    assert by_metric == {
        f"{metric_prefix}/v1/assistant_text": [[0, 0.125]],
        f"{metric_prefix}/v2/assistant_text": [[0, 0.25]],
    }


def test_only_target_keys_captured(tmp_path, monkeypatch):
    # [FUNCTION DOC] Logs a mix of target and non-target keys (including an unversioned tito mismatch
    # rate and a policy-gradient loss). Asserts only the grad norm is recorded and the recorded set is
    # a subset of the target metric keys.
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    backend = CiHistoryBackend()
    backend.init(object(), primary=False)

    backend.log(
        {
            "train/grad_norm": 1.0,
            "train/pg_loss": 9.0,
            "rollout/tito_session_mismatch_rate/assistant_text": 0.125,
            "train/step": 0,
        },
        step=0,
    )
    backend.finish()

    records = _read_jsonl(os.path.join(tmp_path, os.listdir(tmp_path)[0]))
    metrics = {r["metric"] for r in records}
    assert metrics == {"train/grad_norm"}
    assert metrics <= set(TARGET_METRIC_KEYS)


def test_non_numeric_target_metric_errors_without_partial_capture(tmp_path, monkeypatch, caplog):
    # [FUNCTION DOC] Logs a step where one target metric is a list. Asserts a type error naming the
    # metric is raised and an error log line says it must be int or float, got list. After that
    # failed step, logs a valid step and asserts only the valid step's grad norm point is recorded
    # (the failed step captured nothing, not even its valid grad norm).
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    backend = CiHistoryBackend()
    backend.init(object(), primary=False)

    with caplog.at_level(logging.ERROR, logger="miles.utils.tracking_utils.ci_history"):
        with pytest.raises(TypeError, match="train/ppo_kl"):
            backend.log({"train/grad_norm": 1.0, "train/ppo_kl": [0.1], "train/step": 0}, step=0)

    assert "CI history metric 'train/ppo_kl' must be int or float, got list" in caplog.text

    backend.log({"train/grad_norm": 2.0, "train/step": 1}, step=1)
    backend.finish()

    records = _read_jsonl(os.path.join(tmp_path, os.listdir(tmp_path)[0]))
    by_metric = {r["metric"]: r["series"] for r in records}
    assert by_metric["train/grad_norm"] == [[1, 2.0]]


def _reject_bare_constant(name):
    # [FUNCTION DOC] HELPER: JSON parse-constant hook that always fails. Raises an assertion error
    # naming the bare non-finite token that was found; used to enforce strict JSON.
    raise AssertionError(f"bare {name} token in record; non-finite must be a string marker")


def test_non_finite_values_recorded_as_strict_json_markers(tmp_path, monkeypatch):
    # [FUNCTION DOC] Logs NaN, positive infinity and negative infinity values for target metrics.
    # Re-parses every line with a hook that rejects bare non-finite tokens (so the file is strict
    # JSON), then asserts the values are stored as the strings "NaN", "Infinity" and "-Infinity"
    # while finite values stay numbers.
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    backend = CiHistoryBackend()
    backend.init(object(), primary=False)

    backend.log({"train/grad_norm": 1.5, "train/step": 0}, step=0)
    backend.log({"train/grad_norm": float("nan"), "train/step": 1}, step=1)
    backend.log({"train/ppo_kl": float("inf"), "train/step": 1}, step=1)
    backend.log({"rollout/raw_reward": float("-inf"), "rollout/step": 0}, step=0)
    backend.finish()

    path = os.path.join(tmp_path, os.listdir(tmp_path)[0])
    # Strict JSON: parse_constant fires only on bare NaN/Infinity tokens.
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                json.loads(line, parse_constant=_reject_bare_constant)

    by_metric = {r["metric"]: r["series"] for r in _read_jsonl(path)}
    assert by_metric["train/grad_norm"] == [[0, 1.5], [1, "NaN"]]
    assert by_metric["train/ppo_kl"] == [[1, "Infinity"]]
    assert by_metric["rollout/raw_reward"] == [[0, "-Infinity"]]


def test_record_carries_no_identity(tmp_path, monkeypatch):
    # [FUNCTION DOC] Logs one grad norm point and reads the raw file text. Asserts it contains none
    # of the identity-revealing substrings test_path, test_, .py or filename.
    # KNOWN ISSUE: the check for the substring "test_" would also reject a legitimate metric name
    # containing it; it only passes because current target metric names do not.
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    backend = CiHistoryBackend()
    backend.init(object(), primary=False)
    backend.log({"train/grad_norm": 1.0}, step=0)
    backend.finish()

    raw = open(os.path.join(tmp_path, os.listdir(tmp_path)[0]), encoding="utf-8").read()
    for forbidden in ("test_path", "test_", ".py", "filename"):
        assert forbidden not in raw, f"record leaked identity token {forbidden!r}: {raw}"


def test_disabled_when_env_unset(tmp_path, monkeypatch):
    # [FUNCTION DOC] Removes the record directory variable, runs init, log and finish, and asserts
    # nothing was written under the temporary directory (the backend is a no-op when disabled).
    monkeypatch.delenv(RECORD_DIR_ENV, raising=False)
    backend = CiHistoryBackend()
    backend.init(object(), primary=False)
    backend.log({"train/grad_norm": 1.0}, step=0)
    backend.finish()
    # No directory env => nothing written anywhere under tmp_path.
    assert not list(tmp_path.iterdir())


def test_flush_survives_without_finish(tmp_path, monkeypatch):
    # [FUNCTION DOC] Logs two steps without ever calling finish (as actor processes do). Asserts the
    # file already contains both points, i.e. each log call persists a full snapshot.
    # Actor processes never call finish(); each log() must persist a snapshot.
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    backend = CiHistoryBackend()
    backend.init(object(), primary=False)
    backend.log({"train/grad_norm": 1.0, "train/step": 0}, step=0)
    backend.log({"train/grad_norm": 2.0, "train/step": 1}, step=1)
    # Deliberately no finish().

    records = _read_jsonl(os.path.join(tmp_path, os.listdir(tmp_path)[0]))
    by_metric = {r["metric"]: r["series"] for r in records}
    assert by_metric["train/grad_norm"] == [[0, 1.0], [1, 2.0]]


def test_distinct_backend_instances_do_not_clobber_current_record_files(tmp_path, monkeypatch):
    # [FUNCTION DOC] Creates two backends against the same directory, logs a different metric on
    # each, finishes both, and asserts two separate JSONL files exist (one per instance). This pins
    # the current file-naming behavior, not a harness contract to merge writer outputs.
    # This covers the backend's current file naming detail, not a CI harness
    # contract to merge multiple writer outputs.
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    b1 = CiHistoryBackend()
    b1.init(object(), primary=False)
    b2 = CiHistoryBackend()
    b2.init(object(), primary=False)

    b1.log({"train/grad_norm": 1.0}, step=0)
    b2.log({"rollout/raw_reward": 0.5}, step=0)
    b1.finish()
    b2.finish()

    files = [f for f in os.listdir(tmp_path) if f.endswith(".jsonl")]
    assert len(files) == 2, f"expected one file per backend instance, got {files}"


def test_attempt_isolation(tmp_path):
    # [FUNCTION DOC] Computes the attempt record directory for a file name with a space and a
    # subdirectory for attempts 1 and 2. Asserts the two differ, share the same parent, the parent's
    # name starts with the sanitized file name (slash and space replaced by underscores) followed by
    # a hyphen, and the directory ends with attempt-1 / attempt-2.
    base = str(tmp_path)
    filename = "nested dir/test bar.py"

    d1 = _attempt_record_dir(base, filename, attempt=1)
    d2 = _attempt_record_dir(base, filename, attempt=2)
    assert d1 != d2, "attempts must not share a record directory"
    assert os.path.dirname(d1) == os.path.dirname(d2)
    assert os.path.basename(os.path.dirname(d1)).startswith("nested_dir_test_bar.py-")
    assert d1.endswith(os.path.join("attempt-1"))
    assert d2.endswith(os.path.join("attempt-2"))


def test_attempt_record_dir_disambiguates_test_filenames(tmp_path):
    # [FUNCTION DOC] Asserts attempt directories differ for the same base name in different suites,
    # and for names that sanitize to the same string (foo/bar.py versus foo_bar.py) even though both
    # parent names start with foo_bar.py- (the suffix disambiguates).
    base = str(tmp_path)

    same_basename_a = _attempt_record_dir(base, "suite_a/test_a.py", attempt=1)
    same_basename_b = _attempt_record_dir(base, "suite_b/test_a.py", attempt=1)
    assert same_basename_a != same_basename_b

    sanitized_collision_a = _attempt_record_dir(base, "foo/bar.py", attempt=1)
    sanitized_collision_b = _attempt_record_dir(base, "foo_bar.py", attempt=1)
    assert os.path.basename(os.path.dirname(sanitized_collision_a)).startswith("foo_bar.py-")
    assert os.path.basename(os.path.dirname(sanitized_collision_b)).startswith("foo_bar.py-")
    assert sanitized_collision_a != sanitized_collision_b


def test_run_unittest_files_hands_off_attempt_record_dir(tmp_path, monkeypatch):
    # [FUNCTION DOC] Changes into the temporary directory, sets the record directory variable, and
    # writes a child script that records the directory it was given into a marker file and exits with
    # failure the first time (leaving a state file) and success the second. Runs the harness on that
    # one file with retry enabled, two attempts and no wait. Asserts the return code is 0, each
    # attempt saw its own attempt directory, and only the passing attempt's records were merged into
    # a merged JSONL file (the failed attempt has none).
    monkeypatch.chdir(tmp_path)
    record_base = tmp_path / "records"
    monkeypatch.setenv(RECORD_DIR_ENV, str(record_base))

    state_file = tmp_path / "retry_state.txt"
    child = tmp_path / "t retry.py"
    child.write_text(
        f"""
import os
import pathlib
import sys

record_dir = os.environ[{RECORD_DIR_ENV!r}]
pathlib.Path(record_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(record_dir, "seen.txt").write_text(record_dir)

state_file = {str(state_file)!r}
if os.path.exists(state_file):
    sys.exit(0)

pathlib.Path(state_file).write_text("1")
print("accuracy retry", flush=True)
sys.exit(1)
"""
    )

    rc = run_unittest_files(
        [TestFile(name=child.name, estimated_time=1)],
        timeout_per_file=10,
        enable_retry=True,
        max_attempts=2,
        retry_wait_seconds=0,
    )

    assert rc == 0
    attempt_1 = Path(_attempt_record_dir(str(record_base), child.name, attempt=1))
    attempt_2 = Path(_attempt_record_dir(str(record_base), child.name, attempt=2))
    assert (attempt_1 / "seen.txt").read_text() == str(attempt_1)
    assert (attempt_2 / "seen.txt").read_text() == str(attempt_2)
    # Only the PASSING attempt's records are merged for the gate hook; the
    # failed attempt keeps its per-process records unmerged.
    assert not Path(f"{attempt_1}.merged.jsonl").exists()
    assert Path(f"{attempt_2}.merged.jsonl").exists()
