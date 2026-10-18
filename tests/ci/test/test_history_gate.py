# [MODULE DOC]
# Offline tests for the metric-history regression gate (tests.ci.metric_history.gate:
# parse_merged_record, evaluate_gate, GateStatus). Every test builds a fixture test file that
# declares gates with register_ci_gate (a metric key, a step selector such as "last", "all" or a
# list of steps, and a constraint with relative up/down bands and optional absolute floors), writes
# a merged JSONL metric record, optionally seeds trusted baseline runs into an in-memory SQLite
# store, then evaluates the gate and inspects per-metric results (status, current value, baseline
# count and mean, reason, coordinate keys) and the overall trusted flag. Covers parsing and
# non-finite decoding, cold start, historical pass/fail, per-step fan-out, separate coordinates per
# declaration, near-zero floors, missing/empty/ill-formed series, asymmetric corridors, no specs,
# no persistence, and files registered for both CUDA and ROCm. PROVENANCE is a sample pull request
# run; LAST_KEY is the canonical key of the step selector "last". Registers itself as a CPU test in
# suite stage-a-cpu.
"""Offline tests for the metric-history regression gate.

These run fully offline against an in-memory :class:`SQLiteMetricHistoryStore`
and on-disk fixture files (a test file declaring register_ci_gate + a merged
JSONL record). No network, no real DB connection opened by the gate, no wandb.
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import pytest
from tests.ci.ci_register import CIRegistry, HWBackend, register_cpu_ci
from tests.ci.metric_history import MetricSample, RunIdentity, RunProvenance, SQLiteMetricHistoryStore
from tests.ci.metric_history.gate import GateStatus, evaluate_gate, parse_merged_record

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

PROVENANCE = RunProvenance(
    commit_sha="deadbeef",
    pr_number=1,
    github_run_id=100,
    github_run_attempt=1,
    event_name="pull_request",
    ref="refs/pull/1/merge",
)


def _key(literal: object) -> str:
    # [FUNCTION DOC] HELPER: Serialize a literal to compact JSON with sorted keys and no spaces
    # (separators comma and colon), the canonical declaration key.
    """Canonical declaration key, matching register.py's `_canonical_key`."""
    return json.dumps(literal, sort_keys=True, separators=(",", ":"))


# Keys for the fixtures below: steps / constraint literals as written there.
LAST_KEY = _key("last")


@pytest.fixture
def store():
    # [FUNCTION DOC] HELPER: Fixture yielding a fresh in-memory SQLite metric history store, closed
    # afterwards.
    s = SQLiteMetricHistoryStore(":memory:")
    yield s
    s.close()


def _write_test_file(tmp_path: Path, gate_lines: str, *, name: str = "test_e2e_fixture.py") -> str:
    # [FUNCTION DOC] HELPER: Write a fixture test file (default test_e2e_fixture.py) with the two
    # imports, one CUDA registration (est_time 600, stage-c-8-gpu-h100, label megatron, hopper) and
    # then the dedented, stripped gate lines; return its path. Built by concatenation so multi-line
    # gate blocks keep correct indentation.
    # Concatenate instead of interpolating into an indented f-string: a
    # multi-line gate block would otherwise defeat the outer dedent.
    body = (
        "from tests.ci.ci_register import register_cuda_ci\n"
        "from tests.ci.metric_history import register_ci_gate\n"
        'register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100", labels=["megatron"], hardware=["hopper"])\n'
        + textwrap.dedent(gate_lines).strip()
        + "\n"
    )
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def _write_record(tmp_path: Path, by_metric: dict[str, list], *, name: str = "merged.jsonl") -> str:
    # [FUNCTION DOC] HELPER: Write a JSONL record (default name merged.jsonl) with one line per metric
    # holding the metric name and its series; return the path.
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for metric, series in by_metric.items():
            f.write(json.dumps({"metric": metric, "series": series}) + "\n")
    return str(p)


def _seed_baseline(
    store, test_filename, *, metric_key, steps_key, constraint_key, step, values, suite="stage-c-8-gpu-h100"
):
    # [FUNCTION DOC] HELPER: Seed trusted baseline runs: for each value (in order, index i) write one
    # run for the test under cuda and the suite (default stage-c-8-gpu-h100) with the sample
    # provenance, created at "2026-06-0<i+1>T00:00:00+00:00" (so at most nine values), trusted, with
    # a single metric sample for the given metric key, steps key, constraint key and step.
    identity = RunIdentity(
        test_path=test_filename,
        backend="cuda",
        suite=suite,
    )
    for i, v in enumerate(values):
        store.write_run(
            identity,
            PROVENANCE,
            created_at=f"2026-06-0{i + 1}T00:00:00+00:00",
            trusted=True,
            values=[MetricSample(metric_key, steps_key, constraint_key, step, v)],
        )


# --- parse_merged_record ----------------------------------------------------


def test_parse_merged_record(tmp_path):
    # [FUNCTION DOC] Parsing a two-metric record returns a mapping from metric name to its series.
    path = _write_record(tmp_path, {"train/grad_norm": [[0, 0.5], [1, 1.0]], "rollout/raw_reward": [[0, 0.3]]})
    got = parse_merged_record(path)
    assert got == {"train/grad_norm": [[0, 0.5], [1, 1.0]], "rollout/raw_reward": [[0, 0.3]]}


def test_parse_merged_record_decodes_non_finite_markers(tmp_path):
    # [FUNCTION DOC] The string markers "NaN", "Infinity" and "-Infinity" in a series are decoded to
    # real not-a-number, positive infinity and negative infinity floats; finite points are kept.
    path = _write_record(tmp_path, {"train/grad_norm": [[0, 0.5], [1, "NaN"], [2, "Infinity"], [3, "-Infinity"]]})
    series = parse_merged_record(path)["train/grad_norm"]
    assert series[0] == [0, 0.5]
    assert math.isnan(series[1][1])
    assert series[2][1] == math.inf
    assert series[3][1] == -math.inf


def test_non_finite_at_gated_coordinate_errors_and_untrusts(tmp_path, store):
    # [FUNCTION DOC] A last-step gate on grad norm where the last point is a NaN marker yields an
    # error status whose reason mentions non-finite (no silent fallback to the previous finite
    # point), and the run is not trusted.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    # Capture-side marker for a NaN at the last (gated) step: ERROR, not a
    # silent fallback to the previous finite point.
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 0.9], [1, "NaN"]]})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")

    m = result.metrics[0]
    assert m.historical_status == GateStatus.ERROR
    assert "non-finite" in m.reason
    assert result.trusted is False


# --- cold start (no trusted history) ----------------------------------------


def test_cold_start_vacuously_trusted(tmp_path, store):
    # [FUNCTION DOC] With no stored baselines, one result is produced with inactive status (not an
    # error or failure), reason "historical: cold start (0 trusted baselines)", zero baseline count,
    # the "last" steps key and step -1; the run is trusted because it has zero active checks.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 0.9]]})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")

    assert len(result.metrics) == 1
    m = result.metrics[0]
    # No baselines yet: historical gate inactive, NOT an error, NOT a failure.
    assert m.historical_status == GateStatus.INACTIVE
    assert "historical: cold start (0 trusted baselines)" in m.reason
    assert m.baseline_n == 0
    assert m.steps_key == LAST_KEY
    assert m.step == -1
    # Zero active checks: vacuously trusted -- this run seeds the baseline.
    assert result.trusted is True


# --- historical gate --------------------------------------------------------


def test_historical_failure(tmp_path, store):
    # [FUNCTION DOC] Seeds baselines 0.80, 0.82, 0.78 for the last step with a 20 percent band and
    # evaluates a current 0.55. Asserts baseline count 3, baseline mean equal to the average, fail
    # status (deviation 0.25 exceeds the 0.16 band) and an untrusted run.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    # Seed a trusted baseline around 0.80 under the `last` coordinate.
    _seed_baseline(
        store,
        test_file,
        metric_key="rollout/raw_reward",
        steps_key=LAST_KEY,
        constraint_key=_key({"rel_up": 0.20, "rel_down": 0.20}),
        step=-1,
        values=[0.80, 0.82, 0.78],
    )
    # Current 0.55 vs mean 0.80, band = 0.16 -> |0.55-0.80|=0.25 fails historical.
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.55]]})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")
    m = result.metrics[0]
    assert m.baseline_n == 3
    assert m.baseline_mean == pytest.approx((0.80 + 0.82 + 0.78) / 3)
    assert m.historical_status == GateStatus.FAIL
    assert result.trusted is False


def test_historical_pass_within_tolerance(tmp_path, store):
    # [FUNCTION DOC] Same baselines as the failure test with a current of 0.79: pass status and a
    # trusted run.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="rollout/raw_reward",
        steps_key=LAST_KEY,
        constraint_key=_key({"rel_up": 0.20, "rel_down": 0.20}),
        step=-1,
        values=[0.80, 0.82, 0.78],
    )
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.79]]})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")
    m = result.metrics[0]
    assert m.historical_status == GateStatus.PASS
    assert result.trusted is True


def test_drift_beyond_historical_band_not_trusted(tmp_path, store):
    # [FUNCTION DOC] Baselines of 1.0 three times with a 50 percent band and a current of 1.8
    # (deviation 0.8) give fail status, baseline mean 1.0 and an untrusted run.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="last", constraint={"rel_up": 0.50, "rel_down": 0.50})
        """,
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="train/grad_norm",
        steps_key=LAST_KEY,
        constraint_key=_key({"rel_up": 0.50, "rel_down": 0.50}),
        step=-1,
        values=[1.0, 1.0, 1.0],
    )
    # current 1.8: historical |1.8-1.0|=0.8 > 0.5 fail.
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 1.8]]})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")
    m = result.metrics[0]
    assert m.historical_status == GateStatus.FAIL
    assert m.baseline_mean == pytest.approx(1.0)
    assert result.trusted is False


# --- per-step fan-out ---------------------------------------------------------


def test_all_fans_out_one_result_per_step(tmp_path, store):
    # [FUNCTION DOC] An all-steps gate over a two-point series yields two results with step, the
    # recorded step and current value (0, 0.9) and (1, 1.1) in order; with no baselines the run is
    # trusted.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="all", constraint={"rel_up": 0.50, "rel_down": 0.50})
        """,
    )
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 0.9], [1, 1.1]]})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")
    assert len(result.metrics) == 2
    assert [(m.step, m.at_step, m.current) for m in result.metrics] == [
        (0, 0, 0.9),
        (1, 1, 1.1),
    ]
    assert result.trusted is True


def test_all_reads_per_step_baselines(tmp_path, store):
    # [FUNCTION DOC] Seeds step 0 baselines of 0.1 and step 1 baselines of 0.9 (all-steps key, 90
    # percent band) and evaluates 0.1 at step 0 and 0.9 at step 1. Asserts each step's baseline mean
    # is its own (no cross-contamination), both pass, and the run is trusted.
    # Step 0's history and step 1's history must never cross-contaminate.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/ppo_kl",
                         steps="all", constraint={"rel_up": 0.90, "rel_down": 0.90})
        """,
    )
    kl_rule = _key({"rel_up": 0.90, "rel_down": 0.90})
    _seed_baseline(
        store,
        test_file,
        metric_key="train/ppo_kl",
        steps_key=_key("all"),
        constraint_key=kl_rule,
        step=0,
        values=[0.1, 0.1],
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="train/ppo_kl",
        steps_key=_key("all"),
        constraint_key=kl_rule,
        step=1,
        values=[0.9, 0.9],
    )
    record = _write_record(tmp_path, {"train/ppo_kl": [[0, 0.1], [1, 0.9]]})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")
    by_step = {m.step: m for m in result.metrics}
    assert by_step[0].baseline_mean == pytest.approx(0.1)
    assert by_step[1].baseline_mean == pytest.approx(0.9)
    assert by_step[0].historical_status == GateStatus.PASS
    assert by_step[1].historical_status == GateStatus.PASS
    assert result.trusted is True


def test_all_one_bad_step_untrusts_run(tmp_path, store):
    # [FUNCTION DOC] Seeds baselines 1.0 at steps 0 and 1 with a 20 percent band; current 1.1 at
    # step 0 passes and 1.5 at step 1 fails, so the run is untrusted.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="all", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="train/grad_norm",
        steps_key=_key("all"),
        constraint_key=_key({"rel_up": 0.20, "rel_down": 0.20}),
        step=0,
        values=[1.0, 1.0],
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="train/grad_norm",
        steps_key=_key("all"),
        constraint_key=_key({"rel_up": 0.20, "rel_down": 0.20}),
        step=1,
        values=[1.0, 1.0],
    )
    # Step 0 within band 0.2; step 1 drifts past it.
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 1.1], [1, 1.5]]})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")
    by_step = {m.step: m for m in result.metrics}
    assert by_step[0].historical_status == GateStatus.PASS
    assert by_step[1].historical_status == GateStatus.FAIL
    assert result.trusted is False


def test_all_and_explicit_steps_have_separate_coordinates(tmp_path, store):
    # [FUNCTION DOC] Two gates on the same metric, one with all steps and one with the explicit step
    # list [0], same constraint. Only the all-steps coordinate is seeded (two runs at step 0). Both
    # results judge step 0, but the first has baseline count 2 and the second count 0 with inactive
    # status, showing each declaration owns its series.
    # The coordinate is the declaration's literal content: a steps="all" gate and
    # a steps=[0] gate both judge step 0's value, but each owns its own series.
    gate_lines = """
        register_ci_gate(metric_key="train/ppo_kl",
                         steps="all", constraint={"rel_up": 0.50, "rel_down": 0.50})
        register_ci_gate(metric_key="train/ppo_kl",
                         steps=[0], constraint={"rel_up": 0.50, "rel_down": 0.50})
    """
    test_file = _write_test_file(tmp_path, gate_lines)
    _seed_baseline(
        store,
        test_file,
        metric_key="train/ppo_kl",
        steps_key=_key("all"),
        constraint_key=_key({"rel_up": 0.50, "rel_down": 0.50}),
        step=0,
        values=[0.1, 0.1],
    )
    record = _write_record(tmp_path, {"train/ppo_kl": [[0, 0.1]]})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")
    assert len(result.metrics) == 2  # one per spec, both at step 0
    all_m, steps_m = result.metrics
    assert all_m.baseline_n == 2
    assert steps_m.baseline_n == 0  # its own coordinate: cold start
    assert steps_m.historical_status == GateStatus.INACTIVE


def test_rule_is_part_of_coordinate(tmp_path, store):
    # [FUNCTION DOC] Two last-step gates with 50 percent and 1 percent bands; only the loose rule is
    # seeded (three baselines of 1.0). With current 1.2 the two results have different constraint
    # keys; the loose one has count 3 and passes; the tight one has count 0 and is inactive (cold
    # start rather than a shared read).
    # Two gates, same steps, different constraints: different constraint_key,
    # so each judges against its own baseline series.
    gate_lines = """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.50, "rel_down": 0.50})
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.01, "rel_down": 0.01})
    """
    test_file = _write_test_file(tmp_path, gate_lines)
    _seed_baseline(
        store,
        test_file,
        metric_key="rollout/raw_reward",
        steps_key=LAST_KEY,
        constraint_key=_key({"rel_up": 0.50, "rel_down": 0.50}),
        step=-1,
        values=[1.0, 1.0, 1.0],
    )
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 1.2]]})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")
    loose, tight = result.metrics
    assert loose.constraint_key != tight.constraint_key
    assert loose.baseline_n == 3
    assert loose.historical_status == GateStatus.PASS  # band 0.5
    # The tight rule's own series is unseeded: cold start, not a shared read.
    assert tight.baseline_n == 0
    assert tight.historical_status == GateStatus.INACTIVE


# --- near-zero abs constraint -------------------------------------------------


def test_near_zero_not_flagged_on_relative_pct(tmp_path, store):
    # [FUNCTION DOC] A gate on step 0 of ppo kl with 20 percent bands and absolute floors of 1e-6 in
    # both directions, baselines around 1e-9, and a record where step 0 is 1e-7 (and step 1 is 5e-3,
    # which is ignored). The selected current is 1e-7; the absolute deviation (~1e-7) is within the
    # floor, so the status is pass despite the huge relative change, and the run is trusted.
    # ppo_kl rides at ~1e-9. With a positive abs_floor, a tiny absolute
    # deviation must NOT trip even though the *relative* change is huge.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/ppo_kl",
                         steps=[0],
                         constraint={"rel_up": 0.20, "abs_floor_up": 1e-6, "rel_down": 0.20, "abs_floor_down": 1e-6})
        """,
    )
    # Seed a near-zero baseline; current also near-zero but 100x in relative terms.
    _seed_baseline(
        store,
        test_file,
        metric_key="train/ppo_kl",
        steps_key=_key([0]),
        constraint_key=_key({"rel_up": 0.20, "abs_floor_up": 1e-6, "rel_down": 0.20, "abs_floor_down": 1e-6}),
        step=0,
        values=[1e-9, 2e-9, 1e-9],
    )
    record = _write_record(tmp_path, {"train/ppo_kl": [[0, 1e-7], [1, 5e-3]]})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")
    m = result.metrics[0]
    # steps:[0] picks the step-0 value 1e-7.
    assert m.current == pytest.approx(1e-7)
    # historical mean ~1.33e-9; |1e-7 - 1.33e-9| ~ 9.9e-8 <= abs_floor 1e-6.
    assert m.historical_status == GateStatus.PASS
    assert result.trusted is True


def test_near_zero_real_jump_is_flagged(tmp_path, store):
    # [FUNCTION DOC] Same gate and baselines, with current 0.5 at step 0 (far beyond the floor): fail
    # status and an untrusted run.
    # Sanity counterpart: a ppo_kl that jumps well past abs_floor IS flagged.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/ppo_kl",
                         steps=[0],
                         constraint={"rel_up": 0.20, "abs_floor_up": 1e-6, "rel_down": 0.20, "abs_floor_down": 1e-6})
        """,
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="train/ppo_kl",
        steps_key=_key([0]),
        constraint_key=_key({"rel_up": 0.20, "abs_floor_up": 1e-6, "rel_down": 0.20, "abs_floor_down": 1e-6}),
        step=0,
        values=[1e-9, 2e-9, 1e-9],
    )
    record = _write_record(tmp_path, {"train/ppo_kl": [[0, 0.5]]})
    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")
    assert result.metrics[0].historical_status == GateStatus.FAIL
    assert result.trusted is False


# --- missing / empty / ill-formed required series -----------------------------


def test_missing_required_series_verdict_not_crash(tmp_path, store):
    # [FUNCTION DOC] The record lacks the gated metric: the result has error status, no current value,
    # a reason containing "missing", and the run is untrusted (no exception).
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    # Record carries a different metric only.
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 1.0]]})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")
    m = result.metrics[0]
    assert m.historical_status == GateStatus.ERROR
    assert m.current is None
    assert "missing" in m.reason
    assert result.trusted is False


def test_empty_required_series_verdict(tmp_path, store):
    # [FUNCTION DOC] The gated metric has an empty series: error status and an untrusted run.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    record = _write_record(tmp_path, {"rollout/raw_reward": []})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")
    m = result.metrics[0]
    assert m.historical_status == GateStatus.ERROR
    assert result.trusted is False


def test_all_null_step_is_error_verdict(tmp_path, store):
    # [FUNCTION DOC] An all-steps gate over a series containing a point with a null step yields a
    # single result for the spec with error status (no crash, no silent skip) and an untrusted run.
    # A steps="all" gate over a series with a step-less point yields one ERROR
    # verdict for the spec, not a crash and not a silent skip.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="all", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 1.0], [None, 1.1]]})

    result = evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")
    assert len(result.metrics) == 1
    assert result.metrics[0].historical_status == GateStatus.ERROR
    assert result.trusted is False


# --- asymmetric corridor ------------------------------------------------------


def test_asymmetric_corridor_tight_up_loose_down(tmp_path, store):
    # [FUNCTION DOC] Baselines of 2.0 with an up band of 10 percent and a down band of 80 percent
    # (corridor 0.4 to 2.2). Current 0.5 passes (within the loose lower band), 3.0 fails (above the
    # tight upper band), and 0.1 fails (a suspiciously good collapse below the lower band).
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="last",
                         constraint={"rel_up": 0.10, "rel_down": 0.80})
        """,
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="train/grad_norm",
        steps_key=LAST_KEY,
        constraint_key=_key({"rel_up": 0.10, "rel_down": 0.80}),
        step=-1,
        values=[2.0, 2.0],
    )
    # Corridor [2.0 - 1.6, 2.0 + 0.2] = [0.4, 2.2]. A drop within the loose
    # lower band passes.
    low = _write_record(tmp_path, {"train/grad_norm": [[0, 0.5]]}, name="low.jsonl")
    assert (
        evaluate_gate(test_file, low, store, executing_suite="stage-c-8-gpu-h100").metrics[0].historical_status
        == GateStatus.PASS
    )

    # A rise beyond the tight upper band fails.
    high = _write_record(tmp_path, {"train/grad_norm": [[0, 3.0]]}, name="high.jsonl")
    assert (
        evaluate_gate(test_file, high, store, executing_suite="stage-c-8-gpu-h100").metrics[0].historical_status
        == GateStatus.FAIL
    )

    # A collapse past the lower band fails too: a "too good" value is suspect
    # and must not enter the baseline.
    collapse = _write_record(tmp_path, {"train/grad_norm": [[0, 0.1]]}, name="collapse.jsonl")
    assert (
        evaluate_gate(test_file, collapse, store, executing_suite="stage-c-8-gpu-h100").metrics[0].historical_status
        == GateStatus.FAIL
    )


def test_reward_corridor_flags_drop_and_suspicious_jump(tmp_path, store):
    # [FUNCTION DOC] Baselines of 0.60 with up band 50 percent and down band 10 percent (corridor
    # 0.54 to 0.90). Current 0.85 passes, 0.50 (a drop) fails, 0.95 (a suspicious jump) fails.
    # raw_reward regressing means DROPPING (tight lower band); a jump far
    # above baseline smells of a broken env / reward hack (loose upper band).
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last",
                         constraint={"rel_up": 0.50, "rel_down": 0.10})
        """,
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="rollout/raw_reward",
        steps_key=LAST_KEY,
        constraint_key=_key({"rel_up": 0.50, "rel_down": 0.10}),
        step=-1,
        values=[0.60, 0.60],
    )
    # Corridor [0.54, 0.90]: a genuine improvement passes.
    ok = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.85]]}, name="ok.jsonl")
    assert (
        evaluate_gate(test_file, ok, store, executing_suite="stage-c-8-gpu-h100").metrics[0].historical_status
        == GateStatus.PASS
    )

    low = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.50]]}, name="low.jsonl")
    assert (
        evaluate_gate(test_file, low, store, executing_suite="stage-c-8-gpu-h100").metrics[0].historical_status
        == GateStatus.FAIL
    )

    jump = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.95]]}, name="jump.jsonl")
    assert (
        evaluate_gate(test_file, jump, store, executing_suite="stage-c-8-gpu-h100").metrics[0].historical_status
        == GateStatus.FAIL
    )


# --- multiple specs, no specs ------------------------------------------------


def test_no_gate_specs_is_vacuously_trusted(tmp_path, store):
    # [FUNCTION DOC] A test file with a CUDA registration but no gate declarations evaluates to an
    # empty metric list and a trusted result.
    body = textwrap.dedent(
        """
        from tests.ci.ci_register import register_cuda_ci
        register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100", labels=["megatron"], hardware=["hopper"])
        """
    ).lstrip("\n")
    p = tmp_path / "test_nogate.py"
    p.write_text(body)
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.3]]})

    result = evaluate_gate(str(p), record, store, executing_suite="stage-c-8-gpu-h100")
    assert result.metrics == []
    assert result.trusted is True


def test_gate_writes_no_rows(tmp_path, store):
    # [FUNCTION DOC] After evaluating an all-steps gate, the store's runs table is still empty: the
    # gate never persists.
    # The gate must never persist: after evaluation the store has no runs.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="all", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.31]]})
    evaluate_gate(test_file, record, store, executing_suite="stage-c-8-gpu-h100")

    n = store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n == 0


# --- dual-register files + harness-supplied registry ------------------------


def _write_dual_register_file(tmp_path: Path, gate_lines: str, *, name: str = "test_dual_fixture.py") -> str:
    # [FUNCTION DOC] HELPER: Write a fixture file (default test_dual_fixture.py) with both a CUDA
    # registration (est_time 360, stage-c-8-gpu-h100, label short, hopper) and a ROCm registration
    # (est_time 360, stage-c-8-gpu-mi350, label short), followed by the dedented gate lines; return
    # its path.
    """A real-shaped e2e file: BOTH register_cuda_ci and register_rocm_ci.

    Mirrors tests/e2e/short/test_qwen2.5_0.5B_gsm8k_short.py, which trips the
    single-registry reparse (two register_*_ci calls). The harness passes the
    chosen registry in.
    """
    body = (
        "from tests.ci.ci_register import register_cuda_ci, register_rocm_ci\n"
        "from tests.ci.metric_history import register_ci_gate\n"
        'register_cuda_ci(est_time=360, suite="stage-c-8-gpu-h100", labels=["short"], hardware=["hopper"])\n'
        'register_rocm_ci(est_time=360, suite="stage-c-8-gpu-mi350", labels=["short"])\n'
        + textwrap.dedent(gate_lines).strip()
        + "\n"
    )
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def test_dual_register_with_gate_uses_supplied_registry(tmp_path, store):
    # [FUNCTION DOC] For a dual-registered file with a gate, evaluates with an explicitly supplied
    # CUDA registry. Asserts no ambiguity error, one result with inactive status, and that the
    # result's backend is cuda, its suite the supplied stage, and its test path the file.
    # A file with BOTH register_cuda_ci and register_rocm_ci would make the
    # single-registry reparse raise (ambiguous). With the harness passing the
    # CUDA registry explicitly, the gate uses that identity and does not raise.
    test_file = _write_dual_register_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.31]]})
    cuda_registry = CIRegistry(
        backend=HWBackend.CUDA,
        filename=test_file,
        est_time=360,
        suite="stage-c-8-gpu-h100",
        labels=["short"],
    )

    result = evaluate_gate(test_file, record, store, registry=cuda_registry, executing_suite="stage-c-8-gpu-h100")

    assert len(result.metrics) == 1
    assert result.metrics[0].historical_status == GateStatus.INACTIVE
    # Identity is the supplied CUDA registry, not the ROCm one.
    assert result.backend == "cuda"
    assert result.suite == "stage-c-8-gpu-h100"
    assert result.test_path == test_file


def test_dual_register_no_spec_registry_none_vacuously_trusted(tmp_path, store):
    # [FUNCTION DOC] A dual-registered file without any gate, evaluated with no registry, gives an
    # empty metric list, a trusted result and the file path, without raising on the ambiguous
    # registrations.
    # A dual-registered file with NO register_ci_gate spec, evaluated with
    # registry=None, must be vacuously trusted -- it must NOT raise on the
    # ambiguous (two register_*_ci) file because no gate identity is needed.
    body = textwrap.dedent(
        """
        from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
        register_cuda_ci(est_time=360, suite="stage-c-8-gpu-h100", labels=["short"], hardware=["hopper"])
        register_rocm_ci(est_time=360, suite="stage-c-8-gpu-mi350", labels=["short"])
        """
    ).lstrip("\n")
    p = tmp_path / "test_dual_nogate.py"
    p.write_text(body)
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.3]]})

    result = evaluate_gate(str(p), record, store, registry=None, executing_suite="stage-c-8-gpu-h100")

    assert result.metrics == []
    assert result.trusted is True
    assert result.test_path == str(p)


def test_single_register_gate_registry_none_still_reparses(tmp_path, store):
    # [FUNCTION DOC] A single-registration file with a gate evaluated without a registry reparses its
    # identity: one inactive result, backend cuda, suite stage-c-8-gpu-h100, trusted.
    # The isolated unit-test convenience path: a single-register file with a
    # gate spec and registry=None still reparses identity via _registry_for.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.31]]})

    result = evaluate_gate(test_file, record, store, registry=None, executing_suite="stage-c-8-gpu-h100")

    assert len(result.metrics) == 1
    assert result.metrics[0].historical_status == GateStatus.INACTIVE
    assert result.backend == "cuda"
    assert result.suite == "stage-c-8-gpu-h100"
    assert result.trusted is True
