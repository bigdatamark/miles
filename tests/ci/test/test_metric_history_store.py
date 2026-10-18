# [MODULE DOC]
# Offline tests of the metric-history store contract, run against the in-memory SQLite backend.
# Covers: writing a run and reading back trusted values newest-first, rejection of non-finite
# values before any insert, the limit cap, isolation between runs of different identity
# (test path, backend, suite) and between different coordinate columns (metric key, steps key,
# constraint key, step), marking runs untrusted by run id / github run id / commit sha
# (idempotent, exactly one selector required), exclusion of untrusted runs from baselines, a
# guard that the SQLite baseline query keeps the authoritative shape, and that the Neon store
# refuses to construct without a connection string. The module registers itself as a CPU test
# in the stage-a-cpu suite. Module constants: one canonical run identity, one provenance record,
# and canonical-JSON strings standing in for steps keys and constraint keys.
"""Offline unit tests for the metric-history store contract.

These run against :class:`SQLiteMetricHistoryStore` with an in-memory database:
no network, no driver install, no live Postgres. The SQLite backend mirrors the
authoritative baseline query, so passing here means the contract the gate
depends on holds.
"""

from __future__ import annotations

import re

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.ci.metric_history import (
    MetricSample,
    NeonMetricHistoryStore,
    RunIdentity,
    RunProvenance,
    SQLiteMetricHistoryStore,
)

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

IDENTITY = RunIdentity(
    test_path="tests/e2e/test_grpo.py",
    backend="megatron",
    suite="stage-c-8-gpu-h100",
)

PROVENANCE = RunProvenance(
    commit_sha="deadbeef",
    pr_number=42,
    github_run_id=1001,
    github_run_attempt=1,
    event_name="pull_request",
    ref="refs/pull/42/merge",
)

# Canonical-JSON declaration keys, as the parser would derive them from a
# test file's literal steps / constraint values.
LAST_STEPS_KEY = '"last"'
STEP_LIST_KEY = "[0,1]"
REL_CONSTRAINT_KEY = '{"rel_down":0.2,"rel_up":0.2}'
ABS_CONSTRAINT_KEY = '{"abs_floor_down":0.02,"abs_floor_up":0.02}'


def _sample(metric_key, value, *, steps_key=LAST_STEPS_KEY, constraint_key=REL_CONSTRAINT_KEY, step=-1):
    # [FUNCTION DOC] HELPER: Build a metric sample from a metric key and a value, with the
    # steps key, constraint key and step defaulting to the "last" selection, the relative
    # constraint and the reduction sentinel step.
    return MetricSample(metric_key, steps_key, constraint_key, step, value)


def _recent(
    store,
    metric_key,
    *,
    steps_key=LAST_STEPS_KEY,
    constraint_key=REL_CONSTRAINT_KEY,
    step=-1,
    identity=IDENTITY,
    limit=10,
):
    # [FUNCTION DOC] HELPER: Query the store for the most recent trusted values of one metric
    # coordinate under the given run identity (default canonical identity), newest run first,
    # at most the given limit (default 10). Returns the list of values from the store.
    return store.recent_trusted_values(
        identity.test_path,
        identity.backend,
        identity.suite,
        metric_key,
        steps_key,
        constraint_key,
        step,
        limit,
    )


@pytest.fixture
def store():
    # [FUNCTION DOC] HELPER: Test fixture that yields a fresh in-memory SQLite metric-history
    # store and closes it after the test finishes.
    s = SQLiteMetricHistoryStore(":memory:")
    yield s
    s.close()


def _write(
    store,
    *,
    identity=IDENTITY,
    provenance=PROVENANCE,
    created_at,
    trusted=True,
    values,
):
    # [FUNCTION DOC] HELPER: Write one run to the store. The creation timestamp and the list of
    # metric samples are required (keyword only); identity and provenance default to the
    # module constants and the run is trusted by default. Returns the new run id.
    return store.write_run(identity, provenance, created_at, trusted, values)


def test_write_then_recent_returns_written_values(store):
    # [FUNCTION DOC] Writes two trusted runs on consecutive days, each with one reward sample, and
    # asserts the baseline query returns both values with the newest run first.
    _write(
        store,
        created_at="2026-06-01T00:00:00+00:00",
        values=[_sample("reward_mean", 0.81)],
    )
    _write(
        store,
        created_at="2026-06-02T00:00:00+00:00",
        values=[_sample("reward_mean", 0.83)],
    )

    # Newest run first.
    assert _recent(store, "reward_mean") == [0.83, 0.81]


def test_write_run_rejects_non_finite_before_persisting(store):
    # [FUNCTION DOC] For each of nan, positive infinity and negative infinity, writes a run holding
    # one finite sample plus one non-finite sample and asserts a value error mentioning
    # "non-finite". Then asserts the finite sibling was never persisted (baseline is empty),
    # proving validation happens before any insert.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            _write(
                store,
                created_at="2026-06-01T00:00:00+00:00",
                values=[_sample("reward_mean", 0.5), _sample("reward_mean", bad, step=0)],
            )

    # Validation runs before any insert: the finite sibling sample must not
    # have leaked in as a trusted row.
    assert _recent(store, "reward_mean") == []


def test_limit_caps_and_orders_newest_first(store):
    # [FUNCTION DOC] Writes three runs with increasing timestamps and values 0.0, 1.0, 2.0, then
    # queries with a limit of two and asserts only the two newest values come back, newest first.
    for i, created in enumerate(
        ["2026-06-01T00:00:00+00:00", "2026-06-02T00:00:00+00:00", "2026-06-03T00:00:00+00:00"]
    ):
        _write(store, created_at=created, values=[_sample("reward_mean", float(i))])

    assert _recent(store, "reward_mean", limit=2) == [2.0, 1.0]


def test_identity_isolation(store):
    # [FUNCTION DOC] Writes a reference run under the canonical identity, then three newer runs
    # with a large value each differing from the canonical identity in exactly one field (backend,
    # suite, test path). Asserts the canonical baseline contains only its own single value.
    # Reference run under the canonical identity.
    _write(store, created_at="2026-06-01T00:00:00+00:00", values=[_sample("reward_mean", 0.5)])

    # Same metric, but each of these differs in exactly one identity field.
    other_backend = RunIdentity(IDENTITY.test_path, "fsdp", IDENTITY.suite)
    other_suite = RunIdentity(IDENTITY.test_path, IDENTITY.backend, "stage-b-2-gpu-h200")
    other_path = RunIdentity("tests/e2e/test_other.py", IDENTITY.backend, IDENTITY.suite)
    for ident in (other_backend, other_suite, other_path):
        _write(
            store,
            identity=ident,
            created_at="2026-06-02T00:00:00+00:00",
            values=[_sample("reward_mean", 9.9)],
        )

    # Baseline for the canonical identity sees only its own single value.
    assert _recent(store, "reward_mean") == [0.5]


def test_coordinate_isolation(store):
    # [FUNCTION DOC] Writes one run holding four samples of one metric key: a base coordinate and
    # three variants differing in exactly one of steps key, constraint key or step. Asserts each
    # exact coordinate query returns only its own value (plain equality on every column).
    # Four values under one metric_key; each non-base row differs from the
    # base coordinate in exactly one coordinate column.
    _write(
        store,
        created_at="2026-06-01T00:00:00+00:00",
        values=[
            _sample("pass_rate", 0.70),
            _sample("pass_rate", 0.60, steps_key=STEP_LIST_KEY, step=0),
            _sample("pass_rate", 0.80, constraint_key=ABS_CONSTRAINT_KEY),
            _sample("pass_rate", 0.90, step=0),
        ],
    )

    # Each exact coordinate matches only its own row: plain equality on every
    # column, no cross-matching.
    assert _recent(store, "pass_rate") == [0.70]
    assert _recent(store, "pass_rate", steps_key=STEP_LIST_KEY, step=0) == [0.60]
    assert _recent(store, "pass_rate", constraint_key=ABS_CONSTRAINT_KEY) == [0.80]
    assert _recent(store, "pass_rate", step=0) == [0.90]


def test_mark_untrusted_by_run_id_excludes_immediately(store):
    # [FUNCTION DOC] Writes two runs, marks the newer one untrusted by run id, and asserts exactly
    # one run was affected, the very next baseline query excludes it (no rebaseline step needed),
    # the older run remains, and the two run ids differ.
    keep = _write(store, created_at="2026-06-01T00:00:00+00:00", values=[_sample("reward_mean", 0.5)])
    drop = _write(store, created_at="2026-06-02T00:00:00+00:00", values=[_sample("reward_mean", 0.9)])

    affected = store.mark_untrusted(run_id=drop)
    assert affected == 1

    # No rebaseline step: the dropped run is gone from the baseline on the very
    # next query, the kept run remains.
    assert _recent(store, "reward_mean") == [0.5]
    assert keep != drop


def test_mark_untrusted_by_github_run_id(store):
    # [FUNCTION DOC] Writes two runs sharing one provenance (github run id 7777, no pull request
    # number), marks untrusted by github run id, and asserts both runs were affected and the
    # baseline is empty.
    prov = RunProvenance("c1", None, 7777, 1, "push", "refs/heads/main")
    _write(store, provenance=prov, created_at="2026-06-01T00:00:00+00:00", values=[_sample("m", 1.0)])
    _write(store, provenance=prov, created_at="2026-06-02T00:00:00+00:00", values=[_sample("m", 2.0)])

    affected = store.mark_untrusted(github_run_id=7777)
    assert affected == 2
    assert _recent(store, "m") == []


def test_mark_untrusted_by_commit_sha(store):
    # [FUNCTION DOC] Writes one run with a distinctive commit sha and asserts marking untrusted by
    # that commit sha affects exactly one run.
    prov = RunProvenance("badc0de", 5, 1, 1, "pull_request", "refs/pull/5/merge")
    _write(store, provenance=prov, created_at="2026-06-01T00:00:00+00:00", values=[_sample("m", 1.0)])
    affected = store.mark_untrusted(commit_sha="badc0de")
    assert affected == 1


def test_mark_untrusted_is_idempotent(store):
    # [FUNCTION DOC] Marks one run untrusted twice by run id; asserts the first call affects one
    # run and the second affects zero since it is already untrusted.
    rid = _write(store, created_at="2026-06-01T00:00:00+00:00", values=[_sample("m", 1.0)])
    assert store.mark_untrusted(run_id=rid) == 1
    # Already untrusted -> no rows change.
    assert store.mark_untrusted(run_id=rid) == 0


def test_mark_untrusted_requires_exactly_one_key(store):
    # [FUNCTION DOC] Asserts a value error when marking untrusted with no selector at all and when
    # giving two selectors (run id and commit sha) at once.
    with pytest.raises(ValueError):
        store.mark_untrusted()
    with pytest.raises(ValueError):
        store.mark_untrusted(run_id="x", commit_sha="y")


def test_untrusted_run_never_in_baseline(store):
    # [FUNCTION DOC] Writes a run flagged untrusted at write time and asserts it never appears in
    # the baseline query results.
    _write(store, created_at="2026-06-01T00:00:00+00:00", trusted=False, values=[_sample("m", 5.0)])
    assert _recent(store, "m") == []


def test_baseline_sql_matches_authoritative_shape():
    # [FUNCTION DOC] Imports the private baseline query text of the SQLite store, collapses and
    # lowercases its whitespace, and asserts it contains each required fragment: the join of
    # metric values to runs, equality predicates on test path, backend, suite, metric key, steps
    # key, constraint key and step, the trusted filter, newest-first ordering and a limit.
    # Guard against drift from the authoritative query: identity predicates,
    # plain equality on every coordinate column, trusted filter,
    # created_at DESC, LIMIT.
    from tests.ci.metric_history.storage.sqlite_store import _BASELINE_SQL

    sql = re.sub(r"\s+", " ", _BASELINE_SQL).strip().lower()
    for fragment in (
        "from metric_values mv",
        "join runs r using (run_id)",
        "r.test_path = ?",
        "r.backend = ?",
        "r.suite = ?",
        "mv.metric_key = ?",
        "mv.steps_key = ?",
        "mv.constraint_key = ?",
        "mv.step = ?",
        "r.trusted = 1",
        "order by r.created_at desc",
        "limit ?",
    ):
        assert fragment in sql, f"baseline query missing: {fragment!r}"


def test_neon_store_requires_dsn(monkeypatch):
    # [FUNCTION DOC] Removes the Neon connection-string environment variable and asserts that
    # constructing the Neon store without an explicit one raises a runtime error naming that
    # variable, without attempting any connection.
    # No DSN passed and the env var unset: construction fails with a clear error
    # (not NotImplementedError -- the backend is implemented now). It must not
    # attempt a connection, so the failure is purely about the missing DSN.
    from tests.ci.metric_history import NEON_DATABASE_URL_ENV

    monkeypatch.delenv(NEON_DATABASE_URL_ENV, raising=False)
    with pytest.raises(RuntimeError, match=NEON_DATABASE_URL_ENV):
        NeonMetricHistoryStore()
