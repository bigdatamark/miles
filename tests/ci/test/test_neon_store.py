# [MODULE DOC]
# Tests of the Neon (Postgres) metric-history store. Two layers: (1) offline tests that replace
# the psycopg driver module (inserted into the module table by a fixture) with a fake
# connection/cursor recording executed statements, parameters, commits, rollbacks, connects and
# closes, asserting the store's SQL shape, parameter order, single-transaction writes, rollback
# on failure, validation before any statement, connection string resolution (explicit argument
# beats the environment variable; missing is a runtime error), no connection at construction and
# one fresh connection per operation that is always closed; (2) one live round-trip test, skipped
# unless the test Postgres connection-string environment variable is set, which provisions the
# two tables (runs and metric values plus indexes) as an admin and exercises write, read and
# mark-untrusted. Registers itself as a CPU test in the stage-a-cpu suite.
"""Tests for :class:`NeonMetricHistoryStore`.

Two layers:

* SQL/transaction unit tests that run WITHOUT a live database. They monkeypatch
  `psycopg.connect` with a fake connection/cursor that records the executed
  SQL and params plus commit/rollback calls, then assert the store issues the
  right statements within a single transaction.
* A live-Postgres smoke test, guarded by `@pytest.mark.skipif` on the
  `MILES_TEST_POSTGRES_DSN` env var. It provisions the two tables and round-trips
  write_run -> recent_trusted_values -> mark_untrusted against a real server. It
  skips cleanly offline (no network, no import of an absent driver, no failure).
"""

from __future__ import annotations

import os
import re
import sys
import types

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.ci.metric_history import (
    NEON_DATABASE_URL_ENV,
    MetricSample,
    NeonMetricHistoryStore,
    RunIdentity,
    RunProvenance,
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

# Canonical-JSON declaration keys, as the parser derives them.
LAST_STEPS_KEY = '"last"'
REL_CONSTRAINT_KEY = '{"rel_down":0.2,"rel_up":0.2}'


def _sample(metric_key, value, *, steps_key=LAST_STEPS_KEY, constraint_key=REL_CONSTRAINT_KEY, step=-1):
    # [FUNCTION DOC] HELPER: Build a metric sample from a metric key and value, with steps key,
    # constraint key and step defaulting to the "last" selection, the relative constraint and the
    # reduction sentinel step.
    return MetricSample(metric_key, steps_key, constraint_key, step, value)


# --------------------------------------------------------------------------- #
# Fake psycopg: records executed SQL + params and commit/rollback ordering.
# --------------------------------------------------------------------------- #


def _norm(sql: str) -> str:
    # [FUNCTION DOC] HELPER: Normalize SQL text for comparison: collapse every whitespace run to a
    # single space, strip the ends and lowercase. Returns the normalized string.
    return re.sub(r"\s+", " ", sql).strip().lower()


class _FakeCursor:
    # [CLASS DOC] HELPER: Fake database cursor (context manager) bound to a fake connection. Every
    # execute / executemany is appended to the connection's event list with normalized SQL and
    # parameters; it can raise a simulated error and expose a pinned row count and fetched rows.
    def __init__(self, conn: _FakeConn):
        # [FUNCTION DOC] HELPER: Keep the owning fake connection; the row count starts at minus one.
        self._conn = conn
        self.rowcount = -1

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Context manager entry; returns the cursor itself.
        return self

    def __exit__(self, *exc):
        # [FUNCTION DOC] HELPER: Context manager exit; never suppresses exceptions.
        return False

    def execute(self, sql, params=None):
        # [FUNCTION DOC] HELPER: Record an "execute" event with the normalized SQL and the
        # parameters. If the connection is set to fail, raise a runtime error "boom: simulated db
        # error" after recording. Otherwise, if the connection has a pinned next row count, copy it
        # to this cursor's row count. Returns the cursor.
        self._conn.events.append(("execute", _norm(sql), params))
        if self._conn.fail_on_execute:
            raise RuntimeError("boom: simulated db error")
        # Let a test pin a rowcount for the next UPDATE/SELECT.
        if self._conn.next_rowcount is not None:
            self.rowcount = self._conn.next_rowcount
        return self

    def executemany(self, sql, seq):
        # [FUNCTION DOC] HELPER: Materialize the parameter sequence into a list, record an
        # "executemany" event with the normalized SQL and that list, raise the simulated runtime
        # error if the connection is set to fail, else return the cursor.
        seq = list(seq)
        self._conn.events.append(("executemany", _norm(sql), seq))
        if self._conn.fail_on_execute:
            raise RuntimeError("boom: simulated db error")
        return self

    def fetchall(self):
        # [FUNCTION DOC] HELPER: Return a copy of the rows configured on the connection.
        return list(self._conn.fetch_rows)


class _FakeConn:
    # [CLASS DOC] HELPER: Fake database connection. Records an ordered list of events (executes,
    # commits, rollbacks) and counters for commits, rollbacks, connects, closes, plus the list of
    # connection strings used. Test knobs: fail every execute, rows returned by fetch, and a
    # pinned row count for the next statement.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initialize empty event list, zero counters, no failure, no fetch
        # rows and no pinned row count.
        self.events: list = []
        self.commit_count = 0
        self.rollback_count = 0
        self.connect_count = 0
        self.connected_dsns: list = []
        self.close_count = 0
        self.fail_on_execute = False
        self.fetch_rows: list = []
        self.next_rowcount: int | None = None

    def cursor(self):
        # [FUNCTION DOC] HELPER: Return a new fake cursor bound to this connection.
        return _FakeCursor(self)

    def commit(self):
        # [FUNCTION DOC] HELPER: Increment the commit counter and record a "commit" event.
        self.commit_count += 1
        self.events.append(("commit", None, None))

    def rollback(self):
        # [FUNCTION DOC] HELPER: Increment the rollback counter and record a "rollback" event.
        self.rollback_count += 1
        self.events.append(("rollback", None, None))

    def close(self):
        # [FUNCTION DOC] HELPER: Increment the close counter.
        self.close_count += 1


@pytest.fixture
def fake_conn(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture. Creates one shared fake connection and a fake driver module
    # whose connect function counts the call, records the connection string and returns that
    # connection; the fake module replaces the real driver in the module table for the test
    # (restored by the monkeypatch). Returns the fake connection for assertions.
    """Install a fake `psycopg` module whose `connect` hands out one shared
    fake connection while recording every connect call and its DSN."""
    conn = _FakeConn()
    fake_psycopg = types.ModuleType("psycopg")

    def _connect(dsn):
        # [FUNCTION DOC] HELPER: Fake driver connect: count the call, record the connection string
        # and return the shared fake connection.
        conn.connect_count += 1
        conn.connected_dsns.append(dsn)
        return conn

    fake_psycopg.connect = _connect
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    return conn


@pytest.fixture
def store(fake_conn):
    # [FUNCTION DOC] HELPER: Fixture returning a Neon store constructed with a fake connection
    # string, on top of the fake driver (so no connection happens at construction).
    return NeonMetricHistoryStore(dsn="postgresql://fake/db")


# --------------------------------------------------------------------------- #
# Construction / DSN resolution.
# --------------------------------------------------------------------------- #


def test_requires_dsn_when_env_unset(monkeypatch):
    # [FUNCTION DOC] With the connection-string environment variable removed and no argument,
    # constructing the store raises a runtime error that names the variable.
    monkeypatch.delenv(NEON_DATABASE_URL_ENV, raising=False)
    with pytest.raises(RuntimeError, match=NEON_DATABASE_URL_ENV):
        NeonMetricHistoryStore()


def test_reads_dsn_from_env(monkeypatch, fake_conn):
    # [FUNCTION DOC] Sets the environment variable, builds the store with no argument, performs a
    # read, and asserts the fake driver was connected with the environment value.
    monkeypatch.setenv(NEON_DATABASE_URL_ENV, "postgresql://env/db")
    s = NeonMetricHistoryStore()
    s.recent_trusted_values("t", "b", "s", "m", LAST_STEPS_KEY, REL_CONSTRAINT_KEY, -1, limit=1)
    assert fake_conn.connected_dsns == ["postgresql://env/db"]


def test_explicit_dsn_takes_precedence(monkeypatch, fake_conn):
    # [FUNCTION DOC] With the environment variable removed, builds the store with an explicit
    # connection string, performs a read, and asserts the driver was connected with that string.
    monkeypatch.delenv(NEON_DATABASE_URL_ENV, raising=False)
    s = NeonMetricHistoryStore(dsn="postgresql://explicit/db")
    s.recent_trusted_values("t", "b", "s", "m", LAST_STEPS_KEY, REL_CONSTRAINT_KEY, -1, limit=1)
    assert fake_conn.connected_dsns == ["postgresql://explicit/db"]


def test_no_ddl_at_construction(store, fake_conn):
    # [FUNCTION DOC] After building the store (via the fixture) asserts zero connects and no
    # recorded events: schema is provisioned out of band and connections are per operation.
    # Construction must not connect nor issue any statement -- schema is
    # provisioned out-of-band and connections are per-operation.
    assert fake_conn.connect_count == 0
    assert fake_conn.events == []


# --------------------------------------------------------------------------- #
# write_run: one runs INSERT + N metric INSERTs, single commit, rollback on error.
# --------------------------------------------------------------------------- #


def test_write_run_single_transaction_one_commit(store, fake_conn):
    # [FUNCTION DOC] Writes one run with three samples (one reduced, two at steps 0 and 1) and
    # asserts: the returned run id is a 32-character hex string; the event sequence is exactly one
    # execute, one bulk execute, one commit (one commit, no rollback); the first statement inserts
    # into the runs table with the run id first, then test path, backend, suite as the next three
    # parameters, and the trusted flag last as a native boolean; the second statement inserts into
    # the metric values table with one tuple per sample of run id, metric key, steps key,
    # constraint key, step and value.
    values = [
        _sample("reward_mean", 0.83),
        _sample("pass_rate", 0.6, step=0),
        _sample("pass_rate", 0.8, step=1),
    ]
    run_id = store.write_run(IDENTITY, PROVENANCE, "2026-06-02T00:00:00+00:00", True, values)

    assert isinstance(run_id, str) and len(run_id) == 32  # uuid4().hex

    kinds = [e[0] for e in fake_conn.events]
    # Exactly: one execute (runs INSERT), one executemany (metric_values), one commit.
    assert kinds == ["execute", "executemany", "commit"]
    assert fake_conn.commit_count == 1
    assert fake_conn.rollback_count == 0

    run_kind, run_sql, run_params = fake_conn.events[0]
    assert "insert into runs" in run_sql
    # run_id is generated by the store and is the first bound param.
    assert run_params[0] == run_id
    assert run_params[1:4] == (
        IDENTITY.test_path,
        IDENTITY.backend,
        IDENTITY.suite,
    )
    # trusted is bound as a native bool, not 0/1.
    assert run_params[-1] is True

    mv_kind, mv_sql, mv_seq = fake_conn.events[1]
    assert "insert into metric_values" in mv_sql
    assert mv_seq == [
        (run_id, "reward_mean", LAST_STEPS_KEY, REL_CONSTRAINT_KEY, -1, 0.83),
        (run_id, "pass_rate", LAST_STEPS_KEY, REL_CONSTRAINT_KEY, 0, 0.6),
        (run_id, "pass_rate", LAST_STEPS_KEY, REL_CONSTRAINT_KEY, 1, 0.8),
    ]


def test_write_run_rolls_back_on_error(store, fake_conn):
    # [FUNCTION DOC] With every execute failing, a write raises the simulated runtime error;
    # asserts zero commits, one rollback, and a rollback event in the recorded events.
    fake_conn.fail_on_execute = True
    with pytest.raises(RuntimeError, match="boom"):
        store.write_run(IDENTITY, PROVENANCE, "2026-06-02T00:00:00+00:00", True, [_sample("m", 1.0)])
    assert fake_conn.commit_count == 0
    assert fake_conn.rollback_count == 1
    assert ("rollback", None, None) in fake_conn.events


def test_write_run_rejects_non_finite_before_any_statement(store, fake_conn):
    # [FUNCTION DOC] For nan and both infinities mixed with a finite sample, a write raises a value
    # error "non-finite"; asserts no statement was issued and no commit or rollback happened.
    # The DB is the write boundary where validity is enforced (store contract:
    # validate_finite_values); the raise must land before any SQL is issued.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            store.write_run(
                IDENTITY,
                PROVENANCE,
                "2026-06-02T00:00:00+00:00",
                True,
                [_sample("m", 0.5), _sample("m", bad, step=0)],
            )
    assert fake_conn.events == []
    assert fake_conn.commit_count == 0
    assert fake_conn.rollback_count == 0


# --------------------------------------------------------------------------- #
# recent_trusted_values: the parameterized authoritative baseline JOIN.
# --------------------------------------------------------------------------- #


def test_recent_trusted_values_issues_baseline_join(store, fake_conn):
    # [FUNCTION DOC] Configures two fetched rows and reads with a limit of 10. Asserts the values
    # come back flattened in fetch order; exactly one execute event; the SQL contains the join of
    # metric values to runs, percent-s placeholders for each identity and coordinate predicate,
    # the trusted-true filter, newest-first order and a limit placeholder; the parameters are in
    # exact placeholder order (test path, backend, suite, metric key, steps key, constraint key,
    # step, limit); and no commit or rollback occurs for a read.
    fake_conn.fetch_rows = [(0.83,), (0.81,)]
    got = store.recent_trusted_values(
        IDENTITY.test_path,
        IDENTITY.backend,
        IDENTITY.suite,
        "reward_mean",
        LAST_STEPS_KEY,
        REL_CONSTRAINT_KEY,
        -1,
        limit=10,
    )
    assert got == [0.83, 0.81]  # newest-first, passed straight through

    assert len(fake_conn.events) == 1
    kind, sql, params = fake_conn.events[0]
    assert kind == "execute"
    # The authoritative shape, with %s placeholders and Postgres boolean literal.
    for fragment in (
        "from metric_values mv",
        "join runs r using (run_id)",
        "r.test_path = %s",
        "r.backend = %s",
        "r.suite = %s",
        "mv.metric_key = %s",
        "mv.steps_key = %s",
        "mv.constraint_key = %s",
        "mv.step = %s",
        "r.trusted = true",
        "order by r.created_at desc",
        "limit %s",
    ):
        assert fragment in sql, f"baseline query missing: {fragment!r}"
    # Params in the exact positional order the placeholders expect.
    assert params == (
        IDENTITY.test_path,
        IDENTITY.backend,
        IDENTITY.suite,
        "reward_mean",
        LAST_STEPS_KEY,
        REL_CONSTRAINT_KEY,
        -1,
        10,
    )
    # No commit/rollback on a read.
    assert fake_conn.commit_count == 0
    assert fake_conn.rollback_count == 0


# --------------------------------------------------------------------------- #
# mark_untrusted: exactly-one-key guard + parameterized UPDATE.
# --------------------------------------------------------------------------- #


def test_mark_untrusted_requires_exactly_one_key(store, fake_conn):
    # [FUNCTION DOC] Marking untrusted with no selector, or with two selectors, raises a value
    # error; asserts the guard fires before any statement is issued.
    with pytest.raises(ValueError):
        store.mark_untrusted()
    with pytest.raises(ValueError):
        store.mark_untrusted(run_id="x", commit_sha="y")
    # The guard fires before any statement is issued.
    assert fake_conn.events == []


@pytest.mark.parametrize(
    "kwargs,column,value",
    [
        ({"run_id": "abc"}, "run_id", "abc"),
        ({"github_run_id": 7777}, "github_run_id", 7777),
        ({"commit_sha": "badc0de"}, "commit_sha", "badc0de"),
    ],
)
def test_mark_untrusted_issues_parameterized_update(store, fake_conn, kwargs, column, value):
    # [FUNCTION DOC] Parametrized over selecting by run id, github run id and commit sha. Pins the
    # row count to 2 and asserts the returned affected count is 2; exactly one execute, whose SQL
    # updates the runs table setting trusted to false where the selected column equals a
    # placeholder and trusted is currently true, with the single parameter being the value; one
    # commit and no rollback.
    fake_conn.next_rowcount = 2
    affected = store.mark_untrusted(**kwargs)
    assert affected == 2

    update = [e for e in fake_conn.events if e[0] == "execute"]
    assert len(update) == 1
    _, sql, params = update[0]
    assert f"update runs set trusted = false where {column} = %s and trusted = true" in sql
    assert params == (value,)
    assert fake_conn.commit_count == 1
    assert fake_conn.rollback_count == 0


def test_each_operation_opens_and_closes_its_own_connection(store, fake_conn):
    # [FUNCTION DOC] One write and one read on the same store lead to two connects and two closes,
    # so no connection is cached across the long gaps between end-to-end tests.
    # A store built at suite start must not cache a connection across the
    # minutes-long gaps between e2e tests (Neon kills idle sockets); every
    # operation gets a fresh connection and releases it before returning.
    store.write_run(IDENTITY, PROVENANCE, "2026-06-02T00:00:00+00:00", True, [_sample("m", 0.5)])
    store.recent_trusted_values("t", "b", "s", "m", LAST_STEPS_KEY, REL_CONSTRAINT_KEY, -1, limit=1)
    assert fake_conn.connect_count == 2
    assert fake_conn.close_count == 2


def test_connection_closed_even_when_operation_fails(store, fake_conn):
    # [FUNCTION DOC] A failing write raises the simulated error and the connection is still closed
    # exactly once.
    fake_conn.fail_on_execute = True
    with pytest.raises(RuntimeError, match="boom"):
        store.write_run(IDENTITY, PROVENANCE, "2026-06-02T00:00:00+00:00", True, [_sample("m", 1.0)])
    assert fake_conn.close_count == 1


def test_mark_untrusted_rolls_back_on_error(store, fake_conn):
    # [FUNCTION DOC] A failing mark-untrusted raises the simulated error with zero commits and one
    # rollback.
    fake_conn.fail_on_execute = True
    with pytest.raises(RuntimeError, match="boom"):
        store.mark_untrusted(run_id="abc")
    assert fake_conn.commit_count == 0
    assert fake_conn.rollback_count == 1


# --------------------------------------------------------------------------- #
# Live-Postgres smoke test. Skips cleanly when no DSN env var is set.
# --------------------------------------------------------------------------- #

_LIVE_DSN_ENV = "MILES_TEST_POSTGRES_DSN"


# Test-local provisioning DDL, mirroring the out-of-band schema (docs/developer/ci/03:
# the two tables are provisioned outside this repo; the store itself stays
# DML-only). Only the live smoke test ever executes this, as an admin role.
_PROVISION_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    test_path           TEXT NOT NULL,
    backend             TEXT NOT NULL,
    suite               TEXT NOT NULL,
    commit_sha          TEXT NOT NULL,
    pr_number           INTEGER,
    github_run_id       BIGINT,
    github_run_attempt  INTEGER,
    event_name          TEXT,
    ref                 TEXT,
    created_at          TIMESTAMPTZ NOT NULL,
    trusted             BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS metric_values (
    run_id         TEXT NOT NULL REFERENCES runs(run_id),
    metric_key     TEXT NOT NULL,
    steps_key  TEXT NOT NULL,
    constraint_key       TEXT NOT NULL,
    step           INTEGER NOT NULL,
    value          DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS runs_baseline_idx
    ON runs (test_path, backend, suite, trusted, created_at DESC);

CREATE INDEX IF NOT EXISTS metric_values_run_id_idx
    ON metric_values (run_id);
"""


@pytest.mark.skipif(
    not os.environ.get(_LIVE_DSN_ENV),
    reason=f"{_LIVE_DSN_ENV} not set; skipping live-Postgres round-trip",
)
def test_live_postgres_round_trip():
    # [FUNCTION DOC] Live test, skipped unless the test Postgres connection-string environment
    # variable is set. Side effects on that server: as an admin connection drops the metric values
    # and runs tables if present and recreates them from the provisioning script, then commits.
    # Then with the store: writes two trusted runs (values 0.5 then 0.9, a day apart) and asserts
    # their ids differ; reading returns [0.9, 0.5]; marking the newer one untrusted affects one run
    # and the next read returns [0.5]; marking it again affects zero. The store is always closed.
    # Only reached when a real DSN is configured; imports the driver, provisions
    # the two tables on that server, then round-trips the contract.
    import psycopg

    dsn = os.environ[_LIVE_DSN_ENV]

    # Provision as a privileged role (the DSN here is a provisioning DSN, not
    # the least-privilege app role).
    with psycopg.connect(dsn) as admin:
        with admin.cursor() as cur:
            # Clean slate so the round-trip is deterministic across reruns.
            cur.execute("DROP TABLE IF EXISTS metric_values")
            cur.execute("DROP TABLE IF EXISTS runs")
            cur.execute(_PROVISION_SQL)
        admin.commit()

    identity = RunIdentity("tests/e2e/test_smoke.py", "megatron", "stage-x")
    prov = RunProvenance("smoke-sha", 1, 9001, 1, "push", "refs/heads/main")

    store = NeonMetricHistoryStore(dsn=dsn)
    try:
        keep = store.write_run(identity, prov, "2026-06-01T00:00:00+00:00", True, [_sample("m", 0.5)])
        drop = store.write_run(identity, prov, "2026-06-02T00:00:00+00:00", True, [_sample("m", 0.9)])
        assert keep != drop

        got = store.recent_trusted_values(
            identity.test_path,
            identity.backend,
            identity.suite,
            "m",
            LAST_STEPS_KEY,
            REL_CONSTRAINT_KEY,
            -1,
            limit=10,
        )
        assert got == [0.9, 0.5]  # newest-first

        assert store.mark_untrusted(run_id=drop) == 1
        # No rebaseline: the dropped run is gone on the next query.
        got2 = store.recent_trusted_values(
            identity.test_path,
            identity.backend,
            identity.suite,
            "m",
            LAST_STEPS_KEY,
            REL_CONSTRAINT_KEY,
            -1,
            limit=10,
        )
        assert got2 == [0.5]
        # Idempotent.
        assert store.mark_untrusted(run_id=drop) == 0
    finally:
        store.close()
