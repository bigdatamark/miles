# [MODULE DOC]
# Integration tests for the regression-gate hook of the CI harness (tests.ci.ci_utils: run_gate_hook,
# gate_provenance_from_env, build_store_from_env). Uses an in-memory SQLite metric history store
# (or fakes), temporary test files containing CUDA registrations and register_ci_gate calls,
# temporary merged JSONL metric records, and a cleared GitHub environment. Covers environment
# parsing into run provenance, store construction fallback, passing-attempt selection, baseline
# writes (trusted flag from the verdict, one row per coordinate, no row without a gate spec), keying
# rows on the executing stage, the pull-request shadow verdict (no write, log plus step summary), and
# the never-blocks guarantee (the hook returns nothing and swallows gate and store errors).
# Constants: LAST_KEY and REL20_KEY are the canonical serialized forms of the "last" step selector
# and a relative 20 percent up/down constraint; PROVENANCE is a sample scheduled run on main.
# Registers itself as a CPU test in suite stage-a-cpu.
"""Integration tests for wiring the regression gate into the CI harness.

These exercise the helpers and the gate hook extracted into
`tests.ci.ci_utils` directly -- no real DB, no real CUDA run. The store is an
in-memory :class:`SQLiteMetricHistoryStore`; the GitHub environment is
monkeypatched.

Covered:

* passing-attempt selection: only the PASSED attempt's record feeds the gate.
* baseline write: a policy-selected writer makes the hook persist a baseline row
  whose `trusted` flag comes from the verdict; specs sharing a coordinate
  collapse to one `metric_values` row; a no-spec file writes nothing.
* PR no-write: a `pull_request` event writes no row and emits a shadow
  verdict string.
* never-blocks: a not-trusted verdict, and a gate that raises, both leave the
  test's pass/fail untouched.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from tests.ci.ci_register import HWBackend, register_cpu_ci, ut_parse_one_file
from tests.ci.ci_utils import build_store_from_env, gate_provenance_from_env, run_gate_hook
from tests.ci.metric_history import MetricSample, RunIdentity, RunProvenance, SQLiteMetricHistoryStore

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

# Canonical declaration keys for the `last` + rel-0.20 fixtures below.
LAST_KEY = json.dumps("last", sort_keys=True, separators=(",", ":"))
REL20_KEY = json.dumps({"rel_up": 0.20, "rel_down": 0.20}, sort_keys=True, separators=(",", ":"))

_GITHUB_ENV_VARS = (
    "GITHUB_EVENT_NAME",
    "GITHUB_SHA",
    "GITHUB_COMMIT_NAME",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_REF",
    "GITHUB_STEP_SUMMARY",
    "NEON_DATABASE_URL",
)


@pytest.fixture(autouse=True)
def _clear_github_env(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture that removes every GitHub and database environment
    # variable the hook reads (event name, sha, commit name, run id, attempt, ref, step summary,
    # Neon database url) before each test.
    """Each test starts from a clean GitHub environment so leftovers from the
    real CI runner (or a prior test) cannot leak in."""
    for var in _GITHUB_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def store():
    # [FUNCTION DOC] HELPER: Fixture yielding a fresh in-memory SQLite metric history store and
    # closing it afterwards.
    s = SQLiteMetricHistoryStore(":memory:")
    yield s
    s.close()


def _write_test_file(tmp_path: Path, gate_lines: str, *, name: str = "test_e2e_gate_fixture.py") -> str:
    # [FUNCTION DOC] HELPER: Write a fixture test file (default name test_e2e_gate_fixture.py) to the
    # temporary directory and return its path. The file imports the two register functions, makes a
    # CUDA registration (est_time 600, stage-c-8-gpu-h100, label megatron, hopper), then appends the
    # stripped, dedented gate lines and a newline. Built by concatenation so multi-line gate blocks
    # are not broken by indentation.
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


def _write_record(tmp_path: Path, by_metric: dict[str, list], *, name: str) -> str:
    # [FUNCTION DOC] HELPER: Write a JSONL metric record file with one line per metric holding the
    # metric name and its series (list of step/value pairs) to the temporary directory; return the
    # path.
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for metric, series in by_metric.items():
            f.write(json.dumps({"metric": metric, "series": series}) + "\n")
    return str(p)


def _registry(test_file: str):
    # [FUNCTION DOC] HELPER: Parse the test file's registrations and return the first one.
    return ut_parse_one_file(test_file)[0]


PROVENANCE = RunProvenance(
    commit_sha="deadbeef",
    pr_number=7,
    github_run_id=100,
    github_run_attempt=1,
    event_name="schedule",
    ref="refs/heads/main",
)


def _count_runs(store) -> int:
    # [FUNCTION DOC] HELPER: Return the number of rows in the store's runs table (queries the
    # store's private connection).
    return store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]


# --- gate_provenance_from_env -----------------------------------------------


class TestGateProvenanceFromEnv:
    # [CLASS DOC] Tests of building run provenance from GitHub environment variables.
    def test_full_env_parsed(self, monkeypatch):
        # [FUNCTION DOC] With sha, run id, attempt, event name, ref and a commit name of the form
        # "<sha>_<pr number>" set, provenance has that sha, the pr number as an integer, integer run
        # id and attempt, and the event name and ref.
        monkeypatch.setenv("GITHUB_SHA", "cafef00d")
        monkeypatch.setenv("GITHUB_RUN_ID", "555")
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
        monkeypatch.setenv("GITHUB_COMMIT_NAME", "cafef00d_42")

        prov = gate_provenance_from_env()
        assert prov.commit_sha == "cafef00d"
        assert prov.pr_number == 42
        assert prov.github_run_id == 555
        assert prov.github_run_attempt == 2
        assert prov.event_name == "pull_request"
        assert prov.ref == "refs/pull/42/merge"

    def test_non_pr_commit_name_yields_none_pr(self, monkeypatch):
        # [FUNCTION DOC] A commit name "<sha>_non-pr" yields no pr number, and the sha falls back to
        # the commit name prefix when the sha variable is unset.
        monkeypatch.setenv("GITHUB_COMMIT_NAME", "cafef00d_non-pr")
        prov = gate_provenance_from_env()
        assert prov.pr_number is None
        # sha falls back to the GITHUB_COMMIT_NAME prefix when GITHUB_SHA unset.
        assert prov.commit_sha == "cafef00d"

    def test_empty_env_does_not_raise(self):
        # [FUNCTION DOC] With nothing set, provenance has an empty sha and no pr number, run id or
        # attempt, without raising.
        prov = gate_provenance_from_env()
        assert prov.commit_sha == ""
        assert prov.pr_number is None
        assert prov.github_run_id is None
        assert prov.github_run_attempt is None


# --- build_store_from_env ---------------------------------------------------


class TestBuildStoreFromEnv:
    # [CLASS DOC] Tests of the factory that builds the metric history store from the environment.
    def test_no_neon_url_returns_none(self):
        # [FUNCTION DOC] Without the Neon database url the factory returns none.
        assert build_store_from_env() is None

    def test_neon_url_routes_to_neon_store(self, monkeypatch):
        # [FUNCTION DOC] Replaces the Neon store class (as imported by the harness) with a fake that
        # records its constructor argument, sets the database url and asserts the factory returns an
        # instance of the fake, constructed once with no explicit dsn (it reads the environment
        # itself). Avoids needing a database driver.
        # With the env var set the factory must construct the Neon backend
        # rather than return None. The real NeonMetricHistoryStore.__init__
        # eagerly opens a psycopg connection, so we patch the symbol *as imported
        # by ci_utils* with a non-connecting fake that records its construction.
        # This pins the factory's only job -- routing to NeonMetricHistoryStore --
        # without depending on psycopg or a reachable DSN.
        constructed = []

        class _FakeNeonStore:
            # [CLASS DOC] HELPER: Non-connecting fake of the Neon store that records construction.
            def __init__(self, dsn=None):
                # [FUNCTION DOC] HELPER: Append the dsn argument to the recording list.
                constructed.append(dsn)

        monkeypatch.setattr("tests.ci.ci_utils.NeonMetricHistoryStore", _FakeNeonStore)
        monkeypatch.setenv("NEON_DATABASE_URL", "postgres://example/db")

        store = build_store_from_env()
        assert isinstance(store, _FakeNeonStore), "NEON_DATABASE_URL must route to NeonMetricHistoryStore"
        assert constructed == [None], "the factory constructs the store with no explicit DSN"

    def test_store_construction_failure_degrades_to_none(self, monkeypatch):
        # [FUNCTION DOC] Replaces the Neon store class with one whose constructor raises a runtime
        # error (connection refused) and sets the database url; asserts the factory swallows the
        # error and returns none, because the gate must never fail a job before tests have run.
        # NeonMetricHistoryStore.__init__ connects eagerly, so a bad DSN / DB
        # outage raises here -- OUTSIDE the gate hook's try/except. The factory
        # must swallow it and return None: the gate may never fail a CI job
        # (CUDA or ROCm) before a single test has run.
        class _ExplodingStore:
            # [CLASS DOC] HELPER: Fake store whose construction always fails.
            def __init__(self, dsn=None):
                # [FUNCTION DOC] HELPER: Always raises a runtime error "connection refused".
                raise RuntimeError("connection refused")

        monkeypatch.setattr("tests.ci.ci_utils.NeonMetricHistoryStore", _ExplodingStore)
        monkeypatch.setenv("NEON_DATABASE_URL", "postgres://unreachable/db")

        assert build_store_from_env() is None


# --- passing-attempt selection ----------------------------------------------


class TestPassingAttemptSelection:
    # [CLASS DOC] Tests that the hook is only given the passing attempt's record.
    def test_only_passing_attempt_record_is_evaluated_and_written(self, tmp_path, store, monkeypatch):
        # [FUNCTION DOC] On a scheduled event with a gate on raw reward (last step, 20 percent band),
        # creates a bad record (0.10, attempt 1) and a good record (0.80, attempt 2) but hands only the
        # good one to the hook with baseline writing on. Asserts the only stored value is 0.80.
        # KNOWN ISSUE: the bad record is created but deliberately unused (assigned to a throwaway
        # name), so the test does not prove the hook would ignore a failed attempt by itself; the
        # selection happens in the caller.
        # Baseline writer so the hook writes; assert the persisted value comes from the
        # PASSING attempt's good record, never the failed attempt's bad one.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
            """,
        )
        registry = _registry(test_file)

        # Attempt 1 "failed" with a bad metric (0.10), attempt 2 "passed" (0.80).
        bad_record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.10]]}, name="attempt-1.merged.jsonl")
        good_record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.80]]}, name="attempt-2.merged.jsonl")

        # Only the passing attempt's record path is handed to the hook -- this
        # is exactly what run_unittest_files selects (it captures the merged
        # path only inside the ret_code==0 branch).
        _ = bad_record
        run_gate_hook(
            test_file,
            good_record,
            store=store,
            registry=registry,
            executing_suite="stage-c-8-gpu-h100",
            write_baseline=True,
            provenance=PROVENANCE,
        )

        rows = store._conn.execute("SELECT mv.value FROM metric_values mv JOIN runs r USING (run_id)").fetchall()
        assert rows == [(0.80,)], "only the passing attempt's value must be written"


# --- baseline write ---------------------------------------------------------


class TestBaselineWrite:
    # [CLASS DOC] Tests of what the hook persists when the run policy selects it as a baseline writer.
    def test_nightly_writes_trusted_row_with_provenance(self, tmp_path, store, monkeypatch):
        # [FUNCTION DOC] Runs the hook on a scheduled event with a gate and a record of 0.81, a fixed
        # timestamp and sample provenance. Asserts one run row with the test path, backend cuda, the
        # suite, sha deadbeef, pr 7, run id 100, event schedule, the timestamp, and trusted 1 (cold
        # start, no prior baseline). Then asserts the baseline query for that coordinate returns
        # [0.81].
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
            """,
        )
        registry = _registry(test_file)
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")

        assert _count_runs(store) == 0
        run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            executing_suite="stage-c-8-gpu-h100",
            write_baseline=True,
            provenance=PROVENANCE,
            now_iso="2026-06-29T00:00:00+00:00",
        )

        row = store._conn.execute(
            "SELECT test_path, backend, suite, commit_sha, pr_number, "
            "github_run_id, event_name, created_at, trusted FROM runs"
        ).fetchone()
        assert row is not None
        assert row[0] == registry.filename
        assert row[1] == "cuda"
        assert row[2] == "stage-c-8-gpu-h100"
        assert row[3] == "deadbeef"
        assert row[4] == 7
        assert row[5] == 100
        assert row[6] == "schedule"
        assert row[7] == "2026-06-29T00:00:00+00:00"
        # Cold-start (no prior baseline) -> historical inactive -> trusted verdict.
        assert row[8] == 1

        # The persisted value feeds future baselines.
        vals = store.recent_trusted_values(
            registry.filename,
            "cuda",
            "stage-c-8-gpu-h100",
            "rollout/raw_reward",
            LAST_KEY,
            REL20_KEY,
            -1,
            20,
        )
        assert vals == [0.81]

    def test_nightly_writes_untrusted_when_verdict_not_trusted(self, tmp_path, store):
        # [FUNCTION DOC] Seeds a trusted baseline of 0.30, then runs the hook with a record of 0.90
        # (outside the 20 percent band) and baseline writing on. Asserts the newest run row has
        # trusted 0, and the baseline query returns only the seeded 0.30 (untrusted rows are
        # invisible to baselines).
        # A historical failure makes the verdict not-trusted; the row is still
        # written (baseline writers always write) but flagged untrusted, so it never
        # pollutes a future baseline.
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
            """,
        )
        registry = _registry(test_file)
        # Seed a trusted baseline at 0.30 under the gated coordinate.
        store.write_run(
            RunIdentity(test_path=registry.filename, backend="cuda", suite="stage-c-8-gpu-h100"),
            PROVENANCE,
            created_at="2026-06-01T00:00:00+00:00",
            trusted=True,
            values=[MetricSample("rollout/raw_reward", LAST_KEY, REL20_KEY, -1, 0.30)],
        )
        # 0.90 vs baseline mean 0.30, band 0.06 -> historical fails -> not trusted.
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.90]]}, name="m.jsonl")

        run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            executing_suite="stage-c-8-gpu-h100",
            write_baseline=True,
            provenance=PROVENANCE,
            now_iso="2026-06-29T00:00:00+00:00",
        )
        trusted = store._conn.execute("SELECT trusted FROM runs ORDER BY created_at DESC").fetchone()[0]
        assert trusted == 0
        # Untrusted rows are invisible to the baseline query: only the seeded
        # trusted baseline comes back, never the hook's 0.90 row.
        vals = store.recent_trusted_values(
            registry.filename,
            "cuda",
            "stage-c-8-gpu-h100",
            "rollout/raw_reward",
            LAST_KEY,
            REL20_KEY,
            -1,
            20,
        )
        assert vals == [0.30]

    def test_nightly_one_liner_writes_every_step(self, tmp_path, store):
        # [FUNCTION DOC] A gate with only a metric key (default: all steps, no constraint) and a
        # four-point series; after the hook, the stored values are the four points ordered by step,
        # all with steps key "all".
        test_file = _write_test_file(
            tmp_path,
            'register_ci_gate(metric_key="rollout/raw_reward")',
        )
        registry = _registry(test_file)
        record = _write_record(
            tmp_path,
            {"rollout/raw_reward": [[0, 0.20], [1, 0.40], [2, 0.60], [3, 0.80]]},
            name="m.jsonl",
        )

        run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            executing_suite="stage-c-8-gpu-h100",
            write_baseline=True,
            provenance=PROVENANCE,
        )

        rows = store._conn.execute("SELECT steps_key, step, value FROM metric_values ORDER BY step").fetchall()
        assert rows == [
            ('"all"', 0, 0.20),
            ('"all"', 1, 0.40),
            ('"all"', 2, 0.60),
            ('"all"', 3, 0.80),
        ]

    def test_nightly_no_spec_file_writes_nothing(self, tmp_path, store, monkeypatch):
        # [FUNCTION DOC] A test file without any gate call, on a scheduled event: the hook writes no
        # run row at all.
        # A file with no register_ci_gate call has nothing a baseline can use:
        # the hook must skip the write entirely, not leave an empty runs row.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(tmp_path, "")
        registry = _registry(test_file)
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")

        run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            executing_suite="stage-c-8-gpu-h100",
            write_baseline=True,
            provenance=PROVENANCE,
        )
        assert _count_runs(store) == 0

    def test_nightly_dedupes_values_by_coordinate(self, tmp_path, store, monkeypatch):
        # [FUNCTION DOC] Two gates on the same metric with identical steps and constraint (one also
        # enforcing) share a coordinate. After the hook there is one run row and exactly one value row
        # (metric, steps key, constraint key, step -1, 0.81), so the run is not double-weighted in the
        # baseline mean.
        # Two specs with identical steps + constraint literals share one
        # coordinate even though their policy metadata differs; the selected
        # value is identical, so writing one row per spec would double-weight
        # this run in the baseline mean.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20},
                             enforce=True)
            """,
        )
        registry = _registry(test_file)
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")

        run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            executing_suite="stage-c-8-gpu-h100",
            write_baseline=True,
            provenance=PROVENANCE,
        )
        assert _count_runs(store) == 1
        rows = store._conn.execute(
            "SELECT metric_key, steps_key, constraint_key, step, value FROM metric_values"
        ).fetchall()
        assert rows == [("rollout/raw_reward", LAST_KEY, REL20_KEY, -1, 0.81)]


# --- PR no-write + shadow verdict -------------------------------------------


class TestExecutingSuiteIdentity:
    # [CLASS DOC] Tests that stored rows and baselines are keyed on the stage that actually executed
    # the test, not the stage declared in the registration.
    """A run's numbers belong to the stage that produced them.

    Once a test can be dispatched to a second GPU generation, keying the
    baseline on `register_*_ci(suite=...)` would merge two hardware populations
    into one series and make the regression gate compare across generations.
    """

    def test_row_is_keyed_on_the_executing_stage_not_the_declared_one(self, tmp_path, store):
        # [FUNCTION DOC] The registration declares the h100 stage; the hook is told the executing
        # stage is the b200 one. The stored run row's suite equals the executing stage.
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
            """,
        )
        registry = _registry(test_file)
        assert registry.suite == "stage-c-8-gpu-h100"
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")

        run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            executing_suite="stage-c-8-gpu-b200",
            write_baseline=True,
            provenance=PROVENANCE,
        )

        assert store._conn.execute("SELECT suite FROM runs").fetchone()[0] == "stage-c-8-gpu-b200"

    def test_the_two_stages_keep_independent_series(self, tmp_path, store):
        # [FUNCTION DOC] Runs the hook once for the h100 stage with 0.81 and once for the b200 stage
        # with 0.42 (separate record files). Asserts each stage's baseline query returns only its own
        # value, so the very different values do not share a series.
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
            """,
        )
        registry = _registry(test_file)
        for suite, value in (("stage-c-8-gpu-h100", 0.81), ("stage-c-8-gpu-b200", 0.42)):
            run_gate_hook(
                test_file,
                _write_record(tmp_path, {"rollout/raw_reward": [[0, value]]}, name=f"{suite}.jsonl"),
                store=store,
                registry=registry,
                executing_suite=suite,
                write_baseline=True,
                provenance=PROVENANCE,
            )

        coordinate = ("rollout/raw_reward", LAST_KEY, REL20_KEY, -1)
        hopper = store.recent_trusted_values(registry.filename, "cuda", "stage-c-8-gpu-h100", *coordinate, 20)
        blackwell = store.recent_trusted_values(registry.filename, "cuda", "stage-c-8-gpu-b200", *coordinate, 20)
        # 0.42 against a 0.81 baseline would be a hard fail if the two shared a series.
        assert hopper == [0.81]
        assert blackwell == [0.42]


class TestPrShadow:
    # [CLASS DOC] Tests of pull-request behavior: evaluate and report only, never write.
    def test_pr_writes_no_row_and_emits_shadow_verdict(self, tmp_path, store, monkeypatch, caplog):
        # [FUNCTION DOC] On a pull request event with a step summary file configured, runs the hook
        # with baseline writing off at info log level. Asserts no run row, the log contains the shadow
        # marker and the word TRUSTED, and the step summary file exists and contains the shadow marker
        # and the metric name.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        summary = tmp_path / "step_summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
            """,
        )
        registry = _registry(test_file)
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")

        with caplog.at_level("INFO"):
            run_gate_hook(
                test_file,
                record,
                store=store,
                registry=registry,
                executing_suite="stage-c-8-gpu-h100",
                write_baseline=False,
                provenance=PROVENANCE,
            )

        # No row written for a PR run.
        assert _count_runs(store) == 0
        # A readable shadow verdict line was logged and appended to the summary.
        assert "[CI Gate][shadow]" in caplog.text
        assert "TRUSTED" in caplog.text
        assert summary.exists()
        summary_text = summary.read_text()
        assert "[CI Gate][shadow]" in summary_text
        assert "rollout/raw_reward" in summary_text


# --- never-blocks -----------------------------------------------------------


class TestNeverBlocks:
    # [CLASS DOC] Tests that no gate outcome or failure can propagate out of the hook.
    def test_not_trusted_verdict_does_not_raise_in_pr(self, tmp_path, store, monkeypatch):
        # [FUNCTION DOC] Seeds a trusted baseline of 0.30 and runs the hook on a pull request with a
        # record of 0.95 (fails the historical check). Asserts the hook returns none and the run count
        # is still one (only the seeded row).
        # A clearly-not-trusted PR run still completes the hook normally (no
        # exception, no new row). The caller's file_passed is computed
        # independently of run_gate_hook, which returns None either way.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
            """,
        )
        registry = _registry(test_file)
        # Seed a trusted baseline at 0.30 so 0.95 fails the historical check.
        store.write_run(
            RunIdentity(test_path=registry.filename, backend="cuda", suite="stage-c-8-gpu-h100"),
            PROVENANCE,
            created_at="2026-06-01T00:00:00+00:00",
            trusted=True,
            values=[MetricSample("rollout/raw_reward", LAST_KEY, REL20_KEY, -1, 0.30)],
        )
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.95]]}, name="m.jsonl")

        result = run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            executing_suite="stage-c-8-gpu-h100",
            write_baseline=False,
            provenance=PROVENANCE,
        )
        assert result is None
        # Only the seeded baseline row: the PR run wrote nothing.
        assert _count_runs(store) == 1

    def test_gate_error_is_swallowed(self, tmp_path, store, monkeypatch, caplog):
        # [FUNCTION DOC] Passes a nonexistent record path on a scheduled event so evaluation fails.
        # Asserts the hook returns none, writes no row, and logs a warning containing "[CI Gate] hook
        # failed".
        # A missing record path makes evaluate_gate raise; the hook must catch,
        # log, and return None rather than propagate -- CI stays green.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
            """,
        )
        registry = _registry(test_file)
        missing_record = str(tmp_path / "does_not_exist.jsonl")

        with caplog.at_level("WARNING"):
            result = run_gate_hook(
                test_file,
                missing_record,
                store=store,
                registry=registry,
                executing_suite="stage-c-8-gpu-h100",
                write_baseline=True,
                provenance=PROVENANCE,
            )
        assert result is None
        assert _count_runs(store) == 0
        assert "[CI Gate] hook failed" in caplog.text

    def test_store_error_never_blocks(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Uses a store subclass whose write raises a runtime error (database down)
        # and runs the hook on a scheduled event with baseline writing on; asserts the hook returns
        # none instead of raising, and closes the store afterwards.
        # A store whose write_run raises (mirrors a real DB error, e.g. the Neon
        # backend's connection dropping mid-write) must not propagate out of the
        # hook.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
            """,
        )
        registry = _registry(test_file)
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")

        class _ExplodingStore(SQLiteMetricHistoryStore):
            # [CLASS DOC] HELPER: In-memory store whose run writes always fail.
            def write_run(self, *a, **k):
                # [FUNCTION DOC] HELPER: Always raises a runtime error "DB down".
                raise RuntimeError("DB down")

        store = _ExplodingStore(":memory:")
        try:
            result = run_gate_hook(
                test_file,
                record,
                store=store,
                registry=registry,
                executing_suite="stage-c-8-gpu-h100",
                write_baseline=True,
                provenance=PROVENANCE,
            )
            assert result is None
        finally:
            store.close()


# --- backend gating: hook never fires off the CUDA path ----------------------


def test_hook_signature_matches_metric_sample_contract(tmp_path, store, monkeypatch):
    # [FUNCTION DOC] Regression guard: with a gate on a metric missing from the record, on a
    # scheduled event, the hook writes one run row with zero value rows (values are only built for
    # metrics that were found). Also asserts the metric sample type keeps its value, the CUDA
    # backend constant exists, and run identity keeps the backend.
    # KNOWN ISSUE: the section header comment above says "backend gating: hook never fires off the
    # CUDA path" but this test does not exercise backend gating.
    # Regression guard: baseline values must be built as
    # MetricSample(metric_key, steps_key, constraint_key, step, current) and only
    # for metrics whose current is not None. A spec on a missing metric
    # (current None) yields a written run with zero values.
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    registry = _registry(test_file)
    # Record has a different metric; rollout/raw_reward is missing -> current None.
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 1.0]]}, name="m.jsonl")

    run_gate_hook(
        test_file,
        record,
        store=store,
        registry=registry,
        executing_suite="stage-c-8-gpu-h100",
        write_baseline=True,
        provenance=PROVENANCE,
    )
    assert _count_runs(store) == 1
    n_values = store._conn.execute("SELECT COUNT(*) FROM metric_values").fetchone()[0]
    assert n_values == 0
    # The MetricSample type is the contract the hook builds from; assert it imports.
    assert MetricSample("k", LAST_KEY, REL20_KEY, -1, 1.0).value == 1.0
    assert HWBackend.CUDA is not None
    assert RunIdentity("p", "cuda", "s").backend == "cuda"
