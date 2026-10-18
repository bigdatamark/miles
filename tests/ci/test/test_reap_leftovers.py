# [MODULE DOC]
# Tests of the CI utility that reaps leaked accelerator processes between test files, and of the
# helper deciding whether reaping is safe. All process and system calls are faked by replacing
# the subprocess run function and the settle/poll constants in the CI utility module, so no real
# process is killed. Covered: both the ray runtime and the sglang engines are targeted; the engine
# pattern ends in a double colon so it cannot match test file paths; survivors that remain after
# the wait are logged as a warning, while zombie (defunct) entries are not counted; the settle
# window is spent even when nothing survives; a clean reap logs nothing; the unit-test file runner
# reaps only when asked and the suite runner passes the isolation check; isolation is true only
# in a CI container without an external ray cluster; a missing killer binary does not abort.
# Registers itself as a CPU test in the stage-a-cpu suite.
import inspect
import logging
import subprocess
import textwrap
from pathlib import Path

import pytest

from tests.ci import ci_utils, run_suite
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@pytest.fixture
def recorded_argvs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    # [FUNCTION DOC] HELPER: Fixture. Replaces the subprocess run function used by the CI utility
    # with a fake that records each command line and reports success with empty output (what a
    # successful reap looks like in the process listing), and sets the settle and poll delays to
    # zero. Returns the list that accumulates the recorded command lines.
    argvs: list[list[str]] = []

    def fake_run(argv, **kwargs):
        # [FUNCTION DOC] HELPER: Record a copy of the command line and return a successful
        # completed process with empty standard output.
        argvs.append(list(argv))
        # An empty ps listing is what a reap that worked looks like.
        return subprocess.CompletedProcess(argv, 0, stdout="")

    monkeypatch.setattr(ci_utils.subprocess, "run", fake_run)
    monkeypatch.setattr(ci_utils, "_REAP_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(ci_utils, "_REAP_POLL_SECONDS", 0.0)
    return argvs


@pytest.fixture
def one_passing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[ci_utils.TestFile]:
    # [FUNCTION DOC] HELPER: Fixture. Writes a script that exits successfully into the temporary
    # directory, changes the working directory to it (side effect, undone by the monkeypatch), and
    # returns a one-element list with a test file record for it (estimated time one second).
    """A single trivially-passing test file, so run_unittest_files really enters its per-file loop."""
    (tmp_path / "t_pass.py").write_text(
        textwrap.dedent(
            """
        import sys
        sys.exit(0)
    """
        )
    )
    monkeypatch.chdir(tmp_path)
    return [ci_utils.TestFile(name="t_pass.py", estimated_time=1)]


def _count_reaps(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    # [FUNCTION DOC] HELPER: Replace the CI utility's reaper with a stub that appends a marker
    # string to a list on each call. Returns that list so tests can count reaper invocations.
    calls: list[str] = []
    monkeypatch.setattr(ci_utils, "reap_leaked_accelerator_processes", lambda: calls.append("reaped"))
    return calls


class TestReapLeakedAcceleratorProcesses:
    # [CLASS DOC] Tests of the reaper function itself and of how the unit-test file runner and the
    # suite runner invoke it, using faked process calls.
    def test_both_the_ray_runtime_and_the_leaked_engines_are_reaped(self, recorded_argvs: list[list[str]]) -> None:
        # [FUNCTION DOC] Calls the reaper and asserts the recorded commands include a forced ray
        # stop and at least one command whose arguments mention the sglang engine process title.
        """A leaked sglang scheduler is not under ray's control, so stopping ray alone leaves it
        holding accelerator memory and the next test file starts on a dirty device."""
        ci_utils.reap_leaked_accelerator_processes()

        assert ["ray", "stop", "--force"] in recorded_argvs
        assert any("sglang::" in arg for argv in recorded_argvs for arg in argv)

    def test_the_engine_pattern_cannot_match_a_test_file_path(self, recorded_argvs: list[list[str]]) -> None:
        # [FUNCTION DOC] Calls the reaper; takes the last argument of every recorded pattern-kill
        # command; asserts there is at least one and each ends with a double colon, so it cannot
        # match test paths containing the engine name.
        """Test paths under tests/e2e/sglang/ contain "sglang", so a bare pattern would make the
        reaper kill the very test process it is preparing for."""
        ci_utils.reap_leaked_accelerator_processes()

        patterns = [argv[-1] for argv in recorded_argvs if argv[0] == "pkill"]
        assert patterns
        for pattern in patterns:
            assert pattern.endswith("::")

    def test_leftovers_that_outlive_the_wait_are_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # [FUNCTION DOC] Fakes the process listing to always show a live ray actor process, zero
        # settle and poll delays, runs the reaper capturing warning-level logs, and asserts a log
        # record mentions "still alive".
        """A kill that missed leaves the next file on an occupied device, and staying silent
        about it is what makes the infrastructure failure read as that test being broken."""

        def fake_run(argv, **kwargs):
            # [FUNCTION DOC] HELPER: Fake process call that always reports success with a
            # listing showing one live (sleeping state) ray actor process.
            return subprocess.CompletedProcess(argv, 0, stdout="Sl   ray::MegatronTrainRayActor.train()\n")

        monkeypatch.setattr(ci_utils.subprocess, "run", fake_run)
        monkeypatch.setattr(ci_utils, "_REAP_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(ci_utils, "_REAP_POLL_SECONDS", 0.0)

        with caplog.at_level(logging.WARNING):
            ci_utils.reap_leaked_accelerator_processes()

        assert any("still alive" in record.message for record in caplog.records)

    def test_a_killed_process_awaiting_its_parent_is_not_mistaken_for_a_survivor(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # [FUNCTION DOC] Fakes the process listing to show only a zombie (defunct state) ray
        # process, runs the reaper capturing warnings, and asserts no "still alive" record is
        # logged, since zombies awaiting a parent that never reaps them are not survivors.
        """SIGKILL leaves the entry in the table until the parent reaps it, and the job's pid 1
        never will, so counting those reports every successful reap as a failure."""

        def fake_run(argv, **kwargs):
            # [FUNCTION DOC] HELPER: Fake process call that always reports success with a
            # listing showing one zombie (defunct) ray actor process.
            return subprocess.CompletedProcess(argv, 0, stdout="Z    ray::MegatronTrainRayActor.train() <defunct>\n")

        monkeypatch.setattr(ci_utils.subprocess, "run", fake_run)
        monkeypatch.setattr(ci_utils, "_REAP_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(ci_utils, "_REAP_POLL_SECONDS", 0.0)

        with caplog.at_level(logging.WARNING):
            ci_utils.reap_leaked_accelerator_processes()

        assert not [record for record in caplog.records if "still alive" in record.message]

    def test_the_settle_window_is_spent_even_when_nothing_survives(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Fakes the process listing as empty, replaces the sleep function with one
        # recording its durations, sets the settle window to 4 seconds, runs the reaper, and
        # asserts the total slept time is about 4 seconds (tolerance half a second, because the
        # survivor check consumes part of the window).
        """The driver frees the memory after its holders are gone, so an empty process table is
        not yet a clean device; polling exists to tell whether the kill worked, not to cut the
        wait short."""
        slept: list[float] = []
        monkeypatch.setattr(
            ci_utils.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout="")
        )
        monkeypatch.setattr(ci_utils.time, "sleep", slept.append)
        monkeypatch.setattr(ci_utils, "_REAP_SETTLE_SECONDS", 4.0)

        ci_utils.reap_leaked_accelerator_processes()

        # Approximate because the survivor check itself consumes part of the window.
        assert sum(slept) == pytest.approx(4.0, abs=0.5)

    def test_a_clean_reap_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch, recorded_argvs: list[list[str]], caplog: pytest.LogCaptureFixture
    ) -> None:
        # [FUNCTION DOC] Runs the reaper with the empty process listing capturing warnings; asserts
        # a process listing command was issued and no "still alive" record was logged.
        """The warning only earns its place if the ordinary path stays quiet."""
        with caplog.at_level(logging.WARNING):
            ci_utils.reap_leaked_accelerator_processes()

        assert any(argv[0] == "ps" for argv in recorded_argvs)
        assert not [record for record in caplog.records if "still alive" in record.message]

    def test_running_test_files_does_not_reap_unless_asked(
        self, monkeypatch: pytest.MonkeyPatch, one_passing_file: list[ci_utils.TestFile]
    ) -> None:
        # [FUNCTION DOC] Stubs the reaper with a counter, runs the unit-test file runner on the one
        # passing file with a 30 second per-file timeout and no reaping option, and asserts the
        # reaper was never called.
        """Reaping is process-wide, so a run_unittest_files caller that is itself a test would be
        killed by its own reaper; only the CUDA suite runner opts in."""
        calls = _count_reaps(monkeypatch)

        ci_utils.run_unittest_files(one_passing_file, timeout_per_file=30)

        assert calls == []

    def test_running_test_files_reaps_before_a_file_when_asked(
        self, monkeypatch: pytest.MonkeyPatch, one_passing_file: list[ci_utils.TestFile]
    ) -> None:
        # [FUNCTION DOC] Same as the previous test but passing the reap-leftovers option; asserts
        # the reaper was called exactly once for the single file.
        """Without this the CUDA suite is back to starting every file on whatever the previous one
        left holding the accelerators."""
        calls = _count_reaps(monkeypatch)

        ci_utils.run_unittest_files(one_passing_file, timeout_per_file=30, reap_leftovers=True)

        assert calls == ["reaped"]

    def test_the_cuda_suite_runner_asks_for_reaping_only_when_isolated(self) -> None:
        # [FUNCTION DOC] Reads the source text of the suite runner function and asserts it passes
        # the reap-leftovers option computed by the isolation check (a textual check, not a
        # behavioral one).
        """The opt-in only protects CI if run_suite passes it, and only CI's pid namespace makes it safe."""
        source = inspect.getsource(run_suite.run_a_suite)

        assert "reap_leftovers=reaping_is_isolated()" in source


class TestReapingIsIsolated:
    # [CLASS DOC] Tests of the predicate that says whether process reaping is safe, driven by the
    # CI and external-ray-cluster environment variables, plus one robustness test.
    def test_a_ci_container_is_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With the CI variable set to true and no external ray variable, the
        # predicate returns true.
        """GitHub Actions runs every accelerator stage in its own pid namespace, where reaping is safe."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY", raising=False)

        assert ci_utils.reaping_is_isolated() is True

    def test_a_local_run_is_not_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With the CI variable unset, the predicate returns false.
        """A developer running run_suite on a shared host must not have their other workloads killed."""
        monkeypatch.delenv("CI", raising=False)

        assert ci_utils.reaping_is_isolated() is False

    def test_an_external_ray_cluster_is_not_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With the CI variable true but the external ray variable set to 1, the
        # predicate returns false.
        """An externally managed Ray cluster outlives the suite, so its processes are not leftovers."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")

        assert ci_utils.reaping_is_isolated() is False

    def test_a_missing_reaper_binary_does_not_abort_the_suite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Makes every process call raise a file-not-found error (as when the pkill
        # binary is absent), zero settle delay, and asserts the reaper returns without raising.
        """Reaping is best-effort housekeeping; letting OSError escape would fail every test file
        on a machine without pkill."""
        monkeypatch.setattr(ci_utils, "_REAP_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(
            ci_utils.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("pkill"))
        )

        ci_utils.reap_leaked_accelerator_processes()
