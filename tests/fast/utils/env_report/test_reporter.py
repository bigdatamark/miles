# [MODULE DOC] Unit tests for the environment reporter: the one-line summary of a repository's
# state, logging a collected report both as a stored event (read back from the event log
# directory) and as a summary log line, the startup entry point (a quick unprobed report stored
# synchronously, then a background thread), and the background reporter class (timing and
# jitter of its delay, settling delay, rebinding to final arguments, stop behavior, non-positive
# and non-finite intervals, and resilience to failures). Collection of packages is mocked through
# shared fixtures; threads are coordinated with events and generous timeouts.
import logging
import random
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from tests.fast.utils.env_report.conftest import make_args, mock_pip_inspect

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import EnvReportEvent, EnvReportGitRepoInfo
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.env_report.collector import collect_env_report, collect_env_report_snapshot
from miles.utils.env_report.reporter import EnvReporter, _log_env_report, _summarise_repo, start_env_reporting


class TestSummariseRepo:
    # [CLASS DOC] Tests for the short string describing one repository: the commit, followed for a
    # dirty tree by "-dirty-" and the first twelve characters of the uncommitted hash (or
    # "unknown" when there is no hash).
    def test_dirty_repo_summaries_are_distinguished_by_truncated_uncommitted_hash(self) -> None:
        # [FUNCTION DOC] Two dirty repositories at the same commit with different 64-character
        # hashes summarize as the commit plus the first twelve characters of each hash.
        """Dirty trees at one commit must be distinguished by exactly twelve hash characters."""
        first = _repo_info(commit="abc", dirty=True, uncommitted_hash="1" * 64)
        second = _repo_info(commit="abc", dirty=True, uncommitted_hash="2" * 64)

        assert _summarise_repo(first) == "abc-dirty-111111111111"
        assert _summarise_repo(second) == "abc-dirty-222222222222"

    def test_repo_summary_distinguishes_clean_state_from_dirty_state_with_unknown_hash(self) -> None:
        # [FUNCTION DOC] A clean repository summarizes as just the commit; a dirty one without
        # a hash summarizes as the commit with "-dirty-unknown".
        """A clean commit stays verbatim while a dirty tree without a hash is explicitly unknown."""
        clean = _repo_info(commit="abc", dirty=False, uncommitted_hash=None)
        dirty = _repo_info(commit="abc", dirty=True, uncommitted_hash=None)

        assert _summarise_repo(clean) == "abc"
        assert _summarise_repo(dirty) == "abc-dirty-unknown"


def _repo_info(*, commit: str, dirty: bool, uncommitted_hash: str | None) -> EnvReportGitRepoInfo:
    # [FUNCTION DOC] HELPER: Builds a git repository record for package "miles" at location
    # "/repo" with the given commit, dirty flag and uncommitted hash, an empty diff stat, no
    # untracked paths, not truncated, and no unhashed paths. Returns it.
    return EnvReportGitRepoInfo(
        package_name="miles",
        location="/repo",
        commit=commit,
        dirty=dirty,
        diff_stat="",
        uncommitted_hash=uncommitted_hash,
        untracked_paths=[],
        untracked_paths_truncated=False,
        untracked_unhashed_paths=[],
    )


class TestLogEnvReport:
    # [CLASS DOC] Tests for the function that records a collected report: as an event through the
    # global event logger when one is configured, and always as a compact summary log line at info
    # level (not the full report).
    def _log(self, **overrides) -> None:
        # [FUNCTION DOC] HELPER: Collects a report from a snapshot of an argument namespace built
        # from the overrides and passes it to the logging function under test.
        _log_env_report(report=collect_env_report(snapshot=collect_env_report_snapshot(make_args(**overrides))))

    def test_writes_one_event_the_analyzer_can_read_back(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        # [FUNCTION DOC] Logging one report leaves exactly one event in the log directory; it is
        # an environment report event whose source is the main-component process identity, whose
        # dumped arguments include the learning rate 1.0 and whose host name is non-empty.
        """The report is stored as a normal event, so replaying a run's jsonl recovers its environment."""
        self._log(lr=1.0)

        events = read_events(event_log_dir)
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, EnvReportEvent)
        assert event.source == SimpleProcessIdentity(component="main")
        assert event.report.process.args.values["lr"] == 1.0
        assert event.report.process.hostname

    def test_summarises_the_report_on_stdout_instead_of_dumping_it(
        self, mocked_pip_inspect, event_log_dir: Path, caplog
    ) -> None:
        # [FUNCTION DOC] At info level the reporter logger output contains "op=env_report" and
        # "num_packages=4" (the sample package count) but not an environment variable name that
        # would only appear in a full dump, "PYTHONUNBUFFERED".
        """A full report is tens of kilobytes; logging it per process would drown the logs."""
        with caplog.at_level(logging.INFO, logger="miles.utils.env_report.reporter"):
            self._log(lr=1.0)

        assert "op=env_report" in caplog.text
        assert "num_packages=4" in caplog.text
        assert "PYTHONUNBUFFERED" not in caplog.text

    def test_summarises_the_report_when_no_event_logger_is_configured(
        self, mocked_pip_inspect, without_event_logger, caplog
    ) -> None:
        # [FUNCTION DOC] With no global event logger the summary line is still logged and
        # contains "stored=false".
        """A run without an event dir still leaves a trace instead of silently dropping the report."""
        with caplog.at_level(logging.INFO, logger="miles.utils.env_report.reporter"):
            self._log()

        assert "op=env_report" in caplog.text
        assert "stored=false" in caplog.text


class TestTheStartupReport:
    # [CLASS DOC] Tests of the startup entry point, which stores a first report synchronously
    # without probing packages (so the record exists even if the process dies early) and then
    # starts the background reporter. Failures to start are logged as warnings and never raised.
    def test_a_thread_start_failure_only_warns_and_returns_none(self, mocked_pip_inspect, caplog) -> None:
        # [FUNCTION DOC] With thread starting patched to raise a runtime error, the entry point
        # returns none, logs a warning containing "Failed to start the env reporting", and the
        # last log record has exception information.
        """A thread start failure must warn with its traceback without disrupting process startup."""
        with patch.object(threading.Thread, "start", side_effect=RuntimeError("boom")):
            with caplog.at_level(logging.WARNING, logger="miles.utils.env_report.reporter"):
                reporter = start_env_reporting(make_args())

        assert reporter is None
        assert "Failed to start the env reporting" in caplog.text
        assert caplog.records[-1].exc_info is not None

    def test_is_stored_before_the_caller_gets_its_reporter_back(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        # [FUNCTION DOC] The full report collection is patched to block forever (until an event
        # is set), so only the synchronous startup report can exist. After the entry point
        # returns, the log holds exactly one event with the packages-probed flag false and the
        # dumped learning rate 1.0. Releases the blocked thread at the end.
        """A process that dies in its first seconds is exactly the one whose startup has to be on record."""
        never_finish = threading.Event()

        with patch("miles.utils.env_report.reporter.collect_env_report", side_effect=lambda **_: never_finish.wait()):
            start_env_reporting(make_args(lr=1.0))

            events = read_events(event_log_dir)
            never_finish.set()

        assert [event.report.packages_probed for event in events] == [False]
        assert events[0].report.process.args.values["lr"] == 1.0

    def test_names_no_package_it_has_not_probed(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        # [FUNCTION DOC] Same blocking setup; the single stored startup report has an empty
        # package list, an empty git repository list and the probed flag false.
        """An empty pip list that claims to be probed would read as a machine with nothing installed."""
        never_finish = threading.Event()

        with patch("miles.utils.env_report.reporter.collect_env_report", side_effect=lambda **_: never_finish.wait()):
            start_env_reporting(make_args())
            report = read_events(event_log_dir)[0].report
            never_finish.set()

        assert report.full_pip_list == []
        assert report.git_repos == []
        assert report.packages_probed is False


class TestEnvReporter:
    # [CLASS DOC] Tests of the background reporter object (arguments, interval in seconds):
    # runs collection in a thread, waits a settling delay (capped by the interval) before the
    # first full report, then reports periodically with jitter, can be rebound to final
    # arguments, and can be stopped with a bounded wait.
    def _reporter(self, *, interval_seconds: float, **overrides) -> EnvReporter:
        # [FUNCTION DOC] HELPER: Creates a reporter (not started) with arguments built from the
        # overrides and the given interval.
        return EnvReporter(args=make_args(**overrides), interval_seconds=interval_seconds)

    def test_reports_from_a_background_thread(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        # [FUNCTION DOC] Patches report collection with a function that signals it was entered
        # and then waits for a release signal. Starts the reporter through the entry point and
        # asserts collection begins (within 30 seconds) while the caller is not blocked, then
        # releases and stops the reporter. Nothing else is asserted.
        """Collection shells out, so it must not sit on the caller's startup path."""
        collecting = threading.Event()
        release = threading.Event()

        def block(**kwargs):
            # [FUNCTION DOC] HELPER: Fake collection: signals that collecting started, then waits
            # up to 30 seconds for the release signal. Returns nothing.
            collecting.set()
            release.wait(timeout=30.0)

        with patch("miles.utils.env_report.reporter.collect_env_report", side_effect=block):
            reporter = start_env_reporting(make_args())
            assert collecting.wait(timeout=30.0)
            release.set()
            reporter.stop()

    def test_stop_returns_when_an_environment_probe_is_blocked(self, mocked_pip_inspect) -> None:
        # [FUNCTION DOC] Replaces the pip subprocess with a function that blocks until released.
        # With the stop timeout patched to 0.01 seconds, starts the reporter, waits until the
        # probe is running, then calls stop on a separate daemon thread and asserts the stop
        # completes within 1 second even though the probe is still blocked. In a final cleanup
        # releases the probe, joins the stopper thread and stops again.
        """Stopping uses its bounded join even when an active environment probe cannot finish."""
        collecting = threading.Event()
        release = threading.Event()
        stop_completed = threading.Event()

        def block(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            # [FUNCTION DOC] HELPER: Fake subprocess run: signals it started, waits up to 30
            # seconds for release, then returns the standard fake pip inspect result.
            collecting.set()
            release.wait(timeout=30.0)
            return mock_pip_inspect()

        def stop_reporter() -> None:
            # [FUNCTION DOC] HELPER: Stops the reporter of the enclosing test and signals that
            # the stop call returned.
            reporter.stop()
            stop_completed.set()

        with patch("miles.utils.env_report.collector.subprocess.run", side_effect=block):
            reporter = self._reporter(interval_seconds=60.0)
            stopper: threading.Thread | None = None
            try:
                with patch("miles.utils.env_report.reporter._STOP_TIMEOUT_SECONDS", 0.01):
                    reporter.start()
                    assert collecting.wait(timeout=30.0)
                    stopper = threading.Thread(target=stop_reporter, daemon=True)
                    stopper.start()

                    assert stop_completed.wait(timeout=1.0)
            finally:
                release.set()
                if stopper is not None:
                    stopper.join(timeout=30.0)
                reporter.stop()

    def test_re_reads_the_environment_the_process_settled_on(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        # [FUNCTION DOC] Creates a reporter with learning rate 1.0 and a 0.01 second interval,
        # then mutates the same arguments object to 2.0 before starting; with the settling delay
        # patched to 0. Waits for three stored events, stops, and asserts the first event holds
        # 1.0 (the startup snapshot taken at construction) and the last holds 2.0 (later reports
        # re-read the arguments).
        """A rank writes MASTER_ADDR after its logger is configured, and that is what it actually ran with."""
        args = make_args(lr=1.0)
        reporter = EnvReporter(args=args, interval_seconds=0.01)
        args.lr = 2.0

        with patch("miles.utils.env_report.reporter.SETTLED_DELAY_SECONDS", 0.0):
            reporter.start()
            _wait_for_events(event_log_dir, count=3)
            reporter.stop()

        recorded = [event.report.process.args.values["lr"] for event in read_events(event_log_dir)]
        assert recorded[0] == 1.0
        assert recorded[-1] == 2.0

    def test_a_rebind_moves_the_periodic_report_onto_the_final_args(
        self, mocked_pip_inspect, event_log_dir: Path
    ) -> None:
        # [FUNCTION DOC] Creates a reporter with learning rate 1.0, rebinds it to a new
        # arguments object with 2.0, starts it (settling delay 0, interval 0.01 seconds), waits
        # for three events and stops. Asserts the first event has 1.0 (snapshot at creation)
        # while the second and last have 2.0.
        """A trainer is constructed with the launch args and only receives the final ones in init()."""
        reporter = EnvReporter(args=make_args(lr=1.0), interval_seconds=0.01)
        reporter.rebind(make_args(lr=2.0))

        with patch("miles.utils.env_report.reporter.SETTLED_DELAY_SECONDS", 0.0):
            reporter.start()
            _wait_for_events(event_log_dir, count=3)
            reporter.stop()

        recorded = [event.report.process.args.values["lr"] for event in read_events(event_log_dir)]
        assert recorded[0] == 1.0
        assert recorded[1] == 2.0
        assert recorded[-1] == 2.0

    def test_settles_before_it_falls_back_to_the_interval(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        # [FUNCTION DOC] With a one hour interval, replaces the reporter's stop-event wait with a
        # function that records the requested delay and reports the event as set (so the loop
        # ends immediately). Runs the loop body directly and asserts the only delay requested
        # was 300 seconds, the settling delay.
        """One hour is far too long to wait for the environment a process only finishes building at startup."""
        reporter = self._reporter(interval_seconds=3600.0)
        delays: list[float] = []

        with patch.object(reporter._stopped, "wait", side_effect=lambda delay: delays.append(delay) or True):
            reporter._run()

        assert delays == [300.0]

    def test_an_interval_shorter_than_the_settling_delay_wins(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        # [FUNCTION DOC] Same setup with a 60 second interval: the only requested first delay is
        # 60 seconds, i.e. the smaller of the interval and the settling delay.
        """Asking for a report every minute must not be answered with a five minute wait."""
        reporter = self._reporter(interval_seconds=60.0)
        delays: list[float] = []

        with patch.object(reporter._stopped, "wait", side_effect=lambda delay: delays.append(delay) or True):
            reporter._run()

        assert delays == [60.0]

    def test_the_jitter_does_not_disturb_the_process_random_number_generator(self, mocked_pip_inspect) -> None:
        # [FUNCTION DOC] Seeds the global random generator, draws one number as the expected
        # value, reseeds, calls the reporter's next-delay computation, and asserts the next
        # draw equals the expected value, i.e. jitter uses a private generator.
        """Megatron seeds the global RNG for reproducibility, and this thread runs alongside it."""
        reporter = self._reporter(interval_seconds=60.0)
        random.seed(1234)
        expected = random.random()

        random.seed(1234)
        reporter._next_delay_seconds()

        assert random.random() == expected

    def test_an_interval_longer_than_a_wait_can_express_still_waits(self, mocked_pip_inspect) -> None:
        # [FUNCTION DOC] With an interval equal to the maximum timeout a thread wait supports,
        # the computed next delay (including jitter) is clamped to at most that maximum.
        """A wait longer than TIMEOUT_MAX raises, which would end the process's only reporter."""
        reporter = self._reporter(interval_seconds=threading.TIMEOUT_MAX)

        assert reporter._next_delay_seconds() <= threading.TIMEOUT_MAX

    @pytest.mark.parametrize("interval_seconds", [0.0, -1.0])
    def test_a_non_positive_interval_reports_only_at_startup(
        self, mocked_pip_inspect, event_log_dir: Path, interval_seconds: float
    ) -> None:
        # [FUNCTION DOC] Parametrized over interval 0 and -1: after starting, the reporter
        # thread ends by itself within 30 seconds, and exactly two events were stored: the
        # unprobed startup one, then one probed report (flags false then true).
        """The reporter must end by itself, rather than look finished because it was stopped."""
        reporter = self._reporter(interval_seconds=interval_seconds)

        reporter.start()
        reporter._thread.join(timeout=30.0)

        assert not reporter._thread.is_alive()
        assert [event.report.packages_probed for event in read_events(event_log_dir)] == [False, True]

    @pytest.mark.parametrize("interval_seconds", [float("nan"), float("inf")])
    def test_a_non_finite_interval_is_refused(self, interval_seconds: float) -> None:
        # [FUNCTION DOC] Parametrized over not-a-number and infinity: constructing the reporter
        # raises an assertion error.
        """nan never waits and inf overflows the wait, so both silently break the reporter."""
        with pytest.raises(AssertionError):
            self._reporter(interval_seconds=interval_seconds)

    def test_a_failing_report_does_not_stop_the_reporter(
        self, mocked_pip_inspect, event_log_dir: Path, caplog
    ) -> None:
        # [FUNCTION DOC] Patches report collection with a function that raises a runtime error
        # the first time and signals success on later calls. Starts a reporter with a 0.01
        # second interval, waits for the success signal (30 seconds max), stops it, and asserts a
        # warning containing "Failed to log the env report" was logged.
        """A broken environment probe must never take the process, or the next report, down with it."""
        succeeded = threading.Event()
        outcomes = iter([RuntimeError("boom")])

        def fail_once(**kwargs):
            # [FUNCTION DOC] HELPER: Fake collection: raises the next queued error; when the
            # queue is exhausted, signals success instead and returns nothing.
            try:
                raise next(outcomes)
            except StopIteration:
                succeeded.set()

        reporter = self._reporter(interval_seconds=0.01)
        with patch("miles.utils.env_report.reporter.collect_env_report", side_effect=fail_once):
            with caplog.at_level(logging.WARNING, logger="miles.utils.env_report.reporter"):
                reporter.start()
                assert succeeded.wait(timeout=30.0)
                reporter.stop()

        assert "Failed to log the env report" in caplog.text

    def test_a_failing_snapshot_only_warns(self, event_log_dir: Path, caplog) -> None:
        # [FUNCTION DOC] Patches the snapshot function to raise. The entry point must still
        # return a reporter (which is then stopped), a warning containing "Failed to log the
        # startup env report" is logged, and no event is stored.
        """The snapshot runs on the caller's own thread, so an audit failure must not end its startup."""
        with patch("miles.utils.env_report.reporter.collect_env_report_snapshot", side_effect=RuntimeError("boom")):
            with caplog.at_level(logging.WARNING, logger="miles.utils.env_report.reporter"):
                reporter = start_env_reporting(make_args())
                reporter.stop()

        assert "Failed to log the startup env report" in caplog.text
        assert read_events(event_log_dir) == []

    def test_a_failing_startup_report_still_starts_the_periodic_ones(
        self, mocked_pip_inspect, event_log_dir: Path, caplog
    ) -> None:
        # [FUNCTION DOC] Patches the unprobed startup collection to raise, with a 0.01 second
        # interval. Asserts the warning "Failed to log the startup env report" is logged, and
        # that at least one event (from the periodic path) still gets stored before stop.
        """The startup write is one attempt, and losing it must not cost the process every later report."""
        with patch("miles.utils.env_report.reporter.collect_unprobed_env_report", side_effect=RuntimeError("boom")):
            with caplog.at_level(logging.WARNING, logger="miles.utils.env_report.reporter"):
                reporter = start_env_reporting(make_args(env_report_interval_seconds=0.01))
                _wait_for_events(event_log_dir, count=1)
                reporter.stop()

        assert "Failed to log the startup env report" in caplog.text


def _wait_for_events(log_dir: Path, *, count: int) -> None:
    # [FUNCTION DOC] HELPER: Polls the event log directory up to 3000 times, sleeping 0.01
    # seconds between polls (about 30 seconds), returning as soon as at least the given number of
    # events can be read. Raises an assertion error stating how many events were found if the
    # count is never reached.
    deadline = threading.Event()
    for _ in range(3000):
        if len(read_events(log_dir)) >= count:
            return
        deadline.wait(0.01)
    raise AssertionError(f"only {len(read_events(log_dir))} events were written, expected {count}")
