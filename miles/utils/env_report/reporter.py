# [MODULE DOC] Background reporting of the process environment. An EnvReporter logs a cheap
# (unprobed) report immediately on construction and rebind, and runs a daemon thread that logs
# full reports: one right at start, then after min(300 seconds, interval), then at the interval
# with up to 50 percent random extra delay; a non-positive interval means a single full report.
# Reports go to the audit event logger when initialized and always to a structured info log
# summary (versions, per-repo commit with a dirty marker and short hash, counts). Failures are
# logged as warnings and never propagate.
import logging
import math
import random
import threading
from typing import Any

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import EnvReport, EnvReportEvent, EnvReportGitRepoInfo
from miles.utils.env_report.collector import (
    collect_env_report,
    collect_env_report_snapshot,
    collect_unprobed_env_report,
)
from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)
_REPORTER_THREAD_NAME = "env-report"
_INTERVAL_JITTER_RATIO = 0.5
_STOP_TIMEOUT_SECONDS = 5.0
_SUMMARY_HASH_CHARS = 12
SETTLED_DELAY_SECONDS = 300.0


def start_env_reporting(args: Any) -> "EnvReporter | None":
    # [FUNCTION DOC] Creates a reporter with the interval from the arguments' env report interval
    # option and starts its thread; returns it, or none (after a warning) if anything fails,
    # including an invalid interval.
    try:
        reporter = EnvReporter(args=args, interval_seconds=args.env_report_interval_seconds)
        reporter.start()
        return reporter
    except Exception:
        logger.warning("Failed to start the env reporting of this process", exc_info=True)
        return None


class EnvReporter:
    # [CLASS DOC] Owns the reporting thread (daemon, named env-report), a stop event, a private
    # random generator for jitter, the interval and the current arguments object. Lifecycle:
    # construct (logs the startup report), start, optionally rebind to new arguments, stop.
    def __init__(self, *, args: Any, interval_seconds: float) -> None:
        # [FUNCTION DOC] Keyword-only. Asserts the interval is a finite number (zero or negative
        # means report only once; infinity and not-a-number are rejected). Creates the unstarted
        # thread, then rebinds to the arguments, which logs the startup report.
        assert math.isfinite(interval_seconds), (
            f"--env-report-interval-seconds is {interval_seconds}, which is neither a delay nor a way to say "
            f"'only at startup'; pass a finite number"
        )

        self._interval_seconds = interval_seconds
        self._jitter = random.Random()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name=_REPORTER_THREAD_NAME, daemon=True)
        self.rebind(args)

    def rebind(self, args: Any) -> None:
        # [FUNCTION DOC] Replaces the arguments used for later reports and immediately logs an
        # unprobed report built from a fresh snapshot of the new arguments; failures are logged as
        # warnings.
        self._args = args
        try:
            _log_env_report(report=collect_unprobed_env_report(snapshot=collect_env_report_snapshot(args)))
        except Exception:
            logger.warning("Failed to log the startup env report", exc_info=True)

    def start(self) -> None:
        # [FUNCTION DOC] Starts the reporting thread (raises if started twice).
        self._thread.start()

    def stop(self) -> None:
        # [FUNCTION DOC] Signals the thread to stop and waits up to 5 seconds for it to finish
        # (does not interrupt a report already in progress).
        self._stopped.set()
        self._thread.join(timeout=_STOP_TIMEOUT_SECONDS)

    def _run(self) -> None:
        # [FUNCTION DOC] HELPER: Thread body. First delay = min(300 seconds, interval) when the
        # interval is positive, else zero. Loop: log a full report; return if the interval is not
        # positive or the stop event is set during the wait for the delay; then compute the next
        # delay with jitter.
        delay = min(SETTLED_DELAY_SECONDS, self._interval_seconds) if self._interval_seconds > 0 else 0.0
        while True:
            self._report_once()
            if self._interval_seconds <= 0 or self._stopped.wait(delay):
                return
            delay = self._next_delay_seconds()

    def _report_once(self) -> None:
        # [FUNCTION DOC] HELPER: Takes a fresh snapshot of the current arguments, builds a full
        # (probing) report and logs it; any exception is logged as a warning.
        try:
            snapshot = collect_env_report_snapshot(self._args)
            _log_env_report(report=collect_env_report(snapshot=snapshot))
        except Exception:
            logger.warning("Failed to log the env report", exc_info=True)

    def _next_delay_seconds(self) -> float:
        # [FUNCTION DOC] HELPER: The interval multiplied by a random factor between 1.0 and 1.5,
        # capped at the maximum timeout the threading library accepts.
        jittered = self._interval_seconds * (1.0 + self._jitter.random() * _INTERVAL_JITTER_RATIO)
        return min(jittered, threading.TIMEOUT_MAX)


def _log_env_report(*, report: EnvReport) -> EnvReport:
    # [FUNCTION DOC] HELPER: If the audit event logger is initialized, records an env-report event
    # holding the report (without printing it); always logs the summary line. Returns the report.
    if is_event_logger_initialized():
        get_event_logger().log(EnvReportEvent, {"report": report}, print_log=False)
    _log_report_summary(report)

    return report


def _log_report_summary(report: EnvReport) -> None:
    # [FUNCTION DOC] HELPER: Logs one structured info record (tag audit, operation
    # env_report_summary) with host name, key versions, a mapping of package name to a short repo
    # summary, package count, environment-variable count, the probed flag, and whether the event
    # logger stored the report.
    log_structured(
        logger.info,
        tag="audit",
        op="env_report_summary",
        hostname=report.process.hostname,
        versions=report.key_versions,
        repos={repo.package_name: _summarise_repo(repo) for repo in report.git_repos},
        num_packages=len(report.full_pip_list),
        num_env_vars=len(report.process.env_vars),
        packages_probed=report.packages_probed,
        stored=is_event_logger_initialized(),
    )


def _summarise_repo(repo: EnvReportGitRepoInfo) -> str:
    # [FUNCTION DOC] HELPER: The commit hash for a clean repository; for a dirty one, the commit
    # plus "-dirty-" and the first 12 characters of the uncommitted-content hash, or the word
    # unknown when that hash is missing.
    if not repo.dirty:
        return repo.commit
    return f"{repo.commit}-dirty-{repo.uncommitted_hash[:_SUMMARY_HASH_CHARS] if repo.uncommitted_hash else 'unknown'}"
