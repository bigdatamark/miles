# [MODULE DOC]
# Root pytest configuration for the whole tests tree. Re-exports (by import, kept alive with a
# dummy assignment) the generation and rollout test fixtures so they are available everywhere;
# defines two automatic fixtures (disable the logger's background environment reporter; clear
# the legacy-rollout environment gate); a session-scoped local Ray fixture; and an optional
# stall dumper. The stall dumper is enabled only when the stall-dump-seconds environment
# variable is present and the platform has the alarm signal: it then arms a repeating real-time
# timer at the start of each test, and when the timer fires dumps all thread tracebacks and the
# asyncio task stacks of the main thread to the uncaptured standard error, and also registers a
# traceback dump on the termination signal. Hooks arm it at test log start and disarm at log
# finish. Module state: the interval (default 60 seconds), the enabled flag and the single
# dumper instance.
import asyncio
import faulthandler
import os
import signal
import sys
from types import FrameType
from typing import TextIO

import pytest

from tests.fast.fixtures.generation_fixtures import generation_env
from tests.fast.fixtures.rollout_fixtures import rollout_env

_ = rollout_env, generation_env


@pytest.fixture(autouse=True)
def no_env_reporting(monkeypatch):
    # [FUNCTION DOC] HELPER: Automatic fixture for every test: replaces the logging utility's
    # environment-reporting starter with a function that does nothing, so constructing a worker
    # does not start a thread that runs pip and git. Reverted by the monkeypatch.
    """Constructing a worker configures its logger, which in a real process starts a thread that
    shells out to pip and git; tests exercise that reporter directly instead."""
    monkeypatch.setattr("miles.utils.logging_utils.start_env_reporting", lambda args: None)


@pytest.fixture(autouse=True)
def clear_legacy_rollout_gate(monkeypatch):
    # [FUNCTION DOC] HELPER: Automatic fixture for every test: removes the legacy rollout v1
    # environment variable (if set) because its value changes which arguments the parser
    # registers. Restored after the test.
    # an ambient value changes which arguments the parser registers
    monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)


@pytest.fixture(scope="session")
def ray_local_mode():
    # [FUNCTION DOC] HELPER: Session-scoped fixture that makes sure Ray is initialized, yielding
    # nothing. If Ray is not already initialized: reinit errors are ignored, the dashboard is off
    # and driver logging is off. When no cluster address environment variable is set, it forces a
    # fresh local cluster (a bare init would auto-attach to any leaked local cluster) with 32 CPUs
    # and 8 logical GPUs (so placement-group tests with fractional GPU bundles work on GPU-less
    # runners; engines are mocked and no real GPU is used). When an address is set (CI), it joins
    # that cluster without resource arguments, since Ray rejects them when joining. Side effect:
    # starts Ray. It is deliberately not shut down at teardown because other session-scoped
    # suites may share the cluster.
    """Session-scoped Ray init. On CI ``RAY_ADDRESS`` points at an existing
    cluster, so we connect without ``num_cpus`` (Ray rejects it when joining).
    Tests that only need pure-Python helpers should not depend on this."""
    import ray

    if not ray.is_initialized():
        kwargs: dict = dict(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
        )
        if not os.environ.get("RAY_ADDRESS"):
            # address="local" forces a fresh cluster: with no address, ray.init
            # auto-connects to any leaked local cluster (via /tmp/ray), and
            # connecting with num_cpus/num_gpus set is a hard ValueError.
            kwargs["address"] = "local"
            kwargs["num_cpus"] = 32
            # Logical GPU resource so real_ray placement-group tests (engines
            # are mocked via MockSGLangEngine; no real GPU is used) can satisfy
            # their {"GPU": 0.2} bundles on GPU-less CPU CI runners.
            kwargs["num_gpus"] = 8
        ray.init(**kwargs)
    yield
    # Don't shut down — other session-scoped suites may share this cluster.


_STALL_DUMP_SECONDS = float(os.environ.get("MILES_TEST_STALL_DUMP_SECONDS", "60"))
_STALL_DUMP_ENABLED = hasattr(signal, "SIGALRM") and "MILES_TEST_STALL_DUMP_SECONDS" in os.environ


class _StallDumper:
    # [CLASS DOC] HELPER: Watchdog that reports a stalled test. Holds the pytest config, the
    # interval, the current test node id and a duplicated file descriptor of the real (uncaptured)
    # standard error. Construction installs process-wide signal handlers (alarm -> dump method;
    # termination -> traceback dump of all threads to the duplicated descriptor, chained to any
    # previous handler). Use arm at test start and disarm at test end.
    def __init__(self, *, config: pytest.Config, interval_seconds: float) -> None:
        # [FUNCTION DOC] HELPER: Store the config and interval, an empty node id, duplicate the
        # uncaptured standard error descriptor, then register the alarm signal handler and the
        # termination-signal traceback dump. Side effects: signal handlers, a new file descriptor.
        self._config = config
        self._interval_seconds = interval_seconds
        self._nodeid: str = ""
        self._uncaptured_stderr_fd: int = _duplicate_uncaptured_stderr(config)

        signal.signal(signal.SIGALRM, self._dump)
        faulthandler.register(signal.SIGTERM, file=self._uncaptured_stderr_fd, all_threads=True, chain=True)

    def arm(self, *, nodeid: str) -> None:
        # [FUNCTION DOC] HELPER: Remember the test node id and start the real-time interval timer
        # so the alarm fires after one interval (and then repeats via the handler).
        self._nodeid = nodeid
        signal.setitimer(signal.ITIMER_REAL, self._interval_seconds)

    def disarm(self) -> None:
        # [FUNCTION DOC] HELPER: Cancel the real-time interval timer.
        signal.setitimer(signal.ITIMER_REAL, 0)

    def _dump(self, signum: int, frame: FrameType | None) -> None:
        # [FUNCTION DOC] HELPER: Alarm handler. Temporarily suspends pytest's global output capture
        # (if the capture plugin exists) so output reaches the real standard error; prints a
        # banner naming the node id and the interval, dumps all thread tracebacks, dumps the
        # asyncio tasks, prints an end banner; always resumes capture afterwards; then re-arms the
        # timer for another interval. The signal number and frame arguments are unused.
        capture_manager = self._config.pluginmanager.getplugin("capturemanager")
        if capture_manager is not None:
            capture_manager.suspend_global_capture(in_=False)
        try:
            out = sys.stderr
            print(
                f"\n===== MILES STALL DUMP: {self._nodeid} exceeded {self._interval_seconds}s =====",
                file=out,
                flush=True,
            )
            faulthandler.dump_traceback(file=out, all_threads=True)
            self._dump_asyncio_tasks(out)
            print("===== MILES STALL DUMP END =====", file=out, flush=True)
        finally:
            if capture_manager is not None:
                capture_manager.resume_global_capture()
        signal.setitimer(signal.ITIMER_REAL, self._interval_seconds)

    def _dump_asyncio_tasks(self, out: TextIO) -> None:
        # [FUNCTION DOC] HELPER: Write to the given stream the stack of every asyncio task of the
        # running event loop in the calling (main) thread: a line with the task's representation,
        # then up to 30 stack frames as file:line in function. If there is no running loop,
        # writes "no running event loop in the main thread" and returns. Ends with a blank line.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            print("no running event loop in the main thread", file=out, flush=True)
            return
        for task in asyncio.all_tasks(loop):
            print(f"----- {task!r}", file=out)
            for frame in task.get_stack(limit=30):
                print(f"        {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}", file=out)
        print("", file=out, flush=True)


def _duplicate_uncaptured_stderr(config: pytest.Config) -> int:
    # [FUNCTION DOC] HELPER: Return a duplicate of file descriptor 2 (standard error) that points
    # at the real stream. If the capture plugin is absent, just duplicate. Otherwise suspend
    # global capture, duplicate, and always resume capture afterwards. Side effect: a new open
    # file descriptor that the caller owns.
    if (capture_manager := config.pluginmanager.getplugin("capturemanager")) is None:
        return os.dup(2)

    capture_manager.suspend_global_capture(in_=False)
    try:
        return os.dup(2)
    finally:
        capture_manager.resume_global_capture()


_stall_dumper: _StallDumper | None = None


def pytest_configure(config: pytest.Config) -> None:
    # [FUNCTION DOC] Pytest hook run at startup. If the stall dumper is enabled (environment
    # variable present and alarm signal available) creates the module-level dumper with the
    # configured interval (side effect: installs signal handlers); otherwise does nothing.
    global _stall_dumper
    if not _STALL_DUMP_ENABLED:
        return
    _stall_dumper = _StallDumper(config=config, interval_seconds=_STALL_DUMP_SECONDS)


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:
    # [FUNCTION DOC] Pytest hook at the start of each test: if a stall dumper exists, arm it for
    # this node id. The location argument is unused.
    if _stall_dumper is not None:
        _stall_dumper.arm(nodeid=nodeid)


def pytest_runtest_logfinish(nodeid: str, location: tuple[str, int | None, str]) -> None:
    # [FUNCTION DOC] Pytest hook at the end of each test: if a stall dumper exists, disarm it. The
    # node id and location arguments are unused.
    if _stall_dumper is not None:
        _stall_dumper.disarm()
