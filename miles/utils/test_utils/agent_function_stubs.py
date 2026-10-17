# [MODULE DOC] HELPER: Stub agent functions used by tests of the custom-agent-function mode
# (running a user agent in a separate process). Defined at module level so a fresh process can
# import them by path just like a real agent. Each is an async function taking arbitrary keyword
# arguments (extra keywords are ignored) and exercises one behavior: reporting process ids,
# printing, reporting the rollout-limit scope, failing in various ways, exiting abruptly,
# being cancelled, ignoring cancellation, leaving background threads, and holding or dying while
# holding rollout limit primitives (semaphore, lock).
"""Agent functions for the ``--custom-agent-function-mode`` tests.

Module-level so that a fresh process imports them the way it imports a real agent.
"""

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

from miles.rollout.agentic.rollout_limits import current_scope, rollout_lock, rollout_semaphore


class NeedsKeywordError(Exception):
    # [CLASS DOC] HELPER: Exception carrying a required keyword-only detail argument in addition to
    # the message. Because its constructor needs a keyword that the standard exception pickling
    # does not pass back, it can be pickled but not rebuilt on the receiving side (mimicking many
    # HTTP client errors). Stores the detail as an attribute.
    """An error that pickles but cannot be rebuilt from its args, like many HTTP client errors."""

    def __init__(self, message: str, *, detail: str):
        # [FUNCTION DOC] HELPER: Passes only the message to the base exception (so the pickled args
        # lack the detail, causing the failed rebuild) and stores the keyword detail.
        super().__init__(message)
        self.detail = detail


async def report_pid(**kwargs) -> dict:
    # [FUNCTION DOC] HELPER: Returns a mapping with this process's id and parent process id, to let
    # tests check which process the agent ran in.
    return {"pid": os.getpid(), "ppid": os.getppid()}


async def print_and_return(**kwargs) -> str:
    # [FUNCTION DOC] HELPER: Prints a fixed line to standard output (so tests can check output
    # capture) and returns a fixed string.
    print("printed by the agent")
    return "returned by the agent"


async def report_limits_scope(**kwargs) -> str:
    # [FUNCTION DOC] HELPER: Returns the current rollout-limits scope identifier string as seen
    # from inside the agent.
    return current_scope()


async def fail(**kwargs) -> None:
    # [FUNCTION DOC] HELPER: Always raises a value error with a fixed message.
    raise ValueError("agent failed on purpose")


async def fail_unrebuildably(**kwargs) -> None:
    # [FUNCTION DOC] HELPER: Always raises the keyword-requiring error type (message "agent failed
    # on purpose", detail "lost on the way") to test error transport across processes.
    raise NeedsKeywordError("agent failed on purpose", detail="lost on the way")


async def exit_without_returning(**kwargs) -> None:
    # [FUNCTION DOC] HELPER: Terminates the process immediately with exit code 3, skipping all
    # cleanup and never returning a result.
    os._exit(3)


async def call_sys_exit(**kwargs) -> None:
    # [FUNCTION DOC] HELPER: Raises the interpreter exit request with code 2 (a system-exit
    # exception, not a hard exit), so cleanup handlers can run.
    sys.exit(2)


async def wait_until_cancelled(*, pid_file: str, cleaned_up: str, **kwargs) -> None:
    # [FUNCTION DOC] HELPER: Writes this process id to the pid file path, then sleeps for an hour.
    # If cancelled, creates (touches) the cleaned-up marker file and re-raises the cancellation.
    # Lets tests check that cancellation reaches the agent and cleanup runs.
    Path(pid_file).write_text(str(os.getpid()))
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        Path(cleaned_up).touch()
        raise


async def ignore_cancellation(*, pid_file: str, **kwargs) -> None:
    # [FUNCTION DOC] HELPER: Writes this process id to the pid file path, then loops forever
    # sleeping an hour at a time, swallowing every cancellation; only a hard kill stops it. Used to
    # test escalation to forceful termination.
    Path(pid_file).write_text(str(os.getpid()))
    while True:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass


async def leave_cleanup_thread(*, marker: str, delay_s: float, **kwargs) -> int:
    # [FUNCTION DOC] HELPER: Starts a non-daemon background thread that sleeps delay_s seconds then
    # touches the marker file, and immediately returns this process id. Tests that the host process
    # waits for (or not) lingering non-daemon threads.
    def _cleanup() -> None:
        # [FUNCTION DOC] HELPER: Thread body: blocking sleep, then touch the marker file.
        time.sleep(delay_s)
        Path(marker).touch()

    threading.Thread(target=_cleanup, daemon=False).start()
    return os.getpid()


async def leave_endless_thread(**kwargs) -> int:
    # [FUNCTION DOC] HELPER: Starts a non-daemon thread that sleeps for an hour and returns this
    # process id immediately, simulating a thread that would keep the process alive indefinitely.
    threading.Thread(target=time.sleep, args=(3600,), daemon=False).start()
    return os.getpid()


async def hold_semaphore(*, name: str, limit: int, hold_s: float, **kwargs) -> tuple[float, float]:
    # [FUNCTION DOC] HELPER: Acquires the named rollout semaphore with the given limit, records the
    # monotonic start time, sleeps (asynchronously) hold_s seconds, and returns the pair of
    # monotonic start and end times while still holding it, so tests can verify mutual exclusion /
    # concurrency limits from the intervals.
    async with rollout_semaphore(name, limit):
        start = time.monotonic()
        await asyncio.sleep(hold_s)
        return start, time.monotonic()


async def hold_lock(*, name: str, hold_s: float, **kwargs) -> tuple[float, float]:
    # [FUNCTION DOC] HELPER: Acquires the named rollout lock (a synchronous context manager),
    # records the monotonic start time, blocks the thread with a synchronous sleep of hold_s
    # seconds, and returns the monotonic start and end times pair.
    with rollout_lock(name):
        start = time.monotonic()
        time.sleep(hold_s)
        return start, time.monotonic()


async def die_holding_semaphore(*, name: str, **kwargs) -> None:
    # [FUNCTION DOC] HELPER: Acquires the named rollout semaphore with limit one and then kills the
    # process immediately with exit code 1 while holding it, to test that the permit is released
    # when the holder dies.
    async with rollout_semaphore(name, 1):
        os._exit(1)
