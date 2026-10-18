# [MODULE DOC] Tests of the agent-function caller in its inline mode (runs in the calling process)
# and its subprocess mode (runs in a child process). Stub agent functions from the shared test
# utilities report their process ids, print, fail, exit, wait for cancellation, ignore cancellation,
# or leave background threads behind. Covered: child process isolation, output not leaking into the
# result, error propagation with the child's traceback, errors that cannot be rebuilt arriving by
# name, abnormal exit codes, system exit confinement, shared rollout limit scope, cancellation with
# cleanup, forced kill after a grace period, leftover cleanup threads and their deadline, and the
# child dying with its parent on Linux. Several tests poll the file system with real sleeps.
"""A ``subproc`` agent call runs in a child process yet behaves like the awaited coroutine."""

import asyncio
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from miles.rollout.agentic import agent_function
from miles.rollout.agentic.agent_function import call_agent_function
from miles.rollout.agentic.rollout_limits import current_scope
from miles.utils.test_utils import agent_function_stubs as stubs


def _wait_for(path: Path, timeout_s: float = 60.0) -> bool:
    # [FUNCTION DOC] HELPER: Blocking poll (every 0.1 second, real sleeping) until the path exists
    # or the timeout (seconds, default 60) passes. Returns whether the path exists at the end.
    deadline = time.monotonic() + timeout_s
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    return path.exists()


def _is_running(pid: int) -> bool:
    # [FUNCTION DOC] HELPER: Reports whether a process id refers to a live process. Sends the null
    # signal; a missing process means not running. On non-Linux platforms a successful signal means
    # running. On Linux it also reads the process status file and treats zombie and dead states as
    # not running (a killed child not yet reaped still answers the null signal); a vanished status
    # file means not running.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if not sys.platform.startswith("linux"):
        return True
    # a killed process whose new parent has not reaped it yet still answers signal 0
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except FileNotFoundError:
        return False


async def _wait_until_gone(pid: int, timeout_s: float) -> bool:
    # [FUNCTION DOC] HELPER: Asynchronously polls every 0.1 second until the process is no longer
    # running or the timeout (seconds) passes. Returns true if the process is gone.
    deadline = time.monotonic() + timeout_s
    while _is_running(pid) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    return not _is_running(pid)


def test_inline_runs_on_the_calling_process():
    # [FUNCTION DOC] Calls the pid-reporting stub inline and asserts the reported process id is
    # the test process's own.
    assert asyncio.run(call_agent_function(stubs.report_pid, mode="inline"))["pid"] == os.getpid()


def test_subproc_runs_in_a_child_process():
    # [FUNCTION DOC] Calls the pid-reporting stub in subprocess mode and asserts the reported pid
    # differs from the test's and the reported parent pid equals the test's pid.
    reported = asyncio.run(call_agent_function(stubs.report_pid, mode="subproc"))
    assert reported["pid"] != os.getpid()
    assert reported["ppid"] == os.getpid()


def test_what_the_agent_prints_does_not_reach_the_result():
    # [FUNCTION DOC] Runs a stub that prints and returns a string in subprocess mode and asserts
    # the call result equals exactly the returned string, with the printed text excluded.
    assert asyncio.run(call_agent_function(stubs.print_and_return, mode="subproc")) == "returned by the agent"


def test_subproc_raises_the_agent_error_with_its_traceback():
    # [FUNCTION DOC] Runs a stub that raises a value error in subprocess mode and asserts the same
    # error type and message is raised in the caller, with the child's traceback text (mentioning
    # the failing function) attached as the cause.
    with pytest.raises(ValueError, match="agent failed on purpose") as raised:
        asyncio.run(call_agent_function(stubs.fail, mode="subproc"))
    assert "in fail" in str(raised.value.__cause__), "the agent process's traceback should be the cause"


def test_an_error_that_cannot_be_rebuilt_arrives_by_name():
    # [FUNCTION DOC] Runs a stub raising an error type that cannot be reconstructed from its
    # message alone (its constructor needs a keyword) and asserts the caller gets a runtime error
    # whose message begins with the original type name and carries the original message.
    with pytest.raises(RuntimeError, match="NeedsKeywordError: agent failed on purpose"):
        asyncio.run(call_agent_function(stubs.fail_unrebuildably, mode="subproc"))


def test_an_agent_process_that_exits_without_returning_fails_the_call():
    # [FUNCTION DOC] Runs a stub whose process exits with code 3 without returning and asserts the
    # caller gets a runtime error mentioning that exit code.
    with pytest.raises(RuntimeError, match="exited with code 3"):
        asyncio.run(call_agent_function(stubs.exit_without_returning, mode="subproc"))


def test_an_agent_that_exits_fails_only_its_own_call():
    # [FUNCTION DOC] Runs a stub that requests interpreter exit and asserts the caller sees a
    # runtime error naming the system exit, not an actual exit of the test process.
    """``sys.exit`` in an agent must not reach the rollout process as ``SystemExit``."""
    with pytest.raises(RuntimeError, match="SystemExit"):
        asyncio.run(call_agent_function(stubs.call_sys_exit, mode="subproc"))


def test_the_agent_process_shares_the_callers_rollout_limits():
    # [FUNCTION DOC] Asserts the rollout-limits scope reported from inside a subprocess agent
    # equals the scope current in the calling process.
    assert asyncio.run(call_agent_function(stubs.report_limits_scope, mode="subproc")) == current_scope()


def test_cancelling_the_caller_runs_the_agent_cleanup(tmp_path):
    # [FUNCTION DOC] Starts a stub agent that writes its pid to a file and waits until cancelled,
    # running cleanup that writes a marker file. Once the pid file appears the caller task is
    # cancelled; asserts the cancellation propagates to the caller, the child process exits
    # within 30 seconds, and the cleanup marker exists. The whole scenario is bounded by a 120
    # second timeout.
    """A cancelled episode still gets to close what it opened, such as its sandbox."""
    pid_file, cleaned_up = tmp_path / "agent_pid", tmp_path / "cleaned_up"

    async def cancel_and_watch() -> bool:
        # [FUNCTION DOC] HELPER: Starts the cancellable agent call as a task, waits for the pid file,
        # cancels the task, expects the cancelled error from awaiting it, and returns whether the
        # agent process is gone within 30 seconds.
        call = asyncio.create_task(
            call_agent_function(
                stubs.wait_until_cancelled, mode="subproc", pid_file=str(pid_file), cleaned_up=str(cleaned_up)
            )
        )
        while not pid_file.exists():
            await asyncio.sleep(0.1)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        return await _wait_until_gone(int(pid_file.read_text()), timeout_s=30)

    assert asyncio.run(asyncio.wait_for(cancel_and_watch(), timeout=120)), "the cancelled agent never exited"
    assert cleaned_up.exists(), "the cancelled agent never ran its cleanup"


def test_an_agent_that_ignores_cancellation_is_killed_after_the_grace_period(tmp_path, monkeypatch):
    # [FUNCTION DOC] Shrinks the module's cancellation grace period to one second, starts a stub
    # that records its pid and ignores cancellation, cancels the caller, and asserts the agent
    # process is eventually killed (gone within 30 seconds), all under a 120 second bound.
    monkeypatch.setattr(agent_function, "_CANCEL_GRACE_S", 1.0)
    pid_file = tmp_path / "agent_pid"

    async def cancel_and_watch() -> bool:
        # [FUNCTION DOC] HELPER: Same flow as the sibling scenario for a cancellation-ignoring
        # agent: start the call, wait for the pid file, cancel, expect the cancelled error, and
        # return whether the agent process is gone within 30 seconds.
        call = asyncio.create_task(
            call_agent_function(stubs.ignore_cancellation, mode="subproc", pid_file=str(pid_file))
        )
        while not pid_file.exists():
            await asyncio.sleep(0.1)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        return await _wait_until_gone(int(pid_file.read_text()), timeout_s=30)

    assert asyncio.run(asyncio.wait_for(cancel_and_watch(), timeout=120)), "the agent process outlived the grace"


def test_a_cleanup_thread_finishes_after_the_call_returns(tmp_path):
    # [FUNCTION DOC] Runs a stub that returns its pid but leaves a background thread that writes a
    # marker file after a 2 second delay. Asserts the call returned before the marker existed
    # (it did not wait for the thread), then that the agent process eventually exits (within 30
    # seconds) and only after the marker was written.
    """The outcome must not wait for a thread the agent left closing a sandbox, nor may the process cut it short."""
    marker = tmp_path / "cleanup_done"

    async def call_and_watch() -> tuple[bool, bool]:
        # [FUNCTION DOC] HELPER: Calls the stub (which returns the agent pid), records whether the
        # marker file is still absent at return time, then waits up to 30 seconds for the agent
        # process to exit. Returns the pair of returned-first flag and exited flag.
        pid = await call_agent_function(stubs.leave_cleanup_thread, mode="subproc", marker=str(marker), delay_s=2.0)
        returned_first = not marker.exists()
        return returned_first, await _wait_until_gone(pid, timeout_s=30)

    returned_first, exited = asyncio.run(call_and_watch())
    assert returned_first, "the call waited for the agent's cleanup thread"
    assert exited and marker.exists(), "the agent process exited before its cleanup thread finished"


def test_leftover_cleanup_is_cut_at_its_deadline(monkeypatch):
    # [FUNCTION DOC] Shrinks the module's leftover-cleanup deadline to one second, runs a stub that
    # leaves a thread that never ends, and asserts the agent process still exits (gone within 30
    # seconds).
    monkeypatch.setattr(agent_function, "_LEFTOVER_CLEANUP_S", 1.0)

    async def call_and_watch() -> bool:
        # [FUNCTION DOC] HELPER: Calls the endless-thread stub (returns the agent pid) and returns
        # whether that process is gone within 30 seconds.
        pid = await call_agent_function(stubs.leave_endless_thread, mode="subproc")
        return await _wait_until_gone(pid, timeout_s=30)

    assert asyncio.run(call_and_watch()), "the agent process outlived its leftover-cleanup deadline"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG exists only on Linux")
def test_the_agent_process_dies_with_the_rollout_process(tmp_path):
    # [FUNCTION DOC] Linux only (skipped elsewhere because it relies on the parent-death signal).
    # Spawns a separate Python process that runs a cancellation-ignoring stub agent in subprocess
    # mode, which writes its pid to a file. Waits for the pid file, then kills and reaps the
    # rollout process (in a finally block). Asserts the agent process is gone within 10 seconds.
    pid_file = tmp_path / "agent_pid"
    rollout = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import asyncio
                from miles.rollout.agentic.agent_function import call_agent_function
                from miles.utils.test_utils import agent_function_stubs as stubs
                asyncio.run(call_agent_function(stubs.ignore_cancellation, mode="subproc", pid_file={str(pid_file)!r}))
                """
            ),
        ]
    )
    try:
        assert _wait_for(pid_file), "the agent process never started"
    finally:
        rollout.kill()
        rollout.wait()
    assert asyncio.run(_wait_until_gone(int(pid_file.read_text()), timeout_s=10)), "the agent outlived its rollout"
