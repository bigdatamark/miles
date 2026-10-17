# [MODULE DOC] HELPER:
# Helpers for launching a subprocess bound to the lifetime of the current process, and for
# terminating or killing a subprocess's whole process group.
import atexit
import logging
import os
import signal
import subprocess
import sys

logger = logging.getLogger(__name__)

_TRAMPOLINE_MODULE = "miles.utils.workers.process_trampoline"


def launch_bound_subprocess(argv: list[str], *, envs: dict[str, str]) -> subprocess.Popen:
    # [FUNCTION DOC]
    # Start the command in a new session (its own process group) via the trampoline module, which
    # makes the child die if this process dies, passing this process's pid. The child environment
    # is the current environment overlaid with the given variables (the given ones win). Registers
    # an exit hook that terminates the child's process group at interpreter exit (this keeps a
    # reference to the process for the rest of the program). Returns the process object.
    process = subprocess.Popen(
        [sys.executable, "-m", _TRAMPOLINE_MODULE, str(os.getpid()), *argv],
        env={**os.environ, **envs},
        start_new_session=True,
    )
    atexit.register(terminate_process_tree, process)
    return process


def terminate_process_tree(process: subprocess.Popen, *, sigkill_timeout: float = 5.0) -> None:
    # [FUNCTION DOC]
    # Stop a process and its process group: send the terminate signal to the group, wait up to the
    # timeout (default 5 seconds) for the process itself to exit, logging a warning if it did not,
    # then always send the kill signal to the group (to catch lingering descendants) and wait for
    # the process to be reaped. A group that no longer exists is ignored.
    _signal_process_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=sigkill_timeout)
    except subprocess.TimeoutExpired:
        logger.warning(
            "Process %d did not exit within %.1fs after SIGTERM; escalating to SIGKILL", process.pid, sigkill_timeout
        )
    _signal_process_group(process.pid, signal.SIGKILL)
    process.wait()


def kill_process_tree(process: subprocess.Popen) -> None:
    # [FUNCTION DOC]
    # Send the kill signal to the process group whose id equals the process's pid. Does not wait
    # and does not reap; a missing group is ignored.
    _signal_process_group(process.pid, signal.SIGKILL)


def kill_process(process: subprocess.Popen) -> None:
    # [FUNCTION DOC]
    # Send the kill signal to only the process itself (not its group); a process that is already
    # gone is ignored. Does not wait.
    try:
        os.kill(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _signal_process_group(process_group_id: int, signal_number: int) -> None:
    # [FUNCTION DOC] HELPER: Send a signal to a process group, ignoring a group that does not exist.
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        pass
