# [MODULE DOC]
# Tiny launcher run as a module (arguments: the expected parent pid, then the real command) that
# binds a child process to its parent's lifetime on Linux and then replaces itself with the real
# command. It asks the kernel to kill this process when the parent dies, and checks that the
# parent did not already die before that request took effect (closing the race). Diagnostics go
# to standard error.
import ctypes
import os
import signal
import sys

_PR_SET_PDEATHSIG = 1
_EXEC_FAILURE_EXIT_CODE = 127


def main() -> None:
    # [FUNCTION DOC]
    # Read the expected parent pid from the first command-line argument and the command from the
    # rest. On Linux: request a kill signal on parent death through the process-control system
    # call, then, if the current parent pid differs from the expected one, log and exit
    # immediately with status 1 without running the command. Then log and replace this process by
    # the command, found on the executable search path (the environment is inherited). If the
    # replace fails, log the error and exit with status 127. On other platforms the binding is
    # skipped. Does not return on success. Raises on a missing or non-integer first argument; an
    # empty command would raise an index error.
    expected_parent_pid = int(sys.argv[1])
    argv = sys.argv[2:]

    if sys.platform == "linux":
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
        if (parent_pid := os.getppid()) != expected_parent_pid:
            _log(f"parent {expected_parent_pid} is gone (current parent {parent_pid}); exiting without running {argv}")
            os._exit(1)

    _log(f"bound to parent {expected_parent_pid}; exec {argv}")
    try:
        os.execvp(argv[0], argv)
    except OSError as error:
        _log(f"exec of {argv} failed: {error}")
        os._exit(_EXEC_FAILURE_EXIT_CODE)


def _log(message: str) -> None:
    # [FUNCTION DOC] HELPER: Print the message to standard error prefixed with this process's pid,
    # flushing immediately.
    print(f"[process_trampoline pid={os.getpid()}] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
