# [MODULE DOC]
# A small actor that runs one shell command as a supervised subprocess, for workers whose real
# work is an external command rather than Python code. The actor's own process mirrors the
# subprocess: when the subprocess exits on its own, the actor process exits with the same status.
import logging
import os
import subprocess
import threading

from miles.utils.misc import NodeProbeMixin
from miles.utils.test_utils import fault_injector
from miles.utils.workers import process_utils

logger = logging.getLogger(__name__)


class CommandActor(NodeProbeMixin):
    # [CLASS DOC]
    # Runs at most one command in a subprocess (started once) and exposes shutdown, kill and
    # fault-injection controls. Inherits node-probing behavior from the node probe mixin. The
    # shutting-down flag distinguishes an intended exit from a crash of the subprocess.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initialize with no subprocess and the shutting-down flag cleared.
        self._process: subprocess.Popen | None = None
        self._shutting_down = False

    def run(self, cmd: str, envs: dict[str, str]) -> None:
        # [FUNCTION DOC]
        # Start the command given as a string through the system shell, with the given extra
        # environment variables, via the process utility that launches a subprocess bound to this
        # process's lifetime. Asserts it has not been called before. Logs the command and only the
        # sorted names of the environment variables (not values). Then starts a daemon thread that
        # waits on the subprocess. Returns immediately.
        assert self._process is None, "CommandActor.run can only be called once"

        logger.info(f"CommandActor launches subprocess cmd={cmd!r} env_names={sorted(envs)}")
        self._process = process_utils.launch_bound_subprocess(["/bin/sh", "-c", cmd], envs=envs)

        threading.Thread(target=self._babysit, args=(self._process,), daemon=True).start()

    def shutdown(self) -> None:
        # [FUNCTION DOC]
        # Graceful stop: does nothing if no subprocess was started. Otherwise marks the actor as
        # shutting down (so the watcher thread will not kill this process) and terminates the
        # subprocess and its descendants through the process utility.
        if self._process is None:
            return

        self._shutting_down = True
        process_utils.terminate_process_tree(self._process)

    def kill_subprocess(self) -> None:
        # [FUNCTION DOC]
        # Forcefully kill the subprocess and its descendants. Asserts a subprocess exists. Does not
        # set the shutting-down flag, so the watcher thread will then terminate this process with
        # the subprocess's status.
        assert self._process is not None, "CommandActor has no subprocess to kill"
        process_utils.kill_process_tree(self._process)

    def inject_fault(self, mode: str) -> None:
        # [FUNCTION DOC]
        # Fault-injection entry point taking the failure mode as a string. Asserts a subprocess
        # exists and that the mode is the hard-kill mode (other modes are faults a process inflicts
        # on itself and cannot be reproduced from outside; the assertion message says so; an
        # unknown mode string raises a value error from the enum conversion). Logs a warning and
        # kills the subprocess group, which the watcher then turns into a process exit.
        assert self._process is not None, "CommandActor has no subprocess to inject a fault into"
        assert (failure_mode := fault_injector.FailureMode(mode)) is fault_injector.FailureMode.SIGKILL, (
            f"{failure_mode.value} is a fault a process inflicts on itself from the inside, and no signal reproduces "
            f"it from the outside, so only sigkill can be injected into a subprocess"
        )

        logger.warning(f"CommandActor kills its subprocess group pid={self._process.pid}")
        process_utils.kill_process_tree(self._process)

    def _babysit(self, process: subprocess.Popen) -> None:
        # [FUNCTION DOC] HELPER: Body of the watcher thread. Blocks until the subprocess exits. If
        # a shutdown was requested, just logs and returns. Otherwise logs and immediately exits the
        # whole process (skipping cleanup handlers) with the subprocess's exit status; a negative
        # status meaning death by signal is converted to 128 plus the signal number, as a shell
        # reports it.
        returncode = process.wait()

        if self._shutting_down:
            logger.info(f"CommandActor subprocess exited with returncode={returncode} during shutdown")
            return

        logger.info(f"CommandActor exits since its subprocess exited with returncode={returncode}")
        # Popen reports a signal death as -signal; exit 128+signal, as a shell reports it.
        os._exit(128 - returncode if returncode < 0 else returncode)
