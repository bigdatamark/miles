# [MODULE DOC] HELPER: Deliberately crashes or hangs the current process, for fault-tolerance
# tests. Defines the set of failure modes and a function that executes one of them.
"""
Failure modes modeled after torchft's failure.py:
https://github.com/meta-pytorch/torchft/blob/main/examples/monarch/utils/failure.py
"""

import ctypes
import logging
import os
import signal
from enum import Enum

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    # [CLASS DOC] HELPER: Enumeration of the failure modes, valued by lowercase strings: "sigkill"
    # (kill self with the uncatchable kill signal), "exit" (immediate exit with status 1, no
    # cleanup), "segfault" (call a null function pointer to crash with a segmentation fault), and
    # "deadlock" (block the whole process in a native sleep of 600 seconds while holding the
    # interpreter lock).
    SIGKILL = "sigkill"
    EXIT = "exit"
    SEGFAULT = "segfault"
    DEADLOCK = "deadlock"


def inject_fault(mode: str) -> None:
    # [FUNCTION DOC] HELPER: Executes the named failure mode in the current process. Raises a
    # value error (from the enum lookup) for an unknown mode name. Logs a warning with the mode
    # and process id first. Effects per mode: sigkill sends the kill signal to itself; exit
    # terminates immediately with status 1 skipping cleanup; segfault invokes a null native
    # function pointer; deadlock calls the C library sleep for 600 seconds through a Python-holding
    # foreign function interface so the interpreter lock stays held (all Python threads freeze),
    # and then returns normally after ten minutes. In the destructive modes it does not return.
    failure_mode = FailureMode(mode)
    logger.warning("FaultInjector: executing %s (pid=%d)", failure_mode.value, os.getpid())

    match failure_mode:
        case FailureMode.SIGKILL:
            os.kill(os.getpid(), signal.SIGKILL)

        case FailureMode.EXIT:
            os._exit(1)

        case FailureMode.SEGFAULT:
            crash_func = ctypes.CFUNCTYPE(None)()
            crash_func()

        case FailureMode.DEADLOCK:
            libc = ctypes.PyDLL(None)
            libc.sleep.argtypes = (ctypes.c_uint,)
            libc.sleep.restype = ctypes.c_uint
            libc.sleep(600)
