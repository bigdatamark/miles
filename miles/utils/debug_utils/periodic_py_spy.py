# [MODULE DOC] Optional background thread that periodically prints Python stack dumps (via the
# external py-spy tool) of all Python / Ray / sglang processes on the machine, for diagnosing
# hangs. Enabled only when the environment variable MILES_DEBUG_PYSPY_DUMP_INTERVAL is a positive
# number of seconds. Output goes to standard output. Module state: a flag recording that the
# thread was started (at most one per process).
import logging
import os
import shutil
import subprocess
import threading
import time

import psutil

logger = logging.getLogger(__name__)

DUMP_INTERVAL_ENV = "MILES_DEBUG_PYSPY_DUMP_INTERVAL"
_PROCESS_KEYWORDS = ("python", "ray::", "sglang::")
_PER_PROCESS_DUMP_TIMEOUT_SECONDS = 60

_started = False


def maybe_start_periodic_pyspy_dump() -> None:
    # [FUNCTION DOC] Reads the interval (seconds, as a float; unset or empty means zero) from the
    # environment. Returns without doing anything if the interval is not positive or a dump thread
    # was already started. If the py-spy executable is not on the path, logs an error and returns
    # (not marked started). Otherwise sets the started flag, launches a daemon thread named
    # debug-pyspy-dump running the dump loop with that interval, and logs an info message. A
    # non-numeric value raises a value error.
    global _started

    interval = float(os.environ.get(DUMP_INTERVAL_ENV, "0") or "0")
    if interval <= 0 or _started:
        return

    if shutil.which("py-spy") is None:
        logger.error(f"{DUMP_INTERVAL_ENV}={interval} set but py-spy not found on PATH; skipping periodic dump")
        return

    _started = True
    thread = threading.Thread(
        target=_dump_loop,
        args=(interval,),
        daemon=True,
        name="debug-pyspy-dump",
    )
    thread.start()
    logger.info(f"Started periodic py-spy dump (interval={interval}s)")


def _dump_loop(interval: float) -> None:
    # [FUNCTION DOC] HELPER: Thread body that never returns: sleeps for the interval, then dumps
    # all target processes; any exception in an iteration is logged with its stack trace and the
    # loop continues. The first dump happens after one full interval.
    while True:
        time.sleep(interval)
        try:
            _dump_all_processes()
        except Exception:
            logger.exception("Periodic py-spy dump iteration failed")


def _dump_all_processes() -> None:
    # [FUNCTION DOC] HELPER: Collects the target processes, prints a flushed header line with the
    # current unix time and the process count, and dumps each process in turn (sequentially).
    targets = _collect_target_processes()
    print(
        f"===== [debug-pyspy] ts={int(time.time())} processes={len(targets)} =====",
        flush=True,
    )
    for pid, cmdline in targets:
        _dump_one_process(pid=pid, cmdline=cmdline)


def _collect_target_processes() -> list[tuple[int, str]]:
    # [FUNCTION DOC] HELPER: Iterates over all system processes (psutil) and returns a list of
    # (process id, command line truncated to 160 characters) for those whose space-joined command
    # line contains any of the keywords python, "ray::" or "sglang::". Processes whose command
    # line cannot be read (process errors or OS errors) are skipped; a missing command line counts
    # as empty. Includes this process itself.
    targets: list[tuple[int, str]] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except (psutil.Error, OSError):
            continue
        if any(keyword in cmdline for keyword in _PROCESS_KEYWORDS):
            targets.append((proc.info["pid"], cmdline[:160]))
    return targets


def _dump_one_process(pid: int, cmdline: str) -> None:
    # [FUNCTION DOC] HELPER: Runs the py-spy dump subprocess (through the shell, 60 second timeout,
    # failing on non-zero exit) for the process id, first with native frames enabled and, if that
    # fails or times out, again without. On success prints a flushed header with the id and command
    # line followed by the captured output, then returns. If the second attempt also fails, prints
    # a single "DUMP FAILED" line with the first 200 characters of the error output of that last
    # attempt. Failure of the first attempt is silent.
    for native_flag in ("--native", ""):
        cmd = f"py-spy dump {native_flag} --pid {pid}".strip()
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=_PER_PROCESS_DUMP_TIMEOUT_SECONDS,
                check=True,
            )
            print(f"----- [debug-pyspy] pid={pid} cmd={cmdline}\n{result.stdout}", flush=True)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            stderr = (getattr(e, "stderr", "") or "").strip()
            if native_flag == "":
                print(f"----- [debug-pyspy] pid={pid} cmd={cmdline} DUMP FAILED: {stderr[:200]}", flush=True)
