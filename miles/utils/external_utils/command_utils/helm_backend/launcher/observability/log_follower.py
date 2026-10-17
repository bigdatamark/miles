# [MODULE DOC]
# Streams container logs of the pods selected by a label selector into the python logger while a
# helm launch is running. A background poller reconciles the set of running log streams against the
# current pods: one kubectl logs subprocess (with a reader thread) per container run, including the
# previous (terminated) instance of a restarted container. Every emitted line is prefixed with
# pod/container (and "(previous)" for terminated instances). Streams are never started twice for
# the same container id, and a stream whose process died while the container still runs is resumed
# from the last seen timestamp.
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import IO

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.pod_facts import (
    ContainerKey,
    ContainerRun,
    container_runs,
    selected_pods,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.polling import polling_in_background

logger = logging.getLogger(__name__)

_STOP_GRACE_SECONDS = 5.0
_ERROR_TAIL_CHARACTERS = 500
_FRACTIONAL_SECONDS = re.compile(r"(?<=:\d\d)\.(\d+)")
_SUPPORTED_FRACTION_DIGITS = 6


@contextmanager
def with_log_following(*, namespace: str, selector: str) -> Iterator[None]:
    # [FUNCTION DOC]
    # Context manager that follows logs of all pods matching the selector in the namespace for the
    # duration of the with-body. Creates a log follower and polls its reconcile step in the
    # background (every few seconds, see the polling module). On exit (normal or exceptional) the
    # polling stops first and then all log streams are stopped (terminate, then kill after a grace
    # period). Yields nothing.
    follower = _LogFollower(namespace=namespace, selector=selector)
    try:
        with polling_in_background(follower.reconcile, description="look for pods to follow"):
            yield
    finally:
        follower.stop_all_streams()


class _LogFollower:
    # [CLASS DOC] HELPER: Owns the set of active log streams for one namespace and label selector.
    # Keeps a mapping from container key to its stream, and a set of every container id that was
    # ever followed so that a finished container is not streamed again.
    def __init__(self, *, namespace: str, selector: str) -> None:
        # [FUNCTION DOC] HELPER: Stores namespace and selector; starts with no streams and no
        # followed container ids.
        self._namespace = namespace
        self._selector = selector
        self._streams: dict[ContainerKey, _LogStream] = {}
        self._followed_container_ids: set[str] = set()

    def reconcile(self) -> None:
        # [FUNCTION DOC] One polling step. Lists the selected pods and derives the current container
        # runs. Streams whose container key vanished or whose container id changed (container was
        # restarted) are stopped and removed (blocking until stopped). Then for each current run: an
        # existing stream is given the chance to resume if its process died; otherwise, if the
        # container id was never followed before, it is recorded and a new stream is started.
        # Kubectl errors propagate to the poller, which logs and retries.
        runs = container_runs(selected_pods(self._namespace, self._selector))
        for key, stream in list(self._streams.items()):
            if (run := runs.get(key)) is None or run.container_id != stream.container_id:
                self._streams.pop(key).stop()

        for key, run in runs.items():
            if stream := self._streams.get(key):
                stream.resume_if_dropped(run)
            elif run.container_id not in self._followed_container_ids:
                self._followed_container_ids.add(run.container_id)
                self._streams[key] = _LogStream(namespace=self._namespace, run=run)

    def stop_all_streams(self) -> None:
        # [FUNCTION DOC] Removes every stream and stops them in two phases: first ask all of them to
        # stop (so they terminate concurrently), then wait for each one to finish.
        stopping = []
        while self._streams:
            _, stream = self._streams.popitem()
            stream.ask_to_stop()
            stopping.append(stream)

        for stream in stopping:
            stream.wait_until_stopped()


class _LogStream:
    # [CLASS DOC] HELPER: Follows the log of a single container run through one kubectl logs
    # subprocess and a daemon reader thread that forwards lines to the logger. Remembers the last
    # RFC3339 timestamp seen so a restarted process continues from there. Streams are started
    # immediately on construction.
    def __init__(self, *, namespace: str, run: ContainerRun) -> None:
        # [FUNCTION DOC] HELPER: Records namespace and run, builds the log line prefix of the form
        # pod/container in brackets with " (previous)" for terminated instances, initializes state
        # (no timestamp, no process, no thread, not stopped) and starts the process right away.
        self._namespace = namespace
        self._run = run
        self._prefix = f"[{run.key.pod}/{run.key.container}{' (previous)' if run.key.previous else ''}]"
        self._last_timestamp: str | None = None
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._start()

    @property
    def container_id(self) -> str:
        # [FUNCTION DOC] HELPER: Property returning the id of the container run being followed.
        return self._run.container_id

    def resume_if_dropped(self, run: ContainerRun) -> None:
        # [FUNCTION DOC] HELPER: Replaces the stored run with the freshly observed one. If the stream
        # was stopped or the container is no longer running, does nothing more. Otherwise, if the
        # reader thread has ended (the kubectl process dropped), starts a new process which resumes
        # from the last seen timestamp.
        self._run = run
        if self._stopped or not run.running:
            return
        if self._thread is not None and not self._thread.is_alive():
            self._start()

    def stop(self) -> None:
        # [FUNCTION DOC] HELPER: Asks the stream to stop and blocks until process and thread ended.
        self.ask_to_stop()
        self.wait_until_stopped()

    def ask_to_stop(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the stream as stopped (suppressing drop warnings and resume)
        # and terminates the kubectl process if it is still alive. Does not wait.
        self._stopped = True
        if (process := self._process) is not None and process.poll() is None:
            process.terminate()

    def wait_until_stopped(self) -> None:
        # [FUNCTION DOC] HELPER: Waits up to the grace period (5 seconds) for the process to exit,
        # killing it on timeout, then joins the reader thread with the same timeout. A thread that
        # does not finish in time is silently abandoned (it is a daemon thread).
        if (process := self._process) is not None:
            try:
                process.wait(timeout=_STOP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_GRACE_SECONDS)

    def _start(self) -> None:
        # [FUNCTION DOC] HELPER: Spawns a kubectl logs subprocess for the container: follow mode only
        # if the container is currently running, previous-instance flag from the key, and since-time
        # set to the last seen timestamp if any. Stdout is piped; stderr goes to a temporary file
        # used for error reporting. If the process cannot be launched (operating system error), logs
        # a warning, marks the stream stopped permanently and closes the temp file. Otherwise starts
        # a daemon reader thread for the process.
        command = Kubectl.logs_command(
            namespace=self._namespace,
            target=self._run.key.pod,
            container=self._run.key.container,
            follow=self._run.running,
            previous=self._run.key.previous,
            since_time=self._last_timestamp,
        )
        errors = tempfile.TemporaryFile(mode="w+")
        try:
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True)
        except OSError as error:
            logger.warning(f"{self._prefix} could not be followed ({error})")
            self._stopped = True
            errors.close()
            return

        self._thread = threading.Thread(target=self._read_until_done, args=(self._process, errors), daemon=True)
        self._thread.start()

    def _read_until_done(self, process: subprocess.Popen, errors: IO[str]) -> None:
        # [FUNCTION DOC] HELPER: Reader thread body. Emits every stdout line (trailing newline
        # removed) until end of stream, then waits for the process. If it exited with a non-zero
        # code and the stream was not asked to stop, logs a warning containing the last 500
        # characters of stderr (stripped). Always closes the stderr temp file.
        for line in process.stdout:
            self._emit(line.rstrip("\n"))
        process.wait()

        if process.returncode and not self._stopped:
            errors.seek(0)
            logger.warning(f"{self._prefix} dropped: {errors.read()[-_ERROR_TAIL_CHARACTERS:].strip()}")
        errors.close()

    def _emit(self, line: str) -> None:
        # [FUNCTION DOC] HELPER: Logs one line at info level with the prefix. Kubectl is run with
        # timestamps, so if the first space-separated token parses as an RFC3339 timestamp it is
        # remembered as the resume point and stripped from the line; otherwise the line is logged
        # unchanged.
        timestamp, _, rest = line.partition(" ")
        if _parse_rfc3339(timestamp) is not None:
            self._last_timestamp = timestamp
            line = rest
        logger.info(f"{self._prefix} {line}")


def _parse_rfc3339(text: str) -> datetime | None:
    # [FUNCTION DOC] HELPER: Parses an RFC3339 timestamp, returning a datetime or none when invalid.
    # Normalizes the fractional seconds to exactly six digits (truncate or right-pad with zeros) and
    # replaces a trailing upper or lower case Z by a zero offset so the standard ISO parser accepts
    # it. Note the Z replacement applies to any Z in the text, not only the suffix.
    normalized = _FRACTIONAL_SECONDS.sub(_normalized_fraction, text).replace("Z", "+00:00").replace("z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _normalized_fraction(match: re.Match[str]) -> str:
    # [FUNCTION DOC] HELPER: Regex substitution callback; turns the matched fractional-seconds
    # digits into a dot followed by exactly six digits (truncated, or padded with trailing zeros).
    digits = match.group(1)[:_SUPPORTED_FRACTION_DIGITS]
    return f".{digits.ljust(_SUPPORTED_FRACTION_DIGITS, '0')}"
