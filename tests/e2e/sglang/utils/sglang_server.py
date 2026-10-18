# [MODULE DOC] HELPER: Launcher utilities for running a standalone SGLang server as a subprocess
# in end-to-end tests. Provides a small dataclass handle (with a base URL property and a stop
# method), a start function that builds the launch command, spawns the server with output
# redirected to a per-port log file under the system temp directory, and blocks until the health
# endpoint reports ready, plus internal helpers that poll health and read the log tail. Module
# constants: default host (loopback), base port to search from (34000), startup timeout (900 s)
# and shutdown timeout (30 s).
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import requests
import torch

from miles.utils.http_utils import find_available_port

DEFAULT_HOST = "127.0.0.1"
DEFAULT_BASE_PORT = 34000
DEFAULT_STARTUP_TIMEOUT_SECS = 900.0
DEFAULT_SHUTDOWN_TIMEOUT_SECS = 30.0


@dataclass
class SGLangServer:
    # [CLASS DOC] HELPER: Handle for a running server subprocess: the process object, host, port,
    # path of the log file receiving its output, and the open log file object (private). Owns the
    # log file and closes it on stop.
    process: subprocess.Popen
    host: str
    port: int
    log_path: Path
    _log_file: IO[str]

    @property
    def base_url(self) -> str:
        # [FUNCTION DOC] HELPER: Property returning the plain-HTTP base URL from host and port.
        return f"http://{self.host}:{self.port}"

    def stop(self, timeout_secs: float = DEFAULT_SHUTDOWN_TIMEOUT_SECS) -> None:
        # [FUNCTION DOC] HELPER: Stops the server. If the process is still running, sends a
        # terminate signal and waits up to the timeout; on timeout, kills it and waits again up to
        # the timeout (a second timeout raises). Always closes the log file afterwards. Safe to call
        # for an already exited process; calling twice would try to close an already closed file,
        # which is harmless. Returns nothing.
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout_secs)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout_secs)
        self._log_file.close()


def start_sglang_server(
    *,
    model_path: str,
    host: str = DEFAULT_HOST,
    port: int | None = None,
    startup_timeout_secs: float = DEFAULT_STARTUP_TIMEOUT_SECS,
    enable_deterministic_inference: bool = True,
    extra_args: list[str] | None = None,
) -> SGLangServer:
    # [FUNCTION DOC] Starts a server and waits until healthy, returning its handle. Keyword-only
    # inputs: model path; host (default loopback); port (default: first available port searching
    # upward from 34000); startup timeout in seconds (default 900); whether to enable deterministic
    # inference (default true); extra command-line arguments appended verbatim. The log goes to a
    # file named from the port under the system temp directory, opened for writing (overwriting a
    # previous file). The command runs the current Python interpreter with the SGLang server module
    # with remote code trust, host and port. On ROCm builds with deterministic inference and no
    # attention-backend argument present, it adds the triton attention backend. The child inherits
    # the environment with unbuffered output defaulted. Output and errors both go to the log. If
    # readiness fails (early exit gives a runtime error, timeout gives a timeout error) the server
    # is stopped and the error re-raised. Side effects: subprocess, filesystem, port selection.
    # Note the port is chosen before the server binds, so a race with other processes is possible.
    if port is None:
        port = find_available_port(DEFAULT_BASE_PORT)

    log_path = Path(f"/tmp/sglang_e2e_{port}.log")
    log_file = log_path.open("w", encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--trust-remote-code",
    ]
    if enable_deterministic_inference:
        cmd.append("--enable-deterministic-inference")
    if extra_args:
        cmd.extend(extra_args)

    # Use Triton as the deterministic-inference attention backend on ROCm.
    is_rocm = torch.version.hip is not None
    has_attention_backend = any(arg.startswith("--attention-backend") for arg in cmd)

    if enable_deterministic_inference and is_rocm and not has_attention_backend:
        cmd.extend(["--attention-backend", "triton"])

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    server = SGLangServer(process=process, host=host, port=port, log_path=log_path, _log_file=log_file)

    try:
        _wait_for_ready(server, timeout_secs=startup_timeout_secs)
    except (RuntimeError, TimeoutError):
        server.stop()
        raise
    return server


def _wait_for_ready(server: SGLangServer, *, timeout_secs: float) -> None:
    # [FUNCTION DOC] HELPER: Polls once per second until the deadline. Each round: if the process
    # has exited, raises a runtime error with the exit code and the log tail; otherwise requests
    # the health endpoint with a 5 second timeout and returns on status 200, remembering the last
    # status or request error text. On deadline expiry raises a timeout error including the last
    # error and the log tail.
    deadline = time.monotonic() + timeout_secs
    last_error = ""

    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            log_tail = _read_log_tail(server.log_path)
            raise RuntimeError(
                "SGLang server exited early. " f"Exit code: {server.process.returncode}. " f"Log tail:\n{log_tail}"
            )

        try:
            response = requests.get(f"{server.base_url}/health", timeout=5)
            if response.status_code == 200:
                return
            last_error = f"status_code={response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)

        time.sleep(1.0)

    log_tail = _read_log_tail(server.log_path)
    raise TimeoutError(
        "Timed out waiting for SGLang server to become healthy. "
        f"Last error: {last_error}. "
        f"Log tail:\n{log_tail}"
    )


def _read_log_tail(path: Path, max_lines: int = 80) -> str:
    # [FUNCTION DOC] HELPER: Returns the last max_lines lines (default 80) of the log file joined
    # by newlines, or the whole content if it is shorter, or an empty string if the file does not
    # exist. Reads the file as text ignoring undecodable bytes.
    if not path.exists():
        return ""

    content = path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return content
    return "\n".join(lines[-max_lines:])
