# [MODULE DOC] HELPER:
# Process and network helpers for the end-to-end worker tests: launching the serve process as a
# subprocess in its own session, waiting until it answers health checks, stopping and killing it
# by process group, checking for refused ports, and two local tcp relays: a failure-injecting
# http proxy that records requests (and can reject, drop or rewrite responses) and a plain relay
# that counts accepted connections.
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR

REPO_ROOT = Path(__file__).resolve().parents[5]
SPECS_PATH = "tests.fast.utils.workers.e2e.e2e_worker.compute_specs"
POOL_ID = "e2e-pool"
RPC_PORT_FLAG = "--rpc-port"

READY_TIMEOUT_SECONDS = 60.0
STOP_TIMEOUT_SECONDS = 15.0
KILL_TIMEOUT_SECONDS = 10.0


def reserve_port() -> int:
    # [FUNCTION DOC] HELPER: Asks the operating system for a free local tcp port by binding to
    # port zero on the loopback address, releases the socket and returns the port number. There
    # is a small window in which another process could take the port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclasses.dataclass
class ServerProcess:
    # [CLASS DOC] HELPER: Handle on a launched serve subprocess: its port, the process object and
    # the path of the file that receives its combined output.
    port: int
    process: subprocess.Popen
    log_path: Path

    @property
    def url(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the base http url on the loopback address and the port.
        return f"http://127.0.0.1:{self.port}"

    def logs(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the whole log file text (undecodable bytes replaced), or
        # an empty string if the file does not exist yet.
        return self.log_path.read_text(errors="replace") if self.log_path.exists() else ""

    def is_running(self) -> bool:
        # [FUNCTION DOC] HELPER: True while the process has not exited (polls it).
        return self.process.poll() is None

    def signal(self, signal_number: int) -> None:
        # [FUNCTION DOC] HELPER: Sends the given signal to the main server process only (not its
        # group), and only if it is still running.
        if self.is_running():
            self.process.send_signal(signal_number)

    def wait(self, timeout: float) -> int | None:
        # [FUNCTION DOC] HELPER: Waits up to the timeout (seconds) for exit; returns the exit
        # code, or nothing if the timeout expired.
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def stop(self) -> int | None:
        # [FUNCTION DOC] HELPER: Graceful shutdown. If already exited, returns its exit code.
        # Otherwise sends a terminate request, waits up to 15 seconds, and if still alive kills
        # the whole process group and waits up to 10 more seconds. Returns the exit code, or
        # nothing if the process still did not exit.
        if not self.is_running():
            return self.process.returncode

        self.process.terminate()
        exit_code = self.wait(STOP_TIMEOUT_SECONDS)
        if exit_code is None:
            self.kill()
            exit_code = self.wait(KILL_TIMEOUT_SECONDS)
        return exit_code

    def kill(self) -> None:
        # [FUNCTION DOC] HELPER: Force-kills the whole process group of the server (so worker
        # child processes die too); ignores the case where the process is already gone.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)


def spawn_server(
    *,
    state_dir: Path,
    log_path: Path,
    port: int | None = None,
    worker_argv: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    specs_path: str = SPECS_PATH,
) -> ServerProcess:
    # [FUNCTION DOC] HELPER: Launches the serve module as a subprocess in a new session (so its
    # group can be killed as one). Uses the given port or reserves a free one. The child
    # environment is a copy of the current one with the repository root prepended to the python
    # path, unbuffered output, the cell index variable set to "0", then the extra environment
    # applied last (it can override). The command passes the specs path, the pool id, then after
    # a separator the state directory, the rpc port flag with the port, and any extra worker
    # arguments. Standard output and error are written to the log file (truncated). Returns the
    # server process record without waiting for readiness.
    port = reserve_port() if port is None else port

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env[CELL_INDEX_ENV_VAR] = "0"
    env.update(extra_env or {})

    argv = [sys.executable, "-m", "miles.utils.workers.serving.serve"]
    argv += ["--specs", specs_path, "--pool-id", POOL_ID]
    argv += ["--", "--state-dir", str(state_dir), RPC_PORT_FLAG, str(port)]
    argv += worker_argv or []

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            argv, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )

    return ServerProcess(port=port, process=process, log_path=log_path)


def wait_until_serving(server: ServerProcess, timeout: float = READY_TIMEOUT_SECONDS) -> None:
    # [FUNCTION DOC] HELPER: Polls every 50 ms until the health endpoint answers with status 200
    # (2 s per request, environment proxies ignored; transport errors are retried). Fails an
    # assertion with the log if the process exits first, and raises an assertion error with the
    # log if the timeout (default 60 s) passes.
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        exit_code = server.process.poll()
        assert exit_code is None, f"server exited with {exit_code} before serving:\n{server.logs()}"
        with contextlib.suppress(httpx.TransportError):
            if httpx.get(f"{server.url}/v1/health", timeout=2.0, trust_env=False).status_code == 200:
                return
        time.sleep(0.05)

    raise AssertionError(f"server never became ready within {timeout}s:\n{server.logs()}")


def port_is_refused(port: int) -> bool:
    # [FUNCTION DOC] HELPER: True if a connection attempt to the loopback port (1 s timeout)
    # fails for any reason (a timeout also counts as not connectable).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) != 0


@dataclasses.dataclass
class ProxyRequest:
    # [CLASS DOC] HELPER: One request seen by the proxy: monotonic arrival time, http verb, path
    # without query string, and the raw request bytes (head plus body).
    at: float
    verb: str
    path: str
    body: bytes


class FlakyProxy:
    # [CLASS DOC] HELPER: Behavior knobs (all public attributes): a count and status of requests
    # to reject with a synthetic response, a count of requests whose upstream response is dropped
    # (the connection is closed without answering after the upstream already processed the
    # request), flags to strip or replace the boot-uuid response header, and a record-only mode
    # that answers 503 without contacting upstream. Each connection serves one request and is
    # then closed. Not thread-safe; used from one event loop.
    """A local TCP relay that records requests and can inject HTTP failures."""

    def __init__(self, upstream_port: int | None) -> None:
        # [FUNCTION DOC] HELPER: Stores the upstream port (nothing means record-only use) and
        # initializes the request log, the failure counters and the rewrite options to off.
        self._upstream_port = upstream_port
        self._server: asyncio.Server | None = None
        self._connections: dict[asyncio.StreamWriter, asyncio.Task] = {}
        self.requests: list[ProxyRequest] = []
        self.reject_status: int | None = None
        self.reject_remaining = 0
        self.drop_remaining = 0
        self.strip_boot_uuid = False
        self.rewrite_boot_uuid: str | None = None
        self.record_only = False

    @property
    def port(self) -> int:
        # [FUNCTION DOC] HELPER: The local port the proxy listens on; asserts it was started.
        assert self._server is not None
        return self._server.sockets[0].getsockname()[1]

    @property
    def url(self) -> str:
        # [FUNCTION DOC] HELPER: Base http url on the loopback address and the proxy port.
        return f"http://127.0.0.1:{self.port}"

    def submits(self, method: str) -> list[ProxyRequest]:
        # [FUNCTION DOC] HELPER: Returns the recorded post requests whose path is the versioned
        # rpc path of the given method name, in arrival order.
        return [r for r in self.requests if r.verb == "POST" and r.path == f"/v1/{method}"]

    def reject_next(self, count: int, status: int) -> None:
        # [FUNCTION DOC] HELPER: Makes the next count requests (replacing any earlier remaining
        # count) get a synthetic response with the given status instead of being forwarded.
        self.reject_remaining = count
        self.reject_status = status

    def drop_next(self, count: int) -> None:
        # [FUNCTION DOC] HELPER: For the next count requests, forwards to upstream and reads the
        # response but then closes the client connection without answering.
        self.drop_remaining = count

    async def start(self) -> None:
        # [FUNCTION DOC] HELPER: Starts listening on an ephemeral loopback port.
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)

    async def stop(self) -> None:
        # [FUNCTION DOC] HELPER: Stops listening, closes every open client connection, cancels
        # their handler tasks, waits for them (errors ignored), then waits for the listener to
        # finish closing. Does nothing if never started.
        if self._server is not None:
            self._server.close()
            # Python 3.12 waits for accepted connections, including abandoned requests.
            tasks = list(self._connections.values())
            for writer, task in self._connections.items():
                writer.close()
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._server.wait_closed()

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # [FUNCTION DOC] HELPER: Connection callback. If the listener is already closing, closes
        # the new connection. Otherwise starts a handler task for it, tracks it by its writer,
        # and removes the entry when the task finishes.
        # An accept callback may already be queued when stop() closes the listener.
        if not self._server.is_serving():
            writer.close()
            return
        task = asyncio.create_task(self._handle(reader, writer))
        self._connections[writer] = task
        task.add_done_callback(lambda _: self._connections.pop(writer))

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # [FUNCTION DOC] HELPER: Serves one request on a connection. Reads the request (returns if
        # the client sent nothing), records it, then in order: in record-only mode answers 503;
        # else if rejections remain, decrements and answers with the configured status; else
        # forwards to upstream (asserting an upstream exists), and if drops remain decrements and
        # returns without answering; otherwise writes the (possibly header-rewritten) upstream
        # response. Connection resets, broken pipes and incomplete reads are ignored. The
        # connection is always closed at the end, ignoring close errors.
        try:
            request = await _read_http_message(reader)
            if request is None:
                return

            verb, path = _parse_request_line(request)
            self.requests.append(ProxyRequest(at=time.monotonic(), verb=verb, path=path, body=request))

            if self.record_only:
                _write_simple(writer, 503, b"record-only proxy")
                await writer.drain()
                return

            if self.reject_remaining != 0 and self.reject_status is not None:
                self.reject_remaining -= 1
                _write_simple(writer, self.reject_status, b"injected failure")
                await writer.drain()
                return

            assert self._upstream_port is not None
            response = await _forward(self._upstream_port, request)

            if self.drop_remaining != 0:
                self.drop_remaining -= 1
                return

            writer.write(self._maybe_rewrite(response))
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    def _maybe_rewrite(self, response: bytes) -> bytes:
        # [FUNCTION DOC] HELPER: Returns the response unchanged unless stripping or rewriting of
        # the boot-uuid header is enabled. Otherwise splits head from body at the first blank
        # line, and for each header line named x-miles-boot-uuid (case-insensitive) either drops
        # it (strip has priority) or replaces its value with the configured one; other lines are
        # kept. Reassembles head, separator and body.
        if not self.strip_boot_uuid and self.rewrite_boot_uuid is None:
            return response

        head, separator, body = response.partition(b"\r\n\r\n")
        kept = []
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"x-miles-boot-uuid:"):
                if self.strip_boot_uuid:
                    continue
                line = b"x-miles-boot-uuid: " + self.rewrite_boot_uuid.encode()
            kept.append(line)
        return b"\r\n".join(kept) + separator + body


async def _forward(upstream_port: int, request: bytes) -> bytes:
    # [FUNCTION DOC] HELPER: Opens a fresh loopback connection to the upstream port, sends the raw
    # request bytes, reads one complete http message (waiting for the body, including chunked
    # bodies) and returns it, or empty bytes if nothing came back. Always closes the connection.
    reader, writer = await asyncio.open_connection("127.0.0.1", upstream_port)
    try:
        writer.write(request)
        await writer.drain()
        return await _read_http_message(reader, require_body=True) or b""
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def _read_http_message(reader: asyncio.StreamReader, require_body: bool = False) -> bytes | None:
    # [FUNCTION DOC] HELPER: Reads one http message from the stream in 4096-byte reads. Reads
    # until the blank line ending the head; if the stream ends first returns what was read, or
    # nothing if nothing was read. Then reads the body to the declared content length (stopping
    # early at end of stream). If body is required, the length is zero and the head declares
    # chunked transfer encoding, keeps reading until the body ends with the chunked terminator
    # (or the stream ends). Returns head, blank line and body as bytes. Chunked requests are
    # only handled in the require-body mode. A malformed content length raises a value error.
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = await reader.read(4096)
        if not chunk:
            return head or None
        head += chunk

    header_blob, _, rest = head.partition(b"\r\n\r\n")
    length = 0
    for line in header_blob.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])

    body = rest
    while len(body) < length:
        chunk = await reader.read(4096)
        if not chunk:
            break
        body += chunk

    if require_body and length == 0 and b"transfer-encoding: chunked" in header_blob.lower():
        while not body.endswith(b"0\r\n\r\n"):
            chunk = await reader.read(4096)
            if not chunk:
                break
            body += chunk

    return header_blob + b"\r\n\r\n" + body


def _parse_request_line(request: bytes) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Returns the verb and the path (query string removed) from the first
    # line of a raw request, or a pair of empty strings if the line has fewer than two parts.
    first_line = request.split(b"\r\n", 1)[0].decode(errors="replace")
    parts = first_line.split(" ")
    return (parts[0], parts[1].split("?")[0]) if len(parts) >= 2 else ("", "")


def _write_simple(writer: asyncio.StreamWriter, status: int, body: bytes) -> None:
    # [FUNCTION DOC] HELPER: Writes a minimal http/1.1 response with the given status (reason
    # text "Injected"), a content length, a close-connection header and the body. Does not
    # flush; callers drain the writer.
    writer.write(
        f"HTTP/1.1 {status} Injected\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
    )


class ConnectionCountingRelay:
    # [CLASS DOC] HELPER: Transparent byte relay to an upstream loopback port that counts how many
    # client connections it accepted (used to check connection reuse). Does not parse http.
    def __init__(self, upstream_port: int) -> None:
        # [FUNCTION DOC] HELPER: Stores the upstream port; accepted count starts at zero.
        self._upstream_port = upstream_port
        self._server: asyncio.Server | None = None
        self.accepted = 0

    @property
    def port(self) -> int:
        # [FUNCTION DOC] HELPER: Local listening port; asserts the relay was started.
        assert self._server is not None
        return self._server.sockets[0].getsockname()[1]

    @property
    def url(self) -> str:
        # [FUNCTION DOC] HELPER: Base http url on the loopback address and the relay port.
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        # [FUNCTION DOC] HELPER: Starts listening on an ephemeral loopback port.
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)

    async def stop(self) -> None:
        # [FUNCTION DOC] HELPER: Closes the listener and waits for it to finish closing. Open
        # relayed connections are not closed explicitly. Does nothing if never started.
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # [FUNCTION DOC] HELPER: Counts the connection, opens a connection to upstream, and pumps
        # bytes in both directions concurrently until both directions end; then closes both
        # upstream and client writers, ignoring close errors.
        self.accepted += 1
        upstream_reader, upstream_writer = await asyncio.open_connection("127.0.0.1", self._upstream_port)

        try:
            await asyncio.gather(_pump(reader, upstream_writer), _pump(upstream_reader, writer))
        finally:
            for stream in (upstream_writer, writer):
                with contextlib.suppress(Exception):
                    stream.close()


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    # [FUNCTION DOC] HELPER: Copies data from reader to writer in chunks of up to 64 KiB until
    # the reader reaches end of stream, flushing after each chunk (connection resets and broken
    # pipes are ignored), then signals end of stream on the writer, ignoring any error.
    with contextlib.suppress(ConnectionResetError, BrokenPipeError):
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()

    with contextlib.suppress(Exception):
        writer.write_eof()
