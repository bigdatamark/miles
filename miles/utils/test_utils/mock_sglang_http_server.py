# [MODULE DOC] HELPER: A minimal threaded HTTP server that answers every GET and POST request with
# the same fixed JSON body and status 200, recording each request (method, path without query
# string, parsed JSON payload for POSTs) so tests can assert what clients sent. Runs in a daemon
# thread, tracks open client connections so close can forcibly cut them, and supports persistent
# (HTTP/1.1 keep-alive) connections.
from __future__ import annotations

import dataclasses
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from miles.utils.misc import NodeProbeMixin

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RecordedRequest:
    # [CLASS DOC] HELPER: Immutable record of one received request: the HTTP method name, the path
    # with any query string removed, and the decoded JSON body (none when there was no body, such as
    # for GET requests).
    method: str
    path: str
    payload: dict[str, Any] | None


class MockSGLangHttpServer:
    # [CLASS DOC] HELPER: The mock server. Binds all interfaces (any address) on the requested
    # port (0 picks a free one) and immediately starts serving in a background daemon thread. The
    # request log, and the set of open connection sockets, are protected by a lock. Public
    # surface: host, port, url, requests, paths, payloads_of and close.
    def __init__(self, response_payload: dict[str, Any] | None = None, port: int = 0):
        # [FUNCTION DOC] HELPER: Stores the response body mapping (default a mapping with a single
        # key "mock" set to true), creates the request log, lock and connection set, binds the
        # threaded HTTP server on all interfaces at the port (OS error if bound or unavailable)
        # and starts its serve loop in a daemon thread.
        self._response_payload = response_payload if response_payload is not None else {"mock": True}
        self._requests: list[RecordedRequest] = []
        self._lock = threading.Lock()
        self._connections: set[socket.socket] = set()

        self._server = ThreadingHTTPServer(("0.0.0.0", port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def host(self) -> str:
        # [FUNCTION DOC] HELPER: The node's IP address as determined by the node-probe mixin (not
        # the bind address).
        return NodeProbeMixin._get_node_ip()

    @property
    def port(self) -> int:
        # [FUNCTION DOC] HELPER: The actual bound port number (useful when constructed with port
        # 0).
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        # [FUNCTION DOC] HELPER: Base URL built from the node IP and bound port, using plain http.
        return f"http://{self.host}:{self.port}"

    @property
    def requests(self) -> list[RecordedRequest]:
        # [FUNCTION DOC] HELPER: A snapshot copy of the recorded requests in arrival order, taken
        # under the lock.
        with self._lock:
            return list(self._requests)

    @property
    def paths(self) -> list[str]:
        # [FUNCTION DOC] HELPER: The path of each recorded request in arrival order.
        return [request.path for request in self.requests]

    def payloads_of(self, path: str) -> list[dict[str, Any] | None]:
        # [FUNCTION DOC] HELPER: The payloads (possibly none) of recorded requests whose path
        # equals the given path exactly, in arrival order, regardless of HTTP method.
        return [request.payload for request in self.requests if request.path == path]

    def close(self) -> None:
        # [FUNCTION DOC] HELPER: Stops the server: ends the serve loop and waits for it, closes the
        # listening socket, then shuts down both directions of and closes every tracked client
        # connection (ignoring errors for connections already down, logged at debug level), and
        # finally joins the server thread with a 5 second timeout. Safe against idle keep-alive
        # connections that would otherwise keep handler threads alive. Not idempotent: calling it
        # twice blocks or fails because the serve loop already ended.
        self._server.shutdown()
        self._server.server_close()
        for connection in self._drain_connections():
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                logger.debug("mock sglang http server: connection was already down", exc_info=True)
            connection.close()
        self._thread.join(timeout=5)

    def _record(self, method: str, path: str, payload: dict[str, Any] | None) -> None:
        # [FUNCTION DOC] HELPER: Appends a request record with the path truncated at the first
        # question mark, under the lock.
        with self._lock:
            self._requests.append(RecordedRequest(method=method, path=path.split("?")[0], payload=payload))

    def _register_connection(self, connection: socket.socket) -> None:
        # [FUNCTION DOC] HELPER: Adds a client connection socket to the tracked set, under the lock.
        with self._lock:
            self._connections.add(connection)

    def _unregister_connection(self, connection: socket.socket) -> None:
        # [FUNCTION DOC] HELPER: Removes a client connection socket from the tracked set if present,
        # under the lock.
        with self._lock:
            self._connections.discard(connection)

    def _drain_connections(self) -> list[socket.socket]:
        # [FUNCTION DOC] HELPER: Returns a list of all tracked connections and empties the set,
        # atomically under the lock.
        with self._lock:
            connections = list(self._connections)
            self._connections.clear()
        return connections

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        # [FUNCTION DOC] HELPER: Builds the request handler class bound to this server instance
        # (captured by closure). The class speaks HTTP/1.1, registers each connection on setup and
        # unregisters it on finish, records GET requests (no payload) and POST requests (body of
        # Content-Length bytes, parsed as JSON when non-empty, else none; invalid JSON raises and
        # the request is not recorded), always replies with status 200, JSON content type, correct
        # length and the fixed response body, and routes the base class access logging to the
        # debug logger.
        server = self

        class _Handler(BaseHTTPRequestHandler):
            # [CLASS DOC] HELPER: Per-connection request handler; see the enclosing function.
            protocol_version = "HTTP/1.1"

            def setup(self):
                # [FUNCTION DOC] HELPER: Runs the base setup, then registers the connection
                # socket with the server for forced closing.
                super().setup()
                server._register_connection(self.connection)

            def finish(self):
                # [FUNCTION DOC] HELPER: Unregisters the connection socket, then runs the base
                # finish.
                server._unregister_connection(self.connection)
                super().finish()

            def do_GET(self):
                # [FUNCTION DOC] HELPER: Records the GET (path as sent, with no payload) and sends
                # the fixed response.
                server._record("GET", self.path, None)
                self._respond()

            def do_POST(self):
                # [FUNCTION DOC] HELPER: Reads the number of bytes given by the content-length
                # header (missing or empty means zero), parses a non-empty body as JSON, records
                # the POST with the parsed payload, and sends the fixed response.
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                payload = json.loads(raw) if raw else None
                server._record("POST", self.path, payload)
                self._respond()

            def log_message(self, format, *args):
                # [FUNCTION DOC] HELPER: Overrides access logging to emit a debug-level log line
                # with the server prefix instead of writing to standard error.
                logger.debug("mock sglang http server: " + format, *args)

            def _respond(self):
                # [FUNCTION DOC] HELPER: Sends status 200 with a JSON content type and exact
                # content length, and writes the JSON-encoded fixed response payload.
                body = json.dumps(server._response_payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return _Handler
