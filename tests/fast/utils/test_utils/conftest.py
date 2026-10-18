# [MODULE DOC] Shared pytest support for the test utility tests: fake session verification
# backend and config objects that record calls instead of launching anything, a fixture factory
# that creates mock sglang HTTP servers and closes them at teardown, and a helper to open a plain
# HTTP connection to such a server.
from __future__ import annotations

import http.client
import json
import urllib.parse
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from miles.utils.test_utils.mock_sglang_http_server import MockSGLangHttpServer


class RecordingSessionBackend:
    # [CLASS DOC] HELPER: Fake backend that records command strings and training launch keyword
    # arguments in two public lists (CPU commands and train launches) instead of executing them.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the two empty recording lists.
        self.cpu_commands: list[str] = []
        self.train_launches: list[dict[str, Any]] = []

    def exec_command_cpu(self, command: str) -> None:
        # [FUNCTION DOC] HELPER: Appends the command string to the CPU command list; runs nothing.
        self.cpu_commands.append(command)

    def execute_train(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Appends the keyword arguments to the train launch list, then
        # simulates a successful session verification run by writing a single JSON line to the
        # path given by the metrics path entry of the extra environment variables mapping (raises
        # a key error if that mapping or entry is missing). The JSON holds a driver events list
        # with one append-tool event and a false assistant mismatch flag. Overwrites the file.
        self.train_launches.append(kwargs)
        metrics_path = Path(kwargs["extra_env_vars"]["MILES_SESSION_VERIFY_METRICS_PATH"])
        metrics_path.write_text(json.dumps({"driver_events": ["append_tool"], "had_assistant_mismatch": False}) + "\n")


class RecordingSessionConfig:
    # [CLASS DOC] HELPER: Fake session config that always hands out one given backend and counts
    # how many times a backend was requested.
    def __init__(self, backend: RecordingSessionBackend) -> None:
        # [FUNCTION DOC] HELPER: Stores the backend and sets the creation call counter to zero.
        self.backend = backend
        self.create_backend_calls = 0

    def create_backend(self) -> RecordingSessionBackend:
        # [FUNCTION DOC] HELPER: Increments the call counter and returns the stored backend.
        self.create_backend_calls += 1
        return self.backend


@pytest.fixture
def make_server() -> Iterator[Callable[..., MockSGLangHttpServer]]:
    # [FUNCTION DOC] HELPER: Fixture yielding a factory that creates mock sglang HTTP servers
    # from keyword arguments and remembers them; after the test, closes every created server.
    servers: list[MockSGLangHttpServer] = []

    def _make(**kwargs) -> MockSGLangHttpServer:
        # [FUNCTION DOC] HELPER: Creates a mock server with the given keyword arguments, records
        # it for teardown, and returns it.
        server = MockSGLangHttpServer(**kwargs)
        servers.append(server)
        return server

    yield _make

    for server in servers:
        server.close()


def connect_via_url(server: MockSGLangHttpServer) -> http.client.HTTPConnection:
    # [FUNCTION DOC] HELPER: Parses the server's URL, asserts its scheme is http (assertion error
    # otherwise), and returns an unconnected HTTP connection object to that host and port with a
    # 5 second timeout. The connection is lazily opened by the caller's first request.
    parts = urllib.parse.urlsplit(server.url)
    assert parts.scheme == "http"
    return http.client.HTTPConnection(parts.hostname, parts.port, timeout=5)
