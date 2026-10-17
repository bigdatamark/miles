# [MODULE DOC] HELPER: In-memory stand-in for the engine command actor, usable without GPUs, the
# real inference server, or a model. The mock records every call it receives, can be told to
# raise an injected exception on the next call of a given method, and on "run" starts a tiny mock
# HTTP server on the port parsed from the launch command so tests can check addressing and which
# HTTP requests were sent. The class is wrapped as a Ray remote actor class at module end (the
# name is rebound to the remote version).
"""In-memory stand-in for the engine ``CommandActor`` (no CUDA, no sglang, no model)."""

from __future__ import annotations

import logging
import shlex
from typing import Any

import ray

from miles.utils.misc import NodeProbeMixin, get_free_port
from miles.utils.test_utils.mock_sglang_http_server import MockSGLangHttpServer

logger = logging.getLogger(__name__)


def parse_cmd_flags(cmd: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Splits a launch command with shell-style quoting and scans for
    # double-dash flags. Returns a mapping from flag name (leading dashes removed, remaining
    # dashes turned into underscores) to its value: the next token if it exists and does not start
    # with a double dash (converted to an integer if it consists only of decimal digits, otherwise
    # kept as a string), else the boolean true for a bare flag. Tokens that are not flags and
    # not consumed as a value are skipped. Later duplicates overwrite earlier ones. Negative
    # numbers stay strings (a minus sign is not a digit). Raises a value error on unbalanced quotes
    # (from the shell-style splitting).
    """Naive ``--flag value`` scan of a launch command; enough for addressing asserts."""
    tokens = shlex.split(cmd)
    flags: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        name = token[2:].replace("-", "_")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            value = tokens[index + 1]
            flags[name] = int(value) if value.isdigit() else value
            index += 2
        else:
            flags[name] = True
            index += 1
    return flags


class MockSGLangEngine:
    # [CLASS DOC] HELPER: Fake engine actor. State: an ordered log of calls (method name, positional
    # args tuple, keyword args mapping), a one-shot fault table (method name to exception), the
    # mock HTTP server (created on run) and the flags parsed from the run command. Most methods
    # mirror the private probing methods of the real actor with trivial answers and record
    # themselves. Not thread-safe.
    """Records every call into ``self.calls`` so tests can assert sequence and
    arguments. Fault injection is set via ``set_fault(method, exception)``."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts with an empty call log, no faults, no HTTP server and no
        # parsed server arguments.
        self.calls: list[tuple[str, tuple, dict]] = []
        self._faults: dict[str, BaseException] = {}
        self._http_server: MockSGLangHttpServer | None = None
        self._server_args: dict[str, Any] | None = None

    def set_fault(self, method: str, exception: BaseException | None):
        # [FUNCTION DOC] HELPER: Arms (exception given) or disarms (none) a fault for the named
        # method. An armed fault is raised once by the next call of that method that consults the
        # fault table, then removed. Only the run and shutdown methods consult the table.
        if exception is None:
            self._faults.pop(method, None)
        else:
            self._faults[method] = exception

    def inject_fault(self, mode: str) -> None:
        # [FUNCTION DOC] HELPER: Mock of the real fault-injection entry point: only records the
        # call with the mode; does not crash anything.
        self._record("inject_fault", (), {"mode": mode})

    def get_calls(self) -> list[tuple[str, tuple, dict]]:
        # [FUNCTION DOC] HELPER: Returns a shallow copy of the recorded call log.
        return list(self.calls)

    def get_server_args(self) -> dict[str, Any] | None:
        # [FUNCTION DOC] HELPER: Returns a copy of the flags parsed from the run command, or none
        # if run has not been called.
        return dict(self._server_args) if self._server_args is not None else None

    def get_http_paths(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns the request paths the mock HTTP server received so far in
        # arrival order (query strings removed), or an empty list if no server was started.
        return self._http_server.paths if self._http_server is not None else []

    def get_http_payloads_of(self, path: str) -> list[dict | None]:
        # [FUNCTION DOC] HELPER: Returns the JSON payloads (none for GETs or empty bodies) of the
        # requests received on the given path, or an empty list if no server was started.
        return self._http_server.payloads_of(path) if self._http_server is not None else []

    def run(self, cmd: str, envs: dict[str, str]):
        # [FUNCTION DOC] HELPER: Mock of launching the server. Records the call with the command
        # and environment mapping, raises an armed fault for "run" if any, parses the command
        # flags, and starts a mock HTTP server listening on the integer value of the port flag
        # (key error if no port flag, value error if not an integer). Does not stop a previously
        # started server. Returns nothing.
        self._record("run", (), {"cmd": cmd, "envs": envs})
        self._maybe_fault("run")
        self._server_args = parse_cmd_flags(cmd)
        self._http_server = MockSGLangHttpServer(port=int(self._server_args["port"]))
        return None

    def shutdown(self):
        # [FUNCTION DOC] HELPER: Mock of graceful shutdown. Records the call, raises an armed
        # fault for "shutdown" if any (in which case the server is left running), closes the mock
        # HTTP server if one exists, and returns true.
        self._record("shutdown", (), {})
        self._maybe_fault("shutdown")
        if self._http_server is not None:
            self._http_server.close()
        return True

    def kill_subprocess(self):
        # [FUNCTION DOC] HELPER: Mock of killing the engine subprocess: records the call and closes
        # the mock HTTP server if present. Does not exit the process.
        """Real CommandActor takes the actor down with the subprocess; the
        in-process mock cannot exit the test process, so only the server dies."""
        self._record("kill_subprocess", (), {})
        if self._http_server is not None:
            self._http_server.close()

    def _get_free_port_block(self, *, start_port: int, count: int) -> int:
        # [FUNCTION DOC] HELPER: Records the call and returns the real result of searching for a
        # block of consecutive free ports starting at the given port with the given length (so
        # it has real side effects of probing sockets on this host).
        self._record("_get_free_port_block", (), {"start_port": start_port, "count": count})
        return get_free_port(start_port=start_port, consecutive=count)

    def _get_node_ip(self):
        # [FUNCTION DOC] HELPER: Records the call and returns the real node IP address as
        # determined by the node-probe mixin.
        self._record("_get_node_ip", (), {})
        return NodeProbeMixin._get_node_ip()

    def _get_node_external_ip(self):
        # [FUNCTION DOC] HELPER: Records the call and returns the real external node IP as
        # determined by the node-probe mixin (may involve network access).
        self._record("_get_node_external_ip", (), {})
        return NodeProbeMixin._get_node_external_ip()

    def _to_local_gpu_ids(self, *, gpu_ids: list[int]) -> list[int]:
        # [FUNCTION DOC] HELPER: Records the call and returns local ids as 0..n-1 for n given
        # global ids (ignoring the actual id values).
        self._record("_to_local_gpu_ids", (), {"gpu_ids": gpu_ids})
        return list(range(len(gpu_ids)))

    def _is_port_available(self, *, port: int) -> bool:
        # [FUNCTION DOC] HELPER: Records the call and always reports the port as available.
        self._record("_is_port_available", (), {"port": port})
        return True

    def _get_gpu_uuids(self, gpu_ids: list[int]):
        # [FUNCTION DOC] HELPER: Records the call (the ids passed positionally) and returns a list
        # of nones, one per id, in place of GPU unique identifiers.
        self._record("_get_gpu_uuids", (gpu_ids,), {})
        return [None] * len(gpu_ids)

    def _record(self, name: str, args: tuple, kwargs: dict) -> None:
        # [FUNCTION DOC] HELPER: Appends a (method name, positional args, keyword args) entry to
        # the call log.
        self.calls.append((name, args, kwargs))

    def _maybe_fault(self, method: str) -> None:
        # [FUNCTION DOC] HELPER: Removes and raises the armed fault for the method if one exists;
        # otherwise does nothing.
        exc = self._faults.pop(method, None)
        if exc is not None:
            raise exc


MockSGLangEngine = ray.remote(MockSGLangEngine)
