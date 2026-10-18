# [MODULE DOC]
# Tests for the HTTP/network utilities module. Areas: hostname-to-address resolution (literals pass
# through, IPv6 bracketing, first non-loopback address, optional IPv6 preference via an environment
# variable, loud failures); the blocking server-readiness poll (real sockets and threads, plus
# simulated delays with mocked clock, sleep and connect); the async TCP readiness poll (retry on
# refusal and per-attempt timeouts, writer closing, event loop stays free, bracket stripping); the
# per-event-loop shared HTTP client provider and its timeout policy (no read timeout, finite
# connect, write and pool deadlines, unlimited connection pool); and the distributed Ray poster
# actors (keyword-only construction, concurrency split across nodes and GPU slots) and the HTTP
# client concurrency sizing. Helper classes: fake sockets/writers, a blocking threaded HTTP server
# with an arrival gate, and a recording transport.
"""Tests for wait_for_server_ready in http_utils.

wait_for_server_ready() polls a TCP port in a loop until the server is
accepting connections.  Each iteration:
  1. Check if the process is still alive (if a process handle was given).
  2. Try ``socket.create_connection((host, port))``.
  3. If it connects → server is ready, return.
  4. If OSError (connection refused) → sleep 0.5s, retry.
  5. If ``time.time()`` exceeds the deadline → raise RuntimeError.

TestWaitForServerReady uses real sockets/threads to verify end-to-end
behaviour (port already open, delayed open, timeout, dead process).

TestWaitForServerReadySimulatedDelays uses mocks to test long startup
delays (5s / 10s / 20s) without actually waiting.  The trick:
  - Mock ``time.time()``  → returns a fake clock that we control.
  - Mock ``time.sleep()`` → doesn't really wait, just advances the fake clock.
  - Mock ``socket.create_connection()`` → raises OSError for the first N
    calls (simulating "port not ready"), then returns a fake socket
    (simulating "port ready").
This lets us simulate 20 seconds of polling in <1ms of real time.
"""

import asyncio
import inspect
import multiprocessing
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import patch

import httpx
import pytest
import ray
from tests.fast.utils.fake_ray_ids import fake_ray_node_id

from miles.utils import http_utils
from miles.utils.http_utils import (
    MILES_PREFER_IPV6_ENV,
    GeneralHttpClientProvider,
    resolve_ip,
    wait_for_server_ready,
    wait_tcp_ready_async,
)


def _find_free_port() -> int:
    # [FUNCTION DOC] HELPER: Bind a TCP socket to the loopback address with an OS-chosen port, read
    # the port number, close the socket and return the port. Another process could in principle
    # grab the port after it is released.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _listen_after_delay(host: str, port: int, delay: float, stop_event: threading.Event):
    # [FUNCTION DOC] HELPER: Thread body. Sleep for the delay, bind and listen on the host and port
    # (address reuse enabled, backlog one), block until the stop event is set, then close the
    # listening socket. Connections are never accepted.
    """Open a TCP listener after *delay* seconds, keep it open until *stop_event* is set."""
    time.sleep(delay)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    stop_event.wait()
    srv.close()


def _addrinfo_of(records: dict[int, list[str]]):
    # [FUNCTION DOC] HELPER: Build a fake address-lookup function from a mapping of address family
    # to a list of IP strings. The fake ignores the host and returns stream-socket records for the
    # requested family; a family absent from the mapping (or with an empty list) raises a
    # name-resolution error ("Name or service not known").
    def _getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        # [FUNCTION DOC] HELPER: Look up the requested family in the captured mapping and return
        # one record per address, each with the address and port zero, or raise the resolution
        # error. Family zero (unspecified) therefore always fails.
        ips = records.get(family)
        if not ips:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]

    return _getaddrinfo


def _refuse_lookup(*_args, **_kwargs):
    # [FUNCTION DOC] HELPER: Fake lookup that accepts any arguments and always raises an assertion
    # error, proving no name lookup happens.
    raise AssertionError("a name lookup ran for a host that needs none")


class TestResolveIp:
    # [CLASS DOC] Tests of the host-to-bindable-address resolver, with the system address lookup
    # monkeypatched.
    def test_an_ipv4_literal_passes_through_without_a_lookup(self, monkeypatch):
        # [FUNCTION DOC] With lookups forbidden, an IPv4 literal resolves to itself.
        """A literal already is what a socket address parser wants, so resolving it would only add a failure mode."""
        monkeypatch.setattr(socket, "getaddrinfo", _refuse_lookup)

        assert resolve_ip("10.0.0.7") == "10.0.0.7"

    def test_a_wildcard_literal_passes_through(self, monkeypatch):
        # [FUNCTION DOC] With lookups forbidden, the IPv4 wildcard stays as is and the IPv6
        # wildcard comes back bracketed.
        """Kubernetes binds every command worker on the wildcard, which no lookup could produce."""
        monkeypatch.setattr(socket, "getaddrinfo", _refuse_lookup)

        assert resolve_ip("0.0.0.0") == "0.0.0.0"
        assert resolve_ip("::") == "[::]"

    def test_an_ipv6_literal_keeps_its_brackets(self, monkeypatch):
        # [FUNCTION DOC] With lookups forbidden, an already bracketed IPv6 literal is unchanged and
        # an unbracketed one gets brackets.
        """A socket address parser needs the brackets to tell the address from the port."""
        monkeypatch.setattr(socket, "getaddrinfo", _refuse_lookup)

        assert resolve_ip("[fd00::7]") == "[fd00::7]"
        assert resolve_ip("fd00::7") == "[fd00::7]"

    def test_a_hostname_resolves_to_its_first_non_loopback_address(self, monkeypatch):
        # [FUNCTION DOC] With the IPv6 preference variable unset and a lookup returning a loopback
        # address followed by a routable one, the hostname resolves to the routable address.
        """Ray hands out the hostname a slurm node was started with, and the router cannot bind on a name."""
        monkeypatch.delenv(MILES_PREFER_IPV6_ENV, raising=False)
        monkeypatch.setattr(socket, "getaddrinfo", _addrinfo_of({socket.AF_INET: ["127.0.1.1", "10.0.0.7"]}))

        assert resolve_ip("node1234") == "10.0.0.7"

    def test_ipv4_is_the_only_family_by_default(self, monkeypatch):
        # [FUNCTION DOC] With the preference variable unset and both families available, the IPv4
        # address is chosen.
        """A dual-stack name binds the v4 address unless the deployment asks for v6."""
        monkeypatch.delenv(MILES_PREFER_IPV6_ENV, raising=False)
        monkeypatch.setattr(
            socket, "getaddrinfo", _addrinfo_of({socket.AF_INET: ["10.0.0.7"], socket.AF_INET6: ["fd00::7"]})
        )

        assert resolve_ip("node1234") == "10.0.0.7"

    def test_ipv6_is_preferred_when_the_environment_asks_for_it(self, monkeypatch):
        # [FUNCTION DOC] With the preference variable set to one and both families available, the
        # bracketed IPv6 address is chosen.
        """MILES_PREFER_IPV6 picks the v6 address of a dual-stack name."""
        monkeypatch.setenv(MILES_PREFER_IPV6_ENV, "1")
        monkeypatch.setattr(
            socket, "getaddrinfo", _addrinfo_of({socket.AF_INET: ["10.0.0.7"], socket.AF_INET6: ["fd00::7"]})
        )

        assert resolve_ip("node1234") == "[fd00::7]"

    def test_a_v6_only_request_never_falls_back_to_v4(self, monkeypatch):
        # [FUNCTION DOC] With the preference variable set and only IPv4 records available, the
        # resolver raises the name-resolution error instead of falling back.
        """A deployment that asked for v6 has a v4 that does not route, so a silent v4 bind would be unreachable."""
        monkeypatch.setenv(MILES_PREFER_IPV6_ENV, "1")
        monkeypatch.setattr(socket, "getaddrinfo", _addrinfo_of({socket.AF_INET: ["10.0.0.7"]}))

        with pytest.raises(socket.gaierror):
            resolve_ip("node1234")

    def test_a_name_that_only_reaches_loopback_fails_loudly(self, monkeypatch):
        # [FUNCTION DOC] When the only address is loopback, the resolver raises a runtime error
        # whose message contains the hostname.
        """Binding on loopback would start a router nothing else can reach, so the launch must fail instead."""
        monkeypatch.delenv(MILES_PREFER_IPV6_ENV, raising=False)
        monkeypatch.setattr(socket, "getaddrinfo", _addrinfo_of({socket.AF_INET: ["127.0.1.1"]}))

        with pytest.raises(RuntimeError, match="node1234"):
            resolve_ip("node1234")

    def test_an_unknown_name_fails_loudly(self, monkeypatch):
        # [FUNCTION DOC] When the lookup finds nothing, the name-resolution error propagates.
        """An unresolvable name has no address to bind, so the lookup error propagates as is."""
        monkeypatch.setattr(socket, "getaddrinfo", _addrinfo_of({}))

        with pytest.raises(socket.gaierror):
            resolve_ip("node1234")


# ---------------------------------------------------------------------------
# Real-network tests (use actual sockets and threads)
# ---------------------------------------------------------------------------


class TestWaitForServerReady:
    # [CLASS DOC] Real-socket tests of the blocking server-readiness poll.
    def test_returns_when_port_is_already_open(self):
        # [FUNCTION DOC] A real listener already open on a free loopback port: the wait returns
        # without error (timeout 2 s); the listener is closed afterwards.
        """Server is already listening → should return immediately."""
        port = _find_free_port()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        try:
            wait_for_server_ready("127.0.0.1", port, timeout=2)
        finally:
            srv.close()

    def test_waits_until_port_opens(self):
        # [FUNCTION DOC] A daemon thread opens the listener after 1 s; the wait (timeout 10 s)
        # returns after at least about 0.8 s elapsed; the stop event is set at the end.
        """Server starts listening after 1s → should wait and then return."""
        port = _find_free_port()
        stop = threading.Event()
        t = threading.Thread(target=_listen_after_delay, args=("127.0.0.1", port, 1.0, stop))
        t.daemon = True
        t.start()
        try:
            start = time.time()
            wait_for_server_ready("127.0.0.1", port, timeout=10)
            elapsed = time.time() - start
            assert elapsed >= 0.8, f"Should have waited ~1s, waited {elapsed:.2f}s"
        finally:
            stop.set()

    def test_raises_on_timeout(self):
        # [FUNCTION DOC] With nothing listening and a 1 s timeout, the wait raises a runtime error
        # whose message contains "not ready after".
        """No server on the port → should raise after timeout."""
        port = _find_free_port()
        with pytest.raises(RuntimeError, match="not ready after"):
            wait_for_server_ready("127.0.0.1", port, timeout=1)

    def test_raises_when_process_dies(self):
        # [FUNCTION DOC] Start and join a child process that exits at once, then wait for a port
        # while passing that dead process handle; expect a runtime error containing "process died".
        """Process exits before port is ready → should raise immediately."""
        port = _find_free_port()

        def _die_immediately():
            # [FUNCTION DOC] HELPER: Child process target that does nothing and exits.
            pass

        proc = multiprocessing.Process(target=_die_immediately)
        proc.start()
        proc.join()  # ensure it's dead before we call wait

        with pytest.raises(RuntimeError, match="process died"):
            wait_for_server_ready("127.0.0.1", port, process=proc, timeout=5)

    def test_raises_when_subprocess_dies(self) -> None:
        # [FUNCTION DOC] Run a Python subprocess that exits with status 2 and wait for it; the
        # readiness wait given that finished process raises a runtime error containing "process
        # died".
        """Subprocess exits before port is ready and raises immediately."""
        process: subprocess.Popen[bytes] = subprocess.Popen([sys.executable, "-c", "raise SystemExit(2)"])
        process.wait(timeout=5)

        with pytest.raises(RuntimeError, match="process died"):
            wait_for_server_ready("127.0.0.1", 0, process=process, timeout=5)

    def test_waits_for_a_live_subprocess_to_open_its_port(self) -> None:
        # [FUNCTION DOC] A subprocess sleeping for 30 s stays alive while a thread opens a listener
        # after 1 s; the wait with that process handle returns, and the process is still running.
        # The subprocess is killed and the thread stop event is set in a finally step.
        """A still-running subprocess counts as alive, so a port opened later is awaited."""
        port = _find_free_port()
        stop = threading.Event()
        process: subprocess.Popen[bytes] = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        listener = threading.Thread(target=_listen_after_delay, args=("127.0.0.1", port, 1.0, stop))
        listener.daemon = True
        listener.start()

        try:
            wait_for_server_ready("127.0.0.1", port, process=process, timeout=10)
            assert process.poll() is None
        finally:
            stop.set()
            process.kill()
            process.wait(timeout=5)

    def test_raises_when_subprocess_exits_while_waiting_for_port(self) -> None:
        # [FUNCTION DOC] A subprocess that sleeps 1 s is alive at first poll but exits during the
        # wait (timeout 30 s, nothing listening); expect a runtime error containing "process
        # died". The subprocess is killed in a finally step.
        """A subprocess alive at the first poll but exiting later fails as a dead process."""
        port = _find_free_port()
        process: subprocess.Popen[bytes] = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])

        try:
            assert process.poll() is None
            with pytest.raises(RuntimeError, match="process died"):
                wait_for_server_ready("127.0.0.1", port, process=process, timeout=30)
        finally:
            process.kill()
            process.wait(timeout=5)


# ---------------------------------------------------------------------------
# Simulated-delay tests (mock time + socket so tests run instantly)
#
# How it works:
#   wait_for_server_ready() calls time.time() / time.sleep() / socket.create_connection()
#   in a loop.  We replace all three:
#
#   - fake_time_fn():  returns a counter we control (starts at 0.0)
#   - fake_sleep(d):   advances the counter by d (no real waiting)
#   - fake_connect():  raises OSError for the first N calls, then succeeds
#
#   This makes the loop run through all its iterations at full speed while
#   the function "thinks" real time is passing.  A 20s simulated delay
#   finishes in <1ms of wall-clock time.
# ---------------------------------------------------------------------------


class _FakeSocket:
    # [CLASS DOC] HELPER: Minimal fake connected socket usable only as a context manager.
    """Minimal stand-in for a connected socket (used as context manager)."""

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Return the fake socket itself.
        return self

    def __exit__(self, *args):
        # [FUNCTION DOC] HELPER: Do nothing and do not suppress exceptions.
        pass


class TestWaitForServerReadySimulatedDelays:
    # [CLASS DOC] Tests of the blocking readiness poll against a fake clock, fake sleep and fake
    # connect function, so long delays run instantly.
    @pytest.mark.parametrize("simulated_delay_s", [5, 10, 20])
    def test_succeeds_after_simulated_delay(self, simulated_delay_s):
        # [FUNCTION DOC] Parametrized over 5, 10 and 20 simulated seconds. The fake connect fails
        # for the first delay-divided-by-0.5 attempts (poll interval 0.5 s) then succeeds. With
        # clock, sleep and connect patched in the module under test and a timeout of delay plus
        # 10 s, assert the connect function was called exactly failures plus one times.
        """Port becomes available after simulated_delay_s seconds.

        wait_for_server_ready polls every 0.5s, so it should take
        (simulated_delay_s / 0.5) failed attempts before succeeding.
        """
        poll_interval = 0.5  # matches the sleep(0.5) inside wait_for_server_ready
        polls_until_ready = int(simulated_delay_s / poll_interval)
        call_count = 0
        fake_time = [0.0]

        def fake_time_fn():
            # [FUNCTION DOC] HELPER: Return the current fake clock value.
            return fake_time[0]

        def fake_sleep(duration):
            # [FUNCTION DOC] HELPER: Advance the fake clock by the requested duration without
            # sleeping.
            # Don't really sleep — just advance the fake clock.
            fake_time[0] += duration

        def fake_connect(addr, timeout=None):
            # [FUNCTION DOC] HELPER: Count the call; raise an OS error (connection refused) while
            # the count is within the failure budget, otherwise return a fake socket.
            nonlocal call_count
            call_count += 1
            if call_count <= polls_until_ready:
                # Simulate "port not listening yet"
                raise OSError("Connection refused")
            # Simulate "port is now accepting connections"
            return _FakeSocket()

        with (
            patch("miles.utils.http_utils.time.time", side_effect=fake_time_fn),
            patch("miles.utils.http_utils.time.sleep", side_effect=fake_sleep),
            patch("miles.utils.http_utils.socket.create_connection", side_effect=fake_connect),
        ):
            wait_for_server_ready("127.0.0.1", 9999, timeout=simulated_delay_s + 10)

        # Should have polled exactly polls_until_ready times (fail) + 1 (success)
        assert call_count == polls_until_ready + 1

    @pytest.mark.parametrize("timeout", [5, 10, 20])
    def test_timeout_after_simulated_duration(self, timeout):
        # [FUNCTION DOC] Parametrized over timeouts 5, 10 and 20. With a fake clock and a connect
        # that always fails, the wait raises a runtime error saying not ready after that many
        # seconds, and the fake clock has advanced to at least the timeout.
        """Port never opens → should raise after exactly *timeout* simulated seconds."""
        fake_time = [0.0]

        def fake_time_fn():
            # [FUNCTION DOC] HELPER: Return the current fake clock value.
            return fake_time[0]

        def fake_sleep(duration):
            # [FUNCTION DOC] HELPER: Advance the fake clock by the requested duration.
            fake_time[0] += duration

        def fake_connect(addr, timeout=None):
            # [FUNCTION DOC] HELPER: Always raise an OS error (connection refused).
            # Always fail — server never starts
            raise OSError("Connection refused")

        with (
            patch("miles.utils.http_utils.time.time", side_effect=fake_time_fn),
            patch("miles.utils.http_utils.time.sleep", side_effect=fake_sleep),
            patch("miles.utils.http_utils.socket.create_connection", side_effect=fake_connect),
        ):
            with pytest.raises(RuntimeError, match=f"not ready after {timeout}s"):
                wait_for_server_ready("127.0.0.1", 9999, timeout=timeout)

        # The fake clock should have advanced past the timeout
        assert fake_time[0] >= timeout


class _FakeWriter:
    # [CLASS DOC] HELPER: Fake stream writer recording whether it was closed and whether its
    # wait-for-close was awaited.
    """Minimal stand-in for the writer half of an opened connection."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Both flags start false.
        self.closed: bool = False
        self.wait_closed_awaited: bool = False

    def close(self) -> None:
        # [FUNCTION DOC] HELPER: Mark the writer closed.
        self.closed = True

    async def wait_closed(self) -> None:
        # [FUNCTION DOC] HELPER: Mark the wait-for-close as awaited.
        self.wait_closed_awaited = True


class TestWaitTcpReadyAsync:
    # [CLASS DOC] Tests of the async TCP readiness poll, mostly with the connection opener patched.
    async def test_retries_the_timeout_exception_exported_by_asyncio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Replace asyncio's exported timeout error type with a local exception class,
        # replace the bounded-wait function with a fake that raises it on the first attempt and
        # then returns a connection with the fake writer, and set the retry interval to zero. The
        # poll must retry: two attempts total, and the writer's wait-for-close was awaited.
        """Python 3.10's distinct asyncio timeout must be treated as a refused attempt."""

        class LegacyAsyncioTimeout(Exception):
            # [CLASS DOC] HELPER: Stand-in for the legacy asyncio timeout error type.
            pass

        attempts = 0
        writer = _FakeWriter()

        async def fake_wait_for(awaitable: Any, *, timeout: float) -> tuple[object, _FakeWriter]:
            # [FUNCTION DOC] HELPER: Close the given awaitable without running it, count the
            # attempt, raise the legacy timeout on the first attempt, and afterwards return a
            # dummy reader and the fake writer.
            nonlocal attempts
            awaitable.close()
            attempts += 1
            if attempts == 1:
                raise LegacyAsyncioTimeout
            return object(), writer

        monkeypatch.setattr(http_utils.asyncio, "TimeoutError", LegacyAsyncioTimeout)
        monkeypatch.setattr(http_utils.asyncio, "wait_for", fake_wait_for)
        monkeypatch.setattr(http_utils, "_CONNECT_RETRY_INTERVAL_SECONDS", 0)

        await wait_tcp_ready_async("127.0.0.1", 23456, timeout=30)

        assert attempts == 2
        assert writer.wait_closed_awaited

    async def test_a_successful_probe_waits_until_its_writer_is_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With the connection opener patched to return the fake writer, a
        # successful poll leaves the writer closed and its wait-for-close awaited.
        """A successful probe returns only after its connection writer finishes closing."""
        writer = _FakeWriter()

        async def fake_open_connection(host: str, port: int) -> tuple[object, _FakeWriter]:
            # [FUNCTION DOC] HELPER: Return a dummy reader and the fake writer immediately.
            return object(), writer

        monkeypatch.setattr(http_utils.asyncio, "open_connection", fake_open_connection)

        await wait_tcp_ready_async("127.0.0.1", 23456, timeout=30)

        assert writer.closed
        assert writer.wait_closed_awaited

    async def test_keeps_retrying_until_the_port_accepts(self, monkeypatch):
        # [FUNCTION DOC] The patched opener refuses twice then succeeds; polling a bracketed IPv6
        # host yields three attempts, each with the unbracketed host and the port, and the writer
        # is closed.
        """Readiness depends on the endpoint alone, retrying while it refuses connections."""
        attempts: list[tuple[str, int]] = []
        writer = _FakeWriter()

        async def fake_open_connection(host, port):
            # [FUNCTION DOC] HELPER: Record the host and port; raise an OS error for the first two
            # attempts, then return a dummy reader and the fake writer.
            attempts.append((host, port))
            if len(attempts) < 3:
                raise OSError("Connection refused")
            return object(), writer

        monkeypatch.setattr(http_utils, "_CONNECT_RETRY_INTERVAL_SECONDS", 0)
        monkeypatch.setattr(http_utils.asyncio, "open_connection", fake_open_connection)

        await wait_tcp_ready_async("[2001:db8::7]", 23456, timeout=30)

        assert attempts == [("2001:db8::7", 23456)] * 3
        assert writer.closed

    async def test_gives_up_when_the_deadline_passes(self, monkeypatch):
        # [FUNCTION DOC] An opener that always refuses and a 0.05 s timeout make the poll raise a
        # runtime error stating the server at the host and port was not ready after 0.05 s.
        """A port that never opens fails with a timeout instead of blocking forever."""

        async def fake_open_connection(host, port):
            # [FUNCTION DOC] HELPER: Always raise an OS error (connection refused).
            raise OSError("Connection refused")

        monkeypatch.setattr(http_utils, "_CONNECT_RETRY_INTERVAL_SECONDS", 0)
        monkeypatch.setattr(http_utils.asyncio, "open_connection", fake_open_connection)

        with pytest.raises(RuntimeError, match="Server at 127.0.0.1:23456 not ready after 0.05s"):
            await wait_tcp_ready_async("127.0.0.1", 23456, timeout=0.05)

    async def test_a_connection_that_never_answers_is_one_refused_attempt(self, monkeypatch):
        # [FUNCTION DOC] With the per-attempt timeout set to 0.01 s and an opener that hangs for
        # 10 s, a 0.05 s poll raises a not-ready runtime error after making more than one attempt.
        """A syn that hangs must not hold the whole budget; each attempt has its own small timeout."""
        attempts: list[tuple[str, int]] = []

        async def fake_open_connection(host, port):
            # [FUNCTION DOC] HELPER: Record the attempt and sleep 10 s (never answering).
            attempts.append((host, port))
            await asyncio.sleep(10)

        monkeypatch.setattr(http_utils, "_CONNECT_ATTEMPT_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(http_utils, "_CONNECT_RETRY_INTERVAL_SECONDS", 0)
        monkeypatch.setattr(http_utils.asyncio, "open_connection", fake_open_connection)

        with pytest.raises(RuntimeError, match="not ready"):
            await wait_tcp_ready_async("127.0.0.1", 23456, timeout=0.05)

        assert len(attempts) > 1

    async def test_it_returns_once_the_port_accepts(self):
        # [FUNCTION DOC] Start a real asyncio server on a loopback port that closes each connection
        # at once; the poll (5 s limit) returns without error. The server is closed afterwards.
        """The async probe must still answer the question the blocking one answered."""

        async def _close_immediately(reader, writer) -> None:
            # [FUNCTION DOC] HELPER: Server connection handler that closes the writer and waits for
            # the close to finish.
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(_close_immediately, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        try:
            await asyncio.wait_for(wait_tcp_ready_async("127.0.0.1", port, timeout=5), timeout=5)
        finally:
            server.close()
            await asyncio.wait_for(server.wait_closed(), timeout=5)

    async def test_a_closed_port_leaves_the_event_loop_free(self):
        # [FUNCTION DOC] While polling a closed port with a 1.2 s timeout (which raises a not-ready
        # runtime error), a concurrent ticker task still gets scheduled; it completes all five
        # 20 ms ticks, proving the event loop is not blocked.
        """The blocking probe froze the whole startup loop for up to two minutes per router."""
        ticks = 0

        async def _tick() -> None:
            # [FUNCTION DOC] HELPER: Five times, sleep 20 ms and increment the shared tick counter.
            nonlocal ticks
            for _ in range(5):
                await asyncio.sleep(0.02)
                ticks += 1

        ticker = asyncio.create_task(_tick())
        with pytest.raises(RuntimeError, match="not ready after"):
            await wait_tcp_ready_async("127.0.0.1", _find_free_port(), timeout=1.2)
        await asyncio.wait_for(ticker, timeout=1)

        assert ticks == 5

    async def test_it_gives_up_when_the_deadline_passes(self):
        # [FUNCTION DOC] Polling a free (closed) port with a 0.2 s timeout raises a runtime error
        # stating the server at that host and port was not ready after 0.2 s.
        """A port that never opens must fail the caller rather than be awaited forever."""
        port = _find_free_port()

        with pytest.raises(RuntimeError, match=f"Server at 127.0.0.1:{port} not ready after 0.2s"):
            await wait_tcp_ready_async("127.0.0.1", port, timeout=0.2)

    async def test_a_bracketed_ipv6_host_is_unwrapped_before_connecting(self):
        # [FUNCTION DOC] With a patched opener that records hosts and refuses, polling a bracketed
        # IPv6 loopback with a tiny timeout raises a runtime error, and every recorded host is the
        # unbracketed address.
        """Addresses come in wrapped for urls, and the socket layer rejects the brackets."""
        connected: list[str] = []

        async def _fake_open_connection(host: str, port: int):
            # [FUNCTION DOC] HELPER: Record the host and raise a connection-refused error.
            connected.append(host)
            raise ConnectionRefusedError

        with patch("miles.utils.http_utils.asyncio.open_connection", side_effect=_fake_open_connection):
            with pytest.raises(RuntimeError):
                await wait_tcp_ready_async("[::1]", 23456, timeout=0.01)

        assert connected == ["::1"]


class TestGeneralHttpClientProvider:
    # [CLASS DOC] Tests of the shared HTTP client provider (one client per event loop) and its
    # connection pool and timeout configuration.
    """The provider hands out one httpx client per event loop."""

    async def test_the_same_loop_gets_the_same_client(self):
        # [FUNCTION DOC] Two calls on the same running loop return the identical client object.
        """Two calls on one loop must share one connection pool."""
        assert GeneralHttpClientProvider.client() is GeneralHttpClientProvider.client()

    async def test_a_different_loop_gets_a_different_client(self):
        # [FUNCTION DOC] Get the client on this loop, then in a separate thread run a fresh event
        # loop that fetches the client; the thread is joined (10 s). Assert a client was fetched
        # there and it is a different object.
        """A client's connections belong to the loop that opened them; reusing it elsewhere fails."""
        mine = GeneralHttpClientProvider.client()
        others: list[object] = []

        async def _on_the_other_loop():
            # [FUNCTION DOC] HELPER: Return the provider's client as seen from the running loop.
            return GeneralHttpClientProvider.client()

        thread = threading.Thread(target=lambda: others.append(asyncio.run(_on_the_other_loop())))
        thread.start()
        thread.join(timeout=10)

        assert others and others[0] is not mine

    def test_calling_it_off_loop_fails_loudly(self):
        # [FUNCTION DOC] Calling the provider from a plain synchronous test (no running loop)
        # raises a runtime error.
        """Building the client outside a loop would bind it to whichever loop ran next."""
        with pytest.raises(RuntimeError):
            GeneralHttpClientProvider.client()

    async def test_the_client_has_no_read_timeout_but_finite_connect_write_and_pool_deadlines(self):
        # [FUNCTION DOC] The client's timeout has no read limit, a 10 s connect limit, and 60 s
        # write and pool limits.
        """A weight-update request blocks until the collective forms, but a stuck peer, a full send
        buffer or an exhausted pool must never block a control-path caller forever."""
        timeout = GeneralHttpClientProvider.client().timeout

        assert timeout.read is None
        assert timeout.connect == 10.0
        assert timeout.write == 60.0
        assert timeout.pool == 60.0

    async def test_the_pool_is_configured_without_a_connection_cap(self):
        # [FUNCTION DOC] The provider's limits object has neither a maximum connection count nor a
        # maximum keep-alive count.
        """A capped pool queues the last requests behind the collective waiting for them."""
        assert GeneralHttpClientProvider._LIMITS.max_connections is None
        assert GeneralHttpClientProvider._LIMITS.max_keepalive_connections is None

    async def test_more_requests_than_httpxs_default_cap_reach_the_server_at_once(self):
        # [FUNCTION DOC] Start a local threaded server that blocks each request until released and
        # counts arrivals. Fire 101 concurrent GET requests with the shared client and wait (up to
        # 60 s, failing with a message of how many arrived) until all 101 have arrived at the
        # server simultaneously; then release them and assert every response has status 200.
        # The gate is released and the server shut down in a finally step.
        """101 engines must all arrive before the caller joins the collective, and httpx caps
        connections at 100 by default."""
        num_requests = 101
        arrived = _ArrivalGate()
        server = _BlockingHttpServer(arrived)

        try:
            client = GeneralHttpClientProvider.client()
            requests = [asyncio.create_task(client.get(server.url)) for _ in range(num_requests)]

            deadline = time.monotonic() + 60
            while arrived.count < num_requests:
                assert time.monotonic() < deadline, (
                    f"only {arrived.count}/{num_requests} requests reached the server; the rest are "
                    "queued behind the connection cap"
                )
                await asyncio.sleep(0.01)

            arrived.release()
            assert [response.status_code for response in await asyncio.gather(*requests)] == [200] * num_requests
        finally:
            arrived.release()
            server.close()


class TestGeneralHttpClientTimeoutPolicy:
    # [CLASS DOC] Tests of the timeout policy of the shared HTTP client.
    """Only the read deadline of the shared client may be unbounded."""

    async def test_read_is_the_only_deadline_left_unbounded(self):
        # [FUNCTION DOC] Among the client's timeout fields, exactly the read field is unlimited.
        """A peer that stops draining the socket must not be able to block a caller forever."""
        timeout = GeneralHttpClientProvider.client().timeout

        assert [name for name, seconds in timeout.as_dict().items() if seconds is None] == ["read"]

    async def test_the_bounded_deadlines_are_positive_and_small_enough_to_fire(self):
        # [FUNCTION DOC] The connect, write and pool timeouts are each set, greater than zero and
        # at most 300 s.
        """A zero deadline fails every request and an astronomical one is no deadline at all."""
        deadlines = GeneralHttpClientProvider.client().timeout.as_dict()

        for name in ("connect", "write", "pool"):
            assert deadlines[name] is not None and 0 < deadlines[name] <= 300, name

    async def test_the_bounded_deadlines_track_the_class_constants(self):
        # [FUNCTION DOC] The client's connect, write and pool timeouts equal the provider class's
        # connect, write and pool timeout constants.
        """Retuning a timeout constant must move the client's deadline instead of silently desyncing."""
        timeout = GeneralHttpClientProvider.client().timeout

        assert timeout.connect == GeneralHttpClientProvider._CONNECT_TIMEOUT
        assert timeout.write == GeneralHttpClientProvider._WRITE_TIMEOUT
        assert timeout.pool == GeneralHttpClientProvider._POOL_TIMEOUT

    async def test_every_request_carries_the_bounded_deadlines_down_to_the_transport(self):
        # [FUNCTION DOC] Build a separate client with the shared client's timeout and a recording
        # transport, send one POST, and assert the transport saw exactly the timeout mapping, with
        # the write and pool entries set.
        """Deadlines bound a stuck peer only if httpx hands them to the transport on each request."""
        recorded = _RecordingTransport()
        timeout = GeneralHttpClientProvider.client().timeout

        async with httpx.AsyncClient(timeout=timeout, transport=recorded) as client:
            await client.post("http://weight-update.invalid/update_weights", json={})

        assert recorded.deadlines == timeout.as_dict()
        assert recorded.deadlines["write"] is not None
        assert recorded.deadlines["pool"] is not None

    async def test_a_client_built_on_another_loop_keeps_the_same_deadlines(self):
        # [FUNCTION DOC] Read the timeout mapping on this loop and on a new loop in a separate
        # thread (joined within 10 s); the two mappings are equal.
        """Every event loop gets a fresh client, and each one must carry the same timeout policy."""
        mine = GeneralHttpClientProvider.client().timeout.as_dict()
        theirs: list[dict[str, float | None]] = []

        async def _on_the_other_loop() -> dict[str, float | None]:
            # [FUNCTION DOC] HELPER: Return the timeout mapping of the client on the running loop.
            return GeneralHttpClientProvider.client().timeout.as_dict()

        thread = threading.Thread(target=lambda: theirs.append(asyncio.run(_on_the_other_loop())))
        thread.start()
        thread.join(timeout=10)

        assert theirs == [mine]


class _ArrivalGate:
    # [CLASS DOC] HELPER: Thread-safe gate used by the blocking test server: handlers register their
    # arrival (incrementing a shared count under a lock) and then block until released or 60 s.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Arrival count starts at zero, with a lock and an unset release event.
        self.count = 0
        self._lock = threading.Lock()
        self._released = threading.Event()

    def wait_for_release(self) -> None:
        # [FUNCTION DOC] HELPER: Increment the arrival count under the lock, then wait up to 60 s
        # for the release event (returning silently on timeout).
        with self._lock:
            self.count += 1
        self._released.wait(timeout=60)

    def release(self) -> None:
        # [FUNCTION DOC] HELPER: Set the release event, unblocking all waiters.
        self._released.set()


class _BlockingHttpServer:
    # [CLASS DOC] HELPER: Threaded local HTTP server on an OS-chosen loopback port. Each GET request
    # waits at the arrival gate and then answers 200 with an empty body. Serves in a daemon thread.
    def __init__(self, gate: _ArrivalGate):
        # [FUNCTION DOC] HELPER: Create the threading HTTP server with a handler bound to the gate,
        # store its health-generate URL, and start serving forever in a daemon thread.
        handler = self._make_handler(gate)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/health_generate"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        # [FUNCTION DOC] HELPER: Stop the serve loop and close the listening socket.
        self._server.shutdown()
        self._server.server_close()

    @staticmethod
    def _make_handler(gate: _ArrivalGate):
        # [FUNCTION DOC] HELPER: Build and return a request-handler class (HTTP/1.1) whose GET
        # handler waits at the given gate, then replies 200 with a zero content length, and whose
        # request logging is silenced.
        class _Handler(BaseHTTPRequestHandler):
            # [CLASS DOC] HELPER: Request handler answering GET after the gate releases.
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                # [FUNCTION DOC] HELPER: Wait at the gate, then send a 200 response with an empty
                # body.
                gate.wait_for_release()
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, format, *args):
                # [FUNCTION DOC] HELPER: Suppress request logging.
                pass

        return _Handler


class _RecordingTransport(httpx.AsyncBaseTransport):
    # [CLASS DOC] HELPER: HTTP transport that answers every request with an empty 200 and records the
    # timeout mapping that the client attached to the request.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: The recorded deadlines start as none.
        self.deadlines: dict[str, float | None] | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC] HELPER: Store the request's timeout extension and return an empty 200.
        self.deadlines = request.extensions["timeout"]
        return httpx.Response(200)


class TestDistributedPostActors:
    # [CLASS DOC] Tests of the Ray poster actor creation used for distributed HTTP posting.
    def test_the_poster_actor_is_constructed_with_keyword_arguments(self, monkeypatch):
        # [FUNCTION DOC] Fake the cluster node list (one alive node), the remote-wrapper function,
        # the module's actor list and a client concurrency of 7; run the distributed-post init with
        # two GPUs per node. Assert two actors were created, each with no positional arguments and
        # the keyword concurrency equal to 8 (a value the module derives from the client
        # concurrency and the actor count; the exact formula is not shown by this test).
        """A positional handoff silently binds to the wrong parameter once the actor grows another one."""
        recorded: list[tuple[tuple, dict]] = []

        class _FakeActorClass:
            # [CLASS DOC] HELPER: Fake remote actor class that records construction arguments.
            def options(self, **_options):
                # [FUNCTION DOC] HELPER: Ignore the options and return itself.
                return self

            def remote(self, *call_args, **call_kwargs):
                # [FUNCTION DOC] HELPER: Record the positional and keyword arguments and return a
                # dummy handle.
                recorded.append((call_args, call_kwargs))
                return object()

        monkeypatch.setattr(ray, "nodes", lambda: [{"NodeID": fake_ray_node_id(0), "Alive": True}])
        monkeypatch.setattr(ray, "remote", lambda _cls: _FakeActorClass())
        monkeypatch.setattr(http_utils, "_post_actors", [])
        monkeypatch.setattr(http_utils, "_client_concurrency", 7)

        http_utils._init_ray_distributed_post(SimpleNamespace(num_gpus_per_node=2))

        assert recorded == [((), {"concurrency": 8})] * 2


class _PosterActorInit(NamedTuple):
    # [CLASS DOC] HELPER: Result of running the distributed-post init under fakes: the actor class that
    # was wrapped for remote use, and the list of recorded (positional, keyword) construction calls.
    actor_class: type
    calls: list[tuple[tuple, dict]]


class _RecordingRemoteActorClass:
    # [CLASS DOC] HELPER: Fake remote actor wrapper that records construction call arguments and
    # ignores scheduling options.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Start with an empty call list.
        self.calls: list[tuple[tuple, dict]] = []

    def options(self, **_options: Any) -> "_RecordingRemoteActorClass":
        # [FUNCTION DOC] HELPER: Ignore the options and return itself.
        return self

    def remote(self, *call_args: Any, **call_kwargs: Any) -> object:
        # [FUNCTION DOC] HELPER: Record the arguments and return a dummy handle.
        self.calls.append((call_args, call_kwargs))
        return object()


def _run_init_ray_distributed_post(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_gpus_per_node: int = 1,
    client_concurrency: int = 7,
    nodes: list[dict] | None = None,
) -> _PosterActorInit:
    # [FUNCTION DOC] HELPER: Run the module's distributed-post initializer under fakes. Patches the
    # cluster node list (default: one alive node), the remote-wrapper function (capturing the
    # wrapped class and returning the recording wrapper), clears the module's actor list and sets
    # its client concurrency (default 7). Calls the initializer with the given GPUs per node
    # (default 1). Returns the captured actor class plus the recorded construction calls.
    captured: dict[str, type] = {}
    remote_actor_class = _RecordingRemoteActorClass()

    def _fake_remote(cls: type) -> _RecordingRemoteActorClass:
        # [FUNCTION DOC] HELPER: Capture the class being wrapped and return the recording wrapper.
        captured["actor_class"] = cls
        return remote_actor_class

    monkeypatch.setattr(ray, "nodes", lambda: nodes or [{"NodeID": fake_ray_node_id(0), "Alive": True}])
    monkeypatch.setattr(ray, "remote", _fake_remote)
    monkeypatch.setattr(http_utils, "_post_actors", [])
    monkeypatch.setattr(http_utils, "_client_concurrency", client_concurrency)

    http_utils._init_ray_distributed_post(SimpleNamespace(num_gpus_per_node=num_gpus_per_node))

    return _PosterActorInit(actor_class=captured["actor_class"], calls=remote_actor_class.calls)


class TestPosterActorKeywordOnlyConstruction:
    # [CLASS DOC] Tests of the poster actor's keyword-only constructor contract and its call site.
    def test_the_poster_actor_refuses_a_positional_concurrency(self, monkeypatch):
        # [FUNCTION DOC] Instantiating the captured actor class with a positional concurrency
        # raises a type error.
        """Constructing the poster actor positionally must fail so a later parameter cannot silently steal the slot."""
        actor_class = _run_init_ray_distributed_post(monkeypatch).actor_class

        with pytest.raises(TypeError):
            actor_class(7)

    def test_every_poster_actor_constructor_parameter_is_keyword_only(self, monkeypatch):
        # [FUNCTION DOC] Inspect the captured actor class's constructor signature (excluding the
        # instance parameter): every parameter is keyword-only, and there is at least one.
        """No poster actor constructor parameter may be positionally bindable."""
        actor_class = _run_init_ray_distributed_post(monkeypatch).actor_class

        parameters = list(inspect.signature(actor_class.__init__).parameters.values())[1:]

        assert [parameter.kind for parameter in parameters] == [inspect.Parameter.KEYWORD_ONLY] * len(parameters)
        assert parameters

    def test_the_recorded_poster_keywords_bind_to_the_actor_constructor(self, monkeypatch):
        # [FUNCTION DOC] For each recorded construction call, assert it has no positional arguments
        # and that its keywords bind successfully to the actor constructor signature.
        """The keywords the call site sends must name real poster actor constructor parameters."""
        init = _run_init_ray_distributed_post(monkeypatch)
        signature = inspect.signature(init.actor_class.__init__)

        for call_args, call_kwargs in init.calls:
            assert call_args == ()
            signature.bind(object(), *call_args, **call_kwargs)

    def test_the_poster_actor_is_constructed_by_keyword_on_every_node_slot(self, monkeypatch):
        # [FUNCTION DOC] With two nodes, two GPUs per node and client concurrency ten, assert four
        # actors are constructed, each with only the keyword concurrency of six.
        """Every actor created across nodes and per-node slots receives its concurrency by keyword."""
        init = _run_init_ray_distributed_post(
            monkeypatch,
            num_gpus_per_node=2,
            client_concurrency=10,
            nodes=[{"NodeID": fake_ray_node_id(0), "Alive": True}, {"NodeID": fake_ray_node_id(1), "Alive": True}],
        )

        assert init.calls == [((), {"concurrency": 6})] * 4


class TestInitHttpClientConcurrency:
    # [CLASS DOC] Tests of how the HTTP client initializer sizes its concurrency from GPU counts.
    def _args(self, **overrides):
        # [FUNCTION DOC] HELPER: Build an arguments namespace with defaults (no rollout or eval GPUs,
        # one GPU per engine, snapshot-based eval on, server concurrency 512, no distributed post)
        # overridden by the keyword arguments.
        defaults = dict(
            rollout_num_gpus=0,
            rollout_num_gpus_per_engine=1,
            eval_num_gpus=0,
            eval_num_gpus_per_engine=1,
            eval_uses_snapshots=True,
            sglang_server_concurrency=512,
            use_distributed_post=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _init(self, monkeypatch, args):
        # [FUNCTION DOC] HELPER: Reset the module's global client to none and concurrency to zero,
        # run the client initializer with the arguments, close the created client (if any) by
        # running its async close to completion, and return the pair of resulting concurrency and
        # client object.
        monkeypatch.setattr(http_utils, "_http_client", None)
        monkeypatch.setattr(http_utils, "_client_concurrency", 0)
        http_utils.init_http_client(args)
        if http_utils._http_client is not None:
            asyncio.run(http_utils._http_client.aclose())
        return http_utils._client_concurrency, http_utils._http_client

    def test_external_snapshot_eval_without_in_job_gpus_gets_one_engine_of_connections(self, monkeypatch):
        # [FUNCTION DOC] With zero GPUs everywhere and snapshot eval on, a client is created with
        # concurrency 512 (one engine's worth).
        concurrency, client = self._init(monkeypatch, self._args())
        assert client is not None
        assert concurrency == 512

    def test_train_only_eval_fleet_sizes_the_pool_from_the_fleet(self, monkeypatch):
        # [FUNCTION DOC] With rollout GPUs none, four eval GPUs and two eval GPUs per engine, the
        # concurrency is 512 times four divided by two (two engines' worth), with a client created.
        args = self._args(rollout_num_gpus=None, eval_num_gpus=4, eval_num_gpus_per_engine=2)
        concurrency, client = self._init(monkeypatch, args)
        assert client is not None
        assert concurrency == 512 * 4 // 2

    def test_in_job_gpu_sizing_is_unchanged(self, monkeypatch):
        # [FUNCTION DOC] With eight rollout GPUs, two per engine and snapshot eval off, the
        # concurrency is 512 times eight divided by two.
        args = self._args(rollout_num_gpus=8, rollout_num_gpus_per_engine=2, eval_uses_snapshots=False)
        concurrency, _ = self._init(monkeypatch, args)
        assert concurrency == 512 * 8 // 2

    def test_no_client_without_gpus_or_snapshot_eval(self, monkeypatch):
        # [FUNCTION DOC] With no GPUs and snapshot eval off, no client is created (none).
        _, client = self._init(monkeypatch, self._args(eval_uses_snapshots=False))
        assert client is None
