# [MODULE DOC] HELPER:
# Pytest fixtures for the end-to-end worker tests: per-test and session-shared serve processes,
# a factory for rpc worker handles with their own http clients, a raw http client, and proxies
# that sit between client and server to inject or record failures. Also installs a report hook
# that stores each phase's result on the test item so teardown can print server logs of failed
# tests.
from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import httpx
import pytest
from tests.fast.utils.workers.e2e.e2e_worker import E2eWorker
from tests.fast.utils.workers.e2e.harness import (
    READY_TIMEOUT_SECONDS,
    FlakyProxy,
    ServerProcess,
    spawn_server,
    wait_until_serving,
)

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle


@pytest.fixture
def state_dir(tmp_path) -> Path:
    # [FUNCTION DOC] HELPER: Fixture. Creates and returns an empty "state" directory under the
    # test's temporary path; the test worker writes marker files there.
    path = tmp_path / "state"
    path.mkdir()
    return path


@pytest.fixture
def tag() -> str:
    # [FUNCTION DOC] HELPER: Fixture. Returns a fresh random 8-character hex string used to
    # label calls and keep tests independent on a shared server.
    return uuid.uuid4().hex[:8]


@pytest.fixture
def spawn(state_dir, tmp_path, request) -> Iterator[Callable[..., ServerProcess]]:
    # [FUNCTION DOC] HELPER: Fixture providing a function that launches a private serve process.
    # On teardown every server started is stopped gracefully then force-killed; for a test whose
    # call phase failed, the last 4000 characters of each server log are printed.
    started: list[ServerProcess] = []

    def start(*, wait: bool = True, **kwargs) -> ServerProcess:
        # [FUNCTION DOC] HELPER: Launches a server with a numbered log file in the test's
        # temporary directory, forwarding extra keyword arguments to the spawn helper, and
        # remembers it for teardown. When wait is true (default) blocks until it serves health
        # checks (fails an assertion if it exits or times out). Returns the server process.
        log_path = tmp_path / f"server-{len(started)}.log"
        server = spawn_server(state_dir=state_dir, log_path=log_path, **kwargs)
        started.append(server)
        if wait:
            wait_until_serving(server)
        return server

    yield start

    for server in started:
        server.stop()
        server.kill()
        if request.node.rep_call is not None and request.node.rep_call.failed:
            print(f"\n--- server log {server.log_path.name} ---\n{server.logs()[-4000:]}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # [FUNCTION DOC] HELPER: Hook wrapper run first. After the report for a phase is built, stores
    # it on the test item under a per-phase attribute (setup, call, teardown) so fixtures can
    # inspect whether the call phase failed.
    outcome = yield
    setattr(item, f"rep_{call.when}", outcome.get_result())


@pytest.fixture(autouse=True)
def _reset_reports(request):
    # [FUNCTION DOC] HELPER: Autouse fixture. Ensures each phase report attribute exists on the
    # test item (set to nothing if missing) so fixtures can read it even if a phase never ran.
    for phase in ("setup", "call", "teardown"):
        if not hasattr(request.node, f"rep_{phase}"):
            setattr(request.node, f"rep_{phase}", None)


@pytest.fixture(scope="session")
def shared_server(tmp_path_factory) -> Iterator[ServerProcess]:
    # [FUNCTION DOC] HELPER: Session-scoped fixture. Starts one serve process (own temporary
    # directory, state directory and log), waits until it serves, yields it, and stops then
    # kills it at the end of the session. Tests that stop the server must spawn their own.
    directory = tmp_path_factory.mktemp("shared-server")
    state_dir = directory / "state"
    state_dir.mkdir()

    server = spawn_server(state_dir=state_dir, log_path=directory / "server.log")
    wait_until_serving(server)
    yield server

    server.stop()
    server.kill()


@pytest.fixture
async def server(shared_server, request) -> AsyncIterator[ServerProcess]:
    # [FUNCTION DOC] HELPER: Per-test fixture over the shared server. Asserts the shared server is
    # still alive (message includes its log tail), yields it, and afterwards asks the test worker
    # to release every blocking gate (any error ignored) so a blocked call cannot leak into the
    # next test. If the test's call phase failed, prints the shared server's log tail.
    assert shared_server.is_running(), (
        f"the shared server died in an earlier test; a test that stops it must spawn its own:\n"
        f"{shared_server.logs()[-4000:]}"
    )

    yield shared_server

    async with httpx.AsyncClient(base_url=shared_server.url, timeout=30.0, trust_env=False) as client:
        with contextlib.suppress(Exception):
            await RpcWorkerHandle(E2eWorker, server_url=shared_server.url, http_client=client).release_every_gate()

    if request.node.rep_call is not None and request.node.rep_call.failed:
        print(f"\n--- shared server log tail ---\n{shared_server.logs()[-4000:]}")


@pytest.fixture
async def make_handle() -> AsyncIterator[Callable[..., RpcWorkerHandle]]:
    # [FUNCTION DOC] HELPER: Fixture providing a handle factory. Every client it creates is closed
    # at teardown.
    clients: list[httpx.AsyncClient] = []

    def build(target: ServerProcess | FlakyProxy | str, *, worker_cls: type = E2eWorker, **kwargs) -> RpcWorkerHandle:
        # [FUNCTION DOC] HELPER: Builds an rpc worker handle for the worker class (default the
        # test worker). The target is a url string or an object with a url (server or proxy).
        # Uses a new http client (60 s timeout, 10 s connect timeout, environment proxy settings
        # ignored) tracked for closing; extra keyword arguments go to the handle constructor.
        url = target if isinstance(target, str) else target.url
        client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0), trust_env=False)
        clients.append(client)
        return RpcWorkerHandle(worker_cls, server_url=url, http_client=client, **kwargs)

    yield build

    for client in clients:
        await client.aclose()


@pytest.fixture
async def handle(server, make_handle) -> AsyncIterator[RpcWorkerHandle]:
    # [FUNCTION DOC] HELPER: Fixture yielding a handle to the shared server, after waiting up to
    # the ready timeout for the worker to report readiness.
    worker_handle = make_handle(server)
    await worker_handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
    yield worker_handle


@pytest.fixture
async def raw(server) -> AsyncIterator[httpx.AsyncClient]:
    # [FUNCTION DOC] HELPER: Fixture yielding a plain http client with the shared server's url as
    # base url, a 30 s timeout and environment proxy settings ignored; closed at teardown.
    async with httpx.AsyncClient(base_url=server.url, timeout=30.0, trust_env=False) as client:
        yield client


@pytest.fixture
async def proxy_to(server) -> AsyncIterator[Callable[[], FlakyProxy]]:
    # [FUNCTION DOC] HELPER: Fixture providing an async factory of failure-injecting proxies in
    # front of the shared server. All proxies created are stopped at teardown.
    proxies: list[FlakyProxy] = []

    async def build() -> FlakyProxy:
        # [FUNCTION DOC] HELPER: Creates a proxy forwarding to the shared server's port, starts
        # it on an ephemeral local port, registers it for teardown and returns it.
        proxy = FlakyProxy(upstream_port=server.port)
        await proxy.start()
        proxies.append(proxy)
        return proxy

    yield build

    for proxy in proxies:
        await proxy.stop()


@pytest.fixture
async def dead_proxy() -> AsyncIterator[FlakyProxy]:
    # [FUNCTION DOC] HELPER: Fixture yielding a started proxy with no upstream in record-only
    # mode (it records each request and answers 503), stopped at teardown.
    proxy = FlakyProxy(upstream_port=None)
    proxy.record_only = True
    await proxy.start()
    yield proxy
    await proxy.stop()
