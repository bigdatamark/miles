# [MODULE DOC]
# End-to-end tests of the rpc client's retry policy, using a failure-injecting proxy in front of
# a real server and addresses nothing listens on. Retry timing constants of the client module are
# shrunk by a fixture. Covered: submits that reached the server are never retried (server errors,
# bad requests, unknown methods, dropped responses); submits that never reached it are retried
# only within a deadline, and a late-starting server is tolerated; readiness waits give up at
# their deadline and retry unhealthy responses; a handle pinned to a boot id keeps its pin on a
# gateway error without that header.
import asyncio
import time
from typing import Any

import pytest
from tests.fast.utils.workers.e2e.harness import wait_until_serving

from miles.utils.workers.rpc.client import call as client_module
from miles.utils.workers.rpc.client import handle as handle_module
from miles.utils.workers.rpc.client.misc import RpcProtocolError
from miles.utils.workers.worker_handle import WorkerUnreachableError


@pytest.fixture
def short_retry_window(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Fixture that shortens the client's retry parameters for the test
    # (restored afterwards): submit retry window 0.2 s, initial retry delay 10 ms, maximum retry
    # delay 20 ms, default poll timeout 50 ms.
    monkeypatch.setattr(client_module, "SUBMIT_RETRY_WINDOW_SECONDS", 0.2)
    monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "RETRY_MAX_DELAY_SECONDS", 0.02)
    monkeypatch.setattr(client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.05)


class _StaleWorker:
    # [CLASS DOC] HELPER: Worker class with a method the real server does not have, to simulate
    # a client that is newer than the server.
    def demo_removed_method(self) -> int:
        # [FUNCTION DOC] HELPER: Returns zero; never executed locally.
        return 0


class TestSubmitNoRetry:
    # [CLASS DOC] A submit that may have reached the server must be attempted exactly once.
    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    async def test_server_error_gives_up_after_one_attempt(
        self,
        proxy_to: Any,
        make_handle: Any,
        short_retry_window: None,
        tag: str,
        status_code: int,
    ) -> None:
        # [FUNCTION DOC] Parametrized over four server-error statuses. The proxy rejects the next
        # submit with that status; asserts the call raises a worker-unreachable error and the
        # proxy saw exactly one submit of the counting method.
        """A submit 5xx becomes unreachable after exactly one attempt."""
        proxy = await proxy_to()
        proxy.reject_next(count=1, status=status_code)
        handle = make_handle(proxy)

        with pytest.raises(WorkerUnreachableError):
            await handle.demo_count_sync(tag=tag)

        assert len(proxy.submits("demo_count_sync")) == 1

    async def test_bad_request_is_not_retried(
        self,
        proxy_to: Any,
        make_handle: Any,
        short_retry_window: None,
        tag: str,
    ) -> None:
        # [FUNCTION DOC] The proxy rejects one submit with status 400; asserts a protocol error
        # with status code 400 and exactly one recorded submit.
        """A submit 400 raises a protocol error after one attempt."""
        proxy = await proxy_to()
        proxy.reject_next(count=1, status=400)
        handle = make_handle(proxy)

        with pytest.raises(RpcProtocolError) as exc_info:
            await handle.demo_count_sync(tag=tag)

        assert exc_info.value.status_code == 400
        assert len(proxy.submits("demo_count_sync")) == 1

    async def test_unknown_method_is_not_retried(
        self,
        server: Any,
        make_handle: Any,
        short_retry_window: None,
    ) -> None:
        # [FUNCTION DOC] Calls a method only the stale client class has against the real server;
        # asserts a protocol error with status 404.
        """An unknown server method raises 404 after one submit."""
        handle = make_handle(server, worker_cls=_StaleWorker)

        with pytest.raises(RpcProtocolError) as exc_info:
            await handle.demo_removed_method()

        assert exc_info.value.status_code == 404

    async def test_post_wire_disconnect_is_not_retried(
        self,
        proxy_to: Any,
        make_handle: Any,
        short_retry_window: None,
        tag: str,
    ) -> None:
        # [FUNCTION DOC] The proxy forwards one submit but drops its response; asserts a
        # worker-unreachable error and exactly one recorded submit.
        """A dropped submit response becomes unreachable after one attempt."""
        proxy = await proxy_to()
        proxy.drop_next(count=1)
        handle = make_handle(proxy)

        with pytest.raises(WorkerUnreachableError):
            await handle.demo_count_sync(tag=tag)

        assert len(proxy.submits("demo_count_sync")) == 1


class TestNeverReachedRetry:
    # [CLASS DOC] A submit that never reached any server is retried until the submit deadline.
    async def test_unreachable_server_gives_up_after_retry_window(
        self,
        make_handle: Any,
        short_retry_window: None,
    ) -> None:
        # [FUNCTION DOC] Calls against a refused address with the 0.2 s window; asserts a
        # worker-unreachable error after an elapsed time of at least 0.1 s and under 1 s.
        """A refused connection retries only until the submit deadline."""
        handle = make_handle("http://127.0.0.1:9")
        started = time.monotonic()

        with pytest.raises(WorkerUnreachableError):
            await handle.demo_sync(a=1, b=1)

        elapsed = time.monotonic() - started
        assert elapsed >= 0.1
        assert elapsed < 1.0

    async def test_late_server_start_is_tolerated(
        self,
        spawn: Any,
        make_handle: Any,
        monkeypatch: pytest.MonkeyPatch,
        tag: str,
    ) -> None:
        # [FUNCTION DOC] Reserves a port, sets the submit window to the full ready timeout and
        # short retry delays, and starts a counting call against that port. A background task
        # waits 0.1 s then spawns a server on the port without waiting and polls (in a worker
        # thread) until it serves. Asserts the call succeeds with counter value 1, i.e. it was
        # retried until the server appeared and executed exactly once.
        """A submit succeeds when a server appears within its retry window."""
        from tests.fast.utils.workers.e2e.harness import READY_TIMEOUT_SECONDS, reserve_port

        monkeypatch.setattr(client_module, "SUBMIT_RETRY_WINDOW_SECONDS", READY_TIMEOUT_SECONDS)
        monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        monkeypatch.setattr(client_module, "RETRY_MAX_DELAY_SECONDS", 0.05)
        port = reserve_port()
        handle = make_handle(f"http://127.0.0.1:{port}")

        async def start_later() -> None:
            # [FUNCTION DOC] HELPER: Sleeps 0.1 s, spawns the server on the reserved port
            # without waiting, then waits for it to serve in a worker thread.
            await asyncio.sleep(0.1)
            server = spawn(port=port, wait=False)
            await asyncio.to_thread(wait_until_serving, server)

        starter = asyncio.create_task(start_later())
        result = await handle.demo_count_sync(tag=tag)
        await starter

        assert result == 1


class TestWaitReadyRetries:
    # [CLASS DOC] Retry behavior of the readiness wait.
    async def test_wait_ready_gives_up(self, make_handle: Any, short_retry_window: None) -> None:
        # [FUNCTION DOC] Waits for readiness with a 0.2 s timeout on a refused address; asserts
        # a worker-unreachable error within 1 s.
        """wait_ready stops at its own deadline."""
        handle = make_handle("http://127.0.0.1:9")
        started = time.monotonic()

        with pytest.raises(WorkerUnreachableError):
            await handle.wait_ready(timeout=0.2)

        assert time.monotonic() - started < 1.0

    async def test_wait_ready_retries_unhealthy_responses(
        self,
        proxy_to: Any,
        make_handle: Any,
        short_retry_window: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # [FUNCTION DOC] Also shortens the handle module's initial retry delay. The proxy
        # rejects the next two requests with status 503; asserts readiness still completes
        # within a 1 s timeout.
        """wait_ready retries transient health 503 responses."""
        monkeypatch.setattr(handle_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        proxy = await proxy_to()
        proxy.reject_next(count=2, status=503)
        handle = make_handle(proxy)

        await handle.wait_ready(timeout=1.0)


class TestPinnedSubmitFailure:
    # [CLASS DOC] Behavior of a handle pinned to one server boot id.
    async def test_headerless_gateway_error_does_not_repin(
        self,
        proxy_to: Any,
        make_handle: Any,
        short_retry_window: None,
        tag: str,
    ) -> None:
        # [FUNCTION DOC] Builds a handle requiring a stable boot id, waits until ready (pins),
        # makes the proxy reject the next request with 503 (no boot-id header), and asserts the
        # counting call raises a worker-unreachable error after exactly one recorded submit.
        """A pinned handle keeps its pin and gives up on a submit 503."""
        proxy = await proxy_to()
        handle = make_handle(proxy, require_stable_boot_uuid=True)
        await handle.wait_ready(timeout=1.0)
        proxy.reject_next(count=1, status=503)

        with pytest.raises(WorkerUnreachableError):
            await handle.demo_count_sync(tag=tag)

        assert len(proxy.submits("demo_count_sync")) == 1
