# [MODULE DOC]
# End-to-end tests of how worker-side failures surface to callers: exceptions in sync and async
# methods become rpc worker-call errors carrying the remote traceback (the http exchange itself
# stays successful with a failed outcome); the server and executors stay usable afterwards;
# results that cannot be serialized or violate the declared type fail the call; and a system-exit
# request raised in a worker (sync or async) terminates the server process, so callers fail with
# unreachable or timeout errors by their own deadline.
import asyncio

import pytest

from miles.utils.workers.rpc.client.misc import RpcWorkerCallError
from miles.utils.workers.worker_handle import WorkerUnreachableError


async def _submit(raw, method: str, call_id: str, query: dict):
    # [FUNCTION DOC] HELPER: Posts a submission of the method with the call id and query mapping
    # through the raw client and returns the http response unchecked.
    return await raw.post(f"/v1/{method}", json={"call_id": call_id, "query": query})


async def _poll_until_done(raw, call_id: str):
    # [FUNCTION DOC] HELPER: Polls the call with a 2 s server-side wait up to 50 times; returns
    # the decoded body as soon as its status is not pending. Raises an assertion error if the
    # call never finishes.
    for _ in range(50):
        body = (await raw.get(f"/v1/calls/{call_id}", params={"timeout": 2.0})).json()
        if body["status"] != "pending":
            return body
    raise AssertionError("call never finished")


class TestWorkerExceptions:
    # [CLASS DOC] Exceptions raised by worker methods.
    async def test_sync_exception_raises_call_error(self, handle):
        # [FUNCTION DOC] A sync method raises; asserts the client raises a call error whose text
        # names the method.
        """A sync worker exception surfaces as a call error on the client."""
        with pytest.raises(RpcWorkerCallError) as exc_info:
            await handle.demo_sync_raises(message="sync-boom")
        assert "demo_sync_raises" in str(exc_info.value)

    async def test_async_exception_raises_call_error(self, handle):
        # [FUNCTION DOC] An async method raises; asserts the client raises a call error.
        """An async worker exception surfaces the same way."""
        with pytest.raises(RpcWorkerCallError):
            await handle.demo_async_raises(message="async-boom")

    async def test_error_carries_the_remote_traceback(self, handle):
        # [FUNCTION DOC] A sync method raises with message "needle"; asserts the error text
        # contains the error line, the standard traceback header, and the worker source file
        # name.
        """The client sees the worker's traceback, not just a repr."""
        with pytest.raises(RpcWorkerCallError) as exc_info:
            await handle.demo_sync_raises(message="needle")

        text = str(exc_info.value)
        assert "ValueError: needle" in text
        assert "Traceback (most recent call last)" in text
        assert "e2e_worker.py" in text

    async def test_worker_failure_is_http_200(self, raw, tag):
        # [FUNCTION DOC] Submits a raising call over raw http; asserts the submission status is
        # 200, then that the finished body has status failed, no result, and an error text
        # mentioning the error class.
        """A failing worker call is a successful HTTP exchange with a failed envelope."""
        response = await _submit(raw, "demo_sync_raises", tag, {"message": "x"})
        assert response.status_code == 200

        body = await _poll_until_done(raw, tag)
        assert body["status"] == "failed" and body["result"] is None
        assert "ValueError" in body["error"]

    async def test_failure_does_not_poison_the_server(self, handle):
        # [FUNCTION DOC] After a raising call (asserted to fail), a sum call returns 3.
        """The server keeps serving after a worker raises."""
        with pytest.raises(RpcWorkerCallError):
            await handle.demo_sync_raises(message="x")
        assert await handle.demo_sync(a=1, b=2) == 3

    async def test_failure_does_not_poison_its_executor(self, handle, tag):
        # [FUNCTION DOC] After a raising sync call, an instant sync call still runs on an
        # "rpc-" executor thread.
        """The executor thread of a failed sync call keeps working."""
        with pytest.raises(RpcWorkerCallError):
            await handle.demo_sync_raises(message="x")
        assert (await handle.demo_instant_sync(tag=tag)).startswith("rpc-")


class TestResultContract:
    # [CLASS DOC] Results the server cannot return turn into failures of that call only.
    async def test_unserializable_result_becomes_a_failure(self, handle):
        # [FUNCTION DOC] The method returns a plain object; asserts a call error.
        """A result the server cannot serialize fails the call instead of stranding it."""
        with pytest.raises(RpcWorkerCallError):
            await handle.demo_unserializable_result()

    async def test_result_type_mismatch_fails_on_the_server(self, handle):
        # [FUNCTION DOC] The method declares an integer result but returns a string; asserts a
        # call error whose text names the method.
        """A result that does not match the annotation fails in the process that produced it."""
        with pytest.raises(RpcWorkerCallError, match="demo_wrong_result_type"):
            await handle.demo_wrong_result_type()

    async def test_server_survives_a_mismatched_result(self, handle):
        # [FUNCTION DOC] After a mismatched-result failure, a sum call (2+2) returns 4.
        """A failed result serialization leaves the server usable."""
        with pytest.raises(RpcWorkerCallError):
            await handle.demo_wrong_result_type()
        assert await handle.demo_sync(a=2, b=2) == 4


class TestNonExceptionExits:
    # [CLASS DOC] A system-exit request is not an ordinary exception: it ends the server. Each
    # test uses a private server and shortened client retry timings.
    async def test_systemexit_takes_the_server_down_and_fails_the_call(
        self,
        spawn,
        make_handle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # [FUNCTION DOC] Shortens retry delay (10 ms) and poll timeout (50 ms), spawns a server,
        # builds a handle with a 0.3 s call budget and calls the sync method that requests
        # system exit; asserts a worker-unreachable or timeout error, and that the server
        # process exits within 10 s.
        """SystemExit kills the server; the caller fails either at submit or at the poll deadline."""
        from miles.utils.workers.rpc.client import call as client_module

        monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        monkeypatch.setattr(client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.05)
        server = spawn()
        handle = make_handle(server, call_timeout_seconds=0.3)

        with pytest.raises((WorkerUnreachableError, TimeoutError)):
            await handle.demo_system_exit()

        assert server.wait(timeout=10.0) is not None

    async def test_async_systemexit_takes_the_server_down_and_fails_the_call(
        self,
        spawn,
        make_handle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # [FUNCTION DOC] Same as the sync variant but with the async method requesting system
        # exit: expects a worker-unreachable or timeout error and the server exiting within 10 s.
        """SystemExit from an async method escapes the event loop too, killing the server and the call."""
        from miles.utils.workers.rpc.client import call as client_module

        monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        monkeypatch.setattr(client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.05)
        server = spawn()
        handle = make_handle(server, call_timeout_seconds=0.3)

        with pytest.raises((WorkerUnreachableError, TimeoutError)):
            await handle.demo_system_exit_async()

        assert server.wait(timeout=10.0) is not None

    async def test_calls_in_flight_when_worker_exits_finish_by_deadline(
        self,
        spawn,
        make_handle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # [FUNCTION DOC] Shortens client timings, spawns a server, starts a 30 s async sleep
        # call on a handle with a 0.3 s budget, then calls the system-exit method (expects
        # unreachable or timeout). Asserts the in-flight sleeping call ends with a timeout
        # error at its own deadline.
        """A concurrent accepted call dies with the server and reaches its own deadline."""
        from miles.utils.workers.rpc.client import call as client_module

        monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        monkeypatch.setattr(client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.05)
        handle = make_handle(spawn(), call_timeout_seconds=0.3)
        other = asyncio.create_task(handle.demo_sleep_async(tag="doomed", seconds=30.0))
        await asyncio.sleep(0.05)

        with pytest.raises((WorkerUnreachableError, TimeoutError)):
            await handle.demo_system_exit()

        with pytest.raises(TimeoutError):
            await other
