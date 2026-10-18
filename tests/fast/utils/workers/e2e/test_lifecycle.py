# [MODULE DOC]
# End-to-end lifecycle tests of the serve process and its clients: detection of a server restarted
# on the same port (with and without a handle pinned to the boot id), shutdown behavior on
# terminate and kill signals (idle exit, abandoning in-flight calls without any cancellation
# callback in the worker), port release and process reaping, and client behavior against a dead
# server. Each test spawns its own private servers.
import asyncio
import os
import signal
import time
import uuid

import httpx
import pytest
from tests.fast.utils.workers.e2e.harness import (
    READY_TIMEOUT_SECONDS,
    ServerProcess,
    port_is_refused,
    wait_until_serving,
)

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.client.misc import RpcProtocolError, ServerRestartedError
from miles.utils.workers.worker_handle import WorkerUnreachableError


async def _submit_and_disconnect(server: ServerProcess, method: str, query: dict) -> None:
    # [FUNCTION DOC] HELPER: Submits a call with a fresh random call id through a short-lived
    # http client (30 s timeout, environment proxies ignored) and asserts the submission was
    # accepted with status 200. The client is closed on return, so nobody polls the outcome.
    async with httpx.AsyncClient(base_url=server.url, timeout=30.0, trust_env=False) as client:
        response = await client.post(f"/v1/{method}", json={"call_id": uuid.uuid4().hex, "query": query})
        assert response.status_code == 200


async def _wait_until_started(handle: RpcWorkerHandle, tag: str) -> None:
    # [FUNCTION DOC] HELPER: Polls the worker's events up to 200 times at 50 ms intervals until a
    # start event for the tag exists; raises an assertion error otherwise.
    for _ in range(200):
        events = await handle.report_events()
        if any(event.tag == tag and event.phase == "start" for event in events):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"the worker never entered the body of the call tagged {tag}")


class TestRestartDetection:
    # [CLASS DOC] A server stopped and respawned on the same port is a different process with a
    # different boot id; handles pinned to the boot id must notice.
    async def test_pinned_handle_refuses_a_restarted_server(self, spawn, make_handle, tag):
        # [FUNCTION DOC] Pins a handle to the first server and makes one counting call (asserts
        # 1). Stops the server, spawns a new one on the port, and asserts the next call raises
        # a server-restarted error.
        """A pinned handle refuses to talk to a different process on the same port."""
        server = spawn()
        pinned = make_handle(server, require_stable_boot_uuid=True)
        assert await pinned.demo_count_sync(tag=tag) == 1

        server.stop()
        spawn(port=server.port)

        with pytest.raises(ServerRestartedError):
            await pinned.demo_count_sync(tag=tag)

    async def test_pinned_handle_does_not_rerun_on_the_new_server(self, spawn, make_handle, tag):
        # [FUNCTION DOC] Pins a handle (via readiness wait), restarts the server, asserts the
        # pinned call raises a server-restarted error, and then asserts through a fresh handle
        # that the new server's counter for the tag is zero (the call never reached it).
        """The refused call never reaches the restarted worker."""
        server = spawn()
        pinned = make_handle(server, require_stable_boot_uuid=True)
        await pinned.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        server.stop()
        restarted = spawn(port=server.port)

        with pytest.raises(ServerRestartedError):
            await pinned.demo_count_sync(tag=tag)

        fresh = make_handle(restarted)
        assert await fresh.report_counter(tag=tag) == 0

    async def test_unpinned_handle_sees_the_new_process(self, spawn, make_handle, tag):
        # [FUNCTION DOC] An unpinned handle counts once (asserts 1), the server is restarted,
        # and a second call returns 1 again because the new process starts from zero.
        """Without pinning, a new call after a restart simply goes to the new process."""
        server = spawn()
        handle = make_handle(server)
        assert await handle.demo_count_sync(tag=tag) == 1

        server.stop()
        spawn(port=server.port)

        assert await handle.demo_count_sync(tag=tag) == 1

    async def test_unpinned_handle_mid_call_restart_reports_unknown_call(self, spawn, make_handle, tag):
        # [FUNCTION DOC] Starts a gated async call with a long call timeout (twice the ready
        # timeout), waits 0.5 s, stops the server and spawns a replacement in a worker thread;
        # asserts the pending call fails with a protocol error (the new server does not know the
        # call id).
        """A call whose server was replaced mid-flight fails loudly instead of hanging."""
        server = spawn()
        handle = make_handle(server, call_timeout_seconds=2 * READY_TIMEOUT_SECONDS)

        pending = asyncio.create_task(handle.demo_block_async(tag=tag))
        await asyncio.sleep(0.5)

        server.stop()
        await asyncio.to_thread(spawn, port=server.port)

        with pytest.raises(RpcProtocolError):
            await pending

    async def test_pinned_handle_created_after_restart_works(self, spawn, make_handle, tag):
        # [FUNCTION DOC] Spawns, stops and respawns a server on the same port, then pins a new
        # handle to the restarted server; asserts a counting call returns 1.
        """Pinning after the restart is the supported recovery path."""
        server = spawn()
        server.stop()
        restarted = spawn(port=server.port)

        pinned = make_handle(restarted, require_stable_boot_uuid=True)
        assert await pinned.demo_count_sync(tag=tag) == 1


class TestShutdown:
    # [CLASS DOC] Behavior of the server process under signals and after stopping.
    async def test_idle_server_exits_promptly_on_sigterm(self, spawn):
        # [FUNCTION DOC] Sends a terminate signal to an idle server; asserts it exits within
        # 15 s of waiting and the whole thing took under 5 s.
        """An idle server shuts down quickly and cleanly."""
        server = spawn()
        started = time.monotonic()
        server.signal(signal.SIGTERM)

        assert server.wait(timeout=15.0) is not None
        assert time.monotonic() - started < 5.0

    async def test_sigterm_abandons_an_in_flight_async_call(self, spawn, state_dir):
        # [FUNCTION DOC] Submits a call that would create the marker file "abandoned" after
        # 2 s, terminates the server, asserts it exits within 30 s, waits 2.5 s, and asserts the
        # marker file does not exist (the in-flight call was cancelled).
        """An accepted call is cancelled rather than allowed to finish."""
        server = spawn()
        await _submit_and_disconnect(server, "demo_marker_after_sleep_async", {"name": "abandoned", "seconds": 2.0})

        server.signal(signal.SIGTERM)
        assert server.wait(timeout=30.0) is not None
        await asyncio.sleep(2.5)
        assert not (state_dir / "abandoned").exists()

    @pytest.mark.parametrize("stop", ["sigterm", "sigkill"])
    async def test_the_worker_gets_no_cancellation_callback(self, spawn, state_dir, tag, stop: str):
        # [FUNCTION DOC] Parametrized over terminate and kill. Submits a call that waits forever
        # and writes a marker on cancellation, stops the server accordingly (a terminate signal
        # or a kill of the process group), asserts it exits, and asserts the cancellation marker
        # file for the tag was never created.
        """There is no graceful shutdown, so an in-flight worker never observes cancellation."""
        server = spawn()
        await _submit_and_disconnect(server, "demo_block_until_cancelled", {"tag": tag})

        if stop == "sigterm":
            server.signal(signal.SIGTERM)
        else:
            server.kill()
        assert server.wait(timeout=30.0) is not None
        assert not (state_dir / f"cancelled_{tag}").exists()

    async def test_sigterm_after_the_worker_started_does_not_run_its_cancellation_callback(
        self, spawn, make_handle, state_dir, tag
    ):
        # [FUNCTION DOC] Like the previous test but first waits until the worker reports the
        # call started (via a handle), then terminates the server, waits for exit, sleeps 0.5 s
        # and asserts the cancellation marker file was not created.
        """Even once the worker body is provably running, SIGTERM hands it no cancellation callback."""
        server = spawn()
        await _submit_and_disconnect(server, "demo_block_until_cancelled", {"tag": tag})
        await _wait_until_started(make_handle(server), tag)

        server.signal(signal.SIGTERM)
        assert server.wait(timeout=30.0) is not None

        await asyncio.sleep(0.5)
        assert not (state_dir / f"cancelled_{tag}").exists()

    async def test_port_is_released_for_the_next_process(self, spawn):
        # [FUNCTION DOC] Stops a server and asserts its port refuses connections; spawns a
        # successor on the same port and waits until it serves.
        """A stopped server frees its port for an immediate successor."""
        server = spawn()
        server.stop()
        assert port_is_refused(server.port)

        successor = spawn(port=server.port)
        wait_until_serving(successor)

    async def test_no_process_survives_teardown(self, spawn):
        # [FUNCTION DOC] Stops a server and asserts that probing its process id with the null
        # signal raises a process-lookup error, i.e. the process was reaped.
        """Stopping the server really reaps the process."""
        server = spawn()
        pid = server.process.pid
        server.stop()

        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


class TestClientAgainstDeadServer:
    # [CLASS DOC] Error types a client raises when the server dies or never existed.
    async def test_call_submitted_then_server_killed_times_out(
        self,
        spawn,
        make_handle,
        monkeypatch: pytest.MonkeyPatch,
        tag,
    ) -> None:
        # [FUNCTION DOC] Shortens the client's retry delay (10 ms) and poll timeout (50 ms),
        # builds a handle with a 0.3 s call budget, waits for readiness, starts a gated async
        # call, kills the server after 0.05 s, and asserts the call fails with a timeout error.
        """Poll failures after server death exhaust as TimeoutError."""
        from miles.utils.workers.rpc.client import call as client_module

        monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        monkeypatch.setattr(client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.05)
        server = spawn()
        handle = make_handle(server, call_timeout_seconds=0.3)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        pending = asyncio.create_task(handle.demo_block_async(tag=tag))
        await asyncio.sleep(0.05)
        server.kill()

        with pytest.raises(TimeoutError):
            await pending

    async def test_call_to_a_dead_port_raises_plain_unreachable(
        self,
        make_handle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # [FUNCTION DOC] Shortens the submit retry window to 0.2 s and retry delays; asserts a
        # call to a refused address raises a worker-unreachable error.
        """Never reaching the server raises plain WorkerUnreachableError."""
        from miles.utils.workers.rpc.client import call as client_module

        monkeypatch.setattr(client_module, "SUBMIT_RETRY_WINDOW_SECONDS", 0.2)
        monkeypatch.setattr(client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.01)
        monkeypatch.setattr(client_module, "RETRY_MAX_DELAY_SECONDS", 0.02)

        handle = make_handle("http://127.0.0.1:9")
        with pytest.raises(WorkerUnreachableError):
            await handle.demo_sync(a=1, b=1)
