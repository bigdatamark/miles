# [MODULE DOC]
# End-to-end tests of transport-level behavior: the failure-injecting proxy's own shutdown, what
# happens when clients disconnect or abandon long polls, connection pooling and reuse (counted
# with a relay), many short connections, a slow poller not blocking others, and load bursts of
# concurrent calls.
import asyncio
import time

import httpx
import pytest
from tests.fast.utils.workers.e2e.e2e_worker import E2eWorker
from tests.fast.utils.workers.e2e.harness import ConnectionCountingRelay, FlakyProxy

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle


class TestConnectionBehaviour:
    # [CLASS DOC] Connection-level behavior of the client, server and test proxy.
    async def test_proxy_stop_closes_idle_connections(self):
        # [FUNCTION DOC] Starts a proxy without upstream, opens an idle connection to it, and
        # asserts that stopping the proxy completes within 1 s and the client side then reads
        # end of stream (empty bytes) within 1 s. Cleans up the connection and stops the proxy
        # again.
        proxy = FlakyProxy(upstream_port=None)
        await proxy.start()
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
        try:
            await asyncio.wait_for(proxy.stop(), timeout=1.0)
            assert await asyncio.wait_for(reader.read(), timeout=1.0) == b""
        finally:
            writer.close()
            await writer.wait_closed()
            await proxy.stop()

    async def test_client_disconnect_does_not_stop_the_call(self, server, make_handle, tag):
        # [FUNCTION DOC] Submits a 2 s counting sleep over a short-lived client that is closed
        # right after the submission, waits 3 s, and asserts a fresh handle sees the counter
        # equal to 1.
        """Dropping the submitting connection leaves the accepted call running."""
        async with httpx.AsyncClient(base_url=server.url, timeout=5.0, trust_env=False) as client:
            await client.post(
                "/v1/demo_count_after_sleep", json={"call_id": tag, "query": {"tag": tag, "seconds": 2.0}}
            )

        await asyncio.sleep(3.0)
        assert await make_handle(server).report_counter(tag=tag) == 1

    async def test_abandoned_long_poll_does_not_break_the_server(self, server, handle, raw, tag):
        # [FUNCTION DOC] Submits a gated call, then polls with a 30 s server wait but a 1 s
        # client timeout; asserts the client times out. Releases the gate and asserts the health
        # endpoint still answers 200.
        """A long poll the client walks away from leaves the server healthy."""
        await raw.post("/v1/demo_block_async", json={"call_id": tag, "query": {"tag": tag}})

        async with httpx.AsyncClient(base_url=server.url, timeout=1.0, trust_env=False) as impatient:
            with pytest.raises(httpx.TimeoutException):
                await impatient.get(f"/v1/calls/{tag}", params={"timeout": 30.0})

        await handle.release(tag=tag)
        assert (await raw.get("/v1/health")).status_code == 200

    async def test_connections_are_reused_across_calls(self, handle):
        # [FUNCTION DOC] Makes twenty sequential sum calls on one handle and asserts each is 2.
        # Despite the docstring, connection reuse itself is not observed here (the next test
        # measures it).
        # KNOWN ISSUE: the test name and docstring promise connection reuse, but only call
        # correctness is asserted.
        """Sequential calls reuse the pooled connection instead of reconnecting each time."""
        for _ in range(20):
            assert await handle.demo_sync(a=1, b=1) == 2

    async def test_sequential_calls_reuse_the_same_http_connection(self, server):
        # [FUNCTION DOC] Places a connection-counting relay in front of the server, creates a
        # handle through one pooled http client and makes five sequential sum calls (each
        # asserted to be 2). The relay is always stopped. Asserts the relay accepted exactly one
        # connection.
        """Sequential calls travel over one pooled TCP connection instead of reconnecting per request."""
        relay = ConnectionCountingRelay(upstream_port=server.port)
        await relay.start()

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0), trust_env=False) as client:
                handle = RpcWorkerHandle(E2eWorker, server_url=relay.url, http_client=client)
                for _ in range(5):
                    assert await handle.demo_sync(a=1, b=1) == 2
        finally:
            await relay.stop()

        assert relay.accepted == 1

    async def test_server_handles_many_short_connections(self, server):
        # [FUNCTION DOC] Twenty times, creates a new http client, fetches the health endpoint
        # and asserts status 200.
        """Opening a fresh connection per request stays healthy."""
        for _ in range(20):
            async with httpx.AsyncClient(base_url=server.url, timeout=10.0, trust_env=False) as client:
                assert (await client.get("/v1/health")).status_code == 200

    async def test_slow_client_read_does_not_block_other_clients(self, server, handle, raw, tag):
        # [FUNCTION DOC] Submits a gated call and keeps a 10 s long poll open in the
        # background; after 0.3 s asserts a sum call returns 2 in under 5 s. Releases the gate
        # and awaits the background poll.
        """One client holding a long poll open does not stall unrelated requests."""
        await raw.post("/v1/demo_block_async", json={"call_id": tag, "query": {"tag": tag}})
        slow = asyncio.create_task(raw.get(f"/v1/calls/{tag}", params={"timeout": 10.0}))
        await asyncio.sleep(0.3)

        started = time.monotonic()
        assert await handle.demo_sync(a=1, b=1) == 2
        assert time.monotonic() - started < 5.0

        await handle.release(tag=tag)
        await slow


class TestLoad:
    # [CLASS DOC] Bursts of concurrent calls.
    async def test_burst_of_calls_all_complete(self, handle):
        # [FUNCTION DOC] Fires 100 concurrent sum calls (i plus 1); asserts each result is i
        # plus 1 in order.
        """A burst of a hundred concurrent calls all complete correctly."""
        results = await asyncio.gather(*[handle.demo_sync(a=i, b=1) for i in range(100)])
        assert results == [i + 1 for i in range(100)]

    async def test_burst_does_not_leak_call_records(self, handle, raw, tag):
        # [FUNCTION DOC] Fires 50 concurrent counting calls with distinct tags, then asserts the
        # health endpoint answers 200 and a sum call still works. It cannot actually observe
        # record counts.
        """The server keeps answering after a burst, so its bookkeeping survived it."""
        await asyncio.gather(*[handle.demo_count_sync(tag=f"{tag}{i}") for i in range(50)])
        assert (await raw.get("/v1/health")).status_code == 200
        assert await handle.demo_sync(a=1, b=1) == 2

    async def test_mixed_sync_and_async_burst(self, handle, tag):
        # [FUNCTION DOC] Fires 20 instant async and 20 instant sync calls interleaved, all
        # concurrently; asserts exactly 20 report the main thread and exactly 20 report a
        # thread name starting with "rpc-".
        """Mixing executor-bound and loop-bound calls under load stays correct."""
        calls = []
        for i in range(20):
            calls.append(handle.demo_instant_async(tag=f"{tag}a{i}"))
            calls.append(handle.demo_instant_sync(tag=f"{tag}s{i}"))

        names = await asyncio.gather(*calls)
        assert sum(1 for name in names if name == "MainThread") == 20
        assert sum(1 for name in names if name.startswith("rpc-")) == 20
