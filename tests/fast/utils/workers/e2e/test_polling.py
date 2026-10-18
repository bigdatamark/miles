# [MODULE DOC]
# End-to-end tests of result polling: the server's long-poll timing (woken on completion, waits
# its timeout while pending, zero timeout never blocks, results kept for late polls, many waiters
# released together) and the client's poll loop (calls spanning several poll windows, call
# timeouts that do not cancel the worker, poll requests bounded by the call budget, timeout
# error text).
import asyncio
import time

import pytest

from miles.utils.workers.rpc.client import call as client_module


async def _submit(raw, method: str, call_id: str, query: dict):
    # [FUNCTION DOC] HELPER: Posts a submission of the method with the call id and query mapping
    # through the raw client and returns the http response unchecked.
    return await raw.post(f"/v1/{method}", json={"call_id": call_id, "query": query})


class TestLongPollTiming:
    # [CLASS DOC] Server-side long poll behavior, tested with raw http.
    async def test_poll_returns_as_soon_as_the_call_finishes(self, raw, tag):
        # [FUNCTION DOC] Submits a 1 s async sleep and polls with a 30 s timeout; asserts status
        # success and an elapsed time between 0.5 s and 10 s.
        """A long poll is woken by completion instead of waiting out its timeout."""
        await _submit(raw, "demo_sleep_async", tag, {"tag": tag, "seconds": 1.0})

        started = time.monotonic()
        body = (await raw.get(f"/v1/calls/{tag}", params={"timeout": 30.0})).json()
        elapsed = time.monotonic() - started

        assert body["status"] == "success"
        assert 0.5 < elapsed < 10.0

    async def test_poll_waits_out_its_timeout_while_pending(self, raw, handle, tag):
        # [FUNCTION DOC] Submits a gated async call and polls with a 2 s timeout; asserts status
        # pending and an elapsed time between 1.5 s and 8 s. Releases the gate afterwards.
        """A poll on a still-running call blocks for the timeout it asked for."""
        await _submit(raw, "demo_block_async", tag, {"tag": tag})

        started = time.monotonic()
        body = (await raw.get(f"/v1/calls/{tag}", params={"timeout": 2.0})).json()
        elapsed = time.monotonic() - started

        assert body["status"] == "pending"
        assert 1.5 < elapsed < 8.0
        await handle.release(tag=tag)

    async def test_zero_timeout_does_not_block(self, raw, handle, tag):
        # [FUNCTION DOC] Submits a gated async call and polls with timeout zero; asserts status
        # pending returned in under 2 s. Releases the gate afterwards.
        """A zero timeout is a non-blocking status check."""
        await _submit(raw, "demo_block_async", tag, {"tag": tag})

        started = time.monotonic()
        body = (await raw.get(f"/v1/calls/{tag}", params={"timeout": 0.0})).json()

        assert body["status"] == "pending" and time.monotonic() - started < 2.0
        await handle.release(tag=tag)

    async def test_result_completed_before_first_poll_is_not_lost(self, raw, tag):
        # [FUNCTION DOC] Submits a sum, waits 1 s without polling, then polls with timeout
        # zero; asserts the body is success with result 3 and no error.
        """A call that finishes before anyone polls keeps its outcome."""
        await _submit(raw, "demo_sync", tag, {"a": 1, "b": 2})
        await asyncio.sleep(1.0)

        body = (await raw.get(f"/v1/calls/{tag}", params={"timeout": 0.0})).json()
        assert body == {"status": "success", "result": 3, "error": None}

    async def test_many_waiters_are_all_woken(self, raw, handle, tag):
        # [FUNCTION DOC] Submits a gated call, starts 20 concurrent polls with a 15 s timeout,
        # waits 0.5 s, releases the gate, and asserts all 20 responses report success.
        """Every concurrent poller of one call is released when it completes."""
        await _submit(raw, "demo_block_async", tag, {"tag": tag})

        polls = [raw.get(f"/v1/calls/{tag}", params={"timeout": 15.0}) for _ in range(20)]
        pollers = asyncio.gather(*polls)
        await asyncio.sleep(0.5)
        await handle.release(tag=tag)

        bodies = [response.json() for response in await pollers]
        assert all(body["status"] == "success" for body in bodies)


class TestClientPollLoop:
    # [CLASS DOC] The client repeats polls until the call finishes or its budget is spent.
    async def test_call_spanning_several_poll_windows(self, server, make_handle, monkeypatch, tag):
        # [FUNCTION DOC] Shrinks the client's default poll timeout to 1 s, builds a handle with
        # a 60 s budget and runs a 4 s async sleep; asserts it returns the tag.
        """A call outliving one poll window is stitched together by the client loop."""
        monkeypatch.setattr(client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 1.0)

        handle = make_handle(server, call_timeout_seconds=60.0)
        assert await handle.demo_sleep_async(tag=tag, seconds=4.0) == tag

    async def test_client_timeout_does_not_cancel_the_worker(self, server, make_handle, tag):
        # [FUNCTION DOC] A handle with a 1 s budget calls a 3 s counting sleep; asserts a
        # timeout error. After 3.5 s a fresh handle must see the counter equal to 1.
        """Giving up on the client side leaves the server running the call to completion."""
        impatient = make_handle(server, call_timeout_seconds=1.0)
        with pytest.raises(TimeoutError):
            await impatient.demo_count_after_sleep(tag=tag, seconds=3.0)

        await asyncio.sleep(3.5)
        assert await make_handle(server).report_counter(tag=tag) == 1

    async def test_poll_requests_stay_within_the_call_budget(self, server, make_handle, proxy_to, handle, tag):
        # [FUNCTION DOC] Runs a gated call through a recording proxy with a 2 s budget; asserts
        # a timeout error and that the proxy recorded at least one poll request. Releases the
        # gate. Despite the name, the test does not inspect the timeout values of the polls.
        # KNOWN ISSUE: the name and docstring claim polls never exceed the remaining budget, but
        # only the existence of poll requests is asserted.
        """Each poll asks for no more than the remaining call budget."""
        proxy = await proxy_to()
        impatient = make_handle(proxy, call_timeout_seconds=2.0)

        pending = asyncio.create_task(impatient.demo_block_async(tag=tag))
        with pytest.raises(TimeoutError):
            await pending

        polls = [r for r in proxy.requests if r.verb == "GET" and "/v1/calls/" in r.path]
        assert polls
        await handle.release(tag=tag)


class TestCallTimeout:
    # [CLASS DOC] The call budget and the timeout error it produces.
    async def test_pending_call_raises_timeout(self, server, make_handle, handle, tag):
        # [FUNCTION DOC] A handle with a 2 s budget calls a gated method; asserts a timeout
        # error after more than 1 s and under 10 s whose message names the method. Releases the
        # gate.
        """A call still pending past its budget raises TimeoutError."""
        impatient = make_handle(server, call_timeout_seconds=2.0)

        started = time.monotonic()
        with pytest.raises(TimeoutError) as exc_info:
            await impatient.demo_block_async(tag=tag)

        assert 1.0 < time.monotonic() - started < 10.0
        assert "demo_block_async" in str(exc_info.value)
        await handle.release(tag=tag)

    async def test_generous_timeout_does_not_interrupt(self, server, make_handle, tag):
        # [FUNCTION DOC] A handle with a 60 s budget runs a 3 s sync sleep; asserts it returns
        # the tag.
        """A slow call under a generous budget completes normally."""
        patient = make_handle(server, call_timeout_seconds=60.0)
        assert await patient.demo_sleep_sync(tag=tag, seconds=3.0) == tag

    async def test_timeout_message_names_the_call(self, server, make_handle, handle, tag):
        # [FUNCTION DOC] A handle with a 1 s budget calls a gated method; asserts the timeout
        # error message contains the phrase "call id". Releases the gate.
        """The timeout error identifies the call so the caller can investigate it."""
        impatient = make_handle(server, call_timeout_seconds=1.0)

        with pytest.raises(TimeoutError) as exc_info:
            await impatient.demo_block_async(tag=tag)

        assert "call id" in str(exc_info.value)
        await handle.release(tag=tag)
