# [MODULE DOC]
# End-to-end tests of what happens when the caller cancels an awaiting rpc call: the server-side
# work continues and its result stays available, other calls and the handle stay usable, and an
# abandoned synchronous call still occupies its concurrency-group slot.
import asyncio

import httpx
import pytest

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle


async def _wait_until_worker_event(handle: RpcWorkerHandle, *, tag: str, phase: str) -> None:
    # [FUNCTION DOC] HELPER: Polls the worker's recorded events up to 200 times, 50 ms apart
    # (about 10 seconds), returning as soon as an event with the given call tag and phase exists.
    # Raises an assertion error naming the phase and tag if it never appears.
    for _ in range(200):
        events = await handle.report_events()
        if any(event.tag == tag and event.phase == phase for event in events):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"the worker never reported {phase!r} for call tag {tag!r}")


class TestCallerCancellation:
    # [CLASS DOC] Tests for caller-side cancellation of rpc calls against a real server.
    async def test_cancelling_the_await_does_not_stop_the_worker(self, server, make_handle, tag):
        # [FUNCTION DOC] Starts a 2 s blocking-sleep counting call, cancels it after 0.5 s and
        # asserts the caller sees cancellation. After waiting 2.5 s, a fresh handle must see the
        # tag's counter equal to one, i.e. the worker finished the call anyway.
        """Cancelling on the caller side abandons the result, it does not cancel the work."""
        handle = make_handle(server)
        pending = asyncio.create_task(handle.demo_count_after_sleep(tag=tag, seconds=2.0))
        await asyncio.sleep(0.5)

        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        await asyncio.sleep(2.5)
        assert await make_handle(server).report_counter(tag=tag) == 1

    async def test_cancelling_one_call_leaves_the_others(self, handle, tag):
        # [FUNCTION DOC] Runs a 3 s call and a 1 s call concurrently on one handle, cancels the
        # first after 0.3 s, and asserts the second still returns its tag.
        """Cancelling one call does not disturb concurrent calls on the same handle."""
        cancelled = asyncio.create_task(handle.demo_sleep_async(tag=f"{tag}c", seconds=3.0))
        kept = asyncio.create_task(handle.demo_sleep_async(tag=f"{tag}k", seconds=1.0))
        await asyncio.sleep(0.3)

        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        assert await kept == f"{tag}k"

    async def test_handle_is_reusable_after_a_cancelled_call(self, handle, tag):
        # [FUNCTION DOC] Cancels a 3 s call after 0.3 s, then asserts a new simple sum call on
        # the same handle returns 2.
        """A cancelled call leaves the handle in a usable state."""
        pending = asyncio.create_task(handle.demo_sleep_async(tag=tag, seconds=3.0))
        await asyncio.sleep(0.3)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        assert await handle.demo_sync(a=1, b=1) == 2

    async def test_cancelling_wait_ready_is_clean(self, make_handle):
        # [FUNCTION DOC] Waits for readiness (30 s timeout) against an address nothing listens
        # on, cancels after 0.5 s, and asserts cancellation is raised rather than a timeout.
        """Cancelling wait_ready against a dead server raises CancelledError promptly."""
        handle = make_handle("http://127.0.0.1:9")
        pending = asyncio.create_task(handle.wait_ready(timeout=30.0))
        await asyncio.sleep(0.5)

        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    async def test_cancelling_sync_await_does_not_release_its_group_slot(
        self, handle: RpcWorkerHandle, raw: httpx.AsyncClient, tag: str
    ) -> None:
        # [FUNCTION DOC] Starts a synchronous call that blocks on a gate and waits until the
        # worker reports it started, then cancels the caller's await. Submits a second instant
        # synchronous call with a chosen call id through the raw http client (expects 200) and
        # polls it with zero timeout: asserts the status is still pending, since the abandoned
        # call keeps the single worker thread busy. After releasing the gate, polls with a 5 s
        # timeout and asserts success and a result starting with the "rpc-" thread prefix.
        """An abandoned sync call keeps its concurrency group busy, so a queued call cannot overtake it."""
        blocked_tag = f"{tag}block"
        queued_tag = f"{tag}queued"
        queued_call_id = f"{tag}-queued"
        abandoned = asyncio.create_task(handle.demo_block_sync(tag=blocked_tag))
        await _wait_until_worker_event(handle, tag=blocked_tag, phase="start")

        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandoned

        submitted = await raw.post(
            "/v1/demo_instant_sync",
            json={"call_id": queued_call_id, "query": {"tag": queued_tag}},
        )
        assert submitted.status_code == 200

        pending = await raw.get(f"/v1/calls/{queued_call_id}", params={"timeout": 0.0})
        assert pending.status_code == 200
        assert pending.json()["status"] == "pending"

        await handle.release(tag=blocked_tag)
        finished = await raw.get(f"/v1/calls/{queued_call_id}", params={"timeout": 5.0})
        assert finished.status_code == 200
        assert finished.json()["status"] == "success"
        assert finished.json()["result"].startswith("rpc-")

    async def test_result_of_a_cancelled_call_stays_available(self, server, make_handle, raw, tag):
        # [FUNCTION DOC] Cancels a 2 s async-sleep call after 0.3 s, waits 2.5 s, then asserts a
        # fresh handle sees an end event for the tag in the worker's events. The raw client
        # fixture is requested but unused.
        """The outcome the caller walked away from is still on the server."""
        handle = make_handle(server)
        pending = asyncio.create_task(handle.demo_sleep_async(tag=tag, seconds=2.0))
        await asyncio.sleep(0.3)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        await asyncio.sleep(2.5)
        events = await make_handle(server).report_events()
        assert any(event.tag == tag and event.phase == "end" for event in events)
