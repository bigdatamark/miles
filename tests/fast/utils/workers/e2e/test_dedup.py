# [MODULE DOC]
# End-to-end tests of server-side call deduplication, driven mostly with raw http requests: a
# repeated submission of the same call id is refused with status 409 whether the first call is
# finished, still running or raced concurrently; a call id is bound to its method and payload
# (key order ignored); and the client does not retry an ambiguous submit (server error or
# dropped response) so the worker executes the call at most once.
import asyncio
from typing import Any

import httpx
import pytest

from miles.utils.workers.worker_handle import WorkerUnreachableError


async def _submit(
    raw: httpx.AsyncClient,
    *,
    method: str,
    call_id: str,
    query: dict[str, Any],
) -> httpx.Response:
    # [FUNCTION DOC] HELPER: Posts a submission for the method with the given call id and query
    # mapping through the raw client; returns the http response without checking it.
    return await raw.post(f"/v1/{method}", json={"call_id": call_id, "query": query})


async def _poll(raw: httpx.AsyncClient, *, call_id: str, timeout: float = 5.0) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Long-polls the outcome of a call, waiting server-side up to the
    # timeout (default 5 s); returns the decoded json body without checking the status code.
    return (await raw.get(f"/v1/calls/{call_id}", params={"timeout": timeout})).json()


class TestDuplicateSubmits:
    # [CLASS DOC] Resubmitting an already accepted call id is a conflict.
    async def test_identical_resubmit_returns_409(self, raw: httpx.AsyncClient, handle: Any, tag: str) -> None:
        # [FUNCTION DOC] Submits a counting call and waits for it to finish, then submits the
        # identical request again. Asserts the first answer is the submitted status, the second
        # has status 409 with a detail message containing "already submitted", and the worker
        # counter for the tag is exactly 1.
        """Resubmitting the same call id and payload fails loudly."""
        first = await _submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag})
        assert first.json() == {"status": "submitted"}
        await _poll(raw, call_id=tag)

        second = await _submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag})

        assert second.status_code == 409
        assert "already submitted" in second.json()["detail"]
        assert await handle.report_counter(tag=tag) == 1

    async def test_in_flight_duplicate_returns_409_immediately(
        self,
        raw: httpx.AsyncClient,
        handle: Any,
        tag: str,
    ) -> None:
        # [FUNCTION DOC] Submits a gated blocking call, then a duplicate 0.05 s later with a
        # 1 s limit on the request; asserts status 409 (so the duplicate did not wait for the
        # first). Releases the gate and asserts the original finishes with status success.
        """A duplicate in-flight call fails without waiting for completion."""
        await _submit(raw, method="demo_block_sync", call_id=tag, query={"tag": tag})
        await asyncio.sleep(0.05)

        duplicate = await asyncio.wait_for(
            _submit(raw, method="demo_block_sync", call_id=tag, query={"tag": tag}),
            timeout=1.0,
        )

        assert duplicate.status_code == 409
        await handle.release(tag=tag)
        assert (await _poll(raw, call_id=tag, timeout=2.0))["status"] == "success"

    async def test_concurrent_duplicates_have_one_acceptance(
        self,
        raw: httpx.AsyncClient,
        handle: Any,
        tag: str,
    ) -> None:
        # [FUNCTION DOC] Fires ten identical submissions concurrently and waits for the call to
        # finish; asserts exactly one response has status 200, nine have 409, and the worker
        # counter is 1.
        """Concurrent duplicate submits produce one acceptance and nine conflicts."""
        responses = await asyncio.gather(
            *[_submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag}) for _ in range(10)]
        )
        await _poll(raw, call_id=tag)

        assert [response.status_code for response in responses].count(200) == 1
        assert [response.status_code for response in responses].count(409) == 9
        assert await handle.report_counter(tag=tag) == 1

    async def test_duplicate_conflict_leaves_original_pollable(self, raw: httpx.AsyncClient, tag: str) -> None:
        # [FUNCTION DOC] Submits a counting call and immediately a duplicate; asserts the
        # duplicate gets 409 and the original's outcome still polls as success.
        """A duplicate conflict leaves the original outcome pollable."""
        await _submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag})
        duplicate = await _submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag})

        assert duplicate.status_code == 409
        assert (await _poll(raw, call_id=tag))["status"] == "success"


class TestCallIdIdentity:
    # [CLASS DOC] A call id is tied to the exact method and payload it was first used with.
    async def test_different_payload_same_call_id_returns_409(
        self,
        raw: httpx.AsyncClient,
        handle: Any,
        tag: str,
    ) -> None:
        # [FUNCTION DOC] Submits a counting call, then the same call id with another tag in the
        # payload; asserts 409 and that the other tag's counter stayed zero.
        """Reusing a call id with a different payload returns 409."""
        await _submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag})
        response = await _submit(
            raw,
            method="demo_count_sync",
            call_id=tag,
            query={"tag": f"{tag}other"},
        )

        assert response.status_code == 409
        assert await handle.report_counter(tag=f"{tag}other") == 0

    async def test_different_method_same_call_id_returns_409(self, raw: httpx.AsyncClient, tag: str) -> None:
        # [FUNCTION DOC] Submits the sync counting method, then the async counting method with
        # the same call id and payload; asserts 409.
        """A call id reused on another method returns 409."""
        await _submit(raw, method="demo_count_sync", call_id=tag, query={"tag": tag})

        response = await _submit(raw, method="demo_count_async", call_id=tag, query={"tag": tag})

        assert response.status_code == 409

    async def test_key_order_does_not_change_duplicate_failure(self, raw: httpx.AsyncClient, tag: str) -> None:
        # [FUNCTION DOC] Submits the sum method with keys a, b then again with the same values
        # in order b, a; asserts the second answer is 409.
        """A reordered duplicate payload still returns 409."""
        await _submit(raw, method="demo_sync", call_id=tag, query={"a": 1, "b": 2})

        response = await _submit(raw, method="demo_sync", call_id=tag, query={"b": 2, "a": 1})

        assert response.status_code == 409


class TestNoRetryAfterAmbiguousSubmit:
    # [CLASS DOC] The client must not resend a submission whose outcome is unknown, since that
    # would produce a duplicate or a conflict.
    async def test_submit_503_is_not_retried(
        self,
        proxy_to: Any,
        make_handle: Any,
        tag: str,
    ) -> None:
        # [FUNCTION DOC] The proxy rejects the next request with 503; asserts the call raises a
        # worker-unreachable error and the proxy saw exactly one submit.
        """A submit 503 raises after one wire attempt."""
        proxy = await proxy_to()
        proxy.reject_next(count=1, status=503)
        handle = make_handle(proxy)

        with pytest.raises(WorkerUnreachableError):
            await handle.demo_count_sync(tag=tag)

        assert len(proxy.submits("demo_count_sync")) == 1

    async def test_dropped_submit_response_is_not_retried(
        self,
        proxy_to: Any,
        make_handle: Any,
        handle: Any,
        tag: str,
    ) -> None:
        # [FUNCTION DOC] The proxy forwards one submit but drops the response. Asserts the call
        # raises a worker-unreachable error with one recorded submit, then checks (up to 100
        # yields of the event loop, then a final read) via a direct handle that the worker
        # executed the call exactly once.
        """A dropped submit response raises once while the accepted call runs."""
        proxy = await proxy_to()
        proxy.drop_next(count=1)
        retrying = make_handle(proxy)

        with pytest.raises(WorkerUnreachableError):
            await retrying.demo_count_sync(tag=tag)

        assert len(proxy.submits("demo_count_sync")) == 1
        for _ in range(100):
            if await handle.report_counter(tag=tag) == 1:
                break
            await asyncio.sleep(0)
        assert await handle.report_counter(tag=tag) == 1
