# [MODULE DOC]
# End-to-end happy-path tests of the rpc worker stack over real http against a serve subprocess:
# basic round trips, readiness, many concurrent calls, typed and large payloads, and the manual
# submit-then-poll protocol without the client library.
import asyncio
import time

import httpx

from tests.fast.utils.workers.e2e.e2e_worker import Item
from tests.fast.utils.workers.e2e.harness import READY_TIMEOUT_SECONDS


class TestRoundtrip:
    # [CLASS DOC] Simple calls through the handle fixture and what they return.
    async def test_sync_method(self, handle):
        # [FUNCTION DOC] Calls the sync sum with 3 and 4; asserts 7 and that it is an integer.
        """A sync method answers over real HTTP with its declared type."""
        result = await handle.demo_sync(a=3, b=4)
        assert result == 7 and isinstance(result, int)

    async def test_async_method(self, handle):
        # [FUNCTION DOC] Echoes a mapping holding a list with an integer, a string and nothing;
        # asserts equality.
        """An async method roundtrips a nested payload unchanged."""
        assert await handle.demo_async(value={"k": [1, "x", None]}) == {"k": [1, "x", None]}

    async def test_none_return(self, handle):
        # [FUNCTION DOC] Calls a method returning nothing; asserts the result is nothing.
        """A method declared -> None returns None rather than a missing value."""
        assert await handle.demo_none_result() is None

    async def test_default_parameter_omitted(self, handle):
        # [FUNCTION DOC] Calls the greeting method without arguments; asserts "hello world".
        """An omitted defaulted argument uses the worker-side default."""
        assert await handle.demo_default_arg() == "hello world"

    async def test_default_parameter_overridden(self, handle):
        # [FUNCTION DOC] Calls the greeting method with name "miles"; asserts "hello miles".
        """Passing the defaulted argument overrides it."""
        assert await handle.demo_default_arg(name="miles") == "hello miles"

    async def test_explicit_none_differs_from_omitted(self, handle):
        # [FUNCTION DOC] Passes an explicit missing value to the optional-argument method;
        # asserts the worker saw it as the text "None".
        """An explicit None reaches the worker as None."""
        assert await handle.demo_optional_arg(name=None) == "None"

    async def test_sequential_calls_on_one_handle(self, handle):
        # [FUNCTION DOC] Two consecutive sum calls (1+1 and 2+2); asserts 2 and 4.
        """Consecutive calls on one handle stay independent."""
        assert await handle.demo_sync(a=1, b=1) == 2
        assert await handle.demo_sync(a=2, b=2) == 4

    async def test_two_handles_one_server(self, server, make_handle):
        # [FUNCTION DOC] Builds two handles to the shared server, waits for the first to be
        # ready, and asserts both compute sums correctly (3 and 7).
        """Two independent handles can drive the same server."""
        first, second = make_handle(server), make_handle(server)
        await first.wait_ready(timeout=READY_TIMEOUT_SECONDS)
        assert await first.demo_sync(a=1, b=2) == 3
        assert await second.demo_sync(a=3, b=4) == 7

    async def test_call_runs_in_the_server_subprocess(self, handle, server):
        # [FUNCTION DOC] Asserts the process id reported by the worker equals the id of the
        # spawned server process (so the worker runs in-process with the server, not in the test
        # process).
        """The worker really executes in the spawned process, not in the test process."""
        assert await handle.report_pid() == server.process.pid


class TestReadiness:
    # [CLASS DOC] Readiness waiting and the health endpoint.
    async def test_wait_ready_returns_promptly(self, server, make_handle):
        # [FUNCTION DOC] Waits for readiness with a 30 s timeout; asserts it returns in under
        # 5 s.
        """wait_ready returns as soon as the server answers instead of waiting out its timeout."""
        started = time.monotonic()
        await make_handle(server).wait_ready(timeout=30.0)
        assert time.monotonic() - started < 5.0

    async def test_health_endpoint(self, raw):
        # [FUNCTION DOC] Gets the health endpoint; asserts status 200, body status "ok", and a
        # non-empty boot-uuid response header.
        """The health endpoint answers ok with a boot uuid header."""
        response = await raw.get("/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert response.headers["x-miles-boot-uuid"]


class TestConcurrentCalls:
    # [CLASS DOC] Many calls at once do not mix up results.
    async def test_many_concurrent_calls_from_one_handle(self, handle):
        # [FUNCTION DOC] Fires fifty concurrent echo calls with distinct payloads; asserts each
        # result matches its own request in order.
        """Fifty concurrent calls over one connection pool all return their own result."""
        results = await asyncio.gather(*[handle.demo_async(value={"i": i}) for i in range(50)])
        assert results == [{"i": i} for i in range(50)]

    async def test_concurrent_calls_from_several_handles(self, server, make_handle):
        # [FUNCTION DOC] Five handles each compute i plus i concurrently (after the first is
        # ready); asserts results 0, 2, 4, 6, 8.
        """Concurrent calls from several handles do not cross results."""
        handles = [make_handle(server) for _ in range(5)]
        await handles[0].wait_ready(timeout=READY_TIMEOUT_SECONDS)
        results = await asyncio.gather(*[h.demo_sync(a=i, b=i) for i, h in enumerate(handles)])
        assert results == [0, 2, 4, 6, 8]


class TestTypedPayloads:
    # [CLASS DOC] Typed and large payloads over the real wire.
    async def test_pydantic_model_argument_and_result(self, handle):
        # [FUNCTION DOC] Echoes an item model; asserts the result is an item instance equal to
        # the input.
        """A pydantic model survives the wire in both directions as a real model."""
        result = await handle.demo_model(item=Item(name="x", values=[3, 1, 2]))
        assert isinstance(result, Item) and result == Item(name="x", values=[3, 1, 2])

    async def test_scalar_types_keep_their_python_type(self, handle):
        # [FUNCTION DOC] Echoes a mapping with a float, a boolean, the string "1" and nothing;
        # asserts equality (comparison would fail if types collapsed).
        """Scalars keep their type instead of collapsing to strings or ints."""
        assert await handle.demo_async(value={"f": 1.5, "b": True, "s": "1", "n": None}) == {
            "f": 1.5,
            "b": True,
            "s": "1",
            "n": None,
        }

    async def test_unicode_payload(self, handle):
        # [FUNCTION DOC] Echoes a mapping holding Chinese characters, an emoji, a backslash and
        # quotes; asserts equality.
        """Non-ascii text survives the real socket unchanged."""
        text = "中文 🚀 \\ \" '"
        assert await handle.demo_async(value={"t": text}) == {"t": text}

    async def test_large_upload(self, handle):
        # [FUNCTION DOC] Uploads a string of 4 MiB; asserts the worker reports its length.
        """A multi-megabyte request body is transferred intact."""
        blob = "x" * (4 * 1024 * 1024)
        assert await handle.demo_large_upload(blob=blob) == len(blob)

    async def test_large_download(self, handle):
        # [FUNCTION DOC] Requests a list of 200000 integers; asserts its length and that the
        # last element is 199999.
        """A large response body is transferred intact."""
        values = await handle.demo_large_download(size=200_000)
        assert len(values) == 200_000 and values[-1] == 199_999


class TestManualProtocol:
    # [CLASS DOC] Uses the raw client to speak the submit-and-poll protocol by hand. Call ids
    # "manual-1", "manual-2" and "manual-3" are fixed, so each can be used once per server.
    async def test_submit_then_poll_by_hand(self, raw):
        # [FUNCTION DOC] Submits the sum of 2 and 5 under call id "manual-1"; asserts status 200
        # and body status submitted. Polls with a 1 s timeout up to 50 times until the status is
        # not pending; asserts the final body is success, result 7, error nothing.
        """The documented submit + poll pair works without the client."""
        submit = await raw.post("/v1/demo_sync", json={"call_id": "manual-1", "query": {"a": 2, "b": 5}})
        assert submit.status_code == 200
        assert submit.json() == {"status": "submitted"}

        for _ in range(50):
            poll = await raw.get("/v1/calls/manual-1", params={"timeout": 1.0})
            assert poll.status_code == 200
            if poll.json()["status"] != "pending":
                break
        assert poll.json() == {"status": "success", "result": 7, "error": None}

    async def test_finished_call_can_be_polled_repeatedly(self, raw):
        # [FUNCTION DOC] Submits the sum of 1 and 1 as "manual-2", then polls three times with
        # a 5 s timeout; asserts the last body is success with result 2.
        """A finished outcome stays retrievable for later polls."""
        await raw.post("/v1/demo_sync", json={"call_id": "manual-2", "query": {"a": 1, "b": 1}})
        for _ in range(3):
            body = (await raw.get("/v1/calls/manual-2", params={"timeout": 5.0})).json()
        assert body == {"status": "success", "result": 2, "error": None}

    async def test_second_client_sees_the_outcome(self, raw, server):
        # [FUNCTION DOC] Submits the sum of 4 and 4 as "manual-3" on the raw client, then polls
        # it from a separately created http client; asserts the result is 8.
        """Call state belongs to the server, not to the connection that submitted it."""
        await raw.post("/v1/demo_sync", json={"call_id": "manual-3", "query": {"a": 4, "b": 4}})
        async with httpx.AsyncClient(base_url=server.url, timeout=30.0, trust_env=False) as other:
            body = (await other.get("/v1/calls/manual-3", params={"timeout": 5.0})).json()
        assert body["result"] == 8
