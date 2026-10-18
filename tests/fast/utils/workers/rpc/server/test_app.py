# [MODULE DOC] Unit tests for the rpc server web application, driven in process through an ASGI
# http client against a test worker. Covers submit-then-poll round trips (sync and async methods,
# model results, failure envelopes, health), protocol errors (404, 400, 409, 412, boot uuid header
# on errors and its unforgeability, 500 handling), duplicate call ids, pending versus completed
# calls, concurrency group serialization versus parallelism, cancellation outcome recording, boot
# uuid stability, and the in-flight call listing endpoint. Wire shapes: a submit is a post to
# the method path with a json envelope holding a call id and a query object; a poll is a get of the
# call path with a long-poll timeout parameter.
import asyncio
import contextlib
import threading
import uuid
from collections.abc import AsyncIterator
from typing import NamedTuple

import httpx
import pytest
from fastapi.responses import JSONResponse

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs, rpc
from miles.utils.workers.rpc.common.protocol import (
    BOOT_UUID_HEADER,
    EXPECTED_BOOT_UUID_HEADER,
    IN_FLIGHT_PATH,
    CallStatusResponse,
)
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.rpc.server.executor import RpcCallExecutor


class _Item(StrictBaseModel):
    # [CLASS DOC] HELPER: Strict model with a text name and an integer value.
    name: str
    value: int


class _Worker:
    # [CLASS DOC] HELPER: Test worker with thread events and a barrier to coordinate with tests, a call
    # counter and an ordering log. Methods: sync add, async model builder, raising method, two
    # methods in the "serial" group, two methods in distinct groups that rendezvous at a two party
    # barrier, and two tag echo methods.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Sets the call counter to zero, creates the done, release-slow and
        # slow-started thread events, a two party barrier with a 5 second timeout, and an empty order log.
        self.calls = 0
        self.done_event = threading.Event()
        self.release_slow = threading.Event()
        self.slow_started = threading.Event()
        self.barrier = threading.Barrier(parties=2, timeout=5.0)
        self.order: list[str] = []

    def demo_sync(self, a: int, b: int) -> int:
        # [FUNCTION DOC] HELPER: Increments the call counter, sets the done event and returns a plus b.
        self.calls += 1
        self.done_event.set()
        return a + b

    async def demo_async_model(self, name: str) -> _Item:
        # [FUNCTION DOC] HELPER: Returns an item with the given name and its length as value.
        return _Item(name=name, value=len(name))

    def demo_raises(self, message: str) -> None:
        # [FUNCTION DOC] HELPER: Raises a runtime error with the given message.
        raise RuntimeError(message)

    @rpc(concurrency_group="serial")
    def demo_slow(self, tag: str) -> str:
        # [FUNCTION DOC] HELPER: Sets slow-started, logs tag_start, blocks (asserting within 5 seconds)
        # until the release event, logs tag_end and returns the tag.
        self.slow_started.set()
        self.order.append(f"{tag}_start")
        assert self.release_slow.wait(timeout=5.0)
        self.order.append(f"{tag}_end")
        return tag

    @rpc(concurrency_group="serial")
    def demo_fast(self, tag: str) -> str:
        # [FUNCTION DOC] HELPER: Logs tag_start and returns the tag immediately.
        self.order.append(f"{tag}_start")
        return tag

    @rpc(concurrency_group="left")
    def demo_meet_left(self) -> str:
        # [FUNCTION DOC] HELPER: Waits at the shared two party barrier, so it only finishes if the
        # right method runs concurrently; returns "left".
        self.barrier.wait()
        return "left"

    @rpc(concurrency_group="right")
    def demo_meet_right(self) -> str:
        # [FUNCTION DOC] HELPER: Waits at the shared barrier and returns "right".
        self.barrier.wait()
        return "right"

    def demo_tag(self, tag: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the tag unchanged.
        return tag

    def demo_tag_upper(self, tag: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the tag upper-cased.
        return tag.upper()


class _AsyncCancelWorker:
    # [CLASS DOC] HELPER: Worker whose only method raises a cancellation, to test cancellation handling.
    async def demo_cancel_self(self) -> str:
        # [FUNCTION DOC] HELPER: Raises an asyncio cancellation error.
        raise asyncio.CancelledError


@contextlib.asynccontextmanager
async def _client(worker: object) -> AsyncIterator[httpx.AsyncClient]:
    # [FUNCTION DOC] HELPER: Async context manager that creates the rpc application for the worker,
    # runs its lifespan, and yields an http client talking to it in process with base url
    # http://testserver. Everything is closed on exit.
    app = create_rpc_app(worker)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


class _Submitted(NamedTuple):
    # [CLASS DOC] HELPER: Pair of the call id used for a submit and the raw http response.
    call_id: str
    response: httpx.Response


async def _submit(client: httpx.AsyncClient, method: str, query: dict, call_id: str | None = None) -> _Submitted:
    # [FUNCTION DOC] HELPER: Posts a submit for the method with an envelope of call id and query. A
    # random hex call id is generated when none is given. Returns the call id and the response; does
    # not check the status.
    call_id = call_id if call_id is not None else uuid.uuid4().hex
    response = await client.post(f"/v1/{method}", json={"call_id": call_id, "query": query})
    return _Submitted(call_id=call_id, response=response)


async def _poll_until_done(client: httpx.AsyncClient, call_id: str) -> dict:
    # [FUNCTION DOC] HELPER: Polls the call status path with a 1 second long-poll timeout up to 100
    # times, asserting each answer is http 200. Returns the json body of the first answer whose status
    # is not pending; raises an assertion error if the call never finishes.
    for _ in range(100):
        query_response = await client.get(f"/v1/calls/{call_id}", params={"timeout": 1.0})
        assert query_response.status_code == 200
        body = query_response.json()
        if body["status"] != "pending":
            return body
    raise AssertionError("call never finished")


async def _call(client: httpx.AsyncClient, method: str, query: dict) -> dict:
    # [FUNCTION DOC] HELPER: Submits the call, asserts it was accepted with http 200, then polls until
    # done and returns the final json body.
    submitted = await _submit(client, method, query)
    assert submitted.response.status_code == 200
    return await _poll_until_done(client, submitted.call_id)


class TestRoundtrip:
    # [CLASS DOC] Tests complete submit and poll round trips.
    async def test_sync_method_success(self):
        # [FUNCTION DOC] The sync add method with 1 and 2 ends with a success envelope, result 3 and no error.
        """A sync method roundtrips its typed result through submit + query."""
        async with _client(_Worker()) as client:
            body = await _call(client, "demo_sync", {"a": 1, "b": 2})
            assert body == {"status": "success", "result": 3, "error": None}

    async def test_async_method_success(self):
        # [FUNCTION DOC] The async method returns the model as a json object with name and value 3.
        """An async method runs on the event loop and returns a model result."""
        async with _client(_Worker()) as client:
            body = await _call(client, "demo_async_model", {"name": "abc"})
            assert body == {"status": "success", "result": {"name": "abc", "value": 3}, "error": None}

    async def test_business_exception_becomes_failed_envelope(self):
        # [FUNCTION DOC] The raising method ends with a failed status whose error text contains the
        # exception type name and message (the submit itself was http 200).
        """Worker exceptions surface as 200 + failed envelope with a traceback."""
        async with _client(_Worker()) as client:
            body = await _call(client, "demo_raises", {"message": "kaboom"})
            assert body["status"] == "failed"
            assert "RuntimeError" in body["error"] and "kaboom" in body["error"]

    async def test_health_endpoint(self):
        # [FUNCTION DOC] A get of the health path returns 200 with status ok.
        """The health endpoint answers ok."""
        async with _client(_Worker()) as client:
            response = await client.get("/v1/health")
            assert response.status_code == 200 and response.json() == {"status": "ok"}


class TestProtocolErrors:
    # [CLASS DOC] Tests http error statuses for invalid requests and the boot uuid header rules.
    async def test_unknown_method_404(self):
        # [FUNCTION DOC] A submit to an undefined method returns 404.
        """Submitting to an unknown method returns 404."""
        async with _client(_Worker()) as client:
            submitted = await _submit(client, "nope", {})
            assert submitted.response.status_code == 404

    async def test_invalid_query_400(self):
        # [FUNCTION DOC] A string for an integer parameter returns 400.
        """A query failing pydantic validation returns 400."""
        async with _client(_Worker()) as client:
            submitted = await _submit(client, "demo_sync", {"a": "x", "b": 2})
            assert submitted.response.status_code == 400

    async def test_unknown_call_id_404(self):
        # [FUNCTION DOC] A zero-timeout poll of an unknown call id returns 404.
        """Querying an unknown call id returns 404."""
        async with _client(_Worker()) as client:
            response = await client.get("/v1/calls/missing", params={"timeout": 0.0})
            assert response.status_code == 404

    async def test_resubmit_different_payload_409(self):
        # [FUNCTION DOC] A second submit with the same call id but different arguments returns 409
        # after the first returned 200.
        """Reusing a call id with a different payload returns 409."""
        async with _client(_Worker()) as client:
            first = await _submit(client, "demo_sync", {"a": 1, "b": 2}, call_id="fixed")
            assert first.response.status_code == 200
            second = await _submit(client, "demo_sync", {"a": 9, "b": 9}, call_id="fixed")
            assert second.response.status_code == 409

    async def test_same_call_id_across_methods_409(self):
        # [FUNCTION DOC] Submitting the same call id and identical query to two different methods:
        # first 200, second 409.
        """Reusing a call id on a different method returns 409 rather than the other result."""
        async with _client(_Worker()) as client:
            first = await client.post("/v1/demo_tag", json={"call_id": "fixed", "query": {"tag": "x"}})
            assert first.status_code == 200
            second = await client.post("/v1/demo_tag_upper", json={"call_id": "fixed", "query": {"tag": "x"}})
            assert second.status_code == 409

    async def test_missing_query_field_400(self):
        # [FUNCTION DOC] An envelope with only a call id returns 400 (not 422).
        """An envelope without the query field returns 400, not FastAPI's default 422."""
        async with _client(_Worker()) as client:
            response = await client.post("/v1/demo_sync", json={"call_id": "c1"})
            assert response.status_code == 400

    async def test_non_object_query_400(self):
        # [FUNCTION DOC] An envelope whose query is a list returns 400.
        """An envelope whose query is not an object returns 400."""
        async with _client(_Worker()) as client:
            response = await client.post("/v1/demo_sync", json={"call_id": "c1", "query": [1, 2]})
            assert response.status_code == 400

    async def test_extra_envelope_field_400(self):
        # [FUNCTION DOC] An envelope with an extra field returns 400.
        """An envelope carrying unknown fields returns 400."""
        async with _client(_Worker()) as client:
            response = await client.post("/v1/demo_sync", json={"call_id": "c1", "query": {"a": 1, "b": 2}, "junk": 1})
            assert response.status_code == 400

    async def test_boot_uuid_mismatch_header_returns_412(self) -> None:
        # [FUNCTION DOC] A submit with a stale expected-boot-uuid header returns 412; the response
        # carries the server's real boot uuid (different from the stale one), the done event is not
        # set and the worker's call counter is zero (nothing was scheduled).
        """A stale boot UUID request header is refused before scheduling."""
        worker = _Worker()
        async with _client(worker) as client:
            response = await client.post(
                "/v1/demo_sync",
                headers={EXPECTED_BOOT_UUID_HEADER: "stale"},
                json={"call_id": "c1", "query": {"a": 1, "b": 2}},
            )
            assert response.status_code == 412
            assert response.headers[BOOT_UUID_HEADER] != "stale"
            assert not worker.done_event.is_set()
            assert worker.calls == 0

    async def test_stale_boot_uuid_is_refused_before_routing_and_validation(self) -> None:
        # [FUNCTION DOC] With a stale expected-boot-uuid header, both a submit to an unknown method and
        # a malformed envelope get 412 rather than 404 or 400.
        """A stale expectation wins over routing and validation, so a restart is never reported as a 404 or 400."""
        async with _client(_Worker()) as client:
            unknown_route = await client.post(
                "/v1/nope", headers={EXPECTED_BOOT_UUID_HEADER: "stale"}, json={"call_id": "c1", "query": {}}
            )
            malformed = await client.post(
                "/v1/demo_sync", headers={EXPECTED_BOOT_UUID_HEADER: "stale"}, json={"call_id": "c1"}
            )

            assert unknown_route.status_code == 412
            assert malformed.status_code == 412

    @pytest.mark.parametrize(
        ("method", "query", "status_code"),
        [("nope", {}, 404), ("demo_sync", {"a": "x", "b": 2}, 400)],
    )
    async def test_rejected_submission_does_not_reserve_call_id(self, method: str, query: dict, status_code: int):
        # [FUNCTION DOC] Parametrized over an unknown method (404) and an invalid query (400). After the
        # rejected submit, the same call id is reused for a valid sync call, which is accepted with 200
        # and completes with a success envelope of result 3.
        """A submission refused before scheduling leaves its call id free for a corrected retry."""
        async with _client(_Worker()) as client:
            rejected = await _submit(client, method, query, call_id="reused")
            assert rejected.response.status_code == status_code

            accepted = await _submit(client, "demo_sync", {"a": 1, "b": 2}, call_id="reused")
            assert accepted.response.status_code == 200
            assert await _poll_until_done(client, "reused") == {"status": "success", "result": 3, "error": None}

    async def test_invalid_poll_timeout_400(self):
        # [FUNCTION DOC] A poll with a negative timeout returns 400.
        """A negative long-poll timeout is a client error, reported as 400."""
        async with _client(_Worker()) as client:
            response = await client.get("/v1/calls/whatever", params={"timeout": -1.0})
            assert response.status_code == 400

    async def test_error_responses_carry_boot_uuid_header(self):
        # [FUNCTION DOC] Both a 404 response and a 400 response include the boot uuid header.
        """4xx responses carry the boot uuid header just like successful ones."""
        async with _client(_Worker()) as client:
            not_found = await client.post("/v1/nope", json={"call_id": "c1", "query": {}})
            malformed = await client.post("/v1/demo_sync", json={"call_id": "c1"})
            assert BOOT_UUID_HEADER in not_found.headers
            assert BOOT_UUID_HEADER in malformed.headers

    async def test_unhandled_route_error_still_carries_boot_uuid_header(self):
        # [FUNCTION DOC] Adds an extra route that raises a runtime error to the application and calls
        # it through a transport that does not re-raise app exceptions. Asserts a 500 response with the
        # boot uuid header and the json body "unhandled rpc server error" under the detail key.
        """An exception escaping a route becomes a 500 that still identifies the serving process."""
        app = create_rpc_app(_Worker())

        @app.get("/v1/boom")
        async def boom() -> None:
            # [FUNCTION DOC] HELPER: Route that always raises a runtime error "boom".
            raise RuntimeError("boom")

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/v1/boom")

        assert response.status_code == 500
        assert BOOT_UUID_HEADER in response.headers
        assert response.json() == {"detail": "unhandled rpc server error"}

    async def test_a_route_cannot_forge_the_boot_uuid_header(self):
        # [FUNCTION DOC] Adds an extra route that sets a forged boot uuid header; asserts that route's
        # response carries the same boot uuid as the health response (the real one overwrote it).
        """The boot uuid header is authoritative, so a header set downstream is overwritten, not kept."""
        app = create_rpc_app(_Worker())

        @app.get("/v1/forged")
        async def forged() -> JSONResponse:
            # [FUNCTION DOC] HELPER: Route returning an empty json object with a forged boot uuid header.
            return JSONResponse(content={}, headers={BOOT_UUID_HEADER: "forged"})

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                health = await client.get("/v1/health")
                response = await client.get("/v1/forged")

        assert response.headers[BOOT_UUID_HEADER] == health.headers[BOOT_UUID_HEADER]


class TestDuplicateCalls:
    # [CLASS DOC] Tests resubmission of an already known call id.
    async def test_resubmit_same_payload_returns_409(self) -> None:
        # [FUNCTION DOC] The first submit answers status submitted; after the call ran, an identical
        # resubmit returns 409 with a detail mentioning "already submitted", and the worker ran once.
        """Resubmitting an identical call id fails loudly without rerunning."""
        worker = _Worker()
        async with _client(worker) as client:
            first = await _submit(client, "demo_sync", {"a": 1, "b": 2}, call_id="fixed")
            assert first.response.json() == {"status": "submitted"}
            assert await asyncio.to_thread(worker.done_event.wait, 5.0)

            second = await _submit(client, "demo_sync", {"a": 1, "b": 2}, call_id="fixed")
            assert second.response.status_code == 409
            assert "already submitted" in second.response.json()["detail"]
            assert worker.calls == 1


class TestPendingAndCompletion:
    # [CLASS DOC] Tests the pending status and completion without polling.
    async def test_pending_then_success(self):
        # [FUNCTION DOC] A blocked slow call polled with zero timeout reports pending; after release
        # it finishes with a success envelope carrying the tag.
        """A long-running call reports pending until it finishes."""
        worker = _Worker()
        async with _client(worker) as client:
            submitted = await _submit(client, "demo_slow", {"tag": "s"})
            pending = await client.get(f"/v1/calls/{submitted.call_id}", params={"timeout": 0.0})
            assert pending.json()["status"] == "pending"

            worker.release_slow.set()
            body = await _poll_until_done(client, submitted.call_id)
            assert body == {"status": "success", "result": "s", "error": None}

    async def test_sync_call_completes_without_polling(self):
        # [FUNCTION DOC] Submits a sync call and never polls; the done event is set within 5 seconds
        # and the call counter is one.
        """A submitted sync call runs to completion even if never polled."""
        worker = _Worker()
        async with _client(worker) as client:
            await _submit(client, "demo_sync", {"a": 1, "b": 2})
            assert await asyncio.to_thread(worker.done_event.wait, 5.0)
            assert worker.calls == 1


class TestConcurrencyGroups:
    # [CLASS DOC] Tests how calls are scheduled across concurrency groups.
    async def test_same_group_serializes(self):
        # [FUNCTION DOC] A slow call and a fast call, both in the serial group, are submitted; after
        # 0.05 seconds only the slow start is logged. After release and completion, the log shows slow
        # start, slow end, then fast start.
        """Two calls in one concurrency group run strictly one after another."""
        worker = _Worker()
        async with _client(worker) as client:
            slow_submitted = await _submit(client, "demo_slow", {"tag": "demo_slow"})
            assert await asyncio.to_thread(worker.slow_started.wait, 5.0)
            fast_submitted = await _submit(client, "demo_fast", {"tag": "demo_fast"})

            await asyncio.sleep(0.05)
            assert worker.order == ["demo_slow_start"]

            worker.release_slow.set()
            await _poll_until_done(client, slow_submitted.call_id)
            await _poll_until_done(client, fast_submitted.call_id)
            assert worker.order == ["demo_slow_start", "demo_slow_end", "demo_fast_start"]

    async def test_different_groups_run_in_parallel(self):
        # [FUNCTION DOC] Two calls in the left and right groups rendezvous at a two party barrier, so
        # both finish only if run concurrently; asserts results "left" and "right".
        """Calls in different concurrency groups run concurrently."""
        worker = _Worker()
        async with _client(worker) as client:
            left = await _submit(client, "demo_meet_left", {})
            right = await _submit(client, "demo_meet_right", {})
            left_body = await _poll_until_done(client, left.call_id)
            right_body = await _poll_until_done(client, right.call_id)
            assert left_body["result"] == "left" and right_body["result"] == "right"


class TestCancellationOutcome:
    # [CLASS DOC] Tests handling of a worker method raising a cancellation.
    async def test_worker_cancellation_yields_terminal_outcome(self):
        # [FUNCTION DOC] Calling the self-cancelling method through the app ends with a failed status
        # whose error text mentions the cancellation error type.
        """A worker method raising CancelledError still records a terminal failed outcome."""
        async with _client(_AsyncCancelWorker()) as client:
            body = await _call(client, "demo_cancel_self", {})
            assert body["status"] == "failed"
            assert "CancelledError" in body["error"]

    async def test_cancellation_is_re_raised_after_the_outcome_is_recorded(self):
        # [FUNCTION DOC] Calls the executor's private run method directly with a finish callback that
        # appends the outcome to a list. Asserts the run raises a cancellation error afterwards and
        # exactly one outcome with status failed was recorded.
        """The execution records its terminal outcome and then stays cancelled instead of swallowing it."""
        specs = collect_rpc_method_specs(_AsyncCancelWorker)
        executor = RpcCallExecutor(worker=_AsyncCancelWorker(), specs=specs)
        recorded: list[CallStatusResponse] = []

        with pytest.raises(asyncio.CancelledError):
            await executor._run(
                spec=specs["demo_cancel_self"],
                kwargs={},
                call_id="c1",
                finish=lambda *, outcome: recorded.append(outcome),
            )

        assert [outcome.status for outcome in recorded] == ["failed"]


class TestBootUuid:
    # [CLASS DOC] Tests boot uuid generation per server instance.
    async def test_boot_uuid_header_stable_within_server(self):
        # [FUNCTION DOC] Two health responses of one server carry the same boot uuid.
        """All responses of one server carry the same boot uuid header."""
        async with _client(_Worker()) as client:
            first = await client.get("/v1/health")
            second = await client.get("/v1/health")
            assert first.headers[BOOT_UUID_HEADER] == second.headers[BOOT_UUID_HEADER]

    async def test_boot_uuid_differs_across_servers(self):
        # [FUNCTION DOC] Two separately created servers answer with different boot uuids.
        """Two server instances have different boot_uuids."""
        async with _client(_Worker()) as first_client:
            first = (await first_client.get("/v1/health")).headers[BOOT_UUID_HEADER]
        async with _client(_Worker()) as second_client:
            second = (await second_client.get("/v1/health")).headers[BOOT_UUID_HEADER]
        assert first != second


class TestTheInFlightEndpoint:
    # [CLASS DOC] Tests the endpoint listing the ids of calls currently running or queued.
    async def test_a_worker_running_nothing_reports_nothing_in_flight(self) -> None:
        # [FUNCTION DOC] An idle server answers 200 with an empty call id list.
        """A caller waiting a worker out reads this to decide the worker has really let go."""
        async with _client(_Worker()) as client:
            response = await client.get(IN_FLIGHT_PATH)

            assert response.status_code == 200 and response.json() == {"call_ids": []}

    async def test_it_names_the_call_the_worker_is_running(self) -> None:
        # [FUNCTION DOC] While a slow call c1 is blocked, the listing is exactly c1; after release and
        # completion the listing is empty again.
        """This is what a caller waits out, so a running call has to show up under its own id."""
        worker = _Worker()
        async with _client(worker) as client:
            submitted = await _submit(client, "demo_slow", {"tag": "a"}, call_id="c1")
            assert submitted.response.status_code == 200
            assert worker.slow_started.wait(timeout=5.0)

            assert (await client.get(IN_FLIGHT_PATH)).json() == {"call_ids": ["c1"]}

            worker.release_slow.set()
            await _poll_until_done(client, "c1")

            assert (await client.get(IN_FLIGHT_PATH)).json() == {"call_ids": []}
