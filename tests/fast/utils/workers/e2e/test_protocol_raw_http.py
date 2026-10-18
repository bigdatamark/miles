# [MODULE DOC]
# Tests of the rpc server's http protocol using raw requests: method lookup errors (unknown,
# private, dunder and reserved names give 404), envelope validation (malformed or incomplete
# bodies give 400), query validation against the method signature, call lookup and poll-timeout
# parameter handling, the clamping of huge poll timeouts to the server's cap (using an in-process
# application), and the boot-uuid header on every response.
import asyncio
import contextlib
import time
from collections.abc import AsyncIterator

import httpx
import pytest

from miles.utils.workers.rpc.server import core as core_module
from miles.utils.workers.rpc.server.app import create_rpc_app

EXPECTED_BOOT_UUID_HEADER = "x-miles-expected-boot-uuid"
BOOT_UUID_HEADER = "x-miles-boot-uuid"


async def _submit(raw, method: str, call_id: str, query: dict, **extra):
    # [FUNCTION DOC] HELPER: Posts a submission to the method path with a body of the call id,
    # the query and any extra envelope fields; returns the unchecked http response.
    return await raw.post(f"/v1/{method}", json={"call_id": call_id, "query": query, **extra})


class _GatedWorker:
    # [CLASS DOC] HELPER: In-process worker whose single method waits on an event-loop gate that
    # the test opens.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the closed gate event.
        self.gate = asyncio.Event()

    async def demo_wait_for_gate(self) -> str:
        # [FUNCTION DOC] HELPER: Waits until the gate is opened and returns "released".
        await self.gate.wait()
        return "released"


@contextlib.asynccontextmanager
async def _in_process_client(worker: object) -> AsyncIterator[httpx.AsyncClient]:
    # [FUNCTION DOC] HELPER: Async context manager yielding an http client wired directly to an
    # rpc application built around the given worker (no sockets), with the application's startup
    # and shutdown handlers run around the block. The client uses a fake base url.
    app = create_rpc_app(worker)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


class TestMethodLookup:
    # [CLASS DOC] Which method names a submission can reach.
    async def test_unknown_method_404(self, raw, tag):
        # [FUNCTION DOC] Submits to an undefined method; asserts status 404 and that the detail
        # message includes the method name.
        """Submitting to a method the worker does not define is a 404."""
        response = await _submit(raw, "no_such_method", tag, {})
        assert response.status_code == 404
        assert "no_such_method" in response.json()["detail"]

    async def test_private_method_not_exposed(self, raw, tag):
        # [FUNCTION DOC] Submits to the private counter method name; asserts 404.
        """Underscore-prefixed worker methods are not reachable."""
        assert (await _submit(raw, "_bump", tag, {"tag": "x"})).status_code == 404

    async def test_dunder_not_exposed(self, raw, tag):
        # [FUNCTION DOC] Submits to the constructor method name; asserts 404.
        """Dunder attributes are not reachable."""
        assert (await _submit(raw, "__init__", tag, {})).status_code == 404

    async def test_health_is_not_a_submit_target(self, raw, tag):
        # [FUNCTION DOC] Posts to the health path name as a method; asserts 404.
        """POST /v1/health falls through to method lookup and 404s."""
        assert (await _submit(raw, "health", tag, {})).status_code == 404

    async def test_calls_path_is_not_a_submit_target(self, raw):
        # [FUNCTION DOC] Posts a valid-looking envelope to the calls path; asserts 404.
        """POST /v1/calls/... is method lookup, not a call query."""
        assert (await raw.post("/v1/calls", json={"call_id": "x", "query": {}})).status_code == 404


class TestEnvelopeValidation:
    # [CLASS DOC] The submission body must be an object with exactly a call id and a query
    # object; violations give status 400.
    async def test_malformed_json_400(self, raw):
        # [FUNCTION DOC] Posts the invalid json text with a json content type; asserts 400.
        """A body that is not JSON is a client error, normalized to 400."""
        response = await raw.post("/v1/demo_sync", content=b"{not json", headers={"content-type": "application/json"})
        assert response.status_code == 400

    async def test_missing_call_id_400(self, raw):
        # [FUNCTION DOC] Posts a body without a call id; asserts 400.
        """An envelope without call_id is rejected."""
        assert (await raw.post("/v1/demo_sync", json={"query": {"a": 1, "b": 2}})).status_code == 400

    async def test_missing_query_400(self, raw, tag):
        # [FUNCTION DOC] Posts a body without a query; asserts 400.
        """An envelope without query is rejected."""
        assert (await raw.post("/v1/demo_sync", json={"call_id": tag})).status_code == 400

    async def test_extra_envelope_field_400(self, raw, tag):
        # [FUNCTION DOC] Posts a valid envelope plus an unknown field; asserts 400.
        """Unknown envelope fields are rejected rather than ignored."""
        response = await raw.post("/v1/demo_sync", json={"call_id": tag, "query": {"a": 1, "b": 2}, "bogus": 1})
        assert response.status_code == 400

    async def test_query_not_an_object_400(self, raw, tag):
        # [FUNCTION DOC] Posts a list as the query; asserts 400.
        """A non-object query is rejected."""
        assert (await raw.post("/v1/demo_sync", json={"call_id": tag, "query": [1, 2]})).status_code == 400


class TestQueryValidation:
    # [CLASS DOC] The query mapping is validated against the method's parameters.
    async def test_unknown_kwarg_400(self, raw, tag):
        # [FUNCTION DOC] Submits the sum with an extra key "c"; asserts 400 and that the detail
        # mentions "c".
        """An argument the method does not declare is rejected."""
        response = await _submit(raw, "demo_sync", tag, {"a": 1, "b": 2, "c": 3})
        assert response.status_code == 400 and "c" in response.json()["detail"]

    async def test_missing_required_kwarg_400(self, raw, tag):
        # [FUNCTION DOC] Submits the sum without "b"; asserts 400 and that the detail mentions
        # "b".
        """A missing required argument is rejected."""
        response = await _submit(raw, "demo_sync", tag, {"a": 1})
        assert response.status_code == 400 and "b" in response.json()["detail"]

    async def test_wrong_type_400(self, raw, tag):
        # [FUNCTION DOC] Submits a non-numeric string for an integer; asserts 400.
        """An argument that cannot be coerced is rejected."""
        assert (await _submit(raw, "demo_sync", tag, {"a": "not-int", "b": 2})).status_code == 400

    async def test_coercible_strings_accepted(self, raw, tag):
        # [FUNCTION DOC] Submits numeric strings "3" and "4" for integer parameters; asserts
        # status 200 and, on polling, result 7 (the server coerces).
        """Coercible values are accepted, pinning the non-strict query behaviour."""
        assert (await _submit(raw, "demo_sync", tag, {"a": "3", "b": "4"})).status_code == 200
        body = (await raw.get(f"/v1/calls/{tag}", params={"timeout": 5.0})).json()
        assert body["result"] == 7


class TestCallLookup:
    # [CLASS DOC] Validation and defaults of the poll endpoint.
    async def test_unknown_call_id_404(self, raw):
        # [FUNCTION DOC] Polls an id never submitted; asserts 404 and that the detail includes
        # the id.
        """Polling an unknown call id is a 404."""
        response = await raw.get("/v1/calls/deadbeef", params={"timeout": 0.0})
        assert response.status_code == 404 and "deadbeef" in response.json()["detail"]

    async def test_negative_poll_timeout_400(self, raw, tag):
        # [FUNCTION DOC] Submits a call, polls with timeout -1; asserts 400.
        """A negative poll timeout is a client error."""
        await _submit(raw, "demo_sync", tag, {"a": 1, "b": 1})
        assert (await raw.get(f"/v1/calls/{tag}", params={"timeout": -1})).status_code == 400

    async def test_non_numeric_poll_timeout_400(self, raw, tag):
        # [FUNCTION DOC] Submits a call, polls with timeout "abc"; asserts 400.
        """A non-numeric poll timeout is a client error."""
        await _submit(raw, "demo_sync", tag, {"a": 1, "b": 1})
        assert (await raw.get(f"/v1/calls/{tag}", params={"timeout": "abc"})).status_code == 400

    async def test_huge_poll_timeout_accepted(self, raw, tag):
        # [FUNCTION DOC] Submits a counting call, polls with timeout 999999; asserts 200.
        """A poll timeout beyond the server cap is accepted rather than rejected."""
        await _submit(raw, "demo_count_sync", tag, {"tag": tag})
        response = await raw.get(f"/v1/calls/{tag}", params={"timeout": 999999})
        assert response.status_code == 200

    async def test_omitted_poll_timeout_uses_server_default(self, raw, tag):
        # [FUNCTION DOC] Submits a sum, polls with a 5 s timeout until success, then polls
        # without any timeout parameter; asserts the second body is success with result 5.
        """Polling without a timeout parameter uses the server default and returns the finished outcome."""
        await _submit(raw, "demo_sync", tag, {"a": 2, "b": 3})
        assert (await raw.get(f"/v1/calls/{tag}", params={"timeout": 5.0})).json()["status"] == "success"

        body = (await raw.get(f"/v1/calls/{tag}")).json()
        assert body["status"] == "success" and body["result"] == 5


class TestPollTimeoutClamp:
    # [CLASS DOC] The server caps the effective poll timeout.
    async def test_huge_poll_timeout_is_clamped_to_the_server_cap(self, monkeypatch):
        # [FUNCTION DOC] Sets the server module's maximum poll timeout to 0.3 s and builds an
        # in-process app around a gated worker. Submits the gated call (asserts 200), polls with
        # timeout 999999 under a 5 s outer limit, and always opens the gate afterwards. Asserts
        # the answer says pending and arrived in under 3 s, showing the clamp.
        """A huge poll timeout on a still pending call is clamped, so pending comes back within the cap."""
        monkeypatch.setattr(core_module, "MAX_POLL_TIMEOUT_SECONDS", 0.3)
        worker = _GatedWorker()

        async with _in_process_client(worker) as client:
            submitted = await client.post("/v1/demo_wait_for_gate", json={"call_id": "gated", "query": {}})
            assert submitted.status_code == 200

            started_at = time.monotonic()
            try:
                polled = client.get("/v1/calls/gated", params={"timeout": 999999})
                response = await asyncio.wait_for(polled, timeout=5.0)
            finally:
                worker.gate.set()
            elapsed = time.monotonic() - started_at

            assert response.json()["status"] == "pending"
            assert elapsed < 3.0


class TestHeaders:
    # [CLASS DOC] Every response carries the serving process's boot uuid.
    @pytest.mark.parametrize(
        "case",
        ["ok", "unknown_method", "malformed_body", "unknown_call", "stale_boot_uuid"],
    )
    async def test_every_response_carries_the_boot_uuid(self, raw, tag, case):
        # [FUNCTION DOC] Parametrized over five request shapes: a valid submit, an unknown
        # method, a body missing the query, a poll of an unknown call, and a submit carrying an
        # expected boot uuid of 32 zeros (stale). Asserts the boot-uuid header is present on the
        # response in every case.
        """Success and failure responses alike identify the serving process."""
        if case == "ok":
            response = await _submit(raw, "demo_sync", tag, {"a": 1, "b": 1})
        elif case == "unknown_method":
            response = await _submit(raw, "nope", tag, {})
        elif case == "malformed_body":
            response = await raw.post("/v1/demo_sync", json={"call_id": tag})
        elif case == "unknown_call":
            response = await raw.get("/v1/calls/missing", params={"timeout": 0.0})
        else:
            response = await raw.post(
                "/v1/demo_sync",
                headers={EXPECTED_BOOT_UUID_HEADER: "0" * 32},
                json={"call_id": tag, "query": {"a": 1, "b": 1}},
            )

        assert BOOT_UUID_HEADER in response.headers

    async def test_boot_uuid_is_stable_within_a_process(self, raw, tag):
        # [FUNCTION DOC] Reads the boot-uuid header from a health response, a submit response
        # and a poll response of the same server; asserts all three are equal and 32 characters.
        """One server keeps one boot uuid across endpoints."""
        health = (await raw.get("/v1/health")).headers[BOOT_UUID_HEADER]
        submit = (await _submit(raw, "demo_sync", tag, {"a": 1, "b": 1})).headers[BOOT_UUID_HEADER]
        poll = (await raw.get(f"/v1/calls/{tag}", params={"timeout": 5.0})).headers[BOOT_UUID_HEADER]
        assert health == submit == poll and len(health) == 32
