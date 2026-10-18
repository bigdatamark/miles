# [MODULE DOC] Unit tests for the rpc client support module: the boot uuid pin (which records the
# first server boot uuid and refuses later different or missing ones, with unpin and repin support)
# and the rpc transport (status code classification, response validation, request timeout, header
# merging with the pinned uuid header, fallback to the shared general http client, mapping of a 412
# response to a server-restarted error). All http traffic goes to in-memory mock transports.
import asyncio
import json
from collections.abc import Awaitable, Callable

import httpx
import pytest
from pydantic import ValidationError

from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.workers.rpc.client.misc import (
    BootUuidPin,
    RetryableResponseError,
    RpcProtocolError,
    RpcTransport,
    ServerRestartedError,
)
from miles.utils.workers.rpc.common.protocol import BOOT_UUID_HEADER, EXPECTED_BOOT_UUID_HEADER, HealthResponse


def _response(*, status_code: int = 200, boot_uuid: str | None = None) -> httpx.Response:
    # [FUNCTION DOC] HELPER: Builds a bodiless http response with the given status (default 200) and,
    # only when a boot uuid is given, the boot uuid header set to it.
    headers = {} if boot_uuid is None else {BOOT_UUID_HEADER: boot_uuid}
    return httpx.Response(status_code=status_code, headers=headers)


def _transport_over(
    handler: Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]],
    *,
    pin: BootUuidPin | None = None,
) -> tuple[RpcTransport, httpx.AsyncClient]:
    # [FUNCTION DOC] HELPER: Builds an async http client over a mock transport that calls the given
    # handler (sync or async) for each request, and an rpc transport for server url
    # http://testserver using that client and the given pin (a non-required pin for a worker class
    # named "Worker" when none is given). Returns the transport and the client; the caller must close
    # the client (tests use it as an async context manager).
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    boot_uuid_pin = pin or BootUuidPin(required=False, worker_cls_name="Worker")
    transport = RpcTransport(
        server_url="http://testserver",
        http_client=client,
        boot_uuid_pin=boot_uuid_pin,
    )
    return transport, client


class TestBootUuidPin:
    # [CLASS DOC] Tests verify behavior of the boot uuid pin for optional and required pins.
    def test_optional_pin_never_requires_handshake(self) -> None:
        # [FUNCTION DOC] Verifies a response carrying a uuid against an optional pin; asserts no
        # handshake is needed afterwards and nothing was recorded as the expected uuid.
        """An optional pin never records a response boot UUID."""
        pin = BootUuidPin(required=False, worker_cls_name="Worker")

        pin.verify(_response(boot_uuid="boot-a"))

        assert pin.needs_handshake() is False
        assert pin.expected is None

    def test_required_pin_rejects_missing_header(self) -> None:
        # [FUNCTION DOC] Verifies a response without the boot uuid header against a required pin and
        # asserts a server-restarted error mentioning the header name.
        """A required pin rejects a response missing the boot UUID header."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")

        with pytest.raises(ServerRestartedError, match=BOOT_UUID_HEADER):
            pin.verify(_response())

    def test_required_pin_keeps_first_value(self) -> None:
        # [FUNCTION DOC] Verifies the same uuid twice on a required pin; asserts the handshake is no
        # longer needed and the expected value is the first uuid.
        """A required pin keeps the first observed boot UUID."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")

        pin.verify(_response(boot_uuid="boot-a"))
        pin.verify(_response(boot_uuid="boot-a"))

        assert pin.needs_handshake() is False
        assert pin.expected == "boot-a"

    def test_required_pin_refuses_a_different_value_afterwards(self) -> None:
        # [FUNCTION DOC] Pins boot-a, then verifies boot-b; asserts a server-restarted error mentioning
        # boot-b and that the expected value stays boot-a.
        """Once pinned, a response from another boot UUID is a restart."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))

        with pytest.raises(ServerRestartedError, match="boot-b"):
            pin.verify(_response(boot_uuid="boot-b"))

        assert pin.expected == "boot-a"

    def test_required_pin_refuses_a_missing_header_afterwards(self) -> None:
        # [FUNCTION DOC] Pins boot-a, then verifies a response without the header; asserts a
        # server-restarted error mentioning the header name.
        """Once pinned, a response without the header is a restart, not a silent pass."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))

        with pytest.raises(ServerRestartedError, match=BOOT_UUID_HEADER):
            pin.verify(_response())


class TestRpcTransport:
    # [CLASS DOC] Tests the rpc transport request method against mock http handlers. Every request is
    # a GET of the health path with a one second budget unless stated otherwise.
    async def test_returns_validated_response_model(self) -> None:
        # [FUNCTION DOC] The handler answers 200 with a json body saying status ok; asserts the result
        # equals a default health response model.
        """A successful request returns the validated protocol model."""

        def handler(request: httpx.Request) -> httpx.Response:
            # [FUNCTION DOC] HELPER: Answers 200 with the json body status ok.
            return httpx.Response(200, json={"status": "ok"}, request=request)

        transport, client = _transport_over(handler)
        async with client:
            response = await transport.request(
                "GET",
                "/v1/health",
                seconds=1.0,
                response_model=HealthResponse,
            )

        assert response == HealthResponse()

    @pytest.mark.parametrize("status_code", [500, 502, 503, 599])
    async def test_server_status_raises_retryable_response(self, status_code: int) -> None:
        # [FUNCTION DOC] For each of the status codes 500, 502, 503 and 599, the handler answers with
        # that status and text "busy"; asserts a retryable response error whose message contains the
        # status code.
        """Any 5xx response is classified as retryable."""

        def handler(request: httpx.Request) -> httpx.Response:
            # [FUNCTION DOC] HELPER: Answers with the parametrized status and the text "busy".
            return httpx.Response(status_code, text="busy", request=request)

        transport, client = _transport_over(handler)
        async with client:
            with pytest.raises(RetryableResponseError, match=str(status_code)):
                await transport.request(
                    "GET",
                    "/v1/health",
                    seconds=1.0,
                    response_model=HealthResponse,
                )

    @pytest.mark.parametrize("status_code", [201, 400, 404, 409, 499])
    async def test_non_200_non_5xx_raises_protocol_error(self, status_code: int) -> None:
        # [FUNCTION DOC] For each of the status codes 201, 400, 404, 409 and 499, the handler answers
        # with that status; asserts an rpc protocol error whose status code attribute equals it.
        """A non-200 non-5xx response is a protocol error."""

        def handler(request: httpx.Request) -> httpx.Response:
            # [FUNCTION DOC] HELPER: Answers with the parametrized status and the text "rejected".
            return httpx.Response(status_code, text="rejected", request=request)

        transport, client = _transport_over(handler)
        async with client:
            with pytest.raises(RpcProtocolError) as exc_info:
                await transport.request(
                    "GET",
                    "/v1/health",
                    seconds=1.0,
                    response_model=HealthResponse,
                )

        assert exc_info.value.status_code == status_code

    async def test_outer_timeout_enforces_request_budget(self) -> None:
        # [FUNCTION DOC] The handler waits forever on an event; with a 0.01 second budget asserts the
        # transport raises a built-in timeout error.
        """The transport cancels a request exceeding its explicit budget."""

        async def handler(request: httpx.Request) -> httpx.Response:
            # [FUNCTION DOC] HELPER: Blocks forever on an event nobody sets; the return is unreachable.
            await asyncio.Event().wait()
            return httpx.Response(200, request=request)

        transport, client = _transport_over(handler)
        async with client:
            with pytest.raises(TimeoutError):
                await transport.request(
                    "GET",
                    "/v1/health",
                    seconds=0.01,
                    response_model=HealthResponse,
                )

    async def test_pinned_boot_uuid_is_sent_as_request_header(self) -> None:
        # [FUNCTION DOC] Pins boot-a on a required pin, sends two requests whose handler records the
        # expected-boot-uuid header and answers with a matching boot uuid header; asserts both requests
        # carried boot-a.
        """A pinned transport sends the expected boot UUID on every request."""
        seen_headers: list[str | None] = []
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))

        def handler(request: httpx.Request) -> httpx.Response:
            # [FUNCTION DOC] HELPER: Records the expected boot uuid request header (or nothing) and
            # answers 200 with status ok and the boot uuid header set to boot-a.
            seen_headers.append(request.headers.get(EXPECTED_BOOT_UUID_HEADER))
            return httpx.Response(
                200,
                headers={BOOT_UUID_HEADER: "boot-a"},
                json={"status": "ok"},
                request=request,
            )

        transport, client = _transport_over(handler, pin=pin)
        async with client:
            await transport.request(
                "GET",
                "/v1/health",
                seconds=1.0,
                response_model=HealthResponse,
            )
            await transport.request(
                "GET",
                "/v1/health",
                seconds=1.0,
                response_model=HealthResponse,
            )

        assert seen_headers == ["boot-a", "boot-a"]

    async def test_caller_headers_survive_but_cannot_override_the_pin(self) -> None:
        # [FUNCTION DOC] With a pinned uuid boot-a, sends a request with custom headers: a trace header
        # and a caller-chosen expected-boot-uuid value. Asserts the trace header arrives unchanged and
        # the expected-boot-uuid header carries the pinned value, not the caller's.
        """Caller headers are passed through, while a caller-supplied expectation loses to the pinned uuid."""
        seen: list[httpx.Headers] = []
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))

        def handler(request: httpx.Request) -> httpx.Response:
            # [FUNCTION DOC] HELPER: Stores the request headers and answers 200 status ok with a
            # matching boot uuid header.
            seen.append(request.headers)
            return httpx.Response(200, headers={BOOT_UUID_HEADER: "boot-a"}, json={"status": "ok"}, request=request)

        transport, client = _transport_over(handler, pin=pin)
        async with client:
            await transport.request(
                "GET",
                "/v1/health",
                seconds=1.0,
                response_model=HealthResponse,
                headers={"x-trace": "keep-me", EXPECTED_BOOT_UUID_HEADER: "caller-guess"},
            )

        assert seen[0]["x-trace"] == "keep-me"
        assert seen[0][EXPECTED_BOOT_UUID_HEADER] == "boot-a"

    async def test_malformed_200_response_is_rejected(self) -> None:
        # [FUNCTION DOC] The handler answers 200 with a body that is not json; asserts a json decode
        # error is raised by the request.
        """A 200 response whose body is not JSON fails loudly instead of yielding a result."""

        def handler(request: httpx.Request) -> httpx.Response:
            # [FUNCTION DOC] HELPER: Answers 200 with plain text that is not valid json.
            return httpx.Response(200, text="not json at all", request=request)

        transport, client = _transport_over(handler)
        async with client:
            with pytest.raises(json.JSONDecodeError):
                await transport.request(
                    "GET",
                    "/v1/health",
                    seconds=1.0,
                    response_model=HealthResponse,
                )

    async def test_success_status_with_invalid_response_envelope_is_rejected(self) -> None:
        # [FUNCTION DOC] The handler answers 200 with json holding an unknown status value and an extra
        # field; asserts a pydantic validation error is raised.
        """A 200 response whose JSON does not match the protocol model is refused, not passed on."""

        def handler(request: httpx.Request) -> httpx.Response:
            # [FUNCTION DOC] HELPER: Answers 200 with a json body that violates the health model.
            return httpx.Response(200, json={"status": "made-up", "extra": 1}, request=request)

        transport, client = _transport_over(handler)
        async with client:
            with pytest.raises(ValidationError):
                await transport.request(
                    "GET",
                    "/v1/health",
                    seconds=1.0,
                    response_model=HealthResponse,
                )

    async def test_without_client_override_uses_general_http_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Installs a mock-backed client into the general http client provider's per
        # event loop client table (restored by the monkeypatch), builds a transport with no http client
        # of its own, performs a request, and asserts the response is a default health response and the
        # handler saw exactly the url http://testserver/v1/health.
        """Without an injected client the request goes through the shared general http client."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            # [FUNCTION DOC] HELPER: Records the request url as text and answers 200 status ok.
            seen.append(str(request.url))
            return httpx.Response(200, json={"status": "ok"}, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setitem(GeneralHttpClientProvider._clients, asyncio.get_running_loop(), client)
        transport = RpcTransport(
            server_url="http://testserver",
            http_client=None,
            boot_uuid_pin=BootUuidPin(required=False, worker_cls_name="Worker"),
        )

        async with client:
            response = await transport.request(
                "GET",
                "/v1/health",
                seconds=1.0,
                response_model=HealthResponse,
            )

        assert response == HealthResponse()
        assert seen == ["http://testserver/v1/health"]

    async def test_boot_uuid_mismatch_status_raises_server_restarted(self) -> None:
        # [FUNCTION DOC] The handler answers status 412 with a boot uuid header and text "stale boot";
        # asserts a server-restarted error whose message contains "restarted", even though the pin is
        # not required.
        """A 412 response is mapped to ServerRestartedError."""

        def handler(request: httpx.Request) -> httpx.Response:
            # [FUNCTION DOC] HELPER: Answers 412 with boot uuid header boot-b and text "stale boot".
            return httpx.Response(
                412,
                headers={BOOT_UUID_HEADER: "boot-b"},
                text="stale boot",
                request=request,
            )

        transport, client = _transport_over(handler)
        async with client:
            with pytest.raises(ServerRestartedError, match="restarted"):
                await transport.request(
                    "GET",
                    "/v1/health",
                    seconds=1.0,
                    response_model=HealthResponse,
                )


class TestUnpinningThePin:
    # [CLASS DOC] Tests the unpin operation, which forgets the pinned boot uuid so the next verified
    # response is adopted as the new pin.
    def test_unpinning_forgets_the_pinned_value(self):
        # [FUNCTION DOC] Pins boot-a, unpins, and asserts the expected value is nothing and a handshake
        # is needed again.
        """Only a wait that was told to expect a replaced server process may accept a new boot uuid."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))

        pin.unpin()

        assert pin.expected is None
        assert pin.needs_handshake() is True

    def test_an_unpinned_pin_adopts_the_next_boot_uuid(self):
        # [FUNCTION DOC] Pins boot-a, unpins, verifies a response with boot-b, and asserts the
        # expected value is now boot-b.
        """The replacement process is the one this client drives from now on."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))
        pin.unpin()

        pin.verify(_response(boot_uuid="boot-b"))

        assert pin.expected == "boot-b"

    def test_an_unpinned_pin_is_strict_again_afterwards(self):
        # [FUNCTION DOC] Unpins a fresh pin, adopts boot-b, then asserts that verifying boot-c raises a
        # server-restarted error mentioning boot-c.
        """Fencing has to come back the moment readiness is established, or a silent restart passes."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.unpin()
        pin.verify(_response(boot_uuid="boot-b"))

        with pytest.raises(ServerRestartedError, match="boot-c"):
            pin.verify(_response(boot_uuid="boot-c"))

    def test_unpinning_an_optional_pin_changes_nothing(self):
        # [FUNCTION DOC] Unpins an optional pin and asserts the expected value stays nothing and no
        # handshake is needed.
        """A client that does not require a stable server has nothing to forget."""
        pin = BootUuidPin(required=False, worker_cls_name="Worker")

        pin.unpin()

        assert pin.expected is None
        assert pin.needs_handshake() is False


class TestRestoringThePin:
    # [CLASS DOC] Tests that repin restores the value returned by unpin.
    def test_a_restored_pin_fences_against_the_process_it_was_pinned_to(self):
        # [FUNCTION DOC] Pins boot-a, unpins and immediately repins with the value the unpin returned;
        # asserts the expected value is boot-a again and no handshake is needed.
        """A readiness wait that never succeeded must not leave the fence open for the next ordinary call."""
        pin = BootUuidPin(required=True, worker_cls_name="Worker")
        pin.verify(_response(boot_uuid="boot-a"))

        pin.repin(pin.unpin())

        assert pin.expected == "boot-a"
        assert pin.needs_handshake() is False

    def test_unpinning_an_unpinned_pin_answers_nothing(self):
        # [FUNCTION DOC] Asserts that unpinning a required pin that never recorded a uuid returns
        # nothing (no baseline to restore).
        """A handle that never handshook has nothing to restore, and must not invent a baseline."""
        assert BootUuidPin(required=True, worker_cls_name="Worker").unpin() is None
