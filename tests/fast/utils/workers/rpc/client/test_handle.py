# [MODULE DOC] Unit tests for the rpc worker handle (the client object that turns attribute calls
# into rpc submits and polls). Tests run the real rpc server application in process through ASGI
# transports, wrapped by a hook transport that can count requests, inject transport failures or
# replacement responses, record timeouts, or swap the underlying server mid-test (to simulate
# restarts). Covered areas: typed calls and result revival, fire-and-forget submits, local argument
# validation, the readiness handshake, submit retry classification and timing, poll failure handling,
# call timeouts, long poll windows, boot uuid pinning and its restore rules, wait-ready, wait-dead,
# wait-idle, positional argument binding, error chain traversal, and the call store keeping the
# outcome of an abandoned call. Timing constants of the client modules are patched with monkeypatch
# to keep tests fast.
import asyncio
import contextlib
import errno
import json
import logging
import socket
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from tests.fast.utils.workers.rpc.client.fake_transports import PollWindowRecordingTransport, StallingPollTransport

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers import worker_handle as worker_handle_module
from miles.utils.workers.rpc.client import call as rpc_client_module
from miles.utils.workers.rpc.client import handle as rpc_handle_module
from miles.utils.workers.rpc.client import misc as rpc_misc_module
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.client.misc import RpcProtocolError, RpcWorkerCallError, ServerRestartedError
from miles.utils.workers.rpc.common.protocol import (
    BOOT_UUID_HEADER,
    CALL_STATUS_PATH,
    EXPECTED_BOOT_UUID_HEADER,
    HEALTH_PATH,
    IN_FLIGHT_PATH,
)
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.worker_handle import WorkerStillBusyError, WorkerUnreachableError


class _Item(StrictBaseModel):
    # [CLASS DOC] HELPER: Strict pydantic model with a text name and an integer value, used to check
    # that model results are revived on the client.
    name: str
    value: int


class _Worker:
    # [CLASS DOC] HELPER: The main test worker. It exposes a method with a default argument that counts
    # its calls, an async method echoing a model, a method that raises, and a method that blocks on a
    # thread event so tests can keep a call pending. Two thread events (block forever, hang started)
    # coordinate with tests.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the block-forever and hang-started thread events (unset) and
        # sets the call counter to zero.
        self.block_forever = threading.Event()
        self.hang_started = threading.Event()
        self.calls = 0

    def demo_default_arg(self, a: int, b: int = 100) -> int:
        # [FUNCTION DOC] HELPER: Increments the call counter and returns a plus b.
        self.calls += 1
        return a + b

    async def demo_model(self, item: _Item) -> _Item:
        # [FUNCTION DOC] HELPER: Returns the given item unchanged.
        return item

    def demo_raises(self) -> None:
        # [FUNCTION DOC] HELPER: Always raises a value error with the message "exploded".
        raise ValueError("exploded")

    def demo_hang(self) -> str:
        # [FUNCTION DOC] HELPER: Sets the hang-started event, then blocks (in a server thread) until the
        # block-forever event is set, asserting it is set within 30 seconds, and returns "done".
        self.hang_started.set()
        assert self.block_forever.wait(timeout=30.0)
        return "done"


class _WiderWorker(_Worker):
    # [CLASS DOC] HELPER: Subclass of the main worker adding a method the served worker does not have;
    # used to create a client whose class knows more methods than the server.
    def demo_extra(self) -> int:
        # [FUNCTION DOC] HELPER: Returns one.
        return 1


class _PositionalWorker:
    # [CLASS DOC] HELPER: Worker used to test positional argument binding: a join method with two
    # ordinary parameters and one keyword-only parameter, and a method with no parameters.
    def demo_join(self, first: str, second: str, *, separator: str = "-") -> str:
        # [FUNCTION DOC] HELPER: Returns first, separator and second concatenated.
        return f"{first}{separator}{second}"

    def demo_nothing(self) -> int:
        # [FUNCTION DOC] HELPER: Returns zero.
        return 0


class _HookTransport(httpx.AsyncBaseTransport):
    # [CLASS DOC] HELPER: Transport wrapping an ASGI app (or nothing). For every request it counts it,
    # stores it and its monotonic time, then in order: raises the configured error if one was set,
    # lets an optional hook return a replacement response or raise, and otherwise forwards to the inner
    # ASGI transport. The inner app can be swapped at any time to simulate a server replacement.
    def __init__(
        self,
        app: Any,
        hook: Callable[[httpx.Request], httpx.Response | None] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Initializes request counters and lists, wraps the app in an ASGI
        # transport unless the app is nothing, and stores the optional hook and no error.
        self.requests = 0
        self.request_times: list[float] = []
        self.seen: list[httpx.Request] = []
        self._inner = httpx.ASGITransport(app=app) if app is not None else None
        self._hook = hook
        self._error: Exception | None = None

    def switch_to(self, app: Any) -> None:
        # [FUNCTION DOC] HELPER: Replaces the inner transport with an ASGI transport over another app.
        self._inner = httpx.ASGITransport(app=app)

    def fail_with(self, error: Exception) -> None:
        # [FUNCTION DOC] HELPER: Makes every later request raise the given exception.
        self._error = error

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC] HELPER: Counts and records the request (and its time), raises the stored error
        # if any, returns the hook's replacement response if the hook gives one, otherwise forwards to
        # the inner transport (asserting that one exists).
        self.requests += 1
        self.seen.append(request)
        self.request_times.append(time.monotonic())
        if self._error is not None:
            raise self._error
        if self._hook is not None:
            replacement = self._hook(request)
            if replacement is not None:
                return replacement
        assert self._inner is not None
        return await self._inner.handle_async_request(request)

    def polls(self) -> list[httpx.Request]:
        # [FUNCTION DOC] HELPER: Returns the recorded requests that are GETs whose url contains the call
        # status path fragment (the polls for call results).
        return [r for r in self.seen if r.method == "GET" and "/v1/calls/" in str(r.url)]


class _PollRecordingTransport(_HookTransport):
    # [CLASS DOC] HELPER: Hook transport that also records the status field of each poll response body.
    def __init__(self, app: Any) -> None:
        # [FUNCTION DOC] HELPER: Runs the base constructor and starts an empty poll status list.
        super().__init__(app)
        self.poll_statuses: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC] HELPER: Gets the base response; for a poll request, reads the full body and
        # appends its json status field to the status list. Returns the response.
        response = await super().handle_async_request(request)
        if request.method == "GET" and "/v1/calls/" in str(request.url):
            await response.aread()
            self.poll_statuses.append(response.json()["status"])
        return response


def _fail_hook(
    times: int,
    method: str | None = None,
    path_fragment: str = "",
    *,
    error_type: type[httpx.TransportError] = httpx.ConnectError,
) -> Callable[[httpx.Request], httpx.Response | None]:
    # [FUNCTION DOC] HELPER: Builds a hook for the hook transport that raises a transport error of the
    # given type (default connect error) with the message "injected transport failure" for matching
    # requests. A request matches when the method equals the given one (or any method when none is
    # given) and the url contains the path fragment (empty matches all). It fails the first "times"
    # matching requests; a negative count never reaches zero, so it fails forever. Non-matching or
    # later requests get nothing (pass through).
    remaining = [times]

    def hook(request: httpx.Request) -> httpx.Response | None:
        # [FUNCTION DOC] HELPER: Raises the configured error while matching and the budget is not
        # zero (decrementing it); otherwise returns nothing.
        matches = (method is None or request.method == method) and path_fragment in str(request.url)
        if matches and remaining[0] != 0:
            remaining[0] -= 1
            raise error_type("injected transport failure", request=request)
        return None

    return hook


def _status_hook(
    *,
    status_code: int,
    times: int,
    method: str,
    path_fragment: str = "",
) -> Callable[[httpx.Request], httpx.Response | None]:
    # [FUNCTION DOC] HELPER: Builds a hook that answers matching requests with a response of the given
    # status and body text "upstream busy" instead of reaching the server. A request matches when its
    # method equals the given one and its url contains the path fragment. The first "times" matches are
    # answered (a negative count means forever); the rest pass through.
    remaining = [times]

    def hook(request: httpx.Request) -> httpx.Response | None:
        # [FUNCTION DOC] HELPER: Returns the canned error response while matching and the budget is not
        # zero (decrementing it); otherwise returns nothing.
        if request.method == method and path_fragment in str(request.url) and remaining[0] != 0:
            remaining[0] -= 1
            return httpx.Response(status_code, text="upstream busy", request=request)
        return None

    return hook


@pytest.fixture
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Fixture that shortens the client retry timing through monkeypatching: the
    # submit retry window becomes 0.5 seconds; the initial retry delay becomes 0.02 seconds (both in the
    # call module and the handle module); the maximum retry delay becomes 0.1 seconds; and the handle's
    # idle poll interval becomes 0.02 seconds. Undone at test end.
    monkeypatch.setattr(rpc_client_module, "SUBMIT_RETRY_WINDOW_SECONDS", 0.5)
    monkeypatch.setattr(rpc_client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.02)
    monkeypatch.setattr(rpc_client_module, "RETRY_MAX_DELAY_SECONDS", 0.1)
    monkeypatch.setattr(rpc_handle_module, "RETRY_INITIAL_DELAY_SECONDS", 0.02)
    monkeypatch.setattr(rpc_handle_module, "_IDLE_POLL_INTERVAL_SECONDS", 0.02)


@contextlib.asynccontextmanager
async def _running_app(worker: object) -> AsyncIterator:
    # [FUNCTION DOC] HELPER: Async context manager that creates the rpc server application for the
    # given worker object, runs its lifespan startup, yields the app, and runs lifespan shutdown on exit.
    app = create_rpc_app(worker)
    async with app.router.lifespan_context(app):
        yield app


@contextlib.asynccontextmanager
async def _handle_over(
    transport: httpx.AsyncBaseTransport,
    worker_cls: type = _Worker,
    require_stable_boot_uuid: bool = False,
    call_timeout_seconds: float = 3600.0,
    ready_timeout_seconds: float = rpc_handle_module.DEFAULT_READY_TIMEOUT_SECONDS,
    follow_redirects: bool = False,
) -> AsyncIterator[RpcWorkerHandle]:
    # [FUNCTION DOC] HELPER: Async context manager that opens an http client over the given transport
    # (optionally following redirects) and yields an rpc worker handle for the given worker class (the
    # main worker by default) at server url http://testserver, with the given stable boot uuid
    # requirement (default off), call timeout (default 3600 seconds) and ready timeout (default the
    # module default). The client is closed on exit.
    async with httpx.AsyncClient(transport=transport, follow_redirects=follow_redirects) as http_client:
        yield RpcWorkerHandle(
            worker_cls,
            server_url="http://testserver",
            require_stable_boot_uuid=require_stable_boot_uuid,
            call_timeout_seconds=call_timeout_seconds,
            ready_timeout_seconds=ready_timeout_seconds,
            http_client=http_client,
        )


class TestRepresentation:
    # [CLASS DOC] Tests the textual representation of a handle.
    async def test_repr_names_the_worker_class(self) -> None:
        # [FUNCTION DOC] Asserts the repr of a handle for the main worker is exactly the handle type
        # name followed by the worker class name in parentheses.
        """The handle representation names both the handle type and its worker class."""
        async with _handle_over(_HookTransport(None)) as handle:
            assert repr(handle) == "RpcWorkerHandle(_Worker)"


class TestTypedCalls:
    # [CLASS DOC] Tests ordinary end-to-end calls through the handle against the in-process server.
    async def test_scalar_roundtrip(self):
        # [FUNCTION DOC] A call with a=1, b=2 returns 3.
        """A typed call returns the deserialized scalar result."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            assert await handle.demo_default_arg(a=1, b=2) == 3

    async def test_default_argument_applied(self):
        # [FUNCTION DOC] A call with only a=1 returns 101, the default b being 100.
        """An omitted defaulted argument uses the declared default value."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            assert await handle.demo_default_arg(a=1) == 101

    async def test_positional_arguments_are_bound_by_name(self):
        # [FUNCTION DOC] A call with positionals one and two returns 3.
        """A positional call is bound to parameter names client-side, so the wire stays keyword-shaped."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            assert await handle.demo_default_arg(1, 2) == 3

    async def test_a_positional_and_keyword_duplicate_is_rejected_locally(self):
        # [FUNCTION DOC] A call giving a both positionally and by keyword raises a type error
        # mentioning multiple values.
        """The same parameter given twice is a caller bug and must not reach the server."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            with pytest.raises(TypeError, match="multiple values"):
                await handle.demo_default_arg(1, a=2)

    async def test_excess_positional_arguments_are_rejected_locally(self):
        # [FUNCTION DOC] A call with three positionals for a two-parameter method raises a type error
        # saying at most 2 positional arguments.
        """More positionals than the method declares is a caller bug and must not reach the server."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            with pytest.raises(TypeError, match="at most 2 positional arguments"):
                await handle.demo_default_arg(1, 2, 3)

    async def test_model_result_revived(self):
        # [FUNCTION DOC] A call echoing a model returns an equal item that is an instance of the item
        # model class.
        """Pydantic model results come back as real model instances."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            result = await handle.demo_model(item=_Item(name="x", value=7))
            assert result == _Item(name="x", value=7) and isinstance(result, _Item)

    async def test_remote_exception_raises_call_error(self):
        # [FUNCTION DOC] A call whose worker method raises a value error makes the handle raise a
        # worker call error whose message mentions the value error type.
        """A remote business exception raises RpcWorkerCallError carrying the traceback."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            with pytest.raises(RpcWorkerCallError, match="ValueError"):
                await handle.demo_raises()


class TestSubmitWithoutResult:
    # [CLASS DOC] Tests the fire-and-forget submit method of the handle, which posts a call and never
    # polls for its result.
    async def test_first_fire_and_forget_call_handshakes_before_submission(self) -> None:
        # [FUNCTION DOC] With a stable boot uuid requirement, a fire-and-forget call must first send a
        # health request and then the submit. Records the boot uuid header of the health response and
        # asserts the exact request order (health GET, submit POST) and that the submit carried the
        # expected boot uuid header equal to the recorded one.
        """A first stable-boot fire-and-forget call pins the server before submitting."""

        class _HealthBootUuidRecordingTransport(_HookTransport):
            # [CLASS DOC] HELPER: Hook transport that remembers the boot uuid header of the health
            # response.
            def __init__(self, app: Any) -> None:
                # [FUNCTION DOC] HELPER: Runs the base constructor and starts with no recorded uuid.
                super().__init__(app)
                self.health_boot_uuid: str | None = None

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                # [FUNCTION DOC] HELPER: Forwards through the base; for the health GET, stores the boot
                # uuid header of the response. Returns the response.
                response = await super().handle_async_request(request)
                if request.method == "GET" and request.url.path == HEALTH_PATH:
                    self.health_boot_uuid = response.headers[BOOT_UUID_HEADER]
                return response

        async with _running_app(_Worker()) as app:
            transport = _HealthBootUuidRecordingTransport(app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.submit_without_result("demo_default_arg", a=1, b=2)

        assert [(request.method, request.url.path) for request in transport.seen] == [
            ("GET", HEALTH_PATH),
            ("POST", "/v1/demo_default_arg"),
        ]
        assert transport.health_boot_uuid is not None
        assert transport.seen[1].headers[EXPECTED_BOOT_UUID_HEADER] == transport.health_boot_uuid

    async def test_a_call_is_submitted_and_never_polled(self):
        # [FUNCTION DOC] After a fire-and-forget submit, asserts no poll request was made and exactly
        # one request in total was sent.
        """Some calls kill the process that would answer them, so waiting for a result waits for ever."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport) as handle:
                await handle.submit_without_result("demo_default_arg", a=1, b=2)

                assert transport.polls() == []
                assert transport.requests == 1

    async def test_a_method_the_worker_does_not_define_fails_locally(self):
        # [FUNCTION DOC] Submitting a misspelled method name raises an attribute error saying there is
        # no rpc method, and no request is made.
        """A typo would otherwise be submitted as a call nobody ever runs and nobody ever waits for."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport) as handle:
                with pytest.raises(AttributeError, match="no rpc method"):
                    await handle.submit_without_result("demo_defualt_arg")

                assert transport.requests == 0

    async def test_locally_invalid_arguments_fail_before_any_request(self):
        # [FUNCTION DOC] Submitting a string for an integer parameter raises a validation error and no
        # request is made.
        """The result is never read, so an argument the worker rejects would fail where nobody is looking."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport) as handle:
                with pytest.raises(ValidationError):
                    await handle.submit_without_result("demo_default_arg", a="not-an-int")

                assert transport.requests == 0


class TestLocalValidation:
    # [CLASS DOC] Tests checks the handle performs locally, before any network traffic.
    async def test_worker_shadowing_handle_method_rejected(self):
        # [FUNCTION DOC] Creating a handle for a worker class that defines a method named like a
        # reserved handle method (wait ready) raises a type error mentioning shadowing.
        """A worker exposing a reserved handle method name fails at client creation."""

        class Shadowing:
            # [CLASS DOC] HELPER: Worker whose only method collides with a reserved handle method.
            async def wait_ready(self, timeout: float) -> None:
                # [FUNCTION DOC] HELPER: Does nothing; exists only for the name collision.
                pass

        with pytest.raises(TypeError, match="shadow"):
            RpcWorkerHandle(Shadowing, server_url="http://testserver")

    async def test_method_typo_raises_attribute_error(self):
        # [FUNCTION DOC] Accessing a misspelled method on a handle raises an attribute error saying
        # there is no rpc method.
        """Calling a method the worker class does not define fails locally."""
        handle = RpcWorkerHandle(_Worker, server_url="http://testserver")
        with pytest.raises(AttributeError, match="no rpc method"):
            _ = handle.demo_defualt_arg

    async def test_bad_argument_type_fails_before_any_request(self):
        # [FUNCTION DOC] A string for an integer parameter raises a validation error and no request is
        # made.
        """Locally invalid arguments raise ValidationError without touching the network."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport) as handle:
                with pytest.raises(ValidationError):
                    await handle.demo_default_arg(a="not-an-int")
                assert transport.requests == 0

    async def test_unknown_argument_fails_before_any_request(self):
        # [FUNCTION DOC] An unknown keyword argument raises a validation error and no request is made.
        """Unknown keyword arguments raise ValidationError without touching the network."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport) as handle:
                with pytest.raises(ValidationError):
                    await handle.demo_default_arg(a=1, nope=2)
                assert transport.requests == 0


class TestReadyHandshake:
    # [CLASS DOC] Tests how the implicit readiness handshake of a stable boot uuid handle uses the
    # ready timeout.
    @pytest.mark.parametrize(
        ("handle_kwargs", "expected_timeout"),
        [
            ({}, rpc_handle_module.DEFAULT_READY_TIMEOUT_SECONDS),
            ({"ready_timeout_seconds": 2.5}, 2.5),
        ],
    )
    async def test_stable_handshake_uses_configured_ready_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        handle_kwargs: dict[str, float],
        expected_timeout: float,
    ) -> None:
        # [FUNCTION DOC] Parametrized with no ready timeout (expects the default) and 2.5 seconds. The
        # handle's wait-ready method and the call's run method are replaced by fakes (the first records
        # the timeout it receives, the second returns 3), and the http layer raises an assertion error
        # on any request. After one call returning 3, asserts the default ready timeout constant is 600
        # seconds and the single recorded timeout equals the expected one.
        """A stable-boot handshake uses the default or configured ready timeout."""
        observed_timeouts: list[float] = []

        async def wait_ready(*, timeout: float) -> None:
            # [FUNCTION DOC] HELPER: Records the timeout it was called with.
            observed_timeouts.append(timeout)

        async def run(call: rpc_client_module.RpcCall) -> int:
            # [FUNCTION DOC] HELPER: Replacement for the call run method; ignores the call and returns 3.
            return 3

        def fail_if_requested(request: httpx.Request) -> httpx.Response:
            # [FUNCTION DOC] HELPER: Mock transport handler that must never run; raises an assertion
            # error naming the unexpected request.
            raise AssertionError(f"unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(fail_if_requested)) as http_client:
            handle = RpcWorkerHandle(
                _Worker,
                server_url="http://testserver",
                require_stable_boot_uuid=True,
                http_client=http_client,
                **handle_kwargs,
            )
            monkeypatch.setattr(handle, "wait_ready", wait_ready)
            monkeypatch.setattr(rpc_client_module.RpcCall, "run", run)

            assert await handle.demo_default_arg(a=1, b=2) == 3

        assert rpc_handle_module.DEFAULT_READY_TIMEOUT_SECONDS == 600.0
        assert observed_timeouts == [expected_timeout]

    async def test_implicit_stable_handshake_honors_ready_timeout_before_submit(self) -> None:
        # [FUNCTION DOC] With a ready timeout of 2.5 seconds and a stable boot uuid, the first call must
        # health-check first. Records the health request's read timeout and the response boot uuid, and
        # asserts: the first request is the health GET, the submit carries the expected boot uuid
        # header equal to the recorded one, and the health read timeout is greater than 2.0 and at most
        # 2.5 seconds.
        """The first call on a stable-boot handle really health-checks on the ready timeout before submitting."""

        class _HealthTimeoutRecordingTransport(_HookTransport):
            # [CLASS DOC] HELPER: Hook transport recording health read timeouts and the health boot uuid.
            def __init__(self, app: Any) -> None:
                # [FUNCTION DOC] HELPER: Runs the base constructor and starts the uuid and the read
                # timeout list empty.
                super().__init__(app)
                self.health_boot_uuid: str | None = None
                self.health_read_timeouts: list[float] = []

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                # [FUNCTION DOC] HELPER: For a health GET, records the read timeout before sending and
                # the response boot uuid after. Returns the base response.
                is_health = request.method == "GET" and str(request.url).endswith(HEALTH_PATH)
                if is_health:
                    self.health_read_timeouts.append(request.extensions["timeout"]["read"])
                response = await super().handle_async_request(request)
                if is_health:
                    self.health_boot_uuid = response.headers[BOOT_UUID_HEADER]
                return response

        async with _running_app(_Worker()) as app:
            transport = _HealthTimeoutRecordingTransport(app)
            async with _handle_over(transport, require_stable_boot_uuid=True, ready_timeout_seconds=2.5) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

        assert transport.seen[0].method == "GET" and str(transport.seen[0].url).endswith(HEALTH_PATH)
        submit_request = next(request for request in transport.seen if request.method == "POST")
        assert transport.health_boot_uuid is not None
        assert submit_request.headers[EXPECTED_BOOT_UUID_HEADER] == transport.health_boot_uuid
        assert transport.health_read_timeouts
        assert 2.0 < transport.health_read_timeouts[0] <= 2.5


class TestSubmitRetry:
    # [CLASS DOC] Tests which submit failures are retried (never-reached errors, inside a retry
    # window with growing backoff) and which give up at once (post-wire errors, 5xx, protocol errors,
    # redirects), plus the per-attempt time budget.
    async def test_never_reached_errors_retried_until_success(self, fast_retries: None) -> None:
        # [FUNCTION DOC] The first two submit POSTs fail with connect errors; the call still returns 3
        # and at least three requests were made.
        """Never-reached submit failures are retried until one succeeds."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(2, "POST"))
            async with _handle_over(transport) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3
                assert transport.requests >= 3

    async def test_pool_timeout_is_retried_until_submit_succeeds(self, fast_retries: None) -> None:
        # [FUNCTION DOC] The first two submit POSTs fail with pool timeouts; the call returns 3 and
        # exactly three POSTs were sent.
        """A pool timeout never handed the submit to the server, so it is retried until one lands."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(2, "POST", error_type=httpx.PoolTimeout))
            async with _handle_over(transport) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

            assert len([r for r in transport.seen if r.method == "POST"]) == 3

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    async def test_server_error_submit_gives_up_without_retry(
        self,
        status_code: int,
        fast_retries: None,
    ) -> None:
        # [FUNCTION DOC] Parametrized over 500, 502, 503 and 504: the submit POST is answered once with
        # that status. Asserts the call raises a worker unreachable error and exactly one request was
        # sent.
        """A submit 5xx becomes unreachable after exactly one attempt."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(
                app,
                hook=_status_hook(status_code=status_code, times=1, method="POST"),
            )
            async with _handle_over(transport) as handle:
                with pytest.raises(WorkerUnreachableError):
                    await handle.demo_default_arg(a=1, b=2)

        assert transport.requests == 1

    async def test_one_stalled_attempt_does_not_consume_the_whole_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Sets a 20 second retry window, a 0.05 second attempt timeout and a 0.05 second
        # abort slack, and uses a transport that hangs forever on every request. Asserts the call
        # raises a worker unreachable error in under five seconds in total.
        """A submit that stalls is abandoned on its own attempt budget, not on the whole retry window."""
        monkeypatch.setattr(rpc_client_module, "SUBMIT_RETRY_WINDOW_SECONDS", 20.0)
        monkeypatch.setattr(rpc_client_module, "SUBMIT_ATTEMPT_TIMEOUT_SECONDS", 0.05)
        monkeypatch.setattr(rpc_misc_module, "_ABORT_SLACK_SECONDS", 0.05)

        class _StallingTransport(httpx.AsyncBaseTransport):
            # [CLASS DOC] HELPER: Transport whose every request sleeps 60 seconds.
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                # [FUNCTION DOC] HELPER: Sleeps for 60 seconds then raises an assertion error (only
                # reachable if the client failed to abandon the request).
                await asyncio.sleep(60.0)
                raise AssertionError("the stalled request should have been abandoned")

        async with _handle_over(_StallingTransport()) as handle:
            started = time.monotonic()
            with pytest.raises(WorkerUnreachableError):
                await handle.demo_default_arg(a=1, b=2)
            assert time.monotonic() - started < 5.0

    async def test_a_submit_attempt_never_outlives_the_remaining_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Sets a 0.5 second retry window and a 30 second attempt timeout, with a
        # transport that records each request's read timeout and refuses the connection. After the call
        # raises a worker unreachable error, asserts at least one timeout was seen and none exceeded
        # 0.5 seconds.
        """When the retry window is shorter than the attempt budget, the attempt is cut down to the window."""
        monkeypatch.setattr(rpc_client_module, "SUBMIT_RETRY_WINDOW_SECONDS", 0.5)
        monkeypatch.setattr(rpc_client_module, "SUBMIT_ATTEMPT_TIMEOUT_SECONDS", 30.0)

        class _TimeoutRecordingTransport(httpx.AsyncBaseTransport):
            # [CLASS DOC] HELPER: Transport that records read timeouts and always refuses connection.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Starts with an empty read timeout list.
                self.read_timeouts: list[float] = []

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                # [FUNCTION DOC] HELPER: Records the request's read timeout and raises a connect error
                # "refused".
                self.read_timeouts.append(request.extensions["timeout"]["read"])
                raise httpx.ConnectError("refused", request=request)

        transport = _TimeoutRecordingTransport()
        async with _handle_over(transport) as handle:
            with pytest.raises(WorkerUnreachableError):
                await handle.demo_default_arg(a=1, b=2)

        assert transport.read_timeouts
        assert max(transport.read_timeouts) <= 0.5

    async def test_submit_attempt_budget_defaults_stay_put(self) -> None:
        # [FUNCTION DOC] Asserts the default submit attempt timeout constant is 10 seconds and the abort
        # slack constant is 1 second.
        """The submit attempt budget and its abort slack keep the values the retry design assumes."""
        assert rpc_client_module.SUBMIT_ATTEMPT_TIMEOUT_SECONDS == 10.0
        assert rpc_misc_module._ABORT_SLACK_SECONDS == 1.0

    async def test_a_redirected_submit_is_never_followed(self) -> None:
        # [FUNCTION DOC] Uses a client that follows redirects and a hook answering every submit POST
        # with a 307 redirect to another host. Asserts the call raises an rpc protocol error and only
        # one request was sent (the redirect was not followed).
        """A redirect is refused outright, so an injected client cannot deliver one submit twice."""

        def redirect_submits(request: httpx.Request) -> httpx.Response | None:
            # [FUNCTION DOC] HELPER: Answers POSTs with a 307 and a location header pointing elsewhere;
            # other requests pass through (nothing returned).
            if request.method != "POST":
                return None
            return httpx.Response(307, headers={"location": "http://elsewhere/v1/demo_default_arg"}, request=request)

        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=redirect_submits)
            async with _handle_over(transport, follow_redirects=True) as handle:
                with pytest.raises(RpcProtocolError):
                    await handle.demo_default_arg(a=1, b=2)
            assert transport.requests == 1

    async def test_a_late_connect_timeout_is_still_classified_as_never_reached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Sets a 6 second retry window, 0.2 second attempt timeout, tiny retry delays and
        # a 0.4 second abort slack. The transport waits past the connect timeout by 0.2 seconds and
        # then raises a connect timeout. Asserts the call ends with a worker unreachable error after
        # more than one attempt, i.e. the late connect timeout was retried and not treated as an abort.
        """The abort deadline sits behind the http timeout, so a connect timeout stays retryable."""
        monkeypatch.setattr(rpc_client_module, "SUBMIT_RETRY_WINDOW_SECONDS", 6.0)
        monkeypatch.setattr(rpc_client_module, "SUBMIT_ATTEMPT_TIMEOUT_SECONDS", 0.2)
        monkeypatch.setattr(rpc_client_module, "RETRY_INITIAL_DELAY_SECONDS", 0.02)
        monkeypatch.setattr(rpc_client_module, "RETRY_MAX_DELAY_SECONDS", 0.05)
        monkeypatch.setattr(rpc_misc_module, "_ABORT_SLACK_SECONDS", 0.4)

        class _LateConnectTimeout(httpx.AsyncBaseTransport):
            # [CLASS DOC] HELPER: Transport counting attempts and raising a connect timeout late.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Starts the attempt counter at zero.
                self.attempts = 0

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                # [FUNCTION DOC] HELPER: Counts the attempt, sleeps for the request's connect timeout
                # plus 0.2 seconds, then raises a connect timeout error.
                self.attempts += 1
                await asyncio.sleep(request.extensions["timeout"]["connect"] + 0.2)
                raise httpx.ConnectTimeout("connect timed out", request=request)

        transport = _LateConnectTimeout()
        async with _handle_over(transport) as handle:
            with pytest.raises(WorkerUnreachableError):
                await handle.demo_default_arg(a=1, b=2)
        assert transport.attempts > 1

    async def test_window_exhausted_raises_unreachable(self, fast_retries: None) -> None:
        # [FUNCTION DOC] Every request fails forever (no server). Asserts a worker unreachable error,
        # at least one attempt, and that no request started later than the retry window after the call
        # began.
        """A dead server exhausts the retry window, which also bounds when requests stop."""
        transport = _HookTransport(None, hook=_fail_hook(-1))
        async with _handle_over(transport) as handle:
            started = time.monotonic()
            with pytest.raises(WorkerUnreachableError):
                await handle.demo_default_arg(a=1, b=2)

            window = rpc_client_module.SUBMIT_RETRY_WINDOW_SECONDS
            assert transport.request_times, "expected at least one submit attempt"
            assert max(transport.request_times) - started <= window

    async def test_backoff_grows_between_attempts(self, fast_retries: None) -> None:
        # [FUNCTION DOC] Every request fails forever. Computes the gaps between consecutive request
        # times and asserts there are at least three, the first gap is at least the initial retry delay
        # and the second gap is larger than the first.
        """Retry gaps grow instead of hammering the server at a fixed rate."""
        transport = _HookTransport(None, hook=_fail_hook(-1))
        async with _handle_over(transport) as handle:
            with pytest.raises(WorkerUnreachableError):
                await handle.demo_default_arg(a=1, b=2)

            gaps = [b - a for a, b in zip(transport.request_times, transport.request_times[1:], strict=False)]
            assert len(gaps) >= 3, f"expected several retries, got {len(gaps)}"
            assert gaps[0] >= rpc_client_module.RETRY_INITIAL_DELAY_SECONDS
            assert gaps[1] > gaps[0]

    @pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ReadError])
    async def test_post_wire_submit_error_gives_up_without_retry(
        self,
        error_type: type[httpx.TransportError],
        fast_retries: None,
    ) -> None:
        # [FUNCTION DOC] Parametrized over read timeout and read error injected on every submit POST.
        # Asserts a worker unreachable error and exactly one request (no retry, since the server may
        # have received the submit).
        """A post-wire submit transport error gives up after one attempt."""
        transport = _HookTransport(
            None,
            hook=_fail_hook(-1, "POST", error_type=error_type),
        )
        async with _handle_over(transport) as handle:
            with pytest.raises(WorkerUnreachableError):
                await handle.demo_default_arg(a=1, b=2)

        assert transport.requests == 1

    async def test_protocol_error_not_retried(self) -> None:
        # [FUNCTION DOC] A client for the wider worker calls a method the server does not have; asserts
        # an rpc protocol error with status 404 and exactly one request.
        """A 4xx protocol error raises immediately without retries."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport, worker_cls=_WiderWorker) as handle:
                with pytest.raises(RpcProtocolError) as exc_info:
                    await handle.demo_extra()
                assert exc_info.value.status_code == 404
                assert transport.requests == 1


class TestPollFailure:
    # [CLASS DOC] Tests how failures while polling for a submitted call's result are handled.
    async def test_server_death_after_submit_exhausts_as_timeout(self, fast_retries: None) -> None:
        # [FUNCTION DOC] All poll GETs fail forever with a 0.2 second call timeout; the call raises a
        # built-in timeout error.
        """Poll transport failures exhaust as TimeoutError."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(-1, "GET", "/v1/calls/"))
            async with _handle_over(transport, call_timeout_seconds=0.2) as handle:
                with pytest.raises(TimeoutError):
                    await handle.demo_default_arg(a=1, b=2)

    async def test_transient_poll_loss_recovers(self, fast_retries: None) -> None:
        # [FUNCTION DOC] The first two poll GETs fail; with a 10 second call timeout the call still
        # returns 3.
        """A poll failure before the deadline is retried and the call still returns."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(2, "GET", "/v1/calls/"))
            async with _handle_over(transport, call_timeout_seconds=10.0) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

    async def test_transient_poll_5xx_recovers(self, fast_retries: None) -> None:
        # [FUNCTION DOC] The first poll GET is answered with 503; the call returns 3 and at least two
        # polls were made.
        """A transient poll 5xx is retried and the call succeeds."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(
                app,
                hook=_status_hook(
                    status_code=503,
                    times=1,
                    method="GET",
                    path_fragment="/v1/calls/",
                ),
            )
            async with _handle_over(transport, call_timeout_seconds=2.0) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

        assert len(transport.polls()) >= 2

    async def test_poll_non_200_raises_protocol_error(self, fast_retries: None) -> None:
        # [FUNCTION DOC] The first poll GET is answered with 404; asserts an rpc protocol error with
        # status 404 and exactly one poll (no retry).
        """A poll 4xx raises RpcProtocolError without retry."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(
                app,
                hook=_status_hook(
                    status_code=404,
                    times=1,
                    method="GET",
                    path_fragment="/v1/calls/",
                ),
            )
            async with _handle_over(transport, call_timeout_seconds=2.0) as handle:
                with pytest.raises(RpcProtocolError) as exc_info:
                    await handle.demo_default_arg(a=1, b=2)

        assert exc_info.value.status_code == 404
        assert len(transport.polls()) == 1

    async def test_long_poll_timeout_silently_repolls(self, fast_retries: None) -> None:
        # [FUNCTION DOC] The first poll GET fails with a read timeout; the call returns 3 and at least
        # two polls were made.
        """A long-poll timeout is silently retried."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(
                app,
                hook=_fail_hook(1, "GET", "/v1/calls/", error_type=httpx.ReadTimeout),
            )
            async with _handle_over(transport, call_timeout_seconds=2.0) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

        assert len(transport.polls()) >= 2


class TestCallTimeout:
    # [CLASS DOC] Tests the overall call deadline for a call that never finishes.
    async def test_pending_past_deadline_raises_timeout(self):
        # [FUNCTION DOC] Calls the blocking method with a 0.3 second call timeout; asserts a built-in
        # timeout error, at least one poll, and that every poll's server timeout query parameter is at
        # most 0.3. Finally releases the blocked worker thread.
        """A call still pending past the call timeout raises, and polls never outlast the budget."""
        worker = _Worker()
        async with _running_app(worker) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport, call_timeout_seconds=0.3) as handle:
                with pytest.raises(TimeoutError):
                    await handle.demo_hang()

                assert transport.polls(), "expected at least one poll"
                assert all(float(r.url.params["timeout"]) <= 0.3 for r in transport.polls())
                worker.block_forever.set()


class TestLongPoll:
    # [CLASS DOC] Tests the long poll windows (how long the client asks the server to hang and how
    # long it waits locally) by patching the default poll timeout to be small.
    async def test_poll_leaves_the_server_room_to_answer_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With a default poll timeout of 0.4 seconds, starts a blocking call, waits one
        # second, releases the worker, and asserts the call returns "done", at least one poll answered
        # with a pending status, and every poll asked the server to wait strictly less than 0.4 seconds.
        """The client asks the server to hang for less than it will wait, so a pending answer arrives."""
        monkeypatch.setattr(rpc_client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.4)
        worker = _Worker()
        async with _running_app(worker) as app:
            transport = _PollRecordingTransport(app)
            async with _handle_over(transport, call_timeout_seconds=5.0) as handle:
                pending = asyncio.create_task(handle.demo_hang())
                await asyncio.sleep(1.0)
                worker.block_forever.set()

                assert await pending == "done"
                assert "pending" in transport.poll_statuses
                assert all(float(request.url.params["timeout"]) < 0.4 for request in transport.polls())

    async def test_each_poll_waits_only_for_its_own_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With a default poll timeout of 0.4 seconds and a 30 second call timeout, records
        # each poll's local read timeout through a custom transport; asserts the call returns "done",
        # at least one poll was recorded and the largest read timeout is at most 0.4 seconds.
        """A poll's local http timeout follows the poll window, not the whole remaining call budget."""
        monkeypatch.setattr(rpc_client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.4)
        worker = _Worker()

        class _PollTimeoutRecordingTransport(_HookTransport):
            # [CLASS DOC] HELPER: Hook transport recording the read timeout of each poll request.
            def __init__(self, app: Any) -> None:
                # [FUNCTION DOC] HELPER: Runs the base constructor and starts an empty timeout list.
                super().__init__(app)
                self.poll_read_timeouts: list[float] = []

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                # [FUNCTION DOC] HELPER: For a poll GET, appends the request's read timeout; then
                # forwards via the base transport.
                if request.method == "GET" and "/v1/calls/" in str(request.url):
                    self.poll_read_timeouts.append(request.extensions["timeout"]["read"])
                return await super().handle_async_request(request)

        async with _running_app(worker) as app:
            transport = _PollTimeoutRecordingTransport(app)
            async with _handle_over(transport, call_timeout_seconds=30.0) as handle:
                pending = asyncio.create_task(handle.demo_hang())
                await asyncio.sleep(1.0)
                worker.block_forever.set()

                assert await pending == "done"

        assert transport.poll_read_timeouts
        assert max(transport.poll_read_timeouts) <= 0.4

    async def test_outer_poll_timeout_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With a 0.1 second poll timeout and 0.05 second abort slack, the first two polls
        # stall until abandoned. Asserts the call still returns 3 and at least three polls were made.
        """A poll abandoned by its own abort deadline is repolled instead of failing the call."""
        monkeypatch.setattr(rpc_client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.1)
        monkeypatch.setattr(rpc_misc_module, "_ABORT_SLACK_SECONDS", 0.05)

        async with _running_app(_Worker()) as app:
            transport = StallingPollTransport(app, stalled_polls=2)
            async with _handle_over(transport, call_timeout_seconds=5.0) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

        assert transport.polls >= 3

    @pytest.mark.parametrize("slack_seconds", [5.0, 0.1])
    async def test_poll_window_keeps_positive_server_wait_and_bounded_client_slack(
        self, monkeypatch: pytest.MonkeyPatch, slack_seconds: float
    ) -> None:
        # [FUNCTION DOC] Parametrized over poll slack of 5.0 and 0.1 seconds, with a 0.4 second default
        # poll timeout. Records each poll's client read timeout and server timeout parameter while a
        # blocking call is held for one second. For at least two polls, asserts: the server wait is
        # positive and strictly below the client wait, the client wait is at most 0.4, the server wait
        # is at least half the client wait, and the difference is at most the slack.
        """Every poll asks the server for a positive share of the window while keeping client-side slack."""
        monkeypatch.setattr(rpc_client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.4)
        monkeypatch.setattr(rpc_client_module, "POLL_SLACK_SECONDS", slack_seconds)
        worker = _Worker()

        async with _running_app(worker) as app:
            transport = PollWindowRecordingTransport(app)
            async with _handle_over(transport, call_timeout_seconds=5.0) as handle:
                pending = asyncio.create_task(handle.demo_hang())
                await asyncio.sleep(1.0)
                worker.block_forever.set()

                assert await pending == "done"

        assert len(transport.poll_windows) >= 2
        for window in transport.poll_windows:
            assert 0.0 < window.server_seconds < window.client_seconds <= 0.4
            assert window.server_seconds >= window.client_seconds / 2
            assert window.client_seconds - window.server_seconds <= slack_seconds

    async def test_poll_slack_default_stays_put(self) -> None:
        # [FUNCTION DOC] Asserts the default poll slack constant is 5 seconds.
        """The poll slack keeps the value that lets the server answer before the client gives up."""
        assert rpc_client_module.POLL_SLACK_SECONDS == 5.0

    async def test_restart_during_polling_is_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With a stable boot uuid handle, a 0.1 second poll timeout and a 10 second call
        # timeout, starts a blocking call on the first server, waits 0.5 seconds, swaps the transport
        # to a second server, and asserts the pending call fails with a server-restarted error. Then
        # releases the first worker's thread.
        """A pinned handle notices a server swap that happens while it is already polling."""
        monkeypatch.setattr(rpc_client_module, "DEFAULT_POLL_TIMEOUT_SECONDS", 0.1)
        worker = _Worker()
        async with _running_app(worker) as first_app, _running_app(_Worker()) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True, call_timeout_seconds=10.0) as handle:
                pending = asyncio.create_task(handle.demo_hang())
                await asyncio.sleep(0.5)
                transport.switch_to(second_app)

                with pytest.raises(ServerRestartedError):
                    await pending
                worker.block_forever.set()


class _StallingAfterwardsTransport(_HookTransport):
    # [CLASS DOC] HELPER: Hook transport that behaves normally until its stalling flag is set; after
    # that every request sleeps for an hour (until cancelled).
    """Answers normally until `stalling` is set, after which every request hangs until it is cancelled."""

    def __init__(self, app: Any) -> None:
        # [FUNCTION DOC] HELPER: Runs the base constructor and starts with stalling off.
        super().__init__(app)
        self.stalling = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC] HELPER: While stalling, sleeps 3600 seconds (so only cancellation ends it);
        # otherwise defers to the base transport.
        if self.stalling:
            await asyncio.sleep(3600)
        return await super().handle_async_request(request)


class _HealthFailsLaterTransport(_HookTransport):
    # [CLASS DOC] HELPER: Hook transport that behaves normally until a health status is set; after
    # that any request to the health endpoint is answered locally with that status.
    """Answers normally until `health_status` is set, after which the health endpoint answers with it."""

    def __init__(self, app: Any) -> None:
        # [FUNCTION DOC] HELPER: Runs the base constructor and starts with no forced health status.
        super().__init__(app)
        self.health_status: int | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC] HELPER: If a health status is set and the url contains the health path, counts
        # the request and returns a response with that status and body "gone" without recording it in
        # the seen list or reaching the server; otherwise defers to the base transport.
        if self.health_status is not None and HEALTH_PATH in str(request.url):
            self.requests += 1
            return httpx.Response(self.health_status, text="gone", request=request)
        return await super().handle_async_request(request)


class TestBootUuid:
    # [CLASS DOC] Tests boot uuid pinning on the handle: restart detection for stable handles, the
    # rules for submits and wait-ready, and when the pin is kept, dropped or restored by wait-ready
    # with the allow-server-uuid-change option.
    async def test_restart_detected_when_required_stable(self):
        # [FUNCTION DOC] After one successful call on a stable handle, the transport switches to a
        # second server; the next call raises a server-restarted error.
        """A server restart between calls raises ServerRestartedError."""
        async with _running_app(_Worker()) as first_app, _running_app(_Worker()) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

                transport.switch_to(second_app)
                with pytest.raises(ServerRestartedError):
                    await handle.demo_default_arg(a=1, b=2)

    async def test_restart_ignored_by_default(self):
        # [FUNCTION DOC] On a non-stable handle, switching servers between two calls is tolerated and
        # the second call returns 3.
        """Without the stable-boot-uuid requirement a restart is tolerated."""
        async with _running_app(_Worker()) as first_app, _running_app(_Worker()) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

                transport.switch_to(second_app)
                assert await handle.demo_default_arg(a=1, b=2) == 3

    async def test_stale_pin_refused_before_side_effects(self):
        # [FUNCTION DOC] Pins the first server via wait-ready, switches to a second server, and asserts
        # a call raises a server-restarted error and the second worker's call counter is still zero
        # (the server refused before running anything).
        """In stable mode a submit reaching a restarted server never runs there."""
        second_worker = _Worker()
        async with _running_app(_Worker()) as first_app, _running_app(second_worker) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)

                transport.switch_to(second_app)
                with pytest.raises(ServerRestartedError):
                    await handle.demo_default_arg(a=1, b=2)
                assert second_worker.calls == 0

    async def test_post_wire_loss_is_not_retried_on_restarted_server(self, fast_retries: None) -> None:
        # [FUNCTION DOC] The hook raises a read error on the first submit POST (the response is lost
        # after the first server accepted it) and the test then switches to a second server. Asserts
        # the call raises a worker unreachable error (not a retry) and the second worker never ran
        # a call.
        """A post-wire submit loss is not retried onto a restarted server."""
        dropped: list[bool] = []
        second_worker = _Worker()

        def drop_first_submit_response(request: httpx.Request) -> httpx.Response | None:
            # [FUNCTION DOC] HELPER: Raises a read error for the first POST only (tracked by a list
            # flag); everything else passes through.
            if request.method == "POST" and not dropped:
                dropped.append(True)
                raise httpx.ReadError("response lost after the server accepted it", request=request)
            return None

        async with _running_app(_Worker()) as first_app, _running_app(second_worker) as second_app:
            transport = _HookTransport(first_app, hook=drop_first_submit_response)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)

                transport.switch_to(second_app)
                with pytest.raises(WorkerUnreachableError):
                    await handle.demo_default_arg(a=1, b=2)
                assert second_worker.calls == 0

    async def test_missing_header_rejected_in_stable_mode(self):
        # [FUNCTION DOC] A hook answers every request with a 200 body lacking the boot uuid header; a
        # call on a stable handle raises a server-restarted error.
        """A response without the boot uuid header is refused in stable mode."""

        def strip_header(request: httpx.Request) -> httpx.Response:
            # [FUNCTION DOC] HELPER: Answers any request with 200 and status ok, no boot uuid header.
            return httpx.Response(200, json={"status": "ok"}, request=request)

        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=strip_header)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                with pytest.raises(ServerRestartedError):
                    await handle.demo_default_arg(a=1, b=2)

    async def test_server_ignoring_the_expectation_header_is_still_refused(self) -> None:
        # [FUNCTION DOC] After a normal wait-ready pin, a hook answers submit POSTs with 200 and a
        # different boot uuid header; the call raises a server-restarted error whose message mentions
        # that other boot uuid.
        """A rolled-back server that never checks the expectation header is caught by the client."""

        def answer_from_another_boot(request: httpx.Request) -> httpx.Response | None:
            # [FUNCTION DOC] HELPER: For POSTs, answers 200 with status submitted and boot uuid header
            # "some-other-boot"; other requests pass through.
            if request.method != "POST":
                return None
            return httpx.Response(
                200, json={"status": "submitted"}, headers={BOOT_UUID_HEADER: "some-other-boot"}, request=request
            )

        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=answer_from_another_boot)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)

                with pytest.raises(ServerRestartedError, match="some-other-boot"):
                    await handle.demo_default_arg(a=1, b=2)

    async def test_response_losing_the_header_after_pinning_is_refused(self) -> None:
        # [FUNCTION DOC] After wait-ready pins, a hook answers submit POSTs with 200 but no boot uuid
        # header; the call raises a server-restarted error mentioning the header name.
        """Once pinned, a successful response without the header is a restart, not a silent pass."""

        def strip_header(request: httpx.Request) -> httpx.Response | None:
            # [FUNCTION DOC] HELPER: For POSTs, answers 200 with status submitted and no boot uuid
            # header; other requests pass through.
            if request.method != "POST":
                return None
            return httpx.Response(200, json={"status": "submitted"}, request=request)

        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=strip_header)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)

                with pytest.raises(ServerRestartedError, match=BOOT_UUID_HEADER):
                    await handle.demo_default_arg(a=1, b=2)

    @pytest.mark.parametrize("status_code", [500, 502, 503])
    async def test_header_less_error_before_pinning_is_not_a_restart(
        self, fast_retries: None, status_code: int
    ) -> None:
        # [FUNCTION DOC] Parametrized over 500, 502 and 503: the first health GET is answered with that
        # status and no boot uuid header. Asserts wait-ready on a stable handle still succeeds after
        # retrying (more than one request was made) instead of raising a restart error.
        """A transient error response carrying no boot uuid is retried, not reported as a restart."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(
                app, hook=_status_hook(status_code=status_code, times=1, method="GET", path_fragment="/v1/health")
            )
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)
                assert transport.requests > 1

    async def test_wait_ready_keeps_original_pin_after_restart(self) -> None:
        # [FUNCTION DOC] After a call pins the first server and the transport switches to a second
        # server, a plain wait-ready raises a server-restarted error.
        """wait_ready refuses a new boot after the first pin."""
        async with _running_app(_Worker()) as first_app, _running_app(_Worker()) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

                transport.switch_to(second_app)
                with pytest.raises(ServerRestartedError):
                    await handle.wait_ready(timeout=5.0)

    async def test_wait_ready_adopts_a_replacement_when_the_caller_allows_it(self) -> None:
        # [FUNCTION DOC] After pinning the first server and switching to a second, wait-ready with the
        # allow-server-uuid-change option succeeds; the following call returns 3 and ran on the second
        # worker exactly once.
        """A caller that expects the server process to be replaced may drop the pin instead of failing on it."""
        second_worker = _Worker()
        async with _running_app(_Worker()) as first_app, _running_app(second_worker) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                assert await handle.demo_default_arg(a=1, b=2) == 3

                transport.switch_to(second_app)
                await handle.wait_ready(timeout=5.0, allow_server_uuid_change=True)

                assert await handle.demo_default_arg(a=1, b=2) == 3
                assert second_worker.calls == 1

    async def test_a_restart_after_an_allowed_change_is_still_refused(self) -> None:
        # [FUNCTION DOC] Adopts a second server with the allow option, then switches to a third server;
        # a call raises a server-restarted error (the pin was restored to strict mode).
        """Adopting one replacement must not leave the fence off for every restart that follows."""
        async with (
            _running_app(_Worker()) as first_app,
            _running_app(_Worker()) as second_app,
            _running_app(_Worker()) as third_app,
        ):
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)
                transport.switch_to(second_app)
                await handle.wait_ready(timeout=5.0, allow_server_uuid_change=True)

                transport.switch_to(third_app)
                with pytest.raises(ServerRestartedError):
                    await handle.demo_default_arg(a=1, b=2)

    async def test_a_wait_that_never_reached_a_server_leaves_the_pin_it_found(self) -> None:
        # [FUNCTION DOC] After pinning, the transport fails every request with a connect error. A
        # wait-ready with the allow option and a 0.05 second timeout raises a worker unreachable error,
        # and the handle's private pin still holds the previously pinned uuid.
        """Dropping the pin for a wait that failed would let the next ordinary call adopt any server silently."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)
                pinned = handle._boot_uuid_pin.expected

                transport.fail_with(httpx.ConnectError("no route"))
                with pytest.raises(WorkerUnreachableError):
                    await handle.wait_ready(timeout=0.05, allow_server_uuid_change=True)

                assert handle._boot_uuid_pin.expected == pinned

    async def test_a_wait_that_reached_a_replacement_keeps_the_new_pin(self) -> None:
        # [FUNCTION DOC] After adopting a second server with the allow option, the pinned uuid is
        # neither nothing nor the original one.
        """The restore only applies to a wait that failed, never to one that adopted a replacement."""
        async with _running_app(_Worker()) as first_app, _running_app(_Worker()) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)
                pinned = handle._boot_uuid_pin.expected

                transport.switch_to(second_app)
                await handle.wait_ready(timeout=5.0, allow_server_uuid_change=True)

                assert handle._boot_uuid_pin.expected not in (None, pinned)

    async def test_a_wait_that_adopted_a_replacement_is_pinned_again_afterwards(self) -> None:
        # [FUNCTION DOC] After adopting a second server with the allow option, asserts the pin no
        # longer needs a handshake (it is pinned again).
        """The adopt drops the pin before probing, so a probe that answered without repinning leaves it wide open."""
        async with _running_app(_Worker()) as first_app, _running_app(_Worker()) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)

                transport.switch_to(second_app)
                await handle.wait_ready(timeout=5.0, allow_server_uuid_change=True)

                assert handle._boot_uuid_pin.needs_handshake() is False

    async def test_a_wait_a_protocol_error_ended_leaves_the_pin_it_found(self) -> None:
        # [FUNCTION DOC] After pinning, the health endpoint starts answering 404. A wait-ready with
        # the allow option raises an rpc protocol error and the pin keeps its previous uuid.
        """Only a retryable exhaustion restored the pin, so a 4xx left the handle open to any process."""
        async with _running_app(_Worker()) as app:
            transport = _HealthFailsLaterTransport(app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)
                pinned = handle._boot_uuid_pin.expected

                transport.health_status = 404
                with pytest.raises(RpcProtocolError):
                    await handle.wait_ready(timeout=5.0, allow_server_uuid_change=True)

                assert handle._boot_uuid_pin.expected == pinned

    async def test_a_wait_that_was_cancelled_leaves_the_pin_it_found(self) -> None:
        # [FUNCTION DOC] After pinning, all requests stall. A wait-ready with the allow option is
        # cancelled by an outer 0.05 second timeout (a timeout error), and the pin keeps its previous
        # uuid.
        """A take-over cancels these waits, and an unpinned handle is what the take-over is guarding against."""
        async with _running_app(_Worker()) as app:
            transport = _StallingAfterwardsTransport(app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)
                pinned = handle._boot_uuid_pin.expected

                transport.stalling = True
                with pytest.raises((asyncio.TimeoutError, TimeoutError)):
                    await asyncio.wait_for(
                        handle.wait_ready(timeout=30.0, allow_server_uuid_change=True), timeout=0.05
                    )

                assert handle._boot_uuid_pin.expected == pinned

    async def test_a_call_after_a_cancelled_wait_still_refuses_a_replacement(self) -> None:
        # [FUNCTION DOC] After a cancelled allow-option wait, requests stop stalling and the transport
        # switches to a second server; the next call raises a server-restarted error, showing the pin
        # was restored.
        """Restoring the pin is only worth anything if the next call is still checked against it."""
        async with _running_app(_Worker()) as first_app, _running_app(_Worker()) as second_app:
            transport = _StallingAfterwardsTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)

                transport.stalling = True
                with pytest.raises((asyncio.TimeoutError, TimeoutError)):
                    await asyncio.wait_for(
                        handle.wait_ready(timeout=30.0, allow_server_uuid_change=True), timeout=0.05
                    )
                transport.stalling = False
                transport.switch_to(second_app)

                with pytest.raises(ServerRestartedError):
                    await handle.demo_default_arg(a=1, b=2)

    async def test_a_wait_that_succeeded_is_still_pinned_to_the_server_that_answered(self) -> None:
        # [FUNCTION DOC] After adopting a second server with the allow option, asserts the pinned uuid
        # differs from the original and the pin needs no further handshake.
        """The restore runs on every path that does not end pinned and ready, and this path does."""
        async with _running_app(_Worker()) as first_app, _running_app(_Worker()) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)
                pinned = handle._boot_uuid_pin.expected

                transport.switch_to(second_app)
                await handle.wait_ready(timeout=5.0, allow_server_uuid_change=True)

                assert handle._boot_uuid_pin.expected != pinned
                assert handle._boot_uuid_pin.needs_handshake() is False


class TestWaitReady:
    # [CLASS DOC] Tests the wait-ready method, which polls the health endpoint with retries until the
    # server answers or the given timeout expires.
    async def test_wait_ready_returns_when_healthy(self):
        # [FUNCTION DOC] Against a healthy server, wait-ready with a 5 second timeout returns.
        """wait_ready returns promptly against a healthy server."""
        async with _running_app(_Worker()) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            await handle.wait_ready(timeout=5.0)

    async def test_wait_ready_times_out_against_dead_server(self, fast_retries):
        # [FUNCTION DOC] With every request failing, wait-ready with a 0.2 second timeout raises a
        # worker unreachable error within 2 seconds after more than one request.
        """wait_ready keeps polling within its window and then raises WorkerUnreachableError."""
        transport = _HookTransport(None, hook=_fail_hook(-1))
        async with _handle_over(transport) as handle:
            started = time.monotonic()
            with pytest.raises(WorkerUnreachableError):
                await handle.wait_ready(timeout=0.2)
            assert time.monotonic() - started <= 2.0
            assert transport.requests > 1

    async def test_wait_ready_survives_initial_failures(self, fast_retries):
        # [FUNCTION DOC] The first two requests fail; wait-ready with a 5 second timeout succeeds
        # after at least three requests.
        """wait_ready keeps retrying through initial connection failures."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(2))
            async with _handle_over(transport) as handle:
                await handle.wait_ready(timeout=5.0)
                assert transport.requests >= 3

    async def test_wait_ready_probe_timeout_is_clamped_to_remaining_budget(self, fast_retries) -> None:
        # [FUNCTION DOC] A transport records the monotonic time and read timeout of each probe and
        # always refuses the connection. After wait-ready with a 0.4 second timeout raises a worker
        # unreachable error, asserts at least two probes happened, each later probe's read timeout is
        # no larger than the first timeout minus the elapsed time (plus 0.01 tolerance), and the last
        # probe's timeout is more than 0.1 seconds below the first.
        """A health probe never gets more time than the caller's remaining readiness budget."""

        class _HealthTimeoutRecordingTransport(httpx.AsyncBaseTransport):
            # [CLASS DOC] HELPER: Transport recording (time, read timeout) pairs and refusing connection.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Starts with an empty observation list.
                self.observations: list[tuple[float, float]] = []

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                # [FUNCTION DOC] HELPER: Appends the current monotonic time and the request's read
                # timeout, then raises a connect error "refused".
                self.observations.append((time.monotonic(), request.extensions["timeout"]["read"]))
                raise httpx.ConnectError("refused", request=request)

        transport = _HealthTimeoutRecordingTransport()
        async with _handle_over(transport) as handle:
            with pytest.raises(WorkerUnreachableError):
                await handle.wait_ready(timeout=0.4)

        assert len(transport.observations) >= 2
        first_request_at, first_timeout = transport.observations[0]
        for request_at, read_timeout in transport.observations[1:]:
            elapsed = request_at - first_request_at
            assert read_timeout <= first_timeout - elapsed + 0.01
        assert transport.observations[-1][1] < first_timeout - 0.1

    async def test_wait_ready_protocol_error_is_not_retried(self) -> None:
        # [FUNCTION DOC] Every health GET is answered with 404; wait-ready raises an rpc protocol
        # error with status 404 after exactly one request.
        """A 4xx health response is a protocol error, so wait_ready gives up after one request."""
        transport = _HookTransport(
            None,
            hook=_status_hook(status_code=404, times=-1, method="GET", path_fragment=HEALTH_PATH),
        )
        async with _handle_over(transport) as handle:
            with pytest.raises(RpcProtocolError) as exc_info:
                await handle.wait_ready(timeout=5.0)

        assert exc_info.value.status_code == 404
        assert transport.requests == 1

    async def test_wait_ready_persistent_server_error_exhausts_as_unreachable(self, fast_retries) -> None:
        # [FUNCTION DOC] Every health GET is answered with 503 and the timeout is 0.5 seconds;
        # wait-ready raises a worker unreachable error, after at least three requests, and the last
        # request happened no earlier than the timeout minus 0.1 seconds after the start.
        """A health endpoint stuck on 5xx is retried until the deadline and then reported unreachable."""
        timeout = 0.5
        transport = _HookTransport(
            None,
            hook=_status_hook(status_code=503, times=-1, method="GET", path_fragment=HEALTH_PATH),
        )
        async with _handle_over(transport) as handle:
            started = time.monotonic()
            with pytest.raises(WorkerUnreachableError):
                await handle.wait_ready(timeout=timeout)

        assert transport.requests >= 3
        assert transport.request_times[-1] - started >= timeout - 0.1


class TestPositionalCalls:
    # [CLASS DOC] Tests positional argument binding end to end using the positional worker.
    async def test_positional_arguments_reach_the_worker_in_declaration_order(self):
        # [FUNCTION DOC] A join with two positionals returns the two strings joined by the default
        # separator in order.
        """Positionals bind to parameters left to right, so an order-sensitive method sees the caller's order."""
        async with (
            _running_app(_PositionalWorker()) as app,
            _handle_over(httpx.ASGITransport(app=app), worker_cls=_PositionalWorker) as handle,
        ):
            assert await handle.demo_join("left", "right") == "left-right"

    async def test_positional_arguments_travel_as_named_query_fields(self):
        # [FUNCTION DOC] Inspects the submit request bodies: the json query field of the single submit
        # is a mapping with first, second and the default separator, i.e. positionals were converted to
        # names client-side.
        """The wire stays keyword-shaped, so the server never has to know the caller used positionals."""
        async with _running_app(_PositionalWorker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport, worker_cls=_PositionalWorker) as handle:
                await handle.demo_join("left", "right")

            submits = [r for r in transport.seen if r.method == "POST"]
            assert [json.loads(r.content)["query"] for r in submits] == [
                {"first": "left", "second": "right", "separator": "-"}
            ]

    async def test_a_positional_and_keyword_mix_leaves_the_remaining_default_intact(self):
        # [FUNCTION DOC] A join with first positional and second by keyword returns the default
        # separator result.
        """A short positional call omits the parameters it does not fill, so their declared defaults still apply."""
        async with (
            _running_app(_PositionalWorker()) as app,
            _handle_over(httpx.ASGITransport(app=app), worker_cls=_PositionalWorker) as handle,
        ):
            assert await handle.demo_join("left", second="right") == "left-right"

    async def test_a_keyword_only_parameter_is_reachable_alongside_positionals(self):
        # [FUNCTION DOC] A join with two positionals and a keyword separator of plus returns the two
        # strings joined by plus.
        """Positionals fill the ordinary parameters while a keyword-only parameter still arrives by name."""
        async with (
            _running_app(_PositionalWorker()) as app,
            _handle_over(httpx.ASGITransport(app=app), worker_cls=_PositionalWorker) as handle,
        ):
            assert await handle.demo_join("left", "right", separator="+") == "left+right"

    async def test_a_positional_for_a_keyword_only_parameter_fails_before_any_request(self):
        # [FUNCTION DOC] A third positional (meant for the keyword-only parameter) raises a type error
        # saying at most 2 positional arguments, and no request is made.
        """A keyword-only parameter cannot be filled positionally, and the bad call never reaches the network."""
        async with _running_app(_PositionalWorker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport, worker_cls=_PositionalWorker) as handle:
                with pytest.raises(TypeError, match="at most 2 positional arguments"):
                    await handle.demo_join("left", "right", "+")
                assert transport.requests == 0

    async def test_a_duplicated_parameter_fails_before_any_request(self):
        # [FUNCTION DOC] Giving first both positionally and by keyword raises a type error with
        # multiple values for first, and no request is made.
        """The same parameter given positionally and by keyword raises locally instead of one value winning."""
        async with _running_app(_PositionalWorker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport, worker_cls=_PositionalWorker) as handle:
                with pytest.raises(TypeError, match=r"multiple values for \['first'\]"):
                    await handle.demo_join("left", first="other")
                assert transport.requests == 0

    async def test_a_parameterless_method_rejects_a_positional_argument(self):
        # [FUNCTION DOC] One positional for the parameterless method raises a type error saying at
        # most 0 positional arguments, and no request is made.
        """A method taking only the receiver has no parameter name to bind a positional to."""
        async with _running_app(_PositionalWorker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport, worker_cls=_PositionalWorker) as handle:
                with pytest.raises(TypeError, match="at most 0 positional arguments"):
                    await handle.demo_nothing(1)
                assert transport.requests == 0


class TestTraverseErrorChain:
    # [CLASS DOC] Tests the generator that walks an exception's cause/context chain.
    def test_single_error_yields_itself(self) -> None:
        # [FUNCTION DOC] An exception with no cause or context yields a chain of only itself.
        """Include the original exception even without a cause or context."""
        error = RuntimeError("root")

        assert list(rpc_handle_module._traverse_error_chain(error)) == [error]

    def test_cause_takes_precedence_and_context_is_followed(self) -> None:
        # [FUNCTION DOC] Builds root with an explicit cause and an ignored context, where the cause
        # has its own implicit context. The chain is root, cause, cause's context; root's own context
        # is skipped because an explicit cause wins.
        """Follow explicit causes first and fall back to implicit context."""
        error = RuntimeError("root")
        cause = OSError("cause")
        context = ConnectionRefusedError("context")
        error.__cause__ = cause
        error.__context__ = ValueError("ignored context")
        cause.__context__ = context

        assert list(rpc_handle_module._traverse_error_chain(error)) == [error, cause, context]

    def test_cycle_stops_before_repeating_an_exception(self) -> None:
        # [FUNCTION DOC] Builds a cycle (root caused by cause, whose context is root). The generator
        # yields root, then cause, then stops without yielding root a second time.
        """Terminate a cyclic chain without yielding an exception twice."""
        error = RuntimeError("root")
        cause = OSError("cause")
        error.__cause__ = cause
        cause.__context__ = error

        chain = rpc_handle_module._traverse_error_chain(error)
        assert next(chain) is error
        assert next(chain) is cause
        with pytest.raises(StopIteration):
            next(chain)


class TestWaitDead:
    # [CLASS DOC] Tests the dead-probe and wait-dead methods, which decide that a worker's server is
    # gone only on a refused connection or on a replaced boot uuid.
    @pytest.mark.parametrize(
        ("cause", "expected_dead"),
        [
            (None, False),
            (socket.gaierror(socket.EAI_NONAME, "name lookup failed"), False),
            (OSError(errno.ENETUNREACH, "network unreachable"), False),
            (ConnectionRefusedError(errno.ECONNREFUSED, "connection refused"), True),
        ],
    )
    async def test_connection_errors_require_underlying_refusal_before_reporting_death(
        self, cause: Exception | None, expected_dead: bool
    ) -> None:
        # [FUNCTION DOC] Parametrized over the root cause of a connect error: nothing, a name lookup
        # failure and an unreachable network (all expected not dead) and a refused connection
        # (expected dead). The transport raises a connect error whose cause chain goes through a
        # generic OS error to the given cause; asserts the handle's probe-is-dead result equals the
        # expectation.
        """DNS and network failures do not prove death, unlike an explicit refused connection."""
        error = httpx.ConnectError("connection refused")
        intermediate = OSError("all connection attempts failed")
        intermediate.__cause__ = cause
        error.__cause__ = intermediate
        transport = _HookTransport(None)
        transport.fail_with(error)

        async with _handle_over(transport) as handle:
            assert await handle.probe_is_dead() is expected_dead

    async def test_wait_dead_returns_once_the_server_stops_answering(self):
        # [FUNCTION DOC] With a connect error caused by a refused connection, wait-dead with a 5 second
        # timeout returns after exactly one request.
        """A cell is healed only after its ranks are gone, and a refused connection is that proof."""
        error = httpx.ConnectError("connection refused")
        error.__cause__ = ConnectionRefusedError(errno.ECONNREFUSED, "connection refused")
        transport = _HookTransport(None)
        transport.fail_with(error)
        async with _handle_over(transport) as handle:
            await handle.wait_dead(timeout=5.0)

            assert transport.requests == 1

    async def test_wait_dead_gives_up_on_a_worker_that_stays_healthy(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        # [FUNCTION DOC] Shortens the probe interval to 0.01 seconds. Against a healthy server,
        # wait-dead with a 0.05 second timeout returns without raising; asserts at least two probes
        # were made and an error level log containing the text "waiting for" was emitted by the worker
        # handle logger.
        """A worker that refuses to die must not block healing forever, so the wait is bounded and says so."""
        monkeypatch.setattr(worker_handle_module, "_WAIT_DEAD_PROBE_INTERVAL_SECONDS", 0.01)
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport) as handle:
                with caplog.at_level(logging.ERROR, logger=worker_handle_module.__name__):
                    await handle.wait_dead(timeout=0.05)

                assert transport.requests >= 2
                assert "waiting for" in caplog.text

    async def test_wait_dead_treats_a_restarted_server_as_dead(self):
        # [FUNCTION DOC] Pins the first server, swaps in a second, and asserts wait-dead returns after
        # exactly one extra request.
        """The pod that was asked to die is gone once a fresh boot uuid answers in its place."""
        async with _running_app(_Worker()) as first:
            transport = _HookTransport(first)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)
                async with _running_app(_Worker()) as second:
                    transport.switch_to(second)
                    requests_before = transport.requests

                    await handle.wait_dead(timeout=5.0)

                    assert transport.requests == requests_before + 1

    async def test_wait_dead_keeps_waiting_while_the_worker_is_merely_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] Shortens the probe interval to 0.01 seconds and fails every request with a
        # connect timeout (no refusal). Wait-dead with a 0.05 second timeout returns after its bounded
        # wait, having probed at least twice (a timeout does not prove death).
        """A connect timeout says nothing about the worker, so healing must not start on one."""
        monkeypatch.setattr(worker_handle_module, "_WAIT_DEAD_PROBE_INTERVAL_SECONDS", 0.01)
        transport = _HookTransport(None, hook=_fail_hook(-1, error_type=httpx.ConnectTimeout))
        async with _handle_over(transport) as handle:
            await handle.wait_dead(timeout=0.05)

            assert transport.requests >= 2


class _SlowWorker:
    # [CLASS DOC] HELPER: Worker whose async and sync methods each signal start, take 0.2 seconds and
    # signal finish, to model a call that outlives its poller. Uses two thread events.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the started and finished thread events (unset).
        self.started = threading.Event()
        self.finished = threading.Event()

    async def demo_slow(self) -> str:
        # [FUNCTION DOC] HELPER: Sets started, awaits a 0.2 second async sleep, sets finished and
        # returns "done".
        self.started.set()
        await asyncio.sleep(0.2)
        self.finished.set()
        return "done"

    def demo_slow_sync(self) -> str:
        # [FUNCTION DOC] HELPER: Same as the async variant but blocks the executing thread with a
        # 0.2 second sleep.
        self.started.set()
        time.sleep(0.2)
        self.finished.set()
        return "done"


async def _abandon_polled_call(worker: _SlowWorker, *, transport: _HookTransport, method: str) -> None:
    # [FUNCTION DOC] HELPER: Starts the named method on a handle for the slow worker as a task, waits
    # (in a thread, up to 5 seconds) until the worker reports it started, then waits up to 5 seconds
    # (polling every 0.01 seconds) until the transport has seen at least one poll, failing with an
    # assertion message otherwise. Finally cancels the client task and expects a cancellation error,
    # so the call is abandoned mid-flight while the server keeps running it.
    async with _handle_over(transport, worker_cls=_SlowWorker) as handle:
        call = asyncio.create_task(getattr(handle, method)())
        assert await asyncio.to_thread(worker.started.wait, 5.0)

        deadline = time.monotonic() + 5.0
        while not transport.polls():
            assert time.monotonic() < deadline, "the client never polled the call it submitted"
            await asyncio.sleep(0.01)

        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call


class TestTheCallStoreOwnsACallRatherThanItsClient:
    # [CLASS DOC] Tests that the server side call store keeps running and retaining a call even when
    # the polling client goes away. Parametrized over an async and a sync slow method.
    @pytest.mark.parametrize("method", ["demo_slow", "demo_slow_sync"])
    async def test_a_submitted_call_keeps_running_after_its_poller_walks_away(self, method: str) -> None:
        # [FUNCTION DOC] Abandons the polled call, then asserts the worker still reports finished
        # within 5 seconds.
        """A client may walk away mid-call, and the worker has to finish what it accepted regardless."""
        worker = _SlowWorker()
        async with _running_app(worker) as app:
            await _abandon_polled_call(worker, transport=_HookTransport(app), method=method)

            assert await asyncio.to_thread(worker.finished.wait, 5.0)

    @pytest.mark.parametrize("method", ["demo_slow", "demo_slow_sync"])
    async def test_the_store_still_holds_the_outcome_for_whoever_polls_next(self, method: str) -> None:
        # [FUNCTION DOC] Abandons the polled call and waits for the worker to finish. Reads the call id
        # from the recorded submit request body, then polls the status path directly with a fresh
        # client (timeout 1 second); asserts a 200 response whose status field is "success".
        """The next script polls the worker for the call its predecessor submitted, so the outcome must be retained."""
        worker = _SlowWorker()
        async with _running_app(worker) as app:
            transport = _HookTransport(app)
            await _abandon_polled_call(worker, transport=transport, method=method)
            assert await asyncio.to_thread(worker.finished.wait, 5.0)

            submitted = next(request for request in transport.seen if request.method == "POST")
            call_id = json.loads(submitted.content)["call_id"]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as http_client:
                response = await http_client.get(CALL_STATUS_PATH.format(call_id=call_id), params={"timeout": 1.0})

            assert response.status_code == 200
            assert response.json()["status"] == "success"


class TestWaitIdle:
    # [CLASS DOC] Tests the wait-idle method, which polls the server's in-flight endpoint until it
    # reports no running or queued calls, or fails with a still-busy error at the timeout.
    async def test_an_idle_worker_is_reported_idle_at_once(self) -> None:
        # [FUNCTION DOC] Against an idle worker, wait-idle returns after exactly one request.
        """A worker that is running nothing must not cost its caller a poll interval."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app)
            async with _handle_over(transport) as handle:
                await handle.wait_idle(timeout=5.0)

                assert transport.requests == 1

    async def test_a_running_call_is_waited_out(self, fast_retries: None) -> None:
        # [FUNCTION DOC] Starts a blocking call, waits until it started, and releases it after 0.2
        # seconds from a background task. Wait-idle with a 10 second timeout returns only after the
        # release (asserted by the block event being set) and the call result is "done".
        """The wait ends only once the worker has finished the call it was already running."""
        worker = _Worker()

        async def release_soon() -> None:
            # [FUNCTION DOC] HELPER: Sleeps 0.2 seconds and sets the block-forever event.
            await asyncio.sleep(0.2)
            worker.block_forever.set()

        async with _running_app(worker) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            call = asyncio.create_task(handle.demo_hang())
            assert await asyncio.to_thread(worker.hang_started.wait, 5.0)
            releaser = asyncio.create_task(release_soon())

            await handle.wait_idle(timeout=10.0)

            assert worker.block_forever.is_set()
            assert await asyncio.wait_for(call, timeout=5.0) == "done"
            await releaser

    async def test_a_call_that_never_finishes_ends_the_wait_loudly(self, fast_retries: None) -> None:
        # [FUNCTION DOC] With a call blocked, wait-idle with a 0.3 second timeout raises a worker
        # still-busy error whose message says the worker is still running the calls. Afterwards the
        # call is released and finishes with "done".
        """A wait that hung forever would leave its caller wedged with no explanation, so it has to fail."""
        worker = _Worker()
        async with _running_app(worker) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            call = asyncio.create_task(handle.demo_hang())
            assert await asyncio.to_thread(worker.hang_started.wait, 5.0)

            with pytest.raises(WorkerStillBusyError, match="is still running the calls"):
                await handle.wait_idle(timeout=0.3)

            worker.block_forever.set()
            assert await asyncio.wait_for(call, timeout=5.0) == "done"

    async def test_a_call_that_is_queued_but_not_started_still_counts_as_in_flight(self, fast_retries: None) -> None:
        # [FUNCTION DOC] One blocking call occupies the worker and a second call is queued behind it.
        # Wait-idle with a 0.3 second timeout raises a still-busy error. After releasing, the first
        # call returns "done" and the queued one returns 3.
        """A take-over that adopted a worker with work still queued would drive it while that work runs."""
        worker = _Worker()
        async with _running_app(worker) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            first = asyncio.create_task(handle.demo_hang())
            assert await asyncio.to_thread(worker.hang_started.wait, 5.0)
            queued = asyncio.create_task(handle.demo_default_arg(a=1, b=2))
            await asyncio.sleep(0.05)

            with pytest.raises(WorkerStillBusyError, match="is still running the calls"):
                await handle.wait_idle(timeout=0.3)

            worker.block_forever.set()
            assert await asyncio.wait_for(first, timeout=5.0) == "done"
            assert await asyncio.wait_for(queued, timeout=5.0) == 3

    async def test_a_call_that_failed_leaves_the_worker_idle(self, fast_retries: None) -> None:
        # [FUNCTION DOC] After a call that raises a worker call error, wait-idle with a 5 second
        # timeout returns (a failed call is no longer in flight).
        """A call the worker refused is finished, and a take-over that kept waiting for it would never proceed."""
        worker = _Worker()
        async with _running_app(worker) as app, _handle_over(httpx.ASGITransport(app=app)) as handle:
            with pytest.raises(RpcWorkerCallError):
                await handle.demo_raises()

            await handle.wait_idle(timeout=5.0)

    async def test_a_wait_against_a_replaced_server_process_is_refused(self, fast_retries: None) -> None:
        # [FUNCTION DOC] After pinning the first server and switching to a second, wait-idle on a
        # stable handle raises a server-restarted error.
        """A pinned handle waiting out a worker that restarted under it is waiting out the wrong process."""
        async with _running_app(_Worker()) as first_app, _running_app(_Worker()) as second_app:
            transport = _HookTransport(first_app)
            async with _handle_over(transport, require_stable_boot_uuid=True) as handle:
                await handle.wait_ready(timeout=5.0)

                transport.switch_to(second_app)
                with pytest.raises(ServerRestartedError):
                    await handle.wait_idle(timeout=5.0)

    async def test_a_transport_hiccup_does_not_end_the_wait(self, fast_retries: None) -> None:
        # [FUNCTION DOC] The first two in-flight probes fail with connect errors; wait-idle with a 5
        # second timeout still returns after at least three requests.
        """The wait spans a pod that may be restarting under it, so a lost probe is retried rather than fatal."""
        async with _running_app(_Worker()) as app:
            transport = _HookTransport(app, hook=_fail_hook(2, "GET", IN_FLIGHT_PATH))
            async with _handle_over(transport) as handle:
                await handle.wait_idle(timeout=5.0)

                assert transport.requests >= 3
