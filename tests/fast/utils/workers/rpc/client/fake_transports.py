# [MODULE DOC] HELPER: Fake http transports for rpc client tests. Both wrap an in-process ASGI
# transport around a server application and special-case only the long-poll status requests (GET
# requests whose url contains the call status path fragment): one stalls the first N polls, the
# other records the timeout window each poll was sent with. All other requests pass straight through.
from __future__ import annotations

import asyncio
from typing import Any, NamedTuple

import httpx

_CALL_STATUS_FRAGMENT = "/v1/calls/"


def _is_poll(request: httpx.Request) -> bool:
    # [FUNCTION DOC] HELPER: Returns true when the request is a GET whose full url contains the call
    # status path fragment, which identifies a poll for a call result.
    return request.method == "GET" and _CALL_STATUS_FRAGMENT in str(request.url)


class StallingPollTransport(httpx.AsyncBaseTransport):
    # [CLASS DOC] HELPER: Transport that makes the first N poll requests hang for a long time (so the
    # client's own deadline must abandon them) and serves every other request, including later polls,
    # through the wrapped ASGI app. Exposes a counter of polls seen so far.
    def __init__(self, app: Any, *, stalled_polls: int, stall_seconds: float = 60.0) -> None:
        # [FUNCTION DOC] HELPER: Wraps the given ASGI app in an ASGI transport, sets the poll counter to
        # zero, and stores how many polls to stall and for how many seconds (default 60).
        self.polls = 0
        self._inner = httpx.ASGITransport(app=app)
        self._stalled_polls = stalled_polls
        self._stall_seconds = stall_seconds

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC] HELPER: For a poll request, increments the counter; while the counter is within
        # the stalled count it sleeps for the stall duration and then raises an assertion error (reached
        # only if the client failed to abandon the request). Every non-stalled request is forwarded to
        # the inner ASGI transport and its response returned.
        if _is_poll(request):
            self.polls += 1
            if self.polls <= self._stalled_polls:
                await asyncio.sleep(self._stall_seconds)
                raise AssertionError("the stalled poll should have been abandoned")
        return await self._inner.handle_async_request(request)


class PollWindow(NamedTuple):
    # [CLASS DOC] HELPER: Immutable record of one poll's timing: the http read timeout the client set
    # locally and the wait the client asked the server to hang for, both in seconds.
    client_seconds: float
    server_seconds: float


class PollWindowRecordingTransport(httpx.AsyncBaseTransport):
    # [CLASS DOC] HELPER: Transport that records, for every poll request, the client side read timeout
    # and the server side timeout query parameter, then forwards all requests to the wrapped ASGI app
    # unchanged. Exposes the recorded windows in request order.
    def __init__(self, app: Any) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty list of poll windows and wraps the ASGI app in an
        # ASGI transport.
        self.poll_windows: list[PollWindow] = []
        self._inner = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC] HELPER: For a poll request, appends a window made of the request's read
        # timeout extension and its timeout query parameter, both converted to floats (a missing entry
        # would raise a key error). Then forwards the request to the inner transport and returns its
        # response.
        if _is_poll(request):
            self.poll_windows.append(
                PollWindow(
                    client_seconds=float(request.extensions["timeout"]["read"]),
                    server_seconds=float(request.url.params["timeout"]),
                )
            )
        return await self._inner.handle_async_request(request)
