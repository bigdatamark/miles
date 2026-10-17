# [MODULE DOC]
# Shared pieces of the rpc client: error types, retry timing constants and sets of error types
# considered retryable, the http transport that adds boot-uuid headers and maps response
# statuses to errors, and the boot uuid pin that detects a restarted server process.
from __future__ import annotations

import asyncio
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.workers.rpc.common.protocol import (
    BOOT_UUID_HEADER,
    BOOT_UUID_MISMATCH_STATUS,
    EXPECTED_BOOT_UUID_HEADER,
)

_ResponseT = TypeVar("_ResponseT", bound=BaseModel)

RETRY_INITIAL_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 10.0

_LOWEST_SERVER_ERROR_STATUS = 500

_ABORT_SLACK_SECONDS = 1.0


class ServerRestartedError(Exception):
    # [CLASS DOC] Raised when the server answering is not the process previously contacted: boot uuid
    # mismatch, or a missing boot uuid header when one is required. Not retryable.
    pass


class RpcProtocolError(Exception):
    # [CLASS DOC] Raised when the server rejected a request with a client-error status (anything
    # other than 200 below 500 and not the boot uuid mismatch status); carries the status code.
    # Not retryable.
    def __init__(self, message: str, *, status_code: int) -> None:
        # [FUNCTION DOC] Store the message in the base exception and keep the http status code.
        super().__init__(message)
        self.status_code = status_code


class RpcWorkerCallError(Exception):
    # [CLASS DOC] Raised to the caller when the remote method itself failed; the message carries the
    # remote error text.
    pass


class RetryableResponseError(Exception):
    # [CLASS DOC] Raised when the server answered with a server-error status (500 or above); treated
    # as transient and retryable.
    pass


RETRYABLE_ERRORS = (httpx.TransportError, TimeoutError, asyncio.TimeoutError, RetryableResponseError)

NEVER_REACHED_SERVER_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class RpcTransport:
    # [CLASS DOC]
    # Sends requests to one rpc server and validates the answers. Uses either the injected http
    # client or, when none was given, the process-wide general client looked up on every request.
    def __init__(self, *, server_url: str, http_client: httpx.AsyncClient | None, boot_uuid_pin: BootUuidPin) -> None:
        # [FUNCTION DOC] Store the server url without trailing slashes, the optional http client
        # override and the boot uuid pin.
        self._server_url = server_url.rstrip("/")
        self._http_client_override = http_client
        self._boot_uuid_pin = boot_uuid_pin

    async def request(
        self, method: str, path: str, *, seconds: float, response_model: type[_ResponseT], **kwargs: Any
    ) -> _ResponseT:
        # [FUNCTION DOC]
        # Send one http request to the server url plus path, with the given method name, and return
        # the response body validated as the given response model. The seconds value is the http
        # timeout, and an outer hard limit of that plus one second is enforced as a backstop (a
        # timeout error). Extra keyword arguments (json body, query params, headers) are passed to
        # the client; redirects are not followed. If the pin holds an expected boot uuid, it is
        # sent in the expected-boot-uuid header. Status handling: the boot uuid mismatch status raises
        # the server-restarted error; 500 and above raises the retryable response error; any other
        # non-200 status raises the protocol error with the body text; on 200 the boot uuid pin
        # verifies the response headers (may raise server-restarted) before the body is parsed.
        headers = dict(kwargs.pop("headers", {}))
        if self._boot_uuid_pin.expected is not None:
            headers[EXPECTED_BOOT_UUID_HEADER] = self._boot_uuid_pin.expected
        response = await asyncio.wait_for(
            self._client.request(
                method,
                f"{self._server_url}{path}",
                timeout=seconds,
                headers=headers,
                follow_redirects=False,
                **kwargs,
            ),
            timeout=seconds + _ABORT_SLACK_SECONDS,
        )

        if response.status_code == BOOT_UUID_MISMATCH_STATUS:
            raise ServerRestartedError(f"rpc server restarted: {response.text}")
        if response.status_code >= _LOWEST_SERVER_ERROR_STATUS:
            raise RetryableResponseError(f"{method} {path} returned {response.status_code}")
        if response.status_code != 200:
            raise RpcProtocolError(
                f"{method} {path} rejected ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )

        self._boot_uuid_pin.verify(response)
        return response_model.model_validate(response.json())

    @property
    def _client(self) -> httpx.AsyncClient:
        # [FUNCTION DOC] HELPER: Property: the injected http client if any, else the shared general
        # http client.
        if self._http_client_override is not None:
            return self._http_client_override
        return GeneralHttpClientProvider.client()


class BootUuidPin:
    # [CLASS DOC]
    # Remembers the boot uuid (an identifier the server generates once per process start and puts
    # in a response header) of the server the client first talked to, to detect restarts. When not
    # required it is inert: nothing is sent or checked. When required, the first verified response
    # pins its uuid; later responses must carry the same one.
    def __init__(self, *, required: bool, worker_cls_name: str) -> None:
        # [FUNCTION DOC] Store whether pinning is required and the worker class name (for error
        # messages); starts unpinned.
        self._required = required
        self._worker_cls_name = worker_cls_name
        self._value: str | None = None

    @property
    def expected(self) -> str | None:
        # [FUNCTION DOC] Property: the pinned uuid to send to the server, or none if pinning is not
        # required or nothing is pinned yet.
        return self._value if self._required else None

    def needs_handshake(self) -> bool:
        # [FUNCTION DOC] True iff pinning is required and no uuid has been pinned yet.
        return self._required and self._value is None

    def unpin(self) -> str | None:
        # [FUNCTION DOC] Clear the pin and return the previously pinned uuid (or none).
        previous = self._value
        self._value = None
        return previous

    def repin(self, value: str | None) -> None:
        # [FUNCTION DOC] Set the pinned uuid to the given value (restoring an earlier pin, or
        # none).
        self._value = value

    def verify(self, response: httpx.Response) -> None:
        # [FUNCTION DOC] Check a successful response's boot uuid header. Does nothing when pinning
        # is not required. Otherwise raises the server-restarted error if the header is missing or
        # differs from the pinned uuid; if nothing is pinned yet, pins the header's value.
        if not self._required:
            return

        boot_uuid = response.headers.get(BOOT_UUID_HEADER)
        if boot_uuid is None:
            raise ServerRestartedError(f"{self._worker_cls_name} response is missing the {BOOT_UUID_HEADER} header")
        if self._value is None:
            self._value = boot_uuid
        elif boot_uuid != self._value:
            raise ServerRestartedError(
                f"{self._worker_cls_name} response came from boot uuid {boot_uuid}, expected {self._value}"
            )
