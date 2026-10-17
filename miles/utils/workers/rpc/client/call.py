# [MODULE DOC]
# One remote method invocation as a two-phase protocol. Phase one (submit) sends the encoded
# request under a client-generated call id, retrying only errors proving the server was never
# reached, within a fixed window. Phase two (poll) repeatedly long-polls the call status until the
# call finishes or the call timeout elapses; transport failures while polling are retried. The
# result of a failed remote call is raised as a call error with the remote error text.
from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from typing import Any

import httpx

from miles.utils.retry_utils import retry_until_deadline
from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.client.misc import (
    NEVER_REACHED_SERVER_ERRORS,
    RETRY_INITIAL_DELAY_SECONDS,
    RETRY_MAX_DELAY_SECONDS,
    RETRYABLE_ERRORS,
    RetryableResponseError,
    RpcTransport,
    RpcWorkerCallError,
)
from miles.utils.workers.rpc.common.metadata import RpcMethodSpec
from miles.utils.workers.rpc.common.protocol import (
    CALL_STATUS_PATH,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    SUBMIT_PATH,
    CallStatusResponse,
    SubmitRequest,
    SubmitResponse,
)
from miles.utils.workers.worker_handle import WorkerUnreachableError

logger = logging.getLogger(__name__)

SUBMIT_RETRY_WINDOW_SECONDS = 60.0
SUBMIT_ATTEMPT_TIMEOUT_SECONDS = 10.0
POLL_SLACK_SECONDS = 5.0


class _CallStillPendingError(Exception):
    # [CLASS DOC] HELPER: Internal retry signal meaning the poll found no final outcome yet (still
    # pending, long poll timed out, or a transient poll failure), so polling should continue.
    pass


class RpcCall:
    # [CLASS DOC]
    # A single remote call, identified by a fresh random call id generated at construction. The
    # id makes the submit idempotent on the server side, so a retried submit cannot start the call
    # twice. Use run to do everything, or submit alone for fire-and-forget (the result is then
    # never fetched). Not reusable: create a new object per invocation.
    def __init__(
        self,
        *,
        spec: RpcMethodSpec,
        kwargs: dict[str, Any],
        worker_cls_name: str,
        transport: RpcTransport,
        call_timeout_seconds: float,
    ) -> None:
        # [FUNCTION DOC]
        # Capture the method spec, the canonical keyword arguments, the worker class name (for
        # logs and errors), the transport and the overall call timeout in seconds. Asserts the
        # timeout is a positive finite number. Encodes the arguments into the request query using
        # the method's serializer at construction (so encoding errors surface here), and
        # generates the call id as a random hexadecimal string.
        assert (
            math.isfinite(call_timeout_seconds) and call_timeout_seconds > 0
        ), f"call_timeout_seconds must be a positive finite number, got {call_timeout_seconds!r}"
        self._spec = spec
        self._kwargs = kwargs
        self._query = spec.serializer.encode_query(kwargs)
        self._worker_cls_name = worker_cls_name
        self._transport = transport
        self._call_timeout_seconds = call_timeout_seconds
        self._call_id = uuid.uuid4().hex

    async def run(self) -> Any:
        # [FUNCTION DOC]
        # Execute the call end to end: log the start (debug level, with the sorted argument
        # names only), submit, poll until done, log the end with the elapsed time. If the final
        # status is failed, raise a worker call error containing the method label and the remote
        # error text. Otherwise decode the returned result with the method's serializer and return
        # it. Other errors: worker-unreachable if submit could not reach the server, a timeout
        # error if still pending at the call timeout.
        log_structured(logger.debug, op="call", phase="start", **self._log_fields, args=sorted(self._kwargs))

        started_at = time.monotonic()
        await self.submit()
        outcome = await self._poll_until_done()
        elapsed = time.monotonic() - started_at

        ok = outcome.status != "failed"
        log_structured(logger.debug, op="call", phase="end", ok=ok, **self._log_fields, elapsed_s=round(elapsed, 3))
        if not ok:
            raise RpcWorkerCallError(f"{self._method_label} failed remotely:\n{outcome.error}")
        return self._spec.serializer.decode_result(outcome.result)

    async def submit(self) -> None:
        # [FUNCTION DOC]
        # Send the submit request (call id and encoded query) to the per-method submit path. Retries
        # only connect errors, connect timeouts and pool timeouts (errors that prove the server never
        # got the request) with backoff from 1 to 10 seconds, for at most 60 seconds in total. If
        # retries are exhausted on a retryable transport error, logs a warning and raises the
        # worker-unreachable error. Errors outside that set (server errors, protocol rejections,
        # server restart) propagate as raised by the transport.
        request = SubmitRequest(call_id=self._call_id, query=self._query)

        try:
            await retry_until_deadline(
                lambda remaining: self._submit_attempt(request=request, remaining=remaining),
                total_seconds=SUBMIT_RETRY_WINDOW_SECONDS,
                retry_on=NEVER_REACHED_SERVER_ERRORS,
                initial_delay=RETRY_INITIAL_DELAY_SECONDS,
                max_delay=RETRY_MAX_DELAY_SECONDS,
                log_fields={**self._log_fields, "op": "submit"},
            )
        except RETRYABLE_ERRORS as e:
            log_structured(logger.warning, op="submit", phase="gave_up", **self._log_fields, error=repr(e))
            raise WorkerUnreachableError(f"{self._method_label} submit failed: {e!r}") from e

    async def _submit_attempt(self, *, request: SubmitRequest, remaining: float) -> None:
        # [FUNCTION DOC] HELPER: One submit attempt: post the request body (without absent fields) to
        # the submit path of the method, with a timeout of ten seconds or the remaining time, whichever
        # is smaller, validating the response as a submit response. Logs acceptance at debug level.
        await self._transport.request(
            "POST",
            SUBMIT_PATH.format(method_name=self._spec.name),
            seconds=min(SUBMIT_ATTEMPT_TIMEOUT_SECONDS, remaining),
            response_model=SubmitResponse,
            json=request.model_dump(exclude_none=True),
        )
        log_structured(logger.debug, op="submit", phase="accepted", **self._log_fields)

    async def _poll_until_done(self) -> CallStatusResponse:
        # [FUNCTION DOC] HELPER: Repeat single polls with no delay between them (each poll itself
        # long-waits on the server) until one returns a final status, for at most the call timeout.
        # Returns the final status response. If the deadline passes while still pending, logs a
        # warning and raises a timeout error naming the method and call id.
        try:
            return await retry_until_deadline(
                self._poll_once,
                total_seconds=self._call_timeout_seconds,
                retry_on=_CallStillPendingError,
                initial_delay=0.0,
                log_fields={**self._log_fields, "op": "poll"},
            )
        except _CallStillPendingError:
            log_structured(
                logger.warning,
                op="poll",
                phase="still_pending",
                **self._log_fields,
                timeout_s=self._call_timeout_seconds,
            )
            raise TimeoutError(
                f"{self._method_label} (call id {self._call_id}) still pending after {self._call_timeout_seconds}s"
            ) from None

    async def _poll_once(self, remaining: float) -> CallStatusResponse:
        # [FUNCTION DOC] HELPER: Ask the server for the call status as a long poll. The client wait
        # is the smaller of the default poll timeout and the remaining time; the server is asked to
        # wait that long minus a slack of five seconds (or half of it, if smaller) so that the
        # server normally answers before the client gives up. A client timeout, a transport error or
        # a retryable server response means no outcome yet: for transport or server errors it first
        # sleeps one second (or the remaining time if shorter), then signals still-pending. A
        # pending status also signals still-pending. Other errors (protocol rejection, server
        # restart) propagate. Returns the status response when it is not pending.
        poll_seconds = min(DEFAULT_POLL_TIMEOUT_SECONDS, remaining)
        server_seconds = poll_seconds - min(POLL_SLACK_SECONDS, poll_seconds / 2)
        try:
            outcome = await self._transport.request(
                "GET",
                CALL_STATUS_PATH.format(call_id=self._call_id),
                seconds=poll_seconds,
                response_model=CallStatusResponse,
                params={"timeout": server_seconds},
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise _CallStillPendingError(f"long poll timed out after {poll_seconds:.1f}s") from e
        except (httpx.TransportError, RetryableResponseError) as e:
            await asyncio.sleep(min(RETRY_INITIAL_DELAY_SECONDS, remaining))
            raise _CallStillPendingError(f"poll attempt failed: {e!r}") from e

        if outcome.status == "pending":
            raise _CallStillPendingError("call still pending")
        return outcome

    @property
    def _log_fields(self) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Property: structured log fields identifying this call (an rpc tag,
        # the worker class name, the method name and the call id).
        return {"tag": "rpc", "worker": self._worker_cls_name, "method": self._spec.name, "call": self._call_id}

    @property
    def _method_label(self) -> str:
        # [FUNCTION DOC] HELPER: Property: the worker class name and method name joined by a dot,
        # used in error messages.
        return f"{self._worker_cls_name}.{self._spec.name}"
