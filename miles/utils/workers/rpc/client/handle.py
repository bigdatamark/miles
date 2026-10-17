# [MODULE DOC]
# Worker handle for workers served over the http rpc server. The handle is built from the worker
# class: every method of the class marked as an rpc method becomes an attribute that performs a
# remote call. Also implements readiness, idleness and death probing against the server's health
# and in-flight endpoints, with optional pinning of the server's boot identity so a restarted
# server process is detected.
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

import httpx

from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.rpc.client.call import RpcCall
from miles.utils.workers.rpc.client.misc import (
    RETRY_INITIAL_DELAY_SECONDS,
    RETRYABLE_ERRORS,
    BootUuidPin,
    RpcTransport,
    ServerRestartedError,
)
from miles.utils.workers.rpc.common.metadata import (
    RpcMethodSpec,
    canonicalize_method_arguments,
    collect_rpc_method_specs,
)
from miles.utils.workers.rpc.common.protocol import HEALTH_PATH, IN_FLIGHT_PATH, HealthResponse, InFlightResponse
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerStillBusyError, WorkerUnreachableError

DEFAULT_CALL_TIMEOUT_SECONDS = 3600.0
DEFAULT_READY_TIMEOUT_SECONDS = 600.0

_HEALTH_TIMEOUT_SECONDS = 5.0
_IDLE_POLL_INTERVAL_SECONDS = 5.0

logger = logging.getLogger(__name__)


class RpcWorkerHandle(BaseWorkerHandle):
    # [CLASS DOC]
    # Client proxy for one rpc server hosting one worker object. Rpc methods of the worker class
    # are reachable as attributes returning async callables; non-rpc names raise attribute errors.
    # When the stable-boot-identity option is on, the handle pins the server's boot uuid on first
    # contact (a handshake via the readiness probe) and treats any later change as a restart.
    def __init__(
        self,
        worker_cls: type,
        *,
        server_url: str,
        require_stable_boot_uuid: bool = False,
        call_timeout_seconds: float = DEFAULT_CALL_TIMEOUT_SECONDS,
        ready_timeout_seconds: float = DEFAULT_READY_TIMEOUT_SECONDS,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Collect the rpc method specs of the worker class. Raises a type error if any rpc method
        # name coincides with an existing attribute of this handle class (it would be shadowed and
        # unreachable). Stores the worker class name, the per-call timeout (default 3600 seconds)
        # and the readiness timeout used by the automatic handshake (default 600 seconds), creates
        # the boot uuid pin (required only when requested) and the transport for the server url
        # (using the given http client, or else the shared general client at request time).
        self._specs = collect_rpc_method_specs(worker_cls)
        shadowed = sorted(name for name in self._specs if hasattr(type(self), name))
        if shadowed:
            raise TypeError(f"{worker_cls.__name__} rpc methods shadow handle attributes: {shadowed}")

        self._worker_cls_name = worker_cls.__name__
        self._call_timeout_seconds = call_timeout_seconds
        self._ready_timeout_seconds = ready_timeout_seconds
        self._boot_uuid_pin = BootUuidPin(required=require_stable_boot_uuid, worker_cls_name=worker_cls.__name__)
        self._transport = RpcTransport(
            server_url=server_url, http_client=http_client, boot_uuid_pin=self._boot_uuid_pin
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text with the handle class name and the worker class name.
        return f"{type(self).__name__}({self._worker_cls_name})"

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        # [FUNCTION DOC]
        # Dynamic proxy, invoked only for names not found normally. Names starting with an
        # underscore raise attribute errors (so private and protocol lookups are not hijacked), as do
        # names that are not rpc methods of the worker class. For an rpc method returns an async
        # function that canonicalizes the positional and keyword arguments against the method spec
        # (into keyword form, validating them) and performs the full remote call, returning the
        # decoded result.
        if name.startswith("_"):
            raise AttributeError(name)
        spec = self._specs.get(name)
        if spec is None:
            raise AttributeError(f"{self._worker_cls_name} has no rpc method {name!r}")

        async def call(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: The returned proxy call; see the enclosing method.
            return await self._perform_call(
                spec=spec, kwargs=canonicalize_method_arguments(spec=spec, args=args, kwargs=kwargs)
            )

        return call

    async def submit_without_result(self, method_name: str, /, **kwargs: Any) -> None:
        # [FUNCTION DOC]
        # Fire-and-forget: submit the named rpc method with keyword arguments only, and return as
        # soon as the server accepted the submission, never fetching the outcome. Raises an
        # attribute error for an unknown method. Performs the boot identity handshake first if
        # needed, and then the submit with its retry rules.
        spec = self._specs.get(method_name)
        if spec is None:
            raise AttributeError(f"{self._worker_cls_name} has no rpc method {method_name!r}")

        call = await self._prepare_call(
            spec=spec, kwargs=canonicalize_method_arguments(spec=spec, args=(), kwargs=kwargs)
        )
        await call.submit()

    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None:
        # [FUNCTION DOC]
        # Poll the server's health endpoint until it answers, for at most the timeout in seconds,
        # retrying transport errors, timeouts and server errors at a fixed one-second interval
        # (each attempt limited to five seconds or the remaining time). On deadline raises the
        # worker-unreachable error. The first successful answer also pins the boot uuid if pinning
        # is required. With the allow-uuid-change flag, any existing pin is dropped before each
        # attempt so a new server process is accepted and pinned; if the whole wait then fails or is
        # cancelled, the previous pin is restored. With the flag, asserts after success that the
        # answering process was pinned.
        pinned_before = self._boot_uuid_pin.unpin() if allow_server_uuid_change else None

        async def attempt(remaining: float) -> None:
            # [FUNCTION DOC] HELPER: One health probe; drops the pin first when changes are allowed.
            if allow_server_uuid_change:
                self._boot_uuid_pin.unpin()
            await self._transport.request(
                "GET", HEALTH_PATH, seconds=min(_HEALTH_TIMEOUT_SECONDS, remaining), response_model=HealthResponse
            )

        try:
            try:
                await retry_until_deadline(
                    attempt,
                    total_seconds=timeout,
                    retry_on=RETRYABLE_ERRORS,
                    initial_delay=RETRY_INITIAL_DELAY_SECONDS,
                    backoff_factor=1.0,
                )
            except RETRYABLE_ERRORS as e:
                raise WorkerUnreachableError(
                    f"{self._worker_cls_name} rpc server not ready within {timeout}s: {e!r}"
                ) from e

            if allow_server_uuid_change:
                assert (
                    not self._boot_uuid_pin.needs_handshake()
                ), "the readiness probe did not pin the answering process"
        except BaseException:
            if allow_server_uuid_change:
                self._boot_uuid_pin.repin(pinned_before)
            raise

    async def wait_idle(self, *, timeout: float) -> None:
        # [FUNCTION DOC]
        # Poll the server's in-flight endpoint every five seconds until it reports no running call
        # ids, for at most the timeout in seconds; busy answers and retryable transport or server
        # errors are retried. When the deadline passes, the last error (still-busy or transport)
        # propagates.
        async def attempt(remaining: float) -> None:
            # [FUNCTION DOC] HELPER: One in-flight query (limited to five seconds or the remaining
            # time); raises the still-busy error naming the running call ids if there are any.
            response = await self._transport.request(
                "GET",
                IN_FLIGHT_PATH,
                seconds=min(_HEALTH_TIMEOUT_SECONDS, remaining),
                response_model=InFlightResponse,
            )
            if response.call_ids:
                raise WorkerStillBusyError(f"{self._worker_cls_name} is still running the calls {response.call_ids}")

        await retry_until_deadline(
            attempt,
            total_seconds=timeout,
            retry_on=(WorkerStillBusyError, *RETRYABLE_ERRORS),
            initial_delay=_IDLE_POLL_INTERVAL_SECONDS,
            backoff_factor=1.0,
            log_fields={"tag": "rpc", "op": "wait_idle", "target": self._worker_cls_name},
        )

    async def probe_is_dead(self) -> bool:
        # [FUNCTION DOC]
        # One health probe with a five-second limit. Returns true when the server process is
        # definitely gone or replaced: a server restart error (boot identity changed), or a
        # connection error whose cause chain contains a refused connection. Any other failure
        # (timeouts, other transport errors, server errors) is inconclusive and returns false, as
        # does a healthy answer.
        try:
            await self._transport.request(
                "GET", HEALTH_PATH, seconds=_HEALTH_TIMEOUT_SECONDS, response_model=HealthResponse
            )
        except ServerRestartedError:
            return True
        except httpx.ConnectError as error:
            return any(isinstance(e, ConnectionRefusedError) for e in _traverse_error_chain(error))
        except RETRYABLE_ERRORS:
            return False
        return False

    async def _perform_call(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] HELPER: Prepare the call (including any handshake) and run it to
        # completion, returning the decoded result.
        call = await self._prepare_call(spec=spec, kwargs=kwargs)
        return await call.run()

    async def _prepare_call(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any]) -> RpcCall:
        # [FUNCTION DOC] HELPER: Build the call object (which encodes the arguments), then, only if
        # the boot uuid is required but not yet pinned, wait for server readiness first (which pins
        # it). Returns the call object, not yet submitted.
        call = RpcCall(
            spec=spec,
            kwargs=kwargs,
            worker_cls_name=self._worker_cls_name,
            transport=self._transport,
            call_timeout_seconds=self._call_timeout_seconds,
        )

        if self._boot_uuid_pin.needs_handshake():
            await self.wait_ready(timeout=self._ready_timeout_seconds)

        return call


def _traverse_error_chain(error: BaseException) -> Iterator[BaseException]:
    # [FUNCTION DOC] HELPER: Generator yielding the exception, then its explicit cause or, if none,
    # its implicit context, and so on, stopping at the end of the chain or on a cycle.
    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        yield error
        if (cause := error.__cause__ or error.__context__) is None:
            return
        error = cause
