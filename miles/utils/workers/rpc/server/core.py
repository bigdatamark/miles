# [MODULE DOC]
# Transport-independent logic of the rpc server: validates and accepts submitted calls, hands them
# to the executor, and reports their status. Protocol errors are raised as http exceptions with a
# status code and logged as warnings.
from __future__ import annotations

import functools
import logging
import uuid
from typing import NoReturn

from fastapi import HTTPException
from pydantic import ValidationError

from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.rpc.common.protocol import (
    MAX_POLL_TIMEOUT_SECONDS,
    CallStatusResponse,
    InFlightResponse,
    SubmitRequest,
    SubmitResponse,
)
from miles.utils.workers.rpc.server.executor import RpcCallExecutor
from miles.utils.workers.rpc.server.store import CallStore, DuplicateCallError

logger = logging.getLogger(__name__)


class RpcServer:
    # [CLASS DOC]
    # Serves one worker object. Exposes a public boot uuid: a random hexadecimal string generated
    # per instance, identifying this server process incarnation to clients.
    def __init__(self, *, worker: object) -> None:
        # [FUNCTION DOC] Generate the boot uuid, collect the rpc method specs of the worker's class
        # (type errors for invalid methods propagate), create the call store and the executor, and
        # log the boot with the worker class name, uuid, method count and concurrency groups.
        self.boot_uuid = uuid.uuid4().hex
        self._specs = collect_rpc_method_specs(type(worker))
        self._store = CallStore()
        self._executor = RpcCallExecutor(worker=worker, specs=self._specs)
        log_structured(
            logger.info,
            tag="rpc",
            op="server",
            phase="boot",
            worker=type(worker).__name__,
            boot_uuid=self.boot_uuid,
            methods=len(self._specs),
            groups=self._executor.concurrency_groups,
        )

    def submit_call(self, *, method_name: str, request: SubmitRequest) -> SubmitResponse:
        # [FUNCTION DOC]
        # Accept a call and start it in the background; returns immediately with a submitted
        # acknowledgement. Rejections (logged, raised as http errors): 404 if the method is not an
        # exposed rpc method; 400 if the arguments fail validation; 409 if the call id was already
        # submitted (so retried submits are safe but report a conflict). Otherwise the call is
        # registered in the store and handed to the executor with a completion callback that
        # records the outcome in the store. Must run inside the event loop.
        def reject(*, status_code: int, reason: str, detail: str) -> NoReturn:
            # [FUNCTION DOC] HELPER: Log a rejection warning with reason, method, call id and detail,
            # then raise an http exception with the given status and detail. Never returns.
            log_structured(
                logger.warning,
                tag="rpc",
                op="submit",
                phase="reject",
                reason=reason,
                method=method_name,
                call=request.call_id,
                error=detail,
            )
            raise HTTPException(status_code=status_code, detail=detail)

        spec = self._specs.get(method_name)
        if spec is None:
            reject(status_code=404, reason="unknown_method", detail=f"unknown rpc method {method_name!r}")

        try:
            kwargs = spec.serializer.decode_query(request.query)
        except ValidationError as e:
            reject(status_code=400, reason="invalid_query", detail=str(e))

        try:
            self._store.begin(call_id=request.call_id)
        except DuplicateCallError as e:
            reject(status_code=409, reason="duplicate_call", detail=str(e))

        self._executor.start(
            spec=spec,
            kwargs=kwargs,
            call_id=request.call_id,
            finish=functools.partial(self._store.finish, call_id=request.call_id),
        )

        return SubmitResponse()

    def in_flight_calls(self) -> InFlightResponse:
        # [FUNCTION DOC] Return the sorted ids of submitted calls that have no outcome yet.
        return InFlightResponse(call_ids=self._store.in_flight_call_ids())

    async def query_call(self, *, call_id: str, timeout: float) -> CallStatusResponse:
        # [FUNCTION DOC]
        # Long poll. Raises 404 (and logs a warning) for an unknown call id, including calls
        # already purged by expiry. Otherwise waits up to the timeout in seconds, capped at 300,
        # for the call to finish. Returns a pending status if it did not finish in time, else the
        # final status with its encoded result or error text. Waiting marks the first time the
        # outcome was retrieved, which starts the shorter retention timer.
        if not self._store.contains(call_id):
            log_structured(logger.warning, tag="rpc", op="poll", phase="reject", reason="unknown_call", call=call_id)
            raise HTTPException(status_code=404, detail=f"unknown call id {call_id!r}")

        outcome = await self._store.wait(call_id=call_id, timeout=min(timeout, MAX_POLL_TIMEOUT_SECONDS))
        if outcome is None:
            return CallStatusResponse(status="pending")
        return CallStatusResponse(status=outcome.status, result=outcome.result, error=outcome.error)
