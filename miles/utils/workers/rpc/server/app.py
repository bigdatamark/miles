# [MODULE DOC]
# Builds the web application serving one worker over rpc. Routes: health, list of in-flight calls,
# submit a call by method name, and long-poll call status by id. A middleware enforces the boot
# uuid contract and stamps every response with the server's boot uuid. Status codes: 400 for
# malformed requests, 404 unknown method or call, 409 duplicate call id, 412 boot uuid mismatch,
# 500 unhandled error.
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.common.protocol import (
    BOOT_UUID_HEADER,
    BOOT_UUID_MISMATCH_STATUS,
    CALL_STATUS_PATH,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    EXPECTED_BOOT_UUID_HEADER,
    HEALTH_PATH,
    IN_FLIGHT_PATH,
    SUBMIT_PATH,
    CallStatusResponse,
    HealthResponse,
    InFlightResponse,
    SubmitRequest,
    SubmitResponse,
)
from miles.utils.workers.rpc.server.core import RpcServer

logger = logging.getLogger(__name__)


def create_rpc_app(worker: object) -> FastAPI:
    # [FUNCTION DOC]
    # Create the application for the worker object. Builds the request core (which generates the
    # server's boot uuid, collects the worker class's rpc method specs, and may raise a type error
    # for invalid methods), then registers the middleware, the validation error handler and the
    # four routes, which all delegate to the core. Returns the application; nothing is served yet.
    server = RpcServer(worker=worker)

    app = FastAPI()

    @app.middleware("http")
    async def boot_uuid_guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # [FUNCTION DOC] HELPER: Middleware. If the request carries an expected boot uuid header
        # that differs from this server's, log a warning and answer 412 with a detail message
        # without running the route. Otherwise run the route; an unhandled exception is logged and
        # becomes a 500 response. In every case the response gets the server's boot uuid header.
        expected = request.headers.get(EXPECTED_BOOT_UUID_HEADER)
        if expected is not None and expected != server.boot_uuid:
            log_structured(
                logger.warning,
                tag="rpc",
                op="server",
                phase="reject",
                reason="boot_uuid_mismatch",
                expected=expected,
                actual=server.boot_uuid,
            )
            response: Response = JSONResponse(
                status_code=BOOT_UUID_MISMATCH_STATUS,
                content={"detail": f"boot uuid mismatch: client expected {expected}, server is {server.boot_uuid}"},
            )
        else:
            try:
                response = await call_next(request)
            except Exception:
                log_structured(
                    logger.error, tag="rpc", op="server", phase="unhandled_error", path=request.url.path, exc_info=True
                )
                response = JSONResponse(status_code=500, content={"detail": "unhandled rpc server error"})

        response.headers[BOOT_UUID_HEADER] = server.boot_uuid
        return response

    @app.exception_handler(RequestValidationError)
    async def handle_malformed_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # [FUNCTION DOC] HELPER: Turn framework request validation failures into a 400 response
        # with the error text as detail.
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get(HEALTH_PATH)
    async def health() -> HealthResponse:
        # [FUNCTION DOC] HELPER: Route: liveness answer, always ok.
        return HealthResponse()

    @app.get(IN_FLIGHT_PATH)
    async def in_flight_calls() -> InFlightResponse:
        # [FUNCTION DOC] HELPER: Route: ids of calls submitted and not yet finished.
        return server.in_flight_calls()

    @app.post(SUBMIT_PATH)
    async def submit_call(method_name: str, request: SubmitRequest) -> SubmitResponse:
        # [FUNCTION DOC] HELPER: Route: accept a call for the method named in the path; see the
        # core's submit for validation and error statuses.
        return server.submit_call(method_name=method_name, request=request)

    @app.get(CALL_STATUS_PATH)
    async def query_call(
        call_id: str, timeout: float = Query(default=DEFAULT_POLL_TIMEOUT_SECONDS, ge=0.0)
    ) -> CallStatusResponse:
        # [FUNCTION DOC] HELPER: Route: long-poll the call status, waiting up to the timeout query
        # parameter in seconds (default 30, must be at least zero; the core caps it at 300).
        return await server.query_call(call_id=call_id, timeout=timeout)

    return app
