# [MODULE DOC]
# Fault tolerance api server entry point. Builds cell handlers for the enabled fault tolerance
# components (train cells of type actor, rollout cells of type rollout), wraps them in a registry,
# creates a fastapi app with routes (health, list cells, get cell, patch cell to suspend/resume,
# inject fault) and runs it with uvicorn on a daemon thread, blocking until it has started.
# Errors map to kubernetes-style status bodies through a private error type.
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from miles.ray.specs.inference import compute_engine_pool_ids
from miles.ray.specs.train import compute_trainer_pool_id
from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.models import Cell, CellList, CellPatch, FaultInjection, K8sStatus, _OkResponse
from miles.utils.ft_utils.api_server.registry import _CellRegistry
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_handle import BaseWorkerHandle

logger = logging.getLogger(__name__)

_API_SERVER_STARTUP_TIMEOUT_SECONDS = 30.0
_THREAD_READY_POLL_INTERVAL_SECONDS = 0.05


# -------------------------- entrypoint ------------------------------


def start_api_server(
    *,
    args,
    trainer_models: dict[str, BaseWorkerHandle],
    inference_controller: BaseWorkerHandle | None,
    host: str = "127.0.0.1",
    port: int,
    ft_components: list[str],
    cell_operations: BaseCellOperations,
) -> None:
    # [FUNCTION DOC] Starts the api server in the background and returns once it is accepting
    # connections. If the train component is enabled, adds an actor-type handler whose status
    # sources are the trainer model handles and whose pool ids are the trainer pool ids of the
    # trainer ids. If the rollout component is enabled, asserts an inference controller exists and
    # adds a rollout-type handler using it, with the engine pool ids computed from args. Shared cell
    # operations back both. Default bind is loopback. The created server object is discarded (it
    # cannot be stopped by the caller). Raises per startup failure rules of the thread starter.
    handlers: list[_CellHandler] = []

    if "train" in ft_components:
        handlers.append(
            _CellHandler(
                cell_type="actor",
                operations=cell_operations,
                controllers=list(trainer_models.values()),
                pool_ids=[compute_trainer_pool_id(trainer_id) for trainer_id in trainer_models],
            )
        )

    if "rollout" in ft_components:
        assert inference_controller is not None, (
            "rollout cells are suspended and resumed through the inference controller, so a deployment that runs "
            "none of its own cannot answer for them"
        )
        handlers.append(
            _CellHandler(
                cell_type="rollout",
                operations=cell_operations,
                controllers=[inference_controller],
                pool_ids=compute_engine_pool_ids(args),
            )
        )

    _start_api_server_raw(registry=_CellRegistry(handlers), host=host, port=port)


def _start_api_server_raw(*, registry: _CellRegistry, port: int, host: str) -> uvicorn.Server:
    # [FUNCTION DOC] HELPER: Creates the app for the registry and a uvicorn server for the host and
    # port, runs the server on a daemon thread and waits (up to 30 seconds) for the started flag.
    # Returns the server object.
    app = _create_api_app(registry)

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port))
    _start_and_wait_thread(
        target=server.run,
        is_ready=lambda: server.started,
        description=f"Api server on port {port}",
        timeout_seconds=_API_SERVER_STARTUP_TIMEOUT_SECONDS,
    )
    return server


# -------------------------- main app ------------------------------


def _create_api_app(registry: _CellRegistry) -> FastAPI:
    # [FUNCTION DOC] HELPER: Builds the fastapi application with the error handler and the routes
    # (all under the version one api prefix): get health; get cells; get cell by name; patch cell by
    # name; post inject-fault for a cell. Route handlers close over the registry.
    app = FastAPI()

    # -------------------------- exceptions ------------------------------

    @app.exception_handler(_K8sError)
    async def _handle_k8s_error(request: Request, exc: _K8sError) -> JSONResponse:
        # [FUNCTION DOC] HELPER: Converts the private error into a json response with the error's
        # http status code and a kubernetes-style status body (message, reason, code).
        return JSONResponse(
            status_code=exc.status_code,
            content=K8sStatus(message=exc.message, reason=exc.reason, code=exc.status_code).model_dump(),
        )

    # -------------------------- APIs ------------------------------

    @app.get("/api/v1/health")
    async def health() -> _OkResponse:
        # [FUNCTION DOC] HELPER: Liveness route; returns the ok body.
        return _OkResponse()

    @app.get("/api/v1/cells")
    async def get_cells() -> CellList:
        # [FUNCTION DOC] HELPER: Route returning all cells of all handlers as a list resource.
        return CellList(items=await registry.list_cells())

    @app.get("/api/v1/cells/{name}")
    async def get_cell(name: str) -> Cell:
        # [FUNCTION DOC] HELPER: Route returning one cell; 404 status body if no handler owns the id.
        handler = await _resolve(name)
        return await handler.get_cell(name)

    @app.patch("/api/v1/cells/{name}")
    async def patch_cell(name: str, body: CellPatch) -> Cell:
        # [FUNCTION DOC] HELPER: Route that applies a patch. 404 if the cell is unknown. If the body
        # has a spec with a non-none suspend flag, suspends (true) or resumes (false) the cell; any
        # exception is logged and mapped to a 500 internal error status. A body without a suspend
        # flag changes nothing. Returns the cell as read after the operation (state may lag).
        handler = await _resolve(name)

        if body.spec is not None and body.spec.suspend is not None:
            try:
                if body.spec.suspend:
                    await handler.suspend(name)
                else:
                    await handler.resume(name)
            except Exception as err:
                logger.error("Failed to patch cell %s", name, exc_info=True)
                raise _K8sError(
                    status_code=500, reason="InternalError", message=f"Failed to patch cell '{name}'"
                ) from err

        return await handler.get_cell(name)

    @app.post("/api/v1/cells/{name}/inject-fault")
    async def inject_fault(name: str, body: FaultInjection) -> _OkResponse:
        # [FUNCTION DOC] HELPER: Route that injects a fault with the body's mode and sub index.
        # 404 for unknown cells; a not-implemented error from the backend maps to a 400 bad request
        # carrying its text; any other exception is logged and mapped to a 500. Returns the ok body.
        handler = await _resolve(name)
        try:
            await handler.inject_fault(name, mode=body.mode, sub_index=body.sub_index)
        except NotImplementedError as err:
            raise _K8sError(
                status_code=400,
                reason="BadRequest",
                message=str(err),
            ) from err
        except Exception as err:
            logger.error("Failed to inject fault into cell %s", name, exc_info=True)
            raise _K8sError(
                status_code=500,
                reason="InternalError",
                message=f"Failed to inject fault into cell '{name}'",
            ) from err
        return _OkResponse()

    # -------------------------- utils ------------------------------

    async def _resolve(name: str) -> _CellHandler:
        # [FUNCTION DOC] HELPER: Finds the handler for a cell name via the registry; converts the
        # key error into a 404 not-found error whose message names the cell.
        try:
            return await registry.resolve(name)
        except KeyError:
            raise _K8sError(status_code=404, reason="NotFound", message=f"Cell '{name}' not found") from None

    return app


# -------------------------- exception ------------------------------


class _K8sError(Exception):
    # [CLASS DOC] HELPER: Exception carrying http status code, machine readable reason and message;
    # rendered by the app's error handler as a kubernetes-style status body.
    def __init__(self, *, status_code: int, reason: str, message: str) -> None:
        # [FUNCTION DOC] HELPER: Stores the three fields as attributes.
        # KNOWN ISSUE: does not call the base exception constructor, so the exception's own string
        # form and args are empty.
        self.status_code = status_code
        self.reason = reason
        self.message = message


# -------------------------- thread startup ------------------------------


def _start_and_wait_thread(
    *,
    target: Callable[[], None],
    is_ready: Callable[[], bool],
    description: str,
    timeout_seconds: float,
) -> threading.Thread:
    # [FUNCTION DOC] HELPER: Runs the target on a daemon thread and blocks until the readiness
    # predicate is true. Polls every 50 milliseconds; while not ready: if the thread captured an
    # exception raises a runtime error (chained to it) saying startup failed; if the thread ended
    # raises a runtime error saying it exited during startup; if the deadline passed raises a
    # timeout error. When ready logs the start and returns the thread.
    error: list[BaseException] = []

    def _run() -> None:
        # [FUNCTION DOC] HELPER: Thread body; runs the target, and on any exception (even system
        # exit) logs it and records it for the waiting thread.
        try:
            target()
        except BaseException as err:  # noqa: BLE001 - re-raised on the caller thread below
            logger.error("%s died", description, exc_info=True)
            error.append(err)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout_seconds
    while not is_ready():
        if error:
            raise RuntimeError(f"{description} failed during startup") from error[0]
        if not thread.is_alive():
            raise RuntimeError(f"{description} exited during startup")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{description} did not finish startup within {timeout_seconds}s")
        time.sleep(_THREAD_READY_POLL_INTERVAL_SECONDS)

    logger.info("%s started", description)
    return thread
