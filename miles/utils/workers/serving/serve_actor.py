# [MODULE DOC]
# Ray actor body that hosts a worker and serves it over the http rpc server on a background
# thread. If the server ever stops or crashes, the whole process is terminated so the failure is
# visible to the supervisor.
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import Any

import uvicorn

from miles.utils.misc import NodeProbeMixin
from miles.utils.test_utils.fault_injector import inject_fault as _inject_fault
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.serving.utils import create_server_socket

logger = logging.getLogger(__name__)

_SERVER_THREAD_NAME = "miles-rpc-server"


class ServeActor(NodeProbeMixin):
    # [CLASS DOC]
    # Generic actor holding one worker object, built by a factory at construction. Inherits node
    # probing from the node probe mixin. The rpc server is started separately and at most once.
    def __init__(self, *, build_worker: Callable[[], Any]) -> None:
        # [FUNCTION DOC] Call the factory (any exception propagates and fails actor creation) and
        # keep the worker; no server thread yet.
        self._worker = build_worker()
        self._server_thread: threading.Thread | None = None

    def start_rpc_server(self, *, port: int) -> None:
        # [FUNCTION DOC] Asserts the server was not started yet. Builds the rpc application for the
        # worker and starts a daemon thread named for the rpc server that listens on the port (all
        # interfaces). Returns without waiting for the server to be ready.
        assert self._server_thread is None, "the rpc server of this actor is already running"

        app = create_rpc_app(self._worker)
        logger.info(f"ServeActor serves {type(self._worker).__name__} on port {port}")
        self._server_thread = threading.Thread(
            target=serve_until_stopped,
            kwargs=dict(app=app, port=port),
            name=_SERVER_THREAD_NAME,
            daemon=True,
        )
        self._server_thread.start()

    def inject_fault(self, mode: str) -> None:
        # [FUNCTION DOC] Inject a fault of the given mode (string) into this very process through
        # the shared fault injector helper (which may kill or crash the process).
        _inject_fault(mode=mode)


def serve_until_stopped(*, app: Any, port: int) -> None:
    # [FUNCTION DOC]
    # Bind a server socket on the port, run the uvicorn server for the application on it (blocking)
    # and, whenever serving ends, whether by returning normally or by any error including
    # interrupts, log an error and immediately exit the whole process with status 1 without
    # cleanup. Therefore never returns.
    try:
        with create_server_socket(port=port) as server_socket:
            logger.info(f"The rpc server listens on {server_socket.getsockname()}")
            uvicorn.Server(uvicorn.Config(app)).run(sockets=[server_socket])
        logger.error(f"The rpc server on port {port} stopped serving, so the process it lives in exits with it")
    except BaseException:
        logger.error(f"The rpc server on port {port} crashed, so the process it lives in exits with it", exc_info=True)

    os._exit(1)
