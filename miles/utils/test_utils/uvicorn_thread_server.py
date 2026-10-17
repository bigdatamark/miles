# [MODULE DOC] HELPER: Runs an ASGI application with uvicorn in a background daemon thread for
# tests, with a startup wait that polls the TCP port, and a stop that requests exit and joins the
# thread briefly.
import asyncio
import socket
import threading
import time

import uvicorn


class UvicornThreadServer:
    # [CLASS DOC] HELPER: Holds the app, the advertised host and port, the bind host (defaults to the
    # host), and, once started, the uvicorn server object and thread. The public host and port
    # attributes are what clients connect to; the bind host may differ (for example all interfaces).
    def __init__(self, app, host: str, port: int, bind_host: str | None = None):
        # [FUNCTION DOC] HELPER: Stores the app, host, port and bind host (the host itself when no
        # bind host is given); no server or thread exists yet.
        self._app = app
        self.host = host
        self.port = port
        self._bind_host = bind_host if bind_host is not None else host
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        # [FUNCTION DOC] HELPER: Plain http URL for the advertised host and port.
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        # [FUNCTION DOC] HELPER: Builds a uvicorn server for the app bound to the bind host and port
        # with info-level logging, runs its serve coroutine in a new event loop inside a daemon
        # thread, then blocks until the advertised host and port accept a TCP connection. Raises a
        # runtime error if that does not happen (see the wait helper). Calling start twice creates a
        # second server and thread.
        config = uvicorn.Config(self._app, host=self._bind_host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config)

        def run() -> None:
            # [FUNCTION DOC] HELPER: Thread body: runs the server's serve coroutine to completion in
            # a fresh event loop.
            asyncio.run(self._server.serve())

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        self._wait_for_port_open()

    def stop(self) -> None:
        # [FUNCTION DOC] HELPER: If started, sets the server's exit flag, then, if the thread is alive,
        # waits up to 2 seconds for it to end. It may return with the thread still running if
        # shutdown is slower than that. Safe to call when never started.
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _wait_for_port_open(self) -> None:
        # [FUNCTION DOC] HELPER: Polls up to 50 times, 0.1 seconds apart (about 5 seconds), trying a
        # TCP connection to the advertised host and port; returns on the first success. Errors in a
        # probe are swallowed. Raises a runtime error naming the URL if no probe succeeds.
        for _ in range(50):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex((self.host, self.port))
                sock.close()
                if result == 0:
                    return
            except Exception:
                pass
            time.sleep(0.1)
        raise RuntimeError(f"Failed to start server on {self.url}")
