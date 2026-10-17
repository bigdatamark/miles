# [MODULE DOC]
# Standalone session-server process. Defines the server class (a web app plus one shared async
# HTTP client that forwards requests to the inference router and returns raw results, converting
# transport failures into a 502 result) and the subprocess entry points: run_session_server
# (configures logging and the process title, then serves with uvicorn) and main (parses the
# config from the command line). Module constant: request header names never forwarded upstream
# (content length, transfer encoding, host).
"""Standalone session-server process: HTTP chassis + upstream proxy transport.

- ``SessionServer`` is a FastAPI app plus one shared httpx client; ``do_proxy`` forwards a request to the inference router (sglang or miles) — which does the load balancing to worker engines — and returns the raw result, or a 502 JSON error on transport failure.
- Session/TITO logic lives in ``core.SessionCore``; ``setup_session_routes`` (``sessions.py``) wires the HTTP routes to it.
- Standalone (own process, own event loop) so sessions also work with the SGLang Rust Router or any other backend, decoupled from the Miles Router.
- ``run_session_server`` is the subprocess entry point: fresh interpreter, so it configures logging and the process title itself, then serves uvicorn.
"""

import json
import logging

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.core import ProxyRequest
from miles.rollout.session.sessions import setup_session_routes
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.workers.argv_utils import parse_config_argv

logger = logging.getLogger(__name__)

# Request headers that must not be forwarded verbatim to the upstream backend.
_DROP_REQUEST_HEADERS = ("content-length", "transfer-encoding", "host")


class SessionServer:
    # [CLASS DOC]
    # The web application object and proxy backend of the session server. Attributes: backend url,
    # the web app, a shared async HTTP client, and the additional-routing-replay flag. It is also
    # the backend object handed to the session core (which calls its proxy method).
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests through the inference router (sglang or miles)."""

    def __init__(self, config: SessionServerConfig):
        # [FUNCTION DOC] Build the app and HTTP client and register routes. The client allows up
        # to 1024 connections and uses the config timeout for all phases (none disables it). The
        # client's close is registered as a shutdown hook to avoid descriptor leaks. The
        # additional-routing-replay flag is true unless the pause generation mode is retract
        # (retract may recompute earlier rows so full routing data is needed). Session routes are
        # installed on the app with this object as backend (routes are skipped when no HF
        # checkpoint is configured).
        self.backend_url = config.backend_url
        self.app = FastAPI()

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1024),
            timeout=httpx.Timeout(config.timeout),
        )

        # Close the httpx connection pool when uvicorn shuts down to avoid FD leaks.
        self.app.router.on_shutdown.append(self.client.aclose)

        # `retract` may recompute earlier rows and must return full R3; all other
        # pause modes preserve prior rows and can request only the appended R3.
        self.use_addition_r3 = config.pause_generation_mode != "retract"
        setup_session_routes(self.app, self, config, use_addition_r3=self.use_addition_r3)

    async def do_proxy(self, request: ProxyRequest, path: str, *, body: bytes, headers: dict) -> dict:
        # [FUNCTION DOC]
        # Forward one request to the backend (network I/O). The url is backend url, a slash and
        # the path, plus the query string if non-empty. Request headers except content length,
        # transfer encoding and host (case-insensitive) are forwarded, with the body as raw
        # content. Returns a mapping with request body, response body bytes, status code and
        # response headers. On an HTTP transport error (connect, timeout, etc.) it logs a warning
        # and returns status 502 with a JSON error body and JSON content type header instead of
        # raising; HTTP error statuses from the backend are returned as ordinary results.
        url = f"{self.backend_url}/{path}"
        if request.query:
            url = f"{url}?{request.query}"

        headers = {k: v for k, v in headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
        except httpx.TransportError as exc:
            logger.warning("Proxy transport error for %s %s: %s", request.method, path, exc)
            error_body = json.dumps({"error": f"backend transport error: {type(exc).__name__}: {exc}"}).encode()
            return {
                "request_body": body,
                "response_body": error_body,
                "status_code": 502,
                "headers": {"content-type": "application/json"},
            }
        content = await response.aread()
        return {
            "request_body": body,
            "response_body": content,
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }


def run_session_server(config: SessionServerConfig):
    # [FUNCTION DOC] Process entry point (blocks). Side effects: configures raw logging for the
    # session server, lowers the http client libraries' log level to warning, sets the process
    # title so it can be killed by name, constructs the server, logs the bind and backend
    # addresses, and runs uvicorn on the configured host and port with info logging and access
    # logs off.
    """Entry point to start the standalone session server as a subprocess."""
    # Spawned as a fresh interpreter, so it inherits no logging config.
    configure_logger_raw("session_server")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-session-server")

    server = SessionServer(config)
    logger.info(
        "[session-server] Starting on %s:%s, proxying to %s",
        config.host,
        config.port,
        config.backend_url,
    )
    uvicorn.run(server.app, host=config.host, port=config.port, log_level="info", access_log=False)


def main(argv: list[str] | None = None) -> None:
    # [FUNCTION DOC] Parse the server config from the given command line arguments (the process
    # arguments when none given) and run the server.
    run_session_server(parse_config_argv(SessionServerConfig, argv))


if __name__ == "__main__":
    main()
