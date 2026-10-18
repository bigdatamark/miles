# [MODULE DOC] Tests for the thread-hosted uvicorn test server: which interface it listens on
# (the advertised host by default, an explicit bind host to widen it) and which address its url
# advertises. Real sockets and HTTP requests are used; servers get a free port starting at 21000.
from __future__ import annotations

import socket
from collections.abc import Callable, Iterator

import pytest
import requests
from fastapi import FastAPI

from miles.utils.misc import get_current_node_ip, get_free_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


def _make_app() -> FastAPI:
    # [FUNCTION DOC] HELPER: Builds a web app with a single health route returning an ok status.
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        # [FUNCTION DOC] HELPER: Route handler returning a mapping with status ok.
        return {"status": "ok"}

    return app


def _is_reachable(host: str, port: int) -> bool:
    # [FUNCTION DOC] HELPER: Tries a TCP connect to the host and port with a 2 second timeout and
    # returns true when it succeeds (connect result zero).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        return sock.connect_ex((host, port)) == 0


@pytest.fixture
def start_server() -> Iterator[Callable[..., UvicornThreadServer]]:
    # [FUNCTION DOC] HELPER: Fixture yielding a starter function; after the test, stops every
    # server it started.
    servers: list[UvicornThreadServer] = []

    def _start(**kwargs) -> UvicornThreadServer:
        # [FUNCTION DOC] HELPER: Creates a server for the health app on a free port found from
        # 21000, with the given extra keyword arguments (host, bind host), remembers and starts
        # it, and returns it.
        server = UvicornThreadServer(_make_app(), port=get_free_port(start_port=21000), **kwargs)
        servers.append(server)
        server.start()
        return server

    yield _start

    for server in servers:
        server.stop()


class TestBindHost:
    # [CLASS DOC] Tests of the bind host option versus the advertised host.
    def test_the_listener_defaults_to_the_advertised_host(self, start_server) -> None:
        # [FUNCTION DOC] Asserts the node ip is not loopback, starts a server advertising
        # loopback, and asserts it is reachable on loopback but not on the node ip.
        """Without ``bind_host`` the server listens only on the host it advertises."""
        node_ip = get_current_node_ip()
        assert node_ip != "127.0.0.1"

        server = start_server(host="127.0.0.1")

        assert _is_reachable(host="127.0.0.1", port=server.port)
        assert not _is_reachable(host=node_ip, port=server.port)

    def test_an_explicit_bind_host_widens_the_listener_beyond_the_advertised_host(self, start_server) -> None:
        # [FUNCTION DOC] A server advertising loopback with the wildcard bind host answers the
        # health route with 200 over both loopback and the node ip.
        """``bind_host`` must reach uvicorn, else a loopback-advertising server stays unreachable elsewhere."""
        node_ip = get_current_node_ip()

        server = start_server(host="127.0.0.1", bind_host="0.0.0.0")

        assert requests.get(f"http://127.0.0.1:{server.port}/health", timeout=5).status_code == 200
        assert requests.get(f"http://{node_ip}:{server.port}/health", timeout=5).status_code == 200

    def test_the_url_advertises_the_host_and_never_the_wildcard_bind_address(self, start_server) -> None:
        # [FUNCTION DOC] A server advertising the node ip with the wildcard bind host has host
        # equal to the node ip, a url made of http, the node ip and port, and that url serves the
        # health route with 200.
        """Peers dial ``url``, so it must carry the routable host even when the socket binds the wildcard address."""
        node_ip = get_current_node_ip()

        server = start_server(host=node_ip, bind_host="0.0.0.0")

        assert server.host == node_ip
        assert server.url == f"http://{node_ip}:{server.port}"
        assert requests.get(f"{server.url}/health", timeout=5).status_code == 200
