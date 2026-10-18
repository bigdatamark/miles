# [MODULE DOC] Tests of the session server process module: the server class applies the
# configured timeout to its shared proxy HTTP client; the run function silences routine request
# logs of the HTTP client libraries (sets their level to warning) and starts uvicorn with the
# server's application on the configured host and port with info logging and no access log; and
# the command-line entry point parses the config payload from arguments losslessly and exits
# when the payload is missing. Process-level collaborators (logger setup, process title, the
# server class, uvicorn) are replaced by fakes.
from __future__ import annotations

import logging

import httpx
import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session import server as session_server_module
from miles.rollout.session.server import SessionServer, main
from miles.utils.workers.argv_utils import config_to_argv


class TestSessionServer:
    # [CLASS DOC] Tests of the session server class.
    def test_constructor_applies_configured_timeout_to_proxy_client(self):
        # [FUNCTION DOC] Builds a server from a config with timeout 7.5 seconds and asserts its
        # proxy client's timeout equals an HTTP timeout of 7.5.
        """The configured timeout reaches the shared proxy client instead of an httpx default."""
        server = SessionServer(make_session_server_config(timeout=7.5))

        assert server.client.timeout == httpx.Timeout(7.5)


def test_run_session_server_suppresses_routine_request_logs(monkeypatch):
    # [FUNCTION DOC] Resets the httpx and httpcore loggers' levels to unset (restored by the
    # monkeypatch fixture afterwards), replaces the logger configuration and process title
    # functions with no-ops, the server class with a fake exposing a sentinel application, and
    # uvicorn's run function with a recorder. Runs the server for host 127.0.0.1 port 31001 and
    # asserts both client library loggers are at warning level and uvicorn was called with the
    # fake application, that host and port, log level info and access logging off.
    app = object()
    uvicorn_call = {}
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    monkeypatch.setattr(httpx_logger, "level", logging.NOTSET)
    monkeypatch.setattr(httpcore_logger, "level", logging.NOTSET)

    class FakeSessionServer:
        # [CLASS DOC] HELPER: Fake server class whose instances expose the sentinel application.
        def __init__(self, config):
            # [FUNCTION DOC] HELPER: Ignores the config and stores the sentinel application.
            self.app = app

    monkeypatch.setattr(session_server_module, "configure_logger_raw", lambda *_: None)
    monkeypatch.setattr(session_server_module.setproctitle, "setproctitle", lambda *_: None)
    monkeypatch.setattr(session_server_module, "SessionServer", FakeSessionServer)

    def fake_uvicorn_run(received_app, **kwargs):
        # [FUNCTION DOC] HELPER: Records the received application and keyword arguments in the
        # enclosing call record instead of serving.
        uvicorn_call["app"] = received_app
        uvicorn_call.update(kwargs)

    monkeypatch.setattr(session_server_module.uvicorn, "run", fake_uvicorn_run)
    config = make_session_server_config(host="127.0.0.1", port=31001)

    session_server_module.run_session_server(config)

    assert httpx_logger.level == logging.WARNING
    assert httpcore_logger.level == logging.WARNING
    assert uvicorn_call == {
        "app": app,
        "host": "127.0.0.1",
        "port": 31001,
        "log_level": "info",
        "access_log": False,
    }


class TestMain:
    # [CLASS DOC] Tests of the command-line entry point.
    def test_feeds_the_parsed_config_to_the_server(self, monkeypatch):
        # [FUNCTION DOC] Replaces the run function with a recorder, converts a config (port 5005,
        # instance id "abc", backend URL) to its argument list, runs the entry point on it, and
        # asserts the run function received exactly one config equal to the original.
        """The CLI parses the config payload losslessly."""
        calls = []
        monkeypatch.setattr(session_server_module, "run_session_server", lambda config: calls.append(config))
        config = make_session_server_config(port=5005, instance_id="abc", backend_url="http://127.0.0.1:3000")

        main(config_to_argv(config))

        assert calls == [config]

    def test_missing_config_is_rejected(self):
        # [FUNCTION DOC] Runs the entry point with no arguments and asserts it exits (raises a
        # system exit) because the config payload is mandatory.
        """The config payload is mandatory for a session server."""
        with pytest.raises(SystemExit):
            main([])
