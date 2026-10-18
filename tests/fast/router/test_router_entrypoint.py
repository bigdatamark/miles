# [MODULE DOC] Tests for the miles router's process entrypoints: that running the router module as
# a command with the help flag exits successfully and mentions the config-json option (in a real
# subprocess), and that the run-router function builds a router from the given config and starts the
# web server on the configured host and port (with the logger setup, process title, router class and
# server run function replaced by recorders, so nothing is actually served).
import subprocess
import sys

import pytest

from miles.router import router as router_module
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter, run_router


def test_miles_router_module_help_exits_successfully() -> None:
    # [FUNCTION DOC] Runs the router module as a command with the help flag in a subprocess using the
    # current interpreter, capturing text output with a 120 second timeout (generous; it only guards
    # against hangs because the whole stack is imported). Asserts the exit code is 0 (the error
    # output is the failure message) and the standard output mentions the config-json option.
    """The Miles router module exposes its CLI help without starting the server."""
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-m", "miles.router.router", "--help"],
        capture_output=True,
        text=True,
        # Generous because this imports the whole miles stack, and a loaded runner takes longer
        # than the few seconds it costs idle; the bound is here to catch a hang, not to time it.
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config-json" in result.stdout


class RouterRecorder:
    # [CLASS DOC] HELPER: Stand-in factory for the router class that builds real routers and keeps
    # them in a list.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty list of created routers.
        self.routers: list[MilesRouter] = []

    def create(self, config: MilesRouterConfig, verbose: bool = False) -> MilesRouter:
        # [FUNCTION DOC] HELPER: Builds a real router from the config and verbosity flag, appends it
        # to the list, and returns it.
        router = MilesRouter(config, verbose=verbose)
        self.routers.append(router)
        return router


class UvicornRunRecorder:
    # [CLASS DOC] HELPER: Stand-in for the web server run function that records what it was asked to
    # serve instead of binding a port.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with empty lists of applications and (host, port) pairs.
        self.apps: list[object] = []
        self.bind_addresses: list[tuple[str, int]] = []

    def run(self, app: object, *, host: str, port: int, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: Records the application and the host and port; extra keyword
        # options are ignored. Returns immediately without serving.
        self.apps.append(app)
        self.bind_addresses.append((host, port))


class TestRunRouter:
    # [CLASS DOC] Tests of the run-router entrypoint function.
    def test_config_controls_router_construction_and_uvicorn_bind_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Builds a config for host 192.0.2.7, port 21234 (11 max connections, no
        # timeout, 1.0 s health interval, threshold 2). Patches the router module: the logger
        # configuration and process title setters become no-ops, the router class becomes the
        # recorder's factory, and the server run function becomes the recorder. After calling the
        # run-router function: exactly one router was built, it holds the same config object, the
        # server was asked to serve that router's application, and the bind address was
        # ("192.0.2.7", 21234).
        """`run_router` builds the router from the given config and serves its app on the configured host and port."""
        config = MilesRouterConfig(
            host="192.0.2.7",
            port=21234,
            max_connections=11,
            timeout=None,
            health_check_interval=1.0,
            health_check_failure_threshold=2,
        )
        router_recorder = RouterRecorder()
        uvicorn_recorder = UvicornRunRecorder()
        monkeypatch.setattr(router_module, "configure_logger_raw", lambda name: None)
        monkeypatch.setattr(router_module.setproctitle, "setproctitle", lambda title: None)
        monkeypatch.setattr(router_module, "MilesRouter", router_recorder.create)
        monkeypatch.setattr(router_module.uvicorn, "run", uvicorn_recorder.run)

        run_router(config)

        assert len(router_recorder.routers) == 1
        assert router_recorder.routers[0].config is config
        assert uvicorn_recorder.apps == [router_recorder.routers[0].app]
        assert uvicorn_recorder.bind_addresses == [("192.0.2.7", 21234)]
