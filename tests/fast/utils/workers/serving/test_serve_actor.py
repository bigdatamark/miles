# [MODULE DOC] Unit tests for the serve actor (the ray actor that builds a worker in its own process
# and serves it over rpc in a background thread) and the serve-until-stopped function. Covers: the
# worker being built once at construction, the server thread being a named daemon thread started
# once with the port and an application serving the actor's worker, the process exiting with status
# 1 whenever the server returns or crashes, the server receiving the socket bound to the ipv6
# wildcard host and closing it, fault injection being forwarded to the process-level injector, and one
# test with a real thread. Server socket creation, uvicorn and thread creation are replaced by fakes.
from __future__ import annotations

import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from miles.utils.workers.serving import serve_actor as serve_actor_module
from miles.utils.workers.serving.serve_actor import ServeActor, serve_until_stopped
from miles.utils.workers.serving.utils import IPV6_WILDCARD_HOST


class DemoWorker:
    # [CLASS DOC] HELPER: Trivial worker holding a tag and echoing a value.
    def __init__(self, *, tag: str) -> None:
        # [FUNCTION DOC] HELPER: Stores the tag.
        self.tag = tag

    def echo(self, *, value: int) -> int:
        # [FUNCTION DOC] HELPER: Returns the value unchanged.
        return value


def _build_worker(tag: str = "demo"):
    # [FUNCTION DOC] HELPER: Returns a zero-argument builder function that creates a demo worker with
    # the given tag (default "demo") each time it is called.
    return lambda: DemoWorker(tag=tag)


class _FakeServerSocket:
    # [CLASS DOC] HELPER: Fake server socket that reports a fixed address and records being closed
    # when used as a context manager.
    def __init__(self, address: tuple[str, int]) -> None:
        # [FUNCTION DOC] HELPER: Stores the address and starts not closed.
        self.address = address
        self.closed = False

    def getsockname(self) -> tuple[str, int]:
        # [FUNCTION DOC] HELPER: Returns the stored address.
        return self.address

    def __enter__(self) -> _FakeServerSocket:
        # [FUNCTION DOC] HELPER: Returns itself.
        return self

    def __exit__(self, *args: Any) -> None:
        # [FUNCTION DOC] HELPER: Marks the socket closed; never suppresses exceptions.
        self.closed = True


def _patch_serving(monkeypatch, *, run: Callable[..., None]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Replaces, in the serve actor module, the socket creation function with a
    # fake that records the requested port and returns a fake socket at the ipv6 wildcard host and that
    # port; replaces uvicorn's config constructor with one that records the app and returns a dummy
    # config; and replaces uvicorn's server constructor with one returning an object whose run is the
    # given callable. Returns the mapping that records the created port and the app.
    served: dict[str, Any] = {}

    def create_server_socket(*, port: int) -> _FakeServerSocket:
        # [FUNCTION DOC] HELPER: Records the port and returns a fake socket bound to the wildcard host.
        served["created_port"] = port
        return _FakeServerSocket((IPV6_WILDCARD_HOST, port))

    monkeypatch.setattr(serve_actor_module, "create_server_socket", create_server_socket)
    monkeypatch.setattr(serve_actor_module.uvicorn, "Config", lambda app: served.update(app=app) or "config")
    monkeypatch.setattr(serve_actor_module.uvicorn, "Server", lambda config: SimpleNamespace(run=run))
    return served


@pytest.fixture
def started_threads(monkeypatch) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Fixture replacing the thread class in the serve actor module with a fake
    # that records the target, keyword arguments, name and daemon flag of each created thread and does
    # not run anything on start. Returns the list of recorded thread descriptions.
    started: list[dict[str, Any]] = []

    class _FakeThread:
        # [CLASS DOC] HELPER: Thread stand-in that only records its construction arguments.
        def __init__(self, *, target, kwargs, name, daemon) -> None:
            # [FUNCTION DOC] HELPER: Appends a mapping of the four arguments to the recorded list.
            started.append(dict(target=target, kwargs=kwargs, name=name, daemon=daemon))

        def start(self) -> None:
            # [FUNCTION DOC] HELPER: Does nothing.
            pass

    monkeypatch.setattr(serve_actor_module.threading, "Thread", _FakeThread)
    return started


class TestTheWorkerLivesInTheActor:
    # [CLASS DOC] Tests that the actor constructs its worker from the given recipe.
    def test_the_worker_is_built_when_the_actor_is_constructed(self):
        # [FUNCTION DOC] The actor's private worker is a demo worker with the tag from the recipe.
        """RDT and the rest of the ray ecosystem need the worker inside the actor process, not beside it."""
        actor = ServeActor(build_worker=_build_worker(tag="built-here"))

        assert isinstance(actor._worker, DemoWorker) and actor._worker.tag == "built-here"

    def test_the_recipe_is_evaluated_exactly_once(self):
        # [FUNCTION DOC] A recipe that records each call is invoked exactly once by construction.
        """A second evaluation would build a second worker and serve one while the other holds the gpu."""
        calls: list[int] = []

        ServeActor(build_worker=lambda: calls.append(1) or DemoWorker(tag="x"))

        assert calls == [1]


class TestStartingTheServer:
    # [CLASS DOC] Tests start-rpc-server with a fake thread class.
    def test_the_server_runs_in_a_background_thread(self, started_threads):
        # [FUNCTION DOC] Starting the server creates exactly one daemon thread whose keyword arguments
        # carry the requested port.
        """The actor must stay answerable: the launcher still probes ports and kills it through ray."""
        actor = ServeActor(build_worker=_build_worker())

        actor.start_rpc_server(port=12345)

        assert len(started_threads) == 1
        assert started_threads[0]["daemon"] is True
        assert started_threads[0]["kwargs"]["port"] == 12345

    def test_the_app_serves_the_actor_s_own_worker(self, started_threads):
        # [FUNCTION DOC] The application passed to the thread has routes (it was built from the
        # actor's worker). The test only checks the routes are non-empty.
        """Serving a different object would answer calls that never reach the worker holding the gpu."""
        actor = ServeActor(build_worker=_build_worker())

        actor.start_rpc_server(port=12345)

        assert started_threads[0]["kwargs"]["app"].routes

    def test_the_server_can_only_be_started_once(self, started_threads):
        # [FUNCTION DOC] A second start on another port raises an assertion error.
        """Two servers on one worker would each bind a port and answer with a different boot uuid."""
        actor = ServeActor(build_worker=_build_worker())
        actor.start_rpc_server(port=12345)

        with pytest.raises(AssertionError):
            actor.start_rpc_server(port=12346)

    def test_the_server_thread_is_named_for_the_stacks_it_appears_in(self, started_threads):
        # [FUNCTION DOC] The thread's name contains the text "rpc".
        """A hung train step is read off thread names, and 'Thread-7' says nothing."""
        actor = ServeActor(build_worker=_build_worker())

        actor.start_rpc_server(port=12345)

        assert "rpc" in started_threads[0]["name"]


class TestTheActorDiesWithItsServer:
    # [CLASS DOC] Tests that the serve function ends the process (exit code 1, the exit call being
    # replaced by a recorder) whenever the server stops.
    def test_a_server_that_returns_takes_the_process_with_it(self, monkeypatch):
        # [FUNCTION DOC] With a server whose run returns at once, serving ends with exactly one exit
        # call with code 1.
        """A live actor whose server stopped is a worker nobody can call and nothing reports as dead."""
        exits: list[int] = []
        _patch_serving(monkeypatch, run=lambda *, sockets: None)
        monkeypatch.setattr(serve_actor_module.os, "_exit", lambda code: exits.append(code))

        serve_until_stopped(app=object(), port=12345)

        assert exits == [1]

    def test_a_server_that_crashes_takes_the_process_with_it(self, monkeypatch):
        # [FUNCTION DOC] With a server whose run raises an os error (address in use), serving still
        # ends with one exit call with code 1 and does not propagate the error.
        """Binding an already taken port raises rather than returns, and must end the same way."""
        exits: list[int] = []

        def _explode(*, sockets: list[Any]) -> None:
            # [FUNCTION DOC] HELPER: Always raises an os error "address already in use".
            raise OSError("address already in use")

        _patch_serving(monkeypatch, run=_explode)
        monkeypatch.setattr(serve_actor_module.os, "_exit", lambda code: exits.append(code))

        serve_until_stopped(app=object(), port=12345)

        assert exits == [1]

    def test_the_server_is_handed_the_socket_bound_on_the_wildcard(self, monkeypatch):
        # [FUNCTION DOC] Asserts the socket was created for the requested port, uvicorn's config got
        # the given app, the server's run received a socket whose address is the ipv6 wildcard host and
        # that port, and that socket was closed after serving. The exit call is a no-op here.
        """The driver reaches the worker by the node ip the launcher advertised, never by loopback."""
        handed: list[Any] = []
        served = _patch_serving(monkeypatch, run=lambda *, sockets: handed.extend(sockets))
        monkeypatch.setattr(serve_actor_module.os, "_exit", lambda code: None)
        app = object()

        serve_until_stopped(app=app, port=12345)

        assert served["created_port"] == 12345
        assert served["app"] is app
        assert handed[0].address == (IPV6_WILDCARD_HOST, 12345)
        assert handed[0].closed is True


class TestFaultInjection:
    # [CLASS DOC] Tests fault injection forwarding.
    def test_the_actor_forwards_an_injected_fault_to_the_process(self, monkeypatch):
        # [FUNCTION DOC] Replaces the module's private inject-fault function with a recorder; calling
        # the actor's inject-fault with mode "hang" results in exactly that mode recorded.
        """Fault injection targets the process the worker runs in, which is now the actor itself."""
        injected: list[str] = []
        monkeypatch.setattr(serve_actor_module, "_inject_fault", lambda *, mode: injected.append(mode))
        actor = ServeActor(build_worker=_build_worker())

        actor.inject_fault("hang")

        assert injected == ["hang"]


class TestTheRealThreadStarts:
    # [CLASS DOC] Tests start-rpc-server with a real thread class.
    def test_start_rpc_server_really_spawns_a_thread(self, monkeypatch):
        # [FUNCTION DOC] Replaces the serve function with a no-op, starts the server, joins the
        # actor's private server thread (10 second timeout) and asserts it is a real thread that has
        # finished.
        """The fake thread of the other tests must not be the only proof that the server ever starts."""
        monkeypatch.setattr(serve_actor_module, "serve_until_stopped", lambda **kwargs: None)
        actor = ServeActor(build_worker=_build_worker())

        actor.start_rpc_server(port=12345)
        actor._server_thread.join(timeout=10.0)

        assert isinstance(actor._server_thread, threading.Thread) and not actor._server_thread.is_alive()
