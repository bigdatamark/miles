# [MODULE DOC] Unit tests for the inner serve entrypoint: parsing of its own arguments (specs path
# and pool id required, unknown options rejected with usage exit code 2), the address the worker is
# served on (dual-stack ipv6 wildcard when the platform supports it, otherwise the ipv4 wildcard; the
# spec's rpc port in both cases and the socket closed afterwards), and validation that a spec
# declares exactly one rpc port. The main function is run with all heavy collaborators (spec
# computation, worker creation, app creation, socket creation, uvicorn) replaced by fakes.
import socket
from types import SimpleNamespace
from typing import Any

import pytest

from miles.utils.workers.serving import serve_inner
from miles.utils.workers.serving import utils as serving_utils
from miles.utils.workers.serving.serve_inner import _rpc_port_of, parse_own_args
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

SPECS_PATH = "tests.fast.utils.workers.e2e.e2e_worker.compute_specs"
POOL_ID = "e2e-pool"


def _serve_spec() -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a serve spec named by the pool id with one rpc port (static 8000),
    # an empty environment, single cpu-only worker scheduling, the worker class "test.worker" and empty
    # constructor kwargs.
    return ServeWorkerSpec(
        name=POOL_ID,
        port_infos=[PortInfo(name="rpc", static_port=8000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        worker_class="test.worker",
        ctor_kwargs=lambda context: {},
    )


class _FakeServerSocket:
    # [CLASS DOC] HELPER: Fake server socket reporting a fixed address and recording closure when used
    # as a context manager.
    def __init__(self, address: tuple[str, int]) -> None:
        # [FUNCTION DOC] HELPER: Stores the address; starts not closed.
        self.address = address
        self.closed = False

    def getsockname(self) -> tuple[str, int]:
        # [FUNCTION DOC] HELPER: Returns the stored address.
        return self.address

    def __enter__(self) -> "_FakeServerSocket":
        # [FUNCTION DOC] HELPER: Returns itself.
        return self

    def __exit__(self, *args: Any) -> None:
        # [FUNCTION DOC] HELPER: Marks the socket closed.
        self.closed = True


class TestParseOwnArgs:
    # [CLASS DOC] Tests the inner entrypoint's argument parser.
    def test_the_spec_table_and_the_pool_it_serves_are_read(self) -> None:
        # [FUNCTION DOC] Parsing the two options yields the specs path and pool id.
        """These two are the whole of what the pod needs to find the one spec it is a worker of."""
        args = parse_own_args(["--specs", SPECS_PATH, "--pool-id", POOL_ID])

        assert (args.specs, args.pool_id) == (SPECS_PATH, POOL_ID)

    def test_an_omitted_pool_id_is_a_usage_error(self) -> None:
        # [FUNCTION DOC] Missing pool id makes the parser exit with code 2.
        """A process that does not know which pool it serves would pick a spec at random."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--specs", SPECS_PATH])

        assert exc_info.value.code == 2

    def test_an_omitted_spec_table_is_a_usage_error(self) -> None:
        # [FUNCTION DOC] Missing specs path makes the parser exit with code 2.
        """Without the run's spec table there is nothing to match the pool id against."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--pool-id", POOL_ID])

        assert exc_info.value.code == 2

    def test_unknown_inner_option_is_a_usage_error(self) -> None:
        # [FUNCTION DOC] An undefined option makes the parser exit with code 2.
        """The inner entrypoint rejects an option it does not define instead of ignoring it."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--specs", SPECS_PATH, "--pool-id", POOL_ID, "--unknown-option", "1"])

        assert exc_info.value.code == 2


def _served(monkeypatch: pytest.MonkeyPatch, *, has_dualstack_ipv6: bool) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Runs the inner main function with fakes and returns what it served.
    # Sets the process argv to the specs path and pool id; makes the dual-stack ipv6 capability check
    # return the given flag; replaces spec computation, worker creation (returns a plain object), app
    # creation (returns the text "app"), the in-pod index reader (returns 0) and the rpc port lookup
    # (effective static port 8123). Fake socket creation records host, port and extra keyword arguments
    # and returns a fake socket; fake uvicorn config records the app; fake uvicorn server records the
    # config and its run records the sockets passed. After main returns, asserts a socket was created
    # and adds whether it was closed. Returns the mapping with keys host, port, socket_kwargs, app,
    # config, sockets and socket_closed.
    served: dict[str, Any] = {}
    monkeypatch.setattr(serve_inner.sys, "argv", ["serve_inner", "--specs", SPECS_PATH, "--pool-id", POOL_ID])
    monkeypatch.setattr(serving_utils.socket, "has_dualstack_ipv6", lambda: has_dualstack_ipv6)
    monkeypatch.setattr(serve_inner, "compute_serve_worker_spec", lambda **kwargs: SimpleNamespace(worker_class="w"))
    monkeypatch.setattr(serve_inner, "create_worker", lambda spec, **kwargs: object())
    monkeypatch.setattr(serve_inner, "create_rpc_app", lambda worker: "app")
    monkeypatch.setattr(serve_inner, "read_worker_in_pod_index", lambda environ: 0)
    monkeypatch.setattr(
        serve_inner,
        "_rpc_port_of",
        lambda spec: SimpleNamespace(effective_static_port=lambda worker_in_pod_index: 8123),
    )
    server_socket: _FakeServerSocket | None = None

    def create_server(address: tuple[str, int], **kwargs: Any) -> _FakeServerSocket:
        # [FUNCTION DOC] HELPER: Fake socket creation: records the host, port and extra keyword
        # arguments, remembers and returns a fake socket at the address.
        nonlocal server_socket
        served.update(host=address[0], port=address[1], socket_kwargs=kwargs)
        server_socket = _FakeServerSocket(address)
        return server_socket

    def create_uvicorn_server(config: Any) -> SimpleNamespace:
        # [FUNCTION DOC] HELPER: Fake uvicorn server constructor: records the config and returns an
        # object whose run method records the sockets it is given.
        served["config"] = config
        return SimpleNamespace(run=lambda *, sockets: served.update(sockets=sockets))

    monkeypatch.setattr(serving_utils.socket, "create_server", create_server)
    monkeypatch.setattr(serve_inner.uvicorn, "Config", lambda app: served.update(app=app) or "config")
    monkeypatch.setattr(serve_inner.uvicorn, "Server", create_uvicorn_server)

    serve_inner.main()
    assert server_socket is not None
    served["socket_closed"] = server_socket.closed
    return served


class TestTheAddressAWorkerIsServedOn:
    # [CLASS DOC] Tests which wildcard address and port the served socket is bound to.
    def test_binds_the_dual_stack_wildcard_where_the_platform_offers_one(self, monkeypatch):
        # [FUNCTION DOC] With dual-stack support, the host is the ipv6 wildcard and the socket is
        # created with the ipv6 family and dual stack enabled.
        """The cell view publishes the pod ip, and on an ipv6-only cluster that is an ipv6 address."""
        served = _served(monkeypatch, has_dualstack_ipv6=True)

        assert served["host"] == serving_utils.IPV6_WILDCARD_HOST
        assert served["socket_kwargs"] == {"family": socket.AF_INET6, "dualstack_ipv6": True}

    def test_binds_the_ipv4_wildcard_where_ipv6_is_unavailable(self, monkeypatch):
        # [FUNCTION DOC] Without dual-stack support, the host is the ipv4 wildcard and only the ipv4
        # family is requested.
        """Asking for the dual-stack wildcard where there is no ipv6 stack leaves the worker unserved."""
        served = _served(monkeypatch, has_dualstack_ipv6=False)

        assert served["host"] == serving_utils.IPV4_WILDCARD_HOST
        assert served["socket_kwargs"] == {"family": socket.AF_INET}

    def test_the_dual_stack_wildcard_is_the_unspecified_ipv6_address(self):
        # [FUNCTION DOC] The two wildcard constants equal "::" and "0.0.0.0".
        """Only the unspecified address accepts the ipv4-mapped connections an ipv4 client makes."""
        assert (serving_utils.IPV6_WILDCARD_HOST, serving_utils.IPV4_WILDCARD_HOST) == ("::", "0.0.0.0")

    def test_serves_the_rpc_port_the_spec_declares_whichever_wildcard_it_binds(self, monkeypatch):
        # [FUNCTION DOC] Runs the served helper in both modes; in each, the port is 8123, the socket
        # handed to the server has the matching wildcard host and that port, and the socket was closed.
        """The address a client dials is the published pod ip and this port, so the port may not move."""
        dual_stack = _served(monkeypatch, has_dualstack_ipv6=True)
        ipv4 = _served(monkeypatch, has_dualstack_ipv6=False)

        assert dual_stack["port"] == 8123
        assert dual_stack["sockets"][0].address == (serving_utils.IPV6_WILDCARD_HOST, 8123)
        assert dual_stack["socket_closed"] is True
        assert ipv4["port"] == 8123
        assert ipv4["sockets"][0].address == (serving_utils.IPV4_WILDCARD_HOST, 8123)
        assert ipv4["socket_closed"] is True


class TestRpcPortOf:
    # [CLASS DOC] Tests extraction of the single rpc port declaration from a spec.
    @pytest.mark.parametrize(
        "port_infos, expected_count",
        [
            ([PortInfo(name="metrics", static_port=9000)], 0),
            ([PortInfo(name="rpc", static_port=8000), PortInfo(name="rpc", static_port=8001)], 2),
        ],
    )
    def test_a_spec_without_exactly_one_rpc_port_is_rejected(
        self, port_infos: list[PortInfo], expected_count: int
    ) -> None:
        # [FUNCTION DOC] Parametrized over a spec with no rpc port (only a metrics port) and one with
        # two rpc ports; the lookup raises an assertion error saying the spec declares that many rpc ports.
        """A served spec with a missing or ambiguous rpc port cannot choose a listening port."""
        spec = _serve_spec().model_copy(update={"port_infos": port_infos})

        with pytest.raises(AssertionError, match=rf"declares {expected_count} rpc ports"):
            _rpc_port_of(spec)
