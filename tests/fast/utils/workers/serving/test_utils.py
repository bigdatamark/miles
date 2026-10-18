# [MODULE DOC] Unit tests for the serving utilities: the temporary environment override context
# manager, argv splitting at a trailing separator, selection of exactly one serve worker spec for a
# pool, the temporary argv override context manager, and creation of the listening server socket
# (dual-stack ipv6 accepting both ipv4 and ipv6 connections, or ipv4 only when dual-stack is
# unavailable). Socket tests bind to port zero on the local host and make real loopback connections.
import os
import socket
import sys
from typing import Any

import pytest

from miles.utils.function_registry import function_registry
from miles.utils.workers.serving import utils as serving_utils
from miles.utils.workers.serving.utils import compute_serve_worker_spec, override_argv, override_env, split_worker_argv
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

SPECS_FN = "test:serving-utils-specs"
POOL_ID = "trainer"


def _base_spec_fields() -> dict[str, object]:
    # [FUNCTION DOC] HELPER: Returns the fields shared by both spec kinds as a mapping: name (the pool
    # id), one rpc port with static port 8000, an empty environment function, and single worker
    # scheduling with no gpus.
    return {
        "name": POOL_ID,
        "port_infos": [PortInfo(name="rpc", static_port=8000)],
        "env_var": lambda context: {},
        "scheduling": SchedulingSpec.single(num_gpus_per_worker=0),
    }


def _serve_spec() -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a serve spec from the shared fields plus the worker class
    # "test.worker" and empty constructor kwargs.
    return ServeWorkerSpec(
        **_base_spec_fields(),
        worker_class="test.worker",
        ctor_kwargs=lambda context: {},
    )


class TestOverrideEnv:
    # [CLASS DOC] Tests the temporary environment override context manager.
    def test_override_env_restores_existing_and_absent_keys_after_an_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Within the override, an existing key holds the new value and an absent key is
        # introduced; an exception raised inside propagates. Afterwards the existing key has its
        # original value and the introduced key is gone from the environment.
        """Environment overrides restore old values and remove new keys when the context raises."""
        existing_key = "MILES_TEST_OVERRIDE_ENV_EXISTING"
        absent_key = "MILES_TEST_OVERRIDE_ENV_ABSENT"
        monkeypatch.setenv(existing_key, "original")
        monkeypatch.delenv(absent_key, raising=False)

        with pytest.raises(RuntimeError, match="inside context"):
            with override_env({existing_key: "overridden", absent_key: "introduced"}):
                assert os.environ[existing_key] == "overridden"
                assert os.environ[absent_key] == "introduced"
                raise RuntimeError("inside context")

        assert os.environ[existing_key] == "original"
        assert absent_key not in os.environ


class TestSplitWorkerArgv:
    # [CLASS DOC] Tests argv splitting for a trailing separator.
    @pytest.mark.parametrize("argv", [["--"], ["--host", "127.0.0.1", "--"]])
    def test_trailing_separator_yields_empty_worker_argv(self, argv: list[str]) -> None:
        # [FUNCTION DOC] Parametrized over a lone separator and options followed by a separator; the
        # entrypoint part is everything before the separator and the worker part is empty.
        """A separator with nothing after it is accepted and leaves the worker argv empty."""
        own_argv, worker_argv = split_worker_argv(argv)

        assert own_argv == argv[:-1]
        assert worker_argv == []


class TestComputeServeWorkerSpec:
    # [CLASS DOC] Tests the selection of the serve spec for a pool from a registered spec table.
    @pytest.mark.parametrize(
        "specs, error_match",
        [
            ([_serve_spec(), _serve_spec()], "not one spec named"),
            (
                [CommandWorkerSpec(**_base_spec_fields(), launch_command=lambda context: "true")],
                "CommandWorkerSpec, which is not served",
            ),
        ],
    )
    def test_a_pool_must_match_exactly_one_serve_worker_spec(
        self, specs: list[ServeWorkerSpec | CommandWorkerSpec], error_match: str
    ) -> None:
        # [FUNCTION DOC] Parametrized over two specs with the same name (ambiguous) and one command
        # spec with the pool's name (not served). Registers a spec table returning the list, and
        # asserts computing the serve spec raises an assertion error matching the expected text.
        """A pool is rejected when its name is ambiguous or belongs to a non-served spec."""

        def compute_specs(worker_argv: list[str]) -> list[ServeWorkerSpec | CommandWorkerSpec]:
            # [FUNCTION DOC] HELPER: Returns the parametrized spec list regardless of argv.
            return specs

        with function_registry.temporary(SPECS_FN, compute_specs):
            with pytest.raises(AssertionError, match=error_match):
                compute_serve_worker_spec(specs_fn=SPECS_FN, pool_id=POOL_ID, worker_argv=[])


class TestOverrideArgv:
    # [CLASS DOC] Tests the temporary process argv override context manager.
    def test_override_argv_restores_the_original_after_an_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Within the override the process argv is the program name (first original
        # item) followed by the replacement items; after an exception the original list object itself
        # (identity) is back.
        """An exceptional context exit restores the exact argv object that preceded the override."""
        original_argv = ["runner.py", "--original"]
        monkeypatch.setattr(sys, "argv", original_argv)

        with pytest.raises(RuntimeError, match="boom"):
            with override_argv(["--replacement"]):
                assert sys.argv == ["runner.py", "--replacement"]
                raise RuntimeError("boom")

        assert sys.argv is original_argv


class TestCreateServerSocket:
    # [CLASS DOC] Tests the listening socket factory using real sockets on port zero.
    def test_dual_stack_listener_accepts_ipv4_and_ipv6_when_ipv6_sockets_default_to_v6_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Replaces the socket constructor so every ipv6 socket starts ipv6-only, and
        # reports dual-stack as available. The listener must be an ipv6 socket with the ipv6-only
        # option cleared, and accept real connections to 127.0.0.1 and to ::1.
        """An explicitly dual-stack listener accepts both address families despite a v6-only default."""
        real_socket = socket.socket

        def create_v6_only_socket(*args: Any, **kwargs: Any) -> socket.socket:
            # [FUNCTION DOC] HELPER: Creates a real socket; if it is ipv6, sets its ipv6-only option
            # to enabled. Returns the socket.
            server_socket = real_socket(*args, **kwargs)
            if server_socket.family == socket.AF_INET6:
                server_socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            return server_socket

        monkeypatch.setattr(serving_utils.socket, "socket", create_v6_only_socket)
        monkeypatch.setattr(serving_utils.socket, "has_dualstack_ipv6", lambda: True)

        with serving_utils.create_server_socket(port=0) as server_socket:
            port = server_socket.getsockname()[1]

            assert server_socket.family == socket.AF_INET6
            assert server_socket.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 0
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
            with socket.create_connection(("::1", port), timeout=1):
                pass

    def test_ipv4_listener_remains_reachable_without_dual_stack_support(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With dual-stack reported unavailable, the listener is an ipv4 socket that
        # accepts a real connection to 127.0.0.1.
        """A host without dual-stack IPv6 still accepts worker RPC calls over IPv4."""
        monkeypatch.setattr(serving_utils.socket, "has_dualstack_ipv6", lambda: False)

        with serving_utils.create_server_socket(port=0) as server_socket:
            port = server_socket.getsockname()[1]

            assert server_socket.family == socket.AF_INET
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
