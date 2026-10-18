# [MODULE DOC] HELPER:
# Helpers shared by the serving launchers: creating the listening socket, temporarily overriding
# process arguments or environment, splitting and parsing the launcher command line, and finding the
# served spec of a pool.
from __future__ import annotations

import argparse
import os
import socket
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from miles.utils.function_registry import load_function
from miles.utils.workers.worker_spec import ServeWorkerSpec

IPV4_WILDCARD_HOST = "0.0.0.0"
IPV6_WILDCARD_HOST = "::"


def create_server_socket(*, port: int) -> socket.socket:
    # [FUNCTION DOC] Create a listening tcp socket on the port on all interfaces: a dual-stack
    # socket accepting both ipv6 and ipv4 if the platform supports it, else an ipv4-only socket.
    # Raises an operating system error if the port is taken. The caller closes it.
    if socket.has_dualstack_ipv6():
        return socket.create_server(
            (IPV6_WILDCARD_HOST, port),
            family=socket.AF_INET6,
            dualstack_ipv6=True,
        )
    return socket.create_server((IPV4_WILDCARD_HOST, port), family=socket.AF_INET)


@contextmanager
def override_argv(argv: list[str]) -> Iterator[None]:
    # [FUNCTION DOC] Context manager: for its duration replace the process arguments by the program
    # name followed by the given list; restores the original arguments on exit, even on error.
    original_argv = sys.argv
    sys.argv = [original_argv[0], *argv]
    try:
        yield
    finally:
        sys.argv = original_argv


@contextmanager
def override_env(env: dict[str, str]) -> Iterator[None]:
    # [FUNCTION DOC] Context manager: set the given environment variables of the process for its
    # duration; on exit restore each to its earlier value, or remove it if it was unset. Mutates the
    # real process environment (not thread-safe).
    original = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def split_worker_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    # [FUNCTION DOC] Split arguments at the first double dash into the part before (own) and the
    # part after (worker). Without a separator everything is own and the worker part is empty.
    if "--" not in argv:
        return argv, []

    separator_index = argv.index("--")
    return argv[:separator_index], argv[separator_index + 1 :]


def parse_own_args(own_argv: list[str]) -> argparse.Namespace:
    # [FUNCTION DOC] Parse the launcher's own options: a required spec table function path (dotted
    # module and callable) and a required pool id. Bad input exits the process through the argument
    # parser. Returns the namespace.
    parser = argparse.ArgumentParser(description="Serve one pool of a miles run")
    parser.add_argument("--specs", required=True, help="Spec table of the run as 'package.module.callable'")
    parser.add_argument("--pool-id", required=True, help="Which pool of that run this process serves")
    return parser.parse_args(own_argv)


def compute_serve_worker_spec(*, specs_fn: str, pool_id: str, worker_argv: list[str]) -> ServeWorkerSpec:
    # [FUNCTION DOC] Load the spec table function by path, call it with the worker arguments to get
    # all specs of the run, and return the one spec named as the pool id. Asserts exactly one spec
    # has that name and that it is a served worker spec.
    specs = load_function(specs_fn)(worker_argv)
    matched = [spec for spec in specs if spec.name == pool_id]
    assert len(matched) == 1, (
        f"the run described by this pod's argv has {[spec.name for spec in specs]}, not one spec named "
        f"'{pool_id}'; the pod and the launcher disagree about what this run is"
    )

    spec = matched[0]
    assert isinstance(spec, ServeWorkerSpec), f"spec '{pool_id}' is a {type(spec).__name__}, which is not served"
    return spec
