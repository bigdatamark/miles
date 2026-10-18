# [MODULE DOC]
# Inner process of a served worker, run as a module by the outer launcher with the same own options
# and worker arguments. Builds the worker object from the pool's spec (with a lazily created
# backend capability) and serves it over the http rpc server on the pool's rpc port, blocking
# until the server stops.
from __future__ import annotations

import os
import sys
from typing import Any

import uvicorn

from miles.utils.arguments import parse_args
from miles.utils.function_registry import load_function
from miles.utils.workers.backend_capability.base import BackendCapability, DeferredBackendCapability
from miles.utils.workers.backend_capability.factory import get_backend_capability
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.serving.utils import (
    compute_serve_worker_spec,
    create_server_socket,
    override_argv,
    parse_own_args,
    split_worker_argv,
)
from miles.utils.workers.serving.worker_identity import read_worker_identity, read_worker_in_pod_index
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_spec import RPC_PORT_NAME, PortInfo, ServeWorkerSpec


def main() -> None:
    # [FUNCTION DOC]
    # Split and parse the command line as the outer launcher does, load the pool's served spec,
    # build the worker, find the pool's single rpc port declaration and resolve its effective static
    # port for this worker's index within the pod (read from the environment), then run the uvicorn
    # server for the worker's rpc application on a dual-stack socket bound to that port. Blocks
    # until the server ends. Logs to standard output.
    own_argv, worker_argv = split_worker_argv(sys.argv[1:])
    args = parse_own_args(own_argv)
    _log(f"start own_argv={own_argv} worker_argv={worker_argv}")

    spec = compute_serve_worker_spec(specs_fn=args.specs, pool_id=args.pool_id, worker_argv=worker_argv)
    worker = create_worker(spec, specs_fn=args.specs, worker_argv=worker_argv)
    _log(f"pool_id={args.pool_id} worker_class={spec.worker_class}")

    port = _rpc_port_of(spec).effective_static_port(worker_in_pod_index=read_worker_in_pod_index(os.environ))
    app = create_rpc_app(worker)
    with create_server_socket(port=port) as server_socket:
        _log(f"serve address={server_socket.getsockname()}")
        uvicorn.Server(uvicorn.Config(app)).run(sockets=[server_socket])


def create_worker(spec: ServeWorkerSpec, *, specs_fn: str, worker_argv: list[str]) -> Any:
    # [FUNCTION DOC]
    # Construct the worker for a served spec: read the worker identity from the process
    # environment, create a deferred backend capability (built only if the worker uses it), load
    # the worker class by the spec's class path and instantiate it with the constructor keyword
    # arguments that the spec computes from the identity's constructor context. Returns the worker.
    identity = read_worker_identity(scheduling=spec.scheduling, environ=os.environ)
    _log(f"identity={identity}")
    capability = DeferredBackendCapability(create=lambda: _backend_capability(specs_fn, worker_argv))
    return load_function(spec.worker_class)(**spec.ctor_kwargs(identity.ctor_context(capability=capability)))


def _backend_capability(specs_fn: str, worker_argv: list[str]) -> BackendCapability:
    # [FUNCTION DOC] HELPER: With the process arguments temporarily replaced by the worker
    # arguments, parse the full run arguments to learn the cluster backend; then load the spec table
    # function, compute all specs from the worker arguments and build the backend capability for
    # that backend and those specs.
    with override_argv(worker_argv):
        cluster_backend = ClusterBackend(parse_args().cluster_backend)
    return get_backend_capability(specs=load_function(specs_fn)(worker_argv), cluster_backend=cluster_backend)


def _rpc_port_of(spec: ServeWorkerSpec) -> PortInfo:
    # [FUNCTION DOC] HELPER: Return the spec's port declaration named as the rpc port; asserts there
    # is exactly one.
    ports = [port_info for port_info in spec.port_infos if port_info.name == RPC_PORT_NAME]
    assert len(ports) == 1, f"spec '{spec.name}' declares {len(ports)} rpc ports, so this process cannot pick one"
    return ports[0]


def _log(message: str) -> None:
    # [FUNCTION DOC] HELPER: Print the message prefixed with the inner tag to standard output,
    # flushing immediately.
    print(f"[serve_inner] {message}", flush=True)


if __name__ == "__main__":
    main()
