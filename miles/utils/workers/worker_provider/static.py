# [MODULE DOC]
# Worker provider for pools whose workers sit at fixed, known addresses (no discovery): either
# computed from a helm release's naming rules or given explicitly as rpc addresses. It can answer
# endpoints and rpc handles but cannot enumerate workers or watch cells. Also holds helpers to wait
# for such addresses to accept connections and to parse host and port text.
from __future__ import annotations

import asyncio
from collections.abc import Iterable

from miles.utils.function_registry import load_function
from miles.utils.http_utils import wait_tcp_ready_async, wrap_ipv6
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm import naming
from miles.utils.workers.worker_provider.utils import build_rpc_handle
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    BaseWorkerSpec,
    HostAndPort,
    NamedHostAndPorts,
    ServeWorkerSpec,
)

_STATIC_ADDRS_READY_TIMEOUT_SECONDS = 600.0


class StaticWorkerProvider(BaseWorkerProvider):
    # [CLASS DOC]
    # Provider for one pool with a fixed mapping from worker name to named endpoints. The optional
    # worker class path says the workers are served over rpc (needed to build handles).
    def __init__(
        self, *, pool_id: str, addrs_by_worker: dict[str, NamedHostAndPorts], worker_class: str | None
    ) -> None:
        # [FUNCTION DOC] Store the pool id, the endpoints per worker name and the worker class path.
        # Asserts there is at least one worker.
        assert addrs_by_worker, f"pool {pool_id} is addressed statically, so it needs at least one cell"
        self._pool_id = pool_id
        self._addrs_by_worker = addrs_by_worker
        self._worker_class = worker_class

    @classmethod
    def of_release(cls, *, release: str, spec: BaseWorkerSpec) -> StaticWorkerProvider:
        # [FUNCTION DOC] Class-level constructor for a pool deployed by a helm release. Asserts that
        # a cell fits in one pod. Computes, for every worker of every cell, the endpoints from the
        # chart's host naming for the release, pool name and cell index, using the worker's index (which
        # equals its pod-local index since a cell is one pod). The worker class is taken from served
        # specs only.
        scheduling = spec.scheduling
        assert scheduling.pods_per_cell() == 1, (
            f"pool {spec.name} spreads a cell over {scheduling.pods_per_cell()} pods, "
            f"so its workers do not share one host"
        )
        return cls(
            pool_id=spec.name,
            addrs_by_worker={
                compute_worker_name(
                    pool_id=spec.name, cell_index=cell_index, worker_in_cell_index=worker_in_cell_index
                ): naming.static_cell_addrs(
                    spec=spec,
                    release=release,
                    cell_index=cell_index,
                    worker_in_pod_index=worker_in_cell_index,
                )
                for cell_index in range(scheduling.num_cells)
                for worker_in_cell_index in range(scheduling.num_workers_per_cell)
            },
            worker_class=spec.worker_class if isinstance(spec, ServeWorkerSpec) else None,
        )

    @classmethod
    def of_rpc_addrs(cls, *, pool_id: str, addrs: list[HostAndPort], worker_class: str) -> StaticWorkerProvider:
        # [FUNCTION DOC] Class-level constructor from explicit rpc endpoints, one per cell in order:
        # cell i has a single worker (index zero) whose only named endpoint is the rpc port at the
        # i-th address.
        return cls(
            pool_id=pool_id,
            addrs_by_worker={
                compute_worker_name(pool_id=pool_id, cell_index=cell_index): {RPC_PORT_NAME: addr}
                for cell_index, addr in enumerate(addrs)
            },
            worker_class=worker_class,
        )

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] The fixed endpoints of the worker; asserts the worker belongs to this pool.
        return self._addrs_of(worker_name)

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        # [FUNCTION DOC] Not supported: always raises a not-implemented error.
        raise NotImplementedError(f"{type(self).__name__} answers addresses, it does not enumerate workers")

    def get_handle(self, worker_name: str) -> BaseWorkerHandle:
        # [FUNCTION DOC] Build an rpc handle for the worker from the worker class and the worker's
        # rpc endpoint (with stable boot identity required). Asserts the pool is served (has a worker
        # class), the worker belongs to the pool and has an rpc port.
        assert (
            self._worker_class is not None
        ), f"pool {self._pool_id} is launched as a command rather than served, so its rpc methods are unknown"
        return build_rpc_handle(worker_class=load_function(self._worker_class), addrs=self._addrs_of(worker_name))

    def _addrs_of(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] HELPER: Look up the worker's endpoints; asserts it is known, listing the
        # known worker names in the message.
        addrs = self._addrs_by_worker.get(worker_name)
        assert addrs is not None, (
            f"{worker_name} is not a worker of pool {self._pool_id}, which addresses "
            f"{sorted(self._addrs_by_worker)} statically"
        )
        return addrs


async def wait_static_addrs_ready(addrs: Iterable[HostAndPort]) -> None:
    # [FUNCTION DOC] Concurrently wait until each endpoint accepts tcp connections, each with a
    # timeout of 600 seconds; the failure behavior on timeout is that of the tcp wait helper.
    await asyncio.gather(
        *[wait_tcp_ready_async(addr.host, addr.port, timeout=_STATIC_ADDRS_READY_TIMEOUT_SECONDS) for addr in addrs]
    )


def parse_host_and_port(addr: str) -> HostAndPort:
    # [FUNCTION DOC] Parse host-colon-port text by splitting at the last colon. Asserts: a colon and an
    # all-digit port are present, no slash appears anywhere (so no scheme or path), the host is not
    # empty, and a host containing colons is a bracketed ipv6 literal (bare ipv6 is rejected as
    # ambiguous). Returns the endpoint with the host passed through the ipv6 bracketing helper.
    host, separator, port = addr.rpartition(":")
    assert separator and port.isdigit() and "/" not in addr, f"static address {addr!r} must be host:port"
    assert host, f"static address {addr!r} names a port but no host to dial it on"
    assert ":" not in host or (host.startswith("[") and host.endswith("]")), (
        f"static address {addr!r} reads as a bare ipv6 address, whose own colons cannot be told from the port "
        f"separator; write it as [{host}]:<port>"
    )
    return HostAndPort(host=wrap_ipv6(host), port=int(port))
