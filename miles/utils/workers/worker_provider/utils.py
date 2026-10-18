# [MODULE DOC] HELPER:
# Helpers shared by worker providers: applying one observation of a cell to a registry through add
# and remove callbacks, and building rpc worker handles from a worker class and endpoints.
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from miles.utils.function_registry import load_function
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_spec import RPC_PORT_NAME, NamedHostAndPorts

if TYPE_CHECKING:
    from miles.utils.workers.worker_provider.base import CellInfo


async def apply_cell_observation(
    *,
    cell_id: str,
    observed: CellInfo | None,
    actual_workers_hash: str | None,
    add: Callable[[str, CellInfo], Awaitable[None]],
    remove: Callable[[str], Awaitable[None]],
) -> None:
    # [FUNCTION DOC]
    # Bring a registry in line with an observation of one cell. Compares the observed cell info (none
    # if gone) with the workers hash currently registered for the cell (none if not registered):
    # observed and unregistered means add it; gone but registered means remove it; both present with
    # different hashes means remove then add (replacement); otherwise do nothing. The callbacks are
    # awaited in that order and their errors propagate.
    if observed is not None and actual_workers_hash is None:
        await add(cell_id, observed)
    elif observed is None and actual_workers_hash is not None:
        await remove(cell_id)
    elif observed is not None and actual_workers_hash is not None and actual_workers_hash != observed.workers_hash:
        await remove(cell_id)
        await add(cell_id, observed)


def build_rpc_handle(*, worker_class: type, addrs: NamedHostAndPorts) -> BaseWorkerHandle:
    # [FUNCTION DOC] Build an rpc handle for the worker class against the endpoint named as the rpc
    # port (asserts it exists), with the stable boot identity requirement enabled so a restarted
    # server is detected.
    assert (
        RPC_PORT_NAME in addrs
    ), f"a worker addressed by {sorted(addrs)} has no {RPC_PORT_NAME!r} port to be called through"
    return RpcWorkerHandle(worker_class, server_url=addrs[RPC_PORT_NAME].addr, require_stable_boot_uuid=True)


def build_rpc_handle_of_worker_info(info: WorkerInfo) -> BaseWorkerHandle:
    # [FUNCTION DOC] Build an rpc handle for a worker info: asserts the info carries a worker class
    # path, loads the class by that path and builds the handle from the info's endpoints.
    assert info.worker_class is not None, f"{info.name} is not served, so its rpc methods are unknown"
    return build_rpc_handle(worker_class=load_function(info.worker_class), addrs=info.self_addrs)
