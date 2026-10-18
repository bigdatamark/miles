# [MODULE DOC]
# Interface of worker providers and the cell info model. A provider answers worker endpoints and
# worker infos by cell, can optionally watch cells for changes, and builds handles for reaching
# workers (rpc handles for workers that declare a worker class).
import abc
from collections.abc import Awaitable, Callable
from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.naming import cell_id_of_worker
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.utils import build_rpc_handle_of_worker_info
from miles.utils.workers.worker_spec import NamedHostAndPorts


class CellInfo(FrozenStrictBaseModel):
    # [CLASS DOC]
    # Immutable snapshot of one cell: its id, the id of its pool, whether it is alive (all parts
    # present and healthy), the names of its workers, a hash that changes when its membership or
    # incarnation changes, and a free-form metadata mapping. Equality is by value, which
    # change-detection relies on.
    cell_id: str
    pool_id: str
    alive: bool
    worker_names: list[str]
    workers_hash: str
    meta: dict[str, Any]  # TODO: in k8s native mode, may be provided from pod annotations


# args: (cell_id, CellInfo)
CellReconcileFn = Callable[[str, CellInfo | None], Awaitable[None]]
StopWatchFn = Callable[[], Awaitable[None]]


class BaseWorkerProvider(abc.ABC):
    # [CLASS DOC]
    # Base class of providers. Required: endpoint lookup and worker info lookup. Optional: async
    # initialization, cell watching and an expected cell count; handles are built from worker infos.
    async def init(self) -> None:
        # [FUNCTION DOC] Optional one-time asynchronous setup before use; does nothing by default.
        return None

    # [FUNCTION DOC] (get_addrs, abstract, block sits above the decorator) Return the named
    # endpoints of the worker with the given name.
    @abc.abstractmethod
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts: ...

    # [FUNCTION DOC] (get_worker_infos, abstract, block sits above the decorator) For each given
    # cell id, in order, return the list of infos of that cell's workers.
    @abc.abstractmethod
    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]: ...

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        # [FUNCTION DOC] Start watching cells: the reconcile callback is called with a cell id and
        # its current info (none when the cell is gone), first for all existing cells and then on
        # each change. Returns an awaitable stop function. Default: raises not-implemented.
        raise NotImplementedError(f"{type(self).__name__} answers addresses, it does not observe cells")

    def expected_num_cells(self, *, group_id: str) -> int | None:
        # [FUNCTION DOC] Number of cells expected to exist for a pool, or none if unknown. Default:
        # none.
        # KNOWN ISSUE: the parameter is named as a group id, a banned term in the naming rules
        # (should be a pool id), and it is unused here.
        return None

    def get_handle(self, worker_name: str) -> BaseWorkerHandle:
        # [FUNCTION DOC] Build a handle for one worker: fetch the worker infos of its cell (derived
        # from the worker name), build handles for all workers that can have one, and return the
        # named worker's handle. Asserts the worker has a handle (it is served over rpc).
        (infos,) = self.get_worker_infos(cell_ids=[cell_id_of_worker(worker_name)])
        handles = self.get_handles_of_worker_infos(infos)
        assert worker_name in handles, f"{worker_name=} is not one of {sorted(handles)}"
        return handles[worker_name]

    def get_handles_of_worker_infos(self, infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
        # [FUNCTION DOC] Map worker name to handle for those infos that yield a handle; workers
        # without one are left out.
        return {info.name: handle for info in infos if (handle := self._build_handle_of_worker_info(info)) is not None}

    def _build_handle_of_worker_info(self, info: WorkerInfo) -> BaseWorkerHandle | None:
        # [FUNCTION DOC] HELPER: Hook for subclasses. Default: an rpc handle if the info names a
        # worker class, otherwise none.
        return build_rpc_handle_of_worker_info(info) if info.worker_class is not None else None
