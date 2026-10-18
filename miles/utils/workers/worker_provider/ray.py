# [MODULE DOC]
# Worker provider for the Ray backend. It answers endpoint, worker info and cell queries by calling
# the Ray worker manager actor, builds handles (Ray actor handles for command-style and ray-served
# workers, rpc handles for rpc-served workers), and watches cells by polling the manager.
from functools import partial

import ray.actor

from miles.utils.workers.polling_reconcile_loop import PollingReconcileLoop
from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.utils import build_rpc_handle_of_worker_info
from miles.utils.workers.worker_spec import NamedHostAndPorts

POLL_INTERVAL_SECONDS = 5.0


class RayWorkerProvider(BaseWorkerProvider):
    # [CLASS DOC]
    # Provider over the Ray worker manager actor handle. The pool ids restrict which pools are watched
    # (required for watching, unused by other queries); the poll interval (default 5 seconds) is the
    # cell watch period.
    def __init__(
        self,
        worker_manager_handle: ray.actor.ActorHandle,
        *,
        pool_ids: list[str] | None = None,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    ):
        # [FUNCTION DOC] Store the manager handle, the optional pool ids and the poll interval.
        self._worker_manager_handle = worker_manager_handle
        self._pool_ids = pool_ids
        self._poll_interval_seconds = poll_interval_seconds

    # TEMPORARY: this layer is not meant to serve a suspend the inference controller drives, deliberately
    # violated until the weight-update fault tolerance work removes the need
    async def stop_cells(self, *, cell_ids: list[str]) -> None:
        # [FUNCTION DOC] Ask the manager actor to stop the given cells and wait for completion.
        # Not part of the base provider interface (temporary exception, see the note above).
        await self._worker_manager_handle.stop_cells.remote(cell_ids)

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        # [FUNCTION DOC] Issue one remote worker-info request per cell id concurrently and wait for
        # all with a blocking call. Returns one list per cell, in order. Remote errors (such as an
        # unknown cell) propagate.
        # KNOWN ISSUE: blocks the calling thread (and an event loop, if called from async code) while
        # waiting for the remote results.
        refs = [self._worker_manager_handle.get_worker_infos.remote(cell_id) for cell_id in cell_ids]
        return ray.get(refs)

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] Ask the manager actor for the worker's endpoints and return them.
        return await self._worker_manager_handle.get_worker_addrs.remote(worker_name)

    def get_handles_of_worker_infos(self, infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
        # [FUNCTION DOC] Handle for every info (unlike the base class, none is skipped). Workers with
        # no worker class (called directly through Ray) get Ray worker handles around the actor
        # handles fetched from the manager, which checks each info's generation and fails if stale;
        # workers with a worker class get rpc handles. The actor handle fetch blocks the caller.
        actor_handles = self._get_actor_handles([info for info in infos if info.worker_class is None])
        return {
            info.name: (
                RayWorkerHandle(actor_handles[info.name])
                if info.worker_class is None
                else build_rpc_handle_of_worker_info(info)
            )
            for info in infos
        }

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        # [FUNCTION DOC] Start polling the manager for alive cells of the configured pools (asserts
        # pool ids were given), reconciling differences; the first poll completes before returning.
        # Returns the awaitable stop function of the polling loop.
        pool_ids = self._watched_pool_ids()
        loop = PollingReconcileLoop(
            list_cells=partial(self._list_alive_cells, pool_ids=pool_ids),
            poll_interval_seconds=self._poll_interval_seconds,
        )
        return await loop.start(reconcile)

    async def _list_alive_cells(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        # [FUNCTION DOC] HELPER: Fetch the cell infos of the pools from the manager and keep only
        # those marked alive, so a stopped or not yet ready cell looks absent to the watcher.
        all_infos = await self._worker_manager_handle.get_cell_infos.remote(pool_ids=pool_ids)
        return {cell_id: info for cell_id, info in all_infos.items() if info.alive}

    def _get_actor_handles(self, infos: list[WorkerInfo]) -> dict[str, ray.actor.ActorHandle]:
        # [FUNCTION DOC] HELPER: Fetch from the manager the actor handle of each info's worker,
        # requiring the info's generation to be current, with one blocking wait for all. Returns a
        # mapping from worker name to actor handle.
        refs = [
            self._worker_manager_handle.get_actor_handle.remote(info.name, expected_generation=info.generation)
            for info in infos
        ]
        return {info.name: handle for info, handle in zip(infos, ray.get(refs), strict=True)}

    def _watched_pool_ids(self) -> list[str]:
        # [FUNCTION DOC] HELPER: The configured pool ids; asserts they were provided.
        assert self._pool_ids is not None, "this provider was built without the pool_ids it is meant to observe"
        return self._pool_ids
