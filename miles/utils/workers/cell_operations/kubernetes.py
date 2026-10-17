# [MODULE DOC]
# Kubernetes implementation of cell operations. Cell information comes from a watch-based
# provider; suspending a cell deletes its pods through the Kubernetes API; faults are injected by
# sending a remote call to the worker process asking it to crash itself. Resuming is not supported
# because the owning workload recreates deleted pods on its own.
from __future__ import annotations

import asyncio
import logging

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError
from miles.utils.workers.worker_provider.base import CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesWorkerProvider

logger = logging.getLogger(__name__)

INJECT_FAULT_TIMEOUT_SECONDS = 60.0


class KubernetesCellOperations(BaseCellOperations):
    # [CLASS DOC]
    # Cell operations backed by a Kubernetes worker provider and a namespace. The provider's cell
    # watch is started lazily on the first operation and then shared by all later operations; if
    # the watch task ever fails, it is forgotten so that the next operation starts a fresh one.
    def __init__(self, *, provider: KubernetesWorkerProvider, namespace: str) -> None:
        # [FUNCTION DOC]
        # Store the provider and namespace; no watch is started yet.
        self._provider = provider
        self._namespace = namespace
        self._watching: asyncio.Task[StopWatchFn] | None = None

    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        # [FUNCTION DOC]
        # Wait until the cell watch has delivered its initial state, then return a mapping from cell
        # id to cell info, keeping only cells whose info is available and whose pool id is in the
        # requested pools. Reads the provider's in-memory view; no cluster call besides the watch.
        await self._ensure_watching()

        wanted = set(pool_ids)
        infos = (self._provider.cell_info(cell_id) for cell_id in self._provider.cell_ids())
        return {info.cell_id: info for info in infos if info is not None and info.pool_id in wanted}

    async def suspend(self, *, cell_id: str) -> None:
        # [FUNCTION DOC]
        # Ensure the watch is running, look up the pod names of the cell, assert there is at least
        # one (assertion error otherwise), then delete all those pods concurrently through the
        # Kubernetes API. The workload controller later recreates them. Side effect: pod deletion.
        await self._ensure_watching()

        pods = self._provider.pod_names_of_cell(cell_id)
        assert pods, f"cannot suspend {cell_id}, which has no pods"
        await _delete_pods(namespace=self._namespace, pod_names=pods)

    async def resume(self, *, cell_id: str) -> None:
        # [FUNCTION DOC]
        # Not supported: always raises a not-implemented error, because a suspended (deleted) cell
        # is recreated by its workload without any moment at which resume could return.
        raise NotImplementedError(
            "a deleted cell comes back when its workload recreates it, so resume has no moment to return at"
        )

    async def inject_fault(self, *, cell_id: str, mode: FailureMode, sub_index: int) -> None:
        # [FUNCTION DOC]
        # Ensure the watch is running, fetch the worker infos of the single cell, assert the
        # sub-index is within range, then find the worker at that index and its remote-call handle
        # (assert that the worker is served over remote calls). Finally ask that worker to inject
        # the fault via the rpc helper. Assertion errors report the offending index or worker name.
        await self._ensure_watching()

        (infos,) = self._provider.get_worker_infos(cell_ids=[cell_id])
        assert (
            0 <= sub_index < len(infos)
        ), f"sub_index {sub_index} is out of range for cell {cell_id}, which has {len(infos)} workers"

        worker_name = infos[sub_index].name
        handles = self._provider.get_handles_of_worker_infos(infos)
        assert (
            worker_name in handles
        ), f"{worker_name} is not served over rpc, so no call can reach the process to crash it"

        await _inject_fault_over_rpc(handle=handles[worker_name], mode=mode, worker_name=worker_name)

    async def _ensure_watching(self) -> None:
        # [FUNCTION DOC] HELPER: Lazily start the provider's cell watch as a background task
        # (with a callback that ignores every cell event), then await its first completion so
        # callers see a populated view. If awaiting raises anything (including cancellation), the
        # stored task is cleared so a later call restarts the watch, and the error is re-raised.
        if self._watching is None:
            self._watching = asyncio.ensure_future(self._provider.watch_cells(_ignore_cell))
        try:
            await self._watching
        except BaseException:
            self._watching = None
            raise


async def _inject_fault_over_rpc(*, handle: BaseWorkerHandle, mode: FailureMode, worker_name: str) -> None:
    # [FUNCTION DOC] HELPER: Submit the remote inject-fault call to a worker with the failure mode's
    # string value and wait for it up to the module timeout of 60 seconds. An unreachable worker or
    # a timeout is the expected outcome of a successful crash and is only logged at info level;
    # other errors propagate.
    try:
        await asyncio.wait_for(
            handle.submit_without_result("inject_fault", mode=mode.value), timeout=INJECT_FAULT_TIMEOUT_SECONDS
        )
    except (WorkerUnreachableError, TimeoutError, asyncio.TimeoutError):
        logger.info("Injecting %s into %s left it unreachable, which is what was asked for", mode.value, worker_name)


async def _ignore_cell(cell_id: str, info: CellInfo | None) -> None:
    # [FUNCTION DOC] HELPER: No-op cell watch callback; ignores the cell id and info and returns
    # nothing. The watch is used only to keep the provider's view current.
    return None


async def _delete_pods(*, namespace: str, pod_names: list[str]) -> None:
    # [FUNCTION DOC] HELPER: Delete the named pods in the namespace. Loads the in-cluster
    # Kubernetes configuration (so it only works inside a pod; raises otherwise), opens an async
    # API client, and issues all delete requests concurrently, failing on the first error. The
    # kubernetes client imports are local, deferring the optional dependency.
    from kubernetes_asyncio import client as kubernetes_client
    from kubernetes_asyncio import config as kubernetes_config

    kubernetes_config.load_incluster_config()
    async with kubernetes_client.ApiClient() as api_client:
        core_v1_api = kubernetes_client.CoreV1Api(api_client)
        await asyncio.gather(
            *(core_v1_api.delete_namespaced_pod(name=pod_name, namespace=namespace) for pod_name in pod_names)
        )
