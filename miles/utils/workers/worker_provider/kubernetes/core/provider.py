# [MODULE DOC]
# Worker provider for Kubernetes runs. It watches the pods of the selected pools through a
# reflector feeding a reconcile loop, groups them into cells by label, and answers endpoint, cell
# and worker queries from that in-memory view. Queries other than the watch itself require the watch
# to be running.
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable
from contextlib import AsyncExitStack, asynccontextmanager

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.reconcile.k8s_api import KubernetesAsyncioPodApi
from miles.utils.workers.reconcile.k8s_reflector import KubernetesReflector
from miles.utils.workers.reconcile.loop import ReconcileLoop
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.kubernetes.core import cell_view, pod_view
from miles.utils.workers.worker_provider.kubernetes.core.pod_view import CellLabelKeys
from miles.utils.workers.worker_spec import BaseWorkerSpec, NamedHostAndPorts

logger = logging.getLogger(__name__)


class KubernetesRunInfo(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable description of a deployed run: namespace, base label selector selecting
    # its pods, the label and annotation key names, and the worker specs by pool id.
    namespace: str
    label_selector: str
    label_keys: CellLabelKeys
    specs: dict[str, BaseWorkerSpec]


class KubernetesWorkerProvider(BaseWorkerProvider):
    # [CLASS DOC]
    # Provider restricted to the given pool ids. At most one watch is expected: the reconcile loop is
    # stored on the provider by the watch and used by every query. Unlike the registration hub, calling
    # watch again replaces the stored loop.
    def __init__(self, *, run: KubernetesRunInfo, pool_ids: list[str], resync_period: float | None) -> None:
        # [FUNCTION DOC] Store the run info, the pool ids to serve, and the resync period in seconds
        # (none disables resync) for the reconcile loop. No cluster access yet.
        self._run = run
        self._pool_ids = pool_ids
        self._resync_period = resync_period
        self._loop: ReconcileLoop | None = None

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] Search all observed cells for the worker and return its endpoints. Raises an
        # assertion error if no observed pod serves it, or if the watch is not running.
        for cell_id in self.cell_ids():
            for worker in self._workers_of_cell(cell_id):
                if worker.name == worker_name:
                    return cell_view.addrs_of_worker(worker, run=self._run)
        raise AssertionError(f"no observed pod serves {worker_name}")

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        # [FUNCTION DOC] For each cell id, the worker infos built from its observed pods. Asserts the
        # cell has pods and none are missing; requires the watch to be running.
        return [
            cell_view.compute_worker_infos(cell_id, pods=self._pods_of_cell(cell_id), run=self._run)
            for cell_id in cell_ids
        ]

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        # [FUNCTION DOC]
        # Start watching. Opens the in-cluster Kubernetes pod api, creates a reflector limited to
        # the pods of the served pools in the namespace, and starts a reconcile loop over it that
        # groups pods by cell id and reports each changed cell to the callback. The loop waits for
        # its first full sync before this returns. The callback receives the cell's info only while
        # the cell is alive; a cell that is not alive (not ready, terminating or incomplete) or has no
        # pods is reported as gone. Returns the awaitable stop function that stops the loop and closes
        # the api. If setup fails, everything already opened is closed.
        def notify_cell(cell_id: str) -> Awaitable[None]:
            # [FUNCTION DOC] HELPER: Reconcile function of the loop; computes the cell's info and
            # returns the awaitable callback invocation with it if alive, else with none.
            info = self.cell_info(cell_id)
            return reconcile(cell_id, info if info is not None and info.alive else None)

        async with AsyncExitStack() as watching:
            reflector = KubernetesReflector(
                kube_client=await watching.enter_async_context(_kubernetes_pod_api()),
                namespace=self._run.namespace,
                label_selector=_watched_pods_selector(
                    base_selector=self._run.label_selector,
                    pool_label_key=self._run.label_keys.pool_id,
                    pool_ids=self._pool_ids,
                ),
            )
            self._loop = await watching.enter_async_context(
                ReconcileLoop(
                    source=reflector.watch,
                    reconcile=notify_cell,
                    key_map=self._cell_id_of_pod,
                    resync_period=self._resync_period,
                )
            )
            return watching.pop_all().aclose

    def cell_ids(self) -> list[str]:
        # [FUNCTION DOC] Sorted ids of all cells that currently have at least one observed pod
        # (including cells that are not alive). Requires the watch to be running.
        return sorted(self._loop_or_fail().parent_keys())

    def cell_info(self, cell_id: str) -> CellInfo | None:
        # [FUNCTION DOC] The cell's info computed from its observed pods, whether alive or not; none
        # if it has no pods.
        return cell_view.compute_cell_info(cell_id, pods=self._pods_of_cell(cell_id), run=self._run)

    def pod_names_of_cell(self, cell_id: str) -> list[str]:
        # [FUNCTION DOC] Names of the cell's observed pods in pod index order.
        return [pod.name for pod in self._pods_of_cell(cell_id)]

    def _cell_id_of_pod(self, pod: Pod) -> str | None:
        # [FUNCTION DOC] HELPER: Key mapping of the reconcile loop: the pod's cell id, or none if the
        # pod belongs to no cell or to a pool this provider does not serve (such pods are ignored).
        parsed = pod_view.parse_pod(pod, self._run.label_keys)
        if parsed is None or parsed.pool_id not in self._pool_ids:
            return None
        return parsed.cell_id

    def _workers_of_cell(self, cell_id: str) -> list[cell_view.KubernetesWorkerInfo]:
        # [FUNCTION DOC] HELPER: All workers of the cell's observed pods.
        return cell_view.workers_of_pods(self._pods_of_cell(cell_id), run=self._run)

    def _pods_of_cell(self, cell_id: str) -> list[pod_view.ParsedPod]:
        # [FUNCTION DOC] HELPER: The cell's pods from the loop's store, parsed and sorted by pod index.
        return pod_view.parse_pods(self._loop_or_fail().get_by_parent(cell_id), keys=self._run.label_keys)

    def _loop_or_fail(self) -> ReconcileLoop:
        # [FUNCTION DOC] HELPER: The reconcile loop; asserts the watch was started. The loop is not
        # cleared when the watch is stopped, so queries after stopping read stale data.
        assert self._loop is not None, "watch_cells must be running before the cells can be read"
        return self._loop


@asynccontextmanager
async def _kubernetes_pod_api() -> AsyncIterator[KubernetesAsyncioPodApi]:
    # [FUNCTION DOC] HELPER: Async context manager yielding a pod api. Loads the in-cluster
    # configuration (works only inside a pod), opens an api client for the scope of the context and
    # closes it on exit. Kubernetes client imports are local to defer the optional dependency.
    from kubernetes_asyncio import client as kubernetes_client
    from kubernetes_asyncio import config as kubernetes_config

    kubernetes_config.load_incluster_config()
    async with kubernetes_client.ApiClient() as api_client:
        yield KubernetesAsyncioPodApi(core_v1_api=kubernetes_client.CoreV1Api(api_client))


def _watched_pods_selector(*, base_selector: str, pool_label_key: str, pool_ids: list[str]) -> str:
    # [FUNCTION DOC] HELPER: Label selector text for the watch: the base selector, a comma, and a
    # set-membership requirement on the pool label with the sorted pool ids. With no pool ids it
    # appends a requirement that no pod can satisfy, so nothing is watched.
    if not pool_ids:
        return f"{base_selector},{_NO_POD_CARRIES_THIS_LABEL}"
    wanted = ",".join(sorted(pool_ids))
    return f"{base_selector},{pool_label_key} in ({wanted})"


_NO_POD_CARRIES_THIS_LABEL = "never-matches.invalid/watches-nothing"
