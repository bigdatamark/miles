# [MODULE DOC]
# Turns the parsed pods of a cell into cell infos, worker infos and worker endpoints. Pods of a cell
# are expected sorted by their index in the cell. A pod hosts workers-per-pod workers; worker names
# are derived from pool id, cell index and the worker's index in the cell; endpoints come from the
# pod ip (or a headless service name) and the pool's declared static ports.
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from miles.utils.http_utils import wrap_ipv6
from miles.utils.misc import merge_asserting_consistency
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_provider.kubernetes.core import pod_view
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    HostAndPort,
    NamedHostAndPorts,
    ServeWorkerSpec,
    WorkerMetaContext,
)

if TYPE_CHECKING:
    from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesRunInfo


class KubernetesWorkerInfo(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable view of one worker inside a pod: the pod, the worker name, its index in
    # the pod and its gpu ids (pod-local share).
    pod: pod_view.ParsedPod
    name: str
    worker_in_pod_index: int
    gpu_ids: list[int]


def compute_cell_info(cell_id: str, *, pods: list[pod_view.ParsedPod], run: KubernetesRunInfo) -> CellInfo | None:
    # [FUNCTION DOC]
    # Build the cell info from the cell's pods; none if there are no pods. Pool id comes from the
    # first pod. Metadata is the spec's metadata function applied to the first pod's cell index,
    # merged with the pods' own metadata (the pods' values win on conflicts). The cell is alive iff
    # every pod is ready and not being deleted and all expected pods are present. Worker names are
    # those of all pods' workers in pod order. The workers hash is derived from pod membership.
    if not pods:
        return None

    pool_id = pods[0].pool_id
    meta = _spec_meta_of_pod(pods[0], run=run) | _pod_meta_of_cell(pods)

    return CellInfo(
        cell_id=cell_id,
        pool_id=pool_id,
        alive=all(pod.ready and not pod.deleting for pod in pods) and _has_all_pods(pods),
        worker_names=[worker.name for worker in workers_of_pods(pods, run=run)],
        workers_hash=pod_view.cell_members_hash(pods),
        meta=meta,
    )


def compute_worker_infos(cell_id: str, *, pods: list[pod_view.ParsedPod], run: KubernetesRunInfo) -> list[WorkerInfo]:
    # [FUNCTION DOC] Build the worker infos of a complete cell. Asserts there is at least one pod,
    # that the pods' indices in the cell are exactly zero up to the count minus one in order, and
    # that all expected pods are present. Returns one info per worker, in pod then worker order.
    assert pods, f"cell {cell_id} has no observed worker pods, so it cannot be driven"
    indices = [pod.pod_in_cell_index for pod in pods]
    assert indices == list(range(len(pods))) and _has_all_pods(
        pods
    ), f"cell {cell_id} is missing pods: observed {indices} of {max(pod.cell_size for pod in pods)}"

    return [_compute_worker_info(worker, run=run) for worker in workers_of_pods(pods, run=run)]


def workers_of_pods(pods: list[pod_view.ParsedPod], *, run: KubernetesRunInfo) -> list[KubernetesWorkerInfo]:
    # [FUNCTION DOC] All workers of the pods, concatenated in the given pod order.
    return [worker for pod in pods for worker in _workers_of_pod(pod, run=run)]


def addrs_of_worker(worker: KubernetesWorkerInfo, *, run: KubernetesRunInfo) -> NamedHostAndPorts:
    # [FUNCTION DOC]
    # Endpoints of a worker by port name: the pod's host with each static port declared by the
    # pool's spec; the rpc port is shifted by the worker's index in the pod, other ports are used
    # as declared. Asserts the spec declares at least one port.
    # KNOWN ISSUE: only the rpc port is shifted per worker, while the port declaration's own rule
    # (effective static port) shifts every per-worker port by the worker's index and by the number of
    # consecutive ports; the two rules disagree for non-rpc per-worker ports in multi-worker pods.
    host = _host_of_pod(worker.pod, namespace=run.namespace)
    ports = _ports_of_pool(worker.pod.pool_id, run=run)
    assert ports, f"spec {worker.pod.pool_id} declares no ports, so {worker.name} has no address"
    return {
        name: HostAndPort(host=host, port=port + (worker.worker_in_pod_index if name == RPC_PORT_NAME else 0))
        for name, port in ports.items()
    }


def _compute_worker_info(worker: KubernetesWorkerInfo, *, run: KubernetesRunInfo) -> WorkerInfo:
    # [FUNCTION DOC] HELPER: Build a worker info: name; generation taken from the pod's container
    # restart count; endpoints; a copy of the gpu ids; and the worker class path only for served
    # specs.
    pool_id = worker.pod.pool_id
    spec = run.specs[pool_id]

    return WorkerInfo(
        name=worker.name,
        generation=worker.pod.restart_count,
        self_addrs=addrs_of_worker(worker, run=run),
        gpu_ids=list(worker.gpu_ids),
        worker_class=spec.worker_class if isinstance(spec, ServeWorkerSpec) else None,
    )


def _workers_of_pod(pod: pod_view.ParsedPod, *, run: KubernetesRunInfo) -> list[KubernetesWorkerInfo]:
    # [FUNCTION DOC] HELPER: Split one pod into its workers. The pod's gpu ids (from its annotation)
    # are divided into equal consecutive shares per worker (asserts divisibility). Worker names use
    # the pod's cell index and worker-in-cell index = pod index in cell times workers per pod plus
    # the worker's index in the pod.
    workers_per_pod = run.specs[pod.pool_id].scheduling.workers_per_pod()
    assert len(pod.gpu_ids) % workers_per_pod == 0, (
        f"pod {pod.name} was annotated with {len(pod.gpu_ids)} gpus for the {workers_per_pod} workers it serves, "
        f"so no worker owns an equal share of them"
    )
    gpus_per_worker = len(pod.gpu_ids) // workers_per_pod
    return [
        KubernetesWorkerInfo(
            pod=pod,
            name=compute_worker_name(
                pool_id=pod.pool_id,
                cell_index=pod.cell_index,
                worker_in_cell_index=pod.pod_in_cell_index * workers_per_pod + worker_in_pod_index,
            ),
            worker_in_pod_index=worker_in_pod_index,
            gpu_ids=list(
                pod.gpu_ids[worker_in_pod_index * gpus_per_worker : (worker_in_pod_index + 1) * gpus_per_worker]
            ),
        )
        for worker_in_pod_index in range(workers_per_pod)
    ]


def _ports_of_pool(pool_id: str, *, run: KubernetesRunInfo) -> dict[str, int]:
    # [FUNCTION DOC] HELPER: Mapping from port name to its declared static port for the pool's spec.
    return {port.name: port.static_port for port in run.specs[pool_id].port_infos}


def _host_of_pod(pod: pod_view.ParsedPod, *, namespace: str) -> str:
    # [FUNCTION DOC] HELPER: Host to reach a pod: its ip (ipv6 bracketed) if known; otherwise the
    # in-cluster dns name built from pod name, headless service subdomain and namespace (asserts the
    # subdomain exists).
    if pod.pod_ip:
        return wrap_ipv6(pod.pod_ip)
    assert pod.subdomain, f"worker {pod.name} has neither a pod ip nor a headless service"
    return f"{pod.name}.{pod.subdomain}.{namespace}.svc"


def _has_all_pods(pods: list[pod_view.ParsedPod]) -> bool:
    # [FUNCTION DOC] HELPER: True iff the sorted pod indices are exactly zero up to the cell size
    # minus one, where the cell size is the largest size any pod reports; true when that size is
    # zero (unknown).
    expected = max(pod.cell_size for pod in pods)
    if not expected:
        return True
    return sorted(pod.pod_in_cell_index for pod in pods) == list(range(expected))


def _spec_meta_of_pod(pod: pod_view.ParsedPod, *, run: KubernetesRunInfo) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: The spec's metadata function applied to the pod's cell index, as a
    # fresh mapping; empty if the spec has no such function.
    compute_meta = run.specs[pod.pool_id].meta
    if compute_meta is None:
        return {}
    return dict(compute_meta(WorkerMetaContext(cell_index=pod.cell_index)))


def _pod_meta_of_cell(pods: list[pod_view.ParsedPod]) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Merge the metadata of all pods into one mapping, asserting that pods
    # agree on any key they share.
    return functools.reduce(merge_asserting_consistency, (pod.meta for pod in pods), {})
