# [MODULE DOC]
# Parses raw Kubernetes pods into typed views carrying the cell-related facts encoded in their
# labels and annotations: pool id, cell index, index of the pod within the cell, expected pod count of
# the cell, metadata, gpu ids, readiness and deletion state. The names of the labels and
# annotations are configurable through a key set, since they depend on the chart.
from __future__ import annotations

import hashlib

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.k8s_types import Pod, PodStatus
from miles.utils.workers.naming import compute_cell_id


class CellLabelKeys(FrozenStrictBaseModel):
    # [CLASS DOC]
    # Names of the labels and annotations the parser reads: labels for pool id, cell index and pod
    # index in the cell; annotation for the cell's pod count; the prefix of annotations that form
    # pod metadata; the metadata key holding the comma-separated gpu ids; and the annotation holding
    # the base gpu id that offsets those ids.
    pool_id: str
    cell_index: str
    pod_in_cell_index: str
    cell_size_annotation: str
    meta_annotation_prefix: str
    gpu_ids_meta: str
    base_gpu_id_annotation: str


class ParsedPod(FrozenStrictBaseModel):
    # [CLASS DOC]
    # Immutable view of a pod that belongs to a cell: pod name, cell id, cell index, pool id, pod index
    # in the cell, readiness, whether it is being deleted, ip (if assigned), uid, total container
    # restart count, metadata mapping, expected number of pods in the cell (zero if unannotated), the
    # headless service subdomain and the pod's gpu ids (already offset by the base gpu id).
    name: str
    cell_id: str
    cell_index: int
    pool_id: str
    pod_in_cell_index: int
    ready: bool
    deleting: bool
    pod_ip: str | None
    uid: str
    restart_count: int
    meta: dict[str, str]
    cell_size: int
    subdomain: str | None
    gpu_ids: tuple[int, ...]


def parse_pods(pods: list[Pod], *, keys: CellLabelKeys) -> list[ParsedPod]:
    # [FUNCTION DOC] Parse the pods, dropping those that do not belong to any cell, and return the
    # views sorted by pod index in the cell.
    parsed_pods = [parsed for pod in pods if (parsed := parse_pod(pod, keys)) is not None]
    return sorted_by_pod_in_cell_index(parsed_pods)


def parse_pod(pod: Pod, keys: CellLabelKeys) -> ParsedPod | None:
    # [FUNCTION DOC]
    # Parse one pod; none if it lacks the pool id label or the cell index label. Cell index and pod
    # index are read from labels as integers (pod index defaults to zero; non-integer text raises a
    # value error). Ready means the Ready condition has status True. Restart count sums over all
    # containers. Cell size comes from the annotation (default zero). Metadata is every annotation
    # with the metadata prefix, keyed by the remainder. Gpu ids are the comma-separated integers in
    # the gpu ids metadata entry, each added to the base gpu id annotation (default zero).
    metadata = pod.metadata
    labels = metadata.labels
    pool_id = labels.get(keys.pool_id)
    if pool_id is None or keys.cell_index not in labels:
        return None

    status = pod.status
    meta = _read_meta(pod, keys)
    cell_index = int(labels[keys.cell_index])
    return ParsedPod(
        name=metadata.name,
        cell_id=compute_cell_id(pool_id=pool_id, cell_index=cell_index),
        cell_index=cell_index,
        pool_id=pool_id,
        pod_in_cell_index=int(labels.get(keys.pod_in_cell_index, 0)),
        ready=_is_ready(status),
        deleting=metadata.deletion_timestamp is not None,
        pod_ip=status.pod_ip,
        uid=metadata.uid,
        restart_count=sum(container.restart_count for container in status.container_statuses),
        meta=meta,
        cell_size=int(metadata.annotations.get(keys.cell_size_annotation, 0)),
        subdomain=pod.spec.subdomain,
        gpu_ids=_parse_gpu_ids(meta.get(keys.gpu_ids_meta, ""), base=_base_gpu_id(pod, keys)),
    )


def cell_members_hash(pods: list[ParsedPod]) -> str:
    # [FUNCTION DOC] Short fingerprint (first 16 hexadecimal characters of a sha-256 digest) of the
    # pods' names, uids, restart counts and deletion flags, taken in pod index order. It changes when
    # a pod is replaced, restarted or starts terminating.
    parts = [f"{pod.name}:{pod.uid}:{pod.restart_count}:{pod.deleting}" for pod in sorted_by_pod_in_cell_index(pods)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def sorted_by_pod_in_cell_index(pods: list[ParsedPod]) -> list[ParsedPod]:
    # [FUNCTION DOC] A new list of the pods ordered by index in the cell (stable).
    return sorted(pods, key=lambda pod: pod.pod_in_cell_index)


def _read_meta(pod: Pod, keys: CellLabelKeys) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Annotations whose key starts with the metadata prefix, with the prefix
    # removed from the key.
    annotations = pod.metadata.annotations
    prefix = keys.meta_annotation_prefix
    return {key[len(prefix) :]: value for key, value in annotations.items() if key.startswith(prefix)}


def _parse_gpu_ids(value: str, *, base: int) -> tuple[int, ...]:
    # [FUNCTION DOC] HELPER: Split the comma-separated text, skip empty parts, convert each to an
    # integer and add the base. Empty text gives an empty tuple.
    return tuple(base + int(part) for part in value.split(",") if part)


def _base_gpu_id(pod: Pod, keys: CellLabelKeys) -> int:
    # [FUNCTION DOC] HELPER: The base gpu id annotation as an integer; zero if absent or empty.
    return int(pod.metadata.annotations.get(keys.base_gpu_id_annotation) or 0)


def _is_ready(status: PodStatus) -> bool:
    # [FUNCTION DOC] HELPER: True iff some condition is of type Ready with status True.
    return any(condition.type == "Ready" and condition.status == "True" for condition in status.conditions)
