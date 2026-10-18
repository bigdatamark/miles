# [MODULE DOC] Core logic of the colocate pairing controller. At construction every inference pod
# coordinate (pool, cell, pod index) of every configured pool is mapped to the trainer pod it must
# share a node with and the base GPU id on that node. Reconciliation works per trainer pod: once
# the trainer pod has a node, all its gated inference pods are patched (pinned to that node, base
# GPU id annotated, scheduling gate removed). Pods are grouped under the key of their trainer pod;
# unrelated pods get a unique key that makes their reconcile a no-op.
from __future__ import annotations

import asyncio
import logging
from typing import NamedTuple

from kubernetes_asyncio import client

from miles.utils.external_utils.colocate_pairing.config import PairingConfig, PairingLayout
from miles.utils.external_utils.colocate_pairing.pods import (
    PodCoordinate,
    coordinate_of,
    gate_names,
    is_gated,
    release_patch,
)
from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.reconcile.loop import ReconcileLoop

logger = logging.getLogger(__name__)

_UNRELATED_KEY_PREFIX = "__unrelated__/"


class InferencePlacement(NamedTuple):
    # [CLASS DOC] Where an inference pod must go: the coordinate of its trainer pod, and the GPU
    # index on that node at which its GPUs start.
    trainer_coord: PodCoordinate
    base_gpu_id: int


class PairingController:
    # [CLASS DOC] Holds the config, the Kubernetes core API client, precomputed placement tables
    # and (set after construction) the reconcile loop used to look up pods. Precomputed:
    # for each trainer pod coordinate, the list of (inference coordinate, base GPU id) it hosts;
    # and for every inference and trainer coordinate, the key of the trainer pod.
    _loop: ReconcileLoop

    def __init__(self, *, config: PairingConfig, core_v1: client.CoreV1Api) -> None:
        # [FUNCTION DOC] Keyword-only. Computes a placement for every (pool, cell, pod) of every
        # inference pool, groups inference pods by the trainer coordinate they land on, and builds
        # a mapping from every inference and every hosting trainer coordinate to the trainer's
        # key. Trainer pods hosting no inference pod are not in the mapping. Placement assertion
        # errors propagate.
        self._config = config
        self._core_v1 = core_v1
        placement_of_inference = {
            PodCoordinate(
                pool_id=pool.pool_id, cell_index=cell_index, pod_in_cell_index=pod_index
            ): _place_inference_pod(
                inference_cell_index=cell_index,
                inference_pod_index=pod_index,
                layout=pool.layout,
                trainer_pool_id=config.trainer_pool_id,
            )
            for pool in self._config.inference_pools
            for cell_index in range(pool.layout.num_inference_cells)
            for pod_index in range(pool.layout.num_pods_per_inference_cell)
        }
        self._inferences_of_trainer: dict[PodCoordinate, list[tuple[PodCoordinate, int]]] = {}
        for inference, placement in placement_of_inference.items():
            self._inferences_of_trainer.setdefault(placement.trainer_coord, []).append(
                (inference, placement.base_gpu_id)
            )

        self._trainer_key_of_coord = {
            inference: placement.trainer_coord.key for inference, placement in placement_of_inference.items()
        } | {trainer: trainer.key for trainer in self._inferences_of_trainer}

    def set_loop(self, loop: ReconcileLoop) -> None:
        # [FUNCTION DOC] Stores the reconcile loop (needed before any reconcile call, because
        # the loop is created after the controller whose methods it is built from).
        self._loop = loop

    async def reconcile(self, pair_key: str) -> None:
        # [FUNCTION DOC] Reconciles one trainer-pod group. Takes the pods the loop has under the
        # key and indexes those that have a parseable coordinate. Finds the trainer pod (the
        # coordinate whose key equals the pair key); returns if absent. Collects inference pods
        # expected on this trainer that exist and still carry the gate; returns if none. If the
        # trainer pod has no node yet, logs and returns (retry on a later event or resync).
        # Otherwise releases all collected pods concurrently; failures are logged per pod with the
        # stack trace and not raised (the next reconcile retries).
        pods_by_coord = {
            coord: pod for pod in self._loop.get_by_parent(pair_key) if (coord := coordinate_of(pod)) is not None
        }

        trainer_coord = next((coord for coord in pods_by_coord if coord.key == pair_key), None)
        if trainer_coord is None:
            return
        gated_pods_and_base_gpu_ids = [
            (pod, base_gpu_id)
            for inference_coord, base_gpu_id in self._inferences_of_trainer.get(trainer_coord, [])
            if (pod := pods_by_coord.get(inference_coord)) is not None and is_gated(pod)
        ]
        if not gated_pods_and_base_gpu_ids:
            return

        trainer_node_name = pods_by_coord[trainer_coord].spec.node_name
        if not trainer_node_name:
            logger.info(
                "Waiting for %s to be scheduled before releasing %s",
                trainer_coord.key,
                [pod.metadata.name for pod, _ in gated_pods_and_base_gpu_ids],
            )
            return

        outcomes = await asyncio.gather(
            *(
                self._release(
                    inference_pod,
                    node_name=trainer_node_name,
                    base_gpu_id=base_gpu_id,
                    trainer_key=trainer_coord.key,
                )
                for inference_pod, base_gpu_id in gated_pods_and_base_gpu_ids
            ),
            return_exceptions=True,
        )
        for (inference_pod, _), outcome in zip(gated_pods_and_base_gpu_ids, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.error(
                    "Releasing %s failed; the next reconcile of %s tries it again",
                    inference_pod.metadata.name,
                    trainer_coord.key,
                    exc_info=outcome,
                )

    async def _release(self, inference_pod: Pod, *, node_name: str, base_gpu_id: int, trainer_key: str) -> None:
        # [FUNCTION DOC] HELPER: Logs the intended placement, then patches the pod in the
        # configured namespace with the release patch built from the node name, base GPU id, the
        # pod's current gate names and its annotations (see the pod helpers). API errors
        # propagate to the caller.
        logger.info(
            "Releasing %s onto gpu %s of %s, where %s runs",
            inference_pod.metadata.name,
            base_gpu_id,
            node_name,
            trainer_key,
        )
        await self._core_v1.patch_namespaced_pod(
            name=inference_pod.metadata.name,
            namespace=self._config.namespace,
            body=release_patch(
                node_name=node_name,
                base_gpu_id=base_gpu_id,
                gates=gate_names(inference_pod),
                annotations=inference_pod.metadata.annotations,
            ),
        )

    def key_of(self, pod: Pod) -> str:
        # [FUNCTION DOC] The grouping key for a pod: the key of its trainer pod when the pod's
        # coordinate is known to the placement tables; otherwise a unique per-pod key with the
        # prefix "__unrelated__/" plus the pod name, so unrelated pods are reconciled alone (and
        # reconcile ignores them).
        if (coord := coordinate_of(pod)) is not None and (key := self._trainer_key_of_coord.get(coord)) is not None:
            return key
        return f"{_UNRELATED_KEY_PREFIX}{pod.metadata.name}"


def _place_inference_pod(
    *, inference_cell_index: int, inference_pod_index: int, layout: PairingLayout, trainer_pool_id: str
) -> InferencePlacement:
    # [FUNCTION DOC] HELPER: Keyword-only. Asserts the cell and pod indices are inside the
    # layout. The pod's absolute trainer-GPU index = GPU offset + (cell index times pods per cell
    # plus pod index) times GPUs per inference pod. The trainer node index = absolute GPU divided
    # (floor) by GPUs per node; that node index is split into (trainer cell, pod in trainer cell)
    # with the trainer cell's pod count. Returns the trainer coordinate in the trainer pool and the
    # base GPU id = absolute GPU modulo GPUs per node.
    assert 0 <= inference_cell_index < layout.num_inference_cells, f"{inference_cell_index=} outside {layout}"
    assert 0 <= inference_pod_index < layout.num_pods_per_inference_cell, f"{inference_pod_index=} outside {layout}"

    absolute_gpu = (
        layout.gpu_offset
        + (inference_cell_index * layout.num_pods_per_inference_cell + inference_pod_index)
        * layout.num_gpus_per_inference_pod
    )
    trainer_cell_index, trainer_pod_index = divmod(
        absolute_gpu // layout.num_gpus_per_node, layout.num_pods_per_trainer_cell
    )
    return InferencePlacement(
        trainer_coord=PodCoordinate(
            pool_id=trainer_pool_id, cell_index=trainer_cell_index, pod_in_cell_index=trainer_pod_index
        ),
        base_gpu_id=absolute_gpu % layout.num_gpus_per_node,
    )
