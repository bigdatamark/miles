# [MODULE DOC]
# Ray implementation of the backend capability. All providers and cell operations talk to a
# single Ray worker manager actor through its handle.
from __future__ import annotations

from collections.abc import Sequence

import ray.actor

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.cell_operations.ray import RayCellOperations
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.ray import RayWorkerProvider


class RayBackendCapability(BackendCapability):
    # [CLASS DOC]
    # Backend capability for Ray deployments. Holds the actor handle of the Ray worker manager and
    # passes it to every provider and to cell operations it creates.
    def __init__(self, *, worker_manager_handle: ray.actor.ActorHandle) -> None:
        # [FUNCTION DOC]
        # Store the Ray worker manager actor handle. No I/O.
        self._worker_manager_handle = worker_manager_handle

    def dynamic_worker_provider(self, *, pool_ids: Sequence[str]) -> BaseWorkerProvider:
        # [FUNCTION DOC]
        # Create a Ray worker provider bound to the manager handle and restricted to a copy of the
        # given pool ids (no validation of the ids).
        return RayWorkerProvider(worker_manager_handle=self._worker_manager_handle, pool_ids=list(pool_ids))

    def static_worker_provider(self, *, pool_id: str) -> BaseWorkerProvider:
        # [FUNCTION DOC]
        # Create a Ray worker provider bound to the manager handle.
        # KNOWN ISSUE: the pool id argument is ignored; the provider is not restricted to that
        # pool, so it is the same as an unfiltered dynamic provider.
        return RayWorkerProvider(worker_manager_handle=self._worker_manager_handle)

    def cell_operations(self) -> BaseCellOperations:
        # [FUNCTION DOC]
        # Create Ray cell operations bound to the manager handle. The inference controller is
        # resolved lazily through a callback that builds the inference controller handle from this
        # capability; the import of that factory is done locally on purpose (a deliberate layering
        # violation, see the note below) to avoid a dependency from this layer on the ray package.
        # TEMPORARY: this layer is not meant to import miles.ray, deliberately violated until the
        # weight-update fault tolerance work removes the need
        from miles.ray.specs.inference import create_inference_controller_handle

        return RayCellOperations(
            worker_manager_handle=self._worker_manager_handle,
            resolve_inference_controller=lambda: create_inference_controller_handle(capability=self),
        )
