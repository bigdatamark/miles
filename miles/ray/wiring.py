# [MODULE DOC] Thin wiring layer between run arguments and the worker platform: launches or shuts
# down the Ray worker manager (the actor that creates workers from specs), and obtains the backend
# capability object (provider and handle factories) for the configured cluster backend, built from
# the specs computed for the run.
from __future__ import annotations

import ray
from ray.actor import ActorHandle

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.workers.backend_capability import factory
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import ClusterBackend, WorkerCommBackend


def launch_worker_manager(args):
    # [FUNCTION DOC] Start the worker manager for the configured cluster backend. Kubernetes: returns
    # none (the platform manages workers). Ray: creates placement groups and launches the Ray worker
    # manager actor, returning its handle. Any unlisted backend value falls through and returns none
    # implicitly.
    match ClusterBackend(args.cluster_backend):
        case ClusterBackend.KUBERNETES:
            return None
        case ClusterBackend.RAY:
            return _launch_ray_worker_manager(args)


async def shutdown_worker_manager(worker_manager_handle: ActorHandle | None) -> None:
    # [FUNCTION DOC] Stop the worker manager if there is one (none: do nothing). Awaits its remote
    # shutdown call and always kills the actor afterwards, even if shutdown raised (the error then
    # propagates after the kill).
    if worker_manager_handle is None:
        return
    try:
        await worker_manager_handle.shutdown.remote()
    finally:
        ray.kill(worker_manager_handle)


def get_backend_capability(args) -> BackendCapability:
    # [FUNCTION DOC] Build the backend capability from the specs computed from args (filtered by deploy
    # component) and the cluster backend named in args. Recomputes the specs on every call.
    return factory.get_backend_capability(
        specs=compute_specs(args), cluster_backend=ClusterBackend(args.cluster_backend)
    )


def _launch_ray_worker_manager(args):
    # [FUNCTION DOC] HELPER: Compute the specs, create the placement groups (imported lazily, since the
    # placement group module imports this module: circular dependency), and launch the Ray worker
    # manager with args, specs, placement groups and the configured worker communication backend.
    # Returns the manager handle. Reserves GPUs as a side effect. TODO in code: pass specs instead of
    # args to the placement group creation.
    from miles.ray.placement_group import create_placement_groups

    specs = compute_specs(args)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(args, specs, pgs, comm_backend=WorkerCommBackend(args.worker_comm_backend))
