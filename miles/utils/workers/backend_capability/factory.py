# [MODULE DOC]
# Selects the concrete backend capability for a run based on the configured cluster backend.
from __future__ import annotations

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability
from miles.utils.workers.worker_spec import BaseWorkerSpec


def get_backend_capability(*, specs: list[BaseWorkerSpec], cluster_backend: ClusterBackend) -> BackendCapability:
    # [FUNCTION DOC]
    # Build the backend capability for the given cluster backend. For Kubernetes it delegates to the
    # helm builder, which derives the capability from the worker specs. For Ray it wraps the
    # handle of the Ray worker manager actor (fetched via the manager's get-handle lookup, which
    # may touch the Ray cluster). Returns a backend capability. Any other backend value falls
    # through the match and implicitly returns nothing.
    # KNOWN ISSUE: there is no default case, so an unsupported backend yields None instead of an error.
    match cluster_backend:
        case ClusterBackend.KUBERNETES:
            return compute_helm_backend_capability(specs=specs)
        case ClusterBackend.RAY:
            return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())
