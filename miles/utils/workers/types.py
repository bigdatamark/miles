# [MODULE DOC]
# Small shared enumerations and helpers for the workers layer: cluster backend, worker
# communication backend (and which combinations are supported), deployment component selectors,
# platform access level, hot-restart components with their comma-separated parser, and the
# identity of a deployment.
from enum import Enum

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ClusterBackend(Enum):
    # [CLASS DOC] Where the run is deployed: ray or kubernetes (string values as on the command line).
    RAY = "ray"
    KUBERNETES = "kubernetes"


class WorkerCommBackend(Enum):
    # [CLASS DOC] How the driver talks to workers: through ray actor calls or through the http rpc.
    RAY = "ray"
    RPC = "rpc"


class DeployComponent(Enum):
    # [CLASS DOC]
    # Which part of a run a deployment carries. All is a selector meaning every component (never a
    # component of a pool itself). Primary carries the orchestration script; trainer and inference
    # carry their own workers.
    ALL = "all"
    PRIMARY = "primary"
    TRAINER = "trainer"
    INFERENCE = "inference"

    def selects(self, component: "DeployComponent") -> bool:
        # [FUNCTION DOC] True iff this selector covers the given component: all covers every
        # component, any other value covers only itself. Asserts the argument is not all.
        assert component is not DeployComponent.ALL, "`all` is a selector over components, never a component itself"
        return self is DeployComponent.ALL or self is component

    def deploys_orchestration_script(self) -> bool:
        # [FUNCTION DOC] True iff this deployment covers the primary component (all or primary).
        return self.selects(DeployComponent.PRIMARY)

    def deploys_own_inference_engines(self) -> bool:
        # [FUNCTION DOC] True iff this deployment covers the inference component (all or inference).
        return self.selects(DeployComponent.INFERENCE)

    def is_split(self) -> bool:
        # [FUNCTION DOC] True iff the run is split into separate deployments, i.e. not all.
        return self is not DeployComponent.ALL

    def takes_instance_id(self) -> bool:
        # [FUNCTION DOC] True iff deployments of this component are identified by an instance id
        # (trainer and inference).
        return self in (DeployComponent.TRAINER, DeployComponent.INFERENCE)


class PlatformAccess(Enum):
    # [CLASS DOC] How much of the platform's control plane a pool may use: none, read, or read and
    # delete.
    NONE = "none"
    READ = "read"
    READ_DELETE = "read-delete"


class HotRestartComponent(Enum):
    # [CLASS DOC] Parts of a run that can be restarted in place: orchestration or the rollout executor.
    ORCHESTRATION = "orchestration"
    ROLLOUT_EXECUTOR = "rollout_executor"


HOT_RESTART_SEPARATOR = ","


def parse_hot_restart(value: str) -> list[HotRestartComponent]:
    # [FUNCTION DOC] Parse a comma-separated list of component names (whitespace around names and
    # empty items ignored) into components, in order. Raises a value error for an unknown name.
    # Asserts that a non-empty text yields at least one component; empty text yields an empty list.
    components = [
        HotRestartComponent(stripped) for name in value.split(HOT_RESTART_SEPARATOR) if (stripped := name.strip())
    ]
    assert (
        components or not value
    ), f"--hot-restart {value!r} names no component; pass one of {[one.value for one in HotRestartComponent]}"
    return components


class DeploymentIdentity(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable identity of a deployment: run uuid, deploy component (as text), and the
    # optional deploy instance id and trainer id.
    run_uuid: str
    deploy_component: str
    deploy_instance_id: str | None = None
    trainer_id: str | None = None


_SUPPORTED_WORKER_COMM_BACKENDS = {
    ClusterBackend.RAY: (WorkerCommBackend.RAY, WorkerCommBackend.RPC),
    ClusterBackend.KUBERNETES: (WorkerCommBackend.RPC,),
}


def resolve_worker_comm_backend(*, cluster_backend: ClusterBackend, requested: str | None) -> WorkerCommBackend:
    # [FUNCTION DOC] Choose the worker communication backend. Without a request, return the first
    # supported backend for the cluster backend (ray for ray, rpc for kubernetes). With a request,
    # parse it (value error if unknown) and assert it is supported by the cluster backend (ray
    # supports both, kubernetes only rpc).
    if requested is None:
        return _SUPPORTED_WORKER_COMM_BACKENDS[cluster_backend][0]

    backend = WorkerCommBackend(requested)
    supported = _SUPPORTED_WORKER_COMM_BACKENDS[cluster_backend]
    assert backend in supported, (
        f"--worker-comm-backend {backend.value} is not available under --cluster-backend {cluster_backend.value}, "
        f"which speaks {[one.value for one in supported]}"
    )
    return backend
