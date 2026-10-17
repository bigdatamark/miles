# [MODULE DOC] Computes the in-cluster network addresses and matching command line flags that let
# separately installed deployments of one split run (trainer deployments, the primary inference
# deployment) reach each other. All addresses are derived deterministically from the run id, run
# uuid and namespace, so each launch command can be issued independently and still agree.
# Constants: the default trainer id (the actor role) and the flag name carrying the expected
# cell count per model. Every returned flag string ends with a trailing space so pieces can be
# concatenated into one argument string.
from dataclasses import dataclass

from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.ray.specs.inference import (
    INFERENCE_CONTROLLER_ADDR_FLAG,
    INFERENCE_CONTROLLER_POOL_ID,
    INFERENCE_CONTROLLER_WORKER_CLASS,
)
from miles.ray.specs.train import (
    TRAINER_CONTROLLER_ADDRS_FLAG,
    TRAINER_CONTROLLER_WORKER_CLASS,
    compute_trainer_controller_pool_id,
)
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.common import get_mooncake_object_store_args
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName, RunNames
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import static_cell_addrs
from miles.utils.workers.worker_spec import RPC_PORT_NAME, SchedulingSpec, ServeWorkerSpec

DEFAULT_TRAINER_ID: str = ACTOR_ROLE
INIT_EXPECTED_NUM_CELLS_FLAG: str = "--init-expected-num-cells"


@dataclass(frozen=True)
class RunAddressBook:
    # [CLASS DOC] Immutable identity of one split run: run id, run uuid and Kubernetes namespace.
    # Methods compute release names and controller RPC addresses from these three values. Every
    # deployment of the run must be given the same values.
    run_id: str
    run_uuid: str
    namespace: str

    @classmethod
    def of_config(cls, config: command_utils.ExecuteTrainConfig) -> "RunAddressBook":
        # [FUNCTION DOC] Build an address book from a train config (run id, run uuid, namespace).
        # Fails an assertion if the namespace is empty (it is part of every in-cluster name) or if
        # the run uuid is unset (it is the only thing joining the deployments, and every install
        # command of a split launch must be given the same uuid explicitly).
        assert config.namespace, (
            "the deployments of one run reach each other by in-cluster release names, and a namespace is half "
            "of every such name"
        )
        assert config.run_uuid is not None, (
            "the deployments of one run are joined by nothing but the run uuid, and a split launch is given none "
            "of its own, so every command installing a part of it has to carry the same --run-uuid"
        )
        return cls(run_id=config.run_id, run_uuid=config.run_uuid, namespace=config.namespace)

    def release(self, deploy_component: DeployComponent, deploy_instance_id: str | None = None) -> str:
        # [FUNCTION DOC] Helm release name of a deployment of this run, for a deploy component
        # (trainer, primary and so on) and an optional deploy instance id (distinguishes several
        # deployments of the same component, for example one per trainer). Returns the serialized
        # name built from run id, component and instance id.
        return ReleaseName(
            run_id=self.run_id, deploy_component=deploy_component, deploy_instance_id=deploy_instance_id
        ).serialize()

    def trainer_controller_addrs_arg(self, *, deploy_instance_id_of_trainer_id: dict[str, str | None]) -> str:
        # [FUNCTION DOC] Build the command line fragment that tells inference where each trainer's
        # controller listens. Input: a mapping from trainer id to the deploy instance id of that
        # trainer's deployment (or none). For each trainer, the address is the controller RPC
        # address within that trainer deployment's release, using the pool id computed from the
        # trainer id and the trainer controller worker class. Returns the trainer-addresses flag
        # followed by space separated entries of the form trainer id, equals sign, address, ending
        # with a trailing space. An empty mapping yields the flag with no entries.
        entries = [
            f"{trainer_id}="
            + self._rpc_addr(
                release=self.release(DeployComponent.TRAINER, deploy_instance_id),
                pool_id=compute_trainer_controller_pool_id(trainer_id),
                worker_class=TRAINER_CONTROLLER_WORKER_CLASS,
            )
            for trainer_id, deploy_instance_id in deploy_instance_id_of_trainer_id.items()
        ]
        return f"{TRAINER_CONTROLLER_ADDRS_FLAG} {' '.join(entries)} "

    def inference_controller_addr_arg(self) -> str:
        # [FUNCTION DOC] Build the command line fragment telling trainers where the inference
        # controller listens: the controller RPC address in the primary component's release (fixed
        # inference pool id and worker class), preceded by the inference-controller-address flag
        # and followed by a trailing space.
        addr = self._rpc_addr(
            release=self.release(DeployComponent.PRIMARY),
            pool_id=INFERENCE_CONTROLLER_POOL_ID,
            worker_class=INFERENCE_CONTROLLER_WORKER_CLASS,
        )
        return f"{INFERENCE_CONTROLLER_ADDR_FLAG} {addr} "

    def shared_object_store_args(self) -> str:
        # [FUNCTION DOC] Build the object store arguments pointing at the Mooncake master service
        # that lives in the primary deployment: the host is derived from the primary release name
        # and namespace. Returns the argument string produced by the shared command utility.
        return get_mooncake_object_store_args(
            master_host=MooncakeInfo.master_service_host(self.release(DeployComponent.PRIMARY), self.namespace)
        )

    def _rpc_addr(self, *, release: str, pool_id: str, worker_class: str) -> str:
        # [FUNCTION DOC] HELPER: Compute host and port of the single controller worker (cell
        # index 0) of a controller pool in a release. Builds a minimal one-cell, zero-GPU worker
        # spec for the pool, asks the Helm worker provider for its static cell addresses, takes the
        # entry for the RPC port name, and returns the cluster service fully qualified name of the
        # host (within this namespace), a colon and the port, as a string.
        rpc = static_cell_addrs(
            spec=_controller_spec(pool_id=pool_id, worker_class=worker_class), release=release, cell_index=0
        )[RPC_PORT_NAME]
        return f"{RunNames.service_fqdn(name=rpc.host, namespace=self.namespace)}:{rpc.port}"


def init_expected_num_cells_arg(num_cells_per_model: int) -> str:
    # [FUNCTION DOC] Return the expected-cells flag followed by the integer and a trailing space.
    return f"{INIT_EXPECTED_NUM_CELLS_FLAG} {num_cells_per_model} "


def _controller_spec(*, pool_id: str, worker_class: str) -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Minimal serve worker spec describing a controller pool, used only to
    # compute addresses: named by the pool id, no extra ports, empty environment and constructor
    # arguments, one cell with one worker needing 0 GPUs and 1 CPU, and the given worker class.
    return ServeWorkerSpec(
        name=pool_id,
        port_infos=[],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0, num_cpus_per_worker=1),
        worker_class=worker_class,
        ctor_kwargs=lambda _ctx: {},
    )
