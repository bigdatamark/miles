# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

# [MODULE DOC] Fault forms: the concrete ways the injector can crash a cell. Three forms exist:
# asking the API server to inject a fault of a given failure mode, deleting one pod of the cell
# (Kubernetes only), and sending a kill signal to matching processes inside a cell's pod
# (Kubernetes only). Also defines the cell type names (actor, rollout), the form names, the
# failure mode lists (all of kill, exit, segfault; rollout engines under Ray only get kill), and
# the factory choosing forms by cluster backend.

import abc
import random

import requests
from tests.e2e.ft.conftest_ft.fault_injection.pod_manipulation import (
    delete_one_pod_of_cell,
    list_pod_names_of_cell,
    sigkill_process_patterns_in_pod,
)

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.types import ClusterBackend, DeployComponent

FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT]
RAY_ROLLOUT_ENGINE_FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL]

DELETE_POD_FORM_NAME: str = "delete_pod"
EXEC_SIGKILL_FORM_NAME: str = "exec_sigkill"
ENGINE_CONTAINER_NAME: str = "engine"
SGLANG_PROCESS_PATTERN: str = "sglang::"

ACTOR_CELL_TYPE: str = "actor"
ROLLOUT_CELL_TYPE: str = "rollout"


class BaseFaultForm(abc.ABC):
    # [CLASS DOC] Abstract base of a fault form: has a stable name, says whether it harms the cell,
    # and can inject itself into a cell given as an API object mapping.
    @property
    @abc.abstractmethod
    def name(self) -> str: ...
    # [FUNCTION DOC] Abstract property: the form's identifier string used in event logs.

    @property
    def harms_cell(self) -> bool:
        # [FUNCTION DOC] Whether injecting this form damages the cell; true by default.
        return True

    @abc.abstractmethod
    def inject(self, cell: dict, rng: random.Random) -> None: ...
    # [FUNCTION DOC] Abstract: crash the given cell (API object with metadata name), using the
    # random generator for any random choice. Raises on failure.


class InjectFaultForm(BaseFaultForm):
    # [CLASS DOC] Fault form that asks the API server to make the cell fail with one failure mode
    # (kill, exit or segfault). Works for any cluster backend.
    def __init__(self, *, base_url: str, failure_mode: FailureMode) -> None:
        # [FUNCTION DOC] Stores the API server base url and the failure mode to request.
        self._base_url = base_url
        self._failure_mode = failure_mode

    @property
    def name(self) -> str:
        # [FUNCTION DOC] The form name: the prefix for injected faults, a colon, then the failure
        # mode's string value.
        return f"inject_fault:{self._failure_mode.value}"

    def inject(self, cell: dict, rng: random.Random) -> None:
        # [FUNCTION DOC] Posts a request to the cell's inject-fault endpoint on the API server
        # with the failure mode and sub index 0 (5 second timeout). The random generator is
        # unused. Raises on network error or a bad status code.
        resp = requests.post(
            f"{self._base_url}/api/v1/cells/{cell['metadata']['name']}/inject-fault",
            json={"mode": self._failure_mode.value, "sub_index": 0},
            timeout=5,
        )
        resp.raise_for_status()


class DeletePodFaultForm(BaseFaultForm):
    # [CLASS DOC] Kubernetes-only fault form that deletes one randomly chosen pod belonging to the
    # cell, within the run's whole-deployment helm release.
    def __init__(self, *, namespace: str, run_id: str) -> None:
        # [FUNCTION DOC] Requires non-empty namespace and run id (assertion errors otherwise) and
        # precomputes the serialized release name from the run id with the all-components
        # deploy component and no instance id.
        assert namespace, "Deleting a cell's pod needs the namespace the run was installed into"
        assert run_id, "Deleting a cell's pod needs the run_id naming the release that owns it"

        self._namespace = namespace
        self._release = ReleaseName(
            run_id=run_id, deploy_component=DeployComponent.ALL, deploy_instance_id=None
        ).serialize()

    @property
    def name(self) -> str:
        # [FUNCTION DOC] The constant delete-pod form name.
        return DELETE_POD_FORM_NAME

    def inject(self, cell: dict, rng: random.Random) -> None:
        # [FUNCTION DOC] Deletes one pod of the cell (named by the cell's metadata name) in the
        # namespace and release, the pod chosen with the random generator. Delegates to the pod
        # manipulation helper.
        delete_one_pod_of_cell(
            namespace=self._namespace, release=self._release, cell_id=cell["metadata"]["name"], rng=rng
        )


class ExecSigkillFaultForm(BaseFaultForm):
    # [CLASS DOC] Kubernetes-only fault form that runs a kill inside one randomly chosen pod of the
    # cell, targeting processes in a named container whose command matches a pattern.
    def __init__(self, *, namespace: str, run_id: str, container: str, process_pattern: str) -> None:
        # [FUNCTION DOC] Requires non-empty namespace and run id (assertion errors otherwise);
        # stores them, the container name and the process pattern, and precomputes the serialized
        # release name as the delete-pod form does.
        assert namespace, "Crashing a process inside a cell's pod needs the namespace the run was installed into"
        assert run_id, "Crashing a process inside a cell's pod needs the run_id naming the release that owns it"

        self._namespace = namespace
        self._release = ReleaseName(
            run_id=run_id, deploy_component=DeployComponent.ALL, deploy_instance_id=None
        ).serialize()
        self._container = container
        self._process_pattern = process_pattern

    @property
    def name(self) -> str:
        # [FUNCTION DOC] The constant exec-kill form name.
        return EXEC_SIGKILL_FORM_NAME

    def inject(self, cell: dict, rng: random.Random) -> None:
        # [FUNCTION DOC] Lists the pods of the cell in the release; asserts there is at least one;
        # picks one at random with the generator and kills processes matching the pattern inside
        # the configured container of that pod.
        cell_id = cell["metadata"]["name"]
        pod_names = list_pod_names_of_cell(namespace=self._namespace, release=self._release, cell_id=cell_id)
        assert pod_names, f"Release {self._release} has no pod of cell {cell_id} in {self._namespace} to crash"

        sigkill_process_patterns_in_pod(
            namespace=self._namespace,
            pod_name=rng.choice(pod_names),
            container=self._container,
            process_pattern=self._process_pattern,
        )


CellFaultForms = dict[str, list[BaseFaultForm]]


def create_cell_fault_forms(*, base_url: str, config: command_utils.ExecuteTrainConfig) -> CellFaultForms:
    # [FUNCTION DOC] Builds the mapping from cell type to its fault forms for the config's cluster
    # backend. Ray: actor cells get API-injected faults for all three failure modes; rollout cells
    # get only the kill mode. Kubernetes: actor cells get the three API-injected forms plus
    # delete-pod; rollout cells get exec-kill (container named engine, process pattern for sglang
    # worker processes) plus delete-pod. The Kubernetes forms assert namespace and run id are set.
    # KNOWN ISSUE: no default case; an unknown cluster backend makes this return none.
    actor_kill_forms = _inject_fault_forms(base_url=base_url, failure_modes=FAILURE_MODES)

    match config.cluster_backend:
        case ClusterBackend.RAY:
            return {
                ACTOR_CELL_TYPE: actor_kill_forms,
                ROLLOUT_CELL_TYPE: _inject_fault_forms(
                    base_url=base_url, failure_modes=RAY_ROLLOUT_ENGINE_FAILURE_MODES
                ),
            }
        case ClusterBackend.KUBERNETES:
            delete_pod_form = DeletePodFaultForm(namespace=config.namespace, run_id=config.run_id)
            exec_sigkill_form = ExecSigkillFaultForm(
                namespace=config.namespace,
                run_id=config.run_id,
                container=ENGINE_CONTAINER_NAME,
                process_pattern=SGLANG_PROCESS_PATTERN,
            )
            return {
                ACTOR_CELL_TYPE: [*actor_kill_forms, delete_pod_form],
                ROLLOUT_CELL_TYPE: [exec_sigkill_form, delete_pod_form],
            }


def _inject_fault_forms(*, base_url: str, failure_modes: list[FailureMode]) -> list[BaseFaultForm]:
    # [FUNCTION DOC] HELPER: One API-injected fault form per given failure mode, in order.
    return [InjectFaultForm(base_url=base_url, failure_mode=failure_mode) for failure_mode in failure_modes]


CELL_TYPE_OF_FT_COMPONENT: dict[str, str] = {"train": ACTOR_CELL_TYPE, "rollout": ROLLOUT_CELL_TYPE}


def compute_mean_interval_seconds_of_cell_type(
    ft_components: tuple[str, ...], *, trainer_crash_interval_seconds: float, rollout_crash_interval_seconds: float
) -> dict[str, float]:
    # [FUNCTION DOC] Maps the given FT components (train, rollout) to cell types (actor, rollout)
    # and the corresponding crash interval in seconds. Only components listed are included.
    # Raises a key error for an unknown component name.
    interval_seconds_of_component: dict[str, float] = {
        "train": trainer_crash_interval_seconds,
        "rollout": rollout_crash_interval_seconds,
    }

    return {
        CELL_TYPE_OF_FT_COMPONENT[component]: interval_seconds_of_component[component] for component in ft_components
    }
