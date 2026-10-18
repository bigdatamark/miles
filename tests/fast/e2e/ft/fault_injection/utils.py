# [MODULE DOC] HELPER: Shared builders for the fault-injection unit tests: fake cell dictionaries
# shaped like the control-plane api server's cell objects (metadata name and labels for cell type
# and workers hash; status phase and Healthy/Serving conditions), canned http responses, observed
# state aliases, event-log builders, config and fault-form factories, and a stub fault form.
import contextlib
import random
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, patch

from tests.e2e.ft.conftest_ft.fault_injection import core, fault_forms, state

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend


def note_injected(log: state.EventLog, cell_name: str) -> None:
    # [FUNCTION DOC] HELPER: Records in the event log a successful injection attempt of form
    # sigkill against the named cell.
    log.note_injection_attempt(
        cell_name=cell_name,
        form_name="sigkill",
        succeeded=True,
    )


@contextlib.contextmanager
def patched_requests() -> Iterator[MagicMock]:
    # [FUNCTION DOC] HELPER: Context manager that replaces the http client in both the core module
    # and the fault_forms module with one shared mock for the duration, and yields that mock.
    # Patches are undone on exit.
    # the loop lists cells through core and injects through fault_forms, so a mock on core alone
    # leaves every injection reaching the real network and timing out against a host nobody serves
    mock_requests = MagicMock()
    with patch.object(core, "requests", mock_requests), patch.object(fault_forms, "requests", mock_requests):
        yield mock_requests


NAMESPACE = "miles-e2e"
RUN_ID = "abc123"


def cell(
    name: str,
    *,
    healthy: bool,
    cell_type: str = "actor",
    phase: str = "Running",
    serving: bool = True,
    workers_hash: str = "generation-0",
) -> dict:
    # [FUNCTION DOC] HELPER: Builds a fake cell object. Metadata has the name and labels for cell
    # type and workers hash. Status has the phase and a Healthy condition set from healthy; only
    # rollout cells also get a Serving condition set from serving. Returns the dictionary.
    status = "True" if healthy else "False"
    conditions = [{"type": "Healthy", "status": status}]
    if cell_type == "rollout":
        conditions.append({"type": "Serving", "status": "True" if serving else "False"})
    return {
        "metadata": {
            "name": name,
            "labels": {"miles.io/cell-type": cell_type, "miles.io/workers-hash": workers_hash},
        },
        "status": {"phase": phase, "conditions": conditions},
    }


def names(cells: list[dict]) -> set[str]:
    # [FUNCTION DOC] HELPER: Returns the set of metadata names of the given cell objects.
    return {c["metadata"]["name"] for c in cells}


def mock_response(payload: dict) -> MagicMock:
    # [FUNCTION DOC] HELPER: Builds a fake http response whose status check does nothing and whose
    # json method returns the given payload.
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


SERVING = state.ObservedCellState.SERVING
RUNNING_NOT_SERVING = state.ObservedCellState.RUNNING_NOT_SERVING
PENDING = state.ObservedCellState.PENDING
SUSPENDED = state.ObservedCellState.SUSPENDED


def staged(
    name: str,
    cell_state: state.ObservedCellState,
    *,
    cell_type: str = "rollout",
    workers_hash: str = "generation-0",
) -> dict:
    # [FUNCTION DOC] HELPER: Builds a fake cell object that the observer classifies as the given
    # observed state. Phase mapping: suspended to Suspended, pending to Pending, running-not-serving
    # and serving to Running. Only Running cells carry conditions: Healthy true, plus Serving true
    # only for the serving state. Default type is rollout.
    phase = {
        SUSPENDED: "Suspended",
        PENDING: "Pending",
        RUNNING_NOT_SERVING: "Running",
        SERVING: "Running",
    }[cell_state]
    conditions: list[dict] = (
        [
            {"type": "Healthy", "status": "True"},
            {"type": "Serving", "status": "True" if cell_state is SERVING else "False"},
        ]
        if phase == "Running"
        else []
    )
    return {
        "metadata": {
            "name": name,
            "labels": {"miles.io/cell-type": cell_type, "miles.io/workers-hash": workers_hash},
        },
        "status": {"phase": phase, "conditions": conditions},
    }


def log_of(
    cell_states: list[state.ObservedCellState], *, inject_before: dict[int, int] | None = None
) -> state.EventLog:
    # [FUNCTION DOC] HELPER: Builds an event log by observing, in order, one rollout engine cell
    # (rollout-engine-0) in each given state. Before the observation at index i, it first notes
    # inject_before[i] successful injection attempts (default none) against that cell.
    log = state.EventLog()
    for index, cell_state in enumerate(cell_states):
        for _ in range((inject_before or {}).get(index, 0)):
            note_injected(log, "rollout-engine-0")
        log.observe([staged("rollout-engine-0", cell_state)])
    return log


def typed_cell(name: str, cell_type: str, *, healthy: bool = True, serving: bool = True) -> dict:
    # [FUNCTION DOC] HELPER: Shortcut for building a cell of the given type with healthy and serving flags.
    return cell(name, healthy=healthy, cell_type=cell_type, serving=serving)


def config_of(backend: ClusterBackend, *, namespace: str = NAMESPACE) -> command_utils.ExecuteTrainConfig:
    # [FUNCTION DOC] HELPER: Builds a train config for the given cluster backend with the module
    # namespace (overridable) and run id.
    return command_utils.ExecuteTrainConfig(cluster_backend=backend, namespace=namespace, run_id=RUN_ID)


def api_server_fault_forms() -> fault_forms.CellFaultForms:
    # [FUNCTION DOC] HELPER: Builds the per-cell-type fault forms for the Ray backend (api-server
    # inject-fault forms only) with a fake control base url.
    return fault_forms.create_cell_fault_forms(base_url="http://control", config=config_of(ClusterBackend.RAY))


class StubFaultForm(fault_forms.BaseFaultForm):
    # [CLASS DOC] HELPER: Fault form with a caller-chosen name whose injection just calls a given
    # callback with the cell and random generator.
    def __init__(self, form_name: str, on_inject: Callable[[dict, random.Random], None]) -> None:
        # [FUNCTION DOC] HELPER: Stores the name and the injection callback.
        self._name = form_name
        self._on_inject = on_inject

    @property
    def name(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the stored form name.
        return self._name

    def inject(self, cell: dict, rng: random.Random) -> None:
        # [FUNCTION DOC] HELPER: Calls the stored callback with the cell and generator.
        self._on_inject(cell, rng)


def fixed_fault_forms(forms: list[fault_forms.BaseFaultForm]) -> fault_forms.CellFaultForms:
    # [FUNCTION DOC] HELPER: Returns a mapping giving the same form list to both the actor and the
    # rollout cell types.
    return {fault_forms.ACTOR_CELL_TYPE: forms, fault_forms.ROLLOUT_CELL_TYPE: forms}


def intervals(cell_types: tuple[str, ...], mean_interval_seconds: float) -> dict[str, float]:
    # [FUNCTION DOC] HELPER: Returns a mapping from each given cell type to the same mean interval in seconds.
    return {cell_type: mean_interval_seconds for cell_type in cell_types}
