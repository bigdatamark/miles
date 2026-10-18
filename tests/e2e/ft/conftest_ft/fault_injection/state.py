# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

# [MODULE DOC] State model of the fault injector: helpers reading cell API objects (liveness,
# observed serving state, cell type label), frozen event records (injection attempts and whole
# polls of observed cells), and a thread-safe append-only event log that is the injector's only
# mutable state.

import enum
import threading
from datetime import datetime, timezone

from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel


def cell_is_alive(cell: dict) -> bool:
    # [FUNCTION DOC] True when the cell's status conditions include one of type healthy with status
    # true. Raises a key error if the status or conditions are missing.
    return any(cond["type"] == "Healthy" and cond["status"] == "True" for cond in cell["status"]["conditions"])


class ObservedCellState(enum.Enum):
    # [CLASS DOC] Coarse serving state of a cell as read from its API status, ordered from least to
    # most available: suspended, pending, running but not serving, serving.
    SUSPENDED = "Suspended"  # torn down, holding no gpu
    PENDING = "Pending"  # allocated but gated: no engine serving yet
    RUNNING_NOT_SERVING = "RunningNotServing"  # engine is up but not registered in the router
    SERVING = "Serving"  # registered in the router, i.e. actually able to answer requests


def compute_observed_cell_state(cell: dict) -> ObservedCellState:
    # [FUNCTION DOC] Maps the cell's status phase to a state: suspended phase gives suspended,
    # pending phase gives pending; any other phase gives serving if a condition of type serving
    # has status true, else running-not-serving.
    phase = cell["status"]["phase"]
    if phase == "Suspended":
        return ObservedCellState.SUSPENDED
    if phase == "Pending":
        return ObservedCellState.PENDING
    serving = any(cond["type"] == "Serving" and cond["status"] == "True" for cond in cell["status"]["conditions"])
    return ObservedCellState.SERVING if serving else ObservedCellState.RUNNING_NOT_SERVING


class BaseEvent(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable base record carrying a timezone-aware UTC wall-clock timestamp set at
    # creation by default.
    # Wall clock, so an event can be lined up against the timestamps the metric events carry.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InjectionEvent(BaseEvent):
    # [CLASS DOC] Record of one injection attempt: target cell name, fault form name, whether the
    # injection call succeeded, and whether the form harms the cell (default true).
    cell_name: str
    form_name: str
    succeeded: bool
    harmed: bool = True


class CellInfo(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable snapshot of one cell at one poll: its cell type, observed state, and
    # whether it is alive (healthy).
    cell_type: str
    state: ObservedCellState
    alive: bool


class ObservationsEvent(BaseEvent):
    # [CLASS DOC] Record of one whole poll: a mapping from cell name to its snapshot. Cells absent
    # from the mapping were absent from that poll.
    # One whole poll, so a cell that has vanished is as recorded as one that answered.
    cell_infos: dict[str, CellInfo]


Event = InjectionEvent | ObservationsEvent


class EventLog:
    # [CLASS DOC] Thread-safe, append-only list of events guarded by a lock; written by the
    # injector thread and read by test threads.
    """The fault injector's only mutable state: what happened, in order. Every question is a view of it."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates an empty event list and its lock.
        self._events: list[Event] = []
        self._lock = threading.Lock()

    @property
    def events(self) -> list[Event]:
        # [FUNCTION DOC] Returns a copy of the events recorded so far, in order, taken under the lock.
        with self._lock:
            return list(self._events)

    def note_injection_attempt(self, *, cell_name: str, form_name: str, succeeded: bool, harmed: bool = True) -> None:
        # [FUNCTION DOC] Appends an injection event for the cell and form with the success and
        # harmed flags.
        self._append(InjectionEvent(cell_name=cell_name, form_name=form_name, succeeded=succeeded, harmed=harmed))

    def observe(self, cells: list[dict]) -> None:
        # [FUNCTION DOC] Appends one observations event built from the polled cell API objects:
        # per cell name, its cell type label, observed state and liveness.
        self._append(
            ObservationsEvent(
                cell_infos={
                    cell["metadata"]["name"]: CellInfo(
                        cell_type=cell_type_of(cell),
                        state=compute_observed_cell_state(cell),
                        alive=cell_is_alive(cell),
                    )
                    for cell in cells
                }
            )
        )

    def _append(self, event: Event) -> None:
        # [FUNCTION DOC] HELPER: Appends an event under the lock.
        with self._lock:
            self._events.append(event)


def cell_type_of(cell: dict) -> str:
    # [FUNCTION DOC] Returns the cell's cell-type label value; raises a key error if absent.
    return cell["metadata"]["labels"]["miles.io/cell-type"]
