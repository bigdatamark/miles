# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

# [MODULE DOC] Pure read-only views over a list of fault injector events, used by the injector's
# form selection and by test assertions: counts and lists of injections, cells that never
# recovered to serving after being harmed, forms drawn without ever succeeding, injection times
# and the distinct sequence of states each cell passed through. Cell types of cell names are
# learned from the observation events. The stale-status grace period is 120 seconds.

import dataclasses
from datetime import datetime
from typing import Literal

from tests.e2e.ft.conftest_ft.fault_injection.state import Event, InjectionEvent, ObservationsEvent, ObservedCellState

STALE_STATUS_GRACE_SECONDS: float = 120.0


def compute_num_injections(events: list[Event], *, cell_type: str | None = None, harmed_only: bool = True) -> int:
    # [FUNCTION DOC] Number of successful injections (counting only harmful forms by default),
    # optionally restricted to one cell type; equals the length of the injected cell names list.
    return len(compute_injected_cell_names(events, cell_type=cell_type, harmed_only=harmed_only))


def compute_injected_cell_names(
    events: list[Event], *, cell_type: str | None = None, harmed_only: bool = True
) -> list[str]:
    # [FUNCTION DOC] One cell name per successful injection (so a cell hit twice appears twice),
    # grouped by cell in first-seen order, optionally restricted to a cell type and, by default,
    # only to injections of harmful forms.
    return [
        name
        for name, cell_events in _compute_matching_cell_events(
            events, cell_type=cell_type, harmed_only=harmed_only
        ).items()
        for one in cell_events
        if one.kind == "injected"
    ]


def compute_num_successful_injections_of_form(events: list[Event], *, form_name: str) -> int:
    # [FUNCTION DOC] Count of successful injection events whose form name equals the given name,
    # across all cell types.
    return len(
        [
            event
            for event in events
            if isinstance(event, InjectionEvent) and event.succeeded and event.form_name == form_name
        ]
    )


def compute_cells_not_serving_after_injection(
    events: list[Event], *, cell_type: str, grace_seconds: float | None = None
) -> dict[str, list[str]]:
    # [FUNCTION DOC] Finds cells of the given type that were harmed and never observed serving
    # again. For each cell, the last successful harmful injection time is taken. A cell counts as
    # recovered once some later observation shows it alive and serving at least the grace seconds
    # after that injection (default 120 seconds; the grace guards against stale status readings
    # right after a kill). Returns a mapping, sorted by cell name, from each non-recovered cell to
    # the ordered list of distinct state values it was observed in (empty list if never observed).
    if grace_seconds is None:
        grace_seconds = STALE_STATUS_GRACE_SECONDS

    cell_type_of_name = _compute_cell_type_of_name(events)
    last_injection_time_of_name: dict[str, datetime] = {
        event.cell_name: event.timestamp
        for event in events
        if isinstance(event, InjectionEvent)
        and event.succeeded
        and event.harmed
        and cell_type_of_name.get(event.cell_name) == cell_type
    }

    served: set[str] = set()
    for event in events:
        if not isinstance(event, ObservationsEvent):
            continue
        for name, injected_at in last_injection_time_of_name.items():
            info = event.cell_infos.get(name)
            if (
                name not in served
                and info is not None
                and info.alive
                and info.state is ObservedCellState.SERVING
                and (event.timestamp - injected_at).total_seconds() >= grace_seconds
            ):
                served.add(name)

    observed_states = compute_states_of_cell_name(events)
    return {
        name: [one.value for one in observed_states.get(name, [])]
        for name in sorted(set(last_injection_time_of_name) - served)
    }


def compute_successful_form_names(events: list[Event], *, cell_type: str) -> set[str]:
    # [FUNCTION DOC] Names of fault forms with at least one successful injection into a cell of the
    # given type (harmful or not).
    cell_type_of_name = _compute_cell_type_of_name(events)
    return {
        event.form_name
        for event in events
        if isinstance(event, InjectionEvent)
        and event.succeeded
        and cell_type_of_name.get(event.cell_name) == cell_type
    }


def compute_forms_drawn_without_success(events: list[Event]) -> list[tuple[str, str]]:
    # [FUNCTION DOC] Sorted list of (cell type, form name) pairs for which an injection was attempted
    # but none ever succeeded. Cells never observed get an empty cell type string.
    cell_type_of_name = _compute_cell_type_of_name(events)
    drawn: set[tuple[str, str]] = set()
    worked: set[tuple[str, str]] = set()
    for event in events:
        if not isinstance(event, InjectionEvent):
            continue
        key = (cell_type_of_name.get(event.cell_name, ""), event.form_name)
        drawn.add(key)
        if event.succeeded:
            worked.add(key)
    return sorted(drawn - worked)


def compute_injection_times(events: list[Event], *, cell_type: str | None = None) -> list[datetime]:
    # [FUNCTION DOC] Timestamps of successful injections (harmful or not), in event order,
    # optionally restricted to cells of one type.
    cell_type_of_name = _compute_cell_type_of_name(events)
    return [
        event.timestamp
        for event in events
        if isinstance(event, InjectionEvent)
        and event.succeeded
        and (cell_type is None or cell_type_of_name.get(event.cell_name) == cell_type)
    ]


def compute_states_of_cell_name(events: list[Event]) -> dict[str, list[ObservedCellState]]:
    # [FUNCTION DOC] For each cell with any observation, the sequence of distinct consecutive
    # observed states (repeats collapsed). Cells with no observed states are omitted.
    return {
        name: states
        for name, cell_events in _compute_cell_events(events).items()
        if (states := _compute_distinct_states(cell_events))
    }


@dataclasses.dataclass(frozen=True)
class _CellEvent:
    # [CLASS DOC] HELPER: Immutable per-cell timeline entry: kind is either injected or observed;
    # observed entries carry the observed state, injected entries carry none.
    kind: Literal["injected", "observed"]
    state: ObservedCellState | None = None


def _compute_cell_events(events: list[Event], *, harmed_only: bool = True) -> dict[str, list[_CellEvent]]:
    # [FUNCTION DOC] HELPER: Builds, per cell name, a time-ordered timeline. Successful injections
    # become injected entries (only harmful ones when harmed-only, the default); failed ones are
    # dropped. Every cell in every poll becomes an observed entry with its state.
    cell_events_of_name: dict[str, list[_CellEvent]] = {}
    for event in events:
        if isinstance(event, InjectionEvent):
            if event.succeeded and (event.harmed or not harmed_only):
                cell_events_of_name.setdefault(event.cell_name, []).append(_CellEvent(kind="injected"))
            continue
        for name, info in event.cell_infos.items():
            cell_events_of_name.setdefault(name, []).append(_CellEvent(kind="observed", state=info.state))
    return cell_events_of_name


def _compute_matching_cell_events(
    events: list[Event], *, cell_type: str | None, harmed_only: bool
) -> dict[str, list[_CellEvent]]:
    # [FUNCTION DOC] HELPER: The per-cell timelines, filtered to cells of the given type when one is
    # given (type known from observations; cells never observed are dropped when filtering).
    cell_events_of_name = _compute_cell_events(events, harmed_only=harmed_only)
    if cell_type is None:
        return cell_events_of_name
    cell_type_of_name = _compute_cell_type_of_name(events)
    return {
        name: cell_events
        for name, cell_events in cell_events_of_name.items()
        if cell_type_of_name.get(name) == cell_type
    }


def _compute_cell_type_of_name(events: list[Event]) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Mapping from cell name to cell type learned from all observation
    # events (later observations overwrite earlier ones).
    cell_type_of_name: dict[str, str] = {}
    for event in events:
        if isinstance(event, ObservationsEvent):
            cell_type_of_name.update({name: info.cell_type for name, info in event.cell_infos.items()})
    return cell_type_of_name


def _compute_distinct_states(events: list[_CellEvent]) -> list[ObservedCellState]:
    # [FUNCTION DOC] HELPER: Collapses a cell timeline to its observed states, dropping injected
    # entries and consecutive duplicates.
    states: list[ObservedCellState] = []
    for event in events:
        if event.kind == "observed" and event.state is not None and (not states or states[-1] != event.state):
            states.append(event.state)
    return states
