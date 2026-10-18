# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

# [MODULE DOC] The random fault injection loop. Polls the cluster API server for cells, records
# every poll in an event log, and at randomized exponentially-distributed times injects a fault
# into a randomly chosen cell of a cell type, but only when that cell type is quiescent (all its
# replicas present, alive and able to serve for a number of consecutive polls) and has a spare
# replica. Defaults: poll every 2 seconds; require 60 consecutive quiescent polls.

import logging
import random
import threading
import time
from collections.abc import Callable
import requests

from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import ROLLOUT_CELL_TYPE, BaseFaultForm, CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.state import (
    Event,
    EventLog,
    ObservedCellState,
    cell_is_alive,
    cell_type_of,
    compute_observed_cell_state,
)
from tests.e2e.ft.conftest_ft.fault_injection.views import compute_successful_form_names

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS: float = 2.0
QUIESCENT_POLLS_REQUIRED: int = 60


def _compute_next_injection_time(rng: random.Random, mean_interval_seconds: float) -> float:
    # [FUNCTION DOC] HELPER: Returns the monotonic-clock time of the next injection: now plus a
    # sample from an exponential distribution whose mean is the given interval, drawn from the
    # given random generator (advancing it).
    return time.monotonic() + rng.expovariate(1.0 / mean_interval_seconds)


def run_fault_injection_loop(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds_of_cell_type: dict[str, float],
    stop_event: threading.Event,
    event_log: EventLog,
    cell_fault_forms: CellFaultForms,
    get_virtual_cells: Callable[[], list[dict]] | None = None,
    injection_enabled: Callable[[], bool] | None = None,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED,
) -> None:
    # [FUNCTION DOC] Runs until the stop event is set; intended to run on a worker thread.
    # Inputs: API server base url; seed for a private random generator (all random choices, so
    # runs are reproducible in choice order); mapping from cell type to mean seconds between
    # injections; stop event; event log to record into; mapping from cell type to its list of
    # fault forms; optional supplier of extra "virtual" cells appended to each poll; optional
    # predicate that can veto an injection; poll interval; consecutive quiescent polls required.
    # Setup: for each configured cell type (iterated in sorted order, so random draws are
    # deterministic) schedule a first injection time; track per-type quiescent poll counts and the
    # maximum number of cells ever seen.
    # Each iteration: wait one poll interval (exit if stopped); list cells (skip the iteration if
    # listing failed); append virtual cells; record the poll in the event log; exit if stopped;
    # group cells by type; update the per-type maximum cell count; a type is quiescent when it has
    # at least that maximum number of cells, all alive and able to serve; increment its quiescent
    # counter, else reset it to zero. Find types whose scheduled time has passed (due). Among due
    # types choose those with enough quiescent polls and more than one cell (a spare replica to
    # survive the kill); if none, log deferral and continue. Pick a random ready type, a random
    # cell of it, and draw a fault form (preferring forms not yet proven successful for the type).
    # If the injection-enabled predicate says no, skip without rescheduling. Inject the fault:
    # on any exception record a failed attempt, reset that type's quiescent counter, log, and
    # continue (the due time stays, so it retries); on success record a successful attempt,
    # reset the quiescent counter, and schedule the next injection time. Records whether the form
    # harms the cell in the attempt note. Exceptions from injection are swallowed.
    rng = random.Random(seed)
    next_injection_time_of_cell_type: dict[str, float] = {
        cell_type: _compute_next_injection_time(rng, mean_interval_seconds)
        for cell_type, mean_interval_seconds in sorted(mean_interval_seconds_of_cell_type.items())
    }
    quiescent_polls_of_cell_type: dict[str, int] = dict.fromkeys(next_injection_time_of_cell_type, 0)
    max_num_cells_of_cell_type: dict[str, int] = dict.fromkeys(next_injection_time_of_cell_type, 0)

    while not stop_event.is_set():
        if stop_event.wait(timeout=poll_interval_seconds):
            break

        cells = list_cells(base_url=base_url, cell_types=set(mean_interval_seconds_of_cell_type))
        if cells is None:
            continue
        if get_virtual_cells is not None:
            cells.extend(get_virtual_cells())

        # Record every poll so the post-run witnesses see the same stream the injector saw.
        event_log.observe(cells)

        if stop_event.is_set():
            break

        cells_of_type: dict[str, list[dict]] = {cell_type: [] for cell_type in next_injection_time_of_cell_type}
        for cell in cells:
            cells_of_type[cell_type_of(cell)].append(cell)
        for cell_type, kind_cells in sorted(cells_of_type.items()):
            max_num_cells_of_cell_type[cell_type] = max(max_num_cells_of_cell_type[cell_type], len(kind_cells))
            if _kind_is_quiescent(kind_cells, expected_num_cells=max_num_cells_of_cell_type[cell_type]):
                quiescent_polls_of_cell_type[cell_type] += 1
            else:
                quiescent_polls_of_cell_type[cell_type] = 0

        now: float = time.monotonic()
        due_types = sorted(kind for kind, due_at in next_injection_time_of_cell_type.items() if now >= due_at)
        if not due_types:
            continue

        # Inject only at a quiescent point: every replica of the kind present and serving for long
        # enough that the readings cannot all be stale. A due kind that is still recovering (or has
        # no spare replica to survive the kill) waits for a later poll.
        ready_types = [
            kind
            for kind in due_types
            if quiescent_polls_of_cell_type[kind] >= quiescent_polls_required and len(cells_of_type[kind]) > 1
        ]
        if not ready_types:
            logger.info(
                "Deferring injection: no due cell kind is quiescent with a spare replica (due %s, "
                "quiescent polls %s, replicas %s)",
                due_types,
                {kind: quiescent_polls_of_cell_type[kind] for kind in due_types},
                {kind: len(cells_of_type[kind]) for kind in due_types},
            )
            continue

        cell_type = rng.choice(ready_types)
        target = rng.choice(cells_of_type[cell_type])
        cell_name = target["metadata"]["name"]
        form = _draw_form(cell_fault_forms[cell_type], events=event_log.events, cell_type=cell_type, rng=rng)
        if injection_enabled is not None and not injection_enabled():
            continue
        try:
            form.inject(target, rng)
        except Exception:
            event_log.note_injection_attempt(
                cell_name=cell_name, form_name=form.name, succeeded=False, harmed=form.harms_cell
            )
            quiescent_polls_of_cell_type[cell_type] = 0
            logger.info("Failed to inject fault %s into %s", form.name, cell_name, exc_info=True)
            continue

        event_log.note_injection_attempt(
            cell_name=cell_name, form_name=form.name, succeeded=True, harmed=form.harms_cell
        )
        quiescent_polls_of_cell_type[cell_type] = 0
        next_injection_time_of_cell_type[cell_type] = _compute_next_injection_time(
            rng, mean_interval_seconds_of_cell_type[cell_type]
        )
        logger.info("Injected fault %s into %s", form.name, cell_name)


def _kind_is_quiescent(kind_cells: list[dict], *, expected_num_cells: int) -> bool:
    # [FUNCTION DOC] HELPER: True only if the list is non-empty, has at least the expected number
    # of cells, and every cell is alive and able to serve.
    if not kind_cells or len(kind_cells) < expected_num_cells:
        return False
    return all(cell_is_alive(cell) and _cell_can_serve(cell) for cell in kind_cells)


def _cell_can_serve(cell: dict) -> bool:
    # [FUNCTION DOC] HELPER: Non-rollout cells always count as able to serve; a rollout cell
    # counts only when its observed state is serving.
    if cell_type_of(cell) != ROLLOUT_CELL_TYPE:
        return True
    return compute_observed_cell_state(cell) is ObservedCellState.SERVING


def _draw_form(
    forms: list[BaseFaultForm], *, events: list[Event], cell_type: str, rng: random.Random
) -> BaseFaultForm:
    # [FUNCTION DOC] HELPER: Picks a fault form at random from those whose names have not yet been
    # seen to succeed for this cell type in the event history; if every form has already succeeded,
    # picks from all forms. Advances the random generator.
    worked = compute_successful_form_names(events, cell_type=cell_type)
    unproven = [form for form in forms if form.name not in worked]
    return rng.choice(unproven or forms)


def list_cells(*, base_url: str, cell_types: set[str]) -> list[dict] | None:
    # [FUNCTION DOC] Fetches the cell list from the API server cells endpoint (5 second timeout)
    # and returns the items whose cell type is in the given set. Returns none (after logging with
    # the traceback) on any failure, including network errors and bad status codes; never raises.
    try:
        resp = requests.get(f"{base_url}/api/v1/cells", timeout=5)
        resp.raise_for_status()
        return [c for c in resp.json()["items"] if cell_type_of(c) in cell_types]
    except Exception:
        logger.info("Failed to list cells from api server", exc_info=True)
        return None
