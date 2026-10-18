# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

# [MODULE DOC] Public entry point for running the fault injector in the background. Defines the
# fixed API server port used by FT tests (18080) and the 180 second stop-and-join timeout (a pod
# deletion is the slowest form, cannot be cancelled, and is two bounded kubectl calls).

import threading
from collections.abc import Callable

from tests.e2e.ft.conftest_ft.fault_injection.core import (
    POLL_INTERVAL_SECONDS,
    QUIESCENT_POLLS_REQUIRED,
    list_cells,
    run_fault_injection_loop,
)
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.state import EventLog

from miles.utils.test_utils.polling_worker import PollingWorker

API_SERVER_PORT: int = 18080
# A pod deletion, the slowest form, cannot be cancelled and is two kubectl calls bounded at a minute.
STOP_AND_JOIN_TIMEOUT_SECONDS: float = 180.0


class FaultInjectorHandle:
    # [CLASS DOC] Owns a background polling worker thread (named for the random fault injector)
    # running the fault injection loop, plus the event log it fills (public attribute) and the
    # fault forms in use (public attribute). Lifecycle: construct, start, then stop and join.
    def __init__(
        self,
        *,
        base_url: str,
        seed: int,
        mean_interval_seconds_of_cell_type: dict[str, float],
        cell_fault_forms: CellFaultForms,
        get_virtual_cells: Callable[[], list[dict]] | None = None,
        injection_enabled: Callable[[], bool] | None = None,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
        quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED,
    ) -> None:
        # [FUNCTION DOC] Stores configuration, creates a fresh event log, and creates (without
        # starting) the worker. Parameters are forwarded to the injection loop unchanged: base url,
        # seed, mean seconds between injections per cell type (its keys also define which cell
        # types are listed), fault forms per cell type, optional virtual cells supplier, optional
        # injection-enabled predicate, poll interval, quiescent polls required.
        self.event_log = EventLog()
        self.cell_fault_forms = cell_fault_forms
        self._base_url = base_url
        self._cell_types: set[str] = set(mean_interval_seconds_of_cell_type)
        self._get_virtual_cells: Callable[[], list[dict]] | None = get_virtual_cells

        def inject_until_stopped(stop_event: threading.Event) -> None:
            # [FUNCTION DOC] HELPER: Worker body; runs the injection loop with the captured settings
            # and this handle's event log until the stop event is set.
            run_fault_injection_loop(
                base_url=base_url,
                seed=seed,
                mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
                stop_event=stop_event,
                event_log=self.event_log,
                cell_fault_forms=cell_fault_forms,
                get_virtual_cells=get_virtual_cells,
                injection_enabled=injection_enabled,
                poll_interval_seconds=poll_interval_seconds,
                quiescent_polls_required=quiescent_polls_required,
            )

        self._worker = PollingWorker(name="ft-random-fault-injector", run=inject_until_stopped)

    def start(self) -> None:
        # [FUNCTION DOC] Starts the background worker thread.
        self._worker.start()

    def stop_and_join(self) -> None:
        # [FUNCTION DOC] Asks the worker to stop and waits up to 180 seconds for it; then asserts the
        # worker is no longer running (failing with an explanatory message about a possibly
        # still-crashing cell otherwise); finally records one last snapshot of the cells in the
        # event log so post-run witnesses see the final state.
        self._worker.stop_and_join(timeout_seconds=STOP_AND_JOIN_TIMEOUT_SECONDS)
        self._worker.assert_not_running(
            message=(
                f"The fault injector was still mid-injection {STOP_AND_JOIN_TIMEOUT_SECONDS}s after being asked to "
                f"stop: it may still crash a cell nothing will heal, and reading its log would race it"
            )
        )
        self._observe_final_snapshot()

    def _observe_final_snapshot(self) -> None:
        # [FUNCTION DOC] HELPER: Lists cells once (plus virtual cells if a supplier exists) and
        # records them in the event log. Silently does nothing if listing fails.
        cells = list_cells(base_url=self._base_url, cell_types=self._cell_types)
        if cells is None:
            return
        if self._get_virtual_cells is not None:
            cells.extend(self._get_virtual_cells())
        self.event_log.observe(cells)


def spawn_fault_injector(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds_of_cell_type: dict[str, float],
    cell_fault_forms: CellFaultForms,
    get_virtual_cells: Callable[[], list[dict]] | None = None,
    injection_enabled: Callable[[], bool] | None = None,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED,
) -> FaultInjectorHandle:
    # [FUNCTION DOC] Convenience factory: builds a handle from the same parameters as the handle
    # constructor, starts it, and returns it (already running).
    handle = FaultInjectorHandle(
        base_url=base_url,
        seed=seed,
        mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
        cell_fault_forms=cell_fault_forms,
        get_virtual_cells=get_virtual_cells,
        injection_enabled=injection_enabled,
        poll_interval_seconds=poll_interval_seconds,
        quiescent_polls_required=quiescent_polls_required,
    )
    handle.start()
    return handle
