# [MODULE DOC]
# Receiving side of cell registration: the registration hub keeps the latest snapshot of cells
# per reporter, serves them as a worker provider, and drops reporters that have gone silent for
# longer than the time-to-live. Snapshots are full replacements per reporter and are ordered by
# a per-reporter sequence number, so late or duplicate snapshots are ignored. Includes the
# validation asserting that a snapshot's cells are consistently named and addressable.
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from miles.utils.misc import cancel_and_await_task
from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.naming import cell_id_of_worker, parse_cell_id
from miles.utils.workers.polling_reconcile_loop import PollingReconcileLoop
from miles.utils.workers.registration.models import RegisteredCellInfo, RegistrationSnapshot
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import NamedHostAndPorts

logger = logging.getLogger(__name__)

REGISTERED_CELLS_POLL_INTERVAL_SECONDS = 5.0
REPORTER_TTL_SECONDS = 240.0


@dataclass(kw_only=True)
class RegistrationHub(BaseWorkerProvider):
    # [CLASS DOC]
    # Worker provider whose cells come from reporters instead of from the cluster. Fields: the run
    # uuid that every accepted snapshot must carry; a clock function returning seconds (default
    # monotonic time, injectable for tests); and private state (per-reporter sequence number and
    # last ingest time, the registered cell per cell id, and whether a watcher is attached). Only
    # one cell id owner (reporter) is allowed per cell id at a time. It is used from one event
    # loop and has no locking.
    run_uuid: str
    clock: Callable[[], float] = time.monotonic
    _state_of_reporter_id: dict[str, _ReporterState] = field(init=False, default_factory=dict)
    _cell_of_id: dict[str, RegisteredCellInfo] = field(init=False, default_factory=dict)
    _watched: bool = field(init=False, default=False)

    # ========================== Taking in snapshots ===========================

    async def ingest(self, snapshot: RegistrationSnapshot) -> None:
        # [FUNCTION DOC]
        # Accept one reporter snapshot. Asserts the snapshot's run uuid equals the hub's and that
        # every cell in it is addressable (see the validation helpers; assertion errors propagate
        # to the caller). Looks up or creates the reporter state (a new reporter starts with
        # sequence number minus one). If the snapshot's sequence number is not greater than the
        # last accepted one it is logged as late and ignored. Otherwise the reporter's cells are
        # replaced by the snapshot's cells (cells it held before but no longer reports are removed;
        # asserts no cell id is claimed by another reporter), and the reporter's last ingest time
        # and sequence number are updated.
        # KNOWN ISSUE: the reporter state is created before the stale-sequence check and before the
        # cell-owner assertion, so a rejected first snapshot still leaves a registered reporter
        # entry (with no cells and ingest time zero) that the sweeper will eventually remove.
        assert (
            snapshot.run_uuid == self.run_uuid
        ), f"reporter {snapshot.reporter_id} carries run_uuid {snapshot.run_uuid!r}, expected {self.run_uuid!r}"
        _assert_snapshot_addressable(snapshot)
        cell_of_id = {cell.info.cell_id: cell for cell in snapshot.cells}

        state = self._state_of_reporter_id.setdefault(snapshot.reporter_id, _ReporterState())

        if snapshot.sequence_number <= state.sequence_number:
            logger.warning(
                f"Ignoring snapshot {snapshot.sequence_number} of reporter {snapshot.reporter_id}: snapshot "
                f"{state.sequence_number} is at least as new, so this one arrived late"
            )
            return

        self._replace_cells_of_reporter(reporter_id=snapshot.reporter_id, cell_of_id=cell_of_id)
        state.last_ingest_time = self.clock()
        state.sequence_number = snapshot.sequence_number

    def _replace_cells_of_reporter(self, *, reporter_id: str, cell_of_id: dict[str, RegisteredCellInfo]) -> None:
        # [FUNCTION DOC] HELPER: Make the reporter's cells in the hub equal to the given mapping.
        # First asserts that none of the new cell ids is currently owned by a different reporter.
        # Then removes cells owned by this reporter that are not in the mapping, and adds or
        # overwrites the mapped ones.
        for cell_id in cell_of_id:
            assert (
                owner := self._cell_of_id.get(cell_id)
            ) is None or owner.reporter_id == reporter_id, (
                f"cell {cell_id} is reported by both {owner.reporter_id} and {reporter_id}"
            )

        held = {cell_id for cell_id, cell in self._cell_of_id.items() if cell.reporter_id == reporter_id}
        for cell_id in held - set(cell_of_id):
            del self._cell_of_id[cell_id]
        self._cell_of_id.update(cell_of_id)

    # ======================= Serving the reported cells =======================

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        # [FUNCTION DOC]
        # Attach the single watcher (asserts none was attached before; a hub cannot be watched
        # twice, even after stopping). Builds a polling loop over the hub's registered cells
        # (listing a mapping from cell id to cell info) with a five-second interval and starts it,
        # which performs an initial reconcile of all current cells. Also starts a background task
        # sweeping out stale reporters. Returns one stop function (awaitable) that stops the
        # sweeper and the polling loop; if starting the sweeper fails, the already started loop is
        # stopped.
        assert not self._watched, "a registration hub reports to exactly one watcher"
        self._watched = True

        async def _list_cells() -> dict[str, CellInfo]:
            # [FUNCTION DOC] HELPER: Snapshot of the registered cells as a mapping from cell id to
            # the cell info, ignoring worker details.
            return {cell_id: cell.info for cell_id, cell in self._cell_of_id.items()}

        loop = PollingReconcileLoop(
            list_cells=_list_cells,
            poll_interval_seconds=REGISTERED_CELLS_POLL_INTERVAL_SECONDS,
        )
        async with contextlib.AsyncExitStack() as stack:
            stack.push_async_callback(await loop.start(reconcile))
            sweep_task = asyncio.create_task(self._remove_stale_reporters_forever())
            stack.push_async_callback(partial(cancel_and_await_task, sweep_task))
            return stack.pop_all().aclose

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] Return the named endpoints the reporter announced for the worker. Derives
        # the cell id from the worker name; raises a key error if that cell is not registered and
        # fails an assertion if the cell does not list that worker.
        cell = self._cell_of_id[cell_id_of_worker(worker_name)]
        worker = next((one for one in cell.workers if one.name == worker_name), None)
        assert worker is not None, (
            f"{worker_name} is not among the workers {sorted(one.name for one in cell.workers)} "
            f"reporter {cell.reporter_id} announced"
        )
        return worker.self_addrs

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        # [FUNCTION DOC] For each requested cell id, in order, return the list of worker infos the
        # reporter announced for it. Raises a key error for an unregistered cell id.
        return [self._cell_of_id[cell_id].workers for cell_id in cell_ids]

    async def _remove_stale_reporters_forever(self) -> None:
        # [FUNCTION DOC] HELPER: Forever: sleep five seconds (real asyncio sleep, not the injected
        # clock) and then remove reporters that exceeded the time-to-live.
        while True:
            await asyncio.sleep(REGISTERED_CELLS_POLL_INTERVAL_SECONDS)
            self._remove_stale_reporters()

    def _remove_stale_reporters(self) -> None:
        # [FUNCTION DOC] HELPER: Remove every reporter whose last ingest time is at or before the
        # current clock time minus the time-to-live (240 seconds), together with its cells.
        deadline = self.clock() - REPORTER_TTL_SECONDS
        for reporter_id, state in list(self._state_of_reporter_id.items()):
            if state.last_ingest_time <= deadline:
                self._remove_reporter(reporter_id)

    def _remove_reporter(self, reporter_id: str) -> None:
        # [FUNCTION DOC] HELPER: Log a warning, forget the reporter's state and delete all cells it
        # owns. Raises a key error for an unknown reporter. The change reaches the watcher at its
        # next poll.
        logger.warning(
            f"Dropping reporter {reporter_id} and every cell it announced: it has not reported for "
            f"{REPORTER_TTL_SECONDS}s, and this run serves requests to engines only while the deployment that "
            f"carries them keeps saying they are there"
        )
        del self._state_of_reporter_id[reporter_id]
        for cell_id in [cell_id for cell_id, cell in self._cell_of_id.items() if cell.reporter_id == reporter_id]:
            del self._cell_of_id[cell_id]


class _ReporterState(StrictBaseModel):
    # [CLASS DOC] HELPER: Mutable per-reporter bookkeeping: the highest accepted snapshot sequence
    # number (minus one before any) and the clock time of the last accepted snapshot.
    sequence_number: int = -1
    last_ingest_time: float = 0.0


def _assert_snapshot_addressable(snapshot: RegistrationSnapshot) -> None:
    # [FUNCTION DOC] HELPER: Assert that no cell id appears more than once in the snapshot, then
    # validate each cell with the cell check, passing the snapshot's reporter id.
    occurrences = Counter(cell.info.cell_id for cell in snapshot.cells)
    repeated = sorted(cell_id for cell_id, count in occurrences.items() if count > 1)
    assert not repeated, f"reporter {snapshot.reporter_id} carries the cells {repeated} more than once"

    for cell in snapshot.cells:
        _assert_cell_addressable(cell, reporter_id=snapshot.reporter_id)


def _assert_cell_addressable(cell: RegisteredCellInfo, *, reporter_id: str) -> None:
    # [FUNCTION DOC] HELPER: Assert all of the following for one cell, with messages naming the
    # cell: it belongs to the reporter that sent it; its cell id parses into pool id and index
    # and every worker name parses to a cell id (a parse failure becomes an assertion error); the
    # pool id in the cell id equals the cell info's pool id; it has at least one worker; all
    # workers' derived cell ids equal this cell id; worker names are unique; and the worker names
    # announced in the cell info equal (as sets after sorting) the workers actually carried.
    assert (
        cell.reporter_id == reporter_id
    ), f"snapshot of reporter {reporter_id} carries cell {cell.info.cell_id} of reporter {cell.reporter_id}"

    prefix = f"cell {cell.info.cell_id} of reporter {cell.reporter_id} is not addressable:"
    try:
        pool_id = parse_cell_id(cell.info.cell_id).pool_id
        worker_cell_ids = {cell_id_of_worker(worker.name) for worker in cell.workers}
    except ValueError as cause:
        raise AssertionError(
            f"{prefix} its cell id or a worker name does not read as <pool id>-<cell index>"
        ) from cause

    assert pool_id == cell.info.pool_id, f"{prefix} its cell id names pool {pool_id}, not {cell.info.pool_id}"
    assert cell.workers, f"{prefix} it carries no worker"
    assert worker_cell_ids == {
        cell.info.cell_id
    }, f"{prefix} its workers {sorted(one.name for one in cell.workers)} name cells {sorted(worker_cell_ids)}"

    names = sorted(worker.name for worker in cell.workers)
    assert len(set(names)) == len(names), f"{prefix} its workers carry the names {names}, one of them twice"
    assert (
        sorted(cell.info.worker_names) == names
    ), f"{prefix} it announces the workers {sorted(cell.info.worker_names)}, but carries {names}"
