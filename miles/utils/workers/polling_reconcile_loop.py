# [MODULE DOC]
# A polling-based cell watcher: periodically lists all cells through a supplied async function,
# diffs the result against what it last reconciled, and invokes a reconcile callback for each
# cell that appeared, changed or disappeared. Used for backends that cannot push cell events.
import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import partial

from miles.utils.misc import cancel_and_await_task
from miles.utils.workers.worker_provider.base import CellInfo, CellReconcileFn, StopWatchFn

logger = logging.getLogger(__name__)

ListCellsFn = Callable[[], Awaitable[dict[str, CellInfo]]]


class PollingReconcileLoop:
    # [CLASS DOC]
    # Polls the cell listing function every fixed interval and reconciles differences. The
    # per-cell state last successfully reconciled is kept by the running watch, not the object,
    # so one instance may start several independent watches.
    def __init__(self, *, list_cells: ListCellsFn, poll_interval_seconds: float) -> None:
        # [FUNCTION DOC]
        # Store the async listing function (returning a mapping from cell id to cell info) and the
        # poll interval in seconds.
        self._list_cells = list_cells
        self._poll_interval_seconds = poll_interval_seconds

    async def start(self, reconcile: CellReconcileFn) -> StopWatchFn:
        # [FUNCTION DOC]
        # Start watching. First performs one poll synchronously, so reconcile is called for every
        # existing cell and any failure of that first poll propagates to the caller. Then spawns a
        # background task that polls forever at the interval. Returns a stop function (awaitable)
        # that cancels the background task and waits for it to finish.
        seen_infos: dict[str, CellInfo] = {}
        # the initial sync must complete (and raise on failure) before the watch is considered established
        await self._poll_once(reconcile, seen_infos=seen_infos)
        task = asyncio.create_task(self._watch_loop(reconcile, seen_infos))
        return partial(cancel_and_await_task, task)

    async def _watch_loop(self, reconcile: CellReconcileFn, seen_infos: dict[str, CellInfo]) -> None:
        # [FUNCTION DOC] HELPER: Background loop: sleep for the interval, then poll once; any
        # exception from a poll is logged and swallowed so the loop keeps running. The sleep
        # happens before the poll, so the first background poll occurs one interval after start.
        # Runs until cancelled.
        while True:
            await asyncio.sleep(self._poll_interval_seconds)
            try:
                await self._poll_once(reconcile, seen_infos=seen_infos)
            except Exception:
                logger.exception("Worker provider poll failed; retrying")

    async def _poll_once(self, reconcile: CellReconcileFn, *, seen_infos: dict[str, CellInfo]) -> None:
        # [FUNCTION DOC] HELPER: List the current cells, then walk the union of previously seen and
        # currently observed cell ids in sorted order. For each cell whose observed info (absent if
        # it vanished) differs from the last seen one, call reconcile with the cell id and the
        # observed info or nothing for a vanished cell. On success update the seen mapping (remove
        # vanished cells, record the rest). A failing reconcile is logged, leaves that cell's seen
        # state unchanged so it is retried next poll, and does not stop the other cells; after the
        # pass the first collected error is re-raised. Mutates the seen mapping in place.
        observed_infos = await self._list_cells()
        first_error: Exception | None = None
        for cell_id in sorted(set(seen_infos) | set(observed_infos)):
            observed_info = observed_infos.get(cell_id)
            if seen_infos.get(cell_id) == observed_info:
                continue
            try:
                await reconcile(cell_id, observed_info)
            except Exception as error:
                logger.exception(f"Reconciling cell {cell_id} failed; continuing with the other cells")
                first_error = first_error or error
                continue
            if observed_info is None:
                seen_infos.pop(cell_id, None)
            else:
                seen_infos[cell_id] = observed_info

        if first_error is not None:
            raise first_error
