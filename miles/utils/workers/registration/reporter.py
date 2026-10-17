# [MODULE DOC]
# Sending side of cell registration. A reporter watches the cells of its own deployment through a
# worker provider and periodically (and soon after any change) sends a full snapshot of them to
# the registration hub over a worker handle. A worker wrapper runs the reporter on a background
# event loop thread and kills the whole process if the reporter ever stops, so a deployment never
# stays up while silently not registering.
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import random
from dataclasses import dataclass, field

from miles.utils.async_utils import AsyncLoopThread
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.logging_utils import configure_logger
from miles.utils.workers.registration.models import RegisteredCellInfo, RegistrationSnapshot
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo

logger = logging.getLogger(__name__)

SNAPSHOT_INTERVAL_SECONDS = 15.0
SNAPSHOT_JITTER_RATIO = 0.2
SNAPSHOT_DEBOUNCE_SECONDS = 1.0
HUB_READY_TIMEOUT_SECONDS = 3600.0
INGEST_TIMEOUT_SECONDS = 60.0


@dataclass(kw_only=True)
class RegistrationReporter:
    # [CLASS DOC]
    # Keyword-only dataclass. Fields: the run uuid and the reporter id put into each snapshot; the
    # handle of the hub (to send snapshots through and to await readiness); the worker provider
    # that supplies the observed cells and their worker infos; and private state: a debounced
    # interval trigger (default: every 15 seconds with plus or minus 20 percent jitter, and sooner
    # after a change with 1 second debounce), the last observed info per cell id and the snapshot
    # sequence number (starts at zero, so the first snapshot sent is number one).
    run_uuid: str
    reporter_id: str
    hub_endpoint: BaseWorkerHandle
    worker_provider: BaseWorkerProvider
    _trigger: _DebouncedIntervalTrigger = field(
        default_factory=lambda: _DebouncedIntervalTrigger(
            interval_seconds=SNAPSHOT_INTERVAL_SECONDS,
            jitter_ratio=SNAPSHOT_JITTER_RATIO,
            debounce_seconds=SNAPSHOT_DEBOUNCE_SECONDS,
        )
    )
    _info_of_cell_id: dict[str, CellInfo] = field(init=False, default_factory=dict)
    _sequence_number: int = field(init=False, default=0)

    async def run(self) -> None:
        # [FUNCTION DOC]
        # Main loop; runs until cancelled or an unexpected error escapes. Steps: initialize the
        # worker provider; wait for the hub to be ready (up to 3600 seconds); start watching cells
        # with the observer callback, which fills the observed map and pokes the trigger. Then
        # forever: wait for the trigger and send a snapshot. If sending shows the hub restarted
        # (server restarted error), log a warning and wait for hub readiness again, this time
        # accepting the changed server identity. Any other ordinary error while sending is logged
        # as a warning and the loop continues (the next trigger retries). On exit, always stops the
        # cell watch.
        await self.worker_provider.init()
        await self.hub_endpoint.wait_ready(timeout=HUB_READY_TIMEOUT_SECONDS)
        stop_watch = await self.worker_provider.watch_cells(self._observe)

        try:
            logger.info(
                f"Reporter {self.reporter_id} observes {len(self._info_of_cell_id)} cells of its own deployment and "
                f"reports them every {self._trigger.interval_seconds}s"
            )
            while True:
                await self._trigger.wait()
                try:
                    await self._send_once()
                except ServerRestartedError:
                    logger.warning(
                        f"The hub {self.reporter_id} reports into is a new process, so its readiness is awaited "
                        f"again before the next snapshot",
                        exc_info=True,
                    )
                    await self.hub_endpoint.wait_ready(
                        timeout=HUB_READY_TIMEOUT_SECONDS, allow_server_uuid_change=True
                    )
                except Exception:
                    logger.warning(f"Reporting the cells of {self.reporter_id} failed", exc_info=True)
        finally:
            await stop_watch()

    async def _send_once(self) -> None:
        # [FUNCTION DOC] HELPER: Increment the sequence number (also when sending later fails, so
        # numbers may skip), build a snapshot of the observed cells and call the hub's ingest
        # method through the handle, waiting at most 60 seconds (a timeout raises).
        self._sequence_number += 1
        snapshot = self._compute_snapshot()
        await asyncio.wait_for(
            self.hub_endpoint.registration_ingest(snapshot=snapshot), timeout=INGEST_TIMEOUT_SECONDS
        )

    async def _observe(self, cell_id: str, observed: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: Cell watch callback. Removes the cell from the observed map when
        # the observed info is none (cell gone), otherwise stores it, then notifies the trigger so a
        # snapshot is sent soon.
        if observed is None:
            self._info_of_cell_id.pop(cell_id, None)
        else:
            self._info_of_cell_id[cell_id] = observed
        self._trigger.notify()

    def _compute_snapshot(self) -> RegistrationSnapshot:
        # [FUNCTION DOC] HELPER: Build a snapshot from the run uuid, reporter id, current sequence
        # number and the cells made from the observed infos (with worker infos fetched from the
        # worker provider).
        return RegistrationSnapshot(
            run_uuid=self.run_uuid,
            reporter_id=self.reporter_id,
            sequence_number=self._sequence_number,
            cells=_compute_cells(
                self._info_of_cell_id, reporter_id=self.reporter_id, worker_provider=self.worker_provider
            ),
        )


class RegistrationReporterWorker:
    # [CLASS DOC]
    # Serve-style worker that hosts a reporter. Constructing it starts everything; it has no other
    # behavior.
    def __init__(self, *, args, reporter: RegistrationReporter) -> None:
        # [FUNCTION DOC]
        # Configure logging with a registration-reporter process identity, start a background
        # event loop thread, and submit the reporter's run coroutine to it. When that run finishes
        # for any reason (including failure) the process exit hook is called, which terminates the
        # whole process.
        configure_logger(args, source=SimpleProcessIdentity(component="registration_reporter"))

        self._loop_thread = AsyncLoopThread()
        self._loop_thread.submit(reporter.run()).add_done_callback(_exit_on_reporter_stop)


@dataclass(kw_only=True)
class _DebouncedIntervalTrigger:
    # [CLASS DOC] HELPER: Wakes a single consumer either after a jittered interval or, if a change
    # was notified, shortly after it (debounced so bursts of changes produce one wake). Fields:
    # base interval, jitter ratio (the interval is multiplied by a random factor within one plus or
    # minus the ratio), debounce delay, a random generator (injectable for tests) and an internal
    # changed signal.
    interval_seconds: float
    jitter_ratio: float
    debounce_seconds: float
    rng: random.Random = field(default_factory=random.Random)
    _changed: asyncio.Event = field(init=False, default_factory=asyncio.Event)

    def notify(self) -> None:
        # [FUNCTION DOC] HELPER: Record that something changed, waking a pending wait.
        self._changed.set()

    async def wait(self) -> None:
        # [FUNCTION DOC] HELPER: Wait for a change for at most one freshly jittered interval. On
        # timeout return at once. If a change arrived: clear the signal, sleep the debounce delay,
        # clear the signal again (absorbing changes during the debounce), and return. Note that a
        # change signalled before the call makes the wait return immediately, then debounce.
        try:
            await asyncio.wait_for(self._changed.wait(), timeout=self._compute_next_interval_seconds())
        except (TimeoutError, asyncio.TimeoutError):
            return
        self._changed.clear()
        await asyncio.sleep(self.debounce_seconds)
        self._changed.clear()

    def _compute_next_interval_seconds(self) -> float:
        # [FUNCTION DOC] HELPER: The base interval times one plus a uniform random number between
        # minus and plus the jitter ratio.
        return self.interval_seconds * (1.0 + self.rng.uniform(-self.jitter_ratio, self.jitter_ratio))


def _compute_cells(
    info_of_cell_id: dict[str, CellInfo], *, reporter_id: str, worker_provider: BaseWorkerProvider
) -> list[RegisteredCellInfo]:
    # [FUNCTION DOC] HELPER: Build registered cell records for all observed cells in sorted cell id
    # order: asks the worker provider for the worker infos of all those cells in one call and pairs
    # each cell info with its worker list under the reporter id. Raises a value error if the
    # provider returns a different number of lists than cells requested.
    observed = sorted(info_of_cell_id.items())
    worker_infos_per_cell = worker_provider.get_worker_infos(cell_ids=[cell_id for cell_id, _ in observed])
    return [
        RegisteredCellInfo(reporter_id=reporter_id, info=info, workers=worker_infos)
        for (_cell_id, info), worker_infos in zip(observed, worker_infos_per_cell, strict=True)
    ]


def _exit_on_reporter_stop(future: concurrent.futures.Future[None]) -> None:
    # [FUNCTION DOC] HELPER: Done-callback of the reporter run. Logs an error with the future's
    # exception (reading it raises if the future was cancelled) and immediately exits the process
    # with status 1 without cleanup, so the deployment does not look healthy while unregistered.
    logger.error(
        "The registration reporter of this deployment stopped, so the deployment exits with it: nothing else here "
        "registers these cells into the run, and a pod that keeps running would look healthy while the run waits "
        "for cells that are never announced",
        exc_info=future.exception(),
    )
    os._exit(1)
