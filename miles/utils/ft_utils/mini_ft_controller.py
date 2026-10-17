# [MODULE DOC]
# Mini fault tolerance controller: a small loop that polls the fault tolerance api server for cells
# and heals unhealthy ones by suspending them, waiting, and (unless the platform resumes cells
# automatically) resuming them. Layers: an entry point that starts the controller on a daemon thread
# with its own event loop; an http runner translating api calls; pure data models; and a core
# controller that depends only on injected async callables and a clock. Failed healing attempts back
# off exponentially per cell.
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from miles.utils.ft_utils.api_server.models import Cell, CellList, CellPatch, CellPatchSpec, TriState
from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.types import ClusterBackend

logger = logging.getLogger(__name__)


# ------------------------ entrypoint ------------------------


def maybe_start_mini_ft_controller(args: Any) -> None:
    # [FUNCTION DOC] Does nothing unless the enable flag is set on args. Otherwise builds a runner
    # for the local api server (loopback, port from args) with the poll interval and resume delay
    # from args; cells auto-resume is enabled exactly when the cluster backend is kubernetes. Starts
    # a daemon thread running the runner in a fresh event loop and returns immediately. Exceptions
    # in the thread are not propagated (the controller logs and the thread dies).
    if not args.mini_ft_controller_enable:
        return

    runner = _MiniFTControllerRunner(
        api_server_url=f"http://127.0.0.1:{args.api_server_port}",
        poll_interval=args.mini_ft_controller_poll_interval,
        resume_delay=args.mini_ft_controller_resume_delay,
        cells_auto_resume=ClusterBackend(args.cluster_backend) == ClusterBackend.KUBERNETES,
    )

    def _run() -> None:
        # [FUNCTION DOC] HELPER: Thread body; runs the runner to completion in a new event loop.
        asyncio.run(runner.run())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info("Started mini FT controller on daemon thread")


# ------------------------ HTTP transport + thread runner ------------------------


class _MiniFTControllerRunner:
    # [CLASS DOC] HELPER: Adapts the core controller to the http api: owns an async http client and
    # provides the get-cells, suspend and resume callables.
    def __init__(
        self,
        *,
        api_server_url: str,
        poll_interval: float,
        resume_delay: float,
        cells_auto_resume: bool,
    ) -> None:
        # [FUNCTION DOC] HELPER: Creates an async http client with the base url (trailing slash
        # removed) and a 30 second timeout, and a core controller wired to this runner's methods
        # with the given intervals and auto resume flag.
        url = api_server_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=url, timeout=30.0)
        self._controller = _MiniFTController(
            get_cells=self._get_cells,
            suspend_cell=self._suspend_cell,
            resume_cell=self._resume_cell,
            poll_interval=poll_interval,
            resume_delay=resume_delay,
            cells_auto_resume=cells_auto_resume,
        )

    async def run(self) -> None:
        # [FUNCTION DOC] HELPER: Runs the controller until it ends or fails, then always closes the
        # http client.
        try:
            await self._controller.run()
        finally:
            await self._client.aclose()

    async def _get_cells(self) -> list[_CellSnapshot]:
        # [FUNCTION DOC] HELPER: Fetches the cell list from the api, raises for http error statuses,
        # validates the body as a cell list, and converts each cell into a health snapshot.
        resp = await self._client.get("/api/v1/cells")
        resp.raise_for_status()
        cell_list = CellList.model_validate(resp.json())
        return [_compute_cell_snapshot(cell) for cell in cell_list.items]

    async def _suspend_cell(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Patches the cell's suspend flag to true.
        await self._patch_cell_suspend(name=name, suspend=True)

    async def _resume_cell(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Patches the cell's suspend flag to false.
        await self._patch_cell_suspend(name=name, suspend=False)

    async def _patch_cell_suspend(self, *, name: str, suspend: bool) -> None:
        # [FUNCTION DOC] HELPER: Sends an http patch with a json body setting the spec suspend flag
        # for the named cell; raises for http error statuses.
        patch = CellPatch(spec=CellPatchSpec(suspend=suspend))
        resp = await self._client.patch(
            f"/api/v1/cells/{name}",
            content=patch.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()


def _compute_cell_snapshot(cell: Cell) -> _CellSnapshot:
    # [FUNCTION DOC] HELPER: Reduces a cell resource to name and health. Unhealthy if any Healthy
    # condition has status false, or if the phase is Suspended. Otherwise not applicable if there is
    # no Healthy condition at all, else healthy (unknown health counts as healthy).
    healthy_conditions = [c for c in cell.status.conditions if c.type == "Healthy"]
    if any(c.status == TriState.FALSE for c in healthy_conditions):
        status = CellHealthStatus.UNHEALTHY
    elif cell.status.phase == "Suspended":
        status = CellHealthStatus.UNHEALTHY
    elif not healthy_conditions:
        status = CellHealthStatus.NOT_APPLICABLE
    else:
        status = CellHealthStatus.HEALTHY
    return _CellSnapshot(name=cell.metadata.name, status=status)


# ------------------------ data models ------------------------


class CellHealthStatus(str, Enum):
    # [CLASS DOC] Coarse health classes of a cell: healthy, unhealthy, not applicable.
    HEALTHY = "Healthy"
    UNHEALTHY = "Unhealthy"
    NOT_APPLICABLE = "NotApplicable"


class _CellSnapshot(StrictBaseModel):
    # [CLASS DOC] HELPER: Name and coarse health of one cell.
    name: str
    status: CellHealthStatus


@dataclass
class _CellBackoff:
    # [CLASS DOC] HELPER: Per-cell healing retry state: consecutive failed heal attempts and the
    # earliest controller-clock time of the next attempt.
    consecutive_failures: int = 0
    next_attempt_at: float = 0.0


# ------------------------ core controller (pure async, no HTTP) ------------------------


class _MiniFTController:
    # [CLASS DOC] HELPER: Core healing loop with injected dependencies. Every poll interval (minus
    # the time the poll took) it lists cells and heals each unhealthy cell whose backoff time has
    # arrived. Backoff entries exist only for currently unhealthy cells; entries of cells that are no
    # longer unhealthy are dropped each poll.
    def __init__(
        self,
        *,
        get_cells: Callable[[], Awaitable[list[_CellSnapshot]]],
        suspend_cell: Callable[[str], Awaitable[None]],
        resume_cell: Callable[[str], Awaitable[None]],
        poll_interval: float,
        resume_delay: float,
        cells_auto_resume: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the async callables (list cells, suspend a cell by name,
        # resume a cell by name), the poll interval and resume delay in seconds, the auto resume flag
        # (when true the platform resumes suspended cells so the controller never resumes), and a
        # monotonic clock (default monotonic time). Starts not running with no backoffs.
        self._get_cells = get_cells
        self._suspend_cell = suspend_cell
        self._resume_cell = resume_cell
        self._poll_interval = poll_interval
        self._resume_delay = resume_delay
        self._cells_auto_resume = cells_auto_resume
        self._clock = clock

        self._running: bool = False
        self._cell_backoffs: dict[str, _CellBackoff] = {}

    async def run(self) -> None:
        # [FUNCTION DOC] HELPER: Sets running and loops while running: polls and heals, then sleeps
        # for the remainder of the poll interval (never negative). Logs and re-raises unexpected
        # exceptions (the poll step itself swallows its own errors, so these come from sleeping or
        # cancellation handling).
        try:
            self._running = True
            while self._running:
                start = self._clock()
                await self._poll_and_heal()
                elapsed = self._clock() - start
                await asyncio.sleep(max(0.0, self._poll_interval - elapsed))
        except Exception:
            logger.error("Error in run", exc_info=True)
            raise

    def request_stop(self) -> None:
        # [FUNCTION DOC] HELPER: Clears the running flag; the loop ends after its current iteration
        # (including the sleep).
        self._running = False

    async def _poll_and_heal(self) -> None:
        # [FUNCTION DOC] HELPER: One iteration. Fetches cells and logs their statuses. For each
        # unhealthy cell: ensures a backoff entry, skips it if its next attempt time has not
        # arrived, else heals it (sequentially, awaiting each). Afterwards removes backoff entries of
        # cells that are not unhealthy in this poll (including cells that vanished). Any exception
        # is logged and swallowed so the loop continues.
        try:
            cells = await self._get_cells()
            log_structured(
                logger.info,
                tag="ft",
                op="controller",
                phase="poll",
                cells=",".join(f"{c.name}:{c.status.value}" for c in cells),
            )

            unhealthy_names: set[str] = set()
            for cell in cells:
                if cell.status != CellHealthStatus.UNHEALTHY:
                    continue

                unhealthy_names.add(cell.name)
                backoff = self._cell_backoffs.setdefault(cell.name, _CellBackoff())

                now = self._clock()
                if now < backoff.next_attempt_at:
                    continue

                await self._heal(cell_name=cell.name, backoff=backoff)

            stale_keys = set(self._cell_backoffs) - unhealthy_names
            for key in stale_keys:
                del self._cell_backoffs[key]
        except Exception:
            logger.error("Error in _poll_and_heal", exc_info=True)

    async def _heal(self, *, cell_name: str, backoff: _CellBackoff) -> None:
        # [FUNCTION DOC] HELPER: Heals one cell: suspend, sleep for the resume delay, resume (skipped
        # when cells auto resume). On success resets the failure count and sets the next attempt time
        # to now plus the resume delay (a cooldown). On any exception increments the failure count and
        # sets the next attempt after an exponential delay of 5 seconds times two to the failure
        # count, capped at 300 seconds, logging a warning. The sleep inside blocks the whole
        # polling loop, so other cells are not polled during healing.
        try:
            log_structured(logger.info, tag="ft", op="heal", phase="suspend", cell=cell_name)
            await self._suspend_cell(cell_name)

            log_structured(
                logger.info, tag="ft", op="heal", phase="sleep", cell=cell_name, resume_delay_s=self._resume_delay
            )
            await asyncio.sleep(self._resume_delay)

            if not self._cells_auto_resume:
                log_structured(logger.info, tag="ft", op="heal", phase="resume", cell=cell_name)
                await self._resume_cell(cell_name)

            backoff.consecutive_failures = 0
            backoff.next_attempt_at = self._clock() + self._resume_delay
            log_structured(
                logger.info,
                tag="ft",
                op="heal",
                phase="done",
                cell=cell_name,
                cooldown_until=round(backoff.next_attempt_at),
            )
        except Exception:
            backoff.consecutive_failures += 1
            delay = min(5 * (2**backoff.consecutive_failures), 300)
            backoff.next_attempt_at = self._clock() + delay
            log_structured(
                logger.warning,
                tag="ft",
                op="heal",
                phase="fail",
                cell=cell_name,
                attempt=backoff.consecutive_failures,
                next_attempt_in_s=round(delay),
                exc_info=True,
            )
