# [MODULE DOC]
# Abstract client-side handle to a worker, independent of how the worker is reached (ray actor or
# http rpc). Defines the readiness, idleness, fire-and-forget and death-probing contract, plus the
# errors used to signal an unreachable or busy worker. Concrete handles also expose the worker's
# own methods as async callables.
from __future__ import annotations

import abc
import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_WAIT_DEAD_PROBE_INTERVAL_SECONDS = 1.0


class WorkerUnreachableError(Exception):
    # [CLASS DOC] Raised when the worker cannot be reached (died, not ready in time, submit failed).
    pass


class WorkerStillBusyError(Exception):
    # [CLASS DOC] Raised when the worker is still running calls where an idle worker was required.
    pass


class BaseWorkerHandle(abc.ABC):
    # [CLASS DOC]
    # Contract for worker handles. Subclasses implement readiness waiting and death probing; idle
    # waiting and fire-and-forget submission are optional and raise not-implemented by default.
    # [FUNCTION DOC] (wait_ready, abstract, block sits above the decorator) Wait until the worker
    # is ready to take calls, for at most the timeout in seconds; raise the unreachable error if it
    # is not. The flag allows the worker to be a restarted process with a new server identity.
    @abc.abstractmethod
    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None: ...

    async def wait_idle(self, *, timeout: float) -> None:
        # [FUNCTION DOC] Wait until the worker runs no call, for at most the timeout in seconds.
        # Default: raises not-implemented, naming the handle class.
        raise NotImplementedError(f"{type(self).__name__} cannot tell whether the worker is running a call")

    async def submit_without_result(self, method_name: str, /, **kwargs: Any) -> None:
        # [FUNCTION DOC] Start the named worker method with keyword arguments without waiting for
        # or ever fetching its result. Default: raises not-implemented, naming the handle class.
        raise NotImplementedError(f"{type(self).__name__} cannot submit a call it will never get an answer to")

    async def wait_dead(self, *, timeout: float) -> None:
        # [FUNCTION DOC] Poll the death probe once per second until it reports dead, for at most the
        # timeout in seconds. Returns normally in both cases; on timeout it logs an error and
        # proceeds, so callers cannot tell whether the worker really died.
        deadline = time.monotonic() + timeout
        while True:
            if await self.probe_is_dead():
                return
            if time.monotonic() >= deadline:
                logger.error("Timed out after %.0fs waiting for %r to die; proceeding anyway", timeout, self)
                return
            await asyncio.sleep(_WAIT_DEAD_PROBE_INTERVAL_SECONDS)

    # [FUNCTION DOC] (probe_is_dead, abstract, block sits above the decorator) Check once whether
    # the worker is definitely dead. True only on positive evidence; false when alive or
    # inconclusive.
    @abc.abstractmethod
    async def probe_is_dead(self) -> bool: ...
