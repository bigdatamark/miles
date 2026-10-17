# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC]
# Exponential-backoff retry scheduling for failed reconciles. Failures are recorded per key; a
# background poller wakes up every second on the injected clock and fires a callback for each
# key whose retry time has arrived. The callback is expected to re-queue the key.
from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Generic, TypeVar

from miles.utils.test_utils.clock import Clock

KeyT = TypeVar("KeyT", bound=Hashable)

POLL_INTERVAL = 1.0


@dataclass
class _RetryInfo:
    # [CLASS DOC] HELPER: Per-key retry state: the consecutive failure count and the clock time at
    # which to retry (none once the retry has been fired and no new failure recorded yet).
    failures: int
    retry_at: float | None


class RetryScheduler(Generic[KeyT]):
    # [CLASS DOC]
    # Tracks consecutive failures per hashable key and fires a retry callback when the backoff for
    # a key elapses. Must be created while an event loop is running, since construction starts the
    # poller task. Timing resolution is the one-second poll interval, so retries fire up to one
    # interval late. After shutdown, new failures are ignored.
    def __init__(
        self,
        *,
        on_retry: Callable[[KeyT], None],
        failure_base_delay: float,
        failure_max_delay: float,
        clock: Clock,
    ) -> None:
        # [FUNCTION DOC]
        # Store the retry callback (called with the key), the first-failure delay in seconds, the
        # delay cap in seconds and the clock. Asserts the base delay is positive and the cap is at
        # least the base. Precomputes the largest useful backoff exponent: the base-two logarithm
        # of cap over base, rounded up, and not below zero. Starts the background poller task.
        assert failure_base_delay > 0, f"{failure_base_delay=} must be positive"
        assert failure_max_delay >= failure_base_delay, f"{failure_max_delay=} must be >= {failure_base_delay=}"

        self._on_retry = on_retry
        self._failure_base_delay = failure_base_delay
        self._failure_max_delay = failure_max_delay
        self._max_backoff_exponent = max(0, math.ceil(math.log2(failure_max_delay / failure_base_delay)))
        self._clock = clock

        self._infos: dict[KeyT, _RetryInfo] = {}
        self._poller = asyncio.create_task(self._poll())
        self._shutdown = False

    def note_failure(self, key: KeyT) -> None:
        # [FUNCTION DOC]
        # Record a failure for the key (ignored after shutdown). The failure count becomes one more
        # than before (one for a new key). The delay is the base delay times two to the power of
        # failures minus one (the power capped at the precomputed exponent), also capped at the
        # maximum delay. Schedules the retry at the clock's current time plus that delay, replacing
        # any earlier schedule for the key.
        if self._shutdown:
            return

        info = self._infos.get(key)
        failures = 1 if info is None else info.failures + 1
        exponent = min(failures - 1, self._max_backoff_exponent)
        delay = min(self._failure_base_delay * 2**exponent, self._failure_max_delay)

        self._infos[key] = _RetryInfo(failures=failures, retry_at=self._clock.time() + delay)

    def note_success(self, key: KeyT) -> None:
        # [FUNCTION DOC] Forget the key's failure history and any pending retry (no-op if unknown).
        self._infos.pop(key, None)

    async def shutdown(self) -> None:
        # [FUNCTION DOC] Mark as shut down, drop all retry state, cancel the poller task and wait
        # for it to finish, ignoring its errors. Pending retries never fire.
        self._shutdown = True
        self._infos = {}

        self._poller.cancel()
        await asyncio.gather(self._poller, return_exceptions=True)

    async def _poll(self) -> None:
        # [FUNCTION DOC] HELPER: Forever: sleep one second on the clock, then take every key whose
        # retry time is due (not none and at or before now), clear that key's retry time (so it does
        # not fire again until a new failure is noted) and call the retry callback with it. The key's
        # failure count is kept, so the next failure backs off further until a success.
        # An exception from the callback would end the poller.
        while True:
            await self._clock.sleep(POLL_INTERVAL)

            now = self._clock.time()
            due = [key for key, info in self._infos.items() if info.retry_at is not None and info.retry_at <= now]
            for key in due:
                self._infos[key].retry_at = None
                self._on_retry(key)
