# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC]
# The reconcile loop: wires a source stream driver, an object store, a deduplicating work queue,
# a retry scheduler with backoff and a single worker task into one level-triggered controller.
# Source events update the store; each affected parent key is queued; the worker calls the
# user's reconcile function once per queued key; failures are retried with backoff; a periodic
# resync re-queues every known parent key.
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from miles.utils.test_utils.clock import Clock, RealClock
from miles.utils.workers.reconcile.object_store import KeyMapFn, ObjectStore
from miles.utils.workers.reconcile.retry_scheduler import RetryScheduler
from miles.utils.workers.reconcile.source_event import ParentKey, SourceWatchFn
from miles.utils.workers.reconcile.source_stream_driver import SourceStreamDriver
from miles.utils.workers.reconcile.work_queue import WorkQueue

logger = logging.getLogger(__name__)

ReconcileFn = Callable[[ParentKey], Awaitable[None]]

DEFAULT_RESYNC_PERIOD = 10 * 60 * 60.0


class ReconcileLoop:
    # [CLASS DOC]
    # Single-use controller object, usable as an async context manager (enter starts, exit
    # stops). Must be started exactly once and stopped after starting. Owns the store, queue,
    # retry scheduler, driver and the background tasks (worker, driver and optional resync).
    """A source stream feeds a store; every changed parent key is reconciled once, level-triggered.

    Args:
        source: Opens every stream with a `ReplaceEvent` carrying the whole world, and is
            restartable: a stream that ends or fails is reopened.
        reconcile: Takes a parent key only and re-derives from `get_by_parent()`, whose objects
            are read-only. Must be idempotent, since delivery is at-least-once, and must not block
            on I/O, since one worker serves every parent key.
    """

    def __init__(
        self,
        *,
        source: SourceWatchFn,
        reconcile: ReconcileFn,
        key_map: KeyMapFn | None = None,
        resync_period: float | None = DEFAULT_RESYNC_PERIOD,
        failure_base_delay: float = 1.0,
        failure_max_delay: float = 60.0,
        source_retry_delay: float = 1.0,
        clock: Clock | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Build the components. Parameters: the source watch function; the reconcile function
        # (async, takes a parent key); an optional function mapping an object to its parent key
        # (objects mapped to none are ignored; without it, an object's own key is its parent key);
        # the resync period in seconds (default ten hours, or none to disable resyncing); the
        # failure backoff base (1 second) and cap (60 seconds); the delay before reopening a failed
        # or ended source (1 second); and an optional clock (real time by default). Asserts that the
        # resync period, when given, and the source retry delay are positive. Creates the object
        # store, the work queue, the retry scheduler (re-queues a key when its retry is due) and the
        # source driver (queues all affected parent keys, in sorted order, on each source event).
        # Starts nothing.
        assert resync_period is None or resync_period > 0, f"{resync_period=} must be positive or None"
        assert source_retry_delay > 0, f"{source_retry_delay=} must be positive"

        self._reconcile = reconcile
        self._resync_period = resync_period
        self._clock = clock or RealClock()

        self._store = ObjectStore(key_map=key_map)
        self._queue: WorkQueue[ParentKey] = WorkQueue()
        self._retry: RetryScheduler[ParentKey] = RetryScheduler(
            on_retry=self._queue.add,
            failure_base_delay=failure_base_delay,
            failure_max_delay=failure_max_delay,
            clock=self._clock,
        )
        self._driver = SourceStreamDriver(
            source=source,
            store=self._store,
            on_affected=self._enqueue_all,
            retry_delay=source_retry_delay,
            clock=self._clock,
        )

        self._start_called = False
        self._tasks: list[asyncio.Task[None]] = []

    async def __aenter__(self) -> ReconcileLoop:
        # [FUNCTION DOC] Start the loop and return it.
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        # [FUNCTION DOC] Stop the loop; exception details are ignored and not suppressed.
        await self.stop()

    async def start(self) -> None:
        # [FUNCTION DOC]
        # Asserts it was not started before. Starts the source driver task and waits until the
        # driver has completed its first full sync, so the store is populated before any
        # reconcile. If cancelled while waiting, cancels the driver task, waits for it, shuts the
        # retry scheduler down and re-raises. Errors from the driver during the wait are not
        # handled here. On success creates the worker task, keeps the driver task, and adds the
        # resync task when a resync period is configured. Does not wait for any reconcile to run.
        assert not self._start_called, "ReconcileLoop.start() must be called exactly once"
        self._start_called = True

        driver_task = asyncio.create_task(self._driver.run())
        try:
            await self._driver.wait_for_sync()
        except asyncio.CancelledError:
            driver_task.cancel()
            await asyncio.gather(driver_task, return_exceptions=True)
            await self._retry.shutdown()
            raise

        self._tasks = [asyncio.create_task(self._worker_loop()), driver_task]
        if self._resync_period is not None:
            self._tasks.append(asyncio.create_task(self._resync_loop()))

    async def stop(self) -> None:
        # [FUNCTION DOC]
        # Asserts the loop was started and that the caller is not one of the loop's own tasks (a
        # reconcile function must schedule the stop as a separate task instead). Shuts down the
        # queue (waking the worker) and the retry scheduler, cancels all tasks and waits for them
        # ignoring their errors, closes the source driver, and forgets the tasks.
        assert self._start_called, "ReconcileLoop.stop() must come after start()"
        assert asyncio.current_task() not in self._tasks, (
            "ReconcileLoop.stop() waits for the worker, so it cannot be awaited from inside reconcile; "
            "call asyncio.create_task(loop.stop()) instead"
        )

        self._queue.shutdown()
        await self._retry.shutdown()

        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        await self._driver.aclose()
        self._tasks = []

    def get_by_parent(self, parent_key: ParentKey) -> list[Any]:
        # [FUNCTION DOC] Return the objects currently in the store under the parent key, ordered by
        # object key. The objects must be treated as read-only.
        return self._store.get_by_parent(parent_key)

    def parent_keys(self) -> set[ParentKey]:
        # [FUNCTION DOC] Return the set of parent keys that currently have at least one object.
        return self._store.parent_keys()

    def _enqueue_all(self, parent_keys: set[ParentKey]) -> None:
        # [FUNCTION DOC] HELPER: Add each parent key to the work queue in sorted order (the queue
        # deduplicates keys that are already pending).
        for parent_key in sorted(parent_keys):
            self._queue.add(parent_key)

    async def _worker_loop(self) -> None:
        # [FUNCTION DOC] HELPER: The single worker. Repeatedly take the next parent key from the
        # queue (stops when the queue is shut down and returns nothing) and call the reconcile
        # function. On success tell the retry scheduler of the success (clearing its backoff); on an
        # ordinary exception log it and tell the scheduler of the failure, which later re-queues
        # the key with backoff. One key at a time is processed.
        while True:
            parent_key = await self._queue.get()
            if parent_key is None:
                return
            try:
                await self._reconcile(parent_key)
                self._retry.note_success(parent_key)
            except Exception:
                logger.error(f"ReconcileLoop reconcile failed {parent_key=}", exc_info=True)
                self._retry.note_failure(parent_key)

    async def _resync_loop(self) -> None:
        # [FUNCTION DOC] HELPER: Forever: sleep for the resync period on the clock, then queue every
        # parent key currently in the store. Asserts a period is configured.
        assert self._resync_period is not None
        while True:
            await self._clock.sleep(self._resync_period)
            self._enqueue_all(self._store.parent_keys())
