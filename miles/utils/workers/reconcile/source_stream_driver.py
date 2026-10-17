# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC]
# Drives a restartable source stream into the object store: opens the stream, applies each event
# to the store, reports the affected parent keys, signals when the first full sync (first replace
# event) has been applied, and reopens the stream after it ends or fails.
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable

from miles.utils.test_utils.clock import Clock
from miles.utils.workers.reconcile.object_store import ObjectStore
from miles.utils.workers.reconcile.source_event import ParentKey, ReplaceEvent, SourceEvent, SourceWatchFn

logger = logging.getLogger(__name__)


class SourceStreamDriver:
    # [CLASS DOC]
    # Runs the source-to-store pump. The synced signal is set at the first replace event and never
    # cleared, so later reopened streams do not reset it. At most one stream is open at a time.
    def __init__(
        self,
        *,
        source: SourceWatchFn,
        store: ObjectStore,
        on_affected: Callable[[set[ParentKey]], None],
        retry_delay: float,
        clock: Clock,
    ) -> None:
        # [FUNCTION DOC]
        # Store the source (no-argument function returning an async event stream), the object
        # store, the callback receiving the set of affected parent keys after each event, the
        # delay in seconds between stream attempts and the clock. Creates the unset synced signal.
        # Must be created inside a running event loop on older Python versions.
        self._source = source
        self._store = store
        self._on_affected = on_affected
        self._retry_delay = retry_delay
        self._clock = clock
        self._stream: AsyncGenerator[SourceEvent, None] | None = None
        self._synced = asyncio.Event()

    async def run(self) -> None:
        # [FUNCTION DOC]
        # Runs forever (until cancelled). Each round: open the source, remember the stream as
        # current, and pump it until it ends (logged as a warning) or an ordinary exception occurs
        # (logged as an error with trace). Whatever happens, clear the current stream and close the
        # stream, logging any close failure. Then sleep the retry delay on the clock and reopen.
        while True:
            stream: AsyncGenerator[SourceEvent, None] | None = None
            try:
                stream = self._source()
                self._stream = stream
                await self._pump(stream)
                logger.warning("SourceStreamDriver source stream ended, reopening")
            except Exception:
                logger.error("SourceStreamDriver source stream failed, reopening", exc_info=True)
            finally:
                self._stream = None
                await _aclose_logging_failure(stream)
            await self._clock.sleep(self._retry_delay)

    async def wait_for_sync(self) -> None:
        # [FUNCTION DOC] Block until the first replace event has been applied to the store. Returns
        # immediately if that already happened.
        await self._synced.wait()

    async def aclose(self) -> None:
        # [FUNCTION DOC] Forget the current stream and, if there is one, close it (errors from
        # closing propagate). Does not stop the run loop, which the caller cancels separately.
        stream = self._stream
        self._stream = None
        if stream is not None:
            await stream.aclose()

    async def _pump(self, stream: AsyncGenerator[SourceEvent, None]) -> None:
        # [FUNCTION DOC] HELPER: Consume the stream. Raises a runtime error if the first event is
        # not a replace event. For every event: apply it to the store, pass the returned affected
        # parent keys to the callback, and, if it was a replace event, set the synced signal.
        first = True
        async for event in stream:
            if first and not isinstance(event, ReplaceEvent):
                raise RuntimeError(f"A source stream must open with ReplaceEvent, got {event=}")
            first = False
            self._on_affected(self._store.handle_event(event))
            if isinstance(event, ReplaceEvent):
                self._synced.set()


async def _aclose_logging_failure(stream: AsyncGenerator[SourceEvent, None] | None) -> None:
    # [FUNCTION DOC] HELPER: Close the stream if there is one; an ordinary exception while
    # closing is logged as an error and swallowed.
    if stream is None:
        return
    try:
        await stream.aclose()
    except Exception:
        logger.error("SourceStreamDriver failed to close a source stream", exc_info=True)
