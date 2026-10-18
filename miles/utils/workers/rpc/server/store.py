# [MODULE DOC]
# In-memory registry of rpc calls by call id: tracks whether each call is running or finished,
# stores the outcome, lets pollers wait for completion, and expires finished calls (5 minutes
# after the first retrieval of the outcome, or 12 hours after finishing if never retrieved).
# Single event loop only, no locking.
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time

from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.common.protocol import CallStatusResponse

logger = logging.getLogger(__name__)

RETRIEVED_TTL_SECONDS = 300.0
FINISHED_TTL_SECONDS = 12 * 3600.0


class DuplicateCallError(Exception):
    # [CLASS DOC] Raised when a call id that is still tracked is submitted again.
    pass


class CallStore:
    # [CLASS DOC]
    # Map from call id to a record of completion event, outcome and timestamps. Expiry is lazy:
    # it only happens when a new call begins. Running calls never expire.
    def __init__(
        self,
        *,
        retrieved_ttl_seconds: float = RETRIEVED_TTL_SECONDS,
        finished_ttl_seconds: float = FINISHED_TTL_SECONDS,
    ) -> None:
        # [FUNCTION DOC] Store the two retention periods in seconds (after first retrieval; after
        # finishing without retrieval) and start empty.
        self._retrieved_ttl_seconds = retrieved_ttl_seconds
        self._finished_ttl_seconds = finished_ttl_seconds
        self._records: dict[str, _CallRecord] = {}

    def begin(self, *, call_id: str) -> None:
        # [FUNCTION DOC] Drop expired records, then register the call id as running. Raises the
        # duplicate call error if the id is still tracked. Creates the completion signal, so it must
        # be called with an event loop available.
        self._purge_expired()

        if call_id in self._records:
            raise DuplicateCallError(f"call {call_id} already submitted")

        self._records[call_id] = _CallRecord(finished_event=asyncio.Event())
        log_structured(
            logger.debug, tag="rpc", op="call_store", phase="accept", call=call_id, tracked=len(self._records)
        )

    def finish(self, *, call_id: str, outcome: CallStatusResponse) -> None:
        # [FUNCTION DOC] Record the outcome of a running call, stamp the finish time and wake all
        # waiters. Raises a key error for an unknown id and a runtime error if the call already
        # finished.
        record = self._records[call_id]
        if record.outcome is not None:
            raise RuntimeError(f"call {call_id} finished twice")
        record.outcome = outcome
        record.finished_at = time.monotonic()
        record.finished_event.set()
        log_structured(logger.debug, tag="rpc", op="call_store", phase="finish", call=call_id, status=outcome.status)

    async def wait(self, *, call_id: str, timeout: float) -> CallStatusResponse | None:
        # [FUNCTION DOC] Wait up to the timeout in seconds for the call to finish (a timeout is not
        # an error). Returns the outcome, or none if still running. The first time an outcome is
        # returned, stamps the retrieval time, starting the shorter retention period. Raises a key
        # error for an unknown id.
        record = self._records[call_id]

        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
            await asyncio.wait_for(record.finished_event.wait(), timeout=timeout)

        if record.outcome is not None and record.first_retrieved_at is None:
            record.first_retrieved_at = time.monotonic()
        return record.outcome

    def contains(self, call_id: str) -> bool:
        # [FUNCTION DOC] True iff the call id is currently tracked (expired ones are removed only
        # lazily, so an expired record may still be reported until the next begin).
        return call_id in self._records

    def in_flight_call_ids(self) -> list[str]:
        # [FUNCTION DOC] Sorted ids of tracked calls that have no outcome yet.
        return sorted(call_id for call_id, record in self._records.items() if record.outcome is None)

    def _purge_expired(self) -> None:
        # [FUNCTION DOC] HELPER: Rebuild the record map keeping only live records and log the
        # number purged at debug level.
        now = time.monotonic()
        retained = {call_id: record for call_id, record in self._records.items() if self._is_live(record, now=now)}
        if len(retained) != len(self._records):
            log_structured(
                logger.debug, tag="rpc", op="call_store", phase="purge", purged=len(self._records) - len(retained)
            )
        self._records = retained

    def _is_live(self, record: _CallRecord, *, now: float) -> bool:
        # [FUNCTION DOC] HELPER: Running records are live. Finished records that were retrieved are
        # live for the retrieved period after first retrieval; never-retrieved ones for the finished
        # period after finishing.
        if record.finished_at is None:
            return True
        if record.first_retrieved_at is not None:
            return now - record.first_retrieved_at <= self._retrieved_ttl_seconds
        return now - record.finished_at <= self._finished_ttl_seconds


@dataclasses.dataclass
class _CallRecord:
    # [CLASS DOC] HELPER: Mutable per-call state: the completion signal, the outcome (none while
    # running), the monotonic finish time and the monotonic time of first retrieval.
    finished_event: asyncio.Event
    outcome: CallStatusResponse | None = None
    finished_at: float | None = None
    first_retrieved_at: float | None = None
