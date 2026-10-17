# [MODULE DOC]
# In-memory store of request futures for the long-poll style Tinker API: each request gets a
# future id that settles once with a result or an error, can be polled by the owning tenant, and
# is dropped one hour after settling. Module constants: the three state names (pending, done,
# failed) and the retention time in seconds for settled futures (3600). Nothing is persisted, so
# deduplication of retried requests only lasts for the current process.
"""In-memory future results. Request deduplication lasts only for the current process."""

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from miles.tinker.core.types import OwnershipError

PENDING = "pending"
DONE = "done"
FAILED = "failed"

_FINISHED_TTL_S = 3600.0


@dataclass
class RequestFuture:
    # [CLASS DOC] One request's outcome record: request id, model id (or model path / base marker
    # for sampling), owning tenant, state (pending, done or failed), result mapping when done,
    # error text and category when failed, monotonic settle time, and an async event set when the
    # future settles (used to wake long-poll waiters).
    request_id: str
    model_id: str
    tenant: str
    state: str = PENDING
    result: dict | None = None
    error: str | None = None
    error_category: str | None = None
    finished_at: float | None = None
    # long-poll wakeup: set when the future settles
    settled: asyncio.Event = field(default_factory=asyncio.Event)


class RequestFutureStore:
    # [CLASS DOC] Mapping from request id to future with tenant-checked lookup, single-settle
    # semantics (the first resolve or fail wins; later ones are ignored) and lazy expiry of
    # settled futures at lookup time. Not thread-safe; used from the event loop.
    def __init__(self) -> None:
        # [FUNCTION DOC] Start with no futures.
        self._futures: dict[str, RequestFuture] = {}

    def create(self, model_id: str, tenant: str) -> RequestFuture:
        # [FUNCTION DOC] Create a pending future for the model and tenant under a fresh id made of
        # the prefix "req-" and 32 random hex characters, store it and return it.
        future = RequestFuture(request_id=f"req-{uuid.uuid4().hex}", model_id=model_id, tenant=tenant)
        self._futures[future.request_id] = future
        return future

    def resolve(self, request_id: str, result: dict) -> None:
        # [FUNCTION DOC] Settle a pending future as done with the result mapping, recording the
        # monotonic time and waking waiters. Does nothing if the future already settled (for
        # example failed by lease expiry). An unknown id raises a key error.
        future = self._futures[request_id]
        if future.state != PENDING:  # e.g. already failed by lease expiry
            return
        future.state = DONE
        future.result = result
        future.finished_at = time.monotonic()
        future.settled.set()

    def fail(self, request_id: str, error: str, category: str) -> None:
        # [FUNCTION DOC] Settle a pending future as failed with an error text and a category
        # (such as user or server), recording the time and waking waiters. Does nothing if already
        # settled. An unknown id raises a key error.
        future = self._futures[request_id]
        if future.state != PENDING:
            return
        future.state = FAILED
        future.error = error
        future.error_category = category
        future.finished_at = time.monotonic()
        future.settled.set()

    def get(self, request_id: str, tenant: str) -> RequestFuture | None:
        # [FUNCTION DOC] First drops expired futures, then returns the future for the id, none if
        # unknown or expired, or raises an ownership error if it belongs to a different tenant.
        """Return None for unknown or expired futures; enforce tenant ownership otherwise."""
        self._sweep()
        future = self._futures.get(request_id)
        if future is None:
            return None
        if future.tenant != tenant:
            raise OwnershipError(f"request {request_id} does not belong to this tenant")
        return future

    def request_id_for_retry(self, request_id: str, model_id: str, tenant: str) -> str:
        # [FUNCTION DOC] Support idempotent retries: if the earlier future still exists (and the
        # tenant owns it, else an ownership error) return its id unchanged. If it expired, create
        # a replacement future for the model and tenant, immediately fail it as a user error saying
        # the result expired after retention, and return the replacement id (so the retrying client
        # gets a clear error instead of silent re-execution).
        if self.get(request_id, tenant) is not None:
            return request_id
        replacement = self.create(model_id, tenant)
        self.fail(replacement.request_id, "result expired after retention", "user")
        return replacement.request_id

    def _sweep(self) -> None:
        # [FUNCTION DOC] HELPER: Delete every settled future whose settle time is more than the
        # retention period (one hour) in the past; pending futures are never expired.
        now = time.monotonic()
        expired = [
            request_id
            for request_id, future in self._futures.items()
            if future.finished_at is not None and now - future.finished_at > _FINISHED_TTL_S
        ]
        for request_id in expired:
            del self._futures[request_id]
