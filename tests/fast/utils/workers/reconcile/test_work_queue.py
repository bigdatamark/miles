# [MODULE DOC]
# Unit tests for the reconcile work queue, a first-in-first-out queue of unique keys: adding a
# queued key again is a no-op that keeps its position, getting waits when empty, and shutting down
# wakes blocked getters with nothing and drops later additions.
from __future__ import annotations

import asyncio
from collections import deque

from tests.fast.utils.workers.reconcile.utils import settle

from miles.utils.workers.reconcile.work_queue import WorkQueue


class TestWorkQueue:
    # [CLASS DOC] Behavior of the queue's shutdown and deduplication.
    async def test_shutdown_wakes_a_worker_blocked_on_an_empty_queue(self):
        # [FUNCTION DOC] A consumer task blocks on get of an empty queue; after shutdown it
        # completes with nothing.
        """A worker parked on get() must be released by shutdown, not only by cancellation."""
        queue: WorkQueue[str] = WorkQueue()
        got: list[str | None] = []

        async def consume() -> None:
            # [FUNCTION DOC] HELPER: Appends the result of one get to the list.
            got.append(await queue.get())

        task = asyncio.create_task(consume())
        await settle()
        assert got == []

        queue.shutdown()
        await task
        assert got == [None]

    async def test_a_key_added_after_shutdown_is_dropped(self):
        # [FUNCTION DOC] After shutdown, adding a key has no effect: get returns nothing and the
        # private key deque is empty.
        """Nothing enters the queue once it is shut down, not even to sit there unread."""
        queue: WorkQueue[str] = WorkQueue()
        queue.shutdown()
        queue.add("cell-a")

        assert await queue.get() is None
        assert queue._keys == deque()

    async def test_a_requeued_key_keeps_its_original_position(self):
        # [FUNCTION DOC] Adding a, b, a yields a then b.
        """Re-adding a queued key must not push it behind keys that arrived later."""
        queue: WorkQueue[str] = WorkQueue()
        queue.add("cell-a")
        queue.add("cell-b")
        queue.add("cell-a")

        assert [await queue.get(), await queue.get()] == ["cell-a", "cell-b"]

    async def test_a_duplicate_key_is_dispatched_once(self):
        # [FUNCTION DOC] Adding a, a, b yields a then b, and the private key deque ends empty.
        """The queue is a dedup set, not a multiset."""
        queue: WorkQueue[str] = WorkQueue()
        queue.add("cell-a")
        queue.add("cell-a")
        queue.add("cell-b")

        assert [await queue.get(), await queue.get()] == ["cell-a", "cell-b"]
        assert queue._keys == deque()
