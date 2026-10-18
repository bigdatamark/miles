# [MODULE DOC]
# Single-event-loop coordination primitive letting a leader act while every follower is held at a
# round boundary. Not thread safe; all participants must run on one event loop.
import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

PARK_TIMEOUT_SECONDS = 3600.0


class Parker:
    # [CLASS DOC] Holds a gate (an event, open by default) and a count of followers currently parked
    # at the gate. The leader closes the gate once all followers are waiting at it; followers call
    # the park method at round boundaries and pass straight through while the gate is open.
    """Parks every follower at its round boundary while the leader holds `with_all_parked()`."""

    def __init__(self, *, num_followers: int) -> None:
        # [FUNCTION DOC] Stores the follower count; the gate starts open; the ready count starts at
        # zero.
        self._num_followers = num_followers
        self._event = asyncio.Event()
        self._event.set()
        self._num_ready = 0

    @asynccontextmanager
    async def with_all_parked(self) -> AsyncIterator[None]:
        # [FUNCTION DOC] Leader side async context manager. First waits until no follower is still
        # parked from an earlier round (ready count zero), then closes the gate, then waits until all
        # followers are parked, then yields. On exit (even on error) reopens the gate. Each wait polls
        # once per second and asserts out after an hour.
        await self._wait_num_ready(0)
        self._event.clear()
        try:
            await self._wait_num_ready(self._num_followers)
            yield
        finally:
            self._event.set()

    async def maybe_park_follower(self) -> None:
        # [FUNCTION DOC] Follower side. Yields to the loop once (so the leader can close the gate
        # first), counts itself as ready, waits for the gate to be open (returns immediately if it is),
        # then uncounts itself.
        # KNOWN ISSUE: if the follower is cancelled while waiting, it stays counted as ready, which
        # would make the leader's next wait for zero ready followers hang until the timeout.
        await asyncio.sleep(0)
        self._num_ready += 1
        await self._event.wait()
        self._num_ready -= 1

    async def _wait_num_ready(self, target: int) -> None:
        # [FUNCTION DOC] HELPER: Polls every second until the ready count equals the target; asserts
        # (with the counts in the message) once an hour of monotonic time has passed. Checks the
        # deadline only when the count is not yet the target.
        deadline = time.monotonic() + PARK_TIMEOUT_SECONDS
        while self._num_ready != target:
            assert (
                time.monotonic() < deadline
            ), f"waited {PARK_TIMEOUT_SECONDS}s for the followers to park; ready {self._num_ready}, want {target}"
            await asyncio.sleep(1)
