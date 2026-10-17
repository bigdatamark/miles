# [MODULE DOC] HELPER: Clock abstraction for async code under test: an abstract clock with a time
# reading and an async sleep, a real implementation backed by wall-clock time and the event loop's
# sleep, and a deterministic fake whose time only advances when a test explicitly elapses it.
from __future__ import annotations

import abc
import asyncio
import heapq
import time
from typing import NamedTuple


class Clock(abc.ABC):
    # [CLASS DOC] HELPER: Abstract clock interface (time and sleep) that production code can accept
    # so tests can substitute a fake.
    @abc.abstractmethod
    def time(self) -> float:
        # [FUNCTION DOC] HELPER: Abstract: returns the current time in seconds as a float.
        ...

    @abc.abstractmethod
    async def sleep(self, seconds: float) -> None:
        # [FUNCTION DOC] HELPER: Abstract: suspends the caller for the given number of seconds.
        ...


class RealClock(Clock):
    # [CLASS DOC] HELPER: Production clock delegating to the system wall clock and the event loop.
    def time(self) -> float:
        # [FUNCTION DOC] HELPER: Returns the system wall-clock time in seconds since the epoch.
        return time.time()

    async def sleep(self, seconds: float) -> None:
        # [FUNCTION DOC] HELPER: Awaits the real event-loop sleep for the given seconds.
        await asyncio.sleep(seconds)


_DRAIN_ITERATIONS = 20


class _Waiter(NamedTuple):
    # [CLASS DOC] HELPER: One pending fake sleep: the absolute target time, a monotonically
    # increasing sequence number (tie-breaker so equal targets wake in call order and futures are
    # never compared), and the future to complete when the target is reached.
    target: float
    seq: int
    future: asyncio.Future[None]


class FakeClock(Clock):
    # [CLASS DOC] HELPER: Deterministic clock. Keeps a current time and a min-heap of waiters
    # ordered by target time then arrival. Time never moves by itself; only the elapse method
    # advances it. Not thread-safe; must be used from a single running event loop.
    """Deterministic clock for testing async time-dependent code.

    ``sleep()`` suspends the caller until ``elapse()`` advances the clock
    past the target time. This gives tests precise control over which
    sleeps resolve and when.
    """

    def __init__(self, start: float = 0.0) -> None:
        # [FUNCTION DOC] HELPER: Sets the current time to the start value (default zero), with no
        # waiters and a zero sequence counter.
        self._now = start
        self._waiters: list[_Waiter] = []
        self._counter: int = 0

    def time(self) -> float:
        # [FUNCTION DOC] HELPER: Returns the fake current time.
        return self._now

    async def sleep(self, seconds: float) -> None:
        # [FUNCTION DOC] HELPER: A negative duration returns immediately without yielding. Otherwise
        # registers a waiter with target equal to now plus the duration (a zero duration is
        # resolved immediately by the readiness pass, though the call still awaits the already
        # completed future), then waits until elapse moves the clock to or past the target. Needs a
        # running event loop.
        if seconds < 0:
            return

        target = self._now + seconds
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._counter += 1
        heapq.heappush(self._waiters, _Waiter(target=target, seq=self._counter, future=future))
        self._resolve_ready()
        await future

    async def elapse(self, seconds: float) -> None:
        # [FUNCTION DOC] HELPER: Advances the fake time by the given non-negative amount (assertion
        # error if negative), completes every waiter whose target is now reached, then yields to
        # the event loop 20 times (re-checking waiters after each yield) so resumed coroutines can
        # run their synchronous code and register their next sleeps, which may themselves become
        # due within the same elapsed span.
        assert seconds >= 0, f"Cannot elapse negative time: {seconds}"
        self._now += seconds
        self._resolve_ready()
        # Drain: yield enough times for resolved coroutines to run through
        # sync code and register their next sleep.
        for _ in range(_DRAIN_ITERATIONS):
            await asyncio.sleep(0)
            self._resolve_ready()

    def _resolve_ready(self) -> None:
        # [FUNCTION DOC] HELPER: Pops waiters from the heap, earliest first, while their target is
        # at or before the current time, and completes each future that is not already done (for
        # example cancelled ones are skipped).
        while self._waiters and self._waiters[0].target <= self._now:
            waiter = heapq.heappop(self._waiters)
            if not waiter.future.done():
                waiter.future.set_result(None)

    @property
    def pending_count(self) -> int:
        # [FUNCTION DOC] HELPER: Number of waiters whose futures are not yet done (still sleeping).
        return sum(1 for w in self._waiters if not w.future.done())
