# [MODULE DOC] Unit tests for the test-utility clocks: RealClock (wall clock time and real async
# sleep) and FakeClock (virtual time that only advances through elapse, which also wakes sleepers
# whose target time has been reached, in target-time order; negative or zero sleeps return
# immediately; a pending_count property reports sleepers still waiting). Async tests rely on the
# project's async test mode and use asyncio tasks plus single event-loop yields to let sleepers
# register.
import asyncio

import pytest

from miles.utils.test_utils.clock import FakeClock, RealClock


class TestRealClock:
    # [CLASS DOC] Tests of the real clock against the system time.
    def test_time_returns_current_time(self):
        # [FUNCTION DOC] The clock's time lies between two system time readings taken around it.
        clock = RealClock()
        import time

        before = time.time()
        result = clock.time()
        after = time.time()
        assert before <= result <= after

    async def test_sleep_actually_waits(self):
        # [FUNCTION DOC] Sleeping 0.05 s takes at least 0.04 s of measured system time (tolerance
        # for timer granularity).
        clock = RealClock()
        import time

        start = time.time()
        await clock.sleep(0.05)
        elapsed = time.time() - start
        assert elapsed >= 0.04


class TestFakeClockTime:
    # [CLASS DOC] Tests of the fake clock's time reading and elapse behavior.
    def test_initial_time(self):
        # [FUNCTION DOC] A clock built with start 100.0 reports 100.0.
        clock = FakeClock(start=100.0)
        assert clock.time() == 100.0

    def test_default_start_is_zero(self):
        # [FUNCTION DOC] A clock built without arguments starts at 0.0.
        clock = FakeClock()
        assert clock.time() == 0.0

    async def test_elapse_advances_time(self):
        # [FUNCTION DOC] Starting at 10.0, elapsing 5.0 yields time 15.0.
        clock = FakeClock(start=10.0)
        await clock.elapse(5.0)
        assert clock.time() == 15.0

    async def test_multiple_elapse_accumulate(self):
        # [FUNCTION DOC] Elapsing 3.0 then 7.0 from zero yields time 10.0.
        clock = FakeClock()
        await clock.elapse(3.0)
        await clock.elapse(7.0)
        assert clock.time() == 10.0

    async def test_elapse_zero_is_allowed(self):
        # [FUNCTION DOC] Elapsing 0.0 is allowed and leaves the time unchanged at 5.0.
        clock = FakeClock(start=5.0)
        await clock.elapse(0.0)
        assert clock.time() == 5.0

    async def test_elapse_negative_raises(self):
        # [FUNCTION DOC] Elapsing -1.0 raises an assertion error whose message mentions "negative".
        clock = FakeClock()
        with pytest.raises(AssertionError, match="negative"):
            await clock.elapse(-1.0)


class TestFakeClockSleep:
    # [CLASS DOC] Tests of fake sleeping: sleepers are tasks that block until elapsed fake time
    # reaches their target; each test starts tasks and yields once to the loop so they register.
    async def test_sleep_zero_returns_immediately(self):
        # [FUNCTION DOC] A task sleeping 0 on the fake clock completes after a single loop yield,
        # without any elapse call.
        clock = FakeClock()
        completed = False

        async def task() -> None:
            # [FUNCTION DOC] HELPER: Sleeps 0 on the clock, then sets the completed flag.
            nonlocal completed
            await clock.sleep(0)
            completed = True

        asyncio.create_task(task())
        await asyncio.sleep(0)
        assert completed

    async def test_sleep_blocks_until_elapse(self):
        # [FUNCTION DOC] A task sleeping 10.0 has not completed after a loop yield, and completes
        # once the clock elapses 10.0.
        clock = FakeClock()
        completed = False

        async def task() -> None:
            # [FUNCTION DOC] HELPER: Sleeps 10.0 on the clock, then sets the completed flag.
            nonlocal completed
            await clock.sleep(10.0)
            completed = True

        asyncio.create_task(task())
        await asyncio.sleep(0)
        assert not completed

        await clock.elapse(10.0)
        assert completed

    async def test_sleep_does_not_resolve_before_target(self):
        # [FUNCTION DOC] A task sleeping 10.0 is still blocked after elapsing 9.9 and completes after
        # a further 0.1.
        clock = FakeClock()
        completed = False

        async def task() -> None:
            # [FUNCTION DOC] HELPER: Sleeps 10.0 on the clock, then sets the completed flag.
            nonlocal completed
            await clock.sleep(10.0)
            completed = True

        asyncio.create_task(task())
        await asyncio.sleep(0)

        await clock.elapse(9.9)
        assert not completed

        await clock.elapse(0.1)
        assert completed

    async def test_multiple_sleeps_resolve_in_order(self):
        # [FUNCTION DOC] Three tasks sleep 5, 10 and 3 seconds and record their names. After elapsing
        # 5 the record is c then a (ordered by target time); after another 5 it is c, a, b.
        clock = FakeClock()
        order: list[str] = []

        async def task_a() -> None:
            # [FUNCTION DOC] HELPER: Sleeps 5.0 then records "a".
            await clock.sleep(5.0)
            order.append("a")

        async def task_b() -> None:
            # [FUNCTION DOC] HELPER: Sleeps 10.0 then records "b".
            await clock.sleep(10.0)
            order.append("b")

        async def task_c() -> None:
            # [FUNCTION DOC] HELPER: Sleeps 3.0 then records "c".
            await clock.sleep(3.0)
            order.append("c")

        asyncio.create_task(task_a())
        asyncio.create_task(task_b())
        asyncio.create_task(task_c())
        await asyncio.sleep(0)

        await clock.elapse(5.0)
        assert order == ["c", "a"]

        await clock.elapse(5.0)
        assert order == ["c", "a", "b"]

    async def test_same_target_time_all_resolve(self):
        # [FUNCTION DOC] Three tasks sleeping the same 5.0 all complete after elapsing 5.0; the
        # completion counter equals 3.
        clock = FakeClock()
        count = 0

        async def task() -> None:
            # [FUNCTION DOC] HELPER: Sleeps 5.0 then increments the counter.
            nonlocal count
            await clock.sleep(5.0)
            count += 1

        for _ in range(3):
            asyncio.create_task(task())
        await asyncio.sleep(0)

        await clock.elapse(5.0)
        assert count == 3

    async def test_overshoot_resolves_all_pending(self):
        # [FUNCTION DOC] Tasks sleeping 1, 5 and 10 are started; one elapse of 100.0 wakes all of
        # them (checked as the sorted list of recorded delays being 1, 5, 10).
        clock = FakeClock()
        order: list[int] = []

        for delay in [1, 5, 10]:
            d = delay

            async def task(d: int = d) -> None:
                # [FUNCTION DOC] HELPER: Sleeps d seconds (bound via default argument to avoid
                # late binding of the loop variable) then records d.
                await clock.sleep(d)
                order.append(d)

            asyncio.create_task(task())
        await asyncio.sleep(0)

        await clock.elapse(100.0)
        assert sorted(order) == [1, 5, 10]

    async def test_sleep_negative_returns_immediately(self):
        # [FUNCTION DOC] A task sleeping -1.0 completes after one loop yield with no elapse.
        clock = FakeClock()
        completed = False

        async def task() -> None:
            # [FUNCTION DOC] HELPER: Sleeps -1.0 on the clock, then sets the completed flag.
            nonlocal completed
            await clock.sleep(-1.0)
            completed = True

        asyncio.create_task(task())
        await asyncio.sleep(0)
        assert completed


class TestFakeClockPendingCount:
    # [CLASS DOC] Tests of the pending_count property (number of unresolved sleepers).
    async def test_no_pending_initially(self):
        # [FUNCTION DOC] A fresh clock has zero pending sleepers.
        clock = FakeClock()
        assert clock.pending_count == 0

    async def test_pending_count_tracks_sleeps(self):
        # [FUNCTION DOC] Two sleep tasks (5.0 and 10.0) give a count of 2; after elapsing 5.0 the
        # count is 1; after another 5.0 it is 0.
        clock = FakeClock()

        asyncio.create_task(clock.sleep(5.0))
        asyncio.create_task(clock.sleep(10.0))
        await asyncio.sleep(0)
        assert clock.pending_count == 2

        await clock.elapse(5.0)
        assert clock.pending_count == 1

        await clock.elapse(5.0)
        assert clock.pending_count == 0


class TestFakeClockChainedSleeps:
    # [CLASS DOC] Tests of coroutines that sleep repeatedly, checking that each elapse resumes them
    # at the right fake times.
    async def test_sequential_sleeps_in_coroutine(self):
        # [FUNCTION DOC] A task records the clock time, sleeps 5.0, records, sleeps 3.0, records.
        # Asserts checkpoints are [0.0] initially, then [0.0, 5.0] after elapsing 5.0, then [0.0,
        # 5.0, 8.0] after elapsing 3.0.
        """A coroutine that does multiple sleeps in sequence."""
        clock = FakeClock()
        checkpoints: list[float] = []

        async def task() -> None:
            # [FUNCTION DOC] HELPER: Records time, sleeps 5.0, records time, sleeps 3.0, records
            # time.
            checkpoints.append(clock.time())
            await clock.sleep(5.0)
            checkpoints.append(clock.time())
            await clock.sleep(3.0)
            checkpoints.append(clock.time())

        asyncio.create_task(task())
        await asyncio.sleep(0)

        assert checkpoints == [0.0]

        await clock.elapse(5.0)
        assert checkpoints == [0.0, 5.0]

        await clock.elapse(3.0)
        assert checkpoints == [0.0, 5.0, 8.0]

    async def test_periodic_loop(self):
        # [FUNCTION DOC] A looping task sleeps 10.0 first, then appends the time and sleeps 5.0
        # until three ticks exist. Asserts no tick before elapsing, ticks [10.0] after 10, [10.0,
        # 15.0] after 5 more, [10.0, 15.0, 20.0] after another 5; finally cancels the task and
        # swallows the cancellation error.
        """Simulates a periodic loop like SimpleHealthChecker._loop."""
        clock = FakeClock()
        ticks: list[float] = []

        async def loop() -> None:
            # [FUNCTION DOC] HELPER: Sleeps 10.0 once, then while fewer than 3 ticks are recorded
            # appends the current time and sleeps 5.0.
            await clock.sleep(10.0)
            while len(ticks) < 3:
                ticks.append(clock.time())
                await clock.sleep(5.0)

        task = asyncio.create_task(loop())
        await asyncio.sleep(0)

        # Step 1: first_wait=10
        assert ticks == []
        await clock.elapse(10.0)
        assert ticks == [10.0]

        # Step 2: interval=5
        await clock.elapse(5.0)
        assert ticks == [10.0, 15.0]

        # Step 3: interval=5
        await clock.elapse(5.0)
        assert ticks == [10.0, 15.0, 20.0]

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
