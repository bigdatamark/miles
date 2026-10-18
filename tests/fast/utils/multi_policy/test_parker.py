# [MODULE DOC]
# Async tests for the multi-policy follower parker. A parker is created with a number of followers;
# followers call the maybe-park method at every round boundary and are held there while a leader
# runs a body inside the all-parked context manager (used to write a consistent global checkpoint).
# Tests cover release on failure, ordering of body versus parking, staying parked until the body
# ends, consecutive rounds, a timeout failure when a follower never parks, the fast path when no
# checkpoint is running, and that the follower always yields to the event loop.
import asyncio

import pytest

from miles.utils.multi_policy import parker as parker_module
from miles.utils.multi_policy.parker import Parker


class TestWithAllParked:
    # [CLASS DOC] Tests of the all-parked context manager: the leader's body runs only once every
    # follower is parked, followers resume after it ends, even on failure, and a missing follower
    # makes the wait fail.
    async def test_a_body_exception_releases_every_parked_follower(self) -> None:
        # [FUNCTION DOC] With one follower task already trying to park, runs a body inside the
        # all-parked context that raises a runtime error. Asserts the original error propagates
        # unchanged and that the follower task still finishes within five seconds (it is released).
        """A failed checkpoint still releases followers and preserves its original failure."""
        parker = Parker(num_followers=1)
        follower = asyncio.create_task(parker.maybe_park_follower())

        with pytest.raises(RuntimeError, match="checkpoint failed"):
            async with parker.with_all_parked():
                raise RuntimeError("checkpoint failed")

        await asyncio.wait_for(follower, timeout=5)

    async def test_the_body_runs_only_once_every_follower_has_parked(self):
        # [FUNCTION DOC] Two followers each sleep (0.05 and 0.1 seconds) and then try to park. The
        # leader enters the all-parked context and inside the body asserts neither follower task has
        # finished (they are held parked, not completed). After the body, asserts it did run and that
        # both followers finish within five seconds.
        """The body writes the run's checkpoint, so a follower still training would be caught mid round."""
        parker = Parker(num_followers=2)
        entered = False

        async def follower(delay: float) -> None:
            # [FUNCTION DOC] HELPER: Simulated follower: sleeps for the given number of seconds, then
            # calls maybe-park and returns when released.
            await asyncio.sleep(delay)
            await parker.maybe_park_follower()

        tasks = [asyncio.create_task(follower(0.05)), asyncio.create_task(follower(0.1))]

        async with parker.with_all_parked():
            entered = True
            assert all(not task.done() for task in tasks)

        assert entered
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)

    async def test_a_parked_follower_stays_parked_until_the_body_returns(self):
        # [FUNCTION DOC] A follower parks and then sets a resumed flag. While the leader's body sleeps
        # for 0.05 seconds inside the all-parked context, asserts the flag is still unset. After the
        # context exits, the follower finishes within five seconds and the flag is set.
        """A follower that resumes early would train past the round the checkpoint claims to hold."""
        parker = Parker(num_followers=1)
        resumed = False

        async def follower() -> None:
            # [FUNCTION DOC] HELPER: Simulated follower: calls maybe-park, then marks the enclosing
            # resumed flag once it has been released.
            nonlocal resumed
            await parker.maybe_park_follower()
            resumed = True

        task = asyncio.create_task(follower())

        async with parker.with_all_parked():
            await asyncio.sleep(0.05)
            assert not resumed

        await asyncio.wait_for(task, timeout=5)
        assert resumed

    async def test_a_second_round_waits_for_the_first_one_to_disperse(self):
        # [FUNCTION DOC] One follower loops forever: park, increment a lap counter, sleep 0.01 seconds.
        # The leader enters the all-parked context twice in a row, recording the lap counter inside
        # each body. The follower task is cancelled at the end. Asserts the second round saw more laps
        # than the first, i.e. the second entry waited for the follower to leave the first round and
        # did not count its stale arrival.
        """Entering while a follower is still leaving would count its stale arrival as this round's."""
        parker = Parker(num_followers=1)
        laps = 0

        async def follower() -> None:
            # [FUNCTION DOC] HELPER: Endless simulated follower loop: park, increment the enclosing lap
            # counter, then sleep 0.01 seconds, forever until cancelled.
            nonlocal laps
            while True:
                await parker.maybe_park_follower()
                laps += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(follower())
        try:
            async with parker.with_all_parked():
                laps_of_first_round = laps
            async with parker.with_all_parked():
                laps_of_second_round = laps
        finally:
            task.cancel()

        assert laps_of_second_round > laps_of_first_round

    async def test_waiting_for_a_follower_that_never_parks_fails_loudly(self, monkeypatch):
        # [FUNCTION DOC] Sets the parker module's park timeout constant to zero seconds, creates a
        # parker expecting one follower that never parks, and enters the all-parked context. Asserts
        # an assertion error is raised whose message reports zero ready followers while one was
        # wanted, rather than hanging.
        """A silent hang here stalls the whole run, so the wait has to end as a failure instead."""
        monkeypatch.setattr(parker_module, "PARK_TIMEOUT_SECONDS", 0.0)
        parker = Parker(num_followers=1)

        with pytest.raises(AssertionError, match="ready 0, want 1"):
            async with parker.with_all_parked():
                pass


class TestMaybeParkFollower:
    # [CLASS DOC] Tests of the follower-side maybe-park call when no checkpoint is in progress.
    async def test_a_follower_is_not_slowed_down_while_no_checkpoint_is_running(self):
        # [FUNCTION DOC] With no all-parked context active, a follower task calling maybe-park and
        # then setting a finished flag completes within five seconds, and the flag is set.
        """Every round of every follower passes through here, so the common case must not yield."""
        parker = Parker(num_followers=1)
        finished = False

        async def follower() -> None:
            # [FUNCTION DOC] HELPER: Simulated follower: calls maybe-park once and marks the enclosing
            # finished flag.
            nonlocal finished
            await parker.maybe_park_follower()
            finished = True

        task = asyncio.create_task(follower())
        await asyncio.wait_for(task, timeout=5)

        assert finished


class TestTheFollowerAlwaysGivesTheLoopATurn:
    # [CLASS DOC] Guards against the follower's maybe-park call returning without ever yielding to the
    # event loop.
    async def test_an_open_gate_still_suspends_the_follower(self):
        # [FUNCTION DOC] Schedules another task that sets a flag, then awaits maybe-park directly with
        # no checkpoint running. Asserts the other task already ran, proving the call suspended at
        # least once even though it had nothing to wait for.
        """An open gate that returns without suspending lets the follower's unbounded loop starve the loop."""
        parker = Parker(num_followers=1)
        other_task_ran = False

        async def other_task() -> None:
            # [FUNCTION DOC] HELPER: Marks the enclosing flag that it has run; used to detect whether
            # the event loop got a turn.
            nonlocal other_task_ran
            other_task_ran = True

        asyncio.create_task(other_task())

        await parker.maybe_park_follower()

        assert other_task_ran
