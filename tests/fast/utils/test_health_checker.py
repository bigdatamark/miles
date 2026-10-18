# [MODULE DOC]
# Tests for the fault-tolerance simple health checker, its activeness tracker, the no-op checker
# and the failure-threshold command-line argument. The checker runs an async probe function in a
# background loop driven by an injectable fake clock: an optional initial grace wait, then one probe
# per interval, each bounded by a timeout. Verified behaviours: start/stop lifecycle, probe cadence
# and first-wait delay, result callback semantics (raw per-probe results, errors in the callback
# do not kill the loop), activeness gating (no probes while inactive, grace period re-armed and
# failure counter reset when becoming active again, stale in-flight probe results discarded using
# an epoch), the three-valued status (unknown/true/false) including being hidden live while paused,
# failure-threshold debouncing, probe timeouts, cancellation on stop, and the shipped rollout
# health-check defaults. Local helpers: a settle function that yields until the fake clock has a
# pending sleeper, a checker factory, and a mutable activeness provider.
import argparse
import asyncio

import pytest
from _pytest.recwarn import WarningsRecorder

from miles.utils.arguments import get_miles_extra_args_provider
from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.ft_utils.health_checker import (
    ActiveAndEpoch,
    ActivenessTracker,
    NoopHealthChecker,
    SimpleHealthChecker,
    SimpleHealthCheckerConfig,
)
from miles.utils.test_utils.clock import FakeClock


async def _settle(clock: FakeClock) -> None:
    # [FUNCTION DOC] HELPER: Yield control to the event loop up to 1000 times until the fake clock
    # has at least one pending sleeper (meaning the checker loop is parked waiting for time), then
    # return. Returns silently after 1000 yields even if no sleeper appeared.
    for _ in range(1000):
        if clock.pending_count >= 1:
            return
        await asyncio.sleep(0)


def _make_checker(
    *,
    check_fn=None,
    on_result=None,
    interval: float = 10.0,
    timeout: float = 5.0,
    first_wait: float = 0.0,
    failure_threshold: int = 1,
    name: str = "test",
    clock: FakeClock | None = None,
    activeness: "_Activeness | None" = None,
) -> tuple[SimpleHealthChecker, FakeClock]:
    # [FUNCTION DOC] HELPER: Build a simple health checker plus the fake clock driving it, returned
    # as a pair. Parameters: probe function (default succeeds immediately), result callback,
    # interval (default 10 s), per-probe timeout (default 5 s), first wait (default 0), failure
    # threshold (default 1), name, optional shared clock (default a new fake clock) and an
    # activeness provider (default an always-active one). The checker is not started.
    from miles.utils.ft_utils.health_checker import SimpleHealthCheckerConfig

    if check_fn is None:

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Default probe that succeeds and does nothing.
            pass

    c = clock or FakeClock()
    checker = SimpleHealthChecker(
        name=name,
        check_fn=check_fn,
        get_activeness=activeness or _Activeness(),
        on_result=on_result,
        config=SimpleHealthCheckerConfig(
            interval=interval, timeout=timeout, first_wait=first_wait, failure_threshold=failure_threshold
        ),
        clock=c,
    )
    return checker, c


class _Activeness:
    # [CLASS DOC] HELPER: Test activeness provider wrapping an activeness tracker. Calling the
    # instance returns the tracker's current active-and-epoch pair; the active property reads and
    # (by assignment) publishes the activeness, which bumps the epoch only on actual changes.
    def __init__(self, active: bool = True) -> None:
        # [FUNCTION DOC] HELPER: Create the underlying tracker with the given initial activeness
        # (default true).
        self._tracker = ActivenessTracker(active=active)

    @property
    def active(self) -> bool:
        # [FUNCTION DOC] HELPER: Return the current activeness flag from the tracker.
        return self._tracker.get().active

    @active.setter
    def active(self, value: bool) -> None:
        # [FUNCTION DOC] HELPER: Publish the new activeness to the tracker (epoch bumps only when
        # the value differs from the current one).
        self._tracker.bump_active(value)

    def __call__(self) -> ActiveAndEpoch:
        # [FUNCTION DOC] HELPER: Return the tracker's current active-and-epoch pair; this makes the
        # instance usable as the checker's activeness getter.
        return self._tracker.get()


class TestActivenessTracker:
    # [CLASS DOC] Tests of the activeness tracker used to fence stale probes.
    def test_repeating_the_same_activeness_does_not_bump_the_epoch(self):
        # [FUNCTION DOC] Start active, publish inactive (epoch becomes 1), capture that state, then
        # publish inactive again; assert the captured state equals inactive at epoch 1 and that
        # the tracker state is unchanged by the repeat.
        """A redundant state publication must not invalidate in-flight probes or re-arm the grace period."""
        tracker = ActivenessTracker(active=True)
        tracker.bump_active(False)
        after_transition = tracker.get()

        tracker.bump_active(False)

        assert after_transition == ActiveAndEpoch(active=False, epoch=1)
        assert tracker.get() == after_transition


class TestStartStop:
    # [CLASS DOC] Lifecycle tests of starting and stopping the checker loop task.
    async def test_start_creates_task(self):
        # [FUNCTION DOC] No task before start; a task exists after start; stop clears it.
        checker, _ = _make_checker()
        assert checker._task is None

        checker.start()
        assert checker._task is not None

        checker.stop()
        assert checker._task is None

    async def test_start_is_idempotent(self):
        # [FUNCTION DOC] Starting twice keeps the very same task object.
        checker, _ = _make_checker()
        checker.start()
        task = checker._task

        checker.start()
        assert checker._task is task

        checker.stop()

    async def test_stop_without_start_is_noop(self):
        # [FUNCTION DOC] Stopping a never-started checker does not raise.
        checker, _ = _make_checker()
        checker.stop()


class TestCheckFnCalled:
    # [CLASS DOC] Tests of probe cadence relative to the first wait and the interval.
    async def test_check_fn_called_after_first_interval(self):
        # [FUNCTION DOC] With first wait zero and a 10 s interval: the first probe runs right after
        # start; after 5 s still one call; after another 5 s (full interval) two calls.
        call_count = 0

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Probe that counts its calls in the enclosing counter.
            nonlocal call_count
            call_count += 1

        checker, clock = _make_checker(check_fn=check_fn, interval=10.0)
        checker.start()

        # Step 1: first_wait=0, so first check runs immediately after task starts
        await _settle(clock)
        assert call_count == 1

        # Step 2: Elapse less than interval — no second check
        await clock.elapse(5.0)
        assert call_count == 1

        # Step 3: Elapse to interval — second check
        await clock.elapse(5.0)
        assert call_count == 2

        checker.stop()

    async def test_first_wait_delays_first_check(self):
        # [FUNCTION DOC] With a 300 s first wait and 10 s interval: no probe after 100 s; first
        # probe after reaching 300 s; second probe one interval later.
        call_count = 0

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Probe that counts its calls in the enclosing counter.
            nonlocal call_count
            call_count += 1

        checker, clock = _make_checker(check_fn=check_fn, first_wait=300.0, interval=10.0)
        checker.start()
        await _settle(clock)

        # Step 1: Elapse 100s — still in first_wait
        await clock.elapse(100.0)
        assert call_count == 0

        # Step 2: Elapse to 300s — first_wait completes, first check runs
        await clock.elapse(200.0)
        assert call_count == 1

        # Step 3: Elapse interval — second check
        await clock.elapse(10.0)
        assert call_count == 2

        checker.stop()


class TestOnResult:
    # [CLASS DOC] Tests of the per-probe result callback.
    async def test_on_result_true_on_success(self):
        # [FUNCTION DOC] A succeeding probe reports exactly one true result to the callback.
        results: list[bool] = []

        checker, clock = _make_checker(on_result=lambda s: results.append(s))
        checker.start()
        await _settle(clock)
        checker.stop()

        assert results == [True]

    async def test_on_result_false_on_failure(self):
        # [FUNCTION DOC] A raising probe reports exactly one false result.
        results: list[bool] = []

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Probe that always raises a runtime error.
            raise RuntimeError("boom")

        checker, clock = _make_checker(check_fn=check_fn, on_result=lambda s: results.append(s))
        checker.start()
        await _settle(clock)
        checker.stop()

        assert results == [False]

    async def test_loop_survives_on_result_raising(self, caplog):
        # [FUNCTION DOC] The callback records the result then raises. After the first probe and one
        # more interval, assert two true results were recorded (loop survived) and a log record
        # mentioning a failed result callback exists.
        """A raising on_result callback is logged and does not kill the check loop."""
        results: list[bool] = []

        def on_result(success: bool) -> None:
            # [FUNCTION DOC] HELPER: Record the result, then raise a runtime error.
            results.append(success)
            raise RuntimeError("callback boom")

        checker, clock = _make_checker(on_result=on_result, interval=5.0)
        checker.start()

        await _settle(clock)
        await clock.elapse(5.0)
        checker.stop()

        assert results == [True, True]
        assert any("on_result_failed" in r.message for r in caplog.records)

    async def test_loop_continues_after_failure(self):
        # [FUNCTION DOC] An always-failing probe keeps being run each interval: two false results
        # after the initial probe plus one interval.
        results: list[bool] = []

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Probe that always raises a runtime error.
            raise RuntimeError("boom")

        checker, clock = _make_checker(check_fn=check_fn, on_result=lambda s: results.append(s), interval=5.0)
        checker.start()

        await _settle(clock)
        await clock.elapse(5.0)
        checker.stop()

        assert results == [False, False]

    async def test_intermittent_failure(self):
        # [FUNCTION DOC] A probe failing on every even-numbered call: after the first probe results
        # are [true]; after three more intervals results alternate true, false, true, false.
        call_count = 0
        results: list[bool] = []

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Count calls and raise a runtime error on every second call.
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("intermittent")

        checker, clock = _make_checker(check_fn=check_fn, on_result=lambda s: results.append(s), interval=5.0)
        checker.start()
        await _settle(clock)
        # first_wait=0 so first check runs immediately on start
        assert results == [True]

        for _ in range(3):
            await clock.elapse(5.0)
        checker.stop()

        assert results == [True, False, True, False]


class TestActiveness:
    # [CLASS DOC] Tests of activeness gating of probes.
    async def test_an_inactive_checker_does_not_call_check_fn(self):
        # [FUNCTION DOC] Start with an inactive provider and run 20 s of fake time; assert the
        # probe was never called.
        """Probing a cell that is offloaded or not yet serving would report a false failure."""
        call_count = 0

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Probe that counts its calls in the enclosing counter.
            nonlocal call_count
            call_count += 1

        activeness = _Activeness(active=False)
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, activeness=activeness)

        checker.start()
        await _settle(clock)
        await clock.elapse(20.0)
        checker.stop()

        assert call_count == 0

    async def test_becoming_active_resumes_checking(self):
        # [FUNCTION DOC] Inactive for 20 s with no probes; then publish active and advance one
        # interval; assert at least one probe happened (the loop notices the change on its own).
        """Nothing calls a resume method any more, so the loop must pick the change up itself."""
        call_count = 0

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Probe that counts its calls in the enclosing counter.
            nonlocal call_count
            call_count += 1

        activeness = _Activeness(active=False)
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, activeness=activeness)

        checker.start()
        await _settle(clock)
        await clock.elapse(20.0)
        assert call_count == 0

        activeness.active = True
        await clock.elapse(5.0)
        checker.stop()

        assert call_count >= 1

    async def test_becoming_inactive_stops_checking_without_one_last_probe(self):
        # [FUNCTION DOC] After the first probe, publish inactive and advance 20 s; assert the call
        # count did not increase.
        """Activeness is read before deciding to probe, so an offloaded engine is never hit."""
        call_count = 0

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Probe that counts its calls in the enclosing counter.
            nonlocal call_count
            call_count += 1

        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, activeness=activeness)

        checker.start()
        await _settle(clock)
        calls_while_active = call_count

        activeness.active = False
        await clock.elapse(20.0)
        checker.stop()

        assert call_count == calls_while_active


class TestNeedFirstWait:
    # [CLASS DOC] Tests of when the initial grace period applies again.
    async def test_becoming_active_again_triggers_first_wait(self):
        # [FUNCTION DOC] First wait 100 s, interval 5 s: first probe at 100 s, second at 105 s.
        # Then an observed off/on cycle (5 s inactive, then active); assert no probe for the next
        # 5 s and 50 s, and the third probe only after the full 100 s grace passes again.
        """After a pause the engine may have been replaced, so the grace period applies again."""
        call_count = 0

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Probe that counts its calls in the enclosing counter.
            nonlocal call_count
            call_count += 1

        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, first_wait=100.0, interval=5.0, activeness=activeness)
        checker.start()
        await _settle(clock)

        # Step 1: Initial first_wait (100s)
        await clock.elapse(50.0)
        assert call_count == 0
        await clock.elapse(50.0)
        assert call_count == 1

        # Step 2: Normal interval (5s)
        await clock.elapse(5.0)
        assert call_count == 2

        # Step 3: a full off/on cycle observed by the loop resets first_wait
        activeness.active = False
        await clock.elapse(5.0)
        activeness.active = True

        # Step 4: the new first_wait (100s) must elapse before the next check
        await clock.elapse(5.0)
        assert call_count == 2

        await clock.elapse(50.0)
        assert call_count == 2

        await clock.elapse(50.0)
        assert call_count == 3

        checker.stop()

    async def test_going_inactive_does_not_re_arm_first_wait(self):
        # [FUNCTION DOC] After the grace period elapses the need-first-wait flag is false; going
        # inactive and letting the loop observe it keeps the flag false.
        """Only the transition back to active restarts the grace period."""
        activeness = _Activeness()
        checker, clock = _make_checker(first_wait=300.0, activeness=activeness)
        checker.start()
        await _settle(clock)
        await clock.elapse(300.0)
        assert checker._need_first_wait is False

        activeness.active = False
        await clock.elapse(10.0)
        assert checker._need_first_wait is False

        checker.stop()

    async def test_becoming_inactive_during_first_wait_skips_the_due_probe(self):
        # [FUNCTION DOC] First wait 100 s; at 50 s publish inactive; past the 100 s mark no probe
        # has run and the status is unknown; a further 10 s still yields no probe.
        """The probe that becomes due at the end of the grace period must not hit an engine offloaded meanwhile."""
        call_count = 0

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Probe that counts its calls in the enclosing counter.
            nonlocal call_count
            call_count += 1

        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, first_wait=100.0, interval=5.0, activeness=activeness)
        checker.start()
        await _settle(clock)

        await clock.elapse(50.0)
        assert call_count == 0

        activeness.active = False
        await clock.elapse(50.0)
        assert call_count == 0
        assert checker.status == TriState.UNKNOWN

        await clock.elapse(10.0)
        assert call_count == 0

        checker.stop()


class TestProbeTimeout:
    # [CLASS DOC] Tests of the per-probe timeout.
    async def test_a_timed_out_probe_is_reported_failed_and_polling_continues(self):
        # [FUNCTION DOC] Probe hangs for an hour with a 0.05 s timeout. Assert the first published
        # result is false, the probe was cancelled, the status is false; then after one more
        # interval a second false result is published (polling continued).
        """A check hanging past the timeout is cancelled, published as a failure, and followed by later polls."""
        results: list[bool] = []
        published = asyncio.Event()
        cancelled = asyncio.Event()

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Sleep for an hour; if cancelled, set the cancelled event and
            # re-raise the cancellation.
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        def on_result(success: bool) -> None:
            # [FUNCTION DOC] HELPER: Record the result and signal the published event.
            results.append(success)
            published.set()

        checker, clock = _make_checker(check_fn=check_fn, on_result=on_result, timeout=0.05, interval=5.0)
        checker.start()
        await asyncio.wait_for(published.wait(), timeout=10)

        assert results == [False]
        assert cancelled.is_set()
        assert checker.status == TriState.FALSE

        published.clear()
        await _settle(clock)
        await clock.elapse(5.0)
        await asyncio.wait_for(published.wait(), timeout=10)

        assert results == [False, False]
        checker.stop()


class TestTriState:
    # [CLASS DOC] Tests of the three-valued status (unknown, true, false) exposed by the checker.
    async def test_initial_status_is_unknown(self):
        # [FUNCTION DOC] A fresh checker reports unknown.
        checker, _ = _make_checker()
        assert checker.status == TriState.UNKNOWN

    async def test_healthy_after_successful_check(self):
        # [FUNCTION DOC] After one successful probe the status is true.
        checker, clock = _make_checker()
        checker.start()
        await _settle(clock)

        assert checker.status == TriState.TRUE
        checker.stop()

    async def test_unhealthy_after_failed_check(self):
        # [FUNCTION DOC] After one failing probe (threshold one) the status is false.
        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Probe that always raises a runtime error.
            raise RuntimeError("boom")

        checker, clock = _make_checker(check_fn=check_fn)
        checker.start()
        await _settle(clock)

        assert checker.status == TriState.FALSE
        checker.stop()

    async def test_stop_resets_to_unknown(self):
        # [FUNCTION DOC] After a successful probe status is true; stopping the checker resets it to
        # unknown.
        checker, clock = _make_checker()
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        checker.stop()
        assert checker.status == TriState.UNKNOWN

    async def test_going_inactive_resets_to_unknown(self):
        # [FUNCTION DOC] Status is true after a probe; after publishing inactive and advancing 10 s
        # the status is unknown.
        """A stale Healthy verdict about a sleeping engine would be read as a live replica."""
        activeness = _Activeness()
        checker, clock = _make_checker(activeness=activeness)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        activeness.active = False
        await clock.elapse(10.0)
        assert checker.status == TriState.UNKNOWN
        checker.stop()

    async def test_going_inactive_is_reported_before_the_loop_wakes_up(self):
        # [FUNCTION DOC] With a 30 s interval, publish inactive without advancing time; reading the
        # status immediately already returns unknown (the status read checks activeness live).
        """The pause is decided between two probes, and a reader in that window must not see the old verdict."""
        activeness = _Activeness()
        checker, clock = _make_checker(activeness=activeness, interval=30.0)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        activeness.active = False

        assert checker.status == TriState.UNKNOWN
        checker.stop()

    async def test_a_pause_and_resume_between_two_probes_hides_the_old_verdict_until_the_loop_catches_up(self):
        # [FUNCTION DOC] Status true; publish inactive then active before the loop wakes: status
        # reads unknown immediately (the epoch changed); after one interval the new probe makes it
        # true again.
        """A verdict from before the pause describes an engine that may have been replaced meanwhile."""
        activeness = _Activeness()
        checker, clock = _make_checker(activeness=activeness, interval=30.0)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        activeness.active = False
        activeness.active = True

        assert checker.status == TriState.UNKNOWN
        await clock.elapse(30.0)
        assert checker.status == TriState.TRUE
        checker.stop()

    async def test_a_failed_verdict_does_not_outlive_the_pause_that_follows_it(self):
        # [FUNCTION DOC] After a failed probe the status is false; publishing inactive makes the
        # status read unknown immediately.
        """A cell suspended right after a failed probe would otherwise be healed a second time."""

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Probe that always raises a runtime error.
            raise RuntimeError("boom")

        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, activeness=activeness, interval=30.0)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.FALSE

        activeness.active = False

        assert checker.status == TriState.UNKNOWN
        checker.stop()

    async def test_recovers_from_unhealthy_to_healthy(self):
        # [FUNCTION DOC] The first probe fails (status false), the second succeeds one interval
        # later (status true).
        call_count = 0

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Count calls and raise a runtime error only on the first call.
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")

        checker, clock = _make_checker(check_fn=check_fn, interval=5.0)
        checker.start()

        await _settle(clock)
        assert checker.status == TriState.FALSE

        await clock.elapse(5.0)
        assert checker.status == TriState.TRUE

        checker.stop()


class TestStatusWhileProbingIsPaused:
    # [CLASS DOC] Tests of the status read while the loop is mid-probe or after a pause.
    async def test_a_pause_is_reported_while_a_probe_is_still_in_flight(self):
        # [FUNCTION DOC] First probe succeeds (status true); the second probe blocks on a release
        # event. While it is in flight, publish inactive and assert the public status is unknown
        # while the internally stored status is still true; then release and stop.
        """The loop is stuck inside a probe, so only a live read can hide the verdict published before the pause."""
        started = asyncio.Event()
        release = asyncio.Event()
        call_count = 0

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Count calls; on the second call signal started and wait for
            # the release event.
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                started.set()
                await release.wait()

        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, activeness=activeness, interval=30.0, timeout=3600.0)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        await clock.elapse(30.0)
        await asyncio.wait_for(started.wait(), timeout=1)
        activeness.active = False

        assert checker.status == TriState.UNKNOWN
        assert checker._status == TriState.TRUE

        release.set()
        checker.stop()

    async def test_a_resumed_checker_serves_a_fresh_verdict_again(self):
        # [FUNCTION DOC] Status true; pause gives unknown; resume and advance one interval gives
        # true again (the hiding is not permanent).
        """Hiding the verdict lasts exactly as long as the pause instead of latching the checker at unknown."""
        activeness = _Activeness()
        checker, clock = _make_checker(activeness=activeness, interval=5.0)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        activeness.active = False
        assert checker.status == TriState.UNKNOWN

        activeness.active = True
        await clock.elapse(5.0)

        assert checker.status == TriState.TRUE
        checker.stop()


class TestFailureThresholdDebounce:
    # [CLASS DOC] Tests of debouncing: the public status flips to false only after the configured
    # number of consecutive failures, while raw results still go to the callback.
    """With failure_threshold > 1, transient failures must not flip the status to FALSE
    until that many consecutive checks have failed; any success resets the counter."""

    def _flaky_check_fn(self, outcomes: list[bool]):
        # [FUNCTION DOC] HELPER: Build a probe function that follows a scripted list of outcomes,
        # one per call (true succeeds, false raises a runtime error). Running past the end of the
        # list raises an index error from the probe.
        idx = 0

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Consume the next scripted outcome and raise a runtime error
            # if it is a failure.
            nonlocal idx
            ok = outcomes[idx]
            idx += 1
            if not ok:
                raise RuntimeError("boom")

        return check_fn

    async def test_below_threshold_keeps_previous_status(self):
        # [FUNCTION DOC] Threshold three; outcomes success then two failures: status stays true
        # while the consecutive-failure counter goes 1 then 2.
        # success, then 2 failures (< threshold 3): status stays TRUE.
        check_fn = self._flaky_check_fn([True, False, False])
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, failure_threshold=3)
        checker.start()
        await _settle(clock)
        assert checker.status == TriState.TRUE

        await clock.elapse(5.0)
        assert checker.status == TriState.TRUE
        assert checker._consecutive_failures == 1

        await clock.elapse(5.0)
        assert checker.status == TriState.TRUE
        assert checker._consecutive_failures == 2

        checker.stop()

    async def test_status_flips_false_only_at_threshold(self):
        # [FUNCTION DOC] Threshold three with three failures: status remains unknown after the
        # first and second failures and becomes false on the third.
        check_fn = self._flaky_check_fn([False, False, False])
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, failure_threshold=3)
        checker.start()

        await _settle(clock)
        assert checker.status == TriState.UNKNOWN  # 1st failure: below threshold, keep initial UNKNOWN

        await clock.elapse(5.0)
        assert checker.status == TriState.UNKNOWN  # 2nd failure: still below threshold

        await clock.elapse(5.0)
        assert checker.status == TriState.FALSE  # 3rd consecutive failure: threshold reached

        checker.stop()

    async def test_success_resets_failure_counter(self):
        # [FUNCTION DOC] Outcomes fail, fail, success, fail, fail with threshold three: counter
        # reaches two, resets to zero on success (status true), then reaches two again; status
        # stays true.
        # 2 failures, a success, then 2 more failures: never reaches 3 consecutive, stays TRUE.
        check_fn = self._flaky_check_fn([False, False, True, False, False])
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, failure_threshold=3)
        checker.start()

        await _settle(clock)  # fail 1
        await clock.elapse(5.0)  # fail 2
        assert checker._consecutive_failures == 2

        await clock.elapse(5.0)  # success -> reset
        assert checker.status == TriState.TRUE
        assert checker._consecutive_failures == 0

        await clock.elapse(5.0)  # fail 1
        await clock.elapse(5.0)  # fail 2
        assert checker.status == TriState.TRUE
        assert checker._consecutive_failures == 2

        checker.stop()

    async def test_on_result_reports_raw_per_check_not_debounced(self):
        # [FUNCTION DOC] With threshold three and outcomes success then three failures, the
        # callback receives the raw sequence true, false, false, false (no debouncing).
        results: list[bool] = []
        check_fn = self._flaky_check_fn([True, False, False, False])
        checker, clock = _make_checker(
            check_fn=check_fn, on_result=lambda s: results.append(s), interval=5.0, failure_threshold=3
        )
        checker.start()
        await _settle(clock)
        for _ in range(3):
            await clock.elapse(5.0)
        checker.stop()

        assert results == [True, False, False, False]

    async def test_becoming_active_again_resets_the_failure_counter(self):
        # [FUNCTION DOC] Two failures accumulate (counter two); after an observed inactive period
        # and reactivation plus one interval, the counter is at most one.
        """Failures against the old engine must not count toward recycling its replacement."""
        check_fn = self._flaky_check_fn([False, False, False])
        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, failure_threshold=3, activeness=activeness)
        checker.start()
        await _settle(clock)
        await clock.elapse(5.0)
        assert checker._consecutive_failures == 2

        activeness.active = False
        await clock.elapse(5.0)
        activeness.active = True
        await clock.elapse(5.0)

        assert checker._consecutive_failures <= 1
        checker.stop()

    async def test_a_pause_resume_completed_between_two_polls_resets_the_failure_counter(self):
        # [FUNCTION DOC] First wait 100 s, interval 10 s, threshold three; two failures accumulate.
        # Then pause and resume without time passing and advance one interval: the counter is
        # zero and the status is unknown.
        """A reconfigure that flips activeness off and on while the loop sleeps must still clear stale failures."""
        check_fn = self._flaky_check_fn([False, False, False])
        activeness = _Activeness()
        checker, clock = _make_checker(
            check_fn=check_fn, first_wait=100.0, interval=10.0, failure_threshold=3, activeness=activeness
        )
        checker.start()
        await _settle(clock)

        await clock.elapse(100.0)
        await clock.elapse(10.0)
        assert checker._consecutive_failures == 2

        activeness.active = False
        activeness.active = True
        await clock.elapse(10.0)

        assert checker._consecutive_failures == 0
        assert checker.status == TriState.UNKNOWN
        checker.stop()

    async def test_a_pause_resume_completed_between_two_polls_re_arms_the_first_wait(self):
        # [FUNCTION DOC] After the first probe at 100 s, pause and resume without time passing:
        # the next interval (10 s) produces no probe, and the next probe comes only after another
        # full 100 s grace period.
        """The unobserved reconfigure also restarts the grace period, so the replacement is not probed at once."""
        call_count = 0

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Probe that counts its calls in the enclosing counter.
            nonlocal call_count
            call_count += 1

        activeness = _Activeness()
        checker, clock = _make_checker(
            check_fn=check_fn, first_wait=100.0, interval=10.0, failure_threshold=3, activeness=activeness
        )
        checker.start()
        await _settle(clock)

        await clock.elapse(100.0)
        assert call_count == 1

        activeness.active = False
        activeness.active = True
        await clock.elapse(10.0)
        assert call_count == 1

        await clock.elapse(100.0)
        assert call_count == 2
        checker.stop()


class TestDiscardingStaleProbeResults:
    # [CLASS DOC] Tests that results of probes which straddled a pause (epoch change) are dropped.
    def _hanging_check_fn(self, started: asyncio.Event):
        # [FUNCTION DOC] HELPER: Build a probe that signals the started event and then sleeps for an
        # hour (never completing on its own).
        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Signal started, then sleep for an hour.
            started.set()
            await asyncio.sleep(3600)

        return check_fn

    def _gated_check_fn(self, started: asyncio.Event, release: asyncio.Event):
        # [FUNCTION DOC] HELPER: Build a probe that signals started, waits for the release event,
        # and then raises a runtime error (so a landed result is a failure).
        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Signal started, wait for release, then raise.
            started.set()
            await release.wait()
            raise RuntimeError("boom")

        return check_fn

    async def test_a_probe_that_lands_after_a_pause_publishes_no_result(self):
        # [FUNCTION DOC] Start a gated failing probe, publish inactive, release it: assert no
        # result is delivered, the failure counter is zero and the status is unknown.
        """A probe that outlives its window would report a failure about an engine nobody was watching."""
        results: list[bool] = []
        started = asyncio.Event()
        release = asyncio.Event()
        activeness = _Activeness()
        checker, clock = _make_checker(
            check_fn=self._gated_check_fn(started, release),
            on_result=lambda s: results.append(s),
            interval=5.0,
            activeness=activeness,
        )
        checker.start()
        await asyncio.wait_for(started.wait(), timeout=1)

        activeness.active = False
        release.set()
        await _settle(clock)

        assert results == []
        assert checker._consecutive_failures == 0
        assert checker.status == TriState.UNKNOWN
        checker.stop()

    async def test_a_probe_that_spans_a_whole_pause_resume_window_publishes_no_result(self):
        # [FUNCTION DOC] Start a gated failing probe, publish inactive then active, release it:
        # activeness is true again at landing but the epoch changed, so no result is delivered and
        # the failure counter stays zero.
        """Activeness is back to True when the probe lands, so only the epoch tells the result is stale."""
        results: list[bool] = []
        started = asyncio.Event()
        release = asyncio.Event()
        activeness = _Activeness()
        checker, clock = _make_checker(
            check_fn=self._gated_check_fn(started, release),
            on_result=lambda s: results.append(s),
            interval=5.0,
            activeness=activeness,
        )
        checker.start()
        await asyncio.wait_for(started.wait(), timeout=1)

        activeness.active = False
        activeness.active = True
        release.set()
        await _settle(clock)

        assert results == []
        assert checker._consecutive_failures == 0
        checker.stop()

    async def test_the_loop_keeps_polling_after_a_discarded_result(self):
        # [FUNCTION DOC] The first probe blocks until released across a pause/resume (its result is
        # discarded); after settling and one interval a second probe runs: call count is two.
        """Discarding one result must not silently kill the checker for the rest of the run."""
        call_count = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Count calls; on the first call signal started and wait for
            # release; later calls return immediately.
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                started.set()
                await release.wait()

        activeness = _Activeness()
        checker, clock = _make_checker(check_fn=check_fn, interval=5.0, activeness=activeness)
        checker.start()
        await asyncio.wait_for(started.wait(), timeout=1)

        activeness.active = False
        activeness.active = True
        release.set()
        await _settle(clock)

        await clock.elapse(5.0)

        assert call_count == 2
        checker.stop()

    async def test_a_successful_probe_spanning_pause_resume_is_discarded(self):
        # [FUNCTION DOC] A probe that would succeed straddles a pause/resume; after release no
        # result is delivered and the status stays unknown.
        """An old engine's success must not mark its replacement healthy without probing it."""
        results: list[bool] = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Signal started and wait for release, then succeed.
            started.set()
            await release.wait()

        activeness = _Activeness()
        checker, clock = _make_checker(
            check_fn=check_fn, on_result=lambda s: results.append(s), interval=5.0, activeness=activeness
        )
        checker.start()
        await asyncio.wait_for(started.wait(), timeout=1)

        activeness.active = False
        activeness.active = True
        release.set()
        await _settle(clock)

        assert results == []
        assert checker.status == TriState.UNKNOWN
        checker.stop()

    async def test_a_probe_that_lands_inside_its_own_window_is_published(self):
        # [FUNCTION DOC] A gated failing probe released without any activeness change delivers its
        # false result exactly once.
        """Discarding is for stale results only; an undisturbed window must still publish its verdict."""
        results: list[bool] = []
        started = asyncio.Event()
        release = asyncio.Event()
        checker, clock = _make_checker(
            check_fn=self._gated_check_fn(started, release), on_result=lambda s: results.append(s), interval=5.0
        )
        checker.start()
        await asyncio.wait_for(started.wait(), timeout=1)

        release.set()
        await _settle(clock)

        assert results == [False]
        checker.stop()

    async def test_stopping_also_kills_a_probe_still_in_flight(self):
        # [FUNCTION DOC] With a hanging probe in flight, stop the checker; awaiting the saved probe
        # task raises cancellation and the task reports cancelled.
        """A probe left running after stop() outlives the cell and keeps dialing a dead engine."""
        started = asyncio.Event()
        checker, _ = _make_checker(check_fn=self._hanging_check_fn(started), interval=5.0)
        checker.start()
        await asyncio.wait_for(started.wait(), timeout=1)
        probe_task = checker._probe_task

        checker.stop()

        with pytest.raises(asyncio.CancelledError):
            await probe_task
        assert probe_task.cancelled()

    async def test_stopping_before_a_probe_starts_does_not_abandon_its_coroutine(
        self, recwarn: WarningsRecorder
    ) -> None:
        # [FUNCTION DOC] Schedule the checker's probe-run coroutine as a task, yield once, stop the
        # checker, and assert awaiting the task raises cancellation and that no "never awaited"
        # warning was recorded.
        """Stopping a newly scheduled probe must not leave its coroutine unawaited."""
        checker, _ = _make_checker(interval=5.0)
        run_probe = asyncio.create_task(checker._run_probe())
        await asyncio.sleep(0)

        checker.stop()

        with pytest.raises(asyncio.CancelledError):
            await run_probe
        assert not [warning for warning in recwarn if "was never awaited" in str(warning.message)]


class TestNoopHealthChecker:
    # [CLASS DOC] Tests of the do-nothing health checker.
    def test_noop_status_is_always_unknown(self):
        # [FUNCTION DOC] The no-op checker reports unknown status.
        checker = NoopHealthChecker()
        assert checker.status == TriState.UNKNOWN


class TestFailureThresholdArgument:
    # [CLASS DOC] Tests of the failure-threshold command-line option registered by the checker
    # config under a caller-chosen prefix (here "demo-check").
    def _parse(self, extra: list[str], **add_arguments_kwargs: int) -> argparse.Namespace:
        # [FUNCTION DOC] HELPER: Create a fresh parser, register the config's arguments with the
        # demo prefix and any extra keyword options (such as a custom threshold default), and
        # return the namespace parsed from the given argument list.
        parser = argparse.ArgumentParser()
        SimpleHealthCheckerConfig.add_arguments(parser, prefix="demo-check", **add_arguments_kwargs)
        return parser.parse_args(extra)

    def test_a_caller_can_ask_for_a_debounce_of_its_own(self):
        # [FUNCTION DOC] Registering with a threshold default of one and passing no flag yields a
        # threshold of one.
        """A checker polling on a long interval needs to report the first failure, not the third."""
        args = self._parse([], failure_threshold_default=1)

        assert args.demo_check_failure_threshold == 1

    def test_a_caller_that_asks_for_nothing_keeps_the_shared_debounce(self):
        # [FUNCTION DOC] Registering with no custom default and passing no flag yields the shared
        # default threshold of three.
        """Giving one caller a tighter default must not tighten it for every other checker."""
        args = self._parse([])

        assert args.demo_check_failure_threshold == 3

    def test_an_explicit_flag_still_beats_a_caller_supplied_default(self):
        # [FUNCTION DOC] With a custom default of one, passing the flag with value five yields five.
        """A caller default that is nailed into the parser would make the flag unusable."""
        args = self._parse(["--demo-check-failure-threshold", "5"], failure_threshold_default=1)

        assert args.demo_check_failure_threshold == 5


class TestShippedRolloutConfig:
    # [CLASS DOC] Tests against the project's real command-line defaults for the rollout health check.
    async def test_a_dead_engine_is_reported_unhealthy_after_a_single_probe_under_the_shipped_defaults(self):
        # [FUNCTION DOC] Build the project's full extra-argument parser, parse with only a rollout
        # batch size, and build the checker config for the rollout health-check prefix. Run a
        # checker whose probe always raises: after the first probe the status is false and the
        # consecutive-failure counter is one (shipped threshold is one).
        """A cell whose engine is already gone must not read Healthy for two more 30s probe intervals."""
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        config = SimpleHealthCheckerConfig.from_args(
            parser.parse_args(["--rollout-batch-size", "64"]), prefix="rollout_health_check"
        )

        async def check_fn() -> None:
            # [FUNCTION DOC] HELPER: Probe that always raises a runtime error ("engine down").
            raise RuntimeError("engine down")

        clock = FakeClock()
        checker = SimpleHealthChecker(
            name="rollout-cell",
            check_fn=check_fn,
            get_activeness=_Activeness(),
            config=config,
            clock=clock,
        )
        checker.start()
        await _settle(clock)

        assert checker.status == TriState.FALSE
        assert checker._consecutive_failures == 1
        checker.stop()
