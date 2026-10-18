# [MODULE DOC] Tests for the polling worker helper (a daemon thread running a target that receives
# a stop event; stop-and-join with a bounded wait that rethrows a worker failure; a not-running
# assertion) and for the poll-until-stopped loop. Module constants: a 5 second join timeout and a
# fixed worker thread name.
import threading

import pytest

from miles.utils.test_utils.polling_worker import PollingWorker, poll_until_stopped

_JOIN_TIMEOUT_SECONDS = 5.0
_WORKER_NAME = "test-polling-worker"


def _find_thread(name: str) -> threading.Thread | None:
    # [FUNCTION DOC] HELPER: Returns the first live thread with the given name, or none.
    return next((thread for thread in threading.enumerate() if thread.name == name), None)


class TestPollingWorker:
    # [CLASS DOC] Tests of the worker thread wrapper.
    def test_the_target_runs_on_a_daemon_thread_driven_by_the_stop_event(self) -> None:
        # [FUNCTION DOC] Starts a worker whose target records the stop event it received, signals
        # that it started, and waits on the event. Asserts it started, reports running, runs on a
        # daemon thread with the given name; after stop-and-join asserts it is no longer running
        # and the target received exactly one event, which is now set.
        """The worker must not keep the process alive, and its target needs the event to notice a stop."""
        received: list[threading.Event] = []
        started = threading.Event()

        def run(stop_event: threading.Event) -> None:
            # [FUNCTION DOC] HELPER: Records the event, signals start, then waits on the event up
            # to the join timeout.
            received.append(stop_event)
            started.set()
            stop_event.wait(timeout=_JOIN_TIMEOUT_SECONDS)

        worker = PollingWorker(name=_WORKER_NAME, run=run)
        worker.start()

        assert started.wait(timeout=_JOIN_TIMEOUT_SECONDS)
        assert worker.is_running
        thread = _find_thread(_WORKER_NAME)
        assert thread is not None and thread.daemon

        worker.stop_and_join(timeout_seconds=_JOIN_TIMEOUT_SECONDS)

        assert not worker.is_running
        assert len(received) == 1 and received[0].is_set()

    def test_stop_and_join_returns_with_the_worker_gone_within_the_timeout(self) -> None:
        # [FUNCTION DOC] A worker waiting on its stop event is stopped and joined; asserts it is
        # not running afterwards.
        """A worker still alive after the bounded join would race whatever reads its log next."""
        worker = PollingWorker(
            name=_WORKER_NAME, run=lambda stop_event: stop_event.wait(timeout=_JOIN_TIMEOUT_SECONDS)
        )
        worker.start()

        worker.stop_and_join(timeout_seconds=_JOIN_TIMEOUT_SECONDS)

        assert not worker.is_running

    def test_a_failure_on_the_worker_thread_is_rethrown_by_join(self) -> None:
        # [FUNCTION DOC] A target that raises a key error with message missing label: stop-and-join
        # rethrows that error in the caller.
        """A worker that died on an exception must fail the caller instead of silently ending the polling."""

        def fail(stop_event: threading.Event) -> None:
            # [FUNCTION DOC] HELPER: Always raises the key error.
            raise KeyError("missing label")

        worker = PollingWorker(name=_WORKER_NAME, run=fail)
        worker.start()

        with pytest.raises(KeyError, match="missing label"):
            worker.stop_and_join(timeout_seconds=_JOIN_TIMEOUT_SECONDS)

    def test_a_join_that_times_out_does_not_rethrow_a_failure_yet(self) -> None:
        # [FUNCTION DOC] A target waits for a release event then raises a key error. A join with a
        # 0.01 second timeout returns without raising and the worker is still running; after the
        # release, stop-and-join rethrows the late key error.
        """A still-running worker owns its exception slot, so a timed-out join must not read it early."""
        release = threading.Event()

        def fail_after_release(stop_event: threading.Event) -> None:
            # [FUNCTION DOC] HELPER: Waits for the release event (bounded), then raises the key
            # error.
            release.wait(timeout=_JOIN_TIMEOUT_SECONDS)
            raise KeyError("late failure")

        worker = PollingWorker(name=_WORKER_NAME, run=fail_after_release)
        worker.start()

        worker.join(timeout_seconds=0.01)
        assert worker.is_running

        release.set()
        with pytest.raises(KeyError, match="late failure"):
            worker.stop_and_join(timeout_seconds=_JOIN_TIMEOUT_SECONDS)

    def test_assert_not_running_raises_the_given_message_while_the_thread_is_alive(self) -> None:
        # [FUNCTION DOC] While the worker thread is alive, the not-running assertion raises an
        # assertion error carrying the given message; cleanup releases the worker and joins it.
        """The caller's message names which worker outlived its join, so it must reach the failure."""
        release = threading.Event()
        worker = PollingWorker(name=_WORKER_NAME, run=lambda stop_event: release.wait(timeout=_JOIN_TIMEOUT_SECONDS))
        worker.start()

        try:
            with pytest.raises(AssertionError, match="injector still running"):
                worker.assert_not_running(message="injector still running")
        finally:
            release.set()
            worker.stop_and_join(timeout_seconds=_JOIN_TIMEOUT_SECONDS)

    def test_assert_not_running_passes_after_the_worker_stopped(self) -> None:
        # [FUNCTION DOC] After stop-and-join of a cooperative worker, the not-running assertion
        # passes silently.
        """A worker that honoured its stop event must pass the check its callers gate the log read on."""
        worker = PollingWorker(
            name=_WORKER_NAME, run=lambda stop_event: stop_event.wait(timeout=_JOIN_TIMEOUT_SECONDS)
        )
        worker.start()
        worker.stop_and_join(timeout_seconds=_JOIN_TIMEOUT_SECONDS)

        worker.assert_not_running(message="worker should be gone")


class TestPollUntilStopped:
    # [CLASS DOC] Tests of the loop that calls a tick function repeatedly, waiting the poll
    # interval on the stop event between ticks, until the event is set.
    def test_it_keeps_ticking_until_the_stop_event_is_set(self) -> None:
        # [FUNCTION DOC] The tick counts calls and sets the event on the third; with a zero
        # interval, asserts exactly three ticks happened.
        """The loop is what drives the periodic work, so it must not stop before the event is set."""
        stop_event = threading.Event()
        num_ticks = 0

        def tick() -> None:
            # [FUNCTION DOC] HELPER: Increments the outer counter and sets the stop event when it
            # reaches 3.
            nonlocal num_ticks
            num_ticks += 1
            if num_ticks == 3:
                stop_event.set()

        poll_until_stopped(stop_event, tick=tick, poll_interval_seconds=0.0)

        assert num_ticks == 3

    def test_an_already_stopped_event_prevents_any_tick(self) -> None:
        # [FUNCTION DOC] With the event already set, the loop performs no tick.
        """A worker stopped before its loop began must not inject anything the caller no longer expects."""
        stop_event = threading.Event()
        stop_event.set()
        ticks: list[None] = []

        poll_until_stopped(stop_event, tick=lambda: ticks.append(None), poll_interval_seconds=0.0)

        assert ticks == []

    def test_the_wait_between_ticks_uses_the_configured_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Replaces the event's wait method (reverted at teardown) with a recorder
        # that stores the timeout, sets the event and returns true. With interval 0.25 and a no-op
        # tick, asserts the wait was called exactly once with 0.25.
        """A hard-coded sleep would ignore the caller's pacing and make the loop unstoppable for that long."""
        stop_event = threading.Event()
        waited: list[float | None] = []

        def recording_wait(timeout: float | None = None) -> bool:
            # [FUNCTION DOC] HELPER: Records the timeout, sets the event, returns true.
            waited.append(timeout)
            stop_event.set()
            return True

        monkeypatch.setattr(stop_event, "wait", recording_wait)

        poll_until_stopped(stop_event, tick=lambda: None, poll_interval_seconds=0.25)

        assert waited == [0.25]
