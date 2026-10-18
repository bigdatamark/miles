# [MODULE DOC] Tests of the timer utility's event sinks: callables registered on the singleton
# timer that receive (name, start time, end time) whenever a named interval ends, and optionally
# a begin hook called when an interval starts. Verifies the interval reported to a sink, that
# timer behavior is unchanged without sinks, that multiple sinks are called in registration order
# on every end including nested and inverse timers, and that a sink with a begin method hears
# starts while the interval is still open.
import time

import pytest

from miles.utils.timer import Timer, inverse_timer, timer


@pytest.fixture
def fresh_timer():
    # [FUNCTION DOC] HELPER: Fixture. Gets the timer instance, saves its timers mapping, start
    # times and sinks, replaces them with empty containers for the test, yields the instance, and
    # restores the saved values afterwards.
    instance = Timer()
    saved = (instance.timers, instance.start_time, instance.event_sinks)
    instance.timers, instance.start_time, instance.event_sinks = {}, {}, []
    yield instance
    instance.timers, instance.start_time, instance.event_sinks = saved


def test_sink_receives_interval(fresh_timer):
    # [FUNCTION DOC] Registers a recording sink, times a short sleep (0.01 seconds) inside the
    # timer context manager, and asserts exactly one event with the phase name whose start and end
    # times lie between wall-clock readings taken before and after, and whose duration matches
    # the recorded timer total approximately.
    events = []
    fresh_timer.event_sinks.append(lambda name, t0, t1: events.append((name, t0, t1)))

    before = time.time()
    with timer("phase_a"):
        time.sleep(0.01)
    after = time.time()

    [(name, t0, t1)] = events
    assert name == "phase_a"
    assert before <= t0 <= t1 <= after
    assert t1 - t0 == pytest.approx(fresh_timer.timers["phase_a"])


def test_no_sinks_keeps_behavior_identical(fresh_timer):
    # [FUNCTION DOC] With no sinks, timing a trivial phase leaves exactly that phase in the
    # timers mapping and no pending start times.
    with timer("phase_b"):
        pass
    assert list(fresh_timer.timers) == ["phase_b"]
    assert fresh_timer.start_time == {}


def test_multiple_sinks_and_nested_timers(fresh_timer):
    # [FUNCTION DOC] Registers two tagging sinks, starts an outer timer, runs an inverse timer on
    # it (which ends the outer and restarts it afterwards) around an inner timer, then ends the
    # outer. Asserts the sinks alternate in registration order for each of the three ends, and the
    # ended names are outer, inner, outer (two events each).
    events = []
    fresh_timer.event_sinks.append(lambda *e: events.append(("s1", e[0])))
    fresh_timer.event_sinks.append(lambda *e: events.append(("s2", e[0])))

    fresh_timer.start("outer")
    with inverse_timer("outer"):
        with timer("inner"):
            pass
    fresh_timer.end("outer")

    tags = [tag for tag, _ in events]
    assert tags == ["s1", "s2"] * 3  # every end() fans out to every sink, in order
    timer_names = [name for _, name in events]
    # inverse_timer ends "outer" first, "inner" ends inside, "outer" ends last
    assert timer_names == ["outer", "outer", "inner", "inner", "outer", "outer"]


def test_sink_with_begin_hears_starts():
    # [FUNCTION DOC] Uses a timer instance whose name shadows the imported context manager. Adds
    # a sink with a begin hook and an end call, starts a phase and asserts the begin hook was
    # called with the phase name and no end yet; after ending, asserts the end was recorded. In
    # cleanup, clears the sinks and resets the timer.
    timer = Timer()
    begins, ends = [], []

    class Sink:
        # [CLASS DOC] HELPER: Recording sink with a begin hook (name and start time) and an end
        # call (name, start, end).
        def begin(self, name, t0):
            # [FUNCTION DOC] HELPER: Records the name and start time in the begins list.
            begins.append((name, t0))

        def __call__(self, name, t0, t1):
            # [FUNCTION DOC] HELPER: Records only the name in the ends list.
            ends.append(name)

    timer.event_sinks.append(Sink())
    try:
        timer.start("long_phase")
        assert [name for name, _ in begins] == ["long_phase"]  # visible while open
        assert ends == []
        timer.end("long_phase")
        assert ends == ["long_phase"]
    finally:
        timer.event_sinks.clear()
        timer.reset()
