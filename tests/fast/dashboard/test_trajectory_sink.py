# [MODULE DOC]
# Tests for the trajectory event sink in the dashboard hooks: event emission carrying sample
# identity and the latest weight version, explicit-time generation and tool spans, draining of
# per-turn lifecycle metadata at attempt end (including agent gaps bounded by request arrival),
# batching with swallowed push failures, attaching and detaching through the global lifecycle
# seam, and a store round trip of trajectory events across hour partitions. Uses a fake
# collector handle with a recording trajectory push method; an autouse fixture clears the global
# lifecycle sink before and after each test.
import pytest

from miles.dashboard import hooks
from miles.dashboard.hooks import BATCH_MAX_EVENTS, TrajectorySink
from miles.dashboard.store import Stream, TrajectoryEvent, TrajectoryEventKind
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


class FakeRemoteMethod:
    # [CLASS DOC] HELPER: Stand-in for a ray remote method wrapper that records calls.
    def __init__(self, fail=False):
        # [FUNCTION DOC] HELPER: Initialises an empty call list and the failure flag.
        self.calls = []
        self.fail = fail

    def remote(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Raises a runtime error (collector unreachable) if configured to
        # fail; otherwise records the positional and keyword arguments as one entry.
        if self.fail:
            raise RuntimeError("collector unreachable")
        self.calls.append((args, kwargs))


class FakeHandle:
    # [CLASS DOC] HELPER: Fake collector handle exposing only a recording trajectory push method.
    def __init__(self, fail_push=False):
        # [FUNCTION DOC] HELPER: Creates the recording method, failing when fail_push is set.
        self.push_trajectories = FakeRemoteMethod(fail=fail_push)


@pytest.fixture(autouse=True)
def clean_sink():
    # [FUNCTION DOC] HELPER: Autouse fixture. Clears the global trajectory lifecycle's sink before
    # the test and again after it.
    TrajectoryLifecycle().sink = None
    yield
    TrajectoryLifecycle().sink = None


def _sample(index=7, group=2, versions=("3", "4")):
    # [FUNCTION DOC] HELPER: Builds a sample with the given index (default 7), group (default 2)
    # and one generation call per version, each a single span of that version over consecutive
    # one-token ranges.
    return Sample(
        index=index,
        group_index=group,
        weight_versions=[
            WeightVersionsPerCall(spans=[WeightVersionSpan(version, i, i + 1)]) for i, version in enumerate(versions)
        ],
    )


def _pushed(handle):
    # [FUNCTION DOC] HELPER: Flattens all event batches pushed to the fake handle into one list in
    # push order.
    return [event for (args, _) in handle.push_trajectories.calls for event in args[0]]


# --------------------------------- emitters ----------------------------------


def test_attempt_and_gen_events_carry_identity_and_version():
    # [FUNCTION DOC] Emitting attempt start, generation start and attempt end for one sample then
    # flushing pushes events of those kinds in order; each carries sample index 7, group 2 and the
    # last weight version "4"; the attempt end's detail is the sample's pending status value.
    handle = FakeHandle()
    sink = TrajectorySink(handle)
    sample = _sample()
    sink.attempt_start(sample)
    sink.gen_start(sample)
    sink.attempt_end(sample)
    sink.flush()

    kinds = [e.kind for e in _pushed(handle)]
    assert kinds == [
        TrajectoryEventKind.ATTEMPT_START,
        TrajectoryEventKind.GEN_START,
        TrajectoryEventKind.ATTEMPT_END,
    ]
    for event in _pushed(handle):
        assert (event.sample_index, event.group_index) == (7, 2)
        assert event.weight_version == "4"
    assert _pushed(handle)[-1].detail == Sample.Status.PENDING.value


def test_event_without_weight_version_spans_uses_empty_version():
    # [FUNCTION DOC] A sample with no version spans still emits an attempt-start event, with
    # sample index 7 and an empty version string.
    """A sample whose generation reported no weight version is still emitted, with an empty version."""
    handle = FakeHandle()
    sink = TrajectorySink(handle)

    sink.attempt_start(_sample(versions=()))
    sink.flush()

    [event] = _pushed(handle)
    assert (event.kind, event.sample_index, event.weight_version) == (TrajectoryEventKind.ATTEMPT_START, 7, "")


def test_event_uses_last_span_from_multi_span_generation_call():
    # [FUNCTION DOC] For a sample whose last call has spans "4" then "5", the generation-start
    # event's version is "5" (last span of the last call).
    """The reported version is the last span of the last call, not that call's first span."""
    handle = FakeHandle()
    sample = Sample(
        index=7,
        group_index=2,
        weight_versions=[
            WeightVersionsPerCall(spans=[WeightVersionSpan("3", 0, 1)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("4", 1, 2), WeightVersionSpan("5", 2, 3)]),
        ],
    )
    sink = TrajectorySink(handle)

    sink.gen_start(sample)
    sink.flush()

    [event] = _pushed(handle)
    assert event.weight_version == "5"


def test_spans_use_explicit_timestamps_and_turns():
    # [FUNCTION DOC] A generation span 10.0..14.5 (turn 3, detail "120") and a tool span
    # 14.5..16.0 (turn 3) push four events in order: gen start, gen end, tool start, tool end, with
    # the given timestamps and turn; the gen end event carries detail "120".
    handle = FakeHandle()
    sink = TrajectorySink(handle)
    sample = _sample()
    sink.gen_span(sample, 10.0, 14.5, turn=3, detail="120")
    sink.tool_span(sample, 14.5, 16.0, turn=3, detail="2 calls")
    sink.flush()

    events = _pushed(handle)
    assert [(e.kind, e.ts, e.turn) for e in events] == [
        (TrajectoryEventKind.GEN_START, 10.0, 3),
        (TrajectoryEventKind.GEN_END, 14.5, 3),
        (TrajectoryEventKind.TOOL_START, 14.5, 3),
        (TrajectoryEventKind.TOOL_END, 16.0, 3),
    ]
    assert events[1].detail == "120"


def test_attempt_end_drains_lifecycle_metadata_from_turn_samples():
    # [FUNCTION DOC] Two turn samples carry lifecycle metadata (start, end, turn); turn 1 has an
    # unknown start (only its end is emitted); turn 2 has a previous end of 110.0 and a request
    # arrival of 110.6. Ending the attempt with the list of both emits: gen end (turn 1), tool
    # start and tool end (turn 2, spanning 110.0 to 110.6, the arrival and not the generation
    # start, detail "agent gap"), gen start and gen end (turn 2), then the attempt end with turn -1.
    handle = FakeHandle()
    sink = TrajectorySink(handle)
    turns = []
    for i in range(2):
        sample = _sample()
        sample.metadata["lifecycle"] = dict(t0=100.0 + i, t1=110.0 + i, turn=i + 1)
        turns.append(sample)
    turns[0].metadata["lifecycle"]["t0"] = None  # unknown start: only the end emits
    turns[1].metadata["lifecycle"]["prev_t1"] = 110.0  # session gap = agent/tool work
    turns[1].metadata["lifecycle"]["req_ts"] = 110.6  # server-edge arrival bounds it

    sink.attempt_end(turns)
    sink.flush()
    events = _pushed(handle)
    assert [(e.kind, e.turn) for e in events] == [
        (TrajectoryEventKind.GEN_END, 1),
        (TrajectoryEventKind.TOOL_START, 2),
        (TrajectoryEventKind.TOOL_END, 2),
        (TrajectoryEventKind.GEN_START, 2),
        (TrajectoryEventKind.GEN_END, 2),
        (TrajectoryEventKind.ATTEMPT_END, -1),
    ]
    assert (events[1].ts, events[2].ts) == (110.0, 110.6)  # gap ends at arrival, not gen start
    assert events[1].detail == "agent gap"


def test_batching_and_failure_swallowed():
    # [FUNCTION DOC] One fewer than the batch maximum of events pushes nothing; one more pushes a
    # full batch. A sink over a failing handle fed more than a batch of events, then flushed,
    # never raises.
    handle = FakeHandle()
    sink = TrajectorySink(handle)
    sample = _sample()
    for _ in range(BATCH_MAX_EVENTS - 1):
        sink.gen_start(sample)
    assert handle.push_trajectories.calls == []
    sink.gen_start(sample)  # crosses the batch threshold
    assert len(_pushed(handle)) == BATCH_MAX_EVENTS

    failing = TrajectorySink(FakeHandle(fail_push=True))
    for _ in range(BATCH_MAX_EVENTS + 1):
        failing.gen_start(sample)  # must never raise into the rollout path
    failing.flush()


def test_attach_detach_via_lifecycle_seam():
    # [FUNCTION DOC] Initially no lifecycle sink. Attaching installs a trajectory sink; attaching
    # again with another handle keeps the same sink (one per process). An emitted event followed
    # by detach-and-flush clears the lifecycle sink and pushed exactly that one event.
    assert TrajectoryLifecycle().sink is None
    handle = FakeHandle()
    hooks.attach_trajectory_sink(handle)
    sink = TrajectoryLifecycle().sink
    assert isinstance(sink, TrajectorySink)
    hooks.attach_trajectory_sink(FakeHandle())
    assert TrajectoryLifecycle().sink is sink  # one sink per process

    sink.gen_start(_sample())
    hooks.detach_and_flush()
    assert TrajectoryLifecycle().sink is None
    assert len(_pushed(handle)) == 1  # detach flushed the partial batch


def test_trajectory_events_roundtrip_store(tmp_path):
    # [FUNCTION DOC] Writes two generation-start events an hour-plus apart (t=10 and t=3700),
    # flushes and loads. The store reports the trajectory stream present, the partition directory
    # holds two hourly files, the event query returns timestamps [10, 3700] (and [3700] when
    # starting at 3600), and the record iterator yields events with version "2".
    # KNOWN ISSUE: the file-count assertion wraps a lazy glob in a boolean expression, so it
    # effectively measures the list of files and is needlessly convoluted; it also re-imports the
    # store class locally.
    from miles.dashboard.store import MetricStore

    writer = MetricStore(tmp_path)
    for ts in (10.0, 3700.0):
        writer.append(
            TrajectoryEvent(
                ts=ts,
                kind=TrajectoryEventKind.GEN_START,
                sample_index=5,
                group_index=-1,
                turn=1,
                weight_version="2",
                detail="",
            )
        )
    writer.flush()
    store = MetricStore.load(tmp_path)
    assert store.has_stream(Stream.TRAJECTORIES)
    assert len((tmp_path / "trajectories").glob("*.jsonl") and list((tmp_path / "trajectories").glob("*.jsonl"))) == 2
    assert [e.ts for e in store.trajectory_events()] == [10.0, 3700.0]
    assert [e.ts for e in store.trajectory_events(t0=3600.0)] == [3700.0]
    assert store.iter_records(Stream.TRAJECTORIES)[0].weight_version == "2"
