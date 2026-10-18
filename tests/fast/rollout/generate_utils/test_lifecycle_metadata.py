# [MODULE DOC] Tests of attaching trajectory lifecycle timing to a sample's metadata from chat
# records (objects with a timestamp, an optional request-arrival timestamp and a response whose
# first choice's meta info may hold an end-to-end latency). The attached record holds the segment
# start (end timestamp minus latency, none without latency), the end timestamp, the turn number,
# optionally the request-arrival timestamp and the previous turn's end timestamp.
"""Pin for the dashboard lifecycle call site in
``compute_samples_from_openai_records`` (design doc §18.3): sample assembly
must keep chat-record timing on the Sample. If a session refactor moves the
assembly and drops the one-line call, this fails loudly."""

from types import SimpleNamespace

from miles.utils.lifecycle import TrajectoryLifecycle, attach_lifecycle_metadata
from miles.utils.types import Sample


def _record(ts, latency=None, arrived=None):
    # [FUNCTION DOC] HELPER: Builds a fake chat record with the given timestamp, a request
    # timestamp (the arrival time, default none) and a response with one choice whose meta info
    # holds the end-to-end latency only when a latency is given.
    meta_info = {} if latency is None else {"e2e_latency": latency}
    return SimpleNamespace(timestamp=ts, request_timestamp=arrived, response={"choices": [{"meta_info": meta_info}]})


def test_segment_shape_with_and_without_latency():
    # [FUNCTION DOC] Attaches lifecycle data for a first-turn record (timestamp 100, latency 8.5)
    # with no previous record and asserts the metadata holds start 91.5, end 100 and turn 1. Then
    # attaches a second-turn record (timestamp 130, no latency) with the first as previous and
    # asserts start none, end 130, turn 2 and previous end 100.
    first, second = Sample(index=1), Sample(index=1)
    r1, r2 = _record(100.0, latency=8.5), _record(130.0)

    attach_lifecycle_metadata(first, r1, None, turn=1)
    assert first.metadata["lifecycle"] == dict(t0=91.5, t1=100.0, turn=1)

    attach_lifecycle_metadata(second, r2, r1, turn=2)
    assert second.metadata["lifecycle"] == dict(t0=None, t1=130.0, turn=2, prev_t1=100.0)


def test_request_timestamp_bounds_the_agent_gap():
    # [FUNCTION DOC] Attaches a turn-2 record (end 130, latency 10, request arrival 104) with a
    # previous record ending at 100, and asserts the metadata holds start 120, end 130, turn 2,
    # request timestamp 104 and previous end 100 (the request timestamp separates the agent gap
    # from server queueing).
    sample = Sample(index=1)
    prev = _record(100.0)
    record = _record(130.0, latency=10.0, arrived=104.0)
    attach_lifecycle_metadata(sample, record, prev, turn=2)
    # req_ts closes the agent gap at the server edge; [104, 120] is queueing
    assert sample.metadata["lifecycle"] == dict(t0=120.0, t1=130.0, turn=2, req_ts=104.0, prev_t1=100.0)


def test_sink_defaults_to_none():
    # [FUNCTION DOC] Asserts a newly created trajectory lifecycle object has no sink.
    # without the dashboard installed every core probe is a guarded no-op
    assert TrajectoryLifecycle().sink is None
