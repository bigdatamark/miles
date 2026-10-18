# [MODULE DOC] Async tests that the generate-and-reward routines record exactly one lifecycle
# attempt per sample for the dashboard: an attempt start, an optional generation start, and
# exactly one attempt end, including on abort, on generation failure, and when a sibling in a group
# fails and cancels the others. The process-wide trajectory lifecycle gets a recording sink
# installed by a fixture (removed afterwards); the state is a bare namespace with a semaphore of
# two generation slots.
import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.types import Sample


class RecordingSink:
    # [CLASS DOC] HELPER: Lifecycle sink that appends a pair of event name and sample index to an
    # ordered list for each attempt start, generation start and attempt end notification.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts with an empty event list.
        self.events = []

    def attempt_start(self, sample):
        # [FUNCTION DOC] HELPER: Records an attempt start event with the sample's index.
        self.events.append(("attempt_start", sample.index))

    def gen_start(self, sample):
        # [FUNCTION DOC] HELPER: Records a generation start event with the sample's index.
        self.events.append(("gen_start", sample.index))

    def attempt_end(self, sample):
        # [FUNCTION DOC] HELPER: Records an attempt end event with the sample's index.
        self.events.append(("attempt_end", sample.index))


@pytest.fixture
def lifecycle_sink():
    # [FUNCTION DOC] HELPER: Fixture that installs a recording sink on the process-wide
    # trajectory lifecycle singleton, yields the sink, and removes the sink (sets it to none)
    # after the test, including after failure.
    sink = RecordingSink()
    TrajectoryLifecycle().sink = sink
    yield sink
    TrajectoryLifecycle().sink = None


def make_state(generate_function):
    # [FUNCTION DOC] HELPER: Builds a fake rollout state: arguments with partial rollout and
    # off-policy masking off, group reward model on and a round-robin router policy; a semaphore
    # allowing two concurrent generations; not aborted; and the given generate function.
    args = SimpleNamespace(
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=True,
        sglang_router_policy="round_robin",
    )
    return SimpleNamespace(
        args=args,
        generate_fn_semaphore=asyncio.Semaphore(2),
        aborted=False,
        generate_function=generate_function,
    )


async def test_attempt_ends_once_after_success(lifecycle_sink):
    # [FUNCTION DOC] Runs generate-and-reward for one sample with a generate function that
    # succeeds, and asserts the returned object is the input sample and the recorded events are
    # attempt start, generation start, attempt end for index 1.
    async def successful_generate(input):
        # [FUNCTION DOC] HELPER: Returns a generate output whose samples is the input's sample.
        return GenerateFnOutput(samples=input.sample)

    sample = Sample(index=1)
    result = await common.generate_and_rm(make_state(successful_generate), sample, sampling_params={})

    assert result is sample
    assert lifecycle_sink.events == [
        ("attempt_start", 1),
        ("gen_start", 1),
        ("attempt_end", 1),
    ]


async def test_attempt_ends_once_after_abort(lifecycle_sink):
    # [FUNCTION DOC] Marks the state as aborted and runs generate-and-reward for one sample.
    # Asserts the sample is returned with status aborted and the events are attempt start then
    # attempt end only (no generation start; the generate function must not be called).
    async def unexpected_generate(_input):
        # [FUNCTION DOC] HELPER: Fails the test immediately; aborted attempts must not generate.
        pytest.fail("aborted attempts must not generate")

    sample = Sample(index=1)
    state = make_state(unexpected_generate)
    state.aborted = True
    result = await common.generate_and_rm(state, sample, sampling_params={})

    assert result is sample
    assert sample.status == Sample.Status.ABORTED
    assert lifecycle_sink.events == [
        ("attempt_start", 1),
        ("attempt_end", 1),
    ]


async def test_attempt_ends_when_generate_raises(lifecycle_sink):
    # [FUNCTION DOC] Runs generate-and-reward with a generate function that raises a runtime
    # error, asserts the error propagates, and that events are still attempt start, generation
    # start, attempt end (the attempt is ended exactly once on failure).
    async def failing_generate(_input):
        # [FUNCTION DOC] HELPER: Always raises a runtime error saying generate failed.
        raise RuntimeError("generate failed")

    sample = Sample(index=1)
    with pytest.raises(RuntimeError, match="generate failed"):
        await common.generate_and_rm(make_state(failing_generate), sample, sampling_params={})

    assert lifecycle_sink.events == [
        ("attempt_start", 1),
        ("gen_start", 1),
        ("attempt_end", 1),
    ]


async def test_cancelled_group_sibling_ends_attempt(lifecycle_sink):
    # [FUNCTION DOC] Runs the group variant with two samples (indices 1 and 2). The generate
    # function waits until both have started; sample 1 then raises a runtime error and sample 2
    # waits forever until cancelled, recording its index when it sees the cancellation. Asserts the
    # group call raises the runtime error, only sample 2 was cancelled, and each sample has
    # exactly the events attempt start, generation start, attempt end in order.
    all_started = asyncio.Event()
    never_finishes = asyncio.Event()
    started = 0
    cancelled = []

    async def generate(input):
        # [FUNCTION DOC] HELPER: Counts started calls and signals when two have started; waits for
        # that signal; sample index 1 then raises a runtime error saying sample failed, while the
        # other waits on an event that is never set. On cancellation records the sample index and
        # re-raises the cancellation.
        nonlocal started
        started += 1
        if started == 2:
            all_started.set()
        try:
            await all_started.wait()
            if input.sample.index == 1:
                raise RuntimeError("sample failed")
            await never_finishes.wait()
        except asyncio.CancelledError:
            cancelled.append(input.sample.index)
            raise

    samples = [Sample(index=1), Sample(index=2)]
    with pytest.raises(RuntimeError, match="sample failed"):
        await common.generate_and_rm_group(make_state(generate), samples, sampling_params={})

    assert cancelled == [2]
    for sample in samples:
        assert [kind for kind, index in lifecycle_sink.events if index == sample.index] == [
            "attempt_start",
            "gen_start",
            "attempt_end",
        ]
