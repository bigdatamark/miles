# [MODULE DOC] CPU tests for the fully asynchronous rollout function and its data buffers.
# Covers: the producer/consumer rollout function driven with a scripted fake data source and a
# patched group generator (batch drain sorted by sample index, metrics, shared-engine vs dedicated
# fleet eval, aborted / missing-reward / stale / filtered group handling and recycling, error and
# cancellation propagation, in-flight bounds, submission granularity, disposal, buffer selection);
# the default single-policy buffer (capacity back-pressure, staleness filtering and metrics, reward
# metrics); the multi-policy buffer with one queue per trained model (splitting groups by model
# id, per-model metric windows, per-model custom buffer classes and their path parsing); and the
# argument registration hook. Everything runs on one asyncio loop with no GPU or network.
# Registered with the first CPU CI stage, 60 second estimate.
from tests.ci.ci_register import register_cpu_ci
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import argparse
import asyncio
from argparse import Namespace
from collections import deque
from collections.abc import Callable
from dataclasses import replace

import pytest

import miles.rollout.fully_async_data_buffer as data_buffer
import miles.rollout.fully_async_rollout as fully_async
import miles.rollout.inference_rollout.inference_rollout_common as rollout_common
from miles.rollout.base_types import BaseRolloutFn, RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnTrainInput
from miles.rollout.filter_hub.base_types import FilterOutput
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

N_SAMPLES_PER_PROMPT = 2


class FakeGenerateState:
    # [CLASS DOC] HELPER: Minimal stand-in for the generation state used by the rollout function:
    # holds the args, an empty sampling parameter mapping and an aborted flag set to false.
    def __init__(self, args):
        # [FUNCTION DOC] HELPER: Stores the args and initializes the sampling parameters (empty
        # mapping) and the aborted flag (false).
        self.args = args
        self.sampling_params = {}
        self.aborted = False


class FakeDataSource:
    # [CLASS DOC] HELPER: Fake prompt data source. Keeps a queue of scripted groups, a counter for
    # manufactured group indices (starting after 1000), a list of recycled groups and a count of
    # get calls.
    """Serves scripted groups first, then manufactures completed groups forever."""

    def __init__(self, scripted=None):
        # [FUNCTION DOC] HELPER: Loads the optional scripted groups into a queue and zeroes the
        # counters (next manufactured index 1000, no recycled groups, no get calls).
        self.scripted = deque(scripted or [])
        self.next_group_index = 1000
        self.recycled = []
        self.num_get_calls = 0

    def get_samples(self, num_samples):
        # [FUNCTION DOC] HELPER: Requires exactly one group to be requested (asserts). Counts the
        # call, then returns a list of one group: the next scripted group if any remain, otherwise a
        # freshly manufactured completed group with an incrementing group index.
        assert num_samples == 1
        self.num_get_calls += 1
        if self.scripted:
            return [self.scripted.popleft()]
        self.next_group_index += 1
        return [make_group(self.next_group_index)]

    def add_samples(self, groups):
        # [FUNCTION DOC] HELPER: Records recycled groups by appending them to the recycled list.
        self.recycled.extend(groups)


def make_group(
    group_index: int,
    status: Sample.Status = Sample.Status.COMPLETED,
    weight_versions: list[str] | None = None,
    reward: float = 1,
) -> list[Sample]:
    # [FUNCTION DOC] HELPER: Builds one prompt group of samples-per-prompt (2) samples sharing the
    # given group index. Sample i has index group_index*10+i, prompt text naming the group, response
    # "ok" of length 1, label "ok", the given reward and status. Each given weight version becomes
    # one per-call record with a single span over tokens 0 to 1, and every sample gets its own copy
    # of that list. Returns the list.
    versions = [
        WeightVersionsPerCall(spans=[WeightVersionSpan(version=version, abs_start=0, abs_end=1)])
        for version in weight_versions or []
    ]
    return [
        Sample(
            group_index=group_index,
            index=group_index * 10 + i,
            prompt=f"prompt {group_index}",
            response="ok",
            response_length=1,
            label="ok",
            reward=reward,
            status=status,
            weight_versions=list(versions),
        )
        for i in range(N_SAMPLES_PER_PROMPT)
    ]


def make_args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds the args namespace for the rollout function. Defaults: global
    # dataset on, rollout batch size 2, 2 samples per prompt, no staleness limit, no concurrency
    # cap, buffer capacity factor 1000, unused-samples handler "drop", no custom buffer paths, no
    # megatron config, no submission granularity, no dynamic or sample filter, no reward key,
    # router at 127.0.0.1:30000 with a 14400 s request timeout, and zero eval GPUs. Overrides win.
    defaults = dict(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        max_weight_staleness=None,
        async_max_concurrent_samples=None,
        async_data_buffer_capacity_factor=1000.0,
        async_unused_samples_handler="drop",
        custom_async_data_buffer_path=None,
        custom_async_data_buffer_path_per_model=None,
        megatron_config=None,
        rollout_submission_granularity=None,
        dynamic_sampling_filter_path=None,
        reward_key=None,
        rollout_sample_filter_path=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        sglang_router_request_timeout_secs=14400,
        eval_num_gpus=0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def make_fn(monkeypatch, args, data_source, generate=None):
    # [FUNCTION DOC] HELPER: Patches the rollout module so the generation state class is the fake
    # one and the group generator is the given coroutine function (default: yield to the loop once
    # and return the group unchanged). Returns a fully async rollout function built from the args
    # and data source.
    async def default_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        # [FUNCTION DOC] HELPER: Yields to the event loop once and returns the group unchanged.
        await asyncio.sleep(0)
        return group

    monkeypatch.setattr(fully_async, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(fully_async, "generate_and_rm_group", generate or default_generate)
    return fully_async.FullyAsyncRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))


async def test_drain_collects_batch_sorted_with_metrics(monkeypatch):
    # [FUNCTION DOC] With batch size 3, one train call returns 3 groups sorted by first-sample
    # index, each with 2 samples, and zero aborted-group and stale-group filter metrics. A second
    # call on the same function instance returns 3 groups again (the background worker persists).
    args = make_args(rollout_batch_size=3)
    fn = make_fn(monkeypatch, args, FakeDataSource())

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 3
    indices = [group[0].index for group in output.samples]
    assert indices == sorted(indices)
    assert all(len(group) == N_SAMPLES_PER_PROMPT for group in output.samples)
    assert output.metrics["rollout/fully_async/aborted_groups_filtered"] == 0
    assert output.metrics["rollout/fully_async/stale_groups_filtered"] == 0

    # The worker persists across calls; a second drain works on the same instance.
    output2 = await fn(RolloutFnTrainInput(rollout_id=1))
    assert len(output2.samples) == 3


async def test_eval_without_fleet_pauses_producer(monkeypatch):
    # [FUNCTION DOC] Shared-engine eval (no eval GPUs): a train call starts the producer, with
    # generation blocked on an event. After 50 ms the number of data source requests is noted, then
    # an eval call is started with a fake dataset runner that signals when started and waits for a
    # release. Once eval has started and generation is released, 50 ms later no new prompts may have
    # been requested. Releasing eval returns the fake results as the eval output data, and the train
    # call then completes with samples. The fake runner also asserts it got the function's own
    # state.
    """Shared-engine eval: producer submissions pause during eval and resume after."""
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        # [FUNCTION DOC] HELPER: Waits for the release event, then returns the group unchanged.
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(
        monkeypatch, make_args(rollout_batch_size=2, eval_num_gpus=0), data_source, generate=blocking_generate
    )

    eval_started = asyncio.Event()
    eval_release = asyncio.Event()
    eval_results = {"fake_ds": {"rewards": [1.0], "truncated": [False], "samples": []}}

    async def fake_run_eval_datasets(state, cache):
        # [FUNCTION DOC] HELPER: Asserts the state is the rollout function's own state, signals that
        # eval started, waits for the eval release event, and returns the prepared fake results.
        assert state is fn.state  # shared-engine eval uses the train state
        eval_started.set()
        await eval_release.wait()
        return eval_results

    monkeypatch.setattr(fully_async, "run_eval_datasets", fake_run_eval_datasets)

    # Start the producer via a train call, then run eval concurrently.
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    submitted_before_eval = data_source.num_get_calls

    eval_task = asyncio.create_task(fn(RolloutFnEvalInput(rollout_id=0)))
    await eval_started.wait()
    release.set()  # in-flight groups finish and buffer, but no NEW submissions
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == submitted_before_eval

    eval_release.set()
    output = await eval_task
    assert output.data == eval_results

    # Producer resumes and the train drain completes.
    assert (await drain).samples


async def test_eval_runs_on_dedicated_fleet(monkeypatch):
    # [FUNCTION DOC] With eval GPUs configured, an eval call carrying a caller-built generate state
    # and weight version "0" must run the dataset evaluation with exactly that state, return the
    # fake results as output data, and must not start the producer (no worker) or request any
    # training prompts.
    """RolloutManager (not the fn) decides fleet-vs-shared and builds the fleet's
    GenerateState; it hands it in via RolloutFnEvalInput.generate_state. The fn must
    use that state as-is (not self.state) and must not touch the producer/data_source.
    Building/caching the fleet state itself is RolloutExecutorEvalFleet's job, covered in
    tests/fast/rollout/test_checkpoint_eval.py.
    """
    args = make_args(eval_num_gpus=1, eval_num_gpus_per_engine=1)
    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, args, data_source)

    fleet_state = FakeGenerateState(args)
    eval_results = {"fake_ds": {"rewards": [1.0], "truncated": [False], "samples": []}}
    seen_states = []

    async def fake_run_eval_datasets(state, cache):
        # [FUNCTION DOC] HELPER: Records the state it was given and returns the fake results.
        seen_states.append(state)
        return eval_results

    monkeypatch.setattr(fully_async, "run_eval_datasets", fake_run_eval_datasets)

    output = await fn(RolloutFnEvalInput(rollout_id=0, generate_state=fleet_state, weight_version="0"))

    assert output.data == eval_results
    assert seen_states == [fleet_state]  # used the fleet's state, not fn.state
    # Eval must not start the producer or consume training prompts.
    assert fn._worker is None
    assert data_source.num_get_calls == 0


async def test_aborted_group_recycled(monkeypatch):
    # [FUNCTION DOC] An aborted group (rewards cleared) is scripted first, with the retry handler.
    # Asserts it was recycled to the data source, was reset for retry (empty response and no weight
    # versions), is absent from the output, one aborted group is counted in the metrics, and no
    # missing-reward drop metric appears.
    aborted = make_group(1, status=Sample.Status.ABORTED)
    for sample in aborted:
        sample.reward = None
    data_source = FakeDataSource(scripted=[aborted])
    args = make_args(rollout_batch_size=1, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [aborted]
    # reset_for_retry cleared generated outputs so the prompt can be re-sampled
    assert all(sample.response == "" and sample.weight_versions == [] for sample in aborted)
    assert output.samples[0][0].group_index != 1
    assert output.metrics["rollout/fully_async/aborted_groups_filtered"] == 1
    assert "rollout/dynamic_filter/drop_group_has_missing_reward" not in output.metrics


async def test_missing_reward_group_dropped_without_recycling(monkeypatch):
    # [FUNCTION DOC] A group with one missing reward is scripted first, with the retry handler. It is
    # dropped, not recycled (nothing recycled), the output comes from another group, and the
    # missing-reward drop metric is 1.
    missing_reward = make_group(1)
    missing_reward[0].reward = None
    data_source = FakeDataSource(scripted=[missing_reward])
    args = make_args(rollout_batch_size=1, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == []
    assert output.samples[0][0].group_index != 1
    assert output.metrics["rollout/dynamic_filter/drop_group_has_missing_reward"] == 1


async def test_stale_group_recycled(monkeypatch):
    # [FUNCTION DOC] With a maximum staleness of 2, a scripted group carrying version "5" is stale
    # at current version 10, and the retry handler recycles it. The data source's get is wrapped so
    # that groups lacking versions receive version "10" (making replacement groups fresh). Asserts
    # the stale group was recycled, one stale group was filtered, and the max staleness metric is 0.
    stale = make_group(1, weight_versions=["5"])
    data_source = FakeDataSource(scripted=[stale])
    data_source_fresh_versions = ["10"]

    original_make = data_source.get_samples

    def get_samples_with_fresh_versions(num_samples):
        # [FUNCTION DOC] HELPER: Calls the original get and, for every sample without weight
        # versions, attaches one span per fresh version (token range 0 to 1). Mutates the samples
        # and returns the groups.
        groups = original_make(num_samples)
        for group in groups:
            for sample in group:
                if not sample.weight_versions:
                    sample.weight_versions = [
                        WeightVersionsPerCall(spans=[WeightVersionSpan(version=version, abs_start=0, abs_end=1)])
                        for version in data_source_fresh_versions
                    ]
        return groups

    data_source.get_samples = get_samples_with_fresh_versions

    args = make_args(rollout_batch_size=1, max_weight_staleness=2, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0, weight_version=10))

    assert data_source.recycled == [stale]
    assert output.metrics["rollout/fully_async/stale_groups_filtered"] == 1
    assert output.metrics["rollout/fully_async/max_staleness"] == 0


async def test_stale_group_dropped_by_default(monkeypatch):
    # [FUNCTION DOC] With staleness limit 2, a group at version "5" is stale at current version 10;
    # under the default handler it is dropped (nothing recycled) and one stale group is counted.
    stale = make_group(1, weight_versions=["5"])
    data_source = FakeDataSource(scripted=[stale])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=2), data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0, weight_version=10))

    assert data_source.recycled == []
    assert output.metrics["rollout/fully_async/stale_groups_filtered"] == 1


async def test_worker_error_propagates(monkeypatch):
    # [FUNCTION DOC] When group generation raises a runtime error, the train call raises that same
    # error rather than hanging.
    async def failing_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        # [FUNCTION DOC] HELPER: Always raises a runtime error saying generation exploded.
        raise RuntimeError("generation exploded")

    fn = make_fn(monkeypatch, make_args(), FakeDataSource(), generate=failing_generate)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))


@pytest.mark.parametrize("handler", ["drop", "retry"])
@pytest.mark.parametrize("granularity", ["sample", "group"])
async def test_sample_cancellation_aborts_group_without_stopping_worker(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    handler: str,
    granularity: str,
) -> None:
    # [FUNCTION DOC] Parametrized over unused-sample handler (drop, retry) and submission
    # granularity (sample, group). Uses the real group fan-out/cleanup code with a fake per-sample
    # generator: for the scripted group 1, the first sample waits until its sibling has started,
    # then cancels its own task; the sibling blocks forever and records when it is cleaned up.
    # Asserts: the sibling finished (was cancelled and settled), the background worker is still
    # alive, one aborted group is counted, the output has no samples from group 1 and all are
    # completed, a log line says the rollout group was cancelled. With retry, the group is recycled
    # and reset (empty response, no versions); with drop, nothing is recycled and the group's
    # samples keep completed status. A second train call (rollout 1) still returns samples with
    # zero aborted groups, showing the producer survives.
    prompt_group = make_group(1)
    data_source = FakeDataSource(scripted=[prompt_group])
    sibling_started = asyncio.Event()
    sibling_finished = asyncio.Event()

    async def generate_sample(
        state: FakeGenerateState,
        sample: Sample,
        sampling_params: dict,
        evaluation: bool = False,
    ) -> Sample:
        # [FUNCTION DOC] HELPER: Fake per-sample generator. Samples of other groups are returned
        # immediately. For group 1: the first sample waits for the sibling's start signal, cancels
        # the current task and yields once; every group-1 sample (the sibling reaching this point
        # directly) signals started, then blocks forever on a never-set event, setting the
        # sibling-finished event when unwound (in a cleanup block).
        if sample.group_index != 1:
            return sample
        if sample is prompt_group[0]:
            await sibling_started.wait()
            asyncio.current_task().cancel()
            await asyncio.sleep(0)
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_finished.set()
        return sample

    monkeypatch.setattr(rollout_common, "generate_and_rm", generate_sample)
    monkeypatch.setattr(rollout_common, "policy_uses_routing_key", lambda args: False)
    args = make_args(
        rollout_batch_size=1,
        async_max_concurrent_samples=N_SAMPLES_PER_PROMPT,
        async_unused_samples_handler=handler,
        rollout_submission_granularity=granularity,
        group_rm=False,
    )
    # Exercise the real fan-out/cleanup and scheduler callbacks, not a mocked
    # group result: one cancelled sample also cancels and settles its sibling.
    fn = make_fn(monkeypatch, args, data_source, generate=rollout_common.generate_and_rm_group)

    output = await asyncio.wait_for(fn(RolloutFnTrainInput(rollout_id=0)), timeout=2)

    assert sibling_finished.is_set()
    assert not fn._worker.done()
    assert output.metrics["rollout/fully_async/aborted_groups_filtered"] == 1
    assert all(sample.group_index != 1 for group in output.samples for sample in group)
    assert all(sample.status == Sample.Status.COMPLETED for group in output.samples for sample in group)
    assert "Rollout group was cancelled" in caplog.text
    if handler == "retry":
        assert data_source.recycled == [prompt_group]
        assert all(sample.response == "" and sample.weight_versions == [] for sample in prompt_group)
    else:
        assert data_source.recycled == []
        assert all(sample.status == Sample.Status.COMPLETED for sample in prompt_group)

    # The producer remains usable after the affected batch, not just until the
    # first replacement arrives.
    following = await asyncio.wait_for(fn(RolloutFnTrainInput(rollout_id=1)), timeout=2)
    assert following.samples
    assert following.metrics["rollout/fully_async/aborted_groups_filtered"] == 0


async def test_worker_cancellation_still_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Starts a train call whose generation blocks, then cancels the background worker
    # task directly. The waiting train call must fail with a runtime error saying the function was
    # disposed while a step waited; the worker is marked cancelled, nothing was recycled, and the
    # output buffer's aborted-group metric is 0 (the cancellation is not counted as an abort). The
    # generation is released in a final cleanup step.
    """A cancelled worker fails the waiting step loudly instead of being filtered away as an abort."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_generate(
        state: FakeGenerateState,
        group: list[Sample],
        sampling_params: dict,
        evaluation: bool = False,
        sample_done_callback: Callable[[], None] | None = None,
    ) -> list[Sample]:
        # [FUNCTION DOC] HELPER: Signals started, waits for release, returns the group unchanged.
        started.set()
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source, generate=blocking_generate)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.wait_for(started.wait(), timeout=2)
    fn._worker.cancel()

    try:
        with pytest.raises(RuntimeError, match="disposed while a step waited"):
            await asyncio.wait_for(drain, timeout=2)
        assert fn._worker.cancelled()
        assert data_source.recycled == []
        assert fn._output.get_metrics()["rollout/fully_async/aborted_groups_filtered"] == 0
    finally:
        release.set()
        await asyncio.sleep(0)


async def test_async_max_concurrent_samples_caps_in_flight_groups(monkeypatch):
    # [FUNCTION DOC] With batch size 4 and a cap of 3 concurrent samples (2 samples per group, so one
    # group in flight), 50 ms after starting a train call with blocked generation exactly one group
    # has been requested. After release, all 4 groups are produced.
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        # [FUNCTION DOC] HELPER: Waits for the release event, then returns the group unchanged.
        await release.wait()
        return group

    data_source = FakeDataSource()
    # 3 samples // 2 per group -> 1 group in flight, below rollout_batch_size
    args = make_args(rollout_batch_size=4, async_max_concurrent_samples=3)
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == 1

    release.set()
    output = await drain
    assert len(output.samples) == 4


async def test_worker_failure_beats_queued_groups(monkeypatch):
    # [FUNCTION DOC] Installs a default buffer already holding one finished group as the output, and
    # a worker task that fails with a runtime error. A train call must raise that error instead of
    # returning the queued group.
    """A dead worker fails the step even when it left completed groups behind."""
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

    async def boom():
        # [FUNCTION DOC] HELPER: Always raises a runtime error saying generation exploded.
        raise RuntimeError("generation exploded")

    fn._output = make_buffer()[0]
    group = make_group(1)
    await fn._output.put(data_buffer.DataBufferInput(prompt_group=group, group=group))
    fn._worker = asyncio.create_task(boom())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))


async def test_nested_group_recycles_the_flat_prompt_group(monkeypatch):
    # [FUNCTION DOC] A generator first returns a nested result (each sample expanded into an aborted
    # copy plus a normal copy), later returns the group as is. With the retry handler, asserts the
    # data source got back the original flat prompt group (all elements plain samples), the
    # generator was resubmitted only flat groups and was called more than once, and the final output
    # holds one group.
    """A generate function may expand one trajectory into several samples; the retry
    must resubmit the flat prompt group the data source handed out."""
    prompt_group = make_group(1)
    data_source = FakeDataSource(scripted=[prompt_group])
    submitted = []

    async def multi_sample_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        # [FUNCTION DOC] HELPER: Asserts every element of the submitted group is a plain sample and
        # records the group. On the first call returns a nested list: for each sample a pair of an
        # aborted copy and a plain copy. On later calls returns the group unchanged.
        assert all(isinstance(sample, Sample) for sample in group), "resubmitted a nested group"
        submitted.append(group)
        if len(submitted) > 1:
            return group
        expanded = []
        for sample in group:
            aborted = replace(sample, status=Sample.Status.ABORTED)
            expanded.append([aborted, replace(sample)])
        return expanded

    args = make_args(rollout_batch_size=1, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, data_source, generate=multi_sample_generate)
    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [prompt_group]
    assert all(isinstance(sample, Sample) for sample in data_source.recycled[0])
    assert len(submitted) > 1
    assert len(output.samples) == 1


def reject_group_1(args, group, **kwargs):
    # [FUNCTION DOC] HELPER: Dynamic sampling filter loaded by dotted path from this module. Returns
    # a filter output that rejects (keep false, reason "rejected") any group whose first sample has
    # group index 1 and keeps all others (no reason).
    keep = group[0].group_index != 1
    return FilterOutput(keep=keep, reason=None if keep else "rejected")


async def test_dynamic_filter_drops_group_without_recycling(monkeypatch):
    # [FUNCTION DOC] With the reject-group-1 filter and the retry handler, scripted group 1 is
    # dropped without recycling (filter rejections bypass the unused handler); the output has one
    # other group and the dynamic filter drop metric for reason "rejected" is 1.
    rejected = make_group(1)
    data_source = FakeDataSource(scripted=[rejected])
    args = make_args(
        rollout_batch_size=1,
        dynamic_sampling_filter_path=f"{__name__}.reject_group_1",
        async_unused_samples_handler="retry",
    )
    fn = make_fn(monkeypatch, args, data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 1
    assert output.samples[0][0].group_index != 1
    # Dropped even with handler="retry": filter rejections bypass the unused handler.
    assert data_source.recycled == []
    assert output.metrics["rollout/dynamic_filter/drop_rejected"] == 1


async def test_sample_filter_marks_samples_without_shrinking_the_batch(monkeypatch):
    # [FUNCTION DOC] Installs a sample filter (set directly on the function's private attribute)
    # that flags the first sample of each group for removal. The output still has 2 groups, and in
    # the first group the flags read true then false.
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), FakeDataSource())

    def mark_first_of_each_group(args, data):
        # [FUNCTION DOC] HELPER: Sets the remove-sample flag on the first sample of every group in
        # place; returns nothing.
        for group in data:
            group[0].remove_sample = True

    fn._sample_filter = mark_first_of_each_group

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 2
    assert [sample.remove_sample for sample in output.samples[0]] == [True, False]


async def test_staleness_filter_off_before_the_first_weight_update(monkeypatch):
    # [FUNCTION DOC] With staleness limit 0 but no current weight version in the train input, a
    # group at version "5" is not filtered: nothing recycled, group 1 is returned, and the max
    # staleness metric is absent.
    """weight_version is None until the trainer pushes weights; staleness is unknown, not zero."""
    stale = make_group(1, weight_versions=["5"])
    data_source = FakeDataSource(scripted=[stale])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=0), data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == []
    assert output.samples[0][0].group_index == 1
    assert "rollout/fully_async/max_staleness" not in output.metrics


# ── DataBuffer: staleness-bounded buffering ─────────────────────────


def make_buffer(max_groups=None, max_staleness=None):
    # [FUNCTION DOC] HELPER: Builds a default single-policy buffer with batch size 1 (so capacity
    # counts groups): capacity factor is max_groups, or 1000 when none or zero; the staleness limit
    # is as given. The unused-group handler appends to a list. Returns (buffer, unused list).
    unused = []
    args = make_args(
        rollout_batch_size=1,  # capacity is factor * batch size; batch size 1 makes it count groups
        async_data_buffer_capacity_factor=max_groups or 1000.0,
        max_weight_staleness=max_staleness,
    )
    buffer = data_buffer.DefaultDataBuffer(
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=unused.append)
    )
    return buffer, unused


async def put_group(buffer, group):
    # [FUNCTION DOC] HELPER: Puts the group into the buffer as both prompt group and finished group;
    # waits if the buffer is full.
    """These tests reuse one group as both the prompt group and the finished group."""
    await buffer.put(data_buffer.DataBufferInput(prompt_group=group, group=group))


async def test_buffer_reports_unfiltered_raw_reward_across_kept_and_dropped():
    # [FUNCTION DOC] With the reject-group-1 filter, puts group 1 (reward 0, rejected) and group 2
    # (reward 1, kept). The metrics report an unfiltered raw reward of 0.5 (mean over both) and one
    # rejected drop; reading metrics resets the window so a second read lacks the unfiltered key.
    """The accepted-only raw_reward is conditioned by the filter, so this mean must still see dropped groups."""
    args = make_args(rollout_batch_size=1, dynamic_sampling_filter_path=f"{__name__}.reject_group_1")
    buffer = data_buffer.DefaultDataBuffer(
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=lambda group: None)
    )

    await put_group(buffer, make_group(1, reward=0))
    await put_group(buffer, make_group(2, reward=1))

    metrics = buffer.get_metrics()
    assert metrics["rollout/raw_reward_unfiltered"] == 0.5
    assert metrics["rollout/dynamic_filter/drop_rejected"] == 1
    assert "rollout/raw_reward_unfiltered" not in buffer.get_metrics()


async def test_buffer_blocks_producer_when_full():
    # [FUNCTION DOC] With capacity 2, two puts succeed and a third stays blocked after 10 ms; the
    # queue size metric reads 2. Getting returns groups 1, 2, 3 in first-in-first-out order, and the
    # blocked put completes once space frees.
    buffer, _ = make_buffer(max_groups=2)
    await put_group(buffer, make_group(1))
    await put_group(buffer, make_group(2))

    blocked = asyncio.create_task(put_group(buffer, make_group(3)))
    await asyncio.sleep(0.01)
    assert not blocked.done()
    assert buffer.get_metrics()["rollout/fully_async/queue_size"] == 2

    assert (await buffer.get()).group[0].group_index == 1
    await blocked
    assert (await buffer.get()).group[0].group_index == 2
    assert (await buffer.get()).group[0].group_index == 3


async def test_buffer_get_ignores_unknown_context_keys():
    # [FUNCTION DOC] Getting with extra unknown keyword context (current version plus a future key)
    # still returns the buffered group 1 without error.
    """get(**context) lets the driver add keys without breaking existing buffers."""
    buffer, _ = make_buffer()
    await put_group(buffer, make_group(1))

    assert (await buffer.get(current_version=1, some_future_key=2)).group[0].group_index == 1


async def test_buffer_get_skips_groups_stale_at_consumption_time():
    # [FUNCTION DOC] With staleness limit 2, buffers group 1 (version "5") and group 2 (version
    # "8"). Getting at current version 10 skips group 1 (stale, handed to the unused handler) and
    # returns group 2; one stale group is counted in the metrics.
    """Both groups were fresh when buffered; only the version passed to get() decides."""
    buffer, unused = make_buffer(max_staleness=2)
    stale = make_group(1, weight_versions=["5"])
    await put_group(buffer, stale)
    await put_group(buffer, make_group(2, weight_versions=["8"]))

    assert (await buffer.get(current_version=10)).group[0].group_index == 2
    assert unused == [stale]
    assert buffer.get_metrics()["rollout/fully_async/stale_groups_filtered"] == 1


async def test_buffer_staleness_metrics():
    # [FUNCTION DOC] Buffers groups at versions 4, 6 and 8. Before any get the buffered-average
    # staleness metric is absent (engine version not yet seen). After getting at current version 10
    # (consuming group 1), the average staleness is 6.0 (10 minus 4), the buffered average staleness
    # is 3.0 (groups 2 and 3: (4+2)/2), and the buffered max staleness is 4.
    buffer, _ = make_buffer(max_groups=8)
    await put_group(buffer, make_group(1, weight_versions=["4"]))
    assert "rollout/fully_async/buffer_avg_staleness" not in buffer.get_metrics()  # engine version never seen

    await put_group(buffer, make_group(2, weight_versions=["6"]))
    await put_group(buffer, make_group(3, weight_versions=["8"]))
    await buffer.get(current_version=10)  # pops group 1 and tracks the engine version clock
    metrics = buffer.get_metrics()
    assert metrics["rollout/fully_async/avg_staleness"] == 6.0  # consumed group 1: 10 - 4
    assert metrics["rollout/fully_async/buffer_avg_staleness"] == 3.0  # buffered groups 2, 3: (4 + 2) / 2
    assert metrics["rollout/fully_async/buffer_max_staleness"] == 4


async def test_buffer_reports_selected_policy_provenance():
    # [FUNCTION DOC] With staleness limit 2, group 1 has versions 2 and 4 (stale) and group 2 has 8
    # and 10. A get at version 10 returns group 2. The metrics describe only the selected group:
    # average staleness 2 (10 minus its oldest, 8), max staleness 2, post-generation staleness 0,
    # generation version span 2, token-weighted staleness 1, and full (1) weight version sample
    # coverage.
    buffer, _ = make_buffer(max_groups=8, max_staleness=2)
    await put_group(buffer, make_group(1, weight_versions=["2", "4"]))
    await put_group(buffer, make_group(2, weight_versions=["8", "10"]))

    assert (await buffer.get(current_version=10)).group[0].group_index == 2
    metrics = buffer.get_metrics()

    assert metrics["rollout/fully_async/avg_staleness"] == 2
    assert metrics["rollout/fully_async/max_staleness"] == 2
    assert metrics["rollout/fully_async/avg_post_generation_staleness"] == 0
    assert metrics["rollout/fully_async/avg_generation_version_span"] == 2
    assert metrics["rollout/fully_async/token_weighted_staleness"] == 1
    assert metrics["rollout/fully_async/weight_version_sample_coverage"] == 1


async def test_buffer_reports_generation_span_without_current_version():
    # [FUNCTION DOC] A group with versions 2 and 4 is fetched with no current version. The metrics
    # include the generation version span (2) but not the post-generation staleness or the
    # token-weighted staleness, since those need a current version.
    buffer, _ = make_buffer(max_groups=8)
    await put_group(buffer, make_group(1, weight_versions=["2", "4"]))

    await buffer.get(current_version=None)
    metrics = buffer.get_metrics()

    assert metrics["rollout/fully_async/avg_generation_version_span"] == 2
    assert "rollout/fully_async/avg_post_generation_staleness" not in metrics
    assert "rollout/fully_async/token_weighted_staleness" not in metrics


async def test_buffer_reports_missing_weight_version_coverage_without_inventing_lag():
    # [FUNCTION DOC] A group where only one of its two samples has a version ("7") is fetched at
    # current version 10: sample coverage is 0.5 and token-weighted staleness is 3 (lag of the
    # versioned sample only, none invented for the other). A second metrics read returns neither
    # key, because the window was reset.
    buffer, _ = make_buffer(max_groups=8)
    partial_group = make_group(1, weight_versions=["7"])
    partial_group[1].weight_versions = []
    await put_group(buffer, partial_group)

    await buffer.get(current_version=10)
    metrics = buffer.get_metrics()

    assert metrics["rollout/fully_async/weight_version_sample_coverage"] == 0.5
    assert metrics["rollout/fully_async/token_weighted_staleness"] == 3

    metrics = buffer.get_metrics()
    assert "rollout/fully_async/weight_version_sample_coverage" not in metrics
    assert "rollout/fully_async/token_weighted_staleness" not in metrics


def make_multi_policy_group(group_index: int, *trainer_model_ids: str) -> list[Sample]:
    # [FUNCTION DOC] HELPER: Builds a standard group for the index and stamps its samples, in order,
    # with the given trainer model ids. The number of ids must equal the number of samples (strict
    # pairing raises a value error otherwise). Returns the group.
    """One prompt group whose samples train different policy models, as a multi policy generate fn returns."""
    group = make_group(group_index)
    for sample, trainer_model_id in zip(group, trainer_model_ids, strict=True):
        sample.trainer_model_id = trainer_model_id
    return group


def make_multi_buffer(*model_ids: str, max_staleness=None, paths_per_model=None):
    # [FUNCTION DOC] HELPER: Builds a default multi-policy buffer for the given model ids (encoded
    # into the megatron config), batch size 1, capacity factor 1000, the given staleness limit and
    # per-model custom buffer path entries. The unused-group handler appends to a list. Returns
    # (buffer, unused list).
    unused = []
    args = make_args(
        rollout_batch_size=1,
        async_data_buffer_capacity_factor=1000.0,
        max_weight_staleness=max_staleness,
        megatron_config=encode_megatron_config(*model_ids),
        custom_async_data_buffer_path_per_model=paths_per_model,
    )
    buffer = data_buffer.DefaultMultiDataBuffer(
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=unused.append)
    )
    return buffer, unused


class TestPerPolicyQueues:
    # [CLASS DOC] Tests of the multi-policy buffer's per-policy queues for policies "solver" and
    # "verifier".
    async def test_a_group_of_two_policies_lands_in_a_queue_of_each(self):
        # [FUNCTION DOC] One group with a solver sample and a verifier sample yields a queue size of
        # 1 in each policy's metrics.
        """One generate call feeds both policies, and a shared queue would hand them each other's samples."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        await put_group(buffer, make_multi_policy_group(1, "solver", "verifier"))

        assert buffer.get_metrics("solver")["rollout/fully_async/queue_size"] == 1
        assert buffer.get_metrics("verifier")["rollout/fully_async/queue_size"] == 1

    async def test_a_policy_only_ever_drains_its_own_samples(self):
        # [FUNCTION DOC] After putting the mixed group, getting for "verifier" returns an entry whose
        # flattened samples are exactly one, tagged "verifier".
        """Training a policy on another policy's responses is the failure this queue split exists to stop."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        await put_group(buffer, make_multi_policy_group(1, "solver", "verifier"))

        entry = await buffer.get(trainer_model_id="verifier")

        assert [sample.trainer_model_id for sample in data_buffer.iter_samples(entry.group)] == ["verifier"]

    async def test_a_policy_waits_for_its_own_queue_instead_of_taking_from_another(self):
        # [FUNCTION DOC] With only solver samples buffered, a get for "verifier" stays pending after
        # 10 ms (it does not take solver's group); the pending get is then cancelled.
        """A policy that consumed a queue it does not own would starve the policy that does."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        await put_group(buffer, make_multi_policy_group(1, "solver", "solver"))

        waiting = asyncio.create_task(buffer.get(trainer_model_id="verifier"))
        await asyncio.sleep(0.01)

        assert not waiting.done()
        waiting.cancel()

    async def test_an_untagged_sample_is_refused_at_the_split(self):
        # [FUNCTION DOC] Putting a plain group whose samples have no trainer model id fails an
        # assertion saying the generate function must stamp every sample.
        """Every sample of a multi policy run is stamped by the generate function, so an unstamped one is a bug."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        with pytest.raises(AssertionError, match="must stamp every sample"):
            await put_group(buffer, make_group(1))

    async def test_a_sample_of_an_unknown_policy_is_refused(self):
        # [FUNCTION DOC] Putting a group with a sample stamped "reviewer" (not a configured policy)
        # fails an assertion saying the run trains no such policy.
        """Its groups would queue up in a buffer no trainer ever drains, and the run would simply stall."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        with pytest.raises(AssertionError, match="trains no policy of this run"):
            await put_group(buffer, make_multi_policy_group(1, "solver", "reviewer"))

    async def test_the_prompt_group_of_a_split_group_stays_whole(self):
        # [FUNCTION DOC] With staleness limit 0, a group split across solver and verifier (versions
        # "1") is requested by solver at current version 9 and judged stale. After 10 ms the unused
        # handler has received the whole original group (not the per-policy part); the pending get
        # is cancelled.
        """Recycling a rejected group resubmits prompts, which are not owned by either policy."""
        buffer, unused = make_multi_buffer("solver", "verifier", max_staleness=0)
        group = make_group(1, weight_versions=["1"])
        group[0].trainer_model_id, group[1].trainer_model_id = "solver", "verifier"

        await put_group(buffer, group)
        drained = asyncio.create_task(buffer.get(current_version=9, trainer_model_id="solver"))
        await asyncio.sleep(0.01)

        assert unused == [group]
        drained.cancel()

    async def test_getting_for_a_policy_this_run_does_not_train_is_refused(self):
        # [FUNCTION DOC] Getting for the unconfigured policy "reviewer" fails an assertion saying
        # the run trains no such policy (instead of waiting forever).
        """A typo in the trainer's model id would wait forever on a queue that is never fed."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        with pytest.raises(AssertionError, match="trains no policy of this run"):
            await buffer.get(trainer_model_id="reviewer")

    async def test_every_policy_of_the_config_gets_a_queue_of_its_own(self):
        # [FUNCTION DOC] A fresh two-policy buffer reports a queue size of 0 for each policy.
        """The queues are built from --megatron-config, so a policy missing one has nowhere to put its groups."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        assert buffer.get_metrics("solver")["rollout/fully_async/queue_size"] == 0
        assert buffer.get_metrics("verifier")["rollout/fully_async/queue_size"] == 0

    async def test_a_policy_reads_and_resets_only_its_own_metric_window(self):
        # [FUNCTION DOC] After a mixed group is put, reading solver's metrics returns keys carrying
        # no per-policy name prefix, and verifier's queue size metric afterwards is still 1 (not
        # reset by solver's read).
        """Draining one policy used to read and clear every policy's counters, moving them onto the wrong curve."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        await put_group(buffer, make_multi_policy_group(1, "solver", "verifier"))

        solver_metrics = buffer.get_metrics("solver")

        assert set(solver_metrics) == {key for key in solver_metrics if not key.startswith(("solver/", "verifier/"))}
        assert buffer.get_metrics("verifier")["rollout/fully_async/queue_size"] == 1

    async def test_an_inner_buffer_is_told_which_policy_asks_for_a_group(self):
        # [FUNCTION DOC] Replaces solver's inner buffer (via its private mapping) with a recording
        # fake. A get with current version 4 for "solver" returns the fake's entry, and the fake saw
        # exactly one context with both the current version and the trainer model id forwarded.
        """A custom per policy buffer cannot filter or account by policy if the composite eats that context."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        seen: list[dict] = []

        class _RecordingInner:
            # [CLASS DOC] HELPER: Fake inner buffer that only supports get, recording the context.
            async def get(self, **context):
                # [FUNCTION DOC] HELPER: Appends the keyword context mapping to the shared list and
                # returns the string "entry".
                seen.append(context)
                return "entry"

        buffer._inners["solver"] = _RecordingInner()

        assert await buffer.get(current_version=4, trainer_model_id="solver") == "entry"
        assert seen == [{"current_version": 4, "trainer_model_id": "solver"}]


def make_tagged_sample(index: int, trainer_model_id: str | None) -> Sample:
    # [FUNCTION DOC] HELPER: Takes the first sample of a standard group with the given index and
    # sets its trainer model id (possibly none, meaning untagged). Returns it.
    sample = make_group(index)[0]
    sample.trainer_model_id = trainer_model_id
    return sample


def split(group: data_buffer.Group, *, prompt_group=None) -> dict:
    # [FUNCTION DOC] HELPER: Wraps the group and prompt group (empty list when none given) into a
    # buffer input and calls the buffer module's private split-by-trainer-model-id function.
    # Returns its mapping from model id to buffer input.
    return data_buffer._split_by_trainer_model_id(
        data_buffer.DataBufferInput(prompt_group=prompt_group if prompt_group is not None else [], group=group)
    )


class TestSplitByTrainerModelId:
    # [CLASS DOC] Tests of the buffer module's splitting of a finished group into one buffer input
    # per trainer model id.
    def test_a_group_of_one_policy_lands_whole_in_that_policy_alone(self):
        # [FUNCTION DOC] Two solver samples split into a mapping with only the key "solver" whose
        # group equals the original two samples in order.
        """The common group is a group of one policy, and every sample of it must reach that policy."""
        first, second = make_tagged_sample(1, "solver"), make_tagged_sample(2, "solver")

        ans = split([first, second])

        assert list(ans) == ["solver"]
        assert ans["solver"].group == [first, second]

    def test_a_mixed_group_becomes_one_input_per_policy(self):
        # [FUNCTION DOC] A solver sample and a verifier sample split into keys solver then verifier
        # (first-seen order), each holding only its own sample.
        """Each policy trains on its own samples only, so the group has to be cut along the tags."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        ans = split([solver, verifier])

        assert list(ans) == ["solver", "verifier"]
        assert ans["solver"].group == [solver]
        assert ans["verifier"].group == [verifier]

    def test_the_samples_of_a_policy_keep_the_order_they_arrived_in(self):
        # [FUNCTION DOC] Solver, verifier, solver samples split so the solver part holds its two
        # samples in their original relative order.
        """Order carries the trajectory, and a reordered group trains on a reshuffled conversation."""
        first, second = make_tagged_sample(1, "solver"), make_tagged_sample(2, "solver")

        ans = split([first, make_tagged_sample(3, "verifier"), second])

        assert ans["solver"].group == [first, second]

    def test_a_sub_group_of_a_multi_sample_trajectory_is_filtered_per_policy(self):
        # [FUNCTION DOC] One nested sub-group holding a solver and a verifier sample splits into a
        # solver part with a sub-group of just the solver sample and a verifier part likewise.
        """A generate function may return several samples per trajectory, and they need not share a policy."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        ans = split([[solver, verifier]])

        assert ans["solver"].group == [[solver]]
        assert ans["verifier"].group == [[verifier]]

    def test_a_sub_group_no_sample_of_which_survives_is_dropped(self):
        # [FUNCTION DOC] Two sub-groups, one per policy: each policy's part contains only the
        # sub-group of its own sample (the other policy's sub-group does not appear as an empty
        # sub-group).
        """An empty sub-group is a trajectory with no samples, which the consumers cannot make sense of."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        ans = split([[solver], [verifier]])

        assert ans["solver"].group == [[solver]]
        assert ans["verifier"].group == [[verifier]]

    def test_every_trajectory_of_the_group_is_split_on_its_own(self):
        # [FUNCTION DOC] Sub-groups [solver, verifier] and [solver] split into solver part
        # [[first], [third]] and verifier part [[second]]: each sub-group is cut independently.
        """One finished group carries several trajectories, and each of them may mix policies differently."""
        first, second, third = (
            make_tagged_sample(1, "solver"),
            make_tagged_sample(2, "verifier"),
            make_tagged_sample(3, "solver"),
        )

        ans = split([[first, second], [third]])

        assert ans["solver"].group == [[first], [third]]
        assert ans["verifier"].group == [[second]]

    def test_the_prompt_group_travels_whole_into_every_split(self):
        # [FUNCTION DOC] Both policy parts carry the identical (same object) original prompt group.
        """A rejected group is recycled by resubmitting its prompts, which belong to no policy in particular."""
        prompt_group = make_group(7)

        ans = split([make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")], prompt_group=prompt_group)

        assert ans["solver"].prompt_group is prompt_group
        assert ans["verifier"].prompt_group is prompt_group

    def test_an_untagged_sample_is_refused_before_anything_is_routed(self):
        # [FUNCTION DOC] A group with one untagged sample fails an assertion saying the generate
        # function must stamp every sample.
        """Nothing downstream can guess where an unstamped sample belongs, so the split is where it must stop."""
        with pytest.raises(AssertionError, match="must stamp every sample"):
            split([make_tagged_sample(1, "solver"), make_tagged_sample(2, None)])

    def test_a_group_with_no_sample_left_reaches_no_policy(self):
        # [FUNCTION DOC] An empty group splits into an empty mapping.
        """A group every filter emptied belongs to nobody, and inventing a key for it would assert on None."""
        assert split([]) == {}


class TestFilterGroup:
    # [CLASS DOC] Tests of the buffer module's private filter that keeps only one trainer model's
    # samples from a possibly nested group.
    def test_it_keeps_only_the_samples_of_the_policy_asked_for(self):
        # [FUNCTION DOC] Filtering a flat solver+verifier group for "solver" yields only the solver
        # sample.
        """This is what stops a policy from training on another policy's responses."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        assert data_buffer._filter_group([solver, verifier], trainer_model_id="solver") == [solver]

    def test_it_keeps_a_sub_group_that_still_has_samples(self):
        # [FUNCTION DOC] A sub-group mixing both policies filtered for "solver" survives as a
        # sub-group holding just the solver sample.
        """A trajectory whose samples are split across policies survives on both sides, one sample each."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        assert data_buffer._filter_group([[solver, verifier]], trainer_model_id="solver") == [[solver]]

    def test_it_drops_a_sub_group_that_lost_every_sample(self):
        # [FUNCTION DOC] A sub-group holding only a verifier sample filtered for "solver" is removed,
        # giving an empty result rather than an empty sub-group.
        """An empty list left in place would be a trajectory that consumers must special-case forever."""
        assert data_buffer._filter_group([[make_tagged_sample(1, "verifier")]], trainer_model_id="solver") == []

    def test_it_leaves_the_group_it_was_given_untouched(self):
        # [FUNCTION DOC] After filtering a group (a sample plus a nested sub-group) for "solver",
        # the input group is unchanged.
        """It runs once per policy over the same group, so a mutating filter would eat the later policies' samples."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")
        group = [solver, [verifier]]

        data_buffer._filter_group(group, trainer_model_id="solver")

        assert group == [solver, [verifier]]


class RecordingBuffer(data_buffer.DefaultDataBuffer):
    # [CLASS DOC] HELPER: Default buffer subclass that remembers, in a class-level attribute shared
    # by all instances, the constructor input of the most recently built instance. Loaded by dotted
    # path from this module in several tests.
    constructed_with = None

    def __init__(self, input):
        # [FUNCTION DOC] HELPER: Calls the base constructor, then stores the input on the class.
        super().__init__(input)
        RecordingBuffer.constructed_with = input


class TestPerPolicyBufferClass:
    # [CLASS DOC] Tests of choosing a custom buffer class per policy via the per-model path flag.
    def test_every_policy_keeps_the_built_in_buffer_by_default(self):
        # [FUNCTION DOC] With no per-model paths, both inner buffers are of the default class.
        """The flag is opt-in, so a run that does not pass it must compose exactly what it composed before."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        assert [type(inner) for inner in buffer._inners.values()] == [
            data_buffer.DefaultDataBuffer,
            data_buffer.DefaultDataBuffer,
        ]

    def test_a_named_policy_gets_the_class_the_flag_names(self):
        # [FUNCTION DOC] Naming the recording buffer for "solver" gives solver exactly that class
        # while verifier keeps the default class.
        """Two policies can need different dataflow, which is the whole point of one buffer per policy."""
        buffer, _ = make_multi_buffer("solver", "verifier", paths_per_model=[f"solver={__name__}.RecordingBuffer"])

        assert type(buffer._inners["solver"]) is RecordingBuffer
        assert type(buffer._inners["verifier"]) is data_buffer.DefaultDataBuffer

    def test_the_custom_class_is_built_with_the_same_constructor_input(self):
        # [FUNCTION DOC] The custom class receives a constructor input whose unused handler equals
        # the one the test passed (appending to its list) and whose args are the same object the
        # default verifier buffer holds (read from its private args attribute).
        """A custom buffer owns staleness and recycling, so it needs the handler the built-in one gets."""
        buffer, unused = make_multi_buffer(
            "solver", "verifier", paths_per_model=[f"solver={__name__}.RecordingBuffer"]
        )

        assert RecordingBuffer.constructed_with.unused_handler_fn == unused.append
        assert RecordingBuffer.constructed_with.args is buffer._inners["verifier"]._args

    def test_a_policy_this_run_does_not_train_is_refused(self):
        # [FUNCTION DOC] A path entry for the unconfigured policy "reviewer" makes buffer
        # construction fail an assertion saying the run trains no such policy.
        """A typo would silently leave the policy it meant to configure on the built-in buffer."""
        with pytest.raises(AssertionError, match="train no policy of this run"):
            make_multi_buffer("solver", "verifier", paths_per_model=[f"reviewer={__name__}.RecordingBuffer"])


class TestParseDataBufferPaths:
    # [CLASS DOC] Tests of the buffer module's private parser of MODEL_ID=PATH flag entries.
    def test_it_maps_every_model_id_to_its_class_path(self):
        # [FUNCTION DOC] Two entries parse into a mapping from each model id to its path.
        """This is the mapping the composite buffer is built from."""
        assert data_buffer._parse_data_buffer_paths(["solver=pkg.A", "verifier=pkg.B"]) == {
            "solver": "pkg.A",
            "verifier": "pkg.B",
        }

    def test_an_unset_flag_is_an_empty_mapping(self):
        # [FUNCTION DOC] A none value parses into an empty mapping.
        """Default is every policy on the built-in buffer, which is the empty mapping."""
        assert data_buffer._parse_data_buffer_paths(None) == {}

    @pytest.mark.parametrize("entry", ["solver", "=pkg.A", "solver=", "solver =  "])
    def test_a_malformed_entry_is_refused(self, entry):
        # [FUNCTION DOC] Parametrized over entries lacking an equals sign, a model id, a path, or
        # both (whitespace only): each raises a value error saying the expected form is MODEL_ID=PATH.
        """Silently ignoring it would run the policy on a buffer the user did not ask for."""
        with pytest.raises(ValueError, match="expected MODEL_ID=PATH"):
            data_buffer._parse_data_buffer_paths([entry])

    def test_the_whitespace_around_an_entry_is_not_part_of_the_names(self):
        # [FUNCTION DOC] An entry with spaces around the id, the equals sign and the path parses with
        # the whitespace stripped from both id and path.
        """A shell-quoted entry keeps its spaces, and an import path with them resolves to nothing."""
        assert data_buffer._parse_data_buffer_paths([" solver = pkg.A "]) == {"solver": "pkg.A"}

    def test_a_model_id_named_twice_is_refused(self):
        # [FUNCTION DOC] Two entries for the same model id raise a value error mentioning a
        # duplicate model id.
        """One of the two class paths would win silently, and which one is not something to guess."""
        with pytest.raises(ValueError, match="Duplicate model id"):
            data_buffer._parse_data_buffer_paths(["solver=pkg.A", "solver=pkg.B"])


class TestDataBufferArgumentRegistration:
    # [CLASS DOC] Tests of the argument-registration hook of the fully async rollout function.
    def test_the_per_model_flag_is_declared_by_the_rollout_function_that_uses_it(self):
        # [FUNCTION DOC] Registering the function's arguments on a fresh parser makes the per-model
        # custom buffer path flag accept several values, parsed as a list of the given strings.
        """The framework asks the selected rollout function for its flags, so this hook must declare it."""
        parser = argparse.ArgumentParser()
        fully_async.FullyAsyncRolloutFn.add_arguments(parser)

        parsed = parser.parse_args(["--custom-async-data-buffer-path-per-model", "solver=pkg.A", "verifier=pkg.B"])

        assert parsed.custom_async_data_buffer_path_per_model == ["solver=pkg.A", "verifier=pkg.B"]

    def test_a_run_that_never_passes_the_flag_leaves_every_policy_on_the_built_in_buffer(self):
        # [FUNCTION DOC] Parsing no arguments leaves the per-model custom buffer path option as none.
        """The default has to be None, which _parse_data_buffer_paths reads as the empty mapping."""
        parser = argparse.ArgumentParser()
        fully_async.FullyAsyncRolloutFn.add_arguments(parser)

        assert parser.parse_args([]).custom_async_data_buffer_path_per_model is None


async def test_custom_data_buffer_path_replaces_default(monkeypatch):
    # [FUNCTION DOC] Setting the custom buffer path to the recording buffer class makes the function
    # use that class as its output buffer; the class was constructed with the function's own recycle
    # method as the unused handler; and a train call still returns 2 groups (batch size default).
    path = f"{__name__}.RecordingBuffer"
    args = make_args(custom_async_data_buffer_path=path, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, FakeDataSource())

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert type(fn._output) is RecordingBuffer
    assert RecordingBuffer.constructed_with.unused_handler_fn == fn._recycle
    assert len(output.samples) == 2


class MultiPolicyDataSource(FakeDataSource):
    # [CLASS DOC] HELPER: Fake data source that stamps all samples of each manufactured group with
    # one of two model ids, "a" or "b", alternating by get-call parity.
    """Stamps every sample of a group with one policy, alternating, as a multi policy generate function does."""

    def get_samples(self, num_samples):
        # [FUNCTION DOC] HELPER: Gets the single group from the base class, stamps every sample "a"
        # when the call count is odd and "b" when even, and returns the one-group list.
        [group] = super().get_samples(num_samples)
        for sample in group:
            sample.trainer_model_id = "a" if self.num_get_calls % 2 == 1 else "b"
        return [group]


class WedgedBuffer(data_buffer.DataBuffer):
    # [CLASS DOC] HELPER: Buffer whose put and get block forever on an event that is never set, used
    # to park the producer and consumer for disposal tests.
    def __init__(self, input: data_buffer.DataBufferConstructorInput):
        # [FUNCTION DOC] HELPER: Creates the never-set event; ignores the input and the base class.
        self._never = asyncio.Event()

    async def put(self, input: data_buffer.DataBufferInput) -> None:
        # [FUNCTION DOC] HELPER: Blocks forever.
        await self._never.wait()

    async def get(self, **context) -> data_buffer.DataBufferInput:
        # [FUNCTION DOC] HELPER: Blocks forever (the assertion afterwards is unreachable unless the
        # event is set).
        await self._never.wait()
        raise AssertionError("the wedged buffer never hands out a group")

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        # [FUNCTION DOC] HELPER: Returns an empty metrics mapping.
        return {}


class TestDisposal:
    # [CLASS DOC] Tests of disposing the rollout function while its producer is parked on a wedged
    # buffer.
    async def test_disposing_ends_the_producer_before_it_returns(self, monkeypatch):
        # [FUNCTION DOC] A train step is started against the wedged buffer and stays unfinished after
        # 50 ms. Disposing (with a 5 s timeout) returns only after the worker task is cancelled.
        """Teardown has to be done when it returns, or the rest of it races the producer's unwinding."""
        args = make_args(rollout_batch_size=1, custom_async_data_buffer_path=f"{__name__}.WedgedBuffer")
        fn = make_fn(monkeypatch, args, FakeDataSource())
        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
        await asyncio.sleep(0.05)
        assert not step.done()

        await asyncio.wait_for(fn.dispose(), timeout=5)

        assert fn._worker.cancelled()

    async def test_disposing_fails_a_step_that_is_waiting_for_groups(self, monkeypatch):
        # [FUNCTION DOC] After disposing while a train step waits on the wedged buffer, that step
        # fails with a runtime error saying it was disposed while a step waited for groups.
        """A run whose producer is parked when it ends must not leave the waiting step there forever."""
        args = make_args(rollout_batch_size=1, custom_async_data_buffer_path=f"{__name__}.WedgedBuffer")
        fn = make_fn(monkeypatch, args, FakeDataSource())
        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
        await asyncio.sleep(0.05)

        await fn.dispose()

        with pytest.raises(RuntimeError, match="disposed while a step waited for groups"):
            await asyncio.wait_for(step, timeout=5)

    async def test_disposing_a_run_that_never_started_is_quiet(self, monkeypatch):
        # [FUNCTION DOC] Disposing a function that never ran a step raises nothing and leaves the
        # worker unset.
        """dispose runs on every teardown path, including one that failed before the first step."""
        fn = make_fn(monkeypatch, make_args(), FakeDataSource())

        await fn.dispose()

        assert fn._worker is None


class RecordingMultiBuffer(data_buffer.DefaultMultiDataBuffer):
    # [CLASS DOC] HELPER: Multi-policy buffer subclass that records, in a shared class-level list,
    # the keyword context of every get call before delegating to the base class.
    get_calls: list[dict] = []

    async def get(self, **context):
        # [FUNCTION DOC] HELPER: Appends the context mapping to the class-level list and returns
        # the base class result.
        RecordingMultiBuffer.get_calls.append(context)
        return await super().get(**context)


class TestBufferSelection:
    # [CLASS DOC] Tests of which buffer class the rollout function selects for its output.
    async def test_a_single_policy_run_keeps_the_plain_buffer(self, monkeypatch):
        # [FUNCTION DOC] Without a megatron config, after one train call the output buffer is of
        # the exact default single-policy class.
        """Every existing run goes through this line, and a per-policy buffer would key it under a model id."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

        await fn(RolloutFnTrainInput(rollout_id=0))

        assert type(fn._output) is data_buffer.DefaultDataBuffer

    async def test_a_multi_policy_run_defaults_to_the_per_policy_buffer(self, monkeypatch):
        # [FUNCTION DOC] With a megatron config naming policies a and b and a source stamping groups
        # alternately, a train call for trainer model "a" selects the exact default multi-policy
        # buffer class, and the returned samples are two, all tagged "a".
        """One shared queue would hand a policy the groups another policy generated."""
        args = make_args(rollout_batch_size=1, megatron_config=encode_megatron_config("a", "b"))
        fn = make_fn(monkeypatch, args, MultiPolicyDataSource())

        output = await fn(RolloutFnTrainInput(rollout_id=0, trainer_model_id="a"))

        assert type(fn._output) is data_buffer.DefaultMultiDataBuffer
        assert [sample.trainer_model_id for group in output.samples for sample in group] == ["a", "a"]

    async def test_a_custom_buffer_still_wins_in_a_multi_policy_run(self, monkeypatch):
        # [FUNCTION DOC] In a two-policy run, setting the custom buffer path to the recording multi
        # buffer makes the output buffer exactly that class. The class-level call list is reset
        # first.
        """--custom-async-data-buffer-path is how a run replaces the queue, whatever the default would have been."""
        RecordingMultiBuffer.get_calls = []
        args = make_args(
            rollout_batch_size=1,
            megatron_config=encode_megatron_config("a", "b"),
            custom_async_data_buffer_path=f"{__name__}.RecordingMultiBuffer",
        )
        fn = make_fn(monkeypatch, args, MultiPolicyDataSource())

        await fn(RolloutFnTrainInput(rollout_id=0, trainer_model_id="a"))

        assert type(fn._output) is RecordingMultiBuffer

    async def test_the_consumer_asks_the_buffer_for_the_policy_that_called_it(self, monkeypatch):
        # [FUNCTION DOC] With the recording multi buffer, a train call with weight version 4 and
        # trainer model "a" results in exactly one buffer get whose context is the current version 4
        # plus trainer model id "a".
        """The queue is keyed by policy, so a consumer that forgets to name itself drains whoever answers first."""
        RecordingMultiBuffer.get_calls = []
        args = make_args(
            rollout_batch_size=1,
            megatron_config=encode_megatron_config("a", "b"),
            custom_async_data_buffer_path=f"{__name__}.RecordingMultiBuffer",
        )
        fn = make_fn(monkeypatch, args, MultiPolicyDataSource())

        await fn(RolloutFnTrainInput(rollout_id=0, weight_version=4, trainer_model_id="a"))

        assert RecordingMultiBuffer.get_calls == [dict(current_version=4, trainer_model_id="a")]


async def test_worker_defaults_to_sample_granularity(monkeypatch):
    # [FUNCTION DOC] With batch size 1 and no granularity set, generation blocks and records the
    # per-sample completion callback it is given. After 10 ms one prompt group was requested. Calling
    # the first callback once per sample of the pending group (2 times) makes the worker submit a
    # replacement group (2 requests in total) even though the first group has not returned. After
    # release, the train call returns one group.
    """Unset --rollout-submission-granularity: this driver backfills on sample completion."""
    callbacks = []
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        # [FUNCTION DOC] HELPER: Records the completion callback, waits for release, returns the
        # group unchanged.
        callbacks.append(sample_done_callback)
        await release.wait()
        return group

    data_source = FakeDataSource()
    args = make_args(rollout_batch_size=1)
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.01)
    assert data_source.num_get_calls == 1

    # Report every sample of the still-pending group as finished.
    for _ in range(N_SAMPLES_PER_PROMPT):
        callbacks[0]()
    await asyncio.sleep(0.01)

    # A replacement group went out even though the first group has not returned.
    assert data_source.num_get_calls == 2

    release.set()
    output = await drain
    assert len(output.samples) == 1


async def test_group_granularity_opts_the_worker_out_of_backfill(monkeypatch):
    # [FUNCTION DOC] With granularity "group" and batch size 1, generation receives no completion
    # callback (none), and only one prompt group is requested even after waiting a further 10 ms.
    # After release the train call returns one group.
    callbacks = []
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        # [FUNCTION DOC] HELPER: Records the completion callback, waits for release, returns the
        # group unchanged.
        callbacks.append(sample_done_callback)
        await release.wait()
        return group

    data_source = FakeDataSource()
    args = make_args(rollout_batch_size=1, rollout_submission_granularity="group")
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.01)
    assert data_source.num_get_calls == 1
    # no callback wired at group level
    assert callbacks == [None]

    await asyncio.sleep(0.01)
    assert data_source.num_get_calls == 1

    release.set()
    output = await drain
    assert len(output.samples) == 1


class TestRolloutFnContract:
    # [CLASS DOC] Tests of the rollout function's contract with the framework's loader.
    def test_it_is_a_rollout_fn_the_loader_accepts(self):
        # [FUNCTION DOC] The fully async rollout function class is a subclass of the base rollout
        # function class.
        """load_rollout_fn gates on issubclass(fn, BaseRolloutFn), so a class that forgets the
        base is rejected at startup no matter how complete its behaviour is."""
        assert issubclass(fully_async.FullyAsyncRolloutFn, BaseRolloutFn)

    def test_the_constructor_input_reaches_the_base(self, monkeypatch):
        # [FUNCTION DOC] After construction, the function's stored constructor input carries the
        # same data source object that was passed in.
        """The base stores it as constructor_input; skipping super().__init__ leaves the
        attribute missing on every path that reads it."""
        data_source = FakeDataSource()
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)

        assert fn.constructor_input.data_source is data_source


async def test_worker_bounds_in_flight_groups(monkeypatch):
    # [FUNCTION DOC] With batch size 2 and blocked generation, 50 ms after starting a train call
    # exactly 2 prompt groups were requested (the in-flight bound equals the batch size). After
    # release the call returns 2 groups.
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        # [FUNCTION DOC] HELPER: Waits for the release event, then returns the group unchanged.
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == 2  # in-flight bound, not more

    release.set()
    output = await drain
    assert len(output.samples) == 2
