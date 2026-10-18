# [MODULE DOC] Async tests of the training rollout driver's submission scheduling and group
# post-processing. A harness replaces the module's task submission, abort and a few setup calls
# with fakes so that tests control exactly when each submitted group (or individual sample)
# "completes". Group granularity submits replacement groups only when a whole group task returns;
# sample granularity ("backfill") counts individual sample completions as credits and submits a
# replacement group as soon as a whole group's worth of samples finished. Also covered: groups
# with a missing reward being refilled (and the metrics counters for dropped groups), aborted
# groups classified before missing rewards, failure of one sample cancelling its siblings while
# still firing every completion callback, and the scheduler objects' capacity and progress
# waiting behavior. Registered as a CPU CI test in the stage-a CPU suite.
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace

import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args as make_rollout_args

import miles.rollout.inference_rollout.inference_rollout_train as train
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.submission_scheduler import (
    GroupLevelSubmission,
    SampleBackfillSubmission,
    make_submission_scheduler,
)
from miles.utils.types import Sample

GROUP_SIZE = 4


def make_args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds a rollout arguments namespace with defaults: global dataset on,
    # rollout batch size 2, 4 samples per prompt, over-sampling batch size 1, no submission
    # granularity override, round-robin router policy, no dynamic filter, no reward key, no
    # sample filter or all-samples process hooks, router at a local address and port 30000. The
    # given keyword overrides replace defaults (new keys are added).
    defaults = dict(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=GROUP_SIZE,
        over_sampling_batch_size=1,
        rollout_submission_granularity=None,
        sglang_router_policy="round_robin",
        dynamic_sampling_filter_path=None,
        reward_key=None,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def make_group(group_index: int) -> list[Sample]:
    # [FUNCTION DOC] HELPER: Builds a group of 4 completed samples for the group index: sample
    # indices are group index times 10 plus the position, prompt "prompt <group index>", response
    # "ok" of length 1, label "ok" and reward 1.
    return [
        Sample(
            group_index=group_index,
            index=group_index * 10 + i,
            prompt=f"prompt {group_index}",
            response="ok",
            response_length=1,
            label="ok",
            reward=1,
            status=Sample.Status.COMPLETED,
        )
        for i in range(GROUP_SIZE)
    ]


class FakeGenerateState:
    # [CLASS DOC] HELPER: Minimal generation state holding the arguments, empty sampling
    # parameters, an aborted flag and a count of how many times it was reset.
    def __init__(self, args: Namespace):
        # [FUNCTION DOC] HELPER: Stores the arguments; sampling parameters empty, not aborted,
        # reset count zero.
        self.args = args
        self.sampling_params = {}
        self.aborted = False
        self.reset_count = 0

    def reset(self) -> None:
        # [FUNCTION DOC] HELPER: Clears the aborted flag and increments the reset count.
        self.aborted = False
        self.reset_count += 1


class Harness:
    # [CLASS DOC] HELPER: Test harness around the training rollout's asynchronous driver. Records
    # the group indices and groups submitted, the sample-done callbacks received per submission,
    # and one blocking event per submitted group; a group task completes only when its blocker is
    # set. Also acts as the data source producing fresh numbered groups.
    """Drives generate_rollout_async with hand-controlled group and sample completions."""

    def __init__(self, monkeypatch, args: Namespace):
        # [FUNCTION DOC] HELPER: Creates a fake generation state and the empty records, then
        # patches the driver module (through the monkeypatch fixture): task submission is
        # replaced by a fake that, per group, records its group index, group and sample-done
        # callback, creates a blocker and spawns a task waiting on it; abort is replaced by a fake
        # that marks the state aborted, cancels and awaits all pending tasks, and returns no
        # groups; function loading returns none; the sglang dump configuration and the prefill
        # log probability recomputation become no-ops.
        self.args = args
        self.state = FakeGenerateState(args)
        self.submitted_group_indices: list[int] = []
        self.submitted_groups: list[list[Sample]] = []
        self.callbacks: list = []
        self._next_group_index = 0
        self._blockers: list[asyncio.Event] = []

        def fake_submit_generate_tasks(_state, samples, sample_done_callback=None):
            # [FUNCTION DOC] HELPER: Fake submission. For each group records its first sample's
            # group index, the group, a new blocker and the callback, and creates a task that
            # finishes with the group once its blocker is set. Returns the list of tasks.
            tasks = []
            for group in samples:
                self.submitted_group_indices.append(group[0].group_index)
                self.submitted_groups.append(group)
                blocker = asyncio.Event()
                self._blockers.append(blocker)
                self.callbacks.append(sample_done_callback)
                tasks.append(asyncio.create_task(self._run_group(group, blocker)))
            return tasks

        async def fake_abort(_state, pendings, _rollout_id):
            # [FUNCTION DOC] HELPER: Fake abort. Sets the state's aborted flag, cancels every
            # pending task, waits for them (collecting exceptions), and returns an empty list.
            _state.aborted = True
            for task in pendings:
                task.cancel()
            await asyncio.gather(*pendings, return_exceptions=True)
            return []

        async def noop_configure_sglang(_args):
            # [FUNCTION DOC] HELPER: Does nothing; replaces the sglang dump configuration.
            return None

        async def noop_recompute(*_args, **_kwargs):
            # [FUNCTION DOC] HELPER: Does nothing; replaces the prefill log probability
            # recomputation.
            return None

        monkeypatch.setattr(train, "submit_generate_tasks", fake_submit_generate_tasks)
        monkeypatch.setattr(train, "abort", fake_abort)
        monkeypatch.setattr(train, "load_function", lambda _path: None)
        monkeypatch.setattr(train.dumper_utils, "configure_sglang", noop_configure_sglang)
        monkeypatch.setattr(train, "recompute_samples_rollout_logprobs_via_prefill", noop_recompute)

    async def _run_group(self, group, blocker):
        # [FUNCTION DOC] HELPER: Waits until the blocker event is set and returns the group.
        await blocker.wait()
        return group

    def data_source(self, num_groups: int) -> list[list[Sample]]:
        # [FUNCTION DOC] HELPER: Data source callback for the driver. Produces the requested
        # number of groups with consecutively increasing group indices (starting at 1 and
        # continuing across calls). When the arguments have a reward key, each sample's reward is
        # replaced by a mapping from that key to 1.0. Returns the list of groups.
        groups = []
        for _ in range(num_groups):
            self._next_group_index += 1
            group = make_group(self._next_group_index)
            if self.args.reward_key is not None:
                for sample in group:
                    sample.reward = {self.args.reward_key: 1.0}
            groups.append(group)
        return groups

    def run(self):
        # [FUNCTION DOC] HELPER: Starts the driver's asynchronous rollout for rollout id 0 with the
        # fake state and this data source as a background task and returns the task (which
        # resolves to the output and a second value).
        return asyncio.create_task(train.generate_rollout_async(self.state, 0, self.data_source))

    def finish_group(self, submission_index: int) -> None:
        # [FUNCTION DOC] HELPER: Releases the blocker of the submission with the given position
        # (order of submission), letting that group's task return.
        self._blockers[submission_index].set()

    def finish_samples(self, submission_index: int, count: int) -> None:
        # [FUNCTION DOC] HELPER: Calls the sample-done callback that was given for the submission
        # the given number of times without finishing the group task. Asserts the callback exists
        # (sample granularity must be on).
        """Report `count` individual sample completions for a still-pending group."""
        callback = self.callbacks[submission_index]
        assert callback is not None, "sample granularity is off, no sample callback was wired"
        for _ in range(count):
            callback()


async def test_sync_driver_defaults_to_group_granularity(monkeypatch):
    # [FUNCTION DOC] With batch size 2 and no granularity override, starts the driver, yields once,
    # and asserts groups 1 and 2 were submitted with no sample callbacks (group granularity). After
    # finishing both groups, asserts the output holds groups 1 and 2 in order.
    harness = Harness(monkeypatch, make_args(rollout_batch_size=2))
    task = harness.run()
    await asyncio.sleep(0)

    # Two groups fill the batch; no sample callback is wired at all.
    assert harness.submitted_group_indices == [1, 2]
    assert harness.callbacks == [None, None]

    harness.finish_group(0)
    harness.finish_group(1)
    output, _ = await task
    assert [group[0].group_index for group in output.samples] == [1, 2]


@pytest.mark.parametrize(
    ("reward_key", "missing_reward"),
    [(None, None), ("score", None), ("score", {"score": None})],
)
async def test_missing_reward_group_is_refilled_before_default_conversion(monkeypatch, reward_key, missing_reward):
    # [FUNCTION DOC] Parametrized over (no reward key, reward none), (key "score", reward none)
    # and (key "score", mapping with a none score). With batch size 1, sets the first sample of
    # the first submitted group to the missing reward and finishes the group. Asserts the driver
    # submitted a replacement group (indices 1 then 2), and after finishing it the output holds
    # only group 2. Then post-processes the output with rollout arguments (batch 1, group size
    # 4, no reward normalization, same reward key) and converts to training data, asserting 4
    # samples, no missing rewards in the training data, all finite rewards, and a metric counting
    # exactly 1 group dropped for a missing reward.
    harness = Harness(monkeypatch, make_args(rollout_batch_size=1, reward_key=reward_key))
    task = harness.run()
    await asyncio.sleep(0)

    harness.submitted_groups[0][0].reward = missing_reward
    harness.finish_group(0)
    await asyncio.sleep(0.01)

    assert harness.submitted_group_indices == [1, 2]
    harness.finish_group(1)
    output, _ = await task
    assert [group[0].group_index for group in output.samples] == [2]

    args = make_rollout_args(
        rollout_batch_size=1,
        n_samples_per_prompt=GROUP_SIZE,
        global_batch_size=GROUP_SIZE,
        rewards_normalization=False,
        reward_key=reward_key,
    )
    samples, metadata = postprocess_rollout_data(args, output.samples, train_parallel_config={"dp_size": 1})
    train_data = convert_samples_to_train_data(
        args,
        samples,
        metadata=metadata,
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )

    assert len(samples) == GROUP_SIZE
    assert all(reward is not None for reward in train_data["rewards"])
    assert torch.isfinite(torch.tensor(train_data["rewards"], dtype=torch.float32)).all()
    assert output.metrics["rollout/dynamic_filter/drop_group_has_missing_reward"] == 1


async def test_aborted_group_is_classified_before_missing_reward(monkeypatch):
    # [FUNCTION DOC] With batch size 1, marks every sample of the first group aborted with no
    # reward and finishes it; the driver submits a replacement, which is finished. Asserts only
    # group 2 is in the output, the metric for groups dropped as aborted is 1, and the metric for
    # groups dropped for a missing reward is absent (aborted classification wins).
    harness = Harness(monkeypatch, make_args(rollout_batch_size=1))
    task = harness.run()
    await asyncio.sleep(0)

    for sample in harness.submitted_groups[0]:
        sample.status = Sample.Status.ABORTED
        sample.reward = None
    harness.finish_group(0)
    await asyncio.sleep(0.01)

    harness.finish_group(1)
    output, _ = await task

    assert [group[0].group_index for group in output.samples] == [2]
    assert output.metrics["rollout/dynamic_filter/drop_group_has_aborted"] == 1
    assert "rollout/dynamic_filter/drop_group_has_missing_reward" not in output.metrics


async def test_backfill_submits_replacement_before_the_group_returns(monkeypatch):
    # [FUNCTION DOC] With batch size 2 and sample granularity, asserts groups 1 and 2 are
    # submitted first. Reports 3 sample completions on group 1 and 1 on group 2 (a whole group's
    # worth, split over two still-pending groups), waits briefly, and asserts a third group was
    # submitted although no group task returned. Finishes the three groups and asserts the output
    # holds 2 groups.
    harness = Harness(monkeypatch, make_args(rollout_batch_size=2, rollout_submission_granularity="sample"))
    task = harness.run()
    await asyncio.sleep(0)
    assert harness.submitted_group_indices == [1, 2]

    # A whole group's worth of samples finishes, spread across two still-pending groups.
    harness.finish_samples(0, GROUP_SIZE - 1)
    harness.finish_samples(1, 1)
    await asyncio.sleep(0.01)

    # A replacement group is in flight even though no group task has returned.
    assert harness.submitted_group_indices == [1, 2, 3]

    for i in range(3):
        harness.finish_group(i)
    output, _ = await task
    assert len(output.samples) == 2


async def test_backfill_does_not_oversubmit_below_one_group(monkeypatch):
    # [FUNCTION DOC] With batch size 2 and sample granularity, reports 3 sample completions (one
    # short of a group) and asserts no replacement group is submitted. Finishes both groups and
    # asserts the output holds 2 groups.
    harness = Harness(monkeypatch, make_args(rollout_batch_size=2, rollout_submission_granularity="sample"))
    task = harness.run()
    await asyncio.sleep(0)
    assert harness.submitted_group_indices == [1, 2]

    # One sample short of a full group: no replacement yet.
    harness.finish_samples(0, GROUP_SIZE - 1)
    await asyncio.sleep(0.01)
    assert harness.submitted_group_indices == [1, 2]

    harness.finish_group(0)
    harness.finish_group(1)
    output, _ = await task
    assert len(output.samples) == 2


async def test_failed_sample_cancels_siblings_and_conserves_credits(monkeypatch):
    # [FUNCTION DOC] Replaces the single-sample generate-and-reward routine with a fake that
    # raises a runtime error for the sample with index 10 (the first of group 1) and otherwise
    # waits forever until cancelled. Runs the group routine with a sample-done callback that
    # records each firing, asserts the runtime error propagates, and asserts the callback fired
    # once for every sample of the group (4), i.e. all sample tasks settled, cancelled ones
    # included, before the group raised.
    """One sample raising must not leave siblings running or credits unreturned."""
    from miles.rollout.inference_rollout import inference_rollout_common as common

    async def fake_generate_and_rm(state, sample, sampling_params, evaluation=False):
        # [FUNCTION DOC] HELPER: Raises a runtime error saying sample failed for sample index 10;
        # for any other sample waits on an event that is never set (until cancelled).
        if sample.index == 10:  # first sample of make_group(1)
            raise RuntimeError("sample failed")
        await asyncio.Event().wait()  # runs until cancelled

    monkeypatch.setattr(common, "generate_and_rm", fake_generate_and_rm)

    fired = []
    state = FakeGenerateState(make_args())
    with pytest.raises(RuntimeError, match="sample failed"):
        await common.generate_and_rm_group(
            state, make_group(1), sampling_params={}, sample_done_callback=lambda: fired.append(1)
        )
    # every sample task settled (cancelled included) before the group raised
    assert len(fired) == GROUP_SIZE


class TestSubmissionSchedulers:
    # [CLASS DOC] Direct tests of the submission scheduler objects: the factory, capacity
    # accounting (groups versus sample credits) and waiting for progress.
    @pytest.mark.parametrize(
        "granularity,default,expected",
        [
            (None, "group", GroupLevelSubmission),
            (None, "sample", SampleBackfillSubmission),
            ("group", "sample", GroupLevelSubmission),
            ("sample", "group", SampleBackfillSubmission),
        ],
    )
    def test_factory_resolves_granularity(self, granularity, default, expected):
        # [FUNCTION DOC] Parametrized: an unset granularity falls back to the caller's default
        # (group gives the group-level scheduler, sample gives the backfill scheduler); an explicit
        # granularity in the arguments wins over the default. Asserts the exact scheduler type.
        args = make_args(rollout_submission_granularity=granularity)
        assert type(make_submission_scheduler(args, default=default)) is expected

    def test_group_level_counts_groups(self):
        # [FUNCTION DOC] The group-level scheduler has no sample callback and has capacity when
        # pending groups (1) are below the group budget (2) but not when equal (2 of 2).
        scheduler = GroupLevelSubmission()
        assert scheduler.sample_done_callback is None
        assert scheduler.has_capacity(pending_groups=1, group_budget=2)
        assert not scheduler.has_capacity(pending_groups=2, group_budget=2)

    def test_backfill_counts_samples(self):
        # [FUNCTION DOC] Submitting two groups makes the backfill scheduler count 8 samples in
        # flight and report no capacity for 2 pending groups with budget 2. After 3 sample
        # completions there is still no capacity; the 4th completion (a whole group's worth)
        # restores capacity.
        scheduler = SampleBackfillSubmission(GROUP_SIZE)

        scheduler.on_submit([make_group(1), make_group(2)])
        assert scheduler.samples_in_flight == 2 * GROUP_SIZE
        assert not scheduler.has_capacity(pending_groups=2, group_budget=2)

        for _ in range(GROUP_SIZE - 1):
            scheduler.sample_done_callback()
        assert not scheduler.has_capacity(pending_groups=2, group_budget=2)

        scheduler.sample_done_callback()
        assert scheduler.has_capacity(pending_groups=2, group_budget=2)

    def test_orphaned_credits_resync_when_nothing_is_pending(self):
        # [FUNCTION DOC] After submitting two groups whose sample tasks never spawned, asking for
        # capacity with zero pending groups returns true and resets the in-flight sample count to
        # zero (orphaned credits are resynchronized).
        scheduler = SampleBackfillSubmission(GROUP_SIZE)
        # credits whose sample tasks never spawned
        scheduler.on_submit([make_group(1), make_group(2)])

        assert scheduler.has_capacity(pending_groups=0, group_budget=2)
        assert scheduler.samples_in_flight == 0

    def test_credits_survive_while_groups_are_pending(self):
        # [FUNCTION DOC] After submitting one group, a capacity query with one pending group does
        # not reset the credits: the in-flight sample count stays 4.
        scheduler = SampleBackfillSubmission(GROUP_SIZE)
        scheduler.on_submit([make_group(1)])

        scheduler.has_capacity(pending_groups=1, group_budget=2)
        assert scheduler.samples_in_flight == GROUP_SIZE

    @pytest.mark.parametrize("scheduler_cls", [GroupLevelSubmission, lambda: SampleBackfillSubmission(GROUP_SIZE)])
    async def test_wait_for_progress_returns_on_group_completion(self, scheduler_cls):
        # [FUNCTION DOC] Parametrized over both scheduler kinds. Starts waiting for progress on one
        # pending group task, yields once and asserts the wait has not returned, then releases the
        # group and asserts the wait returns the group in the done set and an empty pending set.
        scheduler = scheduler_cls()
        blocker = asyncio.Event()

        async def group():
            # [FUNCTION DOC] HELPER: Waits until the blocker is set.
            await blocker.wait()

        task = asyncio.create_task(group())
        waiter = asyncio.create_task(scheduler.wait_for_progress({task}))
        await asyncio.sleep(0)
        assert not waiter.done()

        blocker.set()
        done, pending = await waiter
        assert done == {task}
        assert pending == set()

    async def test_wait_for_progress_returns_on_sample_completion(self):
        # [FUNCTION DOC] With the backfill scheduler and one never-finishing group task, starts
        # waiting and asserts it has not returned after one yield. A sample completion wakes the
        # wait; asserts the done set is empty (the internal sample waiter is filtered out) and the
        # pending set is the group task. Finally cancels the group task.
        scheduler = SampleBackfillSubmission(GROUP_SIZE)
        never = asyncio.create_task(asyncio.Event().wait())

        waiter = asyncio.create_task(scheduler.wait_for_progress({never}))
        await asyncio.sleep(0)
        assert not waiter.done()

        scheduler.sample_done_callback()
        done, pending = await waiter
        assert done == set()  # the sample waiter is filtered out of the result
        assert pending == {never}

        never.cancel()
