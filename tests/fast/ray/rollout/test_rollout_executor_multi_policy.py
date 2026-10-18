# [MODULE DOC]
# Unit tests for the rollout executor when several policies (trainer model ids) share one
# executor: weight versions and train parallel configs are kept per model id (with none as the
# single-policy key), data fetching uses the requesting policy's own version and layout, timers and
# logged metric names are prefixed per policy, and the "no weight version published" watchdog
# counts per policy. The heavy rollout pipeline steps (debug saving, conversion, splitting, the
# watchdog) are replaced by quiet stand-ins through an autouse fixture, and the process-wide timer
# singleton is cleared before and after each test.
import asyncio
from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest

from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import RolloutFnTrainInput
from miles.utils.timer import Timer
from miles.utils.weight_version import (
    assert_weight_version_is_published,
    max_rollouts_without_published_weight_version,
)


@pytest.fixture(autouse=True)
def _quiet_rollout_pipeline(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture. Clears the timer singleton's timers and start times,
    # replaces in the executor module the debug data saver, the sample-to-train-data conversion
    # (returns an empty mapping), the split by data parallel (returns nothing) and the weight
    # version published assertion with no-ops, yields to the test, then clears the timer again.
    Timer().timers.clear()
    Timer().start_time.clear()
    monkeypatch.setattr(rollout_executor_module, "save_debug_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rollout_executor_module, "convert_samples_to_train_data", lambda *a, **kw: {})
    monkeypatch.setattr(rollout_executor_module, "split_train_data_by_dp", lambda *a, **kw: None)
    monkeypatch.setattr(rollout_executor_module, "assert_weight_version_is_published", lambda *a, **kw: None)
    yield
    Timer().timers.clear()
    Timer().start_time.clear()


def _make_executor() -> RolloutExecutor:
    # [FUNCTION DOC] HELPER: Builds an executor without its constructor, with a minimal arguments
    # namespace (no delayed split, no independent dp, no debug modes, no lora, weight update
    # interval 1), an empty data source, no custom hooks, empty per-model weight version and
    # parallel config mappings, a per-model rollout counter defaulting to zero, and rollout id -1.
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor.args = Namespace(
        delay_split_train_data_by_dp=False,
        indep_dp=False,
        load_debug_rollout_data=None,
        ci_inject_rollout_data_path=None,
        debug_rollout_only=False,
        debug_train_only=False,
        debug_skip_weight_update=False,
        lora_rank=0,
        update_weights_interval=1,
    )
    executor.data_source = Namespace()
    executor.custom_convert_samples_to_train_data_func = None
    executor.custom_reward_post_process_func = None
    executor._weight_versions_of_model_id = {}
    executor._train_parallel_configs_of_model_id = {}
    executor._rollouts_since_publish_of_model_id = defaultdict(int)
    executor.rollout_id = -1
    return executor


def _record_generate_inputs(executor: RolloutExecutor, monkeypatch) -> list[RolloutFnTrainInput]:
    # [FUNCTION DOC] HELPER: Marks the executor as non-legacy with a dummy rollout function and
    # patches the module's rollout function caller to record each train input and return a result
    # with no samples and no metrics; also patches the sample weight-version sanity check to a
    # no-op. Returns the list that collects the recorded inputs.
    received: list[RolloutFnTrainInput] = []
    executor.use_legacy_rollout_v1 = False
    executor.generate_rollout = object()

    def _call_rollout_function(rollout_function, rollout_input: RolloutFnTrainInput):
        # [FUNCTION DOC] HELPER: Records the input and returns an empty result.
        received.append(rollout_input)
        return SimpleNamespace(samples=[], metrics=None)

    monkeypatch.setattr(rollout_executor_module, "call_rollout_function", _call_rollout_function)
    monkeypatch.setattr(rollout_executor_module, "assert_samples_weight_version_sane", lambda *a, **kw: None)
    return received


def _record_postprocess_configs(monkeypatch) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Patches the module's rollout data post-processing to record the train
    # parallel config it receives and to return the data unchanged with no metadata. Returns the
    # list that collects the recorded configs.
    seen: list[dict[str, Any]] = []

    def _postprocess_rollout_data(args, data, *, train_parallel_config: dict[str, Any]):
        # [FUNCTION DOC] HELPER: Records the config; returns the data and no metadata.
        seen.append(train_parallel_config)
        return data, None

    monkeypatch.setattr(rollout_executor_module, "postprocess_rollout_data", _postprocess_rollout_data)
    return seen


def _record_logged_model_ids(monkeypatch) -> list[str | None]:
    # [FUNCTION DOC] HELPER: Patches the module's rollout data logging function to record the
    # trainer model id keyword it receives (none if absent). Returns the list of recorded ids.
    logged: list[str | None] = []
    monkeypatch.setattr(
        rollout_executor_module,
        "log_rollout_data",
        lambda *a, trainer_model_id=None, **kw: logged.append(trainer_model_id),
    )
    return logged


class TestPerPolicyKeying:
    # [CLASS DOC] Tests that the executor keys weight versions and train parallel configs by
    # trainer model id and uses the requesting policy's own values when producing data.
    def test_every_policy_keeps_its_own_weight_version(self):
        # [FUNCTION DOC] Setting versions 3 for policy a and 7 for policy b leaves the executor's
        # per-model mapping equal to a to 3 and b to 7.
        """One shared version would let a policy's samples be judged against another policy's weights."""
        executor = _make_executor()

        executor.set_weight_version(3, trainer_model_id="a")
        executor.set_weight_version(7, trainer_model_id="b")

        assert executor._weight_versions_of_model_id == {"a": 3, "b": 7}

    def test_a_single_policy_run_keys_everything_under_none(self):
        # [FUNCTION DOC] Setting a weight version and a parallel config (data parallel size 2)
        # without a model id stores them under the none key in the respective mappings.
        """None is the single-policy key, so the base path must not grow a name of its own."""
        executor = _make_executor()

        executor.set_weight_version(3)
        executor.set_train_parallel_config({"dp_size": 2})

        assert executor._weight_versions_of_model_id == {None: 3}
        assert executor._train_parallel_configs_of_model_id == {None: {"dp_size": 2}}

    def test_a_version_going_backwards_for_one_policy_is_still_refused(self):
        # [FUNCTION DOC] Sets policy a to 7, policy b to 9 (higher, interleaved), then policy a to 5:
        # an assertion error saying the version went backwards is raised for a.
        """The regression check must compare a policy against itself, not against whoever published last."""
        executor = _make_executor()
        executor.set_weight_version(7, trainer_model_id="a")

        executor.set_weight_version(9, trainer_model_id="b")

        with pytest.raises(AssertionError, match="went backwards"):
            executor.set_weight_version(5, trainer_model_id="a")

    async def test_each_policy_is_sharded_by_its_own_parallel_config(self, monkeypatch):
        # [FUNCTION DOC] Policy a has data parallel size 8 and policy b size 4. Patches the split
        # step to record its config and the executor's private rollout data fetch to return empty
        # data. Calls get for policy b and asserts the split saw only b's config.
        """Splitting a policy's batch by another policy's dp size hands its ranks the wrong shards."""
        executor = _make_executor()
        executor.set_train_parallel_config({"dp_size": 8}, trainer_model_id="a")
        executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id="b")
        seen: list[dict] = []
        monkeypatch.setattr(
            rollout_executor_module, "split_train_data_by_dp", lambda args, data, config: seen.append(config)
        )
        _record_logged_model_ids(monkeypatch)

        async def _get_rollout_data(rollout_id, trainer_model_id=None):
            # [FUNCTION DOC] HELPER: Replacement data fetch returning an empty sample list and two
            # empty extras.
            return [], None, None

        executor._get_rollout_data = _get_rollout_data

        await executor.get(0, trainer_model_id="b")

        assert seen == [{"dp_size": 4}]

    async def test_a_policy_asks_for_data_against_its_own_weight_version(self, monkeypatch):
        # [FUNCTION DOC] Policies a (version 3) and b (version 7, parallel size 4). Calls get for b
        # and asserts exactly one rollout input was produced, carrying weight version 7 and
        # trainer model id b.
        """The rollout function stamps its samples with the version it is told, so the wrong one mislabels a batch."""
        executor = _make_executor()
        executor.set_weight_version(3, trainer_model_id="a")
        executor.set_weight_version(7, trainer_model_id="b")
        executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id="b")
        received = _record_generate_inputs(executor, monkeypatch)
        _record_postprocess_configs(monkeypatch)
        _record_logged_model_ids(monkeypatch)

        await executor.get(0, trainer_model_id="b")

        [rollout_input] = received
        assert (rollout_input.weight_version, rollout_input.trainer_model_id) == (7, "b")

    async def test_the_postprocess_step_uses_the_same_parallel_config_as_the_split(self, monkeypatch):
        # [FUNCTION DOC] Policies a (size 8) and b (size 4); records the configs seen by the
        # post-processing and by the split. Calls get for b and asserts both saw only b's config.
        """Postprocessing and splitting disagreeing on dp size would group samples one way and shard them another."""
        executor = _make_executor()
        executor.set_train_parallel_config({"dp_size": 8}, trainer_model_id="a")
        executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id="b")
        _record_generate_inputs(executor, monkeypatch)
        postprocessed = _record_postprocess_configs(monkeypatch)
        split: list[dict[str, Any]] = []
        monkeypatch.setattr(
            rollout_executor_module, "split_train_data_by_dp", lambda args, data, config: split.append(config)
        )
        _record_logged_model_ids(monkeypatch)

        await executor.get(0, trainer_model_id="b")

        assert postprocessed == split == [{"dp_size": 4}]

    async def test_a_policy_without_a_parallel_config_fails_loudly(self, monkeypatch):
        # [FUNCTION DOC] Only policy b has a config; calling get for policy a raises a key error
        # whose message names a.
        """A policy whose trainer never published its layout must name itself instead of sharding by another's."""
        executor = _make_executor()
        executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id="b")
        _record_generate_inputs(executor, monkeypatch)
        _record_postprocess_configs(monkeypatch)
        _record_logged_model_ids(monkeypatch)

        with pytest.raises(KeyError, match="'a'"):
            await executor.get(0, trainer_model_id="a")


class TestRolloutTimerNaming:
    # [CLASS DOC] Tests how the executor names its rollout timer and logged metrics per policy.
    async def test_two_policies_may_be_generating_at_the_same_time(self, monkeypatch):
        # [FUNCTION DOC] Two policies call get concurrently; each data fetch waits (five second
        # limit) until both have arrived, proving the calls overlap. Asserts the timer singleton
        # recorded timers named a/rollout and b/rollout and that the logging function saw model ids
        # a and b. The test sets the parallel config of both policies twice (dp size 4, then 1).
        """The rollout timer is a process singleton that refuses a second start under the same name."""
        logged = _record_logged_model_ids(monkeypatch)
        executor = _make_executor()
        executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id="a")
        executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id="b")
        both_arrived = asyncio.Event()
        arrivals = 0

        async def _get_rollout_data(rollout_id, trainer_model_id=None):
            # [FUNCTION DOC] HELPER: Counts an arrival; the second arrival sets the shared event;
            # waits (up to five seconds) for the event so both callers overlap; returns empty data.
            nonlocal arrivals
            arrivals += 1
            if arrivals == 2:
                both_arrived.set()
            await asyncio.wait_for(both_arrived.wait(), timeout=5)
            return [], None, None

        executor._get_rollout_data = _get_rollout_data
        for trainer_model_id in ("a", "b"):
            executor.set_train_parallel_config({"dp_size": 1}, trainer_model_id=trainer_model_id)

        await asyncio.wait_for(
            asyncio.gather(executor.get(0, trainer_model_id="a"), executor.get(0, trainer_model_id="b")), timeout=5
        )

        assert sorted(Timer().log_dict()) == ["a/rollout", "b/rollout"]
        assert sorted(logged) == ["a", "b"]

    async def test_a_single_policy_run_keeps_the_timer_and_metric_names_it_had(self, monkeypatch):
        # [FUNCTION DOC] Single policy (no model id), data fetch returning empty data. After get,
        # asserts the only timer is the unprefixed rollout and the logging function got a none
        # model id.
        """Every existing dashboard query is written against the unprefixed names."""
        logged = _record_logged_model_ids(monkeypatch)
        executor = _make_executor()
        executor.set_train_parallel_config({"dp_size": 4})

        async def _get_rollout_data(rollout_id, trainer_model_id=None):
            # [FUNCTION DOC] HELPER: Replacement data fetch returning an empty sample list and two
            # empty extras.
            return [], None, None

        executor._get_rollout_data = _get_rollout_data
        executor.set_train_parallel_config({"dp_size": 1})

        await executor.get(0)

        assert list(Timer().log_dict()) == ["rollout"]
        assert logged == [None]


class TestWeightVersionWatchdog:
    # [CLASS DOC] Tests the executor's watchdog that fails when too many rollouts happen for a
    # policy without a weight version publish; the real watchdog function is restored for these
    # tests, and the counter must be per policy.
    @pytest.fixture(autouse=True)
    def _real_watchdog(self, monkeypatch):
        # [FUNCTION DOC] HELPER: Autouse fixture that puts the real weight-version-published
        # assertion back into the executor module (undoing the quiet no-op).
        monkeypatch.setattr(
            rollout_executor_module, "assert_weight_version_is_published", assert_weight_version_is_published
        )

    @staticmethod
    def _make_ready_executor(*model_ids: str | None) -> RolloutExecutor:
        # [FUNCTION DOC] HELPER: Builds a test executor and, for each given model id, sets weight
        # version 1 and a data parallel size 4 config; replaces the data fetch with one returning
        # empty data. Returns the executor.
        executor = _make_executor()
        for model_id in model_ids:
            executor.set_weight_version(1, trainer_model_id=model_id)
            executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id=model_id)

        async def _get_rollout_data(rollout_id, trainer_model_id=None):
            # [FUNCTION DOC] HELPER: Replacement data fetch returning an empty sample list and two
            # empty extras.
            return [], None, None

        executor._get_rollout_data = _get_rollout_data
        return executor

    async def test_one_round_of_more_policies_than_the_threshold_is_not_a_stall(self, monkeypatch):
        # [FUNCTION DOC] Four policies each fetch one rollout in the same round with no publishes;
        # the test passes if no error is raised (the counter is per policy, not shared).
        """A shared counter turns the fourth policy's first rollout into a false 'nobody published' failure."""
        _record_logged_model_ids(monkeypatch)
        model_ids = ["a", "b", "c", "d"]
        executor = self._make_ready_executor(*model_ids)

        for model_id in model_ids:
            await executor.get(0, trainer_model_id=model_id)

    async def test_a_publishing_policy_does_not_clear_a_frozen_one(self, monkeypatch):
        # [FUNCTION DOC] Policies a and b; computes the maximum rollouts allowed without a publish.
        # For that many rounds both fetch data while only a publishes a new weight version each
        # round. Then fetching once more for b (which never published) raises an assertion error
        # whose message contains "without anyone calling".
        """Otherwise a policy whose weights never move again is covered by its neighbour's updates."""
        _record_logged_model_ids(monkeypatch)
        executor = self._make_ready_executor("a", "b")
        max_rollouts = max_rollouts_without_published_weight_version(executor.args)

        for rollout_id in range(max_rollouts):
            await executor.get(rollout_id, trainer_model_id="b")
            await executor.get(rollout_id, trainer_model_id="a")
            executor.set_weight_version(rollout_id + 2, trainer_model_id="a")

        with pytest.raises(AssertionError, match="without anyone calling"):
            await executor.get(max_rollouts, trainer_model_id="b")
