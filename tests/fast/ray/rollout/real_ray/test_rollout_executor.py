# [MODULE DOC]
# Tests of the rollout executor actor class (run in ray local mode with low-level collaborators
# patched out). Covers process setup (http client, router and session server resolution), rollout
# function construction and loading (including debug replay), generation (dp split, weight version
# checks, custom conversion and reward hooks, delayed split), checkpoint save and load fan-out,
# evaluation, weight versions, debug recorded data replay and injection, the legacy rollout protocol,
# lifecycle disposal, epoch length, and checkpointing without a train rollout function. Fixtures patch
# module-level helpers of the executor module; test doubles record calls.
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.debug_data import save_debug_rollout_data
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.checkpoint_eval import CheckpointEvalFn
from miles.utils.types import WeightVersionSpan, WeightVersionsPerCall
from miles.utils.weight_version import max_rollouts_without_published_weight_version


@pytest.fixture
def http_client_calls(monkeypatch) -> list[str]:
    # [FUNCTION DOC] HELPER: Fixture. Replaces the executor module's http client initializer with a
    # function that appends a marker string to a list, and returns that list.
    import miles.ray.rollout.rollout_executor as rexec

    recorded: list[str] = []
    monkeypatch.setattr(rexec, "init_http_client", lambda args: recorded.append("init_http_client"))
    return recorded


@pytest.fixture
def own_args_resolutions(monkeypatch) -> list[tuple[str, object]]:
    # [FUNCTION DOC] HELPER: Fixture. Replaces the executor module's router address resolver and session
    # server readiness waiter with async recorders that append (name, args) to a list and return an
    # empty mapping. Returns the list.
    import miles.ray.rollout.rollout_executor as rexec

    recorded: list[tuple[str, object]] = []

    async def _record_router(args, **kwargs):
        recorded.append(("resolve_router_addrs", args))
        return {}

    async def _record_session(args, **kwargs):
        recorded.append(("wait_session_server_ready", args))
        return {}

    monkeypatch.setattr(rexec, "resolve_router_addrs", _record_router)
    monkeypatch.setattr(rexec, "wait_session_server_ready", _record_session)
    return recorded


@pytest.fixture
def patch_low_level(monkeypatch, http_client_calls, own_args_resolutions):
    # [FUNCTION DOC] HELPER: Fixture. Replaces the executor module's logger configuration, tracking
    # initialization, function loaders, rollout and eval logging, and debug data saving with no-op
    # functions (loaders return a no-op callable). Also depends on the http client and resolution
    # recorders so they are active. Returns nothing.
    import miles.ray.rollout.rollout_executor as rexec

    monkeypatch.setattr(rexec, "configure_logger", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "init_tracking", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "load_function", lambda path: lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "load_rollout_function", lambda input, path: lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "log_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "log_eval_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "save_debug_rollout_data", lambda *a, **kw: None)


class _NeverUsedProvider:
    # [CLASS DOC] HELPER: Address provider stub that fails if asked for an address.
    async def get_addrs(self, worker_name: str):
        # [FUNCTION DOC] HELPER: Always raises an assertion error saying the stubbed resolution must not
        # query the provider.
        raise AssertionError("the stubbed resolution must not ask the provider for an address")


async def _make_executor(args):
    # [FUNCTION DOC] HELPER: Builds a rollout executor with one never-used router provider, no session
    # server provider and a never-used inference controller provider, awaits its init, and returns it.
    executor = RolloutExecutor(
        args=args,
        router_providers=[_NeverUsedProvider()],
        session_server_provider=None,
        inference_controller_provider=_NeverUsedProvider(),
    )
    await executor.init()
    return executor


def _make_test_args(**overrides):
    # [FUNCTION DOC] HELPER: Builds test args with router ip 127.0.0.1, port 30000, and wandb,
    # tensorboard, mlflow and distributed post logging all off; overrides pass through.
    return make_args(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        use_wandb=False,
        use_tensorboard=False,
        use_mlflow=False,
        use_distributed_post=False,
        **overrides,
    )


@pytest.mark.asyncio
class TestProcessSetup:
    # [CLASS DOC] Tests of executor initialization: creation of the http client and resolution of the
    # router and session servers on the executor's own args, in normal and debug-train-only modes.
    async def test_initializes_the_http_client(self, ray_local_mode, patch_low_level, http_client_calls):
        # [FUNCTION DOC] Builds an executor with default test args and asserts the http client
        # initializer was called exactly once.
        """The rollout functions issue their HTTP from this actor, so the client is created here."""
        await _make_executor(_make_test_args())

        assert http_client_calls == ["init_http_client"]

    async def test_initializes_the_http_client_in_debug_train_only(
        self, ray_local_mode, patch_low_level, http_client_calls
    ):
        # [FUNCTION DOC] With the debug train only flag set, the http client initializer is still called
        # exactly once (it decides by itself whether there is anything to talk to).
        """A snapshot-eval fleet may exist in this mode; init_http_client itself decides whether there is anything to talk to."""
        args = _make_test_args()
        args.debug_train_only = True

        await _make_executor(args)

        assert http_client_calls == ["init_http_client"]

    async def test_it_resolves_the_router_and_session_servers_on_its_own_args(
        self, ray_local_mode, patch_low_level, own_args_resolutions
    ):
        # [FUNCTION DOC] Builds an executor and asserts the recorded resolutions are, in order, the
        # router address resolution and the session server readiness wait, each given the executor's own
        # args object (not a copy or another object).
        """The platform builds the executor before the driver resolves anything, so it must resolve for itself."""
        args = _make_test_args()

        executor = await _make_executor(args)

        assert [name for name, _ in own_args_resolutions] == ["resolve_router_addrs", "wait_session_server_ready"]
        assert all(seen is executor.args for _, seen in own_args_resolutions)

    async def test_it_resolves_nothing_in_debug_train_only(
        self, ray_local_mode, patch_low_level, own_args_resolutions
    ):
        # [FUNCTION DOC] With the debug train only flag set, building an executor must record no router
        # or session server resolution at all.
        """No engines and no session servers exist in this mode, so there is nothing to wait for."""
        args = _make_test_args()
        args.debug_train_only = True

        await _make_executor(args)

        assert own_args_resolutions == []


@pytest.mark.asyncio
class TestRolloutFunctionConstruction:
    # [CLASS DOC] Tests of when the executor builds its train and eval rollout functions during
    # initialization, in debug-train-only mode with and without recorded data replay.
    async def test_debug_rollout_replay_skips_class_based_rollout_construction(
        self,
        ray_local_mode,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        # [FUNCTION DOC] Sets debug train only, a replay path template for recorded rollout data, no
        # rollout gpus, and removes the legacy-protocol environment variable. The rollout function
        # loader is replaced by one that fails the test if called. After building the executor asserts
        # both the train and eval rollout functions are none.
        """Replaying dumped rollout data must not build the rollout functions."""
        import miles.ray.rollout.rollout_executor as rexec

        args = _make_test_args()
        args.debug_train_only = True
        args.load_debug_rollout_data = str(tmp_path / "rollout-{rollout_id}.pt")
        args.rollout_num_gpus = None
        monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)

        def fail_if_loaded(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Fails the running test immediately with a message that replay must
            # not construct rollout functions; accepts and ignores any arguments.
            pytest.fail("debug rollout replay must not construct rollout functions")

        monkeypatch.setattr(rexec, "load_rollout_function", fail_if_loaded)

        executor = await _make_executor(args)

        assert executor.generate_rollout is None
        assert executor.eval_generate_rollout is None

    async def test_debug_train_only_without_replay_constructs_rollout_function(
        self,
        ray_local_mode,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        # [FUNCTION DOC] Sets debug train only without a replay path, removes the legacy-protocol
        # environment variable, and replaces the rollout function loader with a recorder. After
        # building the executor asserts the loader saw the configured rollout function path then the
        # eval function path, and that both rollout functions are set.
        """Without replay data, debug_train_only still builds both rollout functions."""
        import miles.ray.rollout.rollout_executor as rexec

        args = _make_test_args()
        args.debug_train_only = True
        monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)
        loaded_paths: list[str] = []

        def record_load(input, path):
            # [FUNCTION DOC] HELPER: Appends the requested path to the shared list and returns a no-op
            # callable standing in for the rollout function.
            loaded_paths.append(path)
            return lambda *args, **kwargs: None

        monkeypatch.setattr(rexec, "load_rollout_function", record_load)

        executor = await _make_executor(args)

        assert loaded_paths == [args.rollout_function_path, args.eval_function_path]
        assert executor.generate_rollout is not None
        assert executor.eval_generate_rollout is not None


@pytest.mark.asyncio
class TestGenerate:
    # [CLASS DOC] Tests of the executor's get method (generation): input given to the rollout
    # function, the split of samples per data-parallel rank, weight version validation, and independence
    # from inference-side objects. Executors use a train parallel config set by the test and a rollout
    # function replaced by a test double.
    async def test_invokes_rollout_fn_with_correct_input_and_returns_dp_split(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Global batch size 8, data-parallel size 2. The rollout function double captures
        # its input and returns two groups of four samples with one metric. After get for rollout 42
        # asserts: exactly one input captured, of the train input type, with rollout id 42; the result
        # holds two data references (one per rank); and each fetched partition has tokens, rewards and
        # loss masks, with four token rows (eight samples split evenly).
        """generate passes a train input and returns the samples split per dp rank."""
        args = _make_test_args()
        args.global_batch_size = 8

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})

        captured: list = []

        def fake_rollout_fn(input):
            # [FUNCTION DOC] HELPER: Records the input and returns a train output with two groups of
            # four samples and a single metric.
            captured.append(input)
            return RolloutFnTrainOutput(
                samples=[make_samples_grouped(n_groups=2, group_size=4)],
                metrics={"my_metric": 1.23},
            )

        executor.generate_rollout = fake_rollout_fn

        result = await executor.get(rollout_id=42)

        assert len(captured) == 1
        assert isinstance(captured[0], RolloutFnTrainInput)
        assert captured[0].rollout_id == 42
        data_refs = result.data_ref
        assert len(data_refs) == 2
        partitions = ray.get([ref.payload for ref in data_refs])
        for partition in partitions:
            assert "tokens" in partition
            assert "rewards" in partition
            assert "loss_masks" in partition
            assert len(partition["tokens"]) == 4

    async def test_rejects_samples_generated_under_the_default_weight_version(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] One generated sample is marked as generated under the default (never updated)
        # weight version. Asserts get raises an assertion error mentioning that the weights were never
        # updated, so such a batch cannot reach training.
        """A batch carrying the sglang never-updated version must fail get(), not reach training."""
        args = _make_test_args()
        args.global_batch_size = 8

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})

        samples = make_samples_grouped(n_groups=2, group_size=4)
        samples[0].weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan(version="default", abs_start=0, abs_end=1)])
        ]

        executor.generate_rollout = lambda input: RolloutFnTrainOutput(samples=[samples], metrics=None)

        with pytest.raises(AssertionError, match="never updated"):
            await executor.get(rollout_id=42)

    async def test_a_frozen_weight_version_eventually_fails_the_run(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Never publishes a weight version and calls get for rollouts one more than the
        # maximum number allowed without a published version. Asserts an assertion error mentioning the
        # set weight version call is raised along the way.
        """A driver that never forwards the version must fail loudly, not silently disable staleness filtering."""
        args = _make_test_args()
        args.global_batch_size = 8

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=2, group_size=4)], metrics=None
        )

        with pytest.raises(AssertionError, match="set_weight_version"):
            for rollout_id in range(max_rollouts_without_published_weight_version(args) + 1):
                await executor.get(rollout_id=rollout_id)

    async def test_publishing_a_version_every_step_keeps_the_run_alive(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Publishes the rollout id as the weight version before every get, for three times
        # the allowed number of unpublished rollouts; no assertion error may be raised.
        """The supported driver publishes after each update, which must never trip the staleness assert."""
        args = _make_test_args()
        args.global_batch_size = 8

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=2, group_size=4)], metrics=None
        )

        for rollout_id in range(3 * max_rollouts_without_published_weight_version(args)):
            executor.set_weight_version(rollout_id)
            await executor.get(rollout_id=rollout_id)

    async def test_does_not_touch_the_inference_side(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Global batch size 4, data-parallel size 1. After a get for rollout 7 asserts the
        # executor has neither a servers attribute nor a health monitors attribute.
        """The controller is a driver-side object the executor cannot reach, so generate must not need it."""
        args = _make_test_args()
        args.global_batch_size = 4

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 1})
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={}
        )

        await executor.get(rollout_id=7)

        assert not hasattr(executor, "servers")
        assert not hasattr(executor, "_health_monitors")


class _RecordingRolloutFn(BaseRolloutFn):
    # [CLASS DOC] HELPER: Stateful rollout function double. Records save and load calls as triples of
    # its name, the kind of call and the rollout id into a shared list. Calling it as a rollout function
    # is not supported.
    def __init__(self, name: str, log: list[tuple[str, str, object]]) -> None:
        # [FUNCTION DOC] HELPER: Stores the name and the shared log.
        self._name = name
        self._log = log

    def __call__(self, input):
        # [FUNCTION DOC] HELPER: Always raises an assertion error saying the checkpointing tests do not
        # exercise generation.
        raise AssertionError("not exercised by the checkpointing tests")

    def save(self, rollout_id: int) -> None:
        # [FUNCTION DOC] HELPER: Appends the triple of name, save and rollout id to the shared log.
        self._log.append((self._name, "save", rollout_id))

    def load(self, rollout_id: int | None) -> None:
        # [FUNCTION DOC] HELPER: Appends the triple of name, load and rollout id (possibly none) to the
        # shared log.
        self._log.append((self._name, "load", rollout_id))


class _RecordingEventLoggerCheckpoint:
    # [CLASS DOC] HELPER: Event logger checkpoint double that records restore calls (args) and snapshot
    # calls (args and rollout id).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with empty restore and snapshot records.
        self.restored: list = []
        self.snapshots: list[tuple] = []

    def restore(self, args) -> None:
        # [FUNCTION DOC] HELPER: Records the args of a restore call.
        self.restored.append(args)

    def snapshot(self, args, rollout_id) -> None:
        # [FUNCTION DOC] HELPER: Records the pair of args and rollout id of a snapshot call.
        self.snapshots.append((args, rollout_id))


@pytest.mark.asyncio
class TestCheckpointing:
    # [CLASS DOC] Tests of the executor's save and load fan-out to the rollout functions (train and
    # eval), the data source and the event log checkpoint, under the new and legacy rollout protocols.
    # Tests replace the event logger checkpoint with a mock or recorder and use recording rollout
    # function doubles and a mock data source.
    async def test_save_and_load_reach_every_distinct_rollout_function(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        # [FUNCTION DOC] Non-legacy protocol, no global dataset. Installs distinct recording train and
        # eval rollout functions and a mock data source, then saves and loads rollout 7. Asserts the
        # recorded order is train save, eval save, train load, eval load.
        """Both the train and the eval instance carry state, so each is written and read back in turn."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())
        args = _make_test_args(rollout_global_dataset=False)

        executor = await _make_executor(args)
        executor.use_legacy_rollout_v1 = False
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _RecordingRolloutFn("train", calls)
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)
        executor.data_source = MagicMock()

        executor.save(rollout_id=7)
        executor.load(rollout_id=7)

        assert calls == [
            ("train", "save", 7),
            ("eval", "save", 7),
            ("train", "load", 7),
            ("eval", "load", 7),
        ]

    async def test_save_forwards_to_the_data_source_for_a_global_dataset(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        # [FUNCTION DOC] Same setup but with a global dataset; saves rollout 5. Asserts the data source
        # save was called once with 5 and the train rollout function recorded a save for 5.
        """With a global dataset both the data source and the rollout functions are checkpointed."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())
        args = _make_test_args(rollout_global_dataset=True)

        executor = await _make_executor(args)
        executor.use_legacy_rollout_v1 = False
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _RecordingRolloutFn("train", calls)
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)
        executor.data_source = MagicMock()

        executor.save(rollout_id=5)

        executor.data_source.save.assert_called_once_with(5)
        assert ("train", "save", 5) in calls

    async def test_save_forwards_to_the_data_source_without_a_global_dataset(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        # [FUNCTION DOC] No global dataset; saves and loads rollout 3. Asserts the data source save and
        # load were each called once with 3 and the train rollout function recorded a save for 3.
        """A custom data source is saved as unconditionally as it is loaded, so its state can be restored."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())
        args = _make_test_args(rollout_global_dataset=False)

        executor = await _make_executor(args)
        executor.use_legacy_rollout_v1 = False
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _RecordingRolloutFn("train", calls)
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)
        executor.data_source = MagicMock()

        executor.save(rollout_id=3)
        executor.load(rollout_id=3)

        executor.data_source.save.assert_called_once_with(3)
        executor.data_source.load.assert_called_once_with(3)
        assert ("train", "save", 3) in calls

    async def test_legacy_function_path_does_not_get_save_load(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        # [FUNCTION DOC] Legacy protocol with plain no-op callables as rollout functions (which have no
        # save or load, so any checkpoint attempt on them would fail). Saves and loads rollout 1 and
        # asserts only the data source load was made, once with 1.
        """Without the experimental flag the rollout functions are bare callables, so they are not checkpointed."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())
        args = _make_test_args(rollout_global_dataset=False)

        executor = await _make_executor(args)
        executor.use_legacy_rollout_v1 = True
        executor.generate_rollout = lambda *a, **kw: None
        executor.eval_generate_rollout = lambda *a, **kw: None
        executor.data_source = MagicMock()

        executor.save(rollout_id=1)
        executor.load(rollout_id=1)

        executor.data_source.load.assert_called_once_with(1)

    async def test_save_snapshots_the_event_log_for_the_rollout_id(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        # [FUNCTION DOC] Installs a recording event logger checkpoint, a train recording function, no
        # eval function and a mock data source, then saves rollout 9. Asserts the recorder saw exactly
        # one snapshot with the executor's args and rollout id 9.
        """The audit event log is checkpointed alongside the rollout state it explains."""
        import miles.ray.rollout.rollout_executor as rexec

        recorder = _RecordingEventLoggerCheckpoint()
        monkeypatch.setattr(rexec, "event_logger_checkpoint", recorder)
        args = _make_test_args(rollout_global_dataset=False)

        executor = await _make_executor(args)
        executor.use_legacy_rollout_v1 = False
        executor.generate_rollout = _RecordingRolloutFn("train", [])
        executor.eval_generate_rollout = None
        executor.data_source = MagicMock()

        executor.save(rollout_id=9)

        assert recorder.snapshots == [(args, 9)]


@pytest.mark.asyncio
class TestEval:
    # [CLASS DOC] Tests of the executor's eval method.
    async def test_invokes_eval_fn_with_eval_input(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Installs an eval function double that records its input and returns an eval
        # output with one dataset's rewards. After eval for rollout 10 asserts exactly one input was
        # recorded, of the eval input type, with rollout id 10.
        """eval passes an eval input carrying the rollout id."""
        executor = await _make_executor(_make_test_args())

        captured: list = []

        def fake_eval_fn(input):
            # [FUNCTION DOC] HELPER: Records the input and returns an eval output with one dataset's
            # rewards and empty metrics.
            captured.append(input)
            return RolloutFnEvalOutput(data={"my_dataset": {"rewards": [0.5, 1.0]}}, metrics={})

        executor.eval_generate_rollout = fake_eval_fn

        await executor.eval(rollout_id=10)

        assert len(captured) == 1
        assert isinstance(captured[0], RolloutFnEvalInput)
        assert captured[0].rollout_id == 10

    async def test_skipped_in_debug_train_only_mode(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] With debug train only set and an eval function double that records calls,
        # eval for rollout 10 must not call the double at all.
        """debug_train_only short-circuits eval before the rollout function runs."""
        args = _make_test_args()
        args.debug_train_only = True

        executor = await _make_executor(args)

        called: list = []
        executor.eval_generate_rollout = lambda inp: called.append(inp)

        await executor.eval(rollout_id=10)

        assert called == []


class _NamedRolloutFn(BaseRolloutFn):
    # [CLASS DOC] HELPER: Rollout function double that remembers the path it was loaded from.
    def __init__(self, path: str) -> None:
        # [FUNCTION DOC] HELPER: Stores the path.
        self.path = path

    def __call__(self, input):
        # [FUNCTION DOC] HELPER: Always raises an assertion error saying the loading tests do not
        # exercise generation.
        raise AssertionError("not exercised by the loading tests")


@pytest.mark.asyncio
class TestRolloutFunctionLoading:
    # [CLASS DOC] Tests of how the train and eval rollout functions are loaded: one shared instance for
    # equal paths, separate instances otherwise. The loader is replaced to return a named double.
    async def test_matching_train_and_eval_paths_share_one_rollout_instance(
        self, ray_local_mode, patch_low_level, monkeypatch
    ):
        # [FUNCTION DOC] Configures the same rollout function path for train and eval and asserts the
        # eval rollout function is the very same object as the train one.
        """One configured path means one stateful object, so train and eval share its state."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "load_rollout_function", lambda input, path: _NamedRolloutFn(path))
        args = _make_test_args(rollout_function_path="pkg.same_fn", eval_function_path="pkg.same_fn")

        executor = await _make_executor(args)

        assert executor.eval_generate_rollout is executor.generate_rollout

    async def test_distinct_train_and_eval_paths_get_their_own_instance(
        self, ray_local_mode, patch_low_level, monkeypatch
    ):
        # [FUNCTION DOC] Configures different train and eval paths and asserts each rollout function
        # carries its own configured path.
        """Two configured paths must each be loaded, so eval never silently runs the train function."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "load_rollout_function", lambda input, path: _NamedRolloutFn(path))
        args = _make_test_args(rollout_function_path="pkg.train_fn", eval_function_path="pkg.eval_fn")

        executor = await _make_executor(args)

        assert executor.generate_rollout.path == "pkg.train_fn"
        assert executor.eval_generate_rollout.path == "pkg.eval_fn"


@pytest.mark.asyncio
class TestCustomHooks:
    # [CLASS DOC] Tests of user-configured hooks applied by get: the conversion of samples to train data
    # and the reward post-processing. The function loader is replaced so the configured hook path
    # resolves to a test hook and any other path to a no-op.
    async def test_get_converts_the_batch_with_the_configured_conversion_hook(
        self, ray_local_mode, patch_low_level, monkeypatch
    ):
        # [FUNCTION DOC] Global batch size 4, data-parallel size 2, one group of four generated samples.
        # The conversion hook records its samples and returns a mapping with two token rows and sample
        # indices 70 and 71. After get for rollout 1 asserts the hook saw samples with indices 0 to 3,
        # the result sample indices are [70, 71], and the two ranks receive one token row each.
        """The configured conversion hook receives the generated samples and its output is what ships."""
        import miles.ray.rollout.rollout_executor as rexec

        seen_samples: list = []

        def conversion_hook(args, samples):
            # [FUNCTION DOC] HELPER: Records the samples and returns a train data mapping with token
            # rows [1, 2] and [3, 4] and sample indices 70 and 71.
            seen_samples.append(samples)
            return dict(tokens=[[1, 2], [3, 4]], sample_indices=[70, 71])

        monkeypatch.setattr(
            rexec,
            "load_function",
            lambda path: conversion_hook if path == "pkg.convert" else (lambda *a, **kw: None),
        )
        args = _make_test_args(global_batch_size=4, custom_convert_samples_to_train_data_path="pkg.convert")

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={}
        )

        result = await executor.get(rollout_id=1)

        assert [sample.index for sample in seen_samples[0]] == [0, 1, 2, 3]
        assert result.sample_indices == [70, 71]
        partitions = ray.get([box.payload for box in result.data_ref])
        assert [partition["tokens"] for partition in partitions] == [[[1, 2]], [[3, 4]]]

    async def test_get_scores_the_batch_with_the_configured_reward_hook(
        self, ray_local_mode, patch_low_level, monkeypatch
    ):
        # [FUNCTION DOC] Global batch size 4, data-parallel size 2, one group of four samples. The reward
        # hook records its samples and returns a pair of lists, the first all 1.0 and the second all 9.0.
        # After get for rollout 1 asserts the hook saw samples with indices 0 to 3 and every rank's
        # shipped rewards are 9.0 (the second list is what is used).
        """The configured reward post-process hook replaces the built-in reward math."""
        import miles.ray.rollout.rollout_executor as rexec

        seen_samples: list = []

        def reward_hook(args, samples):
            # [FUNCTION DOC] HELPER: Records the samples and returns two lists of per-sample values,
            # all 1.0 and all 9.0.
            seen_samples.append(samples)
            return [1.0] * len(samples), [9.0] * len(samples)

        monkeypatch.setattr(
            rexec,
            "load_function",
            lambda path: reward_hook if path == "pkg.reward" else (lambda *a, **kw: None),
        )
        args = _make_test_args(global_batch_size=4, custom_reward_post_process_path="pkg.reward")

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={}
        )

        result = await executor.get(rollout_id=1)

        assert [sample.index for sample in seen_samples[0]] == [0, 1, 2, 3]
        partitions = ray.get([box.payload for box in result.data_ref])
        assert [partition["rewards"] for partition in partitions] == [[9.0, 9.0], [9.0, 9.0]]


@pytest.mark.asyncio
class TestWeightVersion:
    # [CLASS DOC] Tests of the executor's set weight version method and how the latest version reaches
    # the rollout function input, including monotonicity rules (strict normally, a warning with
    # independent data parallel fault tolerance).
    async def test_get_threads_the_latest_weight_version_into_the_train_input(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Global batch size 4, independent dp off, data-parallel size 1. Publishes weight
        # versions 3 then 7, then runs get for rollout 1 with a rollout function double that captures
        # its input. Asserts the captured input carries weight version 7 (the latest).
        """The rollout function is told which engine weight version this batch is generated under."""
        args = _make_test_args(global_batch_size=4, indep_dp=False)

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 1})
        captured: list = []

        def fake_rollout_fn(input):
            # [FUNCTION DOC] HELPER: Records the input and returns a train output with one group of
            # four samples and no metrics.
            captured.append(input)
            return RolloutFnTrainOutput(samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={})

        executor.generate_rollout = fake_rollout_fn
        executor.set_weight_version(3)
        executor.set_weight_version(7)

        await executor.get(rollout_id=1)

        assert captured[0].weight_version == 7

    async def test_a_decreasing_weight_version_is_rejected(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Independent dp off. After publishing version 7, publishing 3 must raise an
        # assertion error saying the version went backwards, and the stored version for the default
        # (no) model id must remain 7.
        """Weight versions only move forward, so a lower one signals a broken update path."""
        executor = await _make_executor(_make_test_args(indep_dp=False))
        executor.set_weight_version(7)

        with pytest.raises(AssertionError, match="went backwards"):
            executor.set_weight_version(3)

        assert executor._weight_versions_of_model_id[None] == 7

    async def test_independent_dp_accepts_a_decreasing_weight_version_with_a_warning(
        self, ray_local_mode, patch_low_level, caplog
    ):
        # [FUNCTION DOC] Independent dp on. After publishing 7, publishing 3 must not raise; the stored
        # version for the default model id becomes 3 and a warning record containing went backwards is
        # logged.
        """Independent-DP fault tolerance may rewind a replica, so the rewind warns instead of failing."""
        executor = await _make_executor(_make_test_args(indep_dp=True))
        executor.set_weight_version(7)

        with caplog.at_level(logging.WARNING):
            executor.set_weight_version(3)

        assert executor._weight_versions_of_model_id[None] == 3
        assert any("went backwards" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
class TestDelayedDpSplit:
    # [CLASS DOC] Tests of the mode where the split by data-parallel rank is delayed to the training
    # side.
    async def test_get_stores_one_unsplit_batch(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Delay flag on, global batch size 4, data-parallel size 2, one group of four
        # samples. After get for rollout 1 asserts the result data reference is a single reference (not
        # a list), the stored payload has four token rows, and the result sample indices equal the
        # stored sample indices, which are [0, 1, 2, 3].
        """With the split delayed to the training side, one whole-batch reference is published."""
        args = _make_test_args(global_batch_size=4, delay_split_train_data_by_dp=True)

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={}
        )

        result = await executor.get(rollout_id=1)

        assert not isinstance(result.data_ref, list)
        stored = ray.get(result.data_ref.payload)
        assert len(stored["tokens"]) == 4
        assert result.sample_indices == stored["sample_indices"] == [0, 1, 2, 3]


@pytest.mark.asyncio
class TestDebugRolloutData:
    # [CLASS DOC] Tests of the executor's internal rollout data retrieval with recorded data: replay of a
    # recording instead of generating, and CI injection that verifies generated data against a recording
    # and then substitutes the recording. Recordings are written with the debug data save function.
    async def test_recorded_data_is_replayed_instead_of_generated(self, ray_local_mode, patch_low_level, tmp_path):
        # [FUNCTION DOC] Records rollout 3 (one group of four samples, with metadata) to a temporary
        # path, builds an executor configured to load it, and installs a rollout function that fails if
        # called. Asserts the internal data retrieval for rollout 3 returns samples with indices 0 to 3,
        # the recorded metadata verbatim, and no metrics.
        """Replaying a recording must not call the rollout function, and its metadata comes back verbatim."""
        template = str(tmp_path / "rollout_{rollout_id}.pt")
        save_debug_rollout_data(
            make_args(save_debug_rollout_data=template),
            make_samples_grouped(n_groups=1, group_size=4),
            rollout_id=3,
            evaluation=False,
            metadata={"dynamic_global_batch_size": 4},
        )
        args = _make_test_args(load_debug_rollout_data=template)

        executor = await _make_executor(args)

        def unreachable(input):
            # [FUNCTION DOC] HELPER: Always raises an assertion error saying the rollout function must
            # not run when a recording is replayed.
            raise AssertionError("the rollout function must not run when a recording is replayed")

        executor.generate_rollout = unreachable

        data, metadata, metrics = await executor._get_rollout_data(rollout_id=3)

        assert [sample.index for sample in data] == [0, 1, 2, 3]
        assert metadata == {"dynamic_global_batch_size": 4}
        assert metrics is None

    async def test_injected_recording_replaces_the_verified_generated_batch(
        self, ray_local_mode, patch_low_level, tmp_path
    ):
        # [FUNCTION DOC] Records rollout 2 with every sample's reward set to 5.0 and source metadata,
        # and configures injection from it starting at rollout 2 (global batch size 4, dp size 1). First
        # the rollout function returns a batch whose last token is altered; the internal retrieval for
        # rollout 2 must raise an assertion error saying generated responses must match. Then it returns
        # an unaltered batch with a generated metric; the retrieval must return the recorded samples
        # (rewards all 5.0), the recorded metadata, and no metrics (generated metrics are dropped).
        """CI injection swaps in the recording it verified against, and drops the generated metrics with it."""
        template = str(tmp_path / "inject_{rollout_id}.pt")
        injected = make_samples_grouped(n_groups=1, group_size=4)
        for sample in injected:
            sample.reward = 5.0
        save_debug_rollout_data(
            make_args(save_debug_rollout_data=template),
            injected,
            rollout_id=2,
            evaluation=False,
            metadata={"source": "recording"},
        )
        args = _make_test_args(
            global_batch_size=4,
            ci_inject_rollout_data_path=template,
            ci_inject_rollout_data_start_rollout_id=2,
        )

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 1})

        mismatched = make_samples_grouped(n_groups=1, group_size=4)
        for sample in mismatched:
            sample.tokens[-1] = 99
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[mismatched], metrics={"generated": 1.0}
        )

        with pytest.raises(AssertionError, match="generated responses match"):
            await executor._get_rollout_data(rollout_id=2)

        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={"generated": 1.0}
        )
        data, metadata, metrics = await executor._get_rollout_data(rollout_id=2)

        assert [sample.reward for sample in data] == [5.0, 5.0, 5.0, 5.0]
        assert metadata == {"source": "recording"}
        assert metrics is None


@pytest.mark.asyncio
class TestLegacyRolloutProtocol:
    # [CLASS DOC] Tests of the legacy rollout protocol (enabled by a flag on the executor), where rollout
    # functions are bare callables taking args, rollout id, data source and an evaluation flag.
    async def test_train_generation_keeps_the_legacy_call_signature(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Legacy flag on, simple data source, global batch size 4, dp size 1. The train
        # function double records its four arguments and returns one group of four samples. After get
        # for rollout 11 asserts one call with the executor args, 11, the data source and the
        # evaluation flag false.
        """Without the experimental flag the train fn is still called as (args, rollout_id, data_source)."""
        args = _make_test_args(global_batch_size=4)

        executor = await _make_executor(args)
        executor.use_legacy_rollout_v1 = True
        executor.data_source = SimpleNamespace()
        executor.set_train_parallel_config({"dp_size": 1})
        calls: list[tuple] = []

        def legacy_rollout_fn(passed_args, rollout_id, data_source, evaluation):
            # [FUNCTION DOC] HELPER: Records the four arguments as a tuple and returns one group of
            # four samples.
            calls.append((passed_args, rollout_id, data_source, evaluation))
            return make_samples_grouped(n_groups=1, group_size=4)

        executor.generate_rollout = legacy_rollout_fn

        await executor.get(rollout_id=11)

        assert calls == [(args, 11, executor.data_source, False)]

    async def test_eval_keeps_the_legacy_call_signature_with_the_evaluation_flag(
        self, ray_local_mode, patch_low_level
    ):
        # [FUNCTION DOC] Legacy flag on. The eval function double records its four arguments and returns
        # a mapping from one dataset name to its rewards. After eval for rollout 12 asserts one call with
        # the executor args, 12, the data source and the evaluation flag true.
        """Without the experimental flag eval uses the same legacy protocol, flagged as evaluation."""
        args = _make_test_args()

        executor = await _make_executor(args)
        executor.use_legacy_rollout_v1 = True
        executor.data_source = SimpleNamespace()
        calls: list[tuple] = []

        def legacy_eval_fn(passed_args, rollout_id, data_source, evaluation):
            # [FUNCTION DOC] HELPER: Records the four arguments as a tuple and returns a mapping from
            # one dataset name to a mapping with its rewards.
            calls.append((passed_args, rollout_id, data_source, evaluation))
            return {"my_dataset": {"rewards": [1.0]}}

        executor.eval_generate_rollout = legacy_eval_fn

        await executor.eval(rollout_id=12)

        assert calls == [(args, 12, executor.data_source, True)]


class _RecordingMetricChecker:
    # [CLASS DOC] HELPER: Metric checker double that records the metrics passed to its eval hook and
    # whether it was disposed.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with no evaluated metrics and not disposed.
        self.evaluated: list[dict] = []
        self.disposed = False

    def on_eval(self, metrics: dict) -> None:
        # [FUNCTION DOC] HELPER: Appends the logged eval metrics mapping to the record.
        self.evaluated.append(metrics)

    def dispose(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the checker as disposed.
        self.disposed = True


class _RecordingDisposableRolloutFn:
    # [CLASS DOC] HELPER: Rollout function double with an asynchronous dispose that sets a flag.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts not disposed.
        self.disposed = False

    async def dispose(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the function as disposed.
        self.disposed = True


class _RecordingCheckpointEvalFn(CheckpointEvalFn):
    # [CLASS DOC] HELPER: Checkpoint eval function double that only tracks disposal; evaluating a
    # checkpoint is not supported.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts not disposed (does not call the base initializer).
        self.disposed = False

    async def evaluate_checkpoint(self, checkpoint_dir, input):
        # [FUNCTION DOC] HELPER: Always raises an assertion error saying the lifecycle tests do not
        # exercise checkpoint evaluation.
        raise AssertionError("not exercised by the lifecycle tests")

    async def dispose(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the function as disposed.
        self.disposed = True


@pytest.mark.asyncio
class TestLifecycle:
    # [CLASS DOC] Tests of the executor's eval metric reporting to the metric checker and of teardown
    # (dispose).
    async def test_eval_reports_the_logged_metrics_to_the_metric_checker(
        self, ray_local_mode, patch_low_level, monkeypatch
    ):
        # [FUNCTION DOC] Replaces the eval logging function with one returning a single accuracy metric
        # of 0.75, installs a recording metric checker and an eval function double returning one
        # dataset's rewards. After eval for rollout 4 asserts the checker received exactly that one
        # metrics mapping.
        """The CI accuracy checker is fed exactly the eval metrics that were logged."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(
            rexec, "log_eval_rollout_data", lambda rollout_id, args, data, metrics: {"eval/accuracy": 0.75}
        )

        executor = await _make_executor(_make_test_args())
        checker = _RecordingMetricChecker()
        executor._metric_checker = checker
        executor.eval_generate_rollout = lambda input: RolloutFnEvalOutput(
            data={"my_dataset": {"rewards": [1.0]}}, metrics={}
        )

        await executor.eval(rollout_id=4)

        assert checker.evaluated == [{"eval/accuracy": 0.75}]

    async def test_dispose_releases_every_executor_owned_resource(self, ray_local_mode, patch_low_level, monkeypatch):
        # [FUNCTION DOC] Replaces the event analyzer with one recording the args it is run with, and
        # installs a data source whose close is recorded, a recording metric checker, and disposable
        # eval and train rollout function doubles (non-legacy protocol). After dispose asserts the data
        # source was closed once, analysis ran once with the executor args, and the checker and both
        # rollout functions were disposed.
        """Teardown closes the data source, runs event analysis and disposes the checker and both rollout fns."""
        import miles.ray.rollout.rollout_executor as rexec

        analyzed: list = []
        monkeypatch.setattr(rexec, "event_analyzer", SimpleNamespace(run_analysis_from_args=analyzed.append))
        args = _make_test_args()

        executor = await _make_executor(args)
        closed: list = []
        executor.data_source = SimpleNamespace(close=lambda: closed.append("closed"))
        checker = _RecordingMetricChecker()
        executor._metric_checker = checker
        eval_fn = _RecordingCheckpointEvalFn()
        executor.eval_generate_rollout = eval_fn
        train_fn = _RecordingDisposableRolloutFn()
        executor.generate_rollout = train_fn
        executor.use_legacy_rollout_v1 = False

        await executor.dispose()

        assert closed == ["closed"]
        assert analyzed == [args]
        assert checker.disposed
        assert eval_fn.disposed
        assert train_fn.disposed


@pytest.mark.asyncio
class TestNumRolloutPerEpoch:
    # [CLASS DOC] Tests of the executor's rollouts-per-epoch computation from the data source.
    async def test_counts_only_complete_global_batches(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Global dataset on, rollout batch size 8, a data source with twenty dataset
        # entries; asserts the number of rollouts per epoch is 2 (floor of 20 divided by 8).
        """A trailing partial batch is not a rollout, so the epoch length floors the division."""
        executor = await _make_executor(_make_test_args(rollout_global_dataset=True, rollout_batch_size=8))
        executor.data_source = SimpleNamespace(dataset=list(range(20)))

        assert executor.get_num_rollout_per_epoch() == 2

    async def test_rejects_a_non_global_data_source(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Global dataset off with a twenty-entry data source; asking for the rollouts per
        # epoch must raise an assertion error.
        """Without a global dataset there is no epoch length to report."""
        executor = await _make_executor(_make_test_args(rollout_global_dataset=False))
        executor.data_source = SimpleNamespace(dataset=list(range(20)))

        with pytest.raises(AssertionError):
            executor.get_num_rollout_per_epoch()


@pytest.mark.asyncio
class TestCheckpointWithoutARolloutFunction:
    # [CLASS DOC] Tests of save and load when the train rollout function was never built (replay of
    # recorded data), where only the data source and the event log checkpoint are involved.
    async def test_checkpointing_a_replay_run_touches_only_the_data_source(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Builds an executor configured to replay recorded data from a nonexistent path
        # with a mock data source, then saves and loads rollout 3. Asserts the train rollout function is
        # none and the data source save and load were each called once with 3.
        """--load-debug-rollout-data replays recorded samples and builds no rollout function, so there is no rollout state to write or read back."""
        executor = await _make_executor(
            _make_test_args(load_debug_rollout_data="/nonexistent/rollout_{rollout_id}.pt")
        )
        executor.data_source = MagicMock()

        executor.save(3)
        executor.load(3)

        assert executor.generate_rollout is None
        executor.data_source.save.assert_called_once_with(3)
        executor.data_source.load.assert_called_once_with(3)

    async def test_a_replay_run_restores_at_startup_without_a_rollout_id(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        # [FUNCTION DOC] Same replay setup (legacy-protocol environment variable removed); calls load
        # with no rollout id and asserts the data source load was called once with none.
        """Startup restores the executor before the first step, so the default rollout id must reach the data source."""
        monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)
        executor = await _make_executor(
            _make_test_args(load_debug_rollout_data="/nonexistent/rollout_{rollout_id}.pt")
        )
        executor.data_source = MagicMock()

        executor.load()

        executor.data_source.load.assert_called_once_with(None)

    async def test_saving_a_replay_run_still_snapshots_the_event_log(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        # [FUNCTION DOC] Same replay setup with a recording event logger checkpoint; saves rollout 6.
        # Asserts the data source save was called once with 6 and the recorder saw one snapshot with the
        # executor args and 6.
        """Skipping the absent rollout function must not skip the rest of save, which still snapshots the audit log."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)
        recorder = _RecordingEventLoggerCheckpoint()
        monkeypatch.setattr(rexec, "event_logger_checkpoint", recorder)
        args = _make_test_args(load_debug_rollout_data="/nonexistent/rollout_{rollout_id}.pt")

        executor = await _make_executor(args)
        executor.data_source = MagicMock()

        executor.save(6)

        executor.data_source.save.assert_called_once_with(6)
        assert recorder.snapshots == [(args, 6)]

    async def test_an_absent_train_rollout_function_is_not_replaced_by_the_eval_one(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        # [FUNCTION DOC] Non-legacy protocol, train rollout function set to none, a recording eval
        # function and a mock data source; saves and loads rollout 2. Asserts only the eval function
        # recorded save then load for 2 (the eval instance does not stand in for the train one) and the
        # data source save and load were each called once with 2.
        """With no train rollout function, checkpointing skips it rather than falling back to the eval instance."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())

        executor = await _make_executor(_make_test_args())
        executor.use_legacy_rollout_v1 = False
        executor.generate_rollout = None
        calls: list[tuple[str, str, object]] = []
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)
        executor.data_source = MagicMock()

        executor.save(rollout_id=2)
        executor.load(rollout_id=2)

        assert calls == [("eval", "save", 2), ("eval", "load", 2)]
        executor.data_source.save.assert_called_once_with(2)
        executor.data_source.load.assert_called_once_with(2)


class _AlwaysEqualRolloutFn(_RecordingRolloutFn):
    # [CLASS DOC] HELPER: Recording rollout function double that compares equal to any object and has a
    # constant hash, to prove code distinguishes instances by identity rather than equality.
    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Always reports equality.
        return True

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the constant zero.
        return 0


@pytest.mark.asyncio
class TestCheckpointOfADistinctEvalRolloutFunction:
    # [CLASS DOC] Tests of which rollout function instances are checkpointed when the eval rollout
    # function is a separate, shared, equal-comparing or absent-protocol instance. Uses mock or
    # recording doubles and a mock data source with default test args.
    async def test_a_separate_eval_instance_is_checkpointed_on_its_own(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Installs two different mock rollout functions as train and eval, then saves and
        # loads rollout 7. Asserts each mock's save and load were called exactly once with 7.
        """A distinct --eval-function-path instance owns its own state, which restore would otherwise silently drop."""
        executor = await _make_executor(_make_test_args())
        executor.data_source = MagicMock()
        executor.generate_rollout = MagicMock()
        executor.eval_generate_rollout = MagicMock()

        executor.save(7)
        executor.load(7)

        executor.generate_rollout.save.assert_called_once_with(7)
        executor.generate_rollout.load.assert_called_once_with(7)
        executor.eval_generate_rollout.save.assert_called_once_with(7)
        executor.eval_generate_rollout.load.assert_called_once_with(7)

    async def test_a_shared_eval_instance_is_checkpointed_once(self, ray_local_mode, patch_low_level):
        # [FUNCTION DOC] Installs the same mock as both train and eval rollout function, then saves and
        # loads rollout 7. Asserts the mock's save and load were each called once (not twice) with 7.
        """Train and eval reuse one instance when the paths match, so a second hook call would checkpoint it twice."""
        executor = await _make_executor(_make_test_args())
        executor.data_source = MagicMock()
        shared = MagicMock()
        executor.generate_rollout = shared
        executor.eval_generate_rollout = shared

        executor.save(7)
        executor.load(7)

        shared.save.assert_called_once_with(7)
        shared.load.assert_called_once_with(7)

    async def test_an_eval_instance_that_compares_equal_is_still_checkpointed_separately(
        self, ray_local_mode, patch_low_level
    ):
        # [FUNCTION DOC] Installs two distinct always-equal recording doubles as train and eval, then
        # saves and loads rollout 4. Asserts both were checkpointed, in the order train save, eval save,
        # train load, eval load.
        """Two objects that compare equal still hold two independent states, so equality must not stand in for identity."""
        executor = await _make_executor(_make_test_args())
        executor.data_source = MagicMock()
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _AlwaysEqualRolloutFn("train", calls)
        executor.eval_generate_rollout = _AlwaysEqualRolloutFn("eval", calls)

        executor.save(4)
        executor.load(4)

        assert calls == [
            ("train", "save", 4),
            ("eval", "save", 4),
            ("train", "load", 4),
            ("eval", "load", 4),
        ]

    async def test_loading_without_a_rollout_id_forwards_none_to_the_eval_instance(
        self, ray_local_mode, patch_low_level
    ):
        # [FUNCTION DOC] Installs distinct recording train and eval doubles and calls load with no
        # rollout id. Asserts the record is train load with none, then eval load with none.
        """A restore of the latest checkpoint must reach the eval instance with the same unset rollout id."""
        executor = await _make_executor(_make_test_args())
        executor.data_source = MagicMock()
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _RecordingRolloutFn("train", calls)
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)

        executor.load()

        assert calls == [
            ("train", "load", None),
            ("eval", "load", None),
        ]

    async def test_the_legacy_protocol_leaves_a_distinct_eval_function_untouched(
        self, ray_local_mode, patch_low_level
    ):
        # [FUNCTION DOC] Legacy flag on with distinct recording doubles; saves and loads rollout 2.
        # Asserts the doubles recorded nothing and the data source save and load were each called once
        # with 2.
        """Legacy rollout functions are bare callables with no state, so neither instance is checkpointed."""
        executor = await _make_executor(_make_test_args())
        executor.use_legacy_rollout_v1 = True
        executor.data_source = MagicMock()
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _RecordingRolloutFn("train", calls)
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)

        executor.save(2)
        executor.load(2)

        assert calls == []
        executor.data_source.save.assert_called_once_with(2)
        executor.data_source.load.assert_called_once_with(2)
