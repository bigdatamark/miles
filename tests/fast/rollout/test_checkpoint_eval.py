# [MODULE DOC] CPU-only tests for snapshot ("checkpoint") evaluation of the rollout stack.
# Covers three areas: (1) the argument retargeting helper from the checkpoint eval module;
# (2) the rollout executor's eval entry point, driven on a bare executor object built without its
# actor constructor and given stub eval functions and stub fleets, checking input threading,
# logging, skip reasons, marker handling and serialization of fleet pins; (3) the asynchronous eval
# dispatcher (export a snapshot, fire eval on a fake manager, in-flight cap with skip or
# backpressure policy, snapshot ring retirement); and (4) the external-server eval example function,
# with its network, subprocess and dataset-eval helpers replaced by recording fakes.
# Registered with the CPU CI in the first CPU stage with a 60 second estimate.
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

import miles.ray.rollout.rollout_executor as rollout_executor_mod
from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnEvalOutput
from miles.rollout.checkpoint_eval import CheckpointEvalFn, EvalSkip, retarget_args


def make_args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds a plain argument namespace for the tests.
    # Defaults: router at 10.0.0.1:30000 with no policy, 4 rollout GPUs (2 per engine), 1 eval GPU
    # (1 per engine), snapshot eval on, no eval function path, train-only/offload/CI flags off, and a
    # router table with a default and an eval router. Keyword overrides replace or add fields.
    # Returns the namespace; no side effects.
    defaults = dict(
        sglang_router_ip="10.0.0.1",
        sglang_router_port=30000,
        sglang_router_policy=None,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2,
        eval_num_gpus=1,
        eval_num_gpus_per_engine=1,
        eval_uses_snapshots=True,
        eval_function_path=None,
        debug_train_only=False,
        offload_train=False,
        ci_test=False,
        sglang_model_routers={"default": ("10.0.0.1", 30000), "eval": ("10.0.0.2", 31000)},
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_retarget_args_swaps_router_and_sizing():
    # [FUNCTION DOC] Retargeting to a new router address and GPU sizing yields a copy whose router ip,
    # port, rollout GPU count and GPUs-per-engine take the new values, while the original namespace
    # keeps its old router (10.0.0.1:30000) and its 4 rollout GPUs.
    args = make_args()
    eval_args = retarget_args(args, "10.0.0.9", 39000, num_gpus=2, num_gpus_per_engine=2)

    assert (eval_args.sglang_router_ip, eval_args.sglang_router_port) == ("10.0.0.9", 39000)
    assert eval_args.rollout_num_gpus == 2
    assert eval_args.rollout_num_gpus_per_engine == 2
    # The original namespace is untouched.
    assert (args.sglang_router_ip, args.sglang_router_port) == ("10.0.0.1", 30000)
    assert args.rollout_num_gpus == 4


# ---------------- RolloutManager._eval_checkpoint (the single snapshot path) ----------------


class CheckpointFnStub(CheckpointEvalFn):
    # [CLASS DOC] HELPER: Stub checkpoint eval function. Records every eval input it receives, can be
    # told to raise an eval-skip with a given reason, and records whether it was disposed.
    def __init__(self, skip_reason=None):
        # [FUNCTION DOC] HELPER: Stores the optional skip reason, an empty list of received inputs
        # and a disposed flag set to false. Does not call the base constructor.
        self.inputs = []
        self.skip_reason = skip_reason
        self.disposed = False

    async def evaluate_checkpoint(self, checkpoint_dir, input):
        # [FUNCTION DOC] HELPER: Appends the input to the recorded list (the directory argument is
        # ignored). If a skip reason was configured, raises an eval-skip carrying it; otherwise returns
        # an eval output holding one dataset named "ds" with a single reward of 1.0.
        self.inputs.append(input)
        if self.skip_reason is not None:
            raise EvalSkip(self.skip_reason)
        return RolloutFnEvalOutput(data={"ds": {"rewards": [1.0]}})

    async def dispose(self):
        # [FUNCTION DOC] HELPER: Marks the stub as disposed.
        self.disposed = True


def make_manager(args, eval_fn=None, fleet=None):
    # [FUNCTION DOC] HELPER: Builds a bare rollout executor object without running its constructor
    # (unwrapping the remote actor class if the class is wrapped), then sets only the fields the eval
    # path reads: args, rollout id 7, a fresh eval lock, no health monitors, legacy v1 rollout off, no
    # metric checker, the given eval function and the given eval fleet. It also mutates the args:
    # snapshot eval is switched on exactly when an eval function exists and either a fleet is given or
    # the function is a checkpoint eval function. Returns the executor.
    mgr = object.__new__(
        getattr(rollout_executor_mod.RolloutExecutor, "__ray_actor_class__", rollout_executor_mod.RolloutExecutor)
    )
    mgr.args = args
    mgr.rollout_id = 7
    mgr._eval_lock = asyncio.Lock()
    mgr._health_monitors = []
    mgr.use_legacy_rollout_v1 = False
    mgr._metric_checker = None
    mgr.eval_generate_rollout = eval_fn
    mgr._eval_fleet = fleet
    args.eval_uses_snapshots = eval_fn is not None and (fleet is not None or isinstance(eval_fn, CheckpointEvalFn))
    return mgr


@pytest.fixture
def controller_env(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture that replaces three logging hooks of the executor module: saving
    # debug rollout data becomes a no-op; logging eval rollout data records the first call as
    # (rollout id, data, extra metrics) under the key "eval" and returns an empty mapping; logging an
    # eval skip records (rollout id, reason) under "skip". Yields-free: returns a namespace whose
    # logged mapping the tests inspect. Patches are undone by the monkeypatch fixture.
    logged = {}
    monkeypatch.setattr(rollout_executor_mod, "save_debug_rollout_data", lambda *a, **k: None)
    monkeypatch.setattr(
        rollout_executor_mod,
        "log_eval_rollout_data",
        lambda rollout_id, args, data, extra: logged.setdefault("eval", (rollout_id, data, extra)) or {},
    )
    monkeypatch.setattr(
        rollout_executor_mod,
        "log_eval_skip",
        lambda rollout_id, args, reason: logged.setdefault("skip", (rollout_id, reason)),
    )
    return SimpleNamespace(logged=logged)


async def test_eval_checkpoint_threads_input_and_logs(controller_env, tmp_path):
    # [FUNCTION DOC] Sets up a completed snapshot directory (with the completion marker file) and an
    # executor using the stub checkpoint function, then evaluates rollout 5 with an export time of
    # 1.5 s. Asserts: the function got exactly one input carrying the snapshot dir, weight version
    # "5" and no generate state; the logged eval has rollout id 5, a lag of 2 steps (against the
    # executor's rollout id 7) and the export time metric 1.5.
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    fn = CheckpointFnStub()
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, eval_fn=fn)

    await mgr.eval(5, hf_dir=str(snapshot), export_time_seconds=1.5)

    assert len(fn.inputs) == 1
    assert fn.inputs[0].hf_dir == str(snapshot)
    assert fn.inputs[0].weight_version == "5"
    assert fn.inputs[0].generate_state is None
    rollout_id, _data, extra = controller_env.logged["eval"]
    assert rollout_id == 5
    assert extra["eval/lag_steps"] == 2
    assert extra["eval/export_time_seconds"] == 1.5


async def test_debug_train_only_runs_snapshot_eval(controller_env, tmp_path):
    # [FUNCTION DOC] With the train-only debug flag on and a completed snapshot, evaluating rollout 5
    # still calls the checkpoint function once, with weight version "5".
    """Train-only skips shared-engine eval, but a pinned snapshot still evaluates."""
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    fn = CheckpointFnStub()
    args = make_args(debug_train_only=True, hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, eval_fn=fn)

    await mgr.eval(5, hf_dir=str(snapshot))

    assert len(fn.inputs) == 1
    assert fn.inputs[0].weight_version == "5"


async def test_eval_checkpoint_runs_the_eval_fn_on_the_fleet(controller_env, monkeypatch, tmp_path):
    # [FUNCTION DOC] Uses a plain eval function (not a checkpoint function) plus a fake fleet, with
    # the rollout-call helper patched to call the function directly. Evaluating rollout 5 on a
    # completed snapshot must pin the fleet exactly once with (snapshot dir, "5"), and the eval
    # function must receive an input whose generate state is what the fleet returned, with the
    # snapshot dir and weight version "5".
    """The fleet only delivers weights — the configured eval fn still generates, and it
    is the same object the shared posture would call."""
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    seen_inputs = []

    def eval_generate_rollout(input):
        # [FUNCTION DOC] HELPER: Records the input and returns a one-dataset output with reward 1.0.
        seen_inputs.append(input)
        return RolloutFnEvalOutput(data={"ds": {"rewards": [1.0]}})

    class FakeFleet:
        # [CLASS DOC] HELPER: Fake eval fleet that records pin requests.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Starts with an empty list of recorded pins.
            self.pins = []

        async def pin(self, checkpoint_dir, weight_version):
            # [FUNCTION DOC] HELPER: Records the pair (checkpoint dir, weight version) and returns
            # the fixed string "fleet-state" as the generate state.
            self.pins.append((checkpoint_dir, weight_version))
            return "fleet-state"

    fleet = FakeFleet()
    monkeypatch.setattr(rollout_executor_mod, "call_rollout_function", lambda fn, input: fn(input))
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path))
    mgr = make_manager(args, eval_fn=eval_generate_rollout, fleet=fleet)

    await mgr.eval(5, hf_dir=str(snapshot))

    assert fleet.pins == [(str(snapshot), "5")]
    assert seen_inputs[0].generate_state == "fleet-state"
    assert seen_inputs[0].hf_dir == str(snapshot)
    assert seen_inputs[0].weight_version == "5"


async def test_eval_checkpoint_missing_marker_skips(controller_env, tmp_path):
    # [FUNCTION DOC] A snapshot directory without the completion marker file makes the eval skip:
    # the checkpoint function receives no input, and the logged skip is (5, "ckpt_missing").
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()  # no .complete marker

    fn = CheckpointFnStub()
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, eval_fn=fn)

    await mgr.eval(5, hf_dir=str(snapshot))

    assert fn.inputs == []
    assert controller_env.logged["skip"] == (5, "ckpt_missing")


async def test_eval_checkpoint_skip_reason_propagates(controller_env, tmp_path):
    # [FUNCTION DOC] The stub raises an eval-skip with reason "pin_violation" on a complete snapshot.
    # Asserts the skip is logged as (5, "pin_violation") and no eval result is logged.
    """EvalSkip from any checkpoint backend becomes an attributable skipped point."""
    snapshot = tmp_path / "step_5"
    snapshot.mkdir()
    (snapshot / ".complete").touch()

    fn = CheckpointFnStub(skip_reason="pin_violation")
    args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
    mgr = make_manager(args, eval_fn=fn)

    await mgr.eval(5, hf_dir=str(snapshot))

    assert controller_env.logged["skip"] == (5, "pin_violation")
    assert "eval" not in controller_env.logged


@pytest.mark.parametrize("reason", ["busy", "export_failed", "ckpt_missing", "unhealthy", "crashed"])
def test_report_eval_skip_fails_ci_after_logging(controller_env, reason):
    # [FUNCTION DOC] Parametrized over the five skip reasons busy, export_failed, ckpt_missing,
    # unhealthy and crashed. With the CI-test flag on, reporting an eval skip for rollout 5 must raise
    # a runtime error whose message names the eval and the reason, and the skip must have been logged
    # first as (5, reason).
    """Every attributable eval problem logs its reason and then fails a CI run."""
    mgr = make_manager(make_args(ci_test=True))

    with pytest.raises(RuntimeError, match=rf"CI eval 5 skipped: {reason}"):
        mgr.report_eval_skip(5, reason)

    assert controller_env.logged["skip"] == (5, reason)


async def test_eval_shared_path_shape_unchanged(controller_env, monkeypatch):
    # [FUNCTION DOC] With no snapshot posture (plain eval function, no fleet, zero eval GPUs) the
    # eval must call the function once with no generate state, weight version or hf dir, and log with
    # no extra metrics (extra is none). The rollout-call helper is patched to call directly.
    """No snapshot posture must keep today's shared-engine call shape: no snapshot
    fields threaded, no lag/duration metrics added."""
    seen_inputs = []

    def eval_generate_rollout(input):
        # [FUNCTION DOC] HELPER: Records the input and returns an output with empty data.
        seen_inputs.append(input)
        return RolloutFnEvalOutput(data={})

    monkeypatch.setattr(rollout_executor_mod, "call_rollout_function", lambda fn, input: fn(input))
    args = make_args(hf_checkpoint="/base", eval_num_gpus=0)
    mgr = make_manager(args, eval_fn=eval_generate_rollout)

    await mgr.eval(5)

    assert len(seen_inputs) == 1
    assert seen_inputs[0].generate_state is None
    assert seen_inputs[0].weight_version is None
    assert seen_inputs[0].hf_dir is None
    _rollout_id, _data, extra = controller_env.logged["eval"]
    assert extra is None


class TestSnapshotEvalGuards:
    # [CLASS DOC] Tests of input validation and marker bypass in snapshot eval.
    async def test_snapshot_eval_without_an_hf_dir_is_rejected(self, controller_env):
        # [FUNCTION DOC] Evaluating without any hf dir while snapshot eval is on must fail an
        # assertion saying checkpoint eval requires a snapshot dir, and the function gets no input.
        """Snapshot eval has no checkpoint to evaluate without a dir, so it must fail loudly."""
        fn = CheckpointFnStub()
        args = make_args(hf_checkpoint="/base", eval_keep_snapshots=2)
        mgr = make_manager(args, eval_fn=fn)

        with pytest.raises(AssertionError, match="checkpoint eval requires an HF snapshot dir"):
            await mgr.eval(5)

        assert fn.inputs == []

    async def test_marker_bypass_evaluates_a_dir_without_a_complete_marker(self, controller_env, tmp_path):
        # [FUNCTION DOC] A snapshot directory with no completion marker is evaluated anyway when the
        # caller passes the require-marker flag as false: one input with that dir is received and no
        # skip is logged.
        """A caller-supplied checkpoint was never exported here, so there is no marker to wait for."""
        snapshot = tmp_path / "step_5"
        snapshot.mkdir()

        fn = CheckpointFnStub()
        args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path), eval_keep_snapshots=2)
        mgr = make_manager(args, eval_fn=fn)

        await mgr.eval(5, hf_dir=str(snapshot), require_marker=False)

        assert len(fn.inputs) == 1
        assert fn.inputs[0].hf_dir == str(snapshot)
        assert "skip" not in controller_env.logged


class BlockingFleet:
    # [CLASS DOC] HELPER: Fake eval fleet whose pin blocks until a release event is set, used to
    # observe serialization of concurrent evals.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts with an empty list of pinned weight versions and an unset
        # asyncio release event.
        self.pins = []
        self.release = asyncio.Event()

    async def pin(self, checkpoint_dir, weight_version):
        # [FUNCTION DOC] HELPER: Records the weight version, waits until the release event is set,
        # then returns the string "fleet-state". The directory is ignored.
        self.pins.append(weight_version)
        await self.release.wait()
        return "fleet-state"


class TestEvalFleetSerialization:
    # [CLASS DOC] Tests that a single eval fleet handles one checkpoint at a time.
    async def test_set_eval_fleet_serializes_concurrent_checkpoint_pins(self, controller_env, monkeypatch, tmp_path):
        # [FUNCTION DOC] Creates complete snapshots for rollouts 5 and 6, an executor with a plain
        # eval function, and a blocking fleet attached afterwards (snapshot eval forced on). Starts
        # both evals concurrently as tasks and yields to the loop a few times: only version "5" may
        # have been pinned. After releasing the fleet and awaiting both tasks, the pins are "5" then
        # "6", showing the second eval waits for the first.
        """One fleet holds one pinned checkpoint, so a second eval point cannot pin until the first finishes."""
        for rollout_id in (5, 6):
            snapshot = tmp_path / f"step_{rollout_id}"
            snapshot.mkdir()
            (snapshot / ".complete").touch()

        def eval_generate_rollout(input):
            # [FUNCTION DOC] HELPER: Ignores the input and returns a one-dataset output, reward 1.0.
            return RolloutFnEvalOutput(data={"ds": {"rewards": [1.0]}})

        monkeypatch.setattr(rollout_executor_mod, "call_rollout_function", lambda fn, input: fn(input))
        args = make_args(hf_checkpoint="/base", eval_hf_dir=str(tmp_path))
        mgr = make_manager(args, eval_fn=eval_generate_rollout)
        args.eval_uses_snapshots = True
        fleet = BlockingFleet()
        mgr._eval_fleet = fleet

        first = asyncio.create_task(mgr.eval(5, hf_dir=str(tmp_path / "step_5")))
        second = asyncio.create_task(mgr.eval(6, hf_dir=str(tmp_path / "step_6")))
        for _ in range(5):
            await asyncio.sleep(0)

        assert fleet.pins == ["5"]

        fleet.release.set()
        await asyncio.gather(first, second)

        assert fleet.pins == ["5", "6"]


# ---------------- driver (train_async.EvalDispatcher) ----------------


class FakeManagerActor:
    # [CLASS DOC] HELPER: Fake rollout manager actor for dispatcher tests. Its eval and skip-report
    # endpoints are callables returning asyncio futures, and every call is recorded.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates recorders: eval calls as (rollout id, hf dir, export time),
        # the require-marker flag of each eval call, skip calls as (rollout id, reason), and the list
        # of futures handed out by eval. Defines two inner callable classes and installs instances as
        # the eval and report-eval-skip attributes. Eval returns a new pending future per call; skip
        # returns an already-resolved future.
        self.eval_calls = []
        self.marker_flags = []
        self.skip_calls = []
        self._futures = []

        outer = self

        class _Eval:
            # [CLASS DOC] HELPER: Callable standing in for the manager's remote eval method.
            def __call__(self, rollout_id, hf_dir=None, export_time_seconds=None, require_marker=True):
                # [FUNCTION DOC] HELPER: Records the call and the marker flag on the outer fake,
                # creates a pending future on the current event loop, stores it and returns it.
                outer.eval_calls.append((rollout_id, hf_dir, export_time_seconds))
                outer.marker_flags.append(require_marker)
                fut = asyncio.get_event_loop().create_future()
                outer._futures.append(fut)
                return fut

        class _Skip:
            # [CLASS DOC] HELPER: Callable standing in for the manager's skip-report method.
            def __call__(self, rollout_id, reason):
                # [FUNCTION DOC] HELPER: Records (rollout id, reason) on the outer fake and returns a
                # future already resolved with none.
                outer.skip_calls.append((rollout_id, reason))
                fut = asyncio.get_event_loop().create_future()
                fut.set_result(None)
                return fut

        self.eval = _Eval()
        self.report_eval_skip = _Skip()

    def finish(self, index=0):
        # [FUNCTION DOC] HELPER: Resolves the eval future at the given index (default the first)
        # with none. Raises if that future was already resolved or the index is out of range.
        self._futures[index].set_result(None)


class FakeActorModel:
    # [CLASS DOC] HELPER: Fake training actor model whose HF export either records the request or
    # raises.
    def __init__(self, fail=False):
        # [FUNCTION DOC] HELPER: Stores the fail flag and an empty list of recorded exports.
        self.exports = []
        self.fail = fail

    async def export_hf(self, rollout_id, path):
        # [FUNCTION DOC] HELPER: Raises a runtime error if configured to fail; otherwise records
        # (rollout id, path). Writes nothing to disk.
        if self.fail:
            raise RuntimeError("export boom")
        self.exports.append((rollout_id, path))


@pytest.fixture
def dispatcher_env():
    # [FUNCTION DOC] HELPER: Fixture returning the eval dispatch module, imported lazily. Nothing is
    # patched because the dispatcher tracks exports as asyncio tasks.
    """The dispatcher tracks its exports as asyncio tasks, so there is nothing left to stand in for."""
    import miles.ray.rollout.eval_dispatch as eval_dispatch

    return eval_dispatch


def make_dispatcher(eval_dispatch, manager, actor_model, **arg_overrides):
    # [FUNCTION DOC] HELPER: Builds an eval dispatcher from the given dispatch module, manager and
    # actor model. Args default to: snapshot eval on, staging dir /dev/shm/eval_hf, at most 2 evals in
    # flight, backpressure overflow policy, keep 2 snapshots, no save-hf path; overrides replace them.
    # Returns the pair (dispatcher, args).
    dispatcher_defaults = dict(
        eval_uses_snapshots=True,
        eval_hf_dir="/dev/shm/eval_hf",
        eval_max_in_flight=2,
        eval_overflow_policy="backpressure",
        eval_keep_snapshots=2,
        save_hf=None,
    )
    dispatcher_defaults.update(arg_overrides)
    args = make_args(**dispatcher_defaults)
    return eval_dispatch.EvalDispatcher(args, actor_model, manager), args


async def test_dispatcher_exports_and_fires(dispatcher_env):
    # [FUNCTION DOC] Dispatching rollout 4 exports the model once to the step_4 directory under the
    # staging dir, then fires one eval on the manager with that dir and a non-null export time, and
    # leaves exactly one pending eval.
    manager = FakeManagerActor()
    actor_model = FakeActorModel()
    dispatcher, _ = make_dispatcher(dispatcher_env, manager, actor_model)

    await dispatcher.dispatch(4)

    assert actor_model.exports == [(4, "/dev/shm/eval_hf/step_4")]
    assert len(manager.eval_calls) == 1
    rollout_id, hf_dir, export_time = manager.eval_calls[0]
    assert (rollout_id, hf_dir) == (4, "/dev/shm/eval_hf/step_4")
    assert export_time is not None
    assert len(dispatcher.pending) == 1


async def test_dispatcher_export_failure_skips(dispatcher_env):
    # [FUNCTION DOC] When the export raises, no eval is fired and a skip with reason
    # "export_failed" is reported for rollout 4.
    manager = FakeManagerActor()
    dispatcher, _ = make_dispatcher(dispatcher_env, manager, FakeActorModel(fail=True))

    await dispatcher.dispatch(4)

    assert manager.eval_calls == []
    assert manager.skip_calls == [(4, "export_failed")]


async def test_dispatcher_skip_policy_drops_before_export(dispatcher_env):
    # [FUNCTION DOC] With a cap of one in-flight eval and the skip policy, the first dispatch runs
    # and the second (while the first is unfinished) is dropped before any export, reported as a
    # skip with reason "busy". Only rollout 1 was exported.
    manager = FakeManagerActor()
    actor_model = FakeActorModel()
    dispatcher, _ = make_dispatcher(
        dispatcher_env, manager, actor_model, eval_max_in_flight=1, eval_overflow_policy="skip"
    )

    await dispatcher.dispatch(1)
    await dispatcher.dispatch(2)  # at cap: dropped, no export

    assert manager.skip_calls == [(2, "busy")]
    assert actor_model.exports == [(1, "/dev/shm/eval_hf/step_1")]


async def test_dispatcher_force_overrides_skip_policy(dispatcher_env):
    # [FUNCTION DOC] Same cap-one skip setup, but the second dispatch is forced: it must wait for the
    # first eval to finish (a helper task resolves it after 10 ms) rather than be dropped. Asserts no
    # skips and eval calls for rollouts 1 then 2.
    """The final eval point must never be dropped: training is already over."""
    manager = FakeManagerActor()
    actor_model = FakeActorModel()
    dispatcher, _ = make_dispatcher(
        dispatcher_env, manager, actor_model, eval_max_in_flight=1, eval_overflow_policy="skip"
    )

    await dispatcher.dispatch(1)

    async def finish_soon():
        # [FUNCTION DOC] HELPER: Sleeps 10 ms, then resolves the first eval future.
        await asyncio.sleep(0.01)
        manager.finish(0)

    finisher = asyncio.create_task(finish_soon())
    await dispatcher.dispatch(2, force=True)  # at cap: waits instead of dropping
    await finisher

    assert manager.skip_calls == []
    assert [c[0] for c in manager.eval_calls] == [1, 2]


async def test_dispatcher_backpressure_awaits_oldest(dispatcher_env):
    # [FUNCTION DOC] With a cap of one and the backpressure policy, dispatching rollout 2 waits for
    # eval 1 to finish (resolved by a helper after 10 ms). Asserts evals were fired for 1 then 2 and
    # only eval 2 remains pending.
    manager = FakeManagerActor()
    dispatcher, _ = make_dispatcher(dispatcher_env, manager, FakeActorModel(), eval_max_in_flight=1)

    await dispatcher.dispatch(1)

    async def finish_soon():
        # [FUNCTION DOC] HELPER: Sleeps 10 ms, then resolves the first eval future.
        await asyncio.sleep(0.01)
        manager.finish(0)

    finisher = asyncio.create_task(finish_soon())
    await dispatcher.dispatch(2)  # must wait for eval 1 to finish
    await finisher

    assert [c[0] for c in manager.eval_calls] == [1, 2]
    assert len(dispatcher.pending) == 1  # only eval 2 pending


async def test_dispatcher_reuse_mode_uses_save_hf(dispatcher_env):
    # [FUNCTION DOC] With no staging dir but a save-hf path template, dispatching rollout 10 performs
    # no extra export and fires eval with the template filled in (/ckpt/hf/10); the require-marker
    # flag is true because reused checkpoints still carry a marker.
    manager = FakeManagerActor()
    actor_model = FakeActorModel()
    dispatcher, _ = make_dispatcher(
        dispatcher_env, manager, actor_model, eval_hf_dir=None, save_hf="/ckpt/hf/{rollout_id}"
    )

    await dispatcher.dispatch(10)

    assert actor_model.exports == []  # no extra export in reuse mode
    assert manager.eval_calls[0][:2] == (10, "/ckpt/hf/10")
    assert manager.marker_flags == [True]  # reused checkpoints still need their marker


async def test_dispatcher_caller_supplied_dir_skips_marker(dispatcher_env):
    # [FUNCTION DOC] Dispatching rollout 0 with a caller-supplied hf dir fires eval with that dir and
    # the require-marker flag false.
    """A caller-supplied hf_dir is an existing checkpoint (the pre-training baseline),
    not one of the dispatcher's exports: it never wrote a marker to check."""
    manager = FakeManagerActor()
    dispatcher, _ = make_dispatcher(dispatcher_env, manager, FakeActorModel())

    await dispatcher.dispatch(0, hf_dir="/base/hf_checkpoint")

    assert manager.eval_calls[0][:2] == (0, "/base/hf_checkpoint")
    assert manager.marker_flags == [False]


class TestSnapshotOwnership:
    # [CLASS DOC] Tests that the dispatcher deletes (retires) only snapshots it exported itself,
    # keeping a ring of the most recent ones.
    """The dispatcher retires snapshots it successfully exported."""

    def _make(self, dispatcher_env, tmp_path, **overrides):
        # [FUNCTION DOC] HELPER: Builds a fake manager and a dispatcher using the temp path as the
        # staging dir plus argument overrides. Returns (dispatcher, manager).
        manager = FakeManagerActor()
        dispatcher, _ = make_dispatcher(
            dispatcher_env, manager, FakeActorModel(), eval_hf_dir=str(tmp_path), **overrides
        )
        return dispatcher, manager

    async def _dispatch_and_finish(self, dispatcher, manager, rollout_id, tmp_path, *, crash=False):
        # [FUNCTION DOC] HELPER: Creates the step directory for the rollout id under the temp path
        # (standing in for the export), dispatches the rollout, then resolves the newest eval future,
        # with a runtime error if crash is set, else with none, and drains the dispatcher. Returns
        # the snapshot directory path.
        snapshot = tmp_path / f"step_{rollout_id}"
        snapshot.mkdir()
        await dispatcher.dispatch(rollout_id)
        index = len(manager._futures) - 1
        if crash:
            manager._futures[index].set_exception(RuntimeError("eval boom"))
        else:
            manager.finish(index)
        await dispatcher.drain()
        return snapshot

    async def test_keeps_only_the_ring(self, dispatcher_env, tmp_path):
        # [FUNCTION DOC] With keep-snapshots 2, after three completed evals (1, 2, 3) the oldest
        # snapshot directory is gone and the two newest still exist.
        dispatcher, manager = self._make(dispatcher_env, tmp_path, eval_keep_snapshots=2)

        dirs = [await self._dispatch_and_finish(dispatcher, manager, i, tmp_path) for i in (1, 2, 3)]

        assert not dirs[0].exists()
        assert dirs[1].exists() and dirs[2].exists()

    async def test_crashed_eval_still_retires_its_snapshot(self, dispatcher_env, tmp_path):
        # [FUNCTION DOC] With keep-snapshots 1, eval 1 crashes and eval 2 succeeds. Asserts a skip
        # with reason "crashed" was reported for rollout 1 and its snapshot directory was deleted.
        """The leak this ownership move exists to prevent: a failed point used to
        keep its snapshot forever, filling the recommended tmpfs staging dir."""
        dispatcher, manager = self._make(dispatcher_env, tmp_path, eval_keep_snapshots=1)

        first = await self._dispatch_and_finish(dispatcher, manager, 1, tmp_path, crash=True)
        await self._dispatch_and_finish(dispatcher, manager, 2, tmp_path)

        assert manager.skip_calls == [(1, "crashed")]
        assert not first.exists()

    async def test_dispatcher_leaves_failed_export_cleanup_to_writer(self, dispatcher_env, tmp_path):
        # [FUNCTION DOC] A pre-existing step_4 directory holding a weights file is left untouched when
        # the export of rollout 4 fails: the skip "export_failed" is reported and the file content is
        # unchanged, so cleanup of failed exports is the writer's job, not the dispatcher's.
        manager = FakeManagerActor()
        dispatcher, _ = make_dispatcher(dispatcher_env, manager, FakeActorModel(fail=True), eval_hf_dir=str(tmp_path))
        checkpoint = tmp_path / "step_4"
        checkpoint.mkdir()
        (checkpoint / "weights").write_bytes(b"previous checkpoint")

        await dispatcher.dispatch(4)

        assert manager.skip_calls == [(4, "export_failed")]
        assert (checkpoint / "weights").read_bytes() == b"previous checkpoint"

    async def test_never_deletes_what_it_did_not_export(self, dispatcher_env, tmp_path):
        # [FUNCTION DOC] In reuse mode (save-hf path, no staging dir, keep-snapshots 0), one dispatched
        # and drained eval must not delete the save-hf directory, and the dispatcher's internal list
        # of exported snapshots stays empty.
        """--save-hf checkpoints and --hf-checkpoint are not the dispatcher's to delete."""
        save_hf = tmp_path / "save_hf"
        save_hf.mkdir()
        manager = FakeManagerActor()
        dispatcher, _ = make_dispatcher(
            dispatcher_env, manager, FakeActorModel(), eval_hf_dir=None, save_hf=str(save_hf), eval_keep_snapshots=0
        )

        await dispatcher.dispatch(1)
        manager.finish()
        await dispatcher.drain()

        assert save_hf.exists()
        assert dispatcher._exported == []


async def test_dispatcher_shared_engine_blocks_like_today(dispatcher_env):
    # [FUNCTION DOC] With snapshot eval off, dispatch calls the manager's plain eval with only the
    # rollout id (the legacy fake accepts nothing else), awaits it, and keeps no pending evals.
    """No snapshot posture in args -> the plain blocking call, no dispatch machinery."""
    manager = FakeManagerActor()

    class _LegacyEval:
        # [CLASS DOC] HELPER: Eval callable taking only a rollout id, as the old manager did.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Starts with an empty list of recorded rollout ids.
            self.calls = []

        def __call__(self, rollout_id):
            # [FUNCTION DOC] HELPER: Records the id and returns an already-resolved future.
            self.calls.append(rollout_id)
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(None)
            return fut

    manager.eval = _LegacyEval()
    dispatcher, _ = make_dispatcher(dispatcher_env, manager, FakeActorModel(), eval_uses_snapshots=False)

    await dispatcher.dispatch(3)
    assert manager.eval.calls == [3]
    assert len(dispatcher.pending) == 0


# ------- example fn (examples/infra_features/fully_async/external_eval_fn.py) -------


@pytest.fixture
def external_fn_env(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture for the external-eval example module. Imports it and patches its
    # post helper (records a post and sets the fake server's loaded version to the payload's weight
    # version unless the server is "stuck"), get helper (records a get and returns the loaded
    # version), dataset-eval runner (records "eval" with the state and returns one dataset with
    # reward 1.0), generate-state constructor (a namespace wrapping the args) and health-wait helper
    # (records "health"). Also clears the four MILES_EXTERNAL_EVAL_ environment variables (URL, GPUS,
    # PORT, SERVER_ARGS). Returns a namespace with the module, recorded calls, fake server and the
    # monkeypatch object.
    import importlib

    mod = importlib.import_module("examples.infra_features.fully_async.external_eval_fn")
    calls = []

    server = SimpleNamespace(loaded_version=None)

    async def fake_post(url, payload):
        # [FUNCTION DOC] HELPER: Records a post, then sets the server's loaded version to the
        # payload's weight version, unless the server is already stuck, in which case it stays stuck.
        calls.append(("post", url, payload))
        server.loaded_version = payload["weight_version"] if server.loaded_version != "stuck" else "stuck"

    async def fake_get(url):
        # [FUNCTION DOC] HELPER: Records a get and returns a mapping with the loaded weight version.
        calls.append(("get", url))
        return {"weight_version": server.loaded_version}

    async def fake_run_eval(state, cache):
        # [FUNCTION DOC] HELPER: Records "eval" with the state (cache ignored) and returns one
        # dataset "ds" with rewards of 1.0.
        calls.append(("eval", state))
        return {"ds": {"rewards": [1.0]}}

    async def fake_wait_ok(url, **kwargs):
        # [FUNCTION DOC] HELPER: Records "health" with the url and returns immediately.
        calls.append(("health", url))

    monkeypatch.setattr(mod, "post", fake_post)
    monkeypatch.setattr(mod, "get", fake_get)
    monkeypatch.setattr(mod, "run_eval_datasets", fake_run_eval)
    monkeypatch.setattr(mod, "GenerateState", lambda args: SimpleNamespace(args=args))
    monkeypatch.setattr(mod, "wait_http_ok", fake_wait_ok)
    for var in ("URL", "GPUS", "PORT", "SERVER_ARGS"):
        monkeypatch.delenv(f"MILES_EXTERNAL_EVAL_{var}", raising=False)
    return SimpleNamespace(mod=mod, calls=calls, server=server, monkeypatch=monkeypatch)


def make_external_fn(external_fn_env, **env):
    # [FUNCTION DOC] HELPER: Sets each keyword as an environment variable prefixed with
    # MILES_EXTERNAL_EVAL_ (via the fixture's monkeypatch), builds args with hf checkpoint /base, and
    # constructs the external eval function with a constructor input holding the args and no data
    # source. Returns the function; in launch mode the constructor starts the (faked) server process.
    from miles.rollout.base_types import RolloutFnConstructorInput

    for var, value in env.items():
        external_fn_env.monkeypatch.setenv(f"MILES_EXTERNAL_EVAL_{var}", value)
    args = make_args(hf_checkpoint="/base")
    return external_fn_env.mod.ExternalSglangEvalFn(RolloutFnConstructorInput(args=args, data_source=None))


async def test_external_eval_fn_waits_pins_then_evals(external_fn_env):
    # [FUNCTION DOC] With a URL for an existing server, one call for rollout 5 must perform, in
    # order: a health wait on the generate-health path, a post to the update-weights-from-disk path
    # with model path /snap/step_5 and weight version "5", a get of the model info path, then the
    # dataset eval. The eval state's args must point the router at the server's host and port
    # (eval-host, 31000), and the output data is the fake eval result.
    fn = make_external_fn(external_fn_env, URL="http://eval-host:31000")

    output = await fn(RolloutFnEvalInput(rollout_id=5, weight_version="5", hf_dir="/snap/step_5"))

    assert external_fn_env.calls[0] == ("health", "http://eval-host:31000/health_generate")
    assert external_fn_env.calls[1] == (
        "post",
        "http://eval-host:31000/update_weights_from_disk",
        {"model_path": "/snap/step_5", "weight_version": "5"},
    )
    assert external_fn_env.calls[2] == ("get", "http://eval-host:31000/model_info")
    assert external_fn_env.calls[3][0] == "eval"
    # The eval state targets the external server, built from the real training args.
    state = external_fn_env.calls[3][1]
    assert (state.args.sglang_router_ip, state.args.sglang_router_port) == ("eval-host", 31000)
    assert output.data == {"ds": {"rewards": [1.0]}}


async def test_external_eval_fn_pin_failure_retries_then_raises(external_fn_env):
    # [FUNCTION DOC] When the server never reports the requested version (stuck), the call raises a
    # runtime error mentioning a pin failure, after exactly two posts (one retry), and no eval runs.
    fn = make_external_fn(external_fn_env, URL="http://eval-host:31000")
    external_fn_env.server.loaded_version = "stuck"  # server never reports the pinned version

    with pytest.raises(RuntimeError, match="pin failed"):
        await fn(RolloutFnEvalInput(rollout_id=5, weight_version="5", hf_dir="/snap/step_5"))

    assert len([c for c in external_fn_env.calls if c[0] == "post"]) == 2  # one retry
    assert not [c for c in external_fn_env.calls if c[0] == "eval"]


async def test_external_eval_fn_launches_own_server(external_fn_env, monkeypatch):
    # [FUNCTION DOC] With GPUs "6,7" and extra server args set and no URL, construction launches one
    # server subprocess (process creation is faked and recorded). Asserts: the visible-devices
    # environment is "6,7"; the tensor-parallel size flag is 2; the model path flag is /base; the
    # command ends with the extra args; the function's url is http://127.0.0.1:31000 (default port).
    # Disposing the function terminates the process.
    """Launch mode is the black-box promise: init prepares everything, pinned to
    the GPUs the user names, extra sglang flags passed through; dispose tears down."""
    procs = []

    def fake_popen(cmd, env=None):
        # [FUNCTION DOC] HELPER: Fake process launcher. Records a namespace with the command,
        # environment and a terminated flag (false), gives it a terminate method that sets the flag,
        # and returns it.
        procs.append(SimpleNamespace(cmd=cmd, env=env, terminated=False))
        procs[-1].terminate = lambda p=procs[-1]: setattr(p, "terminated", True)
        return procs[-1]

    monkeypatch.setattr(external_fn_env.mod.subprocess, "Popen", fake_popen)

    fn = make_external_fn(external_fn_env, GPUS="6,7", SERVER_ARGS="--attention-backend fa3")

    (proc,) = procs
    assert proc.env["CUDA_VISIBLE_DEVICES"] == "6,7"
    assert proc.cmd[proc.cmd.index("--tp") + 1] == "2"
    assert proc.cmd[proc.cmd.index("--model-path") + 1] == "/base"
    assert proc.cmd[-2:] == ["--attention-backend", "fa3"]
    assert fn._url == "http://127.0.0.1:31000"
    await fn.dispose()
    assert proc.terminated
