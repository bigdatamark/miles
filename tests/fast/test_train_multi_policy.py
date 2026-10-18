# [MODULE DOC] Fast tests (stage-a CPU suite, estimated 30 seconds) of the multi-policy training
# driver (the top-level train_multi_policy script): one training loop per policy, sharing one
# rollout executor and inference controller, with the first policy as leader that owns the run
# length, saving, evals and the external save sentinel, and followers that train unbounded rounds
# and are parked at agreed moments. All out-of-process dependencies are replaced by async mocks
# and monkeypatches (autouse fixtures stub orchestration init, metrics group definition, mini
# fault tolerance, argument validation, consistency assertion, trainer and rollout component
# creation, rollout data reference removal and weight updates). Tests cover initial weight
# publication and API server wiring, terminal lifecycle ordering, per-policy draining, updating,
# resuming and failure propagation, debug exit, checkpoint saving coordination, cancellation
# absorption and evaluation dispatch.
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import train_multi_policy as multi_policy_driver
from tests.fast.fixtures.args_fixtures import parser_defaults
from tests.fast.fixtures.driver_fakes import FakeWorkerManager
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config
from train_multi_policy import train_multi_policy

from miles.ray import placement_group, wiring
from miles.utils.async_utils import with_disposer
from miles.utils.multi_policy.checkpoint_state import MultiPolicyCheckpointState
from miles.utils.multi_policy.parker import Parker
from miles.utils.multi_policy.utils import TrainerInfo


def _make_args(**overrides: Any) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds the driver arguments: the parser defaults overlaid with test
    # defaults (an encoded two-policy Megatron config for policies a and b, 2 rollouts, weight
    # update interval 1, no save settings, no debug exit, weight-equality check off with selector
    # all) and then the keyword overrides. Returns a namespace.
    defaults = dict(
        megatron_config=encode_megatron_config("a", "b"),
        num_rollout=2,
        update_weights_interval=1,
        save=None,
        save_interval=None,
        save_trigger_sentinel=None,
        debug_exit_after_rollout=None,
        check_weight_update_equal=False,
        check_weight_update_allow_quant_error=False,
        check_weight_update_selector="all",
        check_weight_update_skip_list=None,
    )
    defaults.update(overrides)
    return Namespace(**{**parser_defaults(), **defaults})


def _make_trainers(model_ids, handles=None, start_rollout_ids=None) -> dict[str, TrainerInfo]:
    # [FUNCTION DOC] HELPER: Builds a mapping from model id to trainer info. Handles default to
    # one async mock per model id; start rollout ids default to an empty mapping (each missing
    # id is 0). Every handle is first made to yield to the event loop when its train method is an
    # unconfigured async mock. Returns the mapping of model id to info (id, start rollout id,
    # handle).
    handles = {model_id: AsyncMock() for model_id in model_ids} if handles is None else handles
    start_rollout_ids = start_rollout_ids or {}
    for handle in handles.values():
        _let_follower_yield(handle)
    return {
        model_id: TrainerInfo(model_id=model_id, start_rollout_id=start_rollout_ids.get(model_id, 0), handle=handle)
        for model_id, handle in handles.items()
    }


async def _run(
    args: Namespace,
    *,
    model_ids: tuple[str, ...] = ("a", "b"),
    trainers: dict[str, AsyncMock] | None = None,
    start_rollout_ids: dict[str, int] | None = None,
    rollout_executor: AsyncMock | None = None,
    lifecycle_events: list[str] | None = None,
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Async; runs the multi-policy training function end to end under the
    # disposer wrapper with a 30 second timeout. Builds the trainer infos for the model ids
    # (default a and b), a context holding the trainers' handles, an async-mock inference
    # controller and the given or a mock rollout executor. If a lifecycle events list is given,
    # the executor's and controller's dispose calls append executor_dispose and inference_dispose
    # to it. Sets the patched trainer creation to return the infos and the patched rollout
    # component creation to return (controller, executor, none). Returns the context mapping.
    """Drive the whole driver with every out-of-process dependency stubbed out."""
    infos = _make_trainers(model_ids, handles=trainers, start_rollout_ids=start_rollout_ids)
    context = dict(
        trainers={model_id: info.handle for model_id, info in infos.items()},
        inference_controller=AsyncMock(),
        rollout_executor=AsyncMock() if rollout_executor is None else rollout_executor,
    )
    if lifecycle_events is not None:
        context["rollout_executor"].dispose.side_effect = lambda: lifecycle_events.append("executor_dispose")
        context["inference_controller"].dispose.side_effect = lambda: lifecycle_events.append("inference_dispose")
    multi_policy_driver.create_trainers.return_value = infos
    multi_policy_driver.create_rollout_components.return_value = (
        context["inference_controller"],
        context["rollout_executor"],
        None,
    )
    await asyncio.wait_for(with_disposer(train_multi_policy, args), timeout=30)
    return context


@pytest.fixture(autouse=True)
def _stub_driver_environment(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture. Replaces (reverted by monkeypatch) in the driver
    # module orchestration init, policy metric group definition, mini fault-tolerance start,
    # multi-policy argument validation and the consistent-restore assertion with do-nothing
    # callables, trainer creation with an async mock returning an empty mapping, and rollout
    # component creation with a plain async mock.
    """Everything the driver reaches outside its own loop: cluster, tracking and logging."""
    for name in (
        "init_orchestration_script",
        "define_policy_metric_groups",
        "maybe_start_mini_ft_controller",
        "validate_multi_policy_args",
        "assert_consistent_restore",
    ):
        monkeypatch.setattr(multi_policy_driver, name, lambda *a, **kw: None)
    monkeypatch.setattr(multi_policy_driver, "create_trainers", AsyncMock(return_value={}))
    monkeypatch.setattr(multi_policy_driver, "create_rollout_components", AsyncMock())


@pytest.fixture(autouse=True)
def _no_object_store(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture; replaces the driver's rollout-data-reference removal
    # with a do-nothing function so no object store is touched.
    monkeypatch.setattr(multi_policy_driver, "remove_rollout_data_refs", lambda args, ref: None)


@pytest.fixture(autouse=True)
def _stub_update_weights(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture; replaces the driver's weight update with an async
    # mock so tests can inspect its awaited calls.
    monkeypatch.setattr(multi_policy_driver, "update_weights", AsyncMock())


async def _slow_train(rollout_id: int, rollout_data_ref, **kwargs) -> None:
    # [FUNCTION DOC] HELPER: Async fake train step; ignores its arguments and sleeps 0.05 seconds.
    await asyncio.sleep(0.05)


async def _train_never_returning(rollout_id: int, rollout_data_ref: Any, **kwargs: Any) -> None:
    # [FUNCTION DOC] HELPER: Async fake train step that never completes (waits on an event that
    # is never set) until cancelled.
    await asyncio.Event().wait()


def _let_follower_yield(handle) -> None:
    # [FUNCTION DOC] HELPER: If the handle's train method is an async mock with no configured side
    # effect, gives it a side effect that just yields control to the event loop once, so a
    # follower does not spin without letting the leader run. Otherwise changes nothing.
    async def yield_to_leader(rollout_id: int, rollout_data_ref, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Async no-op that sleeps for zero seconds to yield once.
        await asyncio.sleep(0)

    if isinstance(handle.train, AsyncMock) and handle.train.side_effect is None:
        handle.train.side_effect = yield_to_leader


class TestInitialWeightPublication:
    # [CLASS DOC] Tests of what the driver does at startup before any rollout: API server
    # wiring and the initial per-policy weight sync and comparison.
    async def test_api_server_receives_every_configured_trainer_handle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Replaces the API server starter with a mock and the backend capability
        # getter with a stub. Runs with zero rollouts and an API port; asserts one start whose
        # trainer models mapping has the actor entries a-actor and b-actor with the two trainer
        # handles and whose inference controller is the context's controller.
        """Each configured trainer id must expose its model's live handle through the shared API server."""
        start_api_server = Mock()
        monkeypatch.setattr(placement_group, "start_api_server", start_api_server)
        monkeypatch.setattr(
            placement_group,
            "get_backend_capability",
            lambda _args: SimpleNamespace(cell_operations=lambda: object()),
        )

        context = await _run(_make_args(num_rollout=0, api_server_port=18080))

        start_api_server.assert_called_once()
        assert start_api_server.call_args.kwargs["trainer_models"] == {
            "a-actor": context["trainers"]["a"],
            "b-actor": context["trainers"]["b"],
        }
        assert start_api_server.call_args.kwargs["inference_controller"] is context["inference_controller"]

    async def test_every_policy_compares_its_engines_against_its_own_trainer(self):
        # [FUNCTION DOC] With zero rollouts and the equality check on, the controller's awaited
        # check-weights calls cover model ids a and b (sorted list equals a, b).
        """--ci-test asks for this comparison, and running it for one policy would leave the others unchecked."""
        context = await _run(_make_args(num_rollout=0, check_weight_update_equal=True))

        compared = [call.kwargs["model_id"] for call in context["inference_controller"].check_weights.await_args_list]
        assert sorted(compared) == ["a", "b"]

    async def test_each_policy_stamps_its_startup_sync_with_its_own_restore_point(self):
        # [FUNCTION DOC] With start rollout ids a=3 and b=7 and zero rollouts, the weight-update
        # calls that have no rollout id keyword (the startup syncs) carry per-trainer start rollout
        # ids equal to those values.
        """The policies resume at their own rollouts, so one global start id would misattribute their weights."""
        await _run(_make_args(num_rollout=0), start_rollout_ids=dict(a=3, b=7))

        stamped = {
            call.kwargs["trainer_model_id"]: call.args[0].start_rollout_id
            for call in multi_policy_driver.update_weights.await_args_list
            if "rollout_id" not in call.kwargs
        }
        assert stamped == dict(a=3, b=7)

    async def test_a_run_that_does_not_ask_for_the_comparison_does_not_pay_for_it(self):
        # [FUNCTION DOC] With the equality check off and zero rollouts, the controller's
        # check-weights is never awaited.
        """The comparison walks every parameter, so it stays off unless the run turns it on."""
        context = await _run(_make_args(num_rollout=0))

        context["inference_controller"].check_weights.assert_not_awaited()


class TestTerminalLifecycle:
    # [CLASS DOC] Tests of end-of-run cleanup ordering.
    async def test_multi_policy_run_releases_its_worker_manager_after_all_trainers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Uses a fake worker manager and two trainer mocks whose dispose appends a
        # per-model dispose event. Replaces the orchestration init with one registering the real
        # manager shutdown on the disposer, and the wiring's actor-kill with the fake's kill. After
        # a zero-rollout run asserts the manager was killed once, the last two events are manager
        # shutdown then manager kill, and the preceding events are exactly the executor, inference
        # controller and both trainer disposals (as a set).
        """The multi-policy driver releases its own manager after shared and trainer cleanup."""
        events: list[str] = []
        manager = FakeWorkerManager(events)
        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        for model_id, trainer in trainers.items():
            trainer.dispose.side_effect = lambda model_id=model_id: events.append(f"{model_id}_dispose")

        def init_orchestration_script(_args: Namespace, *, disposer: Any) -> FakeWorkerManager:
            # [FUNCTION DOC] HELPER: Fake init; adds the wiring shutdown function bound to the fake
            # manager onto the disposer and returns the manager.
            disposer.add(partial(wiring.shutdown_worker_manager, manager))
            return manager

        monkeypatch.setattr(multi_policy_driver, "init_orchestration_script", init_orchestration_script)
        monkeypatch.setattr(wiring.ray, "kill", manager.kill)
        await _run(_make_args(num_rollout=0), trainers=trainers, lifecycle_events=events)

        assert manager.killed == [manager]
        assert events[-2:] == ["manager_shutdown", "manager_kill"]
        assert set(events[:-2]) == {"executor_dispose", "inference_dispose", "a_dispose", "b_dispose"}


class TestRunPolicies:
    # [CLASS DOC] Tests of the per-policy training loops and their coordination.
    async def test_every_policy_drains_and_updates_only_its_own_model(self, monkeypatch):
        # [FUNCTION DOC] Replaces the weight update with an async mock recording (trainer model
        # id, rollout id) pairs. After a default run asserts the executor was asked for data
        # twice for policy a, drains happened for exactly policies a and b, and policy a's updates
        # are for none (startup), 0 and 1; both policies appear in updates.
        """Two policies sharing one executor must never train on, or publish into, each other's model."""
        updated: list[tuple[str, int]] = []
        monkeypatch.setattr(
            multi_policy_driver,
            "update_weights",
            AsyncMock(
                side_effect=lambda *a, rollout_id=None, trainer_model_id=None, **kw: updated.append(
                    (trainer_model_id, rollout_id)
                )
            ),
        )

        context = await _run(_make_args())

        drained = [call.kwargs["trainer_model_id"] for call in context["rollout_executor"].get.await_args_list]
        assert drained.count("a") == 2
        assert set(drained) == {"a", "b"}
        assert [rollout_id for model_id, rollout_id in updated if model_id == "a"] == [None, 0, 1]
        assert {model_id for model_id, _ in updated} == {"a", "b"}

    async def test_a_policy_only_resumes_the_health_probing_of_its_own_engines(self):
        # [FUNCTION DOC] With one rollout and policy b's training never returning, asserts the
        # controller's prepare-rollout calls are exactly (rollout 0, model a) and (rollout 0,
        # model b), i.e. each is scoped by model id.
        """Resuming the whole fleet here un-pauses probing of a policy that is mid weight broadcast."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        trainers["b"].train = _train_never_returning

        context = await _run(_make_args(num_rollout=1), trainers=trainers)

        prepared = context["inference_controller"].prepare_rollout.await_args_list
        assert sorted((call.args[0], call.kwargs["model_id"]) for call in prepared) == [(0, "a"), (0, "b")]

    async def test_two_policies_are_inside_the_executor_at_the_same_time(self):
        # [FUNCTION DOC] Replaces the executor's get with a coroutine that counts arrivals, sets
        # an event when the second arrives, waits up to 5 seconds for it, and returns a mapping
        # with a none data reference. Runs one rollout and asserts the event was set, i.e. both
        # policies were inside get at once.
        """The whole point of one loop per policy is that they overlap; the executor must tolerate it."""
        arrivals = 0
        both_arrived = asyncio.Event()

        async def _get(rollout_id: int, trainer_model_id: str | None = None):
            # [FUNCTION DOC] HELPER: Async fake executor get; increments the arrival count, sets
            # the shared event on the second arrival, waits for it (5 second timeout, raising a
            # timeout error otherwise) and returns a mapping with a none data reference.
            nonlocal arrivals
            arrivals += 1
            if arrivals == 2:
                both_arrived.set()
            await asyncio.wait_for(both_arrived.wait(), timeout=5)
            return dict(data_ref=None)

        rollout_executor = AsyncMock()
        rollout_executor.get = _get

        await _run(_make_args(num_rollout=1), rollout_executor=rollout_executor)

        assert both_arrived.is_set()

    async def test_a_failing_policy_stops_the_others_instead_of_orphaning_them(self):
        # [FUNCTION DOC] Policy a's train raises a runtime error; policy b's train counts rounds
        # and sleeps 0.05 seconds. With 100 rollouts the run must raise the runtime error with
        # the trainer a message, and policy b must have run at most 2 rounds.
        """A surviving loop keeps training and writing checkpoints while the run is already dead."""
        rounds_of_b = 0

        async def _train(rollout_id: int, rollout_data_ref, **kwargs) -> None:
            # [FUNCTION DOC] HELPER: Async fake train for policy b; increments the round counter
            # and sleeps 0.05 seconds.
            nonlocal rounds_of_b
            rounds_of_b += 1
            await asyncio.sleep(0.05)

        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        trainers["a"].train = AsyncMock(side_effect=RuntimeError("trainer a died"))
        trainers["b"].train = _train

        with pytest.raises(RuntimeError, match="trainer a died"):
            await _run(_make_args(num_rollout=100), trainers=trainers)

        assert rounds_of_b <= 2

    async def test_a_policy_resumes_from_its_own_position(self):
        # [FUNCTION DOC] With 3 rollouts and start rollout ids a=0, b=2, asserts policy a trained
        # rollouts 0, 1, 2 in order, and policy b's trained rollout ids are consecutive starting at 2.
        """Each trainer restores its own checkpoint, so the policies need not stand at the same rollout."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(_make_args(num_rollout=3), trainers=trainers, start_rollout_ids=dict(a=0, b=2))

        assert [call.args[0] for call in trainers["a"].train.await_args_list] == [0, 1, 2]
        rounds_of_b = [call.args[0] for call in trainers["b"].train.await_args_list]
        assert rounds_of_b == list(range(2, 2 + len(rounds_of_b)))

    async def test_a_policy_updates_its_weights_on_its_own_interval(self, monkeypatch):
        # [FUNCTION DOC] With the weight update recorder, 4 rollouts and update interval 2: policy
        # a's updates are none (startup), 1 and 3; policy b's non-startup updates all have odd
        # rollout ids.
        """The rhythm is counted on the absolute rollout id, so publishing every round would be wrong."""
        updated: list[tuple[str, int]] = []
        monkeypatch.setattr(
            multi_policy_driver,
            "update_weights",
            AsyncMock(
                side_effect=lambda *a, rollout_id=None, trainer_model_id=None, **kw: updated.append(
                    (trainer_model_id, rollout_id)
                )
            ),
        )

        await _run(_make_args(num_rollout=4, update_weights_interval=2))

        assert [rollout_id for model_id, rollout_id in updated if model_id == "a"] == [None, 1, 3]
        rounds_of_b = [rollout_id for model_id, rollout_id in updated if model_id == "b" and rollout_id is not None]
        assert all(rollout_id % 2 == 1 for rollout_id in rounds_of_b)

    async def test_a_debug_run_stops_the_leader_after_its_own_rounds(self):
        # [FUNCTION DOC] With 10 rollouts, debug exit after 1 rollout and start ids a=0, b=5,
        # the leader trained only rollout 0.
        """--debug-exit-after-rollout counts from where the policy resumed, not from rollout zero."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(
            _make_args(num_rollout=10, debug_exit_after_rollout=1), trainers=trainers, start_rollout_ids=dict(a=0, b=5)
        )

        assert [call.args[0] for call in trainers["a"].train.await_args_list] == [0]

    async def test_a_debug_run_leaves_the_followers_running(self):
        # [FUNCTION DOC] With the leader using the slow train, 10 rollouts and debug exit after 1,
        # the follower still trains at least 2 rounds.
        """A follower's rounds are the leader's to end, so honouring the flag there would retire it
        from the checkpoint every remaining round waits at."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        trainers["a"].train = _slow_train

        await _run(_make_args(num_rollout=10, debug_exit_after_rollout=1), trainers=trainers)

        assert len(trainers["b"].train.await_args_list) >= 2

    async def test_the_run_ends_when_the_leader_runs_out_of_rounds(self):
        # [FUNCTION DOC] With 2 rollouts, the leader trained exactly rollouts 0 and 1.
        """The leader owns --num-rollout; a follower resuming further back must not extend the run."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(_make_args(num_rollout=2), trainers=trainers, start_rollout_ids=dict(a=0, b=0))

        assert [call.args[0] for call in trainers["a"].train.await_args_list] == [0, 1]

    async def test_a_follower_is_never_the_one_that_ends_the_run(self):
        # [FUNCTION DOC] With a slow leader and 2 rollouts, the follower still trains at least 2
        # rounds.
        """Followers train unbounded rounds, so the run must not stop because one of them reached num_rollout."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        trainers["a"].train = _slow_train

        await _run(_make_args(num_rollout=2), trainers=trainers)

        assert len(trainers["b"].train.await_args_list) >= 2


class TestSaving:
    # [CLASS DOC] Tests of checkpoint saving coordination between the leader and followers.
    async def test_the_leader_parks_everybody_and_records_where_they_stood(self, tmp_path):
        # [FUNCTION DOC] With a slow follower, 2 rollouts, save directory and interval 1, loads the
        # multi-policy checkpoint state saved for leader rollout 1 and asserts the leader model is
        # a, the leader's recorded rollout id is 1, and the follower's recorded rollout id equals
        # the last rollout id its save was awaited with.
        """DataSource and RolloutExecutor are global, so their snapshot needs one agreed moment."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        trainers["b"].train = _slow_train

        await _run(_make_args(num_rollout=2, save=str(tmp_path), save_interval=1), trainers=trainers)

        state = MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=1)
        assert state.leader_model_id == "a"
        assert state.rollout_ids["a"] == 1
        assert state.rollout_ids["b"] == [call.args[0] for call in trainers["b"].save_model.await_args_list][-1]

    async def test_a_parked_follower_is_saved_at_the_round_it_reached(self):
        # [FUNCTION DOC] Replaces the follower's save with a mock side effect recording pairs of
        # (save rollout id, the follower's last trained rollout id). With one rollout and interval
        # 1, exactly one pair is recorded and its elements are equal.
        """A record naming a policy at a rollout it never checkpointed cannot be resumed."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        saves: list[tuple[int, int]] = []

        async def _note_follower_position(rollout_id: int, **kwargs: Any) -> None:
            # [FUNCTION DOC] HELPER: Async fake save; appends the save's rollout id and the
            # follower's last awaited train rollout id to the list.
            saves.append((rollout_id, trainers["b"].train.await_args_list[-1].args[0]))

        trainers["b"].save_model = AsyncMock(side_effect=_note_follower_position)

        await _run(_make_args(num_rollout=1, save=None, save_interval=1), trainers=trainers)

        [(saved_at, reached)] = saves
        assert saved_at == reached

    async def test_every_policy_is_on_disk_before_the_record_claims_the_checkpoint_exists(self):
        # [FUNCTION DOC] With one rollout and save interval 1, every trainer's save was awaited
        # exactly once with the force-synchronous flag true.
        """An asynchronous follower save still running would leave the record pointing at files nobody wrote."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(_make_args(num_rollout=1, save=None, save_interval=1), trainers=trainers)

        for trainer in trainers.values():
            assert [call.kwargs["force_sync"] for call in trainer.save_model.await_args_list] == [True]

    async def test_only_the_leader_snapshots_the_rollout_executor(self):
        # [FUNCTION DOC] With 2 rollouts and save interval 1, the executor's save was awaited with
        # rollout ids 0 and 1 only (once per round).
        """One buffer serves every policy, so a second policy saving it would snapshot it twice per round."""
        kwargs = await _run(_make_args(num_rollout=2, save=None, save_interval=1))

        assert [call.args[0] for call in kwargs["rollout_executor"].save.await_args_list] == [0, 1]

    async def test_every_checkpoint_of_the_leader_is_saved_synchronously(self):
        # [FUNCTION DOC] With 2 rollouts and interval 1, both of the leader's save calls carry the
        # force-synchronous flag true.
        """An asynchronous save would let the run publish a record of files that are still being written."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(_make_args(num_rollout=2, save=None, save_interval=1), trainers=trainers)

        assert [call.kwargs["force_sync"] for call in trainers["a"].save_model.await_args_list] == [True, True]

    async def test_two_checkpoints_in_a_row_both_reach_every_policy(self):
        # [FUNCTION DOC] With 2 rollouts and interval 1, the leader saved rollouts 0 and 1, and the
        # follower saved exactly twice with non-decreasing rollout ids.
        """The second round must wait for the first one to disperse instead of reading a stale arrival."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(_make_args(num_rollout=2, save=None, save_interval=1), trainers=trainers)

        assert [call.args[0] for call in trainers["a"].save_model.await_args_list] == [0, 1]
        rounds_of_b = [call.args[0] for call in trainers["b"].save_model.await_args_list]
        assert len(rounds_of_b) == 2 and rounds_of_b[0] <= rounds_of_b[1]

    async def test_a_run_without_a_save_directory_records_nothing(self, tmp_path):
        # [FUNCTION DOC] Runs with no save directory but interval 1 and asserts the temporary
        # directory (not passed to the run) is still empty.
        # KNOWN ISSUE: the temporary directory is never given to the driver, so the final emptiness
        # assertion cannot fail and does not test the claim in its docstring.
        """--save is what asks for checkpoints on disk; the record must not invent a directory."""
        await _run(_make_args(num_rollout=1, save=None, save_interval=1))

        assert list(tmp_path.iterdir()) == []

    async def test_the_external_save_sentinel_is_read_and_removed_by_the_leader_alone(self, tmp_path):
        # [FUNCTION DOC] Creates an empty sentinel file, runs one rollout with no save interval
        # and the sentinel path set, and asserts the sentinel file was removed and the leader's
        # save was awaited exactly once with rollout 0.
        """Every policy racing to remove one sentinel would leave the losers deleting a missing file."""
        sentinel = tmp_path / "save-now"
        sentinel.write_text("")
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(
            _make_args(num_rollout=1, save=None, save_trigger_sentinel=str(sentinel)),
            trainers=trainers,
        )

        assert not sentinel.exists()
        assert [call.args[0] for call in trainers["a"].save_model.await_args_list] == [0]

    async def test_a_sentinel_triggered_save_is_synchronous_for_the_leader(self, tmp_path):
        # [FUNCTION DOC] Creates an empty sentinel file and runs 2 rollouts with the sentinel path
        # set. Asserts the leader and the follower each saved exactly once, both with the
        # force-synchronous flag true.
        """Whoever wrote the sentinel is waiting for the files, so the leader may not defer its own write."""
        sentinel = tmp_path / "save-now"
        sentinel.write_text("")
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(
            _make_args(num_rollout=2, save=None, save_trigger_sentinel=str(sentinel)),
            trainers=trainers,
        )

        assert [call.kwargs["force_sync"] for call in trainers["a"].save_model.await_args_list] == [True]
        assert [call.kwargs["force_sync"] for call in trainers["b"].save_model.await_args_list] == [True]

    async def test_a_run_without_a_save_rhythm_snapshots_nothing(self):
        # [FUNCTION DOC] With 2 rollouts and no save directory or interval, neither trainer's save
        # nor the executor's save is ever awaited.
        """Parking every policy on a round that saves nothing would cost the run its whole overlap."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        context = await _run(_make_args(num_rollout=2, save=None, save_interval=None), trainers=trainers)

        trainers["a"].save_model.assert_not_awaited()
        trainers["b"].save_model.assert_not_awaited()
        context["rollout_executor"].save.assert_not_awaited()


class TestARunThatCancellationCannotEnd:
    # [CLASS DOC] Test that ending the run does not depend on cancellation reaching a follower.
    async def test_a_follower_that_absorbs_cancellation_still_stops(self):
        # [FUNCTION DOC] The follower's train swallows its first cancellation (setting a flag) and
        # only re-raises on a second; the executor's get sleeps 0.01 seconds and returns a none data
        # reference. With one rollout and a 10 second overall timeout, the run must finish and the
        # flag must show a cancellation was absorbed.
        """A follower loops without bound, so ending the run must not depend on cancellation reaching it."""
        absorbed = False

        async def _train(rollout_id: int, rollout_data_ref: Any, **kwargs: Any) -> None:
            # [FUNCTION DOC] HELPER: Async fake follower train that waits forever; on the first
            # cancellation it records the absorption and returns normally, on any later
            # cancellation it re-raises.
            nonlocal absorbed
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if absorbed:
                    raise
                absorbed = True

        async def _get(rollout_id: int, trainer_model_id: str | None = None) -> dict:
            # [FUNCTION DOC] HELPER: Async fake executor get; sleeps 0.01 seconds and returns a
            # mapping with a none data reference.
            await asyncio.sleep(0.01)
            return dict(data_ref=None)

        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        trainers["b"].train = _train
        rollout_executor = AsyncMock()
        rollout_executor.get = _get

        await asyncio.wait_for(
            _run(_make_args(num_rollout=1), trainers=trainers, rollout_executor=rollout_executor), timeout=10
        )

        assert absorbed


class TestEvalDispatch:
    # [CLASS DOC] Tests of evaluation dispatch in the multi-policy driver.
    async def test_eval_fires_once_per_point_however_many_policies_train(self):
        # [FUNCTION DOC] With 2 rollouts and eval interval 2, the executor's eval was awaited with
        # rollout ids 0 and 1 and the controller prepared eval twice.
        """One shared-engine eval scores every policy through the generate chain; a per-policy dispatch would double it."""
        context = await _run(_make_args(num_rollout=2, eval_interval=2))

        eval_rollout_ids = [call.args[0] for call in context["rollout_executor"].eval.await_args_list]
        assert eval_rollout_ids == [0, 1]
        assert context["inference_controller"].prepare_eval.await_count == 2

    async def test_a_run_without_an_eval_interval_never_evaluates(self):
        # [FUNCTION DOC] Without an eval interval, the executor's eval and the controller's
        # prepare-eval are never awaited.
        """--eval-interval is the only opt-in; a surprise eval pauses production for the whole test split."""
        context = await _run(_make_args(num_rollout=2))

        context["rollout_executor"].eval.assert_not_awaited()
        context["inference_controller"].prepare_eval.assert_not_awaited()

    async def test_skip_eval_before_train_drops_only_the_starting_point(self):
        # [FUNCTION DOC] With 2 rollouts, eval interval 2 and skip-eval-before-train, the only eval
        # is for rollout 1.
        """The flag exists to skip the expensive untrained point, not to turn eval off."""
        context = await _run(_make_args(num_rollout=2, eval_interval=2, skip_eval_before_train=True))

        eval_rollout_ids = [call.args[0] for call in context["rollout_executor"].eval.await_args_list]
        assert eval_rollout_ids == [1]

    async def test_eval_holds_every_follower_parked(self, monkeypatch):
        # [FUNCTION DOC] Replaces the driver's parker class with a spy subclass whose
        # all-parked context manager sets a class-level flag while held, and gives the executor an
        # eval that records the flag value. With 2 rollouts, eval interval 2 and skip-before-train,
        # the recorded values are exactly [true], i.e. eval runs while all followers are parked.
        """A follower pushing weights mid-eval would swap its engines' weights under the running sweep."""
        held_during_eval = []

        class SpyParker(Parker):
            # [CLASS DOC] HELPER: Parker subclass exposing, through a class attribute, whether the
            # all-parked section is currently held.
            holding = False

            @asynccontextmanager
            async def with_all_parked(self):
                # [FUNCTION DOC] HELPER: Async context manager wrapping the parent's: after
                # entering it sets the holding flag true, yields, and clears the flag when
                # leaving (even on error).
                async with super().with_all_parked():
                    SpyParker.holding = True
                    try:
                        yield
                    finally:
                        SpyParker.holding = False

        async def _eval(rollout_id: int) -> None:
            # [FUNCTION DOC] HELPER: Async fake eval; records the spy's holding flag.
            held_during_eval.append(SpyParker.holding)

        rollout_executor = AsyncMock()
        rollout_executor.eval = AsyncMock(side_effect=_eval)
        monkeypatch.setattr(multi_policy_driver, "Parker", SpyParker)

        await _run(
            _make_args(num_rollout=2, eval_interval=2, skip_eval_before_train=True),
            rollout_executor=rollout_executor,
        )

        assert held_during_eval == [True]

    async def test_a_resumed_run_does_not_re_evaluate_the_untrained_model(self):
        # [FUNCTION DOC] With 2 rollouts, eval interval 2 and both policies resuming at rollout 1,
        # the only eval is for rollout 1.
        """The rollout-0 point describes the base checkpoint; a resume from rollout 1 is past it."""
        context = await _run(_make_args(num_rollout=2, eval_interval=2), start_rollout_ids={"a": 1, "b": 1})

        eval_rollout_ids = [call.args[0] for call in context["rollout_executor"].eval.await_args_list]
        assert eval_rollout_ids == [1]
