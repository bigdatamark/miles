# [MODULE DOC] Fast tests of the synchronous training driver (the top-level train script's train
# coroutine) using in-memory fakes (inference controller, rollout executor, training models,
# object store, worker manager) that append named events to a shared list, with the driver's
# collaborators replaced via monkeypatch. Covers: eval-only runs, evaluation before training on
# fresh and resumed runs, final eval on and off cadence (including the handoff and onload order),
# routing of the weight-equality check option, handing critic outputs to the actor and releasing
# them, API server wiring only when a port is set, and disposal ordering (components disposed
# before the worker manager is released and killed).
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest
import train as train_driver
from tests.fast.fixtures.driver_fakes import (
    FakeInferenceController,
    FakeObjectStore,
    FakeRolloutExecutor,
    FakeTrainingModel,
    FakeWorkerManager,
)

from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.ray import placement_group, wiring
from miles.utils import object_store
from miles.utils.async_utils import with_disposer


def _make_args(**overrides: Any) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds a namespace of training arguments with inert defaults (local
    # API server host, no port, weight-check options off, no eval interval, no fully-async, a base
    # checkpoint path, zero rollouts and critic-only steps, no offload, no save triggers, start
    # rollout id 0, no critic) and then sets each keyword override as an attribute (adding new
    # attributes is allowed). Returns the namespace.
    args = SimpleNamespace(
        api_server_host="127.0.0.1",
        api_server_port=None,
        check_weight_update_allow_quant_error=False,
        check_weight_update_equal=False,
        check_weight_update_selector=None,
        check_weight_update_skip_list=None,
        colocate_memory_peak_device="cpu",
        debug_exit_after_rollout=None,
        eval_interval=None,
        eval_uses_snapshots=False,
        ft_components=[],
        fully_async=False,
        hf_checkpoint="/base/checkpoint",
        num_critic_only_steps=0,
        num_rollout=0,
        offload_rollout=False,
        offload_rollout_level="",
        offload_train=False,
        save_interval=None,
        save_trigger_sentinel=None,
        skip_eval_before_train=False,
        start_rollout_id=0,
        use_critic=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _install_driver_fakes(
    monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace, events: list[str]
) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Creates fake components sharing the events list (inference
    # controller, rollout executor, actor training model, and a critic training model only when
    # the arguments use a critic) and patches the driver module (reverted by monkeypatch): the
    # orchestration init does nothing and returns none; rollout component creation returns the
    # fake controller, executor and the number 4; training model creation returns actor and
    # critic; the weight update records an update_weights event with the rollout id; mini fault
    # tolerance controller start returns none; and removing rollout data references does nothing.
    # Returns the components namespace.
    components = SimpleNamespace(
        inference_controller=FakeInferenceController(events),
        rollout_executor=FakeRolloutExecutor(events),
        actor_model=FakeTrainingModel(events, "actor"),
        critic_model=FakeTrainingModel(events, "critic") if args.use_critic else None,
    )

    async def create_rollout_components(_args: SimpleNamespace) -> tuple[Any, Any, int]:
        # [FUNCTION DOC] HELPER: Async fake; returns the fake inference controller, rollout
        # executor and the constant 4.
        return components.inference_controller, components.rollout_executor, 4

    async def create_training_models(_args: SimpleNamespace, _executor: Any) -> tuple[Any, Any]:
        # [FUNCTION DOC] HELPER: Async fake; returns the fake actor and critic (possibly none).
        return components.actor_model, components.critic_model

    async def update_weights(
        _args: Any, _model: Any, _executor: Any, _inference_controller: Any, *, rollout_id: int | None = None
    ) -> None:
        # [FUNCTION DOC] HELPER: Async fake weight update; appends an update_weights event tagged
        # with the rollout id to the events list.
        events.append(f"update_weights:{rollout_id}")

    monkeypatch.setattr(train_driver, "init_orchestration_script", lambda _args, *, disposer: None)
    monkeypatch.setattr(train_driver, "create_rollout_components", create_rollout_components)
    monkeypatch.setattr(train_driver, "create_training_models", create_training_models)
    monkeypatch.setattr(train_driver, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train_driver, "update_weights", update_weights)
    monkeypatch.setattr(train_driver, "remove_rollout_data_refs", lambda *_args, **_kwargs: None)
    return components


class TestEvalOnlyRun:
    # [CLASS DOC] Tests of a run with zero rollouts.
    async def test_eval_only_prepares_inference_and_runs_exactly_one_eval(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With zero rollouts and eval interval 2, runs the driver and asserts one
        # prepare-eval event and one eval for rollout 0 occurred in that order, the actor trained
        # nothing, and no prepare-rollout or generate-start events happened.
        """A run with no rollouts but an eval interval evaluates once and generates or trains nothing."""
        events: list[str] = []
        args = _make_args(num_rollout=0, eval_interval=2)
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, args)

        assert events.count("prepare_eval") == 1
        assert events.count("eval:0") == 1
        assert events.index("prepare_eval") < events.index("eval:0")
        assert components.actor_model.trained == []
        assert not [event for event in events if event.startswith(("prepare_rollout", "generate_start"))]


class TestEvalBeforeTrain:
    # [CLASS DOC] Tests of the evaluation done before the first training step.
    async def test_fresh_run_evaluates_the_initial_checkpoint_at_step_zero(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With 2 rollouts and eval interval 1 from a fresh start, eval 0 happens
        # before preparing rollout 0.
        events: list[str] = []
        args = _make_args(num_rollout=2, eval_interval=1)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, args)

        assert events.index("eval:0") < events.index("prepare_rollout:0")

    async def test_resumed_run_evaluates_the_completed_rollout_not_the_next_one(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With 5 rollouts, eval interval 1 and start rollout id 3, eval 2 (the
        # completed rollout) occurs before preparing rollout 3 and eval 3 does not occur before
        # that point.
        """start_rollout_id is the loaded checkpoint's rollout plus one, so dispatching it
        would ask for a checkpoint that does not exist yet and, in staging mode, export
        into the directory the first resumed iteration goes on to overwrite."""
        events: list[str] = []
        args = _make_args(num_rollout=5, eval_interval=1, start_rollout_id=3)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, args)

        assert events.index("eval:2") < events.index("prepare_rollout:3")
        assert "eval:3" not in events[: events.index("prepare_rollout:3")]


class TestFinalEval:
    # [CLASS DOC] Tests of the evaluation at the end of the run.
    async def test_the_last_rollout_is_always_evaluated_even_off_cadence(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With 3 rollouts and eval interval 2, the eval events are exactly eval 0,
        # eval 1 and eval 2 in order (the final rollout is evaluated though off cadence).
        """The final point carries force=True because training is over and backpressure is
        free, but the cadence check has to reach it first: with num_rollout not a multiple
        of eval_interval, the final weights would otherwise never be measured."""
        events: list[str] = []
        args = _make_args(num_rollout=3, eval_interval=2)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, args)

        assert [event for event in events if event.startswith("eval:")] == ["eval:0", "eval:1", "eval:2"]

    async def test_an_off_cadence_final_eval_follows_the_last_handoff(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With 3 rollouts, eval interval 2 and rollout offload on, the events
        # between actor training of rollout 2 and eval 2 contain onload weights, then the weight
        # update for rollout 2, then onload of the key-value cache, in that order.
        """The last rollout skips its handoff only when no final eval follows; an off-cadence
        final eval must still see the engine onloaded with the trained weights."""
        events: list[str] = []
        args = _make_args(num_rollout=3, eval_interval=2, offload_rollout=True)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, args)

        final = events[events.index("actor_train:2") : events.index("eval:2")]
        assert final.index("onload_weights") < final.index("update_weights:2") < final.index("onload_kv")

    async def test_the_last_rollout_skips_the_handoff_without_an_eval(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With 3 rollouts, no eval interval and offload on, weight update 1 happens
        # but weight update 2 does not and no onload of weights follows actor training of rollout 2.
        events: list[str] = []
        args = _make_args(num_rollout=3, offload_rollout=True)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, args)

        assert "update_weights:1" in events
        assert "update_weights:2" not in events
        assert "onload_weights" not in events[events.index("actor_train:2") :]


class TestWeightEqualityCheck:
    # [CLASS DOC] Tests of the weight-update equality check option.
    async def test_weight_equality_check_is_routed_to_the_inference_controller(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With the equality check, quantization error allowance, a selector and a
        # skip list set, the controller's recorded check-weights calls are exactly one compare
        # action carrying those options.
        """--check-weight-update-equal must reach the inference controller with every comparison option intact."""
        events: list[str] = []
        args = _make_args(
            check_weight_update_equal=True,
            check_weight_update_allow_quant_error=True,
            check_weight_update_selector="layers.0",
            check_weight_update_skip_list=["lm_head", "embed_tokens"],
        )
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, args)

        assert components.inference_controller.check_weights_calls == [
            dict(
                action="compare",
                allow_quant_error=True,
                selector="layers.0",
                skip_list=["lm_head", "embed_tokens"],
            )
        ]

    async def test_no_weight_comparison_without_the_flag(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With the equality check off, the controller records no check-weights calls.
        """The comparison reloads weights on every engine, so an ordinary run must never trigger it."""
        events: list[str] = []
        args = _make_args(check_weight_update_equal=False)
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, args)

        assert components.inference_controller.check_weights_calls == []


class TestCriticValuesHandoff:
    # [CLASS DOC] Tests of the hand-off of critic outputs to the actor.
    async def test_critic_outputs_reach_the_actor_and_are_released_after_training(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] With one rollout and a critic, installs a fake object store as the global
        # store, puts a critic-values payload in it and sets it as the critic's train output for
        # rollout 0. The actor's consume hook is replaced by one asserting it receives that exact
        # list and that the stored payload is still retrievable at that time. After the run asserts
        # the store recorded exactly that reference as consumed and no longer contains it.
        """Critic value references reach actor training and remain live until it consumes them."""
        events: list[str] = []
        args = _make_args(num_rollout=1, use_critic=True)
        components = _install_driver_fakes(monkeypatch, args, events)
        store = FakeObjectStore()
        monkeypatch.setattr(object_store, "_INSTANCE", store)
        ref = store.put({"values": ["critic-values"]})
        values = [TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=ref)]
        components.critic_model.train_outputs[0] = values

        def consume_critic_values(external_data: list[TrainStepOutput]) -> None:
            # [FUNCTION DOC] HELPER: Replacement consume hook; asserts the data is the identical
            # list and that the referenced stored value is still the expected payload.
            assert external_data is values
            assert store.get(external_data[0].values).value == {"values": ["critic-values"]}

        components.actor_model.consume_external_data = consume_critic_values

        await with_disposer(train_driver.train, args)

        assert store.consumed == [ref]
        assert not store.contains(ref)


class TestApiServerWiring:
    # [CLASS DOC] Tests of the API server start gate in the sync driver.
    async def test_api_server_receives_the_sync_driver_handles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Replaces the placement-group module's API server starter with a recorder
        # and its backend capability getter with one returning cell operations. Runs the driver
        # with no API port and asserts nothing was started; then with port 8080 and asserts one
        # start with the trainer models mapping holding just the actor and the live inference
        # controller.
        """Only an enabled API server receives the live actor and inference controller handles."""
        started_with: list[dict[str, Any]] = []
        monkeypatch.setattr(placement_group, "start_api_server", lambda **kwargs: started_with.append(kwargs))
        monkeypatch.setattr(
            placement_group,
            "get_backend_capability",
            lambda args: SimpleNamespace(cell_operations=lambda: object()),
        )
        disabled_events: list[str] = []
        disabled_args = _make_args(api_server_port=None)
        _install_driver_fakes(monkeypatch, disabled_args, disabled_events)
        await with_disposer(train_driver.train, disabled_args)
        assert started_with == []

        enabled_events: list[str] = []
        enabled_args = _make_args(api_server_port=8080)
        components = _install_driver_fakes(monkeypatch, enabled_args, enabled_events)
        await with_disposer(train_driver.train, enabled_args)

        assert len(started_with) == 1
        assert started_with[0]["trainer_models"] == {"actor": components.actor_model}
        assert started_with[0]["inference_controller"] is components.inference_controller


class TestTerminalLifecycle:
    # [CLASS DOC] Tests of end-of-run disposal.
    async def test_train_disposes_all_created_component_controllers(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With a critic, the sorted dispose events are exactly actor, critic,
        # executor and inference controller disposals.
        """Every component the driver created must be disposed, or its watchers outlive the run."""
        events: list[str] = []
        args = _make_args(use_critic=True)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, args)

        assert sorted(event for event in events if event.endswith("_dispose")) == [
            "actor_dispose",
            "critic_dispose",
            "executor_dispose",
            "inference_dispose",
        ]

    async def test_train_releases_its_worker_manager_after_component_disposal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Replaces the orchestration init with one that registers the real worker
        # manager shutdown on the disposer for a fake manager, and replaces the actor-kill call in
        # the wiring module with the fake's kill. After the run asserts the fake was killed once
        # and every component dispose event precedes the manager shutdown event, which precedes
        # the manager kill event.
        """The driver releases its own manager only after every component has been disposed."""
        events: list[str] = []
        args = _make_args(use_critic=True)
        manager = FakeWorkerManager(events)
        _install_driver_fakes(monkeypatch, args, events)

        def init_orchestration_script(_args: SimpleNamespace, *, disposer: Any) -> FakeWorkerManager:
            # [FUNCTION DOC] HELPER: Fake init; adds the wiring shutdown function bound to the
            # fake manager onto the disposer and returns the manager.
            disposer.add(partial(wiring.shutdown_worker_manager, manager))
            return manager

        monkeypatch.setattr(train_driver, "init_orchestration_script", init_orchestration_script)
        monkeypatch.setattr(wiring.ray, "kill", manager.kill)

        await with_disposer(train_driver.train, args)

        assert manager.killed == [manager]
        assert all(
            events.index(event) < events.index("manager_shutdown") for event in events if event.endswith("_dispose")
        )
        assert events.index("manager_shutdown") < events.index("manager_kill")
