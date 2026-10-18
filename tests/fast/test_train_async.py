# [MODULE DOC] Fast tests of the asynchronous (pipelined) training driver (the top-level
# train_async script's train coroutine) using in-memory fakes that append named events to a
# shared list, with the driver's collaborators replaced via monkeypatch. Covers: API server
# handles and port gate, routing of the weight-equality check, pipelined generation ordering
# (next rollout's generation overlaps training but must settle before weight publication; the
# fully-async mode ordering with weight update intervals), critic value hand-off and release, and
# terminal lifecycle (eval draining, component disposal, worker manager shutdown order).
import asyncio
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest
import train_async as train_async_driver
from tests.fast.fixtures.driver_fakes import (
    FakeInferenceController,
    FakeObjectStore,
    FakeRolloutExecutor,
    FakeTrainingModel,
    FakeWorkerManager,
)

from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.ray import placement_group as placement_group_mod
from miles.ray import wiring
from miles.utils import object_store
from miles.utils.async_utils import with_disposer


def _make_args(**overrides: Any) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds a namespace of training arguments with inert defaults for the
    # async driver (no API port, weight-check options off, not colocated, no eval interval, eval
    # in-flight limit 2 with skip overflow policy, snapshot-based eval, not fully async, no
    # checkpoint, no old-actor keeping, zero rollouts, weight update interval 1, no critic, no
    # importance-sampling options) and then sets each keyword override as an attribute. Returns it.
    args = SimpleNamespace(
        api_server_host="127.0.0.1",
        api_server_port=None,
        check_weight_update_allow_quant_error=False,
        check_weight_update_equal=False,
        check_weight_update_selector=None,
        check_weight_update_skip_list=None,
        colocate=False,
        debug_exit_after_rollout=None,
        eval_hf_dir=None,
        eval_interval=None,
        eval_max_in_flight=2,
        eval_overflow_policy="skip",
        eval_uses_snapshots=True,
        fully_async=False,
        ft_components=[],
        hf_checkpoint=None,
        keep_old_actor=False,
        num_critic_only_steps=0,
        num_rollout=0,
        offload_train=False,
        save_hf=None,
        save_interval=None,
        save_trigger_sentinel=None,
        skip_eval_before_train=False,
        start_rollout_id=0,
        update_weights_interval=1,
        use_critic=False,
        use_rollout_logprobs=False,
        use_tis=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _install_driver_fakes(
    monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace, events: list[str]
) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Creates fake components sharing the events list (inference
    # controller, rollout executor, actor model, critic model only if the arguments use a
    # critic, a list recording API server start calls, and a placeholder cell-operations object)
    # and patches (reverted by monkeypatch) the async driver module: orchestration init does
    # nothing; rollout component creation returns the fakes and the number 4; training model
    # creation returns actor and critic; weight update records an update_weights event with the
    # rollout id; mini fault-tolerance start and rollout data reference removal do nothing. Also
    # patches the placement-group module so the API server starter records its keyword arguments
    # and the backend capability getter returns an object whose cell operations give the
    # placeholder. Returns the components namespace.
    components = SimpleNamespace(
        inference_controller=FakeInferenceController(events),
        rollout_executor=FakeRolloutExecutor(events),
        actor_model=FakeTrainingModel(events, "actor"),
        critic_model=FakeTrainingModel(events, "critic") if args.use_critic else None,
        api_server_calls=[],
        cell_operations=object(),
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
        # with the rollout id.
        events.append(f"update_weights:{rollout_id}")

    monkeypatch.setattr(train_async_driver, "init_orchestration_script", lambda _args, *, disposer: None)
    monkeypatch.setattr(train_async_driver, "create_rollout_components", create_rollout_components)
    monkeypatch.setattr(train_async_driver, "create_training_models", create_training_models)
    monkeypatch.setattr(train_async_driver, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train_async_driver, "update_weights", update_weights)
    monkeypatch.setattr(train_async_driver, "remove_rollout_data_refs", lambda *_args, **_kwargs: None)
    # the driver reaches the server through maybe_start_api_server, whose gate the tests exercise
    monkeypatch.setattr(
        placement_group_mod, "start_api_server", lambda **kwargs: components.api_server_calls.append(kwargs)
    )
    monkeypatch.setattr(
        placement_group_mod,
        "get_backend_capability",
        lambda _args: SimpleNamespace(cell_operations=lambda: components.cell_operations),
    )
    return components


class TestApiServer:
    # [CLASS DOC] Tests of the API server start in the async driver.
    async def test_api_server_receives_the_refactored_driver_handles(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With API port 8123 and the rollout fault-tolerance component, exactly one
        # API server start is recorded, its trainer models contain only the actor (the very fake
        # object), it receives the live inference controller, the port 8123 and the component list.
        """The API server acts on the live actor and inference controller, so it must be handed those objects."""
        events: list[str] = []
        args = _make_args(api_server_port=8123, ft_components=["rollout"])
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_async_driver.train, args)

        (call,) = components.api_server_calls
        assert list(call["trainer_models"]) == ["actor"]
        assert call["trainer_models"]["actor"] is components.actor_model
        assert call["inference_controller"] is components.inference_controller
        assert call["port"] == 8123
        assert call["ft_components"] == ["rollout"]

    async def test_no_api_server_without_a_port(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With no API port, no API server start is recorded.
        """An unrequested API server would expose a control plane the operator never asked for."""
        events: list[str] = []
        args = _make_args(api_server_port=None)
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_async_driver.train, args)

        assert components.api_server_calls == []


class TestWeightEqualityCheck:
    # [CLASS DOC] Tests of the weight-update equality check option in the async driver.
    async def test_weight_equality_check_is_routed_to_the_inference_controller(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With the equality check, quantization error allowance, selector and skip
        # list set, the controller's recorded check-weights calls are exactly one compare action
        # carrying those options.
        """--check-weight-update-equal must reach the inference controller with every comparison option intact."""
        events: list[str] = []
        args = _make_args(
            check_weight_update_equal=True,
            check_weight_update_allow_quant_error=True,
            check_weight_update_selector="layers.0",
            check_weight_update_skip_list=["lm_head", "embed_tokens"],
        )
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_async_driver.train, args)

        assert components.inference_controller.check_weights_calls == [
            dict(
                action="compare",
                allow_quant_error=True,
                selector="layers.0",
                skip_list=["lm_head", "embed_tokens"],
            )
        ]


class TestPipelinedGeneration:
    # [CLASS DOC] Tests of the ordering between generation, training and weight publication.
    async def test_inflight_next_rollout_finishes_before_weight_publication(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With 2 rollouts, weight update interval 1, and generation of rollout 1
        # held by an event gate, runs the driver as a task and waits (10 second timeout) for actor
        # training of rollout 0 to start. Asserts generation 1 has started but not finished and
        # weights 0 have not been published; then releases the gate, waits for the driver to
        # finish (10 second timeout), and asserts generation 1 started before actor training 0,
        # finished before weight update 0, and the actor trained rollouts 0 and 1.
        """Generation for the next rollout starts while this one trains, but must settle before new weights ship."""
        events: list[str] = []
        args = _make_args(num_rollout=2, update_weights_interval=1)
        components = _install_driver_fakes(monkeypatch, args, events)
        held_generation = asyncio.Event()
        components.rollout_executor.generation_gates[1] = held_generation

        driver = asyncio.create_task(with_disposer(train_async_driver.train, args))
        await asyncio.wait_for(components.actor_model.train_started[0].wait(), timeout=10)

        assert "generate_start:1" in events
        assert "generate_done:1" not in events
        assert "update_weights:0" not in events

        held_generation.set()
        await asyncio.wait_for(driver, timeout=10)

        assert events.index("generate_start:1") < events.index("actor_train:0")
        assert events.index("generate_done:1") < events.index("update_weights:0")
        assert components.actor_model.trained == [0, 1]

    async def test_fully_async_next_drain_starts_after_weight_publication(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Fully-async with 2 rollouts and interval 1: actor training 0 precedes
        # weight update 0, which precedes generation start 1; generation 2 never starts; the actor
        # trained rollouts 0 and 1.
        """The persistent producer needs no lookahead drain that captures the previous weight version."""
        events: list[str] = []
        args = _make_args(fully_async=True, num_rollout=2, update_weights_interval=1)
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_async_driver.train, args)

        assert events.index("actor_train:0") < events.index("update_weights:0")
        assert events.index("update_weights:0") < events.index("generate_start:1")
        assert "generate_start:2" not in events
        assert components.actor_model.trained == [0, 1]

    async def test_fully_async_keeps_lookahead_between_weight_updates(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Fully-async with 3 rollouts and weight update interval 2: generation 1
        # starts before actor training 0; actor training 1 precedes weight update 1, which precedes
        # generation start 2; the actor trained rollouts 0, 1 and 2.
        """A drain may still overlap training when that step cannot change the current weight version."""
        events: list[str] = []
        args = _make_args(fully_async=True, num_rollout=3, update_weights_interval=2)
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_async_driver.train, args)

        assert events.index("generate_start:1") < events.index("actor_train:0")
        assert events.index("actor_train:1") < events.index("update_weights:1")
        assert events.index("update_weights:1") < events.index("generate_start:2")
        assert components.actor_model.trained == [0, 1, 2]


class TestCriticValuesHandoff:
    # [CLASS DOC] Tests of the hand-off of critic outputs to the actor in the async driver.
    async def test_async_critic_outputs_reach_the_actor_and_are_released_after_training(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] With one rollout, a critic and old-actor keeping, installs a fake object
        # store as the global store, stores a critic-values string and sets it as the critic's
        # train output for rollout 0. The actor's consume hook is replaced by one asserting it
        # receives that exact list and that the stored value is still retrievable. After the run
        # asserts the store recorded that reference as consumed and no longer contains it.
        """Pipelined critic value references stay live through the actor step that consumes them."""
        events: list[str] = []
        args = _make_args(num_rollout=1, use_critic=True, keep_old_actor=True)
        components = _install_driver_fakes(monkeypatch, args, events)
        store = FakeObjectStore()
        monkeypatch.setattr(object_store, "_INSTANCE", store)
        ref = store.put("critic-values")
        values = [TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=ref)]
        components.critic_model.train_outputs[0] = values

        def consume_critic_values(external_data: list[TrainStepOutput]) -> None:
            # [FUNCTION DOC] HELPER: Replacement consume hook; asserts the data is the identical
            # list and that the referenced stored value is still the expected string.
            assert external_data is values
            assert store.get(external_data[0].values).value == "critic-values"

        components.actor_model.consume_external_data = consume_critic_values

        await with_disposer(train_async_driver.train, args)

        assert store.consumed == [ref]
        assert not store.contains(ref)


class TestTerminalLifecycle:
    # [CLASS DOC] Tests of end-of-run behavior in the async driver.
    async def test_async_train_drains_eval_and_disposes_all_component_controllers(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] With a critic, old-actor keeping, eval interval 1 and an HF checkpoint
        # path, runs the driver and asserts eval 0 happened and the sorted dispose events are
        # exactly actor, critic, executor and inference controller disposals.
        """A run ends only once its in-flight eval has settled and every component it created is disposed."""
        events: list[str] = []
        args = _make_args(use_critic=True, keep_old_actor=True, eval_interval=1, hf_checkpoint="/ckpt/hf")
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_async_driver.train, args)

        assert "eval:0" in events
        assert sorted(event for event in events if event.endswith("_dispose")) == [
            "actor_dispose",
            "critic_dispose",
            "executor_dispose",
            "inference_dispose",
        ]

    async def test_async_train_releases_its_worker_manager_after_eval_and_component_cleanup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Replaces the orchestration init with one registering the real worker
        # manager shutdown for a fake manager and replaces the actor-kill call in the wiring module
        # with the fake's kill. After the run asserts the manager was killed once, eval 0 precedes
        # the manager shutdown event, every component dispose event precedes it, and it precedes
        # the manager kill event.
        """The async driver releases its own manager after eval draining and component cleanup."""
        events: list[str] = []
        args = _make_args(use_critic=True, keep_old_actor=True, eval_interval=1, hf_checkpoint="/ckpt/hf")
        manager = FakeWorkerManager(events)
        _install_driver_fakes(monkeypatch, args, events)

        def init_orchestration_script(_args: SimpleNamespace, *, disposer: Any) -> FakeWorkerManager:
            # [FUNCTION DOC] HELPER: Fake init; adds the wiring shutdown function bound to the
            # fake manager onto the disposer and returns the manager.
            disposer.add(partial(wiring.shutdown_worker_manager, manager))
            return manager

        monkeypatch.setattr(train_async_driver, "init_orchestration_script", init_orchestration_script)
        monkeypatch.setattr(wiring.ray, "kill", manager.kill)

        await with_disposer(train_async_driver.train, args)

        assert manager.killed == [manager]
        assert events.index("eval:0") < events.index("manager_shutdown")
        assert all(
            events.index(event) < events.index("manager_shutdown") for event in events if event.endswith("_dispose")
        )
        assert events.index("manager_shutdown") < events.index("manager_kill")
