# [MODULE DOC]
# Tests for the Megatron train actor's lifecycle in the shared PPO setup (actor and critic
# sharing GPUs, with the driver controlling offload): train/sleep/wake-up behavior, log-prob
# forward options, model save sequencing, weight-update connection and reconnection logic, model
# weight switching between tags, skipping the actor's own log-prob forward, replay managers, and
# the critic-to-actor values hand-off through an object store. The actor module is imported under
# stubs for the GPU memory saver and peer-to-peer weight updater (module-scoped fixture); worker
# objects are built without running constructors and filled with mocks, so no GPU or real Ray is
# needed.
import importlib
import sys
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import pytest
import ray
import torch

from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.backends.training_utils.weight_update.conn_status import ConnStatusManager
from miles.utils import object_store
from miles.utils.ray_utils import Box
from miles.utils.replay_base import IndexerReplayManager, RoutingReplayManager
from miles.utils.tensor_backper import MainCastContext, TensorBackuper


@pytest.fixture(scope="module")
def actor_module():
    # [FUNCTION DOC] HELPER: Module-scoped fixture. Imports the megatron backend package and the
    # weight-update protocols package, remembers the current module-table entries and package
    # attributes for the actor module, the peer-to-peer protocol module and the GPU memory saver,
    # then installs a stub memory-saver module (with a mock saver), a stub peer-to-peer module
    # whose updater class raises an assertion error if constructed, attaches that stub to its
    # package, and drops the cached actor module and attribute so it is imported fresh. Yields the
    # freshly imported actor module. On teardown it removes the fresh actor module and restores
    # every remembered entry or attribute (deleting those that did not exist before).
    actor_module_name = "miles.backends.megatron_utils.actor"
    p2p_module_name = "miles.backends.training_utils.weight_update.protocols.p2p"
    actor_package = importlib.import_module("miles.backends.megatron_utils")
    p2p_package = importlib.import_module("miles.backends.training_utils.weight_update.protocols")
    missing = object()
    saved_actor_module = sys.modules.get(actor_module_name, missing)
    saved_p2p_module = sys.modules.get(p2p_module_name, missing)
    saved_saver = sys.modules.get("torch_memory_saver", missing)
    saved_actor_package_attr = getattr(actor_package, "actor", missing)
    saved_p2p_package_attr = getattr(p2p_package, "p2p", missing)

    saver_module = ModuleType("torch_memory_saver")
    saver_module.torch_memory_saver = Mock()
    p2p_module = ModuleType(p2p_module_name)
    p2p_module.UpdateWeightP2P = Mock(
        side_effect=AssertionError("shared PPO lifecycle tests must not construct UpdateWeightP2P")
    )
    sys.modules["torch_memory_saver"] = saver_module
    sys.modules[p2p_module_name] = p2p_module
    p2p_package.p2p = p2p_module
    sys.modules.pop(actor_module_name, None)
    if saved_actor_package_attr is not missing:
        delattr(actor_package, "actor")

    try:
        yield importlib.import_module(actor_module_name)
    finally:
        sys.modules.pop(actor_module_name, None)
        if saved_actor_module is not missing:
            sys.modules[actor_module_name] = saved_actor_module
        if saved_actor_package_attr is missing:
            if hasattr(actor_package, "actor"):
                delattr(actor_package, "actor")
        else:
            actor_package.actor = saved_actor_package_attr
        sys.modules.pop(p2p_module_name, None)
        if saved_p2p_module is not missing:
            sys.modules[p2p_module_name] = saved_p2p_module
        if saved_p2p_package_attr is missing:
            if hasattr(p2p_package, "p2p"):
                delattr(p2p_package, "p2p")
        else:
            p2p_package.p2p = saved_p2p_package_attr
        if saved_saver is missing:
            sys.modules.pop("torch_memory_saver", None)
        else:
            sys.modules["torch_memory_saver"] = saved_saver


def _worker(actor_module, role, *, asleep=True):
    # [FUNCTION DOC] HELPER: Creates a train-actor instance without running its constructor, with
    # arguments (offload train on, not rollout-only), the given role, the given sleeping flag
    # (default asleep), and mocks for its heartbeat, wake-up and sleep methods. Returns it.
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(offload_train=True, debug_rollout_only=False)
    worker.role = role
    worker._asleep = asleep
    worker._heartbeat = Mock()
    worker.wake_up = Mock()
    worker.sleep = Mock()
    return worker


def test_critic_train_wakes_and_leaves_offload_to_driver(actor_module, monkeypatch):
    # [FUNCTION DOC] A sleeping critic worker whose critic-train method returns a normal step
    # output carrying a boxed values reference; rollout-data loading is patched to return a small
    # batch and a null context; the timer is patched to record phase names. Runs train for
    # rollout 3. Asserts: wake-up called once without arguments, the critic-train called once,
    # sleep never called (the driver owns offload), the very same output returned with outcome
    # normal and the same boxed value, and recorded phases exactly data_preprocess then
    # critic_train.
    worker = _worker(actor_module, "critic")
    critic_output = TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=Box("cpu-values-ref"))
    worker._train_critic = Mock(return_value=critic_output)
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )
    phases = []

    @contextmanager
    def capture_timer(name):
        # [FUNCTION DOC] HELPER: Context manager recording the phase name in the enclosing list
        # and yielding immediately.
        phases.append(name)
        yield

    monkeypatch.setattr(actor_module, "timer", capture_timer)

    result = worker.train(3, object())

    worker.wake_up.assert_called_once_with()
    worker._train_critic.assert_called_once()
    worker.sleep.assert_not_called()
    assert result is critic_output
    assert result.outcome is TrainStepOutcome.NORMAL
    assert result.values.inner == "cpu-values-ref"
    assert phases == ["data_preprocess", "critic_train"]


def test_actor_receives_critic_payload_and_leaves_offload_to_driver(actor_module, monkeypatch):
    # [FUNCTION DOC] A sleeping actor worker with a mocked actor-train method. Runs train for
    # rollout 4 with a critic output passed as external data. Asserts wake-up called once, the
    # actor-train called once with that same object as its external-data keyword, sleep never
    # called, and the result is none.
    worker = _worker(actor_module, "actor")
    worker._train_actor = Mock(return_value=None)
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )
    values = TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=Box("cpu-values-ref"))

    result = worker.train(4, object(), external_data=values)

    worker.wake_up.assert_called_once_with()
    worker._train_actor.assert_called_once()
    assert worker._train_actor.call_args.kwargs["external_data"] is values
    worker.sleep.assert_not_called()
    assert result is None


def test_train_keeps_model_resident(actor_module, monkeypatch):
    # [FUNCTION DOC] An actor worker that is already awake runs train for rollout 5; asserts
    # neither wake-up nor sleep is called.
    worker = _worker(actor_module, "actor", asleep=False)
    worker._train_actor = Mock(return_value=None)
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )

    worker.train(5, object())

    worker.wake_up.assert_not_called()
    worker.sleep.assert_not_called()


@pytest.mark.parametrize(
    ("store_prefix", "expected"),
    [("", True), ("ref_", False), ("teacher_", False)],
)
def test_compute_log_prob_replays_sampling_support_only_for_actor_scores(
    actor_module, monkeypatch, store_prefix, expected
):
    # [FUNCTION DOC] Parametrized over the store prefix: empty (actor scores) expects sampling
    # support replay on; "ref_" and "teacher_" expect it off. A worker with sampling-support
    # replay enabled and the forward-only function mocked to return a log-probs mapping computes
    # log probs for rollout 3. Asserts the mocked result is returned, the forward was called with
    # fp32 output off, and with the rollout-sampling-mask flag equal to the expectation.
    """Policy log-prob forwards preserve the model's logits precision."""
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(use_sampling_support_replay=True)
    worker.model = [object()]
    forward_only = Mock(return_value={"log_probs": []})
    monkeypatch.setattr(actor_module, "forward_only", forward_only)
    monkeypatch.setattr(actor_module, "timer", lambda _name: nullcontext())

    result = worker._compute_log_prob([], [], rollout_id=3, store_prefix=store_prefix)

    assert result == {"log_probs": []}
    assert forward_only.call_args.kwargs["fp32_output"] is False
    assert forward_only.call_args.kwargs["use_rollout_sampling_mask"] is expected


def test_save_model_does_not_manage_lifecycle(actor_module, monkeypatch):
    # [FUNCTION DOC] A worker with synchronous save, no HuggingFace save and no post-save hook,
    # with mocked save, process-group reload and destroy. Saves for rollout 6. Asserts the save
    # function was called once with rollout 6, model, optimizer, scheduler and the snapshot
    # publisher keyword, and that wake-up, sleep and both process-group functions were not called.
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(
        async_save=False,
        custom_megatron_post_save_hook_path=None,
        debug_rollout_only=False,
        save_hf=None,
    )
    worker.role = "actor"
    worker._heartbeat = Mock()
    worker.model = object()
    worker.optimizer = object()
    worker.opt_param_scheduler = object()
    worker.snapshot_publisher = object()
    worker.wake_up = Mock()
    worker.sleep = Mock()
    save = Mock()
    reload_groups = Mock()
    destroy_groups = Mock()
    monkeypatch.setattr(actor_module, "save", save)
    monkeypatch.setattr(actor_module, "reload_process_groups", reload_groups)
    monkeypatch.setattr(actor_module, "destroy_process_groups", destroy_groups)

    worker.save_model(6)

    save.assert_called_once_with(
        6, worker.model, worker.optimizer, worker.opt_param_scheduler, snapshot_publisher=worker.snapshot_publisher
    )
    worker.wake_up.assert_not_called()
    worker.sleep.assert_not_called()
    reload_groups.assert_not_called()
    destroy_groups.assert_not_called()


def test_force_sync_save_overlaps_hf_export_with_async_checkpoint(actor_module, monkeypatch):
    # [FUNCTION DOC] A worker configured for asynchronous save plus a HuggingFace save path. The
    # save, async-finalize and HuggingFace-save functions are replaced by recorders. Saving for
    # rollout 6 with force-sync asserts the event order: finalize, save, save_hf, finalize (finish
    # earlier async work, start the checkpoint, export HF while it runs, finish it).
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(
        async_save=True,
        custom_megatron_post_save_hook_path=None,
        debug_rollout_only=False,
        save_hf="/checkpoints/hf/{rollout_id}",
    )
    worker.role = "actor"
    worker._heartbeat = Mock()
    worker.model = object()
    worker.optimizer = object()
    worker.opt_param_scheduler = object()
    worker.snapshot_publisher = object()
    events = []

    monkeypatch.setattr(actor_module, "save", lambda *_args, **_kwargs: events.append("save"))

    monkeypatch.setattr(
        actor_module,
        "maybe_finalize_async_save",
        lambda **_kwargs: events.append("finalize"),
    )
    monkeypatch.setattr(actor_module, "save_hf_model", lambda *_args, **_kwargs: events.append("save_hf"))

    worker.save_model(6, force_sync=True)

    assert events == ["finalize", "save", "save_hf", "finalize"]


@pytest.mark.parametrize("asleep", [False, True])
def test_update_weights_only_uses_temporary_process_groups_when_asleep(actor_module, monkeypatch, asleep):
    # [FUNCTION DOC] Parametrized over the sleeping state. A worker with offload enabled, weight
    # update skipped by debug flag, and a mock weight updater whose connection status never needs
    # reconnect, updates weights with empty rollout-engine info as rank 1. Asserts the process
    # group reload and destroy functions were each called once when asleep and not at all when
    # awake.
    """Weight update reloads and destroys temporary process groups only when the model is offloaded."""
    from miles.ray.rollout.inference_controller import UpdatableEngines

    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(
        debug_rollout_only=False,
        debug_skip_weight_update=True,
        debug_train_only=False,
        offload_train=True,
        rematerialize_param_from_master_weight=False,
    )
    worker._asleep = asleep
    worker._heartbeat = Mock()
    worker.weight_updater = Mock()
    worker.weight_updater.conn_status = Mock(spec=ConnStatusManager)
    worker.weight_updater.conn_status.needs_reconnect.return_value = False
    info = UpdatableEngines(
        rollout_engines=[],
        engine_gpu_counts=[],
        engine_gpu_offsets=[],
        snapshot_cell_id_to_hashes={},
    )
    reload_groups = Mock()
    destroy_groups = Mock()
    monkeypatch.setattr(actor_module, "reload_process_groups", reload_groups)
    monkeypatch.setattr(actor_module, "destroy_process_groups", destroy_groups)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 1)

    worker.update_weights(info)

    assert reload_groups.call_count == int(asleep)
    assert destroy_groups.call_count == int(asleep)


def _lifecycle_worker(actor_module, monkeypatch, asleep):
    # [FUNCTION DOC] HELPER: Creates a worker (no constructor) with offload on, no rematerialize,
    # no quantized-workspace clearing, the given sleeping flag and no paused grad buffer. Patches
    # the actor module's memory saver (mock), memory clearing and printing (mocks), process group
    # destroy (mock) and reload (a kept mock), non-first-replica main rank check (false) and LoRA
    # enabled check (false). Returns the worker, the saver mock and the reload mock.
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(
        offload_train=True,
        rematerialize_param_from_master_weight=False,
        clear_quantized_weight_workspaces_on_offload=False,
    )
    worker._asleep = asleep
    worker._grad_buffer_paused = False
    saver = Mock()
    reload_groups = Mock()
    monkeypatch.setattr(actor_module, "torch_memory_saver", saver)
    monkeypatch.setattr(actor_module, "clear_memory", Mock())
    monkeypatch.setattr(actor_module, "print_memory", Mock())
    monkeypatch.setattr(actor_module, "destroy_process_groups", Mock())
    monkeypatch.setattr(actor_module, "reload_process_groups", reload_groups)
    monkeypatch.setattr(actor_module, "is_first_replica_megatron_main_rank", lambda: False)
    monkeypatch.setattr(actor_module, "is_lora_enabled", lambda _args: False)
    return worker, saver, reload_groups


def test_sleep_is_idempotent(actor_module, monkeypatch):
    # [FUNCTION DOC] An awake worker sleeps twice. Asserts the memory saver was paused exactly
    # once and the worker is marked asleep.
    worker, saver, _ = _lifecycle_worker(actor_module, monkeypatch, asleep=False)

    worker.sleep()
    worker.sleep()

    assert saver.pause.call_count == 1
    assert worker._asleep is True


def test_wake_up_when_resident_skips_resume_but_restores_groups(actor_module, monkeypatch):
    # [FUNCTION DOC] An already-awake worker wakes up. Asserts the saver is not resumed, the
    # process groups are reloaded once without arguments, and the worker stays awake.
    # A retried attempt can die between wake and sleep: memory stays resident but the
    # process groups may already be gone, so wake_up must restore groups without resuming.
    worker, saver, reload_groups = _lifecycle_worker(actor_module, monkeypatch, asleep=False)

    worker.wake_up()

    saver.resume.assert_not_called()
    reload_groups.assert_called_once_with()
    assert worker._asleep is False


def test_wake_up_resumes_offloaded_model_once(actor_module, monkeypatch):
    # [FUNCTION DOC] A sleeping worker wakes up twice. Asserts the saver was resumed exactly once
    # and the worker is awake.
    worker, saver, _ = _lifecycle_worker(actor_module, monkeypatch, asleep=True)

    worker.wake_up()
    worker.wake_up()

    assert saver.resume.call_count == 1
    assert worker._asleep is False


def test_lora_sleep_pauses_the_grad_buffers_and_wake_up_resumes_them(actor_module, monkeypatch):
    # [FUNCTION DOC] With LoRA rollout enabled (patched to true), an awake worker sleeps then
    # wakes. Asserts the saver pause calls are, in order, the grad-buffer tag then the default tag,
    # and the resume calls are, in order, the default tag then the grad-buffer tag.
    """The LoRA grad buffers have no host backup, so a sleep that leaves them out keeps them on the GPU."""
    worker, saver, _ = _lifecycle_worker(actor_module, monkeypatch, asleep=False)
    monkeypatch.setattr(actor_module, "lora_rollout_enabled", lambda _args: True)

    worker.sleep()
    worker.wake_up()

    assert saver.pause.call_args_list == [call(tag="grad_buffer"), call(tag="default")]
    assert saver.resume.call_args_list == [call(tag="default"), call(tag="grad_buffer")]


def test_offload_grad_buffer_frees_them_once_and_sleep_does_not_pause_them_again(actor_module, monkeypatch):
    # [FUNCTION DOC] With LoRA rollout enabled, an awake worker offloads its grad buffer twice,
    # sleeps and wakes. Asserts pause calls are grad-buffer then default (grad buffer paused only
    # once overall), resume calls are default then grad-buffer, and the paused flag ends false.
    """offload_grad_buffer() and sleep() must pause the grad buffers exactly once between them."""
    worker, saver, _ = _lifecycle_worker(actor_module, monkeypatch, asleep=False)
    monkeypatch.setattr(actor_module, "lora_rollout_enabled", lambda _args: True)

    worker.offload_grad_buffer()
    worker.offload_grad_buffer()
    worker.sleep()
    worker.wake_up()

    assert saver.pause.call_args_list == [call(tag="grad_buffer"), call(tag="default")]
    assert saver.resume.call_args_list == [call(tag="default"), call(tag="grad_buffer")]
    assert worker._grad_buffer_paused is False


def _actor_train_args(**overrides):
    # [FUNCTION DOC] HELPER: Returns an arguments namespace with defaults: compute advantages and
    # returns on; rollout log probs, keep-old-actor, mismatch metrics and skip-actor-forward all
    # off. Keyword overrides replace defaults (dictionary merge).
    defaults = dict(
        compute_advantages_and_returns=True,
        use_rollout_logprobs=False,
        keep_old_actor=False,
        get_mismatch_metrics=False,
        skip_actor_forward_only=False,
    )
    return Namespace(**(defaults | overrides))


def _actor_reuse_worker(actor_module, **args_overrides):
    # [FUNCTION DOC] HELPER: Creates an actor worker (no constructor) for the actor-training
    # tests: arguments from the train-args helper with no critic plus overrides, placeholder
    # model/optimizer/scheduler, a mock weights backuper with no backup tags, active tag "actor",
    # mocks for model switching, replay stage setting, log-prob computation (returning a log-probs
    # mapping), profiler, weight updater (pop-metrics returns empty) and heartbeat, and no
    # rollout-data postprocess and no fault-tolerance test executor. Returns the worker.
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = _actor_train_args(use_critic=False, **args_overrides)
    worker.model = [object()]
    worker.optimizer = object()
    worker.opt_param_scheduler = object()
    worker.weights_backuper = Mock(backup_tags=set())
    worker._active_model_tag = "actor"
    worker._switch_model = Mock()
    worker._set_replay_stage = Mock()
    worker._compute_log_prob = Mock(return_value={"log_probs": [object()]})
    worker.rollout_data_postprocess = None
    worker.prof = Mock()
    worker._ft_test_action_executor = None
    worker.weight_updater = Mock()
    worker.weight_updater.pop_metrics.return_value = {}
    worker._heartbeat = Mock()
    return worker


def _patch_actor_reuse_dependencies(actor_module, monkeypatch, *, num_microbatches):
    # [FUNCTION DOC] HELPER: Patches the actor module (via monkeypatch): no replay managers; a
    # data-iterator getter returning a one-element iterator list and the given per-step
    # microbatch counts; mocks for advantage computation, advantage event logging, rollout data
    # logging, perf logging and debug train data saving; pass-through timers; and a training
    # function mock returning the discarded-should-retry outcome.
    @contextmanager
    def passthrough_timer(_name):
        # [FUNCTION DOC] HELPER: Context manager that ignores the name and yields immediately.
        yield

    monkeypatch.setattr(actor_module, "all_replay_managers", [])
    monkeypatch.setattr(
        actor_module,
        "get_data_iterator",
        lambda *_args: ([Namespace(micro_batch_indices=None, micro_batch_size=1)], num_microbatches),
    )
    monkeypatch.setattr(actor_module, "compute_advantages_and_returns", Mock())
    monkeypatch.setattr(actor_module, "log_train_advantage_computation_event", Mock())
    monkeypatch.setattr(actor_module, "log_rollout_data", Mock())
    monkeypatch.setattr(actor_module, "log_perf_data", Mock())
    monkeypatch.setattr(actor_module.train_dump, "save_debug_train_data", Mock())
    monkeypatch.setattr(actor_module, "inverse_timer", passthrough_timer)
    monkeypatch.setattr(actor_module, "timer", passthrough_timer)
    monkeypatch.setattr(
        actor_module,
        "train",
        Mock(return_value=actor_module.TrainStepOutcome.DISCARDED_SHOULD_RETRY),
    )


@pytest.mark.parametrize(
    ("skip_actor_forward_only", "use_rollout_logprobs", "num_microbatches"),
    [
        (False, False, [1]),
        (True, False, [1]),
        (True, True, [1]),
        (True, False, [2]),
    ],
)
def test_actor_logprob_forward_is_explicit_single_step_opt_in(
    actor_module, monkeypatch, skip_actor_forward_only, use_rollout_logprobs, num_microbatches
):
    # [FUNCTION DOC] Parametrized over (skip actor forward, use rollout log probs, microbatch
    # counts per step): (no, no, [1]), (yes, no, [1]), (yes, yes, [1]), (yes, no, [2]). Runs the
    # actor training for rollout 7 with rollout data holding one rollouts entry per step and
    # total lengths summing to the microbatch count. Asserts the actor log-prob computation ran
    # once only when neither skip nor rollout log probs is set, otherwise zero times; advantage
    # computation was called once with the arguments and rollout data; and the train function's
    # seventh positional argument is the rollouts list and its keywords are exactly no witness
    # info, attempt 0 and no fault-tolerance executor.
    worker = _actor_reuse_worker(
        actor_module,
        skip_actor_forward_only=skip_actor_forward_only,
        use_rollout_logprobs=use_rollout_logprobs,
    )
    _patch_actor_reuse_dependencies(actor_module, monkeypatch, num_microbatches=num_microbatches)
    rollout_data = {
        "num_rollouts": [1] * len(num_microbatches),
        "total_lengths": [1] * sum(num_microbatches),
    }

    worker._train_actor(7, rollout_data, witness_info=None, attempt=0)

    assert worker._compute_log_prob.call_count == int(not skip_actor_forward_only and not use_rollout_logprobs)
    actor_module.compute_advantages_and_returns.assert_called_once_with(worker.args, rollout_data)
    train_call = actor_module.train.call_args
    assert train_call.args[6] is rollout_data["num_rollouts"]
    assert train_call.kwargs == {
        "witness_info": None,
        "attempt": 0,
        "ft_test_action_executor": None,
    }


def test_skip_actor_forward_only_preserves_reference_teacher_and_training_forwards(actor_module, monkeypatch):
    # [FUNCTION DOC] With skip-actor-forward on and backup tags for reference and teacher, the
    # log-prob computation mock returns a mapping keyed by its store prefix. Runs actor training
    # for rollout 7. Asserts the log-prob computation was called only with store prefixes "ref_"
    # then "teacher_" (not the actor's own) and the train function ran once.
    worker = _actor_reuse_worker(actor_module, skip_actor_forward_only=True)
    worker.weights_backuper.backup_tags = {"ref", "teacher"}
    worker._compute_log_prob.side_effect = lambda *_args, store_prefix, **_kwargs: {
        f"{store_prefix}log_probs": [object()]
    }
    _patch_actor_reuse_dependencies(actor_module, monkeypatch, num_microbatches=[1])
    rollout_data = {"num_rollouts": [1], "total_lengths": [1]}

    worker._train_actor(7, rollout_data, witness_info=None, attempt=0)

    assert [call.kwargs["store_prefix"] for call in worker._compute_log_prob.call_args_list] == ["ref_", "teacher_"]
    actor_module.train.assert_called_once()


@pytest.mark.parametrize(
    ("manager_cls", "rollout_flag", "data_key"),
    [
        (RoutingReplayManager, "use_rollout_routing_replay", "rollout_routed_experts"),
        (IndexerReplayManager, "use_rollout_indexer_replay", "rollout_indexer_topk"),
    ],
)
def test_skip_actor_forward_only_consumes_preloaded_rollout_replay_during_training(
    actor_module,
    monkeypatch,
    manager_cls,
    rollout_flag,
    data_key,
):
    # [FUNCTION DOC] Parametrized over (routing replay manager, its rollout flag, its rollout
    # data key) and (indexer replay manager, its flag, its key). Sets up an enabled manager with
    # result checking off and one mock replay that queues recorded top-k indices on record and
    # pops the oldest on backward. The worker skips the actor forward and enables the rollout flag;
    # the manager is the only replay manager; the replay-stage setter copies the stage onto the
    # manager. A patched replay-data filler asserts it was given the expected data key and the
    # manager's replay list, and records the first rollout item (removed from the rollout data)
    # into the first replay. The patched train function asks the manager for its top-k function
    # (non-probability mode), applies it with k=1 on scores [[0, 1]] and asserts the result equals
    # the preloaded indices [[1]]. Runs actor training for rollout 7. Asserts the log-prob
    # computation was not called, the filler was called once, the replay's backward pop was called
    # once, and the queue is empty at the end.
    manager = manager_cls()
    manager.enabled = True
    manager.enable_check_replay_result = False
    queued_top_indices = []
    replay = Mock()
    replay.record.side_effect = queued_top_indices.append
    replay.pop_backward.side_effect = lambda: queued_top_indices.pop(0)
    manager.replays = [replay]
    manager.set_current(replay)

    worker = _actor_reuse_worker(
        actor_module,
        skip_actor_forward_only=True,
        **{rollout_flag: True},
    )
    _patch_actor_reuse_dependencies(actor_module, monkeypatch, num_microbatches=[1])
    monkeypatch.setattr(actor_module, "all_replay_managers", [manager])
    worker._set_replay_stage.side_effect = lambda stage: setattr(manager, "stage", stage)

    expected_top_indices = torch.tensor([[1]], dtype=torch.int64)

    def preload_replay_data(**kwargs):
        # [FUNCTION DOC] HELPER: Replacement for the replay-data filler. Takes keyword arguments;
        # asserts the data key equals the expected one and the replay list is the manager's own
        # list; removes the data entry from the rollout data and records its first element into
        # the first replay.
        assert kwargs["data_key"] == data_key
        assert kwargs["replay_list"] is manager.replays
        kwargs["replay_list"][0].record(kwargs["rollout_data"].pop(data_key)[0])

    fill_replay_data = Mock(side_effect=preload_replay_data)
    monkeypatch.setattr(actor_module, "fill_replay_data", fill_replay_data)

    def train_with_replay(*_args, **_kwargs):
        # [FUNCTION DOC] HELPER: Replacement for the train function; ignores its arguments. Gets
        # the manager's top-k function wrapping a plain top-k-indices function, applies it with
        # k=1 to scores [[0.0, 1.0]], asserts the output equals the expected preloaded indices,
        # and returns the discarded-should-retry outcome.
        topk_fn = manager.get_topk_fn(
            lambda scores, topk: torch.topk(scores, topk, dim=1).indices,
            return_probs=False,
        )
        scores = torch.tensor([[0.0, 1.0]])
        torch.testing.assert_close(topk_fn(scores, 1), expected_top_indices)
        return actor_module.TrainStepOutcome.DISCARDED_SHOULD_RETRY

    train = Mock(side_effect=train_with_replay)
    monkeypatch.setattr(actor_module, "train", train)
    rollout_data = {
        "num_rollouts": [1],
        "total_lengths": [1],
        data_key: [expected_top_indices],
    }

    worker._train_actor(7, rollout_data, witness_info=None, attempt=0)

    worker._compute_log_prob.assert_not_called()
    fill_replay_data.assert_called_once()
    replay.pop_backward.assert_called_once()
    assert queued_top_indices == []


def test_skip_actor_forward_only_rejects_multiple_optimizer_steps(actor_module, monkeypatch):
    # [FUNCTION DOC] With skip-actor-forward on and two optimizer steps ([1, 1] microbatches),
    # actor training raises an assertion error saying it requires 1 optimizer step. Asserts no
    # log-prob computation, no advantage computation and no training happened.
    worker = _actor_reuse_worker(actor_module, skip_actor_forward_only=True)
    _patch_actor_reuse_dependencies(actor_module, monkeypatch, num_microbatches=[1, 1])
    rollout_data = {"num_rollouts": [1, 1], "total_lengths": [1, 1]}

    with pytest.raises(AssertionError, match="requires 1 optimizer step"):
        worker._train_actor(7, rollout_data, witness_info=None, attempt=0)

    worker._compute_log_prob.assert_not_called()
    actor_module.compute_advantages_and_returns.assert_not_called()
    actor_module.train.assert_not_called()


def test_skip_actor_forward_only_rejects_existing_actor_log_probs(actor_module, monkeypatch):
    # [FUNCTION DOC] With skip-actor-forward on and rollout data already containing log probs,
    # actor training raises an assertion error saying it must run without actor log probs.
    # Asserts no log-prob computation, no advantage computation and no training happened.
    worker = _actor_reuse_worker(actor_module, skip_actor_forward_only=True)
    _patch_actor_reuse_dependencies(actor_module, monkeypatch, num_microbatches=[1])
    rollout_data = {"num_rollouts": [1], "total_lengths": [1]}
    rollout_data["log_probs"] = [object()]

    with pytest.raises(AssertionError, match="without actor log probs"):
        worker._train_actor(7, rollout_data, witness_info=None, attempt=0)

    worker._compute_log_prob.assert_not_called()
    actor_module.compute_advantages_and_returns.assert_not_called()
    actor_module.train.assert_not_called()


_OBJECT_REF_ID_BYTES = 28


class _FakeRay:
    # [CLASS DOC] HELPER: In-memory stand-in for the Ray put/get API that still hands out real
    # object-reference values (built from 28-byte keys), so type checks on references hold.
    """A store ref is typed as holding a real ObjectRef, so the stand-in hands out real ones."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates an empty mapping from key bytes to stored values.
        self._objects: dict[bytes, Any] = {}

    def put(self, value: Any) -> ray.ObjectRef:
        # [FUNCTION DOC] HELPER: Uses the current object count as a little-endian 28-byte key,
        # stores the value under it and returns a real object reference built from the key.
        key = len(self._objects).to_bytes(_OBJECT_REF_ID_BYTES, "little")
        self._objects[key] = value
        return ray.ObjectRef(key)

    def get(self, ref: ray.ObjectRef) -> Any:
        # [FUNCTION DOC] HELPER: Looks up and returns the value stored under the reference's
        # binary id; raises a key error for an unknown reference.
        return self._objects[ref.binary()]


@contextmanager
def _noop_timer(_name: str) -> Iterator[None]:
    # [FUNCTION DOC] HELPER: Timer replacement; ignores the name and yields immediately.
    yield


def _patch_shared_train_helpers(actor_module: Any, monkeypatch: pytest.MonkeyPatch, fake_ray: _FakeRay) -> None:
    # [FUNCTION DOC] HELPER: Patches (via monkeypatch) the object store module to use the fake Ray
    # and a fresh store instance that does not free objects; and the actor module to have no
    # replay managers, a data-iterator getter returning an object and microbatch counts [1], a
    # no-op advantage computation, a train function returning the normal outcome, a parallel state
    # reporting it is the last pipeline stage, and a no-op timer.
    monkeypatch.setattr(object_store, "ray", fake_ray)
    monkeypatch.setattr(object_store, "_INSTANCE", object_store.RayObjectStore(frees_objects=False))
    monkeypatch.setattr(actor_module, "all_replay_managers", [])
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda *_args, **_kwargs: (object(), [1]))
    monkeypatch.setattr(actor_module, "compute_advantages_and_returns", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "train", lambda *_args, **_kwargs: TrainStepOutcome.NORMAL)
    monkeypatch.setattr(actor_module, "get_parallel_state", lambda: SimpleNamespace(is_pp_last_stage=True))
    monkeypatch.setattr(actor_module, "timer", _noop_timer)


def _critic_worker(actor_module: Any) -> Any:
    # [FUNCTION DOC] HELPER: Creates a critic-role worker (no constructor) with global batch size 1,
    # no loss type, placeholder model/optimizer/scheduler and a mock heartbeat. Returns it.
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(global_batch_size=1, loss_type=None)
    worker.role = "critic"
    worker.model = object()
    worker.optimizer = object()
    worker.opt_param_scheduler = object()
    worker._heartbeat = Mock()
    return worker


def _actor_worker(actor_module: Any) -> Any:
    # [FUNCTION DOC] HELPER: Creates an actor-role worker (no constructor) configured for a
    # colocated run with a critic: arguments with advantages computed, rollout log probs used, no
    # mismatch metrics, global batch size 1, no old actor, no reference update interval, no skip
    # of actor forward; no reference or teacher; placeholder model/optimizer/scheduler; mocks for
    # profiler, weight updater, weights backuper (no backup tags), model switching and heartbeat;
    # active tag "actor"; no postprocess and no fault-tolerance executor. Returns it.
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(
        colocate=True,
        compute_advantages_and_returns=True,
        get_mismatch_metrics=False,
        global_batch_size=1,
        keep_old_actor=False,
        ref_update_interval=None,
        skip_actor_forward_only=False,
        use_critic=True,
        use_rollout_logprobs=True,
    )
    worker.role = "actor"
    worker.with_ref = False
    worker.with_opd_teacher = False
    worker.model = object()
    worker.optimizer = object()
    worker.opt_param_scheduler = object()
    worker.prof = Mock()
    worker.rollout_data_postprocess = None
    worker.weight_updater = Mock()
    worker.weights_backuper = Mock()
    worker.weights_backuper.backup_tags = ()
    worker._active_model_tag = "actor"
    worker._ft_test_action_executor = None
    worker._heartbeat = Mock()
    worker._switch_model = Mock()
    return worker


def test_critic_output_roundtrips_into_actor_external_data(actor_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Patches shared train helpers with a fake Ray, plus no-op forward-only, timer,
    # logging, debug-save, and the CUDA current device reported as CPU. A critic worker trains for
    # rollout 7 with rollout data holding two value tensors ([1, 2] and [3]). Asserts the output
    # is a step output with outcome normal and non-empty values. Then an actor worker trains for
    # rollout 8 with that output as external data and asserts its step output is normal and that
    # the actor's rollout data now has values equal to [[1, 2], [3]] (retrieved from the store).
    """Real _train_critic ships values that real _train_actor reads back through TrainStepOutput.values."""
    fake_ray = _FakeRay()
    _patch_shared_train_helpers(actor_module, monkeypatch, fake_ray)
    monkeypatch.setattr(actor_module, "forward_only", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(actor_module, "inverse_timer", _noop_timer)
    monkeypatch.setattr(actor_module, "log_rollout_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "log_train_advantage_computation_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "log_perf_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module.train_dump, "save_debug_train_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module.torch.cuda, "current_device", lambda: torch.device("cpu"))
    critic_values = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]

    critic_output = _critic_worker(actor_module)._train_critic(rollout_id=7, rollout_data={"values": critic_values})

    assert isinstance(critic_output, TrainStepOutput)
    assert critic_output.outcome is TrainStepOutcome.NORMAL
    assert critic_output.values is not None

    actor_rollout_data: dict[str, Any] = {"tokens": []}
    actor_output = _actor_worker(actor_module)._train_actor(
        8, actor_rollout_data, critic_output, witness_info=None, attempt=0
    )

    assert isinstance(actor_output, TrainStepOutput)
    assert actor_output.outcome is TrainStepOutcome.NORMAL
    assert [value.tolist() for value in actor_rollout_data["values"]] == [[1.0, 2.0], [3.0]]


def test_debug_rollout_only_train_answers_with_a_normal_train_step_output(
    actor_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] An awake actor worker with debug rollout-only on and mocked actor/critic train
    # methods, with patched rollout-data loading, logging and timer. Runs train for rollout 9.
    # Asserts the result equals a normal step output (no values) and neither train method ran.
    """A debug-rollout-only step skips training yet still answers the driver with a NORMAL output."""
    worker = _worker(actor_module, "actor", asleep=False)
    worker.args.debug_rollout_only = True
    worker._train_actor = Mock()
    worker._train_critic = Mock()
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )
    monkeypatch.setattr(actor_module, "log_rollout_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "timer", _noop_timer)

    result = worker.train(9, object())

    assert result == TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
    worker._train_actor.assert_not_called()
    worker._train_critic.assert_not_called()


@pytest.mark.parametrize("is_pp_last_stage,rollout_data_values", [(False, [torch.tensor([1.0])]), (True, None)])
def test_critic_without_shippable_values_returns_an_output_carrying_none(
    actor_module: Any, monkeypatch: pytest.MonkeyPatch, is_pp_last_stage: bool, rollout_data_values: Any
) -> None:
    # [FUNCTION DOC] Parametrized over (not pipeline-last with values present) and (pipeline-last
    # with no values key). With shared helpers patched and a no-op forward-only, the critic worker
    # trains for rollout 7. Asserts it returns a normal step output with values none.
    """A critic rank that is not pp-last or has no values still returns a TrainStepOutput with values=None."""
    _patch_shared_train_helpers(actor_module, monkeypatch, _FakeRay())
    monkeypatch.setattr(actor_module, "forward_only", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(actor_module, "get_parallel_state", lambda: SimpleNamespace(is_pp_last_stage=is_pp_last_stage))
    rollout_data: dict[str, Any] = {} if rollout_data_values is None else {"values": rollout_data_values}

    output = _critic_worker(actor_module)._train_critic(rollout_id=7, rollout_data=rollout_data)

    assert output == TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=None)


def test_actor_last_stage_rejects_a_critic_output_without_values(
    actor_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] With shared helpers patched and a no-op inverse timer, an actor worker on the
    # last pipeline stage trains with an external critic output whose values are none. Asserts an
    # assertion error saying the critic output must have shipped 'values'.
    """A pp-last actor using a critic refuses to train on a TrainStepOutput that shipped no values."""
    _patch_shared_train_helpers(actor_module, monkeypatch, _FakeRay())
    monkeypatch.setattr(actor_module, "inverse_timer", _noop_timer)
    empty_critic_output = TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=None)

    with pytest.raises(AssertionError, match="must have shipped 'values'"):
        _actor_worker(actor_module)._train_actor(8, {"tokens": []}, empty_critic_output, witness_info=None, attempt=0)


class _RecordingWeightUpdater:
    # [CLASS DOC] HELPER: Fake weight updater that records connection calls and counts weight
    # updates, using a real connection-status manager to decide when to reconnect.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a real connection-status manager, an empty list of
        # connect-call records, and zeroed update count and weight version.
        self.conn_status = ConnStatusManager()
        self.connect_calls: list[dict[str, Any]] = []
        self.update_weights_calls: int = 0
        self.weight_version: int = 0

    def connect_rollout_engines(
        self,
        rollout_engines: list[Any],
        engine_gpu_counts: list[int] | None = None,
        engine_gpu_offsets: list[int] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Appends a record holding a copy of the engines list and the
        # optional GPU counts and offsets.
        self.connect_calls.append(
            dict(
                rollout_engines=list(rollout_engines),
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
        )

    def reconnect_if_needed(self, info: Any) -> bool:
        # [FUNCTION DOC] HELPER: Asks the status manager whether the snapshot cell-to-hash mapping
        # in the info requires a reconnect; if not returns false; otherwise reconnects and returns
        # true.
        if not self.conn_status.needs_reconnect(info.snapshot_cell_id_to_hashes):
            return False
        self.reconnect(info)
        return True

    def reconnect(self, info: Any) -> None:
        # [FUNCTION DOC] HELPER: Connects to the info's engines with its GPU counts and offsets,
        # then marks the status manager as reconnected for the info's snapshot mapping.
        self.connect_rollout_engines(
            info.rollout_engines,
            engine_gpu_counts=info.engine_gpu_counts,
            engine_gpu_offsets=info.engine_gpu_offsets,
        )
        self.conn_status.mark_reconnected(info.snapshot_cell_id_to_hashes)

    def verify_engine_version(self, rollout_engines: list[Any]) -> None:
        # [FUNCTION DOC] HELPER: No-op version check.
        pass

    def update_weights(self) -> None:
        # [FUNCTION DOC] HELPER: Increments the update count and the weight version by one.
        self.update_weights_calls += 1
        self.weight_version += 1


def _weight_update_worker(actor_module: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    # [FUNCTION DOC] HELPER: Creates a worker (no constructor) with arguments for weight update
    # tests (CI test, rollout-only, skip-weight-update, train-only, keep-old-actor, offload and
    # rematerialize all off; no actor-weights colocate attribute set), awake, a mock heartbeat and
    # the recording weight updater. Patches the actor module's memory printing (mock), gloo group
    # getter (returns none) and the distributed barrier (no-op). Returns the worker.
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(
        ci_test=False,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        debug_train_only=False,
        keep_old_actor=False,
        offload_train=False,
        rematerialize_param_from_master_weight=False,
    )
    worker._asleep = False
    worker._heartbeat = Mock()
    worker.weight_updater = _RecordingWeightUpdater()
    monkeypatch.setattr(actor_module, "print_memory", Mock())
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(actor_module.dist, "barrier", lambda **_kwargs: None)
    return worker


@pytest.mark.parametrize("asleep", [False, True])
def test_disaggregated_weight_sync_reads_host_backup_only_when_asleep(
    actor_module: Any, monkeypatch: pytest.MonkeyPatch, asleep: bool
) -> None:
    # [FUNCTION DOC] Parametrized over the sleeping state. A non-colocated worker whose
    # weight-sync-reads-backup property is patched false, active tag actor, with a mock backuper
    # returning host weights and a mock live-weights iterator returning device weights. Asserts
    # the actor-weights getter returns host weights when asleep and device weights when awake,
    # and that exactly one of the two sources is queried accordingly.
    worker = _weight_update_worker(actor_module, monkeypatch)
    worker.args.colocate = False
    worker._asleep = asleep
    worker._active_model_tag = "actor"
    monkeypatch.setattr(type(worker), "_weight_sync_reads_tms_backup", property(lambda _self: False))
    host_weights = {"weight": object()}
    device_weights = {"weight": object()}
    worker.weights_backuper = Mock()
    worker.weights_backuper.get.return_value = host_weights
    worker._named_actor_weights = Mock(return_value=device_weights.items())

    assert worker._get_actor_weights() == (host_weights if asleep else device_weights)
    assert worker.weights_backuper.get.call_count == int(asleep)
    assert worker._named_actor_weights.call_count == int(not asleep)


@pytest.mark.parametrize("offload_train", [False, True])
def test_offloading_keeps_weight_backup_without_reference_model(
    actor_module: Any, monkeypatch: pytest.MonkeyPatch, offload_train: bool
) -> None:
    # [FUNCTION DOC] Parametrized over the offload-train flag. A non-colocated worker with no
    # reference and no teacher model and the weight-sync-reads-backup property patched false.
    # Asserts the weight-backup-enabled property equals the offload-train flag.
    worker = _weight_update_worker(actor_module, monkeypatch)
    worker.args.colocate = False
    worker.args.offload_train = offload_train
    worker.with_ref = False
    worker.with_opd_teacher = False
    monkeypatch.setattr(type(worker), "_weight_sync_reads_tms_backup", property(lambda _self: False))

    assert worker._enable_weight_backup is offload_train


@pytest.mark.parametrize("active_tag", ["actor", "ref", "teacher", "old_actor"])
def test_switch_model_restores_the_already_active_tag(
    actor_module: Any, monkeypatch: pytest.MonkeyPatch, active_tag: str
) -> None:
    # [FUNCTION DOC] Parametrized over the active tag (actor, ref, teacher, old_actor). With the
    # weight-backup property patched true and a mock backuper whose tags include the active one,
    # switching to the same tag asserts the backuper restore was called once with that tag and
    # the active tag is unchanged.
    """The active tag's live storage may have been discarded by an offload
    cycle since the last switch (--offload-train disables the param buffers'
    memory-saver backup), so a same-tag switch must still restore."""
    worker = _weight_update_worker(actor_module, monkeypatch)
    monkeypatch.setattr(type(worker), "_enable_weight_backup", property(lambda _self: True))
    worker._active_model_tag = active_tag
    worker.weights_backuper = Mock(backup_tags={active_tag})

    worker._switch_model(active_tag)

    worker.weights_backuper.restore.assert_called_once_with(active_tag)
    assert worker._active_model_tag == active_tag


@pytest.mark.parametrize(
    ("active_tag", "target_tag"),
    [(None, "actor"), ("ref", "actor"), ("teacher", "actor"), ("old_actor", "actor"), ("actor", "ref")],
)
def test_switch_model_restores_on_every_switch(
    actor_module: Any, monkeypatch: pytest.MonkeyPatch, active_tag: str | None, target_tag: str
) -> None:
    # [FUNCTION DOC] Parametrized over (active tag, target tag) pairs including no active tag and
    # changes between tags. With the weight-backup property patched true and a mock backuper with
    # the target tag, switches to the target twice. Asserts restore was called exactly twice, the
    # last time with the target tag (no same-tag skip), and the active tag is the target.
    worker = _weight_update_worker(actor_module, monkeypatch)
    monkeypatch.setattr(type(worker), "_enable_weight_backup", property(lambda _self: True))
    worker._active_model_tag = active_tag
    worker.weights_backuper = Mock(backup_tags={target_tag})

    worker._switch_model(target_tag)
    worker._switch_model(target_tag)

    # No same-tag skip: the repeated switch restores again, because an offload
    # cycle between the two calls may have dropped the live storage.
    assert worker.weights_backuper.restore.call_count == 2
    worker.weights_backuper.restore.assert_called_with(target_tag)
    assert worker._active_model_tag == target_tag


def test_switch_model_rejects_unknown_tag_even_when_marked_active(
    actor_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] The active tag is "missing" but the backuper only knows "actor". Switching to
    # "missing" raises a value error saying it cannot switch to the unknown model tag, and the
    # backuper restore is never called.
    worker = _weight_update_worker(actor_module, monkeypatch)
    monkeypatch.setattr(type(worker), "_enable_weight_backup", property(lambda _self: True))
    worker._active_model_tag = "missing"
    worker.weights_backuper = Mock(backup_tags={"actor"})

    with pytest.raises(ValueError, match="Cannot switch to unknown model tag: missing"):
        worker._switch_model("missing")

    worker.weights_backuper.restore.assert_not_called()


def _updatable_engines(rollout_engines: list[Any], snapshot: dict[str, str], gpu_count: int) -> Any:
    # [FUNCTION DOC] HELPER: Builds an updatable-engines info object from the engines, with every
    # engine's GPU count equal to the given count, GPU offsets as index times count, and the
    # given snapshot mapping of cell id to hash.
    from miles.ray.rollout.inference_controller import UpdatableEngines

    return UpdatableEngines(
        rollout_engines=rollout_engines,
        engine_gpu_counts=[gpu_count] * len(rollout_engines),
        engine_gpu_offsets=[index * gpu_count for index in range(len(rollout_engines))],
        snapshot_cell_id_to_hashes=snapshot,
    )


@pytest.mark.parametrize("offload_train", [False, True])
@pytest.mark.parametrize("connect_fails", [False, True])
def test_connection_uses_safe_allocations_when_offloading(
    actor_module: Any, monkeypatch: pytest.MonkeyPatch, offload_train: bool, connect_fails: bool
) -> None:
    # [FUNCTION DOC] Parametrized over offload-train and whether the connection fails. The worker
    # sleeps exactly when offload is on. The memory saver mock's disable returns a context manager
    # that tracks being inside a "safe allocation region" (asserting it is not nested). The
    # patched connect function asserts that it runs inside the region if and only if offload is
    # on, and raises a runtime error "connection failed" when configured. Updates weights with
    # empty engines and one snapshot entry. Expects the error to propagate when failing. Asserts
    # the region is always exited; the saver's disable was called 0 times without offload, else
    # 1 time when the connection fails or 2 times when it succeeds; and the updater's update
    # count is 1 on success and 0 on failure.
    worker = _weight_update_worker(actor_module, monkeypatch)
    worker.args.offload_train = offload_train
    worker._asleep = offload_train
    inside_safe_region = False

    @contextmanager
    def safe_region() -> Iterator[None]:
        # [FUNCTION DOC] HELPER: Context manager replacing the saver's disable: asserts not
        # already inside, marks inside while active, and always clears the mark on exit.
        nonlocal inside_safe_region
        assert not inside_safe_region
        inside_safe_region = True
        try:
            yield
        finally:
            inside_safe_region = False

    def check_connection(*_args: Any, **_kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Replacement for the connect function: asserts the inside-region
        # mark equals the offload-train flag, then raises a runtime error saying connection failed
        # when failure is configured.
        assert inside_safe_region == offload_train
        if connect_fails:
            raise RuntimeError("connection failed")

    saver = Mock()
    saver.disable.side_effect = safe_region
    monkeypatch.setattr(actor_module, "torch_memory_saver", saver)
    monkeypatch.setattr(actor_module, "reload_process_groups", Mock())
    monkeypatch.setattr(actor_module, "destroy_process_groups", Mock())
    worker.weight_updater.connect_rollout_engines = check_connection
    engines = _updatable_engines([], {"cell-0": "hash-a"}, gpu_count=8)

    if connect_fails:
        with pytest.raises(RuntimeError, match="connection failed"):
            worker.update_weights(engines)
    else:
        worker.update_weights(engines)

    assert not inside_safe_region
    assert saver.disable.call_count == ((1 if connect_fails else 2) if offload_train else 0)
    assert worker.weight_updater.update_weights_calls == int(not connect_fails)


def test_update_weights_reconnects_once_per_rollout_snapshot(
    actor_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Updates weights three times: twice with the same engines (4 GPUs each) and
    # snapshot hash "hash-a", then with two replacement engines (2 GPUs each) and hash "hash-b".
    # Asserts the connect calls are exactly the first engines then the replacement engines, the
    # second record has GPU counts [2, 2] and offsets [0, 2], the updater ran 3 updates, the last
    # update call returned weight version 3, and the status manager no longer needs a reconnect
    # for hash-b.
    """The actor connects to a rollout snapshot once and reconnects with the new topology only when it changes."""
    worker = _weight_update_worker(actor_module, monkeypatch)
    updater = worker.weight_updater
    first_engines = [object()]
    replacement_engines = [object(), object()]

    worker.update_weights(_updatable_engines(first_engines, {"cell-0": "hash-a"}, gpu_count=4))
    worker.update_weights(_updatable_engines(first_engines, {"cell-0": "hash-a"}, gpu_count=4))
    weight_version = worker.update_weights(_updatable_engines(replacement_engines, {"cell-0": "hash-b"}, gpu_count=2))

    assert [call["rollout_engines"] for call in updater.connect_calls] == [first_engines, replacement_engines]
    assert updater.connect_calls[1]["engine_gpu_counts"] == [2, 2]
    assert updater.connect_calls[1]["engine_gpu_offsets"] == [0, 2]
    assert updater.update_weights_calls == 3
    assert weight_version == 3
    assert not updater.conn_status.needs_reconnect({"cell-0": "hash-b"})


def test_reconfigure_indep_dp_forces_the_next_weight_update_to_reconnect(
    actor_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Patches the group reconfiguration function (mock), parallel state (empty
    # object), rank 0 and world size 2. Updates weights, reconfigures the independent
    # data-parallel group with a placeholder and an address, then updates weights again with the
    # same engines and snapshot. Asserts two connect calls happened in total.
    """Rebuilding the independent-DP groups invalidates the trainer side, so the next update reconnects."""
    worker = _weight_update_worker(actor_module, monkeypatch)
    updater = worker.weight_updater
    monkeypatch.setattr(actor_module, "reconfigure_indep_dp_group", Mock())
    monkeypatch.setattr(actor_module, "get_parallel_state", lambda: SimpleNamespace())
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor_module.dist, "get_world_size", lambda: 2)
    engines = [object()]
    snapshot = {"cell-0": "hash-a"}

    worker.update_weights(_updatable_engines(engines, snapshot, gpu_count=4))
    worker.reconfigure_indep_dp(object(), "10.0.0.1:1234")
    worker.update_weights(_updatable_engines(engines, snapshot, gpu_count=4))

    assert len(updater.connect_calls) == 2


def test_switch_model_restores_discarded_values_with_the_normal_backuper(
    actor_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Patches CUDA synchronize to a no-op. Builds a real tensor backuper over a
    # live weight tensor [0, 1, 2, 3], injects an unpinned host backup for the actor tag with a
    # copy of the values, zeroes the live tensor (simulating a discarded storage), and switches to
    # the already-active actor tag with weight backup enabled. Asserts the live tensor again
    # equals [0, 1, 2, 3].
    """--offload-train disables the param buffers' memory-saver backup: sleep()
    discards their contents and wake_up() reallocates the storage without
    values. The same-tag switch after wake must copy the host backup back into
    the live tensors -- value-level, not merely 'restore was called'."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    live = {"w": torch.arange(4, dtype=torch.float32)}
    backuper = TensorBackuper.create(lambda: iter(live.items()))
    # The host backup a training step took; unpinned so the test runs on CPU CI.
    backuper._backups["actor"] = {"w": live["w"].clone()}

    worker = _weight_update_worker(actor_module, monkeypatch)
    monkeypatch.setattr(type(worker), "_enable_weight_backup", property(lambda _self: True))
    worker._active_model_tag = "actor"
    worker.weights_backuper = backuper

    live["w"].zero_()  # the storage an offload cycle discarded and reallocated
    worker._switch_model("actor")

    assert torch.equal(live["w"], torch.arange(4, dtype=torch.float32))


def test_switch_model_rebuilds_the_active_actor_for_main_cast(
    actor_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Patches CUDA synchronize to a no-op. Builds a real tensor backuper with an
    # empty live-weight iterator and a main-cast context holding a mock cast function, no model
    # chunks, no rematerializable ids and checking off. With weight backup enabled and the actor
    # tag already active, switches to the actor tag and asserts the cast function was called
    # once with no arguments.
    """--rematerialize-param-from-master-weight: restore('actor') replays the
    master-weight cast, rebuilding the params update_weights paused -- the
    per-cycle call must never be skipped for the already-active tag."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    cast_main_to_params = Mock()
    backuper = TensorBackuper.create(
        lambda: iter(()),
        MainCastContext(
            cast_main_to_params=cast_main_to_params,
            model_chunks=[],
            extras_getter=lambda: iter(()),
            rematerializable_ids=set(),
            check=False,
        ),
    )
    worker = _weight_update_worker(actor_module, monkeypatch)
    monkeypatch.setattr(type(worker), "_enable_weight_backup", property(lambda _self: True))
    worker._active_model_tag = "actor"
    worker.weights_backuper = backuper

    worker._switch_model("actor")

    cast_main_to_params.assert_called_once_with()
