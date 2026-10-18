# [MODULE DOC] CPU tests for the torch-native train actor's core step and optimizer offload helper.
# The actor's collaborators (routing replay, data iterator factory, log-prob and optimizer-step
# methods, advantage computation, rollout logging, timer) are replaced by recorders; the test asserts
# the order of stages: replay fill, reference log-probs under the fallthrough stage and the reference
# context, actor log-probs under the replay-forward stage, replay rewind, optimizer steps under the
# replay-backward stage, replay reset; and what gets stored in the rollout data. A second test
# checks that moving optimizer state between devices does not create state for parameters that had
# none.
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from miles.backends.training_utils.torch_native import actor as base_module
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor
from miles.backends.training_utils.torch_native.offload import _map_optimizer_state

_MODULE = "miles.backends.training_utils.torch_native.actor"


@contextmanager
def _recording(stages, name):
    # [FUNCTION DOC] HELPER: Context manager appending "enter:<name>" to the stages list on entry
    # and "exit:<name>" on exit (not exception-safe: no try/finally).
    stages.append(f"enter:{name}")
    yield
    stages.append(f"exit:{name}")


@contextmanager
def _noop_timer(_name):
    # [FUNCTION DOC] HELPER: Replacement timer context manager; ignores the name, does nothing.
    yield


class _Profiler:
    # [CLASS DOC] HELPER: Fake profiler whose iteration wrappers return the iterable unchanged and
    # which has a mock step method.
    def iterate_train_actor(self, it):
        # [FUNCTION DOC] HELPER: Returns the iterable unchanged.
        return it

    def iterate_train_log_probs(self, it):
        # [FUNCTION DOC] HELPER: Returns the iterable unchanged.
        return it

    step = MagicMock()


class _DataIterator:
    # [CLASS DOC] HELPER: Fake data iterator counting resets and fetches.
    # KNOWN ISSUE: this class (and the module-name constant above) is never used in this file.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts both counters at zero.
        self.resets = 0
        self.fetches = 0

    def reset(self):
        # [FUNCTION DOC] HELPER: Increments the reset counter and returns itself.
        self.resets += 1
        return self


class _Provider(TorchNativeTrainRayActor):
    # [CLASS DOC] HELPER: Train actor subclass built without the real constructor: minimal args
    # (micro batch 1, no offload, not rollout-only, pad multiplier 1, packed format, CI off, no
    # sampling-support replay), a mock model part and an object optimizer, a fake profiler, a runner
    # object returned by the step-runner hook, an identity side-channel aligner and a mock
    # heartbeat.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Sets the attributes listed in the class description; does not call
        # the parent constructor.
        self.args = SimpleNamespace(
            micro_batch_size=1,
            offload_train=False,
            debug_rollout_only=False,
            data_pad_size_multiplier=1,
            qkv_format="thd",
            ci_test=False,
            use_sampling_support_replay=False,
        )
        self.model_parts = [MagicMock()]
        self.optimizers = [object()]
        self.hf_config = object()
        self.prof = _Profiler()
        self.runner = object()
        self.align_token_side_channel = lambda t, pad: t
        self._heartbeat = MagicMock()

    def _step_runner(self):
        # [FUNCTION DOC] HELPER: Returns the stored runner object.
        return self.runner


@pytest.fixture
def step(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture. Builds a recorder dictionary (log-prob calls, optimizer step
    # calls, a shared stage-name list). Replaces the base module's routing replay with a fake whose
    # stage context records enter/exit, whose fill, rewind and reset record "fill", "rewind",
    # "reset", and whose log-prob stage is replay forward. Patches the data iterator factory (one
    # iterator, count 2), the actor's log-prob method (records runner and store prefix, returns a
    # dictionary with one prefixed log-probs entry) and optimizer-step method (records the runner),
    # advantage computation (marks the data), rollout logging (no-op) and the timer (no-op).
    # Returns the recorder dictionary, which also holds the fake replay object.
    stages: list = []
    calls = {"log_probs": [], "steps": [], "stages": stages}
    replay = SimpleNamespace(
        FALLTHROUGH="fallthrough",
        REPLAY_BACKWARD="replay_backward",
        stage=lambda name: _recording(stages, name),
        fill=MagicMock(side_effect=lambda *a, **k: stages.append("fill")),
        log_prob_stage=lambda args: "replay_forward",
        rewind=MagicMock(side_effect=lambda: stages.append("rewind")),
        reset=MagicMock(side_effect=lambda: stages.append("reset")),
    )
    calls["replay"] = replay
    monkeypatch.setattr(base_module, "routing_replay", replay)
    monkeypatch.setattr(base_module, "get_data_iterator", lambda args, parts, data: ([object()], [2]))
    monkeypatch.setattr(
        TorchNativeTrainRayActor,
        "_log_probs",
        lambda self, runner, it, n, store_prefix="": (
            calls["log_probs"].append((runner, store_prefix)) or {f"{store_prefix}log_probs": [1]}
        ),
    )
    monkeypatch.setattr(
        TorchNativeTrainRayActor,
        "_optimizer_steps",
        lambda self, runner, it, n, rollout_id: calls["steps"].append(runner),
    )
    monkeypatch.setattr(base_module, "compute_advantages_and_returns", lambda args, data: data.update(adv=True))
    monkeypatch.setattr(base_module, "log_rollout_data", lambda rid, args, data: None)
    monkeypatch.setattr(base_module, "timer", _noop_timer)
    return calls


def test_the_rollout_step_runs_ref_then_actor_then_optimizer_under_the_right_stages(step):
    # [FUNCTION DOC] Test: an actor with a reference runner and a recording reference context runs
    # its core step for rollout 3 on empty rollout data. Stages must be exactly: fill, enter / exit
    # fallthrough around enter / exit ref, enter / exit replay_forward, rewind, enter / exit
    # replay_backward, reset. Log-prob calls are (reference runner, "ref_") then (actor runner, "");
    # optimizer steps ran once on the actor runner; rollout data ends as the reference log-probs, the
    # actor log-probs and the advantage marker; the fill call received the actor's side-channel
    # aligner as its align argument.
    actor = _Provider()
    actor.ref_runner = object()
    actor._ref_context = lambda: _recording(step["stages"], "ref")
    rollout_data = {}

    actor._train_core(rollout_id=3, rollout_data=rollout_data)

    assert step["stages"] == [
        "fill",
        "enter:fallthrough",
        "enter:ref",
        "exit:ref",
        "exit:fallthrough",
        "enter:replay_forward",
        "exit:replay_forward",
        "rewind",
        "enter:replay_backward",
        "exit:replay_backward",
        "reset",
    ]
    assert step["log_probs"] == [(actor.ref_runner, "ref_"), (actor.runner, "")]
    assert step["steps"] == [actor.runner]
    assert rollout_data == {"ref_log_probs": [1], "log_probs": [1], "adv": True}
    assert step["replay"].fill.call_args.kwargs["align"] is actor.align_token_side_channel


def test_moving_optimizer_state_does_not_grow_state_for_parameters_that_have_none():
    # [FUNCTION DOC] Test: a linear layer with AdamW takes one step with a gradient only on the
    # weight, so the bias has no optimizer state. Mapping a move-to-CPU function over the optimizer
    # state leaves the bias without state and the weight's first-moment state on CPU.
    model = torch.nn.Linear(4, 4, bias=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    assert model.bias not in optimizer.state

    _map_optimizer_state([optimizer], lambda tensor: tensor.to("cpu"))

    assert model.bias not in optimizer.state
    assert optimizer.state[model.weight]["exp_avg"].device.type == "cpu"
