# [MODULE DOC]
# Tests for the optimizer reset utility of the Megatron backend, which puts a chained Megatron
# optimizer back to a freshly built state: drops Adam moments and step counters (including the
# one fused Adam keeps on the parameter group), optionally re-allocates zeroed moments through the
# optimizer's own state-init function, drops gradients, skips stub optimizers, runs an internal
# self-check afterwards, and refuses configurations it cannot reset (disk-streamed state, fp16
# loss scaler, chunked offload, state offload). Uses hand-written fake members plus real torch
# Adam; tests that need Megatron skip when the optional packages are missing.
from types import SimpleNamespace

import pytest
import torch


class _FakeInnerOptimizer:
    # [CLASS DOC] HELPER: Fake inner (torch-like) optimizer with one zero parameter that has a
    # ones gradient, a state holding ones-valued first and second moments for it, and one
    # parameter group carrying a step counter and the parameter.
    def __init__(self, *, step: int) -> None:
        # [FUNCTION DOC] HELPER: Builds the parameter, gradient, state and param group described
        # above, with the group's step set to the given value.
        self.param = torch.nn.Parameter(torch.zeros(2))
        self.param.grad = torch.ones(2)
        self.state = {self.param: {"exp_avg": torch.ones(2), "exp_avg_sq": torch.ones(2)}}
        self.param_groups = [{"step": step, "params": [self.param]}]


class _FakeChainedMember:
    # [CLASS DOC] HELPER: Fake member of a Megatron chained optimizer, wrapping the fake inner
    # optimizer (step counter 7) and recording every gradient-zeroing call.
    def __init__(self, *, is_stub_optimizer: bool = False, init_state_fn=None) -> None:
        # [FUNCTION DOC] HELPER: Stores the stub flag (default false), creates the inner fake,
        # stores the optional state-initializing function, sets a placeholder config string and an
        # empty list of recorded zero-gradient arguments.
        self.is_stub_optimizer = is_stub_optimizer
        self.optimizer = _FakeInnerOptimizer(step=7)
        self.init_state_fn = init_state_fn
        self.config = "the optimizer config"
        self.zero_grad_calls: list[bool] = []

    def zero_grad(self, set_to_none: bool = True) -> None:
        # [FUNCTION DOC] HELPER: Records the set-to-none argument, then sets the gradient of every
        # parameter in every inner param group to none (regardless of the argument).
        self.zero_grad_calls.append(set_to_none)
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                param.grad = None


class _RealChainedMember:
    # [CLASS DOC] HELPER: Chained-optimizer member wrapping a real torch optimizer; not a stub, no
    # config.
    def __init__(self, optimizer: torch.optim.Optimizer, *, init_state_fn=None) -> None:
        # [FUNCTION DOC] HELPER: Stores the optimizer and the optional state-init function, with
        # the stub flag false and the config none.
        self.is_stub_optimizer = False
        self.optimizer = optimizer
        self.init_state_fn = init_state_fn
        self.config = None

    def zero_grad(self, set_to_none: bool = True) -> None:
        # [FUNCTION DOC] HELPER: Delegates to the wrapped optimizer's zero-grad with the same
        # set-to-none argument.
        self.optimizer.zero_grad(set_to_none=set_to_none)


def _zero_adam_state(inner, config) -> None:
    # [FUNCTION DOC] HELPER: State-initializer matching the signature Megatron members use: for
    # every parameter in every param group of the inner optimizer, sets its state to zero-valued
    # first and second moments shaped like the parameter. The config is ignored. Mutates the
    # inner optimizer's state; leaves any group step counter untouched.
    for param_group in inner.param_groups:
        for param in param_group["params"]:
            inner.state[param] = {"exp_avg": torch.zeros_like(param.data), "exp_avg_sq": torch.zeros_like(param.data)}


def _chained_optimizer(*members) -> object:
    # [FUNCTION DOC] HELPER: Skips the test if the Megatron optimizer module is unavailable.
    # Otherwise creates a Megatron chained optimizer instance without running its constructor,
    # sets its member list to the given members and its config to a simple object with state
    # offload off and fp16 off. Returns it.
    optimizer_module = pytest.importorskip("megatron.core.optimizer.optimizer")
    optimizer = optimizer_module.ChainedOptimizer.__new__(optimizer_module.ChainedOptimizer)
    optimizer.chained_optimizers = list(members)
    optimizer.config = SimpleNamespace(offload_optimizer_states=False, fp16=False)
    return optimizer


def _reset(
    monkeypatch,
    *members,
    using_pytorch_optimizer: bool = True,
    stream_optimizer_state_to_disk: bool = False,
    chunked_optimizer_state_offload: bool = False,
) -> None:
    # [FUNCTION DOC] HELPER: Skips the test if the optimizer utils module cannot be imported.
    # Patches (via monkeypatch) the module's flag saying whether a PyTorch optimizer is used to
    # the given value (default true), builds a chained optimizer from the members, and calls the
    # reset function with the stream-to-disk and chunked-offload flags (both default false).
    # Propagates any assertion errors the reset raises.
    optimizer_utils = pytest.importorskip("miles.backends.megatron_utils.optimizer_utils")
    monkeypatch.setattr(optimizer_utils, "USING_PYTORCH_OPTIMIZER", using_pytorch_optimizer)
    optimizer_utils.reset_optimizer_state(
        _chained_optimizer(*members),
        stream_optimizer_state_to_disk=stream_optimizer_state_to_disk,
        chunked_optimizer_state_offload=chunked_optimizer_state_offload,
    )


def _adam_with_history(*, steps: int) -> tuple[torch.nn.Parameter, torch.optim.Adam]:
    # [FUNCTION DOC] HELPER: Creates a four-element ones parameter and an Adam optimizer (learning
    # rate 0.1), then takes the given number of steps with the fixed gradient helper. Returns the
    # parameter and the optimizer.
    param = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.Adam([param], lr=0.1)
    for _ in range(steps):
        _step_on_fixed_gradient(param, optimizer)
    return param, optimizer


def _step_on_fixed_gradient(param: torch.nn.Parameter, optimizer: torch.optim.Adam) -> None:
    # [FUNCTION DOC] HELPER: Sets the parameter's gradient to a tensor filled with 0.5 and takes
    # one optimizer step (mutating parameter and optimizer state).
    param.grad = torch.full_like(param, 0.5)
    optimizer.step()


def _assert_same_optimizer_state(left: torch.optim.Adam, right: torch.optim.Adam) -> None:
    # [FUNCTION DOC] HELPER: Compares the state dictionaries of two optimizers: the param groups
    # must be equal; the state keys must be the same sequence; for each key the per-parameter
    # entry names must be the same sequence, tensor values must be exactly equal and non-tensor
    # values must compare equal. Raises an assertion error on the first difference.
    left_state_dict, right_state_dict = left.state_dict(), right.state_dict()
    assert left_state_dict["param_groups"] == right_state_dict["param_groups"]
    assert list(left_state_dict["state"]) == list(right_state_dict["state"])
    for key, left_state in left_state_dict["state"].items():
        right_state = right_state_dict["state"][key]
        assert list(left_state) == list(right_state)
        for name, left_value in left_state.items():
            right_value = right_state[name]
            if isinstance(left_value, torch.Tensor):
                assert torch.equal(left_value, right_value)
            else:
                assert left_value == right_value


class TestResettingALiveOptimizer:
    # [CLASS DOC] Tests of the reset effects on a fake member (state, step counter, gradients).
    def test_the_adam_moments_are_dropped(self, monkeypatch):
        # [FUNCTION DOC] After a reset, the fake inner optimizer's state is empty.
        """A trainer put back at the start of the run must not keep moments from the rollouts it is discarding."""
        member = _FakeChainedMember()

        _reset(monkeypatch, member)

        assert member.optimizer.state == {}

    def test_the_step_counter_a_fused_adam_keeps_on_the_param_group_is_dropped(self, monkeypatch):
        # [FUNCTION DOC] After a reset, the first param group no longer has a step entry.
        """TE and apex fused adam count steps there rather than in per-param state, so clearing state misses it."""
        member = _FakeChainedMember()

        _reset(monkeypatch, member)

        assert "step" not in member.optimizer.param_groups[0]

    def test_the_state_is_reallocated_when_the_optimizer_knows_how(self, monkeypatch):
        # [FUNCTION DOC] With a state-init function and the PyTorch-optimizer flag off, the reset
        # leaves the first parameter's state with exactly the first and second moment entries.
        """A fused optimizer that saves a checkpoint before its first step would otherwise write no moments at all."""
        member = _FakeChainedMember(init_state_fn=_zero_adam_state)

        _reset(monkeypatch, member, using_pytorch_optimizer=False)

        assert set(next(iter(member.optimizer.state.values()))) == {"exp_avg", "exp_avg_sq"}

    def test_the_state_is_left_empty_for_a_torch_optimizer(self, monkeypatch):
        # [FUNCTION DOC] With a state-init function but the PyTorch-optimizer flag on, the reset
        # leaves the state empty (the init function is not applied).
        """The initializer writes the moments and not the step, and torch only fills a state it finds empty."""
        member = _FakeChainedMember(init_state_fn=_zero_adam_state)

        _reset(monkeypatch, member, using_pytorch_optimizer=True)

        assert member.optimizer.state == {}

    def test_the_gradients_are_dropped_too(self, monkeypatch):
        # [FUNCTION DOC] The reset calls the member's zero-grad exactly once, with set-to-none true.
        """A gradient computed against the weights being thrown away would be applied to the restored ones."""
        member = _FakeChainedMember()

        _reset(monkeypatch, member)

        assert member.zero_grad_calls == [True]

    def test_a_stub_optimizer_is_left_alone(self, monkeypatch):
        # [FUNCTION DOC] A chain of a stub member and a normal member: after the reset the stub's
        # state is still non-empty and its zero-grad was never called.
        """A rank with no parameters of its own carries a stub whose inner optimizer is not there to reset."""
        stub = _FakeChainedMember(is_stub_optimizer=True)

        _reset(monkeypatch, stub, _FakeChainedMember())

        assert stub.optimizer.state != {} and stub.zero_grad_calls == []


class TestResettingARealAdam:
    # [CLASS DOC] Tests of the reset effects on real torch Adam optimizers.
    def test_a_reset_optimizer_holds_what_a_freshly_built_one_holds(self, monkeypatch):
        # [FUNCTION DOC] An Adam stepped 3 times (state non-empty) is reset; its state dictionary
        # must then equal that of a never-stepped Adam.
        """The fakes above only prove the calls happen; a real Adam has to end up where a new one starts."""
        _, optimizer = _adam_with_history(steps=3)
        assert optimizer.state_dict()["state"]

        _reset(monkeypatch, _RealChainedMember(optimizer))

        _assert_same_optimizer_state(optimizer, _adam_with_history(steps=0)[1])

    def test_a_reset_optimizer_then_steps_like_a_freshly_built_one(self, monkeypatch):
        # [FUNCTION DOC] Resets an Adam stepped 3 times, then takes one more fixed-gradient step on
        # it and on a fresh Adam with a fresh parameter; the two state dictionaries must be equal.
        """Matching state dicts could still diverge on the next step, which is the thing a rolled-back run does next."""
        param, optimizer = _adam_with_history(steps=3)
        _reset(monkeypatch, _RealChainedMember(optimizer))
        fresh_param, fresh_optimizer = _adam_with_history(steps=0)

        _step_on_fixed_gradient(param, optimizer)
        _step_on_fixed_gradient(fresh_param, fresh_optimizer)

        _assert_same_optimizer_state(optimizer, fresh_optimizer)

    def test_a_fused_adam_reallocates_the_moments_a_megatron_optimizer_expects_to_find(self, monkeypatch):
        # [FUNCTION DOC] Resets a stepped real Adam with a state-init function and the
        # PyTorch-optimizer flag off. Asserts the parameter's state has exactly first and second
        # moment entries and both are all zeros.
        """Outside the torch path mcore keeps the moments allocated, and a step onto an absent state raises."""
        param, optimizer = _adam_with_history(steps=3)

        _reset(
            monkeypatch,
            _RealChainedMember(optimizer, init_state_fn=_zero_adam_state),
            using_pytorch_optimizer=False,
        )

        state = optimizer.state[param]
        assert set(state) == {"exp_avg", "exp_avg_sq"}
        assert torch.count_nonzero(state["exp_avg"]) == 0 and torch.count_nonzero(state["exp_avg_sq"]) == 0

    def test_a_reset_drops_the_gradient_a_rolled_back_trainer_still_holds(self, monkeypatch):
        # [FUNCTION DOC] After a reset of an Adam stepped once, the parameter's gradient is none.
        """That gradient was computed against the weights being thrown away, so applying it would corrupt the load."""
        param, optimizer = _adam_with_history(steps=1)

        _reset(monkeypatch, _RealChainedMember(optimizer))

        assert param.grad is None


class TestTheSelfCheckAResetRuns:
    # [CLASS DOC] Tests of the post-reset self-check, which must accept a clean reset and raise an
    # assertion error when any moment, step counter, gradient or unexpected state key remains, or
    # when no real parameter was reset at all.
    def test_it_passes_over_a_fused_optimizer_the_reset_just_reinitialized(self, monkeypatch):
        # [FUNCTION DOC] A fake member with the zeroed-moments initializer and the PyTorch flag
        # off resets without raising.
        """The fused path leaves zeroed moments rather than nothing, and both are what a fresh optimizer holds."""
        _reset(monkeypatch, _FakeChainedMember(init_state_fn=_zero_adam_state), using_pytorch_optimizer=False)

    def test_a_moment_left_behind_is_caught(self, monkeypatch):
        # [FUNCTION DOC] The state-init function writes a ones-valued first moment only; the
        # reset must raise an assertion error.
        """A reset that missed a moment would let the run continue on the very state it meant to discard."""
        member = _FakeChainedMember(
            init_state_fn=lambda inner, config: inner.state.update({inner.param: {"exp_avg": torch.ones(2)}})
        )

        with pytest.raises(AssertionError):
            _reset(monkeypatch, member, using_pytorch_optimizer=False)

    def test_a_step_counter_left_on_the_param_group_is_caught(self, monkeypatch):
        # [FUNCTION DOC] The state-init function re-sets the group's step counter to 7; the reset
        # must raise an assertion error.
        """A fused adam keeps its step there, and a reset that leaves it resumes the schedule mid-run."""
        member = _FakeChainedMember(init_state_fn=lambda inner, config: inner.param_groups[0].update(step=7))

        with pytest.raises(AssertionError):
            _reset(monkeypatch, member, using_pytorch_optimizer=False)

    def test_a_gradient_left_behind_is_caught(self, monkeypatch):
        # [FUNCTION DOC] The member's zero-grad is replaced by a no-op, leaving the gradient; the
        # reset must raise an assertion error.
        """That gradient was computed against the weights being thrown away."""
        member = _FakeChainedMember()
        monkeypatch.setattr(member, "zero_grad", lambda set_to_none=True: None)

        with pytest.raises(AssertionError):
            _reset(monkeypatch, member)

    def test_an_optimizer_that_reset_nothing_at_all_is_caught(self, monkeypatch):
        # [FUNCTION DOC] A chain consisting only of a stub member makes the reset raise an
        # assertion error because no parameter was ever seen.
        """A chain of nothing but stubs would let every assertion above pass without ever seeing a parameter."""
        with pytest.raises(AssertionError):
            _reset(monkeypatch, _FakeChainedMember(is_stub_optimizer=True))

    def test_a_state_holding_keys_no_adam_uses_is_caught(self, monkeypatch):
        # [FUNCTION DOC] The state-init function writes a state with a momentum key that Adam does
        # not use; the reset must raise an assertion error (the set of state keys is validated).
        """The value checks say nothing about a key they never see, so the set of keys is checked as well."""
        member = _FakeChainedMember(
            init_state_fn=lambda inner, config: inner.state.update({inner.param: {"momentum": torch.zeros(2)}})
        )

        with pytest.raises(AssertionError):
            _reset(monkeypatch, member, using_pytorch_optimizer=False)


class TestWhatAResetRefusesToWalk:
    # [CLASS DOC] Tests of configurations for which the reset must refuse with an assertion error.
    def test_an_optimizer_streaming_its_state_to_disk_is_refused(self, monkeypatch):
        # [FUNCTION DOC] With the stream-state-to-disk flag set, the reset raises an assertion
        # error whose message mentions the stream-optimizer-state-to-disk option.
        """Its moments live in the nvme store, so this walk would clear nothing and report a clean reset."""
        with pytest.raises(AssertionError, match="stream-optimizer-state-to-disk"):
            _reset(monkeypatch, _FakeChainedMember(), stream_optimizer_state_to_disk=True)

    def test_an_fp16_optimizer_is_refused(self, monkeypatch):
        # [FUNCTION DOC] A chained optimizer whose config has fp16 on makes the reset raise an
        # assertion error mentioning the loss scaler. Calls the reset function directly (not the
        # local helper) because the helper would rebuild the config.
        """The loss scaler sits beside the state this clears, and a reset that leaves it is a half reset."""
        optimizer_utils = pytest.importorskip("miles.backends.megatron_utils.optimizer_utils")
        optimizer = _chained_optimizer(_FakeChainedMember())
        optimizer.config = SimpleNamespace(offload_optimizer_states=False, fp16=True)

        with pytest.raises(AssertionError, match="loss scaler"):
            optimizer_utils.reset_optimizer_state(
                optimizer, stream_optimizer_state_to_disk=False, chunked_optimizer_state_offload=False
            )

    def test_a_chunked_offload_is_refused(self, monkeypatch):
        # [FUNCTION DOC] With the chunked-offload flag set, the reset raises an assertion error
        # mentioning the chunked-optimizer-state-offload option.
        """The offloader's cpu store keeps canonical moments this reset never reaches."""
        with pytest.raises(AssertionError, match="chunked-optimizer-state-offload"):
            _reset(monkeypatch, _FakeChainedMember(), chunked_optimizer_state_offload=True)

    def test_an_optimizer_offloading_its_states_is_refused(self, monkeypatch):
        # [FUNCTION DOC] A chained optimizer whose config has state offload on makes the reset
        # raise an assertion error. Calls the reset function directly with both flags off.
        """The offloader would copy the old moments back over the reset from the buffer it keeps."""
        optimizer_utils = pytest.importorskip("miles.backends.megatron_utils.optimizer_utils")
        optimizer = _chained_optimizer(_FakeChainedMember())
        optimizer.config = SimpleNamespace(offload_optimizer_states=True, fp16=False)

        with pytest.raises(AssertionError):
            optimizer_utils.reset_optimizer_state(
                optimizer, stream_optimizer_state_to_disk=False, chunked_optimizer_state_offload=False
            )
