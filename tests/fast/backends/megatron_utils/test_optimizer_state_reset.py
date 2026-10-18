# [MODULE DOC]
# Tests for the optimizer-state reset utility of the Megatron backend, which zeroes the training
# history of every leaf optimizer nested inside Megatron's chained, wrapped and hybrid-device
# optimizer containers. Covers Muon-style and Adam-style leaves (including group-level and
# tensor-valued step clocks), skipping of empty wrappers, a no-op before the first step, and
# rejection of unknown state keys or optimizer classes. Requires the real Megatron emerging
# optimizers package at import time; the helper builders below create small CPU optimizers.
"""reset_optimizer_states must reach every leaf optimizer under Megatron's wrappers and clear its
history by class, not by the --optimizer string (which Megatron rewrites before training starts)."""

import sys
from types import SimpleNamespace

import pytest
import torch
from megatron.core.optimizer.emerging_optimizers import TensorParallelAdaptiveMuon, TensorParallelMuon

from miles.backends.megatron_utils.optimizer_state_reset import UnsupportedOptimizerState, reset_optimizer_states


def stepped_adamw(*shapes):
    # [FUNCTION DOC] HELPER: Creates one ones-valued parameter per given shape, gives each a
    # ones gradient, builds an AdamW optimizer (learning rate 1e-3) and takes one step, so the
    # optimizer has nonzero moment estimates and a step counter. Returns the optimizer.
    params = [torch.nn.Parameter(torch.ones(*shape)) for shape in shapes]
    for param in params:
        param.grad = torch.ones_like(param)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    optimizer.step()
    return optimizer


class GroupStepAdamW(torch.optim.AdamW):
    # [CLASS DOC] HELPER: AdamW variant imitating a fused Adam whose step counter lives in the
    # parameter groups instead of the per-parameter state.
    """Shaped like TE FusedAdam: the step clock lives in param_groups, not in state."""

    def step(self, closure=None):
        # [FUNCTION DOC] HELPER: Performs the normal AdamW step, then sets every group's step to
        # 50 and deletes the step entry from every parameter's state. Returns nothing (the base
        # step's return value is discarded).
        super().step(closure)
        for group in self.param_groups:
            group["step"] = 50
        for state in self.state.values():
            del state["step"]


def muon_with_momentum(*shapes, cls=TensorParallelMuon, buffers=("momentum_buffer",)):
    # [FUNCTION DOC] HELPER: Creates an instance of the given Muon class (default tensor-parallel
    # Muon) without running its constructor, with a single param group holding ones-valued
    # parameters of the given shapes and a state mapping each parameter to ones-valued tensors
    # under each name in the buffers tuple (default: one momentum buffer). Returns the optimizer.
    optimizer = cls.__new__(cls)
    optimizer.param_groups = [{"params": [torch.nn.Parameter(torch.ones(*shape)) for shape in shapes]}]
    optimizer.state = {p: {name: torch.ones_like(p) for name in buffers} for p in optimizer.param_groups[0]["params"]}
    return optimizer


def wrap(leaf):
    # [FUNCTION DOC] HELPER: Returns a simple object imitating a Megatron optimizer wrapper whose
    # optimizer attribute is the given leaf (which may be none).
    return SimpleNamespace(optimizer=leaf)


def chain(*children):
    # [FUNCTION DOC] HELPER: Returns a simple object imitating a chained optimizer whose
    # chained-optimizers attribute is a list of the given children.
    return SimpleNamespace(chained_optimizers=list(children))


def assert_adam_reset(optimizer):
    # [FUNCTION DOC] HELPER: Asserts that an Adam-style optimizer is fully reset: every group that
    # has a step entry has it equal to 0, and for every parameter state the first and second
    # moment tensors are all zero and the step entry, if present, is 0.
    for group in optimizer.param_groups:
        if "step" in group:
            assert group["step"] == 0
    for state in optimizer.state.values():
        assert not state["exp_avg"].any()
        assert not state["exp_avg_sq"].any()
        if "step" in state:
            assert state["step"] == 0


def test_layer_wise_muon_tree_resets_both_leaves():
    # [FUNCTION DOC] Builds a Muon leaf and a stepped AdamW leaf, wraps both, nests them in a
    # chain inside another chain (as the layer-wise distributed Muon layout does), resets, and
    # asserts the Muon momentum buffers are zero and the Adam leaf is fully reset.
    muon, adam = muon_with_momentum((4, 4)), stepped_adamw((4,))
    # dist_muon: LayerWiseDistributedOptimizer(ChainedOptimizer) holding Float16 wrappers of Muon + Adam,
    # itself a child of the top-level ChainedOptimizer
    reset_optimizer_states(chain(chain(wrap(muon), wrap(adam))))

    assert all(not state["momentum_buffer"].any() for state in muon.state.values())
    assert_adam_reset(adam)


def test_adaptive_muon_resets_both_moment_buffers():
    # [FUNCTION DOC] An adaptive Muon leaf with momentum and second-moment buffers is reset;
    # asserts both buffers are zero for every parameter.
    adaptive = muon_with_momentum(
        (4, 4), cls=TensorParallelAdaptiveMuon, buffers=("momentum_buffer", "moment2_buffer")
    )

    reset_optimizer_states(chain(wrap(adaptive)))

    for state in adaptive.state.values():
        assert not state["momentum_buffer"].any()
        assert not state["moment2_buffer"].any()


def test_adam_reset_does_not_need_emerging_optimizers(monkeypatch):
    # [FUNCTION DOC] Makes the emerging-optimizers module unimportable (its module table entry
    # set to none, via monkeypatch). Resetting a plain Adam leaf must still work and fully reset
    # it, whereas resetting a Muon-like leaf must raise an import error.
    monkeypatch.setitem(sys.modules, "megatron.core.optimizer.emerging_optimizers", None)
    adam = stepped_adamw((4,))

    reset_optimizer_states(chain(wrap(adam)))

    assert_adam_reset(adam)
    with pytest.raises(ImportError):
        reset_optimizer_states(chain(wrap(muon_with_momentum((4, 4)))))


def test_hybrid_device_optimizer_resets_cpu_and_gpu_children():
    # [FUNCTION DOC] A hybrid container with two sub-optimizers (a normal stepped AdamW and a
    # group-step variant stepped once, with a gradient set directly) is wrapped and reset.
    # Asserts both children are fully reset, showing sub-optimizers are traversed.
    cpu_child, gpu_child = stepped_adamw((4,)), GroupStepAdamW([torch.nn.Parameter(torch.ones(4))], lr=1e-3)
    gpu_child.param_groups[0]["params"][0].grad = torch.ones(4)
    gpu_child.step()
    hybrid = SimpleNamespace(sub_optimizers=[cpu_child, gpu_child], state={}, param_groups=[])

    reset_optimizer_states(chain(wrap(hybrid)))

    assert_adam_reset(cpu_child)
    assert_adam_reset(gpu_child)


def test_tensor_valued_group_step_resets_in_place():
    # [FUNCTION DOC] Sets a group's step to a tensor holding 50, resets, and asserts that same
    # tensor object now holds 0 (it is zeroed in place rather than replaced).
    adam = stepped_adamw((4,))
    clock = torch.tensor(50)
    adam.param_groups[0]["step"] = clock

    reset_optimizer_states(chain(wrap(adam)))

    assert clock.item() == 0


def test_wrapper_without_an_optimizer_is_skipped():
    # [FUNCTION DOC] A chain with an empty wrapper (optimizer none) followed by an Adam wrapper:
    # reset does not fail and the Adam leaf is reset.
    adam = stepped_adamw((4,))
    reset_optimizer_states(chain(wrap(None), wrap(adam)))
    assert_adam_reset(adam)


def test_reset_before_the_first_step_is_a_no_op():
    # [FUNCTION DOC] Resetting a never-stepped AdamW (empty state) completes without error.
    reset_optimizer_states(chain(wrap(torch.optim.AdamW([torch.nn.Parameter(torch.ones(4))], lr=1e-3))))


def test_unknown_adam_state_key_is_rejected():
    # [FUNCTION DOC] Adds an unrecognized state entry (max second moment, as in AMSGrad) to a
    # stepped AdamW's state; reset raises the unsupported-optimizer-state error naming the key.
    adam = stepped_adamw((4,))
    next(iter(adam.state.values()))["max_exp_avg_sq"] = torch.ones(4)
    with pytest.raises(UnsupportedOptimizerState, match="max_exp_avg_sq"):
        reset_optimizer_states(chain(wrap(adam)))


def test_unknown_muon_state_key_is_rejected():
    # [FUNCTION DOC] Adds a second-moment buffer entry to a plain (non-adaptive) Muon's state;
    # reset raises the unsupported-optimizer-state error naming that key.
    muon = muon_with_momentum((4, 4))
    next(iter(muon.state.values()))["moment2_buffer"] = torch.ones(4, 4)
    with pytest.raises(UnsupportedOptimizerState, match="moment2_buffer"):
        reset_optimizer_states(chain(wrap(muon)))


def test_unknown_optimizer_class_is_rejected():
    # [FUNCTION DOC] An SGD optimizer with momentum is not a supported class; reset raises the
    # unsupported-optimizer-state error naming SGD.
    sgd = torch.optim.SGD([torch.nn.Parameter(torch.ones(4))], lr=1e-3, momentum=0.9)
    with pytest.raises(UnsupportedOptimizerState, match="SGD"):
        reset_optimizer_states(chain(wrap(sgd)))
