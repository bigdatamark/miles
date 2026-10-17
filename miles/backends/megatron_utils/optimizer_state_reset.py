# [MODULE DOC] Implements zeroing of optimizer history (moments, momentum buffers and step
# counters) for the reset-optimizer-states option. It walks Megatron's wrapped optimizers down to
# the underlying torch optimizers, picks the set of history state keys per optimizer type (Adam
# and CPU Adam, and the tensor-parallel Muon variants) and zeroes them in place. Any unknown
# optimizer type, unknown state key or unsupported step type raises an unsupported-state error
# rather than silently skipping, so new optimizers must be taught here.
"""Reset optimizer history for --reset-optimizer-states, failing closed on state it does not know.

The Muon classes are imported where they are matched: Megatron builds without
``emerging_optimizers`` (the NPU pin) still run Adam.
"""

from collections.abc import Iterator

import torch
from megatron.core.optimizer import Adam, CPUAdam


class UnsupportedOptimizerState(RuntimeError):
    # [CLASS DOC] Runtime error raised when the reset code meets an optimizer type, state key or
    # step value type it has no handling for.
    """An optimizer or state key this module has no schema for; teach _history_keys rather than guessing."""


def reset_optimizer_states(optimizer) -> None:
    # [FUNCTION DOC] Zeroes the history of the given (possibly wrapped, possibly none) Megatron
    # optimizer in place: for each underlying torch optimizer, determines its history keys and
    # resets its step counters and history tensors. Raises the unsupported-state error for unknown
    # optimizer types or state. A none optimizer is a no-op. Returns nothing.
    for leaf in _leaves(optimizer):
        _reset(leaf, _history_keys(leaf))


def _leaves(optimizer) -> Iterator[torch.optim.Optimizer]:
    # [FUNCTION DOC] HELPER: Lazily yields the leaf optimizers. None yields nothing. If the object
    # has a list of chained optimizers or of sub optimizers (checked in that order), recurses into
    # each child and stops. Otherwise, if it wraps an inner optimizer, recurses into that; else the
    # object itself is a leaf and is yielded.
    """Descend Megatron's wrappers to the torch optimizers that own the state."""
    if optimizer is None:
        return
    for children in ("chained_optimizers", "sub_optimizers"):
        if hasattr(optimizer, children):
            for child in getattr(optimizer, children):
                yield from _leaves(child)
            return
    if hasattr(optimizer, "optimizer"):
        yield from _leaves(optimizer.optimizer)
    else:
        yield optimizer


def _history_keys(optimizer) -> frozenset[str]:
    # [FUNCTION DOC] HELPER: Returns the set of state keys holding history for the optimizer type:
    # first and second moment for Adam and CPU Adam; momentum and second-moment buffers for the
    # adaptive tensor-parallel Muon; momentum buffer for tensor-parallel Muon. The Muon classes
    # are imported lazily (they may be unavailable on some builds). Any other type raises the
    # unsupported-state error naming the type.
    # KNOWN ISSUE: if the emerging-optimizers import is unavailable, a non-Adam optimizer fails with
    # an import error rather than the unsupported-state error.
    if isinstance(optimizer, (Adam, CPUAdam)):
        return frozenset({"exp_avg", "exp_avg_sq"})
    from megatron.core.optimizer.emerging_optimizers import TensorParallelAdaptiveMuon, TensorParallelMuon

    if isinstance(optimizer, TensorParallelAdaptiveMuon):
        return frozenset({"momentum_buffer", "moment2_buffer"})
    if isinstance(optimizer, TensorParallelMuon):
        return frozenset({"momentum_buffer"})
    raise UnsupportedOptimizerState(f"no reset schema for {type(optimizer).__name__}")


def _reset(optimizer, history: frozenset[str]) -> None:
    # [FUNCTION DOC] HELPER: Resets one torch optimizer. Resets the step in each param group, then
    # for every per-parameter state: raises the unsupported-state error if it holds keys other
    # than the history keys, the step, and the master param; resets its step; zeroes in place every
    # history tensor present. The master (fp32) parameters are left untouched.
    for group in optimizer.param_groups:
        _reset_step(group)
    for state in optimizer.state.values():
        unknown = set(state) - history - {"step", "master_param"}
        if unknown:
            raise UnsupportedOptimizerState(f"unknown state keys {sorted(unknown)} on {type(optimizer).__name__}")
        _reset_step(state)
        for key in history & set(state):
            state[key].zero_()


def _reset_step(container) -> None:
    # [FUNCTION DOC] HELPER: Zeroes the "step" entry of a mapping (param group or state): a tensor
    # is zeroed in place, a plain number is replaced by integer zero, a missing entry is ignored,
    # and any other type raises the unsupported-state error.
    step = container.get("step")
    if isinstance(step, torch.Tensor):
        step.zero_()
    elif isinstance(step, (int, float)):
        container["step"] = 0
    elif step is not None:
        raise UnsupportedOptimizerState(f"unsupported step type {type(step).__name__}")
