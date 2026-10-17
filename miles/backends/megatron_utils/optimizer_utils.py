# [MODULE DOC] Utilities to wipe a Megatron chained optimizer's state back to a freshly
# initialized one and self-check that the wipe left nothing behind. Only the Adam-style state of
# non-stub sub-optimizers is understood; unsupported configurations are rejected up front by
# assertions (fp16 loss scaler, optimizer-state offload, chunked offload, disk streaming).
import torch
from megatron.core.optimizer import USING_PYTORCH_OPTIMIZER
from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer


def reset_optimizer_state(
    optimizer: MegatronOptimizer, *, stream_optimizer_state_to_disk: bool, chunked_optimizer_state_offload: bool
) -> None:
    # [FUNCTION DOC] Resets a chained optimizer in place to its just-initialized state; returns
    # nothing. Both flags are keyword-only. Preconditions, each an assertion error with an
    # explanatory message where applicable: the optimizer is a chained optimizer; optimizer-state
    # offload is off in its config; chunked optimizer state offload is off; fp16 is off (the loss
    # scaler would not be reset); disk streaming of optimizer state is off (the store would not be
    # cleared).
    # For each non-stub sub-optimizer: clears the inner optimizer's per-parameter state, removes the
    # step entry from every param group, re-runs the sub-optimizer's state initializer with its
    # config when one exists and the plain PyTorch optimizer is not in use, and zeroes (sets to
    # none) its gradients. Finally verifies the result with the empty-state check.
    assert isinstance(optimizer, ChainedOptimizer)
    assert not optimizer.config.offload_optimizer_states
    assert (
        not chunked_optimizer_state_offload
    ), "--chunked-optimizer-state-offload keeps the canonical moments outside the optimizer state this clears"
    assert not optimizer.config.fp16, (
        "an fp16 optimizer keeps a loss scaler with a scale and a growth tracker beside the state this clears, so "
        "the reset would leave the trainer with the scaler of the rollouts it just discarded"
    )
    assert not stream_optimizer_state_to_disk, (
        "--stream-optimizer-state-to-disk keeps the moments in its own store rather than in the master optimizer "
        "state this walks, so the reset would clear nothing and the self-check would pass on an empty walk"
    )

    for chained in optimizer.chained_optimizers:
        if chained.is_stub_optimizer:
            continue

        inner = chained.optimizer
        inner.state.clear()
        for param_group in inner.param_groups:
            param_group.pop("step", None)

        if chained.init_state_fn is not None and not USING_PYTORCH_OPTIMIZER:
            chained.init_state_fn(inner, chained.config)

        chained.zero_grad(set_to_none=True)

    _check_optimizer_state_empty(optimizer)


def _check_optimizer_state_empty(optimizer: MegatronOptimizer) -> None:
    # [FUNCTION DOC] HELPER: Self-check after a reset; raises assertion errors on any violation.
    # Over non-stub sub-optimizers: no param group has a step entry; every parameter has no
    # gradient; every state value is all-zero (tensors via a nonzero count, other values by
    # equality with zero). Also requires at least one optimizer and one parameter to have been
    # inspected (guards against an empty walk), and that the union of state keys seen equals no keys
    # for the plain PyTorch optimizer, else exactly the Adam first and second moment keys (extend
    # here for other optimizers).
    checked_optimizers = 0
    checked_params = 0
    checked_state_keys: set[str] = set()

    for chained in optimizer.chained_optimizers:
        if chained.is_stub_optimizer:
            continue

        checked_optimizers += 1
        inner = chained.optimizer
        for param_group in inner.param_groups:
            assert "step" not in param_group
            for param in param_group["params"]:
                checked_params += 1
                assert param.grad is None

        for state in inner.state.values():
            checked_state_keys |= set(state)
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    assert torch.count_nonzero(value) == 0
                else:
                    assert value == 0

    assert checked_optimizers > 0
    assert checked_params > 0
    # if you use a non-adam optimizer, add the asserts for its state keys here
    assert checked_state_keys == (set() if USING_PYTORCH_OPTIMIZER else {"exp_avg", "exp_avg_sq"})
