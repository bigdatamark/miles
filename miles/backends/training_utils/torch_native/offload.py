# [MODULE DOC] HELPER: Offload utilities that move all model parameters/buffers and optimizer state
# tensors between GPU memory and pinned CPU memory, in place on the module objects.
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn


def _to_pinned_host(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Allocate an empty pinned CPU tensor like the input and copy into it
    # asynchronously (non-blocking); returns the new CPU tensor.
    return torch.empty_like(tensor, device="cpu", pin_memory=True).copy_(tensor, non_blocking=True)


def _to_cuda(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Copy a tensor to the default CUDA device asynchronously; returns it.
    return tensor.to("cuda", non_blocking=True)


def _map_optimizer_state(optimizers: Iterable[torch.optim.Optimizer], fn: Callable) -> None:
    # [FUNCTION DOC] HELPER: For every optimizer and every parameter's state mapping, replace each
    # tensor-valued entry by fn(entry) in place (non-tensor entries untouched).
    for optimizer in optimizers:
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = fn(value)


@torch.no_grad()
def move_train_state(model_parts: list[nn.Module], optimizers: Iterable[torch.optim.Optimizer], device: str) -> None:
    # [FUNCTION DOC] Run without autograd. Moves all parameters and buffers of each module (using
    # the module's apply mechanism, so parameter objects are kept) and all optimizer state tensors
    # to pinned host memory if the device string is exactly "cpu", otherwise to CUDA. Copies are
    # asynchronous; ends with a CUDA synchronize so the move is complete on return. Mutates the
    # modules and optimizers; returns nothing.
    fn = _to_pinned_host if device == "cpu" else _to_cuda
    for module in model_parts:
        module._apply(fn)
    _map_optimizer_state(optimizers, fn)
    torch.cuda.synchronize()
