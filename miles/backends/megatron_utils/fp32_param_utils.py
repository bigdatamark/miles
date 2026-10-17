# [MODULE DOC] HELPER:
# Utilities letting model definitions pin selected parameters to a specific dtype (typically
# float32) even though Megatron's half-precision wrapper casts all floating parameters to bf16/fp16
# at wrap time. Model code marks a parameter with a private attribute; after the model is built,
# the enforcement function re-casts the marked parameters in place.
import logging
from collections.abc import Sequence

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# Parameter attribute used by model definitions to pin parameter dtype.
FORCED_PARAM_DTYPE_ATTR = "_miles_forced_param_dtype"


def mark_param_dtype(param: torch.nn.Parameter, dtype: torch.dtype) -> None:
    # [FUNCTION DOC] HELPER: Sets the forced-dtype attribute on the parameter object to the dtype.
    # Mutates the parameter's attributes only.
    """Mark a parameter with its required runtime dtype."""
    setattr(param, FORCED_PARAM_DTYPE_ATTR, dtype)


def enforce_marked_param_dtypes(model_chunks: Sequence[torch.nn.Module]) -> list[str]:
    # [FUNCTION DOC]
    # Iterates all named parameters of all chunks; for those carrying the forced-dtype attribute,
    # replaces the parameter's data in place with a copy cast to the target dtype when it differs
    # (keeping the parameter object identity so optimizer and DDP maps stay valid). Returns the
    # names of all marked parameters (including those already of the right dtype), in iteration
    # order. Logs the count on rank zero (rank zero assumed when distributed is not initialized)
    # only when something was marked.
    """Apply dtype overrides declared on parameters via ``mark_param_dtype``.

    This keeps the policy in model definitions and avoids model-name checks in
    the training/conversion mainline.

    Motivation: Megatron's ``Float16Module`` unconditionally casts every
    floating-point parameter to bf16/fp16 at wrap time, and there is no
    declarative opt-out in nn.Module or Megatron. Megatron's MoE router hits the
    same problem and solves it with ``_maintain_float32_expert_bias`` (see
    ``megatron/core/transformer/moe/router.py``), which post-hoc casts the
    expert_bias back to fp32. This function generalizes that pattern: callers
    mark params with their required dtype at the model-definition site, and we
    re-cast after ``get_model`` so the rest of the stack (optimizer, DDP, mbridge
    load path) sees the intended dtype.
    """
    updated_names: list[str] = []
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            target_dtype = getattr(param, FORCED_PARAM_DTYPE_ATTR, None)
            if target_dtype is None:
                continue

            if param.dtype != target_dtype:
                # Keep Parameter identity to avoid breaking optimizer/DDP maps.
                param.data = param.data.to(dtype=target_dtype)
            updated_names.append(name)

    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    if rank == 0 and updated_names:
        logger.info("Enforced marked parameter dtypes for %d tensors.", len(updated_names))
    return updated_names
