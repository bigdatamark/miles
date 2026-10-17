# [MODULE DOC] Numeric precision decisions for the FSDP backend: a policy record (compute dtype,
# gradient reduction dtype, whether to keep an fp32 master copy of weights, optional autocast
# dtype and optional per-parameter dtype resolver for weight sync), a registry of hooks that let
# architectures override the policy, a resolver that starts from a base policy built from run args
# and folds in matching hooks, a forward-context factory and a function that converts a model to
# an fp32 master while remembering each parameter's outward dtype.
"""Precision policy for the FSDP backend.

Resolves the FSDP ``MixedPrecisionPolicy`` dtypes and whether to keep an fp32 master copy. The master copy is enabled by default for bit-exact weight sync and can be disabled for memory-constrained runs that only require forward accuracy.
"""

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass
class PrecisionPolicy:
    # [CLASS DOC] Mutable record. param_dtype: compute dtype of the FSDP mixed precision policy.
    # reduce_dtype: dtype of gradient reduction. keep_fp32_master: whether the model is converted
    # to an fp32 master copy (default true). autocast_dtype: if set, forward passes run under
    # autocast with it (default none). sync_dtype_resolver: optional function from parameter name
    # and checkpoint dtype to the dtype used when syncing that parameter to rollout (default none,
    # meaning the checkpoint dtype).
    param_dtype: torch.dtype  # FSDP MixedPrecisionPolicy compute dtype
    reduce_dtype: torch.dtype  # gradient all-reduce dtype
    keep_fp32_master: bool = True
    autocast_dtype: torch.dtype | None = None
    sync_dtype_resolver: Callable[[str, torch.dtype], torch.dtype] | None = None


@dataclass
class PrecisionPolicyHook:
    # [CLASS DOC] Mutable record of a precision override: a name, a predicate on the HF config and
    # run args, and a resolve function taking the current policy, config and args and returning
    # the policy to use next (it may return a modified copy or a new object).
    name: str
    applies_to: Callable  # (hf_config, args) -> bool
    resolve: Callable  # (base_policy, hf_config, args) -> PrecisionPolicy


_PRECISION_POLICY_HOOKS: list[PrecisionPolicyHook] = []


def register_precision_policy(hook: PrecisionPolicyHook) -> None:
    # [FUNCTION DOC] Append a hook to the module registry (no de-duplication).
    _PRECISION_POLICY_HOOKS.append(hook)


def resolve_precision_policy(hf_config, args) -> PrecisionPolicy:
    # [FUNCTION DOC] Build the effective policy. Base: compute dtype float16 if the args' fp16
    # attribute is truthy (default false) else bfloat16; reduction dtype float32; fp32 master flag
    # from the required keep-fp32-master arg (an attribute error if absent); no autocast and no
    # sync resolver. Then every registered hook, in order, whose predicate is true for the config
    # and args replaces the policy with the result of its resolve function (so later hooks see
    # earlier results). Returns the final policy.
    """Resolve compute, reduction, master-weight, and forward-autocast precision."""
    policy = PrecisionPolicy(
        param_dtype=torch.float16 if getattr(args, "fp16", False) else torch.bfloat16,
        reduce_dtype=torch.float32,
        keep_fp32_master=args.keep_fp32_master,
    )
    for hook in _PRECISION_POLICY_HOOKS:
        if hook.applies_to(hf_config, args):
            policy = hook.resolve(policy, hf_config, args)
    return policy


def precision_forward_context(policy: PrecisionPolicy):
    # [FUNCTION DOC] Return a context manager for forward passes: a no-op context when the policy
    # has no autocast dtype, otherwise a CUDA autocast context with that dtype (the device type
    # name cuda is used as is).
    if policy.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=policy.autocast_dtype)


def apply_fp32_master(
    model,
    sync_dtype_resolver: Callable[[str, torch.dtype], torch.dtype] | None = None,
):
    # [FUNCTION DOC] Turn a model into an fp32 master copy and record outbound sync dtypes.
    # First, for every entry of the model's state dict (taken BEFORE conversion), record the
    # entry's current dtype as the checkpoint dtype, or, when a resolver is given, the dtype the
    # resolver returns for the name and checkpoint dtype. Then convert the whole model to float32
    # (in place for module parameters and buffers; returns the converted module) and attach the
    # recorded name-to-dtype mapping to the model under a private attribute used later by weight
    # sync. Returns the model.
    """Convert ``model`` to an fp32 master and record each parameter's outbound sync dtype.

    The checkpoint dtype is the default. A model-specific precision policy may override it when the
    rollout contract stores selected parameters at a different dtype.
    """
    sync_dtypes = {}
    for name, param in model.state_dict().items():
        checkpoint_dtype = param.dtype
        sync_dtypes[name] = (
            sync_dtype_resolver(name, checkpoint_dtype) if sync_dtype_resolver is not None else checkpoint_dtype
        )
    model = model.to(torch.float32)
    model._fsdp_sync_dtypes = sync_dtypes
    return model
