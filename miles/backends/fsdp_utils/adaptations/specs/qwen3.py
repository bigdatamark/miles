# [MODULE DOC] Registers, at import, a precision policy hook and a model instance patch hook for
# dense Qwen3 models, both active only under the formal true-on-policy contract (true on policy
# mode on and the run's contract name equal to the Qwen3 dense v1 contract): the policy computes
# in float32 parameters with bfloat16 autocast and a custom sync dtype rule, requiring bf16
# training and an fp32 master; the instance patch rewires the built model to match inference.
"""Qwen3 dense adaptations for the formal true-on-policy precision contract."""

from dataclasses import replace

import torch

from miles.true_on_policy.contracts import QWEN3_DENSE_TRUE_ON_POLICY_V1

from ..class_patches import ModelInstancePatchHook, register_model_instance_patch
from ..precision import PrecisionPolicyHook, register_precision_policy


def _is_qwen3(hf_config) -> bool:
    # [FUNCTION DOC] HELPER: True when the config's model type (empty text if missing) equals
    # exactly qwen3 (dense; not the MoE type).
    return str(getattr(hf_config, "model_type", "") or "") == "qwen3"


def _uses_formal_contract(hf_config, args) -> bool:
    # [FUNCTION DOC] HELPER: True when the model is dense Qwen3 AND the true on policy mode arg is
    # truthy (default false) AND the args' sglang true-on-policy contract equals the name of the
    # Qwen3 dense v1 contract (default none). Used as the precision hook predicate.
    return (
        _is_qwen3(hf_config)
        and getattr(args, "true_on_policy_mode", False)
        and getattr(args, "sglang_true_on_policy_contract", None) == QWEN3_DENSE_TRUE_ON_POLICY_V1.name
    )


def _resolve_precision(base_policy, hf_config, args):
    # [FUNCTION DOC] HELPER: Precision hook action. Raises a value error naming the contract if
    # fp16 training is requested or if the base policy does not keep an fp32 master. Otherwise
    # returns a copy of the base policy with parameter dtype float32, autocast dtype bfloat16 and
    # the Qwen3-specific sync dtype resolver. The config is unused.
    if getattr(args, "fp16", False):
        raise ValueError(f"{QWEN3_DENSE_TRUE_ON_POLICY_V1.name} requires bf16 training")
    if not base_policy.keep_fp32_master:
        raise ValueError(f"{QWEN3_DENSE_TRUE_ON_POLICY_V1.name} requires fp32 master weights")
    return replace(
        base_policy,
        param_dtype=torch.float32,
        autocast_dtype=torch.bfloat16,
        sync_dtype_resolver=_resolve_sync_dtype,
    )


def _resolve_sync_dtype(name, checkpoint_dtype):
    # [FUNCTION DOC] HELPER: Sync dtype rule for a parameter name and its checkpoint dtype:
    # lazily imports and delegates to the Qwen3 dense model package's resolver and returns its
    # dtype.
    from ...models.qwen3 import resolve_qwen3_dense_sync_dtype

    return resolve_qwen3_dense_sync_dtype(name, checkpoint_dtype)


def _instance_patch_applies(hf_config, args) -> bool:
    # [FUNCTION DOC] HELPER: Instance hook predicate: the formal contract applies and fp16 is not
    # requested. (The fp16 check is redundant in practice because the precision hook already
    # raises for fp16 under the contract.)
    return _uses_formal_contract(hf_config, args) and not getattr(args, "fp16", False)


def _apply_model_patch(model) -> None:
    # [FUNCTION DOC] HELPER: Instance hook action: lazily import and apply the Qwen3 dense true
    # on policy patch to the built model, mutating it.
    from ...models.qwen3 import apply_qwen3_dense_true_on_policy_patch

    apply_qwen3_dense_true_on_policy_patch(model)


register_precision_policy(PrecisionPolicyHook("qwen3_dense_true_on_policy", _uses_formal_contract, _resolve_precision))
register_model_instance_patch(
    ModelInstancePatchHook("qwen3_dense_true_on_policy", _instance_patch_applies, _apply_model_patch)
)
