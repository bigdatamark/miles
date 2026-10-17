# [PACKAGE DOC] Package for the "true-on-policy" feature: making the trainer (FSDP or Megatron)
# produce log-probabilities bit-aligned with the SGLang rollout engine. It is split into four
# layers: schema (declarative contract identity), contracts (contract objects plus the rules
# deriving kernel switches), model_profiles (per-model-family capabilities and Megatron model
# type names) and config (validation of a run's parallel layout and construction of a launch plan
# with CLI argument lists and environment variables). This file only re-exports the public
# names of those modules via the package-level export list.

"""True-on-policy launch contract helpers."""

from .config import (
    TrueOnPolicyArgList,
    TrueOnPolicyKernelPolicy,
    TrueOnPolicyLaunchPlan,
    TrueOnPolicyParallelLayout,
    apply_true_on_policy_script_defaults,
    build_true_on_policy_launch_plan,
)
from .contracts import QWEN3_DENSE_TRUE_ON_POLICY_V1, TrueOnPolicyContract, get_true_on_policy_contract
from .model_profiles import TrueOnPolicyModelProfile, get_megatron_model_type, get_true_on_policy_model_profile

__all__ = [
    "TrueOnPolicyLaunchPlan",
    "TrueOnPolicyArgList",
    "TrueOnPolicyKernelPolicy",
    "TrueOnPolicyContract",
    "TrueOnPolicyModelProfile",
    "TrueOnPolicyParallelLayout",
    "QWEN3_DENSE_TRUE_ON_POLICY_V1",
    "apply_true_on_policy_script_defaults",
    "build_true_on_policy_launch_plan",
    "get_true_on_policy_contract",
    "get_megatron_model_type",
    "get_true_on_policy_model_profile",
]
