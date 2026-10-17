# [MODULE DOC] Registers, at import, three hooks for the qwen3_moe architecture: the weight sync
# transform that unfuses batched expert tensors into per-expert names, a config-time patch of the
# mixture-of-experts block active only in true on policy mode, and a routing replay adapter on the
# Qwen3 MoE top-k router class.
"""qwen3_moe adaptations: the train->rollout weight transform (unfuse transformers>=5.6 batched experts
into the per-expert names sglang expects, via HF's own reverse conversion), the true-on-policy
config-time MoE-block patch, and the rollout-routing-replay hook."""

from miles.backends.fsdp_utils.models.replay_routers import install_qwen3_router_replay
from ..class_patches import ModelPatchHook, register_model_patch
from ..routing_replay import RoutingReplayAdapter, register_routing_replay_adapter
from ..weight_bridge import _batched_experts_matches, _hf_unfuse_experts_expand, register_param_transform

register_param_transform("qwen3_moe", _batched_experts_matches, _hf_unfuse_experts_expand)


def _is_qwen3_moe(hf_config) -> bool:
    # [FUNCTION DOC] HELPER: True when the config's model type (empty text if missing) equals
    # exactly qwen3_moe.
    return str(getattr(hf_config, "model_type", "") or "") == "qwen3_moe"


def _apply_moe_patch(hf_config, args) -> None:
    # [FUNCTION DOC] HELPER: Config-time hook action. Does nothing unless the args' true on policy
    # mode is truthy (default false); then lazily imports (it needs sglang fused MoE kernels that
    # are not in every build) and calls the function that patches the transformers MoE block class
    # before the model is constructed. The config is unused.
    """Patch the MoE block before construction in true-on-policy mode. The import stays lazy because it
    pulls sglang fused-MoE kernels that don't exist for every sglang build."""
    if not getattr(args, "true_on_policy_mode", False):
        return
    from ...models.qwen3_moe import apply_true_on_policy_patch_for_qwen3_moe

    apply_true_on_policy_patch_for_qwen3_moe()


register_model_patch(ModelPatchHook("qwen3_moe_moe_patch", _is_qwen3_moe, _apply_moe_patch))


register_routing_replay_adapter(
    RoutingReplayAdapter(
        name="qwen3_moe",
        applies_to=_is_qwen3_moe,
        module_cls_name="Qwen3MoeTopKRouter",
        install=install_qwen3_router_replay,
    )
)
