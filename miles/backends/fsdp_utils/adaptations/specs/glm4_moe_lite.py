# [MODULE DOC] Registers, at import, the hooks for the glm4_moe_lite architecture: a weight sync
# transform that unfuses batched expert tensors into per-expert names (same rule as qwen3_moe),
# and a routing replay adapter on its mixture-of-experts module class (the router lives in that
# module). Defines one predicate; everything else is registration side effects.
"""glm4_moe_lite (GLM-4.7-Flash) adaptations: the train-to-rollout weight transform (its batched
expert layout matches qwen3_moe, so it reuses the same HF-native unfuse) and the
rollout-routing-replay hook on its group-limited router."""

from miles.backends.fsdp_utils.models.replay_routers import install_glm4_moe_lite_router_replay
from ..routing_replay import RoutingReplayAdapter, register_routing_replay_adapter
from ..weight_bridge import _batched_experts_matches, _hf_unfuse_experts_expand, register_param_transform

register_param_transform("glm4_moe_lite", _batched_experts_matches, _hf_unfuse_experts_expand)


def _is_glm4_moe_lite(hf_config) -> bool:
    # [FUNCTION DOC] HELPER: True when the config's model type (empty text if missing) equals
    # exactly the glm4_moe_lite type.
    return str(getattr(hf_config, "model_type", "") or "") == "glm4_moe_lite"


register_routing_replay_adapter(
    RoutingReplayAdapter(
        name="glm4_moe_lite",
        applies_to=_is_glm4_moe_lite,
        module_cls_name="Glm4MoeLiteMoE",
        install=install_glm4_moe_lite_router_replay,
    )
)
