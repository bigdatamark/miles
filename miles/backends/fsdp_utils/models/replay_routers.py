# [MODULE DOC]
# Per-architecture hooks that make MoE routers replay recorded expert selections (rollout routing
# replay) by swapping the router's top-k step for a function obtained from the shared routing
# replay manager. Each install function attaches that replacement top-k function to the module
# under a private attribute and rebinds the routing method on the instance. Supported: Qwen3 and
# Qwen3.5 MoE routers (token-level gate module) and GLM4 MoE Lite blocks (routing done in the block).
"""Per-architecture expert-selection hooks for rollout routing replay."""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from miles.utils.replay_base import routing_replay_manager


def _qwen3_router_forward(self, hidden_states):
    # [FUNCTION DOC] HELPER: Replacement forward bound onto a router instance. Flattens the input
    # to (tokens, hidden), computes router logits by a linear map with the router weight, a float32
    # softmax over experts, takes the top-k probabilities and indices via the replay-aware top-k
    # function (second argument is the router's top_k), renormalizes the selected probabilities to
    # sum to one unless the router's norm flag is explicitly false (default true), and casts them to
    # the logits dtype. Returns (router logits, top-k weights, top-k expert indices).
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = self._miles_replay_topk(router_probs, self.top_k)
    if getattr(self, "norm_topk_prob", True):
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    return router_logits, router_top_value, router_indices


def install_qwen3_router_replay(router: nn.Module) -> None:
    # [FUNCTION DOC]
    # Gets from the routing replay manager a top-k function (returning both values and indices;
    # underlying behavior is a plain top-k over the last axis when not replaying), stores it on the
    # router, and rebinds the router's forward to the replacement above. Mutates the instance only.
    """Hook ``Qwen3MoeTopKRouter`` / ``Qwen3_5MoeTopKRouter``, whose forwards are identical apart
    from qwen3_5 always renormalizing."""
    router._miles_replay_topk = routing_replay_manager.get_topk_fn(
        lambda scores, k: torch.topk(scores, k, dim=-1), return_probs=True
    )
    router.forward = types.MethodType(_qwen3_router_forward, router)


def _glm4_moe_lite_route_tokens_to_experts(self, router_logits):
    # [FUNCTION DOC] HELPER: Replacement routing method bound onto a GLM4 MoE Lite block.
    # Algorithm: sigmoid of the logits; add the gate's score-correction bias to get selection
    # scores; split experts into groups, score each group by the sum of its top two selection
    # scores, keep the best topk_group groups (mask built by scattering ones), set scores of experts
    # in other groups to negative infinity, then choose the top_k experts with the replay-aware
    # top-k function (indices only). Weights are gathered from the sigmoid outputs (not the
    # bias-corrected scores), renormalized by their sum plus a tiny constant when the norm flag is
    # on, and multiplied by the routed scaling factor. Returns (top-k indices, top-k weights).
    router_logits = router_logits.sigmoid()
    router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
    group_scores = (
        router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
        .reshape(-1, self.n_routed_experts)
    )
    scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = self._miles_replay_topk(scores_for_choice, self.top_k)
    topk_weights = router_logits.gather(1, topk_indices)
    if self.norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * self.routed_scaling_factor
    return topk_indices, topk_weights


def install_glm4_moe_lite_router_replay(block: nn.Module) -> None:
    # [FUNCTION DOC]
    # Gets from the routing replay manager a top-k function that returns indices only (unsorted
    # top-k when not replaying), stores it on the block and rebinds the block's token-to-expert
    # routing method to the replacement above. Mutates the instance only.
    """Hook ``Glm4MoeLiteMoE``, which selects experts in the block rather than the gate. Only the
    final expert-selection topk is replaced; the group-limited routing above it is left intact, and
    the weights still gather from the sigmoid'd logits so replayed routing stays differentiable."""
    block._miles_replay_topk = routing_replay_manager.get_topk_fn(
        lambda scores, k: torch.topk(scores, k, dim=-1, sorted=False)[1], return_probs=False
    )
    block.route_tokens_to_experts = types.MethodType(_glm4_moe_lite_route_tokens_to_experts, block)
