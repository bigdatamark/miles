# [MODULE DOC]
# Replacement Qwen3 MoE sparse block for true-on-policy FSDP training: routing is done in the
# Hugging Face style (softmax over experts in float32, top-k, optional renormalization) but expert
# computation uses the project's fused experts kernel in bfloat16 with all expert weights stacked
# on the fly, so numerics match the rollout engine. The apply function installs the block into
# transformers' module so models constructed afterwards use it.
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

from miles.kernels.moe.fused_experts import fused_experts


class StandardDispatcher:
    # [CLASS DOC]
    # Maps selected global expert ids to local expert ids for expert parallelism. Currently always
    # configured with expert-parallel size 1 and rank 0, so it is an identity mapping; the mapping
    # table (global id to local id, with -1 for experts not held locally) is only built in the
    # unreachable multi-rank case.
    def __init__(self, num_experts: int, num_local_experts: int):
        # [FUNCTION DOC]
        # Stores the total and local expert counts, hardcodes expert-parallel size 1 and rank 0, and
        # leaves the mapping table unset. The multi-rank branch (build an int32 table on the GPU
        # filled with -1 and with this rank's slice set to 0..local-1) never runs.
        # KNOWN ISSUE: the expert-parallel size is hardcoded to 1, so the mapping branch is dead code.
        self.moe_ep_size = 1
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.moe_ep_rank = 0
        self.local_expert_mapping = None

        if self.moe_ep_size > 1:
            self.local_expert_mapping = torch.full((self.num_experts,), -1, dtype=torch.int32, device="cuda")
            self.local_expert_mapping[
                self.moe_ep_rank * self.num_local_experts : (self.moe_ep_rank + 1) * self.num_local_experts
            ] = torch.arange(0, self.num_local_experts, dtype=torch.int32, device="cuda")

    def dispatch(self, topk_ids) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns the selected expert ids translated through the local mapping table when one
        # exists, otherwise returns them unchanged (the current behavior).
        if self.local_expert_mapping is not None:
            return self.local_expert_mapping[topk_ids]
        return topk_ids


class Qwen3MoeSparseMoeBlock(nn.Module):
    # [CLASS DOC]
    # Drop-in replacement for the library's Qwen3 MoE sparse block. Holds a bias-free linear router
    # and a list of per-expert gated MLPs (kept as separate modules so checkpoint names match).
    # A class-level shared dispatcher is created by the first instance. The class-level runner
    # attribute is declared but never used.
    dispatcher = None
    runner = None

    def __init__(self, config):
        # [FUNCTION DOC]
        # Reads the expert count, experts-per-token (top-k) and top-k normalization flag from the
        # config, creates the router linear (hidden size to expert count) and one MLP per expert
        # sized to the MoE intermediate size, and lazily creates the shared dispatcher with all
        # experts local.
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        if Qwen3MoeSparseMoeBlock.dispatcher is None:
            Qwen3MoeSparseMoeBlock.dispatcher = StandardDispatcher(
                num_experts=config.num_experts, num_local_experts=config.num_experts
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Input: hidden states of shape (batch, sequence, hidden). Flattens to tokens, computes
        # router logits, a float32 softmax over experts, picks the top-k experts and weights per
        # token, renormalizes the weights to sum to one when configured, casts weights to the input
        # dtype, maps ids through the dispatcher, then stacks every expert's gate and up projection
        # weights (concatenated along the output axis) and down projection weights, and calls the
        # fused experts kernel on the bfloat16 tokens. Returns a pair: the combined expert output
        # (token-flattened, shape tokens by hidden) and the raw router logits (tokens by experts).
        # Stacking the weights happens on every call (extra memory/time).
        # KNOWN ISSUE: the return annotation says a single tensor but a pair is returned, and the
        # output is not reshaped back to (batch, sequence, hidden) nor cast back to the input dtype.
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        selected_experts = Qwen3MoeSparseMoeBlock.dispatcher.dispatch(selected_experts)

        w13_weight = torch.stack(
            [torch.cat([layer.gate_proj.weight, layer.up_proj.weight], dim=0) for layer in self.experts]
        )
        w2_weight = torch.stack([layer.down_proj.weight for layer in self.experts], dim=0)

        final_hidden_states = fused_experts(
            hidden_states.to(torch.bfloat16),
            w13_weight,
            w2_weight,
            routing_weights,
            selected_experts,
        )

        return final_hidden_states, router_logits


def apply_true_on_policy_patch_for_qwen3_moe():
    # [FUNCTION DOC]
    # Replaces the sparse MoE block class in the transformers Qwen3 MoE modeling module with the
    # class defined here (global side effect; affects models built afterwards). The import is local
    # to defer loading the modeling module until the patch is requested. Takes and returns nothing.
    from transformers.models.qwen3_moe import modeling_qwen3_moe

    modeling_qwen3_moe.Qwen3MoeSparseMoeBlock = Qwen3MoeSparseMoeBlock
