# [MODULE DOC] Layer-spec provider for GLM-4 style models: a thin wrapper that asks Megatron for the
# standard Transformer-Engine GPT layer spec, configured from command-line arguments (experts,
# grouped GEMM, QK layernorm, multi-latent attention, post-attention and post-MLP layernorms).
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec


def get_glm_spec(args, config, vp_stage):
    # [FUNCTION DOC] Returns a single layer spec built from the argument namespace's expert count,
    # grouped-GEMM flag, QK layernorm flag, multi-latent-attention flag and the two post-layernorm
    # flags. The config and virtual pipeline stage arguments are unused. Pure; no side effects.
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
        multi_latent_attention=args.multi_latent_attention,
        post_self_attn_layernorm=args.post_self_attn_layernorm,
        post_mlp_layernorm=args.post_mlp_layernorm,
    )
    return transformer_layer_spec
