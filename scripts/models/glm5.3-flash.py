# [MODULE DOC] HELPER: Megatron model-argument definition for GLM-5.3 Flash (MoE with multi-latent
# attention, 45 layers by default, first 3 dense): hidden 4096, 64 heads, no rotary sub-dimension in
# the MLA head, 288 routed experts with top-8 sigmoid routing scaled by 2.5, shared expert,
# vocabulary 154880, rotary base 10000, activation clamp 10 matching the checkpoint's SwiGLU limit,
# built with the GLM-5-next spec plugin. Import has no side effects.
from model_args_utils import moe_layer_freq


def model_args(nlayers: int = 45, first_k_dense_replace: int = 3) -> str:
    # [FUNCTION DOC] HELPER: return the flag string for the given total layer count (default 45)
    # and number of leading dense layers (default 3), embedded through the MoE layer frequency
    # pattern.
    return (
        "--spec miles_plugins.models.glm5_next.glm5_next get_glm5_next_spec "
        f"--num-layers {nlayers} "
        "--hidden-size 4096 "
        "--ffn-hidden-size 12288 "
        "--num-attention-heads 64 "
        "--multi-latent-attention "
        "--q-lora-rank 1536 "
        "--kv-lora-rank 512 "
        "--qk-head-dim 256 "
        "--qk-pos-emb-head-dim 0 "
        "--v-head-dim 256 "
        "--kv-channels 256 "
        "--qk-layernorm "
        f"--moe-layer-freq {moe_layer_freq(nlayers=nlayers, first_k_dense_replace=first_k_dense_replace)} "
        "--num-experts 288 "
        "--moe-router-topk 8 "
        "--moe-router-score-function sigmoid "
        "--moe-router-pre-softmax "
        "--moe-router-enable-expert-bias "
        "--moe-router-bias-update-rate 0 "
        "--moe-router-topk-scaling-factor 2.5 "
        "--moe-ffn-hidden-size 2048 "
        "--moe-shared-expert-intermediate-size 2048 "
        "--moe-router-dtype fp32 "
        "--moe-grouped-gemm "
        "--moe-permute-fusion "
        "--vocab-size 154880 "
        "--rotary-base 10000 "
        "--rope-type rope "
        "--no-rope-fusion "
        "--position-embedding-type rope "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-5 "
        "--swiglu "
        # the checkpoint sets swiglu_limit=10, which sglang applies to dense, shared and routed MLPs
        "--activation-func-clamp-value 10 "
        "--disable-bias-linear "
        "--untie-embeddings-and-output-weights "
        "--make-vocab-size-divisible-by 16 "
        "--enable-experimental "
    )
