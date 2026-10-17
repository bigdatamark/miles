# [MODULE DOC] HELPER: Megatron model-argument definition for GLM-4.5 355B-A32B (MoE): 3 dense plus
# 89 MoE layers (module constants), hidden 5120, 96 heads with 8 query groups, QK layer norm,
# 160 experts with top-8 sigmoid routing scaled by 2.5, shared expert, QKV bias, partial rotary
# (50 percent), untied embeddings, vocabulary 151552. Import has no side effects.
N_DENSE_LAYERS = 3
N_MOE_LAYERS = 89


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    # Total layers is the sum of the two constants; MoE frequency is dense zeros then MoE ones.
    return (
        "--disable-bias-linear "
        "--qk-layernorm "
        "--group-query-attention "
        "--num-attention-heads 96 "
        "--num-query-groups 8 "
        "--kv-channels 128 "
        f"--num-layers {N_DENSE_LAYERS + N_MOE_LAYERS} "
        "--hidden-size 5120 "
        "--ffn-hidden-size 12288 "
        "--add-qkv-bias "
        "--normalization RMSNorm "
        "--position-embedding-type rope "
        "--rotary-percent 0.5 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 151552 "
        "--rotary-base 1000000 "
        # moe
        "--moe-ffn-hidden-size 1536 "
        "--moe-shared-expert-intermediate-size 1536 "
        "--moe-router-pre-softmax "
        "--moe-router-score-function sigmoid "
        "--moe-router-enable-expert-bias "
        "--moe-router-bias-update-rate 0 "
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-token-dispatcher-type alltoall "
        "--moe-router-topk 8 "
        "--moe-router-topk-scaling-factor 2.5 "
        f"--moe-layer-freq [0]*{N_DENSE_LAYERS}+[1]*{N_MOE_LAYERS} "
        "--num-experts 160 "
        "--moe-grouped-gemm "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
        "--moe-aux-loss-coeff 0 "
    )
