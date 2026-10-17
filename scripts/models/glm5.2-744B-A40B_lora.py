# [MODULE DOC] HELPER: Megatron model-argument definition used for GLM-5.2 744B-A40B LoRA runs:
# 3 dense layers plus a configurable number of MoE layers (default 75), hidden 6144, 64 heads,
# multi-latent attention, 256 routed experts, GLM-5 spec plugin, rotary base 8,000,000. Import has
# no side effects.
# KNOWN ISSUE: the content is a verbatim copy of the glm5.2-744B-A40B definition rather than a
# derivation of it through the sibling loader, and contains no LoRA-specific flag.
MOE_ROUTED_EXPERTS = 256
MOE_ACTIVE_ROUTED_EXPERTS = 8
MOE_SHARED_EXPERTS = 1
NHIDDEN = 6144
MOE_FFN_HIDDEN = 2048
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE = MOE_FFN_HIDDEN * MOE_SHARED_EXPERTS
FFN_HIDDEN = 12288
N_DENSE_LAYERS = 3
NHEADS = 64


def model_args(n_moe_layers: int = 75) -> str:
    # [FUNCTION DOC] HELPER: return the flag string for the given number of MoE layers (default
    # 75); total layers is 3 plus that number, MoE frequency is 3 zeros then that many ones.
    return (
        "--spec miles_plugins.models.glm5.glm5 get_glm5_spec "
        f"--moe-layer-freq [0]*{N_DENSE_LAYERS}+[1]*{n_moe_layers} "
        f"--num-experts {MOE_ROUTED_EXPERTS} "
        f"--moe-shared-expert-intermediate-size {MOE_SHARED_EXPERT_INTERMEDIATE_SIZE} "
        f"--moe-router-topk {MOE_ACTIVE_ROUTED_EXPERTS} "
        "--moe-grouped-gemm "
        "--moe-permute-fusion "
        f"--moe-ffn-hidden-size {MOE_FFN_HIDDEN} "
        "--moe-router-score-function sigmoid "
        "--moe-router-pre-softmax "
        "--moe-router-enable-expert-bias "
        "--moe-router-bias-update-rate 0 "
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-router-topk-scaling-factor 2.5 "
        "--moe-aux-loss-coeff 0 "
        "--moe-router-dtype fp32 "
        "--make-vocab-size-divisible-by 16 "
        f"--num-layers {N_DENSE_LAYERS + n_moe_layers} "
        f"--hidden-size {NHIDDEN} "
        f"--ffn-hidden-size {FFN_HIDDEN} "
        f"--num-attention-heads {NHEADS} "
        "--disable-bias-linear "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--position-embedding-type rope "
        "--no-position-embedding "
        "--normalization RMSNorm "
        "--qk-layernorm "
        "--multi-latent-attention "
        "--q-lora-rank 2048 "
        "--kv-lora-rank 512 "
        "--qk-head-dim 192 "
        "--v-head-dim 256 "
        "--kv-channels 192 "
        "--qk-pos-emb-head-dim 64 "
        "--vocab-size 154880 "
        "--rotary-base 8000000 "
        "--enable-experimental "
    )
