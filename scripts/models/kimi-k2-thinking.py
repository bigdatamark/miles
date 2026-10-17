# [MODULE DOC] HELPER: Megatron model-argument definition for Kimi-K2-Thinking (DeepSeek-style MoE
# with multi-latent attention): 61 layers by default with 1 dense layer, hidden 7168, 64 heads,
# YaRN rotary scaling factor 64 from original context 4096, rotary base 50000, 384 routed experts
# with top-8 sigmoid routing scaled by 2.827, shared expert, vocabulary 163840. Import has no side
# effects.
from model_args_utils import moe_layer_freq


def model_args(nlayers: int = 61, first_k_dense_replace: int = 1, beta_fast: int = 1) -> str:
    # [FUNCTION DOC] HELPER: return the flag string for the given total layers (default 61),
    # leading dense layer count (default 1, through the MoE frequency pattern) and YaRN fast-beta
    # (default 1). Norm epsilon is 1e-5.
    return (
        "--disable-bias-linear "
        f"--num-layers {nlayers} "
        "--hidden-size 7168 "
        "--ffn-hidden-size 18432 "
        "--num-attention-heads 64 "
        "--kv-channels 64 "
        "--normalization RMSNorm "
        "--position-embedding-type rope "
        "--rope-type yarn "
        "--norm-epsilon 1e-5 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 163840 "
        "--multi-latent-attention "
        "--q-lora-rank 1536 "
        "--kv-lora-rank 512 "
        "--qk-head-dim 128 "
        "--qk-pos-emb-head-dim 64 "
        "--v-head-dim 128 "
        "--qk-layernorm "
        "--rotary-scaling-factor 64.0 "
        "--rotary-base 50000 "
        "--original-max-position-embeddings 4096 "
        f"--beta-fast {beta_fast} "
        "--beta-slow 1 "
        "--mscale 1.0 "
        "--mscale-all-dim 1.0 "
        "--attention-softmax-in-fp32 "
        "--no-rope-fusion "
        # moe
        "--num-experts 384 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=nlayers, first_k_dense_replace=first_k_dense_replace)} "
        "--moe-ffn-hidden-size 2048 "
        "--moe-router-topk 8 "
        "--moe-shared-expert-intermediate-size 2048 "
        "--moe-router-pre-softmax "
        "--moe-router-score-function sigmoid "
        "--moe-router-enable-expert-bias "
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-token-dispatcher-type alltoall "
        "--moe-aux-loss-coeff 0 "
        "--moe-router-bias-update-rate 0 "
        "--moe-router-group-topk 1 "
        "--moe-router-num-groups 1 "
        "--moe-grouped-gemm "
        "--moe-router-topk-scaling-factor 2.827 "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
    )
