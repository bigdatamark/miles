# [MODULE DOC] HELPER: Megatron model-argument definition for Nemotron-3 Nano 30B-A3B (MoE): 52
# layers, hidden 2688, 32 heads with 2 query groups, no position embedding, 128 experts of width
# 1856 with top-6 sigmoid routing scaled by 2.5 and a shared expert of 3712, vocabulary 131072,
# untied embeddings. Import has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    # Routing is configured as a single expert group (group-limited routing is a no-op), with
    # sequence auxiliary loss type but coefficient zero and no bias update, to match the known-good
    # GLM-4.7 Flash settings.
    return (
        "--disable-bias-linear "
        "--group-query-attention "
        "--num-attention-heads 32 "
        "--num-query-groups 2 "
        "--kv-channels 128 "
        "--num-layers 52 "
        "--hidden-size 2688 "
        "--ffn-hidden-size 1856 "
        "--normalization RMSNorm "
        "--position-embedding-type none "
        "--vocab-size 131072 "
        "--make-vocab-size-divisible-by 128 "
        "--untie-embeddings-and-output-weights "
        # MoE specifics
        "--num-experts 128 "
        "--moe-router-topk 6 "
        "--moe-ffn-hidden-size 1856 "
        "--moe-shared-expert-intermediate-size 3712 "
        "--moe-router-score-function sigmoid "
        "--moe-router-enable-expert-bias "
        "--moe-grouped-gemm "
        "--moe-router-dtype fp32 "
        # Routing: config has n_group=1 (MoE groups), topk_group=1,
        # routed_scaling_factor=2.5. `n_groups=8` is Mamba groups — unrelated to MoE.
        # With n_group=1, group-limited routing is a no-op (single group of 128).
        "--moe-router-num-groups 1 "
        "--moe-router-group-topk 1 "
        "--moe-router-topk-scaling-factor 2.5 "
        "--moe-router-pre-softmax "
        # Match glm4.7-flash (known-working MoE RL) settings more closely.
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-router-bias-update-rate 0 "
        "--moe-aux-loss-coeff 0 "
    )
