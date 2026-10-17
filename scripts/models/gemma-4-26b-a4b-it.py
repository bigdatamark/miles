# [MODULE DOC] HELPER: Megatron model-argument definition for Gemma-4 26B-A4B instruct (MoE): 30
# layers, hidden 2816, 16 attention heads with 8 query groups and key-value channels 256, 128
# experts with top-8 softmax routing and expert FFN width 704, tied input/output embeddings (the
# untie flag is deliberately omitted), vocabulary 262144, rotary base 1,000,000. Import has no
# side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    # Router: fp32, one expert group, no load-balancing aux loss coefficient and no bias update.
    return (
        "--disable-bias-linear "
        "--group-query-attention "
        "--num-attention-heads 16 "
        "--num-query-groups 8 "
        "--kv-channels 256 "
        "--num-layers 30 "
        "--hidden-size 2816 "
        "--ffn-hidden-size 2112 "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-06 "
        "--position-embedding-type rope "
        "--rotary-base 1000000 "
        "--vocab-size 262144 "
        "--make-vocab-size-divisible-by 128 "
        "--max-position-embeddings 262144 "
        # tied embeddings: do not pass --untie-embeddings-and-output-weights
        "--num-experts 128 "
        "--moe-router-topk 8 "
        "--moe-ffn-hidden-size 704 "
        "--moe-router-score-function softmax "
        "--moe-grouped-gemm "
        "--moe-router-dtype fp32 "
        "--moe-router-num-groups 1 "
        "--moe-router-group-topk 1 "
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-router-bias-update-rate 0 "
        "--moe-aux-loss-coeff 0 "
    )
