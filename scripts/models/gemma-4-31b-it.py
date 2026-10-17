# [MODULE DOC] HELPER: Megatron model-argument definition for Gemma-4 31B instruct (dense): 60
# layers, hidden 5376, FFN 21504, 32 attention heads with 16 query groups and key-value channels
# 256, tied embeddings (untie flag deliberately omitted), vocabulary 262144, rotary base 1,000,000.
# Import has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    return (
        "--disable-bias-linear "
        "--group-query-attention "
        "--num-attention-heads 32 "
        "--num-query-groups 16 "
        "--kv-channels 256 "
        "--num-layers 60 "
        "--hidden-size 5376 "
        "--ffn-hidden-size 21504 "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-06 "
        "--position-embedding-type rope "
        "--rotary-base 1000000 "
        "--vocab-size 262144 "
        "--make-vocab-size-divisible-by 128 "
        "--max-position-embeddings 262144 "
        # tied embeddings: do not pass --untie-embeddings-and-output-weights
    )
