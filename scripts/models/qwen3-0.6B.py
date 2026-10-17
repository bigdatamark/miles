# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen3 0.6B (dense): 28 layers,
# hidden 1024, FFN 3072, 16 heads with 8 query groups and key-value channels 128, QK layer norm,
# rotary base 1,000,000, vocabulary 151936, tied embeddings (the untie flag is omitted). Import
# has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    return (
        "--swiglu "
        "--num-layers 28 "
        "--hidden-size 1024 "
        "--ffn-hidden-size 3072 "
        "--num-attention-heads 16 "
        "--group-query-attention "
        "--num-query-groups 8 "
        "--use-rotary-position-embeddings "
        "--disable-bias-linear "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-6 "
        "--rotary-base 1000000 "
        "--vocab-size 151936 "
        "--kv-channels 128 "
        "--qk-layernorm "
    )
