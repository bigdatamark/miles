# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen2.5 1.5B (dense): 28 layers,
# hidden 1536, FFN 8960, 12 heads with 2 query groups, QKV bias, rotary base 10000, vocabulary
# 151936, tied embeddings (the untie flag is omitted). Import has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    # Note: the rotary base here is 10000, unlike the other Qwen2.5 definitions (1,000,000).
    return (
        "--swiglu "
        "--num-layers 28 "
        "--hidden-size 1536 "
        "--ffn-hidden-size 8960 "
        "--num-attention-heads 12 "
        "--use-rotary-position-embeddings "
        "--disable-bias-linear "
        "--add-qkv-bias "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-6 "
        "--rotary-base 10000 "
        "--group-query-attention "
        "--num-query-groups 2 "
        "--vocab-size 151936 "
    )
