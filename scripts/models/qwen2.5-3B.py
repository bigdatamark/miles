# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen2.5 3B (dense): 36 layers,
# hidden 2048, FFN 11008, 16 heads with 2 query groups, QKV bias, rotary base 1,000,000,
# vocabulary 151936, tied embeddings (the untie flag is omitted). Import has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    return (
        "--swiglu "
        "--num-layers 36 "
        "--hidden-size 2048 "
        "--ffn-hidden-size 11008 "
        "--num-attention-heads 16 "
        "--use-rotary-position-embeddings "
        "--disable-bias-linear "
        "--add-qkv-bias "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-6 "
        "--rotary-base 1000000 "
        "--group-query-attention "
        "--num-query-groups 2 "
        "--vocab-size 151936 "
    )
