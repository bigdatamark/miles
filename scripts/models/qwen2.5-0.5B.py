# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen2.5 0.5B (dense): 24 layers,
# hidden 896, FFN 4864, 14 heads with 2 query groups, QKV bias, rotary base 1,000,000, vocabulary
# 151936, tied embeddings (the untie flag is omitted). Import has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    return (
        "--swiglu "
        "--num-layers 24 "
        "--hidden-size 896 "
        "--ffn-hidden-size 4864 "
        "--num-attention-heads 14 "
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
