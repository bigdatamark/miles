# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen2.5 32B (dense): 64 layers,
# hidden 5120, FFN 27648, 40 heads with 8 query groups, QKV bias, norm epsilon 1e-5, rotary base
# 1,000,000, vocabulary 152064, untied embeddings. Import has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    return (
        "--swiglu "
        "--num-layers 64 "
        "--hidden-size 5120 "
        "--ffn-hidden-size 27648 "
        "--num-attention-heads 40 "
        "--group-query-attention "
        "--num-query-groups 8 "
        "--use-rotary-position-embeddings "
        "--disable-bias-linear "
        "--add-qkv-bias "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-5 "
        "--rotary-base 1000000 "
        "--vocab-size 152064 "
        "--untie-embeddings-and-output-weights "
    )
