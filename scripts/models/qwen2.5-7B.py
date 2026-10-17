# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen2.5 7B (dense): 28 layers,
# hidden 3584, FFN 18944, 28 heads with 4 query groups, QKV bias, rotary base 1,000,000,
# vocabulary 152064, untied embeddings. Import has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    return (
        "--swiglu "
        "--num-layers 28 "
        "--hidden-size 3584 "
        "--ffn-hidden-size 18944 "
        "--num-attention-heads 28 "
        "--group-query-attention "
        "--num-query-groups 4 "
        "--use-rotary-position-embeddings "
        "--disable-bias-linear "
        "--add-qkv-bias "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-06 "
        "--rotary-base 1000000 "
        "--vocab-size 152064 "
        "--untie-embeddings-and-output-weights "
    )
