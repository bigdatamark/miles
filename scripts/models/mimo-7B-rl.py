# [MODULE DOC] HELPER: Megatron model-argument definition for MiMo-7B-RL (dense): 36 layers,
# hidden 4096, FFN 11008, 32 heads with 8 query groups, QKV bias, rotary base 640000, maximum
# positions 32768, vocabulary 151680, untied embeddings, one multi-token-prediction layer. Import
# has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    return (
        "--swiglu "
        "--num-layers 36 "
        "--hidden-size 4096 "
        "--ffn-hidden-size 11008 "
        "--num-attention-heads 32 "
        "--group-query-attention "
        "--num-query-groups 8 "
        "--use-rotary-position-embeddings "
        "--disable-bias-linear "
        "--add-qkv-bias "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-05 "
        "--rotary-base 640000 "
        "--vocab-size 151680 "
        "--untie-embeddings-and-output-weights "
        "--max-position-embeddings 32768 "
        "--mtp-num-layers 1 "
    )
