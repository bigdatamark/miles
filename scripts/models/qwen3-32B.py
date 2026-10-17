# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen3 32B (dense): 64 layers, hidden
# 5120, FFN 25600, 64 heads with 8 query groups and key-value channels 128, QK layer norm, rotary
# base 1,000,000, vocabulary 151936, untied embeddings. Import has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    return (
        "--swiglu "
        "--num-layers 64 "
        "--hidden-size 5120 "
        "--ffn-hidden-size 25600 "
        "--num-attention-heads 64 "
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
        "--untie-embeddings-and-output-weights "
    )
