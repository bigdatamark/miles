# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen3.5 4B (dense hybrid linear /
# full attention): 32 layers, hidden 2560, FFN 9216, 16 heads with 4 query groups and key-value
# channels 256, partial rotary 0.25 with base 10,000,000, attention output gate, layer norm with
# offset one, vocabulary 248320, tied embeddings (the untie flag is omitted), Qwen3.5 spec plugin.
# Import has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    return (
        "--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec "
        "--disable-bias-linear "
        "--qk-layernorm "
        "--group-query-attention "
        "--num-attention-heads 16 "
        "--num-query-groups 4 "
        "--kv-channels 256 "
        "--num-layers 32 "
        "--hidden-size 2560 "
        "--ffn-hidden-size 9216 "
        "--normalization RMSNorm "
        "--apply-layernorm-1p "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        "--rotary-percent 0.25 "
        "--swiglu "
        "--vocab-size 248320 "
        "--rotary-base 10000000 "
        # qwen3.5 specific
        "--attention-output-gate "
    )
