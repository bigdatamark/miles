# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen3.5 0.8B (dense hybrid
# linear / full attention): 24 layers, hidden 1024, FFN 3584, 8 heads with 2 query groups and
# key-value channels 256, partial rotary 0.25 with base 10,000,000, attention output gate, layer
# norm with offset one, vocabulary 248320, tied embeddings (the untie flag is omitted), built with
# the Qwen3.5 spec plugin. Import has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    return (
        "--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec "
        "--disable-bias-linear "
        "--qk-layernorm "
        "--group-query-attention "
        "--num-attention-heads 8 "
        "--num-query-groups 2 "
        "--kv-channels 256 "
        "--num-layers 24 "
        "--hidden-size 1024 "
        "--ffn-hidden-size 3584 "
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
