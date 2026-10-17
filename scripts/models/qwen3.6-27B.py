# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen3.6 27B (dense hybrid linear /
# full attention, same architecture as Qwen3.5 27B): 64 layers, hidden 5120, FFN 17408, 24 heads
# with 4 query groups and key-value channels 256, partial rotary 0.25 with base 10,000,000,
# attention output gate, vocabulary 248320, untied embeddings, Qwen3.5 spec plugin. Import has no
# side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    # KNOWN ISSUE: the flag text is identical to the qwen3.5-27B definition (a duplicate rather
    # than a sibling derivation like the Qwen3.8 27B file).
    return (
        "--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec "
        "--disable-bias-linear "
        "--qk-layernorm "
        "--group-query-attention "
        "--num-attention-heads 24 "
        "--num-query-groups 4 "
        "--kv-channels 256 "
        "--num-layers 64 "
        "--hidden-size 5120 "
        "--ffn-hidden-size 17408 "
        "--normalization RMSNorm "
        "--apply-layernorm-1p "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        "--rotary-percent 0.25 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 248320 "
        "--rotary-base 10000000 "
        # qwen3.5-family specific
        "--attention-output-gate "
    )
