# [MODULE DOC] HELPER: Megatron model-argument definition for Llama-3.2 3B Instruct (dense) for
# the AMD launchers: 28 layers, hidden 3072, FFN 8192, 24 heads with 8 query groups, rotary base
# 500000 with rope scaling factor 32, maximum positions 131072, vocabulary 128256. Import has no
# side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    # KNOWN ISSUE: the flag text is identical to the non-AMD llama3.2-3B-Instruct definition; the
    # file is a verbatim duplicate. Also, the embeddings are tied (the untie flag is omitted).
    return (
        "--swiglu "
        "--num-layers 28 "
        "--hidden-size 3072 "
        "--ffn-hidden-size 8192 "
        "--num-attention-heads 24 "
        "--group-query-attention "
        "--num-query-groups 8 "
        "--max-position-embeddings 131072 "
        "--use-rotary-position-embeddings "
        "--disable-bias-linear "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-5 "
        "--rotary-base 500000 "
        "--vocab-size 128256 "
        "--kv-channels 128 "
        "--use-rope-scaling "
        "--rotary-scaling-factor 32.0 "
    )
