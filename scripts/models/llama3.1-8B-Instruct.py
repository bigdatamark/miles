# [MODULE DOC] HELPER: Megatron model-argument definition for Llama-3.1 8B Instruct (dense): 32
# layers, hidden 4096, FFN 14336, 32 heads with 8 query groups, rotary base 500000 with rope
# scaling factor 8, maximum positions 131072, vocabulary 128256, untied embeddings. Import has no
# side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    return (
        "--swiglu "
        "--num-layers 32 "
        "--hidden-size 4096 "
        "--ffn-hidden-size 14336 "
        "--num-attention-heads 32 "
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
        "--rotary-scaling-factor 8.0 "
        "--untie-embeddings-and-output-weights "
    )
