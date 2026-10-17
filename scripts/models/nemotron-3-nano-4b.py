# [MODULE DOC] HELPER: Megatron model-argument definition for Nemotron-3 Nano 4B (dense): 42
# layers, hidden 3136, FFN 12544, 40 heads with 8 query groups, no position embedding, vocabulary
# 131072, untied embeddings. Import has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    return (
        "--disable-bias-linear "
        "--group-query-attention "
        "--num-attention-heads 40 "
        "--num-query-groups 8 "
        "--kv-channels 128 "
        "--num-layers 42 "
        "--hidden-size 3136 "
        "--ffn-hidden-size 12544 "
        "--normalization RMSNorm "
        "--position-embedding-type none "
        "--vocab-size 131072 "
        "--make-vocab-size-divisible-by 128 "
        "--untie-embeddings-and-output-weights "
    )
