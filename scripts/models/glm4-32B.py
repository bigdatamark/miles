# [MODULE DOC] HELPER: Megatron model-argument definition for GLM-4 32B (dense): 64 layers, hidden
# 6144, FFN 23040, 48 heads with 8 query groups, interleaved partial rotary embedding (50 percent),
# post-attention and post-MLP layer norms, untied embeddings, built with the GLM-4 spec plugin.
# Import has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    # Sequence and maximum position length are both 32768.
    return (
        "--spec miles_plugins.models.glm4 get_glm_spec "
        "--swiglu "
        "--num-layers 64 "
        "--hidden-size 6144 "
        "--ffn-hidden-size 23040 "
        "--num-attention-heads 48 "
        "--max-position-embeddings 32768 "
        "--seq-length 32768 "
        "--use-rotary-position-embeddings "
        "--disable-bias-linear "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-5 "
        "--rotary-base 10000 "
        "--group-query-attention "
        "--num-query-groups 8 "
        "--vocab-size 151552 "
        "--post-self-attn-layernorm "
        "--post-mlp-layernorm "
        "--rotary-interleaved "
        "--rotary-percent 0.5 "
        "--no-rope-fusion "
        "--untie-embeddings-and-output-weights "
    )
