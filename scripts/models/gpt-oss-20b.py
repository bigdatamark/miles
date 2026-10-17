# [MODULE DOC] HELPER: Megatron model-argument definition for gpt-oss-20b (MoE, 24 layers): hidden
# 2880, 64 heads with 8 query groups and key-value channels 64, rotary base 150000 with maximum
# positions 131072 (matching the checkpoint's YaRN setup), learnable softmax offsets with
# alternating sliding-window (128, 0) and full attention, 32 experts with top-4 routing in an
# fp32 router, vocabulary 201088, untied embeddings. Import has no side effects.
def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    # The expert FFN width is not set separately, so it equals the dense FFN width (2880). Bias
    # in linear layers is left enabled because the checkpoint has attention bias. Masked-softmax
    # and rope fusions are disabled for compatibility with the attention pattern.
    return (
        # Base architecture
        "--num-layers 24 "
        "--hidden-size 2880 "
        "--ffn-hidden-size 2880 "
        "--num-attention-heads 64 "
        "--group-query-attention "
        "--num-query-groups 8 "
        "--kv-channels 64 "
        # Positional embeddings
        "--use-rotary-position-embeddings "
        "--rotary-percent 1.0 "
        "--rotary-base 150000 "
        # Train with a 4k context, but keep max positions aligned with the HF checkpoint (YaRN scaling).
        "--max-position-embeddings 131072 "
        # Normalization
        "--normalization RMSNorm "
        "--norm-epsilon 1e-5 "
        # Activation & embeddings
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 201088 "
        # Note: attention_bias is true in HF config, so we may need bias
        # --disable-bias-linear  # commented out since attention_bias=true
        # Sliding window attention + learnable softmax offset (alternating SWA/full attention).
        "--softmax-type learnable "
        "--window-size 128,0 "
        "--window-attn-skip-freq 2 "
        # Fusions can be incompatible with this attention pattern on some stacks.
        "--no-masked-softmax-fusion "
        "--no-rope-fusion "
        # MoE parameters
        "--num-experts 32 "
        "--moe-router-topk 4 "
        "--moe-aux-loss-coeff 0.0 "
        "--moe-token-dispatcher-type alltoall "
        "--moe-router-dtype fp32 "
        "--moe-grouped-gemm "
    )
