# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen3 1.7B (dense): 28 layers,
# hidden 2048, FFN 6144, 16 heads with 8 query groups and key-value channels 128, QK layer norm,
# vocabulary 151936, tied embeddings (the untie flag is omitted). Import has no side effects.
import os


def model_args(rotary_base: str | None = None) -> str:
    # [FUNCTION DOC] HELPER: return the flag string. Rotary base (string): argument, else the
    # environment variable MODEL_ARGS_ROTARY_BASE if set and nonempty, else "1000000". Reads the
    # environment at call time.
    rotary_base = rotary_base if rotary_base is not None else os.environ.get("MODEL_ARGS_ROTARY_BASE") or "1000000"
    return (
        "--swiglu "
        "--num-layers 28 "
        "--hidden-size 2048 "
        "--ffn-hidden-size 6144 "
        "--num-attention-heads 16 "
        "--group-query-attention "
        "--num-query-groups 8 "
        "--use-rotary-position-embeddings "
        "--disable-bias-linear "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-6 "
        f"--rotary-base {rotary_base} "
        "--vocab-size 151936 "
        "--kv-channels 128 "
        "--qk-layernorm "
    )
