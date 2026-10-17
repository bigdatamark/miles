# [MODULE DOC] HELPER: Megatron model-argument definition used for Qwen3.6 35B-A3B LoRA runs
# (hybrid linear / full attention MoE, 40 layers, all MoE, 256 experts with top-8 softmax routing,
# gated shared expert, one multi-token-prediction layer, Qwen3.5 spec plugin). Import has no side
# effects.
# KNOWN ISSUE: the content is a verbatim copy of the qwen3.6-35B-A3B definition rather than a
# derivation through the sibling loader, and contains no LoRA-specific flag.
from model_args_utils import moe_layer_freq


NLAYERS = 40
FIRST_K_DENSE_REPLACE = 0


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    # The layer count flag is a literal 40 while the MoE frequency pattern uses the NLAYERS constant.
    return (
        "--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec "
        "--disable-bias-linear "
        "--qk-layernorm "
        "--group-query-attention "
        "--num-attention-heads 16 "
        "--num-query-groups 2 "
        "--kv-channels 256 "
        "--num-layers 40 "
        "--hidden-size 2048 "
        "--ffn-hidden-size 512 "
        "--normalization RMSNorm "
        "--apply-layernorm-1p "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        "--rotary-percent 0.25 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 248320 "
        "--rotary-base 10000000 "
        # moe
        "--moe-ffn-hidden-size 512 "
        "--moe-shared-expert-intermediate-size 512 "
        "--moe-router-score-function softmax "
        "--moe-token-dispatcher-type alltoall "
        "--moe-router-topk 8 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=NLAYERS, first_k_dense_replace=FIRST_K_DENSE_REPLACE)} "
        "--num-experts 256 "
        "--moe-grouped-gemm "
        "--moe-token-drop-policy probs "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
        "--moe-aux-loss-coeff 0 "
        # qwen3.6 specific (same architecture as qwen3.5)
        "--attention-output-gate "
        "--moe-shared-expert-gate "
        "--mtp-num-layers 1 "
    )
