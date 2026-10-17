# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen3-Next 80B-A3B (hybrid linear /
# full attention MoE): 48 layers (all MoE), hidden 2048, 16 heads with 2 query groups and
# key-value channels 256, partial rotary 0.25 with base 10,000,000, 512 experts of width 512 with
# top-10 softmax routing and a gated shared expert, attention output gate, one
# multi-token-prediction layer, layer norm with offset one, vocabulary 151936, untied embeddings,
# built with the Qwen3-Next spec plugin. Import has no side effects.
from model_args_utils import moe_layer_freq


NLAYERS = 48
FIRST_K_DENSE_REPLACE = 0


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    # KNOWN ISSUE: the layer count flag is a literal 48 while the MoE frequency pattern uses the
    # NLAYERS constant, so the two could drift apart.
    return (
        "--spec miles_plugins.models.qwen3_next get_qwen3_next_spec "
        "--disable-bias-linear "
        "--qk-layernorm "
        "--group-query-attention "
        "--num-attention-heads 16 "
        "--num-query-groups 2 "
        "--kv-channels 256 "
        "--num-layers 48 "
        "--hidden-size 2048 "
        "--ffn-hidden-size 5120 "
        "--normalization RMSNorm "
        "--apply-layernorm-1p "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        "--rotary-percent 0.25 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 151936 "
        "--rotary-base 10000000 "
        # moe
        "--moe-ffn-hidden-size 512 "
        "--moe-shared-expert-intermediate-size 512 "
        "--moe-router-score-function softmax "
        "--moe-token-dispatcher-type alltoall "
        "--moe-router-topk 10 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=NLAYERS, first_k_dense_replace=FIRST_K_DENSE_REPLACE)} "
        "--num-experts 512 "
        "--moe-grouped-gemm "
        "--moe-token-drop-policy probs "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
        "--moe-aux-loss-coeff 0 "
        # qwen3 specific
        "--attention-output-gate "
        "--moe-shared-expert-gate "
        "--mtp-num-layers 1 "
    )
