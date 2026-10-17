# [MODULE DOC] HELPER: Megatron model-argument definition for Kimi-K3 (MoE with multi-latent
# attention, 93 layers by default, first layer dense): hidden 7168, 96 heads (96 query groups),
# FFN 33792, no rotary position embedding in Megatron, 896 routed experts by default with top-16
# sigmoid routing, expert width 3072, MoE latent size 3584, shared expert 6144, no load-balancing,
# vocabulary 163840, built with the Kimi-K3 spec plugin. Import has no side effects.
import os

from model_args_utils import moe_layer_freq


def model_args(nlayers: int | None = None, num_experts: int = 896) -> str:
    # [FUNCTION DOC] HELPER: return the flag string. Layer count: argument, else environment
    # variable MODEL_ARGS_NUM_LAYERS if set and nonempty, else 93. Expert count: argument (default
    # 896). One leading dense layer through the MoE frequency pattern. Reads the environment at
    # call time.
    nlayers = nlayers if nlayers is not None else int(os.environ.get("MODEL_ARGS_NUM_LAYERS") or 93)
    return (
        "--spec miles_plugins.models.kimi_k3 get_kimi_k3_spec "
        "--disable-bias-linear "
        f"--num-layers {nlayers} "
        "--hidden-size 7168 "
        "--ffn-hidden-size 33792 "
        "--num-attention-heads 96 "
        "--num-query-groups 96 "
        "--kv-channels 256 "
        "--normalization RMSNorm "
        "--position-embedding-type none "
        "--rotary-base 10000 "
        "--norm-epsilon 1e-5 "
        "--hidden-dropout 0 "
        "--attention-dropout 0 "
        "--disable-bf16-reduced-precision-matmul "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 163840 "
        "--make-vocab-size-divisible-by 128 "
        "--multi-latent-attention "
        "--q-lora-rank 1536 "
        "--kv-lora-rank 512 "
        "--qk-head-dim 128 "
        "--qk-pos-emb-head-dim 64 "
        "--v-head-dim 128 "
        "--qk-layernorm "
        "--attention-softmax-in-fp32 "
        f"--num-experts {num_experts} "
        f"--moe-layer-freq {moe_layer_freq(nlayers=nlayers, first_k_dense_replace=1)} "
        "--moe-ffn-hidden-size 3072 "
        "--moe-latent-size 3584 "
        "--moe-shared-expert-intermediate-size 6144 "
        "--moe-router-topk 16 "
        "--moe-router-score-function sigmoid "
        "--moe-router-pre-softmax "
        "--moe-router-enable-expert-bias "
        "--moe-router-load-balancing-type none "
        "--moe-token-dispatcher-type alltoall "
        "--moe-router-bias-update-rate 0 "
        "--moe-aux-loss-coeff 0 "
        "--moe-router-topk-scaling-factor 1.0 "
        "--moe-router-dtype fp32 "
        "--moe-grouped-gemm "
        "--moe-permute-fusion "
    )
