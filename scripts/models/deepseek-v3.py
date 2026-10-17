# [MODULE DOC] HELPER: Megatron model-argument definition for DeepSeek-V3 (671B MoE, 61 layers by
# default): multi-latent attention with query LoRA rank 1536 and key-value LoRA rank 512,
# hidden size 7168, 128 heads, 256 routed experts with top-8 sigmoid routing in 8 groups, shared
# expert, and YaRN-style rotary settings. The module constant gives the number of leading dense
# (non-MoE) layers (3). Truncated variants load this file as a sibling. Import has no side effects.
import os

from model_args_utils import moe_layer_freq


FIRST_K_DENSE_REPLACE = 3


def model_args(nlayers: int | None = None) -> str:
    # [FUNCTION DOC] HELPER: return the flag string. Layer count: the argument if given, else the
    # environment variable MODEL_ARGS_NUM_LAYERS if set and nonempty, else 61. Embeds the layer
    # count and a MoE layer-frequency pattern with the first 3 layers dense. Reads the environment
    # at call time.
    nlayers = nlayers if nlayers is not None else int(os.environ.get("MODEL_ARGS_NUM_LAYERS") or 61)
    return (
        "--disable-bias-linear "
        f"--num-layers {nlayers} "
        "--hidden-size 7168 "
        "--ffn-hidden-size 18432 "
        "--num-attention-heads 128 "
        "--kv-channels 128 "
        "--normalization RMSNorm "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 129280 "
        "--multi-latent-attention "
        "--q-lora-rank 1536 "
        "--kv-lora-rank 512 "
        "--qk-head-dim 128 "
        "--qk-pos-emb-head-dim 64 "
        "--v-head-dim 128 "
        "--qk-layernorm "
        "--rotary-scaling-factor 40 "
        "--rotary-base 10000 "
        "--mscale 1.0 "
        "--mscale-all-dim 1.0 "
        "--attention-softmax-in-fp32 "
        "--no-rope-fusion "
        # moe
        "--num-experts 256 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=nlayers, first_k_dense_replace=FIRST_K_DENSE_REPLACE)} "
        "--moe-ffn-hidden-size 2048 "
        "--moe-router-topk 8 "
        "--moe-shared-expert-intermediate-size 2048 "
        "--moe-router-pre-softmax "
        "--moe-router-score-function sigmoid "
        "--moe-router-enable-expert-bias "
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-token-dispatcher-type alltoall "
        "--moe-aux-loss-coeff 0 "
        "--moe-router-bias-update-rate 0 "
        "--moe-router-group-topk 4 "
        "--moe-router-num-groups 8 "
        "--moe-grouped-gemm "
        "--moe-router-topk-scaling-factor 2.5 "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
    )
