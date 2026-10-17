# [MODULE DOC] HELPER: Megatron model-argument definition for JoyAI LLM Flash (MoE with
# multi-latent attention, 40 layers by default, first layer dense): hidden 2048, 32 heads, query
# LoRA rank 1536, key-value LoRA rank 512, 256 routed experts of width 768 with top-8 sigmoid
# routing scaled by 2.5, shared expert, vocabulary 129280. The module constant gives the leading
# dense layer count (1). Import has no side effects.
import os

from model_args_utils import moe_layer_freq


FIRST_K_DENSE_REPLACE = 1


def model_args(nlayers: int | None = None, rotary_base: str | None = None) -> str:
    # [FUNCTION DOC] HELPER: return the flag string. Layer count: argument, else environment
    # variable MODEL_ARGS_NUM_LAYERS if set and nonempty, else 40. Rotary base (string): argument,
    # else environment variable MODEL_ARGS_ROTARY_BASE if set and nonempty, else "32000000".
    # Reads the environment at call time.
    nlayers = nlayers if nlayers is not None else int(os.environ.get("MODEL_ARGS_NUM_LAYERS") or 40)
    rotary_base = rotary_base if rotary_base is not None else os.environ.get("MODEL_ARGS_ROTARY_BASE") or "32000000"
    return (
        "--disable-bias-linear "
        f"--num-layers {nlayers} "
        "--hidden-size 2048 "
        "--ffn-hidden-size 7168 "
        "--num-attention-heads 32 "
        "--kv-channels 128 "
        "--normalization RMSNorm "
        "--position-embedding-type rope "
        "--rope-type rope "
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
        f"--rotary-base {rotary_base} "
        "--mscale 1.0 "
        "--mscale-all-dim 1.0 "
        "--attention-softmax-in-fp32 "
        "--no-rope-fusion "
        "--num-experts 256 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=nlayers, first_k_dense_replace=FIRST_K_DENSE_REPLACE)} "
        "--moe-ffn-hidden-size 768 "
        "--moe-router-topk 8 "
        "--moe-shared-expert-intermediate-size 768 "
        "--moe-router-pre-softmax "
        "--moe-router-score-function sigmoid "
        "--moe-router-enable-expert-bias "
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-token-dispatcher-type alltoall "
        "--moe-aux-loss-coeff 0 "
        "--moe-router-bias-update-rate 0 "
        "--moe-router-group-topk 1 "
        "--moe-router-num-groups 1 "
        "--moe-grouped-gemm "
        "--moe-router-topk-scaling-factor 2.5 "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
    )
