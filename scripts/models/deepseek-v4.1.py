# [MODULE DOC] HELPER: Megatron model-argument definition for DeepSeek-V4.1 (40 layers by default;
# hidden 5120, 64 heads, query LoRA rank 1280, 384 routed experts with top-6 square-root-softplus
# routing and scaling 1.5, 4 residual streams, output groups 8, no hash-routed layers, DSA indexer
# with 32 heads and top-k 512, and a tiny layer-norm epsilon). Module constants: the per-layer
# compression-ratio list (two zeros, eighteen 2s, then twenty 1s) and the extra SwiGLU clamp
# flags (clamp value 10, no fused bias SwiGLU). Import has no side effects.
import os

from model_args_utils import moe_layer_freq


COMPRESS_RATIOS = "[0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
SWIGLU_LIMIT_ARGS = "--activation-func-clamp-value 10 --no-bias-swiglu-fusion"


def model_args(nlayers: int | None = None, compress_ratios: str = COMPRESS_RATIOS) -> str:
    # [FUNCTION DOC] HELPER: return the flag string. Layer count: argument, else environment
    # variable MODEL_ARGS_NUM_LAYERS if set and nonempty, else 40. The compression ratio list is
    # embedded as given. All layers are MoE (no leading dense layers). Uses the V4.1 spec plugin and
    # appends the SwiGLU clamp flags. Reads the environment at call time.
    nlayers = nlayers if nlayers is not None else int(os.environ.get("MODEL_ARGS_NUM_LAYERS") or 40)
    return (
        "--disable-bias-linear "
        f"--num-layers {nlayers} "
        "--hidden-size 5120 "
        "--ffn-hidden-size 2304 "
        "--num-attention-heads 64 "
        "--normalization RMSNorm "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-20 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 129280 "
        "--make-vocab-size-divisible-by 64 "
        "--hidden-dropout 0.0 "
        "--attention-dropout 0.0 "
        "--multi-latent-attention "
        "--q-lora-rank 1280 "
        "--kv-lora-rank 512 "
        "--qk-head-dim 512 "
        "--qk-pos-emb-head-dim 64 "
        "--v-head-dim 512 "
        "--qk-layernorm "
        "--rotary-scaling-factor 16 "
        "--rotary-base 10000 "
        "--original-max-position-embeddings 65536 "
        "--beta-fast 32 "
        "--beta-slow 1 "
        "--attention-softmax-in-fp32 "
        "--no-rope-fusion "
        "--num-experts 384 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=nlayers, first_k_dense_replace=0)} "
        "--moe-ffn-hidden-size 2304 "
        "--moe-router-topk 6 "
        "--moe-shared-expert-intermediate-size 2304 "
        "--moe-router-pre-softmax "
        "--moe-router-score-function sqrtsoftplus "
        "--moe-router-enable-expert-bias "
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-token-dispatcher-type alltoall "
        "--moe-aux-loss-coeff 0 "
        "--moe-grouped-gemm "
        "--moe-router-topk-scaling-factor 1.5 "
        "--num-residual-streams 4 "
        "--mhc-sinkhorn-iterations 20 "
        f"--csa-compress-ratios {compress_ratios} "
        "--csa-compress-rotary-base 160000 "
        "--o-groups 8 "
        "--o-lora-rank 1024 "
        "--moe-n-hash-layers 0 "
        "--csa-window-size 128 "
        "--dsa-indexer-n-heads 32 "
        "--dsa-indexer-head-dim 128 "
        "--dsa-indexer-topk 512 "
        "--spec miles_plugins.models.deepseek_v4_1.deepseek_v4_1 get_dsv41_spec "
        f"{SWIGLU_LIMIT_ARGS} "
    )
