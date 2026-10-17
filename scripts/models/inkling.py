# [MODULE DOC] HELPER: Megatron model-argument definition for the full Inkling model (66 layers by
# default): hidden 6144, 64 heads with 8 query groups, key-value channels 128, no position
# embedding, maximum positions 1,048,576, 256 experts with top-6 sigmoid routing and expert width
# 3072, shared expert, QK layer norm, vocabulary 201024, and the Inkling custom model provider.
# Import has no side effects.
import os


def model_args(nlayers: int | None = None) -> str:
    # [FUNCTION DOC] HELPER: return the flag string. Layer count: argument, else environment
    # variable MODEL_ARGS_NUM_LAYERS if set and nonempty, else 66 (set to 4 for the 4-layer slice).
    # Reads the environment at call time.
    # Inkling config (66L full model; set MODEL_ARGS_NUM_LAYERS=4 for the 4-layer slice)
    nlayers = nlayers if nlayers is not None else int(os.environ.get("MODEL_ARGS_NUM_LAYERS") or 66)
    return (
        "--disable-bias-linear "
        f"--num-layers {nlayers} "
        "--hidden-size 6144 "
        "--ffn-hidden-size 3072 "
        "--num-attention-heads 64 "
        "--group-query-attention "
        "--num-query-groups 8 "
        "--kv-channels 128 "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-6 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 201024 "
        "--hidden-dropout 0.0 "
        "--attention-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--position-embedding-type none "
        "--no-rope-fusion "
        "--no-masked-softmax-fusion "
        "--max-position-embeddings 1048576 "
        # MoE
        "--num-experts 256 "
        "--moe-ffn-hidden-size 3072 "
        "--moe-router-topk 6 "
        "--moe-shared-expert-intermediate-size 3072 "
        "--moe-router-pre-softmax "
        "--moe-router-score-function sigmoid "
        "--moe-router-enable-expert-bias "
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-token-dispatcher-type alltoall "
        "--moe-aux-loss-coeff 0 "
        "--moe-grouped-gemm "
        "--qk-layernorm "
        # Inkling model provider
        "--custom-model-provider-path miles_plugins.models.inkling.model.inkling_model_provider "
    )
