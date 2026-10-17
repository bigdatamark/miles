# [MODULE DOC] HELPER: Megatron model-argument definition for a 4-layer slice of Nemotron-3 Ultra
# 550B-A55B (source layers 0, 1, 7, 8 renumbered 0 to 3: Mamba, MoE, attention, MoE), used for
# single-node CI. Hidden 8192, 64 heads with 2 query groups, no position embedding, and the same
# MoE settings as the full model (512 experts, top-22 sigmoid routing scaled by 5.0, latent size
# 2048). Written out in full rather than derived from the full definition. Import has no side
# effects.
# 4-layer slice of NVIDIA Nemotron-3-Ultra-550B-A55B, for single-node (8 GPU) CI.
#
# Built by cluster_scripts/debug_tool_set/checkpoint/prune_nemotron_h.py, which
# keeps source layers 0,1,7,8 and renumbers them 0..3. That selection is the
# cheapest one covering every block type the full 108-layer model has:
#
#   layer 0 mamba   layer 1 moe   layer 2 attention   layer 3 moe   -> "ME*E"
#   MTP head: attention + moe                                       -> "*E"
#
# A prefix cut would need 8 layers to reach the first attention layer and drag
# in 4 MoE layers (~44B params) instead of 2. Everything else (512 experts,
# top-22, moe_latent_size=2048, sigmoid router + expert bias) is unchanged from
# nemotron-3-ultra-550b-a55b.py, so the weight-conversion path is identical.


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the constant flag string (no parameters, no environment use).
    return (
        "--disable-bias-linear "
        "--group-query-attention "
        "--num-attention-heads 64 "
        "--num-query-groups 2 "
        "--kv-channels 128 "
        "--num-layers 4 "
        "--hidden-size 8192 "
        "--ffn-hidden-size 5120 "
        "--normalization RMSNorm "
        "--position-embedding-type none "
        "--vocab-size 131072 "
        "--make-vocab-size-divisible-by 128 "
        "--untie-embeddings-and-output-weights "
        # MoE specifics (identical to the full model)
        "--num-experts 512 "
        "--moe-router-topk 22 "
        "--moe-ffn-hidden-size 5120 "
        "--moe-shared-expert-intermediate-size 10240 "
        "--moe-latent-size 2048 "
        "--moe-router-score-function sigmoid "
        "--moe-router-enable-expert-bias "
        "--moe-grouped-gemm "
        "--moe-router-dtype fp32 "
        "--moe-router-num-groups 1 "
        "--moe-router-group-topk 1 "
        "--moe-router-topk-scaling-factor 5.0 "
        "--moe-router-pre-softmax "
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-router-bias-update-rate 0 "
        "--moe-aux-loss-coeff 0 "
    )
