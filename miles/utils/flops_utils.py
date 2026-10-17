# [MODULE DOC]
# Analytic forward-pass FLOPs model of a decoder-only transformer (dense or mixture-of-experts,
# optionally multi-head latent attention), plus a converter from a huggingface style model config
# to the argument namespace the model expects. All counts are multiply-add = 2 FLOPs per matrix
# product element, per sequence, summed over a list of sequence lengths. Reports TFLOPs per gpu by
# dividing by the world size and 1e12. Pure functions; no I/O.
from types import SimpleNamespace


def calculate_embedding_flops(seqlen, hidden_size):
    # [FUNCTION DOC] FLOPs of an embedding step modelled as 2 times sequence length times hidden
    # size.
    # KNOWN ISSUE: not used by the forward FLOPs total in this module.
    return 2 * seqlen * hidden_size


def calculate_lm_head_flops(seqlen, hidden_size, vocab_size):
    # [FUNCTION DOC] FLOPs of the output projection to the vocabulary: 2 times sequence length times
    # hidden size times vocabulary size.
    return 2 * seqlen * hidden_size * vocab_size


def calculate_qkv_projection_flops(args, seqlen, hidden_size, num_attention_heads, num_query_groups):
    # [FUNCTION DOC] FLOPs of the query, key and value projections of one layer. Query: without a
    # query low-rank setting, 2 * seqlen * hidden * heads * kv channels; with it, the low-rank path
    # 2 * seqlen * q rank * (hidden + heads * (qk head dim + qk positional head dim)). Key/value:
    # without a kv low-rank setting, 2 (k and v) * 2 * seqlen * hidden * query groups * kv channels;
    # with it, 2 * seqlen * (kv rank * (hidden + heads * (qk head dim + v head dim)) + hidden * qk
    # positional head dim). Returns the sum.
    # KNOWN ISSUE: the low-rank branches use the hidden size and head count from args while the
    # dense branches use the function parameters; they agree only if the caller passes the same
    # values.
    if args.q_lora_rank is None:
        q_flops = 2 * seqlen * hidden_size * num_attention_heads * args.kv_channels
    else:
        q_flops = (
            2
            * seqlen
            * args.q_lora_rank
            * (args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim))
        )
    if args.kv_lora_rank is None:
        kv_flops = 2 * 2 * seqlen * hidden_size * num_query_groups * args.kv_channels
    else:
        kv_flops = (
            2
            * seqlen
            * (
                args.kv_lora_rank
                * (args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim))
                + args.hidden_size * args.qk_pos_emb_head_dim
            )
        )

    return q_flops + kv_flops


def calculate_attention_flops(args, seqlen, num_attention_heads):
    # [FUNCTION DOC] FLOPs of the attention score and value mixing of one layer. Score product
    # (causal, hence the factor one half): 2 * heads * seqlen^2 * head dim / 2, where head dim is
    # qk head dim plus qk positional head dim if the latter is non-zero, else kv channels. Value
    # product: heads * seqlen^2 * (v head dim if set and non-zero, else kv channels), not halved.
    # KNOWN ISSUE: the causal halving is applied to the score product only, not to the value
    # product, so the value product counts the full square.
    # QK^T with causal
    if args.qk_pos_emb_head_dim:
        flops = 2 * num_attention_heads * seqlen * seqlen * (args.qk_head_dim + args.qk_pos_emb_head_dim) / 2
    else:
        flops = 2 * num_attention_heads * seqlen * seqlen * args.kv_channels / 2
    # A*V
    if args.v_head_dim:
        flops += num_attention_heads * seqlen * seqlen * args.v_head_dim
    else:
        flops += num_attention_heads * seqlen * seqlen * args.kv_channels
    return flops


def calculate_output_flops(seqlen, hidden_size):
    # [FUNCTION DOC] FLOPs of the attention output projection, modelled as a square hidden by hidden
    # matrix: 2 * seqlen * hidden^2.
    return 2 * seqlen * hidden_size * hidden_size


def calculate_mlp_flops(seqlen, hidden_size, ffn_hidden_size):
    # [FUNCTION DOC] FLOPs of a gated feed-forward block with three matrices (gate, up, down):
    # 2 * seqlen * hidden * ffn hidden * 3.
    return 2 * seqlen * hidden_size * ffn_hidden_size * 3


def calculate_layer_flops(args, seqlen, hidden_size, num_attention_heads, num_query_groups, ffn_hidden_size):
    # [FUNCTION DOC] FLOPs of one transformer layer: qkv projections plus attention plus output
    # projection plus feed-forward with the given ffn width.
    return (
        calculate_qkv_projection_flops(args, seqlen, hidden_size, num_attention_heads, num_query_groups)
        + calculate_attention_flops(args, seqlen, num_attention_heads)
        + calculate_output_flops(seqlen, hidden_size)
        + calculate_mlp_flops(seqlen, hidden_size, ffn_hidden_size)
    )


def calculate_fwd_flops(
    seqlens,
    args,
):
    # [FUNCTION DOC] Total forward FLOPs for a list of sequence lengths. Layer split: without
    # experts, all layers are dense. With experts, the MoE ffn width is the expert width times the
    # router top-k plus the shared expert width (zero if absent or none). If a layer frequency
    # attribute exists: for a list, layers with value zero are dense and positive ones are MoE; for
    # an integer, layer indices divisible by it are MoE and the others dense. If the attribute is
    # absent, all layers are MoE. Then for each sequence length: dense layer FLOPs times the dense
    # layer count (if any), MoE layer FLOPs times the MoE layer count (if any), plus the language
    # model head FLOPs. Returns the sum over sequences.
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    num_query_groups = args.num_query_groups
    vocab_size = args.vocab_size

    total_flops = 0

    dense_ffn = args.ffn_hidden_size
    if args.num_experts is None:
        num_dense_layers = args.num_layers
        num_moe_layers = 0
    else:
        shared_expert_ffn = getattr(args, "moe_shared_expert_intermediate_size", None)
        if shared_expert_ffn is None:
            shared_expert_ffn = 0

        moe_ffn = args.moe_ffn_hidden_size * args.moe_router_topk + shared_expert_ffn
        if hasattr(args, "moe_layer_freq"):
            if isinstance(args.moe_layer_freq, list):
                num_dense_layers = sum(1 for freq in args.moe_layer_freq if freq == 0)
                num_moe_layers = sum(1 for freq in args.moe_layer_freq if freq > 0)
            else:
                num_dense_layers = sum(1 for i in range(args.num_layers) if i % args.moe_layer_freq != 0)
                num_moe_layers = sum(1 for i in range(args.num_layers) if i % args.moe_layer_freq == 0)
        else:
            num_dense_layers = 0
            num_moe_layers = args.num_layers

    for seqlen in seqlens:
        if num_dense_layers > 0:
            total_flops += (
                calculate_layer_flops(
                    args,
                    seqlen,
                    hidden_size,
                    num_attention_heads,
                    num_query_groups,
                    dense_ffn,
                )
                * num_dense_layers
            )

        if num_moe_layers > 0:
            total_flops += (
                calculate_layer_flops(
                    args,
                    seqlen,
                    hidden_size,
                    num_attention_heads,
                    num_query_groups,
                    moe_ffn,
                )
                * num_moe_layers
            )

        total_flops += calculate_lm_head_flops(seqlen, hidden_size, vocab_size)

    return total_flops


def fwd_tflops_per_gpu(seqlens, args, world_size):
    # [FUNCTION DOC] Forward FLOPs of the sequences divided evenly across the world size, in units
    # of 1e12 (TFLOPs). Assumes perfect data/model parallel division.
    return calculate_fwd_flops(seqlens, args) / world_size / 1e12


def _first(config, *names, default=None):
    # [FUNCTION DOC] HELPER: Returns the first attribute among the names whose value is present and
    # not none on the config; otherwise the default.
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _moe_layer_pattern(config, num_layers, num_experts):
    # [FUNCTION DOC] HELPER: Per-layer list of zero (dense) or one (MoE). None when there are no
    # experts. If the config has a first-k-dense setting, layers below k are dense and the rest MoE.
    # Otherwise layers listed as mlp-only are dense, and among others a layer is MoE when its
    # one-based index is divisible by the sparse step (default 1; a zero or none step counts as 1).
    if num_experts is None:
        return None
    first_dense = getattr(config, "first_k_dense_replace", None)
    if first_dense is not None:
        return [0 if i < first_dense else 1 for i in range(num_layers)]
    mlp_only = set(getattr(config, "mlp_only_layers", None) or ())
    step = getattr(config, "decoder_sparse_step", 1) or 1
    return [0 if i in mlp_only or (i + 1) % step != 0 else 1 for i in range(num_layers)]


def flops_args_from_hf_config(config):
    # [FUNCTION DOC] Converts a huggingface style model config into the simple namespace of fields
    # the FLOPs functions read. Uses the text sub-config when the config offers one (via its getter
    # or a text config attribute), else the config itself. Layer count comes from either the hidden
    # layers or layers attribute (assertion error if neither). Expert count from the first present
    # of routed experts, experts, local experts. Dense ffn width is the intermediate size; MoE ffn
    # width is the MoE intermediate size, defaulting to the dense width when experts exist. Shared
    # expert width is the explicit shared width, else shared expert count times the MoE width.
    # Layer pattern from the pattern helper. Raises a value error if a needed ffn width (dense for
    # dense layers, MoE for MoE layers) is missing. Other fields: query groups from key/value heads
    # (default heads), kv channels from head dim (default hidden divided by heads), router top-k from
    # experts per token or top-k (default 1), low-rank settings, and head dims (none become zero).
    getter = getattr(config, "get_text_config", None)
    config = (getter() if callable(getter) else getattr(config, "text_config", None)) or config

    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    num_layers = _first(config, "num_hidden_layers", "num_layers")
    assert num_layers is not None, f"no layer count on {type(config).__name__}; cannot size the FLOPs model"
    num_experts = _first(config, "n_routed_experts", "num_experts", "num_local_experts")

    dense_ffn = getattr(config, "intermediate_size", None)

    moe_ffn = getattr(config, "moe_intermediate_size", None)
    if num_experts is not None and moe_ffn is None:
        moe_ffn = dense_ffn

    shared_ffn = getattr(config, "shared_expert_intermediate_size", None)
    if shared_ffn is None and getattr(config, "n_shared_experts", None):
        shared_ffn = config.n_shared_experts * moe_ffn

    moe_layer_freq = _moe_layer_pattern(config, num_layers, num_experts)
    needs_dense = moe_layer_freq is None or any(f == 0 for f in moe_layer_freq)
    needs_moe = moe_layer_freq is not None and any(f > 0 for f in moe_layer_freq)
    if (needs_dense and dense_ffn is None) or (needs_moe and moe_ffn is None):
        raise ValueError(
            f"{type(config).__name__} does not expose the FFN widths the FLOPs model needs "
            f"(dense={dense_ffn}, moe={moe_ffn}); it likely nests them under a sub-config"
        )

    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=_first(config, "num_key_value_heads", default=num_attention_heads),
        vocab_size=config.vocab_size,
        num_layers=num_layers,
        ffn_hidden_size=dense_ffn,
        kv_channels=_first(config, "head_dim", default=hidden_size // num_attention_heads),
        num_experts=num_experts,
        moe_ffn_hidden_size=moe_ffn,
        moe_router_topk=_first(config, "num_experts_per_tok", "moe_topk", default=1),
        moe_shared_expert_intermediate_size=shared_ffn,
        moe_layer_freq=moe_layer_freq,
        q_lora_rank=getattr(config, "q_lora_rank", None),
        kv_lora_rank=getattr(config, "kv_lora_rank", None),
        qk_head_dim=getattr(config, "qk_nope_head_dim", None) or 0,
        qk_pos_emb_head_dim=getattr(config, "qk_rope_head_dim", None) or 0,
        v_head_dim=getattr(config, "v_head_dim", None) or 0,
    )
