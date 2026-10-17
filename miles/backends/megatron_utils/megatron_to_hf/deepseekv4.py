# [MODULE DOC]
# Megatron-to-Hugging Face converter for DeepSeek-V4 models. One function maps a Megatron
# parameter (double "module." prefix) to a list of (HF name, tensor): embeddings, head, final norm,
# the model-level hyper-connection head tensors, per-layer hyper-connection tensors (function,
# base and a packed three-element scale), attention (low-rank query, key/value, grouped output
# projections, norms, attention sink, compressor, indexer and its compressor), routed and shared
# experts (fused first linear split in half into gate and up), router weight/bias/token-to-expert
# table, dense MLP and layernorms. A module-level pending store plus a packing helper assembles
# the hyper-connection scale from three separately arriving scalars (state persists across calls).
import re

import torch


_PENDING_ALPHAS: dict[str, dict[str, torch.Tensor]] = {}


def _packed_alphas(name: str, param):
    # [FUNCTION DOC] HELPER: Collects one of three hyper-connection scalars. Splits the Megatron
    # name at its last ".alpha_" into a site prefix and a segment label, stores the tensor reshaped
    # to length one in a module-level pending store keyed by prefix then label. While fewer than
    # three labels are present returns none. When three are present removes the site from the store
    # and returns the concatenation in the fixed order pre, post, res (a length-three tensor);
    # a key error occurs if the three labels are not exactly those. Global mutable state: stale
    # entries remain if a site is never completed, and a repeated label overwrites.
    """Pack a hyper-connection site's three alphas the way the checkpoint stores them.

    Megatron keeps alpha_pre / alpha_post / alpha_res as separate scalars, one per bias
    segment; the checkpoint carries them as one [pre, post, res] tensor. The three may
    arrive in any order across weight-sync buckets, so stash pieces and emit the packed
    tensor once the site is complete.
    """
    prefix, seg = name.rsplit(".alpha_", 1)
    slots = _PENDING_ALPHAS.setdefault(prefix, {})
    slots[seg] = param.reshape(1)
    if len(slots) < 3:
        return None
    del _PENDING_ALPHAS[prefix]
    return torch.cat([slots[seg] for seg in ("pre", "post", "res")])


def convert_deepseekv4_to_hf(args, name, param):
    # [FUNCTION DOC]
    # Input: args (unused), Megatron name, tensor. Output: list of (HF name, tensor); an empty list
    # means "nothing to emit yet" (a hyper-connection alpha whose siblings have not all arrived).
    # Exact names: embedding, output head, final norm, and decoder hyper-connection head function,
    # base and scale (to the model-level hc_head tensors). Decoder layer names (layer index,
    # remainder): self-attention and MLP hyper-connection mapping projection to the function
    # tensor, bias to the base tensor, alpha pieces to the packed scale tensor (via the packing
    # helper); routed expert weights (first linear halves split along axis 0 into gate then up
    # projections; second linear to down; other kinds raise a value error); shared experts likewise;
    # attention: query down to wq_a, query layernorm to q_norm, query up to wq_b, kv projection to
    # wkv, kv layernorm to kv_norm, grouped output projection to wo_a, output projection to wo_b,
    # attention sink; compressor ape/wkv/wgate/norm; indexer wq_b/wk/k_norm weight and bias/weights
    # projection; indexer compressor ape/wkv/wgate/norm; input layernorm and pre-MLP layernorm
    # (to post-attention layernorm); router weight, expert bias, token-to-expert table (to the topk
    # module); dense MLP first linear split to gate/up and second linear to down. Raises a value
    # error for unknown names.
    # KNOWN ISSUE: the grouped output projection key has no ".weight" suffix, unlike the sibling
    # projections, so it only matches a bare parameter of that name.
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    if name == "module.module.decoder.hc_head_fn":
        return [("model.hc_head_fn", param)]
    if name == "module.module.decoder.hc_head_base":
        return [("model.hc_head_base", param)]
    if name == "module.module.decoder.hc_head_scale":
        return [("model.hc_head_scale", param)]

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        if rest == "self_attention_hyper_connection.mapping_proj.weight":
            return [(f"model.layers.{layer_idx}.hc_attn_fn", param)]
        elif rest == "self_attention_hyper_connection.bias":
            return [(f"model.layers.{layer_idx}.hc_attn_base", param)]
        elif rest.startswith("self_attention_hyper_connection.alpha_"):
            packed = _packed_alphas(name, param)
            return [] if packed is None else [(f"model.layers.{layer_idx}.hc_attn_scale", packed)]
        elif rest == "mlp_hyper_connection.mapping_proj.weight":
            return [(f"model.layers.{layer_idx}.hc_ffn_fn", param)]
        elif rest == "mlp_hyper_connection.bias":
            return [(f"model.layers.{layer_idx}.hc_ffn_base", param)]
        elif rest.startswith("mlp_hyper_connection.alpha_"):
            packed = _packed_alphas(name, param)
            return [] if packed is None else [(f"model.layers.{layer_idx}.hc_ffn_scale", packed)]

        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2":
                return [(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param)]
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_q_down_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wq_a.weight", param)]
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wq_b.weight", param)]
        elif rest == "self_attention.linear_kv_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wkv.weight", param)]
        elif rest == "self_attention.kv_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_norm.weight", param)]
        elif rest == "self_attention.linear_o_group_proj":
            return [(f"model.layers.{layer_idx}.self_attn.wo_a.weight", param)]
        elif rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wo_b.weight", param)]
        elif rest == "self_attention.core_attention.attn_sink":
            return [(f"model.layers.{layer_idx}.self_attn.attn_sink", param)]

        elif rest == "self_attention.core_attention.compressor.ape":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.ape", param)]
        elif rest == "self_attention.core_attention.compressor.linear_wkv.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.wkv.weight", param)]
        elif rest == "self_attention.core_attention.compressor.linear_wgate.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.wgate.weight", param)]
        elif rest == "self_attention.core_attention.compressor.norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.norm.weight", param)]

        elif rest == "self_attention.core_attention.indexer.linear_wq_b.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight", param)]
        elif rest == "self_attention.core_attention.indexer.linear_wk.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wk.weight", param)]
        elif rest == "self_attention.core_attention.indexer.k_norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.weight", param)]
        elif rest == "self_attention.core_attention.indexer.k_norm.bias":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.bias", param)]
        elif rest == "self_attention.core_attention.indexer.linear_weights_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.weights_proj.weight", param)]

        elif rest == "self_attention.core_attention.indexer.compressor.ape":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.ape", param)]
        elif rest == "self_attention.core_attention.indexer.compressor.linear_wkv.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.wkv.weight", param)]
        elif rest == "self_attention.core_attention.indexer.compressor.linear_wgate.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.wgate.weight", param)]
        elif rest == "self_attention.core_attention.indexer.compressor.norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.norm.weight", param)]

        elif rest == "input_layernorm.weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]
        elif rest == "mlp.router.tid2eid":
            return [(f"model.layers.{layer_idx}.mlp.topk.tid2eid", param)]

        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
