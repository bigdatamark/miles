# [MODULE DOC]
# Megatron-to-Hugging Face parameter name/tensor converter for DeepSeek-V3-style models (also used
# for GLM4 MoE Lite and GLM MoE DSA variants). One function maps a single Megatron parameter (names
# start with the double "module." wrapper prefix) to a list of (HF name, tensor) pairs: embeddings,
# output head, final norm, per-layer attention (multi-latent attention projections, norms, the
# sparse-attention indexer with optional RoPE channel reordering), dense and shared-expert MLPs,
# routed experts (the fused gate/up first linear is split in half along the first axis), router
# weight and expert bias, and multi-token-prediction (MTP) layers whose index is shifted past the
# main layers. A table maps names used by Megatron's native sparse attention to HF names.
import re

import torch


_NATIVE_DSA_ATTENTION_MAPPING = {
    "self_attention.core_attention.indexer.linear_wq_b.weight": "indexer.wq_b.weight",
    "self_attention.core_attention.indexer.linear_wk.weight": "indexer.wk.weight",
    "self_attention.core_attention.indexer.linear_weights_proj.weight": "indexer.weights_proj.weight",
    "self_attention.core_attention.indexer.k_norm.weight": "indexer.k_norm.weight",
    "self_attention.core_attention.indexer.k_norm.bias": "indexer.k_norm.bias",
    "self_attention.q_layernorm.weight": "q_a_layernorm.weight",
    "self_attention.kv_layernorm.weight": "kv_a_layernorm.weight",
}


def convert_deepseekv3_to_hf(args, name, param):
    # [FUNCTION DOC]
    # Input: run args, the Megatron parameter name, the tensor. Output: list of (HF name, tensor).
    # Exact-name cases: word embeddings to the embed tokens weight; output layer to the language
    # model head; final decoder layernorm to the final model norm. Then computes the attention head
    # size (kv_channels if set else hidden size over attention heads; falls back to the latter if
    # the attribute is missing) and the query heads per group. For decoder layer names (layer index
    # and remainder captured): (1) names in the native-DSA table map directly to attention-module
    # names (indexer weights/norms, q/kv layernorm to the "a" layernorms); (2) routed expert weights
    # (remainder like experts.<kind>.weight<expert index>): fused first linear split in two equal
    # halves along axis 0 into gate then up projections, second linear to down projection, anything
    # else raises a value error; (3) shared experts likewise (first linear split into gate/up);
    # (4) attention: output projection, query projection, query down projection, query up-projection
    # layernorm to the q "a" layernorm, query up projection to q_b, kv down projection to the MQA kv
    # "a" projection, kv up-projection layernorm and weight, indexer query/key/weights/k-norm
    # (with the indexer RoPE interleave flag set, the 64-wide rotary halves of each 128-wide head
    # are swapped for the query-b weight, and the first 64 and remaining entries swapped for the
    # key weight and key norm weight and bias; sizes 128 and 64 are hard coded); fused QKV bias is
    # reshaped by query group then split along axis 1 into sizes (queries per group times head size,
    # head size, head size), each flattened to separate q, k, v biases; (5) dense MLP first linear
    # split into gate/up, second linear to down; input layernorm (from either the fused QKV layer
    # norm weight or the input layernorm); post-attention layernorm (from the MLP fused layer norm or
    # the pre-MLP layernorm); router weight to the gate weight and router expert bias to the score
    # correction bias. For MTP layers (index shifted by args.num_layers): eh projection, enorm and
    # hnorm map directly, the final layernorm maps to the shared head norm, and anything else is
    # renamed as an ordinary decoder layer at the shifted index (dropping transformer-layer and
    # MTP-model-layer segments) and converted recursively. Raises a value error for unknown names
    # (including unmatched decoder-layer remainders).
    # KNOWN ISSUE: the key-weight swap rebuilds the tensor with a view to a two-dimensional shape
    # while the key norm and bias swaps are one-dimensional; the 128/64 constants assume the indexer
    # head dim and rope dim of one model and are not read from args.
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        if rest in _NATIVE_DSA_ATTENTION_MAPPING:
            # Native Megatron applies indexer RoPE in HF order; only the Miles
            # indexer below stores the reordered channels.
            return [(f"model.layers.{layer_idx}.self_attn.{_NATIVE_DSA_ATTENTION_MAPPING[rest]}", param)]

        # experts
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                outputs = [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
                return outputs
            elif rest == "linear_fc2":
                outputs = [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param),
                ]
                return outputs
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
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

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_q_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_proj.weight", param)]
        elif rest == "self_attention.linear_q_down_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_a_proj.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight", param)]

        # indexer
        elif rest == "self_attention.wq_b.weight":
            wq_b = param
            if args.indexer_rope_interleave:
                wq_b = wq_b.view(-1, 128, wq_b.shape[-1])  # hard code 128
                wq_b = torch.cat([wq_b[:, 64:], wq_b[:, :64]], dim=1).view(-1, wq_b.shape[-1])
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight", wq_b)]
        elif rest == "self_attention.wk.weight":
            wk = param
            if args.indexer_rope_interleave:
                wk = torch.cat([wk[64:], wk[:64]], dim=0).view(-1, wk.shape[-1])
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wk.weight", wk)]
        elif rest == "self_attention.weights_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.weights_proj.weight", param)]
        elif rest == "self_attention.k_norm.weight":
            knorm_weight = param
            if args.indexer_rope_interleave:
                knorm_weight = torch.cat([knorm_weight[64:], knorm_weight[:64]], dim=0)
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.weight", knorm_weight)]
        elif rest == "self_attention.k_norm.bias":
            knorm_bias = param
            if args.indexer_rope_interleave:
                knorm_bias = torch.cat([knorm_bias[64:], knorm_bias[:64]], dim=0)
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.bias", knorm_bias)]

        elif rest == "self_attention.linear_q_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_b_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight" or rest == "input_layernorm.weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "self_attention.linear_kv_down_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight", param)]
        elif rest == "self_attention.linear_kv_up_proj.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight", param)]
        elif rest == "self_attention.linear_kv_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]

    mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
    match = re.match(mtp_layer_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        layer_idx = int(layer_idx) + args.num_layers
        if rest == "eh_proj.weight":
            return [(f"model.layers.{layer_idx}.eh_proj.weight", param)]
        elif rest == "enorm.weight":
            return [(f"model.layers.{layer_idx}.enorm.weight", param)]
        elif rest == "hnorm.weight":
            return [(f"model.layers.{layer_idx}.hnorm.weight", param)]
        elif rest == "final_layernorm.weight":
            return [(f"model.layers.{layer_idx}.shared_head.norm.weight", param)]
        else:
            name = f"module.module.decoder.layers.{layer_idx}.{rest}"
            name = name.replace("transformer_layer.", "").replace("mtp_model_layer.", "")
            return convert_deepseekv3_to_hf(args, name, param)

    raise ValueError(f"Unknown parameter name: {name}")
