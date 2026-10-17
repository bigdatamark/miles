# [MODULE DOC]
# Megatron-to-Hugging Face converter for dense Llama models: maps one Megatron parameter (double
# "module." prefix) to a list of (HF name, tensor) pairs, covering embeddings, head, final norm,
# attention (fused QKV split per query group into q, k, v; output projection), SwiGLU MLP (fused
# first linear split in halves into gate then up; second linear as down) and layernorms. No bias,
# no QK norm and no MoE handling.
import re
import torch


def convert_llama_to_hf(args, name, param):
    # [FUNCTION DOC]
    # Exact names: embedding, output head, final norm. Head size is kv_channels (or hidden over
    # heads if absent/unset); queries per group is heads over query groups. Decoder layers: output
    # projection to o_proj; fused QKV weight viewed as (query groups, queries-per-group + 2, head
    # size, hidden) and split along axis 1 into q (queries-per-group slices), k (one) and v (one),
    # each reshaped to (rows, hidden); fused first MLP linear split in halves along axis 0 into
    # gate then up; second linear to down; QKV-fused layer norm weight to input layernorm; MLP-fused
    # layer norm weight and the pre-MLP layernorm to post-attention layernorm. Unknown names raise
    # a value error.
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
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            # Split QKV weight for Llama
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "mlp.linear_fc1.weight":
            # Split gate and up projections for SwiGLU
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
