# [MODULE DOC]
# Megatron-to-Hugging Face converter for dense GLM4 models: maps one Megatron parameter (double
# "module." prefix) to a list of (HF name, tensor) pairs. Covers embeddings, head, final norm, and
# per-layer attention (fused QKV weight and bias split per query group into separate q/k/v), the
# fused MLP first linear kept fused as one gate-up projection, and GLM4's extra norms (QK norms and
# the post-attention and post-MLP "sandwich" norms).
import re
import torch


def convert_glm4_to_hf(args, name, param):
    # [FUNCTION DOC]
    # Input: args (hidden size, attention heads, query groups, optional kv_channels), Megatron name,
    # tensor. Exact names map to embed tokens, language model head and final model norm. Head size
    # is kv_channels if set (hidden over heads if absent or unset) and queries-per-group is heads
    # over query groups. For decoder layer names: output projection to o_proj; fused QKV weight is
    # viewed as (groups, queries-per-group + 2, head size, hidden) and split along axis 1 into
    # query-per-group, one key and one value slice, each reshaped to (rows, hidden), giving q, k, v
    # weights; fused QKV bias is viewed as (groups, per-group size) and split along axis 1 into
    # (queries-per-group * head size, head size, head size), each made contiguous and flattened;
    # MLP first linear stays fused as the gate-up projection weight (no split), second linear to down
    # projection; fused-QKV layer norm weight to the input layernorm; MLP fused layer norm weight to
    # the post-attention layernorm; query/key layernorm weights to q_norm/k_norm; post-self-attention
    # and post-MLP layernorm weights keep their names. Anything else raises a value error.
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
            return [
                (f"model.layers.{layer_idx}.mlp.gate_up_proj.weight", param),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

        # sandwitch norm
        elif rest == "post_self_attn_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_self_attn_layernorm.weight", param)]
        elif rest == "post_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_mlp_layernorm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
