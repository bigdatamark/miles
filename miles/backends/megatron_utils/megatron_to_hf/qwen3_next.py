# [MODULE DOC]
# Megatron-to-Hugging Face converter for Qwen3-Next (hybrid gated-delta-net linear attention and
# gated full attention, MoE with a shared expert, optional multi-token-prediction layers). Standard
# naming under "model."; MTP layers map to "mtp." names, with their inner transformer layer
# converted by recursion through a proxy decoder name. Gated attention's fused QKV carries a gate
# alongside queries, so queries are de-interleaved. The linear-attention convolution weight is
# converted from Megatron's group-major row layout to HF's flat layout using the head layout read
# from the HF checkpoint config.
import re

import torch

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import gdn_heads_of, qkv_group_major_to_flat


def convert_qwen3_next_to_hf(args, name, param):
    # [FUNCTION DOC]
    # Exact names: embedding, output head, final norm. Head size from kv_channels (or hidden over
    # heads); queries per group = heads over query groups. Decoder layers: routed experts (first
    # linear halves to gate then up, second to down; else value error); shared expert (names
    # under "shared_expert", the gate named gate_weight to the shared-expert gate; else value error);
    # output projection; fused QKV weight viewed as (groups, rows, head size, hidden) and split
    # along axis 1 into (2 * queries-per-group, 1, 1): the first part holds query and gate rows
    # interleaved by half and is reshaped to (groups, 2, queries-per-group, head size, hidden),
    # swapped on axes 1 and 2 and flattened, so the result alternates gate/query per head slot; key
    # and value flattened to rows; QKV bias split per group (sizes queries-per-group*head size, head
    # size, head size) and flattened; dense MLP, layernorm and router names; QK layernorms; the
    # linear-attention convolution weight is converted from group-major to flat layout using the
    # head layout of the HF checkpoint; names listed under self-attention for the input layernorm,
    # linear attention (A_log, dt_bias, in_proj_ba, in_proj_qkvz, norm, out_proj) and gated attention
    # (q/k/v/o projections and norms) are passed through with the "self_attention." prefix dropped.
    # MTP layers (module.module.mtp.layers.<idx>): enorm, hnorm, eh_proj, final_layernorm map to
    # fixed "mtp." names (not indexed); transformer-layer parts are converted recursively as if a
    # decoder layer of the same index and the "model.layers.<idx>" prefix rewritten to
    # "mtp.layers.<idx>". Unknown names raise a value error.
    # KNOWN ISSUE: the weight is split with a gate-doubled query part but the fused QKV bias is split
    # as if there were no gate (queries-per-group * head size), which is inconsistent for gated
    # attention layers.
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
                    (f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight", param)]
            elif rest == "gate_weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[2 * value_num_per_group, 1, 1], dim=1
            )
            q_param = (
                q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
                .transpose(1, 2)
                .reshape(-1, args.hidden_size)
            )
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
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]
        elif rest == "self_attention.linear_attn.conv1d.weight":
            return [
                (
                    f"model.layers.{layer_idx}.linear_attn.conv1d.weight",
                    qkv_group_major_to_flat(param, gdn_heads_of(args.hf_checkpoint)),
                )
            ]
        elif rest.startswith("self_attention.") and rest[len("self_attention.") :] in [
            "input_layernorm.weight",
            # linear attn
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_ba.weight",
            "linear_attn.in_proj_qkvz.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            # gated attn
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ]:
            rest = rest[len("self_attention.") :]
            return [(f"model.layers.{layer_idx}.{rest}", param)]

    # MTP (Multi-Token Prediction) layers
    mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
    match = re.match(mtp_layer_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # MTP-specific components
        if rest == "enorm.weight":
            return [("mtp.pre_fc_norm_embedding.weight", param)]
        elif rest == "hnorm.weight":
            return [("mtp.pre_fc_norm_hidden.weight", param)]
        elif rest == "eh_proj.weight":
            return [("mtp.fc.weight", param)]
        elif rest == "final_layernorm.weight":
            return [("mtp.norm.weight", param)]

        # transformer_layer components: reuse decoder conversion with mtp prefix.
        if rest.startswith(("transformer_layer.", "mtp_model_layer.")):
            transformer_rest = rest.split(".", 1)[1]
            proxy_name = f"module.module.decoder.layers.{layer_idx}.{transformer_rest}"
            results = convert_qwen3_next_to_hf(args, proxy_name, param)
            return [
                (hf_name.replace(f"model.layers.{layer_idx}", f"mtp.layers.{layer_idx}"), p) for hf_name, p in results
            ]

    raise ValueError(f"Unknown parameter name: {name}")
