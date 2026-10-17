# [MODULE DOC]
# Megatron-to-Hugging Face converter for Qwen3.5 / Qwen3.6 (vision-language-style naming under
# "model.language_model."; hybrid gated-delta-net linear attention with separate in-projections
# for qkv, z, b, a, and gated full attention; dense or MoE MLP with grouped fused experts or
# individual experts; optional multi-token-prediction layers mapped to "mtp." names). The group-
# major row layout of the linear attention's fused qkv projection and convolution is converted back
# to HF's flat layout using the head counts read from the HF checkpoint config.
import re

import torch

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import gdn_heads_of, qkv_group_major_to_flat


def _convert_mtp_layer(args, name, param, layer_idx):
    # [FUNCTION DOC] HELPER:
    # Handles a Megatron name that contains the MTP layers segment; layer_idx is the layer number
    # text. By substring in the name: enorm to "mtp.pre_fc_norm_embedding.weight", hnorm to
    # "mtp.pre_fc_norm_hidden.weight", final layernorm to "mtp.norm.weight", eh_proj to
    # "mtp.fc.weight" (all unindexed). Otherwise if the name contains the transformer-layer or
    # MTP-model-layer segment: builds a proxy decoder-layer name (replacing the segment path
    # with decoder.layers.<idx>), converts through the main converter, and rewrites each result name's
    # "model.language_model.layers.<idx>" prefix to "mtp.layers.<idx>" (names without it are kept).
    # Returns none when nothing matches, so the caller falls through to the regular logic.
    """Convert MTP layer parameters from Megatron to HuggingFace format."""
    if "enorm.weight" in name:
        return [("mtp.pre_fc_norm_embedding.weight", param)]
    if "hnorm.weight" in name:
        return [("mtp.pre_fc_norm_hidden.weight", param)]
    if "final_layernorm.weight" in name:
        return [("mtp.norm.weight", param)]
    if "eh_proj.weight" in name:
        return [("mtp.fc.weight", param)]

    _mtp_inner = next((p for p in ("transformer_layer", "mtp_model_layer") if p in name), None)
    if _mtp_inner is not None:
        proxy_name = name.replace(f"mtp.layers.{layer_idx}.{_mtp_inner}", f"decoder.layers.{layer_idx}")
        mapped_params = convert_qwen3_5_to_hf(args, proxy_name, param)

        final_params = []
        for hf_name, tensor in mapped_params:
            target_prefix = f"mtp.layers.{layer_idx}"
            if f"model.language_model.layers.{layer_idx}" in hf_name:
                new_hf_name = hf_name.replace(f"model.language_model.layers.{layer_idx}", target_prefix)
                final_params.append((new_hf_name, tensor))
            else:
                final_params.append((hf_name, tensor))
        return final_params

    return None


def convert_qwen3_5_to_hf(args, name, param):
    # [FUNCTION DOC]
    # MTP names (containing "mtp.layers"): extracts the layer index (the path component after
    # "layers"; value error if it cannot be found) and tries the MTP helper; a non-none result is
    # returned, otherwise falls through. Exact names: embedding to model.language_model.embed_tokens,
    # output head to lm_head, final norm to model.language_model.norm. Head size from kv_channels (or
    # hidden over heads); queries per group = heads over query groups. For decoder layers (HF prefix
    # model.language_model.layers.<idx>): grouped-GEMM expert tensors (names ending
    # experts.linear_fc1 / experts.linear_fc2) map to fused experts gate_up_proj / down_proj without
    # splitting; individual experts: first linear halves to gate then up, second to down (other kinds
    # raise value error); shared expert: first linear halves, second linear, and gate_weight to the
    # shared expert gate (others raise value error); output projection; fused QKV weight viewed as
    # (groups, rows, head size, hidden) and split along axis 1 into (2 * queries-per-group, 1, 1): the
    # query part (which holds query and gate rows) is reshaped to (groups, 2, queries-per-group, head
    # size, hidden), axes 1 and 2 swapped and flattened; key and value flattened; fused QKV bias
    # split per group into q, k, v biases (sizes queries-per-group*head size, head size, head size);
    # dense MLP halves and layernorm names; router weight and expert bias; QK layernorms; linear-
    # attention in-projection for qkv and the convolution weight are converted from group-major to
    # flat layout with HF-config head counts; the listed self-attention names (input layernorm,
    # A_log, dt_bias, in_proj_a/b/z, norm, out_proj, and gated-attention q/k/v/o and norms) pass
    # through with the "self_attention." prefix dropped. Unknown names raise a value error.
    # KNOWN ISSUE: the QKV weight split treats the query part as twice as wide (query plus gate), but
    # the fused QKV bias split uses the non-gated widths, which is inconsistent for gated attention.
    """Convert Qwen3.5 model parameters from Megatron to HuggingFace format.

    Qwen3.5 uses model.language_model.layers prefix and has separate
    in_proj_qkv, in_proj_z, in_proj_b, in_proj_a for linear attention.
    """
    # Handle MTP layers
    if "mtp.layers" in name:
        parts = name.split(".")
        try:
            layer_idx_loc = parts.index("layers") + 1
            layer_idx = parts[layer_idx_loc]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid MTP layer name format: {name}") from e

        result = _convert_mtp_layer(args, name, param, layer_idx)
        if result is not None:
            return result

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        prefix = f"model.language_model.layers.{layer_idx}"

        # experts (grouped gemm - fused format)
        if rest == "mlp.experts.linear_fc1":
            return [(f"{prefix}.mlp.experts.gate_up_proj", param)]
        elif rest == "mlp.experts.linear_fc2":
            return [(f"{prefix}.mlp.experts.down_proj", param)]

        # experts (ungrouped - individual expert format)
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2":
                return [(f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight", param)]
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
                    (f"{prefix}.mlp.shared_expert.gate_proj.weight", gate_weight),
                    (f"{prefix}.mlp.shared_expert.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"{prefix}.mlp.shared_expert.down_proj.weight", param)]
            elif rest == "gate_weight":
                return [(f"{prefix}.mlp.shared_expert_gate.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"{prefix}.self_attn.o_proj.weight", param)]
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
                (f"{prefix}.self_attn.q_proj.weight", q_param),
                (f"{prefix}.self_attn.k_proj.weight", k_param),
                (f"{prefix}.self_attn.v_proj.weight", v_param),
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
                (f"{prefix}.self_attn.q_proj.bias", q_bias),
                (f"{prefix}.self_attn.k_proj.bias", k_bias),
                (f"{prefix}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{prefix}.mlp.gate_proj.weight", gate_weight),
                (f"{prefix}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"{prefix}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{prefix}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{prefix}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"{prefix}.post_attention_layernorm.weight", param)]
        elif rest == "mlp.router.weight":
            return [(f"{prefix}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"{prefix}.mlp.gate.e_score_correction_bias", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"{prefix}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"{prefix}.self_attn.k_norm.weight", param)]
        elif rest in ("self_attention.linear_attn.in_proj_qkv.weight", "self_attention.linear_attn.conv1d.weight"):
            hf_rest = rest[len("self_attention.") :]
            return [(f"{prefix}.{hf_rest}", qkv_group_major_to_flat(param, gdn_heads_of(args.hf_checkpoint)))]
        elif rest.startswith("self_attention.") and rest[len("self_attention.") :] in [
            "input_layernorm.weight",
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            # gated attn (full attention layers)
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ]:
            rest = rest[len("self_attention.") :]
            return [(f"{prefix}.{rest}", param)]

    raise ValueError(f"Unknown parameter name: {name}")
