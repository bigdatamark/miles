# [MODULE DOC]
# Megatron-to-Hugging Face converters for Kimi vision-language models (Kimi-VL and Kimi-K2.5).
# Both wrap a shared language-model converter (DeepSeek-V3-style multi-latent attention with MoE,
# HF names under "language_model.model") and handle the vision parts by prefix renaming only: the
# vision encoder and the multimodal projector keep their sub-names under model-specific prefixes
# (Kimi-VL: vision_model -> vision_tower, multi_modal_projector unchanged; Kimi-K2.5: vision_tower
# and mm_projector unchanged), with the double "module." wrapper prefix dropped.
import re

import torch


def convert_kimivl_to_hf(args, name, param):
    # [FUNCTION DOC] Kimi-VL entry. Names starting with the double "module." plus "vision_model."
    # are renamed with prefix "vision_tower."; those starting with "module.module.multi_modal_
    # projector." are renamed with prefix "multi_modal_projector."; each returns a one-pair list with
    # the tensor unchanged. All other names go to the language-model converter.
    if name.startswith("module.module.vision_model."):
        hf_name = "vision_tower." + name[len("module.module.vision_model.") :]
        return [(hf_name, param)]

    if name.startswith("module.module.multi_modal_projector."):
        hf_name = "multi_modal_projector." + name[len("module.module.multi_modal_projector.") :]
        return [(hf_name, param)]

    return convert_language_model_to_hf(args, name, param)


def convert_kimi_k25_to_hf(args, name, param):
    # [FUNCTION DOC] Kimi-K2.5 entry. Names starting with the double "module." plus "vision_tower."
    # or "mm_projector." keep those prefixes with the wrapper prefix removed (one-pair list, tensor
    # unchanged). All other names go to the language-model converter.
    if name.startswith("module.module.vision_tower."):
        hf_name = "vision_tower." + name[len("module.module.vision_tower.") :]
        return [(hf_name, param)]

    if name.startswith("module.module.mm_projector."):
        hf_name = "mm_projector." + name[len("module.module.mm_projector.") :]
        return [(hf_name, param)]

    return convert_language_model_to_hf(args, name, param)


def convert_language_model_to_hf(args, name, param):
    # [FUNCTION DOC]
    # Language-model converter shared by both Kimi variants. Output names are under
    # "language_model." (head) or "language_model.model." (everything else). Exact names (which must
    # include the "language_model." segment after the wrapper prefix): embedding, output head, final
    # norm. Head size is kv_channels (or hidden over heads if absent/unset) and queries per group is
    # heads over query groups. Decoder layers (the "language_model." segment is optional in the
    # pattern): routed experts (first linear split in halves along axis 0 into gate then up; second
    # linear as down; other kinds raise value error); shared experts likewise; output projection;
    # query projection (no low-rank), query down projection to q_a_proj, the query up-projection
    # layer norm to q_a_layernorm, query up projection to q_b_proj; fused QKV bias split per group
    # into q, k, v biases (flattened); dense MLP first linear split into gate/up, second to down;
    # fused-QKV layer norm weight and the input layernorm to input layernorm; MLP fused layer norm
    # and pre-MLP layernorm to post-attention layernorm; KV down projection to the MQA kv_a
    # projection, KV up-projection layer norm and weight to kv_a_layernorm and kv_b_proj; router
    # weight and expert bias to the gate and its score-correction bias. Unknown names raise a value
    # error.
    # KNOWN ISSUE: the three exact-name cases require the "language_model." segment, while the layer
    # pattern treats it as optional, so a name without the segment works for layers but not for the
    # embedding, head or final norm.
    if name == "module.module.language_model.embedding.word_embeddings.weight":
        return [("language_model.model.embed_tokens.weight", param)]
    if name == "module.module.language_model.output_layer.weight":
        return [("language_model.lm_head.weight", param)]
    if name == "module.module.language_model.decoder.final_layernorm.weight":
        return [("language_model.model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.(?:language_model\.)?decoder\.layers\.(\d+)\.(.+)"
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
                    (
                        f"language_model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                        gate_weight,
                    ),
                    (f"language_model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
                return outputs
            elif rest == "linear_fc2":
                outputs = [
                    (f"language_model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param),
                ]
                return outputs
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            # rest = match.groups()[0]
            rest = match.group(1)
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"language_model.model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight", gate_weight),
                    (f"language_model.model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"language_model.model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"language_model.model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_q_proj.weight":
            return [(f"language_model.model.layers.{layer_idx}.self_attn.q_proj.weight", param)]
        elif rest == "self_attention.linear_q_down_proj.weight":
            return [(f"language_model.model.layers.{layer_idx}.self_attn.q_a_proj.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.layer_norm_weight":
            return [(f"language_model.model.layers.{layer_idx}.self_attn.q_a_layernorm.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.weight":
            return [(f"language_model.model.layers.{layer_idx}.self_attn.q_b_proj.weight", param)]
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
                (f"language_model.model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"language_model.model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"language_model.model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"language_model.model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"language_model.model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"language_model.model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight" or rest == "input_layernorm.weight":
            return [(f"language_model.model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"language_model.model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "self_attention.linear_kv_down_proj.weight":
            return [(f"language_model.model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight", param)]
        elif rest == "self_attention.linear_kv_up_proj.layer_norm_weight":
            return [(f"language_model.model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight", param)]
        elif rest == "self_attention.linear_kv_up_proj.weight":
            return [(f"language_model.model.layers.{layer_idx}.self_attn.kv_b_proj.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"language_model.model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "mlp.router.weight":
            return [(f"language_model.model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"language_model.model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]

    raise ValueError(f"Unknown parameter name: {name}")
