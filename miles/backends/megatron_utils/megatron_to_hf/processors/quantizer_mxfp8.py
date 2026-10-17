# [MODULE DOC]
# MXFP8 (microscaling FP8) quantizer for converted Hugging Face tensors, with the same
# target-selection structure as the FP8 quantizer but using the project's MXFP8 quantization (which
# returns quantized data plus per-block scales). Supports opt-out lists: named high-precision
# layers (substring match on the Megatron name) and, optionally, bf16 first/last decoder layers.
import re

from miles.utils.mxfp8 import mxfp8_quantize


def quantize_params_mxfp8(args, megatron_name, converted_named_params, quantization_config):
    # [FUNCTION DOC]
    # Asserts the config method is mxfp8. Returns the input unchanged if any entry of the args list
    # of extra high-precision layer names occurs as a substring of the Megatron name. Finds a decoder
    # layer (unanchored search) or an MTP layer (transformer-layer wrapper segments dropped); none
    # found returns the input unchanged. If the first/last-layers-bf16 flag is set, layers with index
    # below the start count or at/after (num_layers minus end count) are returned unchanged. Routed
    # experts (linear_fc1/linear_fc2): quantize all pairs except HF names ending in "_scale", which are
    # dropped. Shared experts (linear_fc1/linear_fc2 weights): quantize all pairs. Otherwise if the
    # layer-relative name is a listed target (attention output/QKV, dense MLP linears, MLA
    # projections, indexer query, DeepSeek-V4 kv projection and indexer query; plus indexer key
    # weights unless indexer RoPE interleave is on, since interleaved indexers fuse that weight into
    # a bf16 tensor whose loader would misread the uint8 scales) quantize all pairs. Everything else
    # is returned unchanged.
    # KNOWN ISSUE: the layer-bf16 check applies to MTP layers too (their index is compared against
    # the main layer count), and the high-precision list is read twice via the same attribute.
    assert quantization_config["quant_method"] == "mxfp8"

    if getattr(args, "extra_high_precision_layers_megatron", False):
        for layer_name in getattr(args, "extra_high_precision_layers_megatron", ()):
            if layer_name in megatron_name:
                return converted_named_params

    decoder_layers_pattern = r"decoder\.layers\.(\d+)\.(.+)"
    match = re.search(decoder_layers_pattern, megatron_name)

    if not match:
        # check mtp layers
        mtp_layer_pattern = r"mtp\.layers\.(\d+)\.(.+)"
        match = re.search(mtp_layer_pattern, megatron_name)
        if not match:
            return converted_named_params
        layer_idx, rest = match.groups()
        rest = rest.replace("transformer_layer.", "").replace("mtp_model_layer.", "")
    else:
        layer_idx, rest = match.groups()

    # Skip quantization for BF16 tail of main decoder layers.
    if getattr(args, "first_last_layers_bf16", False):
        num_layers = int(args.num_layers)
        num_layers_at_start_in_bf16 = int(getattr(args, "num_layers_at_start_in_bf16", 0))
        num_layers_at_end_in_bf16 = int(getattr(args, "num_layers_at_end_in_bf16", 0))
        head_end_idx = num_layers_at_start_in_bf16
        tail_start_idx = num_layers - num_layers_at_end_in_bf16
        if int(layer_idx) < head_end_idx or int(layer_idx) >= tail_start_idx:
            return converted_named_params

    # experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, expert_idx = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                # skip bf16 weight_scale and input_scale
                # TODO: find a clearer way.
                if converted_name.endswith("_scale"):
                    continue
                quantize_named_params.extend(_quantize_param(converted_name, param))

            return quantize_named_params

    # shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                quantize_named_params.extend(_quantize_param(converted_name, param))

            return quantize_named_params

    mxfp8_param_names = [
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        # mla
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_q_down_proj.weight",
        "self_attention.linear_q_up_proj.weight",
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.linear_kv_up_proj.weight",
        "self_attention.wq_b.weight",
        # DeepSeek V4 attention
        "self_attention.linear_kv_proj.weight",
        "self_attention.core_attention.indexer.linear_wq_b.weight",
    ]
    if not getattr(args, "indexer_rope_interleave", False):
        # Non-interleaved indexers keep wk as a standalone quantized parameter in
        # SGLang; interleaved ones fuse wk into the bf16 wk_weights_proj, whose
        # loader would misread the uint8 e8m0 mxfp8 scales as integers.
        mxfp8_param_names.extend(
            [
                "self_attention.wk.weight",
                "self_attention.core_attention.indexer.linear_wk.weight",
            ]
        )

    if rest in mxfp8_param_names:
        quantize_named_params = []
        for converted_name, param in converted_named_params:
            quantize_named_params.extend(_quantize_param(converted_name, param))

        return quantize_named_params

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


def _quantize_param(name, weight):
    # [FUNCTION DOC] HELPER: Asserts the name ends with ".weight". Quantizes with the MXFP8 routine
    # (returns quantized data and scale) and returns [(name, quantized), (name with ".weight"
    # replaced by ".weight_scale_inv", scale)].
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    qweight, scale = mxfp8_quantize(weight)
    scale_name = name.replace(".weight", ".weight_scale_inv")
    return [(name, qweight), (scale_name, scale)]
