# [PACKAGE DOC]
# Converters from Megatron-format parameters to Hugging Face-format named tensors, one module per
# model family, plus a processors subpackage (padding removal and weight quantizers). Public entry
# points defined here: convert-to-HF (removes vocab padding, dispatches by lowercase model-name
# substring to a family converter, then quantizes), a post-processing helper, and a deprecated
# LoRA name converter. Each family converter takes (args, Megatron parameter name, tensor) and
# returns a list of (HF name, tensor) pairs.
from .deepseekv3 import convert_deepseekv3_to_hf
from .deepseekv4 import convert_deepseekv4_to_hf
from .deepseekv4_1 import convert_deepseekv4_1_to_hf
from .glm4 import convert_glm4_to_hf
from .glm4moe import convert_glm4moe_to_hf
from .glm5_next import convert_glm5_next_to_hf
from .inkling import convert_inkling_to_hf
from .kimi_k3 import convert_kimi_k3_to_hf
from .kimi_vl import convert_kimi_k25_to_hf, convert_kimivl_to_hf
from .llama import convert_llama_to_hf
from .mimo import convert_mimo_to_hf
from .processors import quantize_params, remove_padding
from .qwen2 import convert_qwen2_to_hf
from .qwen3_5 import convert_qwen3_5_to_hf
from .qwen3_8_next import convert_qwen3_8_next_to_hf
from .qwen3_next import convert_qwen3_next_to_hf
from .qwen3moe import convert_qwen3moe_to_hf


# TODO unify w/ `convert_to_hf`
def postprocess_hf_param(args, megatron_param_name, hf_param_name, param):
    # [FUNCTION DOC] Strips vocabulary padding rows from the tensor based on the Megatron parameter
    # name and args.vocab_size (embedding/output only; see padding remover) and returns the tensor.
    # KNOWN ISSUE: the HF parameter name argument is unused, and quantization is not applied here
    # (marked as a TODO), unlike the main conversion function.
    param = remove_padding(megatron_param_name, param, args.vocab_size)
    # TODO support quant
    return param


# TODO optimize code details
def convert_to_hf(args, model_name, name, param, quantization_config=None, packed_weight_basenames=None):
    # [FUNCTION DOC] Main conversion entry. Removes vocab padding from the tensor, converts to a
    # list of (HF name, tensor) pairs via the family dispatcher, then passes them to the quantizer
    # with the optional quantization config and packed weight base names; returns the quantizer's
    # list. Raises a value error for an unsupported model name (from the dispatcher).
    param = remove_padding(name, param, args.vocab_size)

    converted_named_tensors = _convert_to_hf_core(args, model_name, name, param)

    return quantize_params(args, name, converted_named_tensors, quantization_config, packed_weight_basenames)


# TODO optimize code details
def _convert_to_hf_core(args, model_name, name, param):
    # [FUNCTION DOC] HELPER: Lowercases the model name and picks the converter by the first matching
    # substring, in this order (order matters because several names contain others): glm5_next or
    # glm5next; glm4moelite, deepseekv3, glmmoedsa or glm_moe_dsa (all via the DeepSeek-V3
    # converter); glm4moe; glm4; qwen3moe; qwen3next; qwen3_8_next, qwen4_exp or qwen3.8_next;
    # qwen3_5 or qwen3_6; qwen2 or qwen3; deepseekv41; deepseekv4; inkling; llama; mimo; kimi_k3;
    # kimivl; kimi_k25. Raises a value error for no match. Returns the converter's list of pairs.
    model_name = model_name.lower()
    if "glm5_next" in model_name or "glm5next" in model_name:
        converted_named_tensors = convert_glm5_next_to_hf(args, name, param)
    elif (
        "glm4moelite" in model_name
        or "deepseekv3" in model_name
        or "glmmoedsa" in model_name
        or "glm_moe_dsa" in model_name
    ):
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)
    elif "glm4moe" in model_name:
        converted_named_tensors = convert_glm4moe_to_hf(args, name, param)
    elif "glm4" in model_name:
        converted_named_tensors = convert_glm4_to_hf(args, name, param)
    elif "qwen3moe" in model_name:
        converted_named_tensors = convert_qwen3moe_to_hf(args, name, param)
    elif "qwen3next" in model_name:
        converted_named_tensors = convert_qwen3_next_to_hf(args, name, param)
    elif "qwen3_8_next" in model_name or "qwen4_exp" in model_name or "qwen3.8_next" in model_name:
        converted_named_tensors = convert_qwen3_8_next_to_hf(args, name, param)
    elif "qwen3_5" in model_name or "qwen3_6" in model_name:
        converted_named_tensors = convert_qwen3_5_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "deepseekv41" in model_name:
        converted_named_tensors = convert_deepseekv4_1_to_hf(args, name, param)
    elif "deepseekv4" in model_name:
        converted_named_tensors = convert_deepseekv4_to_hf(args, name, param)
    elif "inkling" in model_name:
        converted_named_tensors = convert_inkling_to_hf(args, name, param)
    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    elif "mimo" in model_name:
        converted_named_tensors = convert_mimo_to_hf(args, name, param)
    elif "kimi_k3" in model_name:
        converted_named_tensors = convert_kimi_k3_to_hf(args, name, param)
    elif "kimivl" in model_name:
        converted_named_tensors = convert_kimivl_to_hf(args, name, param)
    elif "kimi_k25" in model_name:
        converted_named_tensors = convert_kimi_k25_to_hf(args, name, param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return converted_named_tensors


def convert_lora_to_hf(args, model_name, name, param):
    # [FUNCTION DOC]
    # Deprecated name-only conversion of a Megatron LoRA tensor name to a PEFT-style name; emits a
    # deprecation warning. Classifies as A (names containing ".linear_in." or ".lora_A.") or B
    # (".linear_out." or ".lora_B."); any other name is returned unchanged as a one-pair list.
    # Renames by plain substring replacement: double module prefix to base-model prefix, decoder
    # layers to model layers, attention QKV linear to the query projection, attention output linear
    # to the output projection, MLP first linear to the gate projection, MLP second linear to the
    # down projection, then adapter in/out weight suffixes to the A or B weight suffix. Returns a
    # one-element list with (new name, unchanged tensor). Args and model name are unused.
    # KNOWN ISSUE: fused layers are mapped to a single projection only (the fused QKV becomes only
    # the query projection, the fused FC1 only gate), which is wrong, as the docstring itself warns.
    """Convert Megatron LoRA parameter to HuggingFace PEFT format.

    .. deprecated::
        This function uses hardcoded string replacements that do NOT correctly
        handle fused layers (e.g. linear_qkv -> q/k/v_proj, linear_fc1 ->
        gate/up_proj). Use ``AutoBridge.export_adapter_weights`` instead, which
        properly splits fused adapter weights and handles TP gathering.

    Megatron format: module.module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight
    HF PEFT format:  base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    """
    import warnings

    warnings.warn(
        "convert_lora_to_hf uses incorrect hardcoded name mapping for fused layers. "
        "Use AutoBridge.export_adapter_weights instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Determine lora_A vs lora_B
    if ".linear_in." in name or ".lora_A." in name:
        lora_suffix = "lora_A.weight"
    elif ".linear_out." in name or ".lora_B." in name:
        lora_suffix = "lora_B.weight"
    else:
        return [(name, param)]

    # Convert Megatron naming to HF PEFT naming
    hf_name = name
    hf_name = hf_name.replace("module.module.", "base_model.model.")
    hf_name = hf_name.replace(".decoder.layers.", ".model.layers.")
    hf_name = hf_name.replace(".self_attention.linear_qkv", ".self_attn.q_proj")
    hf_name = hf_name.replace(".self_attention.linear_proj", ".self_attn.o_proj")
    hf_name = hf_name.replace(".mlp.linear_fc1", ".mlp.gate_proj")
    hf_name = hf_name.replace(".mlp.linear_fc2", ".mlp.down_proj")

    hf_name = hf_name.replace(".adapter.linear_in.weight", f".{lora_suffix}")
    hf_name = hf_name.replace(".adapter.linear_out.weight", f".{lora_suffix}")
    hf_name = hf_name.replace(".lora_A.weight", f".{lora_suffix}")
    hf_name = hf_name.replace(".lora_B.weight", f".{lora_suffix}")

    return [(hf_name, param)]
