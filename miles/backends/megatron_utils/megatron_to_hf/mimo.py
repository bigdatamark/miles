# [MODULE DOC]
# Megatron-to-Hugging Face converter for MiMo models, which are Qwen2 models with additional
# multi-token-prediction (MTP) layers. Non-MTP names are delegated to the Qwen2 converter; MTP
# names are handled here (direct norm/projection names under model.mtp_layers.<idx>, with the inner
# transformer layer converted by proxying through the Qwen2 converter and renaming the layer
# prefix).
import re
import torch
from .qwen2 import convert_qwen2_to_hf


def convert_mimo_to_hf(args, name, param):
    # [FUNCTION DOC] Entry point: names containing the substring "mtp" go to the MTP converter,
    # everything else to the Qwen2 converter. Returns a list of (HF name, tensor).
    # KNOWN ISSUE: the substring test also matches any non-MTP name that merely contains "mtp".
    """
    Convert MiMo model parameters from Megatron to HuggingFace format.

    MiMo extends Qwen2 with MTP (Multi-Token Prediction) layers.
    """

    if "mtp" in name:
        return convert_mimo_mtp_param(args, name, param)

    return convert_qwen2_to_hf(args, name, param)


def convert_mimo_mtp_param(args, name, param):
    # [FUNCTION DOC]
    # Input: args, Megatron MTP name of the form module.module.mtp.layers.<idx>.<component>, tensor.
    # Raises value error if the name does not fit that form. Direct components map to
    # model.mtp_layers.<idx>: enorm to token_layernorm, hnorm to hidden_layernorm, eh_proj to
    # input_proj, final_layernorm to final_layernorm. For eh_proj the tensor's two column halves
    # (split along axis 1) are swapped before returning (the two orders differ between the formats).
    # Components with prefix "transformer_layer." or "mtp_model_
    # layer.": the prefix is dropped, the rest is converted as if it were a normal decoder layer of
    # the same index by the Qwen2 converter, and the resulting HF names have their model.layers.<idx>
    # prefix rewritten to model.mtp_layers.<idx>. Any other component raises a value error.
    # Returns a list of (HF name, tensor).
    """
    Convert MTP layer parameters from Megatron to HuggingFace format.

    MTP layers in MiMo contain:
    - LayerNorms (token_layernorm, hidden_layernorm, final_layernorm)
    - Input projection (input_proj)
    - Self attention (reuses Qwen2 attention structure)
    - MLP (reuses Qwen2 MLP structure)

    Based on MimoBridge._convert_mtp_param logic (reverse mapping)
    """
    mtp_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
    match = re.match(mtp_pattern, name)

    if not match:
        raise ValueError(f"Invalid MTP parameter name: {name}")

    layer_idx, component = match.groups()

    # Direct mappings for MTP-specific components (Megatron -> HF)
    # Based on MimoBridge direct_name_mapping (reversed)
    direct_mappings = {
        "enorm.weight": f"model.mtp_layers.{layer_idx}.token_layernorm.weight",
        "hnorm.weight": f"model.mtp_layers.{layer_idx}.hidden_layernorm.weight",
        "eh_proj.weight": f"model.mtp_layers.{layer_idx}.input_proj.weight",
        "final_layernorm.weight": f"model.mtp_layers.{layer_idx}.final_layernorm.weight",
    }
    if component == "eh_proj.weight":
        first_half, second_half = param.chunk(2, dim=1)
        param = torch.cat([second_half, first_half], dim=1)

    # Check direct mappings first
    if component in direct_mappings:
        return [(direct_mappings[component], param)]

    transformer_layer_prefixes = ("transformer_layer.", "mtp_model_layer.")
    matched_prefix = next((p for p in transformer_layer_prefixes if component.startswith(p)), None)
    if matched_prefix is not None:
        transformer_component = component[len(matched_prefix) :]

        # Create proxy name for reusing existing Qwen2 conversion functions
        proxy_name = f"module.module.decoder.layers.{layer_idx}.{transformer_component}"

        # Use existing convert_qwen2_to_hf function for transformer components
        results = convert_qwen2_to_hf(args, proxy_name, param)

        # Replace model.layers with mtp_layers in results
        converted_results = []
        for hf_name, hf_param in results:
            # Replace model.layers.{idx} with mtp_layers.{idx}
            hf_name = hf_name.replace(f"model.layers.{layer_idx}", f"model.mtp_layers.{layer_idx}")
            converted_results.append((hf_name, hf_param))

        return converted_results

    raise ValueError(f"Unknown MTP component: {component} in {name}")
