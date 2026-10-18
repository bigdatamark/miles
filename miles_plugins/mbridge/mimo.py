# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# [MODULE DOC]
# mbridge bridge for MiMo models: the Qwen2 bridge plus multi-token prediction (mtp) layers. Adds the
# mtp configuration and block spec, the mtp parameter name mapping (Hugging Face stores mtp layers
# under a separate mtp layers prefix) and swaps the two halves of the embedding-hidden projection
# between the two frameworks' concatenation orders.
import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

from mbridge.core import register_model
from mbridge.models import Qwen2Bridge


@register_model("mimo")
class MimoBridge(Qwen2Bridge):
    # [CLASS DOC] Registered under the name mimo. Supports a single stack of mtp layers numbered from
    # zero.
    """
    Bridge implementation for Mimo models.

    This class extends Qwen2Bridge to provide specific configurations and
    optimizations for Mimo models, handling the conversion between
    Hugging Face Mimo format and Megatron-Core.

    MiMo adds MTP (Multi-Token Prediction) layers on top of Qwen2 architecture.
    """

    def _build_config(self):
        # [FUNCTION DOC] Build the config through the base helper with qkv bias on and no query-key
        # layer norm; when the Hugging Face config contains the next-token-prediction layer count,
        # passes it as the mtp layer count. Returns the config.
        # KNOWN ISSUE: the presence test uses a containment check directly on the config object, which
        # only works if the config type supports it; plain config objects usually do not.
        """Override to add MTP configuration."""
        hf_config = self.hf_config

        # Add MTP configuration if present
        mtp_args = {}
        if "num_nextn_predict_layers" in hf_config:
            mtp_args["mtp_num_layers"] = hf_config.num_nextn_predict_layers

        return self._build_base_config(
            add_qkv_bias=True,
            qk_layernorm=False,
            **mtp_args,
        )

    def _get_gptmodel_args(self) -> dict:
        # [FUNCTION DOC] Model constructor arguments from the base class; if the config has mtp layers,
        # also adds an mtp block spec built with transformer-engine layers.
        # KNOWN ISSUE: the config object itself is passed where the transformer layer spec is
        # expected, which looks like a mistake (the layer spec is never obtained).
        """Override to add MTP block spec if needed."""
        ret = super()._get_gptmodel_args()

        # Add MTP block spec if MTP layers are present
        if self.config.mtp_num_layers is not None:
            transformer_layer_spec = self.config
            mtp_block_spec = get_gpt_mtp_block_spec(self.config, transformer_layer_spec, use_transformer_engine=True)
            ret["mtp_block_spec"] = mtp_block_spec

        return ret

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        # [FUNCTION DOC] Hugging Face names for a Megatron parameter: names containing the mtp marker
        # use the mtp conversion; all others use the parent mapping.
        """Override to handle MTP layer mappings."""
        # Check if this is an MTP layer weight
        if "mtp" in mcore_weights_name:
            return self._convert_mtp_param(mcore_weights_name)

        # Otherwise use parent class mapping
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _convert_mtp_param(self, name: str) -> list[str]:
        # [FUNCTION DOC]
        # Hugging Face names for an mtp parameter. Raises not-implemented unless the name contains the
        # mtp layers path. The mtp layer index is the third dotted segment. Four direct entries (token
        # norm, hidden norm, input projection, final norm) map to the mtp layer path of the index and
        # return a one-element list. Otherwise, if the name contains an inner layer path (either
        # supported inner module spelling), it is rewritten to a regular decoder layer path with the
        # same index and mapped by the attention mapping (self attention or input layer norm names) or
        # the MLP mapping (mlp names; else raises not-implemented), and the resulting regular layer
        # path is replaced by the mtp layer path. Names without such an inner path raise
        # not-implemented.
        # KNOWN ISSUE: the final return statement after the if/else is unreachable.
        """Convert MTP layer parameters from MCore to HF format."""
        # For now, assume single MTP layer support
        if "mtp.layers." not in name:
            raise NotImplementedError(f"Invalid MTP parameter name: {name}")

        # Get the MTP layer index
        parts = name.split(".")
        mtp_layer_idx = parts[2]  # mtp.layers.{idx}

        # Direct mappings for MTP-specific components
        direct_name_mapping = {
            f"mtp.layers.{mtp_layer_idx}.enorm.weight": f"model.mtp_layers.{mtp_layer_idx}.token_layernorm.weight",
            f"mtp.layers.{mtp_layer_idx}.hnorm.weight": f"model.mtp_layers.{mtp_layer_idx}.hidden_layernorm.weight",
            f"mtp.layers.{mtp_layer_idx}.eh_proj.weight": f"model.mtp_layers.{mtp_layer_idx}.input_proj.weight",
            f"mtp.layers.{mtp_layer_idx}.final_layernorm.weight": f"model.mtp_layers.{mtp_layer_idx}.final_layernorm.weight",
        }

        if name in direct_name_mapping:
            return [direct_name_mapping[name]]

        _mtp_inner = next((p for p in ("transformer_layer", "mtp_model_layer") if p in name), None)
        if _mtp_inner is not None:
            # Create a proxy name to use with parent class methods
            # Convert mtp.layers.{idx}.<inner>.* to decoder.layers.{idx}.*
            proxy_name = name.replace(
                f"mtp.layers.{mtp_layer_idx}.{_mtp_inner}",
                f"decoder.layers.{mtp_layer_idx}",
            )

            if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
                convert_names = super()._weight_name_mapping_attention(proxy_name)
            elif "mlp" in proxy_name:
                convert_names = super()._weight_name_mapping_mlp(proxy_name)
            else:
                raise NotImplementedError(f"Unsupported transformer component in MTP: {name}")

            # Replace the layer index in converted names to point to mtp_layers
            convert_names = [
                cn.replace(f"model.layers.{mtp_layer_idx}", f"model.mtp_layers.{mtp_layer_idx}")
                for cn in convert_names
            ]
            return convert_names
        else:
            raise NotImplementedError(f"Unsupported MTP parameter name: {name}")
        return convert_names

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        # [FUNCTION DOC] Convert using the base class, then for the embedding-hidden projection weight
        # split the result in two halves along the second axis and concatenate them in swapped order.
        """Swap halves of eh_proj weights before handing off to Megatron-Core."""
        weight = super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
        if mcore_weights_name.endswith("eh_proj.weight"):
            first_half, second_half = weight.chunk(2, dim=1)
            weight = torch.cat([second_half, first_half], dim=1)
        return weight

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        # [FUNCTION DOC] Export: for the embedding-hidden projection weight, first swap the two halves
        # along the second axis (the inverse of the import), then use the base class export.
        """Swap halves back when exporting eh_proj weights to HuggingFace format."""
        if mcore_weights_name.endswith("eh_proj.weight"):
            first_half, second_half = mcore_weights.chunk(2, dim=1)
            mcore_weights = torch.cat([second_half, first_half], dim=1)
        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)
