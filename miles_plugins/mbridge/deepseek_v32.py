# [MODULE DOC]
# mbridge bridge for DeepSeek V3.2 (sparse attention with a lightning indexer), derived from the V3
# bridge, plus a variant registered for the GLM MoE DSA model type. Adds name mappings for the
# indexer and the renamed attention layer norms, makes the rope fields of the Hugging Face config
# explicit while the base class builds the Megatron config, and permutes indexer weights between the
# Hugging Face channel order and the Miles indexer order when the config says the indexer rope is
# interleaved.
import copy
from contextlib import contextmanager

import torch

from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("deepseek_v32")
class DeepseekV32Bridge(DeepseekV3Bridge):
    # [CLASS DOC]
    # Registered under the name deepseek_v32. The class attribute for the sparse attention mapping
    # maps Megatron parameter names (relative to a layer) to lists of Hugging Face names with a layer
    # number placeholder: both the Miles layout (indexer parameters directly under self attention)
    # and the native Megatron layout (under core attention indexer) are mapped to the same Hugging
    # Face names; the query and key-value layer norms map to the Hugging Face a-layernorms. The
    # attention mapping used by the base class is the V3 mapping with these entries merged in (these
    # win).
    _DSA_ATTENTION_MAPPING = {
        "self_attention.wq_b.weight": ["model.layers.{layer_number}.self_attn.indexer.wq_b.weight"],
        "self_attention.wk.weight": ["model.layers.{layer_number}.self_attn.indexer.wk.weight"],
        "self_attention.weights_proj.weight": ["model.layers.{layer_number}.self_attn.indexer.weights_proj.weight"],
        "self_attention.k_norm.weight": ["model.layers.{layer_number}.self_attn.indexer.k_norm.weight"],
        "self_attention.k_norm.bias": ["model.layers.{layer_number}.self_attn.indexer.k_norm.bias"],
        # Mbridge maps each actual model parameter to HF, so both naming
        # layouts can coexist here without duplicating checkpoint imports.
        "self_attention.core_attention.indexer.linear_wq_b.weight": [
            "model.layers.{layer_number}.self_attn.indexer.wq_b.weight"
        ],
        "self_attention.core_attention.indexer.linear_wk.weight": [
            "model.layers.{layer_number}.self_attn.indexer.wk.weight"
        ],
        "self_attention.core_attention.indexer.linear_weights_proj.weight": [
            "model.layers.{layer_number}.self_attn.indexer.weights_proj.weight"
        ],
        "self_attention.core_attention.indexer.k_norm.weight": [
            "model.layers.{layer_number}.self_attn.indexer.k_norm.weight"
        ],
        "self_attention.core_attention.indexer.k_norm.bias": [
            "model.layers.{layer_number}.self_attn.indexer.k_norm.bias"
        ],
        "self_attention.q_layernorm.weight": ["model.layers.{layer_number}.self_attn.q_a_layernorm.weight"],
        "self_attention.kv_layernorm.weight": ["model.layers.{layer_number}.self_attn.kv_a_layernorm.weight"],
    }
    _ATTENTION_MAPPING = {**DeepseekV3Bridge._ATTENTION_MAPPING, **_DSA_ATTENTION_MAPPING}

    def _get_rope_theta(self):
        # [FUNCTION DOC] HELPER: Rope base frequency as a float: the config's rope theta if present,
        # else the rope theta entry of its rope scaling mapping. Raises a value error if neither
        # exists.
        rope_theta = getattr(self.hf_config, "rope_theta", None)
        if rope_theta is None:
            rope_scaling = getattr(self.hf_config, "rope_scaling", None)
            if isinstance(rope_scaling, dict):
                rope_theta = rope_scaling.get("rope_theta")
        if rope_theta is None:
            raise ValueError("V3.2 rope config must contain rope_theta")
        return float(rope_theta)

    def _normalize_rope_scaling(self, rope_scaling):
        # [FUNCTION DOC] HELPER: Normalize a rope scaling mapping to the shape the base class expects.
        # Returns none for none input or when the scaling type (read from the type key, else the rope
        # type key) is the default. Otherwise returns a copy whose type key is set to the found type
        # (if any) and whose factor, beta fast, beta slow, mscale and mscale all dim entries that are
        # integers are converted to floats. The input is not modified.
        if rope_scaling is None:
            return None
        rope_scaling = dict(rope_scaling)
        rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
        if rope_type == "default":
            return None
        if rope_type is not None:
            rope_scaling["type"] = rope_type
        for key in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
            value = rope_scaling.get(key)
            if isinstance(value, int):
                rope_scaling[key] = float(value)
        return rope_scaling

    def _get_rope_scaling(self):
        # [FUNCTION DOC] HELPER: The config's rope scaling (none if absent), normalized.
        return self._normalize_rope_scaling(getattr(self.hf_config, "rope_scaling", None))

    def _hf_config_with_rope_fields(self):
        # [FUNCTION DOC] HELPER: A shallow copy of the Hugging Face config whose rope theta and rope
        # scaling attributes are set to the values computed by the getters above. The original config
        # is untouched.
        hf_config = copy.copy(self.hf_config)
        hf_config.rope_theta = self._get_rope_theta()
        hf_config.rope_scaling = self._get_rope_scaling()
        return hf_config

    @contextmanager
    def _using_hf_config_with_rope_fields(self):
        # [FUNCTION DOC] HELPER: Context manager that temporarily replaces the bridge's Hugging Face
        # config by the rope-explicit copy and restores the original on exit, even on error.
        original_hf_config = self.hf_config
        self.hf_config = self._hf_config_with_rope_fields()
        try:
            yield
        finally:
            self.hf_config = original_hf_config

    def _build_config(self):
        # [FUNCTION DOC] Build the Megatron transformer config through the base class while the
        # rope-explicit Hugging Face config is in place.
        with self._using_hf_config_with_rope_fields():
            return super()._build_config()

    def _get_gptmodel_args(self) -> dict:
        # [FUNCTION DOC] Build the model constructor arguments through the base class while the
        # rope-explicit Hugging Face config is in place; returns the mapping.
        with self._using_hf_config_with_rope_fields():
            return super()._get_gptmodel_args()

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        # [FUNCTION DOC]
        # Export one Megatron weight to Hugging Face names and tensors. If the config does not set the
        # indexer rope interleave flag, defer entirely to the base class. Otherwise, for the Miles
        # indexer weights (matched by name: query projection b, key projection, key norm weight, key
        # norm bias) look up the Hugging Face names and swap the two halves of the rope channel layout:
        # for the query projection, view rows as groups of 128 channels and exchange the first 64 and
        # last 64 channels within each group; for the others, exchange the first 64 and the remaining
        # entries along the first axis (this assumes length 128 for norms and keys; 64 and 128 are
        # hard-coded). Everything else goes to the base class. Returns the name list and a one-element
        # tensor list for the swapped weights.
        """Apply rope reordering when exporting DSA attention weights to HF format.

        The Miles indexer stores RoPE channels in the last half. Native
        Megatron's core_attention.indexer already uses HF channel order.
        """
        if not bool(getattr(self.hf_config, "indexer_rope_interleave", False)):
            return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

        if "self_attention.wq_b.weight" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            wq_b = mcore_weights
            wq_b = wq_b.view(-1, 128, wq_b.shape[-1])  # hard code 128
            wq_b = torch.cat([wq_b[:, 64:], wq_b[:, :64]], dim=1).view(-1, wq_b.shape[-1])
            return hf_names, [wq_b]
        elif "self_attention.wk.weight" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            wk = mcore_weights
            wk = torch.cat([wk[64:], wk[:64]], dim=0)
            return hf_names, [wk]
        elif "self_attention.k_norm.weight" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            knorm_weight = mcore_weights
            knorm_weight = torch.cat([knorm_weight[64:], knorm_weight[:64]], dim=0)
            return hf_names, [knorm_weight]
        elif "self_attention.k_norm.bias" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            knorm_bias = mcore_weights
            knorm_bias = torch.cat([knorm_bias[64:], knorm_bias[:64]], dim=0)
            return hf_names, [knorm_bias]
        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        # [FUNCTION DOC]
        # Import one weight from Hugging Face tensors into Megatron form. Without the indexer rope
        # interleave flag, defer to the base class. Otherwise the four indexer weights named as in the
        # export direction get the same half swap (it is its own inverse) applied to the first
        # Hugging Face tensor and returned directly (no dtype conversion or tensor parallel
        # splitting from the base class is applied on this path). Other weights use the base class.
        # KNOWN ISSUE: the swapped indexer weights bypass the base class conversion, so they keep the
        # source dtype and are not sharded by the base class logic.
        """Apply inverse rope reordering when importing DSA attention weights from HF format.

        The swap operation is its own inverse: swap the two halves back.
        """
        if not bool(getattr(self.hf_config, "indexer_rope_interleave", False)):
            return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

        if "self_attention.wq_b.weight" in mcore_weights_name:
            wq_b = hf_weights[0]
            wq_b = wq_b.view(-1, 128, wq_b.shape[-1])  # hard code 128
            wq_b = torch.cat([wq_b[:, 64:], wq_b[:, :64]], dim=1).view(-1, wq_b.shape[-1])
            return wq_b
        elif "self_attention.wk.weight" in mcore_weights_name:
            wk = hf_weights[0]
            wk = torch.cat([wk[64:], wk[:64]], dim=0)
            return wk
        elif "self_attention.k_norm.weight" in mcore_weights_name:
            knorm_weight = hf_weights[0]
            knorm_weight = torch.cat([knorm_weight[64:], knorm_weight[:64]], dim=0)
            return knorm_weight
        elif "self_attention.k_norm.bias" in mcore_weights_name:
            knorm_bias = hf_weights[0]
            knorm_bias = torch.cat([knorm_bias[64:], knorm_bias[:64]], dim=0)
            return knorm_bias
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)


@register_model("glm_moe_dsa")
class GlmMoeDsaBridge(DeepseekV32Bridge):
    # [CLASS DOC]
    # Registered under the name glm_moe_dsa. Same as the V3.2 bridge but the rope settings live in
    # the config's rope parameters mapping instead of the rope theta and rope scaling attributes.
    def _get_rope_theta(self):
        # [FUNCTION DOC] HELPER: The rope theta entry of the config's rope parameters (key error if
        # missing).
        return self.hf_config.rope_parameters["rope_theta"]

    def _get_rope_scaling(self):
        # [FUNCTION DOC] HELPER: The config's rope parameters mapping, normalized (so it also
        # contains rope theta and other keys).
        return self._normalize_rope_scaling(self.hf_config.rope_parameters)
