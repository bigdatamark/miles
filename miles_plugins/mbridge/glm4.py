# [MODULE DOC]
# mbridge bridge for dense GLM-4 models: maps Hugging Face parameter names to Megatron names and
# configures the Megatron transformer with the extra post-attention and post-MLP layer norms that
# GLM-4 has.
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

from mbridge.core import LLMBridge, register_model


@register_model("glm4")
class GLM4Bridge(LLMBridge):
    # [CLASS DOC]
    # Registered under the name glm4. Class attributes are mappings from Megatron parameter names
    # (relative to the model or a layer) to Hugging Face names with a layer number placeholder:
    # direct (embedding, final norm, output head), attention (fused query-key-value maps to the three
    # separate projections; the input layer norm is fused into the qkv linear) and MLP (fused gate-up
    # projection; the post attention layer norm is fused into the first MLP linear).
    # KNOWN ISSUE: the class docstring and some comments describe Qwen2, copied from another bridge.
    """
    Bridge implementation for Qwen2 models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for Qwen2 models, handling the conversion between
    Hugging Face Qwen2 format and Megatron-Core.
    """

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": ["model.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.linear_qkv.layer_norm_weight": ["model.layers.{layer_number}.input_layernorm.weight"],
        "self_attention.q_layernorm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
        "self_attention.k_layernorm.weight": ["model.layers.{layer_number}.self_attn.k_norm.weight"],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    }
    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
    }

    def _build_config(self):
        # [FUNCTION DOC] Build the Megatron transformer config through the base helper with: query-key-
        # value bias on, no query-key layer norm, extra layer norms after the attention and after the
        # MLP, and interleaved rotary embeddings. Returns the config.
        # KNOWN ISSUE: the docstring below describes Qwen2 models, not GLM-4.
        """
        Build the configuration for Qwen2 models.

        Configures Qwen2-specific parameters such as QKV bias settings and
        layer normalization options.

        Returns:
            TransformerConfig: Configuration object for Qwen2 models
        """
        return self._build_base_config(
            # qwen2
            add_qkv_bias=True,
            qk_layernorm=False,
            post_mlp_layernorm=True,
            post_self_attn_layernorm=True,
            rotary_interleaved=True,
        )

    def _get_transformer_layer_spec(self):
        # [FUNCTION DOC] Layer spec from the transformer-engine based gpt layer spec helper with the
        # post-attention and post-MLP layer norms enabled. Returns the spec.
        # KNOWN ISSUE: the docstring mentions an assertion on RMSNorm that this code does not perform.
        """
        Gets the transformer layer specification.

        Creates and returns a specification for the transformer layers based on
        the current configuration.

        Returns:
            TransformerLayerSpec: Specification for transformer layers

        Raises:
            AssertionError: If normalization is not RMSNorm
        """
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            post_self_attn_layernorm=True,
            post_mlp_layernorm=True,
        )
        return transformer_layer_spec

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        # [FUNCTION DOC] Hugging Face names for a Megatron parameter. Asserts the name is not an extra
        # state entry. Order of checks: direct mapping; names containing the post attention norm or post
        # MLP norm marker map to the layer's corresponding norm weight (layer number is the third
        # dotted segment of the Megatron name); names with self attention use the attention mapping;
        # names with mlp use the MLP mapping; otherwise raises not-implemented.
        """
        Map MCore weight names to Hugging Face weight names.

        Args:
            mcore_weights_name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names
        """
        assert "_extra_state" not in mcore_weights_name, "extra_state should not be loaded"

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if "post_self_attn_layernorm" in mcore_weights_name:
            layer_number = mcore_weights_name.split(".")[2]
            return [f"model.layers.{layer_number}.post_self_attn_layernorm.weight"]
        elif "post_mlp_layernorm" in mcore_weights_name:
            layer_number = mcore_weights_name.split(".")[2]
            return [f"model.layers.{layer_number}.post_mlp_layernorm.weight"]
        elif "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")
