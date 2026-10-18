# [MODULE DOC]
# mbridge bridge for DeepSeek V4.1, derived from the V4 bridge. The V4.1 checkpoint uses a flatter
# naming scheme (no model prefix; embed, norm and head at top level; attn and ffn instead of
# self_attn and mlp), has no hash-routed layers, an indexer key projection and norm, and engram
# parameters. The mappings are replaced wholesale, and the config builder fills in attributes the
# V4.1 Hugging Face config lacks.
from megatron.core.transformer.enums import AttnBackend

from mbridge.core import register_model

from .deepseek_v4 import DeepseekV4Bridge


@register_model("deepseek_v41")
class DeepseekV41Bridge(DeepseekV4Bridge):
    # [CLASS DOC]
    # Registered under the name deepseek_v41. Class attributes replace (not extend) the V4 mappings,
    # from Megatron parameter names to Hugging Face names with layer number and expert id
    # placeholders: direct mapping (embedding, final norm, output head); attention mapping
    # (including the input layer norm, which in V4 came from the V3 mapping); other mapping
    # (hyper-connection and engram parameters); and the MLP mapping (shared and routed experts,
    # where fused first projections list the gate and up weights separately).
    # KNOWN ISSUE: the replaced direct mapping drops the V4 final hyper-connection head entries, so
    # those parameters have no name mapping in this bridge unless handled elsewhere.
    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "embed.weight",
        "decoder.final_layernorm.weight": "norm.weight",
        "output_layer.weight": "head.weight",
    }

    _ATTENTION_MAPPING = {
        "input_layernorm.weight": ["layers.{layer_number}.attn_norm.weight"],
        "self_attention.linear_q_down_proj.weight": ["layers.{layer_number}.attn.wq_a.weight"],
        "self_attention.q_layernorm.weight": ["layers.{layer_number}.attn.q_norm.weight"],
        "self_attention.linear_q_up_proj.weight": ["layers.{layer_number}.attn.wq_b.weight"],
        "self_attention.linear_kv_proj.weight": ["layers.{layer_number}.attn.wkv.weight"],
        "self_attention.kv_layernorm.weight": ["layers.{layer_number}.attn.kv_norm.weight"],
        "self_attention.linear_o_group_proj": ["layers.{layer_number}.attn.wo_a.weight"],
        "self_attention.linear_proj.weight": ["layers.{layer_number}.attn.wo_b.weight"],
        "self_attention.core_attention.attn_sink": ["layers.{layer_number}.attn.attn_sink"],
        "self_attention.core_attention.compressor.linear_wkv.weight": [
            "layers.{layer_number}.attn.compressor.wkv.weight"
        ],
        "self_attention.core_attention.compressor.linear_wgate.weight": [
            "layers.{layer_number}.attn.compressor.wgate.weight"
        ],
        "self_attention.core_attention.compressor.norm.weight": ["layers.{layer_number}.attn.compressor.norm.weight"],
        "self_attention.core_attention.indexer.linear_wq_b.weight": ["layers.{layer_number}.attn.indexer.wq_b.weight"],
        "self_attention.core_attention.indexer.linear_weights_proj.weight": [
            "layers.{layer_number}.attn.indexer.weights_proj.weight"
        ],
        "self_attention.core_attention.indexer.linear_wk.weight": ["layers.{layer_number}.attn.indexer.wk.weight"],
        "self_attention.core_attention.indexer.k_norm.weight": ["layers.{layer_number}.attn.indexer.k_norm.weight"],
    }

    _OTHER_MAPPING = {
        "self_attention_hyper_connection.mapping_proj.weight": ["layers.{layer_number}.hc_attn_fn"],
        "self_attention_hyper_connection.bias": ["layers.{layer_number}.hc_attn_base"],
        "self_attention_hyper_connection.alpha_pre": ["layers.{layer_number}.hc_attn_scale"],
        "self_attention_hyper_connection.alpha_post": ["layers.{layer_number}.hc_attn_scale"],
        "self_attention_hyper_connection.alpha_res": ["layers.{layer_number}.hc_attn_scale"],
        "mlp_hyper_connection.mapping_proj.weight": ["layers.{layer_number}.hc_ffn_fn"],
        "mlp_hyper_connection.bias": ["layers.{layer_number}.hc_ffn_base"],
        "mlp_hyper_connection.alpha_pre": ["layers.{layer_number}.hc_ffn_scale"],
        "mlp_hyper_connection.alpha_post": ["layers.{layer_number}.hc_ffn_scale"],
        "mlp_hyper_connection.alpha_res": ["layers.{layer_number}.hc_ffn_scale"],
        "v41_engram.linear_wkv.weight": ["layers.{layer_number}.engram.wkv.weight"],
        "v41_engram.q_weight": ["layers.{layer_number}.engram.q_weight"],
        "v41_engram.k_weight": ["layers.{layer_number}.engram.k_weight"],
    }

    _MLP_MAPPING = {
        "pre_mlp_layernorm.weight": ["layers.{layer_number}.ffn_norm.weight"],
        "mlp.router.weight": ["layers.{layer_number}.ffn.gate.weight"],
        "mlp.router.expert_bias": ["layers.{layer_number}.ffn.gate.bias"],
        "mlp.shared_experts.linear_fc1.weight": [
            "layers.{layer_number}.ffn.shared_experts.w1.weight",
            "layers.{layer_number}.ffn.shared_experts.w3.weight",
        ],
        "mlp.shared_experts.linear_fc2.weight": ["layers.{layer_number}.ffn.shared_experts.w2.weight"],
        "mlp.experts.linear_fc1.weight": [
            "layers.{layer_number}.ffn.experts.{expert_id}.w1.weight",
            "layers.{layer_number}.ffn.experts.{expert_id}.w3.weight",
        ],
        "mlp.experts.linear_fc2.weight": ["layers.{layer_number}.ffn.experts.{expert_id}.w2.weight"],
    }

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        # [FUNCTION DOC] Hugging Face names for a Megatron parameter: direct mapping hits return a
        # one-element list; names containing hyper connection or the engram marker use the other
        # mapping; everything else goes to the parent class lookup.
        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]
        if "hyper_connection" in mcore_weights_name or "v41_engram" in mcore_weights_name:
            return self._weight_name_mapping_other(mcore_weights_name)
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _build_config(self):
        # [FUNCTION DOC]
        # Fill in missing Hugging Face config attributes in place (mutates the shared config): zero
        # hash-routed layers; first dense layer count default zero; intermediate size defaults to the
        # mixture-of-experts intermediate size; key-value low-rank size defaults to the head dim;
        # query-key no-position head dim defaults to head dim minus rope head dim; value head dim
        # defaults to the head dim; and if the rope scaling mapping exists without rope theta, a copy
        # of it with the config's rope theta is stored. Then builds the V4 config, resets the backend
        # to automatic, forces zero hash layers, enables the shared-expert activation clamp, and
        # applies the V4.1 specific settings from the models plugin (imported locally here).
        # KNOWN ISSUE: when the rope scaling is none, the parent config builder still reads its rope
        # theta entry and would fail.
        hf = self.hf_config
        hf.n_hash_layers = 0
        hf.first_k_dense_replace = getattr(hf, "first_k_dense_replace", 0)
        hf.intermediate_size = getattr(hf, "intermediate_size", hf.moe_intermediate_size)
        hf.kv_lora_rank = getattr(hf, "kv_lora_rank", hf.head_dim)
        hf.qk_nope_head_dim = getattr(hf, "qk_nope_head_dim", hf.head_dim - hf.qk_rope_head_dim)
        hf.v_head_dim = getattr(hf, "v_head_dim", hf.head_dim)
        if hf.rope_scaling is not None and "rope_theta" not in hf.rope_scaling:
            hf.rope_scaling = dict(hf.rope_scaling, rope_theta=hf.rope_theta)
        config = super()._build_config()
        config.attention_backend = AttnBackend.auto
        config.moe_n_hash_layers = 0
        config.activation_func_clamp_shared_expert = True
        from miles_plugins.models.deepseek_v4_1.deepseek_v4_1 import apply_v41_config

        apply_v41_config(config, hf)
        return config
