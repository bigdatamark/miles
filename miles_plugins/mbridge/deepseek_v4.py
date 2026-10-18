# [MODULE DOC]
# mbridge bridge for DeepSeek V4, derived from the V3 bridge. V4 differs by: low-rank query and joint
# key-value projections with different names, grouped output projection, attention sink, compressor
# and indexer sub-modules, hyper-connections whose three scalar alphas are packed in one Hugging Face
# tensor, hash-routed experts, and several parameters that must stay in float32. This module remaps
# parameter names accordingly and builds the Megatron config with the V4 attention variant settings.
import torch
from megatron.core.transformer.enums import AttnBackend

from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("deepseek_v4")
class DeepseekV4Bridge(DeepseekV3Bridge):
    # [CLASS DOC]
    # Registered under the name deepseek_v4. Class attributes are mappings from Megatron parameter
    # names (relative to a layer or the model) to Hugging Face names (with a layer number placeholder).
    # The attention mapping starts as a copy of the V3 mapping with the fused layer norm entries of
    # the up projections removed, then V4 entries are added. The other mapping covers the
    # hyper-connection parameters. The MLP mapping is V3's plus the hash router table. The direct
    # mapping is V3's plus the final hyper-connection head parameters. The alpha segment table gives
    # each alpha's position within the packed scale tensor.
    _ATTENTION_MAPPING = DeepseekV3Bridge._ATTENTION_MAPPING.copy()

    _ATTENTION_MAPPING.pop("self_attention.linear_q_up_proj.layer_norm_weight", None)
    _ATTENTION_MAPPING.pop("self_attention.linear_kv_up_proj.layer_norm_weight", None)

    _ATTENTION_MAPPING.update(
        {
            "self_attention.linear_q_down_proj.weight": ["model.layers.{layer_number}.self_attn.wq_a.weight"],
            "self_attention.q_layernorm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
            "self_attention.linear_q_up_proj.weight": ["model.layers.{layer_number}.self_attn.wq_b.weight"],
            "self_attention.linear_kv_proj.weight": ["model.layers.{layer_number}.self_attn.wkv.weight"],
            "self_attention.kv_layernorm.weight": ["model.layers.{layer_number}.self_attn.kv_norm.weight"],
            "self_attention.linear_o_group_proj": ["model.layers.{layer_number}.self_attn.wo_a.weight"],
            "self_attention.linear_proj.weight": ["model.layers.{layer_number}.self_attn.wo_b.weight"],
            "self_attention.core_attention.attn_sink": ["model.layers.{layer_number}.self_attn.attn_sink"],
            "self_attention.core_attention.compressor.ape": ["model.layers.{layer_number}.self_attn.compressor.ape"],
            "self_attention.core_attention.compressor.linear_wkv.weight": [
                "model.layers.{layer_number}.self_attn.compressor.wkv.weight"
            ],
            "self_attention.core_attention.compressor.linear_wgate.weight": [
                "model.layers.{layer_number}.self_attn.compressor.wgate.weight"
            ],
            "self_attention.core_attention.compressor.norm.weight": [
                "model.layers.{layer_number}.self_attn.compressor.norm.weight"
            ],
            "self_attention.core_attention.indexer.linear_wq_b.weight": [
                "model.layers.{layer_number}.self_attn.indexer.wq_b.weight"
            ],
            "self_attention.core_attention.indexer.linear_weights_proj.weight": [
                "model.layers.{layer_number}.self_attn.indexer.weights_proj.weight"
            ],
            "self_attention.core_attention.indexer.compressor.ape": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.ape"
            ],
            "self_attention.core_attention.indexer.compressor.linear_wkv.weight": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.wkv.weight"
            ],
            "self_attention.core_attention.indexer.compressor.linear_wgate.weight": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.wgate.weight"
            ],
            "self_attention.core_attention.indexer.compressor.norm.weight": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.norm.weight"
            ],
        }
    )

    # The HF checkpoint keeps the three alphas packed as one hc_*_scale tensor, ordered
    # [pre, post, res] — the same order both implementations feed their kernels.
    _OTHER_MAPPING = {
        "self_attention_hyper_connection.mapping_proj.weight": ["model.layers.{layer_number}.hc_attn_fn"],
        "self_attention_hyper_connection.bias": ["model.layers.{layer_number}.hc_attn_base"],
        "self_attention_hyper_connection.alpha_pre": ["model.layers.{layer_number}.hc_attn_scale"],
        "self_attention_hyper_connection.alpha_post": ["model.layers.{layer_number}.hc_attn_scale"],
        "self_attention_hyper_connection.alpha_res": ["model.layers.{layer_number}.hc_attn_scale"],
        "mlp_hyper_connection.mapping_proj.weight": ["model.layers.{layer_number}.hc_ffn_fn"],
        "mlp_hyper_connection.bias": ["model.layers.{layer_number}.hc_ffn_base"],
        "mlp_hyper_connection.alpha_pre": ["model.layers.{layer_number}.hc_ffn_scale"],
        "mlp_hyper_connection.alpha_post": ["model.layers.{layer_number}.hc_ffn_scale"],
        "mlp_hyper_connection.alpha_res": ["model.layers.{layer_number}.hc_ffn_scale"],
    }

    _MLP_MAPPING = DeepseekV3Bridge._MLP_MAPPING.copy()
    _MLP_MAPPING.update(
        {
            "mlp.router.tid2eid": ["model.layers.{layer_number}.mlp.topk.tid2eid"],
        }
    )

    _DIRECT_MAPPING = DeepseekV3Bridge._DIRECT_MAPPING.copy()
    _DIRECT_MAPPING.update(
        {
            "decoder.hc_head_fn": "model.hc_head_fn",
            "decoder.hc_head_base": "model.hc_head_base",
            "decoder.hc_head_scale": "model.hc_head_scale",
        }
    )

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        # [FUNCTION DOC] Hugging Face names for a Megatron parameter name: try the base class
        # mapping; if it raises not-implemented (name not found), fall back to the other mapping
        # (hyper-connection parameters).
        try:
            return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        except NotImplementedError:
            return self._weight_name_mapping_other(mcore_weights_name)

    _ALPHA_SEGMENTS = {"alpha_pre": 0, "alpha_post": 1, "alpha_res": 2}

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        # [FUNCTION DOC]
        # Convert Hugging Face tensors to the Megatron weight. For a hyper-connection alpha parameter
        # (name ends with alpha pre, post or res) the first tensor is flattened and the one element at
        # that alpha's position is returned as a one-element float32 tensor. For a single float32
        # source tensor, the base class conversion is run with the bridge's target dtype temporarily
        # cleared (restored afterwards even on error), so the weight is not rounded down to the lower
        # precision before reaching float32 model parameters. All other cases use the base class.
        # The alpha segments are matched by name suffix, tried in table order.
        # The checkpoint packs the hyper-connection alphas as one [pre, post, res] tensor;
        # the model stores one parameter per segment, matching the native module.
        for suffix, segment in self._ALPHA_SEGMENTS.items():
            if mcore_weights_name.endswith(suffix):
                return hf_weights[0].reshape(-1)[segment : segment + 1].float()

        # V4 keeps several params in fp32 (attn_sink, compressor.ape, and the
        # hyper-connection hc_* params, all marked _keep_fp32). The base bridge
        # downcasts every loaded weight to self.dtype (bf16), which would silently
        # round these to bf16 before they reach the fp32 mcore params. Run the base
        # reshaping but skip the dtype downcast for fp32-source weights.
        if len(hf_weights) == 1 and hf_weights[0].dtype == torch.float32:
            saved_dtype = getattr(self, "dtype", None)
            self.dtype = None
            try:
                return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
            finally:
                self.dtype = saved_dtype
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _build_config(self):
        # [FUNCTION DOC]
        # Build the Megatron config. First copies the rope theta from the rope scaling mapping into
        # the Hugging Face config in place (key error or type error if the scaling is missing or
        # lacks it). Then takes the base config and sets: automatic attention backend; router score
        # function from the config's scoring function; the V4 attention variant; indexer heads (64),
        # head dim (128) and top-k (512); number of residual streams (hyper-connection multiplier,
        # 4) and sinkhorn iterations (20); compression ratios (none) and compression rope base
        # (160000); when the swiglu limit is positive, disables bias-activation fusion and sets the
        # activation clamp value to the limit; output groups (8), output low-rank size (1024), number
        # of hash-routed layers (3) and window size (128). Defaults in parentheses apply when the
        # config lacks the attribute. Returns the config.
        self.hf_config.rope_theta = self.hf_config.rope_scaling["rope_theta"]
        config = super()._build_config()

        config.attention_backend = AttnBackend.auto
        config.moe_router_score_function = self.hf_config.scoring_func

        config.experimental_attention_variant = "dsv4"
        config.dsa_indexer_n_heads = getattr(self.hf_config, "index_n_heads", 64)
        config.dsa_indexer_head_dim = getattr(self.hf_config, "index_head_dim", 128)
        config.dsa_indexer_topk = getattr(self.hf_config, "index_topk", 512)

        config.num_residual_streams = getattr(self.hf_config, "hc_mult", 4)
        config.mhc_sinkhorn_iterations = getattr(self.hf_config, "hc_sinkhorn_iters", 20)

        config.csa_compress_ratios = getattr(self.hf_config, "compress_ratios", None)
        config.csa_compress_rotary_base = getattr(self.hf_config, "compress_rope_theta", 160000)

        swiglu_limit = getattr(self.hf_config, "swiglu_limit", 0.0)
        if swiglu_limit > 0:
            config.bias_activation_fusion = False
            config.activation_func_clamp_value = swiglu_limit

        config.o_groups = getattr(self.hf_config, "o_groups", 8)
        config.o_lora_rank = getattr(self.hf_config, "o_lora_rank", 1024)
        config.moe_n_hash_layers = getattr(self.hf_config, "n_hash_layers", 3)
        config.csa_window_size = getattr(self.hf_config, "window_size", 128)

        return config
