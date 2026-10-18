# [MODULE DOC]
# mbridge bridge for GLM-4.7 MoE Lite (for example GLM-4.7-Flash), which uses DeepSeek V3 style
# multi-head latent attention. Derived from the V3 bridge but builds its own Megatron config because
# the Hugging Face runtime config stores rope theta inside the rope scaling mapping and describes
# which layers are dense through a per-layer type list. Also handles the single multi-token
# prediction (mtp) layer stored after the 47 regular layers, including tied embedding and head
# weights that appear twice in the checkpoint.
import torch
from megatron.core.transformer.enums import AttnBackend

from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("glm4_moe_lite")
class GLM4MoELiteBridge(DeepseekV3Bridge):
    # [CLASS DOC]
    # Registered under the name glm4_moe_lite. The expected-layers class attribute (47) is the only
    # layer count the mtp handling supports.
    """Bridge for GLM-4.7 MoE Lite models (e.g. GLM-4.7-Flash).

    Here "runtime config" means the config object returned by
    AutoConfig.from_pretrained(..., trust_remote_code=True).

    Glm4MoeLiteConfig differs from DeepseekV3Config:
      - runtime config exposes rope_theta via rope_scaling['rope_theta'] (no top-level rope_theta).
      - runtime config uses mlp_layer_types (raw JSON may still include first_k_dense_replace).
    """

    # NOTE: We cannot use DeepseekV3Bridge._build_config() directly because it
    # reads hf_config.rope_theta, which is not present in Glm4MoeLiteConfig.
    # GLM-4.7 MoE Lite stores this value under hf_config.rope_scaling['rope_theta'].
    _EXPECTED_NUM_LAYERS = 47

    @property
    def rope_theta(self):
        # [FUNCTION DOC] Property: rope base frequency read from the rope theta entry of the config's
        # rope scaling mapping (key error or type error if missing).
        return self.hf_config.rope_scaling["rope_theta"]

    @property
    def moe_layer_freq(self):
        # [FUNCTION DOC] Property: per-layer list with 0 for layers whose configured type is dense and
        # 1 for all others (mixture-of-experts layers).
        return [0 if t == "dense" else 1 for t in self.hf_config.mlp_layer_types]

    def _build_config(self):
        # [FUNCTION DOC]
        # Build the Megatron config through the base helper. Rope settings start from defaults (beta
        # fast 32, beta slow 1, factor 1, mscale 1.0, mscale all dim 1.0, original max positions
        # 4096, type rope) and are overridden by the config's rope scaling mapping if present.
        # Settings: fused attention backend; norm epsilon and ffn size from the config; query-key layer
        # norm on; mixture-of-experts: expert ffn size, all-to-all dispatcher, bias update rate 0.001
        # with expert bias, top-k and expert count from the config, shared expert size equal to the
        # expert ffn size times the number of shared experts (default 1), aux loss coefficient (default
        # 0.001), no load balancing type, shared expert overlap, grouped gemm, sigmoid score with
        # softmax before top-k, top-k scaling factor (default 1.0), per-layer frequency list;
        # multi-head latent attention sizes from the config; rotary base and scaling from the rope
        # settings; float32 router; fusion options on. The original max positions are stored under a
        # different config key depending on the Megatron core version (new key for version at least
        # 0.14). If the config declares next-token-prediction layers, sets the mtp layer count to that
        # and the mtp loss scaling to 0.1. Returns the config.
        # KNOWN ISSUE: the Megatron version check compares version strings lexicographically, so e.g.
        # a version like 0.9 compares greater than 0.14 and a version like 0.100 or 1.0.0 may be
        # misordered.
        hf_config = self.hf_config
        mla_rope_config = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "rope",
        }
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling is not None:
            mla_rope_config.update(rope_scaling)

        base_config = {
            "attention_backend": AttnBackend.fused,
            "layernorm_epsilon": hf_config.rms_norm_eps,
            "ffn_hidden_size": hf_config.intermediate_size,
            "qk_layernorm": True,
            # moe
            "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_bias_update_rate": 0.001,
            "moe_router_enable_expert_bias": True,
            "moe_router_topk": hf_config.num_experts_per_tok,
            "num_moe_experts": hf_config.n_routed_experts,
            "moe_shared_expert_intermediate_size": hf_config.moe_intermediate_size
            * getattr(hf_config, "n_shared_experts", 1),
            "moe_aux_loss_coeff": getattr(hf_config, "aux_loss_alpha", 0.001),
            "moe_router_load_balancing_type": "none",
            "moe_shared_expert_overlap": True,
            "moe_grouped_gemm": True,
            "moe_router_score_function": "sigmoid",
            "moe_router_pre_softmax": True,
            "moe_router_topk_scaling_factor": getattr(hf_config, "routed_scaling_factor", 1.0),
            "moe_layer_freq": self.moe_layer_freq,
            # MLA
            "q_lora_rank": hf_config.q_lora_rank,
            "kv_lora_rank": hf_config.kv_lora_rank,
            "qk_head_dim": hf_config.qk_nope_head_dim,
            "qk_pos_emb_head_dim": hf_config.qk_rope_head_dim,
            "v_head_dim": hf_config.v_head_dim,
            "rotary_base": self.rope_theta,
            "rotary_scaling_factor": mla_rope_config["factor"],
            "rope_type": mla_rope_config["type"],
            "mscale": mla_rope_config["mscale"],
            "mscale_all_dim": mla_rope_config["mscale_all_dim"],
            "beta_fast": mla_rope_config["beta_fast"],
            "beta_slow": mla_rope_config["beta_slow"],
            # mcore 0.12 moe
            "moe_router_dtype": "fp32",
            "disable_bf16_reduced_precision_matmul": True,
            # other
            "persist_layer_norm": True,
            "bias_activation_fusion": True,
            "bias_dropout_fusion": True,
        }

        import megatron.core

        megatron_version = getattr(megatron.core, "__version__", "0.0")
        if megatron_version >= "0.14":
            base_config["original_max_position_embeddings"] = mla_rope_config["original_max_position_embeddings"]
        else:
            base_config["max_position_embeddings"] = mla_rope_config["original_max_position_embeddings"]

        mtp_args = {}
        num_nextn_predict_layers = getattr(hf_config, "num_nextn_predict_layers", None)
        if num_nextn_predict_layers is not None:
            mtp_args["mtp_num_layers"] = num_nextn_predict_layers
            mtp_args["mtp_loss_scaling_factor"] = 0.1
        base_config.update(mtp_args)

        return self._build_base_config(**base_config)

    def _get_gptmodel_args(self) -> dict:
        # [FUNCTION DOC] Model constructor arguments: vocabulary size, maximum sequence length (the
        # config's max position embeddings), rope position embedding type and the rotary base.
        return dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=self.rope_theta,
        )

    def _convert_mtp_param(self, name: str) -> tuple[list[str]]:
        # [FUNCTION DOC]
        # Hugging Face names for a Megatron mtp parameter. Asserts exactly one mtp layer and the
        # expected 47 regular layers; the mtp layer is stored as layer number 47. The four direct
        # entries (embedding norm, hidden norm, projection, final norm) map to that layer's weights and
        # are returned as a one-element list. Otherwise the inner layer path (either supported inner
        # module spelling; asserts one is found) is rewritten to a regular decoder layer path and mapped
        # via the attention mapping (for self attention or input layer norm names) or the MLP mapping
        # (for mlp names). Raises not-implemented for anything else.
        # KNOWN ISSUE: the return annotation says a tuple of lists but a plain list is returned.
        assert self.config.mtp_num_layers == 1, "only support one mtp layer for now"
        assert (
            self.config.num_layers == self._EXPECTED_NUM_LAYERS
        ), f"glm4_moe_lite only supports {self._EXPECTED_NUM_LAYERS} layers for now"

        mtp_layer_id = self.config.num_layers
        direct_name_mapping = {
            "mtp.layers.0.enorm.weight": f"model.layers.{mtp_layer_id}.enorm.weight",
            "mtp.layers.0.hnorm.weight": f"model.layers.{mtp_layer_id}.hnorm.weight",
            "mtp.layers.0.eh_proj.weight": f"model.layers.{mtp_layer_id}.eh_proj.weight",
            "mtp.layers.0.final_layernorm.weight": f"model.layers.{mtp_layer_id}.shared_head.norm.weight",
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]]

        _mtp_inner = next((p for p in ("transformer_layer", "mtp_model_layer") if f"mtp.layers.0.{p}" in name), None)
        assert _mtp_inner is not None, "mtp not found"
        proxy_name = name.replace(f"mtp.layers.0.{_mtp_inner}", f"decoder.layers.{mtp_layer_id}")
        if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
            return self._weight_name_mapping_attention(proxy_name)
        if "mlp" in proxy_name:
            return self._weight_name_mapping_mlp(proxy_name)
        raise NotImplementedError(f"Unsupported parameter name: {name}")

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        # [FUNCTION DOC] Export one weight to Hugging Face. When the model has one mtp layer (asserts the
        # expected 47 layers), the embedding and output head weights are written under two names each
        # (the regular name and the mtp layer's copy) with the same tensor repeated; everything else
        # goes to the base class.
        if self.config.mtp_num_layers == 1:
            assert (
                self.config.num_layers == self._EXPECTED_NUM_LAYERS
            ), f"glm4_moe_lite only supports {self._EXPECTED_NUM_LAYERS} layers for now"
            mtp_layer_id = self.config.num_layers
            shared_state_dict_mapping = {
                "embedding.word_embeddings.weight": [
                    "model.embed_tokens.weight",
                    f"model.layers.{mtp_layer_id}.embed_tokens.weight",
                ],
                "output_layer.weight": [
                    "lm_head.weight",
                    f"model.layers.{mtp_layer_id}.shared_head.head.weight",
                ],
            }
            if mcore_weights_name in shared_state_dict_mapping:
                hf_names = shared_state_dict_mapping[mcore_weights_name]
                return hf_names, [mcore_weights] * len(hf_names)
        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)
