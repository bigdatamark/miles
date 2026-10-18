# [MODULE DOC]
# mbridge bridge for Qwen3-Next (hybrid gated linear attention and full attention mixture-of-experts),
# derived from the Qwen2 MoE bridge. Adds attention name mappings for the linear attention layers,
# the fused query-gate-key-value merge for full attention layers, linear attention convolution
# layout conversion between flat (checkpoint) and group-major (model), multi-token prediction (mtp)
# name mapping, and the Megatron config.
import torch

from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge
from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import (
    gdn_heads,
    qkv_flat_to_group_major,
    qkv_group_major_to_flat,
)
from miles_plugins.mbridge.linear_attn import LinearAttentionBridgeMixin

_GDN_CONV = "self_attention.linear_attn.conv1d.weight"


@register_model("qwen3_next")
class Qwen3NextBridge(LinearAttentionBridgeMixin, Qwen2MoEBridge):
    # [CLASS DOC]
    # Registered under the name qwen3_next. The attention mapping merges the Qwen2 MoE mapping with
    # generated entries for the input norm, linear attention parameters (decay log, convolution,
    # time-step bias, combined input projections, norm, output projection) and gated full attention
    # parameters, then overrides the fused qkv entries (layer norm and three-way projection).
    _ATTENTION_MAPPING = (
        Qwen2MoEBridge._ATTENTION_MAPPING
        | {
            f"self_attention.{weight_name}": ["model.layers.{layer_number}." + weight_name]
            for weight_name in [
                "input_layernorm.weight",
                # linear attn
                "linear_attn.A_log",
                "linear_attn.conv1d.weight",
                "linear_attn.dt_bias",
                "linear_attn.in_proj_ba.weight",
                "linear_attn.in_proj_qkvz.weight",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
                # gated attn
                "self_attn.k_norm.weight",
                "self_attn.k_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.q_proj.weight",
                "self_attn.v_proj.weight",
            ]
        }
        | {
            "self_attention.linear_qkv.layer_norm_weight": ["model.layers.{layer_number}.input_layernorm.weight"],
            "self_attention.linear_qkv.weight": [
                "model.layers.{layer_number}.self_attn.q_proj.weight",
                "model.layers.{layer_number}.self_attn.k_proj.weight",
                "model.layers.{layer_number}.self_attn.v_proj.weight",
            ],
        }
    )

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> tuple[list[str], list[torch.Tensor]]:
        # [FUNCTION DOC]
        # Convert Hugging Face tensors to one Megatron weight. The linear attention convolution weight
        # is converted from flat to group-major layout using head counts from the config. For the fused
        # qkv linear (not its norm; asserts three inputs): the query tensor (which holds query and gate
        # rows) is viewed per key-value group and its query and gate halves swapped into per-group
        # order, key and value are viewed per group, and the three are concatenated per group and
        # flattened to a matrix (or a vector for bias). The group count is derived from the actual
        # query tensor so tensor-parallel slices work. Everything else uses the base class.
        # KNOWN ISSUE: the return annotation says a tuple of names and tensors but a single tensor is
        # returned.
        if mcore_weights_name.endswith(_GDN_CONV):
            return qkv_flat_to_group_major(hf_weights[0], gdn_heads(self.hf_config))

        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            # merge qkv
            assert len(hf_weights) == 3
            num_key_value_heads = self.hf_config.num_key_value_heads
            hidden_dim = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads
            num_querys_per_group = num_attention_heads // self.hf_config.num_key_value_heads
            head_dim = getattr(self.hf_config, "head_dim", hidden_dim // num_attention_heads)
            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights
            # q k v might be tp split
            real_num_key_value_heads = q.shape[0] // (2 * group_dim)
            q = (
                q.view(
                    [
                        real_num_key_value_heads,
                        num_querys_per_group,
                        2,
                        head_dim,
                        -1,
                    ]
                )
                .transpose(1, 2)
                .flatten(1, 3)
            )
            k = k.view([real_num_key_value_heads, head_dim, -1])
            v = v.view([real_num_key_value_heads, head_dim, -1])
            out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]

            qgkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
            return qgkv

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        # [FUNCTION DOC] Export one weight with the parent; for the linear attention convolution weight,
        # convert the single returned tensor from group-major back to flat layout. Returns names and
        # tensors.
        names, weights = super()._weight_to_hf_format(mcore_weights_name, mcore_weights)
        if mcore_weights_name.endswith(_GDN_CONV):
            weights = [qkv_group_major_to_flat(weights[0], gdn_heads(self.hf_config))]
        return names, weights

    def _build_config(self):
        # [FUNCTION DOC] Build the Megatron config via the base helper: mtp layer count if the config has
        # the next-token-prediction attribute; no cpu initialization; expert ffn size, router bias update
        # rate 0.001, top-k, expert count and aux loss coefficient from the config; no load balancing
        # (default for reinforcement learning); grouped gemm; softmax score without softmax before
        # top-k; fusion options on; query-key layer norm, attention output gate and shared expert gate
        # on. Returns the config.
        mtp_args = {}
        if hasattr(self.hf_config, "num_nextn_predict_layers"):
            mtp_args["mtp_num_layers"] = self.hf_config.num_nextn_predict_layers

        return self._build_base_config(
            use_cpu_initialization=False,
            # MoE specific
            moe_ffn_hidden_size=self.hf_config.moe_intermediate_size,
            moe_router_bias_update_rate=0.001,
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.num_experts,
            moe_aux_loss_coeff=self.hf_config.router_aux_loss_coef,
            # moe_router_load_balancing_type="aux_loss",
            moe_router_load_balancing_type="none",  # default None for RL
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # Qwen specific
            moe_router_pre_softmax=False,
            qk_layernorm=True,
            # Qwen3 Next specific
            attention_output_gate=True,
            moe_shared_expert_gate=True,
            **mtp_args,
        )

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        # [FUNCTION DOC] Hugging Face names for a Megatron parameter: names containing the mtp marker
        # use the mtp conversion, all others the parent mapping.
        if "mtp" in mcore_weights_name:
            return self._convert_mtp_param(mcore_weights_name)
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _convert_mtp_param(self, name: str) -> list[str]:
        # [FUNCTION DOC]
        # Hugging Face names for an mtp parameter. The layer index is the third dotted segment (no check
        # that the name has the mtp layers path). Four direct entries (embedding norm, hidden norm,
        # projection, final norm) map to top-level mtp names without a layer index and return a
        # one-element list. Otherwise, if the name has an inner layer path (either supported spelling),
        # it is rewritten to a regular decoder layer path with the same index and mapped via the
        # attention mapping (self attention or input norm names) or the MLP mapping (mlp or pre-mlp
        # norm names), else raises not-implemented; the regular layer prefix in the results is replaced
        # by the mtp layer prefix. Anything else raises not-implemented.
        """Convert MTP layer parameters from MCore to HF format.

        Megatron MTP names:  mtp.layers.{idx}.{enorm,hnorm,eh_proj,final_layernorm,...}
        HF/SGLang MTP names: mtp.{fc,pre_fc_norm_embedding,pre_fc_norm_hidden,norm,...}
                             mtp.layers.{idx}.{decoder layer components}
        """
        parts = name.split(".")
        mtp_layer_idx = parts[2]  # mtp.layers.{idx}

        direct_mappings = {
            f"mtp.layers.{mtp_layer_idx}.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            f"mtp.layers.{mtp_layer_idx}.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            f"mtp.layers.{mtp_layer_idx}.eh_proj.weight": "mtp.fc.weight",
            f"mtp.layers.{mtp_layer_idx}.final_layernorm.weight": "mtp.norm.weight",
        }

        if name in direct_mappings:
            return [direct_mappings[name]]

        _mtp_inner = next((p for p in ("transformer_layer", "mtp_model_layer") if p in name), None)
        if _mtp_inner is not None:
            proxy_name = name.replace(
                f"mtp.layers.{mtp_layer_idx}.{_mtp_inner}",
                f"decoder.layers.{mtp_layer_idx}",
            )

            if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
                convert_names = super()._weight_name_mapping_attention(proxy_name)
            elif "mlp" in proxy_name or "pre_mlp_layernorm" in proxy_name:
                convert_names = super()._weight_name_mapping_mlp(proxy_name)
            else:
                raise NotImplementedError(f"Unsupported transformer component in MTP: {name}")

            return [cn.replace(f"model.layers.{mtp_layer_idx}", f"mtp.layers.{mtp_layer_idx}") for cn in convert_names]

        raise NotImplementedError(f"Unsupported MTP parameter name: {name}")
