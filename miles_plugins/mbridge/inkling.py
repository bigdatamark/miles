# [MODULE DOC]
# mbridge bridge for the Inkling multimodal model (text decoder only), derived from the Qwen2 MoE
# bridge. The Hugging Face checkpoint stores the language model under a nested prefix, fuses
# query/key/value/relative projections, stores all experts of a layer as stacked 3-d tensors with
# interleaved gate and up rows, and has short convolutions and global scale parameters. This module
# supplies the name mappings, the tensor reshaping for import, and a tensor-parallel split that keeps
# per-rank query/key/value/relative blocks together.
import re

import torch

from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge


def _deinterleave_w13(w13: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Convert a fused gate-up weight whose rows alternate gate, up, gate, up
    # into the concatenated layout: all even rows (gate) followed by all odd rows (up), along the
    # first axis. Returns a new tensor.
    gate = w13[0::2]
    up = w13[1::2]
    return torch.cat([gate, up], dim=0)


@register_model("inkling_mm_model")
class InklingBridge(Qwen2MoEBridge):
    # [CLASS DOC]
    # Registered under the name inkling_mm_model. Class attributes: direct mapping (embedding,
    # embedding norm, final norm, output head under the language model prefix); attention mapping (the
    # fused qkv weight takes four Hugging Face tensors: query, key, value and relative; norms, short
    # convolutions and the relative projection); MLP mapping (router, expert and dense weights, shared
    # experts, scales; the router weight and the shared gate share one Hugging Face tensor, split on
    # import); and the config mapping from Megatron config fields to Hugging Face fields (a pair
    # gives field name and default).
    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.llm.embed.weight",
        "embedding.embed_norm.weight": "model.llm.embed_norm.weight",
        "decoder.final_layernorm.weight": "model.llm.norm.weight",
        "output_layer.weight": "model.llm.unembed.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_qkv.layer_norm_weight": ["model.llm.layers.{layer_number}.attn_norm.weight"],
        "self_attention.linear_qkv.weight": [
            "model.llm.layers.{layer_number}.attn.wq_du.weight",
            "model.llm.layers.{layer_number}.attn.wk_dv.weight",
            "model.llm.layers.{layer_number}.attn.wv_dv.weight",
            "model.llm.layers.{layer_number}.attn.wr_du.weight",
        ],
        "self_attention.linear_proj.weight": ["model.llm.layers.{layer_number}.attn.wo_ud.weight"],
        "self_attention.q_norm.weight": ["model.llm.layers.{layer_number}.attn.q_norm.weight"],
        "self_attention.k_norm.weight": ["model.llm.layers.{layer_number}.attn.k_norm.weight"],
        "self_attention.rel_proj": ["model.llm.layers.{layer_number}.attn.rel_logits_proj.proj"],
        "self_attention.k_sconv.weight": ["model.llm.layers.{layer_number}.attn.k_sconv.weight"],
        "self_attention.v_sconv.weight": ["model.llm.layers.{layer_number}.attn.v_sconv.weight"],
        "self_attention.attn_sconv.weight": ["model.llm.layers.{layer_number}.attn_sconv.weight"],
    }

    _MLP_MAPPING = {
        "pre_mlp_layernorm.weight": ["model.llm.layers.{layer_number}.mlp_norm.weight"],
        "mlp.mlp_sconv.weight": ["model.llm.layers.{layer_number}.mlp_sconv.weight"],
        "mlp.router.weight": ["model.llm.layers.{layer_number}.mlp.gate.weight"],
        "mlp.router.shared_gate": ["model.llm.layers.{layer_number}.mlp.gate.weight"],
        "mlp.router.global_scale": ["model.llm.layers.{layer_number}.mlp.gate.global_scale"],
        "mlp.router.expert_bias": ["model.llm.layers.{layer_number}.mlp.gate.bias"],
        "mlp.experts.linear_fc1": ["model.llm.layers.{layer_number}.mlp.experts.w13_weight"],
        "mlp.experts.linear_fc2": ["model.llm.layers.{layer_number}.mlp.experts.w2_weight"],
        "mlp.linear_fc1.layer_norm_weight": ["model.llm.layers.{layer_number}.mlp_norm.weight"],
        "mlp.linear_fc1.weight": ["model.llm.layers.{layer_number}.mlp.w13_dn.weight"],
        "mlp.linear_fc2.weight": ["model.llm.layers.{layer_number}.mlp.w2_md.weight"],
        "mlp.global_scale": ["model.llm.layers.{layer_number}.mlp.global_scale"],
        "mlp.shared_experts.experts": [
            "model.llm.layers.{layer_number}.mlp.shared_experts.shared_w13_weight",
            "model.llm.layers.{layer_number}.mlp.shared_experts.shared_w2_weight",
        ],
    }

    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "ffn_hidden_size": "intermediate_size",
        "attention_dropout": ("attention_dropout", 0.0),
        "layernorm_epsilon": "rms_norm_eps",
        "hidden_dropout": ("hidden_dropout", 0.0),
        "kv_channels": ("head_dim", None),
    }

    def _get_text_config(self):
        # [FUNCTION DOC] HELPER: The nested text config if the Hugging Face config has one, else the
        # config itself.
        if hasattr(self.hf_config, "text_config"):
            return self.hf_config.text_config
        return self.hf_config

    def _is_tied_word_embeddings(self) -> bool:
        # [FUNCTION DOC] Always false: the embedding and output head are separate weights.
        return False

    def _adjust_mapping_for_shared_weights(self):
        # [FUNCTION DOC] Override that disables the base class adjustment for tied weights; does nothing.
        pass

    def _get_hf_shared_weight_keys(self):
        # [FUNCTION DOC] Override reporting that no Hugging Face weights are shared; returns an empty
        # list.
        return []

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        # [FUNCTION DOC] Hugging Face names for an MLP-side Megatron parameter. The layer number is the
        # third dotted segment. Per-expert shared expert weights (name with a shared experts expert
        # index and fc1 or fc2) map to the shared first-projection or second-projection stacked tensor
        # of that layer. Otherwise the first keyword of the MLP mapping contained in the name selects
        # the formatted names (mapping order matters because matching is by substring). Raises
        # not-implemented if none match.
        layer_number = name.split(".")[2]
        m = re.search(r"mlp\.shared_experts\.experts\.(\d+)\.(linear_fc1|linear_fc2)\.weight", name)
        if m is not None:
            which = m.group(2)
            key = (
                "model.llm.layers.{layer_number}.mlp.shared_experts.shared_w13_weight"
                if which == "linear_fc1"
                else "model.llm.layers.{layer_number}.mlp.shared_experts.shared_w2_weight"
            )
            return [key.format(layer_number=layer_number)]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        # [FUNCTION DOC]
        # Convert Hugging Face tensors to the Megatron weight, by name, checked in this order. Fused
        # qkv weight (not its norm): asserts four inputs, casts to the target dtype if set, and
        # concatenates them along the first axis. Router weight: asserts one input and takes the first
        # n-routed-experts rows (cast to target dtype). Shared gate: takes the next n-shared-experts rows.
        # Any global scale: reshaped to one element. Router expert bias: unchanged, made contiguous.
        # Routed expert first or second projection given a stacked 3-d tensor: selects the expert by its
        # global index (expert-parallel rank times local experts, plus the local index parsed from
        # the text after the last weight word in the name); the first projection is de-interleaved.
        # Shared expert per-expert weights given a 3-d tensor: picks the expert by index from the
        # name; first projection de-interleaved. Dense MLP first projection of a 2-d tensor: cast and
        # de-interleaved; second projection of a 2-d tensor: cast only. Everything else uses the base
        # class. All returned tensors are contiguous.
        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            assert len(hf_weights) == 4, f"qkvr expects 4 HF tensors, got {len(hf_weights)}"
            if getattr(self, "dtype", None) is not None:
                hf_weights = [w.to(self.dtype) if w.dtype != self.dtype else w for w in hf_weights]
            return torch.cat(hf_weights, dim=0).contiguous()

        if mcore_weights_name.endswith("mlp.router.weight"):
            assert len(hf_weights) == 1
            tc = self._get_text_config()
            nr = tc.n_routed_experts
            w = hf_weights[0]
            if getattr(self, "dtype", None) is not None and w.dtype != self.dtype:
                w = w.to(self.dtype)
            return w[:nr].contiguous()
        if mcore_weights_name.endswith("mlp.router.shared_gate"):
            assert len(hf_weights) == 1
            tc = self._get_text_config()
            nr = tc.n_routed_experts
            ns = tc.n_shared_experts
            w = hf_weights[0]
            if getattr(self, "dtype", None) is not None and w.dtype != self.dtype:
                w = w.to(self.dtype)
            return w[nr : nr + ns].contiguous()
        if mcore_weights_name.endswith("global_scale"):
            assert len(hf_weights) == 1
            return hf_weights[0].reshape(1).contiguous()
        if mcore_weights_name.endswith("mlp.router.expert_bias"):
            assert len(hf_weights) == 1
            return hf_weights[0].contiguous()

        def _global_eid(name):
            # [FUNCTION DOC] HELPER: Global expert index for a local expert parameter name: the
            # expert-parallel rank times the experts per rank plus the integer found after the last
            # occurrence of the word weight in the name.
            nr = self._get_text_config().n_routed_experts
            return self.mpu.ep_rank * (nr // self.mpu.ep_size) + int(name.split("weight")[-1])

        if "mlp.experts.linear_fc1" in mcore_weights_name and len(hf_weights) == 1 and hf_weights[0].dim() == 3:
            return _deinterleave_w13(hf_weights[0][_global_eid(mcore_weights_name)]).contiguous()
        if "mlp.experts.linear_fc2" in mcore_weights_name and len(hf_weights) == 1 and hf_weights[0].dim() == 3:
            return hf_weights[0][_global_eid(mcore_weights_name)].contiguous()

        sm = re.search(r"mlp\.shared_experts\.experts\.(\d+)\.(linear_fc1|linear_fc2)\.weight", mcore_weights_name)
        if sm is not None and len(hf_weights) == 1 and hf_weights[0].dim() == 3:
            e = int(sm.group(1))
            which = sm.group(2)
            w = hf_weights[0][e]
            if which == "linear_fc1":
                w = _deinterleave_w13(w)
            return w.contiguous()

        if mcore_weights_name.endswith("mlp.linear_fc1.weight") and len(hf_weights) == 1 and hf_weights[0].dim() == 2:
            w = hf_weights[0]
            if getattr(self, "dtype", None) is not None and w.dtype != self.dtype:
                w = w.to(self.dtype)
            return _deinterleave_w13(w).contiguous()
        if mcore_weights_name.endswith("mlp.linear_fc2.weight") and len(hf_weights) == 1 and hf_weights[0].dim() == 2:
            w = hf_weights[0]
            if getattr(self, "dtype", None) is not None and w.dtype != self.dtype:
                w = w.to(self.dtype)
            return w.contiguous()

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_split_across_tp(
        self,
        mcore_weights_name: str,
        mcore_weights: torch.Tensor,
        param: torch.Tensor,
        tp_split_size: int,
    ) -> list[torch.Tensor]:
        # [FUNCTION DOC]
        # Split a weight across tensor-parallel ranks. For the fused qkv weight (not its norm) with more
        # than one rank: reads the head count, head dim and relative dim (default 16) from the text
        # config, derives the query, key/value and relative row counts, asserts the row arithmetic is
        # consistent and that head counts divide by the rank count, splits rows into query, key, value
        # and relative parts, chunks each part across ranks and rebuilds each rank's block as the
        # concatenation of its query, key, value and relative chunks. Returns one contiguous tensor per
        # rank. Everything else uses the base class.
        """Split fused linear_qkv into per-rank [q_i|k_i|v_i|r_i] blocks (base chunk(tp) scrambles them)."""
        if (
            tp_split_size > 1
            and "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            tc = self._get_text_config()
            nh = int(tc.num_attention_heads)
            hd = int(tc.head_dim)
            d_rel = int(getattr(tc, "d_rel", 16) or 16)
            q_rows = nh * hd
            r_rows = nh * d_rel
            total = mcore_weights.shape[0]
            kv_rows = (total - q_rows - r_rows) // 2
            nkv = kv_rows // hd if hd else 0
            assert (
                kv_rows > 0
                and (total - q_rows - r_rows) % 2 == 0
                and nkv * hd == kv_rows
                and q_rows + 2 * kv_rows + r_rows == total
            ), (
                f"qkv split {mcore_weights_name}: total={total} nh={nh} hd={hd} d_rel={d_rel} "
                f"q={q_rows} kv={kv_rows} r={r_rows} nkv={nkv}"
            )
            assert (
                nh % tp_split_size == 0 and nkv % tp_split_size == 0
            ), f"qkv tp={tp_split_size} must divide nh={nh} and nkv={nkv} for {mcore_weights_name}"
            q, k, v, r = mcore_weights.split([q_rows, kv_rows, kv_rows, r_rows], dim=0)
            qs, ks, vs, rs = (
                q.chunk(tp_split_size),
                k.chunk(tp_split_size),
                v.chunk(tp_split_size),
                r.chunk(tp_split_size),
            )
            return [torch.cat([qs[i], ks[i], vs[i], rs[i]], dim=0).contiguous() for i in range(tp_split_size)]
        return super()._weight_split_across_tp(mcore_weights_name, mcore_weights, param, tp_split_size)

    def _build_config(self):
        # [FUNCTION DOC] Build the Megatron config through the base helper, reading the nested text
        # config when present: no cpu initialization; expert count, top-k, expert ffn size and shared
        # expert size from the text config (shared size equals the same intermediate size); sigmoid router
        # score with expert bias and softmax before top-k; sequence aux-loss balancing with coefficient
        # zero; grouped gemm; all-to-all dispatcher; query-key layer norm on; no qkv or linear bias.
        # Returns the config.
        tc = self._get_text_config()
        return self._build_base_config(
            text_config_key="text_config" if hasattr(self.hf_config, "text_config") else None,
            use_cpu_initialization=False,
            num_moe_experts=tc.n_routed_experts,
            moe_router_topk=tc.num_experts_per_tok,
            moe_ffn_hidden_size=tc.intermediate_size,
            moe_shared_expert_intermediate_size=tc.intermediate_size,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_pre_softmax=True,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            qk_layernorm=True,
            add_qkv_bias=False,
            add_bias_linear=False,
        )
