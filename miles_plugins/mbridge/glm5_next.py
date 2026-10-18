# [MODULE DOC]
# mbridge bridge for the GLM-5 next architecture, derived from the GLM MoE DSA bridge. Adds kimi
# delta attention (KDA) and key-pool compression parameters, hyper-connections (with their three
# alphas packed into one Hugging Face scale tensor), checkpoint names nested under a language model
# prefix, and config checks. Export of the packed alphas is intentionally unsupported here.
import torch

from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge

from miles_plugins.mbridge.deepseek_v32 import GlmMoeDsaBridge


_GLM5_NEXT_VOCAB_SIZE = 154880

_HC_ALPHA_SLICES = {
    "alpha_pre": slice(0, 1),
    "alpha_post": slice(1, 2),
    "alpha_res": slice(2, 3),
}


@register_model("glm5_next")
class Glm5NextBridge(GlmMoeDsaBridge):
    # [CLASS DOC]
    # Registered under the name glm5_next. The vocabulary size is fixed (154880) by an assertion in
    # the config builder. Class attributes: the KDA mapping (projection, gate, norm and output weights
    # map one to one under the self attention path; the convolution weight maps to three Hugging
    # Face tensors for query, key and value that are concatenated), the key-pool compression mapping,
    # the attention mapping (parent's merged with these), and the hyper-connection mapping (alphas
    # of one block all map to the same packed scale tensor).
    # The module-level table of alpha slices gives each alpha's position in that packed tensor.

    _KDA_MAPPING = {
        f"self_attention.kda.{weight_name}": ["model.layers.{layer_number}.self_attn." + weight_name]
        for weight_name in [
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "b_proj.weight",
            "f_a_proj.weight",
            "f_b_proj.weight",
            "g_a_proj.weight",
            "g_b_proj.weight",
            "A_log",
            "dt_bias",
            "o_norm.weight",
            "o_proj.weight",
        ]
    }
    _KDA_MAPPING["self_attention.kda.conv1d.weight"] = [
        "model.layers.{layer_number}.self_attn.q_conv1d.weight",
        "model.layers.{layer_number}.self_attn.k_conv1d.weight",
        "model.layers.{layer_number}.self_attn.v_conv1d.weight",
    ]

    _KPOOL_MAPPING = {
        "self_attention.index_kpool_compress_gate": [
            "model.layers.{layer_number}.self_attn.indexer.index_kpool_compress_gate"
        ],
        "self_attention.index_kpool_compress_ape": [
            "model.layers.{layer_number}.self_attn.indexer.index_kpool_compress_ape"
        ],
    }

    _ATTENTION_MAPPING = {**GlmMoeDsaBridge._ATTENTION_MAPPING, **_KDA_MAPPING, **_KPOOL_MAPPING}

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

    def __init__(self, hf_config, *args, **kwargs):
        # [FUNCTION DOC] Use the nested text config if the given Hugging Face config has one, else the
        # config itself, and pass it with the remaining arguments to the parent constructor.
        text_config = getattr(hf_config, "text_config", None) or hf_config
        super().__init__(text_config, *args, **kwargs)

    def _get_rope_theta(self):
        # [FUNCTION DOC] HELPER: Rope base as a float: the config's rope theta, else the rope theta
        # entry of its rope parameters mapping. If neither exists: returns 10000.0 when the rope head
        # dim is zero or absent, else raises a value error.
        rope_theta = getattr(self.hf_config, "rope_theta", None)
        if rope_theta is None:
            rope_parameters = getattr(self.hf_config, "rope_parameters", None)
            if isinstance(rope_parameters, dict):
                rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is None:
            if int(getattr(self.hf_config, "qk_rope_head_dim", 0) or 0) == 0:
                return 10000.0
            raise ValueError("GLM-5.3 config must provide rope_theta (directly or via rope_parameters)")
        return float(rope_theta)

    def _get_rope_scaling(self):
        # [FUNCTION DOC] HELPER: The rope parameters mapping if it is a mapping, else the rope scaling
        # attribute (or none), passed through the normalizer.
        rope_scaling = getattr(self.hf_config, "rope_parameters", None)
        if not isinstance(rope_scaling, dict):
            rope_scaling = getattr(self.hf_config, "rope_scaling", None)
        return self._normalize_rope_scaling(rope_scaling)

    def _build_base_config(self, **kwargs):
        # [FUNCTION DOC] Asserts the vocabulary size equals the fixed one and the hyper-connection
        # epsilon (default 1e-6) equals 1e-6 (the value baked into Megatron). Removes the multi-token
        # prediction layer count and loss scaling from the keyword arguments (not supported here),
        # enables hyper-connections with residual streams (default 4) and sinkhorn iterations (default
        # 20) from the config and turns the fused kernel off, then calls the parent helper. Returns the
        # config. Mutates the passed keyword mapping only locally.
        assert self.hf_config.vocab_size == _GLM5_NEXT_VOCAB_SIZE
        hc_eps = getattr(self.hf_config, "hc_eps", 1e-6)
        assert hc_eps == 1e-6, f"Megatron mHC eps constants are 1e-6, config says {hc_eps}"
        kwargs.pop("mtp_num_layers", None)
        kwargs.pop("mtp_loss_scaling_factor", None)
        kwargs.update(
            enable_hyper_connections=True,
            num_residual_streams=getattr(self.hf_config, "hc_mult", 4),
            mhc_sinkhorn_iterations=getattr(self.hf_config, "hc_sinkhorn_iters", 20),
            use_fused_mhc=False,
        )
        return super()._build_base_config(**kwargs)

    @staticmethod
    def _nest_language_model(name: str) -> str:
        # [FUNCTION DOC] HELPER: Static. Rewrites a Hugging Face name starting with the model prefix
        # into the same name under the language model prefix, unless it already is under the language
        # model or visual prefixes; other names are unchanged.
        if name.startswith("model.") and not name.startswith(("model.language_model.", "model.visual.")):
            return "model.language_model." + name[len("model.") :]
        return name

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        # [FUNCTION DOC] Hugging Face names for a Megatron parameter: the parent mapping, or on
        # not-implemented the other (hyper-connection) mapping, with every name moved under the
        # language model prefix.
        try:
            names = super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        except NotImplementedError:
            names = self._weight_name_mapping_other(mcore_weights_name)
        return [self._nest_language_model(n) for n in names]

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        # [FUNCTION DOC]
        # Convert Hugging Face tensors to the Megatron weight. The KDA convolution weight (asserts
        # three inputs) is the three tensors concatenated along the first axis. A hyper-connection
        # alpha (asserts one input) is the matching slice of the flattened packed tensor, cloned. A
        # single float32 input uses the V3 base conversion with the bridge's target dtype temporarily
        # cleared so it keeps full precision; everything else uses the V3 base conversion directly.
        # KNOWN ISSUE: this skips the intermediate parent classes (DSA bridge) and calls the V3 base
        # conversion directly, so the DSA indexer rope reordering of the parent is never applied here;
        # and unlike the V4 bridge the alpha slice is not cast to float32.
        if mcore_weights_name.endswith("self_attention.kda.conv1d.weight"):
            assert len(hf_weights) == 3
            return torch.cat(hf_weights, dim=0).contiguous()
        for alpha_name, alpha_slice in _HC_ALPHA_SLICES.items():
            if mcore_weights_name.endswith(f"_hyper_connection.{alpha_name}"):
                assert len(hf_weights) == 1
                return hf_weights[0].reshape(-1)[alpha_slice].clone()
        if len(hf_weights) == 1 and hf_weights[0].dtype == torch.float32:
            saved_dtype = getattr(self, "dtype", None)
            self.dtype = None
            try:
                return DeepseekV3Bridge._weight_to_mcore_format(self, mcore_weights_name, hf_weights)
            finally:
                self.dtype = saved_dtype
        return DeepseekV3Bridge._weight_to_mcore_format(self, mcore_weights_name, hf_weights)

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        # [FUNCTION DOC] Export one weight. The KDA convolution weight is split into three equal chunks
        # along the first axis (query, key, value) with the three names. Any hyper-connection alpha
        # raises not-implemented, since the packed scale needs all three alphas at once (the message
        # points to the raw converter). Everything else uses the V3 base export directly, skipping the
        # DSA parent.
        if mcore_weights_name.endswith("self_attention.kda.conv1d.weight"):
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            return hf_names, list(mcore_weights.chunk(3, dim=0))
        for alpha_name in _HC_ALPHA_SLICES:
            if mcore_weights_name.endswith(f"_hyper_connection.{alpha_name}"):
                raise NotImplementedError(
                    "hc_*_scale export needs all three alphas at once; use the raw "
                    "megatron_to_hf converter (miles/backends/megatron_utils/megatron_to_hf/glm5_next.py)"
                )
        return DeepseekV3Bridge._weight_to_hf_format(self, mcore_weights_name, mcore_weights)
