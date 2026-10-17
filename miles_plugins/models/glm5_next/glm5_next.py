# [MODULE DOC] Spec builder for the GLM-5.3 hybrid model. Reads the Hugging Face text config to turn on
# multi-stream hyper-connections (with checked epsilon, unfused kernels), set indexer sizes and pool
# parameters, and decide which layers are full (sparse) attention versus linear (Kimi delta)
# attention. Monkey-patches the hyper-connection constructor to use a reference projection-and-RMS
# routine, and builds Megatron's decoder block spec with per-layer attention specs replaced
# accordingly and a mean output contraction.
import copy

import torch

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer import hyper_connection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.glm5.glm5 import DSASelfAttentionSubmodules
from miles_plugins.models.glm5_next.dsa import Glm5NextDSAAttention
from miles_plugins.models.glm5_next.kda import Glm5NextKDAAttention, _get_text_config

_MHC_EPS = 1e-6


def full_attn_layers(text_config) -> list[int]:
    # [FUNCTION DOC] Returns the zero-based layer indices that use full attention. Priority: (1) the
    # linear-attention config mapping's list of full attention layers, if that is a mapping with a
    # non-null list (sorted, as integers); (2) the config's layer types list, if non-empty: every
    # index whose type is not linear attention; (3) default: layers whose index modulo 4 equals 3,
    # over the number of hidden layers.
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    if isinstance(linear_attn_config, dict) and linear_attn_config.get("full_attn_layers") is not None:
        return sorted(int(i) for i in linear_attn_config["full_attn_layers"])
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types:
        return [i for i, layer_type in enumerate(layer_types) if layer_type != "linear_attention"]
    return [i for i in range(text_config.num_hidden_layers) if i % 4 == 3]


def _apply_glm5_next_config(config, text_config) -> None:
    # [FUNCTION DOC] HELPER: Mutates the Megatron config from the HF text config. Asserts the
    # hyper-connection epsilon (default 1e-6) equals the constant Megatron hard codes. Sets:
    # hyper-connections on, number of residual streams (default 4), Sinkhorn iterations (default 20),
    # fused mHC off; index head count, head dim, top-k, and pool size from the HF config (missing ones
    # raise attribute errors); asserts pool size above one and that the config enables pool
    # compression and always selecting the tail; stores the full-attention layer list on the config.
    hc_eps = getattr(text_config, "hc_eps", _MHC_EPS)
    assert hc_eps == _MHC_EPS, (
        f"GLM-5.3 hc_eps={hc_eps} but Megatron's mHC sinkhorn/compute-h eps are the "
        f"constants _MHC_SINKHORN_EPS=_MHC_COMPUTE_H_EPS={_MHC_EPS}"
    )
    config.enable_hyper_connections = True
    config.num_residual_streams = getattr(text_config, "hc_mult", 4)
    config.mhc_sinkhorn_iterations = getattr(text_config, "hc_sinkhorn_iters", 20)
    config.use_fused_mhc = False

    config.index_num_attention_heads = text_config.index_n_heads
    config.index_head_dim = text_config.index_head_dim
    config.index_topk = text_config.index_topk
    config.index_kpool = text_config.index_kpool
    assert (
        text_config.index_kpool > 1
        and getattr(text_config, "index_kpool_compress", False)
        and getattr(text_config, "index_kpool_always_select_tail", False)
    ), "GLM-5.3 kpool indexer expects index_kpool>1 with compress and always_select_tail"

    config.glm5_next_full_attn_layers = full_attn_layers(text_config)


class Glm5NextMeanOutputContraction(torch.nn.Module):
    # [CLASS DOC] HELPER: Output-stage contraction of the residual streams: delegates to the
    # hyper-connection module's static contraction with the configured stream count.
    """Plain mean over the residual streams, with no learned head weights."""

    def __init__(self, config):
        # [FUNCTION DOC] HELPER: Records the number of residual streams from the config.
        super().__init__()
        self.num_streams = config.num_residual_streams

    def forward(self, hidden_states):
        # [FUNCTION DOC] HELPER: Collapses the stream-expanded hidden states back to a single stream
        # using Megatron's output-contract routine (documented here as a plain mean over streams).
        return HyperConnectionModule.output_contract(hidden_states, self.num_streams)


def _reference_proj_rms(x, weight, eps):
    # [FUNCTION DOC] HELPER: Reference (unfused) computation for hyper-connection mapping: returns
    # the matrix product of x with the transposed weight, and the reciprocal root of the mean square of
    # x over the last dim plus epsilon (kept with a trailing singleton dim).
    proj = torch.matmul(x, weight.t())
    r = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return proj, r


def _patch_reference_proj_rms() -> None:
    # [FUNCTION DOC] HELPER: Idempotent (marker attribute on the hyper-connection module) global
    # monkey patch of the hyper-connection constructor: after the original constructor, asserts the
    # config does not use the fused mHC path, records the layer-norm epsilon, and installs a
    # torch-compiled version of the reference projection-and-RMS routine as the projection op.
    if getattr(hyper_connection, "_glm5_next_reference_proj_rms_patched", False):
        return

    original_init = HyperConnectionModule.__init__

    def _init_with_reference_proj_rms(self, config, layer_number):
        # [FUNCTION DOC] HELPER: Replacement constructor described above; wraps the original.
        original_init(self, config, layer_number)
        assert not config.use_fused_mhc, "GLM-5.3 mHC requires the native (unfused) proj_rms path"
        self.norm_eps = config.layernorm_epsilon
        self._proj_rms_op = torch.compile(_reference_proj_rms)

    HyperConnectionModule.__init__ = _init_with_reference_proj_rms
    hyper_connection._glm5_next_reference_proj_rms_patched = True


def get_glm5_next_spec(args, config, vp_stage=None):
    # [FUNCTION DOC] Entry point. Loads the HF config from the checkpoint path, takes its text config
    # (nested one if present), applies config changes, installs the global patch. Builds Megatron's
    # decoder block spec (Transformer Engine; with the virtual pipeline stage if given) and sets its
    # hyper-connection head contraction to the mean contraction. Asserts no explicit pipeline layout
    # is used. For each layer to build on this stage (global index = local index plus the stage's
    # layer offset): a deep copy of the layer spec gets the sparse-attention spec (causal mask, top-k
    # backend from args, TE-provider submodules, identity layernorms) if the global index is a
    # full-attention layer, otherwise the Kimi delta attention spec taking the args. Returns the
    # block spec.
    hf_config = load_hf_config(args.hf_checkpoint)
    text_config = _get_text_config(hf_config)

    _apply_glm5_next_config(config, text_config)
    _patch_reference_proj_rms()

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
    transformer_layer_spec.hc_head_contraction = ModuleSpec(module=Glm5NextMeanOutputContraction)

    assert config.pipeline_model_parallel_layout is None, "pipeline_model_parallel_layout is not supported"

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    backend = TESpecProvider()
    dsa_module_spec = ModuleSpec(
        module=Glm5NextDSAAttention,
        params={
            "attn_mask_type": AttnMaskType.causal,
            "topk_backend": args.miles_dsa_topk_backend,
        },
        submodules=DSASelfAttentionSubmodules(
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_layer_norm_linear(),
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=backend.column_parallel_layer_norm_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
            linear_v_up_proj=IdentityOp,
            wq_b=backend.linear(),
            wk=backend.linear(),
            k_norm=backend.layer_norm(),
            weights_proj=backend.linear(),
        ),
    )

    dsa_layers = set(config.glm5_next_full_attn_layers)
    for layer_id in range(num_layers_to_build):
        layer_spec = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
        if (layer_id + offset) in dsa_layers:
            layer_spec.submodules.self_attention = dsa_module_spec
        else:
            layer_spec.submodules.self_attention = ModuleSpec(
                module=Glm5NextKDAAttention,
                params={"args": args},
            )
        transformer_layer_spec.layer_specs[layer_id] = layer_spec

    return transformer_layer_spec
