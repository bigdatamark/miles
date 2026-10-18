# [MODULE DOC] Assembly of the Inkling model on Megatron. Builds a transformer config from the
# Hugging Face text config (sigmoid router with expert bias, grouped GEMM, shared experts, RMS norm,
# gated MLP, no rotary or learned position embedding), attaches an "inkling" extras object, builds
# per-layer specs (MoE layers and leading dense layers) and the block spec, defines a GPT model
# subclass that patches embedding norm, final-layernorm dtype handling, logits scaling and global-scale
# freezing, and provides the spec entry point plus the text and multimodal model providers.
# Environment variables (read when the extras object is built): MILES_INKLING_ATTN_BACKEND (flex, te
# or fa4; default flex), MILES_INKLING_SCONV_IMPL (triton or torch), MILES_INKLING_SCONV_PACKED ("1"
# enables), MILES_INKLING_FREEZE_GLOBAL_SCALE (all, router or none; default all).
"""Inkling model assembly: TransformerConfig construction, layer/block specs, and the
GPTModel subclass + provider that miles' custom-model-provider path loads."""

from __future__ import annotations

import logging
import os
from functools import partial

import torch
import transformer_engine.pytorch as te
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.inkling.layers import (
    InklingDenseMLP,
    InklingMoELayer,
    InklingRouter,
    InklingSelfAttention,
    InklingSharedExperts,
)

logger = logging.getLogger(__name__)


class InklingExtra:
    # [CLASS DOC] Plain attribute bag of Inkling-specific settings copied from the Hugging Face text
    # config mapping (attention head/kv/head-dim for global and sliding-window layers, window size,
    # relative feature dim and extent, local layer ids as a set, short-conv settings, expert counts,
    # routing scale, sizes, norm epsilon, dense layer boundary and sizes, optional logits width
    # multiplier, route norm, global scale usage) plus debug knobs from the environment. Also used as a
    # scratch area by layers (segment lengths, shared expert weights are set on it at run time).
    def __init__(self, t: dict):
        # [FUNCTION DOC] Required keys raise key errors when missing. Defaults: relative extent 1024,
        # dense layer index 0, dense intermediate size equal to the intermediate size, optional fields
        # none, global scale usage false. Attention backend default flex, short-conv implementation
        # default triton, packed short-conv off, global-scale freezing default "all".
        self.num_attention_heads = t["num_attention_heads"]
        self.num_key_value_heads = t["num_key_value_heads"]
        self.head_dim = t["head_dim"]
        self.swa_num_attention_heads = t["swa_num_attention_heads"]
        self.swa_num_key_value_heads = t["swa_num_key_value_heads"]
        self.swa_head_dim = t["swa_head_dim"]
        self.sliding_window_size = t["sliding_window_size"]
        self.d_rel = t["d_rel"]
        self.rel_extent = t.get("rel_extent", 1024)
        self.local_layer_ids = set(t["local_layer_ids"])
        self.sconv_kernel_size = t["sconv_kernel_size"]
        self.use_sconv = t["use_sconv"]
        self.use_embed_norm = t["use_embed_norm"]
        self.n_routed_experts = t["n_routed_experts"]
        self.num_experts_per_tok = t["num_experts_per_tok"]
        self.n_shared_experts = t["n_shared_experts"]
        self.route_scale = t["route_scale"]
        self.intermediate_size = t["intermediate_size"]
        self.hidden_size = t["hidden_size"]
        self.num_hidden_layers = t["num_hidden_layers"]
        self.vocab_size = t["vocab_size"]
        self.rms_norm_eps = t["rms_norm_eps"]
        self.dense_mlp_idx = int(t.get("dense_mlp_idx", 0))
        self.dense_intermediate_size = int(t.get("dense_intermediate_size", t["intermediate_size"]))
        self.logits_mup_width_multiplier = t.get("logits_mup_width_multiplier", None)
        self.route_norm = t.get("route_norm", None)
        self.use_global_scale = t.get("use_global_scale", False)
        # Debug knobs: defaults are the production recipe; MILES_INKLING_* env vars override.
        self.attn_backend = os.environ.get("MILES_INKLING_ATTN_BACKEND", "flex")  # flex | te | fa4
        self.sconv_impl = os.environ.get("MILES_INKLING_SCONV_IMPL", "triton")  # triton | torch
        self.sconv_packed = os.environ.get("MILES_INKLING_SCONV_PACKED", "0") == "1"
        self.freeze_global_scale = os.environ.get("MILES_INKLING_FREEZE_GLOBAL_SCALE", "all")  # all | router | none


def build_inkling_config(
    text_cfg: dict,
    tp=1,
    ep=1,
    pp=1,
    bf16=True,
    sp=False,
    etp=1,
    cp=1,
    varlen=True,
    permute_fusion=False,
    fp32_residual=False,
    pp_first_stage_layers=None,
    pp_last_stage_layers=None,
) -> TransformerConfig:
    # [FUNCTION DOC] Builds a Megatron transformer config: dimensions from the text config (kv channels
    # from head dim, grouped query count from kv heads, dense FFN size from the dense intermediate
    # size); parallel sizes and first/last pipeline stage layer counts from arguments; sequence
    # parallel, variable sequence lengths, fp32 residual and MoE permute fusion flags; MoE with routed
    # expert count, top-k, MoE FFN size equal to the intermediate size, shared expert size equal to it
    # when there are shared experts; sigmoid score function with expert bias (update rate zero),
    # pre-softmax, seq-aux-loss balancing with zero coefficient, float32 router, grouped GEMM, all-to-all
    # dispatcher; no bias; RMS norm with the epsilon; QK layernorm flag on; gated linear unit;
    # dtype bfloat16 or float32 per the bf16 flag for params and pipeline. Then attaches the extras
    # object, turns on float32 activation and combine in MoE, sets per-layer MoE frequency as dense
    # (zero) for the first dense-index layers and MoE for the rest, and marks heterogeneous
    # distributed checkpointing (the misspelled attribute name is as in the code).
    inter = text_cfg["intermediate_size"]
    ns = text_cfg["n_shared_experts"]
    denseI = int(text_cfg.get("dense_intermediate_size", inter))
    cfg = TransformerConfig(
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_layers=text_cfg["num_hidden_layers"],
        hidden_size=text_cfg["hidden_size"],
        num_attention_heads=text_cfg["num_attention_heads"],
        num_query_groups=text_cfg["num_key_value_heads"],
        kv_channels=text_cfg["head_dim"],
        ffn_hidden_size=denseI,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        num_layers_in_first_pipeline_stage=pp_first_stage_layers,
        num_layers_in_last_pipeline_stage=pp_last_stage_layers,
        expert_model_parallel_size=ep,
        sequence_parallel=sp,
        expert_tensor_parallel_size=etp,
        context_parallel_size=cp,
        variable_seq_lengths=varlen,
        fp32_residual_connection=fp32_residual,
        moe_permute_fusion=permute_fusion,
        num_moe_experts=text_cfg["n_routed_experts"],
        moe_router_topk=text_cfg["num_experts_per_tok"],
        moe_ffn_hidden_size=inter,
        moe_shared_expert_intermediate_size=inter if ns > 0 else None,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=0.0,
        moe_router_bias_update_rate=0.0,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        add_bias_linear=False,
        normalization="RMSNorm",
        layernorm_epsilon=text_cfg["rms_norm_eps"],
        qk_layernorm=True,
        bf16=bf16,
        params_dtype=torch.bfloat16 if bf16 else torch.float32,
        gated_linear_unit=True,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
    )
    cfg.inkling = InklingExtra(text_cfg)
    cfg.moe_activation_in_fp32 = True
    cfg.moe_combine_in_fp32 = True
    _didx = cfg.inkling.dense_mlp_idx
    cfg.moe_layer_freq = [0] * _didx + [1] * (cfg.num_layers - _didx)
    cfg.hetereogenous_dist_checkpoint = True
    return cfg


def get_inkling_layer_spec(config) -> ModuleSpec:
    # [FUNCTION DOC] Starts from Megatron's Transformer Engine layer spec for the config's expert count
    # and grouped GEMM flag (QK layernorm off in the base spec since attention does its own). Replaces
    # the self-attention with the Inkling attention (causal mask) reusing the base submodules; replaces
    # the router with the Inkling router, the shared experts with the Inkling shared experts (reusing
    # their submodules), and the MLP with the Inkling MoE layer. Returns the layer spec.
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm, qk_layernorm=False
    )
    attn = spec.submodules.self_attention
    spec.submodules.self_attention = ModuleSpec(
        module=InklingSelfAttention, params={"attn_mask_type": AttnMaskType.causal}, submodules=attn.submodules
    )
    moe_subs = spec.submodules.mlp.keywords["submodules"]
    moe_subs.router = InklingRouter
    moe_subs.shared_experts = partial(InklingSharedExperts, submodules=moe_subs.shared_experts.keywords["submodules"])
    spec.submodules.mlp = partial(InklingMoELayer, submodules=moe_subs)
    return spec


def get_inkling_dense_layer_spec(config) -> ModuleSpec:
    # [FUNCTION DOC] Like the MoE layer spec but with no experts: Inkling attention (causal) and the
    # Inkling dense MLP wrapped as an MLP submodule factory. The config argument is unused.
    spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=False)
    attn = spec.submodules.self_attention
    spec.submodules.self_attention = ModuleSpec(
        module=InklingSelfAttention, params={"attn_mask_type": AttnMaskType.causal}, submodules=attn.submodules
    )
    spec.submodules.mlp = partial(
        InklingDenseMLP.as_mlp_submodule, submodules=spec.submodules.mlp.keywords["submodules"]
    )
    return spec


def get_inkling_block_spec(config, vp_stage=None):
    # [FUNCTION DOC] Gets the base block submodules from Megatron for this stage using the MoE layer
    # spec. If the dense layer boundary is zero or less, returns it. Otherwise, using the stage's
    # global layer offset, builds a per-local-layer list choosing the dense spec for global indices
    # below the boundary and the MoE spec elsewhere, and returns block submodules with that list and
    # the base final layer norm.
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, _get_block_submodules
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    base = _get_block_submodules(config, get_inkling_layer_spec(config), vp_stage)
    dense_idx = getattr(config.inkling, "dense_mlp_idx", 0)
    if dense_idx <= 0:
        return base
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    local = [
        (get_inkling_dense_layer_spec(config) if (offset + i) < dense_idx else get_inkling_layer_spec(config))
        for i in range(len(base.layer_specs))
    ]
    return TransformerBlockSubmodules(layer_specs=local, layer_norm=base.layer_norm)


def get_inkling_spec(args, config, vp_stage=None):
    # [FUNCTION DOC] Reads the text sub-config from config.json in the checkpoint directory (file
    # I/O; the file handle is not closed explicitly), sets the extras object on the given config, forces
    # float32 router dtype, zero router bias update rate, no top-k scaling factor, and float32 MoE
    # activation and combine. Returns the MoE layer spec for every layer.
    # KNOWN ISSUE: unlike the provider path, this returns a single layer spec and ignores the
    # dense-layer boundary and the virtual pipeline stage, so leading dense layers are not handled here.
    """--spec entry for the miles standard provider path."""
    import json

    text_cfg = json.load(open(f"{args.hf_checkpoint}/config.json"))["text_config"]
    config.inkling = InklingExtra(text_cfg)
    config.moe_router_dtype = "fp32"
    config.moe_router_bias_update_rate = 0.0
    config.moe_router_topk_scaling_factor = None
    config.moe_activation_in_fp32 = True
    config.moe_combine_in_fp32 = True
    return get_inkling_layer_spec(config)


class InklingGPTModel(GPTModel):
    # [CLASS DOC] GPT model with Inkling tweaks applied in the constructor by patching forwards of
    # submodules: embedding output norm (first stage, if configured), cast of the final-layernorm input
    # back to the parameter dtype when residuals are float32 (last stage), division of the head input by
    # the logits width multiplier (last stage, if configured), and freezing of global-scale parameters.
    def __init__(self, *args, **kw):
        # [FUNCTION DOC] Builds the base model, then: on the first stage with the embed-norm flag,
        # adds an RMS norm to the embedding and wraps its forward to normalize the output (with float32
        # residuals: manual float32 RMS norm, rounded through bfloat16, returned as float32). On the last
        # stage with float32 residuals and a final layernorm present, wraps it to cast input to the
        # parameter dtype. On the last stage with a width multiplier, wraps the output layer to divide
        # its input by it. Finally freezes global scales.
        # KNOWN ISSUE: the float32-residual embedding path rounds through bfloat16 regardless of the
        # configured parameter dtype.
        super().__init__(*args, **kw)
        if getattr(self, "pre_process", False) and getattr(self.config.inkling, "use_embed_norm", False):
            emb = self.embedding
            emb.embed_norm = te.RMSNorm(
                self.config.hidden_size, eps=self.config.inkling.rms_norm_eps, params_dtype=self.config.params_dtype
            )
            _orig_emb_forward = emb.forward

            _fp32res = bool(getattr(self.config, "fp32_residual_connection", False))

            def _emb_forward(*a, _orig=_orig_emb_forward, _norm=emb.embed_norm, _fp32=_fp32res, **k):
                # [FUNCTION DOC] HELPER: Calls the original embedding forward, then normalizes the
                # result (see enclosing constructor for the float32-residual variant).
                out = _orig(*a, **k)
                if _fp32:
                    h = out.float()
                    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + getattr(_norm, "eps", 1e-6))
                    return (h * _norm.weight.float()).bfloat16().float()
                return _norm(out)

            emb.forward = _emb_forward
        if getattr(self, "post_process", False) and bool(getattr(self.config, "fp32_residual_connection", False)):
            fln = getattr(self.decoder, "final_layernorm", None)
            if fln is not None:
                _orig_fln = fln.forward
                _pdt = self.config.params_dtype

                def _fln_forward(x, *a, _orig=_orig_fln, _dt=_pdt, **k):
                    # [FUNCTION DOC] HELPER: Casts the input to the parameter dtype if different,
                    # then calls the original final layernorm forward.
                    return _orig(x.to(_dt) if x.dtype != _dt else x, *a, **k)

                fln.forward = _fln_forward
        mup = getattr(self.config.inkling, "logits_mup_width_multiplier", None)
        if getattr(self, "post_process", False) and mup:
            _ol = self.output_layer
            _orig_ol_forward = _ol.forward
            _mup = float(mup)

            def _ol_forward(x, *a, _orig=_orig_ol_forward, _m=_mup, **k):
                # [FUNCTION DOC] HELPER: Divides the input by the width multiplier, then calls the
                # original output layer forward.
                return _orig(x / _m, *a, **k)

            _ol.forward = _ol_forward
        self._freeze_global_scale()

    def _freeze_global_scale(self):
        # [FUNCTION DOC] HELPER: Mode none does nothing. Otherwise, for each parameter whose last name
        # component is global_scale, if the mode is all, or router and the name contains the router
        # global scale path, and it currently requires gradients, disables gradients and counts it.
        # Logs the count at info level when positive.
        """Freeze per-layer global_scale params per config.inkling.freeze_global_scale
        (all | router | none; router = the MoE gate scale only)."""
        mode = getattr(self.config.inkling, "freeze_global_scale", "all")
        if mode == "none":
            return
        n = 0
        for name, p in self.named_parameters():
            if name.rsplit(".", 1)[-1] != "global_scale":
                continue
            if mode == "all" or (mode == "router" and "router.global_scale" in name):
                if p.requires_grad:
                    p.requires_grad = False
                    n += 1
        if n:
            logger.info(
                f"inkling freeze_global_scale={mode}: froze {n} global_scale params "
                f"(requires_grad=False; excluded from optimizer + grad norm)"
            )


def inkling_model_provider(pre_process=True, post_process=True, vp_stage=None, *, mm_towers=False):
    # [FUNCTION DOC] Reads the global training args. With context parallel size above one, asserts the
    # all-gather CP layout flag is set (zigzag unsupported). Loads the text config from the checkpoint,
    # builds the transformer config from arguments (parallel sizes, bf16, sequence parallel, expert
    # tensor parallel size default 1, variable sequence lengths default true, MoE permute fusion, fp32
    # residual, first/last pipeline stage layer counts), then the Inkling GPT model with the block spec
    # for this stage, vocabulary size from the text config, max positions from args, no position
    # embedding, untied embeddings and output. If mm_towers, wires the multimodal towers from the
    # checkpoint (frozen). Returns the model.
    import json

    from megatron.training import get_args

    args = get_args()
    if getattr(args, "context_parallel_size", 1) > 1:
        assert getattr(args, "allgather_cp", False), "Inkling CP requires --allgather-cp (zigzag CP not supported)"
    text_cfg = json.load(open(f"{args.hf_checkpoint}/config.json"))["text_config"]
    config = build_inkling_config(
        text_cfg,
        tp=args.tensor_model_parallel_size,
        ep=args.expert_model_parallel_size,
        pp=args.pipeline_model_parallel_size,
        bf16=args.bf16,
        sp=args.sequence_parallel,
        etp=getattr(args, "expert_tensor_parallel_size", 1) or 1,
        cp=getattr(args, "context_parallel_size", 1) or 1,
        varlen=getattr(args, "variable_seq_lengths", True),
        permute_fusion=getattr(args, "moe_permute_fusion", False),
        fp32_residual=getattr(args, "fp32_residual_connection", False),
        pp_first_stage_layers=getattr(args, "decoder_first_pipeline_num_layers", None),
        pp_last_stage_layers=getattr(args, "decoder_last_pipeline_num_layers", None),
    )
    model = InklingGPTModel(
        config=config,
        transformer_layer_spec=get_inkling_block_spec(config, vp_stage=vp_stage),
        vocab_size=text_cfg["vocab_size"],
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
        vp_stage=vp_stage,
    )
    if mm_towers:
        from miles_plugins.models.inkling.mm_towers import wire_mm_towers

        wire_mm_towers(model, args.hf_checkpoint)
    return model


def inkling_mm_model_provider(pre_process=True, post_process=True, vp_stage=None):
    # [FUNCTION DOC] Calls the text provider with multimodal towers enabled and returns its model.
    """Multimodal provider: the text model plus the frozen HF vision/audio towers.

    A separate entry point instead of a CLI switch -- multimodal launch scripts pass
    this as --custom-model-provider-path.
    """
    return inkling_model_provider(pre_process, post_process, vp_stage, mm_towers=True)
