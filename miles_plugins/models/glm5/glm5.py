# [MODULE DOC] In-repo implementation of deepseek-sparse multi-latent attention (as used by GLM-5 and
# DeepSeek-V3.2) on packed (THD) sequences for Megatron. Contains: layer-number helpers for optional
# cross-layer sharing of top-k picks (only some "computing" layers run the indexer; "skip" layers reuse
# the latest computing layer's picks via a per-microbatch holder attached to the packed sequence
# parameters); a submodule spec dataclass including the indexer projections; the attention base class
# (rotary setup, output projection, sharing bookkeeping, forward that computes absorbed queries /
# keys, indexer top-k selection and sparse attention); the self-attention subclass that builds the
# projections and implements the query/key/value/indexer computation; and the spec builder that
# reads indexer settings from the Hugging Face config and validates pipeline splits. Context
# parallelism is handled by gathering keys and slicing queries.
import copy
import math
from dataclasses import dataclass
from typing import NoReturn

import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.common.embeddings import RotaryEmbedding, YarnRotaryEmbedding, _yarn_get_mscale
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.post_training.modelopt.layers import Linear
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp

# use fp32 for index weight
from megatron.core.transformer.moe.moe_utils import RouterGatingLinearFunction as WeightLinearFunction
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_config import MLATransformerConfig
from miles.kernels.attention.dsa import causal_ranges, get_dsa_topk_fn, lighting_indexer, sparse_attention
from miles.utils.hf_utils.config import load_hf_config
from miles.utils.replay_base import indexer_replay_manager
from miles_plugins.models.normalization import rms_norm

# Names of the indexer submodules. On a DSA model with *cross-layer index
# sharing* these only exist on "computing" layers; "skip" layers drop them.
_INDEXER_SUBMODULE_NAMES = ("wq_b", "wk", "k_norm", "weights_proj")


def is_skip_topk_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> bool:
    # [FUNCTION DOC] HELPER: Pure arithmetic: true when max(layer number minus offset, 0) is not a
    # multiple of the frequency (layer numbers are one-based).
    """Whether the (1-indexed) Megatron ``layer_number`` reuses a previous layer's top-k.

    Mirrors ``glm-train-prod``'s ``_get_skip_topk_flags``: a layer *computes* its
    own top-k when ``max(layer_number - offset, 0) % freq == 0``; otherwise it is a
    skip layer that reuses the most recent computing layer's indices.
    """
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


def source_compute_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> int:
    # [FUNCTION DOC] HELPER: Walks downward from the given layer number until reaching a layer that is
    # not a skip layer, and returns it. With a skip layer that has no computing layer below, the walk
    # would continue to non-positive numbers; layer numbers at or below the offset are always
    # computing, so it terminates.
    """The computing layer whose ``topk_indices`` a skip layer reuses."""
    layer = layer_number
    while is_skip_topk_layer(layer, skip_topk_offset, topk_freq):
        layer -= 1
    return layer


@dataclass
class DSASelfAttentionSubmodules:
    # [CLASS DOC] HELPER: Dataclass of module specs (all default none) for the attention layer: the
    # standard multi-latent-attention projections, norms, core attention and output projection, plus
    # four indexer pieces: query up projection, key projection, key norm, and per-head weight projection.
    """Submodules for the MLA self-attention layer."""

    linear_q_down_proj: ModuleSpec | type = None
    linear_q_up_proj: ModuleSpec | type = None
    linear_kv_down_proj: ModuleSpec | type = None
    linear_kv_up_proj: ModuleSpec | type = None
    linear_v_up_proj: ModuleSpec | type = None
    core_attention: ModuleSpec | type = None
    linear_proj: ModuleSpec | type = None
    q_layernorm: ModuleSpec | type = None
    kv_layernorm: ModuleSpec | type = None
    # added for indexer
    wq_b: ModuleSpec | type = None
    wk: ModuleSpec | type = None
    k_norm: ModuleSpec | type = None
    weights_proj: ModuleSpec | type = None


class DSAMultiLatentAttention(Attention):
    # [CLASS DOC] Base attention class. Builds rotary embedding (plain or yarn), the output projection,
    # and the sharing schedule; its forward runs absorbed-query sparse attention over packed sequences
    # using top-k picks from the indexer (computed here or reused from an earlier layer). Subclass
    # supplies the tensor computation and the remaining projections. The core-attention submodule is
    # not built here.
    """Multi-Latent Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        topk_backend: str = "torch",
        is_mtp_layer: bool = False,
        cp_comm_type: str | None = None,
        model_comm_pgs=None,
        pg_collection=None,
        name: str | None = None,
    ) -> None:
        # [FUNCTION DOC] Calls the Megatron base constructor, then sets: query projection size
        # (value head dim times heads), query head dim (nope plus rope dims), key/value hidden sizes,
        # whether up-projection recompute is on (selective recompute with the mla up projection module),
        # the softmax scale (yarn mscale squared over the square root of query head dim), a rotary
        # embedding by the config's rope type (plain or yarn; anything else raises a value error), and
        # the row-parallel output projection (skip bias add). The number of selected keys is fixed at
        # 2048, the top-k backend must be torch or flashinfer (value error otherwise), and the module is
        # registered with the indexer replay manager using layer number minus one as stream index.
        # Sharing: frequency (default 1) and skip offset (default 0) read from config, sharing enabled
        # when frequency is above one, skip flag from the layer helper, and the source computing layer
        # (own layer number when sharing is off). Unused arguments: is_mtp_layer, model_comm_pgs.
        # KNOWN ISSUE: the top-k count is hard coded to 2048 rather than read from the config (the
        # Hugging Face index top-k value is not used on this path).

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            name=name,
        )
        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads

        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim

        # Overwrite the base class kv shape to support MLA inference
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.config.v_head_dim

        self.recompute_up_proj = (
            self.config.recompute_granularity == "selective" and "mla_up_proj" in self.config.recompute_modules
        )
        self.qkv_up_checkpoint = None

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == "yarn":

            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f"Unsupported RoPE type: {self.config.rope_type}, supported types are " "'rope' and 'yarn'"
            )

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
            tp_group=self.pg_collection.tp,
        )

        self.index_topk = 2048
        if topk_backend not in ("torch", "flashinfer"):
            raise ValueError(f"Unsupported miles DSA topk backend: {topk_backend}")
        self.topk_backend = topk_backend
        indexer_replay_manager.register_to_module(self, "indexer_replay", stream_idx=self.layer_number - 1)

        # Cross-layer index sharing (optional). When the HF config provides
        # ``index_topk_freq`` / ``index_skip_topk_offset`` (see ``get_glm5_spec``),
        # only a subset of "computing" layers run the indexer top-k; the remaining
        # "skip" layers reuse the most recent computing layer's ``topk_indices``.
        # When those attrs are absent (``freq`` defaults to 1) every layer computes
        # its own top-k and ``skip_topk`` is always False -- i.e. the plain DSA path.
        self.index_topk_freq = getattr(self.config, "index_topk_freq", 1) or 1
        self.skip_topk_offset = getattr(self.config, "index_skip_topk_offset", 0) or 0
        self.index_share = self.index_topk_freq > 1
        self.skip_topk = self.index_share and is_skip_topk_layer(
            self.layer_number, self.skip_topk_offset, self.index_topk_freq
        )
        self._source_layer = (
            source_compute_layer(self.layer_number, self.skip_topk_offset, self.index_topk_freq)
            if self.index_share
            else self.layer_number
        )

    # Attribute name of the per-microbatch top-k holder we attach to the
    # ``packed_seq_params`` object (a plain dict: source layer_number -> topk_indices).
    # Used only on index-share models; see ``forward`` for why it lives on
    # ``packed_seq_params`` (per-microbatch isolation + recompute safety under PP).
    _HOLDER_ATTR = "_dsa_index_share_topk_holder"

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        router_token_masks=None,
        loss_mask=None,
    ):
        # [FUNCTION DOC] Input hidden states (sequence, batch, hidden); packed sequence parameters are
        # required in practice (cumulative lengths are read). Asserts no external rotary embedding,
        # attention bias, or flash-decoding cos/sin are passed; other arguments are ignored. Steps:
        # (1) get absorbed query, shared key, value up-projection weights, and the indexer query, key,
        # per-head weights from the subclass. (2) Choose top-k picks. A nested helper scores queries in
        # blocks of 8192 (whole sequence when replay is enabled, since replay records one tensor per
        # layer), calling the indexer kernel with start/end key bounds as 32-bit integers and the
        # replay-wrapped top-k function; it also computes softmax of the selected scores, but only the
        # picks are used afterwards; returns scores and picks with a head axis inserted. With sharing
        # enabled: fetch or create the per-microbatch holder dictionary on the packed sequence
        # parameters object; a skip layer reads its source layer's picks (raises an assertion error with
        # a descriptive message if absent, e.g. across pipeline stages); a computing layer computes
        # causal key ranges from the cumulative query lengths, scatters the ranges across the context
        # parallel group, squeezes the key's head axis, selects top-k, and stores picks in the holder
        # under its layer number. Without sharing every layer computes picks the same way. (3) Sparse
        # attention over the absorbed latent dimension (kv lora rank as value dim) with the picks and
        # the softmax scale, then expands through the value up-projection by a per-head contraction and
        # flattens heads. (4) If up-projection recompute is on, registers the output for recompute and
        # clears the checkpoint handle (asserting it exists). (5) Output projection. Returns
        # (output, bias).
        """Forward pass for multi-latent attention"""
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert rotary_pos_cos is None and rotary_pos_sin is None, "MLA does not support Flash Decoding"

        # hidden_states: [sq, b, h]
        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 16, 128], key: [96, 16, 128], value: [96, 16, 128]

        # query_absorbed: [96, 16, 576], kv: [96, 1, 576], wv: [16, 128, 512]
        q, kv, wv, index_query, index_key, head_weights = self.get_absorb_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )

        def fused_select_topk(index_q, index_k, w, starts, ends, block_size=8192):
            # [FUNCTION DOC] HELPER: Processes queries in consecutive blocks (block size defaults to
            # 8192; forced to the full length when replay is enabled), converting start/end bounds to
            # 32-bit, calling the indexer kernel for each block, applying softmax to the returned
            # selected scores, and concatenating over blocks. Returns (scores, picks) where picks get
            # an inserted head axis at dim 1.
            seq_len = index_q.shape[0]
            # replay records one topk tensor per layer-forward; don't split it
            if indexer_replay_manager.enabled:
                block_size = seq_len
            indexer_topk_scores = []
            topk_indices = []

            for start in range(0, seq_len, block_size):
                end = min(start + block_size, seq_len)
                index_q_block = index_q[start:end]
                w_block = w[start:end]
                starts_block = starts[start:end]
                ends_block = ends[start:end]
                starts_block = starts_block.to(torch.int32)
                ends_block = ends_block.to(torch.int32)
                indexer_topk_scores_block, topk_indices_block = lighting_indexer(
                    index_q_block,
                    index_k,
                    w_block,
                    starts_block,
                    ends_block,
                    self.index_topk,
                    topk_fn=indexer_replay_manager.get_topk_fn(get_dsa_topk_fn(self.topk_backend), return_probs=False),
                )

                indexer_topk_scores_block = torch.softmax(indexer_topk_scores_block, dim=-1)
                indexer_topk_scores.append(indexer_topk_scores_block)
                topk_indices.append(topk_indices_block)
            return torch.cat(indexer_topk_scores, dim=0), torch.cat(topk_indices, dim=0).unsqueeze(1)

        if self.index_share:
            # Cross-layer index sharing. The top-k holder lives on the per-microbatch
            # ``packed_seq_params`` object: it is closure-captured by Megatron's
            # activation-checkpoint ``custom_forward``, so the same instance is reused at
            # recompute time. That gives per-microbatch isolation (no cross-microbatch
            # clobber under PP 1F1B) AND recompute safety. A stage always starts on a
            # computing layer (asserted in ``get_glm5_spec``), so a skip layer's source is
            # always in-stage.
            holder = getattr(packed_seq_params, self._HOLDER_ATTR, None)
            if holder is None:
                holder = {}
                setattr(packed_seq_params, self._HOLDER_ATTR, holder)

            if self.skip_topk:
                if self._source_layer not in holder:
                    raise AssertionError(
                        "DSA index-share: skip layer "
                        f"(layer_number={self.layer_number}) needs the top-k of its source "
                        f"computing layer (layer_number={self._source_layer}), but that layer "
                        "did not run in this pipeline stage's forward. Cross-PP top-k sharing "
                        "is not supported; ensure every pipeline stage starts on a computing "
                        f"layer (index_topk_freq={self.index_topk_freq}, "
                        f"index_skip_topk_offset={self.skip_topk_offset}). "
                        f"Holder has layers {sorted(holder)}."
                    )
                topk_indices = holder[self._source_layer]
            else:
                starts, ends = causal_ranges(packed_seq_params.cu_seqlens_q)
                index_key = index_key.squeeze(1)
                starts = scatter_to_sequence_parallel_region(starts, group=parallel_state.get_context_parallel_group())
                ends = scatter_to_sequence_parallel_region(ends, group=parallel_state.get_context_parallel_group())
                _, topk_indices = fused_select_topk(index_query, index_key, head_weights, starts, ends)
                holder[self.layer_number] = topk_indices
        else:
            starts, ends = causal_ranges(packed_seq_params.cu_seqlens_q)
            index_key = index_key.squeeze(1)
            starts = scatter_to_sequence_parallel_region(starts, group=parallel_state.get_context_parallel_group())
            ends = scatter_to_sequence_parallel_region(ends, group=parallel_state.get_context_parallel_group())
            _, topk_indices = fused_select_topk(index_query, index_key, head_weights, starts, ends)

        core_attn_out = sparse_attention(
            q.unsqueeze(0),
            kv.unsqueeze(0),
            topk_indices.unsqueeze(0),
            self.softmax_scale,
            d_v=self.config.kv_lora_rank,
        ).squeeze(0)
        core_attn_out = torch.einsum("thm,hdm->thd", core_attn_out, wv)

        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)
        return output, bias


class DSAMLASelfAttention(DSAMultiLatentAttention):
    # [CLASS DOC] Self-attention variant: builds the low-rank query down/up projections, the combined
    # key/value down projection and its up projection (whose weights are absorbed into queries and the
    # output), and the indexer (query, key, key layer norm, per-head weights). Down projections are
    # duplicated across tensor parallel ranks. Skip layers in a sharing schedule drop their indexer
    # modules.
    """MLA Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        topk_backend: str = "torch",
        is_mtp_layer: bool = False,
        cp_comm_type: str | None = None,
        model_comm_pgs=None,
        pg_collection=None,
        name: str | None = None,
    ):
        # [FUNCTION DOC] Calls the base constructor with self attention. Builds, bias-free: query down
        # projection (hidden to q lora rank) and key/value down projection (hidden to kv lora rank plus
        # rope dim): for the Transformer-Engine plain linear they are set to the duplicated parallel
        # mode, for column-parallel kinds output gather is off, and any other module kind raises a value
        # error; query up projection (q lora rank to heads times query head dim) and key/value up
        # projection (kv lora rank to heads times (nope plus value dims)), column parallel without
        # gather; query and key/value layernorms. Indexer modules: query projection (q lora rank to
        # index heads times index head dim, flagged to skip gathering), key projection (hidden to index
        # head dim), key layer norm (temporarily forces the config normalization to layer norm,
        # asserting it was RMS norm, with fixed epsilon 1e-6, then restores it), and weight projection
        # (hidden to index heads, flagged to skip gathering); all indexer linears are duplicated. If
        # this is a skip layer, deletes the four indexer modules so parameters match checkpoints.
        # The loop variable reuses the name parameter name but only after it was consumed.
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            topk_backend=topk_backend,
            is_mtp_layer=is_mtp_layer,
            cp_comm_type=cp_comm_type,
            model_comm_pgs=model_comm_pgs,
            pg_collection=pg_collection,
            name=name,
        )
        q_down_proj_kwargs = {}
        if submodules.linear_q_down_proj in [TELinear]:
            q_down_proj_kwargs["parallel_mode"] = "duplicated"
        elif submodules.linear_q_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
        ]:
            q_down_proj_kwargs["gather_output"] = False
        else:
            raise ValueError(f"Unsupported linear_q_down_proj: {submodules.linear_q_down_proj}")

        self.linear_q_down_proj = build_module(
            submodules.linear_q_down_proj,
            self.config.hidden_size,
            self.config.q_lora_rank,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q_down_proj",
            skip_weight_param_allocation=False,
            **q_down_proj_kwargs,
        )

        self.linear_q_up_proj = build_module(
            submodules.linear_q_up_proj,
            self.config.q_lora_rank,
            self.config.num_attention_heads * self.q_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q_up_proj",
        )

        kv_down_proj_kwargs = {}
        if submodules.linear_kv_down_proj in [TELinear]:
            kv_down_proj_kwargs["parallel_mode"] = "duplicated"
        elif submodules.linear_kv_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
        ]:
            kv_down_proj_kwargs["gather_output"] = False
        else:
            raise ValueError(f"Unsupported linear_kv_down_proj: {submodules.linear_kv_down_proj}")

        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_down_proj",
            skip_weight_param_allocation=False,
            **kv_down_proj_kwargs,
        )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_up_proj",
        )

        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.config.q_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        # added for indexer
        indexer_linear_kwargs = dict(
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            parallel_mode="duplicated",
            skip_weight_param_allocation=False,
        )

        self.wq_b = build_module(
            submodules.wq_b,
            input_size=self.config.q_lora_rank,
            output_size=self.config.index_num_attention_heads * self.config.index_head_dim,
            tp_comm_buffer_name="wq_b",
            **indexer_linear_kwargs,
        )
        self.wq_b.weight._skip_gather = True

        # Build key projection
        self.wk = build_module(
            submodules.wk,
            input_size=self.config.hidden_size,
            output_size=self.config.index_head_dim,
            tp_comm_buffer_name="wk",
            **indexer_linear_kwargs,
        )

        # Build key normalization
        old_value = self.config.normalization
        assert config.normalization == "RMSNorm"
        self.config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            hidden_size=self.config.index_head_dim,
            config=self.config,
            # The layernorm eps is hardcoded at the moment
            eps=1e-6,
        )
        self.config.normalization = old_value

        # Build attention weight projection (per-head gating)
        # not sharded weights
        self.weights_proj = build_module(
            submodules.weights_proj,
            input_size=self.config.hidden_size,
            output_size=self.config.index_num_attention_heads,
            tp_comm_buffer_name="weights_proj",
            **indexer_linear_kwargs,
        )
        self.weights_proj.weight._skip_gather = True

        # Index-share skip layers carry no indexer weights -- drop the modules built
        # above so the parameter set matches the checkpoint (which only stores indexer
        # weights on computing layers) and weight export to HF omits them on skip layers.
        if self.skip_topk:
            for name in _INDEXER_SUBMODULE_NAMES:
                if hasattr(self, name):
                    delattr(self, name)

    def get_absorb_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        # [FUNCTION DOC] Computes the attention inputs for packed sequences. Requires a 3-dimensional
        # hidden state and packed sequence parameters (assertions). Returns (query, key, value
        # up-projection weight per head, index query, index key, head weights); the last three index
        # items are none on skip layers. Steps: rotary table from the rotary module (packed mode).
        # Down projections: query (squeezed) and key/value (gathered over the sequence-parallel group
        # if enabled) split into the compressed latent and the rope part; the key/value layernorm
        # module is applied (an identity in the standard spec). Query up projection to heads, split
        # into nope and rope parts. The key/value up weight is split per head into key-nope and value
        # parts; the query nope part is "absorbed" by contracting with the key-nope weights, giving
        # queries in the latent space. The latent is RMS-normalized in float32 using the fused layernorm
        # weight of the up projection (this is the real key/value norm), then rope part and latent are
        # gathered across the context-parallel group. Rotary is applied with an inner helper (see
        # below) to query rope (local) and key rope (gathered); query = absorbed nope plus rope, key =
        # latent plus rope, both contiguous. Indexer (non-skip layers only, everything detached from the
        # attention graph where noted): the normalized query latent is recomputed from the detached
        # unnormalized latent with the up projection's fused norm weight (plus one if zero-centered
        # gamma), then projected to index queries (gathered over sequence parallel if on); index keys
        # come from the detached hidden states, layer-normalized in float32, cast to bfloat16, gathered
        # over sequence parallel and context parallel, with a singleton head axis; head weights are
        # computed in float32 with the router-gating function, scaled by heads to the minus half and
        # index head dim to the minus half, then gathered over sequence parallel. Rotary on the indexer
        # uses one of two layouts depending on the interleave flag: interleaved places rope dims last,
        # otherwise first (the DeepSeek V3.2 convention, non-interleaved).
        # The key_value_states, position_ids, inference context and inference params arguments are
        # unused. The key/value layernorm module applied early is an identity in the standard spec; the
        # real normalization happens later with the fused weight.
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert hidden_states.ndim == 3, f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"
        assert packed_seq_params is not None

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )
        # TODO: support apply_rope_fusion
        rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq_params is not None)

        cu_seqlens_q = packed_seq_params.cu_seqlens_q
        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv

        # =========================================
        # QKV down projection and layernorm
        # =========================================

        # down proj are `TELinear`s, so the output is gathered and not TP-partitioned.`
        q_compressed, _ = self.linear_q_down_proj(hidden_states)
        q_compressed = q_compressed.squeeze(1)

        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if self.config.sequence_parallel:
            kv_combined = gather_from_sequence_parallel_region(kv_combined)
        kv_compressed, k_pos_emb = torch.split(
            kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )
        kv_compressed = self.kv_layernorm(kv_compressed)

        # =========================================
        # absorb
        # =========================================
        q_compressed = self.q_layernorm(q_compressed)
        q, _ = self.linear_q_up_proj(q_compressed)
        q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)
        q_no_pe, q_pos_emb = torch.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)

        w_kc, w_vc = self.linear_kv_up_proj.weight.unflatten(
            0,
            (-1, self.config.qk_head_dim + self.config.v_head_dim),
        ).split([self.config.qk_head_dim, self.config.v_head_dim], dim=1)

        # absorb
        q_no_pe = torch.einsum("thd,hdm->thm", q_no_pe, w_kc)

        # use scatter and gather here, to make the kv grad all reduce in tp
        kv_compressed = torch.nn.functional.rms_norm(
            kv_compressed.float(),
            normalized_shape=(kv_compressed.shape[-1],),
            weight=self.linear_kv_up_proj.layer_norm_weight.float(),
            eps=self.config.layernorm_epsilon,
        ).to(kv_compressed.dtype)

        k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb, group=parallel_state.get_context_parallel_group())
        kv_compressed = gather_from_sequence_parallel_region(
            kv_compressed, group=parallel_state.get_context_parallel_group()
        )

        def fuse_rope(q, cu_seqlens, gathered=False, interleaved=True):
            # [FUNCTION DOC] HELPER: Applies rotary embedding on packed sequences with the apex fused
            # kernel (imported lazily; import error if apex is missing) using the enclosing rotary
            # table. If interleaved, first de-interleaves the channels: even-indexed channels followed
            # by odd-indexed channels. If the input is already the full gathered sequence, applies
            # directly. Otherwise (local context-parallel slice): repeats the tensor cp-size times along
            # the sequence to span the global packed length, applies, and returns only this rank's
            # slice. Returns the rotated tensor (de-interleaved layout when interleaving was requested).
            # worse precision than apex.
            # from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb_thd
            from apex.transformer.functional import fused_apply_rotary_pos_emb_thd

            if interleaved:
                # mla use rope interleave
                x1 = q[..., 0::2]
                x2 = q[..., 1::2]
                t = torch.cat((x1, x2), dim=-1)
            else:
                t = q
            # TODO remove copy here
            # fuse rope not support this way rope (diff with cp)
            if gathered:
                return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, rotary_pos_emb.squeeze(0))
            else:
                seq_len = q.shape[0]
                cp_size = parallel_state.get_context_parallel_world_size()
                cp_rank = parallel_state.get_context_parallel_rank()
                t = t.repeat(cp_size, 1, 1)
                out = fused_apply_rotary_pos_emb_thd(t, cu_seqlens, rotary_pos_emb.squeeze(0))
                return out[cp_rank * seq_len : (cp_rank + 1) * seq_len]

        q_pos_emb = fuse_rope(q_pos_emb, cu_seqlens_q, gathered=False, interleaved=True)
        k_pos_emb = fuse_rope(k_pos_emb, cu_seqlens_kv, gathered=True, interleaved=True)

        query = torch.cat([q_no_pe, q_pos_emb], dim=-1)
        key = torch.cat([kv_compressed, k_pos_emb], dim=-1)

        query = query.contiguous()
        key = key.contiguous()

        if self.skip_topk:
            # Index-share skip layer: reuse a previous layer's top-k, so the indexer
            # projections are not run here. Return None for the index tensors.
            return query, key, w_vc, None, None, None

        # =========================================
        # Indexer
        # =========================================
        # Project queries and keys
        q_compressed = q_compressed.detach()
        # The query RMSNorm is fused into linear_q_up_proj, so q_compressed
        # still holds its unnormalized input. Share the norm with the indexer
        # without sending indexer gradients into the attention parameters.
        q_norm_weight = self.linear_q_up_proj.layer_norm_weight.detach()
        if self.config.layernorm_zero_centered_gamma:
            q_norm_weight = q_norm_weight.float() + 1
        q_compressed = rms_norm(q_compressed, q_norm_weight, self.config.layernorm_epsilon)
        hidden_states = hidden_states.detach()
        rotary_pos_emb = rotary_pos_emb.detach()

        index_q, _ = self.wq_b(q_compressed)
        index_q = index_q.view(
            *index_q.size()[:-1], self.config.index_num_attention_heads, self.config.index_head_dim
        )  # [total_tokens, index_num_attention_heads_per_partition, index_head_dim]
        if self.config.sequence_parallel:
            index_q = gather_from_sequence_parallel_region(index_q)

        index_k, _ = self.wk(hidden_states)
        index_k = self.k_norm(index_k.squeeze(1).float()).bfloat16()

        if self.config.sequence_parallel:
            index_k = gather_from_sequence_parallel_region(index_k)
        index_k = gather_from_sequence_parallel_region(index_k, group=parallel_state.get_context_parallel_group())
        index_k = index_k.unsqueeze(1)  # [total_tokens, 1, head_dim]

        # head_weights, _ = self.weights_proj(hidden_states.float())
        head_weights = WeightLinearFunction.apply(hidden_states, self.weights_proj.weight, None, torch.float32)
        head_weights = head_weights.squeeze(1) * (
            (self.config.index_num_attention_heads**-0.5) * (self.config.index_head_dim**-0.5)
        )  # [total_tokens, index_num_attention_heads_per_partition]
        if self.config.sequence_parallel:
            head_weights = gather_from_sequence_parallel_region(head_weights)

        if self.config.indexer_rope_interleave:
            index_q_no_pe, index_q_pe = torch.split(
                index_q,
                [self.config.index_head_dim - self.config.qk_pos_emb_head_dim, self.config.qk_pos_emb_head_dim],
                dim=-1,
            )
            index_q_pe = fuse_rope(index_q_pe, cu_seqlens_q, gathered=False, interleaved=True)
            index_query = torch.cat([index_q_no_pe, index_q_pe], dim=-1)

            index_k_no_pe, index_k_pe = torch.split(
                index_k,
                [self.config.index_head_dim - self.config.qk_pos_emb_head_dim, self.config.qk_pos_emb_head_dim],
                dim=-1,
            )
            index_k_pe = fuse_rope(index_k_pe, cu_seqlens_kv, gathered=True, interleaved=True)
            index_key = torch.cat([index_k_no_pe, index_k_pe], dim=-1)
        else:
            # DeepSeek v3.2 indexer uses non-interleaved RoPE with rope dims first.
            index_q_pe, index_q_no_pe = torch.split(
                index_q,
                [self.config.qk_pos_emb_head_dim, self.config.index_head_dim - self.config.qk_pos_emb_head_dim],
                dim=-1,
            )
            index_q_pe = fuse_rope(index_q_pe, cu_seqlens_q, gathered=False, interleaved=False)
            index_query = torch.cat([index_q_pe, index_q_no_pe], dim=-1)

            index_k_pe, index_k_no_pe = torch.split(
                index_k,
                [self.config.qk_pos_emb_head_dim, self.config.index_head_dim - self.config.qk_pos_emb_head_dim],
                dim=-1,
            )
            index_k_pe = fuse_rope(index_k_pe, cu_seqlens_kv, gathered=True, interleaved=False)
            index_key = torch.cat([index_k_pe, index_k_no_pe], dim=-1)

        return query, key, w_vc, index_query, index_key, head_weights

    def get_query_key_value_tensors(self):
        # [FUNCTION DOC] Placeholder that does nothing and returns nothing; the absorbed variant above
        # is used instead.
        pass

    def backward_dw(self) -> NoReturn:
        # [FUNCTION DOC] Triggers deferred weight-gradient computation for key/value, query and output
        # projections in that order. Returns nothing.
        # KNOWN ISSUE: annotated as never returning, but it returns normally.
        """Execute weight update operations"""
        self._backward_kv_proj()
        self._backward_q_proj()
        self._backward_output_proj()

    def _backward_kv_proj(self):
        # [FUNCTION DOC] HELPER: Deferred weight-gradient step for the key/value up then down projection.
        """Update weights for KV projection layers"""
        self.linear_kv_up_proj.backward_dw()
        self.linear_kv_down_proj.backward_dw()

    def _backward_q_proj(self):
        # [FUNCTION DOC] HELPER: Deferred weight-gradient step for the query down then up projection.
        """Update weights for Q projection layers"""
        self.linear_q_down_proj.backward_dw()
        self.linear_q_up_proj.backward_dw()

    def _backward_output_proj(self):
        # [FUNCTION DOC] HELPER: Deferred weight-gradient step for the output projection.
        """Update weights for output projection layer"""
        self.linear_proj.backward_dw()

    def set_for_recompute_input_layernorm(self):
        # [FUNCTION DOC] Prepares for fp8 recompute of the input layernorm: when a query lora rank is
        # set, flags the query down projection, always flags the key/value down projection and the
        # output projection, to save their original (pre-quantization) input. Raises a value error
        # naming the required Transformer-Engine version if any of those modules lacks the attribute.
        # Mutates module attributes.
        """Set the attention layer for recompute input_layernorm. Only needed for fp8."""
        if self.config.q_lora_rank is not None:
            if hasattr(self.linear_q_down_proj, "save_original_input"):
                self.linear_q_down_proj.save_original_input = True
            else:
                raise ValueError(
                    "layernorm recompute for fp8 with MLASelfAttention needs " "transformer-engine>=2.6.0dev0."
                )
        if hasattr(self.linear_kv_down_proj, "save_original_input"):
            self.linear_kv_down_proj.save_original_input = True
        else:
            raise ValueError(
                "layernorm recompute for fp8 with MLASelfAttention needs " "transformer-engine>=2.6.0dev0."
            )
        if hasattr(self.linear_proj, "save_original_input"):
            self.linear_proj.save_original_input = True
        else:
            raise ValueError(
                "layernorm recompute for fp8 with MLASelfAttention needs " "transformer-engine>=2.6.0dev0."
            )


def get_glm5_spec(args, config, vp_stage):
    # [FUNCTION DOC] Spec builder. Loads the Hugging Face config from the checkpoint path and writes
    # onto the Megatron config: index head count, index head dim, interleave flag (default false), and
    # sharing schedule (frequency default 1, skip offset default 0; falsy values become the defaults).
    # Builds Megatron's decoder block spec with Transformer Engine (passing the virtual pipeline stage
    # when given). With sharing on (frequency above one): for the first local layer of this stage,
    # raises an assertion error with a descriptive message if it is a skip layer (since picks cannot
    # cross pipeline boundaries). Then creates the attention module spec for the self-attention class
    # with causal mask, the top-k backend from args, and submodules from the Transformer-Engine spec
    # provider (identity ops for the layernorms and the unused value up projection; layernorm fused
    # into the column-parallel up projections). Replaces the self-attention spec in a deep copy of each
    # layer spec built for this stage and returns the block spec.
    hf_config = load_hf_config(args.hf_checkpoint)
    config.index_num_attention_heads = hf_config.index_n_heads
    config.index_head_dim = hf_config.index_head_dim
    config.indexer_rope_interleave = bool(getattr(hf_config, "indexer_rope_interleave", False))
    # Optional cross-layer index-sharing schedule. Present on DSA checkpoints that only
    # store indexer weights on a subset of "computing" layers (e.g. GLM-5.2). When absent,
    # every layer computes its own top-k (plain DSA) and DSAMLASelfAttention runs the
    # non-shared path.
    config.index_topk_freq = getattr(hf_config, "index_topk_freq", 1) or 1
    config.index_skip_topk_offset = getattr(hf_config, "index_skip_topk_offset", 0) or 0
    # Define the decoder block spec
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    # Cross-layer index sharing keeps the shared top-k in a per-microbatch holder on
    # ``packed_seq_params``, which does not cross PP boundaries. A skip layer must run in
    # the same pipeline stage as the computing layer it reuses, so a (virtual) pipeline
    # stage may not *start* on a skip layer. Forbid that split here (supporting it would
    # need PP send/recv of the top-k).
    if config.index_topk_freq > 1:
        from megatron.core.transformer.transformer_block import get_transformer_layer_offset

        layer_offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for local_id in range(num_layers_to_build):
            layer_number = local_id + layer_offset + 1  # Megatron layer_number is 1-indexed
            if local_id == 0 and is_skip_topk_layer(
                layer_number, config.index_skip_topk_offset, config.index_topk_freq
            ):
                src = source_compute_layer(layer_number, config.index_skip_topk_offset, config.index_topk_freq)
                raise AssertionError(
                    "DSA index-share pipeline split is invalid: this stage starts at global "
                    f"layer_number={layer_number} which is a skip layer whose source computing "
                    f"layer={src} is on a previous pipeline stage. Cross-layer top-k sharing does "
                    "not cross PP boundaries. Choose a pipeline layout where every stage begins on "
                    "a computing layer (index_topk_freq="
                    f"{config.index_topk_freq}, index_skip_topk_offset={config.index_skip_topk_offset})."
                )

    backend = TESpecProvider()

    self_attn_module_spec = ModuleSpec(
        module=DSAMLASelfAttention,
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
    for layer_id in range(num_layers_to_build):
        layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
        layer_specs.submodules.self_attention = self_attn_module_spec
        transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
