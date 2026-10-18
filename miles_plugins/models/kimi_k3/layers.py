# [MODULE DOC] Layer implementations for Kimi K3. Provides the multi-latent attention module (low-rank
# query, shared low-rank key/value latent with a separate extra key field, sigmoid output gate, core
# attention from Transformer Engine with context parallel and packed sequences), the delta-attention
# wrapper on the shared linear-attention layer, and the transformer layer implementing "attention
# residuals": instead of a plain residual stream, each sublayer's input is a learned softmax-weighted
# aggregate of a running prefix sum and a bank of block snapshots, where a snapshot is written every
# block-size layers; the bank travels between pipeline stages packed into the hidden states.
import copy

import torch
import torch.nn as nn
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELinear,
    TERowParallelLinear,
)
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer, get_transformer_layer_offset

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads
from miles_plugins.models.kimi_k3.ops import KimiRMSNorm, attn_res_aggregate
from miles_plugins.models.kimi_k3.pipeline import bank_num_rows, pack_stage_boundary, unpack_stage_boundary
from miles_plugins.models.linear_attn import KimiDeltaAttention, LinearAttentionLayer


def _mark_tp_replicated(module: nn.Module) -> None:
    # [FUNCTION DOC] HELPER: Flags every parameter of the module so that its gradient is summed across
    # the tensor-parallel domain (for parameters replicated across tp ranks that see partial gradients).
    for parameter in module.parameters():
        parameter.sum_gradients_across_tp_domain = True


def _linear(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Calls a Transformer-Engine linear (returns output and bias), asserts the
    # bias is none, and returns the output.
    output, bias = module(inputs)
    assert bias is None
    return output


class KimiK3Attention(MegatronModule):
    # [CLASS DOC] Multi-latent attention. Tensors are kept in sequence-major layout. Components: query
    # path (duplicated down projection, norm, column-parallel up projection to local heads of dimension
    # nope-plus-extra); key/value path (duplicated down projection to a latent plus one extra key field
    # shared by all heads, norm on the latent, column-parallel up projection to per-head nope key and
    # value); a column-parallel gate projection; a row-parallel output projection; and a Transformer
    # Engine core attention (causal, scale head dim to the minus half, separate key and value channel
    # sizes). Rotary embedding is not applied here (the extra field is just concatenated to keys).
    # Class flag is_kda is false.
    """K3's MLA layers; the KDA layers are :class:`KimiK3KDAAttention`."""

    is_kda = False

    def __init__(
        self,
        config,
        layer_number: int,
        cp_comm_type: str | None = None,
        pg_collection=None,
        name: str | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the context-parallel comm type; if no process-group collection is given
        # builds one from global parallel state (tensor and context groups), else asserts it has both.
        # Records group sizes and sequence parallel flag; keeps a shallow config copy with sequence
        # parallelism disabled for the projections (this module gathers and scatters the sequence itself).
        # Asserts the layer is not a delta-attention layer (layer number not in the config's list) and
        # builds the MLA submodules. The name argument is discarded.
        super().__init__(config=config)
        del name  # build_module forwards the module path; K3 constructs its submodules directly
        self.cp_comm_type = cp_comm_type

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        else:
            assert hasattr(pg_collection, "tp") and hasattr(pg_collection, "cp")
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.tp_size = self.tp_group.size()
        self.cp_group = pg_collection.cp
        self.cp_size = self.cp_group.size()
        self.sequence_parallel = config.sequence_parallel
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False

        self.layer_idx = layer_number - 1
        assert layer_number not in config.kimi_kda_layers, "K3's KDA layers are KimiK3KDAAttention"
        self._init_mla(config)

    def _duplicated_linear(self, input_size: int, output_size: int) -> TELinear:
        # [FUNCTION DOC] HELPER: Bias-free Transformer-Engine linear replicated on every tensor-parallel
        # rank (duplicated mode), using the sequence-parallel-free config copy.
        return TELinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def _column_linear(self, input_size: int, output_size: int) -> TEColumnParallelLinear:
        # [FUNCTION DOC] HELPER: Bias-free column-parallel linear without output gather on the layer's
        # tensor-parallel group.
        return TEColumnParallelLinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

    def _row_linear(self, input_size: int, output_size: int) -> TERowParallelLinear:
        # [FUNCTION DOC] HELPER: Bias-free row-parallel linear taking parallel input on the layer's
        # tensor-parallel group.
        return TERowParallelLinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

    def _init_mla(self, config) -> None:
        # [FUNCTION DOC] HELPER: Reads sizes from the config (heads must divide by tensor parallel size,
        # asserted; low-rank ranks; nope key dim; extra key dim; value dim) and builds: query down
        # projection (duplicated) with RMS norm, query up projection (column parallel to all heads times
        # query head dim), key/value down projection (duplicated; latent plus extra), latent norm,
        # key/value up projection (column parallel, heads times nope-plus-value), gate projection
        # (column parallel, hidden to heads times value dim), output projection (row parallel). Norms and
        # created on the current GPU in the parameter dtype. The core attention is built from the full
        # config with the layer number, causal masking, scale, key/value channels, the cp comm type and
        # process groups.
        hidden_size = config.hidden_size
        device = torch.cuda.current_device()
        dtype = config.params_dtype
        self.num_heads = config.num_attention_heads
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = self.num_heads // self.tp_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_head_dim
        self.qk_extra_head_dim = config.qk_pos_emb_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_extra_head_dim

        self.q_a_proj = self._duplicated_linear(hidden_size, self.q_lora_rank)
        self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank, config.layernorm_epsilon, device=device, dtype=dtype)
        self.q_b_proj = self._column_linear(self.q_lora_rank, self.num_heads * self.q_head_dim)
        self.kv_a_proj_with_mqa = self._duplicated_linear(hidden_size, self.kv_lora_rank + self.qk_extra_head_dim)
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank, config.layernorm_epsilon, device=device, dtype=dtype)
        self.kv_b_proj = self._column_linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        self.g_proj = self._column_linear(hidden_size, self.num_heads * self.v_head_dim)
        self.o_proj = self._row_linear(self.num_heads * self.v_head_dim, hidden_size)

        # Megatron's MLA core carries CP and varlen; only its module assumes the rotated 64-wide field K3 lacks
        self.core_attention = TEDotProductAttention(
            config=self.config,
            layer_number=self.layer_idx + 1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=self.q_head_dim**-0.5,
            k_channels=self.q_head_dim,
            v_channels=self.v_head_dim,
            cp_comm_type=self.cp_comm_type,
            pg_collection=self.pg_collection,
        )

    def _forward_mla(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Input (sequence, batch, hidden) full sequence. Query: down projection,
        # norm, up projection, viewed as local heads. Key/value: down projection split into latent and
        # extra key field; up projection of the normed latent split into per-head nope key and value;
        # value made contiguous (TE infers layout from strides); the extra field passes through a
        # tensor-parallel copy op (so its gradient is reduced across tp ranks), gets a head axis
        # broadcast across local heads, and is appended to each head's nope key. For packed (thd) input
        # the singleton batch dim is squeezed from q, k, v. Runs core attention (causal) with the packed
        # parameters; for thd reshapes the output back to (tokens, 1, heads times value dim). Multiplies
        # by a sigmoid gate computed from the input and applies the output projection. Returns the
        # (sequence, batch, hidden) partial output (row parallel; reduction is inside the projection).
        # stay in TE's sbhd [s, b, ...] layout so q/k/v need no transpose; the KDA kernels want [b, s, ...]
        x = hidden_states
        query = _linear(
            self.q_b_proj,
            self.q_a_layernorm(_linear(self.q_a_proj, x)),
        )
        query = query.view(*query.shape[:-1], self.local_num_heads, self.q_head_dim)

        compressed_kv = _linear(self.kv_a_proj_with_mqa, x)
        kv_latent, key_extra = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_extra_head_dim],
            dim=-1,
        )
        key_value = _linear(self.kv_b_proj, self.kv_a_layernorm(kv_latent))
        key_value = key_value.view(
            *key_value.shape[:-1],
            self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key_nope, value = torch.split(
            key_value,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        # TE classifies the qkv layout from strides, so the sliced value needs canonical ones too
        value = value.contiguous()
        key_extra = copy_to_tensor_model_parallel_region(key_extra, group=self.tp_group)
        key_extra = key_extra.unsqueeze(-2).expand(*key_nope.shape[:-1], -1)
        key = torch.cat((key_nope, key_extra), dim=-1)

        # thd packing wants 3D [t, h, d]; mirror Megatron's Attention.forward squeeze/reshape
        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if is_thd:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        # TE returns the head dims already fused, so there is no flatten here.
        output = self.core_attention(
            query,
            key,
            value,
            None,
            packed_seq_params=packed_seq_params,
            attn_mask_type=AttnMaskType.causal,
        )
        if is_thd:
            output = output.reshape(output.size(0), 1, -1)
        output = output * torch.sigmoid(_linear(self.g_proj, x))
        return _linear(self.o_proj, output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        # [FUNCTION DOC] Discards all arguments except the hidden states and packed sequence
        # parameters. With sequence parallel gathers the sequence over the tensor-parallel group (backward
        # split, not reduced, since the gathered computation is partly duplicated), runs the MLA path, and
        # scatters the output. Returns (output, none).
        del (
            attention_mask,
            key_value_states,
            inference_context,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            sequence_len_offset,
            kwargs,
        )
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=self.tp_group,
            )
        output = self._forward_mla(hidden_states, packed_seq_params)
        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
        return output, None


class KimiK3KDAAttention(LinearAttentionLayer):
    # [CLASS DOC] Delta-attention layer: the shared head-sharded linear-attention layer wrapping a Kimi
    # delta attention core, with no input norm (identity; the transformer layer's norm is used) and
    # all-gather-style context parallelism turned off. Class flag is_kda is true.
    """K3's KDA layers on the shared head-sharded linear-attention layer. The ``self_attention`` constructor
    signature K3's layer spec builds with; the input norm is the transformer layer's, so none here."""

    is_kda = True

    def __init__(
        self,
        config,
        layer_number: int,
        cp_comm_type: str | None = None,
        pg_collection=None,
        name: str | None = None,
    ) -> None:
        # [FUNCTION DOC] Ignores the comm type and name. Builds the process groups from global state if
        # not given. Asserts the layer number is in the config's delta-attention layer list. Builds the
        # head layout (key and value heads both equal the linear head count, key and value dims both the
        # linear head dim) and the Kimi delta attention core (conv kernel size, norm epsilon, tp group,
        # gate lower bound from the config), then calls the parent with an identity norm.
        del cp_comm_type, name
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        assert layer_number in config.kimi_kda_layers
        heads = LinearAttnHeads(
            num_k_heads=config.kimi_linear_num_heads,
            num_v_heads=config.kimi_linear_num_heads,
            head_k_dim=config.kimi_linear_head_dim,
            head_v_dim=config.kimi_linear_head_dim,
        )
        core = KimiDeltaAttention(
            config,
            heads,
            config.kimi_linear_conv_kernel_size,
            config.layernorm_epsilon,
            pg_collection.tp,
            gate_lower_bound=config.kimi_kda_gate_lower_bound,
        )
        super().__init__(config, core, nn.Identity(), pg_collection, allgather_cp=False)


class KimiK3TransformerLayer(TransformerLayer):
    # [CLASS DOC] Transformer layer with attention residuals. State flowing between layers is a pair:
    # prefix sum (sequence, batch, hidden) and a block bank (sequence, batch, rows, hidden) of block
    # snapshots. Each layer owns, for attention and for the MLP, a projection to one logit and a norm
    # used to weight the bank rows and the prefix sum; the last layer additionally owns an output
    # aggregation pair. All these small modules are flagged for tensor-parallel gradient summation.
    def __init__(self, *args, **kwargs) -> None:
        # [FUNCTION DOC] Builds the base layer and asserts hidden dropout is zero. Reads block size
        # from the config. Creates the four aggregation modules (and the output pair on the last layer)
        # on the current GPU in the parameter dtype. Computes which layers are pipeline stage entries and
        # exits from each stage's layer offset (virtual pipeline is unsupported, so no virtual stage is
        # passed): an entry is a layer index equal to some stage's first layer and above zero; an exit is
        # the layer just before such a start, excluding the final layer.
        super().__init__(*args, **kwargs)
        assert self.config.hidden_dropout == 0.0, "Kimi K3 requires hidden dropout 0"

        hidden_size = self.config.hidden_size
        eps = self.config.layernorm_epsilon
        device = torch.cuda.current_device()
        dtype = self.config.params_dtype
        self.attn_res_block_size = self.config.kimi_attn_res_block_size
        self.self_attention_res_norm = KimiRMSNorm(hidden_size, eps, device=device, dtype=dtype)
        self.self_attention_res_proj = nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)
        self.mlp_res_norm = KimiRMSNorm(hidden_size, eps, device=device, dtype=dtype)
        self.mlp_res_proj = nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)
        _mark_tp_replicated(self.self_attention_res_norm)
        _mark_tp_replicated(self.self_attention_res_proj)
        _mark_tp_replicated(self.mlp_res_norm)
        _mark_tp_replicated(self.mlp_res_proj)

        if self.layer_number == self.config.num_layers:
            self.output_attn_res_norm = KimiRMSNorm(hidden_size, eps, device=device, dtype=dtype)
            self.output_attn_res_proj = nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)
            _mark_tp_replicated(self.output_attn_res_norm)
            _mark_tp_replicated(self.output_attn_res_proj)

        # stage entry/exit layers from the per-rank offsets; VPP is rejected, so vp_stage is None
        pp_size = self.config.pipeline_model_parallel_size
        stage_starts = {get_transformer_layer_offset(self.config, None, r) for r in range(pp_size)}
        layer_idx = self.layer_number - 1
        self.is_stage_entry = layer_idx in stage_starts and layer_idx > 0
        self.is_stage_exit = (layer_idx + 1) in stage_starts and layer_idx + 1 < self.config.num_layers

    @staticmethod
    def _add_bias(output_with_bias: tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Static method: returns the output plus the bias when a bias exists,
        # otherwise the output alone.
        output, bias = output_with_bias
        return output if bias is None else output + bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] Returns (prefix sum or packed stage-boundary tensor, block bank). The "context"
        # argument is repurposed to carry the block bank between layers within a stage. State setup: with
        # a bank given, the hidden states are the prefix sum; on layer zero the prefix sum is the input
        # and the bank is an empty tensor (zero rows); otherwise (must be a stage entry, asserted) the
        # hidden states are an unpacked pair (prefix sum, bank) with a bank row count derived from the
        # layer index and block size. Attention input: if the bank is non-empty, the attention-residual
        # aggregate of prefix sum and bank (using this layer's attention projection/norm and the input
        # layernorm); else just the input layernorm of the prefix sum. If this layer index is a multiple
        # of the block size (a block-write layer), the prefix sum is appended to the bank as a new row
        # BEFORE attention. Attention output (bias added) replaces the prefix sum on block-write layers
        # (a fresh block starts), else is added to it. MLP input: aggregate of prefix sum and bank with
        # the MLP projection/norm and pre-MLP layernorm; the MLP receives the padding mask, plus token
        # ids for MoE layers; its output is added to the prefix sum. On the last layer, the prefix sum is
        # replaced by the aggregate with the output projection/norm (no further norm). On stage exits the
        # first return value is the prefix sum packed together with the bank for transport. Other
        # arguments (masks, rotary, inference) are passed to attention or ignored.
        del context_mask, kwargs
        layer_idx = self.layer_number - 1

        if context is not None:
            prefix_sum = hidden_states
            block_residual = context
        elif layer_idx == 0:
            prefix_sum = hidden_states
            block_residual = hidden_states.new_empty(*hidden_states.shape[:-1], 0, hidden_states.shape[-1])
        else:
            assert self.is_stage_entry, "Attention-residual snapshot bank is missing"
            prefix_sum, block_residual = unpack_stage_boundary(
                hidden_states,
                self.config.hidden_size,
                bank_num_rows(layer_idx, self.attn_res_block_size),
            )

        if block_residual.shape[-2] > 0:
            attention_input = attn_res_aggregate(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
            )
        else:
            attention_input = self.input_layernorm(prefix_sum)

        is_block_write_layer = layer_idx % self.attn_res_block_size == 0
        if is_block_write_layer:
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)

        attention_output = self._add_bias(
            self.self_attention(
                attention_input,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        )
        prefix_sum = attention_output if is_block_write_layer else prefix_sum + attention_output

        mlp_input = attn_res_aggregate(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            self.pre_mlp_layernorm,
        )
        mlp_kwargs = {"padding_mask": padding_mask}
        if self.is_moe_layer:
            mlp_kwargs["input_ids"] = input_ids
        mlp_output = self._add_bias(self.mlp(mlp_input, **mlp_kwargs))
        prefix_sum = prefix_sum + mlp_output

        if self.layer_number == self.config.num_layers:
            prefix_sum = attn_res_aggregate(
                prefix_sum,
                block_residual,
                self.output_attn_res_proj,
                self.output_attn_res_norm,
                nn.Identity(),
            )

        if self.is_stage_exit:
            return pack_stage_boundary(prefix_sum, block_residual), block_residual
        return prefix_sum, block_residual
