# [MODULE DOC] Hyper-connection (multi-stream residual) modules for Qwen3.8-Next, filling the
# hyper-connection slots of Megatron's layer specs. The hidden state carries n residual streams
# concatenated along the feature dim (width n times hidden). Per layer, a low-rank gate reads the
# streams (normalised per stream) and averages them into the sublayer input; after the sublayer the
# output is injected back into every stream with per-stream weights. Fused kernels live in the miles
# hyper-connection kernel module. Contains: the per-layer connection, the model-level head
# contraction, and a variant that first applies the per-layer-embedding (PLE) increment.
"""Qwen3.8-Next hyper-connections for Megatron's HC ModuleSpec slots.

Differs from Megatron's mHC (DeepSeek-V4): per-stream per-feature low-rank read
gate with a MEAN over streams, and identity residual mixing (h_res is None).
"""

import torch
import torch.nn.functional as F
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.tensor_parallel.random import is_checkpointing
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from miles.kernels.hyper_connection.hc import hc_combine, hc_mix_inject
from miles_plugins.models.qwen3_8_next.ops.ple import Qwen38NextPLE, current_ple_batch


class Qwen38NextHyperConnection(MegatronModule):
    # [CLASS DOC] Per-layer hyper-connection for an attention or MLP slot. Parameters (all in the
    # config's parameter dtype and tagged sequence-parallel so their gradients are reduced across
    # TP): a norm weight over the wide dim (zeros), a down projection (low rank by wide) and an up
    # projection (wide by low rank) forming the low-rank read gate, and, if use_combine, a block
    # inject weight (streams by wide). Wide is stream count times hidden size; low rank comes from
    # the config's Qwen3.8-Next HC low-rank setting. When use_combine is false the module is
    # read-only (no inject weight, set to None) and its forward asserts.
    """Per-layer HC filling the attention/MLP spec slots. Params replicated across TP
    (sequence_parallel attr set); kept bf16 to match sglang."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        hc_count: int | None = None,
        use_combine: bool = True,
    ):
        # [FUNCTION DOC] Stores layer number, stream count n (hc_count if given, else the config's
        # residual stream count), hidden size, layer-norm epsilon, and use_combine. Creates the
        # parameters described on the class (norm weight zeros; the others uninitialised), tags them
        # sequence-parallel per config, then (without gradient tracking) fills the down, up and
        # inject weights with Xavier uniform initialisation.
        super().__init__(config)
        self.layer_number = layer_number
        self.n = hc_count if hc_count is not None else config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.norm_eps = config.layernorm_epsilon
        self.use_combine = use_combine

        lowrank = config.qwen3_8_next_hc_lowrank
        wide = self.n * self.hidden_size
        dtype = config.params_dtype

        self.hc_norm_weight = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.input_mix_weight_down = torch.nn.Parameter(torch.empty(lowrank, wide, dtype=dtype))
        self.input_mix_weight_up = torch.nn.Parameter(torch.empty(wide, lowrank, dtype=dtype))
        params = [self.hc_norm_weight, self.input_mix_weight_down, self.input_mix_weight_up]

        if use_combine:
            self.block_inject_weight = torch.nn.Parameter(torch.empty(self.n, wide, dtype=dtype))
            params.append(self.block_inject_weight)
        else:
            self.block_inject_weight = None

        for p in params:
            p.sequence_parallel = config.sequence_parallel

        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.input_mix_weight_down)
            torch.nn.init.xavier_uniform_(self.input_mix_weight_up)
            if use_combine:
                torch.nn.init.xavier_uniform_(self.block_inject_weight)

    def forward(
        self,
        hidden_states: Tensor,
        mhc_recompute_manager=None,
        output_slot=None,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
        # [FUNCTION DOC] Read step. Input hidden_states: sequence, batch, streams times hidden.
        # Calls the fused mix-and-inject kernel with the norm weight, down / up weights, inject
        # weight, stream count and epsilon, obtaining the aggregated (stream-averaged, gated) sublayer
        # input and the per-stream post weights. Returns a 4-tuple: aggregated, None (no residual
        # mixing matrix), post weights, and the unchanged input as the residual.
        # Raises not-implemented error if the mHC recompute manager or output slot is given; asserts
        # that the module was built with use_combine.
        """Returns (aggregated, h_res=None, h_post, residual)."""
        if mhc_recompute_manager is not None or output_slot is not None:
            raise NotImplementedError(
                "Qwen38NextHyperConnection does not support the mHC recompute arena yet; "
                "run without 'mhc' in --recompute-modules."
            )
        assert self.use_combine, "per-layer HC needs the inject weight; use the Mixer for read-only"
        aggregated, h_post = hc_mix_inject(
            hidden_states,
            self.hc_norm_weight,
            self.input_mix_weight_down,
            self.input_mix_weight_up,
            self.block_inject_weight,
            self.n,
            self.norm_eps,
        )
        return aggregated, None, h_post, hidden_states

    def fused_h_res_h_post_bda(
        self,
        h_res: Tensor | None,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias,
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager=None,
    ) -> Tensor:
        # [FUNCTION DOC] Write step: inject the sublayer output into every stream. Inputs: h_res
        # (must be None; asserted), the original residual streams, post weights from the read step,
        # the sublayer output (a plain tensor or a pair of output and bias), dropout probability,
        # training flag, and a fused flag and manager. The fused flag is accepted but unused; a
        # non-None manager raises not-implemented error. Steps: split output / bias; add the bias
        # (broadcast over the last dim) if present; apply dropout to the output if probability is
        # positive and training (note: dropout is functional with default behaviour, no RNG tracker);
        # return the fused combine of residual, output, post weights and stream count.
        """``X'_c = X_c + a_c * (y + bias)``, dropout applied to the injection."""
        assert h_res is None, "Qwen3.8-Next hyper-connection has no residual mixing matrix"
        if manager is not None:
            raise NotImplementedError("mHC recompute arena not supported yet")

        if isinstance(layer_output_with_bias, tuple):
            x, bias = layer_output_with_bias
        else:
            x, bias = layer_output_with_bias, None

        if bias is not None:
            x = x + bias.view(*([1] * (x.dim() - 1)), -1)
        if dropout_prob > 0.0 and training:
            x = F.dropout(x, p=dropout_prob)
        return hc_combine(original_residual, x, h_post, self.n)


class Qwen38NextHCHeadContraction(MegatronModule):
    # [CLASS DOC] Model-level contraction applied before the output head: reduces the n residual
    # streams (feature width n times hidden) to a single hidden-size stream with the same gated-mean
    # read as the per-layer connection, but without an inject weight. Parameters: norm weight, down
    # and up weights (same shapes and init as the per-layer module), tagged sequence-parallel.
    """Model-level output contraction [s,b,n*C]->[s,b,C]: same gated mean as the
    per-layer HC (per-stream RMS) -- NOT interchangeable with DSv4's built-in."""

    def __init__(self, config: TransformerConfig, hc_count: int | None = None):
        # [FUNCTION DOC] Stores stream count (hc_count or the config's residual stream count),
        # hidden size and epsilon; creates the norm / down / up parameters, tags them
        # sequence-parallel per config, and Xavier-uniform initialises down and up (no gradient).
        super().__init__(config)
        self.n = hc_count if hc_count is not None else config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.norm_eps = config.layernorm_epsilon

        lowrank = config.qwen3_8_next_hc_lowrank
        wide = self.n * self.hidden_size
        dtype = config.params_dtype

        self.hc_norm_weight = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.input_mix_weight_down = torch.nn.Parameter(torch.empty(lowrank, wide, dtype=dtype))
        self.input_mix_weight_up = torch.nn.Parameter(torch.empty(wide, lowrank, dtype=dtype))

        for p in (self.hc_norm_weight, self.input_mix_weight_down, self.input_mix_weight_up):
            p.sequence_parallel = config.sequence_parallel
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.input_mix_weight_down)
            torch.nn.init.xavier_uniform_(self.input_mix_weight_up)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # [FUNCTION DOC] Run the fused mix kernel without an inject weight and return only the mixed
        # (aggregated) tensor of shape sequence, batch, hidden; the post weights are discarded.
        mixed, _ = hc_mix_inject(
            hidden_states,
            self.hc_norm_weight,
            self.input_mix_weight_down,
            self.input_mix_weight_up,
            None,
            self.n,
            self.norm_eps,
        )
        return mixed


class Qwen38NextPLEHyperConnection(Qwen38NextHyperConnection):
    # [CLASS DOC] Hyper-connection for the attention slot of the layer that hosts per-layer
    # embeddings (PLE). Before the normal read it adds the PLE increment (computed from the
    # published n-gram ids side channel) to the hidden states. Under sequence parallelism the
    # hidden states are all-gathered to full length for the PLE computation and the local slice is
    # taken back afterwards. Under activation recompute a first-in-first-out queue of published
    # batches is kept per module (created lazily).
    """Attention-site HC for the PLE layer: applies the PLE increment (full-seq
    under SP) before the read."""

    def __init__(self, config: TransformerConfig, layer_number: int, **kwargs):
        # [FUNCTION DOC] Runs the base constructor (extra keyword arguments pass through), looks up
        # the tensor-parallel group (falling back to none if parallel state is not initialised, e.g.
        # in shape audits), and creates the PLE module for this layer with that group.
        super().__init__(config, layer_number, **kwargs)

        tp_group = None
        try:
            tp_group = get_tensor_model_parallel_group()
        except AssertionError:
            pass  # not initialised (shape audits); falls back to unsharded
        self.ple = Qwen38NextPLE(config, layer_number=layer_number, tp_group=tp_group)

    def _resolve_ple_batch(self):
        # [FUNCTION DOC] HELPER: return the (n-gram ids, cumulative sequence lengths) pair for this
        # forward. Lazily creates the recompute queue. If currently inside a checkpointed region
        # with gradients enabled (the recompute pass), pops the oldest queued batch (raises a runtime
        # error if empty, meaning the original pass did not enqueue or order diverged). Otherwise
        # reads the currently published batch; if inside checkpointing (the original pass with no
        # grad), also appends it to the queue for later recompute.
        """The published batch, made safe under activation recompute.

        The recompute pass runs in backward, after the side channel was cleared or
        overwritten by a later microbatch, so the checkpointed pass enqueues its batch
        and the recompute pass pops it. FIFO matches non-interleaved 1F1B.
        """
        if not hasattr(self, "_ple_recompute_fifo"):
            self._ple_recompute_fifo = []

        if is_checkpointing() and torch.is_grad_enabled():
            if not self._ple_recompute_fifo:
                raise RuntimeError(
                    "PLE recompute ran with no queued n-gram batch; the checkpointed "
                    "original pass did not enqueue (or the recompute order diverged "
                    "from FIFO, e.g. interleaved VPP)."
                )
            return self._ple_recompute_fifo.pop(0)

        batch = current_ple_batch()
        if is_checkpointing():
            self._ple_recompute_fifo.append(batch)
        return batch

    def _apply_ple(self, hidden_states, ngram_ids, cu_seqlens):
        # [FUNCTION DOC] HELPER: add the PLE increment to hidden_states (sequence, batch, features).
        # ngram_ids may be 3-dimensional (batch, sequence, heads), in which case its batch and
        # sequence must match the hidden states (else runtime error) and it is transposed to
        # sequence-first; both ids and states are flattened to rows over sequence times batch, the
        # PLE module is called with the rows and cumulative lengths, its output must match the
        # flattened state's shape (asserted), and it is added to the hidden states. Returns the sum.
        seq, batch = hidden_states.shape[0], hidden_states.shape[1]
        if ngram_ids.dim() == 3:
            if ngram_ids.shape[:2] != (batch, seq):
                raise RuntimeError(
                    f"PLE ngram ids are [B, S, heads]={tuple(ngram_ids.shape)}, which "
                    f"does not match the hidden state's B={batch}, S={seq}"
                )
            ngram_ids = ngram_ids.transpose(0, 1)
        flat_ids = ngram_ids.reshape(seq * batch, ngram_ids.shape[-1])
        flat_state = hidden_states.reshape(seq * batch, hidden_states.shape[-1])

        increment = self.ple(flat_state, flat_ids, cu_seqlens)

        assert (
            increment.shape == flat_state.shape
        ), f"PLE increment {tuple(increment.shape)} != state {tuple(flat_state.shape)}"
        hidden_states = hidden_states + increment.view_as(hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: Tensor,
        mhc_recompute_manager=None,
        output_slot=None,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
        # [FUNCTION DOC] Resolve the PLE batch, add the PLE increment to the hidden states (with
        # the all-gather / local-slice dance when sequence parallelism is enabled and TP size > 1,
        # without gradient to the gathered input's output grad being TP-reduced), then delegate to
        # the base read step with the same recompute manager and output slot arguments (which will
        # raise if either is set). Returns the base 4-tuple.
        ngram_ids, cu_seqlens = self._resolve_ple_batch()

        sp_size = 1
        if getattr(self.config, "sequence_parallel", False):
            sp_size = get_tensor_model_parallel_world_size()
        if sp_size > 1:
            sp_rank = get_tensor_model_parallel_rank()
            local_seq = hidden_states.shape[0]
            full = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=get_tensor_model_parallel_group(),
            )
            updated = self._apply_ple(full, ngram_ids, cu_seqlens)
            hidden_states = updated[sp_rank * local_seq : (sp_rank + 1) * local_seq]
        else:
            hidden_states = self._apply_ple(hidden_states, ngram_ids, cu_seqlens)
        return super().forward(
            hidden_states,
            mhc_recompute_manager=mhc_recompute_manager,
            output_slot=output_slot,
        )
