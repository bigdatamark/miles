# [MODULE DOC] Tensor-parallel, head-sharded linear-attention layers (Gated DeltaNet and Kimi Delta
# Attention) packaged as Megatron modules, plus the helpers that run fla / FlashQLA kernels.
# Layering: LinearAttentionLayer is the module plugged in as a transformer layer's self-attention. It
# owns the input norm and all communication (tensor-parallel in/out collectives, sequence-parallel
# variants, and the context-parallel layout conversion). LinearAttention (abstract) owns this rank's
# heads: projections, short causal convolution(s), the recurrence kernel, and the gated RMS norm.
# GatedDeltaNet and KimiDeltaAttention are the two recurrence families; model plugins subclass them
# to declare their projections under the HuggingFace parameter names.
# Module constants: WEIGHT_LAYOUT_VERSION is stored in a buffer on each LinearAttention so that
# checkpoints record the weight layout; INT32_ELEMENTS is the largest element count the fla conv
# kernel can index; the chunk-elements constant (2**30) and channel-alignment constant (128) control
# how oversized conv inputs are split along channels.
# Importing the fla norm modules is optional at import time (silently skipped when fla is missing);
# the failure then surfaces when a LinearAttention is constructed or run.
"""Head-sharded linear attention (GDN, KDA) as Megatron modules.

:class:`LinearAttentionLayer` is the ``self_attention`` drop-in: input norm, one TP collective in
(identity / all-reduce, or all-gather / reduce-scatter under sequence parallelism), the context-parallel
zigzag relayout, and the row-parallel ``out_proj`` collective out. :class:`LinearAttention` runs this
rank's heads: the model's input projections, the short conv(s) over q/k/v (one over group-major rows
for GDN, see ``megatron_to_hf.linear_attn_layout``; one per tensor for KDA), the family's recurrence
through fla, and a gated RMSNorm whose
replicated weight has its gradient summed across TP. A family subclass supplies the recurrence; a model
subclass declares the projections under the HF names, plain bf16 linears on sharded parameters, so
``--fp8`` training leaves this layer in bf16.
"""

from __future__ import annotations

import functools
import logging
import os
from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint

from miles.backends.megatron_utils.fp32_param_utils import mark_param_dtype
from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads
from miles_plugins.models.cp_utils import build_fla_cp_context, packed_shard_to_zigzag, zigzag_to_packed_shard

try:
    from fla.modules import FusedRMSNormGated
    from fla.modules.fused_norm_gate import rms_norm_gated
except ImportError:
    pass

logger = logging.getLogger(__name__)

WEIGHT_LAYOUT_VERSION = 1
INT32_ELEMENTS = 2**31 - 1
_CHUNK_ELEMENTS = 2**30
_CHANNEL_ALIGN = 128


def gdn_kernel(backend: str):
    # [FUNCTION DOC] Resolve the chunked gated-delta-rule kernel callable for a backend name.
    # Input: backend, either "fla" (flash-linear-attention) or "flashqla" (the FlashQLA package).
    # Returns the kernel function (imported lazily on each call; not cached).
    # Raises ImportError with an install hint if the chosen backend package is missing, and a value
    # error for any other backend name.
    if backend == "fla":
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except ImportError as exc:
            raise ImportError("GDN backend 'fla' requires flash-linear-attention.") from exc
        return chunk_gated_delta_rule

    if backend == "flashqla":
        try:
            from flash_qla import chunk_gated_delta_rule
        except ImportError as exc:
            raise ImportError(
                "GDN backend 'flashqla' requires FlashQLA. Install it from https://github.com/QwenLM/FlashQLA."
            ) from exc
        return chunk_gated_delta_rule

    raise ValueError(f"Unsupported GDN backend: {backend}")


@functools.cache
def kda_kernel():
    # [FUNCTION DOC] Return fla's chunked Kimi-delta-attention kernel (cached after the first call).
    # Side effects: on a CUDA device with compute capability major version 10 or higher (Blackwell),
    # sets the FLA_TILELANG environment variable to "0" only if the user has not already set it, so
    # the Triton backward kernel is used; logs the resulting FLA_TILELANG value once.
    # Raises ImportError if fla lacks the kda ops (needs flash-linear-attention 0.5 or newer).
    """fla's chunk_kda. On Blackwell its backward runs the Triton dqkg kernel (FLA_TILELANG=0), which
    beats fla's TileLang one there at every tiling; an explicit FLA_TILELANG wins."""
    try:
        from fla.ops.kda import chunk_kda
    except ImportError as exc:
        raise ImportError("KDA requires flash-linear-attention >= 0.5 (fla.ops.kda).") from exc
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
        os.environ.setdefault("FLA_TILELANG", "0")
    logger.info(f"KDA backward: FLA_TILELANG={os.environ.get('FLA_TILELANG', 'unset')}")
    return chunk_kda


def gdn_recurrence(q, k, v, beta_logits, decay, A_log, dt_bias, *, backend, cu_seqlens, cp_context):
    # [FUNCTION DOC] Run the Gated DeltaNet recurrence for this rank's heads; returns the core output
    # with shape batch, sequence, value heads, value head dim.
    # Steps: beta is the sigmoid of beta_logits; the log-decay gate g is the negative exponential of
    # A_log (in fp32) times softplus of (decay in fp32 plus dt_bias), one scalar per value head per
    # token. For the flashqla backend, q, k, v, g and beta are first made contiguous. The kernel is
    # called with no initial state, no final state output, in-kernel L2 normalisation of q and k,
    # the packed-sequence boundaries cu_seqlens, and the context-parallel context only when it is
    # not None.
    # Raises NotImplementedError if a context-parallel context is given with a backend other than fla.
    """q/k ``[b, s, G, hk]``, v ``[b, s, H, hv]`` (fla groups value heads per key head), decay ``[b, s, H]``
    -> ``[b, s, H, hv]``."""
    if cp_context is not None and backend != "fla":
        raise NotImplementedError(f"GDN context parallelism requires the 'fla' backend, got {backend!r}.")
    beta = beta_logits.sigmoid()
    g = -A_log.float().exp() * F.softplus(decay.float() + dt_bias)
    if backend == "flashqla":
        q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    out, _ = gdn_kernel(backend)(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        **({"cp_context": cp_context} if cp_context is not None else {}),
    )
    return out


def kda_recurrence(q, k, v, beta_logits, decay, A_log, dt_bias, *, gate_lower_bound, cu_seqlens, cp_context):
    # [FUNCTION DOC] Run the Kimi Delta Attention recurrence; returns the core output with shape
    # batch, sequence, value heads, value head dim.
    # The decay (low-rank forget-gate output) is reshaped to the value tensor's shape and used as the
    # raw gate; the kernel applies A_log and dt_bias itself (in-kernel gating, safe-gate mode with
    # gate_lower_bound as the lower clamp). beta is the fp32 sigmoid of beta_logits. No initial state,
    # no final state output, in-kernel L2 normalisation of q and k, and the transposed state layout.
    # Sequence boundaries: when a context-parallel context is given it is passed (it already carries
    # the boundaries); otherwise cu_seqlens is passed.
    """q/k ``[b, s, H, hk]``, v ``[b, s, H, hv]``, decay ``[b, s, H * hv]`` (the low-rank forget gate,
    gated inside the kernel) -> ``[b, s, H, hv]``."""
    boundaries = {"cp_context": cp_context} if cp_context is not None else {"cu_seqlens": cu_seqlens}
    out, _ = kda_kernel()(
        q=q,
        k=k,
        v=v,
        g=decay.reshape(v.shape),
        beta=beta_logits.float().sigmoid(),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=gate_lower_bound,
        transpose_state_layout=True,
        **boundaries,
    )
    return out


@functools.cache
def short_conv_backend() -> str:
    # [FUNCTION DOC] Choose (once, cached) the fla causal-conv backend name for the short convs.
    # Uses the FLA_CONV_BACKEND environment variable if set. Otherwise returns "mix" when the
    # causal-conv1d CUDA backward function can be imported, else "triton". Logs the choice.
    """fla's "mix" (Triton forward, causal-conv1d CUDA backward) when causal-conv1d's backward imports,
    Triton otherwise; ``FLA_CONV_BACKEND`` overrides, as it does for fla's own ShortConvolution."""
    backend = os.environ.get("FLA_CONV_BACKEND")
    if backend is None:
        try:
            from causal_conv1d.cpp_functions import causal_conv1d_bwd_function  # noqa: F401

            backend = "mix"
        except ImportError:
            backend = "triton"
    logger.info(f"Linear-attention short conv backend: {backend}")
    return backend


class _ContiguousGrad(torch.autograd.Function):
    # [CLASS DOC] HELPER: custom autograd function that is the identity in the forward pass and
    # makes the incoming gradient contiguous in the backward pass. Used only on the chunked conv path
    # of causal_short_conv.
    """Identity whose backward hands on a contiguous gradient. The gradient of one channel chunk of a
    concatenation is a strided view whose row stride is the full width; fla's conv backward indexes it
    with int32 and overflows past 2**31 elements."""

    @staticmethod
    def forward(ctx, x):
        # [FUNCTION DOC] HELPER: returns the input tensor unchanged (no copy, nothing saved).
        return x

    @staticmethod
    def backward(ctx, grad):
        # [FUNCTION DOC] HELPER: returns the gradient as a contiguous tensor (a copy only if it was
        # strided).
        return grad.contiguous()


def causal_short_conv(x, weight, activation, backend, cu_seqlens=None, cp_context=None):
    # [FUNCTION DOC] Apply fla's depthwise causal 1-D convolution (no bias) to x.
    # Inputs: x of shape batch, sequence, channels; weight of shape channels by kernel size;
    # activation name and fla conv backend name; optional packed-sequence boundaries cu_seqlens and
    # context-parallel context, both forwarded to fla. Returns the convolved tensor, same shape as x.
    # Algorithm: if total elements (batch * sequence * channels) fit in int32, one fla call is made
    # and its first output taken. Otherwise the channels are split into chunks whose width is
    # (2**30 divided by the token count) rounded down to a multiple of 128, but never below 128;
    # x and weight are split identically along channels, each chunk (made contiguous) is convolved
    # separately and passed through the contiguous-gradient helper, and the results are
    # concatenated along channels. Exact because the conv is per channel.
    """Depthwise causal conv of ``x`` ``[b, s, C]`` with ``weight`` ``[C, K]`` through fla. fla indexes with
    int32, so inputs past 2**31 elements (long sequences at low TP) run as channel chunks of at most
    2**30; the conv is per channel, so this is exact."""
    from fla.modules.conv.causal_conv1d import causal_conv1d

    tokens = x.shape[0] * x.shape[1]
    kwargs = {
        "bias": None,
        "activation": activation,
        "backend": backend,
        "cu_seqlens": cu_seqlens,
        "cp_context": cp_context,
    }
    if tokens * x.shape[-1] <= INT32_ELEMENTS:
        return causal_conv1d(x=x, weight=weight, **kwargs)[0]
    width = max(_CHANNEL_ALIGN, _CHUNK_ELEMENTS // tokens // _CHANNEL_ALIGN * _CHANNEL_ALIGN)
    chunks = [
        _ContiguousGrad.apply(causal_conv1d(x=chunk.contiguous(), weight=chunk_weight, **kwargs)[0])
        for chunk, chunk_weight in zip(x.split(width, dim=-1), weight.split(width), strict=True)
    ]
    return torch.cat(chunks, dim=-1)


class ShardedShortConv(nn.Conv1d):
    # [CLASS DOC] Torch 1-D convolution subclass used purely as a weight holder: depthwise (one
    # group per channel), no bias, whose forward pass bypasses torch's conv and calls the fla causal
    # conv with a fixed SiLU activation. The weight is tagged as tensor-parallel partitioned along
    # dimension 0 (the channel dimension) so Megatron treats it as sharded across the TP group.
    """Depthwise causal conv (SiLU, no bias) over this rank's channels, holding the TP-sharded weight.
    Built like fla's ``ShortConvolution``, so weights and checkpoints match it."""

    def __init__(self, channels: int, kernel_size: int, tp_group, device=None, dtype=None):
        # [FUNCTION DOC] Build the conv with the local channel count as both input and output
        # channels, groups equal to channels, no bias, torch padding of kernel size minus one (the
        # padding is unused by forward). Stores the activation name "silu", the backend name from
        # short_conv_backend, and the TP group; marks the weight as TP-partitioned on dim 0, stride 1.
        # The weight is initialised by torch's default conv init (not a Megatron init method).
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )
        self.activation = "silu"
        self.backend = short_conv_backend()
        self.tp_group = tp_group
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def forward(self, x: torch.Tensor, cu_seqlens=None, cp_context=None) -> torch.Tensor:
        # [FUNCTION DOC] Convolve x (batch, sequence, local channels) with this module's weight (the
        # middle singleton dimension dropped to channels by kernel size) via causal_short_conv using
        # the stored activation and backend. Returns a tensor of the same shape as x.
        return causal_short_conv(x, self.weight[:, 0], self.activation, self.backend, cu_seqlens, cp_context)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None):
        # [FUNCTION DOC] Megatron distributed-checkpoint hook: returns a mapping of the module's
        # state (just the weight, with live variables kept) to sharded tensors, declaring the weight
        # as partitioned along axis 0 across the TP group, with the data-parallel-by-context-parallel
        # group taken from the metadata (the metadata is first completed with that group if absent).
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"weight": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


class Projections(NamedTuple):
    # [CLASS DOC] Named tuple bundling the outputs of a model's input projections: qkv (a single
    # tensor or a triple, as the family's convolve step expects), the output gate, the beta logits,
    # and the decay input to the recurrence. Pure data container; no behavior.
    """This rank's projections: ``qkv`` as :meth:`LinearAttention.convolve` takes it (one group-major
    ``[b, s, Gl * group_qkv_dim]`` tensor for GDN, a ``(q, k, v)`` tuple for KDA), ``gate``
    ``[b, s, Hl * hv]``, ``beta_logits`` ``[b, s, Hl]``, ``decay`` ``[b, s, Hl]`` (GDN) or
    ``[b, s, Hl * hv]`` (KDA)."""

    qkv: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    gate: torch.Tensor
    beta_logits: torch.Tensor
    decay: torch.Tensor


class LinearAttention(MegatronModule, ABC):
    # [CLASS DOC] Abstract base for one rank's share of a linear-attention block. Pipeline in
    # forward: project -> convolve -> recurrence -> gated RMS norm. Subclasses must implement
    # building the projections, projecting, and the recurrence; may override how convolutions are
    # built and applied. Class attributes: dt_bias_per_channel (dt_bias has one value per local value
    # channel instead of per value head) and dt_bias_dtype (explicit dtype for dt_bias; None means the
    # config's parameter dtype).
    # Parameters owned: A_log (fp32, per local value head, initialised as the log of a uniform
    # sample from 1 to 16), dt_bias (initialised to ones), the norm weight (replicated across TP),
    # the out_proj weight (columns for this rank's value channels), plus whatever the subclass
    # builds. A registry of sharded parameter names to partition dims drives checkpoint sharding.
    """This rank's heads: projections -> conv -> recurrence -> gated norm. ``out_proj`` holds this
    rank's columns; :class:`LinearAttentionLayer` applies it and the TP reduction. ``dt_bias`` is one
    value per value head, or per value channel when the family sets ``dt_bias_per_channel``."""

    dt_bias_per_channel: bool = False
    dt_bias_dtype: torch.dtype | None = None

    def __init__(
        self,
        config,
        heads: LinearAttnHeads,
        conv_kernel_size: int,
        norm_eps: float,
        tp_group,
        norm_activation: str,
    ):
        # [FUNCTION DOC] Construct this rank's modules on the current CUDA device in the config's
        # parameter dtype. Inputs: Megatron transformer config, the global head geometry (localised
        # with the TP group size), conv kernel size, norm epsilon, TP process group, and the
        # activation name for the gated norm.
        # Order: register an int32 one-element buffer holding the weight layout version; build
        # projections (subclass hook); inside the CUDA RNG tracker fork, build convolutions
        # (subclass hook) and create A_log; mark A_log as fp32; create dt_bias (size per value
        # channel if dt_bias_per_channel else per value head; dtype dt_bias_dtype or the param dtype;
        # marked with that dtype when explicit); register A_log and dt_bias as TP-sharded on dim 0;
        # build the fla fused gated RMS norm over the value head dim; build out_proj (local value dim
        # to hidden size, no bias) initialised with the config's output-layer init method inside the
        # RNG fork, sharded on dim 1.
        super().__init__(config=config)
        self.tp_group = tp_group
        self.heads = heads
        self.local = heads.local(tp_group.size())
        self.conv_kernel_size = conv_kernel_size
        self.norm_eps = norm_eps
        self.norm_activation = norm_activation
        device = torch.cuda.current_device()
        dtype = config.params_dtype

        self._sharded_params: dict[str, int] = {}
        self.register_buffer(
            "weight_layout_version", torch.tensor([WEIGHT_LAYOUT_VERSION], dtype=torch.int32, device=device)
        )
        self._build_projections()
        with get_cuda_rng_tracker().fork():
            self._build_convolutions()
            self.A_log = nn.Parameter(
                torch.empty(self.local.num_v_heads, dtype=torch.float32, device=device).uniform_(1, 16).log_()
            )
        mark_param_dtype(self.A_log, torch.float32)
        dt_bias_size = self.local.value_dim if self.dt_bias_per_channel else self.local.num_v_heads
        self.dt_bias = nn.Parameter(torch.ones(dt_bias_size, dtype=self.dt_bias_dtype or dtype, device=device))
        if self.dt_bias_dtype is not None:
            mark_param_dtype(self.dt_bias, self.dt_bias_dtype)
        self._mark_sharded("A_log", self.A_log, dim=0)
        self._mark_sharded("dt_bias", self.dt_bias, dim=0)
        self.norm = FusedRMSNormGated(
            heads.head_v_dim, eps=norm_eps, activation=norm_activation, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(self.local.value_dim, config.hidden_size, bias=False, device=device, dtype=dtype)
        with get_cuda_rng_tracker().fork():
            config.output_layer_init_method(self.out_proj.weight)
        self._mark_sharded("out_proj.weight", self.out_proj.weight, dim=1)

    def sharded_conv(self, channels: int) -> ShardedShortConv:
        # [FUNCTION DOC] Factory for a ShardedShortConv over the given local channel count, using
        # this module's conv kernel size and TP group, on the current CUDA device in the config's
        # parameter dtype.
        return ShardedShortConv(
            channels, self.conv_kernel_size, self.tp_group, torch.cuda.current_device(), self.config.params_dtype
        )

    def _build_convolutions(self) -> None:
        # [FUNCTION DOC] HELPER: default convolution setup: a single conv named conv1d over all local
        # key groups times the per-group q/k/v width (the group-major layout used by GDN).
        # Subclasses override this to create different convs.
        self.conv1d = self.sharded_conv(self.local.num_k_heads * self.local.group_qkv_dim)

    def convolve(self, qkv, cu_seqlens, cp_context):
        # [FUNCTION DOC] Default (group-major) convolution step. Input: the single qkv tensor of shape
        # batch, sequence, local groups times group width, plus packed-sequence boundaries and the
        # context-parallel context. Runs the one conv, views the result as batch, sequence, local key
        # groups, width, and splits the last dim into q (key head dim), k (key head dim) and v (the
        # group's value width). Returns q and k per group, and v reshaped to per value head (a view,
        # since a group's value heads are contiguous).
        """-> q, k ``[b, s, Gl, hk]``, v ``[b, s, Hl, hv]``. One conv over the group-major q/k/v, split per
        group; value heads of a group are contiguous, so ``v`` is a view."""
        batch, seq_len, _ = qkv.shape
        local = self.local
        mixed = self.conv1d(qkv, cu_seqlens=cu_seqlens, cp_context=cp_context)
        q, k, v = mixed.view(batch, seq_len, local.num_k_heads, -1).split(
            [local.head_k_dim, local.head_k_dim, local.group_value_dim], dim=-1
        )
        return q, k, v.reshape(batch, seq_len, local.num_v_heads, local.head_v_dim)

    def _mark_sharded(self, name: str, param: nn.Parameter, dim: int) -> None:
        # [FUNCTION DOC] HELPER: tag a parameter as tensor-parallel partitioned along the given dim
        # (stride 1) for Megatron, and record its name and dim in the registry used by
        # sharded_state_dict. The name is relative to this module (e.g. a submodule weight path).
        set_tensor_model_parallel_attributes(param, True, dim, 1)
        self._sharded_params[name] = dim

    def sharded_linear(self, name: str, input_size: int, local_output_size: int) -> nn.Linear:
        # [FUNCTION DOC] Build a bias-free plain linear layer holding this rank's output rows (the
        # weight is local output size by input size), on the current CUDA device in the config's
        # parameter dtype, initialised with the config's init method inside the CUDA RNG tracker
        # fork, and registered as sharded on dim 0 under "<name>.weight". No communication.
        """This rank's row shard of a head-sharded projection; the input already went through the TP
        collective, so the linear itself communicates nothing."""
        linear = nn.Linear(
            input_size,
            local_output_size,
            bias=False,
            device=torch.cuda.current_device(),
            dtype=self.config.params_dtype,
        )
        with get_cuda_rng_tracker().fork():
            self.config.init_method(linear.weight)
        self._mark_sharded(f"{name}.weight", linear.weight, dim=0)
        return linear

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None):
        # [FUNCTION DOC] Megatron distributed-checkpoint hook. Starts from the base class result,
        # then overrides/adds sharded-tensor entries for every parameter registered as sharded
        # (with its recorded partition dim, over the TP group and the data-parallel-by-context-
        # parallel group from the metadata). Returns the combined mapping.
        sharded = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        params = dict(self.named_parameters())
        sharded.update(
            make_sharded_tensors_for_checkpoint(
                {name: params[name] for name in self._sharded_params},
                prefix,
                self._sharded_params,
                sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
        return sharded

    @abstractmethod
    def _build_projections(self) -> None: ...
        # [FUNCTION DOC] HELPER: abstract hook. Subclasses create their input projection layers
        # (via sharded_linear or plain linears) as attributes during construction.


    @abstractmethod
    def project(self, x: torch.Tensor) -> Projections: ...
        # [FUNCTION DOC] Abstract hook. Subclasses map the layer input (batch, sequence, hidden,
        # after the TP collective) to the Projections tuple for this rank's heads.


    @abstractmethod
    def recurrence(self, q, k, v, beta_logits, decay, cu_seqlens, cp_context) -> torch.Tensor:
        # [FUNCTION DOC] Abstract hook. Subclasses run the family's recurrence kernel on the
        # convolved q, k, v and the projection outputs, returning the core output per value head.
        """q/k ``[b, s, Gl, hk]``, v ``[b, s, Hl, hv]`` -> ``[b, s, Hl, hv]``."""

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None, cp_context=None) -> torch.Tensor:
        # [FUNCTION DOC] Run this rank's heads on x (batch, sequence, hidden), already past the TP
        # input collective. Steps: project; convolve; recurrence; then the gated RMS norm, applied on
        # rows of one value head each: the output gate is the norm's gate input, the norm weight goes
        # through the TP copy op (identity forward, sums gradient across TP in backward), using the
        # configured activation and epsilon. Returns batch, sequence, local value dim (before
        # out_proj and before any reduction).
        """x ``[b, s, hidden]``, TP collective already applied -> ``[b, s, local value_dim]``."""
        batch, seq_len, _ = x.shape
        qkv, gate, beta_logits, decay = self.project(x)
        q, k, v = self.convolve(qkv, cu_seqlens, cp_context)
        core = self.recurrence(q, k, v, beta_logits, decay, cu_seqlens, cp_context)
        weight = copy_to_tensor_model_parallel_region(self.norm.weight, group=self.tp_group)
        core = rms_norm_gated(
            core.reshape(-1, self.heads.head_v_dim),
            gate.reshape(-1, self.heads.head_v_dim),
            weight,
            self.norm.bias,
            self.norm_activation,
            eps=self.norm_eps,
        )
        return core.reshape(batch, seq_len, -1)


class GatedDeltaNet(LinearAttention):
    # [CLASS DOC] Gated DeltaNet family: implements the recurrence with gdn_recurrence (scalar decay
    # per value head). Uses the default group-major single convolution. Concrete models still
    # implement the projection building and projecting.
    """Gated DeltaNet: one softplus-gated decay per value head, through fla's or FlashQLA's chunked
    kernel. Models subclass it with their projections."""

    def __init__(self, config, heads, conv_kernel_size, norm_eps, tp_group, backend="fla", norm_activation="silu"):
        # [FUNCTION DOC] Validate the backend first (resolving the kernel raises ImportError or a
        # value error early, before any parameters are built), then run the base constructor, then
        # store the backend name. Defaults: backend fla, gated-norm activation silu.
        gdn_kernel(backend)
        super().__init__(config, heads, conv_kernel_size, norm_eps, tp_group, norm_activation)
        self.backend = backend

    def recurrence(self, q, k, v, beta_logits, decay, cu_seqlens, cp_context):
        # [FUNCTION DOC] Delegates to gdn_recurrence with this module's A_log, dt_bias and backend.
        return gdn_recurrence(
            q,
            k,
            v,
            beta_logits,
            decay,
            self.A_log,
            self.dt_bias,
            backend=self.backend,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )


class KimiDeltaAttention(LinearAttention):
    # [CLASS DOC] Kimi Delta Attention family with the Kimi-K3 HuggingFace projection names. Sets
    # dt_bias per value channel in fp32 and uses a sigmoid gated-norm activation. Three separate
    # head-sharded q/k/v projections each with their own conv, an output-gate projection, a beta
    # projection, and a low-rank forget gate (replicated down-projection to the value head dim, then
    # head-sharded up-projection to the value dim).
    """KDA in the Kimi-K3 HF layout: ``q_proj`` / ``k_proj`` / ``v_proj`` with one short conv each,
    ``g_proj`` (output gate), ``b_proj`` (beta), and the low-rank forget gate ``f_b_proj(f_a_proj(x))``,
    gated inside fla's kernel. One key head per value head, so the convs see contiguous per-head q / k / v
    and need no group-major permutation. ``f_a_proj`` is replicated and feeds head-sharded ``f_b_proj``,
    so its weight passes the TP copy op like the norm. The conv weights train in fp32, as Kimi's
    checkpoints store them."""

    dt_bias_per_channel = True
    dt_bias_dtype = torch.float32

    def __init__(self, config, heads, conv_kernel_size, norm_eps, tp_group, gate_lower_bound: float):
        # [FUNCTION DOC] Run the base constructor with the sigmoid gated-norm activation, then store
        # gate_lower_bound (the lower clamp for the kernel's safe-gate mode).
        super().__init__(config, heads, conv_kernel_size, norm_eps, tp_group, norm_activation="sigmoid")
        self.gate_lower_bound = gate_lower_bound

    def _build_projections(self):
        # [FUNCTION DOC] HELPER: create q_proj and k_proj (hidden to local key dim), v_proj and
        # g_proj (hidden to local value dim), b_proj (hidden to local value-head count), all
        # head-sharded; f_a_proj, a replicated bias-free linear from hidden to the value head dim
        # initialised with the config's init method (not registered as sharded, and not created
        # inside the RNG fork); and head-sharded f_b_proj from the value head dim to local value dim.
        hidden, local = self.config.hidden_size, self.local
        self.q_proj = self.sharded_linear("q_proj", hidden, local.key_dim)
        self.k_proj = self.sharded_linear("k_proj", hidden, local.key_dim)
        self.v_proj = self.sharded_linear("v_proj", hidden, local.value_dim)
        self.g_proj = self.sharded_linear("g_proj", hidden, local.value_dim)
        self.b_proj = self.sharded_linear("b_proj", hidden, local.num_v_heads)
        self.f_a_proj = nn.Linear(
            hidden,
            self.heads.head_v_dim,
            bias=False,
            device=torch.cuda.current_device(),
            dtype=self.config.params_dtype,
        )
        self.config.init_method(self.f_a_proj.weight)
        self.f_b_proj = self.sharded_linear("f_b_proj", self.heads.head_v_dim, local.value_dim)

    def _build_convolutions(self):
        # [FUNCTION DOC] HELPER: create three convs (q_conv1d and k_conv1d over the local key dim,
        # v_conv1d over the local value dim) and mark each conv weight as an fp32 parameter.
        local = self.local
        self.q_conv1d = self.sharded_conv(local.key_dim)
        self.k_conv1d = self.sharded_conv(local.key_dim)
        self.v_conv1d = self.sharded_conv(local.value_dim)
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            mark_param_dtype(conv.weight, torch.float32)

    def convolve(self, qkv, cu_seqlens, cp_context):
        # [FUNCTION DOC] Apply each of the three convs to its own tensor of the q, k, v triple, then
        # view q and k as batch, sequence, local key heads, key head dim, and v as batch, sequence,
        # local value heads, value head dim. Returns (q, k, v). No group-major permutation is needed.
        batch, seq_len = qkv[0].shape[:2]
        local = self.local
        q, k, v = (
            conv(t, cu_seqlens=cu_seqlens, cp_context=cp_context)
            for conv, t in zip((self.q_conv1d, self.k_conv1d, self.v_conv1d), qkv, strict=True)
        )
        return (
            q.view(batch, seq_len, local.num_k_heads, local.head_k_dim),
            k.view(batch, seq_len, local.num_k_heads, local.head_k_dim),
            v.view(batch, seq_len, local.num_v_heads, local.head_v_dim),
        )

    def project(self, x):
        # [FUNCTION DOC] Compute the Projections: qkv is the triple of q, k, v projections; gate is
        # g_proj; beta logits are b_proj; decay is the low-rank forget gate: x goes through the
        # replicated f_a_proj weight (passed through the TP copy op so its gradient is summed across
        # TP) with a plain linear, then through the head-sharded f_b_proj.
        f_a_weight = copy_to_tensor_model_parallel_region(self.f_a_proj.weight, group=self.tp_group)
        return Projections(
            (self.q_proj(x), self.k_proj(x), self.v_proj(x)),
            self.g_proj(x),
            self.b_proj(x),
            self.f_b_proj(F.linear(x, f_a_weight)),
        )

    def recurrence(self, q, k, v, beta_logits, decay, cu_seqlens, cp_context):
        # [FUNCTION DOC] Delegates to kda_recurrence with this module's A_log, dt_bias and gate
        # lower bound.
        return kda_recurrence(
            q,
            k,
            v,
            beta_logits,
            decay,
            self.A_log,
            self.dt_bias,
            gate_lower_bound=self.gate_lower_bound,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )


class LinearAttentionLayer(MegatronModule):
    # [CLASS DOC] Wrapper that makes a LinearAttention usable as a transformer layer's
    # self-attention. Handles the input norm, the tensor-parallel input collective (all-gather under
    # sequence parallelism, otherwise a copy op that all-reduces gradients), the context-parallel
    # layout conversion (zigzag to contiguous packed shards and back, skipped when allgather_cp is
    # set or CP size is 1), the fla CP context, and the output reduction (reduce-scatter under
    # sequence parallelism, otherwise all-reduce). Norm parameters are tagged with the
    # sequence-parallel flag so their gradients are reduced correctly.
    """``self_attention`` drop-in. ``allgather_cp``: the data pipeline already hands each CP rank a
    contiguous shard (``--allgather-cp``), so no zigzag relayout."""

    def __init__(
        self,
        config,
        linear_attn: LinearAttention,
        input_layernorm: nn.Module,
        pg_collection: ProcessGroupCollection,
        allgather_cp: bool,
    ):
        # [FUNCTION DOC] Store the TP and CP groups from the process-group collection (and CP size),
        # the config's sequence-parallel flag, the allgather_cp flag, the input norm and the wrapped
        # linear attention; set the sequence-parallel attribute on every input-norm parameter.
        super().__init__(config=config)
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.cp_size = self.cp_group.size()
        self.sequence_parallel = config.sequence_parallel
        self.allgather_cp = allgather_cp
        self.input_layernorm = input_layernorm
        self.linear_attn = linear_attn
        for param in self.input_layernorm.parameters():
            param.sequence_parallel = self.sequence_parallel

    def _global_cu_seqlens(self, hidden_states, packed_seq_params):
        # [FUNCTION DOC] HELPER: cumulative sequence lengths over the global (all-CP-ranks) token
        # axis. Returns the packed params' query cumulative lengths when present; otherwise an int32
        # tensor of zero and the total (local token count times CP size) on the hidden states'
        # device, i.e. one sequence spanning everything.
        if packed_seq_params is not None and packed_seq_params.cu_seqlens_q is not None:
            return packed_seq_params.cu_seqlens_q
        total = hidden_states.shape[0] * self.cp_size
        return torch.tensor([0, total], dtype=torch.int32, device=hidden_states.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset=None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        # [FUNCTION DOC] Self-attention forward. Input hidden_states has shape sequence, batch,
        # hidden (sequence-sharded across TP when sequence parallelism is on). All other arguments
        # (mask, key/value states, inference context, rotary embeddings, attention bias, sequence
        # length offset, extra keyword args) are accepted for interface compatibility and ignored;
        # only packed_seq_params is used (for query cumulative lengths).
        # Steps: input norm; TP input collective; global cumulative lengths; if CP size > 1 and not
        # allgather_cp, convert zigzag layout to packed contiguous shard; if CP size > 1, build the
        # fla CP context (kernel size from the wrapped module) and use its cumulative lengths for
        # the kernels; run the wrapped module on the batch-first transposed tensor and transpose
        # back; convert back to zigzag if converted; out_proj; TP output reduction.
        # Returns a pair of (output, None) where the second slot is the unused bias.
        x = self.input_layernorm(hidden_states)
        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True, group=self.tp_group)
        else:
            x = copy_to_tensor_model_parallel_region(x, group=self.tp_group)

        global_cu_seqlens = self._global_cu_seqlens(x, packed_seq_params)
        relayout = self.cp_size > 1 and not self.allgather_cp
        if relayout:
            x = zigzag_to_packed_shard(x, global_cu_seqlens, self.cp_group, self.cp_group.rank(), self.cp_size)
        cp_context = None
        cu_seqlens = global_cu_seqlens
        if self.cp_size > 1:
            cp_context = build_fla_cp_context(
                global_cu_seqlens, self.cp_group, self.linear_attn.conv_kernel_size, x.device
            )
            cu_seqlens = cp_context.cu_seqlens

        core = self.linear_attn(x.transpose(0, 1), cu_seqlens, cp_context).transpose(0, 1)

        if relayout:
            core = packed_shard_to_zigzag(core, global_cu_seqlens, self.cp_group, self.cp_group.rank(), self.cp_size)
        output = self.linear_attn.out_proj(core)
        if self.sequence_parallel:
            output = reduce_scatter_to_sequence_parallel_region(output, group=self.tp_group)
        else:
            output = reduce_from_tensor_model_parallel_region(output, group=self.tp_group)
        return output, None
