# [MODULE DOC] Replicated (non-head-sharded) Gated DeltaNet for Qwen3.8-Next. Every tensor-parallel
# rank holds the full set of heads and weights and runs the whole layer, wrapped in the
# HuggingFace-attention shim (which handles gathering / scattering sequences around the HF-style
# forward). Contains the pure-torch-module GDN (ReplicatedGatedDeltaNet) and the Megatron-facing
# wrapper (ReplicatedGDNAttention). The fla import for the norm and conv is optional at import
# time (skipped silently when fla is missing).
"""Qwen3.8-Next runs its GDN replicated on every TP rank through the HF-attention shim.
Moved from qwen3_5.py when Qwen3.5 / Qwen3-Next switched to the head-sharded layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
except ImportError:
    pass

from miles.backends.megatron_utils.fp32_param_utils import mark_param_dtype
from miles_plugins.models.cp_utils import build_gdn_cp_context
from miles_plugins.models.hf_attention import HuggingfaceAttention
from miles_plugins.models.linear_attn import gdn_kernel
from miles_plugins.models.qwen3_5 import _get_text_config


# Adapted from Qwen3NextGatedDeltaNet but with separate in_proj_qkv and in_proj_z
class ReplicatedGatedDeltaNet(nn.Module):
    # [CLASS DOC] Full (unsharded) Gated DeltaNet layer with variable-length (packed sequence) and
    # context-parallel support. Parameters: separate bias-free linears in_proj_qkv (flat layout:
    # all q, then all k, then all v), in_proj_z (output gate), in_proj_b (beta logits per value head),
    # in_proj_a (decay input per value head); out_proj; fla short convolution over the full qkv width;
    # dt_bias (ones, one per value head); A_log (fp32, log of uniform samples from 0 to 16); fla fused
    # gated RMS norm over the value head dim. The output-gate activation of the norm comes from the
    # config's output gate type (sigmoid for Qwen3.8-Next) falling back to the hidden activation.
    """
    Qwen3.5 GatedDeltaNet with varlen support.
    Unlike Qwen3Next which uses a combined in_proj_qkvz, Qwen3.5 uses
    separate in_proj_qkv (for Q,K,V) and in_proj_z (for Z).
    """

    def __init__(self, config, layer_idx: int, args=None):
        # [FUNCTION DOC] Build all sub-modules from the HuggingFace text config. The kernel backend
        # is taken from args' linear attention backend attribute (default fla if args is missing or
        # lacks it) and resolved eagerly (may raise ImportError / value error). Reads head counts and
        # dims (value heads, key heads, key / value head dims), conv kernel size, hidden activation,
        # rms norm epsilon. Key dim and value dim are head dim times head count; the conv and qkv
        # projection width is twice key dim plus value dim. The A_log parameter is marked as an
        # fp32 parameter (held in fp32 even when the HF checkpoint is bf16, to match the rollout
        # engine). The norm is created on the current CUDA device in the config's dtype.
        # layer_idx is stored but otherwise unused here.
        # KNOWN ISSUE: when config.dtype is None the fallback calls a torch function for the current
        # default dtype that does not exist in torch (the real one is get_default_dtype), so that
        # path would raise an attribute error. Also, the activation module (self.act) and the
        # activation name stored for the conv are not used by forward.
        super().__init__()
        self.gdn_backend = getattr(args, "linear_attention_backend", "fla")
        self.chunk_gated_delta_rule = gdn_kernel(self.gdn_backend)
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        # Qwen3.8-Next gates the output norm with sigmoid while hidden_act stays silu
        self.output_gate_activation = getattr(config, "output_gate_type", None) or config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ShortConvolution(
            hidden_size=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
        )

        # Separate projections for QKV and Z (unlike Qwen3Next which combines QKVZ)
        projection_size_qkv = self.key_dim * 2 + self.value_dim
        projection_size_z = self.value_dim
        self.in_proj_qkv = nn.Linear(self.hidden_size, projection_size_qkv, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, projection_size_z, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # time step projection
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        # Qwen3.5 ships A_log in fp32; Qwen3.6 reverted the HF weight to bf16.
        # We still hold it in fp32 here for engineering simplicity (no special
        # path for 3.6) — this matches the SGLang implementation, which also
        # keeps A_log in fp32 regardless of the HF dtype.
        self.A_log = nn.Parameter(torch.log(A).to(torch.float32))
        mark_param_dtype(self.A_log, torch.float32)

        # HF stores this norm in fp32, but unlike A_log its precision impact is
        # negligible and sglang runs it in bf16 on the rollout side — follow
        # config.dtype (bf16) to stay equivalent to rollout.
        self.norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=self.output_gate_activation,
            device=torch.cuda.current_device(),
            dtype=config.dtype if config.dtype is not None else torch.get_current_dtype(),
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ):
        # [FUNCTION DOC] Forward over hidden_states of shape batch, sequence, hidden; cu_seqlens are
        # the packed-sequence boundaries (global when context parallelism is on). Returns a tensor of
        # shape batch, sequence, hidden.
        # Steps: build a context-parallel context (None when CP is off); project qkv, z, b, a (z
        # reshaped per value head); run the conv on the flat qkv (with the CP context's own
        # boundaries when present); split flat into q, k, v and reshape to heads; beta is the
        # sigmoid of b; the log-decay g is negative exp of A_log (fp32) times softplus of (a in fp32
        # plus dt_bias); if value heads outnumber key heads, repeat q and k along the head dim by the
        # ratio (interleaved); run the chunked gated delta rule with in-kernel q/k L2 norm: with a CP
        # context only the fla backend is allowed (else not-implemented error) and the CP context's
        # boundaries plus the context are passed; without CP, the flashqla backend first makes
        # inputs contiguous, and no initial / final state is used. Finally flatten to rows of one
        # value head, apply the fused gated norm with z as the gate, restore shape and apply out_proj.
        batch_size, seq_len, _ = hidden_states.shape

        cp_context = build_gdn_cp_context(self, cu_seqlens, hidden_states.device)

        # Projections (flat layout: [Q_all, K_all, V_all])
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Convolution on the flat QKV (pass cp_context for boundary handling)
        conv_cu_seqlens = cp_context.cu_seqlens if cp_context is not None else cu_seqlens
        mixed_qkv, _ = self.conv1d(
            x=mixed_qkv,
            cu_seqlens=conv_cu_seqlens,
            cp_context=cp_context,
        )

        # Split into Q, K, V (flat split, matching HF layout)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if cp_context is not None:
            if self.gdn_backend != "fla":
                raise NotImplementedError(
                    f"GDN context parallelism requires the 'fla' backend, got {self.gdn_backend!r}."
                )
            core_attn_out, _ = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cp_context.cu_seqlens,
                cp_context=cp_context,
            )
        else:
            if self.gdn_backend == "flashqla":
                query = query.contiguous()
                key = key.contiguous()
                value = value.contiguous()
                g = g.contiguous()
                beta = beta.contiguous()
            core_attn_out, _ = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


class ReplicatedGDNAttention(HuggingfaceAttention):
    # [CLASS DOC] Megatron self-attention module that hosts a ReplicatedGatedDeltaNet through the
    # HuggingFace-attention shim. Adds an input RMS norm; the shim calls hf_forward with the
    # packed-sequence params.
    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
        name: str | None = None,
    ):
        # [FUNCTION DOC] Runs the shim's constructor (which loads the HF config), then narrows the
        # HF config to its text sub-config if nested (VLM), mutates it to request the
        # flash-attention-2 attention implementation, builds the replicated GDN for the HF layer
        # index with args, and creates the input norm: the HF Qwen3-Next RMS norm when importable,
        # otherwise torch's built-in RMS norm, both over hidden size with the config's epsilon.
        super().__init__(
            args,
            config,
            layer_number,
            cp_comm_type,
            pg_collection,
            name=name,
        )
        # Qwen3.5 is a VLM model with nested text_config
        self.hf_config = _get_text_config(self.hf_config)
        self.hf_config._attn_implementation = "flash_attention_2"

        self.linear_attn = ReplicatedGatedDeltaNet(self.hf_config, self.hf_layer_idx, args=args)

        # Use a simple RMSNorm
        try:
            from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

            self.input_layernorm = Qwen3NextRMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)
        except ImportError:
            from torch.nn import RMSNorm

            self.input_layernorm = RMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)

    def hf_forward(self, hidden_states, packed_seq_params):
        # [FUNCTION DOC] Apply the input norm, then the replicated GDN with the packed params' query
        # cumulative sequence lengths as boundaries. Returns the layer output (shape of the input).
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )
        return hidden_states
