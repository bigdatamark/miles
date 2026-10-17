# [MODULE DOC] Linear-attention layer of the GLM-5.3 hybrid model using Kimi delta attention (KDA)
# from the flash-linear-attention package (optional import; the layer raises an import error at
# construction if missing). Provides config-field extraction helpers, the KDA mixing module
# (short convolution, per-head decay gate, delta-rule chunked kernel, gated RMS norm output), and a
# Megatron attention wrapper that runs it on packed sequences with hybrid context parallelism.
import torch
import torch.nn as nn
from megatron.core.transformer.module import mark_keep_in_fp32

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    FusedRMSNormGated = None
    ShortConvolution = None
    chunk_kda = None
    fused_kda_gate = None

from miles_plugins.models.cp_utils import build_gdn_cp_context
from miles_plugins.models.hf_attention import HuggingfaceAttention


def _get_text_config(hf_config):
    # [FUNCTION DOC] HELPER: Returns the nested text config of a multimodal HF config if present and
    # truthy, otherwise the config itself.
    return getattr(hf_config, "text_config", None) or hf_config


def _linear_attn_fields(text_config) -> dict:
    # [FUNCTION DOC] HELPER: Collects KDA hyper-parameters as a mapping with keys num_heads, head_dim,
    # conv_kernel_size, gate_lower_bound. Each value is looked up first in the linear-attention config
    # mapping (ignored if not a mapping) under its short key, falling back to a text-config attribute
    # (linear_num_heads default 64; linear_head_dim default 128; linear_conv_kernel_dim default 4; the
    # short-conv key is short_conv_kernel_size). The gate lower bound has no default: tries the mapping,
    # then the attribute of the same name, then a linear lower bound attribute; raises a value error if
    # all are missing. Counts become integers and the bound a float.
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    if not isinstance(linear_attn_config, dict):
        linear_attn_config = {}

    def field(key, attr, default):
        # [FUNCTION DOC] HELPER: Returns the mapping's value for the key if not none, else the text
        # config attribute with the given default.
        value = linear_attn_config.get(key)
        if value is None:
            value = getattr(text_config, attr, default)
        return value

    gate_lower_bound = field("gate_lower_bound", "gate_lower_bound", None)
    if gate_lower_bound is None:
        gate_lower_bound = getattr(text_config, "linear_lower_bound", None)
    if gate_lower_bound is None:
        raise ValueError("GLM-5.3 KDA requires gate_lower_bound (safe gate) in the HF config.")
    return dict(
        num_heads=int(field("num_heads", "linear_num_heads", 64)),
        head_dim=int(field("head_dim", "linear_head_dim", 128)),
        conv_kernel_size=int(field("short_conv_kernel_size", "linear_conv_kernel_dim", 4)),
        gate_lower_bound=float(gate_lower_bound),
    )


class Glm5NextKDA(nn.Module):
    # [CLASS DOC] Kimi delta attention mixer on packed sequences of shape (tokens, hidden). Parameters:
    # separate bias-free query, key, value projections to heads times head dim; a causal depthwise
    # short convolution (silu) over the concatenated q/k/v channels; a per-head beta projection; a
    # low-rank forget-gate projection pair (hidden to head dim to heads times head dim); a low-rank
    # output-gate projection pair; float32 per-head decay log-rate and per-channel time-step bias;
    # a gated RMS norm per head (sigmoid gating); and an output projection back to hidden size.

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        gate_lower_bound: float,
        rms_norm_eps: float,
    ):
        # [FUNCTION DOC] Raises an import error if the optional flash-linear-attention kernels are
        # unavailable. Stores sizes (projection size is heads times head dim) and builds the modules
        # listed in the class description; decay log-rate and time-step bias are zero-initialized
        # float32 and kept in float32 under mixed precision (real values come from checkpoints).
        super().__init__()
        if ShortConvolution is None or chunk_kda is None:
            raise ImportError("GLM-5.3 KDA requires flash-linear-attention >= 0.4.2 (fla.ops.kda).")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = num_heads * head_dim
        self.conv_kernel_size = conv_kernel_size
        self.gate_lower_bound = gate_lower_bound

        self.q_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.conv1d = ShortConvolution(
            hidden_size=3 * self.projection_size,
            kernel_size=conv_kernel_size,
            bias=False,
            activation="silu",
        )
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.A_log = mark_keep_in_fp32(nn.Parameter(torch.zeros(num_heads, dtype=torch.float32)))
        self.dt_bias = mark_keep_in_fp32(nn.Parameter(torch.zeros(self.projection_size, dtype=torch.float32)))
        self.o_norm = FusedRMSNormGated(head_dim, eps=rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(self.projection_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor):
        # [FUNCTION DOC] Input hidden states (tokens, hidden) for packed sequences and cumulative
        # sequence lengths. Builds an optional hybrid context-parallel context (none when not using
        # CP); the convolution and kernel then use the context's own cumulative lengths. Steps: project
        # and concatenate q, k, v; causal short convolution restarting at sequence boundaries; split and
        # reshape to heads; beta is sigmoid of the beta projection in float32; the forget gate is the low-rank
        # projection reshaped to heads, transformed by the fused gate function with the decay log-rate,
        # time-step bias and lower bound (the "safe gate"); run the chunked delta-rule kernel with
        # L2 normalization of q and k inside the kernel (with the CP context when present; otherwise
        # with no initial state and not returning the final state); output gate is another low-rank
        # projection; the kernel output is flattened to rows of head dim, passed through the gated RMS
        # norm with the gate, reshaped to (tokens, heads times head dim) and projected back to hidden.
        # Returns (tokens, hidden).
        cp_context = build_gdn_cp_context(self, cu_seqlens, hidden_states.device)

        mixed_qkv = torch.cat(
            (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)),
            dim=-1,
        )
        conv_cu_seqlens = cp_context.cu_seqlens if cp_context is not None else cu_seqlens
        mixed_qkv, _ = self.conv1d(
            x=mixed_qkv,
            cu_seqlens=conv_cu_seqlens,
            cp_context=cp_context,
        )
        query, key, value = torch.split(mixed_qkv, [self.projection_size] * 3, dim=-1)
        query = query.unflatten(-1, (self.num_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_heads, self.head_dim))

        beta = torch.sigmoid(self.b_proj(hidden_states).float())
        forget = self.f_b_proj(self.f_a_proj(hidden_states))
        g = fused_kda_gate(
            forget.unflatten(-1, (self.num_heads, self.head_dim)),
            self.A_log,
            self.dt_bias,
            lower_bound=self.gate_lower_bound,
        )

        if cp_context is not None:
            core_attn_out, _ = chunk_kda(
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
            core_attn_out, _ = chunk_kda(
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

        norm_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        out_shape = core_attn_out.shape
        core_attn_out = self.o_norm(
            core_attn_out.reshape(-1, self.head_dim),
            norm_gate.reshape(-1, self.head_dim),
        )
        core_attn_out = core_attn_out.reshape(out_shape[0], out_shape[1], -1)
        return self.o_proj(core_attn_out)


class Glm5NextKDAAttention(HuggingfaceAttention):
    # [CLASS DOC] Megatron-facing attention wrapper that hosts the KDA mixer (as the kda submodule)
    # built from the Hugging Face text config; the base class handles layout conversion, gathering
    # and the hybrid context-parallel flag (class attribute set true).

    hybrid_cp = True

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
        name: str | None = None,
    ):
        # [FUNCTION DOC] Calls the base constructor (which loads the HF config into the hf config
        # attribute), extracts the text config and KDA fields, and builds the mixer with the text
        # config's hidden size and RMS norm epsilon. Default communication type is point-to-point.
        super().__init__(args, config, layer_number, cp_comm_type, pg_collection, name=name)
        text_config = _get_text_config(self.hf_config)
        fields = _linear_attn_fields(text_config)
        self.kda = Glm5NextKDA(
            hidden_size=text_config.hidden_size,
            num_heads=fields["num_heads"],
            head_dim=fields["head_dim"],
            conv_kernel_size=fields["conv_kernel_size"],
            gate_lower_bound=fields["gate_lower_bound"],
            rms_norm_eps=text_config.rms_norm_eps,
        )

    def hf_forward(self, hidden_states, packed_seq_params):
        # [FUNCTION DOC] Hook called by the base class: runs the KDA mixer on the hidden states using
        # the query cumulative sequence lengths of the packed parameters. Returns its output.
        return self.kda(hidden_states, cu_seqlens=packed_seq_params.cu_seqlens_q)
