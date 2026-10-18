# [MODULE DOC] HELPER: Test-support module for the head-sharded linear-attention tests. Defines
# unsharded ("replicated") torch reference modules in the Hugging Face weight layout for the gated
# delta net family (qwen3_5 and qwen3_next variants) and for Kimi delta attention (KDA), plus helpers
# to convert their weights/gradients to the Megatron layout, shard and gather tensors across a
# process group, build a sharded LinearAttentionLayer holding a rank's slice of a reference's weights,
# and compute a relative error. Requires the fla kernels and Megatron; uses constants: conv kernel
# size 4, norm epsilon 1e-6, KDA gate lower bound -5.
"""Replicated, HF-layout references for the head-sharded linear-attention layers, and helpers to load their
weights into a :class:`LinearAttentionLayer`."""

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.kda import chunk_kda
from megatron.core.process_groups_config import ProcessGroupCollection

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads, qkv_flat_to_group_major
from miles_plugins.models.linear_attn import KimiDeltaAttention, LinearAttentionLayer
from miles_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet
from miles_plugins.models.qwen3_next import Qwen3NextGatedDeltaNet

CONV = 4
EPS = 1e-6
KDA_LOWER_BOUND = -5.0


class ReplicatedGDN(nn.Module):
    # [CLASS DOC] HELPER: Reference gated-delta-net layer with all heads on every rank. Holds input
    # projections whose layout depends on the family: qwen3_5 has separate qkv, z, b and a projections;
    # any other family has a fused qkvz projection and fused ba projection stored group-major. Also a
    # depthwise short convolution over the qkv channels, a per-value-head decay bias and log decay
    # parameter (the latter kept in float32), a gated RMS norm over value head dim with silu gating, and
    # an output projection. All linear layers are bias-free.
    """Pre-sharding GDN: every rank holds all heads. ``family`` picks the HF projection layout."""

    def __init__(self, family: str, hidden: int, heads: LinearAttnHeads, dtype):
        # [FUNCTION DOC] HELPER: Builds the submodules sized from the heads descriptor (qkv dim, value
        # dim, number of value heads, head value dim). Decay bias is uniform random in [0, 1); log decay
        # is the log of values uniform in [1, 16). Parameters use default random initialization. The
        # module is cast to the given dtype, then the log decay parameter is converted back to float32.
        super().__init__()
        self.family, self.heads = family, heads
        h = heads
        if family == "qwen3_5":
            self.in_proj_qkv = nn.Linear(hidden, h.qkv_dim, bias=False)
            self.in_proj_z = nn.Linear(hidden, h.value_dim, bias=False)
            self.in_proj_b = nn.Linear(hidden, h.num_v_heads, bias=False)
            self.in_proj_a = nn.Linear(hidden, h.num_v_heads, bias=False)
        else:
            self.in_proj_qkvz = nn.Linear(hidden, h.qkv_dim + h.value_dim, bias=False)
            self.in_proj_ba = nn.Linear(hidden, 2 * h.num_v_heads, bias=False)
        self.conv1d = ShortConvolution(hidden_size=h.qkv_dim, kernel_size=CONV, bias=False)
        self.dt_bias = nn.Parameter(torch.rand(h.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(h.num_v_heads).uniform_(1, 16)))
        self.norm = FusedRMSNormGated(h.head_v_dim, eps=EPS, activation="silu")
        self.out_proj = nn.Linear(h.value_dim, hidden, bias=False)
        self.to(dtype=dtype)
        self.A_log.data = self.A_log.data.float()

    def _split(self, x):
        # [FUNCTION DOC] HELPER: Applies the input projections to x (batch, seq, hidden) and returns
        # flat (head-major) query, key, value, gate z, and the b and a gating inputs. For qwen3_5 the
        # qkv tensor is split by key dim, key dim, value dim. For other families the fused projections
        # are reshaped per key head group, split into q, k, v, z (and b, a) within each group and
        # flattened across groups.
        h = self.heads
        if self.family == "qwen3_5":
            q, k, v = self.in_proj_qkv(x).split([h.key_dim, h.key_dim, h.value_dim], dim=-1)
            return q, k, v, self.in_proj_z(x), self.in_proj_b(x), self.in_proj_a(x)
        r, hv = h.v_per_k, h.head_v_dim
        grouped = self.in_proj_qkvz(x).view(*x.shape[:-1], h.num_k_heads, 2 * h.head_k_dim + 2 * r * hv)
        q, k, v, z = grouped.split([h.head_k_dim, h.head_k_dim, r * hv, r * hv], dim=-1)
        b, a = self.in_proj_ba(x).view(*x.shape[:-1], h.num_k_heads, 2 * r).split([r, r], dim=-1)
        return q.flatten(-2), k.flatten(-2), v.flatten(-2), z.flatten(-2), b.flatten(-2), a.flatten(-2)

    def forward(self, x, cu_seqlens):
        # [FUNCTION DOC] Input x (batch, seq, hidden) and cumulative sequence lengths for packed
        # sequences (conv and recurrence respect boundaries). Steps: project and split; concatenate q, k,
        # v and run the causal short convolution; split again; repeat q and k heads to match the number
        # of value heads; compute the decay gate as minus exp(log decay) times softplus of (a plus decay
        # bias) in float32; run the chunked gated delta rule kernel with sigmoid of b as beta and
        # in-kernel query/key L2 normalization; apply the gated RMS norm per head with z as gate; finally
        # the output projection. Returns (batch, seq, hidden).
        h = self.heads
        bsz, seq_len, _ = x.shape
        q, k, v, z, b, a = self._split(x)
        mixed, _ = self.conv1d(x=torch.cat([q, k, v], dim=-1), cu_seqlens=cu_seqlens)
        q, k, v = mixed.split([h.key_dim, h.key_dim, h.value_dim], dim=-1)
        q = q.reshape(bsz, seq_len, -1, h.head_k_dim).repeat_interleave(h.v_per_k, dim=2)
        k = k.reshape(bsz, seq_len, -1, h.head_k_dim).repeat_interleave(h.v_per_k, dim=2)
        v = v.reshape(bsz, seq_len, -1, h.head_v_dim)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        out, _ = chunk_gated_delta_rule(
            q, k, v, g=g, beta=b.sigmoid(), use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens
        )
        out = self.norm(out.reshape(-1, h.head_v_dim), z.reshape(-1, h.head_v_dim))
        return self.out_proj(out.reshape(bsz, seq_len, -1))


class ReplicatedKDA(nn.Module):
    # [CLASS DOC] HELPER: Reference Kimi delta attention layer with all heads on every rank. Has
    # separate q, k, v projections each with its own silu short convolution, a low-rank forget gate
    # (down projection to head value dim then up projection to the full size), a per-head beta
    # projection, an output gate projection, float32 log decay (per head) and decay bias (per channel)
    # parameters, a sigmoid-gated RMS norm per head, and an output projection. Bias-free linears.
    """Pre-sharding KDA in the Kimi-K3 / GLM-5.3-flash HF layout."""

    def __init__(self, hidden: int, heads: LinearAttnHeads, dtype):
        # [FUNCTION DOC] HELPER: Builds the submodules with default random init; the size is the
        # value dim and the head dim is the head value dim from the heads descriptor. Log decay is the
        # log of values uniform in [1, 16); decay bias is uniform in [0, 1). The module is cast to the
        # dtype, then log decay and decay bias are made float32.
        super().__init__()
        self.heads = heads
        size, d = heads.value_dim, heads.head_v_dim
        self.q_proj = nn.Linear(hidden, size, bias=False)
        self.k_proj = nn.Linear(hidden, size, bias=False)
        self.v_proj = nn.Linear(hidden, size, bias=False)
        self.q_conv1d = ShortConvolution(size, CONV, bias=False, activation="silu")
        self.k_conv1d = ShortConvolution(size, CONV, bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(size, CONV, bias=False, activation="silu")
        self.f_a_proj = nn.Linear(hidden, d, bias=False)
        self.f_b_proj = nn.Linear(d, size, bias=False)
        self.b_proj = nn.Linear(hidden, heads.num_v_heads, bias=False)
        self.g_proj = nn.Linear(hidden, size, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(heads.num_v_heads).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.rand(size))
        self.o_norm = FusedRMSNormGated(d, eps=EPS, activation="sigmoid")
        self.o_proj = nn.Linear(size, hidden, bias=False)
        self.to(dtype=dtype)
        self.A_log.data = self.A_log.data.float()
        self.dt_bias.data = self.dt_bias.data.float()

    def hf_conv(self):
        # [FUNCTION DOC] HELPER: Returns the q, k and v convolution weights concatenated along the first
        # (channel) dimension, i.e. the fused convolution weight in HF order.
        return torch.cat([self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight])

    def forward(self, x, cu_seqlens):
        # [FUNCTION DOC] Input x (batch, seq, hidden) with cumulative sequence lengths for packed
        # sequences. Projects and convolves q, k, v; reshapes to (batch, seq, value heads, head dim);
        # computes the forget gate input through the low-rank pair; runs the chunked KDA kernel with
        # float32 sigmoid of the beta projection, log decay, decay bias, in-kernel query/key L2
        # normalization and in-kernel gating in "safe" mode with the module's lower bound and transposed
        # state layout; applies the sigmoid-gated RMS norm with the output gate projection as gate per
        # head; then the output projection. Returns (batch, seq, hidden).
        bsz, seq_len, _ = x.shape
        h, d = self.heads.num_v_heads, self.heads.head_v_dim
        q, _ = self.q_conv1d(self.q_proj(x), cu_seqlens=cu_seqlens)
        k, _ = self.k_conv1d(self.k_proj(x), cu_seqlens=cu_seqlens)
        v, _ = self.v_conv1d(self.v_proj(x), cu_seqlens=cu_seqlens)
        out, _ = chunk_kda(
            q=q.view(bsz, seq_len, h, d),
            k=k.view(bsz, seq_len, h, d),
            v=v.view(bsz, seq_len, h, d),
            g=self.f_b_proj(self.f_a_proj(x)).view(bsz, seq_len, h, d),
            beta=self.b_proj(x).float().sigmoid(),
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=KDA_LOWER_BOUND,
            transpose_state_layout=True,
            cu_seqlens=cu_seqlens,
        )
        out = self.o_norm(out.reshape(-1, d), self.g_proj(x).reshape(-1, d))
        return self.o_proj(out.reshape(bsz, seq_len, -1))


def sharded_projections(ref, grad: bool = False) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Returns a mapping from the Megatron core's head-sharded projection
    # attribute names to the full (unsharded) weight tensors (or their gradients when grad is true)
    # taken from the reference. KDA: q, k, v, output gate, beta and forget up projections. GDN
    # qwen3_5: the qkv weight converted from flat to group-major layout plus z, b, a unchanged. Other
    # GDN families: the fused qkvz and ba weights as stored. Gradients may be None if backward was not run.
    """Full (all-head) Megatron tensor of every head-sharded projection, from the reference's HF-layout
    weights or their gradients: what the bridges load."""

    def w(module):
        # [FUNCTION DOC] HELPER: Returns the module's weight gradient if grad is true, else the weight.
        return module.weight.grad if grad else module.weight

    if isinstance(ref, ReplicatedKDA):
        return {
            "q_proj": w(ref.q_proj),
            "k_proj": w(ref.k_proj),
            "v_proj": w(ref.v_proj),
            "g_proj": w(ref.g_proj),
            "b_proj": w(ref.b_proj),
            "f_b_proj": w(ref.f_b_proj),
        }
    if ref.family == "qwen3_5":
        return {
            "in_proj_qkv": qkv_flat_to_group_major(w(ref.in_proj_qkv), ref.heads),
            "in_proj_z": w(ref.in_proj_z),
            "in_proj_b": w(ref.in_proj_b),
            "in_proj_a": w(ref.in_proj_a),
        }
    return {"in_proj_qkvz": w(ref.in_proj_qkvz), "in_proj_ba": w(ref.in_proj_ba)}


def conv_of(ref, grad: bool = False) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Returns a mapping from the core's convolution attribute names to full
    # convolution weights (or gradients when grad is true): three entries (q, k, v convolutions) for
    # KDA, or one entry named conv1d in group-major layout for GDN.
    """Full Megatron conv weight(s) by core attribute: one group-major conv for GDN, one per tensor for KDA."""

    def w(module):
        # [FUNCTION DOC] HELPER: Returns the module's weight gradient if grad is true, else the weight.
        return module.weight.grad if grad else module.weight

    if isinstance(ref, ReplicatedKDA):
        return {name: w(getattr(ref, name)) for name in ("q_conv1d", "k_conv1d", "v_conv1d")}
    return {"conv1d": qkv_flat_to_group_major(w(ref.conv1d), ref.heads)}


def shard(full: torch.Tensor, dim: int, group) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Splits the tensor into group-size equal chunks along dim and returns this
    # rank's chunk as a contiguous tensor. The dimension must be divisible by the group size.
    return full.chunk(group.size(), dim=dim)[group.rank()].contiguous()


def gather(local: torch.Tensor, dim: int, group) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Collective operation: all ranks of the group must call it. Gathers every
    # rank's local tensor (same shape everywhere) with a distributed all-gather, then concatenates in
    # rank order along dim. Not differentiable (the gather is a plain collective).
    parts = [torch.empty_like(local) for _ in range(group.size())]
    dist.all_gather(parts, local.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def build_layer(ref, config, allgather_cp: bool = True) -> LinearAttentionLayer:
    # [FUNCTION DOC] HELPER: Builds a sharded layer from a replicated reference. Gets the tensor- and
    # context-parallel groups from the already-initialized Megatron parallel state. Creates the core
    # attention (KDA with the gate lower bound for a KDA reference; otherwise the qwen3_5 or qwen3_next
    # gated delta net by family) with conv size 4, epsilon 1e-6 and the tp group. Without gradients,
    # copies in this rank's shard (first dim) of each projection and conv weight, log decay and decay
    # bias, copies the norm weight whole, shards the output projection along its input (second) dim,
    # and for KDA copies the replicated forget down projection whole. Wraps the core in the layer with an
    # identity input norm; allgather_cp is forwarded. Requires initialized model-parallel state.
    """A head-sharded layer holding this TP rank's shard of ``ref``'s weights, identity input norm."""
    pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    tp = pg.tp
    if isinstance(ref, ReplicatedKDA):
        core = KimiDeltaAttention(config, ref.heads, CONV, EPS, tp, gate_lower_bound=KDA_LOWER_BOUND)
    else:
        core_cls = Qwen3_5GatedDeltaNet if ref.family == "qwen3_5" else Qwen3NextGatedDeltaNet
        core = core_cls(config, ref.heads, CONV, EPS, tp)
    with torch.no_grad():
        for name, full in sharded_projections(ref).items():
            getattr(core, name).weight.copy_(shard(full, 0, tp))
        for name, full in conv_of(ref).items():
            getattr(core, name).weight.copy_(shard(full, 0, tp))
        core.A_log.copy_(shard(ref.A_log, 0, tp))
        core.dt_bias.copy_(shard(ref.dt_bias, 0, tp))
        norm = ref.o_norm if isinstance(ref, ReplicatedKDA) else ref.norm
        core.norm.weight.copy_(norm.weight)
        out_proj = ref.o_proj if isinstance(ref, ReplicatedKDA) else ref.out_proj
        core.out_proj.weight.copy_(shard(out_proj.weight, 1, tp))
        if isinstance(ref, ReplicatedKDA):
            core.f_a_proj.weight.copy_(ref.f_a_proj.weight)
    return LinearAttentionLayer(config, core, nn.Identity(), pg, allgather_cp=allgather_cp)


def packed(cu_seqlens):
    # [FUNCTION DOC] HELPER: Returns a minimal packed-sequence parameters object (simple namespace)
    # whose only field is the query cumulative sequence lengths.
    return SimpleNamespace(cu_seqlens_q=cu_seqlens)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    # [FUNCTION DOC] HELPER: Returns the norm of the difference over the norm of a (plus 1e-12), both
    # converted to float32, as a Python float. Not symmetric: a is the reference.
    a, b = a.float(), b.float()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()
