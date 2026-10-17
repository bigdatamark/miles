# [MODULE DOC] Differentiable sparse attention where each query attends only to a given list of
# key/value rows (selected indices, minus one meaning padding), with an optional learnable
# per-head "sink" logit that joins the softmax denominator without contributing a value.
# Shapes: queries batch by sequence by heads by (value dim + tail dim), key/value cache batch by
# kv length by kv groups by the same width (keys and values share storage: the value is the first
# value-dim channels), indices batch by sequence by groups by top-k.
# Forward backends: the FlashMLA sparse prefill kernel (when installed, GPU architecture major 9
# or 10, value dim 512, width 512 or 576, one kv group, at most 128 heads) or a tilelang kernel.
# Backward is always the tilelang kernel; it consumes a log-sum-exp in base-2 units with the sink
# already folded in (both forward backends return it in that form), plus a per-row "delta" it
# returns, from which the sink gradient is derived.
# Tuning: a frozen config dataclass selects the forward backend and kernel tiling; a table keyed by
# (architecture major, whether a RoPE tail exists, power-of-two bucket of query heads per group,
# clamped to 8..64) gives measured choices, with a default for missing keys and an automatic
# fallback to tilelang when FlashMLA is unsupported. Caller may pass an explicit config.
# FlashMLA import failure is tolerated (the function is then None).
"""Sparse attention over top-k selected KV rows (DeepSeek Sparse Attention core).

q [B, S, H, d_v + d_tail] bf16, kv [B, S_kv, G, d_v + d_tail] bf16, indices [B, S, G, topk] int32 with
-1 as padding, attn_sink [H] fp32 or None. GLM-5 / DeepSeek-V3.2 pass a RoPE tail (d_tail > 0) and
no sink; DeepSeek-V4 passes a single latent (d_tail = 0, G = 1) and a learnable per-head sink.

The forward runs FlashMLA's sparse prefill kernel (single latent group, d_v = 512; query heads are
zero-padded up to 64 or 128) or the TileLang kernel; the backward is always the TileLang kernel. It takes
the log-sum-exp in log2 space with the sink folded in, which is what the TileLang forward emits and what
FlashMLA's is converted to. Which forward runs and how both kernels are tiled is a SparseAttentionConfig:
measured rows live in _TUNED, keyed by (arch major, RoPE tail, TP-local heads); callers may pass their own.
"""

from dataclasses import dataclass, replace
from typing import Literal

import torch


try:
    from flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None

_LOG2_E = 1.4426950408889634
_TILELANG_TOPK_MULTIPLE = 64
_FLASH_MLA_TOPK_MULTIPLE = 128
_FLASH_MLA_HEADS = (64, 128)
_FLASH_MLA_HEAD_DIMS = (512, 576)
_FLASH_MLA_ARCH_MAJORS = (9, 10)


@dataclass(frozen=True)
class SparseAttentionConfig:
    # [CLASS DOC] Immutable kernel selection and tiling parameters. forward_backend is "flash_mla"
    # (default) or "tilelang"; forward_threads is the thread count of the tilelang forward kernel
    # (default 256); backward_stages is the software pipeline depth of the tilelang backward
    # (default 0); backward_split_store is how many pieces the backward splits its key/value
    # gradient store into (default 2).
    forward_backend: Literal["flash_mla", "tilelang"] = "flash_mla"
    forward_threads: int = 256
    backward_stages: int = 0
    backward_split_store: int = 2


_TILELANG_SMALL_HEADS = SparseAttentionConfig(forward_backend="tilelang", forward_threads=128)

# Measured per layer at 16k / 64k tokens on H200 (9) and B300 / GB300 (10, identical winners) for GLM-5
# (RoPE tail 64, topk 2048) and DeepSeek-V4.1 (no tail, topk 640). FlashMLA pads heads to 64.
_TUNED = {
    (10, True, 8): SparseAttentionConfig(backward_stages=2, backward_split_store=1),
    (10, True, 16): SparseAttentionConfig(backward_stages=2, backward_split_store=1),
    (10, True, 32): SparseAttentionConfig(backward_stages=2, backward_split_store=4),
    (10, True, 64): SparseAttentionConfig(backward_stages=1, backward_split_store=4),
    (10, False, 8): SparseAttentionConfig(backward_stages=1),
    (10, False, 16): SparseAttentionConfig(backward_stages=1),
    (10, False, 32): SparseAttentionConfig(backward_stages=2, backward_split_store=4),
    (10, False, 64): SparseAttentionConfig(backward_stages=1, backward_split_store=4),
    (9, True, 8): SparseAttentionConfig(backward_stages=1),
    (9, True, 16): SparseAttentionConfig(backward_stages=1),
    (9, True, 32): SparseAttentionConfig(backward_stages=1),
    (9, True, 64): SparseAttentionConfig(backward_split_store=4),
    (9, False, 8): replace(_TILELANG_SMALL_HEADS, backward_stages=1, backward_split_store=4),
    (9, False, 16): replace(_TILELANG_SMALL_HEADS, backward_stages=1, backward_split_store=4),
    (9, False, 32): SparseAttentionConfig(backward_stages=1, backward_split_store=4),
    (9, False, 64): SparseAttentionConfig(backward_split_store=4),
}


def _pad_topk_block(indices, multiple: int):
    # [FUNCTION DOC] HELPER: Pads the last dimension of the index tensor with minus one up to a
    # multiple of the given number; returns a contiguous tensor (the input made contiguous when no
    # padding is needed).
    topk = indices.shape[-1]
    padded = (topk + multiple - 1) // multiple * multiple
    if padded == topk:
        return indices.contiguous()
    return torch.nn.functional.pad(indices, (0, padded - topk), value=-1).contiguous()


def _default_config(q, kv, d_v: int) -> SparseAttentionConfig:
    # [FUNCTION DOC] HELPER: Chooses the config for this call. Reads the CUDA capability major of the
    # query's device, computes query heads per kv group, rounds it up to a power of two clamped to
    # the range 8 to 64 as the bucket, and looks up (architecture, whether the width exceeds the
    # value dim i.e. a tail exists, bucket) in the tuned table, falling back to the default config.
    # FlashMLA is deemed supported if the library imported, architecture major is 9 or 10, value dim
    # is 512, width is 512 or 576, there is exactly one kv group and at most 128 heads. When the
    # chosen config wants FlashMLA but it is unsupported, returns a copy using tilelang with 128
    # forward threads for buckets below 64 and 256 otherwise.
    arch_major = torch.cuda.get_device_capability(q.device)[0]
    heads_per_group = q.shape[2] // kv.shape[2]
    heads_bucket = min(max(8, 1 << (heads_per_group - 1).bit_length()), 64)
    config = _TUNED.get((arch_major, q.shape[-1] > d_v, heads_bucket), SparseAttentionConfig())
    flash_mla_supported = (
        flash_mla_sparse_fwd is not None
        and arch_major in _FLASH_MLA_ARCH_MAJORS
        and d_v == 512
        and q.shape[-1] in _FLASH_MLA_HEAD_DIMS
        and kv.shape[2] == 1
        and q.shape[2] <= _FLASH_MLA_HEADS[-1]
    )
    if config.forward_backend == "flash_mla" and not flash_mla_supported:
        config = replace(config, forward_backend="tilelang", forward_threads=128 if heads_bucket < 64 else 256)
    return config


def _flash_mla_forward(q, kv, indices, attn_sink, sm_scale):
    # [FUNCTION DOC] HELPER: Runs the FlashMLA sparse prefill kernel and adapts it to this module's
    # conventions. Flattens batch and sequence of the queries; pads the head count with zero heads
    # up to 64 or 128 (and the sink with zeros; raises a stop-iteration error beyond 128 heads); pads
    # indices to a multiple of 128 with minus one; for batch size one, indices index the kv cache
    # directly, otherwise each batch element's valid indices are shifted by its offset into the
    # flattened kv cache (padding stays minus one). The kv cache is flattened to a single group.
    # Calls the kernel with value dim 512, the scale and the sink; the output is cut back to the real
    # heads, and the returned log-sum-exp (natural-log units, excluding the sink) is converted to
    # base 2 by multiplying by log2(e) and, if a sink exists, combined with the sink logit (also
    # scaled to base 2) by a base-2 log-add-exp. Returns (output batch by sequence by heads by 512,
    # lse batch by sequence by heads, contiguous). The value dim is hard-coded rather than taken
    # from the caller.
    batch, seq_len, heads, _ = q.shape
    seq_len_kv = kv.shape[1]
    padded_heads = next(h for h in _FLASH_MLA_HEADS if h >= heads)
    q = q.reshape(batch * seq_len, heads, -1)
    sink = attn_sink
    if padded_heads != heads:
        q = torch.nn.functional.pad(q, (0, 0, 0, padded_heads - heads))
        sink = None if attn_sink is None else torch.nn.functional.pad(attn_sink, (0, padded_heads - heads))
    indices = _pad_topk_block(indices, _FLASH_MLA_TOPK_MULTIPLE)
    if batch == 1:
        flat_indices = indices.view(seq_len, 1, -1)
    else:
        offsets = torch.arange(batch, device=indices.device, dtype=indices.dtype).view(batch, 1, 1, 1) * seq_len_kv
        flat_indices = torch.where(indices >= 0, indices + offsets, -1).view(batch * seq_len, 1, -1)
    out, _, lse = flash_mla_sparse_fwd(
        q, kv.reshape(batch * seq_len_kv, 1, -1), flat_indices, sm_scale, d_v=512, attn_sink=sink
    )
    out, lse = out[:, :heads].contiguous(), lse[:, :heads] * _LOG2_E
    if attn_sink is not None:
        lse = torch.logaddexp2(lse, attn_sink.float().view(1, heads) * _LOG2_E)
    return out.reshape(batch, seq_len, heads, -1), lse.reshape(batch, seq_len, heads).contiguous()


class _SparseAttention(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function running the selected forward backend and the tilelang
    # backward. Saves q, kv, padded indices, clamped indices (padding mapped to row zero), the sink,
    # the output and the base-2 log-sum-exp; stores the scale, value dim and config on the context.
    @staticmethod
    def forward(ctx, q, kv, indices, attn_sink, sm_scale, d_v, config):
        # [FUNCTION DOC] Makes q and kv contiguous and pads indices to a multiple of 64 with minus
        # one; computes the clamped index copy used for safe memory reads. Dispatches on the config:
        # FlashMLA wrapper, or the tilelang forward (lazily imported) with the given value dim, scale
        # and thread count. Returns the attention output (batch, sequence, heads, value dim).
        q, kv, indices = q.contiguous(), kv.contiguous(), _pad_topk_block(indices, _TILELANG_TOPK_MULTIPLE)
        kv_indices = indices.clamp(min=0)
        if config.forward_backend == "flash_mla":
            out, lse = _flash_mla_forward(q, kv, indices, attn_sink, sm_scale)
        else:
            # tilelang is GPU-only; importing it lazily keeps the package importable on CPU
            from miles.kernels.attention.dsa.tilelang.sparse_attention_fwd import sparse_attention_fwd

            out, lse = sparse_attention_fwd(
                q,
                kv,
                indices,
                kv_indices,
                attn_sink,
                d_v,
                sm_scale=sm_scale,
                threads=config.forward_threads,
            )
        ctx.save_for_backward(q, kv, indices, kv_indices, attn_sink, out, lse)
        ctx.sm_scale = sm_scale
        ctx.d_v = d_v
        ctx.config = config
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # [FUNCTION DOC] Runs the tilelang backward (lazily imported) with the saved tensors, the
        # contiguous output gradient, and the configured pipeline stages and split-store count,
        # obtaining gradients for queries and the key/value cache plus a per-row delta. If a sink was
        # given, its probability mass per row is 2 to the power (sink logit in base 2 minus lse), and
        # the sink gradient is minus the sum over batch and sequence of delta times that mass,
        # giving one value per head. Returns gradients for (q, kv, indices, sink, scale, value dim,
        # config) with none for the non-differentiable ones.
        q, kv, indices, kv_indices, attn_sink, out, lse = ctx.saved_tensors
        from miles.kernels.attention.dsa.tilelang.sparse_attention_bwd import sparse_attention_bwd

        dq, dkv, delta = sparse_attention_bwd(
            q,
            kv,
            out,
            grad_out.contiguous(),
            indices,
            kv_indices,
            lse,
            ctx.d_v,
            sm_scale=ctx.sm_scale,
            num_stages=ctx.config.backward_stages,
            split_store=ctx.config.backward_split_store,
        )
        d_sink = None
        if attn_sink is not None:
            p_sink = torch.exp2(attn_sink.float() * _LOG2_E - lse)
            d_sink = -(delta * p_sink).sum(dim=(0, 1))
        return dq, dkv, None, d_sink, None, None, None


def sparse_attention(
    q,
    kv,
    indices,
    sm_scale: float,
    d_v: int | None = None,
    attn_sink=None,
    config: SparseAttentionConfig | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Public entry. d_v defaults to the full query width (no tail); config defaults to
    # the tuned or fallback choice for the device and shapes (this requires a CUDA device and queries
    # on it). Applies the autograd function and returns the output, batch by sequence by heads by
    # value dim, in the query dtype. Differentiable in q, kv and the sink.
    """Returns out [B, S, H, d_v] bf16. ``d_v`` defaults to the full head dim (no RoPE tail).
    ``config`` defaults to the measured row for this arch and shape."""
    if d_v is None:
        d_v = q.shape[-1]
    if config is None:
        config = _default_config(q, kv, d_v)
    return _SparseAttention.apply(q, kv, indices, attn_sink, sm_scale, d_v, config)
