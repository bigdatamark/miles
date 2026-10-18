# [MODULE DOC]
# Context-parallel (CP) token layout helpers used inside model layers. Megatron stores each CP rank's
# tokens in a zigzag load-balanced layout; flash-linear-attention (fla) CP operators need each rank to
# hold a contiguous chunk of the packed stream. This module builds the fla CP context (when the
# optional dependency is importable) and converts between the two layouts with differentiable
# all-gather based relayouts, computing the index maps on the device.
"""CP token-layout helpers used inside model plugin ``forward`` implementations.

Distinct from ``miles.backends.training_utils.data.context_parallel``, which owns the CP
helpers the training backend applies *around* the model (slicing data, masks,
logprobs and logits). Everything here runs inside a layer.

Megatron CP stores each rank's tokens in the zigzag load-balanced order that
ring attention wants, while fla's CP operators expect a contiguous rank-local
chunk. Anything that hands a sequence to fla under CP therefore has to relayout
first and undo it afterwards.
"""

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from fla.ops.cp import build_cp_context as _fla_build_cp_context
except ImportError:
    _fla_build_cp_context = None


def build_fla_cp_context(cu_seqlens: torch.Tensor, cp_group, conv_kernel_size: int, device: torch.device):
    # [FUNCTION DOC] Build the fla context for this rank. Raises a runtime error if the fla CP module
    # could not be imported, and a value error if the boundaries are missing or have fewer than two
    # entries. Otherwise calls the fla builder with the boundaries moved to the device as 32-bit
    # integers, the CP group and the convolution kernel size, and returns its context.
    """fla CP context for a rank of ``cp_group`` from the global packed boundaries ``cu_seqlens``."""
    if _fla_build_cp_context is None:
        raise RuntimeError(
            "Hybrid CP requires fla.ops.cp (flash-linear-attention >= 0.4.2) " "but it could not be imported."
        )
    if cu_seqlens is None or cu_seqlens.numel() < 2:
        raise ValueError(f"Hybrid CP requires valid cu_seqlens (at least 2 elements) but got {cu_seqlens}")
    return _fla_build_cp_context(
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
        group=cp_group,
        conv1d_kernel_size=conv_kernel_size,
    )


def build_gdn_cp_context(module: nn.Module, cu_seqlens: torch.Tensor, device: torch.device):
    # [FUNCTION DOC] Return none if the module has no CP group attribute (or it is none); otherwise the
    # fla CP context built from the boundaries, the module's CP group and its convolution kernel size.
    """Build fla CP context for a GatedDeltaNet module from packed sequence boundaries.

    Args:
        module: GDN module with ``cp_group`` / ``cp_world_size`` / ``conv_kernel_size``.
        cu_seqlens: Global packed sequence boundaries (e.g. ``packed_seq_params.cu_seqlens_q``).
        device: Target device.

    Returns ``None`` when CP is not configured on the module (``cp_group`` not set).
    """
    cp_group = getattr(module, "cp_group", None)
    if cp_group is None:
        return None
    return build_fla_cp_context(cu_seqlens, cp_group, module.conv_kernel_size, device)


def _relayout_indices(cu_seqlens, cp_rank, cp_size, total):
    # [FUNCTION DOC] HELPER:
    # Compute, entirely with device tensor operations, the row index maps between the zigzag layout and
    # the contiguous packed layout for a stream of the given total token count. Inputs: sequence
    # boundaries (cumulative lengths), this rank, the CP size, the total. Uses deferred device-side
    # assertions that the last boundary equals the total and every sequence length is divisible by the
    # CP size. For each global packed position it finds the owning sequence, the offset in it, the chunk
    # size (a 2*size-th of the sequence when the length divides by twice the CP size, zigzag; otherwise a
    # size-th), the chunk number, and, for zigzag sequences, mirrors chunks beyond the first CP-size
    # ones so rank r owns chunks r and 2*size-1-r. That gives each packed position's position in the
    # rank-concatenated zigzag order; the inverse permutation is built by scattering. Returns the two
    # index vectors restricted to this rank's rows (shard = total / CP size rows starting at rank times
    # shard): first the positions to select from the rank-concatenated zigzag shards to get this rank's
    # contiguous shard, then those to select from the packed stream to get this rank's zigzag shard.
    """Index maps between the two layouts of a ``total``-token packed stream, built on device.

    Rank ``r``'s zigzag shard holds, per sequence, chunks ``r`` and ``2 * cp_size - 1 - r`` of
    ``2 * cp_size`` equal chunks, or its contiguous ``1 / cp_size`` when the length is not a multiple of
    ``2 * cp_size`` (the final padding). Returns ``(to_packed, to_zigzag)``: ``to_packed`` selects this
    rank's contiguous shard from the rank-concatenated zigzag shards, ``to_zigzag`` selects this rank's
    zigzag shard from the rank-concatenated contiguous shards (the packed stream itself)."""
    cu = cu_seqlens.to(torch.int64)
    lengths = cu[1:] - cu[:-1]
    torch._assert_async(cu[-1] == total)
    torch._assert_async((lengths % cp_size == 0).all())
    shard = total // cp_size
    position = torch.arange(total, device=cu.device)
    seq = torch.searchsorted(cu, position, right=True) - 1
    start = cu[seq]
    length = lengths[seq]
    offset = position - start
    zigzag = length % (2 * cp_size) == 0
    chunk = torch.where(zigzag, length // (2 * cp_size), length // cp_size)
    part = offset // chunk
    mirrored = zigzag & (part >= cp_size)
    owner = torch.where(mirrored, 2 * cp_size - 1 - part, part)
    zigzag_position = owner * shard + start // cp_size + offset % chunk + mirrored * chunk
    packed_position = torch.empty_like(zigzag_position)
    packed_position[zigzag_position] = position
    rows = slice(cp_rank * shard, (cp_rank + 1) * shard)
    return zigzag_position[rows], packed_position[rows]


def _gather_select(x, index, cp_group):
    # [FUNCTION DOC] HELPER: All-gather the per-rank tensors along the first axis across the CP group
    # (rank order) into one tensor, then select the given rows. Returns the selected rows. Collective
    # call: every rank must call it.
    gathered = x.new_empty((x.shape[0] * dist.get_world_size(group=cp_group), *x.shape[1:]))
    dist.all_gather_into_tensor(gathered, x.contiguous(), group=cp_group)
    return gathered.index_select(0, index)


class _Relayout(torch.autograd.Function):
    # [CLASS DOC] HELPER: Differentiable relayout. Forward gathers across the CP group and selects rows by
    # an index; backward does the same to the gradient using the inverse index (the inverse index is saved
    # for backward; the group is stored on the context).
    """All-gather over CP, then select this rank's rows; the gradient is the same with the inverse map."""

    @staticmethod
    def forward(ctx, x, index, inverse_index, cp_group):
        # [FUNCTION DOC] HELPER: Save the group and inverse index for backward and return the gathered
        # rows selected by the index.
        ctx.cp_group = cp_group
        ctx.save_for_backward(inverse_index)
        return _gather_select(x, index, cp_group)

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Gradient for the input is the gathered gradient rows selected by the
        # inverse index; the other three arguments get no gradient.
        (inverse_index,) = ctx.saved_tensors
        return _gather_select(grad_output, inverse_index, ctx.cp_group), None, None, None


def zigzag_to_packed_shard(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    # [FUNCTION DOC] Convert this rank's zigzag-layout hidden states (first axis tokens) into this rank's
    # contiguous shard of the packed stream. The total token count is the local count times the CP size.
    # Collective over the CP group; differentiable.
    to_packed, to_zigzag = _relayout_indices(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _Relayout.apply(hidden_states, to_packed, to_zigzag, cp_group)


def packed_shard_to_zigzag(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    # [FUNCTION DOC] Inverse of the previous conversion: take this rank's contiguous shard of the packed
    # stream and return this rank's zigzag-layout rows. Collective over the CP group; differentiable.
    to_packed, to_zigzag = _relayout_indices(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _Relayout.apply(hidden_states, to_zigzag, to_packed, cp_group)
