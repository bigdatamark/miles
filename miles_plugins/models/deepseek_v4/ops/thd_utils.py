# [MODULE DOC] HELPER: Index and layout helpers for the packed variable-length (THD) format used by
# DeepSeek-V4 attention, where many sequences are concatenated into one stream described by cumulative
# lengths. Contains: the layout record; segment lookup; compressed-length computation; sliding-window
# and compressed-key index builders (with per-segment causal visibility); and, for context
# parallelism, the machinery that gives each rank a fixed-capacity compact buffer of the tokens its
# compressed groups need (including a differentiable left-neighbour boundary exchange) and the map
# from sequence-major compressed row ids to rows of the all-gathered buffer. All index tensors live
# on the device of the cumulative-length tensor. Two sections are adapted from NVIDIA Megatron-LM.
"""
Utility functions for DeepSeek V4 THD (packed variable-length) support.

Indices are absolute positions into the concatenated ``[tokens | compressed]`` KV that
``deepseek_v4`` builds, so a query never reaches outside its own segment.

Under context parallelism a rank holds ``[global_start, global_start + total_tokens)`` of a
globally-numbered ``cu_seqlens``, while ``deepseek_v4`` all-gathers the KV. Passing
``global_start`` resolves local rows against the global stream; the returned indices stay
absolute, so the KV layout is unchanged.
"""

from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass
class ThdLayout:
    # [CLASS DOC] HELPER: Mutable record describing this rank's packed stream. Required fields:
    # cumulative sequence lengths tensor (segments plus one entries), a host tuple of segment lengths,
    # the global row offset of this rank's first row, and the longest segment length. Optional fields
    # (default none) are populated during the forward pass: compacted hidden rows, compressed group
    # ids, the map from sequence-major compressed row to gathered row, and cumulative compressed lengths.
    """How this rank's packed stream is laid out; ``None`` stands for the unpacked one.

    The first four fields come from the packed sequence parameters; ``seq_lens`` holds the
    segment lengths on the host, so reading them costs no device sync. The rest are filled in as
    the forward runs: ``cu_seqlens_compressed`` before the compressor is called, and the
    compaction ones only under CP, where a compressed group can straddle the split.
    """

    cu_seqlens: Tensor
    seq_lens: tuple[int, ...]
    global_start: int
    max_seqlen: int
    hidden_compact: Tensor | None = None
    compressed_group_ids: Tensor | None = None
    seq_to_rank_row: Tensor | None = None
    cu_seqlens_compressed: Tensor | None = None

    @classmethod
    def from_packed_seq_params(cls, packed_seq_params, *, cp_rank: int, seqlen_local: int):
        # [FUNCTION DOC] HELPER: Class constructor from Megatron packed-sequence parameters. Returns
        # nothing when the parameters are absent or their format is not the packed one. Otherwise the
        # layout takes the query cumulative lengths as is, host segment lengths as consecutive
        # differences of the host cumulative lengths, global start equal to cp rank times local
        # length (contiguous CP split), and the query max length. Optional fields stay unset.
        """This rank's layout, or None for any format other than thd."""
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            return None
        cu_seqlens_host = packed_seq_params.cu_seqlens_host
        return cls(
            cu_seqlens=packed_seq_params.cu_seqlens_q,
            seq_lens=tuple(end - start for start, end in pairwise(cu_seqlens_host)),
            # CP splits the packed stream contiguously, so this rank's rows start here globally.
            global_start=cp_rank * seqlen_local,
            max_seqlen=packed_seq_params.max_seqlen_q,
        )


def batch_of_row(cu_seqlens: Tensor, total_rows: int, global_start: int = 0) -> Tensor:
    # [FUNCTION DOC] HELPER: Builds global row indices (a 64-bit range of total_rows entries shifted by
    # global start) on the device and returns each row's owning segment index via the position lookup.
    """Segment index owning each row of a THD-packed tensor.

    Args:
        cu_seqlens: ``[n_seg + 1]`` cumulative lengths.
        total_rows: number of rows; rows past ``cu_seqlens[-1]`` clamp to the last segment.
        global_start: first global row this rank holds.
    Returns:
        ``[total_rows]`` int64.
    """
    row_idx = torch.arange(total_rows, device=cu_seqlens.device, dtype=torch.int64) + global_start
    return segment_of_positions(cu_seqlens, row_idx)


def segment_of_positions(cu_seqlens: Tensor, positions: Tensor) -> Tensor:
    # [FUNCTION DOC] HELPER: Bucketizes positions against the segment end boundaries (cumulative
    # lengths without the leading zero) with right-side semantics, so a position equal to a boundary
    # belongs to the next segment; the result is clamped to the last segment index (zero if there are
    # no segments).
    """Segment index owning each global stream position; positions past the end clamp to the last."""
    n_seg = cu_seqlens.size(0) - 1
    return torch.bucketize(positions, cu_seqlens[1:], right=True).clamp(max=max(n_seg - 1, 0))


def compressed_cu_seqlens(cu_seqlens: Tensor, ratio: int) -> Tensor:
    # [FUNCTION DOC] HELPER: Per-segment lengths are divided by ratio (rounding down), cumulatively
    # summed, and a leading zero is prepended; result has the same dtype and device as the input and
    # one more entry than segments.
    """Cumulative compressed lengths, flooring each segment's tail.

    A pure function of ``cu_seqlens``, so the attention forward fills ``ThdLayout`` with it
    before the compressor runs.
    """
    lens = torch.div(cu_seqlens[1:] - cu_seqlens[:-1], ratio, rounding_mode="floor")
    return torch.cat([torch.zeros_like(cu_seqlens[:1]), lens.cumsum(0).to(cu_seqlens.dtype)])


def get_q_positions_thd(cu_seqlens: Tensor, total_tokens: int, global_start: int = 0) -> Tensor:
    # [FUNCTION DOC] HELPER: For each of this rank's rows, returns its global row index minus the start
    # of its segment, i.e. its position inside its own sequence (restarts at zero per segment).
    """Get positions of packed q tokens within their own segment."""
    batch_ids = batch_of_row(cu_seqlens, total_tokens, global_start)
    token_idx = torch.arange(total_tokens, device=cu_seqlens.device) + global_start
    return token_idx - cu_seqlens[batch_ids]


def get_window_topk_idxs_thd(
    cu_seqlens: Tensor, *, window_size: int, total_tokens: int, global_start: int = 0
) -> Tensor:
    # [FUNCTION DOC] HELPER: For each row (global index t) the window starts at the larger of
    # (t minus window size plus one) and its segment's start, and spans window-size slots. Slots
    # beyond t are -1. Indices are global token indices. Result has a leading singleton batch axis.
    """Get window topk indices for a packed stream.

    The window is clamped to the query's own segment start, so it never reaches into the
    preceding segment the way a stream-wide clamp to 0 would. Under CP the window may reach
    rows another rank produced, which the KV all-gather has already delivered.

    Returns:
        ``[1, total_tokens, window_size]`` token indices, ``-1`` past the window.
    """
    device = cu_seqlens.device
    batch_ids = batch_of_row(cu_seqlens, total_tokens, global_start)
    token_idx = torch.arange(total_tokens, device=device) + global_start
    window_start = torch.maximum(token_idx - window_size + 1, cu_seqlens[batch_ids])
    k_pos = window_start.unsqueeze(1) + torch.arange(window_size, device=device)
    topk_idxs = torch.where(k_pos > token_idx.unsqueeze(1), -1, k_pos)
    return topk_idxs.unsqueeze(0)


def to_rank_major_rows(idxs: Tensor, seq_to_rank_row: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC] HELPER: Clamps ids into the table's range, looks them up in the map, and returns
    # (mapped rows, valid mask combined with the mapped row being non-negative). If the table is empty
    # returns all -1 rows and an all-false mask (no clamp is possible). The mapped values of lanes
    # that are not valid are meaningless and must be ignored by the caller.
    """Translate sequence-major compressed ids to their rows in the all-gathered buffer.

    Both the rule-based indices and the indexer's top-k are produced in sequence-major order,
    while CP all-gathers fixed-capacity per-rank blocks; ``-1`` drops a row no rank produced.

    ``valid`` rules a lane out only after the gather, so both ends are clamped first: a segment
    shorter than the longest one never fills its columns, and the indexer leaves its unused
    top-k slots at whatever the kernel wrote. A lane ``valid`` keeps already addresses a row
    the table holds, so the clamp moves no live index.
    """
    n_rows = seq_to_rank_row.numel()
    if n_rows == 0:
        # The whole stream is shorter than the ratio, so no rank produced a compressed row.
        # Clamping would ask for row -1 here; the empty table has no row to fall back on.
        return torch.full_like(idxs, -1), torch.zeros_like(valid)
    rows = seq_to_rank_row[idxs.clamp(0, n_rows - 1).long()]
    return rows, valid & (rows >= 0)


def get_compress_cu_seqlens_thd(
    cu_seqlens: Tensor,
    cu_seqlens_compressed: Tensor,
    *,
    ratio: int,
    total_tokens: int,
    global_start: int = 0,
) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC] HELPER: Builds global positions for this rank's rows (range of total tokens plus
    # global start) and delegates to the bounds-at-positions function; returns its pair of ranges.
    """``compress_bounds_at_positions`` for this rank's contiguous rows."""
    positions = torch.arange(total_tokens, device=cu_seqlens.device) + global_start
    return compress_bounds_at_positions(cu_seqlens, cu_seqlens_compressed, positions, ratio=ratio)


def compress_bounds_at_positions(
    cu_seqlens: Tensor,
    cu_seqlens_compressed: Tensor,
    positions: Tensor,
    *,
    ratio: int,
) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC] HELPER: For each position finds its segment and its offset inside it. Start of
    # the visible range is the segment's first compressed row; end is that start plus (offset plus
    # one) divided by ratio (rounded down), capped at the segment's end in the compressed numbering.
    # Returns two 32-bit integer vectors (start, end), end exclusive. Positions are converted to
    # 64-bit first.
    """Get the compressed rows the packed queries at global stream ``positions`` may see.

    A query sees its own segment only, up to ``(pos_in_seg + 1) // ratio`` and never past
    what that segment produced; the BSHD ``ks = 0`` convention would let it score entries
    of earlier segments once all samples share one flat stream.

    Returns:
        ``(cu_ks, cu_ke)`` int32, one half-open range per position, indices into the compressed
        keys alone, not the concatenated KV. The indexer kernel takes them as is;
        ``get_compress_topk_idxs_thd`` expands them into explicit indices.
    """
    positions = positions.long()
    batch_ids = segment_of_positions(cu_seqlens, positions)
    pos_in_seg = positions - cu_seqlens[batch_ids]
    cu_ks = cu_seqlens_compressed[batch_ids]
    cu_ke = torch.minimum(cu_ks + (pos_in_seg + 1) // ratio, cu_seqlens_compressed[batch_ids + 1])
    return cu_ks.int(), cu_ke.int()


def get_compress_topk_idxs_thd(
    cu_seqlens: Tensor,
    cu_seqlens_compressed: Tensor,
    *,
    ratio: int,
    total_tokens: int,
    max_n_compressed: int,
    kv_offset: int | None = None,
    global_start: int = 0,
    seq_to_rank_row: Tensor | None = None,
) -> Tensor:
    # [FUNCTION DOC] HELPER: Computes per-row visible range, then for each of the given number of
    # columns the sequence-major compressed id is range start plus the column; a column is visible if
    # it is below the range length. When a row map is given (CP), ids are translated to rows of the
    # gathered buffer and visibility further requires the mapped row to exist. Visible entries become
    # the kv offset plus the id, others -1. Result has a leading singleton batch axis. Offset defaults
    # to total tokens.
    """Get static compress topk indices for a packed stream.

    A query sees ``(pos_in_seg + 1) // ratio`` compressed entries, clamped to the number its
    own segment produced, so segments shorter than ``ratio`` fall back to the window alone.

    Args:
        cu_seqlens_compressed: ``[n_seg + 1]`` cumulative compressed lengths, see ``compressed_cu_seqlens``.
        max_n_compressed: column count, an upper bound on any segment's compressed length.
        kv_offset: rows the compressed block starts after; defaults to ``total_tokens``, which
            only holds without CP, where the rank owns the whole stream.
        seq_to_rank_row: sequence-major to all-gather row map, which CP pads to a fixed per-rank
            capacity; ``-1`` drops a row no rank produced.
    Returns:
        ``[1, total_tokens, max_n_compressed]`` indices offset past the uncompressed rows,
        ``-1`` for entries the query cannot see yet.
    """
    device = cu_seqlens.device
    if kv_offset is None:
        kv_offset = total_tokens
    cu_ks, cu_ke = get_compress_cu_seqlens_thd(
        cu_seqlens, cu_seqlens_compressed, ratio=ratio, total_tokens=total_tokens, global_start=global_start
    )

    col_idx = torch.arange(max_n_compressed, device=device).unsqueeze(0)
    seq_major_idx = cu_ks.unsqueeze(1) + col_idx
    visible = col_idx < (cu_ke - cu_ks).unsqueeze(1)
    if seq_to_rank_row is not None:
        seq_major_idx, visible = to_rank_major_rows(seq_major_idx, seq_to_rank_row, visible)
    compress_topk_idxs = torch.where(visible, kv_offset + seq_major_idx, -1)
    return compress_topk_idxs.unsqueeze(0)


# --------------------------------------------------------------------------------------
# Context-parallel compressor input
#
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/95e4bafebaa799d166975ef82066a3c46648e004/megatron/core/transformer/experimental_attention_variant/csa_cp_layout_kernels.py
# --------------------------------------------------------------------------------------


def compressor_boundary_width(ratio: int) -> int:
    # [FUNCTION DOC] HELPER: Returns 8 when ratio is 4 (two groups, to cover the previous-group
    # overlap), otherwise equals the ratio.
    # KNOWN ISSUE: for ratio 128 the width equals the ratio, but a straddling group needs at most
    # ratio minus one rows, so one row is slack; harmless.
    """Hidden rows a rank needs from its left CP neighbour.

    A group straddling the split needs up to ``ratio - 1`` rows the previous rank owns. The
    ratio=4 layers additionally overlap-transform against the preceding group, which
    ``torch.roll`` reads from the row before, so the compact buffer must start one whole
    group early; the wider window is what guarantees that.
    """
    return 8 if ratio == 4 else ratio


def compact_group_capacity(l_local: int, ratio: int) -> int:
    # [FUNCTION DOC] HELPER: Returns max(1, (local rows plus boundary width) divided by ratio, rounded
    # down). Same on every rank for the same inputs.
    """Fixed per-rank compressed-group count, since all-gather needs equal sends.

    A rank's buffer spans ``l_local + d_comp`` global positions, so that many tokens bound the
    groups it can hold. mcore rounds this up for its CuTe tile alignment; the gather here takes
    any length, and every rank derives the same count from the same inputs.
    """
    return max(1, (l_local + compressor_boundary_width(ratio)) // ratio)


def _first_visible_group(range_start: Tensor | int, seg_start: Tensor, ratio: int) -> Tensor:
    # [FUNCTION DOC] HELPER: For a rank whose rows begin at range_start, returns per segment the index
    # of the first group (within the segment) whose tokens begin at or after range_start minus the
    # boundary width: ceiling of max(range_start minus boundary width minus segment start, 0) over ratio.
    """First compressed group a rank starting at ``range_start`` puts in its compact buffer.

    Shared by the compaction and the row map so the slot a group is written to and the slot it
    is looked up in cannot drift apart.
    """
    d_comp = compressor_boundary_width(ratio)
    return torch.div((range_start - d_comp - seg_start).clamp(min=0) + ratio - 1, ratio, rounding_mode="floor")


def compact_gather_index(
    cu_seqlens: Tensor, *, global_start: int, l_local: int, ratio: int, c_cap: int
) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC] HELPER: Plans the compaction for one rank covering global rows
    # [global_start, global_start + l_local). Per segment: groups complete within the segment and
    # ending inside the rank's range are counted (stop group is segment length / ratio rounded down,
    # clipped to range end; first group from the first-visible-group rule; negative counts clamped to
    # zero, and segments not overlapping the range count zero). Counts are cumulated and each of the
    # c_cap slots is mapped to a segment by bucketizing against the cumulative counts; slots at or
    # beyond the total are invalid. For valid slots: group id within segment = first group of that
    # segment plus the slot's rank among that segment's slots; group head = segment start plus id times
    # ratio; the gather indices are head plus 0..ratio-1 minus (range start minus boundary width), i.e.
    # indices into the boundary-then-local concatenation. Invalid slots get -1 everywhere. Returns
    # (flattened 64-bit gather indices of length c_cap times ratio, 32-bit group ids of length c_cap).
    """Source rows feeding each row of a rank's compact compressor input.

    A compressed group belongs to the rank holding its last token, and the group's own tokens
    may start on the previous rank, so sources address the concatenated
    ``[boundary(d_comp) | local(l_local)]`` rows rather than the local rows alone. Trailing
    ``seqlen % ratio`` tokens get no group, matching the non-CP path.

    Returns:
        ``(gather_idx [c_cap * ratio], comp_ids [c_cap])``, both ``-1`` past the groups this
        rank actually produces, which the fixed capacity over-allocates for.
    """
    device = cu_seqlens.device
    d_comp = compressor_boundary_width(ratio)
    n_seg = cu_seqlens.size(0) - 1
    range_start, range_end = global_start, global_start + l_local

    seg_start = cu_seqlens[:-1]
    seg_stop = torch.minimum(cu_seqlens[1:], torch.full_like(cu_seqlens[1:], range_end))
    first_group = _first_visible_group(range_start, seg_start, ratio)
    stop_group = torch.div(seg_stop - seg_start, ratio, rounding_mode="floor")
    n_groups = (stop_group - first_group).clamp(min=0)
    n_groups = torch.where((seg_start < seg_stop) & (range_start < seg_stop), n_groups, 0)

    cu_groups = torch.cat([torch.zeros_like(n_groups[:1]), n_groups.cumsum(0)])
    slot = torch.arange(c_cap, device=device, dtype=cu_seqlens.dtype)
    seg_ids = torch.bucketize(slot, cu_groups[1:], right=True).clamp(max=max(n_seg - 1, 0))
    valid = slot < cu_groups[-1]

    comp_ids = first_group[seg_ids] + (slot - cu_groups[seg_ids])
    group_head = seg_start[seg_ids] + comp_ids * ratio
    # The boundary rows precede the local ones, so both sides of the split share one offset.
    gather_idx = (
        group_head.unsqueeze(1) + torch.arange(ratio, device=device, dtype=cu_seqlens.dtype) - (range_start - d_comp)
    )
    gather_idx = torch.where(valid.unsqueeze(1), gather_idx, -1).flatten()
    return gather_idx.long(), torch.where(valid, comp_ids, -1).int()


class CompressorInputCompact(torch.autograd.Function):
    # [CLASS DOC] HELPER: Differentiable gather of tokens into a fixed number of group slots. Forward
    # inputs: local hidden rows, left-boundary rows, cumulative lengths, global start, ratio, and slot
    # capacity. Outputs a compact tensor (capacity times ratio rows, padding rows zeroed) and the group
    # id per slot (-1 for unused). Backward scatters the compact gradient back to the boundary and
    # local rows; ids get no gradient.
    """Gather each visible compressed group's source tokens into a fixed-capacity buffer.

    Replaces mcore's CuTe kernel, which exists to keep shapes host-known and CUDA-graph
    capturable rather than for the arithmetic: this is integer indexing plus one gather.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_local: Tensor,
        boundary: Tensor,
        cu_seqlens: Tensor,
        global_start: int,
        ratio: int,
        c_cap: int,
    ):
        # [FUNCTION DOC] HELPER: Computes the gather plan, concatenates boundary rows before local
        # rows, selects rows (invalid indices clamped to row zero) and multiplies by a keep mask so
        # padding rows are zero. Saves the indices and mask plus the boundary/local row counts for
        # backward. Returns (compact rows, group ids).
        gather_idx, comp_ids = compact_gather_index(
            cu_seqlens,
            global_start=global_start,
            l_local=hidden_local.size(0),
            ratio=ratio,
            c_cap=c_cap,
        )
        keep = (gather_idx >= 0).view(-1, *([1] * (hidden_local.dim() - 1)))
        source = torch.cat([boundary, hidden_local], dim=0)
        compact = source.index_select(0, gather_idx.clamp(min=0)) * keep
        ctx.save_for_backward(gather_idx, keep)
        ctx.row_split = [boundary.size(0), hidden_local.size(0)]
        return compact, comp_ids

    @staticmethod
    def backward(ctx, grad_compact: Tensor, _grad_comp_ids: Tensor):
        # [FUNCTION DOC] HELPER: Accumulates the masked compact gradient into a zero buffer over the
        # boundary-plus-local rows by index addition, splits it into boundary and local parts, and
        # returns gradients in the order (local, boundary) followed by no gradient for the four other
        # inputs. The group id gradient is ignored.
        gather_idx, keep = ctx.saved_tensors
        # A token belongs to exactly one group, so gather_idx is injective where it is valid
        # and index_add_ never accumulates; keep stops capacity padding reaching row 0.
        grad_source = grad_compact.new_zeros((sum(ctx.row_split),) + grad_compact.shape[1:])
        grad_source.index_add_(0, gather_idx.clamp(min=0), grad_compact * keep)
        grad_boundary, grad_local = grad_source.split(ctx.row_split, dim=0)
        return grad_local, grad_boundary, None, None, None, None


# --------------------------------------------------------------------------------------
# Context-parallel row map and boundary exchange
#
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/95e4bafebaa799d166975ef82066a3c46648e004/megatron/core/transformer/experimental_attention_variant/csa_cp_utils.py
# --------------------------------------------------------------------------------------


class _LeftBoundaryExchange(torch.autograd.Function):
    # [CLASS DOC] HELPER: Differentiable point-to-point halo exchange along the cp group: each rank
    # sends its last rows to the next rank and receives the previous rank's last rows. Backward sends
    # gradients the opposite way.
    """Receive the left CP neighbour's last ``d_comp`` rows."""

    @staticmethod
    def forward(ctx, tensor: Tensor, d_comp: int, cp_group) -> Tensor:
        # [FUNCTION DOC] HELPER: Raises a runtime error if the local row count is smaller than the
        # requested width. Allocates a zero buffer of width rows; non-first ranks post a receive from
        # the previous rank and non-last ranks post a send of their last rows to the next rank; the
        # batched point-to-point operations are issued and waited on (blocking). Stores shape, width and
        # group for backward. Returns the buffer (zeros on rank zero).
        cp_size, cp_rank = cp_group.size(), cp_group.rank()
        if tensor.size(0) < d_comp:
            raise RuntimeError(
                f"DSv4 CP boundary exchange needs at least {d_comp} local rows, " f"got {tensor.size(0)}"
            )
        ctx.input_shape, ctx.d_comp, ctx.cp_group = tensor.shape, d_comp, cp_group
        boundary = tensor.new_zeros((d_comp,) + tuple(tensor.shape[1:]))
        ops = []
        if cp_rank > 0:
            ops.append(dist.P2POp(dist.irecv, boundary, dist.get_global_rank(cp_group, cp_rank - 1), cp_group))
        if cp_rank + 1 < cp_size:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    tensor[-d_comp:].contiguous(),
                    dist.get_global_rank(cp_group, cp_rank + 1),
                    cp_group,
                )
            )
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        return boundary

    @staticmethod
    def backward(ctx, grad_boundary: Tensor):
        # [FUNCTION DOC] HELPER: Sends the boundary gradient back to the previous rank (non-first ranks
        # send) and receives the next rank's boundary gradient (non-last ranks receive), blocking. The
        # input gradient is zeros except the last width rows, which hold the received gradient on
        # non-last ranks. Returns the input gradient and nothing for width and group.
        cp_group = ctx.cp_group
        cp_size, cp_rank = cp_group.size(), cp_group.rank()
        grad_input = grad_boundary.new_zeros(ctx.input_shape)
        # The rows were read from the left neighbour, so their gradient goes back the same way.
        received = grad_boundary.new_zeros((ctx.d_comp,) + tuple(ctx.input_shape[1:]))
        ops = []
        if cp_rank + 1 < cp_size:
            ops.append(dist.P2POp(dist.irecv, received, dist.get_global_rank(cp_group, cp_rank + 1), cp_group))
        if cp_rank > 0:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    grad_boundary.contiguous(),
                    dist.get_global_rank(cp_group, cp_rank - 1),
                    cp_group,
                )
            )
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        if cp_rank + 1 < cp_size:
            grad_input[-ctx.d_comp :] = received
        return grad_input, None, None


def exchange_cp_boundary_hidden(hidden: Tensor, *, ratio: int, cp_group) -> Tensor:
    # [FUNCTION DOC] HELPER: Flattens hidden to two dimensions (rows by the rest), runs the left
    # boundary exchange with the width chosen by the ratio, and restores the trailing shape. Returns
    # the left neighbour's last boundary-width rows (zeros on the first rank). Blocking communication
    # with the cp neighbours; differentiable.
    """Boundary hidden rows the compressor needs from the left CP neighbour.

    Only the compressor needs them: the attention window reads rows the KV all-gather has
    already delivered, so the width is ``d_comp`` rather than mcore's
    ``max(csa_window_size, d_comp)``.
    """
    d_comp = compressor_boundary_width(ratio)
    flat = hidden.reshape(hidden.size(0), -1)
    boundary = _LeftBoundaryExchange.apply(flat, d_comp, cp_group)
    return boundary.reshape((d_comp,) + tuple(hidden.shape[1:]))


def compressed_rank_layout(
    cu_seqlens: Tensor,
    cu_seqlens_compressed: Tensor,
    *,
    l_local: int,
    cp_size: int,
    ratio: int,
    c_cap: int,
) -> Tensor:
    # [FUNCTION DOC] HELPER: For every sequence-major compressed row id (total global rows divided by
    # ratio): find its segment and group index, compute its last token's global position, and its
    # owner rank (position divided by local length, clamped into rank range). For each rank find the
    # sequence-major id of its first slot (cumulative compressed offset of the segment containing
    # its range start plus the first-visible-group rule). The gathered row is owner times capacity
    # plus (id minus owner's first id). Ids beyond the total compressed count get -1. Returns 32-bit.
    """Map sequence-major compressed rows to their rows in the all-gathered buffer.

    All-gather concatenates fixed-capacity per-rank blocks, so a group's sequence-major id is
    not its row: rank ``r`` owns slots ``[r * c_cap, (r + 1) * c_cap)`` and the unused tail of
    each block is padding. A group belongs to the rank holding its last token.

    Returns:
        ``[(l_local * cp_size) // ratio]`` int32, ``-1`` for a row no rank produced.
    """
    device = cu_seqlens.device
    n_seg = cu_seqlens.size(0) - 1
    logical_rows = torch.arange((l_local * cp_size) // ratio, device=device, dtype=cu_seqlens.dtype)
    seg_ids = torch.bucketize(logical_rows, cu_seqlens_compressed[1:], right=True).clamp(max=max(n_seg - 1, 0))
    comp_ids = logical_rows - cu_seqlens_compressed[seg_ids]
    group_last = cu_seqlens[seg_ids] + (comp_ids + 1) * ratio - 1
    owner = torch.div(group_last, l_local, rounding_mode="floor").clamp(0, cp_size - 1)

    rank_starts = torch.arange(cp_size, device=device, dtype=cu_seqlens.dtype) * l_local
    first_seg = torch.bucketize(rank_starts, cu_seqlens[1:], right=True).clamp(max=max(n_seg - 1, 0))
    first_logical = cu_seqlens_compressed[first_seg] + _first_visible_group(rank_starts, cu_seqlens[first_seg], ratio)

    rank_rows = owner * c_cap + (logical_rows - first_logical[owner])
    return torch.where(logical_rows < cu_seqlens_compressed[-1], rank_rows, -1).int()
