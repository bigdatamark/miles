# [MODULE DOC] Derives where each document starts in a packed single-row batch, from the
# position ids (which restart at 0 at each document start). Provides an immutable record type
# for the derived values and one function that builds it; torch is imported lazily inside the
# function. Assumes context parallelism is off, so the full packed row is visible.
"""Shared packed-document boundary derivation for the FSDP backend.

The FSDP backend packs several documents into one ``[1, T]`` forward row; stateful layers (Mamba2,
GatedDeltaNet, attention) must reset per document. This derives the boundaries once from the packed
``position_ids`` (which reset to 0 at each doc start) so every per-arch packing spec shares one
definition. ``packed_seq_context`` returns ``None`` when packing is a no-op (single doc / batch>1).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PackedSeqContext:
    # [CLASS DOC] Immutable record of packed-document boundaries. cu_seqlens: 32-bit integer
    # tensor of length documents plus one holding cumulative start offsets and the total length
    # as the last entry. seq_idx: 32-bit integer tensor of shape one by total length giving each
    # token's zero-based document number. max_seqlen: length of the longest document, as an int.
    cu_seqlens: "object"  # torch.Tensor int32 [num_docs+1]
    seq_idx: "object"  # torch.Tensor int32 [1, T]
    max_seqlen: int


def packed_seq_context(position_ids):
    # [FUNCTION DOC] Build a packed-sequence context, or return none when packing is a no-op.
    # Input: a position id tensor, expected shape one by total tokens; none, a tensor that is not
    # two-dimensional, or a batch dimension other than one all give none. Algorithm: flatten; the
    # document starts are the indices where the position id is zero; zero or one start gives none
    # (single document). cu_seqlens is the starts followed by the total token count, as 32-bit
    # integers; seq_idx is the running count of starts minus one, as a contiguous 32-bit tensor of
    # shape one by tokens; max_seqlen is the largest difference between consecutive offsets.
    # Edge case: if the first position id is not zero, tokens before the first start are not
    # counted in any document boundary (cu_seqlens would not begin at zero).
    """Derive per-document boundaries from packed ``position_ids``, or ``None`` when not packing."""
    import torch

    # FSDP requires context_parallel_size == 1, so this sees the full packed row, not a zigzag CP shard.
    if position_ids is None or position_ids.dim() != 2 or position_ids.shape[0] != 1:
        return None
    pos = position_ids.reshape(-1)
    starts = (pos == 0).nonzero(as_tuple=True)[0]
    if starts.numel() <= 1:
        return None  # single document -> packing is a no-op
    total = torch.tensor([pos.numel()], device=pos.device, dtype=starts.dtype)
    cu_seqlens = torch.cat([starts, total]).to(torch.int32)
    seq_idx = (torch.cumsum((pos == 0).to(torch.int32), dim=0) - 1).to(torch.int32).unsqueeze(0).contiguous()
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
    return PackedSeqContext(cu_seqlens=cu_seqlens, seq_idx=seq_idx, max_seqlen=max_seqlen)
