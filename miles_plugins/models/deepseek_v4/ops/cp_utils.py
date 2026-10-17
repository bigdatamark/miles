# [MODULE DOC] HELPER: Context-parallel (CP) utilities for DeepSeek-V4 attention under a contiguous CP
# layout, where rank r holds the r-th contiguous block of the sequence. Provides a differentiable
# all-gather, and builders for global query positions, sliding-window key indices, compressed-key
# indices (causally masked, with -1 marking invalid slots), and rotary-frequency slices.
from functools import lru_cache

import torch
from torch import Tensor


@lru_cache(1)
def all_gather_cp(tensor: Tensor, dim: int, cp_group: torch.distributed.ProcessGroup) -> Tensor:
    # [FUNCTION DOC] HELPER: Differentiable all-gather of the tensor over the cp group, then
    # concatenation along the given dim; because the layout is contiguous the result is in natural
    # sequence order. Communication: collective across the group.
    # KNOWN ISSUE: wrapped in a one-entry result cache keyed on the tensor object and arguments, so
    # a repeat call with the same tensor object skips the collective and returns the cached result
    # (which may be stale if the tensor was mutated in place) and keeps the tensor alive.
    """All-gather tensor across CP ranks on `dim`. Contiguous CP = result already in natural order."""
    return torch.cat(torch.distributed.nn.functional.all_gather(tensor, group=cp_group), dim=dim)


def get_q_positions_for_cp(
    seqlen_local: int,
    *,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    device,
) -> Tensor:
    # [FUNCTION DOC] HELPER: Returns an integer vector of global positions of this rank's query tokens:
    # a range starting at rank times local length (zero when cp size is at most one or no group is
    # given), of length local sequence length, created on the given device.
    """Get global positions for local q tokens (contiguous CP)."""
    if cp_size <= 1 or cp_group is None:
        return torch.arange(0, seqlen_local, device=device)
    cp_rank = cp_group.rank()
    start = cp_rank * seqlen_local
    return torch.arange(start, start + seqlen_local, device=device)


def get_window_topk_idxs_cp(
    q_positions: Tensor,
    *,
    window_size: int,
    cp_size: int,
    bsz: int,
) -> Tensor:
    # [FUNCTION DOC] HELPER: Builds sliding-window key indices per query. For each query position p
    # the window covers keys from max(p minus window size plus one, 0) upward, for min(global length,
    # window size) slots, where global length is local length times cp size. Slots beyond the query
    # position (future keys) are set to -1. Returns a 32-bit integer tensor of shape batch, local
    # length, window slots, expanded (a view, not a copy) along batch.
    """Get window topk indices (CP-aware)."""
    device = q_positions.device
    seqlen_local = q_positions.shape[0]
    seqlen_global = seqlen_local * cp_size
    base = q_positions.unsqueeze(1)
    k_pos = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen_global, window_size), device=device)
    topk_idxs = torch.where(k_pos > base, -1, k_pos).int()
    result = topk_idxs.unsqueeze(0).expand(bsz, -1, -1)

    return result


def get_compress_topk_idxs_cp(
    q_positions: Tensor,
    *,
    ratio: int,
    cp_size: int,
    bsz: int,
) -> Tensor:
    # [FUNCTION DOC] HELPER: Builds the full (not top-k selected) list of compressed-group indices per
    # query. Group g (0 to global length / ratio minus one) is visible to a query at position p iff g is
    # below (p plus one) integer-divided by ratio; visible slots hold g plus global length (the
    # compressed keys are laid out after the global-length uncompressed keys), invalid slots hold -1.
    # Returns a 32-bit integer tensor of shape batch, local length, group count, as an expanded view.
    """Get static compress topk indices (CP-aware)."""
    device = q_positions.device
    seqlen_local = q_positions.shape[0]
    seqlen_global = seqlen_local * cp_size
    offset = seqlen_global
    k_group_idx = torch.arange(seqlen_global // ratio, device=device).repeat(seqlen_local, 1)
    q_first_invalid_group = (q_positions + 1).unsqueeze(1) // ratio
    invalid_mask = k_group_idx >= q_first_invalid_group
    compress_topk_idxs = torch.where(invalid_mask, -1, k_group_idx + offset).int()
    result = compress_topk_idxs.unsqueeze(0).expand(bsz, -1, -1)

    return result


def get_freqs_cis_for_cp(
    freqs_cis: Tensor,
    seqlen_local: int,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    stride: int = 1,
) -> Tensor:
    # [FUNCTION DOC] HELPER: Slices precomputed rotary frequencies (indexed by global position) for
    # this rank: positions from rank times local length for local-length entries, taking every
    # stride-th entry (start at zero when cp size is one or no group). Returns a view of the input.
    """Get freqs_cis for this CP rank (contiguous slice)."""
    if cp_size == 1 or cp_group is None:
        return freqs_cis[:seqlen_local:stride]
    cp_rank = cp_group.rank()
    start = cp_rank * seqlen_local
    return freqs_cis[start : start + seqlen_local : stride]
