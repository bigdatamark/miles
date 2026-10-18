# [MODULE DOC] CPU unit tests for reassembling the full natural-order token row of a packed
# (THD) batch from per-context-parallel-rank rows in the Qwen3-VL plugin, and re-slicing it per
# rank in zigzag order. Includes a reference copy of the training code's per-sample zigzag slicing
# and a helper that mimics batch packing (slice per sample per rank, concatenate, pad to a
# multiple, scale cumulative boundaries by CP size). Also tests that an indivisible segment makes
# the reassembly give up (return none) so the caller falls back to the dense path.
"""CPU unit test for the Qwen3-VL CP+THD packed mRoPE reconstruction (issue #1296).

Under context parallelism each rank's THD row holds only its zigzag chunks of every packed
segment. `_reassemble_full_row` de-interleaves the all-gathered per-rank rows back to the
full natural-order row so per-segment MRoPE positions can be rebuilt and re-sliced. This
test checks that reconstruction is the exact inverse of `slice_with_cp` (the function miles
uses to shard the tokens), and that re-slicing with `_natural_to_zigzag_slice` round-trips.
"""

import pytest
import torch
import torch.nn.functional as F

from miles_plugins.models.qwen3_vl import _natural_to_zigzag_slice, _reassemble_full_row


def _slice_with_cp(tokens, cp_size, cp_rank, pad_value=0):
    # [FUNCTION DOC] HELPER: Zigzag slice of one sample's 1-D token tensor for a context-parallel
    # rank: the chunk size is the ceiling of length over twice the CP size; the tensor is padded
    # on the right with the pad value (default 0) to twice the CP size times the chunk; the result
    # concatenates chunk number rank and chunk number (2 times CP size minus rank minus one).
    # Returns the concatenated slice.
    """Reference copy of cp_utils.slice_with_cp's THD zigzag slicing (per sample)."""
    token_len = len(tokens)
    chunk = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    pad = 2 * cp_size * chunk - token_len
    if pad:
        tokens = F.pad(tokens, (0, pad), value=pad_value)
    s1, e1 = chunk * cp_rank, chunk * (cp_rank + 1)
    s2, e2 = chunk * (2 * cp_size - cp_rank - 1), chunk * (2 * cp_size - cp_rank)
    return torch.cat([tokens[s1:e1], tokens[s2:e2]])


def _build_like_get_batch(sample_lens, cp_size, pad_size=8):
    # [FUNCTION DOC] HELPER: Creates one sample per length with unique nonzero consecutive ids,
    # builds each rank's row as the concatenation of its zigzag slices of all samples, and builds
    # cumulative boundaries from rank zero's slice lengths. If the row length is not a multiple of
    # the pad size (default 8), pads every rank's row with zeros and appends the pad as a final
    # segment. Finally scales all boundaries by the CP size. Returns (samples, per-rank rows,
    # boundaries).
    """Mimic miles get_batch THD+CP packing: per-sample zigzag slice, concat, pad, cu*cp."""
    samples = []
    base = 1
    for L in sample_lens:
        samples.append(torch.arange(base, base + L))  # unique nonzero ids
        base += L
    per_rank = []
    for r in range(cp_size):
        row = torch.cat([_slice_with_cp(t, cp_size, r) for t in samples])
        per_rank.append(row)
    cu = [0]
    for t in samples:
        cu.append(cu[-1] + _slice_with_cp(t, cp_size, 0).size(0))
    final_pad = (pad_size - per_rank[0].size(0) % pad_size) % pad_size
    if final_pad:
        per_rank = [F.pad(row, (0, final_pad), value=0) for row in per_rank]
        cu.append(cu[-1] + final_pad)
    cu = [x * cp_size for x in cu]
    return samples, per_rank, cu


@pytest.mark.parametrize(
    "cp_size,sample_lens",
    [(2, [10, 7, 13]), (2, [16, 16]), (4, [20, 9, 30, 5]), (2, [3]), (4, [40, 17])],
)
def test_reassemble_is_inverse_of_slice_with_cp(cp_size, sample_lens):
    # [FUNCTION DOC] Parametrized over five (CP size, sample lengths) cases. Packs like the batch
    # builder, asserts the last boundary equals CP size times the local row length, reassembles
    # the full row from the per-rank rows, and asserts it is not none and has that many elements;
    # each sample's tokens appear in order at the start of its segment; and re-slicing each
    # segment of the full row in zigzag order for each rank and concatenating reproduces that
    # rank's row exactly.
    samples, per_rank, cu = _build_like_get_batch(sample_lens, cp_size)
    local_len = per_rank[0].size(0)
    assert cu[-1] == cp_size * local_len

    full = _reassemble_full_row(per_rank, cu, cp_size)
    assert full is not None and full.numel() == cu[-1]

    # Each real sample's tokens reappear (in order) at the start of its segment.
    for i, t in enumerate(samples):
        seg = full[cu[i] : cu[i + 1]]
        assert torch.equal(seg[: t.numel()], t)

    # Re-slicing the full row per segment recovers exactly each rank's local chunks.
    for r in range(cp_size):
        recon = []
        for i in range(len(cu) - 1):
            recon.append(_natural_to_zigzag_slice(full[cu[i] : cu[i + 1]], cp_size, r, dim=0))
        assert torch.equal(torch.cat(recon), per_rank[r])


def test_reassemble_bails_on_indivisible_segment():
    # [FUNCTION DOC] With boundaries [0, 6] and CP size 2 (segment length 6 not divisible by 4),
    # the reassembly returns none.
    # A segment length not divisible by 2*cp -> None (caller falls back to dense path).
    cu = [0, 6]  # 6 not divisible by 2*cp=4
    gathered = [torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    assert _reassemble_full_row(gathered, cu, cp_size=2) is None
