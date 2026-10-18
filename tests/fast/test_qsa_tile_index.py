# [MODULE DOC] Tests (skipped without torch and triton) of the QSA block-sparse attention tile
# index builders: for packed sequences of various lengths (including starts off the block grid),
# the forward tile map (per query tile, the key tiles of size 64) and the forward/backward pair
# (key tiles of size 32 per query tile, and query tiles per key tile) must list every key tile
# that any query in the tile can attend (block selected, causal, same sequence); forward and
# backward pairs must describe the same relation; and block spans stay inside their sequence.
# Constants: compression ratio 4, query tile 64, and a module-level list of packs (fixed ones plus
# four from a generator seeded with 0 with 2-6 sequences of 1-300 tokens, drawn at import time).
"""The QSA tile map must list every key tile a query can attend, including packs off the block grid."""

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from miles.kernels.attention.qsa.block_sparse_attention import build_tile_index, build_tile_index_pair  # noqa: E402
from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import PackedBlockLayout  # noqa: E402

RATIO = 4
BQ = 64

_rng = random.Random(0)
PACKS = [
    [61, 128],
    [61, 61, 61, 200],
    [64, 128],
    [1, 1, 1, 1, 1, 200],
    *[[_rng.randint(1, 300) for _ in range(_rng.randint(2, 6))] for _ in range(4)],
]


def _packed_case(lens, dense: bool, seed: int = 0):
    # [FUNCTION DOC] HELPER: Builds the packed block layout from sequence lengths and a block
    # selection bitmap (tokens by blocks, 8-bit). For each token the allowed blocks run from its
    # sequence's first block through the block containing its own position. Dense selects all of
    # them; otherwise each is kept with probability 0.3 (seeded generator) and the token's own
    # block is always added. Returns (layout, in-sequence positions, selection bitmap).
    """Layout plus a causal, in-sequence block selection; ``dense`` selects every allowed block."""
    cu = torch.tensor([0, *lens]).cumsum(0)
    total = int(cu[-1])
    starts = cu[:-1]
    seg = torch.zeros(total, dtype=torch.long)
    seg[starts[1:]] = 1
    seg = seg.cumsum(0)
    positions = torch.arange(total) - starts[seg]
    layout = PackedBlockLayout(cu, positions, RATIO)

    g = torch.Generator().manual_seed(seed)
    sel = torch.zeros(total, layout.num_blocks, dtype=torch.uint8)
    for t in range(total):
        own = int(layout.token_block_start[t])
        allowed = torch.arange(own, own + int(positions[t]) // RATIO + 1)
        if not dense:
            allowed = allowed[torch.rand(allowed.numel(), generator=g) < 0.3]
            allowed = torch.cat([allowed, torch.tensor([own + int(positions[t]) // RATIO])])
        sel[t, allowed] = 1
    return layout, positions, sel


def _attended_keys(t, layout, positions, sel):
    # [FUNCTION DOC] HELPER: For query token t, takes all tokens from its sequence's first token
    # through t, maps each to its block and returns those whose block is selected for t. The
    # positions argument is unused.
    """Keys the kernel mask lets query ``t`` see: block selected, causal, same sequence."""
    tok_base = int(layout.token_start[t])
    blk_base = int(layout.token_block_start[t])
    keys = torch.arange(tok_base, t + 1)
    blocks = blk_base + (keys - tok_base) // RATIO
    return keys[sel[t, blocks] != 0]


def _tile_sets(lst, cnt):
    # [FUNCTION DOC] HELPER: Converts a padded tile list (rows of tile ids) with per-row counts
    # into a list of sets, each holding the first count entries of its row.
    return [set(lst[i, : int(cnt[i])].tolist()) for i in range(cnt.numel())]


@pytest.mark.parametrize("lens", PACKS, ids=lambda lens: "-".join(map(str, lens)))
@pytest.mark.parametrize("dense", [True, False], ids=["dense", "sparse"])
def test_every_attended_key_tile_is_listed(lens, dense):
    # [FUNCTION DOC] Parametrized over every pack and dense/sparse selection. Builds the forward
    # tile index (key tile size 64) and the backward pair (key tile size 32). Asserts the query
    # tile set count equals the ceiling of tokens over 32; for every query token, every attended
    # key falls in a listed forward key tile (size 64) and a listed backward key tile (size 32) of
    # that token's query tile; and the set of (query tile, key tile) pairs from the key-tile lists
    # equals the one from the query-tile lists.
    layout, positions, sel = _packed_case(lens, dense)
    first, last = layout.block_first_token, layout.block_last_token
    total = sel.shape[0]

    fwd = _tile_sets(*build_tile_index(sel, first, last, BQ, 64))
    klist, kcnt, qlist, qcnt = build_tile_index_pair(sel, first, last, BQ, 32)
    bwd_k = _tile_sets(klist, kcnt)
    bwd_q = _tile_sets(qlist, qcnt)

    assert len(bwd_q) == -(-total // 32)
    for t in range(total):
        keys = _attended_keys(t, layout, positions, sel)
        missing_fwd = sorted({int(k) for k in keys if int(k) // 64 not in fwd[t // BQ]})
        missing_bwd = sorted({int(k) for k in keys if int(k) // 32 not in bwd_k[t // BQ]})
        assert not missing_fwd, f"query {t}: forward tile list misses keys {missing_fwd}"
        assert not missing_bwd, f"query {t}: backward tile list misses keys {missing_bwd}"

    forward_pairs = {(qt, kt) for qt, tiles in enumerate(bwd_k) for kt in tiles}
    backward_pairs = {(qt, kt) for kt, tiles in enumerate(bwd_q) for qt in tiles}
    assert forward_pairs == backward_pairs


def test_block_spans_stay_inside_their_sequence():
    # [FUNCTION DOC] For packs of 61, 128 and 3 tokens, each block's first token equals its
    # sequence start plus ratio times its block-local index, its last token is inside its
    # sequence, and its span is shorter than the ratio.
    layout, _, _ = _packed_case([61, 128, 3], dense=True)
    cu = torch.tensor([0, 61, 189, 192])
    seq = layout.block_seq
    assert torch.equal(layout.block_first_token, cu[seq] + RATIO * layout.block_local)
    assert bool((layout.block_last_token < cu[seq + 1]).all())
    assert bool((layout.block_last_token - layout.block_first_token < RATIO).all())
