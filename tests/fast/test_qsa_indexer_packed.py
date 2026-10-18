# [MODULE DOC] Tests (module skipped without torch) that the Qwen3 8B-next QSA indexer, when run
# on a packed variable-length batch, treats each sequence as if it ran alone: key compression
# by block mean never mixes sequences, the packed block-causal mask equals per-sequence masks
# and hides other sequences' blocks, a global block grid wrongly points later sequences at the
# front of the buffer (documenting the bug the packed path avoids), block-local positions restart
# per sequence, and (CUDA only) the full indexer's packed selection equals single-sequence
# selections shifted by the sequence offset. Constants: compression ratio 4, sequence lengths
# 37, 13, 22 (none a multiple of the ratio) and key dimension 8.
"""A packed (thd) QSA indexer batch must score each sequence as if it ran alone."""

import pytest

torch = pytest.importorskip("torch")

from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import (  # noqa: E402
    PackedBlockLayout,
    block_causal_mask,
    compress_keys_by_mean,
    compress_keys_by_mean_packed,
    packed_block_causal_mask,
)

RATIO = 4
LENS = [37, 13, 22]  # none is a multiple of RATIO
DIM = 8


def _packed_positions(cu, total):
    # [FUNCTION DOC] HELPER: Given cumulative sequence boundaries and the total token count,
    # returns each token's position within its own sequence (token index minus its sequence
    # start), by marking sequence starts, cumulative-summing to get segment ids and indexing the
    # start offsets.
    idx = torch.arange(total)
    starts = cu[:-1].long()
    seg = torch.zeros(total, dtype=torch.long)
    seg[starts[1:]] = 1
    seg = seg.cumsum(0)
    return idx - starts[seg]


def _fixture():
    # [FUNCTION DOC] HELPER: Builds the shared packed case: 32-bit cumulative boundaries of the
    # module lengths, the total, per-token in-sequence positions, and random key vectors (total by
    # key dimension) after seeding the generator with 0. Returns (cu, total, positions, keys).
    cu = torch.tensor([0, *LENS]).cumsum(0).to(torch.int32)
    total = int(cu[-1])
    positions = _packed_positions(cu, total)
    torch.manual_seed(0)
    token_k = torch.randn(total, DIM)
    return cu, total, positions, token_k


def test_packed_blocks_never_mix_sequences():
    # [FUNCTION DOC] Compressing keys by block mean with the packed layout equals concatenating
    # the unpacked per-sequence compressions: same shape and values.
    cu, total, positions, token_k = _fixture()
    layout = PackedBlockLayout(cu, positions, RATIO)

    packed = compress_keys_by_mean_packed(token_k, layout)

    expected = torch.cat(
        [compress_keys_by_mean(token_k[int(cu[i]) : int(cu[i + 1])], RATIO) for i in range(len(LENS))],
        dim=0,
    )
    assert packed.shape == expected.shape
    torch.testing.assert_close(packed, expected)


def test_packed_mask_matches_per_sequence_mask():
    # [FUNCTION DOC] The packed block-causal mask has shape (tokens, total blocks). For each
    # sequence, the window of rows and its own block columns equals the single-sequence mask, and
    # every other column in those rows is false.
    cu, total, positions, _ = _fixture()
    layout = PackedBlockLayout(cu, positions, RATIO)

    packed = packed_block_causal_mask(positions, layout, RATIO)
    assert packed.shape == (total, layout.num_blocks)

    block_start = 0
    for i, length in enumerate(LENS):
        lo, hi = int(cu[i]), int(cu[i + 1])
        n_blocks = -(-length // RATIO)
        alone = block_causal_mask(positions[lo:hi], n_blocks, RATIO)

        window = packed[lo:hi, block_start : block_start + n_blocks]
        torch.testing.assert_close(window, alone)

        # nothing outside the sequence's own block range may be visible
        outside = packed[lo:hi].clone()
        outside[:, block_start : block_start + n_blocks] = False
        assert not bool(outside.any()), f"sequence {i} can see blocks of another sequence"
        block_start += n_blocks


def test_global_block_grid_points_at_the_wrong_sequence():
    # [FUNCTION DOC] Builds a block-causal mask on a global block grid using per-sequence
    # positions. For the last query of the second sequence asserts it sees some blocks, all
    # located before the sequence's own first block, and none of its own blocks.
    """Characterises the bug the packed path exists to avoid.

    A global block grid combined with per-sequence positions sends every sequence
    after the first to the blocks at the front of the buffer, and hides its own.
    """
    cu, total, positions, _ = _fixture()
    num_blocks_global = -(-total // RATIO)
    global_mask = block_causal_mask(positions, num_blocks_global, RATIO)

    lo, hi = int(cu[1]), int(cu[2])  # the second sequence
    own_lo = lo // RATIO
    row = hi - 1  # its last query, which has the most history
    visible = global_mask[row].nonzero().flatten()

    assert visible.numel() > 0
    assert int(visible.max()) < own_lo, "expected the global grid to look before the sequence start"
    assert not bool(global_mask[row, own_lo:].any()), "expected the sequence's own blocks to be hidden"


def test_layout_block_local_positions_restart_per_sequence():
    # [FUNCTION DOC] The layout's block-local index equals the concatenation of ranges from zero
    # to each sequence's block count (ceiling of length over ratio), and the total block count is
    # the sum of those counts.
    cu, total, positions, _ = _fixture()
    layout = PackedBlockLayout(cu, positions, RATIO)

    expected = torch.cat([torch.arange(-(-length // RATIO)) for length in LENS])
    torch.testing.assert_close(layout.block_local, expected)
    assert layout.num_blocks == sum(-(-length // RATIO) for length in LENS)


@pytest.mark.parametrize("lens", [[37, 13], [64, 64], [1, 41]])
def test_selection_equals_single_sequence_runs(lens):
    # [FUNCTION DOC] Parametrized over three length lists; skipped without CUDA. Builds a small
    # Megatron transformer config (hidden 64, 4 heads, bf16) with indexer settings (2 heads, 1
    # key/value head, head dim 32, budget 16, the module compression ratio), creates the QSA
    # indexer on the GPU, packed boundaries/positions, random bf16 hidden states and a neox-style
    # rotary embedding (base 1e7, rotary dim 8). Without gradients runs the indexer on the packed
    # batch and then on each sequence alone; per row compares the sets of non-negative selected
    # token ids (alone results shifted by the sequence start), because an alone result can be
    # narrower.
    """Full indexer: packed selection == per-sequence selection + token offset."""
    if not torch.cuda.is_available():
        pytest.skip("indexer projection needs a CUDA device (TELinear)")

    from megatron.core.transformer.transformer_config import TransformerConfig

    from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import Qwen38NextQSAIndexer

    hidden = 64
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    config.qwen3_8_next_indexer_n_heads = 2
    config.qwen3_8_next_indexer_kv_heads = 1
    config.qwen3_8_next_indexer_head_dim = 32
    config.qwen3_8_next_indexer_budget = 16
    config.qwen3_8_next_indexer_compress_ratio = RATIO

    indexer = Qwen38NextQSAIndexer(config, layer_number=1).cuda()
    cu = torch.tensor([0, *lens]).cumsum(0).to(torch.int32).cuda()
    total = int(cu[-1])
    positions = _packed_positions(cu.cpu(), total).cuda()
    torch.manual_seed(0)
    hidden_states = torch.randn(total, hidden, device="cuda", dtype=torch.bfloat16)
    rotary_dim = 8
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda") / rotary_dim))
    freqs = torch.outer(torch.arange(total, dtype=torch.float32, device="cuda"), inv_freq)
    rotary_pos_emb = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]

    with torch.no_grad():
        packed = indexer(hidden_states, positions, rotary_pos_emb, cu_seqlens=cu)
        for i, length in enumerate(lens):
            lo = int(cu[i])
            sub_cu = torch.tensor([0, length], dtype=torch.int32, device="cuda")
            alone = indexer(
                hidden_states[lo : lo + length].contiguous(),
                positions[lo : lo + length].contiguous(),
                rotary_pos_emb,
                cu_seqlens=sub_cu,
            )
            shifted = torch.where(alone >= 0, alone + lo, alone)
            got = packed[lo : lo + length]

            # a sequence scored alone can come back narrower, so compare token sets
            for row in range(length):
                got_set = {int(t) for t in got[row].tolist() if t >= 0}
                want_set = {int(t) for t in shifted[row].tolist() if t >= 0}
                assert got_set == want_set, (
                    f"sequence {i} row {row}: packed selection {sorted(got_set)} "
                    f"!= single-sequence {sorted(want_set)}"
                )
