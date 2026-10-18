# [MODULE DOC] Tests (module skipped without torch) that the trainer's QSA indexer rotary
# embedding reproduces the serving engine's eager neox-style rotation bit for bit (zero
# tolerance), using an independent re-implementation of the engine's cos/sin cache and rotation
# as reference, and that position zero is an identity rotation. Constants: rotary base 1e7,
# rotary dimension 64 (head dim 256 times partial factor 0.25), head dim 256 and 4096 positions.
"""The trainer's QSA indexer RoPE must reproduce sglang's eager neox rotation bit for bit."""

import pytest

torch = pytest.importorskip("torch")

from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import apply_indexer_rope  # noqa: E402

BASE = 1e7
ROTARY_DIM = 64  # head_dim 256 * partial_rotary_factor 0.25
HEAD_DIM = 256
MAX_POS = 4096


def _megatron_rotary_pos_emb(max_pos: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Builds the non-interleaved Megatron rotary table: inverse frequencies
    # base to the power of (even index over rotary dim), outer product with positions 0 to
    # max_pos minus one in float32, frequencies duplicated along the last axis and given two
    # singleton axes in the middle. Returns shape (max_pos, 1, 1, rotary dim).
    """Megatron ``RotaryEmbedding.get_emb`` for rotary_interleaved=False."""
    inv_freq = 1.0 / (BASE ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM))
    seq = torch.arange(max_pos, dtype=inv_freq.dtype)
    freqs = torch.outer(seq, inv_freq)
    return torch.cat((freqs, freqs), dim=-1)[:, None, None, :]


def _sglang_reference(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Reference rotation of the serving engine. Builds a cos/sin cache for
    # all positions up to the maximum, selects rows for the given positions, repeats the halves to
    # full width and splits into cos and sin truncated to half the rotary dim, casts them to the
    # data dtype, and rotates the first rotary-dim features in neox style (first half becomes x1
    # cos minus x2 sin, second half x2 cos plus x1 sin). Features beyond the rotary dim are left
    # unchanged. Returns a tensor of the same shape as the input.
    """sglang RotaryEmbedding._compute_cos_sin_cache + QSAIndexer.apply_rope + apply_rotary_emb."""
    inv_freq = 1.0 / (BASE ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float) / ROTARY_DIM))
    t = torch.arange(MAX_POS, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    cos_sin = cos_sin_cache.index_select(0, positions.flatten())
    last_dim = cos_sin.size()[-1]
    cos, sin = cos_sin.reshape(-1, 2, last_dim // 2).repeat(1, 1, 2).chunk(2, dim=-2)
    half_rotary_dim = ROTARY_DIM // 2
    cos = cos.reshape(positions.numel(), -1)[:, :half_rotary_dim]
    sin = sin.reshape(positions.numel(), -1)[:, :half_rotary_dim]

    rot = x[..., :ROTARY_DIM]
    cos = cos.unsqueeze(-2).to(rot.dtype)
    sin = sin.unsqueeze(-2).to(rot.dtype)
    x1, x2 = torch.chunk(rot, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([torch.cat((o1, o2), dim=-1), x[..., ROTARY_DIM:]], dim=-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("heads", [1, 8])
def test_indexer_rope_matches_sglang(dtype, heads):
    # [FUNCTION DOC] Parametrized over bfloat16/float32 and 1 or 8 heads. With a generator seeded
    # by the head count, draws 257 tokens of random features and random positions, applies the
    # trainer's rotary function with the Megatron table and the reference, and asserts the output
    # dtype is preserved, equals the reference exactly, and leaves non-rotary features exactly
    # equal to the input.
    g = torch.Generator().manual_seed(heads)
    tokens = 257
    x = torch.randn(tokens, heads, HEAD_DIM, generator=g).to(dtype)
    positions = torch.randint(0, MAX_POS, (tokens,), generator=g)

    got = apply_indexer_rope(x, _megatron_rotary_pos_emb(MAX_POS), positions)
    want = _sglang_reference(x, positions)

    assert got.dtype == dtype
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(got[..., ROTARY_DIM:], x[..., ROTARY_DIM:], rtol=0, atol=0)


def test_indexer_rope_rotates_position_zero_to_identity():
    # [FUNCTION DOC] Rotating random bf16 features at position zero (table of 16 positions) gives
    # exactly the input.
    x = torch.randn(4, 2, HEAD_DIM).to(torch.bfloat16)
    got = apply_indexer_rope(x, _megatron_rotary_pos_emb(16), torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(got, x, rtol=0, atol=0)
