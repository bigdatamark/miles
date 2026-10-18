# [MODULE DOC]
# Cpu tests of the linear-attention weight layout conversions between the flat q,k,v row layout and
# the group-major layout (rows grouped per key head), including a round trip, splitting inversion,
# tensor-parallel chunk consistency, and agreement of activation splits. Constant: a head
# configuration with 4 key heads, 12 value heads, key head dim 16 and value head dim 32.
import torch

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import (
    LinearAttnHeads,
    group_major,
    qkv_flat_to_group_major,
    qkv_group_major_to_flat,
    split_group_major,
)

HEADS = LinearAttnHeads(num_k_heads=4, num_v_heads=12, head_k_dim=16, head_v_dim=32)


def test_qkv_layout_roundtrip():
    # [FUNCTION DOC] For weights of qkv row count with trailing shapes (24) and (1, 4), converts flat to
    # group-major, asserts the shape is unchanged, and that converting back reproduces the original
    # exactly.
    for rest in ((24,), (1, 4)):
        weight = torch.randn(HEADS.qkv_dim, *rest)
        grouped = qkv_flat_to_group_major(weight, HEADS)
        assert grouped.shape == weight.shape
        assert torch.equal(qkv_group_major_to_flat(grouped, HEADS), weight)


def test_group_major_split_inverts():
    # [FUNCTION DOC] Builds three random blocks with rows equal to the key head count times 3, 5 and 1
    # (8 columns each), merges them to group-major, splits back using the per-group sizes 3, 5, 1, and
    # asserts each block is restored exactly.
    blocks = [torch.randn(HEADS.num_k_heads * n, 8) for n in (3, 5, 1)]
    restored = split_group_major(group_major(blocks, HEADS), [3, 5, 1], HEADS)
    assert all(torch.equal(a, b) for a, b in zip(restored, blocks, strict=True))


def test_tp_chunk_is_the_ranks_heads():
    # [FUNCTION DOC] With tensor parallel size 2 and an 8-column random weight, splits the flat weight
    # into q, k, v row blocks. For each rank's chunk of the group-major weight, asserts it equals the
    # group-major conversion of that rank's own q, k, v row slices (using the per-rank head
    # configuration), with the slices taken by local key and value dimensions.
    """Chunk r of the group-major q/k/v equals the group-major tensor built from rank r's heads alone."""
    tp, cols = 2, 8
    weight = torch.randn(HEADS.qkv_dim, cols)
    q, k, v = weight.split([HEADS.key_dim, HEADS.key_dim, HEADS.value_dim])
    local = HEADS.local(tp)
    for rank, chunk in enumerate(qkv_flat_to_group_major(weight, HEADS).chunk(tp)):
        kr = slice(rank * local.key_dim, (rank + 1) * local.key_dim)
        vr = slice(rank * local.value_dim, (rank + 1) * local.value_dim)
        assert torch.equal(chunk, qkv_flat_to_group_major(torch.cat([q[kr], k[kr], v[vr]]), local))


def test_activation_split_matches_weight_layout():
    # [FUNCTION DOC] With hidden size 24, projects random activations (2x5x24) by the flat weight and by
    # the group-major weight. The flat result is split into q, k, v; the group-major result is viewed per
    # key head and split into key dim, key dim and group value dim parts. Asserts each flattened
    # group-major part equals the flat part within 1e-5.
    """A projection with group-major rows splits into the q/k/v a flat projection would give."""
    hidden = 24
    weight = torch.randn(HEADS.qkv_dim, hidden)
    x = torch.randn(2, 5, hidden)
    q_ref, k_ref, v_ref = (x @ weight.T).split([HEADS.key_dim, HEADS.key_dim, HEADS.value_dim], dim=-1)
    grouped = (x @ qkv_flat_to_group_major(weight, HEADS).T).view(2, 5, HEADS.num_k_heads, -1)
    q, k, v = grouped.split([HEADS.head_k_dim, HEADS.head_k_dim, HEADS.group_value_dim], dim=-1)
    for a, b in ((q, q_ref), (k, k_ref), (v, v_ref)):
        assert torch.allclose(a.flatten(-2), b, atol=1e-5)
