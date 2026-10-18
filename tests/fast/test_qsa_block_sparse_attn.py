# [MODULE DOC] GPU tests (module skipped without torch or CUDA) checking the tensor-core
# block-sparse QSA attention kernel against the gather kernel and an fp64 reference: forward
# output agreement, gradient agreement, correctness when packed sequence starts are not
# multiples of the block size (a sequence must not borrow its neighbour's blocks), and the
# consistency of the forward and backward tile index lists. Constants: block size 4, 4 query
# heads, 1 key/value head, head dimension 128 and the default attention scale (head dimension to
# the power minus one half).
"""The tensor-core QSA kernel must match the gather kernel, including off-grid packed boundaries."""

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():  # pragma: no cover - needs a GPU
    pytest.skip("QSA kernels need CUDA", allow_module_level=True)

from miles.kernels.attention.qsa import (  # noqa: E402
    qsa_block_sparse_attention,
    qsa_sparse_attention,
    qsa_sparse_attention_from_indices,
)
from miles.kernels.attention.qsa.block_sparse_attention import (  # noqa: E402
    build_tile_index_pair,
    selection_to_block_bitmap,
)

BLK = 4
HQ, HKV, D = 4, 1, 128
SCALE = D**-0.5


def _single_sequence_case(T=384, budget=64, seed=0):
    # [FUNCTION DOC] HELPER: Builds a random single-sequence case on the GPU with a seeded
    # generator: bfloat16 query (T by heads by head dim), key and value (T by one head by head
    # dim), and a selection index tensor. For each token, random scores over blocks (T // block
    # size plus one) are masked to -1 for blocks after the token's own block; the top
    # (budget // block size) blocks (or all if fewer) are chosen, expanded to token indices, and
    # indices after the token's position are replaced by -1. Returns (q, k, v, indices as 32-bit
    # integers).
    """Selection as the indexer emits it: distinct blocks, tail clamped to the position."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, HQ, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    rows = torch.arange(T, device="cuda")
    nblk = budget // BLK
    scores = torch.rand(T, T // BLK + 1, device="cuda", generator=g)
    allowed = torch.arange(scores.shape[1], device="cuda").unsqueeze(0) <= (rows // BLK).unsqueeze(1)
    scores = torch.where(allowed, scores, torch.full_like(scores, -1.0))
    blk = scores.topk(min(nblk, scores.shape[1]), dim=1).indices
    tok = (blk.unsqueeze(-1) * BLK + torch.arange(BLK, device="cuda")).reshape(T, -1)
    idx = torch.where(tok <= rows.unsqueeze(1), tok, torch.full_like(tok, -1)).to(torch.int32)
    return q, k, v, idx


def _reference(q, k, v, idx, scale):
    # [FUNCTION DOC] HELPER: Slow double-precision reference. For each token and each query head,
    # takes the valid (non-negative) selected indices, computes scaled dot-product scores against
    # the keys of the corresponding key/value head (heads grouped), softmax over the selection,
    # and the weighted sum of values. Tokens with no selection keep zeros. Returns a float64 output
    # of shape tokens by query heads by head dim.
    """fp64, one query at a time, straight from the documented semantics."""
    T, hq, d = q.shape
    group = hq // k.shape[1]
    out = torch.zeros(T, hq, d, device=q.device, dtype=torch.float64)
    qd, kd, vd = q.double(), k.double(), v.double()
    for t in range(T):
        sel = idx[t][idx[t] >= 0].long()
        if sel.numel() == 0:
            continue
        for h in range(hq):
            s = (qd[t, h].unsqueeze(0) * kd[sel, h // group]).sum(-1) * scale
            p = torch.softmax(s, dim=0)
            out[t, h] = (p.unsqueeze(-1) * vd[sel, h // group]).sum(0)
    return out


def _rel(a, b):
    # [FUNCTION DOC] HELPER: Compares two tensors in double precision; returns the maximum
    # absolute difference and the mean absolute difference divided by the mean absolute value of
    # the second tensor (plus a tiny constant), both as Python floats.
    d = (a.double() - b.double()).abs()
    return d.max().item(), (d.mean() / (b.double().abs().mean() + 1e-12)).item()


def test_matches_gather_kernel_and_reference():
    # [FUNCTION DOC] On a single-sequence case runs the reference, the gather kernel and the
    # from-indices tensor-core kernel. Asserts the new kernel's max error is at most four times
    # the old kernel's, its relative error against the reference is under 1e-2, and its relative
    # difference from the old kernel is under 1e-2.
    q, k, v, idx = _single_sequence_case()
    ref = _reference(q, k, v, idx, SCALE)
    old = qsa_sparse_attention(q, k, v, idx, SCALE)
    new = qsa_sparse_attention_from_indices(q, k, v, idx, SCALE)

    old_max, _ = _rel(old, ref)
    new_max, new_rel = _rel(new, ref)
    # neither is closer to exact than the other by more than the bf16 floor
    assert new_max <= old_max * 4, (new_max, old_max)
    assert new_rel < 1e-2, new_rel
    assert _rel(new, old)[1] < 1e-2


def test_gradients_match_gather_kernel():
    # [FUNCTION DOC] Runs forward and backward (with a random output gradient) through the
    # from-indices kernel and the gather kernel on a single-sequence case, and asserts the
    # gradients of query, key and value each differ from each other by relative error under 1e-2.
    q, k, v, idx = _single_sequence_case()
    gout = torch.randn(q.shape, device="cuda", dtype=torch.bfloat16)

    def run(fn):
        # [FUNCTION DOC] HELPER: Clones q, k, v as leaf tensors requiring grad, runs the given
        # kernel with the selection indices and scale, back-propagates the output gradient and
        # returns the three gradients.
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        fn(qq, kk, vv, idx, SCALE).backward(gout)
        return qq.grad, kk.grad, vv.grad

    for a, b, name in zip(run(qsa_sparse_attention_from_indices), run(qsa_sparse_attention), "qkv", strict=True):
        rel = _rel(a, b)[1]
        assert rel < 1e-2, (name, rel)


@pytest.mark.parametrize("lens", [[301, 211], [61, 128]])
def test_packed_boundary_not_a_multiple_of_the_block(lens):
    # [FUNCTION DOC] Parametrized over two packed length lists. Builds random q, k, v for the
    # concatenated tokens and per-token metadata: token base (sequence start), block base
    # (cumulative block counts), position within the sequence, and per-block first and last token
    # (last clipped to the sequence end). For each token picks up to 7 random earlier blocks plus
    # its own block, sets those in a block-selection bitmap and builds a token index list (clipped
    # to the sequence start and to the token itself). Runs the block-sparse kernel with these
    # inputs and compares to the reference both overall and for the tail after the first
    # sequence (relative error under 1e-2), then compares its gradients with those of the gather
    # kernel for the same indices (each under 1e-2).
    """A sequence starting off the global block grid must not borrow its neighbour's blocks."""
    T = sum(lens)
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(T, HQ, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)

    cu = torch.tensor([0, *lens], device="cuda").cumsum(0)
    tok_base = torch.zeros(T, dtype=torch.int32, device="cuda")
    blk_base = torch.zeros(T, dtype=torch.int32, device="cuda")
    pos = torch.zeros(T, dtype=torch.int64, device="cuda")
    block_first, block_last = [], []
    boff = 0
    for i, length in enumerate(lens):
        s0 = int(cu[i])
        tok_base[s0 : s0 + length] = s0
        blk_base[s0 : s0 + length] = boff
        pos[s0 : s0 + length] = torch.arange(length, device="cuda")
        boff += -(-length // BLK)
        block_first += list(range(s0, s0 + length, BLK))
        block_last += [min(b + BLK, s0 + length) - 1 for b in range(s0, s0 + length, BLK)]
    block_first = torch.tensor(block_first, dtype=torch.int32, device="cuda")
    block_last = torch.tensor(block_last, dtype=torch.int32, device="cuda")

    sel = torch.zeros(T, boff, dtype=torch.uint8, device="cuda")
    idx = torch.full((T, 64), -1, dtype=torch.int32, device="cuda")
    for t in range(T):
        p = int(pos[t])
        cand = torch.arange((p + 1) // BLK, device="cuda")
        chosen = cand[torch.randperm(cand.numel(), device="cuda", generator=g)[:7]] if cand.numel() else cand
        blocks = torch.cat([chosen, torch.tensor([p // BLK], device="cuda")]).unique()
        sel[t, int(blk_base[t]) + blocks] = 1
        toks = (blocks.unsqueeze(1) * BLK + torch.arange(BLK, device="cuda")).reshape(-1) + int(tok_base[t])
        toks = toks[(toks >= int(tok_base[t])) & (toks <= t)]
        idx[t, : toks.numel()] = toks.to(torch.int32)

    hi = torch.arange(T, dtype=torch.int32, device="cuda")
    qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
    out = qsa_block_sparse_attention(
        qq, kk, vv, sel, tok_base, hi, blk_base, tok_base, block_first, block_last, SCALE, BLK
    )
    ref = _reference(q, k, v, idx, SCALE)
    assert _rel(out, ref)[1] < 1e-2
    # the second sequence is the one a global grid would corrupt
    assert _rel(out[lens[0] :], ref[lens[0] :])[1] < 1e-2

    gout = torch.randn(q.shape, device="cuda", dtype=torch.bfloat16, generator=g)
    out.backward(gout)
    ref_grads = [t.clone().requires_grad_(True) for t in (q, k, v)]
    qsa_sparse_attention(*ref_grads, idx, SCALE).backward(gout)
    for got, want, name in zip((qq, kk, vv), ref_grads, "qkv", strict=True):
        assert _rel(got.grad, want.grad)[1] < 1e-2, name


def test_tile_index_pair_is_consistent_both_ways():
    # [FUNCTION DOC] For a 256-token case, converts the selection to a block bitmap, builds the
    # forward (key list per query tile) and backward (query list per key tile) tile index pair
    # with tile sizes 64 and 32, and asserts the set of (query tile, key tile) pairs from the
    # forward lists equals the set from the backward lists.
    q, k, v, idx = _single_sequence_case(T=256, budget=32)
    sel = selection_to_block_bitmap(idx, 256, BLK)
    block_first = torch.arange(sel.shape[1], dtype=torch.int32, device="cuda") * BLK
    klist, kcnt, qlist, qcnt = build_tile_index_pair(sel, block_first, block_first + BLK - 1, 64, 32)

    forward = {(qt, int(klist[qt, i])) for qt in range(kcnt.numel()) for i in range(int(kcnt[qt]))}
    backward = {(int(qlist[kt, i]), kt) for kt in range(qcnt.numel()) for i in range(int(qcnt[kt]))}
    assert forward == backward
