# [MODULE DOC] Block-granular sparse attention with Triton tensor-core (tile matrix product)
# kernels, forward and backward, for packed token sequences (one flat axis of T tokens, grouped-query
# heads). Each query attends to the keys of the blocks flagged in its row of a block bitmap
# (blocks are runs of block_size consecutive tokens), restricted to the inclusive token range
# [lo, hi] of that query. Because the bitmap and range together define membership, tokens of a
# flagged block outside [lo, hi] are excluded but tokens inside [lo, hi] of a flagged block are
# always included.
# Pipeline: a bitmap builder from token indices; a tile-visit planner that converts the bitmap into,
# for each query tile, the list of key tiles worth visiting (and the transposed lists for the
# key-gradient kernel); an online-softmax forward kernel (float32 output, log-sum-exp per head and
# query); a query-gradient kernel; a key/value-gradient kernel organized per key tile and kv head so
# that no atomic additions are needed (the query heads of a group are summed inside the program);
# an autograd function wiring them; and two public functions: one taking the bitmap explicitly and
# one deriving it from token indices.
# Block ids for packed sequences: the block of a key for a query is the query's base block id plus
# the integer quotient of (key position minus the query's base token) by the block size, where the
# base block id and base token are those of the first block and first token of the query's own sequence.
# KNOWN ISSUE: the text below refers to a file named qsa_sparse_attn.py, which does not exist in this
# package (the sibling module is sparse_attention.py).
"""Tensor-core QSA sparse attention for training: forward + backward.

Same semantics as ``qsa_sparse_attn.py`` -- each query attends exactly the tokens in
its selection row -- but it walks key tiles with ``tl.dot`` and masks each (query, key)
pair to the query's selection, skipping tiles no query in the tile selected. The result
is exact: block-granular membership is a superset of the row, and ``lo``/``hi`` (the
inclusive key range, sequence start .. query position) removes the extra tokens.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _qsa_bs_fwd_kernel(
    Q,
    K,
    V,
    SEL,
    LO,
    HI,
    BLKBASE,
    TOKBASE,
    KLIST,
    KCNT,
    OUT,
    LSE,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_st,
    stride_kl,
    stride_ot,
    stride_oh,
    T,
    NB,
    scale,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BLK: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton forward kernel; grid of query tiles (BQ queries) by query heads;
    # the kv head is the query head divided by GROUP. Inputs: queries, keys, values (tokens by heads
    # by D, with given strides), SEL block bitmap (queries by NB, non-zero = block selected), per
    # query inclusive key range LO and HI, per query block base and token base, the tile's list of key
    # tiles KLIST with count KCNT, and scale. D must be a power of two. Outputs: OUT float32
    # (tokens by heads by D) and LSE float32 (heads by tokens).
    # Algorithm: for every key tile in the tile's list, compute each (query, key) pair's block id,
    # read the bitmap (out-of-range blocks read as unselected), keep pairs that are selected, inside
    # [lo, hi] and inside the token count; scores = q times k transposed times scale (masked to minus
    # infinity); online softmax update with a safe maximum (zero while still minus infinity) and
    # rescaling; accumulate probabilities (cast to the value dtype) times values. Finally divide by the
    # sum (1 when the sum is zero, so queries without keys output zeros) and store; the log-sum-exp is
    # minus infinity for queries with no key, else maximum plus log of the sum.
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_head = pid_h // GROUP

    offs_q = pid_t * BQ + tl.arange(0, BQ)
    offs_d = tl.arange(0, D)
    q_mask = offs_q < T

    q = tl.load(
        Q + offs_q[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :],
        mask=q_mask[:, None],
        other=0.0,
    )
    lo = tl.load(LO + offs_q, mask=q_mask, other=0)
    hi = tl.load(HI + offs_q, mask=q_mask, other=-1)
    blk_base = tl.load(BLKBASE + offs_q, mask=q_mask, other=0)
    tok_base = tl.load(TOKBASE + offs_q, mask=q_mask, other=0)

    m_i = tl.full((BQ,), float("-inf"), tl.float32)
    l_i = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, D), tl.float32)

    # the tile list is the union over the tile's queries; per-query exactness comes from the mask
    n_tiles = tl.load(KCNT + pid_t)
    for i in range(0, n_tiles):
        kt = tl.load(KLIST + pid_t * stride_kl + i)
        offs_k = kt * BK + tl.arange(0, BK)
        k_in = offs_k < T

        # per-token block lookup: a packed sequence's blocks start at its own first token,
        # which need not be a multiple of BLK
        blk = blk_base[:, None] + (offs_k[None, :] - tok_base[:, None]) // BLK
        sel = tl.load(
            SEL + offs_q[:, None] * stride_st + blk,
            mask=q_mask[:, None] & k_in[None, :] & (blk >= 0) & (blk < NB),
            other=0,
        )
        ok = (sel != 0) & (offs_k[None, :] <= hi[:, None]) & (offs_k[None, :] >= lo[:, None]) & k_in[None, :]

        k_tile = tl.load(
            K + offs_k[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :],
            mask=k_in[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k_tile)) * scale
        s = tl.where(ok, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_use[:, None])
        p = tl.where(ok, p, 0.0)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_use))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_tile = tl.load(
            V + offs_k[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :],
            mask=k_in[:, None],
            other=0.0,
        )
        acc += tl.dot(p.to(v_tile.dtype), v_tile)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    tl.store(
        OUT + offs_q[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :],
        out,
        mask=q_mask[:, None],
    )
    lse = tl.where(m_i == float("-inf"), float("-inf"), m_i + tl.log(l_safe))
    tl.store(LSE + pid_h * T + offs_q, lse, mask=q_mask)


@triton.jit
def _qsa_bs_dq_kernel(
    Q,
    K,
    V,
    SEL,
    LO,
    HI,
    BLKBASE,
    TOKBASE,
    KLIST,
    KCNT,
    OUT,
    LSE,
    DO,
    DQ,
    DELTA,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_st,
    stride_kl,
    stride_ot,
    stride_oh,
    T,
    NB,
    scale,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BLK: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel for the query gradient; same grid, masks and tile lists as
    # the forward kernel. Extra inputs: output gradient DO (read with the output strides), saved
    # log-sum-exp LSE and DELTA (heads by tokens, delta = sum over D of output gradient times output).
    # Queries whose log-sum-exp is minus infinity (no keys) are treated as dead and contribute zero.
    # Per key tile: probabilities = exp(score - lse) masked like the forward; dp = dO times V
    # transposed; ds = p times (dp minus delta) times scale (cast to the key dtype); dq += ds times
    # K. Stores dq (float32, query strides).
    # KNOWN ISSUE: the OUT pointer parameter is never read in this kernel (only its strides, passed
    # as the output-gradient strides, are used), so passing the saved output is unnecessary.
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_head = pid_h // GROUP

    offs_q = pid_t * BQ + tl.arange(0, BQ)
    offs_d = tl.arange(0, D)
    q_mask = offs_q < T

    q = tl.load(Q + offs_q[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :], mask=q_mask[:, None], other=0.0)
    do = tl.load(
        DO + offs_q[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :], mask=q_mask[:, None], other=0.0
    )
    lse = tl.load(LSE + pid_h * T + offs_q, mask=q_mask, other=0.0)
    delta = tl.load(DELTA + pid_h * T + offs_q, mask=q_mask, other=0.0)
    lse_safe = tl.where(lse == float("-inf"), 0.0, lse)
    alive = lse != float("-inf")

    lo = tl.load(LO + offs_q, mask=q_mask, other=0)
    hi = tl.load(HI + offs_q, mask=q_mask, other=-1)
    blk_base = tl.load(BLKBASE + offs_q, mask=q_mask, other=0)
    tok_base = tl.load(TOKBASE + offs_q, mask=q_mask, other=0)

    dq = tl.zeros((BQ, D), tl.float32)
    n_tiles = tl.load(KCNT + pid_t)
    for i in range(0, n_tiles):
        kt = tl.load(KLIST + pid_t * stride_kl + i)
        offs_k = kt * BK + tl.arange(0, BK)
        k_in = offs_k < T

        # per-token block lookup, as in the forward kernel
        blk = blk_base[:, None] + (offs_k[None, :] - tok_base[:, None]) // BLK
        sel = tl.load(
            SEL + offs_q[:, None] * stride_st + blk,
            mask=q_mask[:, None] & k_in[None, :] & (blk >= 0) & (blk < NB),
            other=0,
        )
        ok = (
            (sel != 0)
            & (offs_k[None, :] <= hi[:, None])
            & (offs_k[None, :] >= lo[:, None])
            & k_in[None, :]
            & alive[:, None]
        )

        k_tile = tl.load(
            K + offs_k[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :], mask=k_in[:, None], other=0.0
        )
        v_tile = tl.load(
            V + offs_k[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :], mask=k_in[:, None], other=0.0
        )

        s = tl.dot(q, tl.trans(k_tile)) * scale
        p = tl.exp(s - lse_safe[:, None])
        p = tl.where(ok, p, 0.0)

        dp = tl.dot(do, tl.trans(v_tile))
        ds = (p * (dp - delta[:, None]) * scale).to(k_tile.dtype)

        dq += tl.dot(ds, k_tile)

    tl.store(DQ + offs_q[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :], dq, mask=q_mask[:, None])


@triton.jit
def _qsa_bs_dkdv_kernel(
    Q,
    K,
    V,
    SEL,
    LO,
    HI,
    BLKBASE,
    TOKBASE,
    QLIST,
    QCNT,
    LSE,
    DO,
    DELTA,
    DK,
    DV,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_st,
    stride_ql,
    stride_ot,
    stride_oh,
    T,
    NB,
    scale,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BLK: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel for key and value gradients; grid of key tiles (BK keys) by
    # kv heads. Holds the key and value tiles and float32 accumulators, then iterates over the tile's
    # list of query tiles QLIST (count QCNT). For each query tile it builds the block-membership mask
    # (selected block, inside [lo, hi], valid key and query), then for each of the GROUP query heads
    # sharing the kv head loads queries, output gradients, log-sum-exp and delta (queries with a
    # minus-infinity log-sum-exp are masked out), recomputes probabilities, ds = p times (dp minus
    # delta) times scale, and accumulates dk += ds transposed times q and dv += p transposed (cast to
    # the output-gradient dtype) times dO. Stores float32 dk and dv for the tile's keys (using the key
    # and value strides).
    """dK/dV keyed on the key tile, so nothing needs atomics.

    Launched per KV head: the ``GROUP`` query heads sharing a KV head are summed here,
    since separate programs would overwrite each other's dK/dV.
    """
    pid_k = tl.program_id(0)
    kv_head = tl.program_id(1)

    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    k_in = offs_k < T

    k_tile = tl.load(
        K + offs_k[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :], mask=k_in[:, None], other=0.0
    )
    v_tile = tl.load(
        V + offs_k[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :], mask=k_in[:, None], other=0.0
    )
    dk = tl.zeros((BK, D), tl.float32)
    dv = tl.zeros((BK, D), tl.float32)

    n_q = tl.load(QCNT + pid_k)
    for i in range(0, n_q):
        qt = tl.load(QLIST + pid_k * stride_ql + i)
        offs_q = qt * BQ + tl.arange(0, BQ)
        q_mask = offs_q < T
        lo = tl.load(LO + offs_q, mask=q_mask, other=0)
        hi = tl.load(HI + offs_q, mask=q_mask, other=-1)
        blk_base = tl.load(BLKBASE + offs_q, mask=q_mask, other=0)
        tok_base = tl.load(TOKBASE + offs_q, mask=q_mask, other=0)

        blk = blk_base[:, None] + (offs_k[None, :] - tok_base[:, None]) // BLK
        sel = tl.load(
            SEL + offs_q[:, None] * stride_st + blk,
            mask=q_mask[:, None] & k_in[None, :] & (blk >= 0) & (blk < NB),
            other=0,
        )
        ok = (
            (sel != 0)
            & (offs_k[None, :] <= hi[:, None])
            & (offs_k[None, :] >= lo[:, None])
            & k_in[None, :]
            & q_mask[:, None]
        )

        for gh in range(0, GROUP):
            qh = kv_head * GROUP + gh
            q = tl.load(
                Q + offs_q[:, None] * stride_qt + qh * stride_qh + offs_d[None, :],
                mask=q_mask[:, None],
                other=0.0,
            )
            do = tl.load(
                DO + offs_q[:, None] * stride_ot + qh * stride_oh + offs_d[None, :],
                mask=q_mask[:, None],
                other=0.0,
            )
            lse = tl.load(LSE + qh * T + offs_q, mask=q_mask, other=0.0)
            delta = tl.load(DELTA + qh * T + offs_q, mask=q_mask, other=0.0)
            okh = ok & (lse[:, None] != float("-inf"))

            lse_safe = tl.where(lse == float("-inf"), 0.0, lse)
            sc = tl.dot(q, tl.trans(k_tile)) * scale
            p = tl.exp(sc - lse_safe[:, None])
            p = tl.where(okh, p, 0.0)

            dp = tl.dot(do, tl.trans(v_tile))
            ds = (p * (dp - delta[:, None]) * scale).to(k_tile.dtype)

            dk += tl.dot(tl.trans(ds), q)
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)

    tl.store(DK + offs_k[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :], dk, mask=k_in[:, None])
    tl.store(DV + offs_k[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :], dv, mask=k_in[:, None])


def selection_to_block_bitmap(indices: Tensor, num_tokens: int, block_size: int) -> Tensor:
    # [FUNCTION DOC] Converts per-query selected token indices (queries by K, minus one = padding)
    # into an unsigned 8-bit bitmap of queries by ceil(num_tokens / block_size): entry set to 1 where
    # any valid index of the row falls into that block (index integer-divided by block_size). The
    # block count comes from num_tokens, the key count, which callers take equal to the query count.
    # Allocates the bitmap on the indices' device; indices must be below num_tokens.
    """``[T, K]`` token indices (``-1`` pad) -> ``[T, ceil(T / block_size)]`` uint8 flags.

    A block is flagged when any of its tokens appears in the row. Tokens that the caller
    clamped away inside an otherwise selected block are re-excluded by the ``lo``/``hi``
    range test in the kernel, so this stays exact while being ``block_size``x smaller.
    """
    num_blocks = -(-num_tokens // block_size)
    flags = torch.zeros(indices.shape[0], num_blocks, dtype=torch.uint8, device=indices.device)
    valid = indices >= 0
    rows = torch.arange(indices.shape[0], device=indices.device).unsqueeze(1).expand_as(indices)
    blocks = torch.where(valid, indices // block_size, torch.zeros_like(indices))
    flags[rows[valid], blocks[valid].long()] = 1
    return flags


def _tile_mask(sel: Tensor, block_first: Tensor, block_last: Tensor, bq: int, bk: int) -> Tensor:
    # [FUNCTION DOC] HELPER: Builds the boolean tile visit matrix (query tiles by key tiles). The
    # bitmap is zero-padded in the query axis to a multiple of bq, reduced with a maximum over each
    # query tile, then each block's flag is added (index-add) to the key tile containing the block's
    # first token and to the one containing its last token; entries above zero are true. Assumes
    # every block spans at most two key tiles (block size not above bk; not asserted) and that the
    # number of key tiles equals that of query tiles computed from the same length (key count taken
    # as the query count).
    """``[NQT, NKT]`` bool: some query in query tile ``i`` selected a block with a token in key tile ``j``.

    Key tiles are absolute token ranges ``[j * bk, (j + 1) * bk)``, as the kernels walk them.
    ``block_first`` / ``block_last`` ``[NB]`` are each block's inclusive token span; packed
    block ids drift from ``token // block_size`` once a sequence length is off the grid.
    """
    T, nb = sel.shape
    nqt = -(-T // bq)
    nkt = -(-T // bk)
    pad_q = nqt * bq - T
    if pad_q:
        sel = torch.nn.functional.pad(sel, (0, 0, 0, pad_q))
    picked = sel.view(nqt, bq, nb).amax(dim=1).to(torch.int32)
    tile = torch.zeros(nqt, nkt, dtype=torch.int32, device=sel.device)
    # block_size <= bk, so a block touches at most the tiles of its first and last token
    tile.index_add_(1, block_first.long() // bk, picked)
    tile.index_add_(1, block_last.long() // bk, picked)
    return tile > 0


def _compact_tile_mask(mat: Tensor) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC] HELPER: Turns a boolean matrix into per-row lists of true column indices. Counts
    # trues per row (32-bit), takes the largest count (at least 1; this forces a device-to-host
    # synchronization), and uses a stable sort of the negated matrix so true columns come first in
    # ascending order; returns the first max-count columns of the sort order as 32-bit ids (entries
    # beyond a row's count are arbitrary false columns) plus the counts.
    """``[R, C]`` bool -> (ascending column ids per row ``[R, maxc]`` int32, counts ``[R]`` int32)."""
    cnt = mat.sum(dim=1).to(torch.int32)
    maxc = max(int(cnt.max().item()), 1)
    order = torch.argsort((~mat).to(torch.int8), dim=1, stable=True)
    return order[:, :maxc].contiguous().to(torch.int32), cnt


def build_tile_index(sel: Tensor, block_first: Tensor, block_last: Tensor, bq: int, bk: int) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC] Computes the tile visit matrix from the block bitmap and compacts it into
    # (klist, kcnt): for each query tile the ascending key tile ids and their count (see the two
    # helpers it composes).
    """``[T, NB]`` block flags -> (``klist`` [NQT, maxc] int32, ``kcnt`` [NQT] int32).

    ``klist[i]`` lists, ascending, the key tiles that at least one query in query-tile
    ``i`` selected. Per-query exactness still comes from the in-kernel mask; this only
    decides which tiles are worth visiting.
    """
    return _compact_tile_mask(_tile_mask(sel, block_first, block_last, bq, bk))


def build_tile_index_pair(sel: Tensor, block_first: Tensor, block_last: Tensor, bq: int, bk: int):
    # [FUNCTION DOC] Builds the tile visit matrix once and compacts it twice: per query tile (klist,
    # kcnt) and, from the transposed matrix, per key tile (qlist, qcnt). Returns the four tensors in
    # that order.
    """Both directions of the tile map: (klist, kcnt) per query tile, (qlist, qcnt) per key tile.

    The transposed half is what lets dK/dV be keyed on the key tile and so avoid atomics.
    """
    tile = _tile_mask(sel, block_first, block_last, bq, bk)
    klist, kcnt = _compact_tile_mask(tile)
    qlist, qcnt = _compact_tile_mask(tile.t().contiguous())
    return klist, kcnt, qlist, qcnt


class _QSABlockSparseAttn(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function around the three kernels. Forward saves contiguous q, k,
    # v, the bitmap, range and block-placement tensors, and the float32 output and log-sum-exp for
    # backward. Gradients flow to q, k, v only.
    @staticmethod
    def forward(ctx, q, k, v, sel, lo, hi, blk_base, tok_base, block_first, block_last, scale, block_size):
        # [FUNCTION DOC] Computes attention with 64-query by 64-key tiles. Asserts the query head count
        # is a multiple of the kv head count. Plans the tile lists, allocates a float32 output and
        # float32 log-sum-exp, launches the forward kernel (8 warps, 2 stages) and returns the output
        # cast to the query dtype. The key count unpacked from k is not used; the kernel bounds keys
        # by the query count.
        # KNOWN ISSUE: because keys are bounded by the number of query tokens, key/value tensors longer
        # or shorter than the queries are not supported (self-attention only).
        T, Hq, D = q.shape
        S, Hkv, _ = k.shape
        assert Hq % Hkv == 0
        group = Hq // Hkv
        qc, kc, vc = q.contiguous(), k.contiguous(), v.contiguous()
        selc = sel.contiguous()
        BQ, BK = 64, 64
        klist, kcnt = build_tile_index(selc, block_first, block_last, BQ, BK)
        o = torch.empty(T, Hq, D, device=q.device, dtype=torch.float32)
        lse = torch.empty(Hq, T, device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(T, BQ), Hq)
        _qsa_bs_fwd_kernel[grid](
            qc,
            kc,
            vc,
            selc,
            lo,
            hi,
            blk_base,
            tok_base,
            klist,
            kcnt,
            o,
            lse,
            qc.stride(0),
            qc.stride(1),
            kc.stride(0),
            kc.stride(1),
            vc.stride(0),
            vc.stride(1),
            selc.stride(0),
            klist.stride(0),
            o.stride(0),
            o.stride(1),
            T,
            selc.shape[1],
            scale,
            GROUP=group,
            D=D,
            BQ=BQ,
            BK=BK,
            BLK=block_size,
            num_warps=8,
            num_stages=2,
        )
        ctx.save_for_backward(qc, kc, vc, selc, lo, hi, blk_base, tok_base, block_first, block_last, o, lse)
        ctx.scale = scale
        ctx.group = group
        ctx.block_size = block_size
        return o.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # [FUNCTION DOC] Computes gradients for q, k, v (cast to their dtypes) and none for the nine
        # other forward inputs. The output gradient is made contiguous in the query dtype; delta
        # (heads by tokens, float32) is computed once in torch as the sum over channels of output
        # gradient times saved output. Tile lists are re-planned with 64-query by 32-key tiles, both
        # directions. Runs the query-gradient kernel (grid of query tiles by query heads) and then the
        # key/value kernel (grid of key tiles by kv heads), both with 8 warps and 1 stage, writing
        # into float32 buffers (key and value buffers start at zero).
        qc, kc, vc, selc, lo, hi, blk_base, tok_base, block_first, block_last, o, lse = ctx.saved_tensors
        T, Hq, D = qc.shape
        do = grad_out.contiguous().to(qc.dtype)
        # delta once, in torch: both backward kernels need it and neither should redo it
        delta = (do.float() * o.float()).sum(-1).transpose(0, 1).contiguous()
        dq = torch.empty(T, Hq, D, device=qc.device, dtype=torch.float32)
        dk = torch.zeros(kc.shape, device=kc.device, dtype=torch.float32)
        dv = torch.zeros(vc.shape, device=vc.device, dtype=torch.float32)

        BQ, BK = 64, 32
        klist, kcnt, qlist, qcnt = build_tile_index_pair(selc, block_first, block_last, BQ, BK)
        common = (
            qc.stride(0),
            qc.stride(1),
            kc.stride(0),
            kc.stride(1),
            vc.stride(0),
            vc.stride(1),
            selc.stride(0),
        )
        _qsa_bs_dq_kernel[(triton.cdiv(T, BQ), Hq)](
            qc,
            kc,
            vc,
            selc,
            lo,
            hi,
            blk_base,
            tok_base,
            klist,
            kcnt,
            o,
            lse,
            do,
            dq,
            delta,
            *common,
            klist.stride(0),
            do.stride(0),
            do.stride(1),
            T,
            selc.shape[1],
            ctx.scale,
            GROUP=ctx.group,
            D=D,
            BQ=BQ,
            BK=BK,
            BLK=ctx.block_size,
            num_warps=8,
            num_stages=1,
        )
        _qsa_bs_dkdv_kernel[(triton.cdiv(T, BK), kc.shape[1])](
            qc,
            kc,
            vc,
            selc,
            lo,
            hi,
            blk_base,
            tok_base,
            qlist,
            qcnt,
            lse,
            do,
            delta,
            dk,
            dv,
            *common,
            qlist.stride(0),
            do.stride(0),
            do.stride(1),
            T,
            selc.shape[1],
            ctx.scale,
            GROUP=ctx.group,
            D=D,
            BQ=BQ,
            BK=BK,
            BLK=ctx.block_size,
            num_warps=8,
            num_stages=1,
        )
        return dq.to(qc.dtype), dk.to(kc.dtype), dv.to(vc.dtype), None, None, None, None, None, None, None, None, None


def qsa_block_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sel_blocks: Tensor,
    lo: Tensor,
    hi: Tensor,
    blk_base: Tensor,
    tok_base: Tensor,
    block_first: Tensor,
    block_last: Tensor,
    scale: float,
    block_size: int = 4,
) -> Tensor:
    # [FUNCTION DOC] Public entry point with an explicit block bitmap. Shapes: q tokens by query heads
    # by D (D a power of two), k and v tokens by kv heads by D, sel_blocks queries by block count
    # (uint8), lo/hi and blk_base/tok_base 32-bit per-query tensors, block_first/block_last per-block
    # inclusive token spans. scale multiplies the scores; block_size (default 4) is the number of
    # tokens per block and must not exceed the 32-token key tile used in backward. Returns the output
    # in the query dtype; differentiable in q, k, v.
    """``q`` [T, Hq, D], ``k``/``v`` [S, Hkv, D], ``sel_blocks`` [T, NB] uint8.

    ``lo``/``hi`` are the inclusive key range per query; ``blk_base``/``tok_base`` place
    the query's sequence in the packed block grid (both zero for a single sequence);
    ``block_first``/``block_last`` [NB] are each block's inclusive token span.
    """
    return _QSABlockSparseAttn.apply(
        q, k, v, sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last, scale, block_size
    )


def qsa_sparse_attention_from_indices(
    q: Tensor, k: Tensor, v: Tensor, indices: Tensor, scale: float, block_size: int = 4
) -> Tensor:
    # [FUNCTION DOC] Public entry point taking per-query selected token indices (queries by K, minus
    # one = padding). Derives the block bitmap, then per-query lo (smallest valid index; the largest
    # 32-bit integer when none) and hi (largest valid index; minus one when none, so such queries
    # output zeros), single-sequence placement (zero block and token bases), block spans from the
    # block size (last block clipped to the token count), and calls the bitmap-based function.
    # KNOWN ISSUE: membership is block-granular within [lo, hi], so the result matches a token-exact
    # gather only if every row's selection consists of whole blocks (apart from the clipped ends);
    # a row selecting scattered single tokens also attends the unselected tokens of those blocks
    # that lie between its smallest and largest index.
    """Drop-in for the gather kernel: derives the bitmap and range from ``indices``."""
    T = q.shape[0]
    sel = selection_to_block_bitmap(indices, T, block_size)
    valid = indices >= 0
    big = torch.iinfo(torch.int32).max
    lo = torch.where(valid, indices, torch.full_like(indices, big)).min(dim=1).values.to(torch.int32)
    hi = torch.where(valid, indices, torch.full_like(indices, -1)).max(dim=1).values.to(torch.int32)
    zeros = torch.zeros(T, dtype=torch.int32, device=q.device)
    block_first = torch.arange(sel.shape[1], dtype=torch.int32, device=q.device) * block_size
    block_last = (block_first + block_size).clamp_max(T) - 1
    return qsa_block_sparse_attention(q, k, v, sel, lo, hi, zeros, zeros, block_first, block_last, scale, block_size)
