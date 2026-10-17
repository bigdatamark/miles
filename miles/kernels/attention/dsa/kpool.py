# [MODULE DOC] Key-pooled variant of the lightning-indexer selection. Consecutive groups of kpool
# tokens inside each packed sequence are merged into one pooled key per group (a per-channel softmax
# over the group, using a gate score plus a learned per-slot bias, weights the keys); the indexer
# then scores pooled keys, the best pools are chosen per query, and the chosen pools are expanded
# back into token indices (each pool gives kpool consecutive token positions). The last, incomplete
# group of a query's own prefix ("causal tail") is appended as individual tokens so the query still
# sees its most recent tokens. Short prefixes (up to top-k tokens) bypass pooling and select all
# earlier tokens directly.
# Contents: pooled-key construction (boundaries helper, Triton kernel, host wrapper), two Triton
# kernels (expand pool selection into token indices with optional tail; append tail to an existing
# token index matrix), and host wrappers: a factory of a top-k callable for the pooled scores, a
# tail-append-and-pad function, and a fused select-expand-tail function.
# Constants: SPARSE_MLA_BLOCK (64) is the multiple the output width is padded to; the kernels use
# a column block of 256. Token index tensors are 32-bit integers with minus one as padding.
"""Key-pooled lightning-indexer selection: pooled keys, pool top-k expanded to token indices, causal tail."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


SPARSE_MLA_BLOCK = 64
_SELECT_BLOCK = 256


def pool_boundaries(cu_seqlens: torch.Tensor, kpool: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Converts cumulative token boundaries of packed sequences into
    # cumulative pool boundaries: each sequence contributes floor(length / kpool) pools (a trailing
    # partial group yields no pool), starting from a leading zero. Same shape and dtype as the input.
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    pool_counts = torch.div(seq_lens, kpool, rounding_mode="floor")
    pool_cu_seqlens = torch.zeros_like(cu_seqlens)
    pool_cu_seqlens[1:] = torch.cumsum(pool_counts, dim=0)
    return pool_cu_seqlens


@triton.jit
def _pooled_keys_kernel(
    index_k_ptr,
    gate_ptr,
    ape_ptr,
    cu_seqlens_ptr,
    pool_cu_seqlens_ptr,
    num_seqs,
    out_ptr,
    D: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel; one program per output pooled row (the program index is
    # the global pool number). Rows at or beyond the real pool count (last entry of the pool
    # boundaries) are written as zeros. Otherwise it finds the owning sequence by a binary search
    # for the last sequence whose first pool number is not greater than the program index, derives
    # the first token of the group as sequence start plus local pool number times KPOOL, and for
    # each channel computes in float32 a softmax over the KPOOL slots of (gate score + bias of that
    # slot): first the maximum, then the sum of exponentials, then the weighted sum of the keys using
    # round-to-nearest division and multiplication (separately rounded, no fused multiply-add, with
    # the device exponential) to be bit-reproducible. The pooled value is stored cast to the output
    # dtype. Gate and keys are laid out tokens by D; bias is slots by D in float32.
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    dmask = offs_d < D
    out_offs = pid.to(tl.int64) * D + offs_d
    num_pools = tl.load(pool_cu_seqlens_ptr + num_seqs)
    if pid >= num_pools:
        zeros = tl.zeros((BLOCK_D,), dtype=tl.float32)
        tl.store(out_ptr + out_offs, zeros.to(out_ptr.dtype.element_ty), mask=dmask)
        return
    lo = 0
    hi = num_seqs
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tl.load(pool_cu_seqlens_ptr + mid) <= pid:
            lo = mid
        else:
            hi = mid - 1
    seq_start = tl.load(cu_seqlens_ptr + lo)
    pool_base = tl.load(pool_cu_seqlens_ptr + lo)
    start = (seq_start + (pid - pool_base) * KPOOL).to(tl.int64)
    peak = tl.full((BLOCK_D,), float("-inf"), dtype=tl.float32)
    for r in tl.static_range(KPOOL):
        gate = tl.load(gate_ptr + (start + r) * D + offs_d, mask=dmask, other=0.0).to(tl.float32)
        ape = tl.load(ape_ptr + r * D + offs_d, mask=dmask, other=0.0)
        peak = tl.maximum(peak, gate + ape)
    denom = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for r in tl.static_range(KPOOL):
        gate = tl.load(gate_ptr + (start + r) * D + offs_d, mask=dmask, other=0.0).to(tl.float32)
        ape = tl.load(ape_ptr + r * D + offs_d, mask=dmask, other=0.0)
        denom = denom + libdevice.exp(gate + ape - peak)
    pooled = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for r in tl.static_range(KPOOL):
        gate = tl.load(gate_ptr + (start + r) * D + offs_d, mask=dmask, other=0.0).to(tl.float32)
        ape = tl.load(ape_ptr + r * D + offs_d, mask=dmask, other=0.0)
        key = tl.load(index_k_ptr + (start + r) * D + offs_d, mask=dmask, other=0.0).to(tl.float32)
        weight = libdevice.div_rn(libdevice.exp(gate + ape - peak), denom)
        pooled = pooled + libdevice.mul_rn(weight, key)
    tl.store(out_ptr + out_offs, pooled.to(out_ptr.dtype.element_ty), mask=dmask)


def build_pooled_keys(
    index_k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kpool: int,
) -> torch.Tensor:
    # [FUNCTION DOC] Builds the pooled key matrix. Inputs: index_k (tokens by head dim), gate_score
    # (tokens by head dim, per-channel gate logits), ape (kpool by head dim bias, converted to
    # float32), cu_seqlens (cumulative packed sequence boundaries) and the pool size. The output has
    # total_tokens // kpool rows (an upper bound on the true pool count, which is the sum of
    # per-sequence floors), dtype of index_k; rows past the real pool count are zero. Returns an
    # empty (0 by head dim) tensor when no row is possible. Launches one program per row.
    total_tokens, head_dim = index_k.shape
    max_pools = total_tokens // kpool
    if max_pools == 0:
        return index_k.new_zeros((0, head_dim))
    pool_cu_seqlens = pool_boundaries(cu_seqlens, kpool)
    pooled = torch.empty((max_pools, head_dim), dtype=index_k.dtype, device=index_k.device)
    _pooled_keys_kernel[(max_pools,)](
        index_k.contiguous(),
        gate_score.contiguous(),
        ape.float().contiguous(),
        cu_seqlens.contiguous(),
        pool_cu_seqlens.contiguous(),
        cu_seqlens.shape[0] - 1,
        pooled,
        D=head_dim,
        KPOOL=kpool,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return pooled


@triton.jit
def _expand_topk_kernel(
    pools_ptr,
    scores_ptr,
    seq_base_ptr,
    local_pos_ptr,
    pool_base_ptr,
    out_ptr,
    topk,
    group_topk,
    out_width,
    KPOOL: tl.constexpr,
    WITH_TAIL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel producing, for one query token (grid dimension 0) and one
    # block of BLOCK output columns (dimension 1), the selected token indices. Per-token inputs:
    # seq_base (global index of the first token of the query's sequence), local_pos (query position
    # inside its sequence), pool_base (global number of the sequence's first pool). Pooled
    # selection: pools and scores are tokens by group_topk (chosen pool numbers and their scores).
    # Case 1, the query has at most topk tokens up to and including itself: columns below topk and
    # not after the query position hold seq_base + column (every earlier token), others minus one.
    # Case 2: each group (BLOCK / KPOOL groups per program, group number = column / KPOOL) is
    # expanded to its KPOOL consecutive token indices: seq_base + (pool - pool_base) * KPOOL + slot;
    # groups beyond group_topk, or whose score is not finite (not-a-number or either infinity), give
    # minus one. If WITH_TAIL, columns from topk upward hold the tokens of the incomplete last group
    # (starting at seq_base + ((local_pos + 1) // KPOOL) * KPOOL, count (local_pos + 1) mod KPOOL),
    # overriding the pooled values there; remaining columns stay minus one. Writes 32-bit integers
    # for columns below out_width only. Requires BLOCK divisible by KPOOL.
    t = tl.program_id(0).to(tl.int64)
    col0 = tl.program_id(1) * BLOCK
    cols = col0 + tl.arange(0, BLOCK)
    in_out = cols < out_width
    base = tl.load(seq_base_ptr + t)
    lpos = tl.load(local_pos_ptr + t)
    if (lpos + 1) <= topk:
        val = tl.where((cols < topk) & (cols <= lpos), base + cols, -1)
    else:
        pbase = tl.load(pool_base_ptr + t)
        groups = col0 // KPOOL + tl.arange(0, BLOCK // KPOOL)
        gmask = groups < group_topk
        pool = tl.load(pools_ptr + t * group_topk + groups, mask=gmask, other=0).to(tl.int32)
        score = tl.load(scores_ptr + t * group_topk + groups, mask=gmask, other=float("-inf"))
        finite = (score == score) & (score != float("inf")) & (score != float("-inf"))
        slots = tl.arange(0, KPOOL)
        candidates = (base + (pool - pbase) * KPOOL)[:, None] + slots[None, :]
        val = tl.reshape(tl.where((gmask & finite)[:, None], candidates, -1), (BLOCK,))
        if WITH_TAIL:
            tail_start = base + ((lpos + 1) // KPOOL) * KPOOL
            tail_slot = cols - topk
            tail_ok = (tail_slot >= 0) & (tail_slot < (lpos + 1) % KPOOL)
            val = tl.where(tail_ok, tail_start + tail_slot, val)
    tl.store(out_ptr + t * out_width + cols, val.to(tl.int32), mask=in_out)


@triton.jit
def _append_tail_kernel(
    tokens_ptr,
    seq_base_ptr,
    local_pos_ptr,
    shortcut_ptr,
    out_ptr,
    width,
    out_width,
    KPOOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel; per query token (dimension 0) and column block
    # (dimension 1) it copies the existing token index row (columns below width) into the wider
    # output row and, for columns from width upward, writes the tail tokens of the incomplete last
    # group (first tail token seq_base + ((local_pos + 1) // KPOOL) * KPOOL, count (local_pos + 1)
    # mod KPOOL) unless the per-token shortcut flag is non-zero (query handled by the direct
    # path), filling everything else with minus one. Output is 32-bit integers, masked at out_width.
    t = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_out = cols < out_width
    copy = cols < width
    tokens = tl.load(tokens_ptr + t * width + cols, mask=copy, other=-1).to(tl.int32)
    base = tl.load(seq_base_ptr + t)
    lpos = tl.load(local_pos_ptr + t)
    shortcut = tl.load(shortcut_ptr + t)
    tail_start = base + ((lpos + 1) // KPOOL) * KPOOL
    tail_slot = cols - width
    tail_ok = (tail_slot >= 0) & (tail_slot < (lpos + 1) % KPOOL) & (shortcut == 0)
    val = tl.where(copy, tokens, tl.where(tail_ok, tail_start + tail_slot, -1))
    tl.store(out_ptr + t * out_width + cols, val, mask=in_out)


def _pool_topk(pool_logits: torch.Tensor, topk: int, kpool: int):
    # [FUNCTION DOC] HELPER: Selects the best pools per query from a scores matrix of queries by
    # pools. The number of pools kept is min(topk // kpool, number of pools); asserts that it is
    # positive and that kpool is a power of two dividing the kernel block (256). Returns
    # (scores float32, pool numbers, group count) from a top-k over the last dimension (descending
    # score order); no padding or filtering of minus-infinity scores happens here.
    _, num_pools = pool_logits.shape
    group_topk = min(topk // kpool, num_pools)
    assert group_topk > 0, (topk, kpool, num_pools)
    assert kpool & (kpool - 1) == 0 and _SELECT_BLOCK % kpool == 0, kpool
    scores, pools = torch.topk(pool_logits.float(), group_topk, dim=-1)
    return scores, pools, group_topk


def pool_topk_to_token_fn(seq_token_base, pool_base, local_positions, kpool):
    # [FUNCTION DOC] Factory returning a top-k callable for the indexer: the callable takes pool
    # scores (queries by pools) and a token budget, picks the best pools, and returns a 32-bit
    # token index matrix of queries by topk (each kept pool expands to kpool tokens, minus one where
    # invalid, no causal tail). seq_token_base, pool_base and local_positions are per-query
    # tensors (see the expand kernel) captured by the callable; they are passed to the kernel as is,
    # so they must be contiguous device tensors of a compatible integer type.
    def topk_fn(pool_logits: torch.Tensor, topk: int) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: The callable built by the factory; see the factory description.
        # Launches a grid of queries by ceil(topk / 256) column blocks. Raises assertion errors from
        # the pool selection when topk is smaller than kpool or kpool is not a suitable power of two.
        num_tokens = pool_logits.shape[0]
        scores, pools, group_topk = _pool_topk(pool_logits, topk, kpool)
        tokens = torch.empty((num_tokens, topk), dtype=torch.int32, device=pool_logits.device)
        grid = (num_tokens, triton.cdiv(topk, _SELECT_BLOCK))
        _expand_topk_kernel[grid](
            pools,
            scores,
            seq_token_base,
            local_positions,
            pool_base,
            tokens,
            topk,
            group_topk,
            topk,
            KPOOL=kpool,
            WITH_TAIL=False,
            BLOCK=_SELECT_BLOCK,
        )
        return tokens

    return topk_fn


def append_tail_and_pad(
    tokens: torch.Tensor,
    seq_token_base: torch.Tensor,
    local_positions: torch.Tensor,
    shortcut: torch.Tensor,
    kpool: int,
    pad_multiple: int = SPARSE_MLA_BLOCK,
) -> torch.Tensor:
    # [FUNCTION DOC] Widens a token index matrix (queries by width, 32-bit) with the causal tail:
    # the output width is width + kpool - 1 rounded up to a multiple of pad_multiple (default 64);
    # existing columns are copied, then up to kpool - 1 tail tokens per query follow (none for
    # queries whose shortcut flag is non-zero), the rest minus one. Index-like inputs are converted
    # to 32-bit contiguous tensors first. Returns a new 32-bit matrix.
    num_tokens, width = tokens.shape
    out_width = (width + kpool - 1 + pad_multiple - 1) // pad_multiple * pad_multiple
    out = torch.empty((num_tokens, out_width), dtype=torch.int32, device=tokens.device)
    grid = (num_tokens, triton.cdiv(out_width, _SELECT_BLOCK))
    _append_tail_kernel[grid](
        tokens.contiguous(),
        seq_token_base.to(torch.int32).contiguous(),
        local_positions.to(torch.int32).contiguous(),
        shortcut.to(torch.int32).contiguous(),
        out,
        width,
        out_width,
        KPOOL=kpool,
        BLOCK=_SELECT_BLOCK,
    )
    return out


def select_expand_tail(pool_logits, seq_token_base, pool_base, local_positions, topk, kpool):
    # [FUNCTION DOC] Fused pool selection, expansion and tail: picks the best pools from pool_logits
    # (queries by pools), expands them to token indices and appends the causal tail in one kernel.
    # Output width is topk + kpool - 1 rounded up to a multiple of 64; returns a new 32-bit matrix
    # (minus one padding). The per-query tensors are passed to the kernel unconverted, so they must
    # be contiguous device tensors. Asserts on unsuitable topk or kpool as in the pool selection.
    num_tokens = pool_logits.shape[0]
    scores, pools, group_topk = _pool_topk(pool_logits, topk, kpool)
    out_width = (topk + kpool - 1 + SPARSE_MLA_BLOCK - 1) // SPARSE_MLA_BLOCK * SPARSE_MLA_BLOCK
    out = torch.empty((num_tokens, out_width), dtype=torch.int32, device=pool_logits.device)
    grid = (num_tokens, triton.cdiv(out_width, _SELECT_BLOCK))
    _expand_topk_kernel[grid](
        pools,
        scores,
        seq_token_base,
        local_positions,
        pool_base,
        out,
        topk,
        group_topk,
        out_width,
        KPOOL=kpool,
        WITH_TAIL=True,
        BLOCK=_SELECT_BLOCK,
    )
    return out
