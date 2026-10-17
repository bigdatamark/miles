# [MODULE DOC] Lightning indexer for DeepSeek-style sparse attention. Provides: helpers building
# per-query key ranges (packed causal, or causal over compressed keys); a function producing the
# full score matrix of every query against its key range (delegated to a GPU kernel from the
# tilelang subpackage, imported lazily so the module imports on CPU-only machines); batch-major
# variants; a differentiable gather of the scores at selected top-k key positions whose backward
# recomputes the selected scores inside one fused kernel instead of back-propagating through the
# full score matrix; and a combined function doing score, select, gather.
# Layout convention (packed): queries tokens by heads by head dim in bfloat16, keys tokens by head
# dim in bfloat16, per-head weights tokens by heads in float32, and per-query key range start and
# end (end exclusive) as 32-bit integer tensors. The selected-index tensor uses minus one for
# padding. The top-k selection policy is injected by the caller.
"""DSA lightning indexer: per-query scores over a KV range, top-k selection, and the gathered
top-k scores with a fused backward.

Every function takes the packed layout: q [T, H, D] bf16, k [T_kv, D] bf16, weights [T, H] fp32,
and per-query KV ranges cu_seqlen_ks / cu_seqlen_ke [T] int32 (query t scores keys in [ks, ke)).
A batched sbhd caller runs one packed problem per batch element with `indexer_logits_sbhd`.
"""

import torch


_BWD_MIN_TOPK = 32


def causal_ranges(cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Takes cumulative sequence boundaries (integer tensor, leading zero, one more
    # entry than sequences) and returns (starts, ends) tensors with one entry per packed token: the
    # start of the token's own sequence and the token's global position plus one. The token count is
    # read from the last boundary (a device-to-host synchronization). Each token's sequence is found
    # by a right-sided binary search into the boundaries. Raises an assertion error if any range is
    # empty (e.g. the boundaries are not increasing). starts has the dtype of the boundaries, ends is
    # 64-bit integer.
    """Packed causal ranges: query t attends keys [segment_start(t), t + 1)."""
    seq_len = cu_seqlens[-1].item()
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    assert torch.all((ends - starts) > 0)
    return starts, ends


def causal_ranges_compressed(seq_len_q: int, compress_ratio: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Builds key ranges for one sequence whose keys are compressed groups of
    # compress_ratio tokens: every query position p gets start 0 and end (p+1) integer-divided by
    # the ratio (so a query sees only completed groups; early queries may get an empty range).
    # Returns two 32-bit integer tensors of length seq_len_q on the given device.
    """Causal ranges over compressed KV: query p attends compressed groups [0, (p + 1) // ratio)."""
    positions = torch.arange(seq_len_q, device=device, dtype=torch.int32)
    ks = torch.zeros(seq_len_q, device=device, dtype=torch.int32)
    ke = ((positions + 1) // compress_ratio).to(torch.int32)
    return ks, ke


def indexer_logits(q, k, weights, cu_seqlen_ks, cu_seqlen_ke) -> torch.Tensor:
    # [FUNCTION DOC] Computes the dense indexer score matrix (queries by keys, float32) by calling
    # the tilelang forward kernel with the weights converted to float32 and cleaning enabled, so keys
    # outside each query's range hold minus infinity. Imports the kernel lazily (GPU only). Not
    # differentiable by itself; the gradient path is indexer_topk_scores.
    """Scores [T, T_kv] fp32; keys outside a query's range are -inf."""
    # tilelang is GPU-only; importing it lazily keeps the package importable on CPU
    from miles.kernels.attention.dsa.tilelang.indexer_fwd import indexer_fwd

    return indexer_fwd(q, k, weights.float(), cu_seqlen_ks, cu_seqlen_ke, clean_logits=True)


def indexer_logits_sbhd(q, k, weights, cu_seqlen_ks, cu_seqlen_ke) -> torch.Tensor:
    # [FUNCTION DOC] Sequence-major wrapper: allocates a float32 output of batch by query length by
    # key length and fills it by running the packed score function once per batch element (loop in
    # Python), slicing and making contiguous each element's queries, keys and weights. The same range
    # tensors are used for every batch element. Not differentiable by itself.
    """q [S, B, H, D], k [S_kv, B, D], weights [S, B, H]; ranges are shared across the batch.
    Returns [B, S, S_kv]."""
    seqlen, batch, _, _ = q.shape
    logits = torch.empty([batch, seqlen, k.shape[0]], device=q.device, dtype=torch.float32)
    for b in range(batch):
        logits[b] = indexer_logits(
            q[:, b].contiguous(), k[:, b].contiguous(), weights[:, b].contiguous(), cu_seqlen_ks, cu_seqlen_ke
        )
    return logits


def gather_topk_scores(logits, topk_indices, dim=-1):
    # [FUNCTION DOC] HELPER: Gathers score values from logits at the selected indices along dim
    # (default last). Padding entries (minus one) are looked up at index zero and then replaced by
    # minus infinity in the result. Returns a tensor shaped like the indices.
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    return torch.where(valid_mask, scores, float("-inf"))


def _pad_topk_pow2(topk_indices, grad_scores):
    # [FUNCTION DOC] HELPER: Pads the last dimension of the selected indices (with minus one) and
    # of the score gradient (with zero) up to the larger of 32 and the next power of two of the
    # current width, as the backward kernel needs; returns both as contiguous tensors (unchanged
    # width when already that size).
    topk = topk_indices.shape[-1]
    padded = max(_BWD_MIN_TOPK, 1 << (topk - 1).bit_length())
    if padded == topk:
        return topk_indices.contiguous(), grad_scores.contiguous()
    pad = padded - topk
    topk_indices = torch.nn.functional.pad(topk_indices, (0, pad), value=-1)
    grad_scores = torch.nn.functional.pad(grad_scores, (0, pad), value=0.0)
    return topk_indices.contiguous(), grad_scores.contiguous()


class _IndexerTopkScores(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function returning the gathered top-k scores. The full score
    # matrix is an input only for the forward gather and receives no gradient; instead backward
    # recomputes the selected scores from q, k and weights in a fused kernel and returns the
    # gradients for those three.
    @staticmethod
    def forward(ctx, q, k, weights, logits, topk_indices):
        # [FUNCTION DOC] Saves q, k, weights (already float32) and the indices, and returns the
        # gathered scores (minus infinity at padding).
        ctx.save_for_backward(q, k, weights, topk_indices)
        return gather_topk_scores(logits, topk_indices)

    @staticmethod
    def backward(ctx, grad_scores):
        # [FUNCTION DOC] Pads the indices and score gradient to a kernel-friendly width, lazily
        # imports and runs the tilelang indexer backward kernel, and returns gradients in the order
        # of the forward inputs: for q, for k, for weights, none for logits, none for indices.
        q, k, weights, topk_indices = ctx.saved_tensors
        topk_indices, grad_scores = _pad_topk_pow2(topk_indices, grad_scores)
        from miles.kernels.attention.dsa.tilelang.indexer_bwd import indexer_bwd

        grad_q, grad_w, grad_k = indexer_bwd(q, weights, k, topk_indices, grad_scores)
        return grad_q, grad_k, grad_w, None, None


def indexer_topk_scores(q, k, weights, logits, topk_indices) -> torch.Tensor:
    # [FUNCTION DOC] Public differentiable entry: returns the scores at the selected key positions
    # (queries by top-k, minus infinity at padding), taking logits from a previous score
    # computation. Weights are converted to float32 before the autograd function, so the gradient
    # flows back through that conversion. Gradients reach q, k and weights only.
    """Scores [T, topk] gathered from `logits` at `topk_indices` (-1 = padding, scored -inf).
    The backward recomputes the selected scores from q / k / weights in one TileLang kernel."""
    return _IndexerTopkScores.apply(q, k, weights.float(), logits, topk_indices)


def lighting_indexer(q, k, weights, cu_seqlen_ks, cu_seqlen_ke, topk: int, topk_fn):
    # [FUNCTION DOC] Computes the dense scores, asks the injected callable (taking the score matrix
    # and top-k count, returning 32-bit indices with minus one for padding) for the selection, then
    # returns (differentiable gathered scores, indices). The score matrix is kept alive through the
    # gather only. The function name keeps its original (misspelled) spelling and is the exported API.
    """Scores, top-k by `topk_fn(logits, topk)`, gathered scores. The caller owns the selection policy
    (e.g. routing replay), so the kernel holds no framework state. Returns (scores [T, topk],
    topk_indices [T, topk] int32)."""
    logits = indexer_logits(q, k, weights, cu_seqlen_ks, cu_seqlen_ke)
    topk_indices = topk_fn(logits, topk)
    return indexer_topk_scores(q, k, weights, logits, topk_indices), topk_indices
