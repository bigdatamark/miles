# [MODULE DOC] Top-k token selection for the GLM-5.3 pooled-key indexer. Keys are pooled in groups of
# kpool consecutive tokens per sequence; queries score the pooled keys, the best pools are chosen, and
# the chosen pools are expanded back to token indices with each query's most recent (tail) tokens
# always included. Supports an RL record/replay mode that routes the pool top-k through the replay
# manager.
import torch

from miles.kernels.attention.dsa.kpool import append_tail_and_pad, pool_topk_to_token_fn, select_expand_tail
from miles.utils.replay_base import indexer_replay_manager


def kpool_select_topk(
    index_q: torch.Tensor,
    pooled_k: torch.Tensor,
    head_weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pool_cu_seqlens: torch.Tensor,
    index_topk: int,
    kpool: int,
) -> torch.Tensor:
    # [FUNCTION DOC] Inputs: index queries (tokens, heads, dim), pooled keys (pools, dim), per-head
    # weights, cumulative sequence lengths of tokens and of pools (pool counts per sequence from the
    # pool boundary helper), top-k count, pool size. Per token computes: its sequence (by binary
    # search of the token index in the cumulative lengths), the sequence's token base and pool base,
    # the local position inside the sequence, and the number of eligible pools: (local position plus
    # one) integer-divided by pool size, i.e. only completed pools that end at or before the query.
    # Scores: if there are pooled keys, runs the indexer logits kernel (imported lazily because it is
    # GPU-only) without gradients over each token's pool range [pool base, pool base plus eligible
    # pools) with 32-bit bounds; if there are none, a single column of negative infinity. Selection:
    # in replay mode, wraps a pool-to-token conversion function with the replay manager's top-k
    # function (recording or replaying), gets token indices for the top-k pools, and appends the tail
    # tokens and pads, where a "shortcut" flag marks queries whose visible length is within top-k
    # (so they attend to everything); otherwise one fused helper selects the pools, expands them to
    # tokens, appends the tail, and pads. Returns an integer tensor of token indices with a singleton
    # head axis inserted at dim 1 (tokens, 1, selected count).
    num_tokens = index_q.shape[0]
    device = index_q.device
    token_ids = torch.arange(num_tokens, device=device)
    seq_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    seq_token_base = cu_seqlens[seq_indices].to(torch.int32)
    pool_base = pool_cu_seqlens[seq_indices].to(torch.int32)
    local_positions = (token_ids - seq_token_base).to(torch.int32)
    eligible_pools = torch.div(local_positions + 1, kpool, rounding_mode="floor")

    if pooled_k.shape[0] > 0:
        # tilelang is GPU-only; importing it here keeps this module importable on CPU,
        # where tests/fast pins the per-sequence pool invariants.
        from miles.kernels.attention.dsa import indexer_logits

        with torch.no_grad():
            pool_logits = indexer_logits(
                index_q,
                pooled_k,
                head_weights,
                pool_base.to(torch.int32),
                (pool_base + eligible_pools).to(torch.int32),
            )
    else:
        pool_logits = torch.full((num_tokens, 1), float("-inf"), dtype=torch.float32, device=device)

    if indexer_replay_manager.enabled:
        topk_fn = indexer_replay_manager.get_topk_fn(
            pool_topk_to_token_fn(seq_token_base, pool_base, local_positions, kpool),
            return_probs=False,
        )
        tokens = topk_fn(pool_logits, index_topk)
        shortcut = (local_positions + 1) <= index_topk
        tokens = append_tail_and_pad(tokens, seq_token_base, local_positions, shortcut, kpool)
    else:
        tokens = select_expand_tail(pool_logits, seq_token_base, pool_base, local_positions, index_topk, kpool)
    return tokens.unsqueeze(1)
