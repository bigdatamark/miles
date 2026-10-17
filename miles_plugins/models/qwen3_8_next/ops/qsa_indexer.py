# [MODULE DOC] Indexer for Qwen3.8-Next's QSA sparse attention. For each query token it scores
# compressed key blocks (mean of every compress-ratio consecutive projected keys, per sequence),
# keeps the top blocks among those fully in the past of the query, and expands them into token
# indices. Contents: small numeric helpers (Gemma-style RMS norm, mean compression, neox RoPE,
# causal block masks), PackedBlockLayout (block grid for packed multi-sequence batches), and the
# indexer module. Numerics deliberately mirror the rollout engine's eager implementation (rounding
# order, block rotation positions) so training and rollout choose the same blocks.
"""Qwen3.8-Next QSA (Qwen Sparse Attention) indexer.

Runs on the full-attention layers and picks, per query token, which
``indexer_budget`` key tokens the sparse attention will look at.
"""

import math

import torch
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor


def _indexer_acc_dtype(x):
    # [FUNCTION DOC] HELPER: accumulation dtype for a tensor: its own dtype if it is already 32- or
    # 64-bit floating point, otherwise 32-bit float.
    return x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32


def gemma_rmsnorm_last_dim(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    # [FUNCTION DOC] HELPER: RMS normalisation over the last dim in the accumulation dtype: x times
    # reciprocal square root of (mean square plus eps), times (one plus weight) (weights are stored
    # as offsets from one, zero-initialised), then cast back to x's dtype. Same shape as x.
    """RMSNorm over the last dim with a Gemma-style ``1 + weight`` scale."""
    acc = _indexer_acc_dtype(x)
    xa = x.to(acc)
    var = xa.pow(2).mean(dim=-1, keepdim=True)
    return ((xa * torch.rsqrt(var + eps)) * (1.0 + weight.to(acc))).to(x.dtype)


def compress_keys_by_mean(token_k: Tensor, compress_ratio: int) -> Tensor:
    # [FUNCTION DOC] HELPER: compress per-token keys (tokens, dim) of one sequence into block keys
    # (ceil(tokens / ratio), dim) by averaging each run of ratio consecutive tokens. A trailing
    # partial run is averaged over its real token count only (zero padding is not counted). Exact
    # multiples use a plain mean over reshaped blocks.
    """``[T, head_dim] -> [ceil(T / r), head_dim]`` by averaging each run of ``r``."""
    tokens, dim = token_k.shape
    blocks = -(-tokens // compress_ratio)
    padded = blocks * compress_ratio
    if padded != tokens:
        pad = token_k.new_zeros(padded - tokens, dim)
        counts = token_k.new_ones(padded, 1)
        counts[tokens:] = 0
        summed = torch.cat([token_k, pad], dim=0).view(blocks, compress_ratio, dim).sum(1)
        denom = counts.view(blocks, compress_ratio, 1).sum(1).clamp_min(1)
        return summed / denom
    return token_k.view(blocks, compress_ratio, dim).mean(dim=1)


def apply_indexer_rope(x: Tensor, rotary_pos_emb: Tensor, positions: Tensor) -> Tensor:
    # [FUNCTION DOC] HELPER: rotate the leading rotary dims of x (tokens, heads, head dim) by
    # position, neox style (first half paired with second half). Takes the first half of the
    # flattened rotary table's last dim as frequencies (made contiguous), computes cosine and sine
    # over the whole table, then gathers the rows for the given positions (long), adds a heads axis,
    # and rounds cos / sin to x's dtype before use, as the rollout engine does. The rotated halves
    # are x1*cos - x2*sin and x2*cos + x1*sin; dims beyond the rotary width pass through unchanged.
    # Returns a new tensor of x's shape. Positions index the table directly (must be in range).
    """Neox RoPE on the leading ``rotary_dim`` dims of ``x`` ``[T, heads, head_dim]``.

    ``rotary_pos_emb`` is Megatron's non-interleaved ``[max_pos, 1, 1, rotary_dim]`` fp32
    table, ``cat(freqs, freqs)``. Mirrors sglang's eager ``apply_rotary_emb``: cos/sin
    rounded to ``x.dtype`` first, then per-op rounding.
    """
    freqs = rotary_pos_emb.flatten(1)
    half = freqs.shape[-1] // 2
    # cos/sin over the contiguous table, then gather, like sglang's cache: strided or gathered cos can move an ulp
    freqs = freqs[:, :half].contiguous()
    cos = freqs.cos()[positions.long()].unsqueeze(-2).to(x.dtype)
    sin = freqs.sin()[positions.long()].unsqueeze(-2).to(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., 2 * half :]], dim=-1)


def block_causal_mask(query_positions: Tensor, num_blocks: int, compress_ratio: int) -> Tensor:
    # [FUNCTION DOC] HELPER: boolean mask (queries, blocks) that is true for blocks completely in
    # the query's past, i.e. block index below (position + 1) integer-divided by the ratio. The
    # block containing the query itself, and any partial block, are excluded. Single-sequence form.
    """``[T, num_blocks]`` bool: which compressed blocks a query may attend to."""
    blocks = torch.arange(num_blocks, device=query_positions.device)
    first_invalid = (query_positions + 1) // compress_ratio
    return blocks.unsqueeze(0) < first_invalid.unsqueeze(1)


class PackedBlockLayout:
    # [CLASS DOC] HELPER: immutable bundle of index tensors describing the compressed-block grid of
    # a packed batch, each sequence's grid starting at its own first token. Attributes: num_blocks
    # (total across sequences, a Python int); per token: token_seq (sequence id), token_block
    # (global block id of the token's own block), token_block_start (first global block id of its
    # sequence), token_start (packed index of its sequence's first token); per block: block_seq,
    # block_local (index within its sequence), block_first_token (packed index of first token) and
    # block_last_token (packed index of last token, inclusive, clipped to the sequence end).
    """Per-sequence compressed-block grid for a packed (thd) batch.

    The rollout engine scores one request at a time, so each sequence's block grid
    must start at its own first token.
    """

    def __init__(self, cu_seqlens: Tensor, positions: Tensor, compress_ratio: int):
        # [FUNCTION DOC] Build all attributes on the positions' device. Inputs: cumulative sequence
        # lengths (moved to that device as long), per-token in-sequence positions (their count is
        # the total token count), block size. Blocks per sequence is ceil(length / ratio); the sum
        # is read to the host (a device sync). Segment ids per token come from marking every
        # sequence start after the first and cumulative-summing, so zero-length sequences are not
        # supported. Token block id is the sequence's block start plus position integer-divided by
        # the ratio. Per-block arrays come from repeating sequence ids by their block counts.
        device = positions.device
        cu = cu_seqlens.to(device=device, dtype=torch.long)
        lengths = cu[1:] - cu[:-1]
        blocks_per_seq = -(-lengths // compress_ratio)
        self.num_blocks = int(blocks_per_seq.sum())

        seq_block_start = torch.cat([blocks_per_seq.new_zeros(1), blocks_per_seq.cumsum(0)[:-1]])
        seq_token_start = cu[:-1]

        total = positions.numel()
        seg = torch.zeros(total, dtype=torch.long, device=device)
        seg[seq_token_start[1:]] = 1
        seg = seg.cumsum(0)

        # per token
        self.token_seq = seg
        self.token_block = seq_block_start[seg] + positions // compress_ratio
        self.token_block_start = seq_block_start[seg]
        self.token_start = seq_token_start[seg]
        # per block
        self.block_seq = torch.repeat_interleave(torch.arange(blocks_per_seq.numel(), device=device), blocks_per_seq)
        self.block_local = torch.arange(self.num_blocks, device=device) - seq_block_start[self.block_seq]
        self.block_first_token = seq_token_start[self.block_seq] + self.block_local * compress_ratio
        self.block_last_token = torch.minimum(self.block_first_token + compress_ratio, cu[1:][self.block_seq]) - 1


def compress_keys_by_mean_packed(token_k: Tensor, layout: PackedBlockLayout) -> Tensor:
    # [FUNCTION DOC] HELPER: packed-batch version of block compression. Sums keys into their
    # token's block with an indexed add (accumulation dtype) and divides by the per-block token
    # counts (clamped to at least one), then casts back to the input dtype. Returns (num_blocks,
    # dim); partial last blocks of each sequence are averaged over their real tokens.
    """``[T, head_dim] -> [layout.num_blocks, head_dim]``, averaging inside each sequence."""
    acc = _indexer_acc_dtype(token_k)
    dim = token_k.shape[-1]
    summed = torch.zeros(layout.num_blocks, dim, dtype=acc, device=token_k.device)
    summed.index_add_(0, layout.token_block, token_k.to(acc))
    counts = torch.zeros(layout.num_blocks, 1, dtype=acc, device=token_k.device)
    counts.index_add_(0, layout.token_block, torch.ones(token_k.shape[0], 1, dtype=acc, device=token_k.device))
    return (summed / counts.clamp_min(1)).to(token_k.dtype)


def packed_block_causal_mask(query_positions: Tensor, layout: PackedBlockLayout, compress_ratio: int) -> Tensor:
    # [FUNCTION DOC] HELPER: boolean mask (queries, global blocks), true for blocks of the query's
    # own sequence that are completely in its past: global block id at least the sequence's first
    # block and below that first block plus (position + 1) integer-divided by the ratio.
    """``[T, num_blocks]`` bool, restricted to each query's own sequence."""
    blocks = torch.arange(layout.num_blocks, device=query_positions.device).unsqueeze(0)
    lo = layout.token_block_start.unsqueeze(1)
    first_invalid = lo + ((query_positions + 1) // compress_ratio).unsqueeze(1)
    return (blocks >= lo) & (blocks < first_invalid)


class Qwen38NextQSAIndexer(MegatronModule):
    # [CLASS DOC] Per-layer indexer producing, for each query token, the token indices of the
    # highest-scoring past key blocks. Parameters: a replicated (duplicated parallel mode, no bias)
    # linear projecting hidden states to n_heads query heads plus kv_heads key heads of the indexer
    # head dim, and zero-initialised Gemma-style norm weights for queries and keys (sequence-parallel
    # tagged). The block budget is the token budget integer-divided by the compress ratio.
    # KNOWN ISSUE: the key path reshapes the key projection to rows of head dim, so any kv_heads
    # other than one would multiply the token axis; only a single indexer key head works.
    """Selects the sparse-attention budget for one full-attention layer."""

    def __init__(self, config: TransformerConfig, layer_number: int):
        # [FUNCTION DOC] Read indexer head counts, head dim, token budget, compress ratio from the
        # Qwen3.8-Next config fields and the layer-norm epsilon; derive the block budget; create the
        # projection and the two norm weights (config parameter dtype).
        super().__init__(config)
        self.layer_number = layer_number
        self.n_heads = config.qwen3_8_next_indexer_n_heads
        self.kv_heads = config.qwen3_8_next_indexer_kv_heads
        self.head_dim = config.qwen3_8_next_indexer_head_dim
        self.token_topk = config.qwen3_8_next_indexer_budget
        self.compress_ratio = config.qwen3_8_next_indexer_compress_ratio
        self.block_topk = self.token_topk // self.compress_ratio
        self.norm_eps = config.layernorm_epsilon

        self.index_qk_proj = TELinear(
            config.hidden_size,
            (self.n_heads + self.kv_heads) * self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        dtype = config.params_dtype
        self.q_layernorm = torch.nn.Parameter(torch.zeros(self.head_dim, dtype=dtype))
        self.k_layernorm = torch.nn.Parameter(torch.zeros(self.head_dim, dtype=dtype))
        for p in (self.q_layernorm, self.k_layernorm):
            p.sequence_parallel = config.sequence_parallel

    def project_qk(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        rotary_pos_emb: Tensor,
        layout: PackedBlockLayout | None = None,
    ):
        # [FUNCTION DOC] Project hidden states (tokens, hidden) to indexer queries and block keys.
        # The projection output is split into query part and key part. Queries: reshaped to rows of
        # head dim, Gemma-normed with the query weight, reshaped to (tokens, heads, head dim).
        # Keys: reshaped to rows of head dim, compressed to blocks by mean (per-sequence variant when
        # a layout is given), then normed with the key weight. RoPE: queries are rotated at their
        # positions; block keys at the position of their block's first token within its sequence
        # (block local index times ratio), matching the rollout engine. Returns (queries, block keys
        # of shape blocks by head dim).
        """``[T, hidden] -> (q [T, n_heads, head_dim], block_k [B, head_dim])``."""
        qk, _ = self.index_qk_proj(hidden_states)
        split = self.n_heads * self.head_dim
        q_raw, token_k = qk[..., :split], qk[..., split:]

        q = gemma_rmsnorm_last_dim(q_raw.reshape(-1, self.head_dim), self.q_layernorm, self.norm_eps).reshape(
            -1, self.n_heads, self.head_dim
        )

        token_k = token_k.reshape(-1, self.head_dim)
        if layout is None:
            block_k = compress_keys_by_mean(token_k, self.compress_ratio)
        else:
            block_k = compress_keys_by_mean_packed(token_k, layout)
        block_k = gemma_rmsnorm_last_dim(block_k, self.k_layernorm, self.norm_eps)

        if layout is None:
            block_local = torch.arange(block_k.shape[0], device=positions.device)
        else:
            block_local = layout.block_local
        # sglang rotates a compressed key at the position of its block's first token
        block_positions = block_local * self.compress_ratio
        q = apply_indexer_rope(q, rotary_pos_emb, positions)
        block_k = apply_indexer_rope(block_k.unsqueeze(1), rotary_pos_emb, block_positions).squeeze(1)
        return q, block_k

    def score_blocks(
        self,
        q: Tensor,
        block_k: Tensor,
        query_positions: Tensor,
        layout: PackedBlockLayout | None = None,
    ) -> Tensor:
        # [FUNCTION DOC] Score blocks for every query: in fp32, per-head dot products of queries
        # and block keys, passed through ReLU, summed over heads and divided by the square root of
        # head dim. Blocks not completely in the query's past (or, with a layout, outside its own
        # sequence) are set to negative infinity using the matching causal mask. Returns (queries,
        # blocks) fp32 logits.
        """``[T, num_blocks]`` fp32 logits, invalid blocks at ``-inf``."""
        scores = torch.einsum("mhd,nd->mnh", q.float(), block_k.float())
        logits = torch.relu(scores).sum(dim=-1) / math.sqrt(self.head_dim)
        if layout is None:
            valid = block_causal_mask(query_positions, block_k.shape[0], self.compress_ratio)
        else:
            valid = packed_block_causal_mask(query_positions, layout, self.compress_ratio)
        return logits.masked_fill(~valid, float("-inf"))

    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        rotary_pos_emb: Tensor,
        cu_seqlens: Tensor | None = None,
    ) -> Tensor:
        # [FUNCTION DOC] Select key tokens per query. A packed layout is built only when the
        # cumulative lengths describe more than one sequence (a single sequence uses the simpler
        # non-packed path, which assumes it starts at packed index zero). Steps: project, score,
        # take the top min(block budget, number of blocks) blocks per query; blocks scored negative
        # infinity (invalid) become -1; expand each chosen block into ratio consecutive token indices
        # (absolute packed indices when a layout is used, else block times ratio plus offset);
        # tokens of invalid blocks are -1; also -1 for tokens after the query's own index or (with a
        # layout) before its sequence start; flatten the block and offset axes, keep at most the
        # token budget columns, and return int32.
        # KNOWN ISSUE: the docstring says the result width is the token budget, but the width is
        # actually min(block budget, number of blocks) times the ratio, which is smaller when the
        # budget is not a multiple of the ratio or there are fewer blocks than the budget.
        """``[T, hidden] -> [T, token_topk]`` int32 token indices, ``-1`` where unused.

        ``positions`` restart at 0 per sequence; the returned indices are absolute in
        the packed buffer, so a packed batch must pass ``cu_seqlens``. ``rotary_pos_emb``
        is the attention's RoPE table, see ``apply_indexer_rope``.
        """
        layout = None
        if cu_seqlens is not None and cu_seqlens.numel() > 2:
            layout = PackedBlockLayout(cu_seqlens, positions, self.compress_ratio)

        q, block_k = self.project_qk(hidden_states, positions, rotary_pos_emb, layout=layout)
        logits = self.score_blocks(q, block_k, positions, layout=layout)

        k = min(self.block_topk, logits.shape[-1])
        block_scores, block_idx = torch.topk(logits, k, dim=-1)
        block_idx = block_idx.masked_fill(block_scores == float("-inf"), -1)

        offsets = torch.arange(self.compress_ratio, device=block_idx.device)
        if layout is None:
            tokens = block_idx.unsqueeze(-1) * self.compress_ratio + offsets
        else:
            block_local = block_idx - layout.token_block_start.unsqueeze(-1)
            tokens = layout.token_start.unsqueeze(-1).unsqueeze(-1) + (
                block_local.unsqueeze(-1) * self.compress_ratio + offsets
            )
        tokens = tokens.masked_fill(block_idx.unsqueeze(-1) < 0, -1).flatten(-2)
        limit = positions if layout is None else layout.token_start + positions
        tokens = tokens.masked_fill(tokens > limit.unsqueeze(-1), -1)
        if layout is not None:
            tokens = tokens.masked_fill(tokens < layout.token_start.unsqueeze(-1), -1)
        return tokens[..., : self.token_topk].to(torch.int32)
