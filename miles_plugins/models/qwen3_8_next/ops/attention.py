# [MODULE DOC] Qwen3.8-Next full-attention layer with QSA (query-selected sparse attention). A
# Megatron self-attention subclass first runs the indexer to choose, for every query token, a set
# of key positions (budgeted selection plus the most recent uncompressed "tail" tokens), publishes
# that selection on itself, and then the normal Megatron forward runs; the replaced core attention
# reads the published selection and calls a sparse attention kernel. The selection is cleared after
# every forward. Supports packed (THD, 3-D query) and non-packed (SBHD, 4-D query) layouts.
"""Qwen3.8-Next full-attention layer: QSA indexer + sparse attention.

Each full-attention layer keeps ``indexer_budget`` key tokens per query and attends
only those.
"""

import torch
from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from miles.kernels.attention.qsa import qsa_block_sparse_attention, qsa_sparse_attention
from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import PackedBlockLayout, Qwen38NextQSAIndexer


class Qwen38NextQSACoreAttention(MegatronModule):
    # [CLASS DOC] Core-attention replacement with no parameters. It holds a hidden (non-registered)
    # reference to the owning attention module, from which it reads the selection and block-form
    # data at call time. The reference is set bypassing module registration so the owner is not
    # treated as a child (avoids a cycle). Softmax scale is kv_channels to the power minus one half.
    """Core attention restricted to the indexer's selection."""

    def __init__(self, config, layer_number: int, owner):
        # [FUNCTION DOC] Store layer number, the unregistered owner reference, and the softmax
        # scale computed from the config's per-head key channel count.
        super().__init__(config)
        self.layer_number = layer_number
        object.__setattr__(self, "_owner", owner)
        self.softmax_scale = config.kv_channels**-0.5

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_mask=None, **kwargs):
        # [FUNCTION DOC] Sparse attention restricted to the owner's published selection (per query,
        # a row of selected packed key indices, -1 meaning empty). The attention mask and extra
        # keyword arguments are ignored. Raises a runtime error if the owner has no selection
        # published (called outside the owner's forward), or if the query is neither 3-D nor 4-D.
        # THD query (tokens, heads, dim): if the owner published block-form data, uses the
        # block-sparse tensor-core kernel with the bitmap, bounds, bases and block size; otherwise
        # the token-level sparse kernel with the selection. Output reshaped to tokens by features.
        # SBHD query (sequence, batch, heads, dim): runs the token-level sparse kernel once per
        # batch element with the same selection (the block form is not used), stacks along batch and
        # reshapes to sequence, batch, heads times dim.
        # KNOWN ISSUE: the same selection (computed from batch element 0 only by the owner) is
        # applied to every batch element, so the SBHD path is only correct for batch size 1.
        selection = getattr(self._owner, "_qsa_selection", None)
        if selection is None:
            raise RuntimeError(
                "QSA core attention ran with no selection published. "
                "Qwen38NextAttention.forward sets it before delegating; reaching here "
                "means core_attention was called out of band."
            )
        block_form = getattr(self._owner, "_qsa_block_form", None)
        if query.dim() == 3:
            if block_form is not None:
                sel_bitmap, lo, hi, blk_base, tok_base, block_first, block_last, blk = block_form
                return qsa_block_sparse_attention(
                    query,
                    key,
                    value,
                    sel_bitmap,
                    lo,
                    hi,
                    blk_base,
                    tok_base,
                    block_first,
                    block_last,
                    self.softmax_scale,
                    blk,
                ).reshape(query.shape[0], -1)
            return qsa_sparse_attention(query, key, value, selection, self.softmax_scale).reshape(query.shape[0], -1)

        if query.dim() != 4:
            raise RuntimeError(f"QSA core attention expected a 3D (thd) or 4D (sbhd) query, got {tuple(query.shape)}")

        s, b, hq, d = query.shape
        out = [
            qsa_sparse_attention(query[:, i], key[:, i], value[:, i], selection, self.softmax_scale) for i in range(b)
        ]
        return torch.stack(out, dim=1).reshape(s, b, hq * d)


class Qwen38NextAttention(SelfAttention):
    # [CLASS DOC] Megatron self-attention that swaps in the QSA indexer and QSA core attention.
    # Transient state attributes (selection, block form, cumulative sequence lengths) are set at the
    # start of forward and always cleared in a finally block. Unsupported: selective core-attention
    # recompute (the selection would be gone at backward time) and interleaved rotary embeddings.
    """Megatron self-attention whose key set is chosen by a QSA indexer."""

    def __init__(self, config, submodules, layer_number=1, *args, **kwargs):
        # [FUNCTION DOC] Runs the parent constructor, then raises not-implemented error if
        # core-attention checkpointing is on or if the config uses interleaved rotary embeddings
        # (the indexer mirrors the rollout engine's neox-style RoPE). Creates the indexer, reads the
        # compress ratio (indexer block size in tokens) from the config, replaces core_attention
        # with the QSA core attention owned by this module, and initialises the transient selection
        # and block-form attributes to None. The cumulative-lengths attribute is only created by
        # forward.
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        if self.checkpoint_core_attention:
            # the selection is cleared after forward, so a backward-time core_attn recompute would find none
            raise NotImplementedError(
                "Qwen3.8-Next QSA does not support selective core_attn recompute; "
                "use full recompute or drop core_attn from --recompute-modules"
            )
        if config.rotary_interleaved:
            raise NotImplementedError("the QSA indexer mirrors sglang's neox RoPE; rotary_interleaved is unsupported")
        self.indexer = Qwen38NextQSAIndexer(config, layer_number=layer_number)
        self.compress_ratio = config.qwen3_8_next_indexer_compress_ratio
        self.core_attention = Qwen38NextQSACoreAttention(config, layer_number, owner=self)
        self._qsa_selection = None
        self._qsa_block_form = None

    @staticmethod
    def _packed_positions(cu_seqlens: Tensor, total: int) -> Tensor:
        # [FUNCTION DOC] HELPER: per-token position within its own sequence for a packed batch of
        # total tokens. Marks each sequence start (after the first) in a segment array, cumulative-
        # sums to get segment ids, and subtracts each segment's start index from the token index.
        # Returns a long tensor of length total. Assumes the boundaries cover the tokens and that no
        # sequence is empty (empty sequences would collide on the same start index).
        """Positions restarting at 0 for each sequence in a packed batch."""
        idx = torch.arange(total, device=cu_seqlens.device)
        starts = cu_seqlens[:-1].long()
        seg = torch.zeros(total, dtype=torch.long, device=cu_seqlens.device)
        seg[starts[1:]] = 1
        seg = seg.cumsum(0)
        return idx - starts[seg]

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        # [FUNCTION DOC] Compute the QSA selection, publish it, run the parent forward, clear state.
        # hidden_states is sequence, batch, hidden. Keyword arguments read here: packed_seq_params
        # (query cumulative lengths; raises not-implemented error if the params lack them) and
        # rotary_pos_emb (required; runtime error if absent). All arguments are forwarded unchanged
        # to the parent forward.
        # Steps: (1) indexer input is the hidden states, all-gathered over the sequence dimension
        # without gradient when sequence parallelism is on and TP size > 1. (2) Positions: restart at
        # zero per packed sequence, or a plain range. (3) Without gradient: run the indexer on batch
        # element 0 only to get per-token selected key indices; compute each token's sequence start
        # index; compute the "tail": the not-yet-compressed recent positions of the token's sequence,
        # a window of compress-ratio positions starting at (position + 1) floored to a multiple of the
        # ratio, keeping only those at or before the token itself (others -1); concatenate selection
        # and tail along the key axis; invalidate (set to -1) any entry outside the token's own
        # sequence start up to the token index, or negative. Store as the selection and compute the
        # block form. (4) Call the parent forward; in all cases afterwards reset the selection,
        # cumulative lengths and block form to None. Returns the parent's result.
        # KNOWN ISSUE: only batch element 0 feeds the indexer, so batch sizes above 1 reuse its
        # selection for all elements.
        packed = kwargs.get("packed_seq_params")

        indexer_states = hidden_states
        if getattr(self.config, "sequence_parallel", False):
            if get_tensor_model_parallel_world_size() > 1:
                with torch.no_grad():
                    indexer_states = gather_from_sequence_parallel_region(
                        hidden_states,
                        tensor_parallel_output_grad=False,
                        group=get_tensor_model_parallel_group(),
                    )
        seq = indexer_states.shape[0]
        if packed is not None:
            cu = getattr(packed, "cu_seqlens_q", None)
            if cu is None:
                raise NotImplementedError(
                    "packed_seq_params without cu_seqlens_q: QSA needs the sequence "
                    "boundaries to place positions and to keep attention inside a "
                    "document."
                )
            self._qsa_cu_seqlens = cu
            positions = self._packed_positions(cu, seq)
        else:
            self._qsa_cu_seqlens = None
            positions = torch.arange(seq, device=hidden_states.device)
        rotary_pos_emb = kwargs.get("rotary_pos_emb")
        if rotary_pos_emb is None:
            raise RuntimeError("the QSA indexer reuses the attention RoPE, but no rotary_pos_emb was passed")
        with torch.no_grad():
            selection = self.indexer(indexer_states[:, 0], positions, rotary_pos_emb, cu_seqlens=self._qsa_cu_seqlens)
            seq_start = torch.arange(seq, device=positions.device) - positions
            r = self.compress_ratio
            tail_in_seq = (positions + 1) // r * r
            offs = torch.arange(r, device=positions.device)
            tail_pos = tail_in_seq.unsqueeze(1) + offs.unsqueeze(0)  # in-seq
            tail_idx = seq_start.unsqueeze(1) + tail_pos  # pack index
            tail_idx = torch.where(
                tail_pos <= positions.unsqueeze(1),
                tail_idx,
                torch.full_like(tail_idx, -1),
            )
            merged = torch.cat([selection, tail_idx.to(selection.dtype)], dim=1)
            pack_pos = torch.arange(seq, device=positions.device).unsqueeze(1)
            seg_lo = seq_start.unsqueeze(1)
            ok = (merged >= seg_lo) & (merged <= pack_pos) & (merged >= 0)
            self._qsa_selection = torch.where(ok, merged, torch.full_like(merged, -1))
            self._qsa_block_form = self._publish_block_form(self._qsa_selection, positions, seq_start, seq)
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self._qsa_selection = None
            self._qsa_cu_seqlens = None
            self._qsa_block_form = None

    def _publish_block_form(self, selection: Tensor, positions: Tensor, seq_start: Tensor, seq: int):
        # [FUNCTION DOC] HELPER: convert per-token selected key indices into the block-granular form
        # for the tensor-core kernel. Inputs: selection (tokens by selected keys, -1 for empty),
        # in-sequence positions, per-token sequence start index, total token count seq.
        # Steps: use the stored cumulative lengths, or a single sequence spanning all tokens if they
        # are absent or describe only one sequence; build the packed block layout (blocks of
        # compress-ratio tokens per sequence); then create a zeroed byte bitmap of tokens by total
        # blocks and, for each valid selection entry, set the bit of its block, whose id is the
        # token's sequence block base plus the floor of (key index minus the sequence's token base)
        # divided by the ratio, clamped to the valid range. This test of any-valid forces a
        # host-device sync.
        # Returns an 8-tuple: bitmap, per-token lower bound (sequence token base), per-token upper
        # bound (the token's own packed index), per-token block base, per-token token base, first
        # token of each block, last token of each block, and the block size (the ratio). Index
        # tensors are int32.
        """Selection rows -> the block form the tensor-core kernel consumes.

        Block ids come from the indexer's per-sequence grid (``PackedBlockLayout``); a
        ceil(seq_start / ratio) base collides across sequences for some lengths.
        """
        ratio = self.compress_ratio
        cu = self._qsa_cu_seqlens
        if cu is None or cu.numel() <= 2:
            cu = torch.tensor([0, seq], device=selection.device)
        layout = PackedBlockLayout(cu, positions, ratio)
        blk_base = layout.token_block_start.to(torch.int32)
        tok_base = layout.token_start.to(torch.int32)
        num_blocks = layout.num_blocks

        sel_bitmap = torch.zeros(seq, num_blocks, dtype=torch.uint8, device=selection.device)
        valid = selection >= 0
        if bool(valid.any()):
            rows = torch.arange(seq, device=selection.device).unsqueeze(1).expand_as(selection)
            blk = blk_base.unsqueeze(1).long() + (selection - tok_base.unsqueeze(1)).div(ratio, rounding_mode="floor")
            sel_bitmap[rows[valid], blk[valid].long().clamp_(0, num_blocks - 1)] = 1
        lo = tok_base
        hi = (positions + seq_start).to(torch.int32)
        block_first = layout.block_first_token.to(torch.int32)
        block_last = layout.block_last_token.to(torch.int32)
        return sel_bitmap, lo, hi, blk_base, tok_base, block_first, block_last, ratio
