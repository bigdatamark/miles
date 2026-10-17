# [MODULE DOC] Sparse-attention indexer for DeepSeek-V4 layers with compression ratio 4. Scores each
# query against the compressed keys (produced by the compressor with rotation enabled) with a
# multi-head, weighted, ReLU-style logit kernel, and selects the top-k compressed keys per query.
# Supports unpacked and packed (THD) inputs, sequence-parallel input gathering, contiguous context
# parallelism (keys all-gathered across the cp group), optional row-balanced scoring across cp ranks
# (queries are exchanged so the causal work is even, then picks are returned to the owners), and an
# RL top-k record/replay hook. Output is an int32 index tensor with -1 marking absent picks.
import einops
import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.kernels.attention.dsa import indexer_logits_sbhd
from miles.kernels.attention.dsa.topk import get_dsa_topk_fn
from miles.utils.replay_base import indexer_replay_manager
from miles_plugins.models.deepseek_v4.ops.compressor import DeepSeekV4Compressor
from miles_plugins.models.deepseek_v4.ops.cp_row_balance import (
    LocalRows,
    RowBalancePlan,
    RowExchange,
    plan_causal_row_balance,
    send_rows_to_scorers,
)
from miles_plugins.models.deepseek_v4.ops.cp_utils import all_gather_cp, get_freqs_cis_for_cp, get_q_positions_for_cp
from miles_plugins.models.deepseek_v4.ops.qat import fp8_simulate_qat
from miles_plugins.models.deepseek_v4.ops.rope import apply_rotary_emb, wrapped_precompute_freqs_cis
from miles_plugins.models.deepseek_v4.ops.thd_utils import ThdLayout, compress_bounds_at_positions, get_q_positions_thd
from miles_plugins.models.deepseek_v4.ops.utils import rotate_activation


class V4Indexer(MegatronModule):
    # [CLASS DOC] Megatron module owning: a duplicated (non tensor-parallel) projection from the
    # low-rank query to heads times head dim, a duplicated projection from hidden states to one
    # weight per indexer head, and a compressor (ratio 4, rotation on) that produces the index keys.
    # The compress ratio is fixed at 4.
    """DSA Indexer for DeepSeek-V4 C4 layers."""

    def __init__(self, config: TransformerConfig, pg_collection=None, layer_id: int = 0):
        # [FUNCTION DOC] Reads hidden size, query low-rank rank (falls back to hidden size when unset),
        # indexer head count, head dim, top-k, top-k backend name, rope head dim, and whether fp8 QAT
        # simulation is on (config fp8 set) from the config. If no process-group collection is passed,
        # builds one from the global parallel state with tensor and context groups. Creates the two
        # bias-free duplicated Transformer-Engine linears and the compressor over the cp group.
        # Registers the module with the indexer replay manager, keyed by layer id as stream index, so
        # top-k picks can be recorded or replayed (no effect unless the manager is enabled).
        super().__init__(config=config)

        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size
        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.topk_backend = config.miles_dsa_topk_backend
        self.rope_head_dim = config.qk_pos_emb_head_dim
        self.compress_ratio = 4
        self.use_fp8_qat = config.fp8 is not None

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection

        self.linear_wq_b = TELinear(
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_weights_proj = TELinear(
            self.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.compressor = DeepSeekV4Compressor(
            config=config,
            head_dim=self.index_head_dim,
            compress_ratio=self.compress_ratio,
            rotate=True,
            cp_group=pg_collection.cp,
        )

        # RL rollout-routing-replay (R3) seam for the sparse-attention indexer topk: lets the miles
        # indexer_replay_manager record (on rollout) / replay (on train forward) the top-k KV picks,
        # mirroring the MoE routing-replay seam. No-op unless the manager is enabled.
        indexer_replay_manager.register_to_module(self, "indexer_replay", stream_idx=layer_id)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask=None,
        packed_seq_params=None,
        thd_layout: ThdLayout | None = None,
    ):
        # [FUNCTION DOC] Pipeline: (1) if sequence parallel with tensor-parallel size above one,
        # all-gather x and qr along the sequence. (2) Project qr to per-head queries. (3) Build rotary
        # frequencies for the global length (local length times cp size) using the compress rotary base;
        # unpacked: slice this rank's contiguous part; packed: index the table by each row's position
        # inside its own segment. (4) On a copy of the queries apply rotary in place to the last rope
        # dims (done batch-major), then Hadamard-rotate and, with fp8 QAT, fp8-simulate (block 128).
        # (5) Per-head weights come from the hidden-state projection, scaled by head-count to the minus
        # half and head-dim to the minus half, in float32. (6) Row balancing is enabled only when
        # cp size is above one, a cp group exists, and replay is not enabled; the exchange of queries
        # and weights is started before the compressor runs to overlap, and the local references are
        # deleted. (7) Compress keys from the compacted hidden rows if the layout is pre-grouped, else
        # from x. If the compressor returns nothing, asserts no rows are in flight and returns an
        # int32 tensor filled with -1 of shape batch, local length, top-k. (8) Under cp, all-gather
        # keys along dim 0; if the layout has the sequence-to-rank-row map, reorder keys to
        # sequence-major. (9) Select the top-k function (backend, wrapped by the replay manager) and
        # compute the picks for local rows. The mask and packed_seq_params arguments are unused.
        # Returns int32 picks, shape batch, rows, min(top-k, number of keys).
        # KNOWN ISSUE: the documented output claims a fixed shape of "top-k columns of -1" for the
        # empty case, which holds, but the non-empty case may have fewer than top-k columns; callers
        # must tolerate both widths.
        """Forward pass.

        Args:
            x:  hidden states [seqlen, batch, hidden_size]
            qr: low-rank query [seqlen, batch, q_lora_rank]
            mask: unused (the kernel's per-query KV bounds are generated internally)
            packed_seq_params: unused
            thd_layout: packed-stream layout, or None when unpacked

        Returns:
            topk_indices: [batch, seqlen, min(index_topk, n_kv)] int32, or index_topk columns of -1
            when no segment has a compressed key
        """

        # =========================================
        # Gather inputs if SP is enabled
        # =========================================
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        seqlen, bsz, _ = x.size()

        q, _ = self.linear_wq_b(qr)
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)

        rd = self.rope_head_dim
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_group = self.pg_collection.cp if hasattr(self.pg_collection, "cp") else None
        rope_base = self.config.csa_compress_rotary_base if self.compress_ratio else self.config.rotary_base
        freqs_cis = wrapped_precompute_freqs_cis(
            self.config, self.rope_head_dim, rope_base, False, seqlen * cp_size, x.device
        )
        if thd_layout is None:
            freqs_cis = get_freqs_cis_for_cp(freqs_cis, seqlen, cp_size, cp_group, stride=1)
        else:
            # Packed positions restart at every segment boundary and index the whole table, so
            # the rank's rows are selected here rather than by slicing it first.
            freqs_cis = freqs_cis.index_select(
                0, get_q_positions_thd(thd_layout.cu_seqlens, seqlen, thd_layout.global_start)
            )
        q = q.clone()
        q = einops.rearrange(q, "s b ... -> b s ...")
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = einops.rearrange(q, "b s ... -> s b ...")

        q = rotate_activation(q)
        if self.use_fp8_qat:
            q = fp8_simulate_qat(q, 128)

        weights, _ = self.linear_weights_proj(x)
        softmax_scale = self.index_head_dim**-0.5
        weights = (weights * (self.index_n_heads**-0.5) * softmax_scale).float()

        # Balance the causal scoring work over contiguous CP (cp_row_balance).
        # Replay data holds each rank's own rows, so replay scores them where they are.
        balance = cp_size > 1 and cp_group is not None and not indexer_replay_manager.enabled
        # started before the compressor to overlap it; unpacked, its CP all-gathers wait for the exchange
        exchange = start_row_exchange(q, weights, thd_layout, cp_group, balance=balance)
        del q, weights  # scored from exchange.wait()

        pre_grouped = thd_layout is not None and thd_layout.compressed_group_ids is not None
        k = self.compressor(thd_layout.hidden_compact if pre_grouped else x, thd_layout)
        if k is None:
            # Nothing to score when no segment reaches compress_ratio; -1 leaves each query on its
            # sliding window. The compressor returns None only without CP, so no rows are in flight.
            assert isinstance(exchange, LocalRows), "the compressor returned no keys while rows were in flight"
            return torch.full((bsz, seqlen, self.index_topk), -1, dtype=torch.int32, device=x.device)

        if cp_size > 1 and cp_group is not None:
            k = all_gather_cp(k, dim=0, cp_group=cp_group)
            if thd_layout is not None and thd_layout.seq_to_rank_row is not None:
                # Per-row bounds are sequence-major, so reorder the rank-major gather first.
                k = k.index_select(0, thd_layout.seq_to_rank_row.clamp(min=0).long())

        # RL replay can pin the rollout's top-k picks here; get_topk_fn is transparent when disabled.
        topk_fn = indexer_replay_manager.get_topk_fn(get_dsa_topk_fn(self.topk_backend), return_probs=False)
        return topk_for_local_rows(
            exchange,
            k,
            thd_layout,
            compress_ratio=self.compress_ratio,
            index_topk=self.index_topk,
            topk_fn=topk_fn,
        )


def start_row_exchange(q, weights, thd_layout, cp_group, *, balance: bool) -> RowExchange | LocalRows:
    # [FUNCTION DOC] HELPER: Detaches the queries and weights (the exchange carries no gradient). If
    # balancing is requested, asks for a row-balance plan; if none is requested or the plan says
    # balancing does not pay, returns a no-exchange handle holding the detached tensors and the
    # global positions of local rows (contiguous). Otherwise starts the asynchronous exchange and
    # returns its handle.
    # KNOWN ISSUE: because the tensors are detached, no gradient flows from the indexer top-k back to
    # the queries and weights; this is consistent with top-k selection being non-differentiable.
    """Start sending this rank's indexer rows to their scoring ranks, or keep them if balancing does not pay."""
    tensors = [q.detach(), weights.detach()]
    plan = _row_balance_plan(q.shape[0], thd_layout, cp_group, q.device) if balance else None
    if plan is None:
        cp_size = cp_group.size() if cp_group is not None else 1
        positions = get_q_positions_for_cp(q.shape[0], cp_size=cp_size, cp_group=cp_group, device=q.device)
        return LocalRows(tensors, positions)
    return send_rows_to_scorers(tensors, plan, cp_group)


def topk_for_local_rows(exchange, k, thd_layout, *, compress_ratio, index_topk, topk_fn):
    # [FUNCTION DOC] HELPER: Waits for the exchange to deliver queries and weights, runs the scoring
    # and top-k on the rows this rank scores, then sends the picks (row axis is dim 1) back to the
    # rows' owners. Returns picks in local row order.
    """The top-k picks for this rank's rows, in local order, scored on the rank ``exchange`` sent them to."""
    q, weights = exchange.wait()
    topk_indices = indexer_topk(
        q,
        k,
        weights,
        exchange.scored_positions,
        thd_layout,
        compress_ratio=compress_ratio,
        index_topk=index_topk,
        topk_fn=topk_fn,
    )
    # [batch, rows, topk]: the picks go back along the row dim
    return exchange.return_to_owners(topk_indices, dim=1)


def indexer_topk(q, k, weights, positions, thd_layout, *, compress_ratio, index_topk, topk_fn):
    # [FUNCTION DOC] HELPER: If there are no rows, returns an empty int32 result of the right trailing
    # shape (the scoring kernel cannot launch an empty grid). Visible key range per row: unpacked,
    # start is zero and end is (position plus one) divided by ratio (rounded down); packed, taken from
    # the per-segment bounds. Calls the logit kernel for scores of shape batch, rows, keys; flattens to
    # two dimensions, calls the top-k function with count min(top-k, keys), and reshapes back to
    # batch, rows, count. Returned indices are into the compressed keys alone; the semantics of
    # entries outside the visible range depend on the top-k kernel (unused slots may hold arbitrary
    # values per the surrounding code).
    """Score the query rows at global stream ``positions`` against their visible compressed keys.

    Args:
        q: [rows, batch, heads, head_dim] index queries of those rows
        k: [n_kv, batch, head_dim] every compressed key of the stream, sequence-major under THD
        weights: [rows, batch, heads] fp32 head weights
        positions: [rows] global stream positions of the rows
        thd_layout: packed-stream layout, or None when unpacked

    Returns:
        [batch, rows, min(index_topk, n_kv)] int32 compressed-key indices
    """
    if q.shape[0] == 0:
        # a balanced plan can leave a rank nothing to score, and TileLang cannot launch an empty grid
        return torch.empty(q.shape[1], 0, min(index_topk, k.shape[0]), dtype=torch.int32, device=q.device)
    if thd_layout is None:
        cu_ks = torch.zeros_like(positions, dtype=torch.int32)
        cu_ke = ((positions + 1) // compress_ratio).int()
    else:
        cu_ks, cu_ke = compress_bounds_at_positions(
            thd_layout.cu_seqlens, thd_layout.cu_seqlens_compressed, positions, ratio=compress_ratio
        )
    index_scores = indexer_logits_sbhd(q, k, weights, cu_ks, cu_ke)
    bsz, rows, n_kv = index_scores.shape
    topk_count = min(index_topk, n_kv)
    # flattened to [n_tokens, n_kv], the record/replay convention shared with the MoE seam
    topk_indices = topk_fn(index_scores.reshape(bsz * rows, n_kv), topk_count)
    return topk_indices.reshape(bsz, rows, topk_count)


def _row_balance_plan(seqlen_local, thd_layout, cp_group, device) -> RowBalancePlan | None:
    # [FUNCTION DOC] HELPER: Total rows is local length times cp size. Sequence lengths are one
    # sequence of that length when unpacked, otherwise the layout's host segment lengths (asserted
    # to sum to the total). Returns the memoized plan for this rank, or none when balancing would not
    # help.
    """This micro-batch's balanced exchange; every CP rank derives the same one."""
    cp_size = cp_group.size()
    total_rows = seqlen_local * cp_size
    # each batch row of an unpacked sample is one sequence
    seq_lens = (total_rows,) if thd_layout is None else thd_layout.seq_lens
    assert sum(seq_lens) == total_rows, f"segment lengths cover {sum(seq_lens)} rows of a {total_rows}-row stream"
    return plan_causal_row_balance(seq_lens, cp_rank=cp_group.rank(), cp_size=cp_size, device=device)
