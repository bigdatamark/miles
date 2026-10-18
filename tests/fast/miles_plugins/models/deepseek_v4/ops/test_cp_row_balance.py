# [MODULE DOC]
# Cpu tests for the causal row balancer used by context-parallel indexers in the deepseek v4 plugin.
# The balancer plans, per context-parallel rank, which rows of a packed stream of causal sequences go
# to which scoring rank so the per-row cost (position within its sequence plus one) is balanced. Tests
# compare the planner with a slow reference implementation, replay the two all-to-all exchanges in
# plain python, check balance and pairing properties, the gate that skips rebalancing when the gain is
# too small, error cases, and a real round trip over a gloo process group in subprocesses. The module
# registers itself as a cpu ci test in the stage-a-cpu suite with a 15 second estimate.
"""CPU tests for the causal row balancer shared by CP indexers.

Every rank's plan is built on the host and the two all-to-alls are replayed in Python, so the
routing, the balance and the gate are checked without GPUs; the exchange then runs over a gloo
process group. Rows carry their own global position, which makes a row routed to the wrong rank
or returned out of order visible.
"""

import random
from functools import partial

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles_plugins.models.deepseek_v4.ops.cp_row_balance import (
    plan_causal_row_balance,
    scoring_rank_of_chunk,
    send_rows_to_scorers,
)

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])


def _plans(seq_lens, cp_size, min_gain=0.1):
    # [FUNCTION DOC] HELPER: Returns the planner's result for every context-parallel rank from 0 to cp
    # size minus one, in rank order, on cpu with the given minimum gain (default 0.1). An entry is none
    # when the gate rejects balancing.
    return [
        plan_causal_row_balance(tuple(seq_lens), cp_rank=rank, cp_size=cp_size, device="cpu", min_gain=min_gain)
        for rank in range(cp_size)
    ]


def _reference_plan(seq_lens, cp_rank, cp_size, min_gain):
    # [FUNCTION DOC] HELPER: Slow vectorized reference for one rank's plan. Total rows are split
    # contiguously, rank_rows = total divided by cp size per owner rank. For every global row it derives
    # its sequence, its offset within the sequence, its chunk (the sequence cut into twice cp size chunks
    # by offset times 2 cp size divided by the sequence length, integer division), and its scoring rank
    # (even chunk c goes to rank c over 2, odd chunk c goes to rank cp size minus 1 minus c over 2).
    # Cost of a row is offset plus one. Compares the maximum per-rank cost under contiguous ownership
    # with that under scoring ranks; if balanced max exceeds one minus min gain times the contiguous max,
    # returns none (gate rejects). Otherwise returns a mapping with: send rows (stable argsort of the
    # local rows' scoring ranks), input splits (count of local rows per scoring rank), output splits
    # (for rows this rank scores, count per owner rank), and scored positions (global positions this rank
    # scores, ascending). Sequence lengths of zero are allowed.
    """The balancer spelled out one row at a time: each row's chunk, scoring rank and owner."""
    total = sum(seq_lens)
    rank_rows = total // cp_size
    lens = torch.tensor(seq_lens)
    positions = torch.arange(total)
    seq = torch.repeat_interleave(torch.arange(len(seq_lens)), lens)
    offset = positions - (torch.cumsum(lens, 0) - lens)[seq]
    chunk = offset * 2 * cp_size // lens[seq]
    scorer = torch.where(chunk % 2 == 0, chunk // 2, cp_size - 1 - chunk // 2)
    owner = positions // rank_rows
    cost = (offset + 1).double()
    contiguous = torch.zeros(cp_size, dtype=torch.float64).index_add_(0, owner, cost)
    balanced = torch.zeros(cp_size, dtype=torch.float64).index_add_(0, scorer, cost)
    if balanced.max() > (1 - min_gain) * contiguous.max():
        return None
    local_scorer = scorer[cp_rank * rank_rows : (cp_rank + 1) * rank_rows]
    mine = scorer == cp_rank
    return dict(
        send_rows=torch.argsort(local_scorer, stable=True),
        input_splits=tuple(torch.bincount(local_scorer, minlength=cp_size).tolist()),
        output_splits=tuple(torch.bincount(owner[mine], minlength=cp_size).tolist()),
        scored_positions=positions[mine],
    )


def _all_to_all(send_buffers, send_splits, recv_splits):
    # [FUNCTION DOC] HELPER: Pure python emulation of a single-tensor all-to-all across all ranks. Each
    # rank's send buffer is cut into pieces by its send splits; destination rank d receives piece d from
    # every source, concatenated in source order. Asserts each received piece's row count equals the
    # destination's receive split for that source (message: sender and receiver disagree). Returns the
    # list of received tensors per rank.
    """What all_to_all_single delivers: rank r gets each peer's slice for r, in peer order."""
    cp_size = len(send_buffers)
    chunks = [list(buffer.split(list(splits))) for buffer, splits in zip(send_buffers, send_splits, strict=True)]
    received = []
    for dst in range(cp_size):
        pieces = [chunks[src][dst] for src in range(cp_size)]
        assert [piece.shape[0] for piece in pieces] == list(recv_splits[dst]), "sender and receiver disagree"
        received.append(torch.cat(pieces))
    return received


def _costs(seq_lens):
    # [FUNCTION DOC] HELPER: Returns a double precision vector with the per-row causal cost of the
    # packed stream: for each sequence the numbers 1 up to its length, concatenated.
    return torch.cat([torch.arange(1, n + 1) for n in seq_lens]).double()


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("seed", range(12))
def test_matches_the_row_by_row_reference(cp_size, seed):
    # [FUNCTION DOC] Parametrized over cp size 2, 4, 8 and seeds 0 to 11. Draws one to ten random
    # sequence lengths (each from empty, 1, 3, a random value up to 64, or up to 3000) and appends one
    # extra sequence that pads the total to a multiple of cp size (a full cp size when already a
    # multiple). For min gain of minus one (gate always passes unless balance is worse) and 0.1, and
    # every rank, compares the planner with the reference: both reject or both accept, and when they
    # accept, send rows, input splits, output splits and scored positions are identical.
    rng = random.Random(seed)
    # empty, shorter-than-2cp and long sequences; the last one pads the stream to a multiple of cp
    seq_lens = [rng.choice([0, 1, 3, rng.randint(1, 64), rng.randint(1, 3000)]) for _ in range(rng.randint(1, 10))]
    seq_lens.append(cp_size - sum(seq_lens) % cp_size)
    for min_gain in (-1.0, 0.1):
        for rank in range(cp_size):
            plan = plan_causal_row_balance(
                tuple(seq_lens), cp_rank=rank, cp_size=cp_size, device="cpu", min_gain=min_gain
            )
            reference = _reference_plan(seq_lens, rank, cp_size, min_gain)
            assert (plan is None) == (reference is None), f"gate differs for {seq_lens}"
            if plan is not None:
                assert torch.equal(plan.send_rows, reference["send_rows"])
                assert plan.input_splits == reference["input_splits"]
                assert plan.output_splits == reference["output_splits"]
                assert torch.equal(plan.scored_positions, reference["scored_positions"])


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("seq_lens", [[4096], [3000, 1096], [2048, 1024, 512, 512], [4000, 90, 6]])
def test_rows_reach_their_scorer_and_come_back_in_order(cp_size, seq_lens):
    # [FUNCTION DOC] Parametrized over cp size and four sequence-length packs. Plans with the gate
    # effectively open; each rank's rows are tagged with their global position. Forward emulated
    # exchange: asserts every rank receives exactly its scored positions. Then multiplies the received
    # values by ten, replays the reverse exchange (splits swapped), un-permutes with the send rows, and
    # asserts each rank gets back its own rows times ten in original order.
    plans = _plans(seq_lens, cp_size, min_gain=-1.0)
    rank_rows = sum(seq_lens) // cp_size
    local_rows = [torch.arange(r * rank_rows, (r + 1) * rank_rows) for r in range(cp_size)]

    sent = [rows[plan.send_rows] for rows, plan in zip(local_rows, plans, strict=True)]
    scored = _all_to_all(sent, [p.input_splits for p in plans], [p.output_splits for p in plans])
    for plan, rows in zip(plans, scored, strict=True):
        assert torch.equal(rows, plan.scored_positions)

    results = [rows * 10 for rows in scored]
    returned = _all_to_all(results, [p.output_splits for p in plans], [p.input_splits for p in plans])
    for plan, rows, back in zip(plans, local_rows, returned, strict=True):
        local = torch.empty_like(back)
        local[plan.send_rows] = back
        assert torch.equal(local, rows * 10)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_one_sequence_balances_to_within_a_percent(cp_size):
    # [FUNCTION DOC] Parametrized over cp size. For one sequence of cp size times 8192 rows, sums the
    # row costs each rank scores and asserts the largest per-rank sum is under 1.01 times the smallest.
    seq_lens = [cp_size * 8192]
    costs = _costs(seq_lens)
    per_rank = torch.stack([costs[plan.scored_positions].sum() for plan in _plans(seq_lens, cp_size)])

    assert per_rank.max() / per_rank.min() < 1.01


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_one_sequence_is_a_single_pairwise_swap(cp_size):
    # [FUNCTION DOC] Parametrized over cp size. With one sequence of cp size times 4096 rows, for each
    # rank asserts it sends rows only to itself and to rank cp size minus 1 minus rank, and keeps exactly
    # half of its rows (2048).
    """Rank r keeps its first half and trades its second half with rank cp - 1 - r only."""
    rank_rows = 4096
    for rank, plan in enumerate(_plans([cp_size * rank_rows], cp_size)):
        peers = {dst for dst, rows in enumerate(plan.input_splits) if rows}
        assert peers <= {rank, cp_size - 1 - rank}
        assert plan.input_splits[rank] == rank_rows // 2


def test_chunk_pairing_pairs_early_with_late():
    # [FUNCTION DOC] With cp size 4 and 8 chunks, asserts the chunks scored by each rank are, for ranks
    # 0 to 3, [0,7], [2,5], [3,4], [1,6], and that each rank's chunk indices sum to 2 cp size minus 1
    # (7).
    """Chunk 2r goes to rank r and chunk 2r + 1 to rank cp - 1 - r, so costs sum to 2cp - 1 per rank."""
    cp_size = 4
    ranks = [scoring_rank_of_chunk(chunk, cp_size) for chunk in range(2 * cp_size)]
    by_rank = [[chunk for chunk, rank in enumerate(ranks) if rank == r] for r in range(cp_size)]

    assert by_rank == [[0, 7], [2, 5], [3, 4], [1, 6]]
    assert {sum(chunks) for chunks in by_rank} == {2 * cp_size - 1}


def test_a_pack_of_equal_short_documents_keeps_the_contiguous_layout():
    # [FUNCTION DOC] With 64 documents of 512 rows and cp size 4 at the default gate, asserts every
    # rank's plan is none (no rebalancing).
    """Each rank already holds the same mix of positions, so moving rows would only add traffic."""
    assert all(plan is None for plan in _plans([512] * 64, 4))


def test_a_long_document_in_a_pack_is_balanced():
    # [FUNCTION DOC] With one 24576-row document plus four 2048-row ones over 4 ranks, compares the max
    # per-rank cost under contiguous ownership with that of the planned scoring, and asserts the balanced
    # maximum is below 75 percent of the contiguous maximum.
    seq_lens = [24576, 2048, 2048, 2048, 2048]
    rank_rows = sum(seq_lens) // 4
    costs = _costs(seq_lens)
    contiguous = torch.stack([costs[r * rank_rows : (r + 1) * rank_rows].sum() for r in range(4)])
    balanced = torch.stack([costs[plan.scored_positions].sum() for plan in _plans(seq_lens, 4)])

    assert balanced.max() < 0.75 * contiguous.max()


def test_the_stream_must_split_evenly_over_the_ranks():
    # [FUNCTION DOC] A stream of 121 rows over 2 ranks makes the planner raise a value error mentioning
    # that it must split evenly.
    with pytest.raises(ValueError, match="split evenly"):
        plan_causal_row_balance((100, 21), cp_rank=0, cp_size=2, device="cpu")


def test_differentiable_inputs_are_rejected():
    # [FUNCTION DOC] Plans a 64-row stream for rank 0 of 2, then sending a tensor that requires
    # gradients (no process group needed) raises a value error mentioning no autograd.
    """The exchange carries no autograd, so a gradient would be dropped without a word."""
    plan = plan_causal_row_balance((64,), cp_rank=0, cp_size=2, device="cpu")
    with pytest.raises(ValueError, match="no autograd"):
        send_rows_to_scorers([torch.zeros(32, 4, requires_grad=True)], plan, cp_group=None)


def _exchange_worker(rank, world_size, port, seq_lens):
    # [FUNCTION DOC] HELPER: Per-process body run by the multiprocess launcher. Initializes a gloo
    # group, plans (asserting a plan exists), and builds three row-aligned tensors for this rank's
    # contiguous rows: global positions, a bfloat16 tensor with trailing shape 2 by 3 tagged with the
    # position, and a double tensor of position times 0.5. Sends them through the real exchange over
    # the world group and waits; asserts each received tensor matches the scored positions in the same
    # transformation. Then returns integer results shaped batch 2, rows, 4 (value is scored position
    # times 10) to the owners along dim 1 and asserts they equal the local positions times 10 in the same
    # shape. Always destroys the process group at the end.
    init_gloo(rank, world_size, port=port)
    try:
        rank_rows = sum(seq_lens) // world_size
        plan = plan_causal_row_balance(seq_lens, cp_rank=rank, cp_size=world_size, device="cpu")
        assert plan is not None
        positions = torch.arange(rank * rank_rows, (rank + 1) * rank_rows)
        # mixed dtypes and trailing shapes, each row tagged with its global position
        tagged = positions.view(-1, 1, 1).expand(-1, 2, 3).to(torch.bfloat16).contiguous()
        weights = positions.double().view(-1, 1) * 0.5

        exchange = send_rows_to_scorers([positions, tagged, weights], plan, dist.group.WORLD)
        received = exchange.wait()

        scored = exchange.plan.scored_positions
        assert torch.equal(received[0], scored)
        assert torch.equal(received[1], scored.view(-1, 1, 1).expand(-1, 2, 3).to(torch.bfloat16))
        assert torch.equal(received[2], scored.double().view(-1, 1) * 0.5)
        # results with rows along dim 1, as the indexer's [batch, rows, topk] picks
        picks = (scored * 10).view(1, -1, 1).expand(2, -1, 4).int()
        back = exchange.return_to_owners(picks, dim=1)
        assert torch.equal(back, (positions * 10).view(1, -1, 1).expand(2, -1, 4).int())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("seq_lens", [(512,), (320, 96, 64, 32)])
def test_exchange_round_trips_over_a_process_group(world_size, seq_lens):
    # [FUNCTION DOC] Parametrized over world size 2, 4 and two sequence-length packs. Launches that many
    # processes running the exchange worker with the sequence lengths bound.
    run_multiprocess(partial(_exchange_worker, seq_lens=seq_lens), world_size=world_size)


def test_exchange_round_trips_when_a_rank_scores_nothing():
    # [FUNCTION DOC] Uses a pack of 336 very short documents (235 of length 1, 76 of 2, 25 of 5) over 4
    # ranks. Asserts the planned scored counts per rank are exactly 336, 0, 126, 50 (rank 1 idle) and
    # then runs the multiprocess exchange round trip with 4 processes.
    """Documents shorter than 2cp leave chunks empty, so the gate can accept a plan with an idle rank."""
    seq_lens = (1,) * 235 + (2,) * 76 + (5,) * 25
    assert [plan.num_scored for plan in _plans(seq_lens, 4)] == [336, 0, 126, 50]
    run_multiprocess(partial(_exchange_worker, seq_lens=seq_lens), world_size=4)
