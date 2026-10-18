# [MODULE DOC] Unit test for the debug train-data dump: among the ranks sharing a data and context
# parallel shard, only the rank with tensor-parallel rank zero on the last pipeline stage writes
# the file. A fake parallel state is installed into the parallel module for the test.
"""`save_debug_train_data` must write exactly one file per (dp, cp) shard."""

from types import SimpleNamespace

import pytest
import torch

import miles.backends.training_utils.parallel as parallel
from miles.backends.training_utils.metrics import train_dump
from miles.utils.ft_utils.process_group_utils import GroupInfo


def _parallel_state(*, tp_rank: int, is_pp_last_stage: bool) -> parallel.ParallelState:
    # [FUNCTION DOC] HELPER: Builds a parallel state where every group is trivial (rank 0, size 1,
    # no process group) except tensor parallel, which has the given rank and size 2, and the
    # pipeline-last-stage flag is the given value. Returns it; no side effects.
    trivial = GroupInfo(rank=0, size=1, group=None)
    return parallel.ParallelState(
        intra_dp=trivial,
        intra_dp_cp=trivial,
        cp=trivial,
        tp=GroupInfo(rank=tp_rank, size=2, group=None),
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=trivial,
        is_pp_last_stage=is_pp_last_stage,
    )


@pytest.mark.parametrize(
    ("tp_rank", "is_pp_last_stage", "writes"),
    [(0, True, True), (1, True, False), (0, False, False)],
    ids=["tp0_last_stage_writes", "tp_peer_skips", "earlier_pp_stage_skips"],
)
def test_only_one_rank_per_shard_writes(tmp_path, monkeypatch, tp_rank, is_pp_last_stage, writes):
    # [FUNCTION DOC] Parametrized over (tp rank 0, last stage: writes), (tp rank 1: skips) and
    # (earlier pipeline stage: skips). Patches the module-level parallel state and the global
    # distributed rank (fixed to 5), builds arguments whose dump path template uses rollout id and
    # rank in a temporary directory, calls the dump with rollout id 3 and a minimal rollout data
    # mapping, and asserts the file for rollout 3 and rank 5 exists exactly when a write is
    # expected.
    monkeypatch.setattr(
        parallel, "_parallel_state", _parallel_state(tp_rank=tp_rank, is_pp_last_stage=is_pp_last_stage)
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 5)
    args = SimpleNamespace(
        save_debug_train_data=str(tmp_path / "{rollout_id}_{rank}.pt"),
        qkv_format="thd",
    )

    train_dump.save_debug_train_data(args, rollout_id=3, rollout_data={"sample_indices": [0]})

    assert (tmp_path / "3_5.pt").exists() == writes
