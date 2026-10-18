# [MODULE DOC]
# Unit tests for the Megatron initialize module: random-seed setup from arguments and the check
# for whether the current rank is the main rank of the first data-parallel replica. Parallel
# topology is faked with a hand-built parallel state patched into the module under test; no
# distributed process group or GPU is needed.
import logging
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from miles.backends.megatron_utils import initialize
from miles.backends.training_utils.parallel import GroupInfo, ParallelState


def _parallel_state(
    *,
    tp_rank: int = 0,
    pp_rank: int = 0,
    pp_size: int = 1,
    intra_dp_cp_rank: int = 0,
    intra_dp_cp_size: int = 1,
    indep_dp_rank: int = 0,
    indep_dp_size: int = 1,
) -> ParallelState:
    # [FUNCTION DOC] HELPER: Builds a parallel state object with group-less placeholders for every
    # axis. Takes keyword-only ranks and sizes for tensor, pipeline, intra data-parallel plus
    # context-parallel, and independent data-parallel axes; all other axes (intra data-parallel,
    # context, expert, expert tensor) are rank 0, size 1. The tensor-parallel size is set to
    # max(tp_rank + 1, 1) so the given rank is always valid. Returns the state; no side effects.
    intra_dp = GroupInfo(rank=0, size=1, group=None)
    indep_dp = GroupInfo(rank=indep_dp_rank, size=indep_dp_size, group=None)
    return ParallelState(
        intra_dp=intra_dp,
        intra_dp_cp=GroupInfo(rank=intra_dp_cp_rank, size=intra_dp_cp_size, group=None),
        cp=GroupInfo(rank=0, size=1, group=None),
        tp=GroupInfo(rank=tp_rank, size=max(tp_rank + 1, 1), group=None),
        pp=GroupInfo(rank=pp_rank, size=pp_size, group=None),
        ep=GroupInfo(rank=0, size=1, group=None),
        etp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=indep_dp,
    )


def _patch_state(monkeypatch, state: ParallelState) -> None:
    # [FUNCTION DOC] HELPER: Replaces the parallel-state getter inside the initialize module with
    # one returning the given state, via the pytest monkeypatch fixture (undone at test end).
    monkeypatch.setattr(initialize, "get_parallel_state", lambda: state)


class TestSetRandomSeedFromArgs:
    # [CLASS DOC] Tests for the seed-setting function of the initialize module.
    def test_it_reseeds_rngs_and_only_rank_zero_logs(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # [FUNCTION DOC] Fakes the parallel state and stubs out the Megatron CUDA manual-seed call.
        # Arguments carry seed 1729, data-parallel random init on. Calls the seeding function as
        # rank 0, draws one value each from the Python, NumPy and torch CPU generators, then
        # switches the rank to 1, reseeds and draws again. Asserts both draw triples are equal
        # (the same seed is used regardless of rank here) and that the only log record at info
        # level is the single "setting random seeds to 1729" message, i.e. only rank 0 logged.
        """Repeated setup reproduces CPU RNG streams while only rank zero announces the seed."""
        _patch_state(monkeypatch, _parallel_state())
        monkeypatch.setattr(initialize.tensor_parallel, "model_parallel_cuda_manual_seed", lambda *args: None)
        args = SimpleNamespace(
            rank=0,
            seed=1729,
            data_parallel_random_init=True,
            te_rng_tracker=False,
            inference_rng_tracker=True,
        )

        with caplog.at_level(logging.INFO, logger=initialize.__name__):
            initialize.set_random_seed_from_args(args)
            first_values = (random.random(), np.random.random(), torch.rand(1).item())
            args.rank = 1
            initialize.set_random_seed_from_args(args)
            second_values = (random.random(), np.random.random(), torch.rand(1).item())

        assert second_values == first_values
        assert [record.message for record in caplog.records] == ["> setting random seeds to 1729 ..."]


def test_intra_mode_first_inner_rank_is_main_rank(monkeypatch):
    # [FUNCTION DOC] State: pipeline rank 1 of 2 (last stage), tensor rank 0, intra data-parallel
    # plus context rank 0 of 2, single independent replica. Asserts the main-rank check is true.
    """INTRA mode: last pp stage, tp rank 0, intra_dp_cp rank 0 is the main rank."""
    _patch_state(monkeypatch, _parallel_state(pp_rank=1, pp_size=2, intra_dp_cp_rank=0, intra_dp_cp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is True


def test_intra_mode_non_first_inner_rank_is_not_main_rank(monkeypatch):
    # [FUNCTION DOC] Same as the intra-mode main-rank case but with intra data-parallel plus
    # context rank 1 of 2. Asserts the main-rank check is false.
    """INTRA mode: a non-zero intra_dp_cp rank is not the main rank."""
    _patch_state(monkeypatch, _parallel_state(pp_rank=1, pp_size=2, intra_dp_cp_rank=1, intra_dp_cp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is False


def test_indep_mode_first_replica_first_inner_is_main_rank(monkeypatch):
    # [FUNCTION DOC] State: last pipeline stage, tensor rank 0, intra data-parallel plus context
    # rank 0 of 2, independent data-parallel replica 0 of 2. Asserts the check is true.
    """INDEP mode: first replica plus first inner rank yields the main rank."""
    _patch_state(
        monkeypatch,
        _parallel_state(
            pp_rank=1, pp_size=2, intra_dp_cp_rank=0, intra_dp_cp_size=2, indep_dp_rank=0, indep_dp_size=2
        ),
    )
    assert initialize.is_first_replica_megatron_main_rank() is True


def test_indep_mode_second_replica_main_rank_is_not_first_replica(monkeypatch):
    # [FUNCTION DOC] Regression test. State identical to the first-replica case except the
    # independent data-parallel replica index is 1 of 2. Asserts the check is false: the main
    # rank of a later replica must not count as the first-replica main rank.
    """INDEP regression: second replica's main rank is not the first-replica main rank."""
    _patch_state(
        monkeypatch,
        _parallel_state(
            pp_rank=1, pp_size=2, intra_dp_cp_rank=0, intra_dp_cp_size=2, indep_dp_rank=1, indep_dp_size=2
        ),
    )
    assert initialize.is_first_replica_megatron_main_rank() is False


def test_non_zero_tp_rank_is_not_main_rank(monkeypatch):
    # [FUNCTION DOC] State: tensor rank 1, last pipeline stage of 2, other axes trivial. Asserts
    # the main-rank check is false.
    """A non-zero tp rank is never the main rank even on the first replica's first inner rank."""
    _patch_state(monkeypatch, _parallel_state(tp_rank=1, pp_rank=1, pp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is False


def test_non_last_pp_stage_is_not_main_rank(monkeypatch):
    # [FUNCTION DOC] State: tensor rank 0 but pipeline rank 0 of 2 (not the last stage). Asserts
    # the main-rank check is false.
    """A pp rank that is not the last pipeline stage is never the main rank."""
    _patch_state(monkeypatch, _parallel_state(tp_rank=0, pp_rank=0, pp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is False
