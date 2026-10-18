# [MODULE DOC] CPU tests for the parallel state container. Data-parallel mode: intra mode when only
# intra-dp (or neither) is non-trivial, indep mode when only the independent dp is non-trivial, an
# assertion error when both are non-trivial; the effective dp axis is the active one; the
# effective dp-with-cp axis is a single group in intra mode and an (inner, outer) pair in indep mode
# with rank outer_rank times inner_size plus inner_rank and size the product. A final test checks the
# FSDP parallel state creation gives degree-one axes their own single-rank group (world size 4,
# rank 3) with the dp axis spanning the world.
# KNOWN ISSUE: imports appear mid-file, and the group-info class is imported twice (from the
# training utils parallel module at the top and from the fault-tolerance process group utilities
# below), rebinding the name for the whole module.
import pytest

from miles.backends.training_utils.parallel import GroupInfo, ParallelState, _DPMode


def _parallel_state(
    *,
    intra_dp: GroupInfo,
    indep_dp: GroupInfo,
    intra_dp_cp: GroupInfo | None = None,
) -> ParallelState:
    # [FUNCTION DOC] HELPER: Builds a parallel state with the given intra-dp and independent-dp
    # groups; intra-dp-with-cp defaults to a copy of intra-dp's rank and size with no process group;
    # cp, tp, pp, ep, etp are trivial (rank 0, size 1, no group).
    if intra_dp_cp is None:
        intra_dp_cp = GroupInfo(rank=intra_dp.rank, size=intra_dp.size, group=None)
    return ParallelState(
        intra_dp=intra_dp,
        intra_dp_cp=intra_dp_cp,
        cp=GroupInfo(rank=0, size=1, group=None),
        tp=GroupInfo(rank=0, size=1, group=None),
        pp=GroupInfo(rank=0, size=1, group=None),
        ep=GroupInfo(rank=0, size=1, group=None),
        etp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=indep_dp,
    )


def test_dp_mode_is_intra_when_only_intra_dp_non_trivial():
    # [FUNCTION DOC] Test: intra-dp rank 1 size 4 with trivial independent dp gives intra mode.
    """Non-trivial intra_dp with trivial indep_dp selects INTRA mode."""
    state = _parallel_state(
        intra_dp=GroupInfo(rank=1, size=4, group=None),
        indep_dp=GroupInfo(rank=0, size=1, group=None),
    )
    assert state._dp_mode == _DPMode.INTRA


def test_dp_mode_is_indep_when_only_indep_dp_non_trivial():
    # [FUNCTION DOC] Test: trivial intra-dp with independent dp rank 2 size 3 gives indep mode.
    """Trivial intra_dp with non-trivial indep_dp selects INDEP mode."""
    state = _parallel_state(
        intra_dp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=GroupInfo(rank=2, size=3, group=None),
    )
    assert state._dp_mode == _DPMode.INDEP


def test_dp_mode_is_intra_when_both_trivial():
    # [FUNCTION DOC] Test: both trivial gives intra mode.
    """Both trivial groups fall back to INTRA mode."""
    state = _parallel_state(
        intra_dp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=GroupInfo(rank=0, size=1, group=None),
    )
    assert state._dp_mode == _DPMode.INTRA


def test_dp_mode_raises_when_both_non_trivial():
    # [FUNCTION DOC] Test: both non-trivial (size 2 each) makes reading the mode raise an assertion
    # error saying they cannot both be non-trivial.
    """Both non-trivial groups violate the mutual-exclusion invariant."""
    state = _parallel_state(
        intra_dp=GroupInfo(rank=1, size=2, group=None),
        indep_dp=GroupInfo(rank=1, size=2, group=None),
    )
    with pytest.raises(AssertionError, match="cannot both be non-trivial"):
        _ = state._dp_mode


def test_effective_dp_returns_intra_dp_in_intra_mode():
    # [FUNCTION DOC] Test: in intra mode the effective dp axis is the same object as intra-dp.
    """effective_dp returns intra_dp when in INTRA mode."""
    intra_dp = GroupInfo(rank=3, size=8, group=None)
    state = _parallel_state(
        intra_dp=intra_dp,
        indep_dp=GroupInfo(rank=0, size=1, group=None),
    )
    assert state.effective_dp is intra_dp


def test_effective_dp_returns_indep_dp_in_indep_mode():
    # [FUNCTION DOC] Test: in indep mode the effective dp axis is the same object as indep-dp.
    """effective_dp returns indep_dp when in INDEP mode."""
    indep_dp = GroupInfo(rank=2, size=5, group=None)
    state = _parallel_state(
        intra_dp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=indep_dp,
    )
    assert state.effective_dp is indep_dp


def test_effective_dp_cp_uses_single_group_in_intra_mode():
    # [FUNCTION DOC] Test: intra mode with intra-dp-with-cp rank 1 size 4 gives an effective axis
    # with rank 1, size 4 and single-element lists of process groups and gloo groups (both none).
    """effective_dp_cp wraps intra_dp_cp as a single group in INTRA mode."""
    intra_dp_cp = GroupInfo(rank=1, size=4, group=None)
    state = _parallel_state(
        intra_dp=GroupInfo(rank=1, size=4, group=None),
        indep_dp=GroupInfo(rank=0, size=1, group=None),
        intra_dp_cp=intra_dp_cp,
    )

    result = state.effective_dp_cp

    assert result.rank == 1
    assert result.size == 4
    assert result.groups_inner_to_outer == [None]
    assert result.gloo_groups_inner_to_outer == [None]


def test_effective_dp_cp_uses_inner_outer_pair_in_indep_mode():
    # [FUNCTION DOC] Test: indep mode with inner (dp-with-cp) rank 1 size 4 and outer (independent
    # dp) rank 2 size 3 gives rank 2*4+1 = 9, size 12, and two-element group lists (inner then
    # outer, both none) for process groups and gloo groups.
    """effective_dp_cp combines intra_dp_cp (inner) and indep_dp (outer) in INDEP mode."""
    intra_dp_cp = GroupInfo(rank=1, size=4, group=None)
    indep_dp = GroupInfo(rank=2, size=3, group=None)
    state = _parallel_state(
        intra_dp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=indep_dp,
        intra_dp_cp=intra_dp_cp,
    )

    result = state.effective_dp_cp

    assert result.rank == 2 * 4 + 1
    assert result.size == 3 * 4
    assert result.groups_inner_to_outer == [None, None]
    assert result.gloo_groups_inner_to_outer == [None, None]


from argparse import Namespace
from unittest.mock import MagicMock, patch
from miles.backends.fsdp_utils import parallel as parallel_module
from miles.utils.ft_utils.process_group_utils import GroupInfo

_FSDP_PARALLEL_MODULE = "miles.backends.fsdp_utils.parallel"


def test_the_degree_one_axes_get_their_own_single_rank_group():
    # [FUNCTION DOC] Test: with the FSDP module's distributed handle mocked (world size 4, rank 3,
    # new-group returning a sentinel), meshes mocked for dp and fsdp, the gloo group mocked and the
    # group verification disabled, creating the FSDP parallel state with replicate size 1 creates
    # exactly one new group, over rank 3 only. Pipeline, expert, expert-tensor, independent dp,
    # tensor and context axes all use that group with rank 0 and size 1; intra-dp has rank 3 and
    # size 4.
    self_group = object()
    with (
        patch(f"{_FSDP_PARALLEL_MODULE}.dist") as dist,
        patch(f"{_FSDP_PARALLEL_MODULE}.build_fsdp_meshes", return_value={"dp": MagicMock(), "fsdp": MagicMock()}),
        patch(f"{_FSDP_PARALLEL_MODULE}.get_gloo_group", return_value=object()),
        patch.object(GroupInfo, "_verify_group", lambda self, group, name: None),
    ):
        dist.get_world_size.return_value = 4
        dist.get_rank.return_value = 3
        dist.new_group.return_value = self_group
        state = parallel_module.create_fsdp_parallel_state(Namespace(dp_replicate_size=1))
    dist.new_group.assert_called_once_with([3])
    for axis in (state.pp, state.ep, state.etp, state.indep_dp, state.tp, state.cp):
        assert axis.group is self_group
        assert (axis.rank, axis.size) == (0, 1)
    assert (state.intra_dp.rank, state.intra_dp.size) == (3, 4)
