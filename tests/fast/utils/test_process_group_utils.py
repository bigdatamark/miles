# [MODULE DOC]
# Tests for the fault-tolerance process-group utilities: group info records (rank, size, group,
# optional gloo group; a post-init check verifies rank against the group), combined groups info
# (single or inner/outer pair; pair rank is outer rank times inner size plus inner rank, size is
# the product), the native process-group helper (rank, size, all-reduce, reduce, broadcast,
# barrier, all-gather, gather, gather-object) in 4-process gloo worlds, the raw (torchft-style)
# helper with mock groups (rank from a private attribute, reduce redirected to all-reduce,
# barrier waits on its work and raises on false), the multi-group helper (nested all-reduce giving
# global results and bitwise-equal values, empty group list no-op, max op, nested gather-object
# returning on the global root only), the wait-checking helper (raises when wait returns false,
# propagates wait errors), and the collective boolean-and across ranks.
"""Tests for process_group_utils: GroupInfo, GroupsInfo, GeneralPGUtil, MultiPGUtil."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess
from torch.distributed.device_mesh import init_device_mesh

from miles.utils.ft_utils.process_group_utils import (
    GroupInfo,
    GroupsInfo,
    MultiPGUtil,
    _check_wait,
    _NativePGUtil,
    _RawPGUtil,
    collective_bool_and,
)


def _make_mesh():
    # [FUNCTION DOC] HELPER: Build a 2 by 2 CPU device mesh with dimension names "outer" and
    # "inner"; requires an initialized four-process world.
    return init_device_mesh("cpu", mesh_shape=(2, 2), mesh_dim_names=("outer", "inner"))


# -- GroupInfo / GroupsInfo tests (no distributed needed) --


class TestGroupInfo:
    # [CLASS DOC] Tests of the group info record.
    def test_construction_with_none_group(self) -> None:
        # [FUNCTION DOC] A record with no group stores rank and size and has no gloo group.
        info = GroupInfo(rank=0, size=4, group=None)
        assert info.rank == 0
        assert info.size == 4
        assert info.gloo_group is None


class TestGroupsInfo:
    # [CLASS DOC] Tests of combining group info records.
    def test_from_single(self) -> None:
        # [FUNCTION DOC] A single record gives the same rank and size and one-element lists of
        # groups and gloo groups (both none).
        info = GroupInfo(rank=2, size=4, group=None)
        result = GroupsInfo.from_single(info)
        assert result.rank == 2
        assert result.size == 4
        assert result.groups_inner_to_outer == [None]
        assert result.gloo_groups_inner_to_outer == [None]

    @patch.object(GroupInfo, "__post_init__", lambda self: None)
    def test_from_single_with_gloo(self) -> None:
        # [FUNCTION DOC] With the record's post-init check disabled and sentinel group objects, the
        # single combination carries the sentinel group and gloo group in its lists.
        sentinel_group = object()
        sentinel_gloo = object()
        info = GroupInfo(rank=0, size=2, group=sentinel_group, gloo_group=sentinel_gloo)
        result = GroupsInfo.from_single(info)
        assert result.groups_inner_to_outer == [sentinel_group]
        assert result.gloo_groups_inner_to_outer == [sentinel_gloo]

    def test_from_pair(self) -> None:
        # [FUNCTION DOC] Inner (rank 1, size 3) and outer (rank 2, size 4) give rank 7, size 12 and
        # two none gloo groups.
        inner = GroupInfo(rank=1, size=3, group=None)
        outer = GroupInfo(rank=2, size=4, group=None)
        result = GroupsInfo.from_pair(inner=inner, outer=outer)
        assert result.rank == 2 * 3 + 1  # 7
        assert result.size == 4 * 3  # 12
        assert result.gloo_groups_inner_to_outer == [None, None]

    @patch.object(GroupInfo, "__post_init__", lambda self: None)
    def test_from_pair_with_gloo(self) -> None:
        # [FUNCTION DOC] With the post-init check disabled and sentinel gloo groups, the pair's gloo
        # list is inner first, then outer.
        inner_gloo = object()
        outer_gloo = object()
        inner = GroupInfo(rank=0, size=2, group=None, gloo_group=inner_gloo)
        outer = GroupInfo(rank=0, size=3, group=None, gloo_group=outer_gloo)
        result = GroupsInfo.from_pair(inner=inner, outer=outer)
        assert result.gloo_groups_inner_to_outer == [inner_gloo, outer_gloo]

    def test_from_pair_rank_zero_only_when_both_zero(self) -> None:
        # [FUNCTION DOC] Both ranks zero (sizes 2 and 3) give rank 0 and size 6.
        result = GroupsInfo.from_pair(
            inner=GroupInfo(rank=0, size=2, group=None),
            outer=GroupInfo(rank=0, size=3, group=None),
        )
        assert result.rank == 0
        assert result.size == 6

    def test_from_pair_rank_nonzero_when_inner_nonzero(self) -> None:
        # [FUNCTION DOC] Inner rank 1 with outer rank 0 gives rank 1.
        result = GroupsInfo.from_pair(
            inner=GroupInfo(rank=1, size=2, group=None),
            outer=GroupInfo(rank=0, size=3, group=None),
        )
        assert result.rank == 1

    def test_from_pair_rank_nonzero_when_outer_nonzero(self) -> None:
        # [FUNCTION DOC] Outer rank 1 with inner rank 0 and inner size 2 gives rank 2.
        result = GroupsInfo.from_pair(
            inner=GroupInfo(rank=0, size=2, group=None),
            outer=GroupInfo(rank=1, size=3, group=None),
        )
        assert result.rank == 2


# -- Parameterized GeneralPGUtil tests (native vs torchft code paths) --


def _worker_pg_util_ops(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Per-rank process body. Init gloo, create a group of all ranks, and with
    # the native helper assert: rank and size; all-reduce sum equals 10; reduce to root sums on
    # rank 0; broadcast from root sets 99; barrier; all-gather gives ranks 0..3; gather collects
    # on rank 0 only. Then assert a group info with the right rank passes and a wrong rank raises
    # an assertion error. Always destroys the process group.
    # KNOWN ISSUE: the expected sums are hard-coded for exactly four ranks, and the loop over
    # helper classes has only one entry.
    """Test GeneralPGUtil operations with native gloo groups."""
    init_gloo(rank, world_size, port=port)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

        for util_cls in [_NativePGUtil]:
            util = util_cls()

            # get_rank / get_size
            assert util.get_rank(group) == rank
            assert util.get_size(group) == world_size

            # all_reduce SUM
            tensor = torch.tensor([float(rank + 1)])
            util.all_reduce(tensor, group, op=dist.ReduceOp.SUM)
            assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0

            # reduce to root
            tensor = torch.tensor([float(rank + 1)])
            util.reduce(tensor, group, op=dist.ReduceOp.SUM)
            if rank == 0:
                assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0

            # broadcast from root
            tensor = torch.tensor([99.0]) if rank == 0 else torch.tensor([0.0])
            util.broadcast(tensor, group)
            assert tensor.item() == 99.0

            # barrier (all ranks must reach it; returns nothing)
            util.barrier(group)

            # all_gather
            input_t = torch.tensor([float(rank)])
            output_t = [torch.zeros(1) for _ in range(world_size)]
            util.all_gather(output_t, input_t, group=group)
            assert [t.item() for t in output_t] == [0.0, 1.0, 2.0, 3.0]

            # gather
            input_t = torch.tensor([float(rank)])
            if rank == 0:
                gather_list = [torch.zeros(1) for _ in range(world_size)]
                util.gather(input_t, gather_list=gather_list, group=group)
                assert [t.item() for t in gather_list] == [0.0, 1.0, 2.0, 3.0]
            else:
                util.gather(input_t, gather_list=None, group=group)

        # GroupInfo verification
        GroupInfo(rank=rank, size=world_size, group=group)
        wrong_rank = (rank + 1) % world_size
        with pytest.raises(AssertionError):
            GroupInfo(rank=wrong_rank, size=world_size, group=group)
    finally:
        dist.destroy_process_group()


def test_pg_util_ops() -> None:
    # [FUNCTION DOC] Run the helper-operations worker in a four-process world.
    run_multiprocess(_worker_pg_util_ops, world_size=4)


def _worker_gather_object(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Per-rank process body. For four kinds of Python objects (mapping, list,
    # string, tuple), rank 0 gathers into a preallocated list and asserts no entry is none; other
    # ranks send with no destination list. Destroys the process group at the end.
    """Verify _NativePGUtil gather_object returns correct results on rank 0."""
    init_gloo(rank, world_size, port=port)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

        test_objects = [
            {"rank": rank, "value": rank * 10},
            [rank, rank + 1, "hello"],
            f"string_from_rank_{rank}",
            (rank, {"nested": True}),
        ]

        util = _NativePGUtil()
        for obj in test_objects:
            if rank == 0:
                result: list[Any] = [None] * world_size
                util.gather_object(obj, result, group=group)
                assert all(r is not None for r in result), f"Incomplete gather for obj type={type(obj)}"
            else:
                util.gather_object(obj, None, group=group)
    finally:
        dist.destroy_process_group()


def test_gather_object() -> None:
    # [FUNCTION DOC] Run the gather-object worker in a four-process world.
    run_multiprocess(_worker_gather_object, world_size=4)


class TestRawPGUtilUnit:
    # [CLASS DOC] Mock-based tests of the raw helper.
    """Unit tests for _RawPGUtil using mock groups (torchft-style with _rank attr)."""

    def test_get_rank_returns_group_rank(self) -> None:
        # [FUNCTION DOC] The rank is read from the group's private rank attribute (3).
        group = MagicMock()
        group._rank = 3
        assert _RawPGUtil().get_rank(group) == 3

    def test_get_size_returns_group_size(self) -> None:
        # [FUNCTION DOC] The size is whatever the group's size method returns (8).
        group = MagicMock()
        group.size.return_value = 8
        assert _RawPGUtil().get_size(group) == 8

    @patch("miles.utils.ft_utils.process_group_utils.dist.AllreduceOptions", MagicMock)
    def test_all_reduce_calls_group_allreduce(self) -> None:
        # [FUNCTION DOC] With the all-reduce options class patched to a mock, all-reduce on a mock
        # group calls the group's allreduce once (and waits on the returned work).
        group = MagicMock()
        work = MagicMock()
        work.wait.return_value = True
        group.allreduce.return_value = work
        tensor = torch.tensor([1.0])
        _RawPGUtil().all_reduce(tensor, group, op=dist.ReduceOp.SUM)
        group.allreduce.assert_called_once()

    @patch("miles.utils.ft_utils.process_group_utils.dist.AllreduceOptions", MagicMock)
    def test_reduce_falls_back_to_group_allreduce(self) -> None:
        # [FUNCTION DOC] Reduce on a mock group calls the group's allreduce once and never its
        # reduce method.
        # _RawPGUtil.reduce intentionally redirects to all_reduce because torchft's
        # ProcessGroupWrapper doesn't override reduce() — see implementation comment.
        group = MagicMock()
        work = MagicMock()
        work.wait.return_value = True
        group.allreduce.return_value = work
        tensor = torch.tensor([1.0])
        _RawPGUtil().reduce(tensor, group, op=dist.ReduceOp.SUM)
        group.allreduce.assert_called_once()
        group.reduce.assert_not_called()

    @patch("miles.utils.ft_utils.process_group_utils.dist.BroadcastOptions", MagicMock)
    def test_broadcast_calls_group_broadcast(self) -> None:
        # [FUNCTION DOC] With the broadcast options class patched, broadcast calls the group's
        # broadcast once.
        group = MagicMock()
        work = MagicMock()
        work.wait.return_value = True
        group.broadcast.return_value = work
        tensor = torch.tensor([1.0])
        _RawPGUtil().broadcast(tensor, group)
        group.broadcast.assert_called_once()

    def test_barrier_waits_on_the_work(self) -> None:
        # [FUNCTION DOC] Barrier calls the group's barrier once and waits on the returned work once.
        # Regression: a fire-and-forget group.barrier() (no .wait()) escapes torchft's
        # per-collective timeout, letting a dead peer hang until the NCCL watchdog kills
        # the process. _RawPGUtil.barrier MUST wait on the returned work.
        group = MagicMock()
        work = MagicMock()
        work.wait.return_value = True
        group.barrier.return_value = work
        _RawPGUtil().barrier(group)
        group.barrier.assert_called_once()
        work.wait.assert_called_once()

    def test_barrier_raises_when_wait_returns_false(self) -> None:
        # [FUNCTION DOC] When the work's wait returns false, barrier raises a runtime error saying
        # the distributed operation barrier failed.
        # A failed barrier must raise so callers (e.g. the dumper) catch it and go
        # degraded, instead of silently reporting success on a fire-and-forget call.
        group = MagicMock()
        work = MagicMock()
        work.wait.return_value = False
        group.barrier.return_value = work
        with pytest.raises(RuntimeError, match="distributed operation barrier failed"):
            _RawPGUtil().barrier(group)


# -- MultiPGUtil tests --


def _worker_multi_pg_util_all_reduce(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Per-rank process body on a 2 by 2 mesh. Steps: all-reduce over the
    # inner group only (sums 3 for ranks 0-1 and 7 for ranks 2-3); over inner then outer gives the
    # global sum 10; scaled values are bitwise equal on all ranks (compared via gathered bytes); an
    # empty group list leaves the tensor unchanged; max over both groups gives 4. Destroys the
    # process group at the end.
    init_gloo(rank, world_size, port=port)
    try:
        mesh = _make_mesh()
        inner_group = mesh.get_group("inner")
        outer_group = mesh.get_group("outer")

        # Step 1: single group
        tensor = torch.tensor([float(rank + 1)])
        MultiPGUtil.all_reduce(tensor, [inner_group], op=dist.ReduceOp.SUM)
        expected = {0: 3.0, 1: 3.0, 2: 7.0, 3: 7.0}[rank]
        assert tensor.item() == expected, f"rank {rank}: expected {expected}, got {tensor.item()}"

        # Step 2: two groups = global sum
        tensor = torch.tensor([float(rank + 1)])
        MultiPGUtil.all_reduce(tensor, [inner_group, outer_group], op=dist.ReduceOp.SUM)
        assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0

        # Step 3: bitwise equality across all ranks
        tensor = torch.tensor([float(rank + 1) * 0.1])
        MultiPGUtil.all_reduce(tensor, [inner_group, outer_group], op=dist.ReduceOp.SUM)
        result_bytes = tensor.numpy().tobytes()
        gathered_bytes = [None] * world_size
        dist.all_gather_object(gathered_bytes, result_bytes)
        assert all(b == gathered_bytes[0] for b in gathered_bytes), "Not bitwise equal"

        # Step 4: empty groups = no-op
        tensor = torch.tensor([42.0])
        MultiPGUtil.all_reduce(tensor, [], op=dist.ReduceOp.SUM)
        assert tensor.item() == 42.0

        # Step 5: MAX op
        tensor = torch.tensor([float(rank + 1)])
        MultiPGUtil.all_reduce(tensor, [inner_group, outer_group], op=dist.ReduceOp.MAX)
        assert tensor.item() == 4.0
    finally:
        dist.destroy_process_group()


def test_multi_pg_util_all_reduce() -> None:
    # [FUNCTION DOC] Run the multi-group all-reduce worker in a four-process world.
    run_multiprocess(_worker_multi_pg_util_all_reduce, world_size=4)


def _worker_multi_pg_util_gather_object(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Per-rank process body on a 2 by 2 mesh. Gather over the inner group:
    # inner rank 0 receives two entries (ranks {0,1} or {2,3}), others get none. Gather over inner
    # and outer: only global rank 0 receives all four entries, others get none.
    init_gloo(rank, world_size, port=port)
    try:
        mesh = _make_mesh()
        inner_group = mesh.get_group("inner")
        outer_group = mesh.get_group("outer")

        # Step 1: single group gather
        result = MultiPGUtil.gather_object({"rank": rank}, [inner_group])
        inner_rank = rank % 2
        if inner_rank == 0:
            assert result is not None
            assert len(result) == 2
            ranks_gathered = {item["rank"] for item in result}
            if rank == 0:
                assert ranks_gathered == {0, 1}
            else:
                assert ranks_gathered == {2, 3}
        else:
            assert result is None

        # Step 2: two group gather — global rank 0 gets everything
        result = MultiPGUtil.gather_object({"rank": rank}, [inner_group, outer_group])
        if rank == 0:
            assert result is not None
            assert len(result) == 4
            assert {item["rank"] for item in result} == {0, 1, 2, 3}
        else:
            assert result is None
    finally:
        dist.destroy_process_group()


def test_multi_pg_util_gather_object() -> None:
    # [FUNCTION DOC] Run the multi-group gather-object worker in a four-process world.
    run_multiprocess(_worker_multi_pg_util_gather_object, world_size=4)


# -- _check_wait tests --


class TestCheckWait:
    # [CLASS DOC] Tests of the wait-result checker.
    def test_raises_on_false(self) -> None:
        # [FUNCTION DOC] A work whose wait returns false makes the checker raise a runtime error
        # saying the distributed operation (named allreduce) failed.
        work = MagicMock()
        work.wait.return_value = False

        with pytest.raises(RuntimeError, match="distributed operation allreduce failed"):
            _check_wait(work, "allreduce")

    def test_passes_on_true(self) -> None:
        # [FUNCTION DOC] A work whose wait returns true passes silently.
        work = MagicMock()
        work.wait.return_value = True

        _check_wait(work, "allreduce")

    def test_propagates_exception_from_wait(self) -> None:
        # [FUNCTION DOC] An error raised by wait ("NCCL timeout") propagates unchanged.
        work = MagicMock()
        work.wait.side_effect = RuntimeError("NCCL timeout")

        with pytest.raises(RuntimeError, match="NCCL timeout"):
            _check_wait(work, "allreduce")


def _worker_bool_and_all_true(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Both ranks contribute true; expected result true.
    _worker_bool_and(rank, world_size, port, value_by_rank={0: True, 1: True}, expected=True)


def _worker_bool_and_all_false(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Both ranks contribute false; expected result false.
    _worker_bool_and(rank, world_size, port, value_by_rank={0: False, 1: False}, expected=False)


def _worker_bool_and_mixed(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Rank 0 true, rank 1 false; expected result false.
    _worker_bool_and(rank, world_size, port, value_by_rank={0: True, 1: False}, expected=False)


def _worker_bool_and(rank: int, world_size: int, port: int, *, value_by_rank: dict[int, bool], expected: bool) -> None:
    # [FUNCTION DOC] HELPER: Init gloo, create a group of all ranks, run the collective boolean-and
    # with this rank's value and assert the returned value is exactly the expected boolean. The
    # value mapping assumes a two-rank world. Destroys the process group at the end.
    init_gloo(rank, world_size, port=port)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        result = collective_bool_and(value=value_by_rank[rank], group=group)
        assert result is expected, f"rank {rank}: expected {expected}, got {result}"
    finally:
        dist.destroy_process_group()


class TestCollectiveBoolAnd:
    # [CLASS DOC] Multi-process tests of the collective boolean-and (default world size).
    def test_all_true(self) -> None:
        # [FUNCTION DOC] All ranks true gives true.
        run_multiprocess(_worker_bool_and_all_true)

    def test_all_false(self) -> None:
        # [FUNCTION DOC] All ranks false gives false.
        run_multiprocess(_worker_bool_and_all_false)

    def test_mixed_returns_false(self) -> None:
        # [FUNCTION DOC] A mix of true and false gives false.
        run_multiprocess(_worker_bool_and_mixed)
