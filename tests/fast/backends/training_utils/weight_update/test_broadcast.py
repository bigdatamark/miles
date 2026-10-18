# [MODULE DOC] CPU tests for the distributed-broadcast weight update protocol (connect, disconnect,
# update) with Ray, process group initialization and torch distributed patched. Engines are fakes:
# gated engines record their calls and block until released (to prove ordering), accepting and
# refusing engines for group init, and slow-teardown engines for disconnect. Verified: all engines
# are asked to join their group at their own rank offsets before rank 0 blocks on the handshake
# (heterogeneous gpu counts, with a uniform fallback when counts are absent); a refusing engine fails
# the connect; the protocol forwards discovered gpu counts to the group builder; disconnect awaits
# engine-side teardown even when local destroy fails; and a weight update has every engine's metadata
# call in flight before the first tensor broadcast, stages tensors contiguously, and waits all handles.
import asyncio
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.broadcast import (
    UpdateWeightFromDistributed,
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)
from miles.utils import async_utils

_BROADCAST_MODULE = "miles.backends.training_utils.weight_update.protocols.broadcast"

_BASE_NAMED_TENSORS = [
    ("model.layers.0.mlp.gate_proj.weight", torch.zeros(2, 3)),
    ("model.layers.0.mlp.up_proj.weight", torch.arange(6, dtype=torch.float32).reshape(2, 3).t()),
]


class _GatedEngine:
    # [CLASS DOC] HELPER: Fake rollout engine whose every attribute is an async method that records
    # (name, args, kwargs), signals a shared semaphore that it started, then waits (up to 5 seconds,
    # in a worker thread) for a shared release event; if the wait times out it raises an assertion
    # error saying the caller waited for the engine before reaching the collective; otherwise returns
    # a success dictionary.
    def __init__(self, started: threading.Semaphore, release: threading.Event) -> None:
        # [FUNCTION DOC] HELPER: Stores the shared semaphore and event and an empty call list.
        self.calls: list[tuple[str, tuple, dict]] = []
        self._started = started
        self._release = release

    def __getattr__(self, name: str):
        # [FUNCTION DOC] HELPER: Returns an async method for any requested attribute name (see
        # class description for what it does).
        async def method(*args, **kwargs):
            # [FUNCTION DOC] HELPER: The fake engine call body described in the class description.
            self.calls.append((name, args, kwargs))
            self._started.release()
            released = await asyncio.get_running_loop().run_in_executor(None, self._release.wait, 5)
            if not released:
                raise AssertionError("the caller waited for an engine before reaching the collective")
            return {"success": True}

        return method


class _AcceptingEngine:
    # [CLASS DOC] HELPER: Fake engine that accepts joining a weight update group.
    async def init_weights_update_group(self, *args, **kwargs) -> dict:
        # [FUNCTION DOC] HELPER: Returns a success dictionary regardless of arguments.
        return {"success": True}


class _RefusingEngine:
    # [CLASS DOC] HELPER: Fake engine that refuses joining a weight update group.
    async def init_weights_update_group(self, *args, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Raises a runtime error "engine refused the group".
        raise RuntimeError("engine refused the group")


class _SlowTeardownEngine:
    # [CLASS DOC] HELPER: Fake engine whose group teardown takes 50 ms and records completion.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with no recorded calls and finished false.
        self.calls: list[tuple[str, str]] = []
        self.finished = False

    async def destroy_weights_update_group(self, group_name: str) -> dict:
        # [FUNCTION DOC] HELPER: Records the call with the group name, sleeps 0.05 seconds, marks
        # finished, returns a success dictionary.
        self.calls.append(("destroy_weights_update_group", group_name))
        await asyncio.sleep(0.05)
        self.finished = True
        return {"success": True}


class _RecordingHandle:
    # [CLASS DOC] HELPER: Fake asynchronous collective handle that records whether it was waited on.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with waited false.
        self.waited = False

    def wait(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the handle as waited.
        self.waited = True


class TestConnectRolloutEnginesFromDistributed:
    # [CLASS DOC] Tests for the function that builds the weight update process group with engines.
    def test_group_init_starts_all_engines_before_local_join_with_heterogeneous_offsets(self) -> None:
        # [FUNCTION DOC] Test: three gated engines with gpu counts [2,4,1]; Ray (node IP 10.0.0.1)
        # and process group initialization are patched, the latter with a join function that waits
        # until all engines have started their calls, releases them, and returns a mock group. The
        # connect returns that group; each engine saw exactly one group-init call with host,
        # master port, rank offset (1, 3, 7: trainer is rank 0, then cumulative gpu counts), world
        # size 8 and the nccl backend keyword; the local init was called with nccl, a tcp init
        # method on the same host and port, world size 8, rank 0 and the group name.
        """Every engine is asked to join, at its own rank offset, before rank zero blocks on the handshake."""
        started = threading.Semaphore(0)
        release = threading.Event()
        engines = [_GatedEngine(started, release) for _ in range(3)]
        group = MagicMock(name="nccl_group")

        def join(**kwargs):
            # [FUNCTION DOC] HELPER: Fake local group init; waits (30 s each) for every engine to
            # have started, asserting if one has not, then releases them and returns the mock group.
            for _ in engines:
                assert started.acquire(timeout=30), "an engine had not been asked before the local join"
            release.set()
            return group

        with (
            patch(f"{_BROADCAST_MODULE}.ray") as ray_mock,
            patch(f"{_BROADCAST_MODULE}.init_process_group", side_effect=join) as init_process_group,
        ):
            ray_mock._private.services.get_node_ip_address.return_value = "10.0.0.1"
            result = connect_rollout_engines_from_distributed(
                Namespace(),
                "miles-pp_0",
                engines,
                engine_gpu_counts=[2, 4, 1],
            )

        assert result is group
        master_port = engines[0].calls[0][1][1]
        assert [engine.calls for engine in engines] == [
            [("init_weights_update_group", ("10.0.0.1", master_port, 1, 8, "miles-pp_0"), {"backend": "nccl"})],
            [("init_weights_update_group", ("10.0.0.1", master_port, 3, 8, "miles-pp_0"), {"backend": "nccl"})],
            [("init_weights_update_group", ("10.0.0.1", master_port, 7, 8, "miles-pp_0"), {"backend": "nccl"})],
        ]
        assert init_process_group.call_args.kwargs == {
            "backend": "nccl",
            "init_method": f"tcp://10.0.0.1:{master_port}",
            "world_size": 8,
            "rank": 0,
            "group_name": "miles-pp_0",
        }

    def test_missing_gpu_counts_fall_back_to_the_uniform_engine_size(self) -> None:
        # [FUNCTION DOC] Test: same setup but no gpu counts passed and args saying 2 gpus per engine:
        # the three engines get rank offsets 1, 3, 5 and world size 7, and the local init uses world
        # size 7.
        """Callers without discovered counts retain the uniform engine layout."""
        started = threading.Semaphore(0)
        release = threading.Event()
        engines = [_GatedEngine(started, release) for _ in range(3)]
        group = MagicMock(name="nccl_group")

        def join(**kwargs):
            # [FUNCTION DOC] HELPER: Same fake local group init as in the previous test.
            for _ in engines:
                assert started.acquire(timeout=30), "an engine had not been asked before the local join"
            release.set()
            return group

        with (
            patch(f"{_BROADCAST_MODULE}.ray") as ray_mock,
            patch(f"{_BROADCAST_MODULE}.init_process_group", side_effect=join) as init_process_group,
        ):
            ray_mock._private.services.get_node_ip_address.return_value = "10.0.0.1"
            result = connect_rollout_engines_from_distributed(
                Namespace(rollout_num_gpus_per_engine=2),
                "miles-pp_0",
                engines,
            )

        assert result is group
        master_port = engines[0].calls[0][1][1]
        assert [engine.calls for engine in engines] == [
            [("init_weights_update_group", ("10.0.0.1", master_port, 1, 7, "miles-pp_0"), {"backend": "nccl"})],
            [("init_weights_update_group", ("10.0.0.1", master_port, 3, 7, "miles-pp_0"), {"backend": "nccl"})],
            [("init_weights_update_group", ("10.0.0.1", master_port, 5, 7, "miles-pp_0"), {"backend": "nccl"})],
        ]
        assert init_process_group.call_args.kwargs["world_size"] == 7

    def test_an_engine_that_refuses_the_group_fails_the_connect(self) -> None:
        # [FUNCTION DOC] Test: two accepting engines and one refusing engine, with Ray and local group
        # init patched (init returns a mock group): connecting raises the refusing engine's runtime
        # error.
        """The submitted joins are awaited, so a refusing engine surfaces instead of being dropped."""
        with (
            patch(f"{_BROADCAST_MODULE}.ray") as ray_mock,
            patch(f"{_BROADCAST_MODULE}.init_process_group", return_value=MagicMock(name="nccl_group")),
        ):
            ray_mock._private.services.get_node_ip_address.return_value = "10.0.0.1"
            with pytest.raises(RuntimeError, match="engine refused the group"):
                connect_rollout_engines_from_distributed(
                    Namespace(rollout_num_gpus_per_engine=2),
                    "miles-pp_0",
                    [_AcceptingEngine(), _AcceptingEngine(), _RefusingEngine()],
                )


class TestUpdateWeightFromDistributedConnect:
    # [CLASS DOC] Tests for the protocol object's connect method.
    def test_the_protocol_forwards_discovered_gpu_counts_to_the_group_builder(self) -> None:
        # [FUNCTION DOC] Test: a protocol is constructed under a patched trivial parallel state (pp
        # size 1). Connecting two mock engines with gpu counts [2,4], offsets [0,2], a patched data
        # replica (0 of 1), disconnect and connect functions, placement not gathering pp and selector
        # "all" results in the group builder being called once with the protocol's args, group name
        # "miles-pp_0", the engines, and the gpu counts keyword [2,4] (not replaced by a uniform
        # value).
        """The protocol must not replace heterogeneous discovered counts with the uniform fallback."""
        initial_parallel_state = SimpleNamespace(
            pp=SimpleNamespace(size=1, rank=0),
            tp=SimpleNamespace(rank=0),
            intra_dp_cp=SimpleNamespace(rank=0),
        )
        with patch(f"{_BROADCAST_MODULE}.get_parallel_state", return_value=initial_parallel_state):
            protocol = UpdateWeightFromDistributed(Namespace())

        engines = [MagicMock(name="prefill"), MagicMock(name="decode")]
        parallel_state = SimpleNamespace(pp=SimpleNamespace(rank=0))
        with (
            patch(f"{_BROADCAST_MODULE}.get_data_replica_rank_and_size", return_value=(0, 1)),
            patch(f"{_BROADCAST_MODULE}.disconnect_rollout_engines_from_distributed"),
            patch(f"{_BROADCAST_MODULE}.connect_rollout_engines_from_distributed") as connect,
        ):
            protocol.connect(
                engines,
                engine_gpu_counts=[2, 4],
                engine_gpu_offsets=[0, 2],
                parallel_state=parallel_state,
                placement=SimpleNamespace(gather_pp=False),
                selector="all",
            )

        connect.assert_called_once_with(
            protocol.args,
            "miles-pp_0",
            engines,
            engine_gpu_counts=[2, 4],
        )


class TestDisconnectRolloutEnginesFromDistributed:
    # [CLASS DOC] Tests for the function that tears the weight update group down.
    def test_disconnect_awaits_engine_cleanup_when_local_destroy_fails(self) -> None:
        # [FUNCTION DOC] Test: with the distributed module patched so destroying the local process
        # group raises a runtime error "nccl teardown failed", disconnecting two slow-teardown
        # engines raises that error, yet both engines each received exactly one destroy call with
        # the group name and both finished their teardown (the disconnect waited for them).
        """A failed local teardown still surfaces, and it must not abandon the engine-side teardown."""
        engines = [_SlowTeardownEngine(), _SlowTeardownEngine()]

        with patch(f"{_BROADCAST_MODULE}.dist") as dist_mock:
            dist_mock.destroy_process_group.side_effect = RuntimeError("nccl teardown failed")
            with pytest.raises(RuntimeError, match="nccl teardown failed"):
                disconnect_rollout_engines_from_distributed(
                    Namespace(),
                    "miles-pp_0",
                    MagicMock(name="nccl_group"),
                    engines,
                )

        assert [engine.calls for engine in engines] == [
            [("destroy_weights_update_group", "miles-pp_0")],
            [("destroy_weights_update_group", "miles-pp_0")],
        ]
        assert [engine.finished for engine in engines] == [True, True]


class TestUpdateWeightsFromDistributed:
    # [CLASS DOC] Tests for the function that sends tensors to the engines by broadcast.
    def test_base_metadata_is_in_flight_before_broadcast_and_all_handles_finish(self) -> None:
        # [FUNCTION DOC] Test: three gated engines and two base named tensors (gate projection
        # zeros 2x3, up projection a transposed non-contiguous 3x2 view). The broadcast function is
        # replaced by a fake that records tensors and, at the first call, waits until every engine
        # has started its metadata call (then releases them), and returns recording handles. After
        # updating with selector "target" and waiting for the returned futures: each engine got
        # exactly one call whose names keyword lists both tensor names in order and selector is
        # "target"; both handles were waited; both broadcast tensors are contiguous; the second
        # broadcast tensor equals the (non-contiguous) source tensor in value.
        """No engine is still unasked when the first tensor goes out, every tensor is staged contiguously, and no handle is left unwaited."""
        started = threading.Semaphore(0)
        release = threading.Event()
        engines = [_GatedEngine(started, release) for _ in range(3)]
        handles: list[_RecordingHandle] = []
        broadcast_tensors: list[torch.Tensor] = []

        def broadcast(tensor, src, group=None, async_op=False):
            # [FUNCTION DOC] HELPER: Fake collective broadcast; records the tensor; on the first call
            # waits (30 s per engine) for all engines to have started, asserting otherwise, then
            # releases them; appends and returns a recording handle.
            broadcast_tensors.append(tensor)
            if len(broadcast_tensors) == 1:
                for _ in engines:
                    assert started.acquire(timeout=30), "an engine had not been asked before the first broadcast"
                release.set()
            handles.append(_RecordingHandle())
            return handles[-1]

        with patch(f"{_BROADCAST_MODULE}.dist") as dist_mock:
            dist_mock.broadcast.side_effect = broadcast
            futures = update_weights_from_distributed(
                "miles-pp_0",
                MagicMock(name="nccl_group"),
                engines,
                _BASE_NAMED_TENSORS,
                selector="target",
            )
            async_utils.wait_futures(futures)

        assert [len(engine.calls) for engine in engines] == [1, 1, 1]
        assert engines[0].calls[0][2]["names"] == [name for name, _ in _BASE_NAMED_TENSORS]
        assert engines[0].calls[0][2]["selector"] == "target"
        assert [handle.waited for handle in handles] == [True, True]
        assert [tensor.is_contiguous() for tensor in broadcast_tensors] == [True, True]
        assert torch.equal(broadcast_tensors[1], _BASE_NAMED_TENSORS[1][1])
