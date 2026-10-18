# [MODULE DOC]
# CPU tests for the CUDA IPC (colocated tensor) weight update protocol: how its connect step splits
# rollout engines into colocated (IPC) and remote (distributed) groups from per-engine GPU counts
# and sparse GPU offsets, and how colocated buckets are sent to the engine serially in order. The
# distributed module, bucket class, serializer and engine are replaced by fakes defined here.
import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.cuda_ipc import (
    UpdateWeightFromTensor,
    _send_to_colocated_engine,
)
from miles.utils import async_utils

_TENSOR_MODULE = "miles.backends.training_utils.weight_update.protocols.cuda_ipc"

_ENGINE_GPU_COUNTS = [2, 2, 2, 2]
_SPARSE_GPU_OFFSETS = [0, 2, 6, 8]


class _FakeFlattenedTensorBucket:
    # [CLASS DOC] HELPER: Stand-in for the flattened tensor bucket that does not support mixed
    # dtypes (so the protocol builds one bucket per dtype). It keeps the named tensors it was given.
    supports_multi_dtypes = False

    def __init__(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        # [FUNCTION DOC] HELPER: Stores the list of name and tensor pairs.
        self._named_tensors = named_tensors

    def get_flattened_tensor(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the name of the first tensor (a string, not a tensor) so the
        # serializer fake can embed it and tests can see bucket order.
        return self._named_tensors[0][0]

    def get_metadata(self) -> None:
        # [FUNCTION DOC] HELPER: Returns no metadata.
        return None


class _FakeMultiprocessingSerializer:
    # [CLASS DOC] HELPER: Stand-in for the multiprocessing serializer.
    @staticmethod
    def serialize(data: dict, output_str: bool = False) -> str:
        # [FUNCTION DOC] HELPER: Returns a string made of the prefix "serialized:" followed by the
        # flattened tensor entry of the given mapping. The output_str flag is ignored.
        return f"serialized:{data['flattened_tensor']}"


class _SerialProbeEngine:
    # [CLASS DOC] HELPER: Fake colocated engine that records received payloads and measures the
    # maximum number of concurrent update calls in flight (attribute max_active).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the received list, the in-flight counter and its maximum.
        self.received: list[str] = []
        self.active = 0
        self.max_active = 0

    async def update_weights_from_tensor(self, **kwargs) -> dict:
        # [FUNCTION DOC] HELPER: Async fake of the engine update call. Increments the in-flight
        # counter and updates its maximum, records the first serialized named tensor payload, sleeps
        # ten milliseconds to let overlapping calls show up, decrements the counter and returns a
        # success mapping.
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.received.append(kwargs["serialized_named_tensors"][0])
        await asyncio.sleep(0.01)
        self.active -= 1
        return {"success": True}


def _gather_object_onto_source(obj, object_gather_list=None, dst=None, group=None) -> None:
    # [FUNCTION DOC] HELPER: Fake single-process object gather. Writes the local object into the
    # first slot of the destination list; the destination rank and group are ignored.
    object_gather_list[0] = obj


class TestConnect:
    # [CLASS DOC] Tests of the protocol's connect step with four engines of two GPUs each at sparse
    # GPU offsets zero, two, six and eight (a gap at GPUs four and five). The first three engines
    # are colocated; the last one is remote and reached through distributed groups.
    @staticmethod
    def _make_protocol() -> UpdateWeightFromTensor:
        # [FUNCTION DOC] HELPER: Builds a protocol object bypassing its constructor, with args for
        # two GPUs per engine, one actor node of eight GPUs, colocate and rollout offload enabled,
        # no engines or update groups yet, and a mock gather group from initialization with source
        # rank zero. Returns it.
        protocol = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
        protocol.args = Namespace(
            rollout_num_gpus_per_engine=2,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            check_weight_update_equal=False,
            colocate=True,
            offload_rollout=True,
        )
        protocol.rollout_engines = None
        protocol._model_update_groups = None
        protocol._ipc_gather_group = MagicMock(name="gather_group_from_init")
        protocol._ipc_gather_src = 0
        return protocol

    @staticmethod
    def _connect(
        protocol: UpdateWeightFromTensor, engines: list, *, rank: int, tp_rank: int
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        # [FUNCTION DOC] HELPER: Runs the protocol's connect with the given engines, the module-level
        # GPU counts and offsets, a parallel state whose data-parallel rank and pipeline rank are
        # zero and whose tensor-parallel rank is tp_rank, a placement that gathers over pipeline
        # stages, and selector "all". The distributed module, LoRA check (disabled) and the connect
        # and disconnect helpers are patched, and the mocked world rank is set to rank. Returns the
        # mocked distributed module, the connect mock and the disconnect mock.
        parallel_state = SimpleNamespace(
            intra_dp_cp=SimpleNamespace(rank=0),
            tp=SimpleNamespace(rank=tp_rank),
            pp=SimpleNamespace(rank=0),
        )
        with (
            patch(f"{_TENSOR_MODULE}.dist") as dist_mock,
            patch(f"{_TENSOR_MODULE}.lora_rollout_enabled", return_value=False),
            patch(f"{_TENSOR_MODULE}.connect_rollout_engines_from_distributed") as connect,
            patch(f"{_TENSOR_MODULE}.disconnect_rollout_engines_from_distributed") as disconnect,
        ):
            dist_mock.get_rank.return_value = rank
            protocol.connect(
                engines,
                engine_gpu_counts=_ENGINE_GPU_COUNTS,
                engine_gpu_offsets=_SPARSE_GPU_OFFSETS,
                parallel_state=parallel_state,
                placement=SimpleNamespace(gather_pp=True),
                selector="all",
            )
        return dist_mock, connect, disconnect

    @pytest.mark.parametrize(
        ("rank", "expected_engine_index", "expected_gather_src"),
        [(0, 0, 0), (6, 2, 6)],
        ids=["first_engine", "engine_after_sparse_gap"],
    )
    def test_connect_splits_clients_and_maps_sparse_gpu_offsets(
        self, rank: int, expected_engine_index: int, expected_gather_src: int
    ) -> None:
        # [FUNCTION DOC] Parametrized over world rank zero (engine zero, gather source zero) and rank
        # six (engine two, gather source six, after the sparse gap). Asserts after connect: the
        # distributed-transfer flag is on; the colocated engines are the first three and the
        # distributed engines the last one; the connect helper was called once with the args, the
        # group name "miles", the remote engines and a GPU count list of two, and the disconnect
        # helper never; the update groups are the connect helper's result; the IPC engine is the
        # expected covering engine; the gather group is unchanged and the gather source equals the
        # expected value; the rank is a sender; and no new process group was created.
        """Only remote engines use NCCL, while ranks on either side of a sparse gap map to their covered colocated engine."""
        engines = [MagicMock(name=f"engine{i}") for i in range(4)]
        protocol = self._make_protocol()
        protocol._ipc_gather_src = expected_gather_src
        gather_group_from_init = protocol._ipc_gather_group

        dist_mock, connect, disconnect = self._connect(protocol, engines, rank=rank, tp_rank=0)

        assert protocol.use_distribute is True
        assert protocol.rollout_engines == engines[:3]
        assert protocol.distributed_rollout_engines == engines[3:]
        connect.assert_called_once_with(protocol.args, "miles", engines[3:], engine_gpu_counts=[2])
        disconnect.assert_not_called()
        assert protocol._model_update_groups is connect.return_value
        assert protocol._ipc_engine is engines[expected_engine_index]
        assert protocol._ipc_gather_group is gather_group_from_init
        assert protocol._ipc_gather_src == expected_gather_src
        assert protocol.is_sender is True
        dist_mock.new_group.assert_not_called()

    def test_connect_clears_ipc_state_for_a_placeholder_rank(self) -> None:
        # [FUNCTION DOC] Connecting from world rank four (in the sparse gap, tensor-parallel rank one)
        # must clear the gather group, gather source and IPC engine to none, mark the rank a
        # non-sender, and not call the distributed connect helper.
        """A reserved GPU slot that no engine covers drops its stale gather group, or it would enter a collective nobody else joins."""
        engines = [MagicMock(name=f"engine{i}") for i in range(4)]
        protocol = self._make_protocol()

        _dist_mock, connect, _disconnect = self._connect(protocol, engines, rank=4, tp_rank=1)

        assert protocol._ipc_gather_group is None
        assert protocol._ipc_gather_src is None
        assert protocol._ipc_engine is None
        assert protocol.is_sender is False
        connect.assert_not_called()


class TestColocatedBaseSend:
    # [CLASS DOC] Tests of the function that sends named tensors to the colocated engine.
    def test_multiple_colocated_buckets_are_sent_serially_in_source_order(self) -> None:
        # [FUNCTION DOC] Sends a float32 and a float16 tensor through the colocated send function with
        # a probe engine, a single-process world (rank zero, size one, fake gather), the fake bucket
        # and fake serializer patched in, then waits for the returned futures. Asserts the engine
        # received the two serialized payloads in source order and that at most one update call was
        # ever in flight (serial sending).
        """Two dtype buckets reach the engine one at a time and in the order they were built."""
        engine = _SerialProbeEngine()
        named_tensors = [
            ("model.layers.0.mlp.gate_proj.weight", torch.zeros(2, dtype=torch.float32)),
            ("model.layers.0.mlp.up_proj.weight", torch.zeros(2, dtype=torch.float16)),
        ]

        with (
            patch(f"{_TENSOR_MODULE}.dist") as dist_mock,
            patch(f"{_TENSOR_MODULE}.FlattenedTensorBucket", _FakeFlattenedTensorBucket),
            patch(f"{_TENSOR_MODULE}.MultiprocessingSerializer", _FakeMultiprocessingSerializer),
        ):
            dist_mock.get_rank.return_value = 0
            dist_mock.get_world_size.return_value = 1
            dist_mock.gather_object.side_effect = _gather_object_onto_source
            futures, _long_lived_tensors = _send_to_colocated_engine(
                named_tensors,
                ipc_engine=engine,
                ipc_gather_src=0,
                ipc_gather_group=MagicMock(name="gather_group"),
            )
            async_utils.wait_futures(futures)

        assert engine.received == [
            "serialized:model.layers.0.mlp.gate_proj.weight",
            "serialized:model.layers.0.mlp.up_proj.weight",
        ]
        assert engine.max_active == 1
