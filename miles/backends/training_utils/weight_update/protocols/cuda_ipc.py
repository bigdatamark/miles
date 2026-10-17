# [MODULE DOC]
# Colocated weight transfer: training and inference engines share GPUs, so each training rank
# serializes its bucket as CUDA-IPC handles of a flattened tensor bucket, gathers them (Gloo
# object gather) to its engine's source rank, which hands them to the engine over HTTP. Hybrid
# deployments with extra non-colocated engines also broadcast over NCCL via the broadcast module.
import asyncio
import logging
from argparse import Namespace
from collections.abc import Sequence
from concurrent.futures import Future
from typing import Any

import torch
import torch.distributed as dist
from sglang.srt.utils import MultiprocessingSerializer

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.utils import async_utils
from miles.utils.lora.utils import lora_base_cpu_backup_enabled, lora_rollout_enabled

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

from .broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


class UpdateWeightFromTensor(WeightTransferProtocol):
    # [CLASS DOC] Colocated CUDA-IPC protocol. Requires a full gather over pipeline (tensor and
    # expert always gathered), supports LoRA (not for hybrid deployments). Every rank covered by a
    # colocated engine is a sender, because the per-engine object gather is collective among the
    # group's members. State: per-rank Gloo gather group and its source rank, the engine client
    # for this rank's colocated engine, optional distributed tail engines, the NCCL group and
    # source-rank flag for the tail, and whether the base weights must be re-pushed for LoRA.
    """
    Colocated transfer: every training rank serializes its bucket to CUDA-IPC
    handles and gather_objects them to its engine group's src rank, which hands
    them to the engine over Ray. Hybrid deployments broadcast to the
    non-colocated engine tail over NCCL from the global source rank.
    """

    required_placement = WeightUpdatePlacement(gather_pp=True)
    supports_lora = True

    def __init__(self, args: Namespace) -> None:
        # [FUNCTION DOC] Collective over the whole world. Sets the default group name
        # "miles-colocate" (overwritten with "miles" when a distributed tail exists). Partitions all
        # ranks into consecutive blocks of the configured GPUs per engine and creates a Gloo group
        # for every block (every rank must create every group); this rank remembers the group it
        # belongs to and the block's first rank as gather source. KNOWN ISSUE: the engine ids in
        # connect later assume possibly different layouts; these initial groups are recreated only
        # for ranks without an engine.
        """
        Create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        super().__init__(args)
        self._model_update_groups = None
        # Overwritten with "miles" when connect finds a distributed engine tail.
        self.group_name = "miles-colocate"

        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = min(start_rank + self.args.rollout_num_gpus_per_engine, dist.get_world_size())
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        # [FUNCTION DOC] Steps: (1) default engine GPU counts to the configured per-engine value and
        # offsets to densely packed cumulative counts. (2) Count colocated engines: leading engines
        # whose GPU range lies within the actor GPU total (nodes * GPUs per node); stop at the first
        # one that does not. (3) A distributed tail exists if there are more engines than colocated
        # ones; combined with LoRA rollout this raises a not-implemented error. With a tail, split the
        # engine list, set the group name to "miles", and on the distributed source rank (intra-DP x
        # CP, TP and PP ranks all 0) tear down any old NCCL group and create a new one to the tail
        # engines. (4) Compute the set of ranks covered by colocated engines; covered ranks create
        # a Gloo group per engine only if they have no group yet (groups are created by every covered
        # rank for all engines, a collective); uncovered ranks clear their gather group and source.
        # (5) Find this rank's colocated engine client by GPU range. (6) Sender = has a gather group.
        # (7) Base re-push for LoRA is needed if the weight checker option is on, or base weights do
        # not persist; they persist when there is a distributed tail, or LoRA CPU backup is enabled,
        # or colocated with rollout offload disabled.
        # KNOWN ISSUE: the first-time group recreation test relies on a group from the constructor
        # being none, but the constructor assigns a group for every rank, so in practice it never
        # recreates for covered ranks (matching layouts are assumed); also the uncovered branch
        # discards a group that was created collectively in the constructor.
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines
        self._selector = selector

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums
        if self.use_distribute and lora_rollout_enabled(self.args):
            raise NotImplementedError("LoRA weight sync is not supported for hybrid colocated+distributed deployments")

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                parallel_state.intra_dp_cp.rank == 0 and parallel_state.tp.rank == 0 and parallel_state.pp.rank == 0
            )
            self.group_name = "miles"
            if self._is_distributed_src_rank:
                if (g := self._model_update_groups) is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self.group_name, g, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self.group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Determine whether this rank is covered by any colocated engine.
        all_colocated_ranks = set()
        for offset, count in zip(colocate_gpu_offsets, colocate_gpu_counts, strict=True):
            all_colocated_ranks.update(range(offset, offset + count))
        rank_has_engine = dist.get_rank() in all_colocated_ranks

        # Create IPC Gloo gather groups matching actual engine layout.
        # Re-create on first call or when engine layout changes (placeholder ranks
        # that had a group from __init__ but no actual engine need to be reset).
        if rank_has_engine:
            if self._ipc_gather_group is None:
                for i in range(colocate_engine_nums):
                    group_ranks = list(
                        range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i])
                    )
                    new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                    if dist.get_rank() in group_ranks:
                        self._ipc_gather_group = new_group
                        self._ipc_gather_src = colocate_gpu_offsets[i]
        else:
            # Ranks not covered by any engine (e.g. placeholder GPU slots)
            self._ipc_gather_group = None
            self._ipc_gather_src = None

        # Map training ranks to colocated engine actors.
        self._ipc_engine = None
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

        # Every engine-covered rank sends: the per-engine gather_object is
        # collective among the group's members.
        self.is_sender = self._ipc_gather_group is not None

        # A LoRA sync must re-push the frozen base unless the engines keep it
        # across pauses (CPU backup, persistent GPU copy, or remote engines);
        # the weight checker always needs the full rewrite.
        base_persists = (
            self.use_distribute
            or lora_base_cpu_backup_enabled(self.args)
            or (self.args.colocate and not self.args.offload_rollout)
        )
        self.needs_base_resync_for_lora = self.args.check_weight_update_equal or not base_persists

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        # [FUNCTION DOC] Sends the bucket to this rank's colocated engine via the IPC helper (a
        # collective among the gather group). On the distributed source rank of a hybrid
        # deployment, also broadcasts the bucket to the tail engines and merges the futures. Waits
        # for all futures and checks the engine responses (the check is told the sync is not LoRA).
        # Finally drops the references that kept the IPC tensors alive. KNOWN ISSUE: the LoRA flag
        # for the response check is hard-coded false although this protocol supports LoRA.
        futures, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors=bucket,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            selector=self._selector,
        )
        if self.use_distribute and self._is_distributed_src_rank:
            futures_distributed = update_weights_from_distributed(
                self.group_name,
                self._model_update_groups,
                self.distributed_rollout_engines,
                bucket,
                selector=self._selector,
            )
            if futures_distributed:
                futures = (futures or []) + futures_distributed
        check_weight_sync_results(async_utils.wait_futures(futures or []), is_lora=False)
        del long_lived_tensors

    def after_engines_resumed(self) -> None:
        # [FUNCTION DOC] Release CUDA IPC resources: collect expired IPC handles, then empty the
        # CUDA caching allocator cache.
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    selector: str = "all",
) -> tuple[list[Future], Any]:
    # [FUNCTION DOC] HELPER: Keyword-only after the tensor list. Returns (futures, objects that must
    # stay alive until the engine has copied). Ranks without a gather group return empty results
    # immediately (they skip the collective). Otherwise: if the bucket class supports mixed dtypes,
    # use one flattened bucket for all tensors, else group tensors by dtype into one bucket per
    # dtype. For each bucket, flatten to a single tensor with metadata, keep it alive, and serialize
    # it (handles, as a string). Gather all ranks' serialized lists to the group's source rank
    # (Gloo object gather, collective among group members). On the source rank only, for each dtype
    # index, submit an async call to the engine with the serialized tensors of all ranks, the
    # flattened-bucket load format and the selector, serialized through one lock so engine calls
    # for different dtypes do not interleave; the futures are returned. Non-source ranks return no
    # futures.
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    is_gather_src = dist.get_rank() == ipc_gather_src
    long_live_tensors = []

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors: list = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": flattened_tensor_bucket.get_metadata(),
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    serialized_named_tensors = [None] * dist.get_world_size(ipc_gather_group) if is_gather_src else None
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    futures: list[Future] = []
    if is_gather_src:
        num_dtypes = len(serialized_named_tensors[0])
        ipc_gate = asyncio.Lock()
        for i in range(num_dtypes):
            kwargs = {
                "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                "load_format": "flattened_bucket",
                "selector": selector,
            }
            futures.append(async_utils.submit(_update_weights_from_tensor_gated(ipc_gate, ipc_engine, kwargs)))

    return futures, long_live_tensors


async def _update_weights_from_tensor_gated(
    gate: asyncio.Lock, api_client: SGLangApiClient, kwargs: dict[str, Any]
) -> Any:
    # [FUNCTION DOC] HELPER: Coroutine that acquires the lock, awaits the engine client's update from
    # tensor call with the keyword arguments, and returns its response.
    async with gate:
        return await api_client.update_weights_from_tensor(**kwargs)
