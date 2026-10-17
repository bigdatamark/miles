# [MODULE DOC]
# NCCL broadcast weight transfer for non-colocated engines. A protocol class owns the sender decision
# and the lifecycle of one NCCL group per shard (sender training rank plus all engine GPUs); module
# functions create/destroy that group via HTTP-driven engine calls, and broadcast buckets after
# sending tensor metadata over HTTP. Optionally packs a bucket into one flattened byte tensor.
import socket
from argparse import Namespace
from collections.abc import Sequence
from concurrent.futures import Future
from contextlib import AbstractContextManager, nullcontext

import ray
import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState, get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
from miles.utils import async_utils
from miles.utils.distributed_lock import create_world_ticket_lock
from miles.utils.distributed_utils import init_process_group


class UpdateWeightFromDistributed(WeightTransferProtocol):
    # [CLASS DOC] Broadcast protocol; supports LoRA. Uses the base default required placement.
    # Sender: the rank with data-replica rank 0; one NCCL group named per shard (pipeline rank, or 0
    # when pipeline is gathered). With more than one pipeline stage a world-wide ticket lock
    # serializes broadcasts across stages (participants are intra-DP x CP rank 0 and TP rank 0
    # processes) to prevent NCCL deadlock; otherwise a no-op context.
    """
    Update distributed engines via NCCL. Each PP rank: group "miles-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    """

    supports_lora = True

    def __init__(self, args: Namespace) -> None:
        # [FUNCTION DOC] Initialize the base, no NCCL group yet, and build the engine lock from the
        # global parallel state as described at class level (ticket lock with prefix for weight
        # update when pp size > 1).
        super().__init__(args)
        self._model_update_groups = None
        parallel_state = get_parallel_state()
        self._engine_lock: AbstractContextManager = (
            create_world_ticket_lock(
                prefix="miles/weight_update",
                participates=parallel_state.intra_dp_cp.rank == 0 and parallel_state.tp.rank == 0,
            )
            if parallel_state.pp.size > 1
            else nullcontext()
        )

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        # [FUNCTION DOC] Store engines, selector and engine GPU counts. The data-replica rank comes
        # from the shared helper; this rank is the sender iff it is 0. Shard = 0 if pipeline is
        # gathered else the pipeline rank; the group name becomes "miles-pp_<shard>". On senders:
        # first tears down any previous group (on engines and locally), then creates a new NCCL group
        # with all engine GPUs (blocks until joined). Non-senders do nothing more. KNOWN ISSUE: the
        # docstring says the source is DP=TP=0 and mentions the lock; code decides by replica rank
        # and the lock is created in the constructor; engine_gpu_offsets is unused.
        """
        Create NCCL "miles-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self._selector = selector
        self._engine_gpu_counts = engine_gpu_counts

        # One sender per replica set; one NCCL group (sender + all engines) per shard.
        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0
        shard = 0 if placement.gather_pp else parallel_state.pp.rank
        if self.is_sender:
            self.group_name = f"miles-pp_{shard}"
            disconnect_rollout_engines_from_distributed(
                self.args, self.group_name, self._model_update_groups, self.rollout_engines
            )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self.group_name, rollout_engines, engine_gpu_counts=engine_gpu_counts
            )

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        # [FUNCTION DOC] Under the engine lock: send metadata and broadcast the bucket's tensors to
        # the engines using the configured transfer mode (default broadcast), wait for the engines'
        # HTTP futures, then clear the caller's bucket list in place to free memory.
        """Lock → broadcast → clear → unlock. Lock prevents NCCL deadlock."""
        with self._engine_lock:
            futures = update_weights_from_distributed(
                self.group_name,
                self._model_update_groups,
                self.rollout_engines,
                bucket,
                selector=self._selector,
                transfer_mode=getattr(self.args, "update_weight_transfer_mode", "broadcast"),
            )
            async_utils.wait_futures(futures)
            bucket.clear()


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[SGLangApiClient],
    engine_gpu_counts: Sequence[int] | None = None,
) -> dist.ProcessGroup:
    # [FUNCTION DOC] Create the NCCL group in which the calling training process is rank 0 and every
    # engine GPU gets a consecutive rank. GPU counts default to the configured GPUs per engine for
    # every engine. Master address is this node's IP (from Ray); the master port is a free port found
    # by binding and releasing a socket (small race window). World size = total engine GPUs + 1.
    # Submits an async init call to each engine client (rank offset starts at 1 and advances by that
    # engine's GPU count, backend nccl) and concurrently initializes the local process group over a
    # TCP rendezvous, then waits for all engine futures (raising their errors). Returns the group.
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1

    futures = []
    rank_cursor = 1
    for i, api_client in enumerate(rollout_engines):
        futures.append(
            async_utils.submit(
                api_client.init_weights_update_group(
                    master_address,
                    master_port,
                    rank_cursor,
                    world_size,
                    group_name,
                    backend="nccl",
                )
            )
        )
        rank_cursor += engine_gpu_counts[i]
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    async_utils.wait_futures(futures)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    # [FUNCTION DOC] Ask every engine client to destroy the named weight update group (async), then
    # destroy the local process group if one exists; in all cases (finally) wait for the engine
    # futures. KNOWN ISSUE: the args parameter is unused.
    """
    Destroy NCCL on training and engines.
    """
    futures = [async_utils.submit(client.destroy_weights_update_group(group_name)) for client in rollout_engines]
    try:
        if model_update_groups is not None:
            dist.destroy_process_group(model_update_groups)
    finally:
        async_utils.wait_futures(futures)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    rollout_engines: Sequence[SGLangApiClient],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    selector: str = "all",
    transfer_mode: str = "broadcast",
) -> list[Future]:
    # [FUNCTION DOC] Raises a value error unless the transfer mode is broadcast or broadcast_packed.
    # First prepares the payload (before engines are told to receive, so local failures cannot leave
    # them waiting): packed mode builds one flattened byte tensor of the whole bucket and tells the
    # engines to use the flattened-bucket load format; plain mode makes each tensor contiguous (using
    # the parameter data). Then submits to every engine, asynchronously, an HTTP call with names,
    # dtypes, shapes, selector and group name. Then issues asynchronous NCCL broadcasts from rank 0
    # of the group for each payload tensor and waits for them to complete. Returns the engine
    # futures, which the caller must wait on. Collective on the group.
    """
    Send metadata (HTTP), broadcast tensors (NCCL rank 0 → engines).
    """
    if transfer_mode not in ("broadcast", "broadcast_packed"):
        raise ValueError(f"Expected a broadcast transfer mode, got {transfer_mode!r}")
    # Pack before asking receivers to enter their collective, so allocation or
    # format validation failures cannot leave them waiting for a broadcast.
    if transfer_mode == "broadcast_packed":
        tensors = [_flatten_weight_bucket(converted_named_tensors)]
        format_kwargs = {"load_format": "flattened_bucket"}
    else:
        tensors = [param.data.contiguous() for _, param in converted_named_tensors]
        format_kwargs = {}

    futures = [
        async_utils.submit(
            client.update_weights_from_distributed(
                names=[name for name, _ in converted_named_tensors],
                dtypes=[param.dtype for _, param in converted_named_tensors],
                shapes=[param.shape for _, param in converted_named_tensors],
                selector=selector,
                group_name=group_name,
                **format_kwargs,
            )
        )
        for client in rollout_engines
    ]

    handles = []
    for tensor in tensors:
        handles.append(dist.broadcast(tensor, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return futures


def _flatten_weight_bucket(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Pack the bucket into SGLang's flattened byte tensor, importing the
    # bucket class lazily. Raises a runtime error if that class lacks mixed-dtype support. Because
    # the engine concatenates raw bytes without padding, simulates running byte offsets and raises a
    # value error naming the first tensor whose offset is not a multiple of its element size. Names
    # and order are preserved; tensors are made contiguous.
    # Keep SGLang optional for callers using the existing per-tensor protocol.
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    if not getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        raise RuntimeError("Flattened broadcasts require SGLang's mixed-dtype FlattenedTensorBucket")

    # SGLang concatenates raw bytes without padding. Each reconstructed dtype
    # view requires an aligned offset; preserve names/order and reject unsafe
    # layouts before allocating or sending metadata instead of casting values.
    offset = 0
    for name, tensor in named_tensors:
        if offset % tensor.element_size():
            raise ValueError(f"Unaligned flattened weight {name}: byte offset {offset}, dtype {tensor.dtype}")
        offset += tensor.numel() * tensor.element_size()
    return FlattenedTensorBucket(
        named_tensors=[(name, tensor.contiguous()) for name, tensor in named_tensors]
    ).get_flattened_tensor()
