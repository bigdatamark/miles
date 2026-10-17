# [MODULE DOC]
# Sends and receives an in-memory Megatron checkpoint between training cells through the torchft
# process-group transport, so a freshly started cell can be healed from a healthy one. The sender
# snapshots model/optimizer/scheduler state to an in-memory state dict, encodes it so that tensors
# sharing one underlying storage are transmitted only once, and sends it; the receiver decodes it
# and wraps it in an in-memory checkpoint manager for Megatron to load. Public functions: send and
# receive. The torchft import is optional; its absence is reported only when a transport is built.
# Default timeout is ten minutes (must cover the receiver's model initialization time).
import logging
from collections.abc import Sequence
from datetime import timedelta

import torch

try:
    from torchft.checkpointing.pg_transport import PGTransport
except ImportError:
    PGTransport = None

from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict

from miles.backends.megatron_utils.ft.in_memory_checkpoint import InMemoryCheckpointManager, save_to_memory
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)

# Must accommodate receiver's model init time (can take minutes for large models)
_DEFAULT_TIMEOUT = timedelta(seconds=600)


def send_ckpt(
    *,
    indep_dp: GroupInfo,
    model: Sequence,
    optimizer: object,
    opt_param_scheduler: object,
    iteration: int,
    dst_rank: int,
    timeout: timedelta = _DEFAULT_TIMEOUT,
) -> None:
    # [FUNCTION DOC]
    # Sender side (keyword-only arguments). Steps: snapshot the training state to an in-memory
    # state dict for the iteration; encode it into a transport payload (deduplicated storages plus
    # view metadata); create a transport over the independent-DP process group (raises import
    # error if torchft is missing); log a structured "start" event; send the payload to the single
    # destination rank with step number 0 and the timeout; disallow further checkpoint serving on
    # the transport; log an "end" event. Blocks until sent or timeout. Uses CUDA/NCCL.
    """Send in-memory checkpoint to a destination cell via torchft PGTransport.

    Args:
        indep_dp: Independent DP group info (provides the torchft PG).
        model: Megatron model chunks.
        optimizer: Megatron optimizer.
        opt_param_scheduler: LR scheduler.
        iteration: Current training iteration / rollout_id.
        dst_rank: Destination alive_rank in the indep_dp process group.
        timeout: Timeout for the NCCL send operation.
    """
    state_dict = save_to_memory(
        iteration=iteration,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
    )

    payload = _TransportCodec.encode(state_dict=state_dict, iteration=iteration)

    transport = _create_transport(indep_dp, timeout)
    log_structured(
        logger.info,
        tag="ft",
        op="cross_cell",
        phase="start",
        kind="ckpt_send",
        iteration=iteration,
        to_alive_rank=dst_rank,
    )
    transport.send_checkpoint(
        dst_ranks=[dst_rank],
        step=0,
        state_dict=payload,
        timeout=timeout,
    )
    transport.disallow_checkpoint()
    log_structured(
        logger.info,
        tag="ft",
        op="cross_cell",
        phase="end",
        kind="ckpt_send",
        iteration=iteration,
        to_alive_rank=dst_rank,
    )


def recv_ckpt(
    *,
    indep_dp: GroupInfo,
    src_rank: int,
    timeout: timedelta = _DEFAULT_TIMEOUT,
) -> InMemoryCheckpointManager:
    # [FUNCTION DOC]
    # Receiver side (keyword-only arguments). Creates the transport, logs a start event, receives
    # the payload from the source rank (using the transport's own metadata and step 0), decodes it
    # to the iteration and state dict, logs an end event, and stores them in a new in-memory
    # checkpoint manager, which is returned. Requires the Megatron args to be set for in-memory
    # checkpointing (the manager constructor asserts). Raises import error if torchft is missing.
    """Receive checkpoint from a healthy cell via torchft PGTransport.

    Returns an InMemoryCheckpointManager containing the received state_dict,
    ready to be passed to initialize_model_and_optimizer.

    Args:
        indep_dp: Independent DP group info (provides the torchft PG).
        src_rank: Source alive_rank in the indep_dp process group.
        timeout: Timeout for the NCCL recv operation.

    Returns:
        InMemoryCheckpointManager with state_dict loaded, ready for
        initialize_model_and_optimizer to consume.
    """
    transport = _create_transport(indep_dp, timeout)
    log_structured(logger.info, tag="ft", op="cross_cell", phase="start", kind="ckpt_recv", from_alive_rank=src_rank)
    payload = transport.recv_checkpoint(
        src_rank=src_rank,
        metadata=transport.metadata(),
        step=0,
        timeout=timeout,
    )

    iteration, state_dict = _TransportCodec.decode(payload)
    log_structured(
        logger.info,
        tag="ft",
        op="cross_cell",
        phase="end",
        kind="ckpt_recv",
        iteration=iteration,
        from_alive_rank=src_rank,
    )

    manager = InMemoryCheckpointManager()
    manager.save(state_dict, iteration=iteration)
    return manager


class _TransportCodec:
    # [CLASS DOC] HELPER: Namespace (static methods only) converting between a tensor-aware state
    # dict plus iteration and a payload mapping suitable for the transport. The payload has four
    # entries: unique storages, view metadata, the hollow state dict (tensors removed) and the
    # iteration.
    @staticmethod
    def encode(
        *,
        state_dict: MCoreTensorAwareStateDict,
        iteration: int,
    ) -> dict[str, object]:
        # [FUNCTION DOC] HELPER: Pops all tensors out of the state dict (mutating it into a hollow
        # structure), converts any non-plain-tensor (e.g. parameter) objects to detached plain
        # tensors sharing storage (the transport rejects subclasses), encodes them with the tensor
        # view codec, and returns the payload mapping with keys for unique storages, view metas,
        # the hollow state dict and the iteration.
        """Serialize for transport, deduping by underlying storage.

        `pop_tensors()` returns ShardedTensor `.data` views. Many of those views share
        one big underlying storage (e.g. Megatron distributed-optimizer grad buckets).
        `torchft.PGTransport._cast_tensor` casts each tensor to a uint8 view of its
        FULL storage and sends that — so naively sending N views of one bucket sends
        bucket_size * N bytes (we measured ~110x amplification: 12.5 GB real data
        sent as 1387 GB).

        Fix: send each unique storage exactly once as `unique_storages`, plus per-view
        `view_metas` (storage_id, dtype, shape, stride, storage_offset) so the
        receiver can reconstruct the original views by `as_strided`.
        """
        tensors: list[torch.Tensor] = state_dict.pop_tensors()
        # PGTransport._cast_tensor uses `type(t) is torch.Tensor` (strict),
        # which rejects torch.nn.Parameter. Detach into plain Tensors that share
        # storage but pass the type check.
        tensors = [t.detach() if type(t) is not torch.Tensor else t for t in tensors]

        unique_storages, view_metas = _TensorViewCodec.encode(tensors)

        return {
            "unique_storages": unique_storages,
            "view_metas": view_metas,
            "hollow_state_dict": state_dict,
            "iteration": iteration,
        }

    @staticmethod
    def decode(
        payload: dict[str, object],
    ) -> tuple[int, MCoreTensorAwareStateDict]:
        # [FUNCTION DOC] HELPER: Reads the four payload entries, rebuilds the tensor views from the
        # received storages and metadata, inserts them back into the hollow state dict (in the same
        # order they were popped), and returns the pair (iteration, restored state dict).
        """Reverse of `encode`: reconstruct per-tensor views from received
        unique_storages using view_metas.
        """
        iteration: int = payload["iteration"]
        hollow_state_dict: MCoreTensorAwareStateDict = payload["hollow_state_dict"]
        unique_storages: list[torch.Tensor] = payload["unique_storages"]
        view_metas: list[dict] = payload["view_metas"]

        tensors = _TensorViewCodec.decode(unique_storages, view_metas)

        hollow_state_dict.insert_tensors(tensors)
        return iteration, hollow_state_dict


class _TensorViewCodec:
    # [CLASS DOC] HELPER: Namespace (static methods) for lossless storage-deduplicated tensor
    # transfer: each distinct underlying memory block is represented once as a byte tensor and each
    # input tensor as a record of which block it lives in plus dtype, shape, stride and offset.
    """Encode tensors as (unique_storages, view_metas) and decode back.

    Many input tensors may share underlying storage (e.g. Megatron
    distributed-optimizer grad buckets). `encode` dedups by storage data_ptr —
    each unique storage is wrapped once as a uint8 tensor (no copy), plus a
    per-input view_meta record (storage_id, dtype, shape, stride,
    storage_offset). `decode` reconstructs the original views with
    `as_strided` over the storage bytes reinterpreted at the original dtype.
    """

    @staticmethod
    def encode(tensors: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[dict]]:
        # [FUNCTION DOC] HELPER: For each tensor, identifies its storage by (device, storage start
        # address); the first time a storage is seen, wraps the full storage as a byte tensor on
        # the same device without copying and assigns it the next index. Appends a record with the
        # storage index, dtype, shape tuple, stride tuple and storage offset (in elements of the
        # tensor's dtype). Returns (list of unique storage byte tensors, list of records in input
        # order).
        storage_id_by_key: dict[tuple[torch.device, int], int] = {}
        unique_storages: list[torch.Tensor] = []
        view_metas: list[dict] = []
        for t in tensors:
            storage = t.untyped_storage()
            key = (t.device, storage.data_ptr())
            if key not in storage_id_by_key:
                storage_id_by_key[key] = len(unique_storages)
                # Wrap full storage as uint8 tensor (no copy, shares memory).
                unique_storages.append(torch.tensor(storage, dtype=torch.uint8, device=t.device))
            view_metas.append(
                {
                    "storage_id": storage_id_by_key[key],
                    "dtype": t.dtype,
                    "shape": tuple(t.shape),
                    "stride": tuple(t.stride()),
                    "storage_offset": t.storage_offset(),
                }
            )
        return unique_storages, view_metas

    @staticmethod
    def decode(unique_storages: list[torch.Tensor], view_metas: list[dict]) -> list[torch.Tensor]:
        # [FUNCTION DOC] HELPER: For each record, takes its byte-tensor storage, reinterprets the
        # bytes as the recorded dtype, and returns a strided view with the recorded shape, stride
        # and offset. Views share memory with the received storages. Output order matches records.
        tensors: list[torch.Tensor] = []
        for vm in view_metas:
            storage_t = unique_storages[vm["storage_id"]]  # uint8 view of received storage
            # Reinterpret bytes as the original dtype, then apply stride/offset.
            dtype_view = storage_t.view(vm["dtype"])
            view = torch.as_strided(
                dtype_view,
                size=vm["shape"],
                stride=vm["stride"],
                storage_offset=vm["storage_offset"],
            )
            tensors.append(view)
        return tensors


def _create_transport(indep_dp: GroupInfo, timeout: timedelta) -> PGTransport:
    # [FUNCTION DOC] HELPER: Raises import error if torchft's transport class could not be imported.
    # Otherwise returns a transport bound to the independent-DP process group, the timeout and the
    # CUDA device.
    if PGTransport is None:
        raise ImportError("torchft is required for checkpoint transfer but could not be imported.")
    return PGTransport(
        pg=indep_dp.group,
        timeout=timeout,
        device=torch.device("cuda"),
    )
