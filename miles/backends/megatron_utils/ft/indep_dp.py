# [MODULE DOC]
# Independent data parallelism across training cells (fault-tolerant replicas that can join and
# leave). Provides creation of the cross-cell process groups (torchft NCCL and Gloo groups tagged
# with a quorum id), reconfiguration when membership changes, and a gradient/loss all-reduce
# across cells that reports failure through a cell-wide consensus flag instead of raising, so the
# caller can discard the step. All events are logged in structured form with tag "ft".
import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import TYPE_CHECKING

import torch.distributed as dist
from megatron.core import mpu

from miles.backends.training_utils.metrics.log_utils import aggregate_train_losses
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.ft_utils.process_group_utils import GeneralPGUtil, GroupInfo, collective_bool_and
from miles.utils.tracking_utils.structured_log import log_structured

from ...training_utils.parallel import ParallelState

if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def create_indep_dp_group(
    store_addr: str | None,
    indep_dp_info: IndepDPInfo,
    megatron_rank: int,
    megatron_world_size: int,
) -> GroupInfo:
    # [FUNCTION DOC]
    # Builds the group info for the independent-DP axis. With at most one alive cell, returns a
    # trivial info (rank 0, size 1, no group) carrying the quorum id as debug info. Otherwise
    # lazily imports torchft's NCCL and Gloo process-group classes (raises import error with an
    # install hint if missing) and creates one of each, with a 120-second timeout, configured with
    # a store address of the form: store address, "/indep_dp/", backend name, quorum id, megatron
    # rank; replica id = cell index; rank/size = the cell's alive rank/size; quorum id; and group
    # rank/size = the megatron rank/world size. Logs creation and returns the info with the NCCL
    # group as the main group and the Gloo group as the gloo group. Network/rendezvous side effects.
    if indep_dp_info.alive_size <= 1:
        return GroupInfo(rank=0, size=1, group=None, debug_info={"quorum": indep_dp_info.quorum_id})

    try:
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL
    except ImportError as e:
        raise ImportError("torchft is required for indep_dp. Install with: pip install torchft") from e

    _TIMEOUT = timedelta(seconds=120)

    def _create(pg_cls: type, backend_name: str) -> dist.ProcessGroup:
        # [FUNCTION DOC] HELPER: Instantiates the given process-group class with the timeout and
        # configures it for the current quorum (store address includes the backend name); returns it.
        pg = pg_cls(timeout=_TIMEOUT)
        pg.configure(
            store_addr=f"{store_addr}/indep_dp/{backend_name}/{indep_dp_info.quorum_id}/{megatron_rank}",
            replica_id=str(indep_dp_info.cell_index),
            rank=indep_dp_info.alive_rank,
            world_size=indep_dp_info.alive_size,
            quorum_id=indep_dp_info.quorum_id,
            group_rank=megatron_rank,
            group_world_size=megatron_world_size,
        )
        return pg

    nccl_pg = _create(ProcessGroupNCCL, "nccl")
    gloo_pg = _create(ProcessGroupGloo, "gloo")
    log_structured(
        logger.info,
        tag="ft",
        op="create_pg",
        cell=indep_dp_info.cell_index,
        cell_rank=indep_dp_info.alive_rank,
        members=indep_dp_info.alive_size,
        quorum=indep_dp_info.quorum_id,
        megatron_rank=megatron_rank,
        megatron_world_size=megatron_world_size,
    )
    return GroupInfo(
        rank=indep_dp_info.alive_rank,
        size=indep_dp_info.alive_size,
        group=nccl_pg,
        gloo_group=gloo_pg,
        debug_info={"quorum": indep_dp_info.quorum_id},
    )


def reconfigure_indep_dp_group(
    parallel_state: ParallelState,
    store_addr: str | None,
    indep_dp_info: IndepDPInfo,
    megatron_rank: int,
    megatron_world_size: int,
) -> None:
    # [FUNCTION DOC]
    # Logs a start event, shuts down the existing independent-DP main and gloo groups (those that
    # exist) on the parallel state, creates new groups for the new membership/quorum and assigns
    # them to the parallel state's independent-DP slot (mutation), then logs an end event.
    """Shut down old indep_dp PGs and create new ones with a fresh quorum_id."""
    old = parallel_state.indep_dp
    log_structured(
        logger.info,
        tag="ft",
        op="reconfig",
        phase="start",
        cell=indep_dp_info.cell_index,
        quorum_to=indep_dp_info.quorum_id,
        alive_rank=indep_dp_info.alive_rank,
        members=indep_dp_info.alive_size,
    )
    for g in [old.group, old.gloo_group]:
        if g is not None:
            g.shutdown()

    parallel_state.indep_dp = create_indep_dp_group(
        store_addr=store_addr,
        indep_dp_info=indep_dp_info,
        megatron_rank=megatron_rank,
        megatron_world_size=megatron_world_size,
    )
    log_structured(
        logger.info,
        tag="ft",
        op="reconfig",
        phase="end",
        cell=indep_dp_info.cell_index,
        quorum=indep_dp_info.quorum_id,
    )


def allreduce_grads_and_losses_across_replicas(
    args,
    model: Sequence["DDP"],
    parallel_state: ParallelState,
    losses_reduced: list,
    num_rollouts: int | None = None,
) -> tuple[bool, dict[str, float]]:
    # [FUNCTION DOC]
    # Sums gradients (and aggregates losses) across cells. Asserts per-token loss is off and the
    # intra-cell data-parallel size is 1 (assertion errors; combining intra and independent DP is
    # unsupported). Steps: on the last pipeline stage (ignoring virtual stages) aggregates the
    # reduced training losses (over num_rollouts when given) into a loss mapping; for each model
    # chunk, for every bucket in its dense and expert-parallel bucket groups, all-reduces the
    # bucket's gradient data with sum across the independent-DP group through the generic
    # process-group utility. Any exception during this is caught, logged as an error with the
    # traceback, and marks failure. Then queries the group for an asynchronous error exactly once
    # (the query can force a stream sync); if one is reported it marks failure and logs loudly.
    # Finally takes a cell-wide logical AND of success over the cell-local gloo group so that if any
    # rank failed all ranks agree. Returns (consensus success flag, loss mapping; empty on non-last
    # pipeline stages or when the aggregation did not run). Gradients are summed, not averaged.
    # Gradient buffers may be partially reduced after a failure.
    assert not args.calculate_per_token_loss, "calculate_per_token_loss is not supported with indep_dp yet"
    assert parallel_state.intra_dp.size == 1, (
        f"indep_dp requires intra_dp.size == 1, got {parallel_state.intra_dp.size}. "
        "Simultaneous intra and indep DP is not supported."
    )

    pg = parallel_state.indep_dp.group
    util = GeneralPGUtil.create(pg)
    log_structured(
        logger.info,
        tag="ft",
        op="cross_cell",
        phase="start",
        kind="grad_allreduce",
        cell_rank=parallel_state.indep_dp.rank,
        members=parallel_state.indep_dp.size,
        **parallel_state.indep_dp.debug_info,
    )

    loss_reduced: dict[str, float] = {}
    allreduce_success = True
    try:
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            loss_reduced = aggregate_train_losses(losses_reduced, num_rollouts)
        for model_chunk in model:
            # mimic: DistributedDataParallel.start_grad_sync
            for bucket_group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
                for bucket in bucket_group.buckets:
                    util.all_reduce(bucket.grad_data, pg, op=dist.ReduceOp.SUM)
    except Exception:
        allreduce_success = False
        log_structured(
            logger.error,
            tag="ft",
            op="cross_cell",
            phase="fail",
            kind="grad_allreduce",
            cell_rank=parallel_state.indep_dp.rank,
            members=parallel_state.indep_dp.size,
            exc_info=True,
        )

    # pg.errored() can force a CUDA/stream sync, so call it exactly once per step here -- do NOT
    # sprinkle extra errored() probes. When it does report an async error it MUST be logged loudly:
    # a swallowed cross-cell error means an un-reduced (wrong) gradient would be applied silently.
    if (e := pg.errored()) is not None:
        allreduce_success = False
        log_structured(
            logger.error,
            tag="ft",
            op="cross_cell",
            phase="async_error",
            kind="grad_allreduce",
            cell_rank=parallel_state.indep_dp.rank,
            members=parallel_state.indep_dp.size,
            error=str(e),
        )

    # Intra-cell consensus: if ANY rank's allreduce failed, ALL ranks discard.
    # get_gloo_group() is cell-local (created from the default world PG).
    consensus = collective_bool_and(value=allreduce_success, group=get_gloo_group())
    log_structured(
        logger.info,
        tag="ft",
        op="cross_cell",
        phase="end",
        kind="grad_allreduce",
        cell_rank=parallel_state.indep_dp.rank,
        members=parallel_state.indep_dp.size,
        **parallel_state.indep_dp.debug_info,
        this_rank_ok=allreduce_success,
        consensus_ok=consensus,
    )
    return consensus, loss_reduced
