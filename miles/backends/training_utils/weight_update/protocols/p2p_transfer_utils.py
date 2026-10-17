# [MODULE DOC] HELPER: Support code for the peer-to-peer weight transfer protocol: a static plan
# mapping trainer ranks to rollout engine ranks, small records describing remote engine memory, a
# thread-pool manager for background RDMA writes, and helpers to register local pinned memory with the
# Mooncake transfer engine, create that engine, and query engines for their registered weight memory,
# parallelism info and server args.
import dataclasses
import logging
from argparse import Namespace
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import ray
import torch
from sglang.srt.server_args import ServerArgs
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
from miles.utils import async_utils
from miles.utils.workers.argv_utils import _record_field_names

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TransferTaskP2PMeta:
    # [CLASS DOC] HELPER: Plain record identifying one transfer target: engine index, rank within
    # that engine (tensor-parallel rank of the engine), and the source pipeline shard index (default
    # 0).
    """Specifies a engine rollout rank to connect to."""

    # The index of the target rollout engine.
    engine_ind: int
    # The rank of the target shard within the rollout engine (corresponds to `sglang_tp_rank`).
    engine_rank: int
    # The source pp shard index.
    source_shard: int = 0


class RemoteTransferPlan:
    # [CLASS DOC] HELPER: Static planner. On construction reads trainer-side pipeline and
    # data-replica info from the global parallel state and rollout-side sizes from args; exposes the
    # private fields for pipeline rank/size, gathered data-replica rank/size, rollout GPUs per
    # engine, engine count and total rollout GPUs, which the P2P protocol reads directly.
    """
    Plans and manages remote weight transfers for p2p backends,
    assuming static training and rollout placements.
    """

    def __init__(self, args: Namespace) -> None:
        # [FUNCTION DOC] Collects parallelism information immediately (needs the global parallel
        # state to be set).
        self._get_parallelism(args)

    def _get_parallelism(self, args: Namespace) -> None:
        # [FUNCTION DOC] HELPER: Stores the pipeline rank and size; the gathered data-replica rank and
        # size (from the shared helper with pipeline not gathered); the rollout pipeline size from
        # args (raises a not-implemented error unless it is 1); GPUs per engine; engine count = total
        # rollout GPUs integer-divided by GPUs per engine; total rollout GPUs.
        """
        Collect parallelism information for source (trainer) and target (rollout engines).

        All ranks with the same PP rank (gathered_dp_rank) hold a complete weight replica
        after bucketed all-gather across TP/EP/ETP dims. The size of this group is
        gathered_dp_size, and each rank has a unique gathered_dp_rank in [0, gathered_dp_size).
        """
        self._pp_rank = get_parallel_state().pp.rank
        self._pp_size = get_parallel_state().pp.size

        self._gathered_dp_rank, self._gathered_dp_size = get_data_replica_rank_and_size(
            get_parallel_state(), WeightUpdatePlacement(gather_pp=False)
        )

        self._rollout_pp_size = args.sglang_pp_size
        if self._rollout_pp_size != 1:
            raise NotImplementedError("Rollout pipeline parallelism is not tested yet.")
        self._rollout_num_gpu_per_engine = args.rollout_num_gpus_per_engine
        self._rollout_engine_count = args.rollout_num_gpus // self._rollout_num_gpu_per_engine
        self._rollout_num_gpus = args.rollout_num_gpus

    def plan_p2p(self) -> list[TransferTaskP2PMeta]:
        # [FUNCTION DOC] Deterministic (identical on all ranks) assignment of every (engine index,
        # engine rank) target, enumerated engine-major, to a source data-replica rank, returning the
        # tasks of the calling rank (stamped with its pipeline rank as source shard). Step 1: the
        # first N targets (N = number of source ranks, or fewer if targets are fewer) are assigned
        # one-to-one in order, and the number assigned is counted. Step 2, for each remaining target:
        # if some source already holds a target with the same engine rank, choose among those sources
        # the one with the fewest such assignments (ties go to the lowest source index); otherwise
        # pick sources round-robin through a separate cursor. Sources with no assignment get an empty
        # list. Pure computation.
        """
        Plan P2P transfer within each pp_group -> all target engine ranks.

        For each pp shard source rank, it plans the mapping relationship between n source dp ranks, m target rollout engines with k ranks each.
        The Transfer Plan Mapping Heuristics works as follows:
        1. for each target engine (idx, rank), assign source ranks in a round-robin manner until all source ranks are assigned at least once.
        2. for the reminder target (idx, rank), assign them to source ranks by priotizing the source with existing assignmeng of same rank.

        For example, 4 source ranks (0,1,2,3), 2 target engines with 3 ranks each (0,0),(0,1),(0,2),(1,0),(1,1),(1,2).
        The first round of assignment:
        source_rank=0 -> target (0,0)
        source_rank=1 -> target (0,1)
        source_rank=2 -> target (0,2)
        source_rank=3 -> target (1,0)
        The reminder assignment:
        source_rank=1 -> target (1,1)  # prioritize source_rank=1 as it had (0,1) assigned already.
        source_rank=2 -> target (1,2)

        Finally extract the transfer tasks matching the current dp_rank(self._gathered_dp_rank).

        """
        all_targets = [
            (engine_idx, engine_rank)
            for engine_idx in range(self._rollout_engine_count)
            for engine_rank in range(self._rollout_num_gpu_per_engine)
        ]
        assignments = defaultdict(lambda: defaultdict(list))

        # Total number of source-to-target P2P connections established.
        p2p_count = 0
        # step 1: assign engine ranks in a round-robin way
        for source_rank, (_, target) in zip(range(self._gathered_dp_size), enumerate(all_targets), strict=False):
            p2p_count += 1
            engine_idx, engine_rank = target
            assignments[source_rank][engine_rank].append(engine_idx)

        def count_engine_index_assignments(engine_rank: int) -> list[int]:
            # [FUNCTION DOC] HELPER: For each source rank, the number of engines already assigned to
            # it for the given engine rank (list indexed by source rank).
            return [len(assignments[source][engine_rank]) for source in range(self._gathered_dp_size)]

        cur_source_index = 0
        # step 2: assign the left engine ranks.
        if p2p_count < len(all_targets):
            for target in all_targets[p2p_count:]:
                engine_idx, engine_rank = target
                counted = count_engine_index_assignments(engine_rank)
                if max(counted) > 0:
                    # assign it to existing source rank assigned to the same target engine_rank, with lowest load
                    _, select_source = min((val, idx) for (idx, val) in enumerate(counted) if val > 0)
                else:
                    # otherwise round robin
                    select_source = cur_source_index % self._gathered_dp_size
                    cur_source_index += 1
                assignments[select_source][engine_rank].append(engine_idx)

        transfer_tasks = []
        for engine_rank, engine_indices in assignments[self._gathered_dp_rank].items():
            for engine_ind in engine_indices:
                transfer_tasks.append(
                    TransferTaskP2PMeta(source_shard=self._pp_rank, engine_ind=engine_ind, engine_rank=engine_rank)
                )

        return transfer_tasks


@dataclasses.dataclass
class RemoteWeightInfo:
    # [CLASS DOC] HELPER: Record for one remote transfer session: the session id string and a mapping
    # from parameter name to (remote address, element count, element size in bytes).
    """
    The remote weight info related to one specific engine_rank.
    """

    session_id: str
    weights_info: dict[str, tuple[int, int, int]]  # name -> (remote_address, numel, element_size)


class P2PTransferManager:
    # [CLASS DOC] HELPER: Lazily-started thread pool (size num_workers, default 8) plus a list of all
    # submitted futures. The timeout (default 30.0 seconds) applies to each future in the bulk wait.
    # Not thread-safe for concurrent submitters.
    """Generic async task manager for P2P writes.

    Accepts arbitrary callables via submit(), runs them in a thread pool,
    and tracks futures for bulk waiting.
    """

    def __init__(self, num_workers: int = 8, transfer_timeout: float = 30.0):
        # [FUNCTION DOC] Stores settings; no pool is created yet and the future list is empty.
        self.num_workers = num_workers
        self.transfer_timeout = transfer_timeout
        self.executor: ThreadPoolExecutor | None = None
        self.transfer_futures: list[Future] = []

    def ensure_started(self) -> None:
        # [FUNCTION DOC] Create the thread pool on first use (idempotent).
        if self.executor is None:
            # NOTE: RDMA ops won't be affected by the python GIL
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

    def submit(self, fn: Callable, *args) -> None:
        # [FUNCTION DOC] Start the pool if needed, run fn(*args) in it and track the future. Returns
        # nothing.
        """Submit a callable to the thread pool."""
        self.ensure_started()
        future = self.executor.submit(fn, *args)
        self.transfer_futures.append(future)

    def submit_returning_future(self, fn: Callable, *args) -> torch.Future:
        # [FUNCTION DOC] Same as submit but returns the tracked future. KNOWN ISSUE: the return
        # annotation names the torch future type, but the object returned is a standard concurrent
        # futures future.
        """Submit a callable and return its future (also tracked for bulk waiting)."""
        self.ensure_started()
        future = self.executor.submit(fn, *args)
        self.transfer_futures.append(future)
        return future

    def wait_transfers(self) -> None:
        # [FUNCTION DOC] Waits on each tracked future in submission order with the per-future
        # timeout. Any exception, including a timeout, is only logged as an error (not raised), then
        # the future list is cleared. KNOWN ISSUE: failed or timed-out background writes do not
        # propagate; callers cannot detect transfer failure from this call.
        """Wait for all submitted tasks to complete."""
        for future in self.transfer_futures:
            try:
                future.result(timeout=self.transfer_timeout)
            except Exception as e:
                logger.error(f"[P2P] Transfer future failed: {e}")

        self.transfer_futures.clear()


def create_server_args_from_dict(data_dict: dict) -> ServerArgs:
    # [FUNCTION DOC] Build an SGLang server args object from a mapping, silently dropping keys that
    # are not fields of the server args record.
    valid_fields = set(_record_field_names(ServerArgs))
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return ServerArgs(**filtered_data)


def register_cpu_memory(params_dict: dict, transfer_engine) -> dict:
    # [FUNCTION DOC] For each named tensor, registers its memory (address and byte size) with the
    # transfer engine; a nonzero return code raises a runtime error naming the weight. Returns a
    # mapping from name to (address, element count, element size).
    """Register CPU pinned memory with the transfer engine."""
    weight_dict = {}

    for name, cpu_tensor in params_dict.items():
        addr = cpu_tensor.data_ptr()
        size = cpu_tensor.numel() * cpu_tensor.element_size()
        # NOTE: theoretically using huge page allocator
        # in torch backend could imporve registration speed.
        ret = transfer_engine.register_memory(addr, size)
        if ret != 0:
            raise RuntimeError(f"register CPU memory failed for weight {name}, error: {ret}")
        weight_dict[name] = (addr, cpu_tensor.numel(), cpu_tensor.element_size())

    return weight_dict


def create_transfer_engine():
    # [FUNCTION DOC] Lazily imports the Mooncake transfer engine class, creates an instance and
    # initializes it with this node's IP (from Ray), the peer handshake metadata mode, the RDMA
    # protocol and an empty device filter. Returns the engine. Network/RDMA side effects.
    from mooncake.engine import TransferEngine

    transfer_engine = TransferEngine()
    local_ip = ray._private.services.get_node_ip_address()
    transfer_engine.initialize(local_ip, "P2PHANDSHAKE", "rdma", "")
    return transfer_engine


def query_remote_weight_infos(
    rollout_engines: Sequence[SGLangApiClient],
    targets,
) -> tuple[dict, dict, dict]:
    # [FUNCTION DOC] For each unique (engine index, engine rank) in the targets (iterated in set
    # order, so not deterministic), blocks on three engine calls: transfer-engine info (session id
    # and per-parameter memory mapping), parallelism info for that rank, and server info (filtered
    # into a server args object). Returns three mappings: session id to (weights info, parallelism
    # info); (engine index, engine rank) to session id; session id to server args. KNOWN ISSUE: the
    # check that the session id is not none comes after it was already used as a key for server
    # args, so a missing id is stored before the assertion fires.
    """Query remote rollout engines for weight info, session IDs, and server args."""
    remote_weight_infos_by_session_id = {}
    targets_to_session_id = {}
    session_id_to_server_args = {}
    targets_to_query = set((target.engine_ind, target.engine_rank) for target in targets)

    for engine_ind, engine_rank in targets_to_query:
        session_id, weights_info = async_utils.run(
            rollout_engines[engine_ind].get_remote_instance_transfer_engine_info(rank=engine_rank)
        )
        parallelism_info = async_utils.run(rollout_engines[engine_ind].get_parallelism_info(rank=engine_rank))

        session_id_to_server_args[session_id] = create_server_args_from_dict(
            async_utils.run(rollout_engines[engine_ind].get_server_info())
        )
        assert session_id is not None, f"Failed to get session id from rollout engine {engine_ind} rank {engine_rank}"
        remote_weight_infos_by_session_id[session_id] = (weights_info, parallelism_info)
        targets_to_session_id[(engine_ind, engine_rank)] = session_id

    return remote_weight_infos_by_session_id, targets_to_session_id, session_id_to_server_args
