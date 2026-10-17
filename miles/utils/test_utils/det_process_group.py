# [MODULE DOC] HELPER: A torch distributed backend ("det_nccl") whose sum and average reductions are
# bitwise deterministic, for debugging and tests. Components: a registration function; a process
# group wrapper that forwards all collectives to an inner NCCL group except all-reduce, reduce and
# reduce-scatter (sum-like ops), which are implemented as all-gather followed by a local fold in a
# fixed order; standalone functions for deterministic all-reduce and reduce-scatter that accept
# any process group; chunked gathering to bound memory; and a completed-work placeholder. Fold
# order: pairwise tree when the world size is a power of two, ascending rank order otherwise.
# Only max and min reductions are delegated to the inner group; every other reduce operation is
# treated as sum (average divides by group size afterwards). Costs extra bandwidth and
# synchronization.
"""Process group with bitwise-deterministic SUM reductions.

``DetProcessGroup`` wraps an inner c10d NCCL group. Every collective delegates to
the inner group except the order-sensitive reductions — ``allreduce`` and
``reduce_scatter`` — which are computed as all-gather (pure data movement, no
arithmetic) plus a fixed local fold: a pairwise
tree for power-of-two world sizes, an ascending-rank fold otherwise. The summation
order is therefore independent of the NCCL version, topology, communicator
instance, or buffer layout, and reduce-scatter takes its shard from the same full
fold, so reduce-scatter and all-reduce agree bitwise by construction.

Debug/test use only: the fold trades bandwidth and synchrony for determinism.
"""

import logging

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup as BaseProcessGroup
from torch.distributed import Work
from torch.distributed.distributed_c10d import AllgatherOptions

logger = logging.getLogger(__name__)


DET_NCCL_BACKEND_NAME = "det_nccl"
_backend_registered = False

# Cap on the gather buffer so the fold never allocates world_size x the full tensor
# (a 30B MoE grad buffer x 8 ranks is >100 GiB). Chunking is bitwise-neutral: each
# output element is still folded over the same per-rank operands in the same order.
_GATHER_BUFFER_CAP_BYTES = 1 << 30


def register_det_nccl_backend() -> None:
    # [FUNCTION DOC] Registers the "det_nccl" backend with torch distributed for the cuda device
    # type, using the backend factory below with the extended (options-object) API. Idempotent
    # via a module-level flag; mutates global torch distributed state and logs an info line on
    # first registration. Must be called before creating groups with that backend name.
    """Register the ``det_nccl`` torch.distributed backend (DetProcessGroup over NCCL).

    After registration, groups created with ``backend="det_nccl"`` (via
    ``init_process_group`` or ``new_group``) run SUM/AVG reductions through the
    deterministic fold. Idempotent.
    """
    global _backend_registered
    if _backend_registered:
        return
    dist.Backend.register_backend(DET_NCCL_BACKEND_NAME, _create_det_nccl_backend, extended_api=True, devices=["cuda"])
    _backend_registered = True
    logger.info("Registered torch.distributed backend %s", DET_NCCL_BACKEND_NAME)


def _create_det_nccl_backend(dist_backend_opts: object, pg_options: object) -> "DetProcessGroup":
    # [FUNCTION DOC] HELPER: Backend factory called by torch distributed. Builds an NCCL process
    # group from the store, group rank and group size in the backend options (default NCCL
    # options; the separately supplied process-group options argument is ignored) and returns a
    # deterministic wrapper around it. Imports the NCCL group class lazily.
    from torch.distributed import ProcessGroupNCCL

    inner = ProcessGroupNCCL(
        dist_backend_opts.store,
        dist_backend_opts.group_rank,
        dist_backend_opts.group_size,
        ProcessGroupNCCL.Options(),
    )
    return DetProcessGroup(inner)


class DetProcessGroup(BaseProcessGroup):
    # [CLASS DOC] Process group that wraps an inner group (the inner group is stored privately).
    # Overrides collective entry points: sum-like reductions go through the deterministic fold
    # and return an already-completed work handle (they run synchronously); everything else
    # (all-gather, barrier, broadcast, send, receive, all-to-all, max/min reductions) is forwarded
    # to the inner group, returning its work handle. Reports its backend name as the registered
    # name. Coalescing is emulated: starting is a no-op and ending returns a completed work, with
    # the coalesced collective variants executing each element in turn.
    """Wrapper process group whose SUM/AVG reductions use a fixed-order fold."""

    def __init__(self, inner: dist.ProcessGroup) -> None:
        # [FUNCTION DOC] Initializes the base group with the inner group's rank and size, keeps
        # the inner group, marks the default backend as custom and registers the inner group as
        # the cuda-device backend so device-type lookups and object collectives choose cuda.
        # This only feeds lookups; collectives still dispatch to this class's methods.
        super().__init__(inner.rank(), inner.size())
        self._inner = inner
        # Register the cuda backend as torchft's ProcessGroupNCCL._create_pg does, so
        # _device_types reports cuda and object collectives pick cuda over cpu. Only
        # feeds device/backend lookups; collectives still dispatch to our methods.
        self._set_default_backend(BaseProcessGroup.BackendType.CUSTOM)
        self._register_backend(torch.device("cuda"), BaseProcessGroup.BackendType.CUSTOM, self._inner)

    # ------------------------------------------------------------------ #
    # Deterministic reductions
    # ------------------------------------------------------------------ #

    def allreduce(self, tensors: list[torch.Tensor], opts: object) -> Work:
        # [FUNCTION DOC] All-reduce. Max and min operations are delegated to the inner group
        # unchanged. Any other operation is treated as sum (or average): each tensor in the list is
        # reduced in place by the deterministic all-reduce, synchronously, and a completed work
        # is returned. The reduce operation is read from the options object (or the bare op).
        # KNOWN ISSUE: only max and min are delegated, so other operations (product, bitwise and
        # so on) are silently computed as a sum; the same applies to reduce and reduce-scatter.
        reduce_op = _reduce_op_of(opts)
        if reduce_op == dist.ReduceOp.MAX or reduce_op == dist.ReduceOp.MIN:
            return self._inner.allreduce(tensors, opts)

        for tensor in tensors:
            det_all_reduce(tensor, group=self._inner, reduce_op=reduce_op)
        return _CompletedWork()

    def allreduce_coalesced(self, tensors: list[torch.Tensor], opts: object) -> Work:
        # [FUNCTION DOC] Same as the plain all-reduce on the tensor list.
        return self.allreduce(tensors, opts)

    def _reduce_scatter_base(self, output: torch.Tensor, input: torch.Tensor, opts: object) -> Work:
        # [FUNCTION DOC] Flat reduce-scatter. Max and min are delegated to the inner group.
        # Otherwise runs the deterministic reduce-scatter writing this rank's equal slice of the
        # reduced input into the output (input element count must be world size times output
        # element count, assertion error otherwise) and returns a completed work.
        reduce_op = _reduce_op_of(opts)
        if reduce_op == dist.ReduceOp.MAX or reduce_op == dist.ReduceOp.MIN:
            return self._inner._reduce_scatter_base(output, input, opts)

        det_reduce_scatter(
            output, input, group=self._inner, rank=self.rank(), world_size=self.size(), reduce_op=reduce_op
        )
        return _CompletedWork()

    def reduce_scatter(
        self, output_tensors: list[torch.Tensor], input_tensors: list[list[torch.Tensor]], opts: object
    ) -> Work:
        # [FUNCTION DOC] List-form reduce-scatter. Max and min are delegated. Otherwise, for each
        # pair of an output tensor and its list of per-rank input tensors (lists must be the same
        # length, strict pairing raises a value error otherwise): asserts the input slot for this
        # rank has the same element count as the output, then every rank participates in folding
        # each slot in order (slots may differ in size); only the slot belonging to this rank
        # writes its result into the output. Returns a completed work.
        reduce_op = _reduce_op_of(opts)
        if reduce_op == dist.ReduceOp.MAX or reduce_op == dist.ReduceOp.MIN:
            return self._inner.reduce_scatter(output_tensors, input_tensors, opts)

        for output, inputs in zip(output_tensors, input_tensors, strict=True):
            # Slot j has one size on every rank (sizes may differ between slots);
            # every rank joins each slot's fold, only rank j keeps the result.
            assert (
                inputs[self.rank()].numel() == output.numel()
            ), f"slot {self.rank()} numel {inputs[self.rank()].numel()} != output numel {output.numel()}"
            for slot_idx, slot_input in enumerate(inputs):
                _det_reduce_scatter_slot(
                    output if slot_idx == self.rank() else None, slot_input, group=self._inner, reduce_op=reduce_op
                )
        return _CompletedWork()

    # ------------------------------------------------------------------ #
    # Plain delegation
    # ------------------------------------------------------------------ #

    def allgather(
        self, output_tensors: list[list[torch.Tensor]], input_tensors: list[torch.Tensor], opts: object
    ) -> Work:
        # [FUNCTION DOC] Forwards to the inner group's list-form all-gather and returns its work.
        return self._inner.allgather(output_tensors, input_tensors, opts)

    def allgather_into_tensor_coalesced(
        self, output_tensors: list[torch.Tensor], input_tensors: list[torch.Tensor], opts: object = None
    ) -> Work:
        # [FUNCTION DOC] Coalesced flat all-gather. For each (output, input) pair (strictly paired)
        # issues the inner group's flat all-gather with the given options (default: fresh
        # all-gather options) and waits on the returned work if it is not none; returns a
        # completed work.
        # The coalescing manager's flush lands here with its region still open,
        # so the inner calls are batched and hand back None; it waits on the
        # aggregate itself.
        effective_opts = opts if opts is not None else AllgatherOptions()
        for output, input in zip(output_tensors, input_tensors, strict=True):
            work = self._inner._allgather_base(output, input, effective_opts)
            if work is not None:
                work.wait()
        return _CompletedWork()

    def _allgather_base(self, output: torch.Tensor, input: torch.Tensor, opts: object) -> Work:
        # [FUNCTION DOC] Forwards the flat all-gather to the inner group.
        return self._inner._allgather_base(output, input, opts)

    def barrier(self, opts: object) -> Work:
        # [FUNCTION DOC] Forwards the barrier to the inner group.
        return self._inner.barrier(opts)

    def broadcast(self, tensor_list: list[torch.Tensor], opts: object) -> Work:
        # [FUNCTION DOC] Forwards the broadcast to the inner group.
        return self._inner.broadcast(tensor_list, opts)

    def reduce(self, tensors: list[torch.Tensor], opts: object) -> Work:
        # [FUNCTION DOC] Rooted reduce. Max and min are delegated to the inner group. Any other
        # operation is implemented as a deterministic all-reduce (so every rank, not only the
        # root, ends up with the result in its tensors).
        reduce_op = _reduce_op_of(opts)
        if reduce_op == dist.ReduceOp.MAX or reduce_op == dist.ReduceOp.MIN:
            return self._inner.reduce(tensors, opts)
        return self.allreduce(tensors, opts)

    def reduce_scatter_tensor_coalesced(
        self, output_tensors: list[torch.Tensor], input_tensors: list[torch.Tensor], opts: object
    ) -> Work:
        # [FUNCTION DOC] Runs the flat reduce-scatter for each (output, input) pair in order
        # (strictly paired, value error on length mismatch), ignoring the returned work handles,
        # and returns a completed work.
        for output, input in zip(output_tensors, input_tensors, strict=True):
            self._reduce_scatter_base(output, input, opts)
        return _CompletedWork()

    # torch 2.13 routes dist.reduce_scatter_tensor through these names; without
    # them it reaches the C++ base, which dispatches to the NCCL backend
    # registered above and silently drops the fixed-order fold.
    def reduce_scatter_single(self, output: torch.Tensor, input: torch.Tensor, opts: object) -> Work:
        # [FUNCTION DOC] Entry point used by newer torch for the tensor reduce-scatter; routes to
        # the flat reduce-scatter so the deterministic fold is not bypassed.
        return self._reduce_scatter_base(output, input, opts)

    def reduce_scatter_single_coalesced(
        self, output_tensors: list[torch.Tensor], input_tensors: list[torch.Tensor], opts: object
    ) -> Work:
        # [FUNCTION DOC] Newer-torch coalesced entry point; routes to the coalesced flat
        # reduce-scatter.
        return self.reduce_scatter_tensor_coalesced(output_tensors, input_tensors, opts)

    def alltoall_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        opts: object,
    ) -> Work:
        # [FUNCTION DOC] Forwards the all-to-all (with optional split sizes) to the inner group.
        return self._inner.alltoall_base(output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts)

    def all_to_all_single(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        opts: object,
    ) -> Work:
        # [FUNCTION DOC] Newer-torch entry point for the single-tensor all-to-all; forwards to the
        # all-to-all base method (and so the inner group) because the C++ base implementation
        # faults on a wrapper group.
        # torch 2.13's entry point for dist.all_to_all_single. No reduction is
        # involved, but the C++ base's alltoall faults in NCCL on a wrapper
        # group, so forward it the way alltoall_base does.
        return self.alltoall_base(output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts)

    def send(self, tensors: list[torch.Tensor], dst_rank: int, tag: int) -> Work:
        # [FUNCTION DOC] Forwards the point-to-point send to the inner group.
        return self._inner.send(tensors, dst_rank, tag)

    def recv(self, tensors: list[torch.Tensor], src_rank: int, tag: int) -> Work:
        # [FUNCTION DOC] Forwards the point-to-point receive to the inner group.
        return self._inner.recv(tensors, src_rank, tag)

    def _start_coalescing(self, device: torch.device) -> None:
        # [FUNCTION DOC] No-op: coalesced operations are queued at the Python level and flushed
        # through the coalesced methods; nothing is started on the inner group.
        # Ops queue at the Python level and flush via the *_coalesced methods.
        return None

    def _end_coalescing(self, device: torch.device) -> Work:
        # [FUNCTION DOC] Returns a completed work handle without doing anything.
        return _CompletedWork()

    def getBackendName(self) -> str:
        # [FUNCTION DOC] Returns the registered backend name, "det_nccl".
        return DET_NCCL_BACKEND_NAME


def det_all_reduce(tensor: torch.Tensor, *, group: dist.ProcessGroup, reduce_op: object = dist.ReduceOp.SUM) -> None:
    # [FUNCTION DOC] In-place deterministic all-reduce of a tensor across the group. A
    # non-contiguous tensor is copied to a contiguous temporary, reduced, and copied back.
    # Otherwise folds the flat view in chunks (gather then fixed-order local sum; input and output
    # alias the same flat view) and, for the average operation, divides by the group size
    # afterwards. Any operation other than average is treated as sum. Collective: all ranks of
    # the group must call it with equal shapes and dtypes.
    """SUM/AVG ``tensor`` across ranks in-place with the fixed fold.

    ``group`` may be a raw c10d backend (flat ``_allgather_base``) or any ``ProcessGroup``
    such as torchft's wrappers (list-form ``allgather``): the gather is pure data
    movement and the local fold defines the (shared) summation order.
    """
    if not tensor.is_contiguous():
        work = tensor.contiguous()
        det_all_reduce(work, group=group, reduce_op=reduce_op)
        tensor.copy_(work)
        return

    flat = tensor.view(-1)
    _det_chunked_fold(flat, flat, group=group)
    if reduce_op == dist.ReduceOp.AVG:
        flat.div_(group.size())


def det_reduce_scatter(
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    rank: int,
    world_size: int,
    reduce_op: object = dist.ReduceOp.SUM,
) -> None:
    # [FUNCTION DOC] Deterministic reduce-scatter of an input tensor into an output tensor.
    # Asserts the input element count equals world size times the output element count. Makes the
    # input contiguous and flat, cuts it into world-size equal slots, and for each slot in order
    # runs the slot fold across the group; only the slot index equal to this rank writes into the
    # output (average divides by group size). Collective over the group.
    """SUM/AVG-reduce ``input`` across ranks with the fixed fold and write this rank's
    ``1/world_size`` slice into ``output`` (mirrors ``dist.reduce_scatter_tensor``).
    """
    assert (
        input.numel() == world_size * output.numel()
    ), f"uneven reduce_scatter: input numel {input.numel()} != {world_size} x output numel {output.numel()}"

    flat = input.contiguous().view(-1)
    slot_numel = output.numel()
    for slot_idx in range(world_size):
        slot_input = flat[slot_idx * slot_numel : (slot_idx + 1) * slot_numel]
        _det_reduce_scatter_slot(output if slot_idx == rank else None, slot_input, group=group, reduce_op=reduce_op)


def _det_reduce_scatter_slot(
    output: torch.Tensor | None, slot_input: torch.Tensor, *, group: dist.ProcessGroup, reduce_op: object
) -> None:
    # [FUNCTION DOC] HELPER: Folds one slot across ranks. With no output (not the owner rank) it
    # only takes part in the gathers. With an output: folds into the output's flat view if the
    # output is contiguous, else into a freshly allocated flat staging buffer that is then copied
    # back into the output with its shape; for average divides the output by the group size.
    """Fold one slot across ranks into ``output``; ``None`` joins the collective only."""
    flat_input = slot_input.contiguous().view(-1)
    if output is None:
        _det_chunked_fold(flat_input, None, group=group)
        return

    # NOTE: empty_like preserves the (non-contiguous) layout, so it cannot be view(-1)'d;
    # allocate the staging buffer flat directly.
    out_flat = (
        output.view(-1)
        if output.is_contiguous()
        else torch.empty(output.numel(), dtype=output.dtype, device=output.device)
    )
    _det_chunked_fold(flat_input, out_flat, group=group)
    if not output.is_contiguous():
        output.copy_(out_flat.view(output.shape))
    if reduce_op == dist.ReduceOp.AVG:
        output.div_(group.size())


def _det_chunked_fold(
    flat_input: torch.Tensor,
    out_flat: torch.Tensor | None,
    *,
    group: dist.ProcessGroup,
) -> None:
    # [FUNCTION DOC] HELPER: Core of the deterministic reductions. Splits the flat input into
    # chunks sized so that the gather buffer (world size times chunk size times element size)
    # stays within 1 GiB (at least one element per chunk, at most the whole tensor). For each
    # chunk: gathers every rank's chunk into one buffer (a rank-major layout), and, if an output
    # is given, folds the per-rank rows in the fixed order and copies the result into the output
    # slice. Every rank must call with equal sizes since the gather is collective; a none output
    # joins gathers without writing. Allocates one gather buffer on the input's device and dtype.
    """Fold ``flat_input`` across ranks chunk by chunk into ``out_flat`` (same numel;
    may alias ``flat_input``). ``None`` joins the gathers without folding.
    """
    world_size = group.size()
    total = flat_input.numel()
    chunk_numel = max(1, min(total, _GATHER_BUFFER_CAP_BYTES // (world_size * flat_input.element_size())))
    gather_buf = torch.empty(world_size * chunk_numel, dtype=flat_input.dtype, device=flat_input.device)

    for start in range(0, total, chunk_numel):
        count = min(chunk_numel, total - start)
        buf = gather_buf[: world_size * count]
        _gather_into(group, buf, flat_input[start : start + count])
        if out_flat is not None:
            folded = _fold_gathered_sum(list(buf.view(world_size, count).unbind(dim=0)))
            out_flat[start : start + count].copy_(folded)


def _gather_into(group: dist.ProcessGroup, output: torch.Tensor, input: torch.Tensor) -> None:
    # [FUNCTION DOC] HELPER: Blocking all-gather of each rank's input into the flat output buffer
    # (rank-major). If the group is a high-level process group object (including wrappers such as
    # fault-tolerance groups) it uses the list-form all-gather with rows viewed from the output;
    # otherwise (a raw backend) uses the flat all-gather base. Waits for completion.
    if isinstance(group, dist.ProcessGroup):
        # ProcessGroup wrappers (torchft) inherit ``_allgather_base`` from the C++ base, but it
        # dispatches to a per-device backend they never register; only the overridden list-form
        # ``allgather`` is safe. ``hasattr`` cannot discriminate here.
        rows = list(output.view(group.size(), -1).unbind(dim=0))
        group.allgather([rows], [input], AllgatherOptions()).wait()
    else:
        group._allgather_base(output, input, AllgatherOptions()).wait()


def _reduce_op_of(opts: object) -> object:
    # [FUNCTION DOC] HELPER: Returns the reduce operation carried by an options object (its reduce
    # operation attribute) or, when the argument has no such attribute, the argument itself.
    """Extract the ReduceOp from an options object (or pass a bare ReduceOp through)."""
    return opts.reduceOp if hasattr(opts, "reduceOp") else opts


class _CompletedWork(Work):
    # [CLASS DOC] HELPER: Work handle representing an operation that has already finished
    # synchronously.
    """Work handle for an operation that already completed synchronously."""

    def wait(self, timeout: object = None) -> bool:
        # [FUNCTION DOC] HELPER: Returns true immediately; the timeout is ignored.
        return True

    def get_future(self) -> torch.futures.Future:
        # [FUNCTION DOC] HELPER: Returns a new future already completed with no value.
        future: torch.futures.Future = torch.futures.Future()
        future.set_result(None)
        return future


def _fold_gathered_sum(gathered: list[torch.Tensor]) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Sums the per-rank tensors in a fixed order. If the count is a power
    # of two (and nonzero) it reduces pairwise level by level: adjacent pairs (0+1, 2+3, ...) form
    # new partials until one remains. Otherwise it accumulates in ascending rank order into the
    # first tensor in place (mutating it). Returns the sum. An empty list raises an index error.
    """Sum a per-rank gathered list in a fixed order (pairwise tree for power-of-two).

    May reuse (mutate) the gathered buffers as accumulators.
    """
    world_size = len(gathered)
    if world_size > 0 and (world_size & (world_size - 1)) == 0:
        partials = gathered
        while len(partials) > 1:
            partials = [partials[i] + partials[i + 1] for i in range(0, len(partials), 2)]
        return partials[0]

    running = gathered[0]
    for index in range(1, world_size):
        running += gathered[index]
    return running
