# [MODULE DOC]
# Cross-model scheduler for the Tinker gateway. Looks at the ready work of all model queues,
# picks the earliest-arrived item (a datum of a batch request or a barrier) as the seed, and builds
# one execution unit: a batch unit packing compatible datums from any models up to a token budget,
# or a barrier unit (optimizer steps of several models are merged into one unit; other barriers
# run alone). Also reports queue heads that must be rejected for validation errors.
"""Choose the earliest ready seed, then pack compatible work across model queues."""

from dataclasses import dataclass

from miles.tinker.core.model_queue import ModelRequestQueue, PendingRequest
from miles.tinker.core.types import CommandOp


@dataclass
class DatumRef:
    # [CLASS DOC] Reference to one datum of a pending batch request: its model queue, the request,
    # and the datum's index within the request. Outputs are written back through the request.
    """Pointer to ``request.datums[local_index]``; the datum's output is written back through it."""

    model_queue: ModelRequestQueue
    request: PendingRequest
    local_index: int

    @property
    def datum(self) -> dict:
        # [FUNCTION DOC] Property: the referenced datum mapping.
        return self.request.datums[self.local_index]

    @property
    def arrival(self) -> int:
        # [FUNCTION DOC] Property: the arrival counter of the owning command (global submit order).
        return self.request.command.arrival


@dataclass
class BatchUnit:
    # [CLASS DOC] A unit of trainer work for batch ops: the operation (forward-backward or
    # forward-only), the shared loss function name and config (all packed datums agree on them),
    # and the datum references in packing order.
    """One forward pass on the trainer: datums packed from any number of requests."""

    op: CommandOp  # FORWARD_BACKWARD | FORWARD_ONLY
    loss_fn: str | None
    loss_fn_config: dict | None
    datums: list[DatumRef]


@dataclass
class BarrierUnit:
    # [CLASS DOC] A unit of trainer work for a barrier op: the operation and the list of
    # (model queue, pending request) entries; more than one entry only for optimizer steps.
    """One barrier call after all preceding batch requests in its model queue finish."""

    op: CommandOp  # OPTIM_STEP | SAVE_STATE | LOAD_STATE | SAVE_WEIGHTS_FOR_SAMPLER
    entries: list[tuple[ModelRequestQueue, PendingRequest]]


class RequestScheduler:
    # [CLASS DOC] Holds the registry of model queues (by model id) and the token budget per batch
    # unit; stateless otherwise (selection state lives in the queues' pending requests, whose issued
    # datum counters it advances).
    def __init__(self, batch_token_budget: int) -> None:
        # [FUNCTION DOC] Store the per-batch token budget and start with no queues.
        self.batch_token_budget = batch_token_budget
        self._model_queues: dict[str, ModelRequestQueue] = {}

    def add_model_queue(self, model_queue: ModelRequestQueue) -> None:
        # [FUNCTION DOC] Register (or replace) the queue under its model id.
        self._model_queues[model_queue.model_id] = model_queue

    def remove_model_queue(self, model_id: str) -> None:
        # [FUNCTION DOC] Unregister the model's queue; unknown ids raise a key error.
        del self._model_queues[model_id]

    def model_queue(self, model_id: str) -> ModelRequestQueue:
        # [FUNCTION DOC] Look up a registered queue; unknown ids raise a key error.
        return self._model_queues[model_id]

    def ready_rejections(self) -> list[tuple[ModelRequestQueue, PendingRequest]]:
        # [FUNCTION DOC] List (queue, head request) for every queue whose head request carries a
        # validation error; the dispatcher fails these without running them.
        return [
            (model_queue, model_queue.queue[0])
            for model_queue in self._model_queues.values()
            if model_queue.queue and model_queue.queue[0].command.validation_error is not None
        ]

    def schedule_next(self) -> BatchUnit | BarrierUnit | None:
        # [FUNCTION DOC] Choose the next unit or none if nothing is ready. Collect all issuable
        # datums and all ready barriers. The seed is whichever has the smallest arrival number:
        # the earliest datum (ties among datums broken by the minimum function's first element) or
        # the earliest barrier; a barrier wins only if strictly earlier than the earliest datum.
        # A barrier seed produces a merged barrier unit, a datum seed a packed batch unit (which
        # marks its datums as issued). Work compatible with the seed may overtake earlier
        # incompatible requests.
        """Arrival order selects the seed; compatible work may overtake intervening requests."""
        datums = self._ready_datums()
        barriers = self._ready_barriers()

        datum_seed = min(datums, key=lambda ref: ref.arrival) if datums else None
        barrier_seed = min(barriers, key=lambda e: e[1].command.arrival) if barriers else None
        if datum_seed is None and barrier_seed is None:
            return None
        if barrier_seed is not None and (datum_seed is None or barrier_seed[1].command.arrival < datum_seed.arrival):
            return self._merge_barriers(barrier_seed, barriers)
        return self._pack_batch(datum_seed, datums)

    def _ready_datums(self) -> list[DatumRef]:
        # [FUNCTION DOC] HELPER: Refs for every not yet issued datum of every request in each
        # queue's leading batch run.
        datums = []
        for model_queue in self._model_queues.values():
            for request in model_queue.open_batch_run():
                datums.extend(
                    DatumRef(model_queue, request, index)
                    for index in range(request.num_issued_datums, len(request.datums))
                )
        return datums

    def _ready_barriers(self) -> list[tuple[ModelRequestQueue, PendingRequest]]:
        # [FUNCTION DOC] HELPER: (queue, barrier) pairs for queues whose head is an executable
        # barrier.
        return [
            (model_queue, barrier)
            for model_queue in self._model_queues.values()
            if (barrier := model_queue.ready_barrier()) is not None
        ]

    def _pack_batch(self, seed: DatumRef, datums: list[DatumRef]) -> BatchUnit:
        # [FUNCTION DOC] HELPER: Build a batch unit. Candidates are the ready datums whose request
        # pack key (op, loss function, config) equals the seed's, ordered by arrival then datum
        # index. Take candidates in order while the accumulated token count stays within the
        # budget; the first candidate is always taken even if alone over budget; the first one
        # that would overflow stops packing (later smaller ones are not tried). Each picked
        # datum's request has its issued counter incremented (mutation). The unit takes its
        # operation, loss function and config from the seed's command. Because candidates are
        # sorted by datum index within a request, the picked datums of each request always form a
        # prefix of its not yet issued datums, which keeps the issued counter consistent.
        pack_key = seed.request.pack_key()
        compatible = sorted(
            (ref for ref in datums if ref.request.pack_key() == pack_key),
            key=lambda ref: (ref.arrival, ref.local_index),
        )
        picked: list[DatumRef] = []
        tokens = 0
        for ref in compatible:
            datum_tokens = len(ref.datum["tokens"])
            if picked and tokens + datum_tokens > self.batch_token_budget:
                break
            picked.append(ref)
            tokens += datum_tokens
        for ref in picked:
            ref.request.num_issued_datums += 1
        command = seed.request.command
        return BatchUnit(
            op=command.op,
            loss_fn=command.payload.get("loss_fn"),
            loss_fn_config=command.payload.get("loss_fn_config"),
            datums=picked,
        )

    def _merge_barriers(
        self,
        seed: tuple[ModelRequestQueue, PendingRequest],
        barriers: list[tuple[ModelRequestQueue, PendingRequest]],
    ) -> BarrierUnit:
        # [FUNCTION DOC] HELPER: Build a barrier unit from the seed. For an optimizer step, include
        # every ready barrier of the same op across models so they step in one trainer call;
        # otherwise the unit holds only the seed.
        op = seed[1].command.op
        if op == CommandOp.OPTIM_STEP:
            # optim barriers of different models step in one trainer call
            entries = [(model_queue, barrier) for model_queue, barrier in barriers if barrier.command.op == op]
        else:
            entries = [seed]
        return BarrierUnit(op=op, entries=entries)
