# [MODULE DOC]
# Per-model ordered command queue for the Tinker gateway. Commands are released into the queue
# strictly in sequence id order (gaps hold later commands back). Batch commands (forward ops)
# at the head of the queue form a run whose datums may be issued in parallel; any other command is
# a barrier that can run only after all batch commands ahead of it finished. Commands carrying a
# validation error are queued too but never issued; the dispatcher fails them when they reach head.
"""Per-model command ordering and batch runs separated by barriers."""

from collections import deque
from dataclasses import dataclass, field

from miles.tinker.core.types import Command


@dataclass
class PendingRequest:
    # [CLASS DOC] A queued command with accounting for batch ops: the datum list (taken from the
    # payload), how many datums were already handed to the scheduler, one output slot per datum
    # (none until reported), and the count of datums still awaiting output. Non-batch commands
    # and commands with a validation error keep these empty.
    """One submitted command and its completion accounting."""

    command: Command
    datums: list[dict] = field(default_factory=list)  # batch ops only
    num_issued_datums: int = 0
    outputs: list[dict | None] = field(default_factory=list)
    remaining: int = 0

    @property
    def is_batch_op(self) -> bool:
        # [FUNCTION DOC] Property: whether the command's operation is a batch op.
        return self.command.op.is_batch()

    def record_output(self, local_index: int, output: dict) -> bool:
        # [FUNCTION DOC] Save the output of the datum at the local index, decrement the remaining
        # count, and return true when no datums remain (the request is complete). An index out of
        # range raises an index error; reporting the same datum twice would miscount.
        """Store one datum's result; True once every datum has reported."""
        self.outputs[local_index] = output
        self.remaining -= 1
        return self.remaining == 0

    def pack_key(self) -> tuple:
        # [FUNCTION DOC] Compatibility key for packing datums from different requests into one
        # trainer batch: (operation, loss function name, sorted tuple of loss config items; empty
        # when no config). Requires the payload to hold a loss function name (key error otherwise)
        # and config values to be hashable for later comparison/grouping.
        """Datums pack into one BatchUnit only within the same (op, loss_fn, config)."""
        config = self.command.payload.get("loss_fn_config") or {}
        return (self.command.op, self.command.payload["loss_fn"], tuple(sorted(config.items())))


class ModelRequestQueue:
    # [CLASS DOC] Ordered queue for one model: identity (model id, tenant, adapter slot), the
    # highest sequence id already moved into the queue, a table of commands waiting for earlier
    # sequence ids, and the double-ended queue of pending requests in sequence order.
    def __init__(self, model_id: str, tenant: str, slot: int) -> None:
        # [FUNCTION DOC] Record the identity; start with last enqueued sequence id 0 (ids are
        # 1-based) and empty tables.
        self.model_id = model_id
        self.tenant = tenant
        self.slot = slot
        # seq_ids are 1-based
        self.last_enqueued_seq_id = 0
        self.pending_by_seq: dict[int, Command] = {}
        self.queue: deque[PendingRequest] = deque()

    def submit(self, command: Command) -> None:
        # [FUNCTION DOC] Hold the command by its sequence id and release every command that is now
        # contiguous with the last released id. Submitting an id twice would overwrite a held
        # command (callers deduplicate beforehand); ids at or below the released id would stay stuck.
        """Accept one deduplicated command; feed the queue in seq order."""
        self.pending_by_seq[command.seq_id] = command
        self._enqueue_contiguous_commands()

    def _enqueue_contiguous_commands(self) -> None:
        # [FUNCTION DOC] HELPER: While the next sequence id is held, move that command into the
        # queue as a pending request. Valid batch commands get their datum list from the payload,
        # the remaining count set to the datum count, and one empty output slot per datum.
        while self.last_enqueued_seq_id + 1 in self.pending_by_seq:
            self.last_enqueued_seq_id += 1
            next_command = self.pending_by_seq.pop(self.last_enqueued_seq_id)
            pending = PendingRequest(command=next_command)
            if pending.is_batch_op and pending.command.validation_error is None:
                pending.datums = next_command.payload["datums"]
                pending.remaining = len(pending.datums)
                pending.outputs = [None] * len(pending.datums)
            self.queue.append(pending)

    def open_batch_run(self) -> list[PendingRequest]:
        # [FUNCTION DOC] Walk the queue from the head and collect consecutive batch requests
        # without a validation error; stop at the first barrier or errored request. Read-only.
        """The leading run of batch-op commands; their datums are all issuable."""
        run = []
        for pending in self.queue:
            if not pending.is_batch_op or pending.command.validation_error is not None:
                break
            run.append(pending)
        return run

    def ready_barrier(self) -> PendingRequest | None:
        # [FUNCTION DOC] The head request if it exists, is not a batch op and has no validation
        # error (batch requests ahead of a barrier are finished and removed, so a barrier at head
        # means everything before it completed); otherwise none.
        """The head barrier, executable once every batch op ahead of it completed."""
        if self.queue and not self.queue[0].is_batch_op and self.queue[0].command.validation_error is None:
            return self.queue[0]
        return None

    def finish(self, pending: PendingRequest) -> None:
        # [FUNCTION DOC] Remove the request from the queue (raises a value error if absent). Any
        # position may be removed, since batch requests can complete out of order.
        self.queue.remove(pending)
