# [MODULE DOC] Unit tests of the per-model request queue in the tinker core: commands are queued
# in sequence-id order regardless of arrival order, a gap in sequence ids holds later commands,
# and batch runs (consecutive forward/backward commands) alternate with barrier commands (such
# as optimizer steps), where a barrier is ready only after the preceding batch run has finished.
"""Model queues order commands and hold barriers until all preceding datums finish."""

from tests.fast.tinker.harness import command, datum, fb_payload

from miles.tinker.core.model_queue import ModelRequestQueue


def _model_queue() -> ModelRequestQueue:
    # [FUNCTION DOC] HELPER: Creates a queue for model "model", tenant "tenant", slot 0.
    return ModelRequestQueue("model", "tenant", slot=0)


def test_out_of_order_arrival_queues_in_seq_order():
    # [FUNCTION DOC] Submitting sequence 2 first leaves the queue empty (it waits for 1);
    # submitting sequence 1 then queues both in order 1, 2.
    model_queue = _model_queue()
    model_queue.submit(command("model", 2, "forward_backward", fb_payload("model", 2, [datum()]), arrival=1))
    assert not model_queue.queue, "seq 2 must wait for seq 1"
    model_queue.submit(command("model", 1, "forward_backward", fb_payload("model", 1, [datum()]), arrival=2))
    assert [pending.command.seq_id for pending in model_queue.queue] == [1, 2]


def test_a_gap_holds_later_commands():
    # [FUNCTION DOC] Submitting sequences 1 and 3 queues only 1; adding 2 releases all three in order.
    model_queue = _model_queue()
    model_queue.submit(command("model", 1, "forward_backward", fb_payload("model", 1, [datum()]), arrival=1))
    model_queue.submit(command("model", 3, "forward_backward", fb_payload("model", 3, [datum()]), arrival=2))
    assert [pending.command.seq_id for pending in model_queue.queue] == [1]
    model_queue.submit(command("model", 2, "forward_backward", fb_payload("model", 2, [datum()]), arrival=3))
    assert [pending.command.seq_id for pending in model_queue.queue] == [1, 2, 3]


def test_batch_runs_alternate_with_barriers():
    # [FUNCTION DOC] Submits two forward/backward commands, an optimizer step (barrier) and
    # another forward/backward. The open batch run is sequences 1 and 2 and no barrier is ready.
    # After finishing the run, the barrier (sequence 3) is ready and no batch run is open; after
    # finishing the barrier the open batch run is sequence 4.
    model_queue = _model_queue()
    model_queue.submit(command("model", 1, "forward_backward", fb_payload("model", 1, [datum()]), arrival=1))
    model_queue.submit(command("model", 2, "forward_backward", fb_payload("model", 2, [datum()]), arrival=2))
    model_queue.submit(
        command("model", 3, "optim_step", {"model_id": "model", "seq_id": 3, "adam_params": {}}, arrival=3)
    )
    model_queue.submit(command("model", 4, "forward_backward", fb_payload("model", 4, [datum()]), arrival=4))

    assert [pending.command.seq_id for pending in model_queue.open_batch_run()] == [1, 2]
    assert model_queue.ready_barrier() is None, "the barrier must wait for its batch run"

    for pending in list(model_queue.open_batch_run()):
        model_queue.finish(pending)
    assert model_queue.ready_barrier().command.seq_id == 3
    assert not model_queue.open_batch_run(), "the next batch run opens only after the barrier"

    model_queue.finish(model_queue.ready_barrier())
    assert [pending.command.seq_id for pending in model_queue.open_batch_run()] == [4]
