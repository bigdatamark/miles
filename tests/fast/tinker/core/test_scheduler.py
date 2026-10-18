# [MODULE DOC] Unit tests of the tinker request scheduler: datums with the same pack key (same
# loss function) from different model queues are packed into one batch unit within a token
# budget; different losses do not pack; large requests are split by budget while an oversized
# datum still ships alone; the oldest arrival is scheduled first; optimizer barriers from
# different models merge while save barriers do not; issued datums are not reissued; and
# forward-only work packs only within one loss.
"""The scheduler selects the oldest ready seed and packs compatible work within its budget."""

from tests.fast.tinker.harness import ADAM, command, datum, fb_payload

from miles.tinker.core.model_queue import ModelRequestQueue
from miles.tinker.core.scheduler import BarrierUnit, BatchUnit, RequestScheduler


def _scheduler_with_model_queues(n: int, budget: int = 1_000_000) -> tuple[RequestScheduler, list[ModelRequestQueue]]:
    # [FUNCTION DOC] HELPER: Creates a scheduler with the batch token budget (default one million)
    # and n model queues (models model-0.., tenants tenant-0.., slots 0..) registered in order.
    # Returns the scheduler and the list of queues.
    scheduler = RequestScheduler(budget)
    model_queues = [ModelRequestQueue(f"model-{i}", f"tenant-{i}", slot=i) for i in range(n)]
    for model_queue in model_queues:
        scheduler.add_model_queue(model_queue)
    return scheduler, model_queues


def _submit_fb(model_queue: ModelRequestQueue, seq_id: int, arrival: int, datums: list[dict], loss_fn="cross_entropy"):
    # [FUNCTION DOC] HELPER: Submits a forward/backward command with the given sequence id,
    # arrival order, datums and loss function (default cross entropy) to the model queue.
    model_queue.submit(
        command(
            model_queue.model_id,
            seq_id,
            "forward_backward",
            fb_payload(model_queue.model_id, seq_id, datums, loss_fn),
            arrival,
        )
    )


def _submit_optim(model_queue: ModelRequestQueue, seq_id: int, arrival: int):
    # [FUNCTION DOC] HELPER: Submits an optimizer-step command with the default Adam parameters
    # to the model queue.
    payload = {"model_id": model_queue.model_id, "seq_id": seq_id, "adam_params": dict(ADAM)}
    model_queue.submit(command(model_queue.model_id, seq_id, "optim_step", payload, arrival))


def test_same_pack_key_datums_pack_across_model_queues():
    # [FUNCTION DOC] Two models submit cross-entropy work (2 and 1 datums); the next scheduled
    # unit is a batch unit with 3 datums from both models.
    scheduler, (a, b) = _scheduler_with_model_queues(2)
    _submit_fb(a, 1, arrival=1, datums=[datum(), datum()])
    _submit_fb(b, 1, arrival=2, datums=[datum()])

    unit = scheduler.schedule_next()
    assert isinstance(unit, BatchUnit)
    assert len(unit.datums) == 3
    assert {ref.model_queue.model_id for ref in unit.datums} == {"model-0", "model-1"}


def test_different_loss_functions_do_not_pack():
    # [FUNCTION DOC] Cross-entropy work from model 0 (older) and ppo work from model 1: the first
    # unit contains only model 0's datum.
    scheduler, (a, b) = _scheduler_with_model_queues(2)
    _submit_fb(a, 1, arrival=1, datums=[datum()], loss_fn="cross_entropy")
    _submit_fb(b, 1, arrival=2, datums=[datum()], loss_fn="ppo")

    unit = scheduler.schedule_next()
    assert [ref.model_queue.model_id for ref in unit.datums] == ["model-0"]


def test_the_token_budget_splits_a_large_request():
    # [FUNCTION DOC] With a token budget of 10 and three datums of 5 tokens each, two scheduled
    # units hold 2 and 1 datums respectively.
    scheduler, (a,) = _scheduler_with_model_queues(1, budget=10)
    _submit_fb(a, 1, arrival=1, datums=[datum(4), datum(4), datum(4)])  # 5 tokens each with the appended target

    first, second = scheduler.schedule_next(), scheduler.schedule_next()
    assert [len(first.datums), len(second.datums)] == [2, 1]


def test_an_oversized_datum_still_ships_alone():
    # [FUNCTION DOC] With a budget of 2 and a single 10-token datum, the scheduled unit still has
    # that one datum.
    scheduler, (a,) = _scheduler_with_model_queues(1, budget=2)
    _submit_fb(a, 1, arrival=1, datums=[datum(10)])

    assert len(scheduler.schedule_next().datums) == 1


def test_the_oldest_arrival_wins():
    # [FUNCTION DOC] An optimizer step arriving first beats forward/backward work arriving later:
    # the first scheduled unit is a barrier unit.
    scheduler, (a, b) = _scheduler_with_model_queues(2)
    _submit_optim(a, 1, arrival=1)
    _submit_fb(b, 1, arrival=2, datums=[datum()])

    assert isinstance(scheduler.schedule_next(), BarrierUnit)


def test_optim_barriers_merge_and_save_does_not():
    # [FUNCTION DOC] Two models submit optimizer steps and a third a save-state command. The first
    # unit is a merged optimizer-step barrier with entries for the first two models; after
    # finishing those entries, the next unit is a save-state barrier with a single entry.
    scheduler, (a, b, c) = _scheduler_with_model_queues(3)
    _submit_optim(a, 1, arrival=1)
    _submit_optim(b, 1, arrival=2)
    c.submit(command("model-2", 1, "save_state", {"model_id": "model-2", "seq_id": 1, "name": "x"}, arrival=3))

    merged = scheduler.schedule_next()
    assert merged.op == "optim_step"
    assert {model_queue.model_id for model_queue, _ in merged.entries} == {"model-0", "model-1"}

    for model_queue, pending in merged.entries:
        model_queue.finish(pending)
    save = scheduler.schedule_next()
    assert save.op == "save_state"
    assert len(save.entries) == 1


def test_issued_datums_are_not_reissued():
    # [FUNCTION DOC] After scheduling a single datum, the next schedule call returns none.
    scheduler, (a,) = _scheduler_with_model_queues(1)
    _submit_fb(a, 1, arrival=1, datums=[datum()])

    assert len(scheduler.schedule_next().datums) == 1
    assert scheduler.schedule_next() is None


def test_forward_only_packs_only_within_one_loss():
    # [FUNCTION DOC] Two models submit one forward-only request each with different losses (cross
    # entropy and dro) to a scheduler with a large token budget. The first unit contains only the
    # cross-entropy request's datum, so a dro request never runs under another request's loss.
    scheduler = RequestScheduler(batch_token_budget=1 << 20)
    for index, (model, loss_fn) in enumerate([("m1", "cross_entropy"), ("m2", "dro")], start=1):
        model_queue = ModelRequestQueue(model, "tenant", slot=index)
        scheduler.add_model_queue(model_queue)
        model_queue.submit(
            command(
                model,
                1,
                "forward_only",
                {"datums": [{"tokens": [1, 2], "target_len": 1}], "loss_fn": loss_fn, "loss_fn_config": {}},
                arrival=index,
            )
        )
    first = scheduler.schedule_next()
    assert [ref.request.command.payload["loss_fn"] for ref in first.datums] == [
        "cross_entropy"
    ], "a DRO request must not execute under another request's loss"
