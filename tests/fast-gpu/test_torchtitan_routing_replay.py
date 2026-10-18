# [MODULE DOC] GPU tests of the torchtitan routing-replay mechanism's queue alignment: the queued
# routing decisions must line up one entry per microbatch even though a pipeline schedule runs
# extra metadata forwards. Uses a minimal model part holding a real torchtitan top-k router
# subclass (so the installed replacement forward is exercised), a pytest fixture that configures the
# global replay manager in forward-replay mode and resets it afterwards, and a helper that queues
# one routing entry per microbatch (entry k assigns expert k everywhere). Tests cover: normal
# per-microbatch replay; the schedule's metadata forward not consuming entries (bypass); multiple
# passes sharing one queue; the bypass ending at the first real microbatch; error reporting for a
# lost recompute cursor and for a stray extra forward; and leaving non-actor parts alone.
"""The routing queues must line up with the microbatches, one entry each.

Nothing in the replay mechanism enforces that alignment: a queue serves whoever
asks next, so one unexpected forward shifts every later lookup by a microbatch.
That is not a crash and not obviously wrong in the metrics either -- adjacent
microbatches of a GRPO group share a prompt, so a shifted replay reproduces the
prompt's routing exactly and only diverges over the response.

The unexpected forward is real: a pipeline schedule infers the shapes its stages
exchange by running one forward per stage over microbatch 0, and repeats that
whenever the pass changes direction (in RL, every log-prob-then-train pair).
"""

from tests.ci.ci_register import register_cuda_ci

# Needs a GPU only because the shared replay queue hands its entries out on the
# current CUDA device; the alignment being tested is device-independent.
register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["torchtitan", "replay"], hardware=["hopper"])

import pytest
import torch
import torch.nn as nn

from torchtitan.models.common.moe import TokenChoiceTopKRouter

from miles.backends.torchtitan_utils import routing_replay
from miles.backends.training_utils.replay import routing_replay as shared_replay
from miles.utils.replay_base import routing_replay_manager


class _Gate(nn.Module):
    # [CLASS DOC] HELPER: Identity stand-in for a router's gate projection.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input unchanged, so the router's scores are the input.
        return x


class _Router(TokenChoiceTopKRouter):
    # [CLASS DOC] HELPER: Minimal router subclassing torchtitan's top-k router so the replay
    # installer recognizes it by type; carries only the attributes the router forward reads: identity
    # gate, softmax scoring, no expert groups, top-k 2, no route normalization, scale 1, and load
    # balancing override off.
    """A real router with the attributes its forward reads, and nothing else.

    Subclassing the real class matters: install() finds routers by type and
    rebinds torchtitan's own forward, so the queue is exercised through the code
    that actually runs in training.
    """

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes only the base module class (deliberately skipping the
        # router's own constructor, which would need real sizes) and sets the attributes above.
        nn.Module.__init__(self)
        self.gate = _Gate()
        self.score_func = "softmax"
        self.num_expert_groups = None
        self.top_k = 2
        self.route_norm = False
        self.route_scale = 1.0
        self._debug_force_load_balance = False


class _Part(nn.Module):
    # [CLASS DOC] HELPER: Fake pipeline-stage model part holding a module dictionary with one router
    # under key "0"; its forward returns the router's chosen expert ids.
    """A model part with one router, like a pipeline stage's submodule."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the module dictionary with a single fresh router.
        super().__init__()
        self.layers = nn.ModuleDict({"0": _Router()})

    def forward(self, scores):
        # [FUNCTION DOC] HELPER: Runs the router on the scores and returns only the second output
        # (the selected expert ids).
        _, expert_ids, _ = self.layers["0"](scores)
        return expert_ids


@pytest.fixture
def part():
    # [FUNCTION DOC] HELPER: Generator fixture. Setup: turns on the global replay manager with the
    # replay-result check off, sets its stage to forward replay, clears its replay list, creates a model
    # part and installs routing replay on it (which registers replay queues in the manager). Yields the
    # part. Teardown: disables the manager, clears the replays and sets the stage to fall-through.
    routing_replay_manager.enabled = True
    routing_replay_manager.enable_check_replay_result = False
    routing_replay_manager.stage = shared_replay.REPLAY_FORWARD
    routing_replay_manager.replays = []
    part = _Part()
    routing_replay.install([part])
    yield part
    routing_replay_manager.enabled = False
    routing_replay_manager.replays = []
    routing_replay_manager.stage = shared_replay.FALLTHROUGH


def _queue_microbatches(count):
    # [FUNCTION DOC] HELPER: Takes the first replay queue from the manager and records count
    # entries, each a 4 by 2 integer tensor filled with the microbatch index. Returns the queue.
    """Queue one entry per microbatch, entry k naming expert k."""
    replay = routing_replay_manager.replays[0]
    for k in range(count):
        replay.record(torch.full((4, 2), k, dtype=torch.long))
    return replay


def test_each_microbatch_replays_its_own_entry(part):
    # [FUNCTION DOC] Queues 3 entries; inside a consumption guard expecting 3 forwards, calls the part
    # three times on random scores and asserts the i-th call returns only expert id i.
    _queue_microbatches(3)
    scores = torch.rand(1, 4, 8, device="cuda")

    with routing_replay.consumption_guard([part], 3):
        for expected in range(3):
            assert part(scores).unique().tolist() == [expected]


def test_the_schedules_metadata_forward_does_not_consume(part):
    # [FUNCTION DOC] Queues 2 entries and, in a guard expecting 2 forwards, marks the schedule
    # initialization bypass, then runs one forward (the metadata probe) and asserts its result is
    # not entry 0 (the router chose by itself); then two real forwards must replay entries 0 and 1.
    _queue_microbatches(2)
    scores = torch.rand(1, 4, 8, device="cuda")

    with routing_replay.consumption_guard([part], 2):
        routing_replay.bypass_schedule_initialization([part])
        inferred = part(scores)
        # Bypassed: the router chose for itself rather than replaying entry 0.
        assert inferred.unique().tolist() != [0]

        for expected in range(2):
            assert part(scores).unique().tolist() == [expected]


def test_consecutive_passes_share_one_queue(part):
    # [FUNCTION DOC] Queues 4 entries and runs two consecutive guarded passes of 2 forwards each,
    # asserting the forwards replay entries 0,1 then 2,3 (the queue persists across passes).
    """The queues are filled per rollout and read across its optimizer steps, so
    a pass is held to how far it advanced, not to where it ended up."""
    _queue_microbatches(4)
    scores = torch.rand(1, 4, 8, device="cuda")

    for step in range(2):
        with routing_replay.consumption_guard([part], 2):
            for offset in range(2):
                assert part(scores).unique().tolist() == [step * 2 + offset]


def test_the_bypass_ends_at_the_first_real_microbatch(part):
    # [FUNCTION DOC] Queues 2 entries; in a guard of 2, bypasses initialization and runs the probing
    # forward, asserting the manager stage is fall-through afterward; then two real forwards replay
    # entries 0 and 1, and the stage is back to forward replay at the end.
    """The probing forward and the backward that follows it both fall through;
    the next forward is microbatch 0 and replays again."""
    _queue_microbatches(2)
    scores = torch.rand(1, 4, 8, device="cuda")

    with routing_replay.consumption_guard([part], 2):
        routing_replay.bypass_schedule_initialization([part])
        part(scores)
        assert routing_replay_manager.stage == shared_replay.FALLTHROUGH
        for expected in range(2):
            assert part(scores).unique().tolist() == [expected]
        assert routing_replay_manager.stage == shared_replay.REPLAY_FORWARD


def test_a_recompute_pass_that_lost_its_place_is_reported(part):
    # [FUNCTION DOC] Queues 3 entries, then inside a guard expecting 2 forwards pops two forward
    # entries and one backward (recompute) entry directly on the queue, so the recompute cursor has
    # advanced once instead of twice; asserts leaving the guard raises a runtime error mentioning that
    # the recompute pass is replaying.
    """Activation checkpointing reads a second cursor, one entry per microbatch.
    Anything else means the recompute is replaying other microbatches' routing --
    which is what the schedule's probing backward caused."""
    replay = _queue_microbatches(3)

    with pytest.raises(RuntimeError, match="recompute pass is replaying"):
        with routing_replay.consumption_guard([part], 2):
            replay.pop_forward()
            replay.pop_forward()
            replay.pop_backward()


def test_a_stray_forward_is_reported(part):
    # [FUNCTION DOC] Queues 3 entries, then runs 3 forwards inside a guard that expects a pass of 2;
    # asserts leaving the guard raises a runtime error saying the queue advanced 3 times over a pass of 2.
    _queue_microbatches(3)
    scores = torch.rand(1, 4, 8, device="cuda")

    with pytest.raises(RuntimeError, match="advanced 3 times over a pass of 2"):
        with routing_replay.consumption_guard([part], 2):
            for _ in range(3):
                part(scores)


def test_the_reference_model_is_left_alone(part):
    # [FUNCTION DOC] Queues 1 entry for the installed part, creates a second, never-installed part,
    # and enters a consumption guard on that other part with an expected pass of 7 forwards without
    # running anything; asserts no error is raised (no check applies to a part without replay queues).
    """Only the actor's routers replay; a second trainer's parts must not be
    held to the actor's queues."""
    _queue_microbatches(1)
    other = _Part()

    with routing_replay.consumption_guard([other], 7):
        pass


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
