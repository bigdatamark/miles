# [MODULE DOC]
# Tests for the base replay manager's top-k function wrapper, which replays recorded routing
# choices (top-k expert indices) instead of recomputing them. Registers as a CPU CI test (estimated
# 60 s, stage-a-cpu, no labels) at import. In replay-forward stage, the wrapped function returns
# the recorded indices; rows that are entirely -1 (masked or pad tokens) are replaced by consecutive
# positions starting at zero (arange of the top-k size), while rows with some valid entries keep
# their -1 padding.
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import torch

from miles.utils.replay_base import BaseReplayManager


class _FakeReplay:
    # [CLASS DOC] HELPER: Fake replay source holding a queue of recorded top-index tensors.
    def __init__(self, *top_indices):
        # [FUNCTION DOC] HELPER: Store the given recorded tensors as a list queue.
        self.top_indices = list(top_indices)

    def pop_forward(self):
        # [FUNCTION DOC] HELPER: Remove and return the first recorded tensor (index error if empty).
        return self.top_indices.pop(0)

    def pop_backward(self):
        # [FUNCTION DOC] HELPER: Same as the forward pop.
        return self.pop_forward()


def _topk(scores, topk):
    # [FUNCTION DOC] HELPER: Real top-k along dimension one, returning only the indices as 32-bit
    # integers.
    return torch.topk(scores, topk, dim=1).indices.to(torch.int32)


def _make_replay_manager(top_indices):
    # [FUNCTION DOC] HELPER: Build a base replay manager with replay checking disabled, enabled
    # state on, stage "replay_forward", and a fake replay holding the given recorded indices set as
    # the current replay.
    manager = BaseReplayManager()
    manager.enable_check_replay_result = False
    manager.enabled = True
    manager.stage = "replay_forward"
    manager.set_current(_FakeReplay(top_indices))
    return manager


def test_get_topk_fn_fills_all_invalid_rows_with_arange():
    # [FUNCTION DOC] Recorded indices are one row of three -1 values; the wrapped top-k function
    # (no probabilities) on one row of five scores with k 3 returns [[0, 1, 2]] as 32-bit ints.
    # an all-(-1) row is a masked/pad token; fill it with arange to avoid reading
    # invalid (-1) positions downstream
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    manager = _make_replay_manager(torch.tensor([[-1, -1, -1]], dtype=torch.int32))

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), torch.tensor([[0, 1, 2]], dtype=torch.int32))


def test_get_topk_fn_preserves_partial_padding():
    # [FUNCTION DOC] Recorded indices [[2, -1, -1]]: the wrapped function returns them unchanged.
    # a row with some valid picks keeps its -1 padding (only all-(-1) rows are filled)
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    replayed_top_indices = torch.tensor([[2, -1, -1]], dtype=torch.int32)
    manager = _make_replay_manager(replayed_top_indices)

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), replayed_top_indices)
