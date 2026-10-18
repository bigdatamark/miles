# [MODULE DOC] CPU unit tests for the on-policy-distillation (OPD) advantage adjustment:
# subtraction of the weighted reverse KL (student minus teacher log-probs) from per-token
# advantages with the metric stored in rollout data, detaching of inputs and outputs so no
# gradient leaks into persistent rollout data, use of a precomputed reverse KL, a no-op without
# student log-probs, and the guard-rail errors (missing teacher log-probs, sample count mismatch,
# per-sample scalar advantage shape mismatch).
"""Unit tests for the decoupled on-policy-distillation (OPD) loss path.

`apply_opd_kl_to_advantages` is orthogonal to the advantage estimator: it adds a
reverse-KL penalty (student_logp - teacher_logp) to per-token advantages. These
tests cover the math and the guard rails without needing the external loss
snapshot artifacts.
"""

from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.loss import objective as loss_utils
from miles.backends.training_utils.loss.hub.opd import apply_opd_kl_to_advantages

from .loss_test_utils import make_parallel_state

# This module intentionally has no explicit CI registration call: modules under
# tests/fast are implicitly assigned to the stage-a-cpu suite by the CI collector
# (an explicit default-form call would be rejected by the AC-9 meta-test).


def _args(opd_kl_coef: float = 1.0) -> Namespace:
    # [FUNCTION DOC] HELPER: Returns args with OPD enabled, type sglang and the given KL
    # coefficient (default 1.0).
    return Namespace(use_opd=True, opd_type="sglang", opd_kl_coef=opd_kl_coef)


def test_subtracts_weighted_reverse_kl_and_stores_metric():
    # [FUNCTION DOC] Test: coefficient 0.5, student log-probs [0, 1] and teacher [0, 0] (both
    # requiring grad), advantages [2, 2]. After the call advantages are [2.0, 1.5] (reverse KL
    # [0, 1] weighted by 0.5 subtracted), the stored reverse KL is [0, 1], neither requires grad.
    # Backpropagating advantages times fresh log-probs gives that tensor a gradient equal to the
    # advantages, and no gradient reaches the student or teacher tensors.
    args = _args(opd_kl_coef=0.5)
    student = [torch.tensor([0.0, 1.0], requires_grad=True)]
    teacher = [torch.tensor([0.0, 0.0], requires_grad=True)]
    advantages = [torch.tensor([2.0, 2.0])]
    rollout_data = {"teacher_log_probs": teacher}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student)

    # reverse_kl = student - teacher = [0, 1]; adv - 0.5 * reverse_kl = [2.0, 1.5]
    assert torch.allclose(advantages[0], torch.tensor([2.0, 1.5]))
    assert torch.allclose(rollout_data["opd_reverse_kl"][0], torch.tensor([0.0, 1.0]))
    assert advantages[0].requires_grad is False
    assert rollout_data["opd_reverse_kl"][0].requires_grad is False

    current_student_log_probs = torch.tensor([0.2, -0.3], requires_grad=True)
    (advantages[0] * current_student_log_probs).sum().backward()
    torch.testing.assert_close(current_student_log_probs.grad, advantages[0])
    assert student[0].grad is None
    assert teacher[0].grad is None


def test_precomputed_reverse_kl_is_detached_before_weighting_advantages():
    # [FUNCTION DOC] Test: coefficient 0.25 and a precomputed reverse KL [0.4, -0.2] requiring grad
    # already in rollout data; zero advantages. Advantages become [-0.1, 0.05] and stop requiring
    # grad, the stored reverse KL is detached, and a backward through advantages times fresh
    # log-probs gives gradients equal to the advantages and none to the precomputed tensor.
    args = _args(opd_kl_coef=0.25)
    precomputed = torch.tensor([0.4, -0.2], requires_grad=True)
    advantages = [torch.tensor([0.0, 0.0])]
    rollout_data = {"opd_reverse_kl": [precomputed]}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs=[torch.zeros(2)])

    torch.testing.assert_close(advantages[0], torch.tensor([-0.1, 0.05]))
    assert advantages[0].requires_grad is False
    assert rollout_data["opd_reverse_kl"][0].requires_grad is False

    current_student_log_probs = torch.tensor([0.3, -0.1], requires_grad=True)
    (advantages[0] * current_student_log_probs).sum().backward()
    torch.testing.assert_close(current_student_log_probs.grad, advantages[0])
    assert precomputed.grad is None


def test_fixed_opd_inputs_are_detached_in_persistent_rollout_data(monkeypatch):
    # [FUNCTION DOC] Test: rollout data whose log-prob tensors (old, rollout, reference, teacher)
    # are computed from leaf tensors requiring grad, with OPD on (coefficient 0.5) and KL
    # coefficient 0. The advantage estimator is replaced by a fake that asserts its log-probs input
    # has no gradient history and returns zero advantages and returns. After running the full
    # advantage-and-returns routine, log-probs, rollout log-probs, reference log-probs, teacher
    # log-probs, the OPD reverse KL and the advantages all lack gradient history and do not require
    # grad, and none of the source leaf tensors received a gradient.
    make_parallel_state()
    old_source = torch.tensor([0.2, 0.4], requires_grad=True)
    rollout_source = torch.tensor([0.3, 0.5], requires_grad=True)
    reference_source = torch.tensor([0.4, 0.6], requires_grad=True)
    teacher_source = torch.tensor([0.1, 0.2], requires_grad=True)
    rollout_data = {
        "log_probs": [old_source.sin()],
        "rollout_log_probs": [rollout_source.cos()],
        "ref_log_probs": [reference_source.exp()],
        "teacher_log_probs": [teacher_source.square()],
        "rewards": [0.0],
        "values": None,
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "total_lengths": [2],
    }
    args = Namespace(
        skip_actor_forward_only=False,
        use_rollout_logprobs=False,
        kl_coef=0.0,
        use_opd=True,
        opd_type="sglang",
        opd_kl_coef=0.5,
        normalize_advantages=False,
    )

    def fake_compute_advantages(**kwargs):
        # [FUNCTION DOC] HELPER: Fake estimator; asserts the first log-probs entry has no gradient
        # function, then returns one all-zero advantages tensor and one zero returns tensor shaped
        # like it.
        assert kwargs["log_probs"][0].grad_fn is None
        zeros = torch.zeros_like(kwargs["log_probs"][0])
        return [zeros], [zeros.clone()]

    monkeypatch.setattr(loss_utils, "compute_advantages", fake_compute_advantages)

    loss_utils.compute_advantages_and_returns(args, rollout_data)

    for key in (
        "log_probs",
        "rollout_log_probs",
        "ref_log_probs",
        "teacher_log_probs",
        "opd_reverse_kl",
        "advantages",
    ):
        assert rollout_data[key][0].grad_fn is None
        assert rollout_data[key][0].requires_grad is False

    assert old_source.grad is None
    assert rollout_source.grad is None
    assert reference_source.grad is None
    assert teacher_source.grad is None


def test_noop_when_student_log_probs_none():
    # [FUNCTION DOC] Test: with no student log-probs the call leaves advantages unchanged ([1, 2])
    # and stores no reverse KL in rollout data.
    args = _args()
    advantages = [torch.tensor([1.0, 2.0])]
    rollout_data = {"teacher_log_probs": [torch.tensor([0.0, 0.0])]}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, None)

    assert torch.allclose(advantages[0], torch.tensor([1.0, 2.0]))
    assert "opd_reverse_kl" not in rollout_data


def test_raises_when_teacher_log_probs_missing():
    # [FUNCTION DOC] Test: empty rollout data (no teacher log-probs) raises a value error saying
    # teacher log-probs are required.
    args = _args()
    with pytest.raises(ValueError, match="requires teacher_log_probs"):
        apply_opd_kl_to_advantages(args, {}, [torch.tensor([1.0])], [torch.tensor([1.0])])


def test_raises_on_length_mismatch():
    # [FUNCTION DOC] Test: one teacher sample against two advantage and two student samples raises
    # a value error reporting an OPD length mismatch.
    args = _args()
    rollout_data = {"teacher_log_probs": [torch.tensor([0.0])]}  # 1 sample
    advantages = [torch.tensor([1.0]), torch.tensor([1.0])]  # 2 samples
    student = [torch.tensor([1.0]), torch.tensor([1.0])]

    with pytest.raises(ValueError, match="OPD length mismatch"):
        apply_opd_kl_to_advantages(args, rollout_data, advantages, student)


def test_raises_on_scalar_advantage_broadcast_trap():
    # [FUNCTION DOC] Test: a per-sample scalar advantage (shape 1) against per-token student and
    # teacher log-probs (shape 2) raises a value error reporting an OPD shape mismatch rather than
    # silently broadcasting.
    # GRPO-style per-sample scalar advantage must be expanded to per-token first.
    args = _args()
    student = [torch.tensor([0.0, 1.0])]
    teacher = [torch.tensor([0.0, 0.0])]
    advantages = [torch.tensor([2.0])]  # shape (1,) != student shape (2,)
    rollout_data = {"teacher_log_probs": teacher}

    with pytest.raises(ValueError, match="OPD shape mismatch"):
        apply_opd_kl_to_advantages(args, rollout_data, advantages, student)
