# [MODULE DOC]
# Tests for the per-slot LoRA optimizer stepping function: with duck-typed fake slot optimizers
# it verifies that a nonfinite gradient norm skips the step and zeroes gradients, and that
# unknown failures during gradient preparation or the clip-and-step stage propagate instead of
# being swallowed (and prevent parameter all-gather from being reported as success).
"""Slot steps preserve nonfinite skips and propagate unknown execution failures."""

import pytest

from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer, step_slot_optimizers


class _FakeSlotOptimizer(SlotOptimizer):
    # [CLASS DOC] HELPER: Fake slot optimizer subclass that skips the real constructor and records
    # which stages ran. It can be configured to raise an error in gradient preparation and to
    # return a chosen outcome (or raise an error) from the clip-and-step stage.
    """Duck instance: exercises the choreography without a megatron build."""

    def __init__(self, slot, *, prepare_error=None, step_outcome=None):
        # [FUNCTION DOC] HELPER: Stores the slot index and the optional error to raise during
        # preparation; the step outcome defaults to a mapping with grad norm 1.5. Initializes
        # recorded state: no adam params, not stepped, not zeroed, zero parameter gathers.
        self.slot = slot
        self._prepare_error = prepare_error
        self._step_outcome = step_outcome or {"grad_norm": 1.5}
        self.adam_params = None
        self.stepped = False
        self.zeroed = False
        self.param_gathers = 0

    def apply_adam_params(self, adam_params):
        # [FUNCTION DOC] HELPER: Records the received optimizer hyperparameters.
        self.adam_params = adam_params

    def prepare_grads(self):
        # [FUNCTION DOC] HELPER: Raises the configured preparation error if any; otherwise does
        # nothing.
        if self._prepare_error is not None:
            raise self._prepare_error

    def clip_and_step(self, clip_grad):
        # [FUNCTION DOC] HELPER: If the configured outcome has an error entry, raises a runtime
        # error with that message. Otherwise marks stepped true exactly when the outcome carries a
        # grad norm entry, and returns the outcome mapping. The clip value is ignored.
        if "error" in self._step_outcome:
            raise RuntimeError(self._step_outcome["error"])
        self.stepped = "grad_norm" in self._step_outcome
        return self._step_outcome

    def zero_grads(self):
        # [FUNCTION DOC] HELPER: Marks gradients as zeroed.
        self.zeroed = True

    def allgather_params(self):
        # [FUNCTION DOC] HELPER: Counts a parameter all-gather.
        self.param_gathers += 1


def _step(slot_optimizers):
    # [FUNCTION DOC] HELPER: Builds a per-slot hyperparameter mapping (learning rate 1e-4, gradient
    # clip norm 1.0 for each slot key) and calls the slot-stepping function with the optimizers
    # mapping and that mapping. Returns its per-slot outcomes.
    adam = {slot: {"learning_rate": 1e-4, "grad_clip_norm": 1.0} for slot in slot_optimizers}
    return step_slot_optimizers(slot_optimizers, adam)


def test_an_unknown_step_failure_escapes_before_success_is_reported():
    # [FUNCTION DOC] Two slots: slot 0 healthy, slot 1 whose clip-and-step raises "boom". Stepping
    # both raises a runtime error matching boom. Asserts the healthy slot did step but no
    # parameter all-gather happened for it (success is not reported after a failure).
    healthy = _FakeSlotOptimizer(0)
    failing = _FakeSlotOptimizer(1, step_outcome={"error": "boom"})
    with pytest.raises(RuntimeError, match="boom"):
        _step({0: healthy, 1: failing})
    assert healthy.stepped
    assert healthy.param_gathers == 0


def test_a_nonfinite_grad_norm_skips_the_step():
    # [FUNCTION DOC] A slot whose clip-and-step returns a skipped-nonfinite outcome. Asserts the
    # per-slot result is exactly that outcome, the slot did not step, its gradients were zeroed
    # and no parameter all-gather occurred.
    """BF16 has no grad scaler, so the all-reduced norm is the only inf/nan gate."""
    skipped = _FakeSlotOptimizer(0, step_outcome={"skipped_nonfinite": 1.0})
    outcomes = _step({0: skipped})
    assert outcomes[0] == {"skipped_nonfinite": 1.0}
    assert not skipped.stepped and skipped.zeroed and skipped.param_gathers == 0


def test_an_unknown_preparation_failure_escapes():
    # [FUNCTION DOC] Slot 0 raises a runtime error "prep died" while preparing gradients, slot 1
    # is healthy. Stepping both raises a runtime error matching that message; asserts neither
    # slot stepped.
    broken = _FakeSlotOptimizer(0, prepare_error=RuntimeError("prep died"))
    healthy = _FakeSlotOptimizer(1)
    with pytest.raises(RuntimeError, match="prep died"):
        _step({0: broken, 1: healthy})
    assert not broken.stepped and not healthy.stepped
