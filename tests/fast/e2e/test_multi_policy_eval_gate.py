# [MODULE DOC]
# Tests for the eval-score learning gate of the multi-policy end-to-end support module: the function
# asserting that a policy's evaluation score series on a dataset shows learning, given bounds with a
# maximum for the first point, a minimum for the peak, and an optional minimum growth from first to
# peak. The series reader is replaced so tests supply score lists directly; the events directory
# argument is unused. Failure cases check the assertion messages.
from pathlib import Path

import pytest

import tests.e2e.conftest_multi_policy as conftest_multi_policy
from tests.e2e.conftest_multi_policy import EvalScoreBounds


def _patch_scores(monkeypatch: pytest.MonkeyPatch, scores: list[float]) -> None:
    # [FUNCTION DOC] HELPER: Replaces the module's eval score series reader with one that ignores
    # its arguments and returns the given list.
    monkeypatch.setattr(conftest_multi_policy, "_read_eval_score_series", lambda *_args, **_kwargs: scores)


def _assert_learned(bounds: EvalScoreBounds) -> None:
    # [FUNCTION DOC] HELPER: Runs the learning assertion with an unused events path, the given
    # bounds for the solver policy, and the gsm8k dataset name. Raises whatever the assertion raises.
    conftest_multi_policy.assert_policy_eval_scores_learned(
        Path("unused"), bounds={"solver": bounds}, dataset_name="gsm8k"
    )


def test_peak_not_last_point_passes(monkeypatch):
    # [FUNCTION DOC] Series 0.45, 0.55, 0.60, 0.52 with first-point maximum 0.5, peak minimum 0.58
    # and growth minimum 0.1 passes: the peak 0.60 is used, not the last value.
    """A series that dips after its best point is judged on the peak, not the final value."""
    _patch_scores(monkeypatch, [0.45, 0.55, 0.60, 0.52])
    _assert_learned(EvalScoreBounds(initial_max=0.5, peak_min=0.58, min_growth=0.1))


def test_starting_solved_fails(monkeypatch):
    # [FUNCTION DOC] First point 0.9 exceeds the 0.5 first-point maximum; asserts an assertion
    # error saying the policy starts already solved.
    """A first point above initial_max fails because it cannot demonstrate learning."""
    _patch_scores(monkeypatch, [0.9, 0.95])
    with pytest.raises(AssertionError, match="starts already solved"):
        _assert_learned(EvalScoreBounds(initial_max=0.5, peak_min=0.6))


def test_peak_below_threshold_fails(monkeypatch):
    # [FUNCTION DOC] Peak 0.45 below the 0.6 peak minimum; asserts an assertion error mentioning the peak eval score.
    """A peak below peak_min fails the accuracy gate."""
    _patch_scores(monkeypatch, [0.4, 0.45])
    with pytest.raises(AssertionError, match="peak eval score"):
        _assert_learned(EvalScoreBounds(initial_max=0.5, peak_min=0.6))


def test_growth_below_threshold_fails(monkeypatch):
    # [FUNCTION DOC] Growth 0.05 (0.55 to 0.60) below the 0.2 growth minimum while the other
    # bounds hold; asserts an assertion error mentioning eval growth.
    """A peak-minus-first gap below min_growth fails the growth gate."""
    _patch_scores(monkeypatch, [0.55, 0.60])
    with pytest.raises(AssertionError, match="eval growth"):
        _assert_learned(EvalScoreBounds(initial_max=0.6, peak_min=0.58, min_growth=0.2))


def test_none_growth_skips_growth_gate(monkeypatch):
    # [FUNCTION DOC] Flat series 0.6, 0.6 with growth minimum disabled (none) passes.
    """min_growth=None requires no growth at all."""
    _patch_scores(monkeypatch, [0.6, 0.6])
    _assert_learned(EvalScoreBounds(initial_max=0.65, peak_min=0.58, min_growth=None))


def test_no_points_fails(monkeypatch):
    # [FUNCTION DOC] Empty series; asserts an assertion error saying there are no eval points for gsm8k solver.
    """An empty eval series fails loudly instead of passing vacuously."""
    _patch_scores(monkeypatch, [])
    with pytest.raises(AssertionError, match="no eval/gsm8k/solver points"):
        _assert_learned(EvalScoreBounds(initial_max=0.5, peak_min=0.6))
