# doc-dev: docs/developer/ci/03-metric-history-gate.md
# [MODULE DOC] Pure band-constraint logic for the regression gate: a result type holding the verdict
# and the corridor, the parse-time parameter schema (four non-negative float parameters, all
# defaulting to zero) consumed by the registration parser, and the evaluation function. No I/O.
"""The band constraint for the CI regression gate.

* A constraint decides whether one extracted scalar `cur` passes against a
  reference `ref` -- the mean of the trusted baseline for the historical
  gate.
* Every constraint is two-sided: `cur` must land in the corridor
  `[ref - band_down, ref + band_up]`. There is no unbounded side -- a value
  far from baseline in the "improving" direction is usually a broken metric,
  and a passing run's values become future baselines, so admitting it would
  drag the mean. A side meant to be lenient gets a wide band, not no band.
* Each side's band scales from the reference only:
  `band = max(rel * |ref|, abs_floor)`, so `rel_up = 0.5` reads as "at most
  50% above the baseline mean". Scaling from `cur` would let a deviating
  value widen its own tolerance (at rel 0.5 the effective ceiling is 2x ref,
  not 1.5x). `abs_floor` keeps metrics near zero from flagging on a
  meaningless relative percentage.
* The declaration literal is validated against :data:`CONSTRAINT_SCHEMA` at
  parse time; `evaluate_constraint` expects the normalized dict (defaults
  filled). The literal as written is the spec's `constraint_key`, part of the
  stored value's identity (see register.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintOutcome:
    # [CLASS DOC] Immutable result of a constraint check: ok flag, and the lower and upper corridor
    # bounds (floats) the value was compared with, inclusive on both ends.
    """Whether `cur` passed, and the corridor `[lo, hi]` it was judged against."""

    ok: bool
    lo: float
    hi: float


# Parse-time param schema, consumed by register.py. Each entry:
# param -> (validator_key, required, default). A declaration must write at
# least one band param per side -- the parser enforces it, since an
# all-default side has band 0 and fails on any deviation.
CONSTRAINT_SCHEMA: dict[str, tuple[str, bool, object]] = {
    "rel_up": ("float_nonneg", False, 0.0),
    "abs_floor_up": ("float_nonneg", False, 0.0),
    "rel_down": ("float_nonneg", False, 0.0),
    "abs_floor_down": ("float_nonneg", False, 0.0),
}


def evaluate_constraint(constraint: dict, cur: float, ref: float) -> ConstraintOutcome:
    # [FUNCTION DOC] Computes the corridor from the reference: upper bound is ref plus the larger of
    # (relative-up times absolute ref) and the up floor; lower bound is ref minus the larger of
    # (relative-down times absolute ref) and the down floor. The constraint mapping must already be
    # normalized (all four keys present, else a key error). Returns ok when cur lies within the
    # corridor inclusive. A NaN current value yields not ok (comparisons are false).
    """Apply a normalized constraint dict to `cur` vs `ref`."""
    hi = ref + max(constraint["rel_up"] * abs(ref), constraint["abs_floor_up"])
    lo = ref - max(constraint["rel_down"] * abs(ref), constraint["abs_floor_down"])
    return ConstraintOutcome(lo <= cur <= hi, lo, hi)
