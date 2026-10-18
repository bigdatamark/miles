# [MODULE DOC]
# HELPER: Small math utilities.
from __future__ import annotations


def exact_div(numerator: int, denominator: int) -> int:
    # [FUNCTION DOC] HELPER: Integer division that must be exact. Asserts the denominator is not zero
    # and divides the numerator evenly (messages state the failing values); returns the floor
    # quotient.
    assert denominator != 0, f"cannot divide {numerator} by zero"
    assert numerator % denominator == 0, f"{numerator} is not a whole number of {denominator}"
    return numerator // denominator
