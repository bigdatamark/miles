# [MODULE DOC]
# Tests for the exact-division helper in the math utilities module: integer division that must come
# out whole, otherwise it fails an assertion with a descriptive message (not a remainder floor and
# not a zero-division error).
import pytest

from miles.utils.math_utils import exact_div


class TestExactDiv:
    # [CLASS DOC] Tests of the exact-division helper.
    def test_divides_when_the_division_comes_out_whole(self):
        # [FUNCTION DOC] Dividing 24 by 8 returns 3.
        """The only case a caller wants: a count that tiles the divisor exactly."""
        assert exact_div(24, 8) == 3

    def test_refuses_a_remainder(self):
        # [FUNCTION DOC] Dividing 12 by 8 raises an assertion error whose message says 12 is not a
        # whole number of 8.
        """Flooring a remainder away is what this helper exists to prevent."""
        with pytest.raises(AssertionError, match="12 is not a whole number of 8"):
            exact_div(12, 8)

    def test_refuses_a_zero_divisor(self):
        # [FUNCTION DOC] Dividing 8 by zero raises an assertion error (not a zero-division error)
        # whose message says it cannot divide 8 by zero.
        """A caller that forgot to fill in the divisor must hear about it, not see a ZeroDivisionError."""
        with pytest.raises(AssertionError, match="divide 8 by zero"):
            exact_div(8, 0)
