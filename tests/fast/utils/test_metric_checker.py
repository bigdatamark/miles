# [MODULE DOC]
# Tests for the CI metric checker, which watches evaluation results for one configured metric key
# against a threshold (here key "eval/gsm8k", threshold 0.4) and on disposal decides whether the
# run passes. Default expectation: pass if any observed check meets the threshold. Exact
# expectation (an expected count is configured): pass only if exactly that many checks occurred
# and all succeeded. Having observed no metrics at all always fails with an assertion error.
from argparse import Namespace

import pytest

from miles.utils.metric_checker import MetricChecker


def _make_checker(expect_num: int | None = None) -> MetricChecker:
    # [FUNCTION DOC] HELPER: Build a metric checker from an arguments namespace with key
    # "eval/gsm8k", threshold 0.4 and the given expected check count (default none, meaning the
    # default any-pass expectation).
    args = Namespace(
        ci_metric_checker_key="eval/gsm8k",
        ci_metric_checker_threshold=0.4,
        ci_metric_checker_expect_num=expect_num,
    )
    return MetricChecker(args)


class TestDefaultExpectation:
    # [CLASS DOC] Tests of the default (any-pass) expectation.
    def test_one_passing_eval_satisfies_the_gate(self):
        # [FUNCTION DOC] Feed one failing (0.3) then one passing (0.5) eval, then dispose; disposal
        # does not raise.
        """The default expectation accepts a run when any observed eval meets the threshold."""
        checker = _make_checker()

        checker.on_eval({"eval/gsm8k": 0.3})
        checker.on_eval({"eval/gsm8k": 0.5})

        checker.dispose()


class TestExactExpectation:
    # [CLASS DOC] Tests of the exact-count expectation (two checks, all must succeed).
    def test_one_failing_eval_fails_the_gate(self):
        # [FUNCTION DOC] Feed one passing and one failing eval with expected count two; disposal
        # raises an assertion error saying exactly 2 checks and all to succeed were expected.
        """An exact expectation rejects the run when any observed eval misses the threshold."""
        checker = _make_checker(expect_num=2)

        checker.on_eval({"eval/gsm8k": 0.5})
        checker.on_eval({"eval/gsm8k": 0.3})

        with pytest.raises(AssertionError, match="expected exactly 2 checks and all to succeed"):
            checker.dispose()

    def test_every_passing_eval_satisfies_the_gate(self):
        # [FUNCTION DOC] Feed two evals (0.4, exactly at the threshold, and 0.5) with expected count
        # two; disposal does not raise, showing the threshold is inclusive.
        """An exact expectation passes when it receives that many successful checks."""
        checker = _make_checker(expect_num=2)

        checker.on_eval({"eval/gsm8k": 0.4})
        checker.on_eval({"eval/gsm8k": 0.5})

        checker.dispose()

    @pytest.mark.parametrize("num_results", [1, 3])
    def test_a_non_exact_result_count_fails_the_gate(self, num_results: int):
        # [FUNCTION DOC] Parametrized over one and three passing evals with expected count two;
        # disposal raises the same exact-count assertion error.
        """An exact expectation rejects both missing and additional successful checks."""
        checker = _make_checker(expect_num=2)

        for _ in range(num_results):
            checker.on_eval({"eval/gsm8k": 0.5})

        with pytest.raises(AssertionError, match="expected exactly 2 checks and all to succeed"):
            checker.dispose()


@pytest.mark.parametrize("expect_num", [None, 2])
def test_no_eval_always_fails(expect_num: int | None):
    # [FUNCTION DOC] Parametrized over no expected count and an expected count of two; disposing a
    # checker that saw no eval raises an assertion error saying no metrics were checked.
    """The checker rejects a run with no eval under either expectation mode."""
    with pytest.raises(AssertionError, match="no metrics checked"):
        _make_checker(expect_num=expect_num).dispose()
