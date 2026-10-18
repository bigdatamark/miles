# [MODULE DOC]
# CI helper that checks evaluation metrics against a threshold during a test run.
import logging

logger = logging.getLogger(__name__)


class MetricChecker:
    # [CLASS DOC] Collects pass/fail results of metric threshold checks on each evaluation, and on
    # dispose asserts the overall expectation: if no expected count is configured at least one check
    # must have passed; otherwise exactly that many checks must have run and all must have passed.
    @staticmethod
    def maybe_create(args):
        # [FUNCTION DOC] Returns a checker only in ci test mode with a configured metric key; else
        # none.
        if args.ci_test and (args.ci_metric_checker_key is not None):
            return MetricChecker(args)
        return None

    def __init__(self, args):
        # [FUNCTION DOC] Stores args and starts with no results.
        self.args = args
        self._check_results: list[bool] = []

    def on_eval(self, metrics: dict[str, float]):
        # [FUNCTION DOC] Looks up the configured key in the metrics mapping (assertion error if
        # absent), compares the value with the threshold (success when greater or equal), logs, and
        # records the boolean.
        actual_value = metrics.get(self.args.ci_metric_checker_key)
        assert actual_value is not None, f"{metrics=} {self.args.ci_metric_checker_key=}"

        check_success = actual_value >= self.args.ci_metric_checker_threshold
        logger.info(f"[MetricChecker] {check_success=} {actual_value=} {self.args.ci_metric_checker_threshold=}")

        self._check_results.append(check_success)

    def dispose(self):
        # [FUNCTION DOC] Final check: asserts at least one result exists; with no expected count
        # asserts any success; with an expected count asserts the number of results equals it and
        # all succeeded. Logs on pass.
        assert self._check_results, "[MetricChecker] no metrics checked"
        if (expected := self.args.ci_metric_checker_expect_num) is None:
            assert any(self._check_results), f"[MetricChecker] accuracy check failed: {self._check_results=}"
        else:
            assert len(self._check_results) == expected and all(self._check_results), (
                f"[MetricChecker] expected exactly {expected} checks and all to succeed: " f"{self._check_results=}"
            )
        logger.info("[MetricChecker] pass dispose check")
