# [MODULE DOC]
# HELPER: Reporting utilities for preflight: an accumulator of check results (overall failure flag
# and permission-denial statistics) and functions that log individual results and warnings with
# fixed prefixes.
from __future__ import annotations

import logging

from miles.utils.external_utils.miles_workbench.preflight.checkers import (
    CheckOutcome,
    CheckResult,
    ResourceVerbAvailabilityChecker,
    Status,
)

logger = logging.getLogger(__name__)


class Verdict:
    # [CLASS DOC] Mutable accumulator of preflight results. The failed flag becomes true once any
    # absorbed result failed. It also counts permission checks and how many of them failed, to
    # detect the situation where every permission check was denied.
    def __init__(self) -> None:
        # [FUNCTION DOC] Starts not failed with zero permission counters.
        self.failed = False
        self._permission_checks = 0
        self._permission_failures = 0

    def absorb(self, outcomes: list[CheckOutcome]) -> None:
        # [FUNCTION DOC] Reports each outcome and folds it into the verdict (see the single-result
        # variant); additionally counts outcomes of permission checkers and their failures.
        for outcome in outcomes:
            self.absorb_result(outcome.result)
            if isinstance(outcome.checker, ResourceVerbAvailabilityChecker):
                self._permission_checks += 1
                if outcome.result.status is Status.FAIL:
                    self._permission_failures += 1

    def absorb_result(self, result: CheckResult) -> None:
        # [FUNCTION DOC] Logs the result and sets the failed flag if its status is fail (unknown
        # does not fail).
        report(result)
        if result.status is Status.FAIL:
            self.failed = True

    def observe(self, outcomes: list[CheckOutcome]) -> None:
        # [FUNCTION DOC] Only logs the outcomes' results; does not affect the verdict or counters.
        for outcome in outcomes:
            report(outcome.result)

    def announce(self) -> None:
        # [FUNCTION DOC] Logs the final summary: an info line when nothing failed; otherwise, if
        # every counted permission check failed, a hint to verify the namespace name and kubectl
        # context, then an error line that preflight failed.
        if not self.failed:
            logger.info("Preflight checks passed")
            return

        if self._all_denied:
            logger.error(
                "Every check was denied: confirm the namespace name and your kubectl context before "
                "treating this as missing RBAC"
            )
        logger.error("Preflight checks failed")

    @property
    def _all_denied(self) -> bool:
        # [FUNCTION DOC] HELPER: Property: true when at least one permission check ran and all of
        # them failed.
        return bool(self._permission_checks) and self._permission_failures == self._permission_checks


def report(result: CheckResult) -> None:
    # [FUNCTION DOC] HELPER: Logs the message prefixed by the status: info level for pass and
    # unknown, error level for fail.
    if result.status is Status.PASS:
        logger.info("PASS  %s", result.message)
        return
    if result.status is Status.UNKNOWN:
        logger.info("UNKNOWN  %s", result.message)
        return
    logger.error("FAIL  %s", result.message)


def warn(message: str) -> None:
    # [FUNCTION DOC] HELPER: Logs a warning-level line prefixed by the word warn.
    logger.warning("WARN  %s", message)
