# [MODULE DOC] Tests for the preflight reporting utilities: the verdict object (absorbed failures
# gate installation while observed failures only report; announcing explains a uniform denial of
# every permission probe as a probable wrong context, and reports success when nothing absorbed
# failed), the report function (each status has a stable text prefix and a log level: pass and
# unknown at info, fail at error) and the warn function (warning prefix).
import logging

import pytest

from miles.utils.external_utils.miles_workbench.preflight.checkers import (
    CheckOutcome,
    CheckResult,
    ResourceVerb,
    ResourceVerbAvailabilityChecker,
    Status,
)
from miles.utils.external_utils.miles_workbench.preflight.utils import Verdict, report, warn


def _permission_outcome(status: Status) -> CheckOutcome:
    # [FUNCTION DOC] HELPER: Builds a check outcome for a get permission on pods in namespace rl
    # whose result has the given status and the message permission.
    checker = ResourceVerbAvailabilityChecker(namespace="rl", resource_verb=ResourceVerb(resource="pods", verb="get"))
    return CheckOutcome(checker=checker, result=CheckResult(status=status, message="permission"))


class TestVerdict:
    # [CLASS DOC] Tests of the verdict accumulator.
    def test_absorb_marks_failures_while_observe_only_reports_them(self, caplog: pytest.LogCaptureFixture) -> None:
        # [FUNCTION DOC] One verdict absorbs a failing result and another observes a failing
        # outcome, under info level log capture. Afterwards the absorbing verdict is failed, the
        # observing one is not, and the failure line appears twice in the log.
        """Absorbed failures gate installation while observed failures remain informational."""
        result = CheckResult(status=Status.FAIL, message="failed check")
        absorbed = Verdict()
        observed = Verdict()

        with caplog.at_level(logging.INFO):
            absorbed.absorb_result(result)
            observed.observe(
                [
                    CheckOutcome(
                        checker=ResourceVerbAvailabilityChecker(
                            namespace="rl", resource_verb=ResourceVerb(resource="pods", verb="get")
                        ),
                        result=result,
                    )
                ]
            )

        assert absorbed.failed is True
        assert observed.failed is False
        assert caplog.text.count("FAIL  failed check") == 2

    def test_announce_explains_when_every_permission_probe_was_denied(self, caplog: pytest.LogCaptureFixture) -> None:
        # [FUNCTION DOC] Absorbing two failing permission outcomes and announcing at error level
        # logs a message that every check was denied and a message that preflight checks failed.
        """A uniformly denied permission set emits the wrong-context diagnostic."""
        verdict = Verdict()
        verdict.absorb([_permission_outcome(Status.FAIL), _permission_outcome(Status.FAIL)])

        with caplog.at_level(logging.ERROR):
            verdict.announce()

        assert "Every check was denied" in caplog.text
        assert "Preflight checks failed" in caplog.text

    def test_announce_reports_success_when_no_absorbed_check_failed(self, caplog: pytest.LogCaptureFixture) -> None:
        # [FUNCTION DOC] Absorbing one passing outcome and announcing at info level logs that
        # preflight checks passed.
        """A verdict with no absorbed failure announces success."""
        verdict = Verdict()
        verdict.absorb([_permission_outcome(Status.PASS)])

        with caplog.at_level(logging.INFO):
            verdict.announce()

        assert "Preflight checks passed" in caplog.text


class TestReporting:
    # [CLASS DOC] Tests of the report and warn log helpers.
    @pytest.mark.parametrize(
        ("status", "level", "prefix"),
        [
            (Status.PASS, logging.INFO, "PASS"),
            (Status.UNKNOWN, logging.INFO, "UNKNOWN"),
            (Status.FAIL, logging.ERROR, "FAIL"),
        ],
    )
    def test_each_status_uses_its_protocol_prefix_and_severity(
        self, caplog: pytest.LogCaptureFixture, status: Status, level: int, prefix: str
    ) -> None:
        # [FUNCTION DOC] Parametrized over pass (info, prefix PASS), unknown (info, prefix UNKNOWN)
        # and fail (error, prefix FAIL). Reporting a result with the message probe, captured at the
        # expected level, logs the prefix, two spaces and the message.
        """Each check status is reported with its stable prefix and severity."""
        with caplog.at_level(level):
            report(CheckResult(status=status, message="probe"))

        assert f"{prefix}  probe" in caplog.text

    def test_warn_emits_the_warning_protocol_prefix(self, caplog: pytest.LogCaptureFixture) -> None:
        # [FUNCTION DOC] Warning with the message foreign object, captured at warning level, logs
        # the text WARN, two spaces and the message.
        """Preflight warnings use the stable warning prefix."""
        with caplog.at_level(logging.WARNING):
            warn("foreign object")

        assert "WARN  foreign object" in caplog.text
