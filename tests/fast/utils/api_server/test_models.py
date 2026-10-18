# [MODULE DOC] Tests for the api server data models (miles.utils.ft_utils.api_server.models):
# the health-checker-derived Healthy condition, serialization of the cell status including its
# workers hash generation stamp, and the requirement that the stamp is mandatory.
import pytest
from pydantic import ValidationError

from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus, TriState


class TestCellConditionFromHealthCheckerStatus:
    # [CLASS DOC] Tests of building a Healthy condition from a health checker tri-state.
    def test_a_failed_probe_carries_a_reason(self):
        # [FUNCTION DOC] A false tri-state yields a condition of type Healthy, status false and
        # reason "HealthCheckFailed".
        """The reason is what tells an operator why a cell was recycled."""
        condition = CellCondition.from_health_checker_status(TriState.FALSE)

        assert (condition.type, condition.status, condition.reason) == ("Healthy", TriState.FALSE, "HealthCheckFailed")

    def test_no_verdict_yet_is_distinguishable_from_a_failure(self):
        # [FUNCTION DOC] An unknown tri-state yields status unknown with reason "HealthCheckUnknown".
        """A checker that has not probed yet must not be read as a dead cell."""
        condition = CellCondition.from_health_checker_status(TriState.UNKNOWN)

        assert (condition.status, condition.reason) == (TriState.UNKNOWN, "HealthCheckUnknown")

    def test_a_passing_probe_needs_no_reason(self):
        # [FUNCTION DOC] A true tri-state yields status true and no reason.
        """A healthy cell has nothing to explain."""
        condition = CellCondition.from_health_checker_status(TriState.TRUE)

        assert (condition.status, condition.reason) == (TriState.TRUE, None)


class TestCellStatusSerialization:
    # [CLASS DOC] Tests of how a cell status is dumped for the published document.
    def test_the_worker_generation_stamp_reaches_the_response(self):
        # [FUNCTION DOC] Builds a Running status with an Allocated true condition and workers hash
        # "pseudo-hash-0" and asserts the dumped dictionary contains that hash under workers_hash.
        """A client comparing two polls needs the generation a status describes, so it is part of the published document."""
        status = CellStatus(
            phase="Running",
            conditions=[CellCondition.allocated(TriState.TRUE)],
            workers_hash="pseudo-hash-0",
        )

        assert status.model_dump()["workers_hash"] == "pseudo-hash-0"


class TestCellStatusRequiresItsGeneration:
    # [CLASS DOC] Tests that the workers hash is a required field of the cell status.
    def test_a_status_cannot_be_built_without_the_generation_it_describes(self):
        # [FUNCTION DOC] Asserts that constructing a status without a workers hash raises a model
        # validation error.
        """An optional stamp would let an unstamped status be published as the current generation's verdict."""
        with pytest.raises(ValidationError):
            CellStatus(phase="Running", conditions=[CellCondition.allocated(TriState.TRUE)])

    def test_a_published_status_survives_a_round_trip_with_its_generation(self):
        # [FUNCTION DOC] Dumps a status with workers hash "pseudo-hash-7", validates the dump back
        # into a status and asserts equality with the original.
        """A client polling the api server compares the stamp across polls, so it must reparse into the same value."""
        status = CellStatus(
            phase="Running",
            conditions=[CellCondition.allocated(TriState.TRUE)],
            workers_hash="pseudo-hash-7",
        )

        assert CellStatus.model_validate(status.model_dump()) == status
