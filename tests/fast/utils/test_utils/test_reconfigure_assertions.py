# [MODULE DOC] Tests for the reconfigure assertion helpers of the test utilities, which read cell
# reconfigure events from an event log directory and check them against an expected sequence of
# shrink and healing reconfigures (rollout id, source cell, healed cells, alive cells afterwards),
# the minimum soak injection count, and the soak witness (enough injections and enough healed
# cells). Module constants: a shrink and a healing event payload and their expected-info
# counterparts. Events are written with the real event logger.
from pathlib import Path
from typing import Any

import pytest

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent, TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.reconfigure_assertions import (
    MIN_SOAK_INJECTIONS,
    ReconfigureInfo,
    assert_min_soak_injections,
    assert_reconfigure_events,
    assert_soak_reconfigure_events,
    load_reconfigure_events,
)

_SHRINK_PARTIAL: dict[str, Any] = dict(
    rollout_id=2,
    quorum_id=1,
    src_cell_index=None,
    healed_cell_indices=[],
    alive_cell_indices_after=[0],
)
_HEALING_PARTIAL: dict[str, Any] = dict(
    rollout_id=3,
    quorum_id=2,
    src_cell_index=0,
    healed_cell_indices=[1],
    alive_cell_indices_after=[0, 1],
)

_SHRINK_EXPECTED = ReconfigureInfo(
    rollout_id=2, src_cell_index=None, healed_cell_indices=[], alive_cell_indices_after=[0]
)
_HEALING_EXPECTED = ReconfigureInfo(
    rollout_id=3, src_cell_index=0, healed_cell_indices=[1], alive_cell_indices_after=[0, 1]
)


def _write_events(log_dir: Path, partials: list[dict[str, Any]]) -> None:
    # [FUNCTION DOC] HELPER: Logs each payload as a cell reconfigure event through a real event
    # logger (main component identity) into the directory, without printing, then closes it.
    # With an empty list the logger is still created and closed.
    event_logger = EventLogger(log_dir=log_dir, source=SimpleProcessIdentity(component="main"))
    for partial in partials:
        event_logger.log(CellReconfigureEvent, partial, print_log=False)
    event_logger.close()


class TestLoadReconfigureEvents:
    # [CLASS DOC] Tests of loading reconfigure events from a directory.
    def test_filters_other_event_types_and_preserves_order(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Logs reconfigure, step-end, reconfigure events; asserts only the two
        # reconfigure events come back, in emission order (rollouts 2 and 3).
        """Only CellReconfigureEvents are returned, in file (emission) order."""
        event_logger = EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"))
        event_logger.log(CellReconfigureEvent, _SHRINK_PARTIAL, print_log=False)
        event_logger.log(TrainGroupStepEndEvent, dict(rollout_id=2, cell_outcomes={}), print_log=False)
        event_logger.log(CellReconfigureEvent, _HEALING_PARTIAL, print_log=False)
        event_logger.close()

        events = load_reconfigure_events(tmp_path)

        assert [e.rollout_id for e in events] == [2, 3]
        assert all(isinstance(e, CellReconfigureEvent) for e in events)

    def test_empty_dir_returns_no_events(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A directory with no log files gives an empty list.
        """A directory without any JSONL files yields an empty event list."""
        assert load_reconfigure_events(tmp_path) == []


class TestAssertReconfigureEvents:
    # [CLASS DOC] Tests of the exact-sequence reconfigure assertion (also checks quorum ids are
    # contiguous).
    def test_passes_on_exact_sequence(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Shrink then healing events matching the expected infos pass.
        """An exactly matching shrink+healing sequence with contiguous quorum ids passes."""
        _write_events(tmp_path, [_SHRINK_PARTIAL, _HEALING_PARTIAL])

        assert_reconfigure_events(tmp_path, expected=[_SHRINK_EXPECTED, _HEALING_EXPECTED])

    def test_passes_on_empty_expectation(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] No events logged and an empty expectation pass (the directory is not even
        # written).
        """Expecting zero reconfigures passes when no events were emitted."""
        assert_reconfigure_events(tmp_path, expected=[])

    def test_missing_healing_fails_sequence_check(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] No events but a healing expected: assertion error mentioning a sequence
        # mismatch.
        """A run that never healed fails the exact-sequence comparison (expected healing, got nothing)."""
        _write_events(tmp_path, [])

        with pytest.raises(AssertionError, match="sequence mismatch"):
            assert_reconfigure_events(tmp_path, expected=[_HEALING_EXPECTED])

    def test_unexpected_extra_healing_fails(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] One healing event (quorum 1) but nothing expected: sequence mismatch.
        """A healing event in a run expected to have none fails the exact-sequence comparison."""
        _write_events(tmp_path, [dict(_HEALING_PARTIAL, quorum_id=1)])

        with pytest.raises(AssertionError, match="sequence mismatch"):
            assert_reconfigure_events(tmp_path, expected=[])

    def test_wrong_rollout_id_fails_sequence_check(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A healing at rollout 9 where rollout 3 is expected: sequence mismatch.
        """A healing at the wrong rollout fails the exact-sequence comparison."""
        _write_events(tmp_path, [dict(_HEALING_PARTIAL, rollout_id=9, quorum_id=1)])

        with pytest.raises(AssertionError, match="sequence mismatch"):
            assert_reconfigure_events(tmp_path, expected=[_HEALING_EXPECTED])


class TestAssertMinSoakInjections:
    # [CLASS DOC] Tests of the standalone minimum soak injection check.
    def test_minimum_injection_failure_names_the_caller_context(self) -> None:
        # [FUNCTION DOC] One injection is below the minimum: assertion error containing the given
        # context text.
        """The standalone check reports which soak (e.g. a rollout-only one) fell short, not just that one did."""
        with pytest.raises(AssertionError, match="rollout-only soak on engine cells"):
            assert_min_soak_injections(1, context="rollout-only soak on engine cells")

    def test_reaching_the_minimum_injection_count_passes(self) -> None:
        # [FUNCTION DOC] Exactly the minimum constant passes.
        """Exactly the required number of successful injections satisfies the standalone check."""
        assert_min_soak_injections(MIN_SOAK_INJECTIONS, context="rollout-only soak on engine cells")


class TestAssertSoakReconfigureEvents:
    # [CLASS DOC] Tests of the soak witness, which needs enough successful injections and enough
    # healed cells in the logged events.
    def test_passes_when_enough_injections_and_healings(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A shrink and two healing events with 2 injections pass.
        """>=2 successful injections with >=2 healing events pass the soak witness."""
        _write_events(tmp_path, [_SHRINK_PARTIAL, _HEALING_PARTIAL, dict(_HEALING_PARTIAL, rollout_id=5, quorum_id=3)])

        assert_soak_reconfigure_events(tmp_path, num_successful_injections=2)

    def test_fails_when_no_injections(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Zero injections: assertion error saying the soak proved too little.
        """Zero successful injections means no fault tolerance was exercised, so the witness fails."""
        with pytest.raises(AssertionError, match="proved too little"):
            assert_soak_reconfigure_events(tmp_path, num_successful_injections=0)

    def test_fails_when_too_few_injections(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] One injection with two healing events: proved-too-little error.
        """A single injection is below the soak minimum even when healing events are present."""
        _write_events(tmp_path, [_HEALING_PARTIAL, dict(_HEALING_PARTIAL, rollout_id=5, quorum_id=3)])

        with pytest.raises(AssertionError, match="proved too little"):
            assert_soak_reconfigure_events(tmp_path, num_successful_injections=1)

    def test_fails_when_too_few_healings(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Three injections but only one healed cell: assertion error saying the
        # healing witness failed.
        """Enough injections but fewer than the required healed cells fail the witness."""
        _write_events(tmp_path, [_SHRINK_PARTIAL, _HEALING_PARTIAL])

        with pytest.raises(AssertionError, match="Healing witness failed"):
            assert_soak_reconfigure_events(tmp_path, num_successful_injections=3)

    def test_one_event_that_readmits_two_cells_counts_as_two_healings(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A single event healing cells 0 and 1 with 2 injections passes, so healed
        # cells are counted rather than events.
        """A single reconfigure can heal several cells at once, and counting events would under-count them."""
        _write_events(tmp_path, [dict(_HEALING_PARTIAL, healed_cell_indices=[0, 1], alive_cell_indices_after=[0, 1])])

        assert_soak_reconfigure_events(tmp_path, num_successful_injections=2)
