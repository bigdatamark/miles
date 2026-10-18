# [MODULE DOC] Unit tests for the witness rule of the event analyzer. A "witness" id is allocated
# per sample in a ring buffer and shows up as non-zero in the parameter snapshot of a training cell
# when the sample contributed to the gradient. The rule takes a chronological list of events
# (witness id allocation events mapping witness id to sample index, snapshot events listing
# non-zero and stale witness ids per cell and instance, advantage computation events, and group
# step end events with per-cell outcomes) and, for each cell whose step outcome is normal,
# compares the expected cumulative set of witness ids with the snapshot's actual set, returning
# data mismatch issues (expected versus actual ids) or missing snapshot issues. Rules covered:
# retries use the latest attempt, stale ids are ignored on both sides, discarded or error steps
# are skipped, and ids of samples whose advantages are all zero are excused unless the id was
# reallocated to a nonzero-advantage sample later. Events are built with helpers below.
"""Tests for event_analyzer rules/witness."""

from datetime import datetime, timezone

from pydantic import TypeAdapter

from miles.backends.training_utils.types import TrainStepOutcome
from miles.utils.audit_utils.event_analyzer.rules.witness import (
    WitnessDataMismatchIssue,
    WitnessMissingSnapshotIssue,
    check,
)
from miles.utils.audit_utils.event_logger.models import (
    Event,
    TrainAdvantageComputationEvent,
    TrainGroupStepEndEvent,
    WitnessAllocateIdEvent,
    WitnessSnapshotParamEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainProcessIdentity

_event_adapter = TypeAdapter(Event)

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_MAIN_SOURCE = SimpleProcessIdentity(component="main")


def _make_source(cell_index: int = 0, rank_within_cell: int = 0) -> TrainProcessIdentity:
    # [FUNCTION DOC] HELPER: Builds a training process identity of component "actor" with the
    # given cell index and rank within cell (both default 0).
    return TrainProcessIdentity(component="actor", cell_index=cell_index, rank_within_cell=rank_within_cell)


def _make_snapshot(
    rollout_id: int,
    nonzero_witness_ids: list[int],
    instance_id: str = "pp0.head",
    cell_index: int = 0,
    rank_within_cell: int = 0,
    stale_ids: list[int] | None = None,
    attempt: int = 0,
) -> WitnessSnapshotParamEvent:
    # [FUNCTION DOC] HELPER: Builds a witness snapshot event with the fixed timestamp, a training
    # source for the given cell and rank, the rollout id, attempt number (default 0), instance id
    # (default "pp0.head"), the list of witness ids that were non-zero, and the stale ids
    # (default an empty list).
    return WitnessSnapshotParamEvent(
        timestamp=_FIXED_TS,
        source=_make_source(cell_index=cell_index, rank_within_cell=rank_within_cell),
        rollout_id=rollout_id,
        attempt=attempt,
        instance_id=instance_id,
        nonzero_witness_ids=nonzero_witness_ids,
        stale_ids=stale_ids or [],
    )


def _make_allocate(
    rollout_id: int,
    witness_id_to_sample_index: dict[int, int],
    attempt: int = 0,
) -> WitnessAllocateIdEvent:
    # [FUNCTION DOC] HELPER: Builds a witness id allocation event from the main source for the
    # rollout and attempt, with the given mapping from witness id to sample index. The counter
    # after allocation is set to one more than the largest allocated id (0 when the mapping is
    # empty).
    return WitnessAllocateIdEvent(
        timestamp=_FIXED_TS,
        source=_MAIN_SOURCE,
        rollout_id=rollout_id,
        attempt=attempt,
        witness_id_to_sample_index=witness_id_to_sample_index,
        counter_after=max(witness_id_to_sample_index.keys(), default=-1) + 1,
    )


def _make_step_end(
    rollout_id: int,
    cell_outcomes: dict[int, str | list[TrainStepOutcome]],
) -> TrainGroupStepEndEvent:
    # [FUNCTION DOC] HELPER: Builds a group step end event from the main source for the rollout
    # with the given mapping from cell index to either an error string or a list of step
    # outcomes.
    return TrainGroupStepEndEvent(
        timestamp=_FIXED_TS,
        source=_MAIN_SOURCE,
        rollout_id=rollout_id,
        cell_outcomes=cell_outcomes,
    )


def _make_advantage(
    rollout_id: int,
    advantages: list[list[float]],
    witness_ids: list[list[int]],
    cell_index: int = 0,
    attempt: int = 0,
) -> TrainAdvantageComputationEvent:
    # [FUNCTION DOC] HELPER: Builds an advantage computation event from a training source of
    # the given cell for the rollout and attempt, with per-sample lists of per-token advantages
    # and per-sample lists of per-token witness ids.
    return TrainAdvantageComputationEvent(
        timestamp=_FIXED_TS,
        source=_make_source(cell_index=cell_index),
        rollout_id=rollout_id,
        attempt=attempt,
        advantages=advantages,
        witness_ids=witness_ids,
    )


class TestWitnessCheck:
    # [CLASS DOC] Tests for the rule's check function on sequences of allocation, snapshot and
    # step end events: when expected and actual witness id sets disagree an issue is returned
    # carrying the rollout id, cell index and the expected and actual id collections.
    def test_empty_events(self) -> None:
        # [FUNCTION DOC] An empty event list gives an empty result.
        assert check([]) == []

    def test_normal_step_with_correct_cumulative_witness_ids_returns_no_issues(self) -> None:
        # [FUNCTION DOC] Ids 10 and 11 allocated and both present in the snapshot of a normal
        # step give no issues.
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_normal_step_with_missing_witness_id_returns_issue(self) -> None:
        # [FUNCTION DOC] Id 11 allocated but absent from the snapshot gives one issue for
        # rollout 0 with 11 among the expected ids and not among the actual ids.
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert issues[0].rollout_id == 0
        assert 11 in issues[0].expected_witness_ids
        assert 11 not in issues[0].actual_witness_ids

    def test_normal_step_with_extra_witness_id_returns_issue(self) -> None:
        # [FUNCTION DOC] An id 99 in the snapshot that was never allocated gives one issue with
        # 99 among the actual ids and not among the expected ids.
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11, 99]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert 99 in issues[0].actual_witness_ids
        assert 99 not in issues[0].expected_witness_ids

    def test_discarded_step_is_ignored(self) -> None:
        # [FUNCTION DOC] A step whose outcome is discarded-should-retry is not checked even
        # though the snapshot lacks an allocated id.
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.DISCARDED_SHOULD_RETRY]}),
        ]
        assert check(events) == []

    def test_stale_ids_are_ignored(self) -> None:
        # [FUNCTION DOC] Allocated ids 2, 5, 8 with a snapshot showing 5 and 8 and stale ids
        # 0, 1, 2 give no issues: stale ids are removed from the expected set (and the actual
        # set).
        """IDs in stale_ids are ignored in both expected and actual."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={2: 0, 5: 1, 8: 2}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[5, 8], stale_ids=[0, 1, 2]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_multiple_cells_independent_checking(self) -> None:
        # [FUNCTION DOC] Two cells each with a complete snapshot and a normal outcome give no
        # issues.
        """Each cell is checked independently."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11], cell_index=0),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11], cell_index=1),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_retry_uses_latest_attempt_allocation(self) -> None:
        # [FUNCTION DOC] Two allocations of the same rollout (attempt 0 with ids 10, 11 and
        # attempt 1 with ids 20, 21) and a snapshot with 20, 21 give no issues: only the latest
        # attempt's allocation is expected.
        """When retries happen, only the latest attempt's allocation is used."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}, attempt=0),
            _make_allocate(rollout_id=0, witness_id_to_sample_index={20: 0, 21: 1}, attempt=1),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[20, 21]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_snapshot_latest_attempt_discards_stale_crashed_snapshot(self) -> None:
        # [FUNCTION DOC] An attempt 0 snapshot with wrong ids (10, 11) followed by a correct
        # attempt 1 snapshot (20, 21) matching the attempt 1 allocation give no issues: the
        # earlier snapshot is discarded.
        """A crashed attempt-0 snapshot with wrong ids is discarded; only the latest attempt-1 snapshot is compared."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={20: 0, 21: 1}, attempt=1),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11], attempt=0),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[20, 21], attempt=1),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_snapshot_latest_attempt_is_the_one_compared(self) -> None:
        # [FUNCTION DOC] The attempt 0 snapshot is correct but the attempt 1 snapshot lists
        # only a bogus id 99: exactly one data mismatch issue is returned, with 99 actual and
        # not expected.
        """The latest attempt-1 snapshot has wrong ids (attempt-0 was correct), so exactly one mismatch is reported."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}, attempt=1),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11], attempt=0),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[99], attempt=1),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert isinstance(issues[0], WitnessDataMismatchIssue)
        assert 99 in issues[0].actual_witness_ids
        assert 99 not in issues[0].expected_witness_ids

    def test_cumulative_across_rollouts(self) -> None:
        # [FUNCTION DOC] Rollout 0 allocates id 10; rollout 1 allocates id 11; the rollout 1
        # snapshot shows both: no issues, since expected ids accumulate across rollouts.
        """Expected witness IDs are cumulative from rollout 0 to current."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={11: 1}),
            _make_snapshot(rollout_id=1, nonzero_witness_ids=[10, 11]),
            _make_step_end(rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_missing_snapshot_for_normal_cell_returns_issue(self) -> None:
        # [FUNCTION DOC] A normal step outcome for cell 0 without any snapshot event gives
        # exactly one missing snapshot issue with rollout id 0 and cell index 0.
        """Cell claims NORMAL but has no WitnessSnapshotParamEvent — should return WitnessMissingSnapshotIssue."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0}),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert isinstance(issues[0], WitnessMissingSnapshotIssue)
        assert issues[0].rollout_id == 0
        assert issues[0].cell_index == 0

    def test_error_cell_outcome_is_skipped(self) -> None:
        # [FUNCTION DOC] A cell outcome that is the error string (not a list of outcomes)
        # produces no issue even with no snapshot.
        """cell_outcomes with 'error' string should not produce any issue."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0}),
            _make_step_end(rollout_id=0, cell_outcomes={0: "error"}),
        ]
        assert check(events) == []

    def test_multi_element_cell_outcome_not_all_normal_is_skipped(self) -> None:
        # [FUNCTION DOC] An outcome list mixing normal and discarded-should-retry is skipped
        # (only an all-normal list is checked), even though the snapshot lacks the allocated id.
        """A cell whose outcome list mixes NORMAL with a non-NORMAL outcome is skipped, even with a missing witness id."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[]),
            _make_step_end(
                rollout_id=0,
                cell_outcomes={0: [TrainStepOutcome.NORMAL, TrainStepOutcome.DISCARDED_SHOULD_RETRY]},
            ),
        ]
        assert check(events) == []

    def test_multiple_snapshots_per_cell(self) -> None:
        # [FUNCTION DOC] A cell with a head snapshot showing 10 and 11 and a tail snapshot
        # showing only 10: exactly one data mismatch issue (for the tail), with 11 expected and
        # not actual.
        """Same cell has head and tail snapshots; only the mismatched one produces an issue."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11], instance_id="pp0.head"),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10], instance_id="pp0.tail"),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert isinstance(issues[0], WitnessDataMismatchIssue)
        assert 11 in issues[0].expected_witness_ids
        assert 11 not in issues[0].actual_witness_ids

    def test_ring_buffer_wrap_with_stale_ids(self) -> None:
        # [FUNCTION DOC] Simulates a ring buffer of size 10: rollout 0 allocates ids 0 to 7 and
        # rollout 1 allocates 8, 9, 0, 1, 2 (wrapping). The rollout 1 snapshot has non-zero ids
        # 0, 1, 2, 6, 7, 8, 9 with stale ids 3, 4, 5. Expected cumulative ids 0 to 9 minus stale
        # equals the actual ids, so no issues.
        """After wrap, stale_ids contain wrapped IDs (e.g. [8,9,0]). These should be excluded from comparison."""
        # buffer_size=10, allocated IDs 0..7 in rollout 0, then 8,9,0,1,2 in rollout 1 (wrap)
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={i: i for i in range(8)}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=list(range(8))),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={8: 8, 9: 9, 0: 10, 1: 11, 2: 12}),
            # After wrap: stale_ids=[3,4,5] (old IDs cleaned), actual nonzero = [0,1,2,6,7,8,9]
            _make_snapshot(
                rollout_id=1,
                nonzero_witness_ids=[0, 1, 2, 6, 7, 8, 9],
                stale_ids=[3, 4, 5],
            ),
            _make_step_end(rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        # expected cumulative = {0..9}, minus stale {3,4,5} = {0,1,2,6,7,8,9} — matches actual
        assert check(events) == []

    def test_ring_buffer_wrap_detects_mismatch(self) -> None:
        # [FUNCTION DOC] Same wrap scenario but the rollout 1 snapshot lacks id 7 (which is not
        # stale): exactly one issue with 7 expected and not actual.
        """After wrap, a genuinely missing non-stale ID should still be caught."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={i: i for i in range(8)}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=list(range(8))),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={8: 8, 9: 9, 0: 10, 1: 11, 2: 12}),
            # ID 7 is NOT stale but missing from actual — should be caught
            _make_snapshot(
                rollout_id=1,
                nonzero_witness_ids=[0, 1, 2, 6, 8, 9],  # missing 7
                stale_ids=[3, 4, 5],
            ),
            _make_step_end(rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert 7 in issues[0].expected_witness_ids
        assert 7 not in issues[0].actual_witness_ids


class TestWitnessEventSerialization:
    # [CLASS DOC] Tests that witness events survive a JSON round trip through the generic event
    # type adapter.
    def test_roundtrip(self) -> None:
        # [FUNCTION DOC] Serializes a snapshot event to JSON and parses it back through the
        # union event adapter; asserts the parsed object is a snapshot event with the same
        # rollout id 5, instance id, non-zero ids and stale ids.
        event = _make_snapshot(
            rollout_id=5,
            nonzero_witness_ids=[10, 20],
            instance_id="pp0.tail",
            cell_index=1,
            stale_ids=[0, 1, 2],
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, WitnessSnapshotParamEvent)
        assert parsed.rollout_id == 5
        assert parsed.instance_id == "pp0.tail"
        assert parsed.nonzero_witness_ids == [10, 20]
        assert parsed.stale_ids == [0, 1, 2]


class TestZeroAdvantageExclusion:
    # [CLASS DOC] Tests of the rule that witness ids of samples whose per-token advantages are
    # all zero are excused from the expected set (their gradient contribution is zero, so they
    # show no signal). Details: only the final attempt's advantage events count, the exclusion is
    # global across cells, only the first witness id of each sample's id list is used, an empty
    # advantage shard is not "all zero", and re-allocating an id after ring buffer wrap resets its
    # excusal according to the new sample's advantage.
    def test_zero_advantage_sample_excluded_from_expected(self) -> None:
        # [FUNCTION DOC] Sample with id 10 has all-zero advantages and is absent from the
        # snapshot while sample 11 is present: no issues.
        """Witness ID with zero advantage should not cause a mismatch when missing from actual."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_advantage(rollout_id=0, advantages=[[0.0, 0.0], [2.0, 3.0]], witness_ids=[[10], [11]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[11]),  # 10 missing but zero-adv
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_zero_advantage_sample_stays_excluded_at_later_rollouts(self) -> None:
        # [FUNCTION DOC] Id 11 has zero advantage in rollout 0 and stays absent from the later
        # rollout 1 snapshot (10 and 12 present): no issues, so the excusal persists over later
        # rollouts.
        """Regression: a zero-advantage id stays absent from every later snapshot and must stay excused."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_advantage(rollout_id=0, advantages=[[0.5, 0.5], [0.0, 0.0]], witness_ids=[[10], [11]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={12: 0}),
            _make_snapshot(rollout_id=1, nonzero_witness_ids=[10, 12]),
            _make_step_end(rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_zero_advantage_sample_with_nonzero_signal_is_flagged_extra(self) -> None:
        # [FUNCTION DOC] Id 11 has zero advantage but still appears in the snapshot: one data
        # mismatch issue with 11 actual and not expected.
        """A zero-advantage id that nevertheless shows a nonzero witness signal is still reported."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_advantage(rollout_id=0, advantages=[[0.5, 0.5], [0.0, 0.0]], witness_ids=[[10], [11]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert isinstance(issues[0], WitnessDataMismatchIssue)
        assert 11 in issues[0].actual_witness_ids
        assert 11 not in issues[0].expected_witness_ids

    def test_empty_advantage_shard_under_cp_is_not_treated_as_zero_advantage(self) -> None:
        # [FUNCTION DOC] Two advantage events for the same cell: one from the context parallel
        # rank owning sample 1's response (nonzero advantages) and one from a rank with an empty
        # advantage list for that sample but the full witness ids. Id 11 is present in the
        # snapshot, and no issue is expected: an empty shard must not count as all zero.
        """CP regression: an empty per-rank advantage shard must not excuse a present witness.

        Under CP>1 a short sample's response can land entirely on the other CP rank, leaving
        this rank's advantage shard empty while witness_ids still spans the full sample. The
        sample's real advantage is nonzero on the rank that owns it, so its witness is
        legitimately present and must not be flagged 'extra' (all(v == 0.0 for v in []) is
        vacuously True).
        """
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            # CP rank that owns sample 1's response: real nonzero advantage.
            _make_advantage(rollout_id=0, advantages=[[0.5, 0.5], [-0.54, -0.54]], witness_ids=[[10, 10], [11, 11]]),
            # CP rank without sample 1's response tokens: empty advantage shard, full witness_ids.
            _make_advantage(rollout_id=0, advantages=[[0.5, 0.5], []], witness_ids=[[10, 10], [11, 11]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_zero_advantage_exclusion_uses_final_attempt_only(self) -> None:
        # [FUNCTION DOC] Attempt 0 had all-zero advantages for ids 10 and 11, attempt 1 has a
        # nonzero advantage for id 10 and zero for 11. The snapshot shows only 10. No issues:
        # only the attempt 1 advantages are used (11 excused; 10 required and present).
        """A crashed attempt's all-zero advantage events must not excuse ids the successful retry trains."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}, attempt=1),
            _make_advantage(rollout_id=0, advantages=[[0.0], [0.0]], witness_ids=[[10], [11]], attempt=0),
            _make_advantage(rollout_id=0, advantages=[[5.0], [0.0]], witness_ids=[[10], [11]], attempt=1),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_nonzero_advantage_sample_still_required(self) -> None:
        # [FUNCTION DOC] Ids 10 and 11 both have nonzero advantages but 11 is absent from the
        # snapshot: one issue with 11 expected and not actual.
        """Witness ID with nonzero advantage must still appear — missing produces an issue."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_advantage(rollout_id=0, advantages=[[5.0], [3.0]], witness_ids=[[10], [11]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),  # 11 missing, nonzero adv
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert 11 in issues[0].expected_witness_ids
        assert 11 not in issues[0].actual_witness_ids

    def test_zero_advantage_exclusion_is_global_across_cells(self) -> None:
        # [FUNCTION DOC] Two cells; cell 0 reports zero advantage for sample 10 while cell 1
        # never sees sample 10. Both snapshots show only id 11. No issues: the excusal of id 10
        # is the union over all cells.
        """Zero-adv exclusion is unioned across cells: the per-cell snapshot reflects the global
        (allreduced) gradient, so an id observed all-zero on its owning shard is excused for every cell."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            # Cell 0 owns sample 10 and observes zero advantage for it.
            _make_advantage(rollout_id=0, advantages=[[0.0, 0.0], [5.0]], witness_ids=[[10], [11]], cell_index=0),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[11], cell_index=0),
            # Cell 1 never owned sample 10; its snapshot also lacks it and must not be flagged.
            _make_advantage(rollout_id=0, advantages=[[5.0]], witness_ids=[[11]], cell_index=1),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[11], cell_index=1),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_no_advantage_event_means_no_exclusion(self) -> None:
        # [FUNCTION DOC] With no advantage event at all, a missing id 11 is reported (one
        # issue).
        """Without TrainAdvantageComputationEvent at all, no zero-adv exclusion — missing ID is caught."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1

    def test_mixed_token_advantage_is_not_excused(self) -> None:
        # [FUNCTION DOC] Sample 10 has per-token advantages 0.0 and 2.0 (not all zero) and is
        # absent from the snapshot: one issue with 10 expected and not actual.
        """A sample with mixed per-token advantages [0.0, 2.0] is not all-zero, so its missing witness id is flagged."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_advantage(rollout_id=0, advantages=[[0.0, 2.0], [3.0, 3.0]], witness_ids=[[10], [11]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[11]),  # 10 missing but only partially zero-adv
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert 10 in issues[0].expected_witness_ids
        assert 10 not in issues[0].actual_witness_ids

    def test_zero_advantage_uses_first_token_of_witness_id_list(self) -> None:
        # [FUNCTION DOC] Sample 0 has two per-token witness ids (10, 10) and all-zero
        # advantages; the snapshot lacks 10. No issues: the first witness id of the list
        # identifies the sample to excuse.
        """A zero-advantage sample with a multi-token witness id list excuses its first id (wid_tokens[0])."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_advantage(rollout_id=0, advantages=[[0.0, 0.0], [5.0]], witness_ids=[[10, 10], [11]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[11]),  # 10 missing but zero-adv
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_zero_advantage_id_reused_after_wrap_is_not_excused(self) -> None:
        # [FUNCTION DOC] Id 0 has zero advantage in rollout 0 and is reallocated in rollout 1
        # to a sample with nonzero advantage; the rollout 1 snapshot shows id 0. No issues: the
        # reuse cancels the earlier excusal, so the id is required and it is present.
        """Reallocation after ring wrap cancels the zero-adv excusal; the reused nonzero-adv id is required."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={0: 0}),
            _make_advantage(rollout_id=0, advantages=[[0.0]], witness_ids=[[0]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={0: 1}),  # id 0 reused after wrap
            _make_advantage(rollout_id=1, advantages=[[5.0]], witness_ids=[[0]]),
            _make_snapshot(rollout_id=1, nonzero_witness_ids=[0]),
            _make_step_end(rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_zero_advantage_id_reused_after_wrap_missing_is_flagged(self) -> None:
        # [FUNCTION DOC] Same scenario but the rollout 1 snapshot lacks id 0: exactly one issue
        # for rollout 1 with 0 expected and not actual.
        """A reused nonzero-adv id that stays absent from the snapshot is reported missing, not excused."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={0: 0}),
            _make_advantage(rollout_id=0, advantages=[[0.0]], witness_ids=[[0]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={0: 1}),  # id 0 reused after wrap
            _make_advantage(rollout_id=1, advantages=[[5.0]], witness_ids=[[0]]),
            _make_snapshot(rollout_id=1, nonzero_witness_ids=[]),
            _make_step_end(rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert issues[0].rollout_id == 1
        assert 0 in issues[0].expected_witness_ids
        assert 0 not in issues[0].actual_witness_ids

    def test_zero_advantage_id_reused_after_wrap_and_zero_again_stays_excused(self) -> None:
        # [FUNCTION DOC] Id 0 has zero advantage in both rollouts (reused after wrap) and is
        # absent from both snapshots: no issues.
        """A reused id whose new sample is also zero-adv is excused again at the later rollout."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={0: 0}),
            _make_advantage(rollout_id=0, advantages=[[0.0]], witness_ids=[[0]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={0: 1}),  # id 0 reused after wrap
            _make_advantage(rollout_id=1, advantages=[[0.0]], witness_ids=[[0]]),
            _make_snapshot(rollout_id=1, nonzero_witness_ids=[]),
            _make_step_end(rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_zero_advantage_id_three_lives_alternating(self) -> None:
        # [FUNCTION DOC] Id 0 is allocated in three consecutive rollouts with zero, nonzero,
        # then zero advantage; snapshots are empty, present, empty respectively. No issues: the
        # id is excused, required (and present), then excused again.
        """An id alternating zero/nonzero/zero advantage across three allocations is excused, required, then excused again."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={0: 0}),
            _make_advantage(rollout_id=0, advantages=[[0.0]], witness_ids=[[0]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={0: 1}),
            _make_advantage(rollout_id=1, advantages=[[5.0]], witness_ids=[[0]]),
            _make_snapshot(rollout_id=1, nonzero_witness_ids=[0]),
            _make_step_end(rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=2, witness_id_to_sample_index={0: 2}),
            _make_advantage(rollout_id=2, advantages=[[0.0]], witness_ids=[[0]]),
            _make_snapshot(rollout_id=2, nonzero_witness_ids=[]),
            _make_step_end(rollout_id=2, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []

    def test_zero_advantage_id_reuse_cancellation_applies_per_cell(self) -> None:
        # [FUNCTION DOC] Two cells. Rollout 0: id 0 is zero-advantage (reported by cell 0) and
        # absent from both snapshots. Rollout 1: id 0 is reallocated with nonzero advantage;
        # cell 0's snapshot shows it but cell 1's does not. Exactly one issue is returned, for
        # rollout 1 and cell 1.
        """After reuse with nonzero adv the id is required on every cell — missing on one cell is flagged."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={0: 0}),
            _make_advantage(rollout_id=0, advantages=[[0.0]], witness_ids=[[0]], cell_index=0),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[], cell_index=0),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[], cell_index=1),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: [TrainStepOutcome.NORMAL]}),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={0: 1}),  # id 0 reused after wrap
            _make_advantage(rollout_id=1, advantages=[[5.0]], witness_ids=[[0]], cell_index=0),
            _make_snapshot(rollout_id=1, nonzero_witness_ids=[0], cell_index=0),
            _make_snapshot(rollout_id=1, nonzero_witness_ids=[], cell_index=1),  # missing on cell 1
            _make_step_end(rollout_id=1, cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: [TrainStepOutcome.NORMAL]}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert issues[0].rollout_id == 1
        assert issues[0].cell_index == 1

    def test_zero_advantage_id_never_allocated_has_no_effect(self) -> None:
        # [FUNCTION DOC] An advantage event refers to id 99 with zero advantage, but 99 was
        # never allocated; the snapshot shows only the allocated, nonzero-advantage id 10. No
        # issues.
        """A zero-adv observation for an id that was never allocated neither excuses nor expects anything."""
        events: list[Event] = [
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0}),
            _make_advantage(rollout_id=0, advantages=[[5.0], [0.0]], witness_ids=[[10], [99]]),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
        ]
        assert check(events) == []
