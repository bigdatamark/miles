# [MODULE DOC] Rule: for every training cell that reports a normal step outcome, the witness ids
# visible in its parameter snapshot must equal exactly the set of witness ids allocated for
# rollouts up to the current one, minus ids made irrelevant by ring-buffer staleness or by
# zero-advantage samples. Issue types: data mismatch (missing/extra ids) and missing snapshot.
# Only the highest attempt of each rollout (or rollout and cell) is considered, so crashed
# attempts are ignored.

import logging
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterator, Sequence
from typing import Protocol, TypeVar

from miles.backends.training_utils.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import (
    Event,
    TrainAdvantageComputationEvent,
    TrainGroupStepEndEvent,
    WitnessAllocateIdEvent,
    WitnessSnapshotParamEvent,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)


class WitnessDataMismatchIssue(FrozenStrictBaseModel):
    # [CLASS DOC] Issue: a cell's snapshot witness ids differ from the expected ones. Holds the
    # rollout id, cell index, a human description (naming the missing and extra ids) and the
    # sorted expected and actual id lists after filtering.
    rollout_id: int
    cell_index: int
    description: str
    expected_witness_ids: list[int]
    actual_witness_ids: list[int]


class WitnessMissingSnapshotIssue(FrozenStrictBaseModel):
    # [CLASS DOC] Issue: a cell reported a normal outcome for a rollout but no witness snapshot
    # event exists for it. Holds the rollout id, cell index and a description.
    rollout_id: int
    cell_index: int
    description: str


WitnessIssue = WitnessDataMismatchIssue | WitnessMissingSnapshotIssue


def check(events: list[Event]) -> list[WitnessIssue]:
    # [FUNCTION DOC] Rule entry point. Steps: collect the allocated witness ids per rollout from
    # the allocation events; compute the per-rollout zero-advantage ids from the advantage events;
    # then find mismatches between the step-end events, snapshot events and the cumulative
    # expected ids. Returns the list of issues (data mismatches and missing snapshots).
    """
    Related events:
    * WitnessAllocateIdEvent: when allocating `witness_id` to `sample_index`
    * WitnessSnapshotParamEvent: near the end of each train() step in MegatronTrainRayActor
        * If a witness_id appears in the weight, it means the corresponding data is consumed at least once.
    * TrainGroupStepEndEvent: after each train() step in TrainerController

    Check:
    1. For each (rollout_id, cell_index),
       if TrainGroupStepEndEvent claims the cell ends with TrainStepOutcome.NORMAL,
       then its WitnessSnapshotParamEvent should observe *EXACTLY* the training data in rollout_id=0~curr.

    Remarks:
    * To correlate witness_id vs sample_index utilize WitnessAllocateIdEvent.
    * Witness' ring buffer will remove old data, thus we need to ignore the appearance/disappearance of
      all values in `WitnessSnapshotParamEvent.stale_ids`
    """

    allocated_witness_ids_by_rollout = _compute_allocated_witness_ids_by_rollout(
        _filter_by_type(events, WitnessAllocateIdEvent)
    )

    return list(
        _find_mismatches(
            all_step_events=_filter_by_type(events, TrainGroupStepEndEvent),
            all_witness_events=_filter_by_type(events, WitnessSnapshotParamEvent),
            expected_witness_ids_of_step=_compute_expected_witness_ids_of_step(allocated_witness_ids_by_rollout),
            allocated_witness_ids_by_rollout=allocated_witness_ids_by_rollout,
            zero_adv_witness_ids_by_rollout=_compute_zero_advantage_witness_ids(
                _filter_by_type(events, TrainAdvantageComputationEvent)
            ),
        )
    )


_EventT = TypeVar("_EventT")


def _filter_by_type(arr: Sequence[Event], ty: type[_EventT]) -> list[_EventT]:
    # [FUNCTION DOC] HELPER: Returns, in order, the elements that are instances of the given
    # class.
    return [x for x in arr if isinstance(x, ty)]


def _compute_zero_advantage_witness_ids(
    events: list[TrainAdvantageComputationEvent],
) -> dict[int, set[int]]:
    # [FUNCTION DOC] HELPER: Only events of the latest attempt per rollout are used. Per event,
    # the per-sample advantage lists and witness id lists are paired up (equal lengths required;
    # a mismatch raises a value error). Each sample's witness ids must be uniform (assertion
    # otherwise). A sample whose non-empty advantage list is all exactly zero contributes its
    # witness id to the set for the event's rollout. Returns a plain mapping from rollout id to
    # the set of ids. The rationale in the docstring below was checked against the code and holds.
    """Return witness_ids where all per-token advantages == 0.0, keyed by rollout_id.

    Unioned across cells: under indep_dp the per-cell weight snapshot reflects the
    GLOBAL (allreduced) gradient, so a zero-advantage sample contributes nothing
    and its witness is absent from EVERY cell — even cells that never owned it.
    Keying per (rollout_id, cell_index) would let a cell excuse only its own shard,
    falsely flagging peers' zero-advantage witnesses as missing.

    Snapshot checks excuse these ids cumulatively (see _zero_adv_excused_ids_at): a
    zero-advantage sample stays absent from every later snapshot while the expected
    set keeps it forever — until the ring buffer reallocates the witness id to a new
    sample, which cancels the excusal.

    Only the highest-attempt events per rollout count, mirroring the allocate-event
    handling: a crashed attempt's partial advantage events would otherwise excuse ids
    that the successful retry trains for real.

    An EMPTY per-sample advantage list is skipped, not treated as zero-advantage.
    Under CP>1 the advantage tensor logged by a rank holds only that rank's local
    shard of the response tokens, while witness_ids spans the full sample; a short
    sample whose response lands entirely on the other CP rank leaves this rank's
    advantage shard empty. `all(v == 0.0 for v in [])` is vacuously True, so without
    this guard such a sample would be falsely excused and then flagged 'extra' once
    its (real, nonzero-advantage) witness shows up present on the rank that owns it.
    """
    result: dict[int, set[int]] = defaultdict(set)
    for event in _filter_to_latest_attempt(events, group_key=lambda e: e.rollout_id):
        for adv_tokens, wid_tokens in zip(event.advantages, event.witness_ids, strict=True):
            assert len(set(wid_tokens)) <= 1, f"witness ids within one sample must be uniform, got {set(wid_tokens)}"
            if adv_tokens and all(v == 0.0 for v in adv_tokens):
                result[event.rollout_id].add(wid_tokens[0])

    return dict(result)


def _compute_allocated_witness_ids_by_rollout(events: list[WitnessAllocateIdEvent]) -> dict[int, set[int]]:
    # [FUNCTION DOC] HELPER: Considering only the latest attempt of each rollout, unions the keys
    # of each event's witness-id-to-sample-index mapping per rollout. Returns a plain mapping from
    # rollout id to a set of witness ids.
    result: dict[int, set[int]] = defaultdict(set)
    for e in _filter_to_latest_attempt(events, group_key=lambda e: e.rollout_id):
        result[e.rollout_id] |= set(e.witness_id_to_sample_index.keys())
    return dict(result)


def _compute_expected_witness_ids_of_step(
    allocated_witness_ids_by_rollout: dict[int, set[int]],
) -> dict[int, set[int]]:
    # [FUNCTION DOC] HELPER: Builds the cumulative expected ids: for each rollout id in ascending
    # order, the union of the allocated ids of all rollouts up to and including it. Returns a
    # mapping from rollout id to a fresh set. Rollout ids with no allocation event are absent.
    ans: dict[int, set[int]] = {}
    running: set[int] = set()
    for rollout_id in sorted(allocated_witness_ids_by_rollout.keys()):
        running |= allocated_witness_ids_by_rollout[rollout_id]
        ans[rollout_id] = set(running)
    return ans


def _find_mismatches(
    *,
    all_step_events: list[TrainGroupStepEndEvent],
    all_witness_events: list[WitnessSnapshotParamEvent],
    expected_witness_ids_of_step: dict[int, set[int]],
    allocated_witness_ids_by_rollout: dict[int, set[int]],
    zero_adv_witness_ids_by_rollout: dict[int, set[int]],
) -> Iterator[WitnessIssue]:
    # [FUNCTION DOC] HELPER: Generator of issues. Keeps only the latest attempt of snapshot events
    # per (rollout, cell). For every step-end event and every cell outcome in it: skips cells whose
    # outcome is the error marker string, and skips cells where any recorded outcome is not the
    # normal outcome. For remaining cells, selects that cell's snapshots for the rollout; if none,
    # yields a missing-snapshot issue and moves on. Otherwise computes the zero-advantage excused
    # ids at that rollout and yields a mismatch issue for each snapshot that fails comparison
    # against the expected ids (an empty set if the rollout has no expectation).
    latest_attempt_witness_events = _filter_to_latest_attempt(
        all_witness_events, group_key=lambda e: (e.rollout_id, e.source.cell_index)
    )

    for step_event in all_step_events:
        rollout_id = step_event.rollout_id

        for cell_index, cell_outcome in step_event.cell_outcomes.items():
            if cell_outcome == "error":
                continue
            if not all(r == TrainStepOutcome.NORMAL for r in cell_outcome):
                continue

            witness_events_of_cell = [
                e
                for e in latest_attempt_witness_events
                if e.rollout_id == rollout_id and e.source.cell_index == cell_index
            ]

            if not witness_events_of_cell:
                yield WitnessMissingSnapshotIssue(
                    rollout_id=rollout_id,
                    cell_index=cell_index,
                    description=f"Cell {cell_index} reported NORMAL for rollout {rollout_id} but no WitnessSnapshotParamEvent was found",
                )
                continue

            zero_adv_excused_ids = _zero_adv_excused_ids_at(
                zero_adv_witness_ids_by_rollout=zero_adv_witness_ids_by_rollout,
                allocated_witness_ids_by_rollout=allocated_witness_ids_by_rollout,
                rollout_id=rollout_id,
            )

            for event in witness_events_of_cell:
                issue = _compare_snapshot(
                    event=event,
                    expected=expected_witness_ids_of_step.get(rollout_id, set()),
                    rollout_id=rollout_id,
                    cell_index=cell_index,
                    zero_adv_excused_ids=zero_adv_excused_ids,
                )
                if issue is not None:
                    yield issue


def _zero_adv_excused_ids_at(
    *,
    zero_adv_witness_ids_by_rollout: dict[int, set[int]],
    allocated_witness_ids_by_rollout: dict[int, set[int]],
    rollout_id: int,
) -> set[int]:
    # [FUNCTION DOC] HELPER: Computes the cumulative excused (zero-advantage) ids at a rollout.
    # Walks all rollout ids known to either input in ascending order, stopping after the given
    # rollout. At each rollout, first removes ids newly allocated there (an id reallocated to a
    # new sample is no longer excused), then adds that rollout's zero-advantage ids. Returns the
    # resulting set.
    excused: set[int] = set()
    for rid in sorted(set(zero_adv_witness_ids_by_rollout) | set(allocated_witness_ids_by_rollout)):
        if rid > rollout_id:
            break
        excused -= allocated_witness_ids_by_rollout.get(rid, set())
        excused |= zero_adv_witness_ids_by_rollout.get(rid, set())
    return excused


class _HasAttempt(Protocol):
    # [CLASS DOC] HELPER: Structural type for any event with an integer attempt number.
    attempt: int


_EventWithAttemptT = TypeVar("_EventWithAttemptT", bound=_HasAttempt)


def _filter_to_latest_attempt(
    events: Sequence[_EventWithAttemptT],
    *,
    group_key: Callable[[_EventWithAttemptT], Hashable],
) -> list[_EventWithAttemptT]:
    # [FUNCTION DOC] HELPER: Groups events by the key function, finds the maximum attempt number
    # in each group, and returns, in original order, only the events whose attempt equals their
    # group's maximum (so several events of the winning attempt are all kept).
    max_attempt_by_group: dict[Hashable, int] = {}
    for event in events:
        key = group_key(event)
        prev = max_attempt_by_group.get(key)
        if prev is None or event.attempt > prev:
            max_attempt_by_group[key] = event.attempt

    return [e for e in events if e.attempt == max_attempt_by_group[group_key(e)]]


def _compare_snapshot(
    *,
    event: WitnessSnapshotParamEvent,
    expected: set[int],
    rollout_id: int,
    cell_index: int,
    zero_adv_excused_ids: set[int],
) -> WitnessDataMismatchIssue | None:
    # [FUNCTION DOC] HELPER: Compares one snapshot with the expected ids. Expected ids exclude the
    # snapshot's stale ids and the excused zero-advantage ids; actual ids are the snapshot's
    # non-zero witness ids minus the stale ids. Returns None when equal; otherwise a mismatch
    # issue whose description names the snapshot's instance id and the sorted missing and extra
    # ids, with both filtered sets attached as sorted lists.
    stale_set = set(event.stale_ids)
    filtered_expected = expected - stale_set - zero_adv_excused_ids
    filtered_actual = set(event.nonzero_witness_ids) - stale_set

    if filtered_expected == filtered_actual:
        return None

    return WitnessDataMismatchIssue(
        rollout_id=rollout_id,
        cell_index=cell_index,
        description=(
            f"Witness data mismatch for instance {event.instance_id}: "
            f"missing={sorted(filtered_expected - filtered_actual)}, "
            f"extra={sorted(filtered_actual - filtered_expected)}"
        ),
        expected_witness_ids=sorted(filtered_expected),
        actual_witness_ids=sorted(filtered_actual),
    )
