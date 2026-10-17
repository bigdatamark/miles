# [MODULE DOC] HELPER: Test assertions over the audit events that record cell reconfiguration
# (fault recovery) in fault-tolerance tests: an exact-sequence check and soak-test checks that
# enough faults were injected and enough cells healed.
from pathlib import Path

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ReconfigureInfo(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: Immutable comparable summary of one reconfigure event: the rollout id, the
    # source cell index used for healing (may be none), the list of healed cell indices, and the
    # list of alive cell indices after the reconfiguration.
    rollout_id: int
    src_cell_index: int | None
    healed_cell_indices: list[int]
    alive_cell_indices_after: list[int]

    @staticmethod
    def from_event(event: CellReconfigureEvent) -> "ReconfigureInfo":
        # [FUNCTION DOC] HELPER: Copies the four fields of the event into a summary object.
        return ReconfigureInfo(
            rollout_id=event.rollout_id,
            src_cell_index=event.src_cell_index,
            healed_cell_indices=event.healed_cell_indices,
            alive_cell_indices_after=event.alive_cell_indices_after,
        )


def assert_reconfigure_events(event_dir: Path, *, expected: list[ReconfigureInfo]) -> None:
    # [FUNCTION DOC] HELPER: Asserts the event directory exists, loads its reconfigure events in
    # read order, summarizes them, and asserts the list equals the expected list exactly (order and
    # values); the failure message shows both lists.
    assert event_dir.is_dir(), f"Event directory {event_dir} does not exist or is not a directory"
    actual = [ReconfigureInfo.from_event(event) for event in load_reconfigure_events(event_dir)]
    assert actual == expected, (
        f"CellReconfigureEvent sequence mismatch in {event_dir}:\n" f"  expected: {expected}\n" f"  actual:   {actual}"
    )


MIN_SOAK_INJECTIONS: int = 2
MIN_SOAK_HEALINGS: int = 2


def assert_min_soak_injections(num_successful_injections: int, *, context: str) -> None:
    # [FUNCTION DOC] HELPER: Asserts at least two successful fault injections occurred (so recovery
    # was exercised more than once); the failure message includes the context string.
    assert num_successful_injections >= MIN_SOAK_INJECTIONS, (
        f"Soak proved too little in {context}: the fault injector reported only "
        f"{num_successful_injections} successful injection(s), need >= {MIN_SOAK_INJECTIONS} "
        f"to exercise fault recovery more than once"
    )


def assert_soak_reconfigure_events(event_dir: Path, *, num_successful_injections: int) -> None:
    # [FUNCTION DOC] HELPER: Soak-test check. Asserts the event directory exists; loads reconfigure
    # events and flattens all healed cell indices (a cell healed in several events counts each
    # time); asserts the minimum injection count; asserts at least two healed cells in total (message
    # lists the summaries of all events); prints a pass line with event, healed cell and injection
    # counts.
    assert event_dir.is_dir(), f"Event directory {event_dir} does not exist or is not a directory"
    events = load_reconfigure_events(event_dir)
    healed_cell_indices = [cell_index for event in events for cell_index in event.healed_cell_indices]

    assert_min_soak_injections(num_successful_injections, context=str(event_dir))
    assert len(healed_cell_indices) >= MIN_SOAK_HEALINGS, (
        f"Healing witness failed in {event_dir}: {num_successful_injections} successful injection(s) "
        f"but only {len(healed_cell_indices)} healed cell(s), need >= {MIN_SOAK_HEALINGS} "
        f"(reconfigure events: {[ReconfigureInfo.from_event(event) for event in events]})"
    )

    print(
        f"Soak reconfigure witness assertion passed: {len(events)} reconfigure event(s) "
        f"({len(healed_cell_indices)} healed cell(s)) for {num_successful_injections} successful injection(s) "
        f"in {event_dir}"
    )


def load_reconfigure_events(event_dir: Path) -> list[CellReconfigureEvent]:
    # [FUNCTION DOC] HELPER: Reads all audit events from the directory and returns only the cell
    # reconfigure events, in read order.
    return [event for event in read_events(event_dir) if isinstance(event, CellReconfigureEvent)]
