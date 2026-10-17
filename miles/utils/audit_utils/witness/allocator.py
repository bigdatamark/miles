# [MODULE DOC] Ring-buffer allocator of witness ids. Each training sample gets an id equal to a
# running counter modulo the buffer size; ids older than a 70 percent keep window are reported as
# stale so the witness module can clear them. Also recovers the counter from persisted events.

from pathlib import Path

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import WitnessAllocateIdEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class WitnessInfo(FrozenStrictBaseModel):
    # [CLASS DOC] Result of one allocation: the ids assigned (in order) and the stale ids to be
    # cleared after the step.
    witness_ids: list[int]
    stale_ids: list[int]


class WitnessIdAllocator:
    # [CLASS DOC] Stateful allocator holding a buffer size and a monotonically increasing counter
    # starting at 0. Not thread safe.
    def __init__(self, *, buffer_size: int) -> None:
        # [FUNCTION DOC] Stores the (keyword-only) buffer size and zeroes the counter. Raises a
        # value error if the size is not positive.
        if buffer_size <= 0:
            raise ValueError(f"buffer_size ({buffer_size}) must be positive.")
        self._buffer_size = buffer_size
        self._counter: int = 0

    @property
    def counter(self) -> int:
        # [FUNCTION DOC] Returns the number of ids allocated so far (including resumed ones).
        return self._counter

    def resume(self, counter: int) -> None:
        # [FUNCTION DOC] Fast-forwards the counter to a persisted value; asserts it is not lower
        # than the current counter. Mutates the allocator.
        assert counter >= self._counter
        self._counter = counter

    def allocate(self, num_ids: int) -> WitnessInfo:
        # [FUNCTION DOC] Allocates a consecutive run of ids. Raises a value error for a negative
        # count; asserts the count does not exceed the buffer size (message advises raising the
        # witness buffer size option). Ids are the counter plus offset, modulo the buffer size.
        # Stale ids are computed for the counter value after this allocation, keeping the most
        # recent 70 percent of the buffer (rounded down). Then the counter advances. Returns the
        # ids and stale ids.
        if num_ids < 0:
            raise ValueError(f"num_ids ({num_ids}) must be non-negative.")
        assert num_ids <= self._buffer_size, (
            f"num_ids ({num_ids}) exceeds buffer_size ({self._buffer_size}). " f"Increase --witness-buffer-size."
        )
        ids = [(self._counter + i) % self._buffer_size for i in range(num_ids)]
        stale_ids = _compute_stale_ids(
            keep_count=int(self._buffer_size * 0.7),
            counter=self._counter + num_ids,
            buffer_size=self._buffer_size,
        )
        self._counter += num_ids
        return WitnessInfo(witness_ids=ids, stale_ids=stale_ids)


def read_persisted_witness_counter(event_dir: Path) -> int:
    # [FUNCTION DOC] Reads all events in the directory and returns the maximum counter-after value
    # among witness allocation events, or 0 if there are none. Reads files.
    """Recover the allocator counter from the (checkpoint-restored) event directory."""
    events = read_events(event_dir)
    return max((e.counter_after for e in events if isinstance(e, WitnessAllocateIdEvent)), default=0)


def _compute_stale_ids(*, keep_count: int, counter: int, buffer_size: int) -> list[int]:
    # [FUNCTION DOC] HELPER: Given the counter after allocation, returns the ring slots that are
    # considered stale. Empty if the counter is zero. The number of stale slots is the buffer size
    # minus the smallest of keep count, counter and buffer size (empty if that is zero). The stale
    # slots are consecutive starting at counter modulo buffer size, wrapping around, i.e. the
    # oldest slots, which are next to be reused.
    if counter == 0:
        return []
    num_stale = buffer_size - min(keep_count, counter, buffer_size)
    if num_stale == 0:
        return []

    head = counter % buffer_size
    return [(head + i) % buffer_size for i in range(num_stale)]
