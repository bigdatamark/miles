# [MODULE DOC] Tests for the witness id allocator: a ring buffer of ids handed out monotonically
# with wraparound, each allocation also reporting stale ids (slots not among the most recent
# kept ones, at 70 percent of the buffer); the pure stale id computation; resuming from a
# persisted counter; and reading the persisted counter back from event log files.
"""Tests for miles.utils.audit_utils.witness.allocator: WitnessIdAllocator, _compute_stale_ids."""

import json

import pytest

from miles.utils.audit_utils.witness.allocator import (
    WitnessIdAllocator,
    WitnessInfo,
    _compute_stale_ids,
    read_persisted_witness_counter,
)


class TestWitnessIdAllocator:
    # [CLASS DOC] Tests of allocation behavior.
    def test_monotonic_and_wraps(self) -> None:
        # [FUNCTION DOC] With buffer size 5, allocating 3 gives ids 0,1,2 and then 4 more gives
        # 3,4,0,1 (wrapping modulo the buffer size).
        allocator = WitnessIdAllocator(buffer_size=5)

        info1 = allocator.allocate(3)
        assert info1.witness_ids == [0, 1, 2]

        info2 = allocator.allocate(4)
        assert info2.witness_ids == [3, 4, 0, 1]

    def test_allocate_returns_correct_count(self) -> None:
        # [FUNCTION DOC] Allocating 7 returns 7 ids.
        allocator = WitnessIdAllocator(buffer_size=100)
        info = allocator.allocate(7)
        assert len(info.witness_ids) == 7

    def test_allocate_returns_witness_info(self) -> None:
        # [FUNCTION DOC] The result is a witness info record with 3 ids and a list of stale ids.
        allocator = WitnessIdAllocator(buffer_size=10)
        info = allocator.allocate(3)
        assert isinstance(info, WitnessInfo)
        assert len(info.witness_ids) == 3
        assert isinstance(info.stale_ids, list)

    def test_stale_ids_computed_on_allocate(self) -> None:
        # [FUNCTION DOC] With buffer size 10, allocating 8 gives ids 0 to 7 and stale ids 8, 9, 0
        # (three slots: ten minus the seven kept).
        """Allocating 8 from buffer_size=10 should produce stale IDs (keep 70% = 7)."""
        allocator = WitnessIdAllocator(buffer_size=10)
        info = allocator.allocate(8)
        assert info.witness_ids == [0, 1, 2, 3, 4, 5, 6, 7]
        assert set(info.stale_ids) == {8, 9, 0}

    def test_allocate_zero_ids(self) -> None:
        # [FUNCTION DOC] Allocating zero ids after a prior allocation returns no ids and a list of
        # stale ids.
        """allocate(num_ids=0) should return empty witness_ids and reasonable stale_ids."""
        allocator = WitnessIdAllocator(buffer_size=5)
        allocator.allocate(3)
        info = allocator.allocate(0)
        assert info.witness_ids == []
        assert isinstance(info.stale_ids, list)

    def test_non_positive_buffer_size_raises(self) -> None:
        # [FUNCTION DOC] Buffer sizes 0 and -1 both raise a value error saying it must be
        # positive.
        """buffer_size <= 0 raises ValueError at construction."""
        with pytest.raises(ValueError, match="must be positive"):
            WitnessIdAllocator(buffer_size=0)
        with pytest.raises(ValueError, match="must be positive"):
            WitnessIdAllocator(buffer_size=-1)

    def test_negative_num_ids_raises(self) -> None:
        # [FUNCTION DOC] Allocating -1 ids raises a value error saying it must be non-negative and
        # leaves the counter at 0.
        """allocate(num_ids<0) raises ValueError and leaves the counter untouched."""
        allocator = WitnessIdAllocator(buffer_size=5)
        with pytest.raises(ValueError, match="must be non-negative"):
            allocator.allocate(num_ids=-1)
        assert allocator.counter == 0

    def test_allocate_exceeds_buffer_size_raises(self) -> None:
        # [FUNCTION DOC] Allocating 10 from a buffer of 5 raises an assertion error mentioning
        # that it exceeds the buffer size.
        """num_ids > buffer_size raises AssertionError."""
        allocator = WitnessIdAllocator(buffer_size=5)
        with pytest.raises(AssertionError, match="exceeds buffer_size"):
            allocator.allocate(num_ids=10)

    def test_consecutive_allocations_stale_ids_evolve(self) -> None:
        # [FUNCTION DOC] Three allocations of 3 from a buffer of 10; asserts the stale id sets are
        # not all identical and each is a subset of the buffer's ids.
        """Consecutive allocate calls should produce evolving stale_ids as counter grows."""
        allocator = WitnessIdAllocator(buffer_size=10)

        info1 = allocator.allocate(3)
        stale1 = set(info1.stale_ids)

        info2 = allocator.allocate(3)
        stale2 = set(info2.stale_ids)

        info3 = allocator.allocate(3)
        stale3 = set(info3.stale_ids)

        assert stale1 != stale2 or stale2 != stale3, "stale_ids should evolve across allocations"
        all_ids = set(range(10))
        for stale in [stale1, stale2, stale3]:
            assert stale.issubset(all_ids)


class TestComputeStaleIds:
    # [CLASS DOC] Direct tests of the pure stale id function, called with keep count, counter
    # (total ids allocated so far) and buffer size. The active slots are the minimum of those
    # three; the stale ids are the remaining slots starting at the head (counter modulo buffer)
    # and wrapping.
    """Direct tests for _compute_stale_ids module-level function."""

    def test_counter_zero_returns_empty(self) -> None:
        # [FUNCTION DOC] Counter 0, keep 5, buffer 10: empty result.
        assert _compute_stale_ids(keep_count=5, counter=0, buffer_size=10) == []

    def test_counter_zero_keep_zero_returns_empty(self) -> None:
        # [FUNCTION DOC] Counter 0 and keep 0: empty result.
        assert _compute_stale_ids(keep_count=0, counter=0, buffer_size=10) == []

    def test_counter_zero_large_buffer_returns_empty(self) -> None:
        # [FUNCTION DOC] Counter 0 with a large buffer and keep 100: empty result.
        assert _compute_stale_ids(keep_count=100, counter=0, buffer_size=1000) == []

    def test_counter_less_than_keep_count_returns_all_unused(self) -> None:
        # [FUNCTION DOC] Counter 3 below keep 7 in a buffer of 10: 3 active slots, so stale ids
        # are the seven slots 3 to 9.
        # counter=3, buffer=10, keep=7 → active=min(7,3,10)=3 → stale=7 slots
        result = _compute_stale_ids(keep_count=7, counter=3, buffer_size=10)
        assert set(result) == {3, 4, 5, 6, 7, 8, 9}

    def test_keep_count_equals_buffer_size_returns_empty(self) -> None:
        # [FUNCTION DOC] Keep equal to the buffer size: all slots active, empty result.
        # All slots are active
        assert _compute_stale_ids(keep_count=10, counter=15, buffer_size=10) == []

    def test_keep_count_exceeds_buffer_size_returns_empty(self) -> None:
        # [FUNCTION DOC] Keep larger than the buffer: empty result.
        assert _compute_stale_ids(keep_count=20, counter=15, buffer_size=10) == []

    def test_basic_no_wrap(self) -> None:
        # [FUNCTION DOC] Counter 8, keep 5, buffer 10: stale ids start at head 8 and wrap, giving
        # 8, 9, 0, 1, 2 in that order.
        # counter=8, buffer=10, keep=5 → stale=5, head=8 → stale=[8,9,0,1,2]
        result = _compute_stale_ids(keep_count=5, counter=8, buffer_size=10)
        assert result == [8, 9, 0, 1, 2]

    def test_basic_wrap(self) -> None:
        # [FUNCTION DOC] Counter 3, keep 5, buffer 10: only 3 slots active, stale ids 3 to 9 in
        # order.
        # counter=3, buffer=10, keep=5 → active=min(5,3,10)=3 → stale=7, head=3 → stale=[3,4,5,6,7,8,9]
        result = _compute_stale_ids(keep_count=5, counter=3, buffer_size=10)
        assert result == [3, 4, 5, 6, 7, 8, 9]

    def test_head_at_zero(self) -> None:
        # [FUNCTION DOC] Counter 10, keep 3, buffer 10: head is 0, stale ids 0 to 6 in order.
        # counter=10, buffer=10, keep=3 → stale=7, head=0 → stale=[0,1,2,3,4,5,6]
        result = _compute_stale_ids(keep_count=3, counter=10, buffer_size=10)
        assert result == [0, 1, 2, 3, 4, 5, 6]

    def test_keep_one(self) -> None:
        # [FUNCTION DOC] Counter 5, keep 1, buffer 10: nine stale ids and slot 4 (the single
        # active one) is not among them.
        # counter=5, buffer=10, keep=1 → stale=9, head=5 → stale=[5,6,7,8,9,0,1,2,3]
        result = _compute_stale_ids(keep_count=1, counter=5, buffer_size=10)
        assert len(result) == 9
        assert 4 not in result

    def test_keep_zero(self) -> None:
        # [FUNCTION DOC] Keep 0: every slot of the buffer is stale.
        # All slots stale
        result = _compute_stale_ids(keep_count=0, counter=5, buffer_size=10)
        assert len(result) == 10
        assert set(result) == set(range(10))

    def test_stale_and_active_are_disjoint_and_cover_buffer(self) -> None:
        # [FUNCTION DOC] Sweeps counters 0 to 100 and keeps 0 to 15 on a buffer of 10. For counter
        # 0 asserts empty; otherwise the stale count equals buffer size minus the active count
        # (minimum of keep, counter, 10), with no duplicates and all ids in range.
        for counter in [0, 1, 5, 10, 13, 20, 100]:
            for keep in [0, 1, 3, 7, 10, 15]:
                stale = _compute_stale_ids(keep_count=keep, counter=counter, buffer_size=10)
                if counter == 0:
                    assert stale == [], "counter=0 → no stale IDs"
                    continue
                active_count = min(keep, counter, 10)
                assert len(stale) == 10 - active_count, f"counter={counter}, keep={keep}"
                assert len(set(stale)) == len(stale), "no duplicates"
                assert all(0 <= x < 10 for x in stale), "all in range"

    def test_buffer_size_one(self) -> None:
        # [FUNCTION DOC] Buffer of 1: keep 1 gives no stale ids, keep 0 gives slot 0.
        assert _compute_stale_ids(keep_count=1, counter=5, buffer_size=1) == []
        assert _compute_stale_ids(keep_count=0, counter=5, buffer_size=1) == [0]


class TestWitnessIdAllocatorResume:
    # [CLASS DOC] Tests of resuming an allocator from a saved counter.
    def test_resume_continues_allocation_without_reusing_ids(self) -> None:
        # [FUNCTION DOC] An allocator that issued 10 ids has counter 10; a fresh allocator resumed
        # from it allocates ids 10 to 14 and reaches counter 15.
        """Resuming from a persisted counter issues fresh ids, as if the run never stopped."""
        saved = WitnessIdAllocator(buffer_size=100)
        saved.allocate(10)
        assert saved.counter == 10

        resumed = WitnessIdAllocator(buffer_size=100)
        resumed.resume(saved.counter)
        info = resumed.allocate(5)

        assert info.witness_ids == [10, 11, 12, 13, 14]
        assert resumed.counter == 15

    def test_resume_backwards_is_rejected(self) -> None:
        # [FUNCTION DOC] Resuming to counter 3 after 10 ids were issued raises an assertion error.
        """resume() must never move the counter backwards."""
        allocator = WitnessIdAllocator(buffer_size=100)
        allocator.allocate(10)

        with pytest.raises(AssertionError):
            allocator.resume(3)

    def test_counter_matches_uninterrupted_run(self) -> None:
        # [FUNCTION DOC] With a buffer of 7, four allocations of 3 on one allocator give the same
        # id lists as two allocations, then a new allocator resumed from the counter doing two more.
        """A save/resume sequence allocates the same ids as one uninterrupted allocator."""
        uninterrupted = WitnessIdAllocator(buffer_size=7)
        expected = [uninterrupted.allocate(3).witness_ids for _ in range(4)]

        first = WitnessIdAllocator(buffer_size=7)
        actual = [first.allocate(3).witness_ids, first.allocate(3).witness_ids]
        second = WitnessIdAllocator(buffer_size=7)
        second.resume(first.counter)
        actual += [second.allocate(3).witness_ids, second.allocate(3).witness_ids]

        assert actual == expected


class TestReadPersistedWitnessCounter:
    # [CLASS DOC] Tests of reading the resume counter from witness allocate event log files.
    def _write_events(self, event_dir, counters) -> None:
        # [FUNCTION DOC] HELPER: Creates the directory (with parents) and writes a file main.jsonl
        # with one witness allocate event line per counter value: type, a fixed timestamp, main
        # source, rollout id equal to the position, attempt 0, an empty id-to-sample map, and the
        # counter after value. Overwrites the file.
        event_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for i, counter in enumerate(counters):
            lines.append(
                json.dumps(
                    {
                        "type": "witness_allocate_id",
                        "timestamp": "2026-06-12T00:00:00Z",
                        "source": {"component": "main"},
                        "rollout_id": i,
                        "attempt": 0,
                        "witness_id_to_sample_index": {},
                        "counter_after": counter,
                    }
                )
            )
        (event_dir / "main.jsonl").write_text("\n".join(lines) + "\n")

    def test_reads_latest_counter_from_events(self, tmp_path) -> None:
        # [FUNCTION DOC] Events with counters 3, 6, 9 give a persisted counter of 9.
        """The max counter_after across allocate events is the resume point."""
        self._write_events(tmp_path / "events", [3, 6, 9])

        assert read_persisted_witness_counter(tmp_path / "events") == 9

    def test_empty_or_missing_dir_resumes_from_zero(self, tmp_path) -> None:
        # [FUNCTION DOC] A nonexistent directory gives 0.
        """A fresh run (no events yet) starts allocation at zero."""
        assert read_persisted_witness_counter(tmp_path / "missing") == 0
