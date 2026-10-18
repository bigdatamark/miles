# [MODULE DOC] Unit tests for the polling reconcile loop, which periodically lists cells (a mapping
# from cell id to cell info) through an injected async lister and tells a reconciler about changes:
# new or changed cells are delivered with their info, vanished cells are delivered once with nothing.
# Starting performs a synchronous initial sync whose failures propagate and leave no background task;
# afterwards a background task polls at the given interval, tolerating listing and reconcile failures
# (a failed delivery is retried on a later tick) until the returned stop function cancels it. Tests use
# a one millisecond poll interval with scripted listers and recording reconcilers.
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pytest

from miles.utils.workers.polling_reconcile_loop import PollingReconcileLoop
from miles.utils.workers.worker_provider.base import CellInfo

POLL_INTERVAL_SECONDS = 0.001


def _cell_info(cell_id: str, *, workers_hash: str = "hash-0") -> CellInfo:
    # [FUNCTION DOC] HELPER: Builds an alive cell info for the cell id in the pool
    # "inference-engine-0-0", with one worker named by the cell id plus "-0", the given workers hash
    # (default "hash-0") and an empty meta mapping.
    return CellInfo(
        cell_id=cell_id,
        pool_id="inference-engine-0-0",
        alive=True,
        worker_names=[f"{cell_id}-0"],
        workers_hash=workers_hash,
        meta={},
    )


@dataclass
class _FakeLister:
    # [CLASS DOC] HELPER: Scripted cell lister. Each call counts and returns the next scripted answer
    # (the last answer repeats forever); an answer that is an exception is raised instead.
    answers: list[Any]
    calls: int = 0

    async def __call__(self) -> dict[str, CellInfo]:
        # [FUNCTION DOC] HELPER: Increments the call counter, picks the answer at the index of the
        # call number minus one, capped at the last answer, raises it if it is an exception, otherwise
        # returns the mapping.
        self.calls += 1
        answer = self.answers[min(self.calls - 1, len(self.answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        return answer


class _RecordingReconciler:
    # [CLASS DOC] HELPER: Reconcile callback that records every (cell id, info or nothing) delivery.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty call list.
        self.calls: list[tuple[str, CellInfo | None]] = []

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: Appends the delivery to the call list.
        self.calls.append((cell_id, info))


class _FailingOnceReconciler(_RecordingReconciler):
    # [CLASS DOC] HELPER: Recording reconciler that raises a runtime error the first time it is given
    # one chosen cell id (without recording that delivery) and behaves normally afterwards.
    def __init__(self, *, failing_cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Runs the base constructor and remembers the cell id to fail once.
        super().__init__()
        self._failing_cell_id: str | None = failing_cell_id

    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: For the chosen cell id on its first delivery, clears the failing id
        # and raises a runtime error "reconcile failed" without recording; otherwise records as usual.
        if cell_id == self._failing_cell_id:
            self._failing_cell_id = None
            raise RuntimeError("reconcile failed")
        await super().__call__(cell_id, info)


class _AlwaysFailingReconciler(_RecordingReconciler):
    # [CLASS DOC] HELPER: Recording reconciler that records each delivery and then always raises.
    async def __call__(self, cell_id: str, info: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: Records the delivery, then raises a runtime error "reconcile rejected
        # the cell".
        await super().__call__(cell_id, info)
        raise RuntimeError("reconcile rejected the cell")


def _make_loop(*answers: Any) -> tuple[PollingReconcileLoop, _FakeLister]:
    # [FUNCTION DOC] HELPER: Builds a scripted lister from the given answers and a polling loop over it
    # with the one millisecond poll interval. Returns the loop and the lister.
    lister = _FakeLister(answers=list(answers))
    loop = PollingReconcileLoop(list_cells=lister, poll_interval_seconds=POLL_INTERVAL_SECONDS)
    return loop, lister


async def _wait_until(predicate, *, timeout_seconds: float = 2.0) -> None:
    # [FUNCTION DOC] HELPER: Polls the zero-argument predicate every millisecond until it is true,
    # failing with an assertion error ("timed out waiting for the reconcile loop") after the timeout
    # (default 2 seconds).
    deadline = time.monotonic() + timeout_seconds
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the reconcile loop"
        await asyncio.sleep(0.001)


class TestPollingReconcileLoopInitialSync:
    # [CLASS DOC] Tests the synchronous initial sync performed by start.
    async def test_every_initial_cell_is_reconciled_before_start_returns(self):
        # [FUNCTION DOC] With two cells listed, right after start the reconciler has been given both,
        # in listing order, with their infos. The loop is stopped in cleanup.
        """Callers may assume the whole listing is observed once start returns."""
        infos = {"cell-a": _cell_info("cell-a"), "cell-b": _cell_info("cell-b")}
        loop, _lister = _make_loop(infos)
        reconciler = _RecordingReconciler()

        stop = await loop.start(reconciler)
        try:
            assert reconciler.calls == [("cell-a", infos["cell-a"]), ("cell-b", infos["cell-b"])]
        finally:
            await stop()

    async def test_a_failing_initial_listing_propagates_instead_of_starting_the_loop(self):
        # [FUNCTION DOC] A lister that fails on the first call makes start raise that error; after
        # waiting 0.02 seconds the lister was called exactly once (no background polling started).
        """A source we never managed to read must not look like an empty source."""
        loop, lister = _make_loop(RuntimeError("source unreachable"))

        with pytest.raises(RuntimeError, match="source unreachable"):
            await loop.start(_RecordingReconciler())

        await asyncio.sleep(0.02)

        assert lister.calls == 1

    async def test_an_initial_reconcile_failure_prevents_the_loop_from_starting(self):
        # [FUNCTION DOC] A reconciler that always fails makes start raise its error; after 0.02
        # seconds the lister was called exactly once.
        """A caller that never learned about the initial cells must not be left with a live loop."""
        loop, lister = _make_loop({"cell-a": _cell_info("cell-a")})

        with pytest.raises(RuntimeError, match="reconcile rejected the cell"):
            await loop.start(_AlwaysFailingReconciler())

        await asyncio.sleep(0.02)

        assert lister.calls == 1

    async def test_a_failed_cell_does_not_prevent_later_cells_from_reconciling(self):
        # [FUNCTION DOC] The reconciler fails once for the first of two cells; start still raises that
        # error, but the second cell was delivered.
        """A failed cell must not prevent later cells in the same listing from being delivered."""
        infos = {"cell-a": _cell_info("cell-a"), "cell-b": _cell_info("cell-b")}
        loop, _lister = _make_loop(infos)
        reconciler = _FailingOnceReconciler(failing_cell_id="cell-a")

        with pytest.raises(RuntimeError, match="reconcile failed"):
            await loop.start(reconciler)

        assert reconciler.calls == [("cell-b", infos["cell-b"])]


class TestPollingReconcileLoopPolling:
    # [CLASS DOC] Tests the background polling after the initial sync. Answers of the scripted lister
    # advance one per tick; the last answer repeats.
    async def test_an_unchanged_cell_is_not_reconciled_again(self):
        # [FUNCTION DOC] With a constant listing, after at least three list calls the reconciler has
        # seen the cell only once.
        """Re-reconciling every tick would restart cells every interval."""
        info = _cell_info("cell-a")
        loop, lister = _make_loop({"cell-a": info})
        reconciler = _RecordingReconciler()

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: lister.calls >= 3)
            assert reconciler.calls == [("cell-a", info)]
        finally:
            await stop()

    async def test_a_cell_appearing_later_is_reconciled_once(self):
        # [FUNCTION DOC] An empty first listing and a later listing with a cell: after at least four
        # list calls the cell was delivered exactly once.
        """A cell created after the initial sync must still reach the consumer, and only once."""
        info = _cell_info("cell-a")
        loop, lister = _make_loop({}, {"cell-a": info})
        reconciler = _RecordingReconciler()

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: lister.calls >= 4)
            assert reconciler.calls == [("cell-a", info)]
        finally:
            await stop()

    async def test_a_disappeared_cell_is_reported_as_gone_exactly_once(self):
        # [FUNCTION DOC] A cell present at first and absent afterwards is delivered with its info and
        # then once with nothing, and not reported again on later ticks.
        """A vanished cell must be delivered as None, and must not be re-reported afterwards."""
        info = _cell_info("cell-a")
        loop, lister = _make_loop({"cell-a": info}, {})
        reconciler = _RecordingReconciler()

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: lister.calls >= 4)
            assert reconciler.calls == [("cell-a", info), ("cell-a", None)]
        finally:
            await stop()

    async def test_a_changed_cell_info_is_reconciled_again(self):
        # [FUNCTION DOC] The same cell id with a different workers hash is delivered a second time
        # with the new info.
        """A replaced cell keeps its id, so only its info can reveal the change."""
        first = _cell_info("cell-a", workers_hash="hash-0")
        second = _cell_info("cell-a", workers_hash="hash-1")
        loop, _lister = _make_loop({"cell-a": first}, {"cell-a": second})
        reconciler = _RecordingReconciler()

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: len(reconciler.calls) >= 2)
            assert reconciler.calls[:2] == [("cell-a", first), ("cell-a", second)]
        finally:
            await stop()

    async def test_a_failing_listing_is_retried_instead_of_killing_the_loop(self):
        # [FUNCTION DOC] Listings: empty, an error, then one cell; the cell is eventually delivered
        # exactly once, so the error did not end the loop.
        """One unreachable listing call must not silently end the observation."""
        info = _cell_info("cell-a")
        loop, _lister = _make_loop({}, RuntimeError("transient"), {"cell-a": info})
        reconciler = _RecordingReconciler()

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: reconciler.calls == [("cell-a", info)])
        finally:
            await stop()

    async def test_a_failed_reconcile_is_retried_before_the_cell_is_marked_seen(self):
        # [FUNCTION DOC] A cell appears after the initial sync and the reconciler fails on its first
        # delivery; the cell is delivered again on a later tick and recorded exactly once.
        """A cell whose reconcile raised must be delivered again instead of being silently forgotten."""
        info = _cell_info("cell-a")
        loop, _lister = _make_loop({}, {"cell-a": info})
        reconciler = _FailingOnceReconciler(failing_cell_id="cell-a")

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: reconciler.calls == [("cell-a", info)])
        finally:
            await stop()

    async def test_a_cell_added_by_a_partially_failed_tick_can_still_disappear(self):
        # [FUNCTION DOC] A tick lists cells a and b where the reconciler fails once for b (a is
        # delivered); a later tick lists only b. Eventually cell a is reported gone, and its earlier
        # delivery is in the call record.
        """A reconcile that raises mid-tick must not lose the bookkeeping of the cells already delivered."""
        info_a = _cell_info("cell-a")
        info_b = _cell_info("cell-b")
        loop, _lister = _make_loop({}, {"cell-a": info_a, "cell-b": info_b}, {"cell-b": info_b})
        reconciler = _FailingOnceReconciler(failing_cell_id="cell-b")

        stop = await loop.start(reconciler)
        try:
            await _wait_until(lambda: ("cell-a", None) in reconciler.calls)
            assert ("cell-a", info_a) in reconciler.calls
        finally:
            await stop()


class TestPollingReconcileLoopStop:
    # [CLASS DOC] Tests the stop function returned by start.
    async def test_stopping_ends_the_polling(self):
        # [FUNCTION DOC] After at least two list calls, stopping freezes the call counter: it does not
        # change over the next 0.02 seconds.
        """The returned stop function must actually stop the loop, not just detach from it."""
        loop, lister = _make_loop({})

        stop = await loop.start(_RecordingReconciler())
        await _wait_until(lambda: lister.calls >= 2)
        await stop()
        settled = lister.calls

        await asyncio.sleep(0.02)

        assert lister.calls == settled

    async def test_stopping_twice_is_not_an_error(self):
        # [FUNCTION DOC] Calling the stop function twice raises nothing.
        """A provider disposed twice is ordinary teardown, and raising there would mask the first failure."""
        loop, _lister = _make_loop({})

        stop = await loop.start(_RecordingReconciler())
        await stop()
        await stop()

    async def test_stopping_cancels_an_in_flight_listing(self):
        # [FUNCTION DOC] A custom lister returns an empty mapping on the first call (initial sync) and
        # blocks on an event on the second. With a zero poll interval, after the blocked listing is
        # entered, stopping cancels it: the lister observes the cancellation, and it was called exactly
        # twice.
        """Stopping while listing must cancel and await the active external operation."""
        listing_entered = asyncio.Event()
        listing_cancelled = asyncio.Event()
        release_listing = asyncio.Event()
        calls = 0

        async def list_cells() -> dict[str, CellInfo]:
            # [FUNCTION DOC] HELPER: Counts calls. The first call returns an empty mapping. Later calls
            # signal entry, wait for the release event (never set), and on cancellation signal that
            # and re-raise it; the final return is unreachable in practice.
            nonlocal calls
            calls += 1
            if calls == 1:
                return {}
            listing_entered.set()
            try:
                await release_listing.wait()
            except asyncio.CancelledError:
                listing_cancelled.set()
                raise
            return {}

        loop = PollingReconcileLoop(list_cells=list_cells, poll_interval_seconds=0)
        stop = await loop.start(_RecordingReconciler())
        await listing_entered.wait()

        await stop()

        assert listing_cancelled.is_set()
        assert calls == 2

    async def test_a_failed_initial_sync_leaves_no_polling_behind(self):
        # [FUNCTION DOC] After start raises for a failing first listing, the lister call counter does
        # not change over the next 0.02 seconds.
        """start() raises, so the caller never receives a stop function and could not stop a task it left running."""
        loop, lister = _make_loop(RuntimeError("list failed"))

        with pytest.raises(RuntimeError, match="list failed"):
            await loop.start(_RecordingReconciler())
        settled = lister.calls

        await asyncio.sleep(0.02)

        assert lister.calls == settled


class TestTwoLoopsOverOneProvider:
    # [CLASS DOC] Tests that several loops sharing one lister do not share state.
    async def test_each_loop_keeps_its_own_view_of_what_it_has_seen(self):
        # [FUNCTION DOC] Two loops over the same lister each start with their own recording
        # reconciler and are stopped; both reconcilers were given cell-0 (a shared seen set would have
        # starved the second).
        """A shared seen-set would make the second loop skip the cells the first one already reported."""
        lister = _FakeLister(answers=[{"cell-0": _cell_info("cell-0")}])
        first = PollingReconcileLoop(list_cells=lister, poll_interval_seconds=POLL_INTERVAL_SECONDS)
        second = PollingReconcileLoop(list_cells=lister, poll_interval_seconds=POLL_INTERVAL_SECONDS)

        first_reconciler, second_reconciler = _RecordingReconciler(), _RecordingReconciler()
        first_stop = await first.start(first_reconciler)
        second_stop = await second.start(second_reconciler)
        await first_stop()
        await second_stop()

        assert [cell_id for cell_id, _ in first_reconciler.calls] == ["cell-0"]
        assert [cell_id for cell_id, _ in second_reconciler.calls] == ["cell-0"]
