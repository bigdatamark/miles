# [MODULE DOC]
# Tests for killing and replacing trainer cells: direct kill of a failed cell's workers without
# the worker manager, a replacement cell picking up fresh actor handles, idempotent repeated
# kills, the timeout that bounds a kill rpc which never returns, falling through to death
# confirmation after hanging kills, and the ordering rule that every kill request is sent before
# any death confirmation is awaited. Uses dummy ray actors and fake handles.
import asyncio

import pytest
import ray
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_cell

from miles.ray.train import cell as cell_module
from miles.utils.workers.worker_handle import BaseWorkerHandle

pytestmark = pytest.mark.asyncio


class _HangingKillSelfHandle(BaseWorkerHandle):
    # [CLASS DOC] HELPER: Concrete worker handle used to simulate a hung kill rpc. Counts calls to
    # kill and wait-dead; the kill never returns (it waits on an event nobody sets); wait-ready and
    # the dead probe are not implemented.
    """A worker handle whose kill_self never returns, so _kill_worker has to time it out."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts both call counters at zero.
        self.kill_self_call_count: int = 0
        self.wait_dead_call_count: int = 0

    async def kill_self(self) -> None:
        # [FUNCTION DOC] HELPER: Counts the call, then waits forever (until cancelled).
        self.kill_self_call_count += 1
        await asyncio.Event().wait()

    async def wait_ready(self, *, timeout: float) -> None:
        # [FUNCTION DOC] HELPER: Not implemented; raises a not-implemented error.
        raise NotImplementedError

    async def wait_dead(self, *, timeout: float) -> None:
        # [FUNCTION DOC] HELPER: Counts the call and returns at once; the timeout is ignored.
        self.wait_dead_call_count += 1

    async def probe_is_dead(self) -> bool:
        # [FUNCTION DOC] HELPER: Not implemented; raises a not-implemented error.
        raise NotImplementedError


class TestCellKillAndRestart:
    # [CLASS DOC] Tests killing a cell's workers directly and replacing the cell.
    async def test_killing_a_failed_cell_reaches_the_workers_directly(self):
        # [FUNCTION DOC] Killing a fresh cell's workers records no stop-cells request on the fake
        # manager, and every raw actor raises an actor death error on a later call.
        """Waiting for an external controller would leave the other cells hanging in NCCL."""
        cell = make_cell(2)
        handles = get_raw_actor_handles(cell)

        await cell._kill_workers_and_confirm_dead()

        assert train_conftest.fake_worker_manager.stopped_cell_ids == []
        for handle in handles:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())

    async def test_a_replacement_cell_picks_up_the_fresh_actor_handles(self):
        # [FUNCTION DOC] After the old cell's workers are killed and the manager stops the cell, a
        # replacement cell for the same index has raw actor handles different from the old ones.
        """Reusing the dead handles would make every later call fail."""
        cell = make_cell(0)
        old_handles = get_raw_actor_handles(cell)
        await cell._kill_workers_and_confirm_dead()

        train_conftest.fake_worker_manager._stop_cells([cell.cell_id])
        replacement = make_cell(0)

        assert get_raw_actor_handles(replacement) != old_handles

    async def test_killing_twice_is_harmless(self):
        # [FUNCTION DOC] Killing and confirming death of the same cell's workers twice in a row
        # raises no error.
        """Healing may tear down an already dead cell, which must not raise."""
        cell = make_cell(0)
        await cell._kill_workers_and_confirm_dead()

        await cell._kill_workers_and_confirm_dead()


class TestKillRpcTimeout:
    # [CLASS DOC] Tests the timeout applied to the kill rpc (patched to 0.05 seconds in these tests).
    async def test_a_kill_rpc_that_never_returns_gives_up_at_the_timeout(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Killing through a handle whose kill never returns completes within 10
        # seconds (it gives up at the patched timeout) and the kill was attempted exactly once.
        """A worker whose kill_self RPC hangs forever must not block _kill_worker forever."""
        monkeypatch.setattr(cell_module, "KILL_RPC_TIMEOUT_S", 0.05)
        handle = _HangingKillSelfHandle()

        await asyncio.wait_for(cell_module._kill_worker(handle), timeout=10.0)

        assert handle.kill_self_call_count == 1

    async def test_a_hanging_kill_rpc_still_reaches_the_death_confirmation(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] The cell's handles are replaced by two hanging-kill handles. Kill and
        # confirm completes within 10 seconds, and each handle saw exactly one kill and one
        # wait-dead call.
        # KNOWN ISSUE: the wait-dead count assertion is written twice (a duplicate line), and the
        # cell is built with 2 as its first positional argument, which is the cell index, not the
        # actor count (the handles are replaced anyway).
        """When every kill_self RPC hangs, teardown must fall through to the death probe instead of stalling."""
        monkeypatch.setattr(cell_module, "KILL_RPC_TIMEOUT_S", 0.05)
        cell = make_cell(2)
        hanging_handles: list[_HangingKillSelfHandle] = [_HangingKillSelfHandle(), _HangingKillSelfHandle()]
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: hanging_handles)

        await asyncio.wait_for(cell._kill_workers_and_confirm_dead(), timeout=10.0)

        assert [handle.wait_dead_call_count for handle in hanging_handles] == [1, 1]
        assert [handle.kill_self_call_count for handle in hanging_handles] == [1, 1]
        assert [handle.wait_dead_call_count for handle in hanging_handles] == [1, 1]


class _OrderRecordingKillHandle:
    # [CLASS DOC] HELPER: Fake worker handle recording kill and wait-dead calls as events named
    # kill-N and wait-N (N is the handle index) in a shared list, yielding to the event loop once
    # in each call.
    def __init__(self, events: list[str], index: int) -> None:
        # [FUNCTION DOC] HELPER: Stores the shared event list and this handle's index.
        self._events = events
        self._index = index

    async def kill_self(self) -> None:
        # [FUNCTION DOC] HELPER: Appends a kill event for this index and yields once.
        self._events.append(f"kill-{self._index}")
        await asyncio.sleep(0)

    async def wait_dead(self, *, timeout: float) -> None:
        # [FUNCTION DOC] HELPER: Appends a wait event for this index and yields once; the timeout is ignored.
        self._events.append(f"wait-{self._index}")
        await asyncio.sleep(0)


class TestKillOrdering:
    # [CLASS DOC] Tests the ordering of kill and death confirmation across a cell's workers.
    async def test_every_kill_request_is_sent_before_any_death_is_confirmed(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With three recording handles substituted into the cell, kill and confirm
        # produces three kill events followed by three wait events (all kills before any wait), and
        # the set of events is exactly kill and wait for indices 0 to 2.
        """Confirming rank 0's death first blocks the kill of ranks still stuck inside a collective."""
        events: list[str] = []
        cell = make_cell(2)
        handles = [_OrderRecordingKillHandle(events, index) for index in range(3)]
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: handles)

        await cell._kill_workers_and_confirm_dead()

        assert [event.split("-")[0] for event in events] == ["kill", "kill", "kill", "wait", "wait", "wait"]
        assert sorted(events) == ["kill-0", "kill-1", "kill-2", "wait-0", "wait-1", "wait-2"]
