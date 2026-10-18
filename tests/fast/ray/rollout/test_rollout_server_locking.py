# [MODULE DOC]
# Unit tests for the lock discipline of the rollout server: its methods and engine list properties
# may only be used by a caller that holds the very context lock object the server was given (an
# assertion error otherwise, mentioning "must be called with"), one lock can be shared by several
# servers, and the wait for the expected number of cells needs no lock and must not hold it while
# polling so cells can still be added.
import asyncio
from types import SimpleNamespace

import pytest

from miles.ray.rollout.rollout_server import RolloutServer
from miles.utils.context_lock import ContextLock
from miles.utils.workers.worker_spec import NamedHostAndPorts


class _StubProvider:
    # [CLASS DOC] HELPER: Stub provider that fails the test if any address is requested.
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] HELPER: Always raises an assertion error naming the worker.
        raise AssertionError(f"no cell in this module is ever addressed ({worker_name=})")


def _make_server(context_lock: ContextLock | None = None, **overrides) -> RolloutServer:
    # [FUNCTION DOC] HELPER: Builds a rollout server with no cells, minimal colocated arguments, the
    # given lock or a new one named after the inference controller, the stub provider and any extra
    # constructor keyword overrides. Returns the server.
    return RolloutServer(
        server_cells={},
        args=SimpleNamespace(colocate=True),
        context_lock=context_lock or ContextLock("InferenceController"),
        engine_provider=_StubProvider(),
        **overrides,
    )


class TestRolloutServerLockDiscipline:
    # [CLASS DOC] Tests which callers the server accepts, depending on the lock they hold.
    @pytest.mark.asyncio
    async def test_methods_reject_callers_that_do_not_hold_the_lock(self):
        # [FUNCTION DOC] Calling offload without holding any lock raises an assertion error with
        # the text "must be called with".
        """Driving RolloutServer directly, bypassing InferenceController, is rejected."""
        srv = _make_server()
        with pytest.raises(AssertionError, match="must be called with"):
            await srv.offload()

    def test_properties_reject_callers_that_do_not_hold_the_lock(self):
        # [FUNCTION DOC] Synchronous test: reading the api clients property without the lock raises
        # an assertion error with the text "must be called with".
        """Engine list snapshots are only meaningful under the controller lock."""
        srv = _make_server()
        with pytest.raises(AssertionError, match="must be called with"):
            _ = srv.api_clients

    @pytest.mark.asyncio
    async def test_methods_accept_callers_inside_the_lock(self):
        # [FUNCTION DOC] Inside the server's own lock, offload and onload succeed and the api
        # clients, engine GPU counts and engine GPU offsets properties are all empty lists.
        """A caller holding the controller lock may drive RolloutServer freely."""
        srv = _make_server()
        async with srv.context_lock:
            await srv.offload()
            await srv.onload()
            assert srv.api_clients == []
            assert srv.engine_gpu_counts == []
            assert srv.engine_gpu_offsets == []

    @pytest.mark.asyncio
    async def test_holding_a_different_lock_object_does_not_authorize_the_server(self):
        # [FUNCTION DOC] Holding a different lock object (same name) while calling offload still
        # raises the "must be called with" assertion error.
        """The server must be handed the controller's own lock, not merely some lock."""
        srv = _make_server()
        async with ContextLock("InferenceController"):
            with pytest.raises(AssertionError, match="must be called with"):
                await srv.offload()

    @pytest.mark.asyncio
    async def test_all_servers_of_one_controller_share_a_single_lock(self):
        # [FUNCTION DOC] Two servers built on one shared lock can both be offloaded while that one
        # lock is held.
        """One controller lock guards every model's server, so one acquire covers them all."""
        shared_lock = ContextLock("InferenceController")
        first = _make_server(shared_lock)
        second = _make_server(shared_lock)

        async with shared_lock:
            await first.offload()
            await second.offload()


class TestWaitExpectedNumCellsIsLockFree:
    # [CLASS DOC] Tests that waiting for the expected number of cells does not require or hold
    # the lock.
    @pytest.mark.asyncio
    async def test_it_can_be_awaited_without_the_lock(self):
        # [FUNCTION DOC] A server expecting zero cells returns from the wait without any lock held.
        """It runs during startup from create(), which deliberately holds no lock."""
        srv = _make_server(init_expected_num_cells=0)
        await srv.wait_init_expected_num_cells()

    @pytest.mark.asyncio
    async def test_cells_can_still_be_added_while_it_polls(self):
        # [FUNCTION DOC] A server expecting one cell starts the wait as a task and yields once.
        # Then, taking the server's lock, a cell is put directly into the cell mapping; the wait
        # task completes within five seconds, proving polling did not hold the lock.
        """Polling must not hold the lock, otherwise reconcile could never add the cells it waits for."""
        srv = _make_server(init_expected_num_cells=1)
        waiter = asyncio.create_task(srv.wait_init_expected_num_cells())
        await asyncio.sleep(0)

        async with srv.context_lock:
            srv.server_cells["inference-engine-0-0-0"] = SimpleNamespace(meta=SimpleNamespace(needs_offload=True))

        await asyncio.wait_for(waiter, timeout=5)
