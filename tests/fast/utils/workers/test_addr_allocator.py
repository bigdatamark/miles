# [MODULE DOC] Unit tests for the per-node port allocator. The allocator keeps one cursor per node
# ip (the next port to hand out), asks an engine actor for a block of free consecutive ports starting
# at the cursor, and advances the cursor past the returned block. Tests cover cursor start (base port
# 20000 for an unseen node), block reservation, skipping of occupied candidates, error handling (the
# cursor and lock stay intact after a failed probe), per-node independence, wrapping back to the base
# port when a block would pass the last port (65535), serialization of concurrent allocations, and
# non-blocking of the event loop. The engine is a mock from the ray rollout test fixtures whose free
# port probe is replaced where needed.
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from tests.fast.ray.rollout.conftest import fake_engine

from miles.utils.workers.addr_allocator import PortAllocator


def _make_probe_fail(engine: MagicMock) -> None:
    # [FUNCTION DOC] HELPER: Makes the mock engine's free port block probe fail: the remote call's
    # side effect becomes a coroutine function that raises a runtime error "free port probe failed".
    async def _fail(start_port: int = 15000, count: int = 1):
        # [FUNCTION DOC] HELPER: Ignores its arguments and raises the runtime error.
        raise RuntimeError("free port probe failed")

    engine._get_free_port_block.remote.side_effect = _fail


def _make_probe_slow(engine: MagicMock, *, delay_seconds: float) -> None:
    # [FUNCTION DOC] HELPER: Makes the mock engine's free port block probe slow but correct: the
    # remote call's side effect becomes a coroutine function that sleeps for the delay, then returns
    # the larger of the engine's own port cursor and the requested start port, and moves the engine
    # cursor past the returned block.
    async def _slow(start_port: int = 15000, count: int = 1):
        # [FUNCTION DOC] HELPER: Sleeps, picks the port as described by the enclosing helper, advances
        # the engine's cursor by count and returns the port.
        await asyncio.sleep(delay_seconds)
        port = max(engine._port_cursor, start_port)
        engine._port_cursor = port + count
        return port

    engine._get_free_port_block.remote.side_effect = _slow


class TestPortAllocator:
    # [CLASS DOC] Tests of the port allocator's cursor logic, using fake engines per node ip.
    def test_a_fresh_allocator_has_no_cursors(self):
        # [FUNCTION DOC] A new allocator's private per-node cursor mapping is empty.
        """A brand new allocator starts with no per-node cursors."""
        c = PortAllocator()
        assert c._next_port_of_ip == {}

    async def test_alloc_advances_the_cursor_of_its_node(self):
        # [FUNCTION DOC] Two allocations on one node return increasing ports and the node's cursor
        # ends one past the second port.
        """Two allocations on the same node must hand out non-overlapping ports."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        first = await cursors.alloc(engine, node_ip="10.0.0.1")
        second = await cursors.alloc(engine, node_ip="10.0.0.1")
        assert second > first
        assert cursors._next_port_of_ip["10.0.0.1"] == second + 1

    async def test_alloc_starts_from_the_base_port_on_an_unseen_node(self):
        # [FUNCTION DOC] The first allocation on an unseen node returns 20000.
        """A node with no cursor yet starts at the base port, away from ray's range."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        assert await cursors.alloc(engine, node_ip="10.0.0.1") == 20000

    async def test_alloc_consecutive_reserves_a_whole_block(self):
        # [FUNCTION DOC] Requesting five consecutive ports leaves the node cursor five past the
        # returned first port.
        """A consecutive=N allocation must move this node's cursor past the entire block."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        first = await cursors.alloc(engine, node_ip="10.0.0.1", consecutive=5)
        assert cursors._next_port_of_ip["10.0.0.1"] == first + 5

    async def test_a_skipped_candidate_range_advances_from_the_returned_block(self):
        # [FUNCTION DOC] An engine whose first free block starts at 25000 returns 25000 for a block of
        # four; the cursor ends at 25004 (counted from the returned port, not the candidate).
        """When the actor skips occupied candidates, the cursor must advance from the returned port plus the block size."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=25000)
        first = await cursors.alloc(engine, node_ip="10.0.0.1", consecutive=4)
        assert first == 25000
        assert cursors._next_port_of_ip["10.0.0.1"] == 25004

    async def test_a_failed_probe_does_not_advance_the_node_cursor(self):
        # [FUNCTION DOC] With a cursor preset to 20005 and a failing probe, the allocation raises the
        # probe's runtime error and the cursor mapping is unchanged.
        """A probe that fails on result retrieval propagates its error and leaves this node's cursor untouched."""
        cursors = PortAllocator()
        cursors._next_port_of_ip["10.0.0.1"] = 20005
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        _make_probe_fail(engine)
        with pytest.raises(RuntimeError, match="free port probe failed"):
            await cursors.alloc(engine, node_ip="10.0.0.1", consecutive=3)
        assert cursors._next_port_of_ip == {"10.0.0.1": 20005}

    async def test_a_failed_probe_does_not_create_a_cursor_for_an_unseen_node(self):
        # [FUNCTION DOC] A failing probe on an unseen node raises and leaves the cursor mapping empty.
        """A probe that fails on result retrieval leaves a never-seen node without any cursor."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        _make_probe_fail(engine)
        with pytest.raises(RuntimeError, match="free port probe failed"):
            await cursors.alloc(engine, node_ip="10.0.0.1", consecutive=3)
        assert cursors._next_port_of_ip == {}

    async def test_a_failed_probe_releases_the_lock_for_the_next_allocation(self):
        # [FUNCTION DOC] After a failed allocation, a second allocation with a healthy engine
        # completes within one second and returns 20000 (the lock was released).
        """A failed probe must not leave the allocator locked, so a later allocation still goes through."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        _make_probe_fail(engine)
        with pytest.raises(RuntimeError, match="free port probe failed"):
            await cursors.alloc(engine, node_ip="10.0.0.1")

        healthy = fake_engine(host="10.0.0.1", port_seed=0)
        assert await asyncio.wait_for(cursors.alloc(healthy, node_ip="10.0.0.1"), timeout=1.0) == 20000

    async def test_alloc_tracks_nodes_independently(self):
        # [FUNCTION DOC] Allocations on two different node ips create two separate cursor entries.
        """Each node ip owns its own cursor."""
        cursors = PortAllocator()
        engine_a = fake_engine(host="10.0.0.1", port_seed=0)
        engine_b = fake_engine(host="10.0.0.2", port_seed=0)
        await cursors.alloc(engine_a, node_ip="10.0.0.1")
        await cursors.alloc(engine_b, node_ip="10.0.0.2")
        assert set(cursors._next_port_of_ip.keys()) == {"10.0.0.1", "10.0.0.2"}

    async def test_a_cursor_past_the_last_port_restarts_at_the_base_port(self):
        # [FUNCTION DOC] With a cursor at 65535 and a block of four (which does not fit), the
        # allocation returns 20000.
        """Ports are never reclaimed, so a long fault-tolerance run walks the cursor off the end."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        cursors._next_port_of_ip["10.0.0.1"] = 65535

        assert await cursors.alloc(engine, node_ip="10.0.0.1", consecutive=4) == 20000

    async def test_a_cursor_that_still_fits_is_left_alone(self):
        # [FUNCTION DOC] With a cursor at 65530 and a block of four (which still fits), the allocation
        # returns 65530.
        """Resetting early would hand out ports that are still in use by live cells."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        cursors._next_port_of_ip["10.0.0.1"] = 65530

        assert await cursors.alloc(engine, node_ip="10.0.0.1", consecutive=4) == 65530

    async def test_concurrent_allocations_on_one_node_never_overlap(self):
        # [FUNCTION DOC] Five concurrent blocks of two on one node with a slow probe return exactly
        # 20000, 20002, 20004, 20006 and 20008 (sorted), and the cursor ends at 20010.
        """Allocations racing on the same node while a probe is in flight must still get disjoint blocks."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        _make_probe_slow(engine, delay_seconds=0.01)

        ports = await asyncio.gather(*[cursors.alloc(engine, node_ip="10.0.0.1", consecutive=2) for _ in range(5)])

        assert sorted(ports) == [20000, 20002, 20004, 20006, 20008]
        assert cursors._next_port_of_ip["10.0.0.1"] == 20010

    async def test_a_probe_in_flight_does_not_block_the_event_loop(self):
        # [FUNCTION DOC] While a 0.05 second probe is pending, a concurrent ticker coroutine (three
        # ticks, 0.01 seconds apart) must complete all its ticks.
        """The probe is awaited, so unrelated coroutines keep running while it is pending."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        _make_probe_slow(engine, delay_seconds=0.05)
        ticks: list[int] = []

        async def _tick() -> None:
            # [FUNCTION DOC] HELPER: Appends the numbers 0, 1 and 2 to the tick list, sleeping 0.01
            # seconds after each.
            for i in range(3):
                ticks.append(i)
                await asyncio.sleep(0.01)

        await asyncio.gather(cursors.alloc(engine, node_ip="10.0.0.1"), _tick())

        assert ticks == [0, 1, 2]
