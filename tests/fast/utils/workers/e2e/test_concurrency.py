# [MODULE DOC]
# End-to-end tests of the server's execution model for rpc calls, using the shared-server handle
# and per-call start/end events recorded by the test worker: synchronous methods of one
# concurrency group run one at a time on one dedicated thread in submission order; different
# declared groups run in parallel; async methods run on the event loop and are not blocked by
# blocked sync groups; and a blocked group never stalls health checks.
import asyncio
import time


def _span(events, tag):
    # [FUNCTION DOC] HELPER: From a list of worker events returns the pair of start time and end
    # time (monotonic seconds) of the call with the given tag. Raises a stop-iteration error if
    # either event is missing.
    start = next(e for e in events if e.tag == tag and e.phase == "start")
    end = next(e for e in events if e.tag == tag and e.phase == "end")
    return start.at, end.at


def _overlaps(first, second) -> bool:
    # [FUNCTION DOC] HELPER: True if two (start, end) time spans overlap strictly; spans that
    # merely touch do not overlap.
    return first[0] < second[1] and second[0] < first[1]


class TestSyncSerialization:
    # [CLASS DOC] Sync methods in the default group execute serially on one thread.
    async def test_same_group_calls_do_not_overlap(self, handle, tag):
        # [FUNCTION DOC] Starts two 0.4 s sync sleeps 0.1 s apart and asserts, from the worker's
        # events, that their execution spans do not overlap.
        """Two sync calls in one concurrency group run strictly one after another."""
        first = asyncio.create_task(handle.demo_sleep_sync(tag=f"{tag}a", seconds=0.4))
        await asyncio.sleep(0.1)
        second = asyncio.create_task(handle.demo_sleep_sync(tag=f"{tag}b", seconds=0.4))
        await asyncio.gather(first, second)

        events = await handle.report_events()
        assert not _overlaps(_span(events, f"{tag}a"), _span(events, f"{tag}b"))

    async def test_same_group_keeps_submission_order(self, handle, tag):
        # [FUNCTION DOC] Submits four sync sleeps (the first 0.3 s, the rest 0.05 s) spaced 0.05 s
        # apart so they reach the server in a known order; asserts the start events for the tag
        # prefix are in index order.
        """Queued sync calls execute in the order they were accepted."""
        # Concurrent sends are separate HTTP requests whose arrival order is not their send
        # order, so space the sends out while the first call holds the group: every later
        # call then reaches the server's queue in a known order.
        calls = []
        for i in range(4):
            calls.append(asyncio.create_task(handle.demo_sleep_sync(tag=f"{tag}{i}", seconds=0.3 if i == 0 else 0.05)))
            await asyncio.sleep(0.05)
        await asyncio.gather(*calls)

        events = await handle.report_events()
        starts = [e.tag for e in events if e.phase == "start" and e.tag.startswith(tag)]
        assert starts == sorted(starts, key=lambda name: int(name[len(tag) :]))

    async def test_same_group_uses_one_thread(self, handle, tag):
        # [FUNCTION DOC] Runs five instant sync calls concurrently; asserts they all report the
        # same thread name and that it starts with the "rpc-" prefix.
        """A concurrency group is backed by a single executor thread."""
        names = await asyncio.gather(*[handle.demo_instant_sync(tag=f"{tag}{i}") for i in range(5)])
        assert len(set(names)) == 1 and names[0].startswith("rpc-")

    async def test_blocked_group_drains_after_release(self, handle, tag):
        # [FUNCTION DOC] Blocks the default group with a gated call, queues an instant call and
        # asserts it is not done after 0.2 s; then releases the gate and asserts both finish.
        """A blocked group resumes its backlog once the blocker is released."""
        blocked = asyncio.create_task(handle.demo_block_sync(tag=f"{tag}block"))
        await asyncio.sleep(0.2)
        queued = asyncio.create_task(handle.demo_instant_sync(tag=f"{tag}queued"))

        await asyncio.sleep(0.2)
        assert not queued.done(), "a queued call must wait for its group"

        await handle.release(tag=f"{tag}block")
        await asyncio.gather(blocked, queued)


class TestDeclaredConcurrencyGroups:
    # [CLASS DOC] Behavior of methods declared with explicit concurrency groups ("left" and
    # "right") through the rpc decorator.
    async def test_declared_group_method_is_callable(self, handle, tag):
        # [FUNCTION DOC] Calls an instant method of the "left" group; asserts the executing
        # thread name starts with "rpc-left".
        """An @rpc-decorated method is exposed and runs on an executor thread."""
        assert (await handle.demo_instant_on_left(tag=tag)).startswith("rpc-left")

    async def test_same_declared_group_serializes(self, handle, tag):
        # [FUNCTION DOC] Two 0.4 s sleeps in the "left" group, started 0.1 s apart; asserts
        # their spans do not overlap.
        """Two calls declaring the same group run one after another, never overlapping."""
        first = asyncio.create_task(handle.demo_sleep_on_left(tag=f"{tag}a", seconds=0.4))
        await asyncio.sleep(0.1)
        second = asyncio.create_task(handle.demo_sleep_on_left(tag=f"{tag}b", seconds=0.4))
        await asyncio.gather(first, second)

        events = await handle.report_events()
        assert not _overlaps(_span(events, f"{tag}a"), _span(events, f"{tag}b"))

    async def test_distinct_declared_groups_run_in_parallel(self, handle, tag):
        # [FUNCTION DOC] Runs a 0.5 s sleep in the "left" group and one in the "right" group
        # concurrently; asserts total time under 0.9 s and thread names starting with
        # "rpc-left" and "rpc-right" respectively.
        """Two sync methods in different declared groups overlap, each on its own thread."""
        started = time.monotonic()
        threads = await asyncio.gather(
            handle.demo_sleep_on_left(tag=f"{tag}left", seconds=0.5),
            handle.demo_sleep_on_right(tag=f"{tag}right", seconds=0.5),
        )
        assert time.monotonic() - started < 0.9
        assert threads[0].startswith("rpc-left") and threads[1].startswith("rpc-right")

    async def test_declared_group_is_isolated_from_the_default_group(self, handle, tag):
        # [FUNCTION DOC] Blocks the "left" group, then asserts a default-group sum call still
        # returns 2 promptly; releases the gate and awaits the blocked call.
        """Blocking a declared group leaves the default group free."""
        blocked = asyncio.create_task(handle.demo_block_on_left(tag=tag))
        await asyncio.sleep(0.2)

        assert await handle.demo_sync(a=1, b=1) == 2

        await handle.release(tag=tag)
        await blocked

    async def test_default_rpc_decorator_keeps_async_method_on_loop(self, handle, tag):
        # [FUNCTION DOC] Calls the async method that has the decorator without a group; asserts
        # it runs on the main (event loop) thread.
        """The default rpc decorator keeps an async method on the event loop."""
        assert await handle.demo_async_on_left(tag=tag) == "MainThread"

    async def test_async_method_ignores_blocked_named_sync_group(self, handle, tag):
        # [FUNCTION DOC] Blocks the "left" group; asserts the async method still completes on
        # the main thread; then releases the gate and awaits the blocked call.
        """An async method remains free while a named sync group is blocked."""
        blocked = asyncio.create_task(handle.demo_block_on_left(tag=tag))
        await asyncio.sleep(0.2)

        assert await handle.demo_async_on_left(tag=f"{tag}free") == "MainThread"

        await handle.release(tag=tag)
        await blocked


class TestGroupIsolation:
    # [CLASS DOC] Sync work, async work and health checks do not block each other.
    async def test_sync_and_async_run_in_parallel(self, handle, tag):
        # [FUNCTION DOC] Runs a 0.5 s sync sleep and a 0.5 s async sleep concurrently; asserts
        # the total time is under 0.9 s.
        """A sync call on an executor and an async call on the loop overlap in time."""
        started = time.monotonic()
        await asyncio.gather(
            handle.demo_sleep_sync(tag=f"{tag}default", seconds=0.5),
            handle.demo_sleep_async(tag=f"{tag}async", seconds=0.5),
        )
        assert time.monotonic() - started < 0.9

    async def test_blocked_sync_group_leaves_the_event_loop_free(self, handle, tag, raw):
        # [FUNCTION DOC] Blocks the default sync group, then asserts the health endpoint answers
        # 200 in under 2 s; releases the gate and awaits the blocked call.
        """A blocked sync call must not stall the server's event loop."""
        blocked = asyncio.create_task(handle.demo_block_sync(tag=tag))
        await asyncio.sleep(0.2)

        started = time.monotonic()
        response = await raw.get("/v1/health")
        assert response.status_code == 200 and time.monotonic() - started < 2.0

        await handle.release(tag=tag)
        await blocked

    async def test_blocked_sync_group_leaves_async_methods_free(self, handle, tag):
        # [FUNCTION DOC] Blocks the default sync group, then asserts an instant async call
        # completes on the main thread; releases the gate and awaits the blocked call.
        """Async methods keep running while a sync group is blocked."""
        blocked = asyncio.create_task(handle.demo_block_sync(tag=tag))
        await asyncio.sleep(0.2)

        assert await handle.demo_instant_async(tag=f"{tag}free") == "MainThread"

        await handle.release(tag=tag)
        await blocked

    async def test_blocked_async_call_leaves_others_free(self, handle, tag):
        # [FUNCTION DOC] Suspends an async call on a gate, then asserts a sync sum call still
        # returns 2; releases the gate and awaits the suspended call.
        """A suspended async call does not stall other calls."""
        blocked = asyncio.create_task(handle.demo_block_async(tag=tag))
        await asyncio.sleep(0.2)

        assert await handle.demo_sync(a=1, b=1) == 2

        await handle.release(tag=tag)
        await blocked


class TestAsyncExecution:
    # [CLASS DOC] Which thread each kind of method runs on, and that async calls interleave.
    async def test_async_methods_run_on_the_loop_thread(self, handle, tag):
        # [FUNCTION DOC] Asserts an instant async call reports the main thread name.
        """Async methods run on the event loop rather than an executor thread."""
        assert await handle.demo_instant_async(tag=tag) == "MainThread"

    async def test_sync_methods_run_off_the_loop_thread(self, handle, tag):
        # [FUNCTION DOC] Asserts an instant sync call reports a thread name starting with "rpc-".
        """Sync methods run on a worker thread, never on the loop."""
        assert (await handle.demo_instant_sync(tag=tag)).startswith("rpc-")

    async def test_async_calls_interleave(self, handle, tag):
        # [FUNCTION DOC] Runs twenty 0.3 s async sleeps concurrently; asserts the total time is
        # under 3 s (serial execution would take 6 s).
        """Twenty concurrent async sleeps finish concurrently, not serially."""
        started = time.monotonic()
        await asyncio.gather(*[handle.demo_sleep_async(tag=f"{tag}{i}", seconds=0.3) for i in range(20)])
        assert time.monotonic() - started < 3.0
