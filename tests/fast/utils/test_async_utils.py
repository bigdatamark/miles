# [MODULE DOC] Unit tests for the async helpers module: eager task creation, the gather error
# helpers, the background event loop thread (submit, run, wait_futures), the singleton accessor of
# that loop, wait_cancelling_pending_on_first_completion, gather_and_raise_first, the Disposer
# async context manager, with_disposer, maybe_await and the note-or-log fallback for secondary
# failures. All tests are CPU only; they use real asyncio loops and threads with short sleeps,
# monkeypatch the module-level loop singleton, and use caplog to check logging.
"""Tests for eager_create_task, AsyncioGatherUtils, and the get_async_loop singleton."""

import asyncio
import concurrent.futures
import logging
import threading
import time
from functools import partial

import pytest

from miles.utils import async_utils
from miles.utils.async_utils import (
    AsyncioGatherUtils,
    AsyncLoopThread,
    Disposer,
    eager_create_task,
    maybe_await,
    wait_cancelling_pending_on_first_completion,
    with_disposer,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("create_mode", ["eager", "plain"])
class TestCreateTaskComparison:
    # [CLASS DOC] Every test runs twice, once with the eager task creator and once with the plain
    # asyncio task creator, to contrast their behavior. The parametrized create_mode argument is
    # either "eager" or "plain".
    async def test_returns_asyncio_task(self, create_mode):
        # [FUNCTION DOC] Creates a coroutine returning 42 in the chosen mode (the eager creator is
        # awaited to obtain the task). Asserts the result is a real asyncio task and that awaiting
        # it yields 42.
        async def coro():
            # [FUNCTION DOC] HELPER: Coroutine that returns the integer 42.
            return 42

        if create_mode == "eager":
            task = await eager_create_task(coro())
        else:
            task = asyncio.create_task(coro())

        assert isinstance(task, asyncio.Task)
        assert await task == 42

    async def test_started_before_next_line(self, create_mode):
        # [FUNCTION DOC] Creates a task whose coroutine sets a started flag and then sleeps. In eager
        # mode asserts the flag is already set right after creation; in plain mode asserts it is
        # not yet set. Then cancels the task and asserts awaiting it raises a cancelled error.
        """eager starts immediately; plain does not."""
        started = False

        async def coro():
            # [FUNCTION DOC] HELPER: Sets the enclosing started flag, then sleeps 10 seconds
            # (expected to be cancelled by the test).
            nonlocal started
            started = True
            await asyncio.sleep(10)

        if create_mode == "eager":
            task = await eager_create_task(coro())
            assert started, "eager_create_task should have started the task"
        else:
            task = asyncio.create_task(coro())
            assert not started, "plain create_task should NOT have started the task yet"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_dispatch_order(self, create_mode):
        # [FUNCTION DOC] Creates a critic task and then directly awaits an actor coroutine, each
        # appending its name to a shared order list before sleeping 0.1 s. Asserts the order is
        # critic then actor in eager mode, and actor then critic in plain mode (the plain task only
        # starts when the caller yields).
        """eager preserves critic-before-actor dispatch order; plain reverses it."""
        order: list[str] = []

        async def critic():
            # [FUNCTION DOC] HELPER: Records "critic" in the order list, then sleeps 0.1 s.
            order.append("critic")
            await asyncio.sleep(0.1)

        async def actor():
            # [FUNCTION DOC] HELPER: Records "actor" in the order list, then sleeps 0.1 s.
            order.append("actor")
            await asyncio.sleep(0.1)

        if create_mode == "eager":
            critic_task = await eager_create_task(critic())
        else:
            critic_task = asyncio.create_task(critic())

        await actor()
        await critic_task

        if create_mode == "eager":
            assert order == ["critic", "actor"]
        else:
            assert order == ["actor", "critic"]

    async def test_exception_propagates(self, create_mode):
        # [FUNCTION DOC] Creates a task whose coroutine raises a value error with message "boom" and
        # asserts awaiting the task re-raises that error, in both modes.
        async def failing():
            # [FUNCTION DOC] HELPER: Always raises a value error with message "boom".
            raise ValueError("boom")

        if create_mode == "eager":
            task = await eager_create_task(failing())
        else:
            task = asyncio.create_task(failing())

        with pytest.raises(ValueError, match="boom"):
            await task

    async def test_result_available(self, create_mode):
        # [FUNCTION DOC] Creates a task whose coroutine returns a one-entry dict and asserts awaiting
        # the task returns an equal dict, in both modes.
        async def compute():
            # [FUNCTION DOC] HELPER: Returns a dict mapping "key" to "value".
            return {"key": "value"}

        if create_mode == "eager":
            task = await eager_create_task(compute())
        else:
            task = asyncio.create_task(compute())

        assert await task == {"key": "value"}


# ---- get_async_loop ----


class TestGetAsyncLoopSingleton:
    # [CLASS DOC] Race test for the lazy singleton accessor of the background loop thread.
    def test_concurrent_first_callers_share_one_loop(self, monkeypatch):
        # [FUNCTION DOC] Replaces the loop thread constructor with a slow one (records each
        # constructed instance, sleeps 0.2 s, then runs the real constructor) and clears the
        # module-level singleton. Two threads released together by a barrier each call the accessor
        # and store the result. Asserts exactly one instance was constructed and both threads got the
        # same loop object. Side effects: monkeypatches module attributes (restored afterwards); the
        # constructed loop thread is left running.
        """Two threads racing the first call must construct exactly one loop; a
        primitive bound to a losing loop later raises "bound to a different event loop"."""
        constructed = []
        real_init = async_utils.AsyncLoopThread.__init__

        def slow_init(self):
            # [FUNCTION DOC] HELPER: Replacement constructor: records self, sleeps 0.2 s to keep
            # the check-then-act window open, then delegates to the real constructor.
            constructed.append(self)
            time.sleep(0.2)  # hold the check-then-act window open
            real_init(self)

        monkeypatch.setattr(async_utils.AsyncLoopThread, "__init__", slow_init)
        monkeypatch.setattr(async_utils, "async_loop", None)

        barrier = threading.Barrier(2)
        results = []

        def call():
            # [FUNCTION DOC] HELPER: Thread body: waits on the shared two-party barrier, then
            # appends the accessor's result to the results list.
            barrier.wait()
            results.append(async_utils.get_async_loop())

        threads = [threading.Thread(target=call) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(constructed) == 1
        assert results[0] is results[1]


# ---- AsyncioGatherUtils ----

_ERR1 = RuntimeError("boom")
_ERR2 = ValueError("kaboom")


class TestAsyncioGatherUtilsHasError:
    # [CLASS DOC] Tests of the has_error predicate over lists of gathered results (exceptions are
    # returned as values, as with a gather that returns exceptions).
    def test_no_errors(self):
        # [FUNCTION DOC] A list of plain values (number, string, none) has no error: returns false.
        assert AsyncioGatherUtils.has_error([1, "ok", None]) is False

    def test_empty_list(self):
        # [FUNCTION DOC] An empty list has no error: returns false.
        assert AsyncioGatherUtils.has_error([]) is False

    def test_single_exception(self):
        # [FUNCTION DOC] A list holding one exception instance reports an error.
        assert AsyncioGatherUtils.has_error([_ERR1]) is True

    def test_exception_among_successes(self):
        # [FUNCTION DOC] An exception mixed among successful values is detected.
        assert AsyncioGatherUtils.has_error(["ok", _ERR1, 42]) is True

    def test_multiple_exceptions(self):
        # [FUNCTION DOC] Several exceptions in the list are reported as an error.
        assert AsyncioGatherUtils.has_error([_ERR1, _ERR2]) is True

    def test_base_exception_detected(self):
        # [FUNCTION DOC] A keyboard interrupt (a base exception, not a regular exception) is also
        # detected as an error.
        assert AsyncioGatherUtils.has_error([KeyboardInterrupt()]) is True

    def test_exception_subclass_detected(self):
        # [FUNCTION DOC] An OS error (a subclass of exception) is detected as an error.
        assert AsyncioGatherUtils.has_error([OSError("disk")]) is True


class TestAsyncioGatherUtilsLogError:
    # [CLASS DOC] Tests of log_error, which logs one record per exception in a gathered result list,
    # carrying the index, a debug name or custom description, and the exception info. Uses caplog.
    def test_logs_nothing_on_all_success(self, caplog):
        # [FUNCTION DOC] With only successful values, no log record mentions the debug name.
        with caplog.at_level(logging.WARNING):
            AsyncioGatherUtils.log_error(["ok", 42], debug_name="test_op")

        assert not any("test_op" in r.message for r in caplog.records)

    def test_logs_nothing_on_empty(self, caplog):
        # [FUNCTION DOC] An empty result list produces no log records at all.
        with caplog.at_level(logging.WARNING):
            AsyncioGatherUtils.log_error([], debug_name="test_op")

        assert len(caplog.records) == 0

    def test_logs_single_error_with_index(self, caplog):
        # [FUNCTION DOC] One exception at position 1 yields exactly one record containing the debug
        # name, the text "index=1", and non-empty exception info.
        with caplog.at_level(logging.WARNING):
            AsyncioGatherUtils.log_error(["ok", _ERR1], debug_name="test_op")

        error_records = [r for r in caplog.records if "test_op" in r.message]
        assert len(error_records) == 1
        assert "index=1" in error_records[0].message
        assert error_records[0].exc_info is not None

    def test_logs_multiple_errors_with_correct_indices(self, caplog):
        # [FUNCTION DOC] Exceptions at positions 0 and 2 (with a success between) yield two records,
        # in order, containing "index=0" and "index=2" respectively.
        with caplog.at_level(logging.WARNING):
            AsyncioGatherUtils.log_error([_ERR1, "ok", _ERR2], debug_name="my_gather")

        error_records = [r for r in caplog.records if "my_gather" in r.message]
        assert len(error_records) == 2
        assert "index=0" in error_records[0].message
        assert "index=2" in error_records[1].message

    def test_a_caller_can_name_each_failure_itself(self, caplog):
        # [FUNCTION DOC] Passing a describe_failure callback (index to message) makes the logged
        # message exactly the callback's text; asserts the only record's message is "cell-0 refused".
        """A cell of a server means more to whoever reads the log than the index it happened to gather at."""
        with caplog.at_level(logging.WARNING):
            AsyncioGatherUtils.log_error([_ERR1, "ok"], describe_failure=lambda index: f"cell-{index} refused")

        assert [r.message for r in caplog.records] == ["cell-0 refused"]

    def test_a_caller_can_choose_the_level_it_is_logged_at(self, caplog):
        # [FUNCTION DOC] Passing a custom log function (a logger's error method) makes the record be
        # emitted at error level; asserts the only record has the error level number.
        """A failure that is about to be raised is an error, while a gather that carries on is a warning."""
        with caplog.at_level(logging.ERROR):
            AsyncioGatherUtils.log_error([_ERR1], debug_name="test_op", log=logging.getLogger("x").error)

        assert [r.levelno for r in caplog.records] == [logging.ERROR]

    def test_logs_include_debug_name(self, caplog):
        # [FUNCTION DOC] The debug name string appears in at least one logged message.
        with caplog.at_level(logging.WARNING):
            AsyncioGatherUtils.log_error([_ERR1], debug_name="refresh_cells#coop")

        assert any("refresh_cells#coop" in r.message for r in caplog.records)

    def test_logs_exc_info_is_the_exception(self, caplog):
        # [FUNCTION DOC] The exception info attached to the record holds the very same exception
        # object that was in the result list (checked by identity on the info's second element).
        err = TypeError("specific")
        with caplog.at_level(logging.WARNING):
            AsyncioGatherUtils.log_error([err], debug_name="op")

        record = [r for r in caplog.records if "op" in r.message][0]
        assert record.exc_info[1] is err


# ---- AsyncLoopThread.submit / wait_futures ----


@pytest.fixture
def loop_thread():
    # [FUNCTION DOC] HELPER: Pytest fixture that builds a fresh background loop thread, yields it,
    # and on teardown asks its loop to stop (thread-safe call). Side effect: starts a thread.
    thread = AsyncLoopThread()
    yield thread
    thread.loop.call_soon_threadsafe(thread.loop.stop)


class TestSubmit:
    # [CLASS DOC] Tests of the loop thread's submit (non-blocking dispatch returning a concurrent
    # future) and run (blocking variant). All use the loop_thread fixture.
    def test_submit_returns_before_the_coroutine_finishes(self, loop_thread):
        # [FUNCTION DOC] Submits a coroutine that blocks on a thread event (via the default
        # executor), asserts the returned future is not done, then sets the event and asserts the
        # future's result (10 s timeout) is "done".
        """Non-blocking dispatch is the whole point: the caller must get control back."""
        release = threading.Event()

        async def blocked():
            # [FUNCTION DOC] HELPER: Waits in an executor for the release event, returns "done".
            await asyncio.get_running_loop().run_in_executor(None, release.wait)
            return "done"

        future = loop_thread.submit(blocked())
        assert not future.done()

        release.set()
        assert future.result(timeout=10) == "done"

    def test_run_blocks_until_the_result_is_available(self, loop_thread):
        # [FUNCTION DOC] Runs a coroutine (yields once, returns 7) through the blocking run method
        # and asserts the call returns 7.
        """``run`` is the blocking sibling used by the synchronous ray actor shells."""

        async def compute():
            # [FUNCTION DOC] HELPER: Yields control once and returns the integer 7.
            await asyncio.sleep(0)
            return 7

        assert loop_thread.run(compute()) == 7

    def test_exceptions_surface_at_the_collection_point(self, loop_thread):
        # [FUNCTION DOC] Submits a coroutine raising a value error "boom"; asserts reading the
        # future's result raises that error.
        """A failed request must not disappear into the background loop."""

        async def failing():
            # [FUNCTION DOC] HELPER: Always raises a value error "boom".
            raise ValueError("boom")

        future = loop_thread.submit(failing())

        with pytest.raises(ValueError, match="boom"):
            future.result(timeout=10)

    def test_run_propagates_the_coroutine_exception(self, loop_thread):
        # [FUNCTION DOC] The blocking run method re-raises the coroutine's value error "boom" in the
        # calling thread.
        """``run`` blocks on the future, so a failure must land on the synchronous caller."""

        async def failing():
            # [FUNCTION DOC] HELPER: Always raises a value error "boom".
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            loop_thread.run(failing())

    def test_cancelling_submitted_future_cancels_the_coroutine(self, loop_thread):
        # [FUNCTION DOC] Submits a coroutine that signals a started event then sleeps 30 s, setting a
        # cancelled event if it sees cancellation. After the started event (10 s timeout), cancels
        # the future and asserts the cancel call succeeded, the coroutine observed cancellation
        # (10 s timeout) and the future reports cancelled.
        """Dropping a fired request must stop the background work, not orphan it on the loop."""
        started = threading.Event()
        cancelled = threading.Event()

        async def blocked():
            # [FUNCTION DOC] HELPER: Signals started, sleeps 30 s; on cancellation signals
            # cancelled and re-raises.
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        future = loop_thread.submit(blocked())
        assert started.wait(timeout=10)

        assert future.cancel()
        assert cancelled.wait(timeout=10)
        assert future.cancelled()

    def test_submitting_from_inside_the_loop_thread_is_refused(self, loop_thread):
        # [FUNCTION DOC] Runs a coroutine on the loop thread that tries to submit another coroutine
        # from inside the loop; asserts the submit raises an assertion error whose message mentions
        # "deadlock".
        """Blocking on that future would freeze the very loop that must resolve it."""

        async def submit_from_the_loop():
            # [FUNCTION DOC] HELPER: Runs on the loop thread; expects the nested submit of a sleep
            # coroutine to raise an assertion error matching "deadlock".
            with pytest.raises(AssertionError, match="deadlock"):
                loop_thread.submit(asyncio.sleep(0))

        loop_thread.run(submit_from_the_loop())

    def test_a_fan_out_larger_than_any_thread_pool_still_all_lands(self, loop_thread):
        # [FUNCTION DOC] Submits 200 coroutines that each signal arrival on a semaphore and then
        # block in a 256-worker executor until a release event. Asserts all 200 arrivals happen
        # (30 s timeout each) before release, i.e. dispatch is not limited by a small thread pool,
        # then releases and asserts all 200 results equal "ok".
        """This is why the dispatch is a loop and not a thread pool."""
        arrived = threading.Semaphore(0)
        release = threading.Event()

        async def blocked():
            # [FUNCTION DOC] HELPER: Signals arrival, waits for release in the large executor,
            # returns "ok".
            arrived.release()
            await asyncio.get_running_loop().run_in_executor(_UNBOUNDED_POOL, release.wait)
            return "ok"

        futures = [loop_thread.submit(blocked()) for _ in range(200)]
        for _ in range(200):
            assert arrived.acquire(timeout=30)

        release.set()
        assert [f.result(timeout=30) for f in futures] == ["ok"] * 200


_UNBOUNDED_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=256)


class TestFireThenRendezvous:
    # [CLASS DOC] Tests that all requests are fired before the caller blocks on a collective
    # rendezvous (modelled with a thread barrier), and that blocking early deadlocks.
    """The blocking endpoints need every engine asked before the caller joins."""

    def test_every_request_is_on_the_wire_before_the_caller_joins_the_collective(self, loop_thread):
        # [FUNCTION DOC] Submits 32 fake engine calls that each wait on a 33-party barrier (the extra
        # party is the test thread), then the test thread waits on the barrier (30 s timeout) and
        # asserts the gathered results are the engine indices 0..31 in order.
        num_engines = 32
        barrier = threading.Barrier(num_engines + 1)

        async def engine_call(engine_index: int):
            # [FUNCTION DOC] HELPER: Fake engine request: waits on the barrier in the large
            # executor and returns its index.
            await asyncio.get_running_loop().run_in_executor(_UNBOUNDED_POOL, barrier.wait)
            return engine_index

        futures = [loop_thread.submit(engine_call(i)) for i in range(num_engines)]

        barrier.wait(timeout=30)

        assert async_utils.wait_futures(futures) == list(range(num_engines))

    def test_collecting_before_firing_the_rest_deadlocks(self, loop_thread):
        # [FUNCTION DOC] Submits only one fake engine call that waits on a two-party barrier with a
        # 1 s timeout. Asserts collecting its result with a 0.5 s timeout raises a futures timeout
        # error, and that a later collection (10 s) raises a broken-barrier error because the second
        # party never arrived.
        """A caller that blocks on the first response never sends the second."""
        barrier = threading.Barrier(2)

        async def engine_call(engine_index: int):
            # [FUNCTION DOC] HELPER: Waits on the barrier (1 s timeout) in the large executor and
            # returns its index.
            await asyncio.get_running_loop().run_in_executor(_UNBOUNDED_POOL, barrier.wait, 1.0)
            return engine_index

        first = loop_thread.submit(engine_call(0))

        with pytest.raises(concurrent.futures.TimeoutError):
            first.result(timeout=0.5)

        with pytest.raises(threading.BrokenBarrierError):
            first.result(timeout=10)


async def _failing(message: str):
    # [FUNCTION DOC] HELPER: Coroutine that always raises a value error with the given message.
    raise ValueError(message)


class TestWaitFutures:
    # [CLASS DOC] Tests of wait_futures: ordered results, draining every future before raising the
    # first error in submission order, logging every failure, interrupt handling, and the
    # module-level submit facade.
    def test_it_preserves_the_submission_order(self, loop_thread):
        # [FUNCTION DOC] Submits a slow task returning 1 (0.05 s) and a fast one returning 2; asserts
        # wait_futures returns [1, 2], i.e. positional order, not completion order.
        """Results are positional: caller i's result must stay at index i."""

        async def slow_first(delay: float, value: int):
            # [FUNCTION DOC] HELPER: Sleeps for the delay and returns the value.
            await asyncio.sleep(delay)
            return value

        futures = [loop_thread.submit(slow_first(0.05, 1)), loop_thread.submit(slow_first(0.0, 2))]

        assert async_utils.wait_futures(futures) == [1, 2]

    def test_it_drains_every_future_before_raising(self, loop_thread):
        # [FUNCTION DOC] Submits an immediately failing coroutine and a slow (0.1 s) succeeding one
        # that records its index. Asserts wait_futures raises the value error "boom" and that the slow
        # one had finished (recorded index 1) by then.
        """Bailing early leaves requests running against engines the caller considers done."""
        finished: list[int] = []

        async def failing():
            # [FUNCTION DOC] HELPER: Always raises a value error "boom".
            raise ValueError("boom")

        async def slow_success(index: int):
            # [FUNCTION DOC] HELPER: Sleeps 0.1 s, records the index in the finished list, returns it.
            await asyncio.sleep(0.1)
            finished.append(index)
            return index

        futures = [loop_thread.submit(failing()), loop_thread.submit(slow_success(1))]

        with pytest.raises(ValueError, match="boom"):
            async_utils.wait_futures(futures)

        assert finished == [1]

    def test_it_raises_the_first_error_in_submission_order(self, loop_thread):
        # [FUNCTION DOC] Submits two failing coroutines: "second" fails after 0.1 s but is submitted
        # first, "first" fails immediately but is submitted second. Asserts the raised error is the
        # one from the first submitted future (message "second"), not the earliest in time.
        """A stable error keeps failures reproducible when several engines fail."""

        async def failing(message: str, delay: float):
            # [FUNCTION DOC] HELPER: Sleeps for the delay then raises a value error with the message.
            await asyncio.sleep(delay)
            raise ValueError(message)

        futures = [loop_thread.submit(failing("second", 0.1)), loop_thread.submit(failing("first", 0.0))]

        with pytest.raises(ValueError, match="second"):
            async_utils.wait_futures(futures)

    def test_an_empty_fan_out_is_not_an_error(self):
        # [FUNCTION DOC] wait_futures on an empty list returns an empty list without raising.
        """A group with no live engines must not blow up the weight update."""
        assert async_utils.wait_futures([]) == []

    def test_every_failure_is_logged_not_just_the_one_that_is_raised(self, caplog):
        # [FUNCTION DOC] Submits two failing coroutines through the module-level submit helper
        # (shared background loop). Asserts the first error is raised and that exactly two warning
        # messages mentioning wait_futures were logged, "wait_futures index=0 failed" and
        # "wait_futures index=1 failed", in that order.
        """Only the first error propagates, so the others are only ever seen in the log."""
        futures = [async_utils.submit(_failing("first")), async_utils.submit(_failing("second"))]

        with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="first"):
            async_utils.wait_futures(futures)

        assert [record.message for record in caplog.records if "wait_futures" in record.message] == [
            "wait_futures index=0 failed",
            "wait_futures index=1 failed",
        ]

    def test_each_wait_future_failure_logs_its_exception_info(self, caplog):
        # [FUNCTION DOC] Same setup as the previous logging test; asserts the exception info attached
        # to the two wait_futures records carries the exceptions with texts "first" and "second".
        """The attached exception is the only detail available for failures that never propagate."""
        futures = [async_utils.submit(_failing("first")), async_utils.submit(_failing("second"))]

        with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="first"):
            async_utils.wait_futures(futures)

        records = [record for record in caplog.records if "wait_futures" in record.message]
        assert [str(record.exc_info[1]) for record in records] == ["first", "second"]

    def test_an_interrupt_is_not_held_back_by_the_remaining_futures(self):
        # [FUNCTION DOC] Passes two fake futures to wait_futures: the first raises a keyboard
        # interrupt when its result is read, the second records a touch and raises an assertion error
        # if its result is ever read. Asserts the keyboard interrupt propagates and the second
        # future was never touched.
        """One Ctrl-C must kill a hung weight update, not queue behind a silent engine."""
        touched: list[str] = []

        class _Interrupted(concurrent.futures.Future):
            # [CLASS DOC] HELPER: Fake future whose result method always raises a keyboard interrupt.
            def result(self, timeout=None):
                # [FUNCTION DOC] HELPER: Raises a keyboard interrupt, ignoring the timeout.
                raise KeyboardInterrupt

        class _NeverFinishes(concurrent.futures.Future):
            # [CLASS DOC] HELPER: Fake future that records a touch when its result is read and then
            # fails with an assertion error, proving the caller should not have reached it.
            def result(self, timeout=None):
                # [FUNCTION DOC] HELPER: Appends "second" to the touched list and raises an
                # assertion error.
                touched.append("second")
                raise AssertionError("the interrupt should have propagated before this")

        with pytest.raises(KeyboardInterrupt):
            async_utils.wait_futures([_Interrupted(), _NeverFinishes()])

        assert touched == []

    def test_a_module_level_submit_reaches_the_shared_background_loop(self):
        # [FUNCTION DOC] Submits 50 coroutines through the module-level submit helper, each returning
        # the id of the thread it ran on. Asserts all ids are the same single thread and that it is
        # not the calling thread.
        """Production fires through the module-level facade, not through a hand-built loop thread."""

        async def record():
            # [FUNCTION DOC] HELPER: Returns the identifier of the thread it runs on.
            return threading.get_ident()

        idents = async_utils.wait_futures([async_utils.submit(record()) for _ in range(50)])

        assert len(set(idents)) == 1, "every request must run on the one background loop"
        assert threading.get_ident() not in idents, "the caller thread must stay free to enter the collective"


@pytest.mark.asyncio
class TestWaitCancellingPendingOnFirstCompletion:
    # [CLASS DOC] Async tests of wait_cancelling_pending_on_first_completion: when the first task
    # finishes (or fails) the others are cancelled and awaited; failures propagate, ordering of
    # failures is deterministic, and secondary failures are attached as notes or logged.
    async def test_callback_runs_before_pending_tasks_are_cancelled(self) -> None:
        # [FUNCTION DOC] Starts a leader task waiting on an event and a follower task waiting forever
        # (recording "follower cancelled" if cancelled). After the follower started, releases the
        # leader and waits with an on-first-completion callback that asserts the follower has not
        # been marked as cancelling yet and records "callback". Asserts the event order is the
        # callback first, then the follower cancellation.
        """The first-completion callback runs once before a pending follower observes cancellation."""
        follower_started = asyncio.Event()
        leader_release = asyncio.Event()
        events: list[str] = []

        async def leader() -> None:
            # [FUNCTION DOC] HELPER: Waits until the leader release event is set.
            await leader_release.wait()

        async def follower() -> None:
            # [FUNCTION DOC] HELPER: Signals started, then waits forever; on cancellation records
            # "follower cancelled" and re-raises.
            follower_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("follower cancelled")
                raise

        leader_task: asyncio.Task[None] = asyncio.create_task(leader())
        follower_task: asyncio.Task[None] = asyncio.create_task(follower())
        await follower_started.wait()

        def on_first_completion() -> None:
            # [FUNCTION DOC] HELPER: Callback passed to the waiter; asserts the follower task has no
            # pending cancellation requests yet and appends "callback" to the events.
            assert follower_task.cancelling() == 0
            events.append("callback")

        leader_release.set()
        await wait_cancelling_pending_on_first_completion(
            [leader_task, follower_task], on_first_completion=on_first_completion
        )

        assert events == ["callback", "follower cancelled"]

    async def test_the_first_task_to_return_cancels_the_rest(self):
        # [FUNCTION DOC] A leader that returns after 0.01 s and a follower sleeping 30 s (flagging
        # cancellation) are waited on together; asserts the follower observed cancellation.
        """The leader owns the length of a multi policy run, and its followers loop until it is over."""
        cancelled = False

        async def leader():
            # [FUNCTION DOC] HELPER: Sleeps 0.01 s and returns.
            await asyncio.sleep(0.01)

        async def follower():
            # [FUNCTION DOC] HELPER: Sleeps 30 s; on cancellation sets the enclosing flag and
            # re-raises.
            nonlocal cancelled
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled = True
                raise

        await wait_cancelling_pending_on_first_completion(
            [asyncio.create_task(leader()), asyncio.create_task(follower())]
        )

        assert cancelled

    async def test_the_failure_reaches_the_caller(self):
        # [FUNCTION DOC] One task fails immediately with "boom" while another sleeps 30 s; asserts
        # the waiter raises the value error.
        """A member that raises must fail the whole fan-out instead of being swallowed."""

        async def failing():
            # [FUNCTION DOC] HELPER: Always raises a value error "boom".
            raise ValueError("boom")

        async def slow():
            # [FUNCTION DOC] HELPER: Sleeps 30 s (expected to be cancelled).
            await asyncio.sleep(30)

        with pytest.raises(ValueError, match="boom"):
            await wait_cancelling_pending_on_first_completion(
                [asyncio.create_task(failing()), asyncio.create_task(slow())]
            )

    async def test_a_failure_cancels_the_members_that_are_still_running(self):
        # [FUNCTION DOC] One task fails after 0.01 s while another sleeps 30 s and sets an event on
        # cancellation. Asserts the waiter raises the value error "boom" and the event is set.
        """A survivor left running would keep working for a run that is already dead."""
        cancelled = asyncio.Event()

        async def failing():
            # [FUNCTION DOC] HELPER: Sleeps 0.01 s then raises a value error "boom".
            await asyncio.sleep(0.01)
            raise ValueError("boom")

        async def slow():
            # [FUNCTION DOC] HELPER: Sleeps 30 s; on cancellation sets the event and re-raises.
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with pytest.raises(ValueError, match="boom"):
            await wait_cancelling_pending_on_first_completion(
                [asyncio.create_task(failing()), asyncio.create_task(slow())]
            )

        assert cancelled.is_set()

    async def test_two_members_failing_at_once_report_the_first_one_and_log_the_rest(self, caplog):
        # [FUNCTION DOC] Two tasks fail immediately ("first" and "second"). Asserts the waiter raises
        # the one listed first ("first") and that exactly two error records with the message "task
        # failed" were logged (one per failed task).
        """Which failure surfaced used to depend on set iteration order, and the other root cause vanished."""

        async def failing(message: str):
            # [FUNCTION DOC] HELPER: Raises a value error with the given message.
            raise ValueError(message)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="first"):
                await wait_cancelling_pending_on_first_completion(
                    [asyncio.create_task(failing("first")), asyncio.create_task(failing("second"))]
                )

        assert len([record for record in caplog.records if record.getMessage() == "task failed"]) == 2

    async def test_a_member_that_fails_while_being_cancelled_is_not_lost(self):
        # [FUNCTION DOC] One task fails with "boom" after 0.01 s; the other, when cancelled, raises a
        # runtime error "cleanup exploded". Asserts the waiter still raises the original value error
        # (the cleanup failure must not replace it or crash the waiter).
        """Cleanup that raises is the second root cause of a run that is already failing."""

        async def failing():
            # [FUNCTION DOC] HELPER: Sleeps 0.01 s then raises a value error "boom".
            await asyncio.sleep(0.01)
            raise ValueError("boom")

        async def slow():
            # [FUNCTION DOC] HELPER: Sleeps 30 s; when cancelled raises a runtime error "cleanup
            # exploded" instead of re-raising the cancellation.
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise RuntimeError("cleanup exploded") from None

        with pytest.raises(ValueError, match="boom"):
            await wait_cancelling_pending_on_first_completion(
                [asyncio.create_task(failing()), asyncio.create_task(slow())]
            )

    async def test_primary_failure_precedes_a_cleanup_failure_from_an_earlier_task(self):
        # [FUNCTION DOC] The cleanup-failing task is listed first, the primary failing task second.
        # Asserts the raised error is the primary value error "training failed" and that its notes
        # include the text "RuntimeError: cleanup exploded" from the cleanup failure.
        """Cancellation cleanup is attached without masking the failure that ended the run."""

        async def cleanup_failure():
            # [FUNCTION DOC] HELPER: Sleeps 30 s; when cancelled raises a runtime error "cleanup
            # exploded".
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise RuntimeError("cleanup exploded") from None

        async def primary_failure():
            # [FUNCTION DOC] HELPER: Sleeps 0.01 s then raises a value error "training failed".
            await asyncio.sleep(0.01)
            raise ValueError("training failed")

        with pytest.raises(ValueError, match="training failed") as exc_info:
            await wait_cancelling_pending_on_first_completion(
                [asyncio.create_task(cleanup_failure()), asyncio.create_task(primary_failure())]
            )

        assert any("RuntimeError: cleanup exploded" in note for note in exc_info.value.__notes__)

    async def test_the_cancelled_members_are_awaited_before_the_error_is_raised(self):
        # [FUNCTION DOC] One task fails at once; the other has an async cleanup (one yield, then sets
        # a flag) in its finally block. Asserts the waiter raises "boom" and the flag is already set,
        # proving cancelled tasks were awaited to completion before raising.
        """Raising before the cleanup lands would let a half-cancelled task outlive the caller."""
        cleaned_up = False

        async def failing():
            # [FUNCTION DOC] HELPER: Always raises a value error "boom".
            raise ValueError("boom")

        async def slow():
            # [FUNCTION DOC] HELPER: Sleeps 30 s; in its finally clause yields once and then sets
            # the enclosing cleaned-up flag.
            nonlocal cleaned_up
            try:
                await asyncio.sleep(30)
            finally:
                await asyncio.sleep(0)
                cleaned_up = True

        with pytest.raises(ValueError, match="boom"):
            await wait_cancelling_pending_on_first_completion(
                [asyncio.create_task(failing()), asyncio.create_task(slow())]
            )

        assert cleaned_up


class TestGatherAndRaiseFirst:
    # [CLASS DOC] Tests of gather_and_raise_first: gathers awaitables, optionally logs every failure
    # via a describe-failure callback, then raises the first failure. Not marked for asyncio
    # explicitly (relies on the project's async test mode).
    async def test_it_answers_every_result_when_nothing_fails(self):
        # [FUNCTION DOC] Gathers two awaitables returning 1 and 2; asserts the result list is [1, 2].
        """Callers use the results, so the happy path has to hand back what a plain gather would."""

        async def answer(value: int) -> int:
            # [FUNCTION DOC] HELPER: Returns the given value.
            return value

        assert await async_utils.gather_and_raise_first([answer(1), answer(2)]) == [1, 2]

    async def test_it_logs_a_failure_and_raises_it(self, caplog):
        # [FUNCTION DOC] Gathers one failing awaitable with a describe-failure callback; asserts the
        # value error is raised and the log text contains "awaitable 0 failed".
        """A caller that describes its failures wants them named, and still wants the error to reach it."""
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="only one"):
                await async_utils.gather_and_raise_first(
                    [_failing("only one")], describe_failure=lambda index: f"awaitable {index} failed"
                )

        assert "awaitable 0 failed" in caplog.text

    async def test_it_logs_every_failure_before_raising_the_first(self, caplog):
        # [FUNCTION DOC] Gathers failing, succeeding, failing awaitables with a describe-failure
        # callback. Asserts the first error ("first") is raised, the log names failures at indices 0
        # and 2, and does not mention index 1 (the success).
        """The point of gathering is to see the whole set of sick peers, not whichever one finished first."""

        async def answer() -> int:
            # [FUNCTION DOC] HELPER: Returns 0.
            return 0

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="first"):
                await async_utils.gather_and_raise_first(
                    [_failing("first"), answer(), _failing("second")],
                    describe_failure=lambda index: f"awaitable {index} failed",
                )

        assert "awaitable 0 failed" in caplog.text and "awaitable 2 failed" in caplog.text
        assert "awaitable 1 failed" not in caplog.text

    async def test_it_says_nothing_when_no_description_is_given(self, caplog):
        # [FUNCTION DOC] Without a describe-failure callback the failure is still raised ("quiet")
        # but nothing is logged: the captured log text is empty.
        """A caller that logs its own failures would otherwise have every one of them reported twice."""
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="quiet"):
                await async_utils.gather_and_raise_first([_failing("quiet")])

        assert caplog.text == ""


class TestGetAsyncLoop:
    # [CLASS DOC] Tests of the thread-safe lazy construction of the shared loop (double-checked
    # locking), using a fake loop thread class.
    def test_threads_arriving_together_share_one_loop(self, monkeypatch):
        # [FUNCTION DOC] Clears the singleton and swaps the loop thread class for a fake that sleeps
        # 0.05 s while constructing and records each instance. Eight pool threads call the accessor
        # concurrently. Asserts exactly one instance was built and every caller got that instance.
        """A second loop would strand whatever already awaits on the first, which no caller can recover from."""
        monkeypatch.setattr(async_utils, "async_loop", None)
        built: list[object] = []

        class SlowToBuild:
            # [CLASS DOC] HELPER: Fake loop thread whose constructor is slow and registers itself.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sleeps 0.05 s then appends self to the built list.
                time.sleep(0.05)
                built.append(self)

        monkeypatch.setattr(async_utils, "AsyncLoopThread", SlowToBuild)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            loops = [future.result() for future in [pool.submit(async_utils.get_async_loop) for _ in range(8)]]

        assert len(built) == 1
        assert all(loop is built[0] for loop in loops)

    def test_a_later_caller_is_answered_from_the_loop_already_built(self, monkeypatch):
        # [FUNCTION DOC] Clears the singleton and swaps the loop thread class for a factory returning
        # a plain object. Asserts a second accessor call returns the same object as the first.
        """The guard must not rebuild the loop once one exists, nor pay a lock on every rollout call."""
        monkeypatch.setattr(async_utils, "async_loop", None)
        monkeypatch.setattr(async_utils, "AsyncLoopThread", lambda: object())

        first = async_utils.get_async_loop()

        assert async_utils.get_async_loop() is first


class TestDisposer:
    # [CLASS DOC] Tests of the Disposer async context manager: items added (objects with an async
    # dispose method, sync or async callables, partials, none) are released in reverse order on exit.
    async def test_everything_added_is_released_in_reverse(self):
        # [FUNCTION DOC] Adds two engine objects in one call, then a sync callable, then an async
        # callable, and exits the context. Asserts release order is reversed overall: the async
        # callable, the sync callable, then the second engine, then the first engine.
        """The later object was built on the earlier one, so releasing it first is the only safe order."""
        events: list[str] = []

        class Engine:
            # [CLASS DOC] HELPER: Fake resource with a name and an async dispose method.
            def __init__(self, name: str) -> None:
                # [FUNCTION DOC] HELPER: Stores the name.
                self.name = name

            async def dispose(self) -> None:
                # [FUNCTION DOC] HELPER: Appends the engine's name to the events list.
                events.append(self.name)

        def close_reader() -> None:
            # [FUNCTION DOC] HELPER: Sync teardown; appends "reader" to the events.
            events.append("reader")

        async def close_writer() -> None:
            # [FUNCTION DOC] HELPER: Async teardown; appends "writer" to the events.
            events.append("writer")

        async with Disposer() as disposer:
            disposer.add(Engine("first"), Engine("second"))
            disposer.add(close_reader)
            disposer.add(close_writer)

        assert events == ["writer", "reader", "second", "first"]

    async def test_a_partial_is_awaited_like_the_coroutine_function_it_wraps(self):
        # [FUNCTION DOC] Adds a partial application of an async shutdown function bound to a handle;
        # after exit asserts the shutdown ran with that handle (the partial was awaited).
        """A teardown that needs an argument arrives as a partial, and dropping its await would skip it."""
        released: list[object] = []

        async def shutdown(handle: object) -> None:
            # [FUNCTION DOC] HELPER: Async teardown that records the handle it was given.
            released.append(handle)

        handle = object()

        async with Disposer() as disposer:
            disposer.add(partial(shutdown, handle))

        assert released == [handle]

    async def test_everything_added_is_released_when_the_body_raises(self):
        # [FUNCTION DOC] Adds a partial that records "released", then raises a value error inside the
        # context. Asserts the error propagates and the release still ran.
        """The exit path nobody wrote code for is the one that used to leave the whole run behind."""
        events: list[str] = []

        with pytest.raises(ValueError, match="training failed"):
            async with Disposer() as disposer:
                disposer.add(partial(events.append, "released"))
                raise ValueError("training failed")

        assert events == ["released"]

    def test_a_value_that_can_neither_dispose_nor_be_called_is_refused(self):
        # [FUNCTION DOC] Adding a plain object (no dispose method, not callable) raises an assertion
        # error.
        """Registering one silently would leave the object it stands for released by nobody."""
        with pytest.raises(AssertionError):
            Disposer().add(object())

    async def test_an_object_that_was_never_created_is_skipped(self):
        # [FUNCTION DOC] Adds a none value together with an async callable in one call; after exit
        # asserts only the callable ran (none is silently skipped).
        """A driver whose critic is optional registers both models in one call instead of guarding one."""
        events: list[str] = []

        async def close_actor() -> None:
            # [FUNCTION DOC] HELPER: Async teardown appending "actor" to the events.
            events.append("actor")

        async with Disposer() as disposer:
            disposer.add(None, close_actor)

        assert events == ["actor"]


class TestWithDisposer:
    # [CLASS DOC] Tests of with_disposer, which creates a Disposer, passes it as a keyword argument
    # to an async body function, and releases everything registered when the body finishes.
    async def test_what_the_body_registered_is_released_once_it_returns(self):
        # [FUNCTION DOC] Runs a body (given a positional label and keyword suffix) that registers a
        # partial and an async release function and returns label plus suffix. Asserts the return
        # value is "driver!" and the events are body first, then the engine release, then the
        # tracking release (reverse registration order).
        """The entry point owns the disposer so no driver has to remember to close one."""
        events: list[str] = []

        async def body(label: str, *, disposer: Disposer, suffix: str) -> str:
            # [FUNCTION DOC] HELPER: Registers two releases on the disposer, records "body" and
            # returns the label with the suffix appended.
            async def release_engine() -> None:
                # [FUNCTION DOC] HELPER: Appends "engine" to the events.
                events.append("engine")

            disposer.add(partial(events.append, "tracking"), release_engine)
            events.append("body")
            return label + suffix

        assert await with_disposer(body, "driver", suffix="!") == "driver!"
        assert events == ["body", "engine", "tracking"]

    async def test_what_the_body_registered_is_released_when_the_body_raises(self):
        # [FUNCTION DOC] The body registers a release then raises a value error "training failed".
        # Asserts the error propagates out of with_disposer and the release ran.
        """The exit path nobody wrote code for is the one that used to leave the whole run behind."""
        events: list[str] = []

        async def body(*, disposer: Disposer) -> None:
            # [FUNCTION DOC] HELPER: Registers a release on the disposer, then raises a value error.
            disposer.add(partial(events.append, "released"))
            raise ValueError("training failed")

        with pytest.raises(ValueError, match="training failed"):
            await with_disposer(body)

        assert events == ["released"]


class TestMaybeAwait:
    # [CLASS DOC] Tests of maybe_await, which awaits its argument only if it is awaitable.
    async def test_a_plain_result_is_handed_back_as_it_is(self):
        # [FUNCTION DOC] maybe_await of none returns none (no await error).
        """An out-of-tree hook written against the older contract returns None, and awaiting it would raise."""
        assert await maybe_await(None) is None

    async def test_an_awaitable_result_is_awaited(self):
        # [FUNCTION DOC] maybe_await of a coroutine runs it to completion and returns its value
        # ("done"); asserts the coroutine's side effect (finished flag) happened.
        """The in-tree hooks are coroutines, and their teardown must still finish before the caller moves on."""
        finished = False

        async def dispose():
            # [FUNCTION DOC] HELPER: Sets the enclosing finished flag and returns "done".
            nonlocal finished
            finished = True
            return "done"

        assert await maybe_await(dispose()) == "done"
        assert finished


class _ErrorWithoutAddNote(Exception):
    # [CLASS DOC] HELPER: Exception subclass that simulates Python versions lacking the exception
    # note feature by raising an attribute error whenever the add-note attribute is accessed.
    def __getattribute__(self, name: str):
        # [FUNCTION DOC] HELPER: Attribute lookup hook: raises an attribute error for the name
        # "add_note", otherwise defers to the normal lookup.
        if name == "add_note":
            raise AttributeError(name)
        return super().__getattribute__(name)


class TestKeepingASecondaryFailureWhereNotesDoNotExist:
    # [CLASS DOC] Tests of the private note-or-log helper: a secondary failure is attached as a note
    # to the primary exception when possible, otherwise logged at error level.
    def test_a_secondary_failure_is_logged_when_the_primary_cannot_carry_notes(self, caplog):
        # [FUNCTION DOC] Calls the helper with a primary exception lacking add-note and a text
        # "RuntimeError: cleanup"; asserts the text appears in the captured error log.
        """On 3.10 the annotation raised AttributeError from the error path and replaced the failure it described."""
        primary = _ErrorWithoutAddNote("training failed")

        with caplog.at_level(logging.ERROR):
            async_utils._exception_add_note_or_log(primary, "RuntimeError: cleanup")

        assert "RuntimeError: cleanup" in caplog.text

    def test_a_primary_that_can_carry_notes_is_annotated_rather_than_logged(self, caplog):
        # [FUNCTION DOC] Calls the helper with an ordinary value error; asserts the text is added to
        # the exception's notes and nothing is logged.
        """On 3.11 and later the note travels with the exception, which is where a caller reads it."""
        primary = ValueError("training failed")

        with caplog.at_level(logging.ERROR):
            async_utils._exception_add_note_or_log(primary, "RuntimeError: cleanup")

        assert any("RuntimeError: cleanup" in note for note in primary.__notes__)
        assert caplog.text == ""

    async def test_the_primary_failure_still_reaches_the_caller_unreplaced(self, caplog):
        # [FUNCTION DOC] End-to-end through wait_cancelling_pending_on_first_completion with a
        # primary failure of the note-less exception type and a cleanup failure in a peer. Asserts
        # the primary error propagates unchanged and the log contains "Additional task failure while
        # cancelling peers" and "RuntimeError: cleanup exploded".
        """Annotating the failure must never be able to become the failure the run reports."""

        async def cleanup_failure():
            # [FUNCTION DOC] HELPER: Sleeps 30 s; when cancelled raises a runtime error "cleanup
            # exploded".
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise RuntimeError("cleanup exploded") from None

        async def primary_failure():
            # [FUNCTION DOC] HELPER: Sleeps 0.01 s then raises the note-less exception "training
            # failed".
            await asyncio.sleep(0.01)
            raise _ErrorWithoutAddNote("training failed")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(_ErrorWithoutAddNote, match="training failed"):
                await wait_cancelling_pending_on_first_completion(
                    [asyncio.create_task(cleanup_failure()), asyncio.create_task(primary_failure())]
                )

        assert "Additional task failure while cancelling peers" in caplog.text
        assert "RuntimeError: cleanup exploded" in caplog.text
