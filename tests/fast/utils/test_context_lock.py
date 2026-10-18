# [MODULE DOC] Unit tests for the context lock module: ContextLock (an async lock that tracks
# which logical context holds it, with acquire/release, async-with, detach/reattach for
# cross-call windows, a released-section context manager and periodic "still waiting" reminders),
# the method decorators with_lock (hold the lock for a whole call), requires_lock (assert the
# caller holds the owning object's lock), acquires_lock / releases_lock (open and close a
# cross-call window), lock_exempt, and the class decorator enforce_lock_discipline (every method
# must carry one of these decorators, and decorated methods must live on an enforcing class).
# The tests use a shared guarded test double, a background holder task helper, and many small
# module-level async helpers; all are CPU only with the project's async test mode.
import asyncio
import dataclasses
import functools
import inspect
import itertools
import logging
from collections.abc import Callable, Coroutine
from types import SimpleNamespace
from typing import Any

import pytest

from miles.utils import context_lock
from miles.utils.context_lock import (
    ContextLock,
    acquires_lock,
    enforce_lock_discipline,
    lock_exempt,
    releases_lock,
    requires_lock,
    with_lock,
)


class _ConcurrencyTracker:
    # [CLASS DOC] HELPER: Counts how many calls are inside a section at once; max_concurrent_calls
    # records the high-water mark so tests can assert serialization (value 1).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with the high-water mark and the current count both zero.
        self.max_concurrent_calls = 0
        self._concurrent_calls = 0

    def enter(self) -> None:
        # [FUNCTION DOC] HELPER: Increments the current count and raises the high-water mark if
        # exceeded.
        self._concurrent_calls += 1
        self.max_concurrent_calls = max(self.max_concurrent_calls, self._concurrent_calls)

    def exit(self) -> None:
        # [FUNCTION DOC] HELPER: Decrements the current count.
        self._concurrent_calls -= 1


@enforce_lock_discipline
class _Guarded:
    # [CLASS DOC] HELPER: Fully disciplined test double used across the file. Holds a context lock
    # named "guarded", a concurrency tracker and a counter of calls to the async lock-requiring
    # method. Provides one method of each decorator kind: lock-holding calls (with_lock),
    # lock-requiring helpers (sync, async and property), and cross-call window start/end methods.
    @lock_exempt
    def __init__(self, tracker: _ConcurrencyTracker | None = None) -> None:
        # [FUNCTION DOC] HELPER: Exempt constructor: creates the lock named "guarded", uses the given
        # tracker or a new one, and zeroes the async-private-method call counter.
        self.context_lock = ContextLock("guarded")
        self.tracker = tracker if tracker is not None else _ConcurrencyTracker()
        self.async_private_method_calls = 0

    @with_lock
    async def locked_method(self, delay: float = 0) -> bool:
        # [FUNCTION DOC] HELPER: Under the lock: enters the tracker, sleeps for the delay, leaves the
        # tracker and returns whether the current context holds the lock (expected true).
        self.tracker.enter()
        await asyncio.sleep(delay)
        self.tracker.exit()
        return self.context_lock.held_in_current_context

    @with_lock
    async def locked_method_that_raises(self) -> None:
        # [FUNCTION DOC] HELPER: Under the lock: always raises a runtime error "boom".
        raise RuntimeError("boom")

    @with_lock
    async def locked_method_returning(self, value: int) -> int:
        # [FUNCTION DOC] HELPER: Under the lock: returns the given value.
        return value

    @with_lock
    async def locked_method_calling_private(self) -> bool:
        # [FUNCTION DOC] HELPER: Under the lock: returns the result of the sync lock-requiring helper.
        return self._private_method()

    @with_lock
    async def locked_method_waiting_in_a_released_section(self, resume: asyncio.Event) -> None:
        # [FUNCTION DOC] HELPER: Under the lock: temporarily releases the lock (released-section
        # context) and waits for the resume event inside it.
        async with self.context_lock.with_released():
            await resume.wait()

    @with_lock
    async def locked_method_fanning_out(self) -> list[bool]:
        # [FUNCTION DOC] HELPER: Under the lock: runs the async lock-requiring helper twice
        # concurrently with a gather and returns the two results (child tasks inherit the hold).
        return await asyncio.gather(self.async_private_method(), self.async_private_method())

    @requires_lock
    async def async_private_method(self) -> bool:
        # [FUNCTION DOC] HELPER: Requires the lock: increments the call counter and returns true. The
        # decorator asserts before the body runs, so a rejected call leaves the counter unchanged.
        self.async_private_method_calls += 1
        return True

    @requires_lock
    def _private_method(self) -> bool:
        # [FUNCTION DOC] HELPER: Requires the lock: returns true.
        return True

    @property
    @requires_lock
    def guarded_value(self) -> int:
        # [FUNCTION DOC] HELPER: Lock-requiring property returning the integer 42.
        return 42

    @acquires_lock
    async def start_window(self) -> int:
        # [FUNCTION DOC] HELPER: Opens a cross-call window (acquires the lock and leaves it locked
        # but detached after returning); inside, returns the guarded value 42.
        return self.guarded_value

    @acquires_lock
    async def start_window_that_raises(self) -> None:
        # [FUNCTION DOC] HELPER: Window start whose body raises a runtime error "boom".
        raise RuntimeError("boom")

    @acquires_lock
    async def start_window_that_waits(self, delay: float) -> None:
        # [FUNCTION DOC] HELPER: Window start whose body sleeps for the delay (cancellation target).
        await asyncio.sleep(delay)

    @acquires_lock
    async def start_window_waiting_in_a_released_section(self, resume: asyncio.Event) -> None:
        # [FUNCTION DOC] HELPER: Window start whose body temporarily releases the lock (released
        # section) and waits for the resume event inside it.
        async with self.context_lock.with_released():
            await resume.wait()

    @acquires_lock
    async def start_window_that_hands_the_lock_back_before_raising(self) -> None:
        # [FUNCTION DOC] HELPER: Window start whose body releases the lock itself and then raises a
        # runtime error "boom" (the decorator must tolerate the missing hold on the error path).
        self.context_lock.release()
        raise RuntimeError("boom")

    @releases_lock
    async def end_window(self) -> int:
        # [FUNCTION DOC] HELPER: Closes a cross-call window (reattaches the lock, runs the body, then
        # releases); the body returns the guarded value 42.
        return self.guarded_value

    @releases_lock
    async def end_window_that_raises(self) -> None:
        # [FUNCTION DOC] HELPER: Window end whose body raises a runtime error "boom".
        raise RuntimeError("boom")


class _HolderTask:
    # [CLASS DOC] HELPER: Holds a given context lock inside a separate asyncio task, so the test body
    # itself is a context that does not hold it. start launches the task and returns once the lock is
    # acquired; finish asks it to release and waits for the task to end.
    """Holds the lock in a separate task, so the test body stays a non-holding context."""

    def __init__(self, lock: ContextLock) -> None:
        # [FUNCTION DOC] HELPER: Stores the lock and creates the two events (acquired, release
        # requested); the task handle starts empty.
        self._lock = lock
        self._acquired = asyncio.Event()
        self._release_requested = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the holding task and waits until it signals acquisition.
        self._task = asyncio.create_task(self._hold())
        await self._acquired.wait()

    async def finish(self) -> None:
        # [FUNCTION DOC] HELPER: Requests release and awaits the holding task to completion.
        self._release_requested.set()
        await self._task

    async def _hold(self) -> None:
        # [FUNCTION DOC] HELPER: Task body: acquires the lock, signals acquired, waits for the release
        # request, then releases the lock.
        await self._lock.acquire()
        self._acquired.set()
        await self._release_requested.wait()
        self._lock.release()


class TestContextLock:
    # [CLASS DOC] Tests of the basic ContextLock behavior: locked and held-in-context state,
    # name, reentrancy assertions, waiting, visibility across tasks, async-with usage and its
    # exception-exit handling.
    @pytest.mark.asyncio
    async def test_acquire_and_release_toggle_locked_and_held_state(self):
        # [FUNCTION DOC] After acquire, the lock is locked and held by the current context; after
        # release both are false.
        """acquire marks the lock locked and held by the current context; release clears both."""
        lock = ContextLock("test")
        await lock.acquire()
        assert lock.locked and lock.held_in_current_context
        lock.release()
        assert not lock.locked and not lock.held_in_current_context

    def test_the_lock_exposes_its_name(self):
        # [FUNCTION DOC] The name given at construction is available as the name attribute.
        """Diagnostics identify which lock is involved."""
        assert ContextLock("InferenceController").name == "InferenceController"

    @pytest.mark.asyncio
    async def test_acquire_asserts_when_the_same_lock_is_already_held_in_context(self):
        # [FUNCTION DOC] A second acquire of a lock already held by this context raises an assertion
        # error matching "already held" instead of waiting.
        """Reentrant acquisition is a bug, not a wait."""
        lock = ContextLock("test")
        await lock.acquire()
        with pytest.raises(AssertionError, match="already held"):
            await lock.acquire()

    @pytest.mark.asyncio
    async def test_acquire_asserts_when_another_lock_is_held_in_context(self):
        # [FUNCTION DOC] Acquiring a second, different lock while holding the first raises an
        # assertion error matching "already held" (a context holds at most one context lock).
        """Holding two context locks at once would make held-by ambiguous."""
        first_lock = ContextLock("first")
        second_lock = ContextLock("second")
        await first_lock.acquire()
        with pytest.raises(AssertionError, match="already held"):
            await second_lock.acquire()

    @pytest.mark.asyncio
    async def test_release_asserts_when_not_held_by_the_current_context(self):
        # [FUNCTION DOC] Releasing a never-acquired lock raises an assertion error matching "must be
        # held".
        """A context that never acquired the lock cannot release it."""
        lock = ContextLock("test")
        with pytest.raises(AssertionError, match="must be held"):
            lock.release()

    @pytest.mark.asyncio
    async def test_release_asserts_when_the_lock_is_held_by_another_task(self):
        # [FUNCTION DOC] While a helper task holds the lock, releasing from the test context raises
        # an assertion error matching "must be held"; the holder is then finished cleanly.
        """Only the holding context may release, even though the lock is locked."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()
        with pytest.raises(AssertionError, match="must be held"):
            lock.release()
        await holder.finish()

    @pytest.mark.asyncio
    async def test_a_second_acquirer_waits_until_release(self):
        # [FUNCTION DOC] While a helper task holds the lock, an acquire task stays pending after five
        # loop yields; after the holder finishes, the waiter completes and the lock is locked (now
        # by the waiter).
        """A caller blocks on acquire until the holding task releases."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()
        waiter = asyncio.create_task(lock.acquire())
        for _ in range(5):
            await asyncio.sleep(0)
        assert not waiter.done()

        await holder.finish()
        await waiter
        assert lock.locked

    @pytest.mark.asyncio
    async def test_held_state_is_not_visible_to_an_unrelated_context(self):
        # [FUNCTION DOC] While a helper task holds the lock, the test context sees the lock as
        # locked but not as held in its own context.
        """held_in_current_context tracks the holder, not the global locked flag."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()
        assert lock.locked
        assert not lock.held_in_current_context
        await holder.finish()

    @pytest.mark.asyncio
    async def test_holding_one_lock_does_not_look_like_holding_another(self):
        # [FUNCTION DOC] After acquiring one lock, it reports held in this context while a second,
        # distinct lock does not.
        """Two locks are tracked as distinct objects, not as one global flag."""
        lock = ContextLock("test")
        other_lock = ContextLock("other")
        await lock.acquire()
        assert lock.held_in_current_context
        assert not other_lock.held_in_current_context

    @pytest.mark.asyncio
    async def test_work_spawned_inside_the_critical_section_sees_the_lock_as_held(self):
        # [FUNCTION DOC] Inside an async-with block, two tasks gathered together each read the held
        # state through a helper and both see true.
        """Tasks fanned out from inside the section (e.g. asyncio.gather) are still inside it."""
        lock = ContextLock("test")
        async with lock:
            assert all(await asyncio.gather(_read_held(lock), _read_held(lock)))

    @pytest.mark.asyncio
    async def test_async_with_acquires_and_releases(self):
        # [FUNCTION DOC] Inside async-with the lock is locked and held in context; after the block it
        # is not locked.
        """The lock is a context manager for plain lexical critical sections."""
        lock = ContextLock("test")
        async with lock:
            assert lock.locked and lock.held_in_current_context
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_async_with_binds_the_context_lock_instance(self):
        # [FUNCTION DOC] The value bound by async-with "as" is the very same lock object.
        """`async with lock as acquired` must hand back the lock itself, not the raw asyncio primitive."""
        lock = ContextLock("test")
        async with lock as acquired:
            assert acquired is lock

    @pytest.mark.asyncio
    async def test_async_with_releases_on_exception(self):
        # [FUNCTION DOC] A runtime error "boom" raised in the block propagates and the lock ends up
        # not locked.
        """An exception inside the critical section must not leak a held lock."""
        lock = ContextLock("test")
        with pytest.raises(RuntimeError, match="boom"):
            async with lock:
                raise RuntimeError("boom")
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_async_with_reports_the_bodys_own_error_when_the_lock_is_already_gone(self):
        # [FUNCTION DOC] The block releases the lock itself and then raises "boom"; the exit must not
        # raise an assertion over the missing hold, so the runtime error "boom" is what propagates,
        # and the lock is not locked afterwards.
        """Asserting on the way out would bury the failure the caller actually needs to see."""
        lock = ContextLock("test")

        with pytest.raises(RuntimeError, match="boom"):
            async with lock:
                lock.release()
                raise RuntimeError("boom")

        assert not lock.locked

    @pytest.mark.asyncio
    async def test_async_with_logs_the_skipped_release_when_the_lock_is_already_gone(self, caplog):
        # [FUNCTION DOC] Same scenario as above, with error-level capture on the context lock logger:
        # the only skipped-release message is exactly "Lock 'named-lock' is not held by the current
        # context on exception exit; skipping release".
        """Skipping the release silently would hide broken release discipline from the operator."""
        lock = ContextLock("named-lock")

        with caplog.at_level(logging.ERROR, logger="miles.utils.context_lock"):
            with pytest.raises(RuntimeError, match="boom"):
                async with lock:
                    lock.release()
                    raise RuntimeError("boom")

        assert _skipped_release_messages(caplog) == [
            "Lock 'named-lock' is not held by the current context on exception exit; skipping release"
        ]

    @pytest.mark.asyncio
    async def test_async_with_still_asserts_when_a_successful_body_gave_the_lock_away(self):
        # [FUNCTION DOC] If the block releases the lock and then ends without error, the exit's own
        # release raises an assertion error matching "must be held" (tolerance applies only on the
        # exception path).
        """Tolerating a missing lock on the success path would let a stray release go unnoticed."""
        lock = ContextLock("test")

        with pytest.raises(AssertionError, match="must be held"):
            async with lock:
                lock.release()


class TestWithLock:
    # [CLASS DOC] Tests of the with_lock method decorator: acquire/release around the body,
    # release on error and cancellation, serialization (also across instances sharing a lock),
    # pass-through of values, decoration-time and call-time validation.
    @pytest.mark.asyncio
    async def test_the_lock_is_held_during_the_call_and_released_after(self):
        # [FUNCTION DOC] A locked method returns true (held inside), and after the call the lock is
        # not locked.
        """with_lock wraps the method body in an acquire/release pair."""
        guarded = _Guarded()
        assert await guarded.locked_method() is True
        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_the_lock_is_released_when_the_method_raises(self):
        # [FUNCTION DOC] A locked method raising "boom" propagates the error and leaves the lock not
        # locked.
        """An exception inside the method must not leak a held lock."""
        guarded = _Guarded()
        with pytest.raises(RuntimeError, match="boom"):
            await guarded.locked_method_that_raises()
        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_cancelling_a_with_lock_call_releases_the_lock(self):
        # [FUNCTION DOC] Starts a locked call with delay 1 s as a task, yields five times and asserts
        # the lock is locked; cancels it with a helper; asserts the lock is free and that a new
        # locked call succeeds.
        """A caller that gives up mid-body must not wedge the lock for everyone else."""
        guarded = _Guarded()
        call = asyncio.create_task(guarded.locked_method(delay=1))
        for _ in range(5):
            await asyncio.sleep(0)
        assert guarded.context_lock.locked

        await _cancel(call)

        assert not guarded.context_lock.locked
        assert await guarded.locked_method() is True

    @pytest.mark.asyncio
    async def test_concurrent_calls_are_serialized(self):
        # [FUNCTION DOC] Two locked calls with 0.01 s delay gathered together never overlap: the
        # tracker's maximum concurrency is 1.
        """Two decorated calls never run their bodies at the same time."""
        guarded = _Guarded()
        await asyncio.gather(guarded.locked_method(delay=0.01), guarded.locked_method(delay=0.01))
        assert guarded.tracker.max_concurrent_calls == 1

    @pytest.mark.asyncio
    async def test_the_return_value_is_passed_through(self):
        # [FUNCTION DOC] A locked method called with 42 returns 42.
        """The wrapper is transparent for arguments and return values."""
        guarded = _Guarded()
        assert await guarded.locked_method_returning(42) == 42

    @pytest.mark.asyncio
    async def test_two_instances_sharing_a_lock_serialize_against_each_other(self):
        # [FUNCTION DOC] Two guarded instances share one tracker and the first's lock object; locked
        # calls gathered on both never overlap (maximum concurrency 1).
        """Two instances handed the very same lock object never overlap their bodies."""
        shared_tracker = _ConcurrencyTracker()
        first = _Guarded(tracker=shared_tracker)
        second = _Guarded(tracker=shared_tracker)
        second.context_lock = first.context_lock

        await asyncio.gather(first.locked_method(delay=0.01), second.locked_method(delay=0.01))
        assert shared_tracker.max_concurrent_calls == 1

    def test_rejects_sync_functions_at_decoration_time(self):
        # [FUNCTION DOC] Decorating a plain (non-async) function with with_lock raises an assertion
        # error matching "must be async".
        """with_lock cannot await inside a sync function, so it must refuse one."""
        with pytest.raises(AssertionError, match="must be async"):

            @with_lock
            def sync_method(self) -> None:
                # [FUNCTION DOC] HELPER: Dummy sync function that must be rejected by the decorator.
                pass

    @pytest.mark.asyncio
    async def test_a_nested_call_to_another_locked_method_asserts(self):
        # [FUNCTION DOC] A locked method calling another locked method of the same object raises an
        # assertion error matching "already held" rather than deadlocking.
        """Lock reentrancy through decorated methods is surfaced as an assertion, not a deadlock."""

        @enforce_lock_discipline
        class _Nested:
            # [CLASS DOC] HELPER: Disciplined class with an outer locked method that calls an inner
            # locked method.
            @lock_exempt
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the lock named "nested".
                self.context_lock = ContextLock("nested")

            @with_lock
            async def outer(self) -> None:
                # [FUNCTION DOC] HELPER: Under the lock, awaits the inner locked method.
                await self.inner()

            @with_lock
            async def inner(self) -> None:
                # [FUNCTION DOC] HELPER: Locked no-op method.
                pass

        with pytest.raises(AssertionError, match="already held"):
            await _Nested().outer()

    @pytest.mark.asyncio
    async def test_a_missing_lock_attribute_is_reported(self):
        # [FUNCTION DOC] Calling a locked method on an object lacking the context lock attribute
        # raises an attribute error mentioning "context_lock".
        """A class that never got handed its lock fails loudly at call time."""

        @enforce_lock_discipline
        class _NoLock:
            # [CLASS DOC] HELPER: Disciplined class with a locked method but no lock attribute.
            @with_lock
            async def method(self) -> None:
                # [FUNCTION DOC] HELPER: Locked no-op method.
                pass

        with pytest.raises(AttributeError, match="context_lock"):
            await _NoLock().method()

    @pytest.mark.asyncio
    async def test_a_lock_attribute_of_the_wrong_type_is_reported(self):
        # [FUNCTION DOC] A locked method on an object whose lock attribute is a plain asyncio lock
        # raises an assertion error matching "must be a ContextLock".
        """Handing over a bare asyncio.Lock loses the held-by tracking, so it is refused."""

        @enforce_lock_discipline
        class _WrongLock:
            # [CLASS DOC] HELPER: Disciplined class whose lock attribute is a bare asyncio lock.
            @lock_exempt
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the lock attribute to a plain asyncio lock.
                self.context_lock = asyncio.Lock()

            @with_lock
            async def method(self) -> None:
                # [FUNCTION DOC] HELPER: Locked no-op method.
                pass

        with pytest.raises(AssertionError, match="must be a ContextLock"):
            await _WrongLock().method()


async def _read_held(lock: ContextLock) -> bool:
    # [FUNCTION DOC] HELPER: Returns whether the lock is held in the current context.
    return lock.held_in_current_context


async def _reattach_and_release(lock: ContextLock) -> None:
    # [FUNCTION DOC] HELPER: In the current context, reattaches a detached lock and releases it.
    lock.reattach()
    lock.release()


async def _acquire_release_and_report(lock: ContextLock, taken: list) -> None:
    # [FUNCTION DOC] HELPER: Acquires the lock, appends whether it is held in this context to the
    # given list, then releases.
    await lock.acquire()
    taken.append(lock.held_in_current_context)
    lock.release()


async def _acquire_and_report_held(lock: ContextLock) -> bool:
    # [FUNCTION DOC] HELPER: Acquires the lock (without releasing) and returns whether it is held
    # in the current context.
    await lock.acquire()
    return lock.held_in_current_context


async def _release_and_report_rejection(lock: ContextLock, gate: asyncio.Event) -> bool:
    # [FUNCTION DOC] HELPER: Waits for the gate, then tries to release; returns true if the release
    # raised an assertion error, false if it succeeded.
    await gate.wait()
    try:
        lock.release()
    except AssertionError:
        return True
    return False


async def _reattach_and_report_held(lock: ContextLock, gate: asyncio.Event) -> bool:
    # [FUNCTION DOC] HELPER: Waits for the gate, reattaches the lock and returns whether it is now
    # held in this context.
    await gate.wait()
    lock.reattach()
    return lock.held_in_current_context


async def _leave_the_released_block_after(lock: ContextLock, gate: asyncio.Event) -> bool:
    # [FUNCTION DOC] HELPER: Enters the released-section block, waits for the gate inside it, leaves
    # the block (which must reacquire) and returns whether the lock is held in this context.
    async with lock.with_released():
        await gate.wait()
    return lock.held_in_current_context


async def _read_held_after(lock: ContextLock, gate: asyncio.Event) -> bool:
    # [FUNCTION DOC] HELPER: Waits for the gate, then returns whether the lock is held in this
    # context.
    await gate.wait()
    return lock.held_in_current_context


async def _cancel(task: asyncio.Task) -> None:
    # [FUNCTION DOC] HELPER: Cancels the task and awaits it, asserting that the cancelled error is
    # raised.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _cancel_while_reacquiring(lock: ContextLock) -> tuple[asyncio.Task, _HolderTask]:
    # [FUNCTION DOC] HELPER: Reproduces a cancellation during reacquisition. A section task acquires
    # the lock and enters a released block waiting on a resume event. After one yield, a holder task
    # takes the lock; the resume event is set and four yields let the section try to reacquire (and
    # block); then the section is cancelled and awaited (exceptions collected). Returns the section
    # task and the still-holding holder (the caller must finish it).
    resume = asyncio.Event()

    async def _released_section() -> None:
        # [FUNCTION DOC] HELPER: Acquires the lock via async-with, then releases it temporarily and
        # waits for resume.
        async with lock:
            async with lock.with_released():
                await resume.wait()

    section = asyncio.create_task(_released_section())
    await asyncio.sleep(0)
    holder = _HolderTask(lock)
    await holder.start()
    resume.set()
    for _ in range(4):
        await asyncio.sleep(0)
    section.cancel()
    await asyncio.gather(section, return_exceptions=True)
    return section, holder


async def _cancel_a_reacquiring_call(
    lock: ContextLock, call: Callable[[asyncio.Event], Coroutine[Any, Any, Any]]
) -> asyncio.Task:
    # [FUNCTION DOC] HELPER: Like the previous helper but for a decorated method call that takes a
    # resume event: starts the call as a task, lets it run one yield, has a holder task take the
    # lock, sets the resume event, yields four times so the call tries to reacquire, cancels the
    # call and awaits it (exceptions collected), finishes the holder, and returns the call task.
    resume = asyncio.Event()

    task = asyncio.create_task(call(resume))
    await asyncio.sleep(0)
    holder = _HolderTask(lock)
    await holder.start()
    resume.set()
    for _ in range(4):
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await holder.finish()
    return task


def _skipped_release_messages(caplog) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the messages of captured log records containing "skipping
    # release".
    return [record.message for record in caplog.records if "skipping release" in record.message]


def _reminder_messages(caplog) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the messages of captured log records starting with "Still
    # waiting for lock".
    return [record.message for record in caplog.records if record.message.startswith("Still waiting for lock")]


async def _set_after_yields(event: asyncio.Event, yields: int) -> None:
    # [FUNCTION DOC] HELPER: Yields to the event loop the given number of times, then sets the event.
    for _ in range(yields):
        await asyncio.sleep(0)
    event.set()


async def _wait_until(condition: Callable[[], bool], *, timeout_seconds: float = 5.0) -> None:
    # [FUNCTION DOC] HELPER: Polls the condition (sleeping 1 ms between checks, checking first with
    # no delay) until it is true. Raises a timeout error if it is still false after timeout_seconds
    # (default 5).
    async def _poll() -> None:
        # [FUNCTION DOC] HELPER: Loops sleeping 1 ms until the condition is true.
        while not condition():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(_poll(), timeout=timeout_seconds)


class TestWaitReminder:
    # [CLASS DOC] Tests of the periodic "Still waiting for lock '<name>' after <N>s" info log emitted
    # while an acquirer is blocked: repetition, content, elapsed-time progression, absence when
    # uncontended, stopping on acquisition or cancellation, and independence from wall-clock time.
    @pytest.fixture
    def fast_reminder(self, monkeypatch):
        # [FUNCTION DOC] HELPER: Sets the module's reminder interval to 0.01 s for the test.
        monkeypatch.setattr(context_lock, "WAIT_LOG_INTERVAL_SECONDS", 0.01)

    @pytest.fixture
    def instant_reminder(self, monkeypatch):
        # [FUNCTION DOC] HELPER: Sets the module's reminder interval to 0 so reminders fire on every
        # loop turn.
        monkeypatch.setattr(context_lock, "WAIT_LOG_INTERVAL_SECONDS", 0)

    @pytest.mark.asyncio
    async def test_a_blocked_acquirer_reminds_repeatedly(self, fast_reminder, caplog):
        # [FUNCTION DOC] A holder task holds the lock and a waiter blocks on acquire. Polls (with a
        # helper, not a fixed sleep) until at least two reminder messages are captured, then cancels
        # the waiter, finishes the holder and asserts at least two reminders exist.
        """A caller stuck on the lock keeps reporting that it is still waiting."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()

        waiter = asyncio.create_task(lock.acquire())
        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            await _wait_until(lambda: len(_reminder_messages(caplog)) >= 2)
        await _cancel(waiter)
        await holder.finish()

        assert len(_reminder_messages(caplog)) >= 2

    @pytest.mark.asyncio
    async def test_the_reminder_names_the_lock_and_the_elapsed_time(self, fast_reminder, caplog):
        # [FUNCTION DOC] Blocks a waiter for 0.05 s behind a holder; asserts at least one reminder was
        # logged and all start with "Still waiting for lock 'stuck-lock' after " and end with "s".
        """The operator needs to know which lock is stuck and for how long."""
        lock = ContextLock("stuck-lock")
        holder = _HolderTask(lock)
        await holder.start()

        waiter = asyncio.create_task(lock.acquire())
        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            await asyncio.sleep(0.05)
        await _cancel(waiter)
        await holder.finish()

        messages = _reminder_messages(caplog)
        assert messages
        assert all(message.startswith("Still waiting for lock 'stuck-lock' after ") for message in messages)
        assert all(message.endswith("s") for message in messages)

    @pytest.mark.asyncio
    async def test_the_reminder_counts_up_while_the_wait_drags_on(self, fast_reminder, caplog, monkeypatch):
        # [FUNCTION DOC] Replaces the module's time source with a fake whose monotonic clock returns
        # 0, 6, 12, ... on successive calls. After blocking a waiter for 0.05 s, the first two
        # reminders are exactly "Still waiting for lock 'stuck-lock' after 6s" and "... after 12s".
        """A reminder that always says the same thing cannot tell a short stall from a stuck one."""
        ticks = itertools.count(start=0, step=6)
        monkeypatch.setattr(context_lock, "time", SimpleNamespace(monotonic=lambda: float(next(ticks))))
        lock = ContextLock("stuck-lock")
        holder = _HolderTask(lock)
        await holder.start()

        waiter = asyncio.create_task(lock.acquire())
        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            await asyncio.sleep(0.05)
        await _cancel(waiter)
        await holder.finish()

        assert _reminder_messages(caplog)[:2] == [
            "Still waiting for lock 'stuck-lock' after 6s",
            "Still waiting for lock 'stuck-lock' after 12s",
        ]

    @pytest.mark.asyncio
    async def test_an_uncontended_acquire_reminds_nothing(self, fast_reminder, caplog):
        # [FUNCTION DOC] Acquiring a free lock, sleeping 0.05 s and releasing logs no reminder.
        """The reminder must only fire while actually blocked."""
        lock = ContextLock("test")
        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            await lock.acquire()
            await asyncio.sleep(0.05)
            lock.release()

        assert _reminder_messages(caplog) == []

    @pytest.mark.asyncio
    async def test_the_reminder_stops_once_the_lock_is_acquired(self, fast_reminder, caplog):
        # [FUNCTION DOC] A waiter blocks for 0.03 s, then the holder finishes and the waiter gets the
        # lock. Clears captured logs and observes a further 0.05 s; no reminder appears in that
        # window.
        """No reminder may keep firing after the caller got in."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()
        waiter = asyncio.create_task(lock.acquire())
        await asyncio.sleep(0.03)
        await holder.finish()
        await waiter

        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            # Only the window after the acquisition is under test; the reminders
            # from the legitimate wait above are captured too and are expected.
            caplog.clear()
            await asyncio.sleep(0.05)

        assert _reminder_messages(caplog) == []

    @pytest.mark.asyncio
    async def test_a_waiter_that_eventually_wins_acquires_normally(self, fast_reminder):
        # [FUNCTION DOC] A waiter blocked for 0.05 s behind a holder acquires after the holder
        # finishes: its held-state report is true and the lock is locked.
        """Reminding must not interfere with the acquisition itself."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()

        waiter = asyncio.create_task(_acquire_and_report_held(lock))
        await asyncio.sleep(0.05)
        await holder.finish()

        assert await waiter is True
        assert lock.locked

    @pytest.mark.asyncio
    async def test_a_cancelled_waiter_leaves_the_lock_untouched(self, fast_reminder):
        # [FUNCTION DOC] A waiter is cancelled after 0.03 s of blocking; once the holder finishes the
        # lock is not locked (the cancelled waiter neither took nor kept it).
        """Giving up on the lock must neither mark it held nor steal it."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()

        waiter = asyncio.create_task(lock.acquire())
        await asyncio.sleep(0.03)
        await _cancel(waiter)
        await holder.finish()

        assert not lock.locked

    @pytest.mark.asyncio
    async def test_a_cancelled_waiter_stops_reminding(self, fast_reminder, caplog):
        # [FUNCTION DOC] After cancelling a waiter, clears captured logs and observes 0.05 s while the
        # holder still holds the lock; no reminder is logged in that window.
        """A leaked reminder would keep claiming a caller is waiting long after it gave up."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()

        waiter = asyncio.create_task(lock.acquire())
        await asyncio.sleep(0.03)
        await _cancel(waiter)
        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            # Only the window after the cancellation is under test.
            caplog.clear()
            await asyncio.sleep(0.05)
        await holder.finish()

        assert _reminder_messages(caplog) == []

    @pytest.mark.asyncio
    async def test_the_reminders_repeat_without_spending_wall_clock_time(self, instant_reminder, caplog):
        # [FUNCTION DOC] With a zero reminder interval and a blocked waiter, twenty bare loop yields
        # (no sleeping) produce at least three reminders.
        """Repeated reminders must follow the wait going on, not how much real time a runner grants."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()

        waiter = asyncio.create_task(lock.acquire())
        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            for _ in range(20):
                await asyncio.sleep(0)
        await _cancel(waiter)
        await holder.finish()

        assert len(_reminder_messages(caplog)) >= 3


class TestWaitReminderTestDeterminism:
    # [CLASS DOC] Meta-test guarding the determinism of a reminder test by inspecting its source.
    def test_the_repeated_reminder_test_waits_for_the_reminders_not_for_a_sleep_window(self):
        # [FUNCTION DOC] Reads the source text of the repeated-reminder test and asserts it uses the
        # polling helper and contains no direct asyncio sleep call.
        """A fixed sleep window can end before the second reminder arrives on a loaded runner."""
        source = inspect.getsource(TestWaitReminder.test_a_blocked_acquirer_reminds_repeatedly)

        assert "_wait_until(" in source
        assert "asyncio.sleep(" not in source


class TestWaitUntil:
    # [CLASS DOC] Tests of the polling helper used by the reminder tests.
    @pytest.mark.asyncio
    async def test_an_already_satisfied_condition_returns_without_a_polling_round(self):
        # [FUNCTION DOC] A condition that is true at once is evaluated exactly once.
        """Waiting for something that already happened must cost nothing."""
        calls = 0

        def _condition() -> bool:
            # [FUNCTION DOC] HELPER: Counts calls and always returns true.
            nonlocal calls
            calls += 1
            return True

        await _wait_until(_condition)

        assert calls == 1

    @pytest.mark.asyncio
    async def test_the_condition_is_polled_until_it_turns_true_and_no_further(self):
        # [FUNCTION DOC] A condition that turns true on its third evaluation stops the wait there:
        # the remaining-polls counter ends at exactly 0 (not negative).
        """The wait ends on the condition itself, not after a preset number of rounds."""
        remaining_polls = 3

        def _condition() -> bool:
            # [FUNCTION DOC] HELPER: Decrements the counter and returns true once it is zero or less.
            nonlocal remaining_polls
            remaining_polls -= 1
            return remaining_polls <= 0

        await _wait_until(_condition)

        assert remaining_polls == 0

    @pytest.mark.asyncio
    async def test_a_condition_satisfied_by_another_task_ends_the_wait(self):
        # [FUNCTION DOC] A concurrent task sets an event after five yields; waiting on the event's
        # is-set check returns, the event is set, and the setter task is awaited.
        """Progress made concurrently must be what releases the waiter."""
        satisfied = asyncio.Event()
        setter = asyncio.create_task(_set_after_yields(event=satisfied, yields=5))

        await _wait_until(satisfied.is_set)

        assert satisfied.is_set()
        await setter

    @pytest.mark.asyncio
    async def test_a_condition_that_never_turns_true_times_out(self):
        # [FUNCTION DOC] An always-false condition with a 0.02 s timeout raises a timeout error, and
        # the elapsed loop time is under 1 s.
        """A never satisfied condition must fail the test instead of hanging it."""
        loop = asyncio.get_running_loop()
        started_at = loop.time()

        with pytest.raises(asyncio.TimeoutError):
            await _wait_until(lambda: False, timeout_seconds=0.02)

        assert loop.time() - started_at < 1.0


class TestDetachAndReattach:
    # [CLASS DOC] Tests of detach (hand the held lock to no one, keeping it locked) and reattach
    # (any context adopts a detached lock, issuing a fresh ownership generation, so inherited grants
    # from before become stale).
    @pytest.mark.asyncio
    async def test_detach_keeps_the_lock_locked_but_not_held(self):
        # [FUNCTION DOC] After acquire and detach the lock is locked but not held in this context.
        """A detached lock stays locked so other contexts keep blocking."""
        lock = ContextLock("test")
        await lock.acquire()
        lock.detach()
        assert lock.locked and not lock.held_in_current_context

    @pytest.mark.asyncio
    async def test_detach_asserts_when_not_held_by_the_current_context(self):
        # [FUNCTION DOC] Detaching a never-acquired lock raises an assertion error matching "must be
        # held".
        """Only the holding context may hand the lock over."""
        lock = ContextLock("test")
        with pytest.raises(AssertionError, match="must be held"):
            lock.detach()

    @pytest.mark.asyncio
    async def test_a_detached_lock_still_blocks_other_acquirers(self):
        # [FUNCTION DOC] With the lock detached, an acquire task remains pending after five yields;
        # reattaching and releasing lets it complete.
        """Detaching hands the lock across calls; it does not open it up."""
        lock = ContextLock("test")
        await lock.acquire()
        lock.detach()
        waiter = asyncio.create_task(lock.acquire())
        for _ in range(5):
            await asyncio.sleep(0)
        assert not waiter.done()

        lock.reattach()
        lock.release()
        await waiter

    @pytest.mark.asyncio
    async def test_a_detached_lock_can_be_reattached_and_released_by_another_task(self):
        # [FUNCTION DOC] A separate task reattaches and releases a lock detached by the test; the lock
        # ends not locked.
        """The start/end pair of a cross-call window may run in different tasks."""
        lock = ContextLock("test")
        await lock.acquire()
        lock.detach()
        await asyncio.create_task(_reattach_and_release(lock))
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_reattach_asserts_when_the_lock_was_not_detached(self):
        # [FUNCTION DOC] Reattaching a lock that a holder task holds normally raises an assertion
        # error matching "was not detached".
        """reattach must not steal a lock that is held normally by someone else."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()
        with pytest.raises(AssertionError, match="was not detached"):
            lock.reattach()
        await holder.finish()

    @pytest.mark.asyncio
    async def test_reattach_asserts_when_a_lock_is_already_held_in_context(self):
        # [FUNCTION DOC] Detach one lock, acquire another, then reattach the first: raises an
        # assertion error matching "already held".
        """A context cannot adopt a detached lock while holding another lock."""
        lock = ContextLock("test")
        await lock.acquire()
        lock.detach()
        other_lock = ContextLock("other")
        await other_lock.acquire()
        with pytest.raises(AssertionError, match="already held"):
            lock.reattach()

    @pytest.mark.asyncio
    async def test_reattach_restores_the_held_state(self):
        # [FUNCTION DOC] After detach then reattach the lock is held in this context again; release
        # then leaves it not locked.
        """After reattach the context may again call lock-requiring helpers."""
        lock = ContextLock("test")
        await lock.acquire()
        lock.detach()
        lock.reattach()
        assert lock.held_in_current_context
        lock.release()
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_a_child_task_loses_its_authorization_when_the_holder_detaches(self):
        # [FUNCTION DOC] A child task created while the lock is held (inheriting the grant) waits for
        # a gate; the parent detaches then opens the gate. The child reports not held.
        """Detach hands the lock away, so a task spawned in the section must stop believing it holds it."""
        lock = ContextLock("test")
        await lock.acquire()
        task = asyncio.create_task(_read_held_after(lock, gate := asyncio.Event()))
        lock.detach()
        gate.set()

        assert not await task

    @pytest.mark.asyncio
    async def test_a_child_task_cannot_release_a_detached_lock(self):
        # [FUNCTION DOC] A child task created while held tries to release after the parent detached;
        # its release is rejected with an assertion error (task returns true) and the lock remains
        # locked.
        """Releasing across the detach boundary would open a window someone else is meant to close."""
        lock = ContextLock("test")
        await lock.acquire()
        task = asyncio.create_task(_release_and_report_rejection(lock, gate := asyncio.Event()))
        lock.detach()
        gate.set()

        assert await task
        assert lock.locked

    @pytest.mark.asyncio
    async def test_reattach_is_not_blocked_by_a_stale_grant(self):
        # [FUNCTION DOC] A child task holding a stale inherited grant reattaches the detached lock
        # after the gate; it succeeds, reporting held in its context, and the lock stays locked.
        """Any context may close the window; an expired grant must not veto that."""
        lock = ContextLock("test")
        await lock.acquire()
        task = asyncio.create_task(_reattach_and_report_held(lock, gate := asyncio.Event()))
        lock.detach()
        gate.set()

        assert await task
        assert lock.locked

    @pytest.mark.asyncio
    async def test_reattach_issues_a_fresh_generation(self):
        # [FUNCTION DOC] A child created while held waits for a gate; the parent detaches and
        # reattaches before opening the gate. The child still reports not held (its old grant does
        # not revive).
        """A grant inherited before the detach must not come back to life after the reattach."""
        lock = ContextLock("test")
        await lock.acquire()
        task = asyncio.create_task(_read_held_after(lock, gate := asyncio.Event()))
        lock.detach()
        lock.reattach()
        gate.set()

        assert not await task

    @pytest.mark.asyncio
    async def test_reattach_asserts_when_the_lock_is_no_longer_locked(self):
        # [FUNCTION DOC] After detaching, the underlying primitive lock is released directly (bypassing
        # the wrapper); reattaching raises an assertion error matching "no longer locked".
        """A grant must never be handed out for a lock that nobody actually holds."""
        lock = ContextLock("test")
        await lock.acquire()
        lock.detach()
        lock._lock.release()

        with pytest.raises(AssertionError, match="no longer locked"):
            lock.reattach()


class TestWithReleased:
    # [CLASS DOC] Tests of the released-section context manager: the lock is temporarily open
    # inside the block and reacquired (queueing behind other holders) afterwards.
    @pytest.mark.asyncio
    async def test_the_lock_is_open_inside_the_block(self):
        # [FUNCTION DOC] Inside the block the lock is neither locked nor held in this context.
        """A long wait must not keep everyone else out while it polls."""
        lock = ContextLock("test")
        await lock.acquire()

        async with lock.with_released():
            assert not lock.locked and not lock.held_in_current_context

    @pytest.mark.asyncio
    async def test_the_lock_is_held_again_after_the_block(self):
        # [FUNCTION DOC] After an empty released block the lock is locked and held in this context.
        """The caller was inside a locked region, so it must get its lock back."""
        lock = ContextLock("test")
        await lock.acquire()

        async with lock.with_released():
            pass

        assert lock.locked and lock.held_in_current_context

    @pytest.mark.asyncio
    async def test_another_task_can_take_the_lock_inside_the_block(self):
        # [FUNCTION DOC] Inside the block another task acquires, records the held state (true) and
        # releases; the recorded list is [true].
        """This is the whole point: the waiter lets the work it waits for proceed."""
        lock = ContextLock("test")
        await lock.acquire()
        taken = []

        async with lock.with_released():
            await asyncio.create_task(_acquire_release_and_report(lock, taken))

        assert taken == [True]

    @pytest.mark.asyncio
    async def test_the_block_waits_for_a_concurrent_holder_before_returning(self):
        # [FUNCTION DOC] Inside the block a holder task takes the lock (locked, not held by us),
        # then finishes; after leaving the block the lock is held in this context again.
        """Re-acquiring must queue behind whoever took the lock, not steal it."""
        lock = ContextLock("test")
        await lock.acquire()
        holder = _HolderTask(lock)

        async with lock.with_released():
            await holder.start()
            assert lock.locked and not lock.held_in_current_context
            await holder.finish()

        assert lock.held_in_current_context

    @pytest.mark.asyncio
    async def test_with_released_waits_to_reacquire_until_a_concurrent_holder_releases(self):
        # [FUNCTION DOC] A task enters the released block and waits on a gate; the lock is free. A
        # holder takes the lock and the gate opens: the block task stays pending after five yields
        # (it must reacquire). Once the holder finishes the block task completes returning true.
        """Leaving the block must block while someone else still holds the lock, not return unheld."""
        lock = ContextLock("test")
        await lock.acquire()
        block = asyncio.create_task(_leave_the_released_block_after(lock, gate := asyncio.Event()))
        for _ in range(5):
            await asyncio.sleep(0)
        assert not lock.locked

        holder = _HolderTask(lock)
        await holder.start()
        gate.set()
        for _ in range(5):
            await asyncio.sleep(0)
        assert not block.done()

        await holder.finish()
        assert await block is True

    @pytest.mark.asyncio
    async def test_a_raising_block_still_gets_the_lock_back(self):
        # [FUNCTION DOC] A runtime error "boom" raised inside the block propagates, and afterwards
        # the lock is held in this context again.
        """The caller's own lock discipline continues after the failure is handled."""
        lock = ContextLock("test")
        await lock.acquire()

        with pytest.raises(RuntimeError, match="boom"):
            async with lock.with_released():
                raise RuntimeError("boom")

        assert lock.held_in_current_context

    @pytest.mark.asyncio
    async def test_a_caller_that_does_not_hold_the_lock_is_rejected(self):
        # [FUNCTION DOC] Entering the released block without holding the lock raises an assertion
        # error matching "must be held".
        """Releasing a lock one does not hold would open someone else's critical section."""
        lock = ContextLock("test")

        with pytest.raises(AssertionError, match="must be held"):
            async with lock.with_released():
                pass

    @pytest.mark.asyncio
    async def test_lock_requiring_helpers_are_rejected_inside_the_block(self):
        # [FUNCTION DOC] Inside the released block, calling a lock-requiring helper on the guarded
        # double raises an assertion error matching "must be called with".
        """Inside the block the invariants the lock guards no longer hold."""
        guarded = _Guarded()
        await guarded.context_lock.acquire()

        async with guarded.context_lock.with_released():
            with pytest.raises(AssertionError, match="must be called with"):
                guarded._private_method()


class TestCancelledReacquire:
    # [CLASS DOC] Tests that cancelling a task while it waits to reacquire after a released section
    # keeps the cancellation (not an assertion error), leaves the lock consistent, and does not
    # steal the lock from its new holder.
    @pytest.mark.asyncio
    async def test_the_cancellation_reaches_the_caller_unchanged(self):
        # [FUNCTION DOC] After the cancelled-reacquire scenario, the section task reports cancelled;
        # the holder is then finished.
        """An AssertionError from the outer release would replace the caller's own CancelledError."""
        lock = ContextLock("test")

        section, holder = await _cancel_while_reacquiring(lock)

        assert section.cancelled()
        await holder.finish()

    @pytest.mark.asyncio
    async def test_a_cancelled_reacquire_leaves_the_lock_free(self):
        # [FUNCTION DOC] After the scenario and the holder finishing, a fresh acquire succeeds and the
        # lock is held in this context.
        """A half-acquired lock would deadlock every later waiter on a lock nobody can release."""
        lock = ContextLock("test")

        _, holder = await _cancel_while_reacquiring(lock)
        await holder.finish()

        await lock.acquire()

        assert lock.held_in_current_context

    @pytest.mark.asyncio
    async def test_a_cancelled_reacquire_does_not_take_the_lock_away_from_its_new_holder(self):
        # [FUNCTION DOC] Right after the scenario the lock is still locked (by the holder); after the
        # holder finishes it is not locked.
        """Releasing the underlying lock on the way out would hand a second context the same section."""
        lock = ContextLock("test")

        _, holder = await _cancel_while_reacquiring(lock)

        assert lock.locked
        await holder.finish()
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_a_locked_method_cancelled_while_reacquiring_stays_cancelled(self):
        # [FUNCTION DOC] For a with_lock method that waits in a released section, the cancelled call
        # task reports cancelled, the lock ends free, and a later locked call works (returns true).
        """A with_lock body that gave the lock away must still report cancellation, not a release assertion."""
        guarded = _Guarded()

        call = await _cancel_a_reacquiring_call(
            lock=guarded.context_lock, call=guarded.locked_method_waiting_in_a_released_section
        )

        assert call.cancelled()
        assert not guarded.context_lock.locked
        assert await guarded.locked_method() is True

    @pytest.mark.asyncio
    async def test_a_start_call_cancelled_while_reacquiring_stays_cancelled(self):
        # [FUNCTION DOC] Same as above for a window-start method: the call reports cancelled and the
        # lock ends free.
        """An acquires_lock body that gave the lock away must still report cancellation to its caller."""
        guarded = _Guarded()

        call = await _cancel_a_reacquiring_call(
            lock=guarded.context_lock, call=guarded.start_window_waiting_in_a_released_section
        )

        assert call.cancelled()
        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_the_window_can_be_opened_again_after_a_start_call_is_cancelled_while_reacquiring(self):
        # [FUNCTION DOC] After a cancelled window-start, a normal start then end pair succeeds and
        # leaves the lock free.
        """The controller must keep working after a caller abandons a released section."""
        guarded = _Guarded()

        await _cancel_a_reacquiring_call(
            lock=guarded.context_lock, call=guarded.start_window_waiting_in_a_released_section
        )

        await guarded.start_window()
        await guarded.end_window()

        assert not guarded.context_lock.locked


class TestAcquiresAndReleasesLock:
    # [CLASS DOC] Tests of the cross-call window decorators: acquires_lock keeps the lock locked
    # (detached) after returning, releases_lock reattaches and releases; includes failure,
    # cancellation, misuse and multi-task cases.
    @pytest.mark.asyncio
    async def test_the_lock_stays_locked_between_the_start_and_end_calls(self):
        # [FUNCTION DOC] After the start call the lock is locked; after the end call it is not.
        """acquires_lock opens a cross-call window that releases_lock closes."""
        guarded = _Guarded()
        await guarded.start_window()
        assert guarded.context_lock.locked

        await guarded.end_window()
        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_the_lock_is_held_inside_both_the_start_and_end_bodies(self):
        # [FUNCTION DOC] Both window bodies call the lock-requiring property successfully, each
        # returning 42.
        """Both window methods may call lock-requiring helpers in their bodies."""
        guarded = _Guarded()
        assert await guarded.start_window() == 42
        assert await guarded.end_window() == 42

    @pytest.mark.asyncio
    async def test_the_lock_is_not_held_by_the_caller_between_the_two_calls(self):
        # [FUNCTION DOC] Between start and end the caller's context does not hold the lock; the
        # window is then closed.
        """The window belongs to the lock, not to whoever happens to call end."""
        guarded = _Guarded()
        await guarded.start_window()
        assert not guarded.context_lock.held_in_current_context

        await guarded.end_window()

    @pytest.mark.asyncio
    async def test_lock_requiring_helpers_are_rejected_between_the_two_calls(self):
        # [FUNCTION DOC] Between start and end, calling a lock-requiring helper raises an assertion
        # error matching "must be called with"; the window is then closed.
        """An open window is not an invitation to touch guarded state from outside."""
        guarded = _Guarded()
        await guarded.start_window()
        with pytest.raises(AssertionError, match="must be called with"):
            guarded._private_method()

        await guarded.end_window()

    @pytest.mark.asyncio
    async def test_the_lock_is_released_when_the_start_call_raises(self):
        # [FUNCTION DOC] A start whose body raises "boom" propagates the error and leaves the lock
        # free.
        """A failed start must not leave the lock stuck forever."""
        guarded = _Guarded()
        with pytest.raises(RuntimeError, match="boom"):
            await guarded.start_window_that_raises()
        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_a_failed_start_leaves_the_lock_acquirable_again(self):
        # [FUNCTION DOC] After a failed start, a normal start and end pair works and the lock ends
        # free.
        """After the failure another caller may open a window normally."""
        guarded = _Guarded()
        with pytest.raises(RuntimeError, match="boom"):
            await guarded.start_window_that_raises()

        await guarded.start_window()
        await guarded.end_window()
        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_cancelling_the_start_call_releases_the_lock(self):
        # [FUNCTION DOC] A start whose body sleeps 1 s is run as a task, five yields later the lock is
        # locked; cancelling it frees the lock, and a normal start and end pair then works.
        """Cancellation is not an Exception, but an abandoned start must still hand the lock back."""
        guarded = _Guarded()
        call = asyncio.create_task(guarded.start_window_that_waits(delay=1))
        for _ in range(5):
            await asyncio.sleep(0)
        assert guarded.context_lock.locked

        await _cancel(call)

        assert not guarded.context_lock.locked
        await guarded.start_window()
        await guarded.end_window()
        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_a_start_call_that_handed_the_lock_back_reports_its_own_error(self):
        # [FUNCTION DOC] A start whose body releases the lock and then raises "boom": the runtime
        # error "boom" is what propagates, and the lock is free.
        """A start that fails after giving the lock away must surface its error, not a release assertion."""
        guarded = _Guarded()

        with pytest.raises(RuntimeError, match="boom"):
            await guarded.start_window_that_hands_the_lock_back_before_raising()

        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_a_start_call_that_handed_the_lock_back_logs_the_skipped_release(self, caplog):
        # [FUNCTION DOC] Same scenario with error-level capture: the only skipped-release message is
        # exactly "Lock 'guarded' is not held by the current context on exception exit; skipping
        # release".
        """The operator needs to see which lock the failed start could no longer release."""
        guarded = _Guarded()

        with caplog.at_level(logging.ERROR, logger="miles.utils.context_lock"):
            with pytest.raises(RuntimeError, match="boom"):
                await guarded.start_window_that_hands_the_lock_back_before_raising()

        assert _skipped_release_messages(caplog) == [
            "Lock 'guarded' is not held by the current context on exception exit; skipping release"
        ]

    @pytest.mark.asyncio
    async def test_a_raising_end_call_still_releases_the_lock(self):
        # [FUNCTION DOC] After a normal start, an end whose body raises "boom" propagates the error
        # and the lock ends free.
        """A failure while closing the window must not leave the window open forever."""
        guarded = _Guarded()
        await guarded.start_window()

        with pytest.raises(RuntimeError, match="boom"):
            await guarded.end_window_that_raises()

        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_the_end_call_asserts_without_a_matching_start(self):
        # [FUNCTION DOC] Calling end with no start raises an assertion error matching "was not
        # detached".
        """releases_lock without an open window is a bug."""
        guarded = _Guarded()
        with pytest.raises(AssertionError, match="was not detached"):
            await guarded.end_window()

    @pytest.mark.asyncio
    async def test_concurrent_locked_calls_block_during_the_window(self):
        # [FUNCTION DOC] With a window open, a locked call as a task is still pending after five
        # yields; after the end call it completes returning true.
        """with_lock methods wait out an open window instead of interleaving with it."""
        guarded = _Guarded()
        await guarded.start_window()
        blocked_call = asyncio.create_task(guarded.locked_method())
        for _ in range(5):
            await asyncio.sleep(0)
        assert not blocked_call.done()

        await guarded.end_window()
        assert await blocked_call is True

    @pytest.mark.asyncio
    async def test_the_window_can_be_opened_and_closed_from_different_tasks(self):
        # [FUNCTION DOC] Start and end are each run in their own new task; the lock is locked after
        # the first and free after the second.
        """start and end are separate scheduler invocations in the real caller."""
        guarded = _Guarded()
        await asyncio.create_task(guarded.start_window())
        assert guarded.context_lock.locked

        await asyncio.create_task(guarded.end_window())
        assert not guarded.context_lock.locked

    def test_rejects_sync_functions_at_decoration_time(self):
        # [FUNCTION DOC] Decorating plain functions with the start or end window decorator raises an
        # assertion error matching "must be async".
        """Both window decorators need to await the lock, so sync functions are refused."""
        with pytest.raises(AssertionError, match="must be async"):

            @acquires_lock
            def sync_start(self) -> None:
                # [FUNCTION DOC] HELPER: Dummy sync function that must be rejected.
                pass

        with pytest.raises(AssertionError, match="must be async"):

            @releases_lock
            def sync_end(self) -> None:
                # [FUNCTION DOC] HELPER: Dummy sync function that must be rejected.
                pass


class TestRequiresLock:
    # [CLASS DOC] Tests of the requires_lock decorator: passes only when the current context holds
    # the very lock object stored on the instance; works for sync and async methods and properties;
    # asserts before running an async body; reports a missing lock attribute.
    @pytest.mark.asyncio
    async def test_passes_when_called_from_a_lock_holding_method(self):
        # [FUNCTION DOC] A locked method calling the sync lock-requiring helper returns true.
        """Private helpers run fine inside a with_lock caller."""
        guarded = _Guarded()
        assert await guarded.locked_method_calling_private() is True

    @pytest.mark.asyncio
    async def test_passes_inside_an_explicit_lock_context(self):
        # [FUNCTION DOC] Inside async-with on the instance's lock, the async helper, the sync helper
        # and the property (42) all work.
        """Holding the lock via async with also satisfies the requirement."""
        guarded = _Guarded()
        async with guarded.context_lock:
            assert await guarded.async_private_method() is True
            assert guarded._private_method() is True
            assert guarded.guarded_value == 42

    @pytest.mark.asyncio
    async def test_passes_in_tasks_fanned_out_from_inside_the_lock(self):
        # [FUNCTION DOC] A locked method that gathers two async helper calls returns [true, true].
        """asyncio.gather from a locked method must not trip the check in its children."""
        guarded = _Guarded()
        assert await guarded.locked_method_fanning_out() == [True, True]

    @pytest.mark.asyncio
    async def test_raises_when_no_lock_is_held(self):
        # [FUNCTION DOC] Calling the sync helper without holding the lock raises an assertion error
        # matching "must be called with the 'guarded' context lock held".
        """Calling a lock-requiring method outside the lock is rejected."""
        guarded = _Guarded()
        with pytest.raises(AssertionError, match="must be called with the 'guarded' context lock held"):
            guarded._private_method()

    @pytest.mark.asyncio
    async def test_raises_for_async_methods_before_running_the_body(self):
        # [FUNCTION DOC] Awaiting the async helper without the lock raises an assertion error
        # matching "must be called with", and the body's call counter remains 0.
        """The decorator asserts before awaiting async bodies too, so the body leaves no trace."""
        guarded = _Guarded()
        with pytest.raises(AssertionError, match="must be called with"):
            await guarded.async_private_method()
        assert guarded.async_private_method_calls == 0

    @pytest.mark.asyncio
    async def test_raises_for_property_access_outside_the_lock(self):
        # [FUNCTION DOC] Reading the guarded property without the lock raises an assertion error
        # matching "must be called with".
        """Guarded snapshots must not be readable without the lock."""
        guarded = _Guarded()
        with pytest.raises(AssertionError, match="must be called with"):
            _ = guarded.guarded_value

    @pytest.mark.asyncio
    async def test_raises_when_a_different_lock_is_held(self):
        # [FUNCTION DOC] Holding an unrelated lock named "unrelated" still raises the "must be called
        # with the 'guarded' context lock held" assertion.
        """Holding some unrelated lock does not authorize touching this object."""
        guarded = _Guarded()
        async with ContextLock("unrelated"):
            with pytest.raises(AssertionError, match="must be called with the 'guarded' context lock held"):
                guarded._private_method()

    @pytest.mark.asyncio
    async def test_passes_when_a_collaborator_shares_the_very_same_lock(self):
        # [FUNCTION DOC] A collaborator instance given the controller's lock object passes the check
        # while the controller's lock is held.
        """Collaborators guarded by one controller are handed that controller's lock object."""
        controller = _Guarded()
        collaborator = _Guarded()
        collaborator.context_lock = controller.context_lock

        async with controller.context_lock:
            assert collaborator._private_method() is True

    @pytest.mark.asyncio
    async def test_raises_when_a_collaborator_holds_a_look_alike_lock(self):
        # [FUNCTION DOC] A collaborator with its own separate lock (same name) fails the check while
        # the controller's lock is held: identity matters, not the name.
        """A separate lock object with the same name is still the wrong lock."""
        controller = _Guarded()
        collaborator = _Guarded()

        async with controller.context_lock:
            with pytest.raises(AssertionError, match="must be called with"):
                collaborator._private_method()

    @pytest.mark.asyncio
    async def test_raises_when_the_lock_is_held_by_another_task(self):
        # [FUNCTION DOC] With a holder task holding the instance's lock, calling the helper from the
        # test context raises "must be called with"; the holder is then finished.
        """held-by matters: someone else holding the lock does not authorize this context."""
        guarded = _Guarded()
        holder = _HolderTask(guarded.context_lock)
        await holder.start()
        with pytest.raises(AssertionError, match="must be called with"):
            guarded._private_method()
        await holder.finish()

    @pytest.mark.asyncio
    async def test_the_requirement_lapses_again_after_the_holder_returns(self):
        # [FUNCTION DOC] After a locked method that used the helper returns, calling the helper
        # directly raises "must be called with" again.
        """The requirement is scoped to the critical section, not sticky afterwards."""
        guarded = _Guarded()
        assert await guarded.locked_method_calling_private() is True
        with pytest.raises(AssertionError, match="must be called with"):
            guarded._private_method()

    @pytest.mark.asyncio
    async def test_raises_in_a_child_task_that_outlives_the_critical_section(self):
        # [FUNCTION DOC] A child task spawned inside async-with waits for a resume event. After the
        # section ends and a holder task takes the lock, resuming the child makes it record that it
        # does not hold the lock (false) and its helper call raises the "must be called with the
        # 'guarded' context lock held" assertion.
        """A task spawned inside the section loses its authorization once the holder releases and someone else acquires."""
        guarded = _Guarded()
        resume = asyncio.Event()
        child_saw_held: list[bool] = []

        async def child() -> None:
            # [FUNCTION DOC] HELPER: Waits for resume, records the held state, then calls the helper.
            await resume.wait()
            child_saw_held.append(guarded.context_lock.held_in_current_context)
            guarded._private_method()

        async with guarded.context_lock:
            child_task = asyncio.create_task(child())
            await asyncio.sleep(0)

        holder = _HolderTask(guarded.context_lock)
        await holder.start()
        resume.set()
        with pytest.raises(AssertionError, match="must be called with the 'guarded' context lock held"):
            await child_task
        assert child_saw_held == [False]
        await holder.finish()

    @pytest.mark.asyncio
    async def test_a_child_task_that_outlives_the_section_can_still_acquire_the_lock(self):
        # [FUNCTION DOC] A child spawned inside the section resumes after it ended, acquires the lock
        # itself with async-with (not blocked by its stale grant) and calls the helper successfully;
        # afterwards the lock is free.
        """A stale authorization must not be mistaken for a live one and block a fresh acquire."""
        guarded = _Guarded()
        resume = asyncio.Event()

        async def child() -> None:
            # [FUNCTION DOC] HELPER: Waits for resume, acquires the lock via async-with and asserts
            # the helper returns true.
            await resume.wait()
            async with guarded.context_lock:
                assert guarded._private_method() is True

        async with guarded.context_lock:
            child_task = asyncio.create_task(child())
            await asyncio.sleep(0)

        resume.set()
        await child_task
        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_reports_a_missing_lock_attribute(self):
        # [FUNCTION DOC] A lock-requiring method on an object without the lock attribute raises an
        # attribute error mentioning "context_lock".
        """A collaborator that never got handed the lock fails loudly."""

        @enforce_lock_discipline
        class _NoLock:
            # [CLASS DOC] HELPER: Disciplined class with a lock-requiring method but no lock.
            @requires_lock
            def method(self) -> None:
                # [FUNCTION DOC] HELPER: No-op lock-requiring method.
                pass

        with pytest.raises(AttributeError, match="context_lock"):
            _NoLock().method()


class TestEnforceLockDiscipline:
    # [CLASS DOC] Tests of the class decorator enforce_lock_discipline, which at class creation
    # asserts that every method (sync, async, private, dunder, property getter/setter/deleter,
    # static and class methods) carries a lock decorator or the exemption marker, and which makes
    # decorated methods verify at call time that their owning class was enforced (including shared
    # and wrapped methods). The many small classes defined inside tests are throwaway fixtures.
    def test_rejects_an_undecorated_async_method(self):
        # [FUNCTION DOC] A class with an undecorated async method fails at creation with an assertion
        # error matching "_Bad.method must be decorated".
        """Every method must opt into a lock discipline explicitly."""
        with pytest.raises(AssertionError, match="_Bad.method must be decorated"):

            @enforce_lock_discipline
            class _Bad:
                # [CLASS DOC] HELPER: Invalid class with an undecorated async method.
                async def method(self) -> None:
                    # [FUNCTION DOC] HELPER: Undecorated no-op async method.
                    pass

    def test_rejects_an_undecorated_sync_method(self):
        # [FUNCTION DOC] An undecorated sync method is rejected too: assertion error matching
        # "_BadSync.method must be decorated".
        """Sync methods are checked, not just coroutines."""
        with pytest.raises(AssertionError, match="_BadSync.method must be decorated"):

            @enforce_lock_discipline
            class _BadSync:
                # [CLASS DOC] HELPER: Invalid class with an undecorated sync method.
                def method(self) -> None:
                    # [FUNCTION DOC] HELPER: Undecorated no-op sync method.
                    pass

    def test_rejects_undecorated_private_methods(self):
        # [FUNCTION DOC] An undecorated private (underscore) method is rejected: assertion error
        # matching "_BadPrivate._method must be decorated".
        """Private methods must be disciplined too, not just the public surface."""
        with pytest.raises(AssertionError, match="_BadPrivate._method must be decorated"):

            @enforce_lock_discipline
            class _BadPrivate:
                # [CLASS DOC] HELPER: Invalid class with an undecorated private method.
                def _method(self) -> None:
                    # [FUNCTION DOC] HELPER: Undecorated no-op private method.
                    pass

    def test_rejects_an_undecorated_dunder_method(self):
        # [FUNCTION DOC] A hand-written undecorated constructor is rejected: assertion error matching
        # "_BadInit.__init__ must be decorated".
        """Dunders are methods too: a hand-written __init__ must state its discipline."""
        with pytest.raises(AssertionError, match=r"_BadInit.__init__ must be decorated"):

            @enforce_lock_discipline
            class _BadInit:
                # [CLASS DOC] HELPER: Invalid class with an undecorated constructor.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Undecorated no-op constructor.
                    pass

    def test_rejects_an_undecorated_dunder_other_than_init(self):
        # [FUNCTION DOC] An undecorated representation dunder is rejected: assertion error matching
        # "_BadRepr.__repr__ must be decorated".
        """Any dunder can touch guarded state, so none of them are waved through."""
        with pytest.raises(AssertionError, match=r"_BadRepr.__repr__ must be decorated"):

            @enforce_lock_discipline
            class _BadRepr:
                # [CLASS DOC] HELPER: Invalid class with an undecorated representation method.
                def __repr__(self) -> str:
                    # [FUNCTION DOC] HELPER: Undecorated method returning the text "bad".
                    return "bad"

    def test_rejects_an_undecorated_property_getter(self):
        # [FUNCTION DOC] An undecorated property getter is rejected: assertion error matching
        # "_BadProperty.value must be decorated".
        """Property getters are methods too and must be disciplined."""
        with pytest.raises(AssertionError, match="_BadProperty.value must be decorated"):

            @enforce_lock_discipline
            class _BadProperty:
                # [CLASS DOC] HELPER: Invalid class with an undecorated property getter.
                @property
                def value(self) -> int:
                    # [FUNCTION DOC] HELPER: Undecorated getter returning 0.
                    return 0

    def test_rejects_an_undecorated_property_setter(self):
        # [FUNCTION DOC] A property whose getter is exempt but whose setter is undecorated is
        # rejected: assertion error matching "_BadSetter.value must be decorated".
        """A disciplined getter does not excuse an undisciplined setter."""
        with pytest.raises(AssertionError, match="_BadSetter.value must be decorated"):

            @enforce_lock_discipline
            class _BadSetter:
                # [CLASS DOC] HELPER: Invalid class with an exempt getter and undecorated setter.
                @property
                @lock_exempt
                def value(self) -> int:
                    # [FUNCTION DOC] HELPER: Exempt getter returning 0.
                    return 0

                @value.setter
                def value(self, new_value: int) -> None:
                    # [FUNCTION DOC] HELPER: Undecorated no-op setter.
                    pass

    def test_rejects_an_undecorated_property_deleter(self):
        # [FUNCTION DOC] A property whose getter is exempt but whose deleter is undecorated is
        # rejected: assertion error matching "_BadDeleter.value must be decorated".
        """A disciplined getter does not excuse an undisciplined deleter either."""
        with pytest.raises(AssertionError, match="_BadDeleter.value must be decorated"):

            @enforce_lock_discipline
            class _BadDeleter:
                # [CLASS DOC] HELPER: Invalid class with an exempt getter and undecorated deleter.
                @property
                @lock_exempt
                def value(self) -> int:
                    # [FUNCTION DOC] HELPER: Exempt getter returning 0.
                    return 0

                @value.deleter
                def value(self) -> None:
                    # [FUNCTION DOC] HELPER: Undecorated no-op deleter.
                    pass

    def test_rejects_an_undecorated_staticmethod(self):
        # [FUNCTION DOC] An undecorated static method is rejected: assertion error matching
        # "_BadStatic.method must be decorated".
        """Static methods are unwrapped and checked like the rest."""
        with pytest.raises(AssertionError, match="_BadStatic.method must be decorated"):

            @enforce_lock_discipline
            class _BadStatic:
                # [CLASS DOC] HELPER: Invalid class with an undecorated static method.
                @staticmethod
                def method() -> None:
                    # [FUNCTION DOC] HELPER: Undecorated no-op static method.
                    pass

    def test_rejects_an_undecorated_classmethod(self):
        # [FUNCTION DOC] An undecorated class method is rejected: assertion error matching
        # "_BadClassmethod.method must be decorated".
        """Class methods are unwrapped and checked like the rest."""
        with pytest.raises(AssertionError, match="_BadClassmethod.method must be decorated"):

            @enforce_lock_discipline
            class _BadClassmethod:
                # [CLASS DOC] HELPER: Invalid class with an undecorated class method.
                @classmethod
                def method(cls) -> None:
                    # [FUNCTION DOC] HELPER: Undecorated no-op class method.
                    pass

    def test_accepts_a_fully_disciplined_class(self):
        # [FUNCTION DOC] A class mixing exempt constructor, exempt static and class methods, a locked
        # async method and an exempt property passes the check; the property returns 0.
        """A class whose members are all decorated or exempt passes the check."""

        @enforce_lock_discipline
        class _Good:
            # [CLASS DOC] HELPER: Valid class using every kind of discipline marker.
            @lock_exempt
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Exempt constructor creating a lock named "good".
                self.context_lock = ContextLock("good")

            @staticmethod
            @lock_exempt
            def create() -> None:
                # [FUNCTION DOC] HELPER: Exempt static no-op.
                pass

            @classmethod
            @lock_exempt
            def build(cls) -> None:
                # [FUNCTION DOC] HELPER: Exempt class method no-op.
                pass

            @with_lock
            async def method(self) -> None:
                # [FUNCTION DOC] HELPER: Locked no-op async method.
                pass

            @property
            @lock_exempt
            def value(self) -> int:
                # [FUNCTION DOC] HELPER: Exempt property returning 0.
                return 0

        assert _Good().value == 0

    def test_accepts_a_dataclass_when_the_check_runs_before_field_generation(self):
        # [FUNCTION DOC] With the dataclass decorator outermost and the enforcement inside it, the
        # generated dunders are not checked and an exempt method works (returns the count 1).
        """Dataclass-generated dunders cannot be decorated, so the check must run inside the dataclass."""

        @dataclasses.dataclass
        @enforce_lock_discipline
        class _GoodDataclass:
            # [CLASS DOC] HELPER: Dataclass with an integer count (default 0), a name (default
            # "default") and one exempt method.
            count: int = 0
            name: str = "default"

            @lock_exempt
            def method(self) -> int:
                # [FUNCTION DOC] HELPER: Exempt method returning the count.
                return self.count

        assert _GoodDataclass(count=1).method() == 1

    def test_rejects_a_dataclass_whose_generated_dunders_get_checked(self):
        # [FUNCTION DOC] With enforcement outermost, the dataclass-generated methods are seen and
        # rejected: assertion error matching "_BadOrder.* must be decorated".
        """Applying the check outside @dataclass sees generated methods it cannot possibly discipline."""
        with pytest.raises(AssertionError, match="_BadOrder.* must be decorated"):

            @enforce_lock_discipline
            @dataclasses.dataclass
            class _BadOrder:
                # [CLASS DOC] HELPER: Dataclass with a single integer field, used to trigger the
                # rejection of generated methods.
                count: int = 0

    def test_accepts_plain_and_annotated_class_attributes(self):
        # [FUNCTION DOC] A class with only data attributes (annotated and plain) passes; the class
        # attribute keeps its value 30.
        """Only real methods are checked; data attributes and their annotations are not."""

        @enforce_lock_discipline
        class _GoodAttributes:
            # [CLASS DOC] HELPER: Class with data attributes only (timeout 30, retries 3).
            timeout_seconds: int = 30
            retries = 3

        assert _GoodAttributes.timeout_seconds == 30

    def test_accepts_a_class_whose_only_members_are_inherited(self):
        # [FUNCTION DOC] A subclass of the guarded test double adding nothing passes the check and can
        # be instantiated.
        """A subclass adding nothing has nothing of its own to check."""

        @enforce_lock_discipline
        class _Derived(_Guarded):
            # [CLASS DOC] HELPER: Empty subclass of the guarded test double.
            pass

        assert _Derived() is not None

    def test_the_guarded_test_double_passes_the_check(self):
        # [FUNCTION DOC] Applying the enforcement again to the shared guarded double returns the same
        # class object (it is fully disciplined).
        """The _Guarded helper used across this file is itself fully disciplined."""
        assert enforce_lock_discipline(_Guarded) is _Guarded

    @pytest.mark.asyncio
    async def test_a_with_lock_method_on_an_undecorated_class_is_rejected(self):
        # [FUNCTION DOC] Calling a locked method on a class never decorated with enforcement raises
        # an assertion error matching "not decorated with @enforce_lock_discipline".
        """Decorating methods but forgetting the class decorator is exactly the missed check to catch."""

        class _Forgotten:
            # [CLASS DOC] HELPER: Class with a locked method but without the class decorator.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a lock named "forgotten".
                self.context_lock = ContextLock("forgotten")

            @with_lock
            async def method(self) -> None:
                # [FUNCTION DOC] HELPER: Locked no-op async method.
                pass

        with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
            await _Forgotten().method()

    @pytest.mark.asyncio
    async def test_a_requires_lock_method_on_an_undecorated_class_is_rejected_inside_the_lock(self):
        # [FUNCTION DOC] Even while holding the instance's lock, calling a lock-requiring method on
        # an unenforced class raises "not decorated with @enforce_lock_discipline".
        """The missing class decorator is reported even where the lock check itself would pass."""

        class _Forgotten:
            # [CLASS DOC] HELPER: Class with a lock-requiring method but without the class decorator.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a lock named "forgotten".
                self.context_lock = ContextLock("forgotten")

            @requires_lock
            def method(self) -> None:
                # [FUNCTION DOC] HELPER: Lock-requiring no-op method.
                pass

        forgotten = _Forgotten()
        async with forgotten.context_lock:
            with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
                forgotten.method()

    @pytest.mark.asyncio
    async def test_window_decorators_on_an_undecorated_class_are_rejected(self):
        # [FUNCTION DOC] Both window start and end calls on an unenforced class raise "not decorated
        # with @enforce_lock_discipline".
        """Cross-call window decorators demand the class-level check as well."""

        class _Forgotten:
            # [CLASS DOC] HELPER: Class with window start and end methods but no class decorator.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a lock named "forgotten".
                self.context_lock = ContextLock("forgotten")

            @acquires_lock
            async def start(self) -> None:
                # [FUNCTION DOC] HELPER: Window start no-op.
                pass

            @releases_lock
            async def end(self) -> None:
                # [FUNCTION DOC] HELPER: Window end no-op.
                pass

        forgotten = _Forgotten()
        with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
            await forgotten.start()
        with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
            await forgotten.end()

    @pytest.mark.asyncio
    async def test_an_undecorated_subclass_may_still_call_inherited_methods(self):
        # [FUNCTION DOC] An unenforced subclass of the guarded double can call an inherited locked
        # method (returns true), since the check is per declaring class.
        """The check applies to the class that declared the method, so inheriting is fine."""

        class _Derived(_Guarded):
            # [CLASS DOC] HELPER: Empty, unenforced subclass of the guarded double.
            pass

        assert await _Derived().locked_method() is True

    @pytest.mark.asyncio
    async def test_a_subclass_adding_its_own_decorated_method_needs_its_own_check(self):
        # [FUNCTION DOC] An unenforced subclass adding a locked method: inherited methods still work
        # (true) but the new method raises an assertion error matching "_Derived.extra_method uses a
        # context-lock decorator".
        """An inherited flag must not excuse methods newly decorated on the subclass."""

        class _Derived(_Guarded):
            # [CLASS DOC] HELPER: Unenforced subclass adding one locked method.
            @with_lock
            async def extra_method(self) -> None:
                # [FUNCTION DOC] HELPER: Locked no-op async method.
                pass

        derived = _Derived()
        assert await derived.locked_method() is True
        with pytest.raises(AssertionError, match="_Derived.extra_method uses a context-lock decorator"):
            await derived.extra_method()

    @pytest.mark.asyncio
    async def test_a_guarded_classmethod_on_an_undecorated_class_is_rejected(self):
        # [FUNCTION DOC] A locked class method on an unenforced class (lock stored on the class)
        # raises "not decorated with @enforce_lock_discipline" when called on the class.
        """Looking the owner up from the first argument silently let this shape through before."""

        class _Forgotten:
            # [CLASS DOC] HELPER: Unenforced class with a class-level lock and a locked class method.
            context_lock = ContextLock("forgotten")

            @classmethod
            @with_lock
            async def method(cls) -> None:
                # [FUNCTION DOC] HELPER: Locked no-op class method.
                pass

        with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
            await _Forgotten.method()

    @pytest.mark.asyncio
    async def test_a_guarded_method_wrapped_by_another_decorator_is_still_checked(self):
        # [FUNCTION DOC] A locked method additionally wrapped by a pass-through decorator that copies
        # metadata, on an unenforced class, still raises "not decorated with
        # @enforce_lock_discipline".
        """functools.wraps copies the discipline marker, so the outer wrapper must not become invisible."""

        def _passthrough(fn):
            # [FUNCTION DOC] HELPER: Decorator factory returning an async wrapper (metadata copied
            # from the wrapped function) that simply awaits the wrapped function with the same
            # arguments and returns its result.
            @functools.wraps(fn)
            async def outer(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards the call to the wrapped function and returns its
                # result.
                return await fn(self, *args, **kwargs)

            return outer

        class _Forgotten:
            # [CLASS DOC] HELPER: Unenforced class whose locked method is wrapped by the pass-through.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a lock named "forgotten".
                self.context_lock = ContextLock("forgotten")

            @_passthrough
            @with_lock
            async def method(self) -> None:
                # [FUNCTION DOC] HELPER: Locked no-op async method.
                pass

        with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
            await _Forgotten().method()

    @pytest.mark.asyncio
    async def test_a_guarded_method_wrapped_by_another_decorator_runs_on_an_enforced_class(self):
        # [FUNCTION DOC] Same wrapping on an enforced class works: the call returns true (lock held
        # inside).
        """An outer functools.wraps decorator must not hide the owner marker from the inner lock wrapper."""

        def _passthrough(fn):
            # [FUNCTION DOC] HELPER: Decorator factory returning an async pass-through wrapper (see
            # the identical helper in the previous test).
            @functools.wraps(fn)
            async def outer(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards the call to the wrapped function and returns its
                # result.
                return await fn(self, *args, **kwargs)

            return outer

        @enforce_lock_discipline
        class _Stacked:
            # [CLASS DOC] HELPER: Enforced class whose locked method is wrapped by the pass-through.
            @lock_exempt
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Exempt constructor creating a lock named "stacked".
                self.context_lock = ContextLock("stacked")

            @_passthrough
            @with_lock
            async def method(self) -> bool:
                # [FUNCTION DOC] HELPER: Locked method returning whether the lock is held in context.
                return self.context_lock.held_in_current_context

        assert await _Stacked().method() is True

    @pytest.mark.asyncio
    async def test_a_guarded_method_shared_with_an_unenforced_class_is_still_rejected(self):
        # [FUNCTION DOC] One locked function is attached as a method to an enforced class and to an
        # unenforced class. The enforced one works (true); the unenforced one raises an assertion
        # error matching "was called on a _Forgotten".
        """Stamping the inner wrapper must authorize the enforcing class only, not everyone reusing it."""

        @with_lock
        async def _shared(self) -> bool:
            # [FUNCTION DOC] HELPER: Locked function returning whether the lock is held in context.
            return self.context_lock.held_in_current_context

        @enforce_lock_discipline
        class _Enforced:
            # [CLASS DOC] HELPER: Enforced class reusing the shared locked function as a method.
            @lock_exempt
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Exempt constructor creating a lock named "enforced".
                self.context_lock = ContextLock("enforced")

            method = _shared

        class _Forgotten:
            # [CLASS DOC] HELPER: Unenforced class reusing the shared locked function as a method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a lock named "forgotten".
                self.context_lock = ContextLock("forgotten")

            method = _shared

        assert await _Enforced().method() is True
        with pytest.raises(AssertionError, match="was called on a _Forgotten"):
            await _Forgotten().method()

    @pytest.mark.asyncio
    async def test_a_guarded_method_shared_by_two_enforced_classes_authorizes_both(self):
        # [FUNCTION DOC] One locked function shared by two enforced classes works on both (true each);
        # the second class's registration must not remove the first's.
        """Stamping the second owner must accumulate, not evict the class that claimed the method first."""

        @with_lock
        async def _shared(self) -> bool:
            # [FUNCTION DOC] HELPER: Locked function returning whether the lock is held in context.
            return self.context_lock.held_in_current_context

        @enforce_lock_discipline
        class _FirstOwner:
            # [CLASS DOC] HELPER: First enforced class reusing the shared locked function.
            @lock_exempt
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Exempt constructor creating a lock named "first".
                self.context_lock = ContextLock("first")

            method = _shared

        @enforce_lock_discipline
        class _SecondOwner:
            # [CLASS DOC] HELPER: Second enforced class reusing the shared locked function.
            @lock_exempt
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Exempt constructor creating a lock named "second".
                self.context_lock = ContextLock("second")

            method = _shared

        assert await _FirstOwner().method() is True
        assert await _SecondOwner().method() is True

    @pytest.mark.asyncio
    async def test_a_guarded_classmethod_runs_on_an_enforced_class(self):
        # [FUNCTION DOC] A locked class method on an enforced class (class-level lock) called on the
        # class returns true.
        """A classmethod is called with the class itself, which still belongs to the enforcing class."""

        @enforce_lock_discipline
        class _WithClassmethod:
            # [CLASS DOC] HELPER: Enforced class with a class-level lock and a locked class method.
            context_lock = ContextLock("classmethod")

            @lock_exempt
            def __init__(self) -> None: ...

            @classmethod
            @with_lock
            async def method(cls) -> bool:
                # [FUNCTION DOC] HELPER: Locked class method returning whether the class-level lock
                # is held in context.
                return cls.context_lock.held_in_current_context

        assert await _WithClassmethod.method() is True

    @pytest.mark.asyncio
    async def test_a_guarded_method_wrapping_a_bound_method_is_accepted(self):
        # [FUNCTION DOC] Builds an exempt async source method, binds it to an instance, wraps it with
        # a metadata-copying function (so the wrapped-chain ends at a bound method), and attaches the
        # wrapper to an enforced class; class creation succeeds and calling returns "ok".
        """Walking __wrapped__ onto a bound method must not fail the class definition."""

        class _Source:
            # [CLASS DOC] HELPER: Plain class with one exempt async method.
            @lock_exempt
            async def implementation(self) -> str:
                # [FUNCTION DOC] HELPER: Exempt method returning "ok".
                return "ok"

        bound = _Source().implementation

        @functools.wraps(bound)
        async def _outer(self) -> str:
            # [FUNCTION DOC] HELPER: Wrapper that ignores its own instance and awaits the bound
            # source method, returning its result.
            return await bound()

        @enforce_lock_discipline
        class _Delegating:
            # [CLASS DOC] HELPER: Enforced class using the wrapper as a method.
            @lock_exempt
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Exempt constructor creating a lock named "delegating".
                self.context_lock = ContextLock("delegating")

            method = _outer

        assert await _Delegating().method() == "ok"

    def test_lock_exempt_leaves_the_function_behaviour_untouched(self):
        # [FUNCTION DOC] An exempt function adding two numbers still returns the sum (1 + 2 = 3).
        """The exemption is a marker only; it must not wrap or alter the call."""

        @lock_exempt
        def add(left: int, right: int) -> int:
            # [FUNCTION DOC] HELPER: Returns the sum of the two integers.
            return left + right

        assert add(1, 2) == 3
