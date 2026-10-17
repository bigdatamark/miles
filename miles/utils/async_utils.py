# [MODULE DOC] Helpers for mixing synchronous code with asyncio. Provides a lazily created global
# background event loop thread (with blocking run and non-blocking submit entry points), a
# fan-out future collector, task/gather helpers that cancel peers or surface the first error, a
# helper to await maybe-awaitable values, and an async-context teardown collector (Disposer).
# State: one process-global loop thread, created on first use and never stopped (daemon thread).

import asyncio
import concurrent.futures
import inspect
import logging
import threading
import traceback
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, TypeVar

logger = logging.getLogger(__name__)


__all__ = [
    "get_async_loop",
    "run",
    "submit",
    "wait_futures",
    "wait_cancelling_pending_on_first_completion",
    "eager_create_task",
    "gather_and_raise_first",
    "maybe_await",
    "with_disposer",
    "Disposer",
]

_T = TypeVar("_T")


# Create a background event loop thread
class AsyncLoopThread:
    # [CLASS DOC] Owns a private event loop running forever on a daemon thread, started at
    # construction. Exposes the loop as an attribute, plus submit (non-blocking) and run
    # (blocking) for coroutines from other threads.
    def __init__(self):
        # [FUNCTION DOC] Creates a new event loop and starts the daemon thread that runs it.
        # Side effect: spawns a thread.
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        # [FUNCTION DOC] HELPER: Thread entry point; binds the loop as this thread's current loop
        # and runs it until the process exits.
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro: Coroutine[Any, Any, _T]) -> concurrent.futures.Future[_T]:
        # [FUNCTION DOC] Schedules the coroutine on the background loop in a thread-safe way and
        # returns a standard (thread-blocking) future. Fails an assertion if called from the loop
        # thread itself, since blocking on the result there would deadlock.
        assert (
            threading.current_thread() is not self._thread
        ), "submitting from the loop thread and then blocking on the result would deadlock the loop"
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        # [FUNCTION DOC] Submits the coroutine and blocks the calling thread until it finishes,
        # returning its result or re-raising its exception. Same loop-thread assertion as submit.
        # Schedule a coroutine onto the loop and block until it's done
        return self.submit(coro).result()


# Create one global instance
async_loop = None
_async_loop_init_lock = threading.Lock()


def get_async_loop():
    # [FUNCTION DOC] Returns the process-global background loop thread object, creating it on
    # first call using double-checked locking so concurrent callers create only one. Side effect:
    # sets the module global and may spawn the thread.
    global async_loop
    if async_loop is None:
        with _async_loop_init_lock:
            if async_loop is None:
                async_loop = AsyncLoopThread()
    return async_loop


# TODO: rename these functions and classes
def run(coro: Coroutine[Any, Any, _T]) -> _T:
    # [FUNCTION DOC] Blocking run of the coroutine on the global background loop; returns its
    # result or raises its exception. Must not be called from the loop thread.
    """Run a coroutine in the background event loop."""
    return get_async_loop().run(coro)


def submit(coro: Coroutine[Any, Any, _T]) -> concurrent.futures.Future[_T]:
    # [FUNCTION DOC] Non-blocking scheduling of the coroutine on the global background loop;
    # returns a standard future. Must not be called from the loop thread.
    """Fire a coroutine on the background event loop and return its future."""
    return get_async_loop().submit(coro)


def wait_futures(futures: Sequence[concurrent.futures.Future]) -> list[Any]:
    # [FUNCTION DOC] Waits for each future in order (blocking). A failing future is logged as a
    # warning with its index and traceback, and its result slot is None. After all futures have
    # settled, re-raises the first collected error (by index order); otherwise returns the list of
    # results in input order. Only ordinary exceptions are caught.
    """Collect a fan-out, raising the first error once every future has settled."""
    results: list[Any] = []
    errors: list[Exception] = []
    for index, future in enumerate(futures):
        try:
            results.append(future.result())
        except Exception as e:
            logger.warning(f"wait_futures index={index} failed", exc_info=e)
            results.append(None)
            errors.append(e)

    if errors:
        raise errors[0]
    return results


async def wait_cancelling_pending_on_first_completion(
    tasks: Sequence[asyncio.Task], *, on_first_completion: Callable[[], None] | None = None
) -> None:
    # [FUNCTION DOC] Waits until the first of the given tasks finishes (successfully or not), then
    # calls the optional callback (no arguments), cancels all still-pending tasks and waits for
    # them to settle, swallowing their exceptions. Next it collects the error of every task that
    # was not cancelled and failed, logging each at error level. If any failed, the primary error
    # is the first failure among tasks that were already done at first completion (falling back to
    # the first failure overall); every other failure is attached to the primary as a note holding
    # its formatted traceback (or logged when notes are unsupported), and the primary is raised.
    # Returns nothing when no task failed.
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    if on_first_completion is not None:
        on_first_completion()

    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    task_errors = [(task, error) for task in tasks if (error := _compute_task_error(task)) is not None]
    for _, error in task_errors:
        logger.error("task failed", exc_info=error)
    if task_errors:
        primary_index = next((index for index, (task, _) in enumerate(task_errors) if task in done), 0)
        primary_error = task_errors[primary_index][1]
        for index, (_, error) in enumerate(task_errors):
            if index != primary_index:
                note = "Additional task failure while cancelling peers:\n" + "".join(traceback.format_exception(error))
                _exception_add_note_or_log(primary_error, note)
        raise primary_error


def _exception_add_note_or_log(e: BaseException, msg: str) -> None:
    # [FUNCTION DOC] HELPER: Appends the message as a note on the exception when the interpreter
    # supports exception notes; otherwise logs the message at error level.
    if hasattr(e, "add_note"):
        e.add_note(msg)
    else:
        logger.error(msg)


def _compute_task_error(task: asyncio.Task) -> BaseException | None:
    # [FUNCTION DOC] HELPER: Returns None for a cancelled task, otherwise the task's exception
    # (None if it succeeded). Must only be called on a finished task, otherwise the underlying
    # call raises an invalid-state error.
    if task.cancelled():
        return None
    return task.exception()


async def eager_create_task(coro: Coroutine[object, object, _T]) -> asyncio.Task[_T]:
    # [FUNCTION DOC] Creates a task for the coroutine, then sleeps for zero seconds so the loop
    # runs the task up to its first suspension point before returning the task to the caller.
    """Create a task and yield so it starts executing immediately.

    Unlike bare ``asyncio.create_task``, this ensures the task's first code
    (including any ``.remote()`` calls) runs before the caller continues.
    """
    task = asyncio.create_task(coro)
    await asyncio.sleep(0)
    return task


class AsyncioGatherUtils:
    # [CLASS DOC] Namespace of static helpers for inspecting the output list of a gather call
    # made with exceptions returned instead of raised.
    @staticmethod
    def has_error(outputs):
        # [FUNCTION DOC] Returns true if any element of the outputs is an exception object.
        return any(isinstance(output, BaseException) for output in outputs)

    @staticmethod
    def log_error(
        outputs,
        debug_name: str = "",
        *,
        describe_failure: Callable[[int], str] | None = None,
        log: Callable[..., None] = logger.warning,
    ) -> None:
        # [FUNCTION DOC] For each output that is an exception, calls the log function with a
        # message and the exception as exception info. The message is the failure description
        # callback applied to the index when given, else the debug name followed by "error
        # index=" and the index. The default log function is the module logger's warning method.
        for i, output in enumerate(outputs):
            if isinstance(output, BaseException):
                message = f"{debug_name} error index={i}" if describe_failure is None else describe_failure(i)
                log(message, exc_info=output)


async def gather_and_raise_first(
    awaitables: Sequence[Awaitable[_T]], *, describe_failure: Callable[[int], str] | None = None
) -> list[_T]:
    # [FUNCTION DOC] Runs all awaitables concurrently, collecting exceptions as results so none is
    # abandoned. If a description callback is given, logs each failure at error level using it.
    # Then raises the first exception in input order, or returns the list of results in input
    # order. Without a callback, failures other than the first are silently dropped.
    results = await asyncio.gather(*awaitables, return_exceptions=True)

    if describe_failure is not None:
        AsyncioGatherUtils.log_error(results, describe_failure=describe_failure, log=logger.error)

    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise failures[0]
    return results


async def maybe_await(value: Awaitable[_T] | _T) -> _T:
    # [FUNCTION DOC] Awaits and returns the value if it is awaitable; otherwise returns it as is.
    return await value if inspect.isawaitable(value) else value


async def with_disposer(fn: Callable[..., Awaitable[_T]], *args: Any, **kwargs: Any) -> _T:
    # [FUNCTION DOC] Opens a fresh teardown collector, awaits the async function with the given
    # positional and keyword arguments plus the collector passed under the keyword name
    # "disposer", and returns its result. On exit (normal or error) the registered teardowns run
    # in reverse registration order. The function must accept a disposer keyword.
    async with Disposer() as disposer:
        return await fn(*args, disposer=disposer, **kwargs)


class Disposer:
    # [CLASS DOC] Async context manager that collects teardown callables and runs them in
    # reverse order on exit, backed by an async exit stack.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the empty underlying exit stack.
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "Disposer":
        # [FUNCTION DOC] Enters the underlying stack and returns this disposer.
        await self._stack.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> bool:
        # [FUNCTION DOC] Unwinds the stack, running teardowns in reverse order with the exception
        # information of the block. Returns the stack's verdict on whether to suppress the
        # exception (true only if a teardown suppressed it).
        return await self._stack.__aexit__(exc_type, exc, exc_traceback)

    def add(self, *items: Any) -> None:
        # [FUNCTION DOC] Registers teardowns. None items are skipped. For other items, the
        # teardown is the item's dispose attribute when present, otherwise the item itself; it
        # must be callable (assertion failure otherwise). A coroutine function teardown is
        # registered as an async callback, any other callable as a plain callback (called with no
        # arguments on exit).
        for item in items:
            if item is None:
                continue
            teardown = item.dispose if hasattr(item, "dispose") else item
            assert callable(teardown), teardown
            if inspect.iscoroutinefunction(teardown):
                self._stack.push_async_callback(teardown)
            else:
                self._stack.callback(teardown)
