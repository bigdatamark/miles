# [MODULE DOC] Async retry helpers with exponential backoff. Two strategies: a count-bounded (or
# unbounded) retry that retries on any exception except a designated non-retryable one, and a
# deadline-bounded retry that retries only chosen exception types (plus per-attempt timeouts) with
# jittered backoff and structured logging, until a total time budget is spent.
import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)

_DEFAULT_INITIAL_DELAY = 1.0
_DEFAULT_MAX_DELAY = 60.0
_DEFAULT_BACKOFF_FACTOR = 2.0
_DEFAULT_JITTER_RATIO = 0.1

_T = TypeVar("_T")


class NonRetryableError(Exception):
    # [CLASS DOC] Marker exception: when the function being retried raises it, the plain retry
    # helper re-raises immediately instead of retrying. No extra fields or behavior.
    pass


class AttemptTimeoutError(TimeoutError):
    # [CLASS DOC] Raised by the deadline retry helper when a single attempt exceeded its own
    # per-attempt time budget (as opposed to the whole deadline). It is a subclass of the built-in
    # timeout error and is always treated as retryable by the deadline helper.
    pass


async def retry(
    fn: Callable[[int], Awaitable[_T]],
    *,
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    max_attempts: int | None = None,
) -> _T:
    # [FUNCTION DOC] Repeatedly awaits the async function, passing it the zero-based attempt
    # number, and returns the first successful result. Any exception other than the non-retryable
    # marker counts as a failure: the attempt counter is incremented, then if a maximum number of
    # attempts is set and reached the original exception is re-raised (after a warning log);
    # otherwise a warning with the traceback is logged, the injected sleep function is awaited for
    # the current delay, and the delay is multiplied by the backoff factor, capped at the maximum
    # delay (starting from the initial delay). The non-retryable marker exception propagates
    # immediately without logging. Raises an assertion error up front if max_attempts is given
    # and less than 1. With no max_attempts it loops forever. No jitter is applied. Cancellation
    # is not special-cased (cancelled errors are not ordinary exceptions in modern Python so
    # they propagate).
    """Retry until ``fn`` does not throw, with exponential backoff."""
    assert max_attempts is None or max_attempts >= 1

    attempt = 0
    delay = initial_delay
    while True:
        try:
            return await fn(attempt)
        except NonRetryableError:
            raise
        except Exception:
            attempt += 1
            if max_attempts is not None and attempt >= max_attempts:
                logger.warning(f"retry: attempt {attempt} failed, giving up (max_attempts={max_attempts})")
                raise
            logger.warning(f"retry: attempt {attempt} failed, retrying in {delay:.1f}s", exc_info=True)
            await sleep_fn(delay)
            delay = min(delay * backoff_factor, max_delay)


async def retry_until_deadline(
    fn: Callable[[float], Awaitable[_T]],
    *,
    total_seconds: float,
    retry_on: type[Exception] | tuple[type[Exception], ...],
    attempt_seconds: float | None = None,
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    jitter_ratio: float = _DEFAULT_JITTER_RATIO,
    log_fields: dict[str, Any] | None = None,
) -> _T:
    # [FUNCTION DOC] Retries the async function until it succeeds or a total time budget
    # (total_seconds, measured on the monotonic clock from the call) would be exceeded. The
    # function receives the time budget in seconds for that attempt: the remaining total time, or
    # the smaller of the per-attempt limit and the remaining time when attempt_seconds is given
    # (the per-attempt limit is also enforced by cancelling the attempt, see the run-attempt
    # helper). Only exceptions of the retry_on type(s) (single type or tuple) and the
    # per-attempt timeout error are retried; anything else propagates immediately. After a
    # retryable failure it computes a sleep of the current delay plus a random extra between zero
    # and delay times jitter_ratio; if the remaining time is not larger than that sleep, it logs a
    # warning and re-raises the last exception (so the caller sees the real error, and giving up
    # may happen before the deadline actually passes). Otherwise emits a structured info log
    # (operation name "retry_until_deadline", merged with the optional extra log fields, phase
    # "retrying", attempt number starting at 1, sleep and remaining seconds rounded to
    # milliseconds, and the repr of the error), sleeps with the real asyncio sleep, then grows the
    # delay by the backoff factor capped at the max delay.
    expires_at = time.monotonic() + total_seconds
    retryable = retry_on if isinstance(retry_on, tuple) else (retry_on,)
    delay = initial_delay
    attempt = 0

    while True:
        try:
            return await _run_attempt(
                fn, remaining=max(0.0, expires_at - time.monotonic()), attempt_seconds=attempt_seconds
            )
        except (AttemptTimeoutError, *retryable) as e:
            attempt += 1
            sleep_seconds = delay + random.uniform(0.0, delay * jitter_ratio)
            if expires_at - time.monotonic() <= sleep_seconds:
                logger.warning(f"retry_until_deadline: giving up after {total_seconds:.1f}s", exc_info=True)
                raise
            log_structured(
                logger.info,
                **{"op": "retry_until_deadline", **(log_fields or {})},
                phase="retrying",
                attempt=attempt,
                sleep_s=round(sleep_seconds, 3),
                remaining_s=round(expires_at - time.monotonic(), 3),
                error=repr(e),
            )
            await asyncio.sleep(sleep_seconds)
            delay = min(delay * backoff_factor, max_delay)


async def _run_attempt(fn: Callable[[float], Awaitable[_T]], *, remaining: float, attempt_seconds: float | None) -> _T:
    # [FUNCTION DOC] HELPER: Runs one attempt of the deadline retry. Without a per-attempt limit,
    # awaits the function with the remaining total seconds and returns its result. With a limit,
    # the budget is the smaller of the limit and the remaining time; the function is started as a
    # task with that budget and awaited with a matching timeout. If a timeout error fires and the
    # task was actually cancelled by the timeout, it is converted into the per-attempt timeout
    # error (chained from the original); if the timeout error came from inside the function
    # itself (task not cancelled), the original timeout error is re-raised unchanged, so it can
    # still be matched by the caller's retry_on types.
    if attempt_seconds is None:
        return await fn(remaining)

    budget = min(attempt_seconds, remaining)
    attempt = asyncio.ensure_future(fn(budget))
    try:
        return await asyncio.wait_for(attempt, timeout=budget)
    except (TimeoutError, asyncio.TimeoutError) as e:
        if not attempt.cancelled():
            raise
        raise AttemptTimeoutError(f"one attempt did not answer within {budget:.2f}s") from e
