# [MODULE DOC]
# Tests for the async retry helpers. The retry function calls an async callback with the attempt
# number (from 0), retrying on any exception with exponential backoff (initial delay, factor, cap)
# via an injectable sleep function, up to an optional maximum attempt count (minimum one, default
# unlimited), logging a warning per retry with the delay and exception info and a giving-up warning
# at the end; a non-retryable error aborts at once. The deadline variant retries only listed
# exception types within a total time budget, tells each attempt the remaining time, adds
# jitter (random fraction of the base delay), re-raises the last error (logging a giving-up
# warning) when the next sleep would exceed the budget, logs structured info records per retry
# (merged with caller fields), and can bound each attempt (raising an attempt timeout error that
# is always retryable but distinct from timeouts raised by the attempt itself). All tests are
# asyncio tests; fakes replace sleep and the random module so no real time passes.
import asyncio
import logging

import pytest

from miles.utils import retry_utils
from miles.utils.retry_utils import AttemptTimeoutError, NonRetryableError, retry, retry_until_deadline

pytestmark = pytest.mark.asyncio


class _FakeSleep:
    # [CLASS DOC] HELPER: Awaitable sleep replacement that records requested delays.
    """Records sleep calls without actually sleeping."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Start with an empty delay list.
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        # [FUNCTION DOC] HELPER: Record the delay and return immediately.
        self.delays.append(delay)


class _FakeRandom:
    # [CLASS DOC] HELPER: Random module replacement whose uniform draw returns the upper bound and
    # records the requested ranges.
    """Stands in for the random module and always returns the top of the requested range."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Start with an empty list of requested ranges.
        self.ranges: list[tuple[float, float]] = []

    def uniform(self, low: float, high: float) -> float:
        # [FUNCTION DOC] HELPER: Record the range and return its upper bound.
        self.ranges.append((low, high))
        return high


class TestRetryBasic:
    # [CLASS DOC] Basic tests of the retry function.
    async def test_succeeds_immediately(self):
        # [FUNCTION DOC] A callback that succeeds first time runs once and no sleep is recorded.
        """A successful first attempt does not sleep or retry."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call; ignores the attempt number.
            nonlocal call_count
            call_count += 1

        await retry(fn, sleep_fn=fake_sleep)

        assert call_count == 1
        assert fake_sleep.delays == []

    async def test_retries_then_succeeds(self):
        # [FUNCTION DOC] A callback failing three times then succeeding runs four times with three
        # recorded sleeps.
        """Retry keeps attempting until the callback succeeds."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call; raise a value error while the count is below 4.
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ValueError("not yet")

        await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep)

        assert call_count == 4
        assert len(fake_sleep.delays) == 3

    async def test_single_retry(self):
        # [FUNCTION DOC] One failure then success gives two calls and one sleep.
        """One failure followed by success produces one retry."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call; raise a runtime error on the first call.
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail once")

        await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep)

        assert call_count == 2
        assert len(fake_sleep.delays) == 1

    async def test_fn_receives_correct_attempt_number(self):
        # [FUNCTION DOC] The callback fails three times; the attempt numbers it received are 0, 1,
        # 2, 3.
        """First call gets attempt=0, first retry gets attempt=1, etc."""
        received_attempts: list[int] = []
        fake_sleep = _FakeSleep()

        async def fn(attempt):
            # [FUNCTION DOC] HELPER: Record the attempt number; raise while fewer than four recorded.
            received_attempts.append(attempt)
            if len(received_attempts) < 4:
                raise ValueError("not yet")

        await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep)

        assert received_attempts == [0, 1, 2, 3]


class TestRetryNonRetryable:
    # [CLASS DOC] Tests of the non-retryable error fast path.
    async def test_non_retryable_error_is_raised_after_a_single_attempt(self):
        # [FUNCTION DOC] A callback raising the non-retryable error (max attempts 5) is called once,
        # the error propagates with its message, and no sleep is recorded.
        """A NonRetryableError aborts immediately: fn is called once and no backoff sleep happens."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt: int) -> None:
            # [FUNCTION DOC] HELPER: Count the call and raise the non-retryable error.
            nonlocal call_count
            call_count += 1
            raise NonRetryableError("cannot heal anymore")

        with pytest.raises(NonRetryableError, match="cannot heal anymore"):
            await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep, max_attempts=5)

        assert call_count == 1
        assert fake_sleep.delays == []

    async def test_ordinary_error_is_retried_up_to_max_attempts(self):
        # [FUNCTION DOC] A callback always raising a value error with max attempts 5 is called five
        # times with four sleeps, and the last error propagates.
        """The same setup with an ordinary exception keeps retrying, proving the fast path is what stops it."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt: int) -> None:
            # [FUNCTION DOC] HELPER: Count the call and raise a value error.
            nonlocal call_count
            call_count += 1
            raise ValueError("transient")

        with pytest.raises(ValueError, match="transient"):
            await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep, max_attempts=5)

        assert call_count == 5
        assert len(fake_sleep.delays) == 4


class TestRetryLogging:
    # [CLASS DOC] Tests of the warning logs emitted by retry.
    async def test_logs_on_retry(self, caplog):
        # [FUNCTION DOC] A callback failing twice then succeeding produces exactly two warning
        # records containing the word "retrying".
        """Each retry emits a warning log."""
        call_count = 0

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call; raise a value error while the count is below 3.
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("boom")

        with caplog.at_level("WARNING"):
            await retry(fn, initial_delay=1.0, sleep_fn=_FakeSleep())

        retry_messages = [r for r in caplog.records if "retrying" in r.message]
        assert len(retry_messages) == 2

    async def test_no_log_on_first_success(self, caplog):
        # [FUNCTION DOC] A callback that succeeds first time produces no record containing
        # "retrying".
        """A successful first attempt emits no retry warning."""

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Succeed immediately.
            pass

        with caplog.at_level("WARNING"):
            await retry(fn, sleep_fn=_FakeSleep())

        assert not any("retrying" in r.message for r in caplog.records)

    async def test_logs_include_exc_info(self, caplog):
        # [FUNCTION DOC] After one failure, exactly one "retrying" record exists and it carries
        # exception info.
        """Retry warnings retain exception information."""
        call_count = 0

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call; raise a value error on the first call.
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("detail")

        with caplog.at_level("WARNING"):
            await retry(fn, initial_delay=1.0, sleep_fn=_FakeSleep())

        retry_records = [r for r in caplog.records if "retrying" in r.message]
        assert len(retry_records) == 1
        assert retry_records[0].exc_info is not None

    async def test_log_message_includes_delay(self, caplog):
        # [FUNCTION DOC] With initial delay 2.5 and one failure, the single "retrying" record
        # contains "2.5s".
        """A retry warning includes its sleep delay."""
        call_count = 0

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call; raise a runtime error on the first call.
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail")

        with caplog.at_level("WARNING"):
            await retry(fn, initial_delay=2.5, sleep_fn=_FakeSleep())

        retry_records = [r for r in caplog.records if "retrying" in r.message]
        assert len(retry_records) == 1
        assert "2.5s" in retry_records[0].message


class TestRetryMaxAttempts:
    # [CLASS DOC] Tests of the maximum attempt count.
    async def test_raises_after_max_attempts(self):
        # [FUNCTION DOC] With max attempts 3 and an always-failing callback, the third error ("fail
        # 3") propagates after three calls and two sleeps.
        """The last exception propagates once max_attempts calls have all failed."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call and raise a value error numbered by the count.
            nonlocal call_count
            call_count += 1
            raise ValueError(f"fail {call_count}")

        with pytest.raises(ValueError, match="fail 3"):
            await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep, max_attempts=3)

        assert call_count == 3
        assert len(fake_sleep.delays) == 2

    async def test_succeeds_on_last_allowed_attempt(self):
        # [FUNCTION DOC] With max attempts 3, a callback succeeding on its third call completes
        # without error after three calls.
        """No exception when fn succeeds exactly at the max_attempts-th call."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call; raise a runtime error while below 3.
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("not yet")

        await retry(fn, initial_delay=1.0, sleep_fn=fake_sleep, max_attempts=3)

        assert call_count == 3

    async def test_max_attempts_one_never_retries(self):
        # [FUNCTION DOC] With max attempts 1 and an always-failing callback, the error propagates
        # after one call and no sleep.
        """max_attempts=1 means a single call with no retry."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call and raise a runtime error.
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            await retry(fn, sleep_fn=fake_sleep, max_attempts=1)

        assert call_count == 1
        assert fake_sleep.delays == []

    async def test_default_is_unlimited(self):
        # [FUNCTION DOC] With no maximum and zero delay, a callback failing 49 times succeeds on the
        # 50th call.
        """Without max_attempts, retry keeps going far beyond any small cap."""
        call_count = 0

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call; raise a runtime error while below 50.
            nonlocal call_count
            call_count += 1
            if call_count < 50:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=0.0, sleep_fn=_FakeSleep())

        assert call_count == 50

    async def test_invalid_max_attempts_rejected(self):
        # [FUNCTION DOC] A maximum of zero raises an assertion error.
        """max_attempts below 1 is a programming error."""

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Succeed immediately.
            pass

        with pytest.raises(AssertionError):
            await retry(fn, sleep_fn=_FakeSleep(), max_attempts=0)

    async def test_gives_up_log_message(self, caplog):
        # [FUNCTION DOC] With max attempts 2 and an always-failing callback, the error propagates, a
        # "giving up" warning exists, and exactly one "retrying" warning exists.
        """The final failure logs a giving-up warning instead of a retrying one."""

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Always raise a runtime error.
            raise RuntimeError("fail")

        with caplog.at_level("WARNING"):
            with pytest.raises(RuntimeError):
                await retry(fn, initial_delay=1.0, sleep_fn=_FakeSleep(), max_attempts=2)

        assert any("giving up" in r.message for r in caplog.records)
        assert len([r for r in caplog.records if "retrying" in r.message]) == 1


class TestRetryBackoff:
    # [CLASS DOC] Tests of the delay schedule.
    async def test_delay_doubles_each_retry(self):
        # [FUNCTION DOC] Initial delay 1, cap 100, factor 2, four failures: sleeps are 1, 2, 4, 8.
        """Default exponential backoff doubles each delay."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call; raise a runtime error for the first four calls.
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=1.0, max_delay=100.0, backoff_factor=2.0, sleep_fn=fake_sleep)

        assert fake_sleep.delays == [1.0, 2.0, 4.0, 8.0]

    async def test_delay_capped_at_max(self):
        # [FUNCTION DOC] Initial delay 1, cap 3, factor 2, five failures: sleeps are 1, 2, 3, 3, 3.
        """Backoff delays stop growing at max_delay."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call; raise a runtime error for the first five calls.
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=1.0, max_delay=3.0, backoff_factor=2.0, sleep_fn=fake_sleep)

        assert fake_sleep.delays == [1.0, 2.0, 3.0, 3.0, 3.0]

    async def test_custom_backoff_factor(self):
        # [FUNCTION DOC] Factor 3 with three failures: sleeps are 1, 3, 9.
        """A custom backoff factor scales each retry delay."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call; raise a runtime error for the first three calls.
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=1.0, max_delay=100.0, backoff_factor=3.0, sleep_fn=fake_sleep)

        assert fake_sleep.delays == [1.0, 3.0, 9.0]

    async def test_zero_initial_delay(self):
        # [FUNCTION DOC] Initial delay zero with two failures: three calls and sleeps [0, 0].
        """A zero initial delay keeps immediate retries nonblocking."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call; raise a runtime error while below 3.
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=0.0, sleep_fn=fake_sleep)

        assert call_count == 3
        assert fake_sleep.delays == [0.0, 0.0]

    async def test_default_params_are_reasonable(self):
        # [FUNCTION DOC] The private default constants are initial delay 1.0, max delay 60.0 and
        # backoff factor 2.0.
        """The public retry defaults remain stable."""
        from miles.utils.retry_utils import _DEFAULT_BACKOFF_FACTOR, _DEFAULT_INITIAL_DELAY, _DEFAULT_MAX_DELAY

        assert _DEFAULT_INITIAL_DELAY == 1.0
        assert _DEFAULT_MAX_DELAY == 60.0
        assert _DEFAULT_BACKOFF_FACTOR == 2.0

    async def test_many_retries_stay_capped(self):
        # [FUNCTION DOC] Initial 1, cap 5, factor 2, eight failures: sleeps 1, 2, 4, then five 5s.
        """After hitting max_delay, all subsequent delays remain at max."""
        call_count = 0
        fake_sleep = _FakeSleep()

        async def fn(_attempt):
            # [FUNCTION DOC] HELPER: Count the call; raise a runtime error for the first eight calls.
            nonlocal call_count
            call_count += 1
            if call_count <= 8:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=1.0, max_delay=5.0, backoff_factor=2.0, sleep_fn=fake_sleep)

        # 1, 2, 4, 5, 5, 5, 5, 5
        assert fake_sleep.delays == [1.0, 2.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0]


class TestRetryUntilDeadline:
    # [CLASS DOC] Tests of the deadline-bounded retry helper (callback receives remaining seconds).
    async def test_returns_the_value_of_the_first_success(self):
        # [FUNCTION DOC] An attempt returning 7 immediately makes the helper return 7.
        """The helper hands back whatever the attempt returned."""
        result = await retry_until_deadline(lambda remaining: _immediately(7), total_seconds=1.0, retry_on=ValueError)
        assert result == 7

    async def test_retries_until_success(self):
        # [FUNCTION DOC] An attempt failing twice with the retryable error then returning "done":
        # the result is "done" after exactly three attempts.
        """A retryable failure is retried and the later success is returned."""
        attempts = []

        async def attempt(remaining: float) -> str:
            # [FUNCTION DOC] HELPER: Record the remaining budget; raise a value error for the first
            # two calls, then return "done".
            attempts.append(remaining)
            if len(attempts) < 3:
                raise ValueError("not yet")
            return "done"

        result = await retry_until_deadline(
            attempt, total_seconds=5.0, retry_on=ValueError, initial_delay=0.01, max_delay=0.05
        )
        assert result == "done" and len(attempts) == 3

    async def test_remaining_budget_shrinks(self):
        # [FUNCTION DOC] An always-failing attempt with a 0.2 s budget and 0.05 s initial delay is
        # called at least twice; the second remaining value is smaller than the first.
        """Each attempt is told how much of the budget is left."""
        seen: list[float] = []

        async def attempt(remaining: float) -> None:
            # [FUNCTION DOC] HELPER: Record the remaining budget and raise a value error.
            seen.append(remaining)
            raise ValueError("always")

        with pytest.raises(ValueError):
            await retry_until_deadline(attempt, total_seconds=0.2, retry_on=ValueError, initial_delay=0.05)
        assert len(seen) >= 2 and seen[1] < seen[0]

    async def test_unlisted_error_propagates_immediately(self):
        # [FUNCTION DOC] A type error (not in the retry list) propagates after exactly one attempt.
        """An error outside retry_on is not retried."""
        attempts = []

        async def attempt(remaining: float) -> None:
            # [FUNCTION DOC] HELPER: Record the call and raise a type error.
            attempts.append(1)
            raise TypeError("fatal")

        with pytest.raises(TypeError):
            await retry_until_deadline(attempt, total_seconds=5.0, retry_on=ValueError, initial_delay=0.01)
        assert len(attempts) == 1

    async def test_last_error_reraised_when_budget_runs_out(self):
        # [FUNCTION DOC] An always-failing attempt with a 0.1 s budget ends by raising the value
        # error ("still down").
        """The budget bounds the retries and the final failure surfaces."""

        async def attempt(remaining: float) -> None:
            # [FUNCTION DOC] HELPER: Always raise a value error.
            raise ValueError("still down")

        with pytest.raises(ValueError, match="still down"):
            await retry_until_deadline(attempt, total_seconds=0.1, retry_on=ValueError, initial_delay=0.02)

    async def test_a_retried_attempt_emits_structured_info_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        # [FUNCTION DOC] Two failures then success, zero delay and jitter. Exactly two info records
        # with operation "retry_until_deadline" and phase "retrying" exist; record n contains the
        # attempt number n, a zero sleep, a remaining-seconds field and the error text for failure n.
        """Every condition the caller declared retryable is expected, so the log says retrying, not failed."""
        attempts = 0

        async def attempt(remaining: float) -> str:
            # [FUNCTION DOC] HELPER: Count calls; raise a value error "failure-N" for the first two,
            # then return "done".
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise ValueError(f"failure-{attempts}")
            return "done"

        with caplog.at_level(logging.INFO, logger="miles.utils.retry_utils"):
            result = await retry_until_deadline(
                attempt,
                total_seconds=1.0,
                retry_on=ValueError,
                initial_delay=0.0,
                jitter_ratio=0.0,
            )

        records = [
            record
            for record in caplog.records
            if "op=retry_until_deadline" in record.message and "phase=retrying" in record.message
        ]
        assert result == "done"
        assert len(records) == 2
        for attempt_number, record in enumerate(records, start=1):
            assert f"attempt={attempt_number}" in record.message
            assert "sleep_s=0.0" in record.message
            assert "remaining_s=" in record.message
            assert f"error=ValueError('failure-{attempt_number}')" in record.message

    async def test_caller_log_fields_are_merged_into_the_attempt_log(self, caplog: pytest.LogCaptureFixture) -> None:
        # [FUNCTION DOC] With caller log fields op "submit" and call "c1" and one failure, the single
        # retrying record contains both fields and no longer contains the default operation name.
        """Caller-supplied fields land in the retry log and may override the default op."""
        attempts = 0

        async def attempt(remaining: float) -> str:
            # [FUNCTION DOC] HELPER: Count calls; raise a value error on the first, then return "done".
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ValueError("failure")
            return "done"

        with caplog.at_level(logging.INFO, logger="miles.utils.retry_utils"):
            await retry_until_deadline(
                attempt,
                total_seconds=1.0,
                retry_on=ValueError,
                initial_delay=0.0,
                jitter_ratio=0.0,
                log_fields={"op": "submit", "call": "c1"},
            )

        records = [record for record in caplog.records if "phase=retrying" in record.message]
        assert len(records) == 1
        assert "op=submit" in records[0].message
        assert "call=c1" in records[0].message
        assert "op=retry_until_deadline" not in records[0].message

    async def test_backoff_grows_and_caps_before_each_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With asyncio sleep replaced by the recorder, initial 0.5, cap 4, factor 3,
        # no jitter and five failures: result "done" and sleeps 0.5, 1.5, 4, 4, 4.
        """Sleeps between retries grow by the backoff factor and then stop at max_delay."""
        fake_sleep = _FakeSleep()
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        attempts = 0

        async def attempt(remaining: float) -> str:
            # [FUNCTION DOC] HELPER: Count calls; raise a value error for the first five, then
            # return "done".
            nonlocal attempts
            attempts += 1
            if attempts <= 5:
                raise ValueError("still down")
            return "done"

        result = await retry_until_deadline(
            attempt,
            total_seconds=1000.0,
            retry_on=ValueError,
            initial_delay=0.5,
            max_delay=4.0,
            backoff_factor=3.0,
            jitter_ratio=0.0,
        )

        assert result == "done"
        assert fake_sleep.delays == [0.5, 1.5, 4.0, 4.0, 4.0]

    async def test_jitter_ratio_is_applied_to_each_base_delay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With sleep and the module's random replaced by fakes (random returns the
        # range top), initial 1, factor 2, jitter ratio 0.25 and three failures: uniform ranges
        # requested are (0, 0.25), (0, 0.5), (0, 1.0) and the sleeps are 1.25, 2.5, 5.0.
        """Jitter is drawn from zero to jitter_ratio times the base delay and added on top of it."""
        fake_sleep = _FakeSleep()
        fake_random = _FakeRandom()
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(retry_utils, "random", fake_random)
        attempts = 0

        async def attempt(remaining: float) -> str:
            # [FUNCTION DOC] HELPER: Count calls; raise a value error for the first three, then
            # return "done".
            nonlocal attempts
            attempts += 1
            if attempts <= 3:
                raise ValueError("still down")
            return "done"

        result = await retry_until_deadline(
            attempt,
            total_seconds=1000.0,
            retry_on=ValueError,
            initial_delay=1.0,
            max_delay=10.0,
            backoff_factor=2.0,
            jitter_ratio=0.25,
        )

        assert result == "done"
        assert fake_random.ranges == [(0.0, 0.25), (0.0, 0.5), (0.0, 1.0)]
        assert fake_sleep.delays == [1.25, 2.5, 5.0]

    async def test_budget_exhaustion_reraises_without_sleep_and_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # [FUNCTION DOC] Budget 0.5 s, initial delay 10 s, always failing: the value error is raised
        # after one attempt with no sleep, and exactly one warning "retry_until_deadline: giving up
        # after 0.5s" with exception info is logged.
        """A sleep that would outlast the remaining budget re-raises at once and logs the giving-up warning."""
        fake_sleep = _FakeSleep()
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        attempts = 0

        async def attempt(remaining: float) -> None:
            # [FUNCTION DOC] HELPER: Count the call and raise a value error.
            nonlocal attempts
            attempts += 1
            raise ValueError("still down")

        with caplog.at_level(logging.WARNING, logger="miles.utils.retry_utils"):
            with pytest.raises(ValueError, match="still down"):
                await retry_until_deadline(
                    attempt,
                    total_seconds=0.5,
                    retry_on=ValueError,
                    initial_delay=10.0,
                    jitter_ratio=0.0,
                )

        assert attempts == 1
        assert fake_sleep.delays == []
        records = [record for record in caplog.records if "giving up" in record.message]
        assert len(records) == 1
        assert records[0].message == "retry_until_deadline: giving up after 0.5s"
        assert records[0].exc_info is not None

    async def test_retry_on_tuple_retries_each_listed_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With a retry list of OS error and value error, the attempt raises the OS
        # error, then the value error, then succeeds; the result is "done", both kinds were raised,
        # and sleeps are 0.5 then 1.0.
        """Every member of a retry_on tuple, including OSError, is treated as retryable."""
        fake_sleep = _FakeSleep()
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        raised: list[type[Exception]] = []

        async def attempt(remaining: float) -> str:
            # [FUNCTION DOC] HELPER: First call raises an OS error, second a value error, later
            # calls return "done"; the raised kinds are recorded.
            if not raised:
                raised.append(OSError)
                raise OSError("connection refused")
            if len(raised) == 1:
                raised.append(ValueError)
                raise ValueError("bad payload")
            return "done"

        result = await retry_until_deadline(
            attempt,
            total_seconds=1000.0,
            retry_on=(OSError, ValueError),
            initial_delay=0.5,
            jitter_ratio=0.0,
        )

        assert result == "done"
        assert raised == [OSError, ValueError]
        assert fake_sleep.delays == [0.5, 1.0]

    async def test_an_attempt_is_not_bounded_unless_a_bound_is_asked_for(self):
        # [FUNCTION DOC] Without a per-attempt bound, an attempt sleeping 0.05 s returns "done"
        # within a 5 s budget.
        """The existing callers hand the whole remaining budget to one attempt, and that must keep working."""

        async def attempt(remaining: float) -> str:
            # [FUNCTION DOC] HELPER: Sleep 0.05 s then return "done".
            await asyncio.sleep(0.05)
            return "done"

        assert await retry_until_deadline(attempt, total_seconds=5.0, retry_on=ValueError) == "done"

    async def test_an_attempt_that_never_answers_is_cut_short_and_tried_again(self):
        # [FUNCTION DOC] With a 0.02 s per-attempt bound, the first attempt hangs and is cut short;
        # the second returns "done": result "done" after exactly two attempts.
        """Without a per-attempt bound one call that hangs eats the whole nominal budget in a single attempt."""
        attempts = 0

        async def attempt(remaining: float) -> str:
            # [FUNCTION DOC] HELPER: Count calls; the first sleeps 30 s, then every call returns "done".
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                await asyncio.sleep(30.0)
            return "done"

        result = await retry_until_deadline(
            attempt, total_seconds=5.0, retry_on=ValueError, attempt_seconds=0.02, initial_delay=0.01
        )

        assert result == "done"
        assert attempts == 2

    async def test_an_attempt_is_told_it_only_has_its_own_bound(self):
        # [FUNCTION DOC] Budget 100 s with a 0.5 s per-attempt bound: the attempt is told remaining
        # time 0.5 (once).
        """An attempt that sizes its own request off the remaining budget would overshoot the bound it runs under."""
        seen: list[float] = []

        async def attempt(remaining: float) -> str:
            # [FUNCTION DOC] HELPER: Record the remaining value and return "done".
            seen.append(remaining)
            return "done"

        await retry_until_deadline(attempt, total_seconds=100.0, retry_on=ValueError, attempt_seconds=0.5)

        assert seen == [0.5]

    async def test_a_bound_attempt_that_never_answers_reports_the_bound_it_ran_out_of(self):
        # [FUNCTION DOC] An attempt hanging for 30 s with a 0.02 s bound and 0.05 s budget ends with
        # the attempt timeout error saying it did not answer within 0.02s.
        """An operator seeing only the nominal deadline would never learn the attempt stopped answering."""

        async def attempt(remaining: float) -> None:
            # [FUNCTION DOC] HELPER: Sleep 30 s (never answering in time).
            await asyncio.sleep(30.0)

        with pytest.raises(AttemptTimeoutError, match="did not answer within 0.02s"):
            await retry_until_deadline(
                attempt, total_seconds=0.05, retry_on=ValueError, attempt_seconds=0.02, initial_delay=0.01
            )


async def _immediately(value):
    # [FUNCTION DOC] HELPER: Async function returning the given value.
    return value


class TestWhoTheAttemptTimeoutBelongsTo:
    # [CLASS DOC] Tests that only timeouts imposed by the helper count as attempt timeouts.
    async def test_a_timeout_the_attempt_raised_itself_reaches_the_caller_unretried(self):
        # [FUNCTION DOC] An attempt raising its own timeout error (not in the retry list) with a 1 s
        # bound propagates that error after exactly one attempt.
        """`AttemptTimeoutError` is always retryable, so mislabelling one hides an error the caller never allowed."""
        attempts = 0

        async def attempt(remaining: float) -> None:
            # [FUNCTION DOC] HELPER: Count the call and raise a timeout error ("the worker did not
            # answer").
            nonlocal attempts
            attempts += 1
            raise TimeoutError("the worker did not answer")

        with pytest.raises(TimeoutError, match="the worker did not answer"):
            await retry_until_deadline(
                attempt, total_seconds=5.0, retry_on=ValueError, attempt_seconds=1.0, initial_delay=0.01
            )

        assert attempts == 1

    async def test_an_attempt_is_never_given_longer_than_the_deadline_that_is_left(self):
        # [FUNCTION DOC] A 100 s per-attempt bound with a 0.5 s budget: the attempt is told a
        # remaining time of at most 0.5.
        """A per-attempt bound wider than the remaining budget would push the last attempt past the deadline."""
        seen: list[float] = []

        async def attempt(remaining: float) -> str:
            # [FUNCTION DOC] HELPER: Record the remaining value and return "done".
            seen.append(remaining)
            return "done"

        await retry_until_deadline(attempt, total_seconds=0.5, retry_on=ValueError, attempt_seconds=100.0)

        assert seen and seen[0] <= 0.5
