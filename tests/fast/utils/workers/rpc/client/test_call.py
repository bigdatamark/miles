# [MODULE DOC] Unit tests for the rpc client call object. Covers validation of the call timeout
# (must be finite and positive), how a single poll attempt backs off after a retryable failure (the
# sleep is capped by the remaining budget and a long poll timeout does not sleep at all), and that
# the poll loop gives up close to the call timeout when every poll fails. Uses fake transports that
# always fail, and the shared fixture that records sleeps instead of sleeping.
from __future__ import annotations

import math
import time
from typing import Any

import httpx
import pytest

from miles.utils.workers.rpc.client.call import RpcCall, _CallStillPendingError
from miles.utils.workers.rpc.client.handle import DEFAULT_CALL_TIMEOUT_SECONDS
from miles.utils.workers.rpc.client.misc import RETRY_INITIAL_DELAY_SECONDS, RetryableResponseError
from miles.utils.workers.rpc.common.metadata import RpcMethodSpec, collect_rpc_method_specs


class _Worker:
    # [CLASS DOC] HELPER: Minimal worker class with one public method, used only to derive a method spec.
    def demo(self, a: int) -> int:
        # [FUNCTION DOC] HELPER: Returns its argument unchanged; never actually executed here.
        return a


class _FailingTransport:
    # [CLASS DOC] HELPER: Transport stand-in whose every request fails with a connection error.
    async def request(self, method: str, path: str, *, seconds: float, response_model: type, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Ignores all arguments and always raises an http connect error with the
        # message "injected transport failure".
        raise httpx.ConnectError("injected transport failure")


class _RaisingTransport:
    # [CLASS DOC] HELPER: Transport stand-in that counts requests and raises a configured exception on
    # each one.
    def __init__(self, error: Exception) -> None:
        # [FUNCTION DOC] HELPER: Sets the request counter to zero and stores the exception to raise.
        self.requests = 0
        self._error = error

    async def request(self, method: str, path: str, *, seconds: float, response_model: type, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Increments the request counter and raises the stored exception;
        # other arguments are ignored.
        self.requests += 1
        raise self._error


def _spec() -> RpcMethodSpec:
    # [FUNCTION DOC] HELPER: Returns the collected rpc method spec of the demo method of the test worker.
    return collect_rpc_method_specs(_Worker)["demo"]


def _make_call(*, call_timeout_seconds: float, transport: Any | None = None) -> RpcCall:
    # [FUNCTION DOC] HELPER: Builds an rpc call for the demo method with the argument a equal to 1, the
    # worker class name "_Worker", the given call timeout, and the given transport (a transport that
    # always fails with a connect error when none is given). The constructor may raise an assertion
    # error for an invalid timeout.
    return RpcCall(
        spec=_spec(),
        kwargs={"a": 1},
        worker_cls_name="_Worker",
        transport=transport if transport is not None else _FailingTransport(),
        call_timeout_seconds=call_timeout_seconds,
    )


class TestCallTimeoutValidation:
    # [CLASS DOC] Tests that the call constructor rejects call timeouts that are not finite and
    # positive and accepts normal ones.
    @pytest.mark.parametrize("call_timeout_seconds", [float("inf"), float("nan"), 0.0, -1.0])
    def test_a_non_finite_or_non_positive_call_timeout_is_rejected(self, call_timeout_seconds: float) -> None:
        # [FUNCTION DOC] Parametrized over infinity, not-a-number, zero and minus one; asserts that
        # building a call raises an assertion error whose message mentions the call timeout name.
        """The retry loop compares the remaining budget against the deadline, so such a value would never expire."""
        with pytest.raises(AssertionError, match="call_timeout_seconds"):
            _make_call(call_timeout_seconds=call_timeout_seconds)

    def test_a_finite_call_timeout_is_accepted(self) -> None:
        # [FUNCTION DOC] Asserts that a 30 second timeout builds a call without error.
        """A normal deadline is the whole point of the parameter and must stay usable."""
        assert _make_call(call_timeout_seconds=30.0) is not None

    def test_a_negative_infinite_call_timeout_is_rejected(self) -> None:
        # [FUNCTION DOC] Asserts that minus infinity raises an assertion error mentioning the call
        # timeout name.
        """Minus infinity is neither finite nor positive, and would make every remaining budget comparison nonsense."""
        with pytest.raises(AssertionError, match="call_timeout_seconds"):
            _make_call(call_timeout_seconds=float("-inf"))

    def test_the_rejection_message_names_the_offending_value(self) -> None:
        # [FUNCTION DOC] Asserts that the assertion message for an infinite timeout contains the text
        # "inf", so the bad value is visible.
        """An operator reading the crash needs to see which configured timeout was refused."""
        with pytest.raises(AssertionError, match="inf"):
            _make_call(call_timeout_seconds=float("inf"))

    def test_the_default_handle_call_timeout_is_a_usable_finite_deadline(self) -> None:
        # [FUNCTION DOC] Asserts the handle's default call timeout constant is finite and positive and
        # that a call can be built with it.
        """The shipped default must satisfy the finite-deadline check, or every rpc call would assert at runtime."""
        assert math.isfinite(DEFAULT_CALL_TIMEOUT_SECONDS)
        assert DEFAULT_CALL_TIMEOUT_SECONDS > 0
        assert _make_call(call_timeout_seconds=DEFAULT_CALL_TIMEOUT_SECONDS) is not None


class TestPollBackoffAgainstTheRemainingBudget:
    # [CLASS DOC] Tests the sleep requested by a single poll attempt (the private poll-once method)
    # after a retryable failure; the sleep function is replaced by a recorder. The argument of
    # poll-once is the remaining budget in seconds, and each test expects a still-pending error.
    async def test_a_transport_failure_sleeps_no_longer_than_the_remaining_budget(
        self, recorded_sleeps: list[float]
    ) -> None:
        # [FUNCTION DOC] With a transport that fails and a remaining budget of 0.05 seconds, asserts the
        # poll raises the still-pending error and slept exactly once for 0.05 seconds (the backoff
        # capped to the budget).
        """Sleeping the full backoff past the deadline burns the caller's budget on a retry that can never run."""
        call = _make_call(call_timeout_seconds=30.0)

        with pytest.raises(_CallStillPendingError):
            await call._poll_once(0.05)

        assert recorded_sleeps == [0.05]

    async def test_a_transport_failure_sleeps_the_full_backoff_while_budget_remains(
        self, recorded_sleeps: list[float]
    ) -> None:
        # [FUNCTION DOC] With a remaining budget of 30 seconds, asserts the single sleep equals the
        # initial retry delay constant, i.e. the backoff is not shortened.
        """With plenty of budget left the backoff must not be shortened, or the retry loop would spin."""
        call = _make_call(call_timeout_seconds=30.0)

        with pytest.raises(_CallStillPendingError):
            await call._poll_once(30.0)

        assert recorded_sleeps == [RETRY_INITIAL_DELAY_SECONDS]

    @pytest.mark.parametrize(
        "error",
        [
            httpx.ConnectError("connect refused"),
            httpx.ReadError("connection reset"),
            httpx.PoolTimeout("no free connection"),
            RetryableResponseError("GET /v1/calls/x returned 503"),
        ],
        ids=["connect_error", "read_error", "pool_timeout", "retryable_response"],
    )
    async def test_every_retryable_poll_failure_caps_its_backoff_at_the_remaining_budget(
        self, error: Exception, recorded_sleeps: list[float]
    ) -> None:
        # [FUNCTION DOC] Parametrized over connect error, read error, pool timeout and a retryable
        # response error raised by the transport; with a remaining budget of 0.02 seconds asserts the
        # poll raises still-pending and slept exactly 0.02 seconds.
        """Each retryable poll failure shares one backoff, so all of them must respect the remaining budget."""
        call = _make_call(call_timeout_seconds=30.0, transport=_RaisingTransport(error))

        with pytest.raises(_CallStillPendingError):
            await call._poll_once(0.02)

        assert recorded_sleeps == [0.02]

    async def test_an_exhausted_budget_makes_the_backoff_sleep_zero(self, recorded_sleeps: list[float]) -> None:
        # [FUNCTION DOC] With a remaining budget of zero, asserts the poll raises still-pending and
        # requested one sleep of zero seconds.
        """Once the budget is gone the backoff must collapse to nothing instead of a whole extra second."""
        call = _make_call(call_timeout_seconds=30.0)

        with pytest.raises(_CallStillPendingError):
            await call._poll_once(0.0)

        assert recorded_sleeps == [0.0]

    async def test_a_long_poll_timeout_does_not_sleep_at_all(self, recorded_sleeps: list[float]) -> None:
        # [FUNCTION DOC] With a transport raising a built-in timeout error, asserts the poll raises
        # still-pending and recorded no sleep at all.
        """A long poll already consumed its wait, so retrying it must not add any backoff on top."""
        call = _make_call(call_timeout_seconds=30.0, transport=_RaisingTransport(TimeoutError("long poll expired")))

        with pytest.raises(_CallStillPendingError):
            await call._poll_once(30.0)

        assert recorded_sleeps == []


class TestPollLoopDeadline:
    # [CLASS DOC] Tests the overall poll loop deadline with real (unpatched) sleeping.
    async def test_the_poll_loop_gives_up_close_to_the_call_timeout_when_every_poll_fails(self) -> None:
        # [FUNCTION DOC] Builds a call with a 0.1 second timeout over a transport that always raises a
        # connect error, runs the poll-until-done loop and measures wall time. Asserts a built-in
        # timeout error is raised, the transport saw at least one request, and elapsed time is below
        # half of the initial retry delay (so the loop did not sleep a full backoff past the deadline).
        """A fixed one-second backoff overruns a short deadline, so the caller waits far past the budget it set."""
        transport = _RaisingTransport(httpx.ConnectError("injected transport failure"))
        call = _make_call(call_timeout_seconds=0.1, transport=transport)

        started_at = time.monotonic()
        with pytest.raises(TimeoutError):
            await call._poll_until_done()
        elapsed = time.monotonic() - started_at

        assert transport.requests >= 1
        assert elapsed < RETRY_INITIAL_DELAY_SECONDS / 2
