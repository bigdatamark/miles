# [MODULE DOC] HELPER: Logging helper for the dashboard: a rate-limited warning emitter.
"""Logging helpers shared by the dashboard's background collectors."""

from __future__ import annotations

import logging
import time


class RateLimitedWarner:
    # [CLASS DOC] HELPER: Wraps a logger; allows one warning per time window (default 300 seconds,
    # public interval attribute), measured with the monotonic clock. Not thread-safe (races can let
    # two warnings through).
    """Warn at most once per ``interval_seconds``.

    Background loops (scraper ticks, NVML sampling) hit the same failure
    thousands of times during an outage; observability must neither spam the
    training logs nor die silently, so the first warning goes through and
    repeats are suppressed for the window.
    """

    def __init__(self, logger: logging.Logger, *, interval_seconds: float = 300.0):
        # [FUNCTION DOC] HELPER: Store the logger and window; the last-warning time starts at negative
        # infinity so the first warning always passes.
        self._logger = logger
        self.interval_seconds = interval_seconds
        self._last_warn = float("-inf")

    def warn(self, message: str) -> None:
        # [FUNCTION DOC] HELPER: If at least the window has elapsed since the last emitted warning,
        # records the time and logs the message at warning level with a suffix saying further
        # warnings are suppressed for the window length (whole seconds); otherwise drops it silently.
        now = time.monotonic()
        if now - self._last_warn >= self.interval_seconds:
            self._last_warn = now
            self._logger.warning("%s (further warnings suppressed for %.0fs)", message, self.interval_seconds)

    def reset_window_for_test(self) -> None:
        # [FUNCTION DOC] HELPER: Test aid; forgets the last warning time so the next warning passes.
        self._last_warn = float("-inf")
