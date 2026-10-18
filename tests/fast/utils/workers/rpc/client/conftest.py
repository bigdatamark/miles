# [MODULE DOC] HELPER: Pytest configuration for the rpc client tests. Provides one shared fixture
# that replaces the asynchronous sleep used by the rpc call module, so backoff durations can be
# observed without actually waiting.
from __future__ import annotations

import pytest

from miles.utils.workers.rpc.client import call as rpc_call_module


@pytest.fixture
def recorded_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    # [FUNCTION DOC] HELPER: Fixture. Creates an empty list and patches the sleep function of the
    # asyncio module as seen through the rpc call module with a fake coroutine that appends the
    # requested seconds to the list and returns at once (no real waiting). The patch is undone at test
    # end. Yields-by-return the list, so tests can assert the exact sleep durations requested.
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        # [FUNCTION DOC] HELPER: Records the requested duration in the list; never waits.
        sleeps.append(seconds)

    monkeypatch.setattr(rpc_call_module.asyncio, "sleep", fake_sleep)
    return sleeps
