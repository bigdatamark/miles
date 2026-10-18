# [MODULE DOC] Tests of rollout limits (a counting semaphore and a lock, shared by name) across
# several agent subprocesses started concurrently from one event loop, using stub agents that hold
# a limit for a given time and report their hold start and end times. Real time is used, so tests
# take a few seconds.
"""Rollout limits hold across a rollout's agent processes and free a slot when its holder dies."""

import asyncio

import pytest

from miles.rollout.agentic.agent_function import call_agent_function
from miles.utils.test_utils import agent_function_stubs as stubs


async def _in_agent_processes(fn, count: int, **kwargs) -> list:
    # [FUNCTION DOC] HELPER: Concurrently runs the stub agent function the given number of times,
    # each in its own subprocess with the same keyword arguments, and returns the list of their
    # results in launch order.
    return await asyncio.gather(*(call_agent_function(fn, mode="subproc", **kwargs) for _ in range(count)))


def _peak_overlap(spans: list[tuple[float, float]]) -> int:
    # [FUNCTION DOC] HELPER: Given start/end time pairs, returns the maximum number of spans that
    # contain any span's start time (start inclusive, end exclusive), i.e. peak concurrency.
    return max(sum(1 for start, end in spans if start <= t < end) for t, _ in spans)


def test_semaphore_caps_holders_across_agent_processes():
    # [FUNCTION DOC] Runs six agents each holding a named semaphore of limit two for one second.
    # Asserts the peak overlap never exceeds two and the total span from earliest start to latest
    # end is at least 2.9 seconds (three serial rounds of one second).
    spans = asyncio.run(_in_agent_processes(stubs.hold_semaphore, 6, name="test-cap", limit=2, hold_s=1.0))
    assert _peak_overlap(spans) <= 2
    assert max(end for _, end in spans) - min(start for start, _ in spans) >= 2.9, "six 1s holds under 2 slots"


def test_lock_serializes_holders_across_agent_processes():
    # [FUNCTION DOC] Runs three agents each holding a named lock for half a second and asserts the
    # peak overlap is exactly one.
    spans = asyncio.run(_in_agent_processes(stubs.hold_lock, 3, name="test-lock", hold_s=0.5))
    assert _peak_overlap(spans) == 1


def test_a_dead_holder_frees_its_slot():
    # [FUNCTION DOC] Runs an agent that dies while holding a named semaphore and asserts the call
    # fails with a runtime error saying it ended before returning a result. Then a second agent
    # takes the same semaphore (limit one, no hold time) and must complete within 30 seconds,
    # proving the dead holder's slot was released.
    with pytest.raises(RuntimeError, match="before returning a result"):
        asyncio.run(call_agent_function(stubs.die_holding_semaphore, mode="subproc", name="test-dead"))
    asyncio.run(
        asyncio.wait_for(
            call_agent_function(stubs.hold_semaphore, mode="subproc", name="test-dead", limit=1, hold_s=0.0), 30
        )
    )
