# [MODULE DOC]
# CPU tests that the weight update session begin and end brackets are fanned out to every rollout
# engine concurrently, and that failures propagate only after every engine has settled. Uses several
# fake engine clients (recording, failing at begin, failing at end, gated at end) that log strings
# like "begin-<index>" into a shared list, plus an observed-future wrapper that signals when the
# session code starts waiting on a result.
import asyncio
import concurrent.futures
import threading
from unittest.mock import patch

import pytest

from miles.backends.training_utils.weight_update.session import begin_weight_update, end_weight_update
from miles.utils import async_utils

_SESSION_MODULE = "miles.backends.training_utils.weight_update.session"


class _RecordingClient:
    # [CLASS DOC] HELPER: Fake engine that always succeeds and logs "begin-<index>" and
    # "end-<index>" into the shared list; it also records begin selectors in its own list.
    def __init__(self, calls: list[str], engine_index: int, selectors: list[str] | None = None):
        # [FUNCTION DOC] HELPER: Stores the shared log, engine index and a selector list (a new
        # empty one if none is given).
        self._calls = calls
        self._engine_index = engine_index
        self._selectors = selectors if selectors is not None else []

    async def begin_weight_update(self, selector: str = "all", sync_base: bool = True):
        # [FUNCTION DOC] HELPER: Async; logs the begin entry, records the selector (default "all";
        # the base sync flag is ignored) and returns a success mapping.
        self._calls.append(f"begin-{self._engine_index}")
        self._selectors.append(selector)
        return {"success": True}

    async def end_weight_update(self, expected_lora_checksums=None):
        # [FUNCTION DOC] HELPER: Async; logs the end entry (the checksums argument is ignored) and
        # returns a success mapping.
        self._calls.append(f"end-{self._engine_index}")
        return {"success": True}


class _FailingClient:
    # [CLASS DOC] HELPER: Fake engine whose begin either raises a runtime error saying boom (no
    # gate) or, when given a gate event, blocks on the gate (up to thirty seconds, blocking the
    # event loop thread), logs "slow-<index>" and succeeds. Its end always succeeds.
    def __init__(self, calls: list[str], engine_index: int, gate: threading.Event | None = None):
        # [FUNCTION DOC] HELPER: Stores the shared log, engine index and optional gate.
        self._calls = calls
        self._engine_index = engine_index
        self._gate = gate

    async def begin_weight_update(self, selector: str = "all", sync_base: bool = True):
        # [FUNCTION DOC] HELPER: Async; behavior as described in the class documentation.
        if self._gate is not None:
            self._gate.wait(timeout=30)
            self._calls.append(f"slow-{self._engine_index}")
            return {"success": True}
        raise RuntimeError("boom")

    async def end_weight_update(self, expected_lora_checksums=None):
        # [FUNCTION DOC] HELPER: Async; returns a success mapping without logging.
        return {"success": True}


class _EndFailingClient:
    # [CLASS DOC] HELPER: Fake engine whose begin logs "begin-<index>" and succeeds, and whose end
    # raises a runtime error saying close failed.
    def __init__(self, calls: list[str], engine_index: int):
        # [FUNCTION DOC] HELPER: Stores the shared log and engine index.
        self._calls = calls
        self._engine_index = engine_index

    async def begin_weight_update(self, selector: str = "all", sync_base: bool = True):
        # [FUNCTION DOC] HELPER: Async; logs the begin entry and returns a success mapping.
        self._calls.append(f"begin-{self._engine_index}")
        return {"success": True}

    async def end_weight_update(self, expected_lora_checksums=None):
        # [FUNCTION DOC] HELPER: Async; always raises a runtime error with the message close failed.
        raise RuntimeError("close failed")


class _GatedEndClient:
    # [CLASS DOC] HELPER: Fake engine whose begin logs and succeeds and whose end waits (off the
    # event loop, up to five seconds) for a gate event, raising a timeout error if it never opens,
    # then logs "end-<index>" and succeeds.
    def __init__(self, calls: list[str], engine_index: int, gate: threading.Event):
        # [FUNCTION DOC] HELPER: Stores the shared log, engine index and the gate event.
        self._calls = calls
        self._engine_index = engine_index
        self._gate = gate

    async def begin_weight_update(self, selector: str = "all", sync_base: bool = True):
        # [FUNCTION DOC] HELPER: Async; logs the begin entry and returns a success mapping.
        self._calls.append(f"begin-{self._engine_index}")
        return {"success": True}

    async def end_weight_update(self, expected_lora_checksums=None):
        # [FUNCTION DOC] HELPER: Async; behavior as described in the class documentation.
        if not await asyncio.to_thread(self._gate.wait, 5):
            raise TimeoutError("end_weight_update gate timed out")
        self._calls.append(f"end-{self._engine_index}")
        return {"success": True}


class _ObservedFuture:
    # [CLASS DOC] HELPER: Wrapper around a concurrent future that sets an event when its result is
    # first requested, letting tests know the session code is waiting on it.
    def __init__(self, future: concurrent.futures.Future, result_started: threading.Event):
        # [FUNCTION DOC] HELPER: Stores the wrapped future and the event.
        self._future = future
        self._result_started = result_started

    def result(self):
        # [FUNCTION DOC] HELPER: Sets the event, then blocks for and returns the wrapped result
        # (re-raising its exception).
        self._result_started.set()
        return self._future.result()


class TestWeightUpdateSessionFanOut:
    # [CLASS DOC] Tests of the session begin and end helper functions over multiple engines.
    """The session brackets must reach every engine, not just the first."""

    def test_begin_and_end_reach_every_engine(self):
        # [FUNCTION DOC] With four recording engines, begin then end: the first four log entries are
        # the begin entries of all four engines (any order), and the next four are the end entries.
        """A missed engine would load weights outside a session and corrupt them."""
        calls: list[str] = []
        clients = [_RecordingClient(calls, i) for i in range(4)]

        begin_weight_update(clients)
        end_weight_update(clients)

        assert sorted(calls[:4]) == ["begin-0", "begin-1", "begin-2", "begin-3"]
        assert sorted(calls[4:]) == ["end-0", "end-1", "end-2", "end-3"]

    def test_a_failing_engine_fails_the_session(self):
        # [FUNCTION DOC] Begin over a recording engine and a failing engine raises the failing
        # engine's runtime error (boom); the recording engine still logged its begin.
        """A silently swallowed failure would leave that engine loading outside a session."""
        calls: list[str] = []

        with pytest.raises(RuntimeError, match="boom"):
            begin_weight_update([_RecordingClient(calls, 0), _FailingClient(calls, 1)])

        assert calls == ["begin-0"]

    def test_every_engine_is_asked_before_the_failure_surfaces(self):
        # [FUNCTION DOC] Engine zero fails immediately while engine one (with an already open gate)
        # succeeds with a "slow" log entry. Begin must raise the boom error yet the log must contain
        # the slow entry of engine one, showing all engines were launched before the error surfaced.
        """A later engine must already be in flight when an earlier engine fails."""
        calls: list[str] = []
        gate = threading.Event()
        gate.set()
        clients = [_FailingClient(calls, 0), _FailingClient(calls, 1, gate=gate)]

        with pytest.raises(RuntimeError, match="boom"):
            begin_weight_update(clients)

        assert calls == ["slow-1"]

    def test_end_failure_propagates_after_every_engine_has_settled(self):
        # [FUNCTION DOC] Runs the end bracket in a worker thread over an end-failing engine and a
        # gated engine, with the session module's async submit wrapped so the second submission
        # (the gated engine) is observed. Waits up to five seconds until the session starts waiting
        # on the second result; asserts the end bracket has not finished and nothing was logged;
        # opens the gate; then expects the close-failed runtime error from the end bracket. Finally
        # asserts exactly two submissions and that the gated engine logged its end, proving the
        # failure surfaced only after all engines settled. The gate is always opened in cleanup.
        """A failure must surface only after every engine has finished closing its session."""
        calls: list[str] = []
        gate = threading.Event()
        result_started = threading.Event()
        clients = [_EndFailingClient(calls, 0), _GatedEndClient(calls, 1, gate)]
        original_submit = async_utils.submit
        submission_count = 0

        def observed_submit(coro):
            # [FUNCTION DOC] HELPER: Submits through the original submit, counts submissions, and
            # wraps the second one in an observed future; others are returned unchanged.
            nonlocal submission_count
            future = original_submit(coro)
            submission_count += 1
            return _ObservedFuture(future, result_started) if submission_count == 2 else future

        with patch(f"{_SESSION_MODULE}.async_utils.submit", side_effect=observed_submit):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                outcome = executor.submit(end_weight_update, clients)
                try:
                    assert result_started.wait(timeout=5)
                    assert not outcome.done()
                    assert calls == []
                    gate.set()
                    with pytest.raises(RuntimeError, match="close failed"):
                        outcome.result(timeout=5)
                finally:
                    gate.set()

        assert submission_count == 2
        assert calls == ["end-1"]
