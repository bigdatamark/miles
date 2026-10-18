# [MODULE DOC]
# CPU tests of the weight updater's engine session frame: for every rollout engine the updater must
# pause generation, flush the cache (skipped in in-place pause mode), begin the weight update
# session, send weights, end the session, publish the new weight version and only then continue
# generation. Engines are fake API clients that log every call as engine index, method name and
# keyword arguments, can fail a chosen method on a chosen engine, and can block named methods on
# thread events (gates) so tests can prove one phase completes on all engines before the next
# phase starts. The transfer protocol is a simple namespace of mocks. Two engines are used.
import asyncio
import concurrent.futures
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.utils import async_utils

_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"
_SESSION_MODULE = "miles.backends.training_utils.weight_update.session"

_ENGINE_COUNT = 2
_PREPARE_PHASES = ["pause_generation", "flush_cache", "begin_weight_update"]
_FINALIZE_PHASES = ["end_weight_update", "update_weight_version", "continue_generation"]


class _ObservedFuture:
    # [CLASS DOC] HELPER: Wrapper around a concurrent future that signals an event the moment a
    # consumer asks for its result, so tests know the updater has started waiting on that phase.
    def __init__(self, future: concurrent.futures.Future, result_started: threading.Event):
        # [FUNCTION DOC] HELPER: Stores the wrapped future and the event to set on result access.
        self._future = future
        self._result_started = result_started

    def result(self):
        # [FUNCTION DOC] HELPER: Sets the started event, then blocks for and returns the wrapped
        # future's result (re-raising its exception if any).
        self._result_started.set()
        return self._future.result()


class _RecordingApiClient:
    # [CLASS DOC] HELPER: Fake engine API client accepting any method name. Each call first waits
    # (up to five seconds, off the event loop) on the gate event registered for that method name,
    # raising a timeout error if the gate is not opened; then, if the method is the configured
    # failing one, raises a runtime error saying it failed (without logging); otherwise logs a
    # triple of engine index, method name and keyword arguments and returns a success mapping.
    def __init__(
        self,
        calls: list[tuple[int, str, dict]],
        engine_index: int,
        failing_method: str | None = None,
        gates: dict[str, threading.Event] | None = None,
    ):
        # [FUNCTION DOC] HELPER: Stores the shared call log, this engine's index, the optional
        # failing method name and the optional mapping of method name to gate event (default empty).
        self._calls = calls
        self._engine_index = engine_index
        self._failing_method = failing_method
        self._gates = gates if gates is not None else {}

    def __getattr__(self, name: str):
        # [FUNCTION DOC] HELPER: Returns an async function standing in for the named engine method.
        async def method(**kwargs):
            # [FUNCTION DOC] HELPER: Gate wait, optional failure and call logging as described in
            # the class documentation.
            if (gate := self._gates.get(name)) is not None and not await asyncio.to_thread(gate.wait, 5):
                raise TimeoutError(f"{name} gate timed out")
            if name == self._failing_method:
                raise RuntimeError(f"{name} failed")
            self._calls.append((self._engine_index, name, kwargs))
            return {"success": True}

        return method


def _make_engines(
    calls: list[tuple[int, str, dict]],
    *,
    failing_method: str | None = None,
    failing_engine_index: int = 0,
    later_engine_gates: dict[str, threading.Event] | None = None,
) -> list[_RecordingApiClient]:
    # [FUNCTION DOC] HELPER: Builds two recording engines sharing the given call log. Only the
    # engine at the failing engine index (default zero) gets the failing method, and only the last
    # engine gets the gate mapping. Returns the list of engines.
    return [
        _RecordingApiClient(
            calls,
            engine_index,
            failing_method if engine_index == failing_engine_index else None,
            later_engine_gates if engine_index == _ENGINE_COUNT - 1 else None,
        )
        for engine_index in range(_ENGINE_COUNT)
    ]


def _make_updater(engines: list[_RecordingApiClient], *, pause_generation_mode: str = "retract") -> WeightUpdater:
    # [FUNCTION DOC] HELPER: Builds a weight updater around a fake protocol (session support on,
    # sender, group name "test", the given engines, mock hooks that begin sync returning true,
    # send bucket, after base weights, finalize and after engines resumed) with the protocol
    # factory patched to return it. The iterator mock yields no weights and selects "all". Args
    # carry the pause mode (default retract) and LoRA equality check off. Model name is "qwen", no
    # quantization config, LoRA disabled. Returns the updater.
    protocol = SimpleNamespace(
        use_weight_update_session=True,
        needs_base_resync_for_lora=False,
        is_sender=True,
        group_name="test",
        rollout_engines=engines,
        required_placement=MagicMock(),
        supports_lora=False,
        begin_sync=lambda weight_version, iter_buckets: True,
        send_bucket=MagicMock(),
        after_base_weights=MagicMock(),
        finalize=MagicMock(),
        after_engines_resumed=MagicMock(),
    )
    iterator = MagicMock()
    iterator.iter_hf_weights.return_value = iter([])
    iterator.weight_update_selector = "all"
    args = Namespace(pause_generation_mode=pause_generation_mode, check_lora_weight_equal=False)
    with patch(f"{_UPDATER_MODULE}.get_weight_transfer_protocol", return_value=protocol):
        return WeightUpdater(
            args,
            [MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen",
            quantization_config=None,
            iterator_factory=lambda *a, **k: iterator,
            parallel_state=MagicMock(),
            is_lora=False,
        )


def _run(updater: WeightUpdater, *, rank: int = 0) -> None:
    # [FUNCTION DOC] HELPER: Runs one weight update on the updater with the distributed module and
    # gloo group patched and the mocked world rank set to the given rank (default zero). Exceptions
    # from the update propagate.
    with (
        patch(f"{_UPDATER_MODULE}.dist") as dist_mock,
        patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        dist_mock.get_rank.return_value = rank
        updater.update_weights()


def _phases(calls: list[tuple[int, str, dict]]) -> list[str]:
    # [FUNCTION DOC] HELPER: Collapses the call log into the sequence of method names with
    # consecutive repeats merged (so one phase on all engines counts once).
    phases: list[str] = []
    for _engine_index, name, _kwargs in calls:
        if not phases or phases[-1] != name:
            phases.append(name)
    return phases


def _engines_called(calls: list[tuple[int, str, dict]], method: str) -> list[int]:
    # [FUNCTION DOC] HELPER: Returns the sorted engine indices that logged a call to the method.
    return sorted(engine_index for engine_index, name, _kwargs in calls if name == method)


def _kwargs_of(calls: list[tuple[int, str, dict]], method: str) -> list[dict]:
    # [FUNCTION DOC] HELPER: Returns the keyword argument mappings of all logged calls to the
    # method, in log order.
    return [kwargs for _engine_index, name, kwargs in calls if name == method]


def _run_with_gated_later_engine(
    updater: WeightUpdater,
    calls: list[tuple[int, str, dict]],
    phases: list[str],
    gates: dict[str, threading.Event],
) -> None:
    # [FUNCTION DOC] HELPER: Runs a full weight update in a single worker thread while stepping
    # through the given ordered phases using the gates (which block the last engine). The async
    # submit used by the session module is wrapped so that, for every phase, the submission made
    # for the last engine (every second submission, as there are two engines) is returned as an
    # observed future signalling when the updater starts waiting on it. For each phase in order:
    # waits up to five seconds for that signal, asserts the update has not finished and that no
    # later phase has been called on any engine, then opens that phase's gate. Afterwards waits up
    # to five seconds for the update to finish (re-raising its error), always opening all gates in
    # a final cleanup. Finally asserts the number of submissions equals phases times engines.
    result_started = {phase: threading.Event() for phase in phases}
    submission_count = 0
    original_submit = async_utils.submit

    def observed_submit(coro):
        # [FUNCTION DOC] HELPER: Submits the coroutine through the original submit and counts it;
        # every second submission is wrapped in an observed future tied to the phase it belongs to.
        nonlocal submission_count
        future = original_submit(coro)
        submission_count += 1
        if submission_count % _ENGINE_COUNT == 0:
            phase = phases[submission_count // _ENGINE_COUNT - 1]
            return _ObservedFuture(future, result_started[phase])
        return future

    with patch(f"{_SESSION_MODULE}.async_utils.submit", side_effect=observed_submit):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            outcome = executor.submit(_run, updater)
            try:
                for phase_index, phase in enumerate(phases):
                    assert result_started[phase].wait(timeout=5)
                    assert not outcome.done()
                    assert not any(name in phases[phase_index + 1 :] for _engine_index, name, _kwargs in calls)
                    gates[phase].set()
                outcome.result(timeout=5)
            finally:
                for gate in gates.values():
                    gate.set()

    assert submission_count == len(phases) * _ENGINE_COUNT


class TestWeightUpdateSessionFrame:
    # [CLASS DOC] Tests of the engine session frame around a weight update; see the module
    # documentation for the expected phase order.
    """The session frame must pause, flush, open, close, publish the version and only then resume, on every engine."""

    @pytest.mark.parametrize("pause_generation_mode", ["retract", "abort"])
    def test_every_engine_walks_the_frame_in_order(self, pause_generation_mode):
        # [FUNCTION DOC] For retract and abort pause modes, runs an update with the last engine
        # gated on every phase and steps through all six phases. Asserts the collapsed phase
        # sequence is the full ordered list, every phase reached both engines, pause carried the
        # chosen mode, begin carried selector "all" with base sync true, and the version publish
        # carried weight version "1", for each engine.
        """Loading weights into a still-generating engine, or resuming a half-closed one, corrupts the rollout."""
        calls: list[tuple[int, str, dict]] = []
        phases = _PREPARE_PHASES + _FINALIZE_PHASES
        gates = {phase: threading.Event() for phase in phases}
        updater = _make_updater(
            _make_engines(calls, later_engine_gates=gates), pause_generation_mode=pause_generation_mode
        )

        _run_with_gated_later_engine(updater, calls, phases, gates)

        assert _phases(calls) == phases
        for method in phases:
            assert _engines_called(calls, method) == list(range(_ENGINE_COUNT))
        assert _kwargs_of(calls, "pause_generation") == [{"mode": pause_generation_mode}] * _ENGINE_COUNT
        assert _kwargs_of(calls, "begin_weight_update") == [{"selector": "all", "sync_base": True}] * _ENGINE_COUNT
        assert _kwargs_of(calls, "update_weight_version") == [{"weight_version": "1"}] * _ENGINE_COUNT

    def test_in_place_pause_mode_skips_the_flush(self):
        # [FUNCTION DOC] In in-place pause mode the phase sequence omits the flush cache phase and
        # pause carried the in-place mode on both engines.
        """in_place pause keeps the running requests, so their cache must survive."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls), pause_generation_mode="in_place")

        _run(updater)

        assert _phases(calls) == ["pause_generation", "begin_weight_update"] + _FINALIZE_PHASES
        assert _kwargs_of(calls, "pause_generation") == [{"mode": "in_place"}] * _ENGINE_COUNT

    def test_a_failed_pause_opens_no_update_session(self):
        # [FUNCTION DOC] Engine zero fails the pause: the update raises a runtime error about the
        # pause failure, and the only logged phase is pause, logged by engine one only.
        """A session opened on an engine that never paused would load weights under generation."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls, failing_method="pause_generation"))

        with pytest.raises(RuntimeError, match="pause_generation failed"):
            _run(updater)

        assert _phases(calls) == ["pause_generation"]
        assert _engines_called(calls, "pause_generation") == [1]

    def test_a_failed_flush_opens_no_update_session(self):
        # [FUNCTION DOC] Engine zero fails the flush: runtime error raised; logged phases are pause
        # then flush only, and flush was logged by engine one only.
        """A session opened over a stale cache would serve tokens generated from the old weights."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls, failing_method="flush_cache"))

        with pytest.raises(RuntimeError, match="flush_cache failed"):
            _run(updater)

        assert _phases(calls) == ["pause_generation", "flush_cache"]
        assert _engines_called(calls, "flush_cache") == [1]

    def test_a_failed_begin_prevents_the_update_from_starting(self):
        # [FUNCTION DOC] Engine one fails beginning the session: runtime error raised; the logged
        # phases are exactly the prepare phases, begin was logged by engine zero only, and the
        # protocol's send bucket was never called (no weights transferred).
        """A begin failure must escape instead of letting weight transfer proceed with a closed engine."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls, failing_method="begin_weight_update", failing_engine_index=1))

        with pytest.raises(RuntimeError, match="begin_weight_update failed"):
            _run(updater)

        assert _phases(calls) == _PREPARE_PHASES
        assert _engines_called(calls, "begin_weight_update") == [0]
        updater.protocol.send_bucket.assert_not_called()

    def test_a_failed_session_close_neither_publishes_the_version_nor_resumes(self):
        # [FUNCTION DOC] Engine zero fails ending the session: runtime error raised; logged phases
        # are the prepare phases plus end only (no version publish, no continue), logged by engine
        # one only.
        """An engine that resumed without a post-load pass would serve packed weights."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls, failing_method="end_weight_update"))

        with pytest.raises(RuntimeError, match="end_weight_update failed"):
            _run(updater)

        assert _phases(calls) == _PREPARE_PHASES + ["end_weight_update"]
        assert _engines_called(calls, "end_weight_update") == [1]

    def test_a_failed_version_publication_does_not_resume(self):
        # [FUNCTION DOC] Engine zero fails publishing the version: runtime error raised; logged
        # phases stop after the version publish (no continue generation), which only engine one
        # logged.
        """An engine resuming under a version it never acknowledged would mislabel its samples."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls, failing_method="update_weight_version"))

        with pytest.raises(RuntimeError, match="update_weight_version failed"):
            _run(updater)

        assert _phases(calls) == _PREPARE_PHASES + ["end_weight_update", "update_weight_version"]
        assert _engines_called(calls, "update_weight_version") == [1]

    def test_non_source_rank_issues_no_requests(self):
        # [FUNCTION DOC] Running the update as rank one results in no engine calls at all.
        """Every rank runs the updater, but only rank 0 may drive the engines."""
        calls: list[tuple[int, str, dict]] = []
        updater = _make_updater(_make_engines(calls))

        _run(updater, rank=1)

        assert calls == []
