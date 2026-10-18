# [MODULE DOC]
# Unit tests for the controller's periodic cell tick sweep and for its disposal. Fake cells count
# ticks and can raise or sleep inside a tick; stub servers hold a cell mapping and count disposal;
# the controller is built without its constructor with just servers, a context lock, an empty
# watcher disposer list and no ticker. Covers: every cell of every server ticked, failures and
# wedged cells not stopping the sweep, per-failure error logging, picking up newly added cells,
# and disposal stopping the ticker and every server.
from __future__ import annotations

import asyncio
import logging

from types import SimpleNamespace

from miles.ray.rollout import inference_controller as inference_controller_module
from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.context_lock import ContextLock
from miles.utils.misc import SimpleTicker


class _RecordingCell:
    # [CLASS DOC] HELPER: Fake server cell whose tick counts calls, then optionally raises a
    # configured error or sleeps for a configured delay, and counts a finished tick only when it
    # completes normally. It is never pending-weights-or-serving and has a meta with a cell id.
    def __init__(self, *, error: Exception | None = None, delay: float = 0.0, cell_id: str = "cell"):
        # [FUNCTION DOC] HELPER: Stores the optional error, the delay in seconds and a meta object
        # with the cell id; tick and finished counters start at zero.
        self.tick_count = 0
        self.finished_count = 0
        self.meta = SimpleNamespace(cell_id=cell_id)
        self.is_pending_weights_or_serving = False
        self._error = error
        self._delay = delay

    async def tick(self) -> None:
        # [FUNCTION DOC] HELPER: Counts the tick; raises the configured error if any; otherwise
        # sleeps for the delay if nonzero and counts a finished tick.
        self.tick_count += 1
        if self._error is not None:
            raise self._error
        if self._delay:
            await asyncio.sleep(self._delay)
        self.finished_count += 1


class _StubServer:
    # [CLASS DOC] HELPER: Minimal server stand-in with a mutable cell mapping and a disposal that
    # counts the call and clears the mapping.
    def __init__(self, server_cells: dict):
        # [FUNCTION DOC] HELPER: Stores the shared cell mapping and a dispose counter of zero.
        self.server_cells = server_cells
        self.dispose_count = 0

    async def dispose(self) -> None:
        # [FUNCTION DOC] HELPER: Counts the call and removes all cells from the mapping.
        self.dispose_count += 1
        self.server_cells.clear()


def _make_controller(servers: dict) -> InferenceController:
    # [FUNCTION DOC] HELPER: Builds a controller without running its constructor, with the given
    # servers mapping, a fresh context lock, an empty watcher disposer list and no ticker.
    controller = InferenceController.__new__(InferenceController)
    controller.servers = servers
    controller.context_lock = ContextLock("InferenceController")
    controller._watcher_disposers = []
    controller._ticker = None
    return controller


def _start_ticker(controller: InferenceController) -> None:
    # [FUNCTION DOC] HELPER: Installs and starts a simple ticker on the controller that repeatedly
    # runs the controller's private tick-cells sweep with a zero second interval.
    controller._ticker = SimpleTicker(controller._tick_cells, interval_seconds=0.0)


class TestTickCells:
    # [CLASS DOC] Tests the controller's tick-cells sweep, which ticks every cell of every server
    # (re-reading the cell mappings each time), bounds each tick by a timeout constant, logs each
    # failure with the cell id and traceback, and never lets one failure end the sweep early.
    async def test_it_drives_every_cell_of_every_server(self):
        # [FUNCTION DOC] Three cells across two servers; one sweep ticks each exactly once.
        """A cell only makes progress when ticked, so no server may be left out of the sweep."""
        first, second, third = _RecordingCell(), _RecordingCell(), _RecordingCell()
        controller = _make_controller(
            {"default": _StubServer({"a": first, "b": second}), "frozen": _StubServer({"c": third})}
        )

        await controller._tick_cells()

        assert [cell.tick_count for cell in (first, second, third)] == [1, 1, 1]

    async def test_one_failing_cell_does_not_let_its_siblings_escape_the_sweep(self):
        # [FUNCTION DOC] One cell raises immediately and a sibling takes 20 ms. The sweep does not
        # raise and does not return before the slow sibling finished its tick.
        """A sweep that returns early would release the lock while sibling ticks still mutate state."""
        broken = _RecordingCell(error=RuntimeError("cell exploded"), cell_id="broken")
        slow = _RecordingCell(delay=0.02, cell_id="slow")
        controller = _make_controller({"default": _StubServer({"a": broken, "b": slow})})

        await controller._tick_cells()

        assert slow.finished_count == 1

    async def test_a_wedged_cell_cannot_stall_the_sweep_forever(self, monkeypatch):
        # [FUNCTION DOC] A cell whose tick sleeps 60 seconds and a healthy cell, with the per-tick
        # timeout constant set to 10 ms. The sweep returns; the wedged cell never finished and the
        # healthy cell finished once.
        """A hung engine holds the controller lock for as long as its tick runs, so it must be bounded."""
        wedged = _RecordingCell(delay=60.0, cell_id="wedged")
        healthy = _RecordingCell(cell_id="healthy")
        controller = _make_controller({"default": _StubServer({"a": wedged, "b": healthy})})
        monkeypatch.setattr(inference_controller_module, "CELL_TICK_TIMEOUT_SECONDS", 0.01)

        await controller._tick_cells()

        assert wedged.finished_count == 0
        assert healthy.finished_count == 1

    async def test_a_cell_added_after_the_loop_started_is_picked_up(self):
        # [FUNCTION DOC] Starts the ticker over an empty server, adds a cell after 10 ms, waits 20
        # ms, disposes the controller, and asserts the late cell was ticked at least once.
        """Cells appear from reconcile long after startup, so the sweep must re-read the bookkeeping."""
        srv = _StubServer({})
        controller = _make_controller({"default": srv})

        _start_ticker(controller)
        await asyncio.sleep(0.01)
        late = _RecordingCell()
        srv.server_cells["late"] = late
        await asyncio.sleep(0.02)
        await controller.dispose()

        assert late.tick_count > 0

    async def test_tick_cells_logs_each_failure_with_cell_id_and_traceback(self, caplog, monkeypatch):
        # [FUNCTION DOC] One cell raises a runtime error and another is wedged (timeout constant
        # 10 ms). Captures error-level logs of one sweep and asserts exactly two records starting
        # with the ticking-cell prefix, in order: "Ticking cell broken failed" and "Ticking cell
        # wedged failed". The first carries the original error object, the second a timeout error,
        # and both have a traceback attached.
        """The log line is the only signal that a cell is stuck, and it has to name the cell and carry the cause."""
        error = RuntimeError("cell exploded")
        broken = _RecordingCell(error=error, cell_id="broken")
        wedged = _RecordingCell(delay=60.0, cell_id="wedged")
        controller = _make_controller({"default": _StubServer({"a": broken, "b": wedged})})
        monkeypatch.setattr(inference_controller_module, "CELL_TICK_TIMEOUT_SECONDS", 0.01)

        with caplog.at_level(logging.ERROR):
            await controller._tick_cells()

        failures = [record for record in caplog.records if record.message.startswith("Ticking cell ")]
        assert [record.message for record in failures] == [
            "Ticking cell broken failed",
            "Ticking cell wedged failed",
        ]
        assert failures[0].exc_info[1] is error
        assert isinstance(failures[1].exc_info[1], TimeoutError)
        assert all(record.exc_info[2] is not None for record in failures)

    async def test_the_sweep_keeps_running_after_one_cell_raises(self):
        # [FUNCTION DOC] Starts the ticker over a raising cell and a healthy cell, waits 20 ms,
        # disposes, and asserts both cells were ticked more than once (the loop kept going).
        """One wedged engine must not stop every other cell from making progress."""
        broken, healthy = _RecordingCell(error=RuntimeError("cell exploded")), _RecordingCell()
        controller = _make_controller({"default": _StubServer({"a": broken, "b": healthy})})

        _start_ticker(controller)
        await asyncio.sleep(0.02)
        await controller.dispose()

        assert broken.tick_count > 1
        assert healthy.tick_count > 1


class TestControllerDisposal:
    # [CLASS DOC] Tests the controller's dispose: it stops the ticker and disposes every server.
    async def test_dispose_stops_the_ticker(self):
        # [FUNCTION DOC] Runs the ticker for 20 ms, disposes, notes the tick count, waits another
        # 20 ms and asserts no more ticks happened.
        """A surviving loop would keep dialing engines after the controller is gone."""
        cell = _RecordingCell()
        controller = _make_controller({"default": _StubServer({"a": cell})})

        _start_ticker(controller)
        await asyncio.sleep(0.02)
        await controller.dispose()
        ticks_after_dispose = cell.tick_count
        await asyncio.sleep(0.02)

        assert cell.tick_count == ticks_after_dispose

    async def test_dispose_tears_down_every_server_so_no_cell_keeps_probing(self):
        # [FUNCTION DOC] Disposes a controller with two stub servers and asserts each server was
        # disposed exactly once.
        """A cell health checker keeps calling /health_generate unless dispose reaches its cell."""
        first, second = _StubServer({"a": _RecordingCell()}), _StubServer({"b": _RecordingCell()})
        controller = _make_controller({"default": first, "frozen": second})

        await controller.dispose()

        assert (first.dispose_count, second.dispose_count) == (1, 1)

    async def test_dispose_without_a_running_ticker_is_harmless(self):
        # [FUNCTION DOC] Disposes a controller with no servers and no ticker without error and
        # asserts the ticker attribute is still unset.
        """debug_train_only never starts the ticker, and teardown still has to work."""
        controller = _make_controller({})

        await controller.dispose()

        assert controller._ticker is None
