# [MODULE DOC]
# Unit tests for the rollout server's fan-out of operations (offload, onload, check weights, abort
# all) to its cells: which cells are addressed (only those needing offload for memory operations,
# only those that are pending weights or serving for everything), that arguments reach every cell
# unchanged, how results are collected, and that a cell that never answers an abort is given up on
# after a timeout. Cells are recording fakes; the provider stub fails on address lookup.
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import rollout_server as rollout_server_module
from miles.ray.rollout.rollout_server import RolloutServer
from miles.utils.context_lock import ContextLock
from miles.utils.workers.worker_spec import NamedHostAndPorts


class _RecordingCell:
    # [CLASS DOC] HELPER: Fake server cell that records each operation (name plus keyword record)
    # and returns a result string naming the operation and cell id for offload, onload and check
    # weights (abort all returns nothing). Its meta carries the offload flag, cell id, one GPU per
    # engine and offset zero; the addressable flag drives the pending-weights-or-serving property.
    def __init__(self, *, cell_id: str, needs_offload: bool, addressable: bool = True):
        # [FUNCTION DOC] HELPER: Stores meta from the cell id and offload flag, sets the
        # pending-weights-or-serving flag to the addressable flag (default true), and creates the
        # empty call list.
        self.meta = SimpleNamespace(needs_offload=needs_offload, cell_id=cell_id, num_gpus_per_engine=1, gpu_offset=0)
        self.is_pending_weights_or_serving = addressable
        self.calls: list[tuple[str, dict]] = []

    async def offload(self, tags):
        # [FUNCTION DOC] HELPER: Records an offload call with the tags and returns the text
        # "offloaded-" followed by the cell id.
        self.calls.append(("offload", dict(tags=tags)))
        return f"offloaded-{self.meta.cell_id}"

    async def onload(self, tags):
        # [FUNCTION DOC] HELPER: Records an onload call with the tags and returns the text
        # "onloaded-" followed by the cell id.
        self.calls.append(("onload", dict(tags=tags)))
        return f"onloaded-{self.meta.cell_id}"

    async def check_weights(self, action, allow_quant_error, selector, skip_list):
        # [FUNCTION DOC] HELPER: Records a check-weights call with all four arguments and returns
        # the text "checked-" followed by the cell id.
        self.calls.append(
            (
                "check_weights",
                dict(action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list),
            )
        )
        return f"checked-{self.meta.cell_id}"

    async def abort_all(self):
        # [FUNCTION DOC] HELPER: Records an abort-all call with an empty record. Returns nothing.
        self.calls.append(("abort_all", {}))


class _StubProvider:
    # [CLASS DOC] HELPER: Stub provider that fails the test if asked to resolve worker addresses.
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] HELPER: Always raises an assertion error naming the worker.
        raise AssertionError(f"fanning out to the cells must not resolve {worker_name}")


def _make_server(cells: list[_RecordingCell], **overrides) -> RolloutServer:
    # [FUNCTION DOC] HELPER: Builds a real rollout server over the given fake cells (keyed by cell
    # id), with colocated arguments, a named context lock, the stub provider, and any extra keyword
    # overrides passed through to the constructor. Returns the server.
    return RolloutServer(
        server_cells={cell.meta.cell_id: cell for cell in cells},
        args=make_args(colocate=True),
        context_lock=ContextLock("InferenceController"),
        engine_provider=_StubProvider(),
        **overrides,
    )


class TestMemoryFanOut:
    # [CLASS DOC] Tests the server's offload and onload fan-out. All calls are made under the
    # server's context lock.
    async def test_only_the_cells_sharing_gpus_with_the_trainer_give_memory_back(self):
        # [FUNCTION DOC] One cell needing offload and one resident cell: offloading the server calls
        # offload only on the first; the resident cell sees no call.
        """A resident engine told to release would drop the weights nobody is going to reload."""
        colocated = _RecordingCell(cell_id="a", needs_offload=True)
        resident = _RecordingCell(cell_id="b", needs_offload=False)
        srv = _make_server([colocated, resident])

        async with srv.context_lock:
            await srv.offload()

        assert [name for name, _ in colocated.calls] == ["offload"]
        assert resident.calls == []

    @pytest.mark.parametrize("op", ["offload", "onload"])
    async def test_the_requested_tags_reach_every_cell_unchanged(self, op):
        # [FUNCTION DOC] Parametrized over offload and onload with three offloading cells. Calls the
        # operation on the server with the weights tag and asserts the returned list holds each
        # cell's result in order and each cell recorded exactly one call with that tag list.
        """Resuming the wrong tag set brings back a different slice of the engine's memory."""
        cells = [_RecordingCell(cell_id=str(i), needs_offload=True) for i in range(3)]
        srv = _make_server(cells)

        async with srv.context_lock:
            results = await getattr(srv, op)(tags=["weights"])

        assert results == [f"{op}ed-{i}" for i in range(3)]
        assert all(cell.calls == [(op, dict(tags=["weights"]))] for cell in cells)

    async def test_a_cell_without_an_address_yet_is_left_alone(self):
        # [FUNCTION DOC] One non-addressable offloading cell and one serving offloading cell:
        # offloading calls only the serving cell; the gated one sees no call.
        """Reconcile can add a gated cell mid-window; dialling it asserts inside the cell and
        takes down a weight update the cell was never part of."""
        gated = _RecordingCell(cell_id="gated", needs_offload=True, addressable=False)
        serving = _RecordingCell(cell_id="serving", needs_offload=True)
        srv = _make_server([gated, serving])

        async with srv.context_lock:
            await srv.offload()

        assert gated.calls == []
        assert [name for name, _ in serving.calls] == ["offload"]


class TestCheckWeightsFanOut:
    # [CLASS DOC] Tests the server's check-weights fan-out to addressable cells.
    async def test_every_addressable_cell_is_checked_with_the_same_arguments(self):
        # [FUNCTION DOC] Three cells (not offloading); a snapshot check with quantization error
        # allowed, selector lora and skip list x returns each cell's result in order and every cell
        # recorded exactly one call with those four arguments.
        """Narrowing this to one cell would verify one engine of N and call them all equal."""
        cells = [_RecordingCell(cell_id=str(i), needs_offload=False) for i in range(3)]
        srv = _make_server(cells)

        async with srv.context_lock:
            results = await srv.check_weights(
                action="snapshot", allow_quant_error=True, selector="lora", skip_list=["x"]
            )

        assert results == [f"checked-{i}" for i in range(3)]
        assert all(
            cell.calls
            == [
                (
                    "check_weights",
                    dict(action="snapshot", allow_quant_error=True, selector="lora", skip_list=["x"]),
                )
            ]
            for cell in cells
        )

    async def test_a_cell_without_an_address_yet_is_not_checked(self):
        # [FUNCTION DOC] A single non-addressable cell: the check returns an empty list and the cell
        # sees no call.
        """The check runs during the weight update window, which a gated cell has not entered."""
        gated = _RecordingCell(cell_id="gated", needs_offload=False, addressable=False)
        srv = _make_server([gated])

        async with srv.context_lock:
            assert await srv.check_weights(action="snapshot") == []

        assert gated.calls == []


class _NeverAnsweringCell(_RecordingCell):
    # [CLASS DOC] HELPER: Recording cell whose abort-all never completes in practice (sleeps for an
    # hour after recording the call).
    async def abort_all(self):
        # [FUNCTION DOC] HELPER: Records the abort-all call, then sleeps for 3600 seconds.
        self.calls.append(("abort_all", {}))
        await asyncio.sleep(3600)


class TestAbortFanOut:
    # [CLASS DOC] Tests the server's abort-all fan-out.
    async def test_every_addressable_cell_is_asked_to_abort(self):
        # [FUNCTION DOC] Two cells; aborting all through the server results in each cell recording
        # exactly one abort-all call.
        """A take-over aborts what the previous script left generating, on every engine it can still reach."""
        cells = [_RecordingCell(cell_id="a", needs_offload=False), _RecordingCell(cell_id="b", needs_offload=False)]
        srv = _make_server(cells)

        async with srv.context_lock:
            await srv.abort_all()

        assert all(cell.calls == [("abort_all", {})] for cell in cells)

    async def test_a_cell_that_never_answers_is_given_up_on(self, monkeypatch):
        # [FUNCTION DOC] Sets the module's abort-all timeout constant to 10 ms with one normal cell
        # and one never-answering cell; under the lock, aborting all raises a timeout error.
        """The abort runs under the controller's lock, so an engine that never answers would wedge every caller."""
        monkeypatch.setattr(rollout_server_module, "ABORT_ALL_TIMEOUT_SECONDS", 0.01)
        cells = [
            _RecordingCell(cell_id="a", needs_offload=False),
            _NeverAnsweringCell(cell_id="b", needs_offload=False),
        ]
        srv = _make_server(cells)

        async with srv.context_lock:
            with pytest.raises(TimeoutError):
                await srv.abort_all()
