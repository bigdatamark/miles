# [MODULE DOC]
# Unit tests for the rollout server's wait for the expected number of cells (the startup barrier).
# Colocated servers only need the cells to appear, disaggregated servers need every cell pending
# weights or serving, and in a colocated run a cell that does not share GPUs with the trainer (a
# dedicated eval cell) must also be up. Also covers no cells expected, more cells than expected,
# and timing out with an error message. Cells are tiny fakes with a mutable ready flag; polling
# delays are shrunk to one millisecond by an autouse fixture.
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import rollout_server as rollout_server_module
from miles.ray.rollout.rollout_server import RolloutServer
from miles.utils.context_lock import ContextLock
from miles.utils.workers.worker_spec import NamedHostAndPorts


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture setting the rollout server module's initial and
    # maximum wait-for-cells poll delays to 1 ms.
    monkeypatch.setattr(rollout_server_module, "WAIT_CELLS_INITIAL_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(rollout_server_module, "WAIT_CELLS_MAX_DELAY_SECONDS", 0.001)


class _FakeCell:
    # [CLASS DOC] HELPER: Fake server cell with a ready flag (default false) that doubles as its
    # pending-weights-or-serving state, and a meta carrying the needs-offload flag (default true).
    def __init__(self, *, ready: bool = False, needs_offload: bool = True):
        # [FUNCTION DOC] HELPER: Stores the ready flag and a meta object with the offload flag.
        self.ready = ready
        self.meta = SimpleNamespace(needs_offload=needs_offload)

    @property
    def is_pending_weights_or_serving(self) -> bool:
        # [FUNCTION DOC] HELPER: Property returning the current ready flag.
        return self.ready


class _StubProvider:
    # [CLASS DOC] HELPER: Stub provider that fails the test if any address is requested.
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] HELPER: Always raises an assertion error naming the worker.
        raise AssertionError(f"no cell in this module is ever addressed ({worker_name=})")


def _make_server(*, colocate: bool, init_expected_num_cells: int, cells: dict | None = None) -> RolloutServer:
    # [FUNCTION DOC] HELPER: Builds a rollout server over the given cell mapping (shared, so tests
    # can mutate it; empty if none), test arguments with the given colocate flag, a named context
    # lock, the stub provider and the given expected cell count. Returns the server.
    return RolloutServer(
        server_cells=cells if cells is not None else {},
        args=make_args(colocate=colocate),
        context_lock=ContextLock("InferenceController"),
        engine_provider=_StubProvider(),
        init_expected_num_cells=init_expected_num_cells,
    )


class TestWaitExpectedNumCellsWhenColocated:
    # [CLASS DOC] Tests the wait on a colocated server with offloading cells: only the number of
    # cells matters, not their readiness.
    async def test_cells_only_have_to_appear(self):
        # [FUNCTION DOC] Two expected cells, two present but not ready: the wait completes within
        # one second.
        """Colocated engines cannot load until the first weight update window, so readiness cannot be required."""
        srv = _make_server(colocate=True, init_expected_num_cells=2, cells={"a": _FakeCell(), "b": _FakeCell()})

        await asyncio.wait_for(srv.wait_init_expected_num_cells(), timeout=1)

    async def test_it_waits_while_cells_are_still_missing(self):
        # [FUNCTION DOC] Two expected, one present: the wait task is not done after a yield; adding
        # the second cell lets it finish within five seconds.
        """Starting a rollout with half the pool would run the first step on far too few engines."""
        cells: dict = {"a": _FakeCell()}
        srv = _make_server(colocate=True, init_expected_num_cells=2, cells=cells)

        task = asyncio.create_task(srv.wait_init_expected_num_cells())
        await asyncio.sleep(0)
        assert not task.done()

        cells["b"] = _FakeCell()
        await asyncio.wait_for(task, timeout=5)


class TestWaitExpectedNumCellsWhenDisaggregated:
    # [CLASS DOC] Tests the wait on a non-colocated server: every expected cell must be up.
    async def test_appearing_is_not_enough_the_engines_must_be_up(self):
        # [FUNCTION DOC] One expected cell present but not ready: the wait is still pending after a
        # yield (the task is then cancelled).
        """A cell that has not finished loading yet cannot serve the first rollout."""
        srv = _make_server(colocate=False, init_expected_num_cells=1, cells={"a": _FakeCell(ready=False)})

        task = asyncio.create_task(srv.wait_init_expected_num_cells())
        await asyncio.sleep(0)

        assert not task.done()
        task.cancel()

    async def test_it_returns_once_every_engine_is_up(self):
        # [FUNCTION DOC] One cell not ready; after a yield it is marked ready and the wait finishes
        # within five seconds.
        """This is the startup barrier that replaced the blocking wait inside cell startup."""
        cell = _FakeCell(ready=False)
        srv = _make_server(colocate=False, init_expected_num_cells=1, cells={"a": cell})

        task = asyncio.create_task(srv.wait_init_expected_num_cells())
        await asyncio.sleep(0)
        cell.ready = True
        await asyncio.wait_for(task, timeout=5)


class TestWaitExpectedNumCellsWithADedicatedEvalFleet:
    # [CLASS DOC] Tests the colocated wait when some cells do not share GPUs with the trainer
    # (needs offload false): those must be up, the others only need to appear.
    async def test_a_cell_outside_the_trainer_gpus_still_has_to_come_up(self):
        # [FUNCTION DOC] Colocated, one non-offloading eval cell that is not ready: the wait is
        # still pending after a yield (then cancelled).
        """A colocated run whose eval cells count as ready on arrival snapshots api clients that have no address yet."""
        srv = _make_server(
            colocate=True, init_expected_num_cells=1, cells={"eval": _FakeCell(ready=False, needs_offload=False)}
        )

        task = asyncio.create_task(srv.wait_init_expected_num_cells())
        await asyncio.sleep(0)

        assert not task.done()
        task.cancel()

    async def test_it_returns_once_the_eval_cell_is_up(self):
        # [FUNCTION DOC] Same setup; after the eval cell becomes ready the wait finishes within five
        # seconds.
        """The barrier is what makes the eval fleet see engines that are actually addressable."""
        cell = _FakeCell(ready=False, needs_offload=False)
        srv = _make_server(colocate=True, init_expected_num_cells=1, cells={"eval": cell})

        task = asyncio.create_task(srv.wait_init_expected_num_cells())
        await asyncio.sleep(0)
        cell.ready = True

        await asyncio.wait_for(task, timeout=5)

    async def test_a_mixed_pool_waits_for_the_eval_cell_but_not_for_the_deferred_ones(self):
        # [FUNCTION DOC] Two expected cells: a shared (offloading) cell that never becomes ready and
        # a non-ready eval cell. The wait is pending until the eval cell is ready, then completes
        # even though the shared cell stays not ready.
        """One colocated run holds both kinds of cell, so the wait cannot be decided per run."""
        eval_cell = _FakeCell(ready=False, needs_offload=False)
        srv = _make_server(
            colocate=True,
            init_expected_num_cells=2,
            cells={"shared": _FakeCell(ready=False, needs_offload=True), "eval": eval_cell},
        )

        task = asyncio.create_task(srv.wait_init_expected_num_cells())
        await asyncio.sleep(0)
        assert not task.done()

        eval_cell.ready = True
        await asyncio.wait_for(task, timeout=5)


class TestWaitExpectedNumCellsEdges:
    # [CLASS DOC] Tests edge cases of the wait: zero expected, surplus cells, and the timeout.
    async def test_a_model_without_cells_does_not_wait(self):
        # [FUNCTION DOC] Zero expected cells (disaggregated) completes within one second.
        """A server expecting nothing must not hold up startup."""
        srv = _make_server(colocate=False, init_expected_num_cells=0)

        await asyncio.wait_for(srv.wait_init_expected_num_cells(), timeout=1)

    async def test_more_cells_than_expected_do_not_hang_the_wait(self):
        # [FUNCTION DOC] One cell expected, three present (colocated): completes within one second.
        """An exact-match check would stall forever the moment the pool is bigger than planned."""
        srv = _make_server(
            colocate=True, init_expected_num_cells=1, cells={"a": _FakeCell(), "b": _FakeCell(), "c": _FakeCell()}
        )

        await asyncio.wait_for(srv.wait_init_expected_num_cells(), timeout=1)

    async def test_it_gives_up_instead_of_waiting_forever(self):
        # [FUNCTION DOC] One cell expected, none present, wait with a zero timeout: an error whose
        # message contains "Only 0/1 cells" is raised.
        """A pool that never comes up must surface as a failure rather than a silent hang."""
        srv = _make_server(colocate=True, init_expected_num_cells=1)

        with pytest.raises(Exception, match="Only 0/1 cells"):
            await srv.wait_init_expected_num_cells(timeout=0)
