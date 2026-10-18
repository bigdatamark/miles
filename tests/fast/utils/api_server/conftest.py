# [MODULE DOC] HELPER: Shared mocks and pytest fixtures for the api server tests.
# Provides in-memory stand-ins for the pieces the api server talks to: a mock cell handler (with
# per-cell mutable state and call counters), gated variants used to prove concurrent polling, a
# mock remote-call wrapper imitating a Ray actor method, mock inference and trainer controllers, a
# mock worker manager and stop-cell controller, builders for cell summaries and a mock trainer
# controller, and fixtures that assemble a cell registry and an in-process HTTP client around the
# server application. Nothing here starts real servers, processes or GPUs.
from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest

from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.models import Cell, CellCondition, CellSpec, CellStatus
from miles.utils.ft_utils.api_server.registry import _CellRegistry
from miles.utils.ft_utils.api_server.server import _create_api_app
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.worker_provider.base import CellInfo


class MockCellState:
    # [CLASS DOC] HELPER: Mutable record of one fake cell held by the mock handler: phase string,
    # list of condition dictionaries, suspended flag, optional errors to raise on suspend and on
    # resume, a workers hash, and counters of how many times suspend and resume succeeded.
    # KNOWN ISSUE: the workers hash stored here is never read; the mock handler always reports the
    # fixed hash "pseudo-hash-0" regardless of this value.
    def __init__(
        self,
        *,
        phase: str = "Running",
        conditions: list[dict[str, str | None]] | None = None,
        is_suspended: bool = False,
        suspend_error: Exception | None = None,
        resume_error: Exception | None = None,
        workers_hash: str = "pseudo-hash-1",
    ) -> None:
        # [FUNCTION DOC] HELPER: Store the keyword-only settings. Defaults: phase Running; when no
        # conditions are given (or the list is empty) two conditions, Allocated true and Healthy
        # true; not suspended; no errors; workers hash "pseudo-hash-1". Both call counters start at 0.
        self.phase = phase
        self.conditions = conditions or [
            {"type": "Allocated", "status": "True"},
            {"type": "Healthy", "status": "True"},
        ]
        self.is_suspended = is_suspended
        self.suspend_error = suspend_error
        self.resume_error = resume_error
        self.workers_hash = workers_hash
        self.suspend_calls: int = 0
        self.resume_calls: int = 0


class MockHandler(_CellHandler):
    # [CLASS DOC] HELPER: In-memory replacement for the real cell handler. Cells live in a mapping
    # from cell id to a mock cell state; suspend and resume mutate that state (phase and
    # conditions flip between Running/healthy and Suspended/unhealthy). Fault injection is recorded
    # only when enabled by a flag, and can be made to fail with a configured error.
    def __init__(self, cell_type: str) -> None:
        # [FUNCTION DOC] HELPER: Remember the cell type, start with no cells and an empty record of
        # injected faults (tuples of cell id, failure mode, sub index), fault injection unsupported,
        # and no injection error. Deliberately does not call the base class initializer.
        self._cell_type = cell_type
        self.cells: dict[str, MockCellState] = {}
        self.injected: list[tuple[str, FailureMode, int]] = []
        self.supports_inject_fault = False
        self.inject_fault_error: Exception | None = None

    @property
    def cell_type(self) -> str:
        # [FUNCTION DOC] HELPER: Property returning the cell type given at construction.
        return self._cell_type

    def add(self, cell_id: str = "0", **overrides) -> MockCellState:
        # [FUNCTION DOC] HELPER: Create a mock cell state from the keyword overrides, register it
        # under the cell id (default "0", replacing any existing one) and return it so tests can
        # inspect its counters.
        state = MockCellState(**overrides)
        self.cells[cell_id] = state
        return state

    async def list_cell_ids(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Return the registered cell ids in insertion order.
        return list(self.cells)

    async def list_cells(self) -> list[Cell]:
        # [FUNCTION DOC] HELPER: Fetch all cells concurrently. Lists the ids, then gathers the
        # per-cell lookups; returns the cells in id order.
        cell_ids = await self.list_cell_ids()
        return list(await asyncio.gather(*(self.get_cell(cell_id) for cell_id in cell_ids)))

    async def get_cell(self, cell_id: str) -> Cell:
        # [FUNCTION DOC] HELPER: Render the stored state as a cell document. Metadata comes from
        # the base class metadata computation, the spec carries the suspended flag, and the status
        # carries the stored phase and conditions. Raises a key error for an unknown id.
        # KNOWN ISSUE: the status workers hash is hard-coded to "pseudo-hash-0" instead of using
        # the hash stored in the cell state.
        state = self.cells[cell_id]
        return Cell(
            metadata=self._compute_metadata(cell_id),
            spec=CellSpec(suspend=state.is_suspended),
            status=CellStatus(
                phase=state.phase,
                conditions=[CellCondition(**c) for c in state.conditions],
                workers_hash="pseudo-hash-0",
            ),
        )

    async def suspend(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Raise the configured suspend error if one is set (checked before any
        # change and not counted). Otherwise count the call, mark the cell suspended, set the phase to
        # Suspended and the conditions to Allocated false and Healthy false. Called on an already
        # suspended cell it still counts and re-applies. Raises a key error for an unknown id.
        state = self.cells[cell_id]
        if state.suspend_error:
            raise state.suspend_error
        state.suspend_calls += 1
        state.is_suspended = True
        state.phase = "Suspended"
        state.conditions = [
            {"type": "Allocated", "status": "False"},
            {"type": "Healthy", "status": "False"},
        ]

    async def resume(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Mirror of suspend: raise the configured resume error if set;
        # otherwise count the call, clear the suspended flag, set the phase to Running and the
        # conditions to Allocated true and Healthy true. Raises a key error for an unknown id.
        state = self.cells[cell_id]
        if state.resume_error:
            raise state.resume_error
        state.resume_calls += 1
        state.is_suspended = False
        state.phase = "Running"
        state.conditions = [
            {"type": "Allocated", "status": "True"},
            {"type": "Healthy", "status": "True"},
        ]

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        # [FUNCTION DOC] HELPER: Raise a not-implemented error naming the class unless injection
        # support was enabled; raise the configured injection error if one is set; otherwise append
        # (cell id, mode, sub index) to the injected record. The cell id is not checked for existence.
        if not self.supports_inject_fault:
            raise NotImplementedError(f"{type(self).__name__} does not support fault injection")
        if self.inject_fault_error is not None:
            raise self.inject_fault_error
        self.injected.append((cell_id, mode, sub_index))


class MockGatedHandler(MockHandler):
    # [CLASS DOC] HELPER: Mock handler whose cell listing blocks until an externally controlled
    # event is set; used to show a stalled handler does not delay polling of the others.
    def __init__(self, cell_type: str, *, gate: asyncio.Event) -> None:
        # [FUNCTION DOC] HELPER: Initialize as the parent for the given type and keep the gate event.
        super().__init__(cell_type)
        self._gate = gate

    async def list_cells(self) -> list[Cell]:
        # [FUNCTION DOC] HELPER: Wait until the gate event is set, then return the parent listing.
        await self._gate.wait()
        return await super().list_cells()


class MockGateOpeningHandler(MockHandler):
    # [CLASS DOC] HELPER: Mock handler that sets a shared event when its cells are listed; the
    # counterpart of the gated handler that waits on the same event.
    def __init__(self, cell_type: str, *, gate: asyncio.Event) -> None:
        # [FUNCTION DOC] HELPER: Initialize as the parent for the given type and keep the gate event.
        super().__init__(cell_type)
        self._gate = gate

    async def list_cells(self) -> list[Cell]:
        # [FUNCTION DOC] HELPER: Set the gate event first, then return the parent listing.
        self._gate.set()
        return await super().list_cells()


class MockRemoteCall:
    # [CLASS DOC] HELPER: Imitates a Ray actor method handle, exposing a remote method that returns
    # an already resolved future. Records every call's positional and keyword arguments, optionally
    # runs a side-effect function with the same arguments, and produces the result either from a
    # factory function called with the arguments or from a fixed return value.
    def __init__(
        self,
        return_value: object,
        effect: Callable[..., None] | None = None,
        factory: Callable[..., object] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Store the fixed return value, the optional side-effect function
        # and the optional result factory (the factory wins over the fixed value), and start with
        # an empty list of recorded calls (each a pair of argument tuple and keyword mapping).
        self._return_value = return_value
        self._effect = effect
        self._factory = factory
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args: object, **kwargs: object) -> asyncio.Future[object]:
        # [FUNCTION DOC] HELPER: Record the call, run the side-effect function if any (its
        # exceptions propagate synchronously), then create a future on the current event loop that is
        # already completed with the factory result (or the fixed value) and return it for awaiting.
        # Must be called where an event loop is available.
        self.calls.append((args, kwargs))
        if self._effect is not None:
            self._effect(*args, **kwargs)
        future: asyncio.Future[object] = asyncio.get_event_loop().create_future()
        future.set_result(self._factory(*args, **kwargs) if self._factory is not None else self._return_value)
        return future


class MockInferenceController:
    # [CLASS DOC] HELPER: Fake inference controller holding a mapping from cell id to cell status
    # and counting how many times the statuses were requested.
    def __init__(self, statuses: dict[str, CellStatus] | None = None) -> None:
        # [FUNCTION DOC] HELPER: Copy the given statuses (default empty) and zero the call counter.
        self._statuses = dict(statuses or {})
        self.status_calls: int = 0

    async def get_cell_statuses(self) -> dict[str, CellStatus]:
        # [FUNCTION DOC] HELPER: Increment the call counter and return a shallow copy of the status
        # mapping (the status objects themselves are shared, not copied).
        self.status_calls += 1
        return dict(self._statuses)

    def observe_cell(self, cell_id: str, status: CellStatus) -> None:
        # [FUNCTION DOC] HELPER: Set or replace the status stored for the cell id, imitating the
        # controller observing a new generation.
        self._statuses[cell_id] = status


class MockWorkerManager:
    # [CLASS DOC] HELPER: Fake Ray worker manager holding a mapping from cell id to cell info.
    # Exposes remote-call style properties for listing cell infos, stopping cells and starting
    # cells, and records every call. Stop and start flip the alive flag of the affected cells.
    # Note: it has no fault-injection attribute; tests that need one assign it themselves.
    def __init__(self, summaries: dict[str, CellInfo] | None = None) -> None:
        # [FUNCTION DOC] HELPER: Copy the given summaries (default empty) and start with empty
        # logs of stopped cell id lists, started cell id lists, and keyword arguments of the
        # cell-info queries. KNOWN ISSUE: one test passes an empty list instead of a mapping, which
        # only works because an empty list copies to an empty mapping.
        self._summaries = dict(summaries or {})
        self.stopped_cells: list[list[str]] = []
        self.started_cells: list[list[str]] = []
        self.cell_info_calls: list[dict[str, object]] = []

    @property
    def get_cell_infos(self) -> MockRemoteCall:
        # [FUNCTION DOC] HELPER: Property returning a new remote-call mock for listing cell infos.
        # Each call records its keyword arguments in the manager's query log and returns a copy of
        # all summaries, or only those whose pool id is in the optional pool_ids keyword when given.
        def _filtered(**kwargs: object) -> dict[str, CellInfo]:
            # [FUNCTION DOC] HELPER: Result factory implementing the optional pool filter.
            pool_ids = kwargs.get("pool_ids")
            if pool_ids is None:
                return dict(self._summaries)
            return {cell_id: info for cell_id, info in self._summaries.items() if info.pool_id in pool_ids}

        return MockRemoteCall(
            None,
            effect=lambda **kwargs: self.cell_info_calls.append(kwargs),
            factory=_filtered,
        )

    @property
    def stop_cells(self) -> MockRemoteCall:
        # [FUNCTION DOC] HELPER: Property returning a remote-call mock that, given a list of cell
        # ids, appends the list to the stopped log and marks those cells not alive. Returns none.
        return MockRemoteCall(None, effect=lambda ids: self._record(self.stopped_cells, ids, suspended=True))

    @property
    def start_cells(self) -> MockRemoteCall:
        # [FUNCTION DOC] HELPER: Property returning a remote-call mock that, given a list of cell
        # ids, appends the list to the started log and marks those cells alive. Returns none.
        return MockRemoteCall(None, effect=lambda ids: self._record(self.started_cells, ids, suspended=False))

    def _record(self, log: list[list[str]], cell_ids: list[str], *, suspended: bool) -> None:
        # [FUNCTION DOC] HELPER: Append a copy of the id list to the given log and, for each id,
        # replace the stored summary with a copy whose alive flag is the negation of the suspended
        # argument. Raises a key error for an unknown id (after the log entry was already appended).
        # The worker names of the summary are left unchanged.
        log.append(list(cell_ids))
        for cell_id in cell_ids:
            previous = self._summaries[cell_id]
            self._summaries[cell_id] = previous.model_copy(update=dict(alive=not suspended))


class MockStopCellController:
    # [CLASS DOC] HELPER: Fake inference controller as resolved by the cell operations; forwards
    # stop and fault-injection requests to the mock worker manager.
    def __init__(self, worker_manager: MockWorkerManager) -> None:
        # [FUNCTION DOC] HELPER: Keep a reference to the mock worker manager.
        self._worker_manager = worker_manager

    async def stop_cell_between_weight_updates(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Stop the single cell by awaiting the manager's stop-cells call
        # with a one-element id list.
        await self._worker_manager.stop_cells.remote([cell_id])

    async def inject_fault_between_weight_updates(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        # [FUNCTION DOC] HELPER: Await the manager's inject-fault remote call with the cell id,
        # the failure mode's string value and the sub index passed as the worker-in-cell index.
        # The mock manager has no such attribute by default, so it fails with an attribute error
        # unless a test assigned one.
        await self._worker_manager.inject_fault.remote(
            cell_id,
            mode=mode.value,
            worker_in_cell_index=sub_index,
        )


def make_cell_summaries(
    *cell_ids: str, suspended: bool = False, workers_hash: str = "pseudo-hash-0"
) -> dict[str, CellInfo]:
    # [FUNCTION DOC] HELPER: Build a mapping from cell id to cell info for the given ids. The pool
    # id is the cell id with its last dash-separated segment removed; alive is the opposite of
    # suspended; worker names are empty when suspended, else a single name made of the cell id and
    # "-0"; the workers hash is the given one; metadata holds a model id of "default".
    return {
        cell_id: CellInfo(
            cell_id=cell_id,
            pool_id=cell_id.rsplit("-", 1)[0],
            alive=not suspended,
            worker_names=[] if suspended else [f"{cell_id}-0"],
            workers_hash=workers_hash,
            meta={"model_id": "default"},
        )
        for cell_id in cell_ids
    }


class MockTrainerCell:
    # [CLASS DOC] HELPER: Fake trainer cell as held by the trainer controller: exposes phase,
    # conditions and workers hash, and can render itself as a cell status. The controller builder
    # later attaches a cell index and a cell id to each instance.
    def __init__(
        self,
        *,
        phase: str = "Running",
        conditions: list[dict[str, str | None]] | None = None,
        workers_hash: str = "pseudo-hash-0",
    ) -> None:
        # [FUNCTION DOC] HELPER: Store phase (default Running), conditions (default Allocated true
        # and Healthy true when none or an empty list is given) and workers hash (default
        # "pseudo-hash-0").
        self._phase = phase
        self._conditions = conditions or [
            {"type": "Allocated", "status": "True"},
            {"type": "Healthy", "status": "True"},
        ]
        self.workers_hash = workers_hash

    @property
    def phase(self) -> str:
        # [FUNCTION DOC] HELPER: Property returning the stored phase.
        return self._phase

    @property
    def conditions(self) -> list[dict[str, str | None]]:
        # [FUNCTION DOC] HELPER: Property returning the stored list of condition dictionaries.
        return self._conditions

    def cell_status(self) -> CellStatus:
        # [FUNCTION DOC] HELPER: Build a cell status from the stored phase, conditions (each
        # dictionary turned into a condition object) and workers hash. Returns it.
        # KNOWN ISSUE: the local import of the condition and status models duplicates the
        # identical module-level import and is redundant.
        from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus

        return CellStatus(
            phase=self._phase,
            conditions=[CellCondition(**c) for c in self._conditions],
            workers_hash=self.workers_hash,
        )


def make_mock_controller(cells: list[MockTrainerCell], *, pool_id: str = "trainer-engine-actor") -> object:
    # [FUNCTION DOC] HELPER: Build a real trainer controller object without running its
    # initializer (created bare, imported lazily), then fill only the state the api server reads.
    # Each mock cell gets a cell index equal to its position and a cell id of pool id, dash, index
    # (mutating the passed cells). The controller gets a mapping from cell id to cell, the pool id
    # (default "trainer-engine-actor") and an independent-data-parallel quorum id of 0. Returns it.
    from miles.ray.train.group import TrainerController

    group = object.__new__(TrainerController)
    for cell_index, cell in enumerate(cells):
        cell.cell_index = cell_index
        cell.cell_id = f"{pool_id}-{cell_index}"
    group._cells_by_id = {cell.cell_id: cell for cell in cells}
    group._pool_id = pool_id
    group._indep_dp_quorum_id = 0
    return group


@pytest.fixture
def actor_handler() -> MockHandler:
    # [FUNCTION DOC] HELPER: Fixture returning an empty mock handler of cell type "actor".
    return MockHandler("actor")


@pytest.fixture
def rollout_handler() -> MockHandler:
    # [FUNCTION DOC] HELPER: Fixture returning an empty mock handler of cell type "rollout".
    return MockHandler("rollout")


@pytest.fixture
def registry(actor_handler: MockHandler, rollout_handler: MockHandler) -> _CellRegistry:
    # [FUNCTION DOC] HELPER: Fixture returning a cell registry over the actor and rollout mock
    # handlers, in that order.
    return _CellRegistry([actor_handler, rollout_handler])


@pytest.fixture
def async_client(registry: _CellRegistry) -> httpx.AsyncClient:
    # [FUNCTION DOC] HELPER: Fixture returning an asynchronous HTTP client wired directly to the
    # api server application built over the registry (in-process ASGI transport, base address
    # http://test); no network sockets are opened. The client is not closed by the fixture.
    app = _create_api_app(registry)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")
