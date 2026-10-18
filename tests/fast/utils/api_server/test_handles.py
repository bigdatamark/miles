# [MODULE DOC] Tests for the cell handler of the fault-tolerance api server (_CellHandler in
# miles.utils.ft_utils.api_server.handles) in its two flavors, the "actor" (trainer) handler and
# the "rollout" (inference engine) handler. Handlers combine two sources: the worker manager
# (owns processes, knows alive/suspended state, worker names and the workers hash of the current
# generation) and controllers (know cell statuses). The tests use the mocks from the package
# conftest and check: listing and lookup of cells, phase and condition derivation (Running,
# Pending, Suspended, health), delegation of suspend, resume and fault injection to the worker
# manager, listing cost (one remote read per listing), concurrent status fetching, and blanking
# of statuses that describe an older worker generation.
from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus, TriState
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.ray import RayCellOperations

from .conftest import (
    MockInferenceController,
    MockRemoteCall,
    MockStopCellController,
    MockTrainerCell,
    MockWorkerManager,
    make_cell_summaries,
    make_mock_controller,
)

_WORKERS_HASH = "pseudo-hash-0"
_PENDING_STATUS = CellStatus(
    phase="Pending", conditions=[CellCondition.allocated(TriState.TRUE)], workers_hash=_WORKERS_HASH
)
_SUSPENDED_STATUS = CellStatus(
    phase="Suspended", conditions=[CellCondition.allocated(TriState.FALSE)], workers_hash=_WORKERS_HASH
)


def _running_status(health: TriState, *, workers_hash: str = _WORKERS_HASH) -> CellStatus:
    # [FUNCTION DOC] HELPER: Build a Running cell status with two conditions, Allocated true and
    # Healthy derived from the given health tri-state, stamped with the given workers hash
    # (default the module's standard hash).
    return CellStatus(
        phase="Running",
        conditions=[CellCondition.allocated(TriState.TRUE), CellCondition.from_health_checker_status(health)],
        workers_hash=workers_hash,
    )


ACTOR_CELL_ID = "trainer-engine-actor-0"


def _make_actor_handler(
    *,
    cells: list[MockTrainerCell] | None = None,
    suspended: bool = False,
) -> tuple[_CellHandler, object, MockWorkerManager]:
    # [FUNCTION DOC] HELPER: Build an "actor" cell handler over a mock trainer controller and a
    # mock worker manager. The controller holds the given trainer cells (default a single default
    # mock trainer cell); the manager knows one cell with the module's actor cell id, suspended
    # when requested. The handler uses Ray-style cell operations over the manager and a stop-cell
    # controller, and pool id "trainer-engine-actor". Returns (handler, controller, manager).
    group = make_mock_controller(cells if cells is not None else [MockTrainerCell()])
    manager = MockWorkerManager(make_cell_summaries(ACTOR_CELL_ID, suspended=suspended))
    handler = _CellHandler(
        cell_type="actor",
        operations=RayCellOperations(
            worker_manager_handle=manager,
            resolve_inference_controller=lambda: MockStopCellController(manager),
        ),
        controllers=[group],
        pool_ids=["trainer-engine-actor"],
    )
    return handler, group, manager


class TestActorCellHandler:
    # [CLASS DOC] Tests of the handler flavor serving trainer (actor) cells.
    @pytest.mark.asyncio
    async def test_every_cell_of_the_group_is_listed(self) -> None:
        # [FUNCTION DOC] Asserts listing cell ids returns exactly the one actor cell id known to the
        # worker manager.
        """The api server addresses trainer cells by the manager's cell id."""
        handler, _group, _manager = _make_actor_handler()

        assert await handler.list_cell_ids() == [ACTOR_CELL_ID]

    def test_cell_type(self) -> None:
        # [FUNCTION DOC] Asserts the actor handler reports the cell type "actor".
        """The cell type is the api-server-visible label the ft controller filters on."""
        handler, _group, _manager = _make_actor_handler()

        assert handler.cell_type == "actor"

    @pytest.mark.asyncio
    async def test_a_running_cell_reports_the_controller_status(self) -> None:
        # [FUNCTION DOC] Fetches the actor cell and asserts its name is the cell id, its cell-type
        # label is "actor", it is not suspended and its phase is Running (taken from the controller).
        """The trainer group is the only place that knows a cell's own phase."""
        handler, _group, _manager = _make_actor_handler()

        cell = await handler.get_cell(ACTOR_CELL_ID)

        assert cell.metadata.name == ACTOR_CELL_ID
        assert cell.metadata.labels["miles.io/cell-type"] == "actor"
        assert cell.spec.suspend is False
        assert cell.status.phase == "Running"

    @pytest.mark.asyncio
    async def test_suspension_comes_from_the_worker_manager(self) -> None:
        # [FUNCTION DOC] With the manager reporting the cell suspended, asserts the fetched cell has
        # the suspend flag set and phase Suspended.
        """The manager owns the processes, so it alone knows a cell was suspended."""
        handler, _group, _manager = _make_actor_handler(suspended=True)

        cell = await handler.get_cell(ACTOR_CELL_ID)

        assert cell.spec.suspend is True
        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_a_cell_the_group_has_not_observed_yet_is_pending(self) -> None:
        # [FUNCTION DOC] Empties the mock controller's cell mapping (the manager still knows the
        # cell) and asserts the fetched cell's phase is Pending.
        """Between a manager restart and the next poll the group knows nothing about it."""
        handler, group, _manager = _make_actor_handler()
        group._cells_by_id = {}

        cell = await handler.get_cell(ACTOR_CELL_ID)

        assert cell.status.phase == "Pending"

    @pytest.mark.asyncio
    async def test_suspend_stops_the_cell_in_the_worker_manager(self) -> None:
        # [FUNCTION DOC] Suspends the actor cell and asserts the manager's stop log is exactly one
        # call with a one-element list holding the cell id.
        """The manager owns the processes, so suspension must go through it, not the controller."""
        handler, _group, manager = _make_actor_handler()

        await handler.suspend(ACTOR_CELL_ID)

        assert manager.stopped_cells == [[ACTOR_CELL_ID]]

    @pytest.mark.asyncio
    async def test_resume_starts_the_cell_in_the_worker_manager(self) -> None:
        # [FUNCTION DOC] Resumes the actor cell and asserts the manager's start log is exactly one
        # call with a one-element list holding the cell id.
        """Resume relaunches the cell, which reconcile then observes as a new generation."""
        handler, _group, manager = _make_actor_handler()

        await handler.resume(ACTOR_CELL_ID)

        assert manager.started_cells == [[ACTOR_CELL_ID]]

    @pytest.mark.asyncio
    async def test_injection_is_forwarded_to_the_worker_manager(self) -> None:
        # [FUNCTION DOC] Replaces the manager's inject-fault call with a recording remote-call mock,
        # injects a kill-signal fault at sub index 1 and asserts exactly one call was forwarded with
        # the cell id positionally and the keywords mode "sigkill" and worker-in-cell index 1.
        """The manager owns the actors, so it is the one that can crash them."""
        handler, _group, manager = _make_actor_handler()
        manager.injected = []
        manager.inject_fault = MockRemoteCall(None, effect=lambda *a, **kw: manager.injected.append((a, kw)))

        await handler.inject_fault(ACTOR_CELL_ID, mode=FailureMode.SIGKILL, sub_index=1)

        assert manager.injected == [((ACTOR_CELL_ID,), {"mode": "sigkill", "worker_in_cell_index": 1})]


ENGINE_CELL_ID = "inference-engine-0-0-0"


def _pool_ids_of(manager: MockWorkerManager) -> list[str]:
    # [FUNCTION DOC] HELPER: Return the sorted distinct pool ids of the manager's cell summaries.
    # Reads the manager's private summaries mapping directly.
    return sorted({summary.pool_id for summary in manager._summaries.values()})


def _make_rollout_handler(
    *,
    cell_id: str = ENGINE_CELL_ID,
    suspended: bool = False,
    health: TriState | None = TriState.TRUE,
    status: CellStatus | None = None,
) -> _CellHandler:
    # [FUNCTION DOC] HELPER: Build a "rollout" handler over a mock manager with one cell (given id,
    # default an engine cell id; suspended when requested) and a mock inference controller. The
    # controller's status for the cell is: the explicit status if given; else a Running status with
    # the given health (default true) if health is not none; else no status. A suspended cell with
    # no explicit status gets no controller status at all. The pool id is the cell id minus its
    # last dash segment. Returns the handler only.
    manager = MockWorkerManager(make_cell_summaries(cell_id, suspended=suspended))
    resolved = status if status is not None else (_running_status(health) if health is not None else None)
    if suspended and status is None:
        resolved = None
    controller = MockInferenceController({cell_id: resolved} if resolved is not None else {})
    return _CellHandler(
        cell_type="rollout",
        operations=RayCellOperations(
            worker_manager_handle=manager,
            resolve_inference_controller=lambda: MockStopCellController(manager),
        ),
        controllers=[controller],
        pool_ids=[cell_id.rsplit("-", 1)[0]],
    )


class TestRolloutCellHandler:
    # [CLASS DOC] Tests of the handler flavor serving inference-engine (rollout) cells.
    @pytest.mark.asyncio
    async def test_a_healthy_cell_is_reported_running(self) -> None:
        # [FUNCTION DOC] With a healthy running status, asserts the cell name, cell-type label
        # "rollout", phase Running, not suspended, and conditions exactly Allocated true then
        # Healthy true.
        """A serving engine that answers its probe is what the heal loop must leave alone."""
        handler = _make_rollout_handler()

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.metadata.name == "inference-engine-0-0-0"
        assert cell.metadata.labels["miles.io/cell-type"] == "rollout"
        assert cell.status.phase == "Running"
        assert cell.spec.suspend is False
        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.TRUE),
        ]

    @pytest.mark.asyncio
    async def test_a_failing_probe_is_reported_unhealthy(self) -> None:
        # [FUNCTION DOC] With a failed health probe, asserts the phase stays Running while the
        # conditions are Allocated true and Healthy false.
        """This is the signal the mini ft controller heals on."""
        handler = _make_rollout_handler(health=TriState.FALSE)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Running"
        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.FALSE),
        ]

    @pytest.mark.asyncio
    async def test_suspension_comes_from_the_controller_status(self) -> None:
        # [FUNCTION DOC] With the manager reporting the cell suspended and no controller status,
        # asserts the cell has the suspend flag and phase Suspended (not Pending).
        # KNOWN ISSUE: the test name says the controller status is the source, but the docstring
        # and the setup show the suspension comes from the worker manager.
        """Suspension is read off the worker manager, so a cell with no controller document still
        reports Suspended rather than falling through to pending."""
        handler = _make_rollout_handler(suspended=True)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.spec.suspend is True
        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_a_suspended_cell_reports_no_health(self) -> None:
        # [FUNCTION DOC] For a suspended cell asserts the only condition is Allocated false (no
        # Healthy condition).
        """Its engine is gone, so nothing may claim a health verdict for it."""
        handler = _make_rollout_handler(suspended=True)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert [(c.type, c.status) for c in cell.status.conditions] == [("Allocated", TriState.FALSE)]

    @pytest.mark.asyncio
    async def test_a_cell_the_controller_does_not_track_yet_carries_no_health(self) -> None:
        # [FUNCTION DOC] For an alive cell with no controller status, asserts phase Pending and the
        # only condition Allocated true.
        """Reconcile only hands the manager's cell to the controller a poll period later."""
        handler = _make_rollout_handler(health=None)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Pending"
        assert [(c.type, c.status) for c in cell.status.conditions] == [("Allocated", TriState.TRUE)]

    @pytest.mark.asyncio
    async def test_the_controllers_status_is_passed_through_verbatim(self) -> None:
        # [FUNCTION DOC] Gives the controller a Pending status with matching workers hash and
        # asserts the cell's status equals it exactly (no re-derivation of phase).
        """The handle renders what the controller computed; re-deriving phase here would let the two disagree."""
        handler = _make_rollout_handler(status=_PENDING_STATUS)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status == _PENDING_STATUS

    @pytest.mark.asyncio
    async def test_health_is_read_without_awaiting_the_controller(self) -> None:
        # [FUNCTION DOC] Builds a handler directly over a manager and a controller, fetches one cell
        # and asserts the controller's status request counter is exactly 1.
        # KNOWN ISSUE: the name and docstring talk about reading synchronously without awaiting, but
        # the assertion only checks that the status was requested once.
        """The api server serves from its own event loop, so the controller is read synchronously."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID))
        controller = MockInferenceController()
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[controller],
            pool_ids=_pool_ids_of(manager),
        )

        await handler.get_cell(ENGINE_CELL_ID)

        assert controller.status_calls == 1

    @pytest.mark.asyncio
    async def test_suspend_stops_the_cell_in_the_worker_manager(self) -> None:
        # [FUNCTION DOC] Suspends an engine cell and asserts the manager's stop log is one call
        # holding the cell id.
        """The manager owns the processes, so healing goes through it, not the controller."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID))
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[MockInferenceController()],
            pool_ids=_pool_ids_of(manager),
        )

        await handler.suspend(ENGINE_CELL_ID)

        assert manager.stopped_cells == [[ENGINE_CELL_ID]]

    @pytest.mark.asyncio
    async def test_resume_starts_the_cell_in_the_worker_manager(self) -> None:
        # [FUNCTION DOC] Resumes a suspended engine cell and asserts the manager's start log is one
        # call holding the cell id.
        """Resume relaunches the cell, which reconcile then observes as a new generation."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID, suspended=True))
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[MockInferenceController()],
            pool_ids=_pool_ids_of(manager),
        )

        await handler.resume(ENGINE_CELL_ID)

        assert manager.started_cells == [[ENGINE_CELL_ID]]

    @pytest.mark.asyncio
    async def test_a_suspended_cell_reports_suspended_afterwards(self) -> None:
        # [FUNCTION DOC] Suspends a healthy running cell, fetches it again and asserts the phase is
        # now Suspended (mock manager flips the alive flag).
        """The heal loop reads back the status it just asked for."""
        handler = _make_rollout_handler()

        await handler.suspend(ENGINE_CELL_ID)
        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_a_resumed_cell_is_pending_without_health_until_reconcile_sees_it(self) -> None:
        # [FUNCTION DOC] Resumes a suspended cell (controller has no status) and asserts phase
        # Pending, the suspend flag cleared, and only the Allocated true condition.
        """The new generation is still gated, so calling it Running would be the old process' verdict."""
        handler = _make_rollout_handler(suspended=True)

        await handler.resume(ENGINE_CELL_ID)
        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Pending"
        assert cell.spec.suspend is False
        assert [(c.type, c.status) for c in cell.status.conditions] == [("Allocated", TriState.TRUE)]

    @pytest.mark.asyncio
    async def test_the_status_comes_back_once_reconcile_observes_the_new_generation(self) -> None:
        # [FUNCTION DOC] Starts with a suspended cell and a stale Suspended status in the controller,
        # resumes the cell, then lets the controller observe a healthy running status with the same
        # hash. Asserts the fetched cell is Running with Allocated true and Healthy true.
        """The published document must recover on its own, without another suspend or resume."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID, suspended=True))
        controller = MockInferenceController({ENGINE_CELL_ID: _SUSPENDED_STATUS})
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[controller],
            pool_ids=_pool_ids_of(manager),
        )
        await handler.resume(ENGINE_CELL_ID)

        controller.observe_cell(ENGINE_CELL_ID, _running_status(TriState.TRUE))
        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Running"
        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.TRUE),
        ]

    def test_cell_type_is_rollout(self) -> None:
        # [FUNCTION DOC] Asserts the handler's cell type is "rollout".
        """The cell type is what lets a soak target engines without crashing trainer cells."""
        handler = _make_rollout_handler(cell_id="inference-engine-0-0-3")
        assert handler.cell_type == "rollout"

    async def test_only_the_engine_specs_are_listed(self) -> None:
        # [FUNCTION DOC] The manager knows an engine cell and a router cell; the handler is limited
        # to the engine pool id. Asserts only the engine cell id is listed.
        """Routers and session servers are cells of the manager too, but not rollout cells."""
        manager = MockWorkerManager(
            {**make_cell_summaries("inference-engine-0-0-0"), **make_cell_summaries("miles-router-0")}
        )
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[MockInferenceController()],
            pool_ids=["inference-engine-0-0"],
        )

        assert await handler.list_cell_ids() == ["inference-engine-0-0-0"]

    async def test_a_suspended_cell_is_still_listed(self) -> None:
        # [FUNCTION DOC] Asserts a suspended engine cell still appears in the cell id listing.
        """A suspended cell that vanished from the listing could never be resumed."""
        handler = _make_rollout_handler(suspended=True)

        assert await handler.list_cell_ids() == [ENGINE_CELL_ID]

    async def test_listing_reads_its_sources_once_for_all_cells(self) -> None:
        # [FUNCTION DOC] With three cells, lists all cells and asserts three cells were returned and
        # the controller's status request counter is 1 (not one request per cell).
        """This listing is polled for the life of the run, so it must not scale in round trips."""
        manager = MockWorkerManager(make_cell_summaries("engine-a", "engine-b", "engine-c"))
        controller = MockInferenceController()
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[controller],
            pool_ids=_pool_ids_of(manager),
        )

        cells = await handler.list_cells()

        assert len(cells) == 3
        assert controller.status_calls == 1

    async def test_listing_fetches_worker_infos_once_for_all_cells(self) -> None:
        # [FUNCTION DOC] With three cells, lists all cells and asserts three cells were returned and
        # the manager received exactly one cell-info query, filtered by the pool id of the cells.
        """The worker manager lives behind ray, so one listing must cost one remote round trip."""
        manager = MockWorkerManager(make_cell_summaries("engine-a", "engine-b", "engine-c"))
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[MockInferenceController()],
            pool_ids=_pool_ids_of(manager),
        )

        cells = await handler.list_cells()

        assert len(cells) == 3
        assert manager.cell_info_calls == [{"pool_ids": ["engine"]}]


class TestRolloutCellHandlerInjectFault:
    # [CLASS DOC] Tests of fault injection through the rollout handler.
    @pytest.mark.asyncio
    async def test_injection_is_forwarded_to_the_worker_manager(self) -> None:
        # [FUNCTION DOC] Gives the manager a recording remote-call mock for fault injection, injects
        # a kill-signal fault at sub index 1 and asserts one recorded call with the cell id
        # positionally and keywords mode "sigkill" and worker-in-cell index 1.
        """The manager owns the actors, so it is the one that can crash them."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID))
        manager.inject_fault = MockRemoteCall(None)
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[MockInferenceController()],
            pool_ids=_pool_ids_of(manager),
        )

        await handler.inject_fault(ENGINE_CELL_ID, mode=FailureMode.SIGKILL, sub_index=1)

        assert manager.inject_fault.calls == [
            ((ENGINE_CELL_ID,), {"mode": "sigkill", "worker_in_cell_index": 1}),
        ]


class TestCellStatusGeneration:
    # [CLASS DOC] Tests of how the handler merges controller statuses and handles statuses that
    # describe a different (older) workers-hash generation than the live worker manager listing.
    async def test_fetches_controller_statuses_concurrently(self) -> None:
        # [FUNCTION DOC] Builds an actor handler with two custom controllers: the first blocks on an
        # event after recording its entry, the second records entry, signals an event, and returns
        # at once. Starts the handler's private status gathering as a task, waits up to one second
        # for the second controller to be entered (proving concurrency), asserts the task is still
        # pending, releases the first, and asserts the merged result holds both controllers'
        # statuses, the entry order was first then second, and the second was entered while the
        # first was still blocked.
        """Every controller status request starts before a blocked peer is released, and all results are merged."""
        entered: list[str] = []
        release_first = asyncio.Event()
        second_entered = asyncio.Event()
        first_status = _running_status(TriState.TRUE)
        second_status = _running_status(TriState.FALSE)

        class BlockingController:
            # [CLASS DOC] HELPER: Controller whose status request records "first" and then waits
            # for the release event before answering.
            async def get_cell_statuses(self) -> dict[str, CellStatus]:
                # [FUNCTION DOC] HELPER: Record entry, wait for the release event, return a mapping
                # with one status for the first trainer cell.
                entered.append("first")
                await release_first.wait()
                return {"trainer-a-0": first_status}

        class RecordingController:
            # [CLASS DOC] HELPER: Controller that records "second", signals that it was entered
            # and answers immediately.
            async def get_cell_statuses(self) -> dict[str, CellStatus]:
                # [FUNCTION DOC] HELPER: Record entry, set the entered event, return a mapping with
                # one status for the second trainer cell.
                entered.append("second")
                second_entered.set()
                return {"trainer-b-0": second_status}

        manager = MockWorkerManager([])
        handler = _CellHandler(
            cell_type="actor",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[BlockingController(), RecordingController()],
            pool_ids=[],
        )
        statuses_task = asyncio.create_task(handler._get_cell_statuses())

        try:
            await asyncio.wait_for(second_entered.wait(), timeout=1)
            fetched_concurrently = True
        except TimeoutError:
            fetched_concurrently = False

        assert not statuses_task.done()

        release_first.set()
        assert await statuses_task == {"trainer-a-0": first_status, "trainer-b-0": second_status}
        assert fetched_concurrently
        assert entered == ["first", "second"]

    @pytest.mark.asyncio
    async def test_a_status_about_an_older_generation_carries_no_verdict(self) -> None:
        # [FUNCTION DOC] The controller's status carries an old workers hash while the manager
        # reports the standard one. Asserts the published conditions are Allocated unknown and
        # Healthy unknown and the status hash is the manager's current one.
        """The two sources are polled apart, so a status from the previous process must not be published as this one's."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID))
        stale = _running_status(TriState.TRUE, workers_hash="pseudo-hash-old")
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[MockInferenceController({ENGINE_CELL_ID: stale})],
            pool_ids=_pool_ids_of(manager),
        )

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.UNKNOWN),
            ("Healthy", TriState.UNKNOWN),
        ]
        assert cell.status.workers_hash == _WORKERS_HASH

    @pytest.mark.asyncio
    async def test_a_status_about_the_current_generation_is_published_verbatim(self) -> None:
        # [FUNCTION DOC] With matching hashes and an unhealthy status, asserts conditions remain
        # Allocated true and Healthy false (no blanking).
        """Blanking a status whose generation matches would hide every real verdict."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID))
        current = _running_status(TriState.FALSE)
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[MockInferenceController({ENGINE_CELL_ID: current})],
            pool_ids=_pool_ids_of(manager),
        )

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.FALSE),
        ]

    @pytest.mark.asyncio
    async def test_a_suspended_cell_is_stamped_from_the_live_listing(self) -> None:
        # [FUNCTION DOC] A suspended cell with hash "gen-2" in the manager and a controller status
        # for "gen-1". Asserts phase Suspended and the status hash "gen-2" (stamped from the manager).
        """The suspended document is built here rather than read from the controller, so it needs its own stamp."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID, suspended=True, workers_hash="gen-2"))
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[
                MockInferenceController({ENGINE_CELL_ID: _running_status(TriState.TRUE, workers_hash="gen-1")})
            ],
            pool_ids=_pool_ids_of(manager),
        )

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Suspended"
        assert cell.status.workers_hash == "gen-2"

    @pytest.mark.asyncio
    async def test_a_cell_the_controller_does_not_track_yet_is_stamped_from_the_live_listing(self) -> None:
        # [FUNCTION DOC] An alive cell with hash "gen-2" and no controller status. Asserts phase
        # Pending and status hash "gen-2".
        """The synthesised pending status is about the generation the listing reports, not about no generation."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID, workers_hash="gen-2"))
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[MockInferenceController()],
            pool_ids=_pool_ids_of(manager),
        )

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Pending"
        assert cell.status.workers_hash == "gen-2"

    @pytest.mark.asyncio
    async def test_a_status_about_an_older_generation_keeps_its_phase_and_loses_every_condition(self) -> None:
        # [FUNCTION DOC] A stale Running status for "gen-1" with Allocated, Healthy and Serving
        # conditions while the manager is at "gen-2". Asserts the phase stays Running and all three
        # conditions (in order) turn Unknown.
        """A Serving condition left standing would witness a recovery that the replaced process never made."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID, workers_hash="gen-2"))
        stale = CellStatus(
            phase="Running",
            conditions=[
                CellCondition.allocated(TriState.TRUE),
                CellCondition.from_health_checker_status(TriState.TRUE),
                CellCondition.serving(TriState.TRUE),
            ],
            workers_hash="gen-1",
        )
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[MockInferenceController({ENGINE_CELL_ID: stale})],
            pool_ids=_pool_ids_of(manager),
        )

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Running"
        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.UNKNOWN),
            ("Healthy", TriState.UNKNOWN),
            ("Serving", TriState.UNKNOWN),
        ]

    @pytest.mark.asyncio
    async def test_a_status_about_an_older_generation_drops_the_reasons_behind_its_verdicts(self) -> None:
        # [FUNCTION DOC] A stale status whose conditions carry reasons and messages. Asserts after
        # blanking each condition is Unknown with reason and message cleared, except that the
        # health condition keeps its dedicated "HealthCheckUnknown" reason.
        """An Unknown verdict that still quotes the old process's failure reason reads as a verdict about the new one."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID, workers_hash="gen-2"))
        stale = CellStatus(
            phase="Running",
            conditions=[
                CellCondition(type="Allocated", status=TriState.TRUE, reason="Placed", message="on node 1"),
                CellCondition.from_health_checker_status(TriState.FALSE),
                CellCondition(type="Serving", status=TriState.FALSE, message="draining"),
            ],
            workers_hash="gen-1",
        )
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[MockInferenceController({ENGINE_CELL_ID: stale})],
            pool_ids=_pool_ids_of(manager),
        )

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert [(c.type, c.status, c.reason, c.message) for c in cell.status.conditions] == [
            ("Allocated", TriState.UNKNOWN, None, None),
            ("Healthy", TriState.UNKNOWN, "HealthCheckUnknown", None),
            ("Serving", TriState.UNKNOWN, None, None),
        ]

    @pytest.mark.asyncio
    async def test_blanking_a_stale_status_leaves_the_controllers_own_copy_untouched(self) -> None:
        # [FUNCTION DOC] Fetches a cell whose controller status is stale and asserts the original
        # status object still equals a freshly built identical stale status (not mutated in place).
        """The controller keeps serving this object to its own callers, so blanking it in place would corrupt them."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID, workers_hash="gen-2"))
        stale = _running_status(TriState.TRUE, workers_hash="gen-1")
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[MockInferenceController({ENGINE_CELL_ID: stale})],
            pool_ids=_pool_ids_of(manager),
        )

        await handler.get_cell(ENGINE_CELL_ID)

        assert stale == _running_status(TriState.TRUE, workers_hash="gen-1")

    @pytest.mark.asyncio
    async def test_a_listing_blanks_a_stale_status_the_same_way_a_lookup_does(self) -> None:
        # [FUNCTION DOC] Lists cells (one cell with a stale controller status) and asserts that the
        # listed cell has Unknown Allocated and Healthy conditions and the manager's current hash.
        """Clients poll the whole list, so a check applied only to the single-cell route publishes the stale verdict."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID, workers_hash="gen-2"))
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[
                MockInferenceController({ENGINE_CELL_ID: _running_status(TriState.TRUE, workers_hash="gen-1")})
            ],
            pool_ids=_pool_ids_of(manager),
        )

        [cell] = await handler.list_cells()

        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.UNKNOWN),
            ("Healthy", TriState.UNKNOWN),
        ]
        assert cell.status.workers_hash == "gen-2"

    @pytest.mark.asyncio
    async def test_the_verdict_returns_once_the_controller_observes_the_new_generation(self) -> None:
        # [FUNCTION DOC] Starts with a stale status, then the controller observes an unhealthy
        # status stamped with the manager's current hash. Asserts conditions are Allocated true and
        # Healthy false and the hash is "gen-2".
        """Blanking must be tied to the mismatch alone, otherwise the cell never regains a verdict."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID, workers_hash="gen-2"))
        controller = MockInferenceController({ENGINE_CELL_ID: _running_status(TriState.TRUE, workers_hash="gen-1")})
        handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
            controllers=[controller],
            pool_ids=_pool_ids_of(manager),
        )

        controller.observe_cell(ENGINE_CELL_ID, _running_status(TriState.FALSE, workers_hash="gen-2"))
        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.FALSE),
        ]
        assert cell.status.workers_hash == "gen-2"
