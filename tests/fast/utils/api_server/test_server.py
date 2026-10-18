# [MODULE DOC] Tests for the fault-tolerance cell api server (miles.utils.ft_utils.api_server.server).
# Two kinds of tests: HTTP behavior exercised in process through an ASGI client over mock handlers
# (health, cell list and single cell golden JSON documents, patch to suspend/resume, 404 and 500
# Kubernetes-style status bodies, dynamic cells, fault injection, request validation), and
# server assembly/startup tests (which handlers get registered for which fault-tolerance
# components, port and host passing, a real uvicorn server on a daemon thread, startup failure
# propagation, operations selection). Some tests bind real local sockets on ports found from
# fixed starting numbers; others patch the raw server start function to capture the registry.
from __future__ import annotations

import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from tests.fast.ray.rollout.conftest import make_args as make_rollout_args

from miles.ray.rollout.server_cell import compute_pending_rollout_cell_status
from miles.utils.ft_utils.api_server import server
from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.registry import _CellRegistry
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.ray import RayCellOperations

from .conftest import (
    MockHandler,
    MockInferenceController,
    MockStopCellController,
    MockTrainerCell,
    MockWorkerManager,
    make_cell_summaries,
    make_mock_controller,
)


class TestGetHealth:
    # [CLASS DOC] Tests of the health endpoint.
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, async_client: httpx.AsyncClient) -> None:
        # [FUNCTION DOC] GET the health path; asserts status 200 and body with status "ok".
        resp = await async_client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestGetCells:
    # [CLASS DOC] Tests of the cell list endpoint.
    @pytest.mark.asyncio
    async def test_empty_registry_returns_empty_cell_list(self, async_client: httpx.AsyncClient) -> None:
        # [FUNCTION DOC] With no cells, asserts status 200 and a list document with API version
        # "miles.io/v1", kind "CellList" and no items.
        resp = await async_client.get("/api/v1/cells")
        assert resp.status_code == 200
        assert resp.json() == {
            "apiVersion": "miles.io/v1",
            "kind": "CellList",
            "items": [],
        }

    @pytest.mark.asyncio
    async def test_returns_all_cells_golden(
        self, actor_handler: MockHandler, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] Adds a running actor cell and a suspended rollout cell (both conditions
        # false), GETs the cell list and asserts the entire JSON body: each item has API version,
        # kind Cell, metadata with name and labels (cell-type and cell-id), spec with the suspend
        # flag, and status with phase, the two conditions (each with type, status, null reason,
        # null message, null last transition time) and workers_hash "pseudo-hash-0". Items are in
        # handler order (actor then rollout).
        """Golden test: full JSON response for GET /api/v1/cells with two cells."""
        actor_handler.add("actor-0", phase="Running")
        rollout_handler.add(
            "rollout-0",
            phase="Suspended",
            is_suspended=True,
            conditions=[
                {"type": "Allocated", "status": "False"},
                {"type": "Healthy", "status": "False"},
            ],
        )

        resp = await async_client.get("/api/v1/cells")
        assert resp.status_code == 200
        assert resp.json() == {
            "apiVersion": "miles.io/v1",
            "kind": "CellList",
            "items": [
                {
                    "apiVersion": "miles.io/v1",
                    "kind": "Cell",
                    "metadata": {
                        "name": "actor-0",
                        "labels": {
                            "miles.io/cell-type": "actor",
                            "miles.io/cell-id": "actor-0",
                        },
                    },
                    "spec": {"suspend": False},
                    "status": {
                        "phase": "Running",
                        "conditions": [
                            {
                                "type": "Allocated",
                                "status": "True",
                                "reason": None,
                                "message": None,
                                "lastTransitionTime": None,
                            },
                            {
                                "type": "Healthy",
                                "status": "True",
                                "reason": None,
                                "message": None,
                                "lastTransitionTime": None,
                            },
                        ],
                        "workers_hash": "pseudo-hash-0",
                    },
                },
                {
                    "apiVersion": "miles.io/v1",
                    "kind": "Cell",
                    "metadata": {
                        "name": "rollout-0",
                        "labels": {
                            "miles.io/cell-type": "rollout",
                            "miles.io/cell-id": "rollout-0",
                        },
                    },
                    "spec": {"suspend": True},
                    "status": {
                        "phase": "Suspended",
                        "conditions": [
                            {
                                "type": "Allocated",
                                "status": "False",
                                "reason": None,
                                "message": None,
                                "lastTransitionTime": None,
                            },
                            {
                                "type": "Healthy",
                                "status": "False",
                                "reason": None,
                                "message": None,
                                "lastTransitionTime": None,
                            },
                        ],
                        "workers_hash": "pseudo-hash-0",
                    },
                },
            ],
        }


class TestGetCell:
    # [CLASS DOC] Tests of the single cell endpoint.
    @pytest.mark.asyncio
    async def test_returns_single_cell_golden(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] Adds a running actor cell, GETs it by name and asserts status 200 and the
        # complete JSON cell document (same shape as one item of the list golden test).
        """Golden test: full JSON response for GET /api/v1/cells/{name}."""
        actor_handler.add("actor-0", phase="Running")

        resp = await async_client.get("/api/v1/cells/actor-0")
        assert resp.status_code == 200
        assert resp.json() == {
            "apiVersion": "miles.io/v1",
            "kind": "Cell",
            "metadata": {
                "name": "actor-0",
                "labels": {
                    "miles.io/cell-type": "actor",
                    "miles.io/cell-id": "actor-0",
                },
            },
            "spec": {"suspend": False},
            "status": {
                "phase": "Running",
                "conditions": [
                    {
                        "type": "Allocated",
                        "status": "True",
                        "reason": None,
                        "message": None,
                        "lastTransitionTime": None,
                    },
                    {"type": "Healthy", "status": "True", "reason": None, "message": None, "lastTransitionTime": None},
                ],
                "workers_hash": "pseudo-hash-0",
            },
        }

    @pytest.mark.asyncio
    async def test_not_found_returns_k8s_status_golden(self, async_client: httpx.AsyncClient) -> None:
        # [FUNCTION DOC] GETs an unknown cell; asserts status 404 and the exact Kubernetes-style
        # status body: API version "v1", kind Status, status Failure, message "Cell 'nonexistent'
        # not found", reason NotFound, code 404.
        """Golden test: K8s Status error response for 404."""
        resp = await async_client.get("/api/v1/cells/nonexistent")
        assert resp.status_code == 404
        assert resp.json() == {
            "apiVersion": "v1",
            "kind": "Status",
            "status": "Failure",
            "message": "Cell 'nonexistent' not found",
            "reason": "NotFound",
            "code": 404,
        }


class TestPatchCell:
    # [CLASS DOC] Tests of the cell patch endpoint (suspend or resume via the spec).
    @pytest.mark.asyncio
    async def test_suspend_cell_via_patch(self, actor_handler: MockHandler, async_client: httpx.AsyncClient) -> None:
        # [FUNCTION DOC] Patches a running cell with suspend true; asserts status 200, exactly one
        # suspend call on the mock cell, and a response with phase Suspended and suspend true.
        cell = actor_handler.add("actor-0", phase="Running")

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 200
        assert cell.suspend_calls == 1
        assert resp.json()["status"]["phase"] == "Suspended"
        assert resp.json()["spec"]["suspend"] is True

    @pytest.mark.asyncio
    async def test_resume_cell_via_patch(self, actor_handler: MockHandler, async_client: httpx.AsyncClient) -> None:
        # [FUNCTION DOC] Patches a suspended cell with suspend false; asserts status 200, exactly
        # one resume call, and response phase Running.
        cell = actor_handler.add("actor-0", phase="Suspended", is_suspended=True)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": False}})
        assert resp.status_code == 200
        assert cell.resume_calls == 1
        assert resp.json()["status"]["phase"] == "Running"

    @pytest.mark.asyncio
    async def test_patch_with_no_spec_is_noop(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] Patches with an empty body; asserts status 200 and zero suspend and resume calls.
        cell = actor_handler.add("actor-0", phase="Running")

        resp = await async_client.patch("/api/v1/cells/actor-0", json={})
        assert resp.status_code == 200
        assert cell.suspend_calls == 0
        assert cell.resume_calls == 0

    @pytest.mark.asyncio
    async def test_patch_with_empty_spec_does_not_suspend_or_resume(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] Patches with an empty spec object; asserts status 200, no suspend or resume
        # calls, and the response still shows suspend false and phase Running.
        """A spec that omits suspend carries no instruction, so the cell must be left exactly as it was."""
        cell = actor_handler.add("actor-0", phase="Running")

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {}})

        assert resp.status_code == 200
        assert (cell.suspend_calls, cell.resume_calls) == (0, 0)
        assert resp.json()["spec"]["suspend"] is False
        assert resp.json()["status"]["phase"] == "Running"

    @pytest.mark.asyncio
    async def test_patch_not_found_returns_k8s_status(self, async_client: httpx.AsyncClient) -> None:
        # [FUNCTION DOC] Patches an unknown cell; asserts status 404 with kind Status and reason NotFound.
        resp = await async_client.patch("/api/v1/cells/nonexistent", json={"spec": {"suspend": True}})
        assert resp.status_code == 404
        assert resp.json()["kind"] == "Status"
        assert resp.json()["reason"] == "NotFound"

    @pytest.mark.asyncio
    async def test_patch_suspend_idempotent(self, actor_handler: MockHandler, async_client: httpx.AsyncClient) -> None:
        # [FUNCTION DOC] Patches an already suspended cell with suspend true; asserts status 200 and
        # that the mock cell's suspend counter is 1 (the request is forwarded to the handler, which
        # tolerates it).
        cell = actor_handler.add("actor-0", phase="Suspended", is_suspended=True)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 200
        assert cell.suspend_calls == 1

    @pytest.mark.asyncio
    async def test_patch_error_returns_500_k8s_status(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] The cell raises an error on suspend; asserts the patch answers status 500
        # with kind Status and reason InternalError.
        actor_handler.add("actor-0", suspend_error=RuntimeError("engine crashed"))

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 500
        assert resp.json()["kind"] == "Status"
        assert resp.json()["reason"] == "InternalError"

    @pytest.mark.asyncio
    async def test_resume_error_returns_500_k8s_status(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] The suspended cell raises an error on resume; asserts status 500 and the
        # exact body: API version "v1", kind Status, status Failure, message "Failed to patch cell
        # 'actor-0'", reason InternalError, code 500.
        """A resume that fails must be reported as loudly as a suspend that fails, or the cell looks resumed."""
        actor_handler.add("actor-0", is_suspended=True, resume_error=RuntimeError("relaunch rejected"))

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": False}})

        assert resp.status_code == 500
        assert resp.json() == {
            "apiVersion": "v1",
            "kind": "Status",
            "status": "Failure",
            "message": "Failed to patch cell 'actor-0'",
            "reason": "InternalError",
            "code": 500,
        }


class TestStartApiServerRegistration:
    # [CLASS DOC] Tests of which cell handlers start_api_server registers for each combination of
    # fault-tolerance components, using the real start function with the raw server start patched.
    def _start(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        ft_components: list[str],
        cell_ids: list[str],
        actor_cells: list[MockTrainerCell] | None = None,
    ) -> _CellRegistry:
        # [FUNCTION DOC] HELPER: Run the real api server start function with fakes and return the
        # registry it built. Builds a mock worker manager for the cell ids; patches the engine pool
        # id computation to return the single pool "inference-engine-0-0" and the raw server start to
        # just capture the registry (so no server runs). Calls start with rollout test args, a
        # trainer model mapping "actor" to a mock controller over the given trainer cells (default
        # none), a mock inference controller holding a pending status per cell id, port 18080, the
        # given fault-tolerance components and Ray-style cell operations over the manager. Expects
        # exactly one captured registry and returns it.
        manager = MockWorkerManager(make_cell_summaries(*cell_ids))
        registries: list[_CellRegistry] = []

        monkeypatch.setattr(server, "compute_engine_pool_ids", lambda args: ["inference-engine-0-0"])
        monkeypatch.setattr(
            server, "_start_api_server_raw", lambda *, registry, port, host: registries.append(registry)
        )

        server.start_api_server(
            args=make_rollout_args(),
            trainer_models={"actor": make_mock_controller(actor_cells if actor_cells is not None else [])},
            inference_controller=MockInferenceController(
                {cell_id: compute_pending_rollout_cell_status(workers_hash="pseudo-hash-0") for cell_id in cell_ids}
            ),
            port=18080,
            ft_components=ft_components,
            cell_operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
        )

        (registry,) = registries
        return registry

    @pytest.mark.asyncio
    async def test_the_rollout_handler_enumerates_every_engine_cell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With rollout fault tolerance and two engine cells given in reverse order,
        # asserts the registry lists both, sorted by cell id.
        """A cell the handler never enumerates is invisible to the heal loop forever."""
        registry = self._start(
            monkeypatch,
            ft_components=["rollout"],
            cell_ids=["inference-engine-0-0-1", "inference-engine-0-0-0"],
        )

        assert [cell.metadata.name for cell in await registry.list_cells()] == [
            "inference-engine-0-0-0",
            "inference-engine-0-0-1",
        ]

    @pytest.mark.asyncio
    async def test_no_rollout_handler_exists_when_rollout_ft_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With only train fault tolerance and an engine cell known, asserts the
        # registry lists no cells.
        """Exposing suspend on engines nobody heals would let a request strand the pool."""
        registry = self._start(monkeypatch, ft_components=["train"], cell_ids=["inference-engine-0-0-0"])

        assert await registry.list_cells() == []

    @pytest.mark.asyncio
    async def test_the_actor_handler_enumerates_the_real_trainer_spec_cells(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] With train fault tolerance, one trainer cell and an unpatched trainer pool
        # name, asserts the registry lists exactly the cell "trainer-engine-actor-0" with phase Running.
        """The unpatched trainer spec name must match the real trainer cells, or the heal loop sees no trainer."""
        registry = self._start(
            monkeypatch,
            ft_components=["train"],
            cell_ids=["trainer-engine-actor-0"],
            actor_cells=[MockTrainerCell(phase="Running")],
        )

        cells = await registry.list_cells()
        assert [cell.metadata.name for cell in cells] == ["trainer-engine-actor-0"]
        assert cells[0].status.phase == "Running"

    @pytest.mark.asyncio
    async def test_both_handlers_coexist_under_mixed_ft(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With both train and rollout components, asserts the registry's handlers
        # (read from its private list) have cell types actor then rollout.
        """Mixed ft heals trainer and rollout cells through the one endpoint."""
        registry = self._start(monkeypatch, ft_components=["train", "rollout"], cell_ids=["inference-engine-0-0-0"])

        assert [handler.cell_type for handler in registry._handlers] == ["actor", "rollout"]

    @pytest.mark.asyncio
    async def test_mixed_ft_serves_the_trainer_and_the_engine_cells(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With both components, a trainer cell and an engine cell, asserts the listing
        # is the trainer cell labeled actor followed by the engine cell labeled rollout.
        """Mixed ft is the only mode where one endpoint must answer for two pools, so both must be enumerable."""
        registry = self._start(
            monkeypatch,
            ft_components=["train", "rollout"],
            cell_ids=["trainer-engine-actor-0", "inference-engine-0-0-0"],
            actor_cells=[MockTrainerCell(phase="Running")],
        )

        cells = await registry.list_cells()
        assert [(cell.metadata.name, cell.metadata.labels["miles.io/cell-type"]) for cell in cells] == [
            ("trainer-engine-actor-0", "actor"),
            ("inference-engine-0-0-0", "rollout"),
        ]

    @pytest.mark.asyncio
    async def test_no_actor_handler_exists_when_train_ft_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With only rollout fault tolerance and a trainer cell present, asserts the
        # registry lists no cells (the manager's cell id does not belong to the engine pool).
        """Exposing suspend on trainer cells nobody heals would let one request strand the whole run."""
        registry = self._start(
            monkeypatch,
            ft_components=["rollout"],
            cell_ids=["trainer-actor-0"],
            actor_cells=[MockTrainerCell(phase="Running")],
        )

        assert await registry.list_cells() == []

    @pytest.mark.asyncio
    async def test_the_requested_port_reaches_the_server_that_binds_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Patches the raw server start to record the port, calls start with port
        # 19137 and train fault tolerance, and asserts the recorded ports are exactly [19137].
        """The FT controller is told this port out of band, so binding any other one makes the api unreachable."""
        ports: list[int] = []
        manager = MockWorkerManager(make_cell_summaries("trainer-engine-actor-0"))
        monkeypatch.setattr(server, "_start_api_server_raw", lambda *, registry, port, host: ports.append(port))

        server.start_api_server(
            args=make_rollout_args(),
            trainer_models={"actor": make_mock_controller([MockTrainerCell()])},
            inference_controller=MockInferenceController(),
            port=19137,
            ft_components=["train"],
            cell_operations=RayCellOperations(
                worker_manager_handle=manager,
                resolve_inference_controller=lambda: MockStopCellController(manager),
            ),
        )

        assert ports == [19137]


class TestStartApiServerRaw:
    # [CLASS DOC] Tests of the raw server start function, which really starts a uvicorn server on
    # local sockets (ports probed from fixed starting numbers). Each test signals the server to exit
    # at the end where it keeps a handle.
    def test_uvicorn_serves_the_registry_app_on_a_daemon_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Wraps the thread-start helper to record the thread it returns, starts the
        # server on loopback with an empty registry on a free port at or above 21200, and asserts
        # exactly one thread was started, it is a daemon, the served app is a web application object,
        # and the server config host and port are the requested ones. Always requests server exit.
        """A foreground server blocks the training driver, and a non-loopback bind exposes unauthenticated cell control."""
        serving_threads: list[threading.Thread] = []
        start_and_wait_thread = server._start_and_wait_thread

        def _record_thread(**kwargs) -> threading.Thread:
            # [FUNCTION DOC] HELPER: Call the real thread-start helper with the same keyword
            # arguments, remember the returned thread in the enclosing list and return it.
            thread = start_and_wait_thread(**kwargs)
            serving_threads.append(thread)
            return thread

        monkeypatch.setattr(server, "_start_and_wait_thread", _record_thread)
        port = find_available_port(21200)

        running = server._start_api_server_raw(registry=_CellRegistry([]), port=port, host="127.0.0.1")

        try:
            [thread] = serving_threads
            assert thread.daemon is True
            assert isinstance(running.config.app, FastAPI)
            assert (running.config.host, running.config.port) == ("127.0.0.1", port)
        finally:
            running.should_exit = True

    def test_a_caller_supplied_host_overrides_the_loopback_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Starts the server with host "0.0.0.0" on a free port at or above 21300 and
        # asserts the server config reports exactly that host and port. Always requests exit.
        # KNOWN ISSUE: the wrapped thread-recording helper and the serving thread list are set up
        # but never asserted on in this test (dead setup copied from the previous test); also the
        # name mentions a loopback default although the start call takes the host explicitly.
        """Remote FT controllers are opted into by passing a wider bind host explicitly."""
        serving_threads: list[threading.Thread] = []
        start_and_wait_thread = server._start_and_wait_thread

        def _record_thread(**kwargs) -> threading.Thread:
            # [FUNCTION DOC] HELPER: Call the real thread-start helper with the same keyword
            # arguments, remember the returned thread in the enclosing list and return it.
            thread = start_and_wait_thread(**kwargs)
            serving_threads.append(thread)
            return thread

        monkeypatch.setattr(server, "_start_and_wait_thread", _record_thread)
        port = find_available_port(21300)

        running = server._start_api_server_raw(registry=_CellRegistry([]), port=port, host="0.0.0.0")

        try:
            assert (running.config.host, running.config.port) == ("0.0.0.0", port)
        finally:
            running.should_exit = True

    def test_a_bound_port_serves_and_can_be_reached(self) -> None:
        # [FUNCTION DOC] Starts the server on loopback at a free port at or above 21000, performs a
        # real HTTP GET of the health path with a 10 second timeout and asserts status 200. Always
        # requests server exit afterwards.
        """The happy path must still return once uvicorn is actually accepting connections."""
        port = find_available_port(21000)

        running = server._start_api_server_raw(registry=_CellRegistry([]), port=port, host="127.0.0.1")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/v1/health", timeout=10.0)
            assert resp.status_code == 200
        finally:
            running.should_exit = True

    def test_a_port_already_taken_fails_the_caller(self) -> None:
        # [FUNCTION DOC] Occupies a free port (at or above 21100) with a listening socket on all
        # interfaces, then asserts starting the server on that port raises a runtime error whose
        # message says the port failed during startup.
        """A second job silently losing the port would then poll the first job's cell registry."""
        port = find_available_port(21100)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            occupied.bind(("0.0.0.0", port))
            occupied.listen()

            with pytest.raises(RuntimeError, match=f"port {port} failed during startup"):
                server._start_api_server_raw(registry=_CellRegistry([]), port=port, host="127.0.0.1")


class TestDynamicCells:
    # [CLASS DOC] Tests that the HTTP layer reflects cells added or removed while it runs.
    @pytest.mark.asyncio
    async def test_a_cell_that_appears_after_startup_is_served(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] Asserts a cell lookup is 404, adds the cell to the handler, and asserts the
        # same lookup is now 200.
        """Engine cells are reconciled in, so the server cannot snapshot them once at startup."""
        assert (await async_client.get("/api/v1/cells/rollout-engine-0")).status_code == 404

        rollout_handler.add("rollout-engine-0")

        assert (await async_client.get("/api/v1/cells/rollout-engine-0")).status_code == 200

    @pytest.mark.asyncio
    async def test_a_cell_that_disappears_stops_being_served(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] Asserts a present cell lookup is 200, deletes the cell from the handler and
        # asserts the lookup becomes 404.
        """A removed cell must 404 instead of reporting stale status."""
        rollout_handler.add("rollout-engine-0")
        assert (await async_client.get("/api/v1/cells/rollout-engine-0")).status_code == 200

        del rollout_handler.cells["rollout-engine-0"]

        assert (await async_client.get("/api/v1/cells/rollout-engine-0")).status_code == 404


class TestInjectFault:
    # [CLASS DOC] Tests of the fault-injection endpoint (POST to a cell's inject-fault path with a
    # body holding a failure mode and an optional sub index).
    @pytest.mark.asyncio
    async def test_injection_reaches_the_handler_of_that_cell(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] With injection enabled on the rollout handler, posts mode "sigkill" and sub
        # index 1; asserts status 200 and that the handler recorded (cell id, kill-signal mode, 1).
        """CI fault injection targets one cell by name."""
        rollout_handler.supports_inject_fault = True
        rollout_handler.add("rollout-engine-0")

        resp = await async_client.post(
            "/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "sigkill", "sub_index": 1}
        )

        assert resp.status_code == 200
        assert rollout_handler.injected == [("rollout-engine-0", FailureMode.SIGKILL, 1)]

    @pytest.mark.asyncio
    async def test_a_handler_without_injection_support_answers_bad_request(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] Posts a fault to an actor cell whose handler does not support injection
        # (the mock raises a not-implemented error); asserts status 400.
        """Not every kind of cell can be crashed on demand."""
        actor_handler.add("actor-0")

        resp = await async_client.post("/api/v1/cells/actor-0/inject-fault", json={"mode": "sigkill"})

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_inject_fault_uses_zero_sub_index_by_default(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] Posts only mode "exit"; asserts status 200, body with status "ok", and that
        # the handler recorded the exit mode with sub index 0.
        """The documented default targets worker zero, and a client omitting sub_index relies on it."""
        rollout_handler.supports_inject_fault = True
        rollout_handler.add("rollout-engine-0")

        resp = await async_client.post("/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "exit"})

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert rollout_handler.injected == [("rollout-engine-0", FailureMode.EXIT, 0)]

    @pytest.mark.asyncio
    async def test_inject_fault_rejects_missing_or_unknown_mode(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] Posts an empty body and a body with mode "nuke"; asserts both answer status
        # 422 and the handler recorded no injections.
        """An unrecognised failure mode must be refused by the schema rather than forwarded to the cell."""
        rollout_handler.supports_inject_fault = True
        rollout_handler.add("rollout-engine-0")

        missing = await async_client.post("/api/v1/cells/rollout-engine-0/inject-fault", json={})
        unknown = await async_client.post("/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "nuke"})

        assert (missing.status_code, unknown.status_code) == (422, 422)
        assert rollout_handler.injected == []

    @pytest.mark.asyncio
    async def test_an_injection_that_blows_up_returns_500_k8s_status(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] The handler raises a runtime error on injection; asserts status 500 and the
        # exact body: API version "v1", kind Status, status Failure, message "Failed to inject
        # fault into cell 'rollout-engine-0'", reason InternalError, code 500.
        """A crashed injection is not a bad request, and the CI harness needs the difference to fail the run."""
        rollout_handler.supports_inject_fault = True
        rollout_handler.inject_fault_error = RuntimeError("worker manager unreachable")
        rollout_handler.add("rollout-engine-0")

        resp = await async_client.post(
            "/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "sigkill", "sub_index": 1}
        )

        assert resp.status_code == 500
        assert resp.json() == {
            "apiVersion": "v1",
            "kind": "Status",
            "status": "Failure",
            "message": "Failed to inject fault into cell 'rollout-engine-0'",
            "reason": "InternalError",
            "code": 500,
        }


class TestStartAndWaitThread:
    # [CLASS DOC] Tests of the thread-start helper that runs a target on a daemon thread and waits
    # until a readiness predicate is true, the thread fails, the thread exits, or a timeout passes.
    def test_it_returns_once_the_thread_reports_ready(self):
        # [FUNCTION DOC] The target sets an event that is also the readiness check; asserts the
        # helper returns a thread object and the event is set.
        """The caller may only proceed after the thing it started is actually usable."""
        ready = threading.Event()

        thread = server._start_and_wait_thread(
            target=ready.set, is_ready=ready.is_set, description="probe", timeout_seconds=5.0
        )

        assert ready.is_set()
        assert isinstance(thread, threading.Thread)

    def test_a_failure_on_the_thread_reaches_the_caller(self):
        # [FUNCTION DOC] The target raises a value error and is never ready; asserts the helper
        # raises a runtime error saying the described component failed during startup, chained to
        # the original value error as cause.
        """A daemon thread that dies alone is invisible, which is how a lost port went unnoticed."""

        def _boom() -> None:
            # [FUNCTION DOC] HELPER: Thread target that always raises a value error "could not start".
            raise ValueError("could not start")

        with pytest.raises(RuntimeError, match="probe failed during startup") as excinfo:
            server._start_and_wait_thread(
                target=_boom, is_ready=lambda: False, description="probe", timeout_seconds=5.0
            )

        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_a_thread_that_exits_without_becoming_ready_fails_the_caller(self):
        # [FUNCTION DOC] The target returns immediately and readiness is never true; asserts a
        # runtime error saying the described component exited during startup.
        """Returning quietly is its own failure: nothing is serving afterwards."""
        with pytest.raises(RuntimeError, match="probe exited during startup"):
            server._start_and_wait_thread(
                target=lambda: None, is_ready=lambda: False, description="probe", timeout_seconds=5.0
            )

    def test_a_thread_that_never_becomes_ready_times_out(self):
        # [FUNCTION DOC] The target blocks on an event and readiness is never true; with a 0.2
        # second timeout asserts a timeout error saying the described component did not finish
        # startup. The blocking event is always set at the end so the thread can end.
        """A wedged startup must not block the caller forever."""
        stop = threading.Event()

        try:
            with pytest.raises(TimeoutError, match="probe did not finish startup"):
                server._start_and_wait_thread(
                    target=stop.wait, is_ready=lambda: False, description="probe", timeout_seconds=0.2
                )
        finally:
            stop.set()

    def test_a_ready_thread_is_not_judged_by_a_later_failure(self):
        # [FUNCTION DOC] The target sets the ready event and then raises; asserts the helper still
        # returns without raising (readiness wins over a later failure).
        """Readiness wins the race: a server that starts and later dies is the caller's problem, not startup's."""
        ready = threading.Event()

        def _ready_then_raise() -> None:
            # [FUNCTION DOC] HELPER: Thread target that sets the ready event, then raises a value
            # error "died after serving".
            ready.set()
            raise ValueError("died after serving")

        server._start_and_wait_thread(
            target=_ready_then_raise, is_ready=ready.is_set, description="probe", timeout_seconds=5.0
        )


class TestRequestValidation:
    # [CLASS DOC] Tests of request body validation on the write endpoints.
    @pytest.mark.asyncio
    async def test_invalid_write_bodies_return_422_without_side_effects(
        self, actor_handler: MockHandler, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        # [FUNCTION DOC] Sends a patch with an extra unknown field and an inject-fault request with
        # a misspelled sub-index field; asserts both answer status 422, the actor cell saw no
        # suspend or resume and the rollout handler recorded no injection.
        """Unknown fields must be refused outright, so a typo cannot silently half-apply a write."""
        cell = actor_handler.add("actor-0", phase="Running")
        rollout_handler.supports_inject_fault = True
        rollout_handler.add("rollout-engine-0")

        patch_resp = await async_client.patch(
            "/api/v1/cells/actor-0", json={"spec": {"suspend": True, "gracePeriod": 5}}
        )
        inject_resp = await async_client.post(
            "/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "sigkill", "subIndex": 1}
        )

        assert (patch_resp.status_code, inject_resp.status_code) == (422, 422)
        assert (cell.suspend_calls, cell.resume_calls) == (0, 0)
        assert rollout_handler.injected == []


class TestSeveralTrainers:
    # [CLASS DOC] Tests of the server when several trainer models exist.
    def test_every_trainer_pool_is_listed_under_the_actor_cell_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Patches the trainer pool id computation to prefix "trainer-engine-", the
        # engine pool computation and the raw server start (capturing the registry). Starts with two
        # trainer models and train fault tolerance; asserts there is a single handler whose pool id
        # list holds both trainer pool ids in the models' order.
        """A run of several trainer models heals none of them if the api server only knows the legacy actor pool."""
        registries: list[_CellRegistry] = []
        monkeypatch.setattr(server, "compute_trainer_pool_id", lambda trainer_id: f"trainer-engine-{trainer_id}")
        monkeypatch.setattr(server, "compute_engine_pool_ids", lambda args: ["engine"])
        monkeypatch.setattr(
            server, "_start_api_server_raw", lambda *, registry, port, host: registries.append(registry)
        )

        server.start_api_server(
            args=SimpleNamespace(),
            trainer_models={
                "a-actor": make_mock_controller([], pool_id="trainer-engine-a-actor"),
                "b-actor": make_mock_controller([], pool_id="trainer-engine-b-actor"),
            },
            inference_controller=MockInferenceController(),
            port=1234,
            ft_components=["train"],
            cell_operations=object(),
        )

        [handler] = registries[0]._handlers
        assert handler._pool_ids == ["trainer-engine-a-actor", "trainer-engine-b-actor"]

    @pytest.mark.asyncio
    async def test_the_cell_statuses_of_every_trainer_are_merged(self) -> None:
        # [FUNCTION DOC] Builds an actor handler over two mock trainer controllers with different
        # pool ids (placeholder operations, no pools) and asserts its merged status mapping has the
        # sorted keys of both trainers' cell ids.
        """Each trainer answers only about its own cells, and a cell missing from the merge reads as pending."""
        first = make_mock_controller([MockTrainerCell()], pool_id="trainer-engine-a-actor")
        second = make_mock_controller([MockTrainerCell()], pool_id="trainer-engine-b-actor")

        statuses = await _CellHandler(
            cell_type="actor", operations=object(), controllers=[first, second], pool_ids=[]
        )._get_cell_statuses()

        assert sorted(statuses) == ["trainer-engine-a-actor-0", "trainer-engine-b-actor-0"]


class TestOperationsSelection:
    # [CLASS DOC] Tests that the cell operations backend passed to the server is what handlers use.
    def test_every_handler_gets_the_operations_of_the_process_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Starts with a placeholder operations object and both train and rollout
        # components (raw server start patched to capture the registry); asserts two handlers exist
        # and each holds that very operations object.
        """Under Kubernetes the api server must act on pods, not on a Ray worker manager."""
        operations = object()
        registries: list[_CellRegistry] = []
        monkeypatch.setattr(server, "compute_trainer_pool_id", lambda role: f"trainer-{role}")
        monkeypatch.setattr(server, "compute_engine_pool_ids", lambda args: ["engine"])
        monkeypatch.setattr(
            server, "_start_api_server_raw", lambda *, registry, port, host: registries.append(registry)
        )

        server.start_api_server(
            args=SimpleNamespace(),
            trainer_models={"actor": make_mock_controller([])},
            inference_controller=MockInferenceController(),
            port=1234,
            ft_components=["train", "rollout"],
            cell_operations=operations,
        )

        handlers = registries[0]._handlers
        assert len(handlers) == 2
        assert [handler._operations for handler in handlers] == [operations, operations]

    def test_the_api_server_names_no_ray_worker_manager(self) -> None:
        # [FUNCTION DOC] Reads the server module's source text and asserts it never mentions the Ray
        # worker manager class name and that it contains the typed operations parameter declaration
        # for the base cell operations (a textual check of the source file).
        """Naming it would make the Kubernetes assembly pointless for the half that heals cells."""
        source = Path(server.__file__).read_text()

        assert "RayWorkerManager" not in source
        assert "cell_operations: BaseCellOperations" in source
