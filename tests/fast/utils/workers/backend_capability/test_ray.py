# [MODULE DOC] Test for the ray backend capability's cell operations: suspending a cell through
# the capability goes via the worker manager (lookup of the cell's worker info, then the actor
# handle of the inference controller at the recorded generation) to the controller's stop-cell
# method. The worker manager handle, controller actor and their remote methods are small fakes.
from __future__ import annotations

from dataclasses import dataclass

import pytest

import miles.utils.workers.worker_provider.ray as ray_worker_provider_mod
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_spec import HostAndPort


class _FakeRemoteMethod:
    # [CLASS DOC] HELPER: Fake async remote method that records the cell ids it was called with.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the empty list of suspended cell ids.
        self.suspended_cell_ids: list[str] = []

    async def remote(self, *, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Appends the cell id to the list.
        self.suspended_cell_ids.append(cell_id)


@dataclass
class _FakeInferenceControllerActor:
    # [CLASS DOC] HELPER: Fake controller actor exposing the stop-cell-between-weight-updates
    # remote method.
    stop_cell_between_weight_updates: _FakeRemoteMethod


@dataclass
class _FakeGetWorkerInfosMethod:
    # [CLASS DOC] HELPER: Fake manager method that returns a fixed worker info for any cell.
    controller_info: WorkerInfo

    def remote(self, cell_id: str) -> list[WorkerInfo]:
        # [FUNCTION DOC] HELPER: Returns a one-element list holding the fixed worker info,
        # whatever the cell id.
        return [self.controller_info]


@dataclass
class _FakeGetActorHandleMethod:
    # [CLASS DOC] HELPER: Fake manager method that returns a fixed controller actor.
    controller: _FakeInferenceControllerActor

    def remote(self, worker_name: str, *, expected_generation: int) -> _FakeInferenceControllerActor:
        # [FUNCTION DOC] HELPER: Returns the fixed controller regardless of worker name and
        # generation.
        return self.controller


@dataclass
class _FakeWorkerManagerHandle:
    # [CLASS DOC] HELPER: Fake worker manager handle with the two lookup methods.
    get_worker_infos: _FakeGetWorkerInfosMethod
    get_actor_handle: _FakeGetActorHandleMethod


class TestRayBackendCapabilityCellOperations:
    # [CLASS DOC] Async test of suspending a cell through the ray capability.
    async def test_suspend_reaches_the_inference_controller_served_by_the_worker_manager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Builds a fake controller, a worker info for an inference controller
        # (generation 3, one primary address), and a fake manager serving them. Patches the ray
        # get function used by the ray worker provider module to return its argument unchanged
        # (reverted at teardown). Suspends cell cell-2 through the capability's cell operations
        # and asserts the controller's stop method was called with exactly that cell id.
        """Suspending through a capability reaches its registered inference controller."""
        controller = _FakeInferenceControllerActor(stop_cell_between_weight_updates=_FakeRemoteMethod())
        controller_info = WorkerInfo(
            name="inference-controller-00000-00000",
            generation=3,
            self_addrs={"primary": HostAndPort(host="10.0.0.7", port=15000)},
            gpu_ids=[],
            worker_class=None,
        )
        worker_manager = _FakeWorkerManagerHandle(
            get_worker_infos=_FakeGetWorkerInfosMethod(controller_info=controller_info),
            get_actor_handle=_FakeGetActorHandleMethod(controller=controller),
        )
        monkeypatch.setattr(ray_worker_provider_mod.ray, "get", lambda value: value)
        capability = RayBackendCapability(worker_manager_handle=worker_manager)

        operations = capability.cell_operations()
        await operations.suspend(cell_id="cell-2")

        assert controller.stop_cell_between_weight_updates.suspended_cell_ids == ["cell-2"]
