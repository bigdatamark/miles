# [MODULE DOC]
# Ray implementation of cell operations. Trainer cells are handled directly by the Ray worker
# manager actor; every other cell (inference) is routed through the inference controller so that
# the operation does not interleave with a weight update.
from __future__ import annotations

from collections.abc import Callable

import ray.actor

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import CellInfo


class RayCellOperations(BaseCellOperations):
    # [CLASS DOC]
    # Cell operations over Ray. Holds the worker manager actor handle and a lazily resolved,
    # cached handle to the inference controller. A cell is a trainer cell iff its id starts with
    # the trainer engine prefix; all other cells are treated as inference cells.
    def __init__(
        self,
        *,
        worker_manager_handle: ray.actor.ActorHandle,
        resolve_inference_controller: Callable[[], BaseWorkerHandle],
    ) -> None:
        # [FUNCTION DOC]
        # Store the manager handle and the zero-argument resolver for the inference controller
        # handle; the controller itself is not resolved until first needed.
        self._worker_manager_handle = worker_manager_handle
        # TEMPORARY: this layer is not meant to know the inference controller, deliberately violated
        # until the weight-update fault tolerance work removes the need
        self._resolve_inference_controller = resolve_inference_controller
        self._inference_controller: BaseWorkerHandle | None = None

    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        # [FUNCTION DOC]
        # Ask the Ray worker manager actor, remotely, for the cell infos of the given pools and
        # return its result (a mapping from cell id to cell info) unchanged.
        return await self._worker_manager_handle.get_cell_infos.remote(pool_ids=pool_ids)

    async def suspend(self, *, cell_id: str) -> None:
        # [FUNCTION DOC]
        # For a trainer cell, ask the worker manager actor to stop that cell directly. For any
        # other cell, ask the inference controller to stop the cell at a point between weight
        # updates (so it takes the lock the weight update holds). Waits for completion.
        if _is_trainer_cell_id(cell_id):
            await self._worker_manager_handle.stop_cells.remote([cell_id])
            return

        # TEMPORARY: taking the lock the weight update holds, reverted with that fault tolerance work
        await self._controller().stop_cell_between_weight_updates(cell_id=cell_id)

    async def resume(self, *, cell_id: str) -> None:
        # [FUNCTION DOC]
        # Ask the worker manager actor to start the cell, for both trainer and inference cells.
        await self._worker_manager_handle.start_cells.remote([cell_id])

    async def inject_fault(self, *, cell_id: str, mode: FailureMode, sub_index: int) -> None:
        # [FUNCTION DOC]
        # For a trainer cell, ask the worker manager actor to inject the fault (passing the failure
        # mode's string value and the sub-index as the worker index within the cell). For any other
        # cell, ask the inference controller to inject it between weight updates, passing the cell
        # id, mode and sub-index. Waits for completion.
        if _is_trainer_cell_id(cell_id):
            await self._worker_manager_handle.inject_fault.remote(
                cell_id, mode=mode.value, worker_in_cell_index=sub_index
            )
            return

        # TEMPORARY: taking the lock the weight update holds, reverted with that fault tolerance work
        await self._controller().inject_fault_between_weight_updates(
            cell_id=cell_id,
            mode=mode,
            sub_index=sub_index,
        )

    def _controller(self) -> BaseWorkerHandle:
        # [FUNCTION DOC] HELPER: Return the inference controller handle, resolving it through the
        # injected resolver on first use and caching it afterwards.
        if self._inference_controller is None:
            self._inference_controller = self._resolve_inference_controller()
        return self._inference_controller


# TEMPORARY: matched by pool prefix only, until the fault tolerance rework routes trainer cells properly
def _is_trainer_cell_id(cell_id: str) -> bool:
    # [FUNCTION DOC] HELPER: True iff the cell id starts with the trainer engine prefix.
    return cell_id.startswith("trainer-engine-")
