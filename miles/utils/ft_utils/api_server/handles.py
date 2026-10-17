# [MODULE DOC]
# Per cell-type handler used by the api server. A handler combines the cell operations backend
# (listing, suspend, resume, fault injection) with status sources (controllers) to produce cell
# resources. Cells that are not alive are reported as suspended; alive cells report the status
# supplied by the controllers, marked unknown when the status belongs to an older workers
# generation.
from __future__ import annotations

import asyncio
from typing import Protocol

from miles.ray.rollout.server_cell import compute_pending_rollout_cell_status
from miles.utils.ft_utils.api_server.models import Cell, CellCondition, CellMetadata, CellSpec, CellStatus, TriState
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_provider.base import CellInfo


class _CellStatusSource(Protocol):
    # [CLASS DOC] HELPER: Interface of anything that reports cell statuses.
    async def get_cell_statuses(self) -> dict[str, CellStatus]:
        # [FUNCTION DOC] HELPER: Asynchronously returns a mapping from cell id to its current status.
        ...


class _CellHandler:
    # [CLASS DOC] HELPER: Serves the cells of one cell type (such as actor or rollout). Holds the
    # shared operations backend, the status sources for this type, and the pool ids whose cells
    # belong to the type.
    def __init__(
        self,
        *,
        cell_type: str,
        operations: BaseCellOperations,
        controllers: list[_CellStatusSource],
        pool_ids: list[str],
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the cell type name, operations backend, status sources and
        # pool ids (all keyword-only).
        self._cell_type = cell_type
        self._operations = operations
        self._controllers = controllers
        self._pool_ids = pool_ids

    @property
    def cell_type(self) -> str:
        # [FUNCTION DOC] HELPER: Property: the cell type name.
        return self._cell_type

    def _compute_metadata(self, cell_id: str) -> CellMetadata:
        # [FUNCTION DOC] HELPER: Metadata whose name is the cell id, with labels for the cell type and
        # the cell id.
        return CellMetadata(
            name=cell_id,
            labels={
                "miles.io/cell-type": self.cell_type,
                "miles.io/cell-id": cell_id,
            },
        )

    async def list_cell_ids(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Sorted ids of all cells of this handler's pools (queries the
        # operations backend).
        return sorted(await self._get_cell_infos())

    async def list_cells(self) -> list[Cell]:
        # [FUNCTION DOC] HELPER: Fetches cell infos and statuses once, then builds cell resources for
        # every cell id in sorted order.
        cell_infos = await self._get_cell_infos()
        statuses = await self._get_cell_statuses()
        return [
            self._compute_cell(cell_id, cell_infos=cell_infos, statuses=statuses) for cell_id in sorted(cell_infos)
        ]

    async def get_cell(self, cell_id: str) -> Cell:
        # [FUNCTION DOC] HELPER: Builds the resource of one cell from freshly fetched infos and
        # statuses. Raises a key error if the cell id is unknown.
        return self._compute_cell(
            cell_id,
            cell_infos=await self._get_cell_infos(),
            statuses=await self._get_cell_statuses(),
        )

    def _compute_cell(self, cell_id: str, *, cell_infos: dict[str, CellInfo], statuses: dict[str, CellStatus]) -> Cell:
        # [FUNCTION DOC] HELPER: Pure assembly of a cell resource. A cell that is not alive has spec
        # suspend true and status phase Suspended with a single Allocated-false condition and the
        # info's workers hash. An alive cell has suspend false and the status derived from the
        # controller-reported status for its generation (workers hash). Key error for unknown ids.
        info = cell_infos[cell_id]
        suspended = not info.alive
        return Cell(
            metadata=self._compute_metadata(cell_id),
            spec=CellSpec(suspend=suspended),
            status=(
                CellStatus(
                    phase="Suspended",
                    conditions=[CellCondition.allocated(TriState.FALSE)],
                    workers_hash=info.workers_hash,
                )
                if suspended
                else _compute_status_of_generation(statuses.get(cell_id), workers_hash=info.workers_hash)
            ),
        )

    async def _get_cell_statuses(self) -> dict[str, CellStatus]:
        # [FUNCTION DOC] HELPER: Queries all status sources concurrently and merges the mappings into
        # one (later sources win on duplicate ids).
        return {
            cell_id: status
            for statuses in await asyncio.gather(*(c.get_cell_statuses() for c in self._controllers))
            for cell_id, status in statuses.items()
        }

    async def _get_cell_infos(self) -> dict[str, CellInfo]:
        # [FUNCTION DOC] HELPER: Mapping from cell id to info for the handler's pools, from the
        # operations backend.
        return await self._operations.cell_infos(pool_ids=self._pool_ids)

    async def suspend(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Asks the operations backend to suspend the cell.
        await self._operations.suspend(cell_id=cell_id)

    async def resume(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Asks the operations backend to resume the cell.
        await self._operations.resume(cell_id=cell_id)

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        # [FUNCTION DOC] HELPER: Asks the operations backend to inject a fault of the given mode into
        # the sub-component of the cell at the given index. Backends may raise not-implemented.
        await self._operations.inject_fault(cell_id=cell_id, mode=mode, sub_index=sub_index)


def _compute_status_of_generation(status: CellStatus | None, *, workers_hash: str) -> CellStatus:
    # [FUNCTION DOC] HELPER: Status of an alive cell for the current workers generation. No reported
    # status: the pending rollout cell status for that hash. Reported status with a different hash
    # (stale generation): a copy whose conditions are all unknown and whose hash is the current one.
    # Same hash: the reported status unchanged.
    if status is None:
        return compute_pending_rollout_cell_status(workers_hash=workers_hash)
    if status.workers_hash != workers_hash:
        return status.model_copy(
            update={
                "conditions": [_compute_unknown_condition(condition) for condition in status.conditions],
                "workers_hash": workers_hash,
            }
        )
    return status


def _compute_unknown_condition(condition: CellCondition) -> CellCondition:
    # [FUNCTION DOC] HELPER: Same condition type with unknown status. The health condition is
    # rebuilt through the health checker constructor (so it carries its unknown reason); others
    # have no reason or message.
    if condition.type == "Healthy":
        return CellCondition.from_health_checker_status(TriState.UNKNOWN)
    return CellCondition(type=condition.type, status=TriState.UNKNOWN)
