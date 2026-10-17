# [MODULE DOC]
# Abstract interface for operations on cells, implemented per cluster backend.
from __future__ import annotations

import abc

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.worker_provider.base import CellInfo


class BaseCellOperations(abc.ABC):
    # [CLASS DOC]
    # Abstract set of asynchronous operations on cells identified by cell id. Backends implement
    # them against their own control plane.
    # [FUNCTION DOC]
    # Abstract (one-line body, so this block sits above the decorator). Return a mapping from cell
    # id to cell information for all cells of the given pools.
    @abc.abstractmethod
    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]: ...

    # [FUNCTION DOC]
    # Abstract (block sits above the decorator). Suspend the cell with the given id.
    @abc.abstractmethod
    async def suspend(self, *, cell_id: str) -> None: ...

    # [FUNCTION DOC]
    # Abstract (block sits above the decorator). Resume a previously suspended cell.
    @abc.abstractmethod
    async def resume(self, *, cell_id: str) -> None: ...

    # [FUNCTION DOC]
    # Abstract (block sits above the decorator). Inject a fault of the given failure mode into the
    # cell, targeting the sub-component at the given index within the cell. For fault-tolerance
    # tests.
    @abc.abstractmethod
    async def inject_fault(self, *, cell_id: str, mode: FailureMode, sub_index: int) -> None: ...
