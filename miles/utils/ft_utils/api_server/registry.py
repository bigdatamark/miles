# [MODULE DOC]
# HELPER: Registry that aggregates cell handlers of the api server: lists cells across all handlers
# and routes a cell id to the handler that owns it.
from __future__ import annotations

import asyncio

from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.models import Cell


class _CellRegistry:
    # [CLASS DOC] HELPER: Ordered collection of cell handlers with unique cell types.
    def __init__(self, handlers: list[_CellHandler]) -> None:
        # [FUNCTION DOC] HELPER: Asserts all handlers have distinct cell types, then stores them.
        cell_types = [handler.cell_type for handler in handlers]
        assert len(set(cell_types)) == len(cell_types), f"Duplicate cell types: {cell_types}"
        self._handlers = handlers

    async def list_cells(self) -> list[Cell]:
        # [FUNCTION DOC] HELPER: Lists cells from all handlers concurrently and concatenates them in
        # handler order.
        per_handler = await asyncio.gather(*(handler.list_cells() for handler in self._handlers))
        return [cell for cells in per_handler for cell in cells]

    async def resolve(self, cell_id: str) -> _CellHandler:
        # [FUNCTION DOC] HELPER: Returns the first handler (in order, queried sequentially) whose
        # cell id list contains the id. Raises a key error with the id if none does.
        for handler in self._handlers:
            if cell_id in await handler.list_cell_ids():
                return handler
        raise KeyError(cell_id)
