# [MODULE DOC] Tests for the cell registry (_CellRegistry in miles.utils.ft_utils.api_server.registry),
# which aggregates several cell handlers (one per cell type): listing the cells of all handlers
# concurrently and in handler order, resolving a cell id to the handler that reports it (key error
# when none does), and rejecting duplicate cell types. Uses the mock handlers from the conftest.
from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft_utils.api_server.registry import _CellRegistry

from .conftest import MockGatedHandler, MockGateOpeningHandler, MockHandler


class TestListCells:
    # [CLASS DOC] Tests of the registry's aggregated cell listing.
    async def test_lists_the_cells_of_every_handler(self) -> None:
        # [FUNCTION DOC] One actor and one rollout cell; asserts the listing names them in handler
        # order (actor first, then rollout).
        """One kind of cell must not hide another from the ft controller."""
        actor, rollout = MockHandler("actor"), MockHandler("rollout")
        actor.add("actor-0")
        rollout.add("inference-engine-0-0-0")
        registry = _CellRegistry([actor, rollout])

        assert [cell.metadata.name for cell in await registry.list_cells()] == [
            "actor-0",
            "inference-engine-0-0-0",
        ]

    async def test_a_handler_without_cells_contributes_nothing(self) -> None:
        # [FUNCTION DOC] Two empty handlers; asserts the listing is an empty list.
        """A train-only or rollout-only deployment still answers the same endpoint."""
        registry = _CellRegistry([MockHandler("actor"), MockHandler("rollout")])

        assert await registry.list_cells() == []

    async def test_cells_added_after_construction_are_listed(self) -> None:
        # [FUNCTION DOC] Asserts an empty listing, adds a cell to the handler afterwards and asserts
        # the next listing includes it (nothing is cached).
        """Cells come and go while the server runs, so the set is resolved per request."""
        rollout = MockHandler("rollout")
        registry = _CellRegistry([rollout])
        assert await registry.list_cells() == []

        rollout.add("0")

        assert [cell.metadata.name for cell in await registry.list_cells()] == ["0"]

    async def test_cells_removed_after_construction_disappear(self) -> None:
        # [FUNCTION DOC] Removes the handler's only cell after the registry was built and asserts
        # the listing is empty.
        """A cell that was scaled away must stop being reported as existing."""
        rollout = MockHandler("rollout")
        rollout.add("0")
        registry = _CellRegistry([rollout])

        del rollout.cells["0"]

        assert await registry.list_cells() == []

    async def test_a_later_handler_is_queried_before_a_stalled_earlier_one_finishes(self) -> None:
        # [FUNCTION DOC] The first handler waits on a gate event in its listing and the second sets
        # that event when listed. Listing must finish within 5 seconds (it would time out if handlers
        # were polled one after another) and return both cells in handler order.
        """Handlers are polled concurrently, so one slow cell kind cannot delay the others from starting."""
        gate = asyncio.Event()
        actor = MockGatedHandler("actor", gate=gate)
        rollout = MockGateOpeningHandler("rollout", gate=gate)
        actor.add("actor-3")
        rollout.add("inference-engine-1-0-0")
        registry = _CellRegistry([actor, rollout])

        cells = await asyncio.wait_for(registry.list_cells(), timeout=5)

        assert [cell.metadata.name for cell in cells] == ["actor-3", "inference-engine-1-0-0"]


class TestResolve:
    # [CLASS DOC] Tests of resolving a cell id to its owning handler.
    async def test_a_cell_id_resolves_to_the_handler_that_owns_it(self) -> None:
        # [FUNCTION DOC] An engine cell exists only on the rollout handler; asserts resolving it
        # returns that very handler object.
        """Cell ids are global, so the owner is whoever reports the id, not a name prefix."""
        actor, rollout = MockHandler("actor"), MockHandler("rollout")
        rollout.add("inference-engine-0-0-2")
        registry = _CellRegistry([actor, rollout])

        assert await registry.resolve("inference-engine-0-0-2") is rollout

    async def test_an_unknown_cell_id_raises(self) -> None:
        # [FUNCTION DOC] Asserts resolving an id no handler reports raises a key error.
        """An unknown cell must 404 rather than resolve to a neighbour."""
        registry = _CellRegistry([MockHandler("actor")])

        with pytest.raises(KeyError):
            await registry.resolve("actor-7")

    async def test_a_cell_of_an_unregistered_kind_raises(self) -> None:
        # [FUNCTION DOC] Asserts resolving an id of an unregistered kind raises a key error.
        """Only registered kinds of cells are addressable."""
        registry = _CellRegistry([MockHandler("actor")])

        with pytest.raises(KeyError):
            await registry.resolve("critic-0")

    def test_duplicate_cell_types_are_rejected(self) -> None:
        # [FUNCTION DOC] Asserts constructing a registry from two handlers of the same cell type
        # raises an assertion error whose message mentions duplicate cell types.
        """Two handlers of one type would make cell names ambiguous."""
        with pytest.raises(AssertionError, match="Duplicate cell types"):
            _CellRegistry([MockHandler("rollout"), MockHandler("rollout")])
