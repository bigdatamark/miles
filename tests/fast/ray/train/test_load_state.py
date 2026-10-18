# [MODULE DOC] Async tests for reloading training state ("load_state") through the trainer
# controller and through a single trainer cell.
# Controller tests use fake cells (recording how often load_state was called and returning canned
# restored positions) injected into a controller built without its constructor, with an
# initialization guard that is either already entered or never entered. Cell tests use real alive
# cells from the shared conftest backed by fake Ray actor workers, inspecting their recorded calls.
# An autouse fixture makes the expected cell count come from the arguments namespace.
from types import SimpleNamespace

import pytest
import ray
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_alive_cell

from miles.ray.specs.train import compute_trainer_pool_id
from miles.ray.train import group as group_module
from miles.ray.train.group import TrainerController
from miles.utils.init_once import InitOnce

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _expected_controller_fleet(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture. Patches the controller module's function that
    # computes the number of trainer cells for a role so it simply returns the expected cell count
    # stored on the arguments namespace, letting tests set the expected fleet size directly.
    monkeypatch.setattr(group_module, "compute_trainer_num_cells", lambda args, *, role: args.expected_num_cells)


class _FakeCell:
    # [CLASS DOC] HELPER: Minimal stand-in for a trainer cell. Exposes the cell index, a
    # configurable alive flag, a derived cell id and a counter of load_state invocations.
    def __init__(self, *, cell_index: int, restored: list[int], is_alive: bool = True) -> None:
        # [FUNCTION DOC] HELPER: Stores the index, the list of restored positions that load_state
        # will return, the alive flag (default alive), builds the id from the index and zeroes
        # the call counter.
        self.cell_index = cell_index
        self.restored = restored
        self.is_alive = is_alive
        self.cell_id = f"trainer-engine-actor-{cell_index}"
        self.load_state_calls = 0

    async def load_state(self) -> list[int]:
        # [FUNCTION DOC] HELPER: Increments the call counter and returns the canned restored
        # positions.
        self.load_state_calls += 1
        return self.restored


def _guard(*, initialized: bool) -> InitOnce:
    # [FUNCTION DOC] HELPER: Creates an initialize-once guard named for the trainer controller.
    # When initialized is true it enters and leaves the guarding context once so the guard counts
    # as already initialized; otherwise it is left untouched. Returns the guard.
    guard = InitOnce("TrainerController")
    if initialized:
        with guard.guarding():
            pass
    return guard


def _make_controller(
    cells: list[_FakeCell], *, initialized: bool, expected_num_cells: int | None = None
) -> TrainerController:
    # [FUNCTION DOC] HELPER: Builds a trainer controller without running its constructor. The
    # cells are stored keyed by an id derived from each cell's index (the sort order of the
    # mapping is the order given, not by index), the init guard is initialized or not per the
    # flag, the role is actor with the matching pool id, and the arguments namespace carries the
    # expected cell count (defaults to the number of cells given). Returns the controller.
    controller = object.__new__(TrainerController)
    controller._cells_by_id = {f"trainer-engine-actor-{cell.cell_index}": cell for cell in cells}
    controller._init_once = _guard(initialized=initialized)
    controller._role = "actor"
    controller._pool_id = compute_trainer_pool_id("actor")
    controller.args = SimpleNamespace(
        expected_num_cells=len(cells) if expected_num_cells is None else expected_num_cells
    )
    return controller


class TestTrainerControllerLoadState:
    # [CLASS DOC] Tests of the controller-level load_state, which reloads every cell and returns
    # the concatenated restored positions, with preconditions on cell liveness and initialization.
    async def test_every_cell_reloads_and_its_positions_reach_the_caller_in_cell_order(self):
        # [FUNCTION DOC] Builds an initialized controller with cells registered in the order index
        # 1 then 0, calls load_state, and asserts the result is cell 0's positions followed by
        # cell 1's (ordered by cell index) and each cell was asked to reload exactly once.
        """Independent DP ranks are positional, so a reordered position list misreads which cell resumed where."""
        cells = [_FakeCell(cell_index=1, restored=[4, 4]), _FakeCell(cell_index=0, restored=[3, 3])]
        controller = _make_controller(cells, initialized=True)

        assert await controller.load_state() == [3, 3, 4, 4]
        assert [cell.load_state_calls for cell in cells] == [1, 1]

    async def test_a_controller_whose_cell_is_not_alive_refuses_to_reload_anything(self):
        # [FUNCTION DOC] With one alive and one not-alive cell, asserts load_state raises an
        # assertion error mentioning "not alive" and that no cell was asked to reload.
        """A take-over adopts the fleet a previous script left running; healing one is a different job."""
        cells = [_FakeCell(cell_index=0, restored=[3]), _FakeCell(cell_index=1, restored=[3], is_alive=False)]
        controller = _make_controller(cells, initialized=True)

        with pytest.raises(AssertionError, match="not alive"):
            await controller.load_state()

        assert [cell.load_state_calls for cell in cells] == [0, 0]

    async def test_a_controller_that_never_built_its_model_refuses_to_reload_it(self):
        # [FUNCTION DOC] With a controller whose init guard was never entered, asserts load_state
        # raises an assertion error.
        """Loading a checkpoint into a model that was never built would read as a successful take-over."""
        controller = _make_controller([_FakeCell(cell_index=0, restored=[3])], initialized=False)

        with pytest.raises(AssertionError):
            await controller.load_state()


class TestTrainerCellLoadState:
    # [CLASS DOC] Tests of the cell-level load_state against a real alive cell backed by fake
    # Ray actor workers.
    async def test_the_reload_reaches_every_worker_of_the_cell(self):
        # [FUNCTION DOC] Calls load_state on an alive cell, asserts the result is the two
        # per-worker positions (7, 7) the fake workers return, and that every worker's recorded
        # calls consist of exactly one load_state call.
        """A worker left on its old weights would train a model the other ranks already rolled back."""
        cell = make_alive_cell(0, alive_cell_indices=[0])

        assert await cell.load_state() == [7, 7]

        for handle in get_raw_actor_handles(cell):
            assert [method for method, _args, _kwargs in ray.get(handle.get_calls.remote())] == ["load_state"]


class TestAFailedReloadRecyclesTheCell:
    # [CLASS DOC] Tests that a failing worker during cell-level load_state marks the cell errored
    # and tears down all its workers.
    async def test_a_rank_that_refuses_the_reload_errors_the_cell_and_kills_every_rank(self):
        # [FUNCTION DOC] Configures the first worker to fail its load_state call, asserts the
        # cell's load_state raises a runtime error matching the injected failure, that the cell is
        # then flagged errored, and that every worker actor is dead (querying its calls raises an
        # actor error).
        """A surviving rank would still be in the checkpoint collective while the take-over reports the failure."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        handles = get_raw_actor_handles(cell)
        ray.get(handles[0].set_fail_methods.remote(["load_state"]))

        with pytest.raises(RuntimeError, match="Injected failure"):
            await cell.load_state()

        assert cell.is_errored
        for handle in handles:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())


class TestAReloadWaitsForTheWholeFleet:
    # [CLASS DOC] Tests that controller-level load_state waits for all expected cells to be
    # observed before reloading anything.
    async def test_a_controller_still_short_of_a_cell_waits_rather_than_reloading_what_it_has(self, monkeypatch):
        # [FUNCTION DOC] Shrinks the controller module's cells-ready timeout to 0.05 seconds,
        # builds an initialized controller with one cell while two are expected, and asserts
        # load_state raises a timeout error whose message reports "of 2 trainer cells observed",
        # with the existing cell never asked to reload.
        """Reloading half a fleet would leave the missing cell holding the state the run has moved off."""
        monkeypatch.setattr(group_module, "_CELLS_READY_TIMEOUT_SECONDS", 0.05)
        cells = [_FakeCell(cell_index=0, restored=[3, 3])]
        controller = _make_controller(cells, initialized=True, expected_num_cells=2)

        with pytest.raises(TimeoutError, match="of 2 trainer cells observed"):
            await controller.load_state()

        assert [cell.load_state_calls for cell in cells] == [0]
