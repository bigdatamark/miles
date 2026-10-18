# [MODULE DOC] Async tests for what the trainer controller's train call returns: the per-worker
# train step outputs, their concatenation across cells in cell order, exclusion of failed cells,
# and retry behavior on a discarded-should-retry outcome. Cells are real alive cells from the shared
# conftest backed by fake Ray actors whose train return values the tests configure; the controller
# is built without its constructor.
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import ray
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_alive_cell

from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.ray.train.group import TrainerController
from miles.utils.data import RolloutDataPack
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.object_store import _MooncakeStoreObjectRef
from miles.utils.ray_utils import Box

pytestmark = pytest.mark.asyncio

_DUMMY_DATA_PACK = RolloutDataPack(sample_indices=[0], data_ref=_MooncakeStoreObjectRef(payload="data"))


def _make_controller(cells: list) -> TrainerController:
    # [FUNCTION DOC] HELPER: Builds a trainer controller without its constructor, holding the
    # given cells keyed by cell id, with event analysis and debug event saving disabled, no
    # witness allocator, quorum id zero, an active activeness tracker, and a stub test-action
    # executor whose asynchronous run-after-step hook is a mock. Returns the controller.
    group = object.__new__(TrainerController)
    group._cells_by_id = {cell.cell_id: cell for cell in cells}
    group.args = SimpleNamespace(enable_event_analyzer=False, save_debug_event_data=None)
    group._witness_allocator = None
    group._indep_dp_quorum_id = 0
    group._health_checker_activeness = ActivenessTracker(active=True)
    group._test_action_executor = SimpleNamespace(run_after_step=AsyncMock())
    return group


def _set_train_return_value(cell: Any, value: Any) -> None:
    # [FUNCTION DOC] HELPER: Configures every worker actor of the cell to return the given value
    # from its train call, waiting for each configuration remote call to complete.
    for handle in get_raw_actor_handles(cell):
        ray.get(handle.set_train_return_value.remote(value))


def _count_train_calls(cell: Any) -> int:
    # [FUNCTION DOC] HELPER: Sums over all worker actors of the cell the number of recorded calls
    # whose method name is train. Returns the total (workers times attempts).
    return sum(
        sum(1 for method, _args, _kwargs in ray.get(handle.get_calls.remote()) if method == "train")
        for handle in get_raw_actor_handles(cell)
    )


class TestTrainReturnValue:
    # [CLASS DOC] Tests of the list of train step outputs the controller returns to its caller.
    async def test_one_result_per_worker_reaches_the_caller(self):
        # [FUNCTION DOC] Configures a single two-worker cell to return a normal outcome carrying a
        # boxed critic loss value, trains, and asserts two results (one per worker) each with the
        # normal outcome and the original boxed value.
        """The critic values leave the group per worker so the driver can feed them to the actor."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        _set_train_return_value(
            cell, TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=Box({"critic_loss": 1.5}))
        )
        group = _make_controller([cell])

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert [result.outcome for result in results] == [TrainStepOutcome.NORMAL] * 2
        assert [result.values.inner for result in results] == [{"critic_loss": 1.5}] * 2

    async def test_results_of_several_cells_are_concatenated_in_cell_order(self):
        # [FUNCTION DOC] Gives two cells distinct boxed values (their indices), trains, and asserts
        # the four results are ordered cell 0's workers then cell 1's.
        """Independent DP ranks are positional, so a reordered result list misroutes values."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        for index, cell in enumerate(cells):
            _set_train_return_value(cell, TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=Box(index)))
        group = _make_controller(cells)

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert [result.values.inner for result in results] == [0, 0, 1, 1]

    async def test_a_failed_cell_contributes_no_result(self):
        # [FUNCTION DOC] Makes the first worker of cell 0 fail its train call and gives cell 1 a
        # successful boxed value. Trains and asserts only cell 1's two results are returned and
        # no exception object appears in the list.
        """A raw exception object in the returned list would be fed straight into the next train call."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        ray.get(get_raw_actor_handles(cells[0])[0].set_fail_methods.remote(["train"]))
        _set_train_return_value(cells[1], TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=Box("ok")))
        group = _make_controller(cells)

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert [result.values.inner for result in results] == ["ok", "ok"]


class TestRetryReturnsTheValue:
    # [CLASS DOC] Tests that the retry utility returns the value of the successful attempt.
    async def test_the_value_of_the_successful_attempt_is_returned(self):
        # [FUNCTION DOC] Retries an asynchronous function that fails on attempt zero and returns a
        # string on attempt one, with a no-op sleep. Asserts the string is returned and the
        # function was invoked with attempts zero then one.
        """train() reads its result through retry, so retry must stop swallowing it."""
        from miles.utils.retry_utils import retry

        attempts = []

        async def _fn(attempt: int) -> str:
            # [FUNCTION DOC] HELPER: Records the attempt number; raises a runtime error on attempt
            # zero and returns the string "second" otherwise.
            attempts.append(attempt)
            if attempt == 0:
                raise RuntimeError("boom")
            return "second"

        async def _no_sleep(_seconds: float) -> None:
            # [FUNCTION DOC] HELPER: Sleep replacement that returns immediately.
            return None

        assert await retry(_fn, sleep_fn=_no_sleep) == "second"
        assert attempts == [0, 1]


class TestWorkerResultShape:
    # [CLASS DOC] Tests how the controller interprets worker train results, in particular the
    # discarded-should-retry outcome.
    async def test_a_critic_payload_does_not_break_the_discarded_check(self):
        # [FUNCTION DOC] Returns a normal outcome whose boxed value is a mapping containing a key
        # that looks like an outcome field. Trains and asserts two results come back and the
        # workers were called only once each (two train calls in total, so no retry).
        """A dict payload carried inside TrainStepOutput.values must not be mistaken for a retry request."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        _set_train_return_value(
            cell, TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=Box({"train_step_outcome": "whatever"}))
        )
        group = _make_controller([cell])

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert len(results) == 2
        assert _count_train_calls(cell) == 2

    async def test_a_discarded_outcome_is_seen_by_the_outcome_check(self):
        # [FUNCTION DOC] Feeds the controller's static attempt-outcome computation one cell and
        # one result list holding a discarded-should-retry output, and asserts the cell index is
        # in the discarded bucket and the normal bucket is empty.
        """The critic's DISCARDED_SHOULD_RETRY must land in the discarded bucket rather than the normal one."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        results = [TrainStepOutput(outcome=TrainStepOutcome.DISCARDED_SHOULD_RETRY)]

        outcomes = TrainerController._compute_attempt_outcomes([cell], [results])

        assert outcomes["discarded"] == [0]
        assert outcomes["normal"] == []

    async def test_a_discarded_outcome_makes_the_group_retry_the_step(self):
        # [FUNCTION DOC] Configures every worker to return a discarded-should-retry output on the
        # first attempt and a normal output with boxed value "after_retry" on the second. Trains
        # and asserts the returned results carry the post-retry value for both workers and that
        # four train calls happened in total (two workers over two attempts).
        """A worker asking for a retry must cost a whole extra train attempt, not be silently accepted."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        for handle in get_raw_actor_handles(cell):
            ray.get(
                handle.set_train_return_values_per_attempt.remote(
                    [
                        TrainStepOutput(outcome=TrainStepOutcome.DISCARDED_SHOULD_RETRY),
                        TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=Box("after_retry")),
                    ]
                )
            )
        group = _make_controller([cell])

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert [result.values.inner for result in results] == ["after_retry"] * 2
        assert _count_train_calls(cell) == 4
