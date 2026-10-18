# [MODULE DOC]
# Tests for the creation and disposal of the trainer controller against a recording ray worker
# provider with a very short poll interval: init subscribes the controller's reconcile callback to
# the watch over its own trainer pool, the initial sync populates the cells before init returns,
# dispose stops the watch loop (no further polls) and clears the disposer handle, and dispose can
# be called repeatedly. The provider is backed by the fake worker manager of the shared conftest.
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import make_deployment_identity

from miles.ray.specs.train import compute_trainer_pool_id
from miles.ray.train.group import TrainerController
from miles.utils.workers.worker_provider.base import CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

pytestmark = pytest.mark.asyncio

_POOL_ID = compute_trainer_pool_id("actor")
_POLL_INTERVAL_SECONDS = 0.01


class _RecordingWorkerProvider(RayWorkerProvider):
    # [CLASS DOC] HELPER: Ray worker provider subclass that records every watch subscription
    # (callback plus the watched pool ids) and counts how many times the alive cells were polled.
    def __init__(self, *, worker_manager_handle: object, pool_ids: list[str] | None = None) -> None:
        # [FUNCTION DOC] HELPER: Initializes the base provider with the module's short poll
        # interval (0.01 seconds) and starts with no recorded watches and a zero poll count.
        super().__init__(
            worker_manager_handle=worker_manager_handle,
            pool_ids=pool_ids,
            poll_interval_seconds=_POLL_INTERVAL_SECONDS,
        )
        self.watch_calls: list[tuple[CellReconcileFn, list[str]]] = []
        self.poll_count: int = 0

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        # [FUNCTION DOC] HELPER: Records the reconcile callback together with a copy of the watched
        # pool ids, then delegates to the base implementation and returns its stop function.
        self.watch_calls.append((reconcile, list(self._watched_pool_ids())))
        return await super().watch_cells(reconcile)

    async def _list_alive_cells(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        # [FUNCTION DOC] HELPER: Increments the poll count and delegates to the base implementation.
        self.poll_count += 1
        return await super()._list_alive_cells(pool_ids=pool_ids)


def _make_args(*, num_cells: int) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds the minimal controller argument object: all-in-one deploy
    # component, an external actor controller address, independent dp on, witness off, standard
    # heartbeat checker settings, no fault test actions, tensor, pipeline and context parallel
    # sizes 1, one actor node whose gpu count equals the requested number of cells, ray object
    # store and worker comm backends, and no trainer model id.
    return SimpleNamespace(
        deploy_component="all",
        trainer_controller_addrs=["actor=10.0.0.1:8000"],
        api_server_port=1234,
        indep_dp=True,
        enable_witness=False,
        witness_buffer_size=100,
        save_debug_event_data=None,
        trainer_heartbeat_checker_interval=10.0,
        trainer_heartbeat_checker_timeout=10.0,
        trainer_heartbeat_checker_first_wait=300.0,
        trainer_heartbeat_checker_failure_threshold=3,
        ci_ft_test_actions=None,
        ci_ft_test_actions_path=None,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        actor_num_nodes=1,
        actor_num_gpus_per_node=num_cells,
        object_store_backend="ray",
        worker_comm_backend="ray",
        trainer_model_id=None,
    )


@pytest.fixture
def provider() -> _RecordingWorkerProvider:
    # [FUNCTION DOC] HELPER: Fixture returning a recording provider over the current fake worker
    # manager, watching only the actor trainer pool.
    return _RecordingWorkerProvider(worker_manager_handle=train_conftest.fake_worker_manager, pool_ids=[_POOL_ID])


async def _create_controller(*, num_cells: int, provider: _RecordingWorkerProvider) -> TrainerController:
    # [FUNCTION DOC] HELPER: Sets the fake manager's cell count, builds an actor-role controller
    # over the given provider (mock cell operations, default deployment identity, no reference
    # model), awaits its init with arguments for that many cells, and returns it. Init starts the
    # provider watch, so the caller must dispose the controller.
    train_conftest.fake_worker_manager.num_cells = num_cells
    controller = TrainerController(
        deployment_identity=make_deployment_identity(),
        trainer_id="actor",
        role="actor",
        with_ref=False,
        cell_provider=provider,
        cell_operations=MagicMock(),
    )
    await controller.init(_make_args(num_cells=num_cells))
    return controller


class TestCreate:
    # [CLASS DOC] Tests controller initialization and disposal with respect to the provider watch.
    async def test_create_subscribes_reconcile_to_the_trainer_spec(self, provider):
        # [FUNCTION DOC] After creating a 2-cell controller, the provider recorded exactly one
        # watch whose callback is the controller's own reconcile method and whose pool ids are just
        # the actor trainer pool. The controller is disposed at the end even if assertions fail.
        """create() must watch its own trainer spec with the controller's reconcile callback."""
        controller = await _create_controller(provider=provider, num_cells=2)
        try:
            assert len(provider.watch_calls) == 1
            reconcile, pool_ids = provider.watch_calls[0]
            assert reconcile == controller._reconcile
            assert pool_ids == [_POOL_ID]
        finally:
            await controller.dispose()

    async def test_create_populates_cells_from_the_initial_sync(self, provider):
        # [FUNCTION DOC] After creating a 2-cell controller, its cells-by-id mapping and its cell
        # list both contain cell indices 0 and 1 (sorted). Disposed in a finally block.
        """The initial watch sync must fill in the cells before create() returns."""
        controller = await _create_controller(provider=provider, num_cells=2)
        try:
            assert sorted(cell.cell_index for cell in controller._cells_by_id.values()) == [0, 1]
            assert [cell.cell_index for cell in controller._cells] == [0, 1]
        finally:
            await controller.dispose()

    async def test_dispose_stops_the_watch_loop(self, provider):
        # [FUNCTION DOC] Creates a 1-cell controller, lets polling run for five poll intervals,
        # disposes, records the poll count, waits another five intervals, and requires the count
        # unchanged and the controller's watcher disposer cleared to none. Uses real sleeps of
        # about 0.05 seconds each.
        """Without dispose() the 5-second poll loop outlives training and keeps logging failures."""
        controller = await _create_controller(provider=provider, num_cells=1)
        await asyncio.sleep(_POLL_INTERVAL_SECONDS * 5)

        await controller.dispose()
        polls_after_dispose: int = provider.poll_count
        await asyncio.sleep(_POLL_INTERVAL_SECONDS * 5)

        assert provider.poll_count == polls_after_dispose
        assert controller._watcher_disposer is None

    async def test_dispose_is_idempotent(self, provider):
        # [FUNCTION DOC] Disposing a 1-cell controller twice in a row raises no error.
        """Teardown paths overlap, so a second dispose must not raise."""
        controller = await _create_controller(provider=provider, num_cells=1)

        await controller.dispose()
        await controller.dispose()
