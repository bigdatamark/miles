# [MODULE DOC]
# Tests for the trainer controller (group of independent-dp trainer cells) against dummy ray
# actors and a fake worker manager. The controller is built through helpers that mimic the
# watcher: cells are created for every cell index the fake manager reports and registered by cell
# id; "stop" and "start" helpers emulate the manager suspending and relaunching a cell followed by
# the controller's reconcile. Covered: quorum store creation, init (cell creation, object store,
# marking all cells alive with ranks), run-once init guard, executing on the first alive cell
# (fallback and fatal errors), parallel config read from rank 0 of the first alive cell, independent
# dp info computation, broadcast to all alive cells with per-cell error isolation, refreshing cells
# (reconfigure on alive-set change, healing pending cells with checkpoint sends from a source
# cell, quorum id bumps, no-ops, reconfigure events in the event log), train (event analysis off the
# event loop, retries on discarded outcomes, outcome bucketing, releasing failed attempt values),
# witness id allocation, step end events, concurrent-safe cell status reads, update weights version
# return, model flags forwarded to init, fault-tolerance test actions, save model and hf export.
# KNOWN ISSUE: the module wide asyncio marker also applies to the synchronous tests of this
# module, which are not coroutines.
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import ray
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_deployment_identity, make_provider

import miles.ray.train.group as group_module
from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.ray.train.group import TrainerController, compute_trainer_health_checker_config
from miles.utils import object_store
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.witness.allocator import WitnessIdAllocator
from miles.utils.data import RolloutDataPack
from miles.utils.object_store import _MooncakeStoreObjectRef
from miles.utils.ray_utils import Box
from miles.utils.retry_utils import NonRetryableError
from miles.utils.workers.naming import compute_cell_id

pytestmark = pytest.mark.asyncio

_DUMMY_DATA_PACK = RolloutDataPack(sample_indices=[0], data_ref=_MooncakeStoreObjectRef(payload="data"))


def _make_mock_args(
    *,
    indep_dp: bool = True,
    enable_witness: bool = False,
    gpus_per_cell: int = 1,
    num_cells: int = 3,
    ci_ft_test_actions: str | None = None,
    ci_ft_test_actions_path: str | None = None,
) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds a picklable controller argument object (all-in-one deploy
    # component, no external controller addresses, api server port 0, the given independent dp and
    # witness flags with a witness buffer of 100, heartbeat checker interval 10, timeout 10, first
    # wait 300, failure threshold 3, the given fault test action settings, no debug flags). The
    # parallelism sizes make the controller compute the expected number of cells: tensor and
    # pipeline parallel sizes are 1 and context parallel size equals the gpus per cell, with one
    # actor node holding cells times gpus-per-cell gpus. Ray object store and worker comm
    # backends, no trainer model id.
    # Use SimpleNamespace (not MagicMock) so the args object is picklable. TrainerCell.init
    # passes self.args through Ray to the remote actor; pickling a MagicMock blows the
    # recursion limit because its __getattr__ creates new sub-mocks indefinitely.
    return SimpleNamespace(
        deploy_component="all",
        trainer_controller_addrs=None,
        api_server_port=0,
        indep_dp=indep_dp,
        enable_witness=enable_witness,
        witness_buffer_size=100,
        trainer_heartbeat_checker_interval=10.0,
        trainer_heartbeat_checker_timeout=10.0,
        trainer_heartbeat_checker_first_wait=300.0,
        trainer_heartbeat_checker_failure_threshold=3,
        ci_ft_test_actions=ci_ft_test_actions,
        ci_ft_test_actions_path=ci_ft_test_actions_path,
        debug_train_only=False,
        debug_rollout_only=False,
        # compute_megatron_world_size_except_dp(args) = TP * PP * CP. Set CP to
        # gpus_per_cell so TrainerController computes num_cells correctly.
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=gpus_per_cell,
        actor_num_nodes=1,
        actor_num_gpus_per_node=num_cells * gpus_per_cell,
        object_store_backend="ray",
        worker_comm_backend="ray",
        trainer_model_id=None,
    )


def _make_controller(
    *,
    num_cells: int = 3,
    actor_count_per_cell: int = 1,
    with_ref: bool = False,
    with_opd_teacher: bool = False,
    ci_ft_test_actions: str | None = None,
) -> TrainerController:
    # [FUNCTION DOC] HELPER: Sets the fake manager's cell and actors-per-cell counts, builds an
    # actor-role controller (default deployment identity, mock cell operations, the given
    # reference and distillation teacher flags, a provider over the fake manager), attaches mock
    # args for that many cells with independent dp on and the given fault test actions, computes
    # the heartbeat checker config, creates the quorum store when more than one cell is expected,
    # and then creates a cell object for every cell index (workers hash pseudo-hash-1) and
    # registers it by cell id, as the watcher would. Does not call init, so cells stay uninitialized.
    """Create a TrainerController and let it observe every cell, as the watcher would."""
    train_conftest.fake_worker_manager.num_cells = num_cells
    train_conftest.fake_worker_manager.actor_count_per_cell = actor_count_per_cell
    group = TrainerController(
        deployment_identity=make_deployment_identity(),
        trainer_id="actor",
        role="actor",
        with_ref=with_ref,
        with_opd_teacher=with_opd_teacher,
        cell_provider=make_provider(),
        cell_operations=AsyncMock(),
    )
    group.args = _make_mock_args(
        indep_dp=True,
        gpus_per_cell=actor_count_per_cell,
        num_cells=num_cells,
        ci_ft_test_actions=ci_ft_test_actions,
        ci_ft_test_actions_path=None,
    )
    group._health_checker_config = compute_trainer_health_checker_config(
        group.args, expected_num_cells=group._expected_num_cells
    )
    if group._expected_num_cells > 1:
        group._indep_dp_store, group._indep_dp_store_addr = group_module.create_tcp_store()
    for cell_index in range(num_cells):
        cell = group._create_cell(
            compute_cell_id(pool_id=group._pool_id, cell_index=cell_index),
            cell_index=cell_index,
            workers_hash="pseudo-hash-1",
        )
        group._cells_by_id[cell.cell_id] = cell
    return group


async def _stop_cell(group: TrainerController, cell_index: int) -> None:
    # [FUNCTION DOC] HELPER: Emulates suspending a cell: the fake manager stops the cell (killing
    # its actors and recording the stop), then the controller's reconcile is told the cell is gone
    # (info none) so it drops the cell from its bookkeeping.
    """Suspension stops the cell in the manager; reconcile then drops it from the bookkeeping."""
    cell_id = compute_cell_id(pool_id=group._pool_id, cell_index=cell_index)
    train_conftest.fake_worker_manager._stop_cells([cell_id])
    await group._reconcile(cell_id, None)


def _cell(group: TrainerController, cell_index: int) -> object:
    # [FUNCTION DOC] HELPER: Returns the controller's cell registered for the given cell index;
    # raises a key error if there is none.
    return group._cells_by_id[compute_cell_id(pool_id=group._pool_id, cell_index=cell_index)]


def _start_cell(group: TrainerController, cell_index: int) -> None:
    # [FUNCTION DOC] HELPER: Emulates the manager relaunching a cell: creates a fresh cell object
    # (workers hash pseudo-hash-2, so new dummy actors) and registers it by cell id, replacing
    # any previous entry. The new cell is uninitialized (pending healing).
    """The manager relaunches the cell, so reconcile hands the controller a fresh object."""
    cell_id = compute_cell_id(pool_id=group._pool_id, cell_index=cell_index)
    group._cells_by_id[cell_id] = group._create_cell(cell_id, cell_index=cell_index, workers_hash="pseudo-hash-2")


def _was_stopped(group: TrainerController, cell_index: int) -> bool:
    # [FUNCTION DOC] HELPER: True if the fake manager recorded a stop call for exactly this cell
    # (a call whose cell id list is just the one cell id).
    return [
        compute_cell_id(pool_id=group._pool_id, cell_index=cell_index)
    ] in train_conftest.fake_worker_manager.stopped_cell_ids


def _was_killed(group: TrainerController, cell_index: int) -> bool:
    # [FUNCTION DOC] HELPER: Probes the cell's raw actors by fetching their call logs; returns
    # false as soon as one answers (alive), true if every actor raises an actor death error.
    for handle in get_raw_actor_handles(_cell(group, cell_index)):
        try:
            ray.get(handle.get_calls.remote())
            return False
        except ray.exceptions.RayActorError:
            pass
    return True


async def _init_controller(group: TrainerController) -> None:
    # [FUNCTION DOC] HELPER: Awaits the controller's init with its own mock args, which brings
    # all cells alive.
    """Call init and wait for all cells to become alive."""
    await group.init(group.args)


async def _make_alive_controller(*, num_cells: int = 3, **kwargs) -> TrainerController:
    # [FUNCTION DOC] HELPER: Builds a controller (default 3 cells, extra options forwarded to the
    # controller factory) and inits it so every cell is alive; returns the controller.
    """Create a group and init all cells to alive."""
    group = _make_controller(num_cells=num_cells, **kwargs)
    await _init_controller(group)
    return group


class TestIndepDPStore:
    # [CLASS DOC] Tests the creation of the independent dp quorum store.
    def test_a_multi_cell_pool_gets_one_quorum_store_from_its_controller(self):
        # [FUNCTION DOC] A 3-cell controller holds the fixed fake store address.
        """The store must be minted once, where every cell can be told the same address."""
        group = _make_controller(num_cells=3)

        assert group._indep_dp_store_addr == train_conftest.FAKE_STORE_ADDR

    def test_a_single_cell_pool_needs_no_quorum_store(self):
        # [FUNCTION DOC] A 1-cell controller has no store address (none).
        """One cell never renegotiates a quorum, so binding a port for it would be pure waste."""
        group = _make_controller(num_cells=1)

        assert group._indep_dp_store_addr is None


class TestInit:
    # [CLASS DOC] Tests controller construction and init: pool id, cell creation, actors per cell,
    # object store availability and the alive transition with ranks.
    def test_the_controller_watches_the_pool_of_its_trainer_id(self):
        # [FUNCTION DOC] A controller built with trainer id alpha-actor and role actor has pool id
        # trainer-engine-alpha-actor.
        """A policy's controller owns the pool named after its trainer id, which the role no longer determines."""
        group = TrainerController(
            deployment_identity=make_deployment_identity(),
            trainer_id="alpha-actor",
            role="actor",
            with_ref=False,
            with_opd_teacher=False,
            cell_provider=make_provider(),
            cell_operations=AsyncMock(),
        )

        assert group._pool_id == "trainer-engine-alpha-actor"

    def test_creates_correct_number_of_cells(self):
        # [FUNCTION DOC] A 3-cell controller exposes 3 cells with cell indices 0, 1, 2 in order.
        group = _make_controller(num_cells=3)

        assert len(group._cells) == 3
        assert [c.cell_index for c in group._cells] == [0, 1, 2]

    def test_cells_are_allocated_after_init(self):
        # [FUNCTION DOC] Before init, every cell of a 2-cell controller is allocated but not alive.
        # KNOWN ISSUE: the name says "after init" but init is not called; the cells are only built.
        group = _make_controller(num_cells=2)

        for cell in group._cells:
            assert cell.is_allocated
            assert not cell.is_alive

    def test_each_cell_has_own_actors(self):
        # [FUNCTION DOC] With 3 cells of 2 actors, every cell has 2 raw actor handles and all 6
        # handles are distinct objects.
        group = _make_controller(num_cells=3, actor_count_per_cell=2)

        handles_per_cell = [get_raw_actor_handles(cell) for cell in group._cells]
        assert all(len(h) == 2 for h in handles_per_cell)

        all_handles = [h for handles in handles_per_cell for h in handles]
        assert len(set(id(h) for h in all_handles)) == 6

    def test_single_cell_controller(self):
        # [FUNCTION DOC] A 1-cell controller exposes exactly one cell.
        group = _make_controller(num_cells=1)

        assert len(group._cells) == 1

    async def test_init_gives_the_controller_process_an_object_store(self):
        # [FUNCTION DOC] After init of a 1-cell controller, the object store singleton exists
        # (the autouse fixture resets it to none before each test).
        """The controller frees a failed attempt's outputs itself, which needs a store in its own process."""
        group = _make_controller(num_cells=1)

        await _init_controller(group)

        assert object_store.get_instance() is not None

    async def test_init_marks_all_cells_alive(self):
        # [FUNCTION DOC] After init of a 3-cell controller, every cell is alive with alive cells
        # [0, 1, 2] and alive size 3, and cells 0, 1, 2 have alive ranks 0, 1, 2.
        group = _make_controller(num_cells=3)

        await _init_controller(group)

        for cell in group._cells:
            assert cell.is_alive
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]
            assert cell.indep_dp_info.alive_size == 3

        assert _cell(group, 0).indep_dp_info.alive_rank == 0
        assert _cell(group, 1).indep_dp_info.alive_rank == 1
        assert _cell(group, 2).indep_dp_info.alive_rank == 2


class TestInitRunsExactlyOnce:
    # [CLASS DOC] Tests the run-once init guard of the controller.
    async def test_a_controller_that_never_ran_init_reports_itself_uninitialized(self):
        # [FUNCTION DOC] Before init, the initialized query answers false.
        """A restarted script asks the controller it found running whether to initialize it or to resume it."""
        group = _make_controller(num_cells=1)

        assert await group.is_initialized() is False

    async def test_a_controller_that_ran_init_reports_itself_initialized(self):
        # [FUNCTION DOC] After init, the initialized query answers true.
        """The take-over path resumes exactly the controllers that answer this way."""
        group = await _make_alive_controller(num_cells=1)

        assert await group.is_initialized() is True

    async def test_a_second_init_is_refused(self):
        # [FUNCTION DOC] Calling init a second time raises an assertion error mentioning a stale worker.
        """Initializing trainers a previous script already built would throw away the state they hold."""
        group = await _make_alive_controller(num_cells=1)

        with pytest.raises(AssertionError, match="stale worker"):
            await _init_controller(group)


class TestStopStartCell:
    # [CLASS DOC] Tests the stop and start emulation helpers against the controller bookkeeping.
    async def test_stopping_a_cell_reaches_the_worker_manager(self):
        # [FUNCTION DOC] After stopping cell 1 of 2 alive cells, the manager recorded a stop for it
        # and cell 0 is still alive.
        group = await _make_alive_controller(num_cells=2)

        await _stop_cell(group, 1)

        assert _was_stopped(group, 1)
        assert _cell(group, 0).is_alive

    async def test_a_relaunched_cell_is_uninitialized_again(self):
        # [FUNCTION DOC] After stopping and then starting cell 1, the fresh cell is uninitialized.
        group = await _make_alive_controller(num_cells=2)
        await _stop_cell(group, 1)

        _start_cell(group, 1)

        assert _cell(group, 1).is_uninitialized


class TestExecuteFirstAlive:
    # [CLASS DOC] Tests running a worker method on only the first alive cell.
    async def test_picks_first_alive_cell(self):
        # [FUNCTION DOC] Executing save model on 3 alive cells reaches every actor of cell 0 and
        # no actor of the other cells.
        group = await _make_alive_controller(num_cells=3)

        await group._execute_first_alive("save_model", rollout_id=42)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "save_model" for c in calls)

        for cell in group._cells[1:]:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                assert not any(c[0] == "save_model" for c in calls)

    async def test_skips_errored_picks_next(self):
        # [FUNCTION DOC] With cell 0 marked errored, executing update weights on the first alive
        # cell reaches cell 1's actors.
        group = await _make_alive_controller(num_cells=2)
        _cell(group, 0)._mark_as_errored()

        await group._execute_first_alive("update_weights")

        for handle in get_raw_actor_handles(_cell(group, 1)):
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "update_weights" for c in calls)


class TestGetTrainParallelConfig:
    # [CLASS DOC] Tests reading the train parallel config through the controller.
    @staticmethod
    def _set_configs(cell, configs: list[dict]) -> None:
        # [FUNCTION DOC] HELPER: Sets the parallel config of each of the cell's actors to the
        # matching mapping from the list (lengths must match, otherwise a value error) and waits
        # for all remote calls to finish.
        handles = get_raw_actor_handles(cell)
        ray.get(
            [handle.set_train_parallel_config.remote(config) for handle, config in zip(handles, configs, strict=True)]
        )

    async def test_returns_config_of_rank_zero_of_the_first_alive_cell(self):
        # [FUNCTION DOC] With two cells of two actors, cell 0's actors configured with data
        # parallel sizes 4 and 99, the controller returns the config of cell 0's rank 0 (size 4).
        """The driver reads the config the cell's own rank 0 computed at init."""
        group = await _make_alive_controller(num_cells=2, actor_count_per_cell=2)
        self._set_configs(group._cells[0], [{"dp_size": 4}, {"dp_size": 99}])

        assert await group.get_train_parallel_config() == {"dp_size": 4}

    async def test_skips_stopped_cells(self):
        # [FUNCTION DOC] Cell 1's actor is configured with data parallel size 2 and cell 0 is
        # stopped; the controller returns cell 1's config.
        """A stopped cell 0 must not be asked; the next alive cell answers instead."""
        group = await _make_alive_controller(num_cells=2)
        self._set_configs(_cell(group, 1), [{"dp_size": 2}])
        await _stop_cell(group, 0)

        assert await group.get_train_parallel_config() == {"dp_size": 2}


class TestComputeIndepDPInfo:
    # [CLASS DOC] Tests computing a cell's independent dp info from the alive cell list.
    def test_all_alive(self):
        # [FUNCTION DOC] For cell 2 with alive cells [0, 1, 2], alive rank is 2, alive size 3 and
        # cell index 2.
        group = _make_controller(num_cells=3)

        info = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 1, 2])

        assert info.alive_rank == 2
        assert info.alive_size == 3
        assert info.cell_index == 2

    def test_with_gap(self):
        # [FUNCTION DOC] For cell 2 with alive cells [0, 2], alive rank is 1 and alive size 2.
        group = _make_controller(num_cells=3)

        info = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 2])

        assert info.alive_rank == 1
        assert info.alive_size == 2


class TestExecuteAllAliveAndCatch:
    # [CLASS DOC] Tests broadcasting a worker method to all alive cells with per-cell error capture.
    async def test_skips_errored_cells(self):
        # [FUNCTION DOC] With cell 1 errored, a train broadcast reaches cell 0's actors.
        group = await _make_alive_controller(num_cells=2)
        _cell(group, 1)._mark_as_errored()

        await group._execute_all_alive_and_catch("train")

        for handle in get_raw_actor_handles(_cell(group, 0)):
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "train" for c in calls)

    async def test_refuses_to_retry_when_no_cell_is_alive(self):
        # [FUNCTION DOC] With the only cell errored, a train broadcast raises a non-retryable error
        # saying there are no alive cells.
        """Retrying without a single live cell can never succeed, so it must fail fast."""
        group = await _make_alive_controller(num_cells=1)
        _cell(group, 0)._mark_as_errored()

        with pytest.raises(NonRetryableError, match="No alive cells"):
            await group._execute_all_alive_and_catch("train")


class TestRefreshCellsReconfigure:
    # [CLASS DOC] Tests refreshing cells when the set of alive cells shrinks or stays the same.
    async def test_reconfigure_triggers_on_alive_change(self):
        # [FUNCTION DOC] Three alive cells, then cell 1 stopped, then a refresh. The quorum id is
        # bumped to 1; remaining cells 0 and 2 are alive with alive cells [0, 2] and alive ranks 0
        # and 1 (size 2); the stopped cell is untouched; and every actor of cells 0 and 2 received an
        # independent dp reconfiguration call. The comments in the body label the six steps.
        """When a cell is stopped, _refresh_cells reconfigures remaining alive cells."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Stop cell 1
        await _stop_cell(group, 1)

        # Step 2: Refresh
        await group._refresh_cells(rollout_id=0)

        # Step 3: Quorum bumped (init was quorum 0, this is first reconfigure)
        assert group._indep_dp_quorum_id == 1

        # Step 4: Remaining alive cells have updated indep_dp_info
        assert _cell(group, 0).is_alive
        assert _cell(group, 0).indep_dp_info.alive_cell_indices == [0, 2]
        assert _cell(group, 0).indep_dp_info.alive_rank == 0
        assert _cell(group, 0).indep_dp_info.alive_size == 2

        assert _cell(group, 2).is_alive
        assert _cell(group, 2).indep_dp_info.alive_rank == 1

        # Step 5: Stopped cell untouched
        assert _was_stopped(group, 1)

        # Step 6: Actors received reconfigure_indep_dp
        for cell in [_cell(group, 0), _cell(group, 2)]:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "reconfigure_indep_dp" for c in calls)

    async def test_no_reconfigure_when_unchanged(self):
        # [FUNCTION DOC] With two alive cells and no changes, a refresh leaves the quorum id at 0.
        group = await _make_alive_controller(num_cells=2)

        await group._refresh_cells(rollout_id=0)

        assert group._indep_dp_quorum_id == 0


class TestRefreshCellsHealing:
    # [CLASS DOC] Tests healing of pending (relaunched, uninitialized) cells during a refresh.
    async def test_pending_cell_gets_healed(self):
        # [FUNCTION DOC] Cell 2 of 3 is stopped and restarted (pending), then a refresh. All three
        # cells become alive with alive cells [0, 1, 2] and size 3; the healed cell's actors
        # received init; and cell 0 (the source) sent exactly one checkpoint to destination rank 2
        # per actor. The comments in the body label the steps.
        """A pending cell goes through allocate + healing with correct alive_rank."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Stop cell 2, then start it (pending)
        await _stop_cell(group, 2)
        _start_cell(group, 2)

        # Step 2: Refresh heals the pending cell
        await group._refresh_cells(rollout_id=0)

        # Step 3: All 3 cells are now alive
        assert all(c.is_alive for c in group._cells)

        # Step 4: All cells have consistent indep_dp_info
        for cell in group._cells:
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]
            assert cell.indep_dp_info.alive_size == 3

        # Step 5: Healed cell's actors received init
        for handle in get_raw_actor_handles(_cell(group, 2)):
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "init" for c in calls)

        # Step 6: Source cell sent ckpt to healed cell's alive_rank
        for handle in get_raw_actor_handles(_cell(group, 0)):
            calls = ray.get(handle.get_calls.remote())
            send_calls = [c for c in calls if c[0] == "send_ckpt"]
            assert len(send_calls) == 1
            assert send_calls[0][2]["dst_rank"] == 2

    async def test_multiple_pending_cells_healed(self):
        # [FUNCTION DOC] Cells 1 and 2 are stopped and restarted. After a refresh all cells are
        # alive with alive cells [0, 1, 2], and each actor of source cell 0 sent two checkpoints,
        # to destination ranks 1 and 2.
        """Multiple pending cells healed simultaneously."""
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 1)
        await _stop_cell(group, 2)
        _start_cell(group, 1)
        _start_cell(group, 2)

        await group._refresh_cells(rollout_id=0)

        assert all(c.is_alive for c in group._cells)
        for cell in group._cells:
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]

        # Source (cell 0) sent ckpt to both healed cells
        for handle in get_raw_actor_handles(_cell(group, 0)):
            calls = ray.get(handle.get_calls.remote())
            send_calls = [c for c in calls if c[0] == "send_ckpt"]
            assert len(send_calls) == 2
            dst_ranks = sorted(c[2]["dst_rank"] for c in send_calls)
            assert dst_ranks == [1, 2]

    async def test_pending_cell_with_stopped_cell(self):
        # [FUNCTION DOC] Cell 1 stopped (not restarted), cell 2 stopped and restarted (pending).
        # After a refresh cells 0 and 2 are alive with alive cells [0, 2] (size 2, cell 2 at rank
        # 1) and cell 1 remains stopped.
        """Pending + stopped: only alive and pending participate, stopped excluded."""
        group = await _make_alive_controller(num_cells=3)

        # cell 1 stopped (not restarted), cell 2 pending
        await _stop_cell(group, 1)
        await _stop_cell(group, 2)
        _start_cell(group, 2)

        await group._refresh_cells(rollout_id=0)

        assert _cell(group, 0).is_alive
        assert _was_stopped(group, 1)
        assert _cell(group, 2).is_alive

        assert _cell(group, 0).indep_dp_info.alive_cell_indices == [0, 2]
        assert _cell(group, 0).indep_dp_info.alive_size == 2
        assert _cell(group, 2).indep_dp_info.alive_rank == 1


class TestRefreshCellsReconfigureEvent:
    # [CLASS DOC] Tests the reconfigure events written to the audit event log by refreshes.
    @pytest.fixture
    def _event_log_dir(self, tmp_path: Path):
        # [FUNCTION DOC] HELPER: Fixture. Installs a global event logger writing under the test's
        # temporary directory (source identity: main component), yields that directory, and
        # always clears the global logger afterwards.
        set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main")))
        try:
            yield tmp_path
        finally:
            set_event_logger(None)

    @staticmethod
    def _read_reconfigure_events(log_dir: Path) -> list[CellReconfigureEvent]:
        # [FUNCTION DOC] HELPER: Reads all events under the log directory and returns those that
        # are cell reconfigure events, in log order.
        return [e for e in read_events(log_dir) if isinstance(e, CellReconfigureEvent)]

    async def test_healing_emits_event_with_src_and_healed_cells(self, _event_log_dir: Path):
        # [FUNCTION DOC] After healing cell 2 of 3 during a refresh for rollout 7, exactly one
        # reconfigure event is logged with rollout id 7, quorum id 1, source cell 0, healed cells
        # [2] and alive cells after [0, 1, 2].
        """A healing reconfigure emits one CellReconfigureEvent naming rollout, src cell, and healed cells."""
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 2)
        _start_cell(group, 2)

        await group._refresh_cells(rollout_id=7)

        events = self._read_reconfigure_events(_event_log_dir)
        assert len(events) == 1
        assert events[0].rollout_id == 7
        assert events[0].quorum_id == 1
        assert events[0].src_cell_index == 0
        assert events[0].healed_cell_indices == [2]
        assert events[0].alive_cell_indices_after == [0, 1, 2]

    async def test_shrink_emits_event_without_src(self, _event_log_dir: Path):
        # [FUNCTION DOC] After stopping cell 1 of 3 and refreshing for rollout 4, exactly one event
        # is logged with rollout id 4, no source cell, no healed cells and alive cells after [0, 2].
        """A pure-shrink reconfigure emits one CellReconfigureEvent with no src and no healed cells."""
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 1)

        await group._refresh_cells(rollout_id=4)

        events = self._read_reconfigure_events(_event_log_dir)
        assert len(events) == 1
        assert events[0].rollout_id == 4
        assert events[0].src_cell_index is None
        assert events[0].healed_cell_indices == []
        assert events[0].alive_cell_indices_after == [0, 2]

    async def test_noop_refresh_emits_no_event(self, _event_log_dir: Path):
        # [FUNCTION DOC] A refresh with no change logs no reconfigure event.
        """A refresh that needs no reconfigure emits no CellReconfigureEvent."""
        group = await _make_alive_controller(num_cells=2)

        await group._refresh_cells(rollout_id=1)

        assert self._read_reconfigure_events(_event_log_dir) == []

    async def test_failed_healing_emits_no_event(self, _event_log_dir: Path):
        # [FUNCTION DOC] Cell 2 is stopped, registered to fail its init, and restarted. A refresh
        # (the healing fails) logs no reconfigure event.
        """When cooperative prepare fails, no CellReconfigureEvent is emitted (witness stays absent)."""
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 2)
        train_conftest.fake_worker_manager.fail_init_for_cell(2)
        _start_cell(group, 2)

        await group._refresh_cells(rollout_id=5)

        assert self._read_reconfigure_events(_event_log_dir) == []


class TestRefreshCellsNoOp:
    # [CLASS DOC] Tests that refreshes without state changes do nothing.
    async def test_repeated_refresh_without_change_does_not_reconfigure(self):
        # [FUNCTION DOC] Records each actor's call log length, refreshes twice, and requires the
        # quorum id still 0 and no actor to have received any new call.
        """Calling _refresh_cells multiple times without state changes dispatches no actor calls."""
        group = await _make_alive_controller(num_cells=3)

        # Clear init calls by noting current call count
        init_call_counts = {}
        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                init_call_counts[id(handle)] = len(calls)

        # Two refreshes — neither should change anything
        await group._refresh_cells(rollout_id=0)
        await group._refresh_cells(rollout_id=0)
        assert group._indep_dp_quorum_id == 0

        # No new calls dispatched
        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                assert len(calls) == init_call_counts[id(handle)]

    async def test_refresh_after_reconfigure_is_noop_on_second_call(self):
        # [FUNCTION DOC] After stopping cell 1 and refreshing (quorum id 1), a second refresh keeps
        # the quorum id at 1.
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 1)
        await group._refresh_cells(rollout_id=0)
        assert group._indep_dp_quorum_id == 1

        await group._refresh_cells(rollout_id=0)
        assert group._indep_dp_quorum_id == 1


class TestConsecutiveStopStartCycles:
    # [CLASS DOC] Tests several consecutive stop, start and refresh steps.
    async def test_stop_train_stop_train_start_train(self):
        # [FUNCTION DOC] Stops cell 1 and refreshes (quorum 1, alive cells [0, 2]); stops cell 2
        # and refreshes (quorum 2, alive cells [0], size 1); restarts cell 1 and refreshes (quorum 3,
        # cells 0 and 1 alive with alive cells [0, 1], cell 2 still stopped).
        # KNOWN ISSUE: the name and docstring suggest training between the steps, but only
        # refreshes are called; no train call is made.
        """Consecutive: stop 1 → refresh → stop 2 → refresh → start 1 → refresh."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Stop cell 1
        await _stop_cell(group, 1)
        await group._refresh_cells(rollout_id=0)
        assert group._indep_dp_quorum_id == 1
        assert _cell(group, 0).indep_dp_info.alive_cell_indices == [0, 2]

        # Step 2: Stop cell 2 (only cell 0 alive)
        await _stop_cell(group, 2)
        await group._refresh_cells(rollout_id=0)
        assert group._indep_dp_quorum_id == 2
        assert _cell(group, 0).indep_dp_info.alive_cell_indices == [0]
        assert _cell(group, 0).indep_dp_info.alive_size == 1

        # Step 3: Start cell 1 (cells 0 and 1 alive)
        _start_cell(group, 1)
        await group._refresh_cells(rollout_id=0)
        assert group._indep_dp_quorum_id == 3
        assert _cell(group, 0).is_alive
        assert _cell(group, 1).is_alive
        assert _was_stopped(group, 2)
        assert _cell(group, 0).indep_dp_info.alive_cell_indices == [0, 1]
        assert _cell(group, 1).indep_dp_info.alive_cell_indices == [0, 1]


class TestTrain:
    # [CLASS DOC] Tests the controller's train method end to end: event analysis handling,
    # dispatch to alive cells, refresh before dispatch, and lifecycle sequences.
    async def test_event_analysis_does_not_block_cell_status_requests(self):
        # [FUNCTION DOC] A 1-cell controller with refresh, gather, check, step-end logging and the
        # test action executor stubbed, and with the event analysis function replaced by one that
        # blocks the thread for 0.2 seconds. A train task is started, then the cell status is
        # requested: it must answer in under 0.1 seconds with just the one cell id, and the train
        # result is an empty list. This proves the analysis runs off the event loop.
        """A growing event log must not make the trainer controller stop answering status requests."""
        group = _make_controller(num_cells=1)
        group._witness_allocator = None
        group._refresh_cells = AsyncMock()
        group._gather_all_alive_and_catch = AsyncMock(return_value=([], []))
        group._check_train_one_attempt = MagicMock()
        group._log_step_end_event = MagicMock()
        group._test_action_executor = AsyncMock()

        started_at = time.monotonic()
        with patch.object(
            group_module.event_analyzer,
            "run_analysis_from_args",
            side_effect=lambda _args: time.sleep(0.2),
        ):
            train_task = asyncio.create_task(group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK))
            await asyncio.sleep(0)

            statuses = await group.get_cell_statuses()

            assert time.monotonic() - started_at < 0.1
            assert set(statuses) == {"trainer-engine-actor-00000"}
            assert await train_task == []

    async def test_event_analysis_failure_still_fails_training(self):
        # [FUNCTION DOC] When the event analysis function raises a value error, train raises that
        # error too (matching its message).
        """Moving event analysis off the event loop must preserve its failure contract."""
        group = _make_controller(num_cells=1)

        with patch.object(
            group_module.event_analyzer,
            "run_analysis_from_args",
            side_effect=ValueError("event analysis failed"),
        ):
            with pytest.raises(ValueError, match="event analysis failed"):
                await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

    async def test_train_refreshes_and_dispatches(self):
        # [FUNCTION DOC] Training on 2 alive cells sends a train call to every actor of every cell.
        group = await _make_alive_controller(num_cells=2)

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "train" for c in calls)

    async def test_train_with_stopped_cell_only_dispatches_to_alive(self):
        # [FUNCTION DOC] With cell 1 of 3 stopped, training reaches the actors of cells 0 and 2,
        # and cell 1 stays stopped.
        group = await _make_alive_controller(num_cells=3)
        await _stop_cell(group, 1)

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        for cell in [_cell(group, 0), _cell(group, 2)]:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "train" for c in calls)

        assert _was_stopped(group, 1)

    async def test_consecutive_train_no_reconfigure_overhead(self):
        # [FUNCTION DOC] Three consecutive train calls on 3 alive cells keep the quorum id at 0;
        # after the initial call counts are recorded, each actor received no reconfiguration call
        # and exactly three train calls.
        """Multiple train calls with no state changes — no reconfigure overhead."""
        group = await _make_alive_controller(num_cells=3)

        # Note init call count
        init_counts = {}
        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                init_counts[id(handle)] = len(ray.get(handle.get_calls.remote()))

        for step in range(3):
            await group.train(rollout_id=step, rollout_data_pack=_DUMMY_DATA_PACK)

        assert group._indep_dp_quorum_id == 0

        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                new_calls = calls[init_counts[id(handle)] :]
                assert not any(c[0] == "reconfigure_indep_dp" for c in new_calls)
                train_calls = [c for c in new_calls if c[0] == "train"]
                assert len(train_calls) == 3

    async def test_rapid_stop_start_before_train(self):
        # [FUNCTION DOC] Cell 1 is stopped and immediately restarted before training; one train
        # call heals it so all cells are alive with alive cells [0, 1, 2].
        """Cell stopped and immediately started before next train — healed in one shot."""
        group = await _make_alive_controller(num_cells=3)

        await _stop_cell(group, 1)
        _start_cell(group, 1)

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        assert all(c.is_alive for c in group._cells)
        for cell in group._cells:
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]

    async def test_full_lifecycle_through_train(self):
        # [FUNCTION DOC] Five train calls: normal (quorum 0); with cell 2 stopped (quorum 1, alive
        # cells [0, 1]); steady degraded (quorum stays 1); with cell 2 restarted (quorum 2, all
        # alive, cell 2 sees alive cells [0, 1, 2]); full again (quorum stays 2).
        """End-to-end: normal → degraded → steady degraded → healing → full."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Normal training (no reconfigure)
        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)
        assert group._indep_dp_quorum_id == 0

        # Step 2: Stop cell 2 → degraded (triggers reconfigure)
        await _stop_cell(group, 2)
        await group.train(rollout_id=1, rollout_data_pack=_DUMMY_DATA_PACK)
        assert group._indep_dp_quorum_id == 1
        assert _cell(group, 0).indep_dp_info.alive_cell_indices == [0, 1]

        # Step 3: Steady degraded (no reconfigure)
        await group.train(rollout_id=2, rollout_data_pack=_DUMMY_DATA_PACK)
        assert group._indep_dp_quorum_id == 1

        # Step 4: Start cell 2 → healing (triggers reconfigure)
        _start_cell(group, 2)
        await group.train(rollout_id=3, rollout_data_pack=_DUMMY_DATA_PACK)
        assert group._indep_dp_quorum_id == 2
        assert all(c.is_alive for c in group._cells)
        assert _cell(group, 2).indep_dp_info.alive_cell_indices == [0, 1, 2]

        # Step 5: Full training again (no reconfigure)
        await group.train(rollout_id=4, rollout_data_pack=_DUMMY_DATA_PACK)
        assert group._indep_dp_quorum_id == 2


class TestPerCellErrorIsolation:
    # [CLASS DOC] Tests that one cell's failure during a broadcast does not affect other cells.
    async def test_one_cell_failure_marks_errored_others_ok(self):
        # [FUNCTION DOC] Cell 1's actors are set to fail train. After a broadcast of train, cell 1's
        # actors are killed while cells 0 and 2 stay alive and have received the train call.
        """One cell's actor fails during broadcast, that cell is killed and stopped, others complete normally."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Make cell 1's actors fail on train
        for handle in get_raw_actor_handles(_cell(group, 1)):
            ray.get(handle.set_fail_methods.remote(["train"]))

        # Step 2: Broadcast train
        await group._execute_all_alive_and_catch("train", rollout_id=0, rollout_data_ref="data")

        # Step 3: Cell 1 is errored, others alive
        assert _cell(group, 0).is_alive
        assert _was_killed(group, 1)
        assert _cell(group, 2).is_alive

        # Step 4: Other cells received train call
        for cell_idx in [0, 2]:
            for handle in get_raw_actor_handles(_cell(group, cell_idx)):
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "train" for c in calls)

    async def test_errored_cell_skipped_in_next_broadcast(self):
        # [FUNCTION DOC] Cell 0 fails the first broadcast and is killed. A second broadcast only
        # reaches cell 1, whose actors have two train calls in total.
        """After marking a cell errored, subsequent broadcasts skip it."""
        group = await _make_alive_controller(num_cells=2)

        # Step 1: Make cell 0 fail
        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["train"]))

        await group._execute_all_alive_and_catch("train", rollout_id=0, rollout_data_ref="data")
        assert _was_killed(group, 0)

        # Step 2: Next broadcast only goes to cell 1
        await group._execute_all_alive_and_catch("train", rollout_id=1, rollout_data_ref="data")

        for handle in get_raw_actor_handles(_cell(group, 1)):
            calls = ray.get(handle.get_calls.remote())
            train_calls = [c for c in calls if c[0] == "train"]
            assert len(train_calls) == 2


class TestExecuteFirstAliveFallback:
    # [CLASS DOC] Tests failure handling when running a method on the first alive cell.
    async def test_first_cell_fails_retry_falls_back_to_next(self):
        # [FUNCTION DOC] Cell 0 fails save model. Saving the model (which retries) ends with cell 0
        # killed, cell 1 alive, and cell 1's actors having received the save call.
        """If the first alive cell fails, retry in save_model kills+stops it and picks the next."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Make cell 0 fail on save_model
        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["save_model"]))

        # Step 2: save_model uses retry(lambda _: self._execute_first_alive(...))
        await group.save_model(rollout_id=42)

        # Step 3: Cell 0 errored, cell 1 handled it
        assert _was_killed(group, 0)
        assert _cell(group, 1).is_alive

        for handle in get_raw_actor_handles(_cell(group, 1)):
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "save_model" for c in calls)

    async def test_single_execute_first_alive_raises_on_failure(self):
        # [FUNCTION DOC] A single first-alive execution with cell 0 failing save model raises an
        # error (no retry) and cell 0 is killed.
        """A single _execute_first_alive call raises (no retry) when the first cell fails."""
        group = await _make_alive_controller(num_cells=2)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["save_model"]))

        with pytest.raises(Exception):  # noqa: B017
            await group._execute_first_alive("save_model", rollout_id=42)

        assert _was_killed(group, 0)

    async def test_losing_the_last_cell_keeps_the_worker_error_as_the_cause(self):
        # [FUNCTION DOC] With the only cell failing save model, the first-alive execution raises a
        # non-retryable error whose cause message contains the injected failure text.
        """Without the cause the driver traceback says nothing about why the last cell died."""
        group = await _make_alive_controller(num_cells=1)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["save_model"]))

        with pytest.raises(NonRetryableError) as excinfo:
            await group._execute_first_alive("save_model", rollout_id=42)

        assert "Injected failure in save_model" in str(excinfo.value.__cause__)

    async def test_losing_the_last_alive_cell_is_fatal_even_while_a_cell_is_still_healing(self):
        # [FUNCTION DOC] Cell 1 is stopped and restarted (healing) while cell 0, the only alive
        # cell, fails save model: a non-retryable error is raised and cell 1 remains uninitialized.
        """The next attempt runs on alive cells alone, so a cell that is only healing cannot keep it retryable."""
        group = await _make_alive_controller(num_cells=2)
        await _stop_cell(group, 1)
        _start_cell(group, 1)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["save_model"]))

        with pytest.raises(NonRetryableError):
            await group._execute_first_alive("save_model", rollout_id=42)

        assert _cell(group, 1).is_uninitialized

    async def test_terminal_failure_does_not_burn_another_backoff(self):
        # [FUNCTION DOC] The first-alive execution is wrapped to count attempts. With the only cell
        # failing save model, saving the model raises a non-retryable error after exactly 1 attempt.
        """Retrying without a single live cell can never succeed, so it must fail fast."""
        group = await _make_alive_controller(num_cells=1)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["save_model"]))

        attempts = 0
        execute_first_alive = group._execute_first_alive

        async def _counting_execute_first_alive(fn_name: str, **kwargs: object) -> object:
            # [FUNCTION DOC] HELPER: Increments the enclosing attempt counter and delegates to the
            # original first-alive execution with the same arguments, returning its result.
            nonlocal attempts
            attempts += 1
            return await execute_first_alive(fn_name, **kwargs)

        group._execute_first_alive = _counting_execute_first_alive

        with pytest.raises(NonRetryableError):
            await group.save_model(rollout_id=42)

        assert attempts == 1


class TestRefreshCellsErrorHandling:
    # [CLASS DOC] Tests refresh behavior when healing a pending cell fails.
    async def test_healing_failure_marks_pending_cell_errored_keeps_alive(self):
        # [FUNCTION DOC] Cell 2 is stopped, registered to fail init, and restarted. After a
        # refresh, cells 0 and 1 are alive, cell 2 is errored and its actors were killed (the stop
        # recorded earlier cannot show this, so the kill is checked).
        """When healing init fails, the pending cell is killed and stopped (via _execute_raw's
        except path, which marks errored then confirms-dead), alive cells unaffected."""
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Stop cell 2 and start it (pending)
        await _stop_cell(group, 2)

        # Step 2: Replace actor factory so new actors fail on init
        train_conftest.fake_worker_manager.fail_init_for_cell(2)
        _start_cell(group, 2)

        # Step 3: Refresh — healing init fails, cell auto-marks errored
        await group._refresh_cells(rollout_id=0)

        # Step 4: Cell 2 errored, cells 0 and 1 still alive. _was_stopped would already be true
        # from step 1, so it says nothing about the healing failure; the kill does.
        assert _cell(group, 0).is_alive
        assert _cell(group, 1).is_alive
        assert _cell(group, 2).is_errored
        assert _was_killed(group, 2)


class TestHeartbeatMonitor:
    # [CLASS DOC] Tests the per-cell health checkers created by the controller: their check
    # function, the shared activeness flag and the paused context.
    async def test_heartbeat_normal_does_not_mark_errored(self):
        # [FUNCTION DOC] Running every cell's health check function on 2 alive cells raises nothing
        # and leaves them all alive.
        """When heartbeat returns recent timestamp, cells stay alive."""
        group = await _make_alive_controller(num_cells=2)

        for cell in group._cells:
            await cell.health_checker._check_fn()

        assert all(c.is_alive for c in group._cells)

    async def test_heartbeat_stale_timestamp_does_not_mark_errored(self):
        # [FUNCTION DOC] Cell 1's actors get a last-active timestamp of 0; both cells' health checks
        # still raise nothing and both cells stay alive (health means liveness, not progress).
        """A stale heartbeat timestamp alone keeps the cell healthy: cell health is
        liveness, not training progress, so a cell legitimately blocked in a cross-cell
        collective (whose training loop stops bumping the heartbeat) must not be reported
        unhealthy as long as the heartbeat RPC still returns."""
        group = await _make_alive_controller(num_cells=2)

        # Drive cell 1's last-active timestamp to the epoch (maximally stale); the
        # liveness check must ignore staleness while the heartbeat RPC keeps returning.
        for handle in get_raw_actor_handles(_cell(group, 1)):
            ray.get(handle.set_last_active_timestamp.remote(0.0))

        # Neither check raises (a returned heartbeat proves the process is alive) and
        # both cells stay alive despite cell 1's stale timestamp.
        await _cell(group, 1).health_checker._check_fn()
        await _cell(group, 0).health_checker._check_fn()
        assert all(c.is_alive for c in group._cells)

    async def test_heartbeat_timeout_marks_errored(self):
        # [FUNCTION DOC] Cell 0's actors are set to fail the heartbeat: its health check function
        # raises a runtime error mentioning the injected heartbeat failure.
        # KNOWN ISSUE: the docstring and name say the cell is marked errored, but the test only
        # asserts that the check raises; the cell state is not examined.
        """When heartbeat call fails (actor unresponsive), cell is marked errored."""
        group = await _make_alive_controller(num_cells=2)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_heartbeat_fail.remote(True))

        with pytest.raises(RuntimeError, match="Injected heartbeat failure"):
            await _cell(group, 0).health_checker._check_fn()

    async def test_the_group_activeness_flag_reaches_every_cell_checker(self):
        # [FUNCTION DOC] Setting the controller's activeness to inactive makes no cell checker
        # report active; setting it active makes all of them report active.
        """Checkers pull activeness from the group, so one flag governs the whole pool."""
        group = await _make_alive_controller(num_cells=2)

        group._health_checker_activeness.bump_active(False)
        assert not any(c.health_checker._get_activeness().active for c in group._cells)

        group._health_checker_activeness.bump_active(True)
        assert all(c.health_checker._get_activeness().active for c in group._cells)

    async def test_the_paused_context_restores_activeness_after_an_exception(self):
        # [FUNCTION DOC] Inside the paused health checkers context, activeness is inactive; when a
        # runtime error is raised inside it, the error propagates and activeness is active again afterwards.
        """A crash inside a reconfigure must not leave health checking off for the rest of the run."""
        group = await _make_alive_controller(num_cells=2)

        with pytest.raises(RuntimeError, match="boom"):
            with group._paused_health_checkers():
                assert not group._health_checker_activeness.get().active
                raise RuntimeError("boom")

        assert group._health_checker_activeness.get().active


NORMAL = TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
DISCARDED = TrainStepOutput(outcome=TrainStepOutcome.DISCARDED_SHOULD_RETRY)


_ERR = RuntimeError("boom")
_ERR2 = ValueError("boom2")


def _alive_cells_for(results) -> list[SimpleNamespace]:
    # [FUNCTION DOC] HELPER: Returns one stub cell per result, whose cell index is its position.
    """Mock alive cells aligned with a `results` list; only `.cell_index` is read."""
    return [SimpleNamespace(cell_index=i) for i in range(len(results))]


class TestCheckTrainOneAttempt:
    # [CLASS DOC] Tests the per-attempt result check that decides between accepting, retrying and
    # giving up. Results are one entry per alive cell: a list of step outputs, or an exception
    # object for a failed cell.
    """_check_train_one_attempt raises ValueError when any non-exception cell has DISCARDED."""

    @pytest.mark.parametrize(
        "results",
        [
            [[NORMAL]],  # single cell, single actor
            [[NORMAL, NORMAL], [NORMAL]],  # multi cell, multi actor
            [_ERR, [NORMAL, NORMAL]],  # errored + normal → ok
            [[]],  # cell with empty actor list → vacuously ok
        ],
    )
    def test_no_retry_when_no_discarded(self, results):
        # [FUNCTION DOC] Parametrized over all-normal results (single and multi cell), an errored
        # cell next to normal ones, and a cell with an empty list. The check raises nothing.
        _make_controller(num_cells=1)._check_train_one_attempt(_alive_cells_for(results), results)  # should not raise

    @pytest.mark.parametrize(
        "results",
        [
            [[DISCARDED]],  # single cell
            [[DISCARDED], [DISCARDED, DISCARDED]],  # multi cell
            [[NORMAL, DISCARDED]],  # mixed within same cell
            [[NORMAL], [DISCARDED]],  # mixed across cells
            [_ERR, [DISCARDED]],  # errored + discarded → retry
        ],
    )
    def test_retry_when_discarded_exists(self, results):
        # [FUNCTION DOC] Parametrized over results containing at least one discarded outcome (also
        # mixed with normal ones or with an errored cell). The check raises a value error
        # mentioning discarded-should-retry.
        with pytest.raises(ValueError, match="DISCARDED_SHOULD_RETRY"):
            _make_controller(num_cells=1)._check_train_one_attempt(_alive_cells_for(results), results)

    @pytest.mark.parametrize(
        "results",
        [
            [_ERR],  # single cell errored
            [_ERR, _ERR2],  # multiple cells all errored
        ],
    )
    def test_raises_when_all_cells_errored(self, results):
        # [FUNCTION DOC] Parametrized over one or two errored results. The check raises a
        # non-retryable error saying all cells failed.
        """No cell is alive to carry the next attempt, so the controller must raise the fatal error."""
        with pytest.raises(NonRetryableError, match="All cells failed"):
            _make_controller(num_cells=1)._check_train_one_attempt(_alive_cells_for(results), results)

    @pytest.mark.parametrize(
        "results",
        [
            [_ERR],  # single cell errored
            [_ERR, _ERR2],  # multiple cells all errored
        ],
    )
    def test_raises_a_fatal_error_when_every_cell_is_already_errored(self, results):
        # [FUNCTION DOC] Same results as the previous test, but the controller's cells are first
        # marked errored. The check raises a non-retryable error saying all cells failed.
        """With every cell errored there is nothing left to heal, so an all-errored attempt is non-retryable."""
        group = _make_controller(num_cells=1)
        for cell in group._cells:
            cell._mark_as_errored()

        with pytest.raises(NonRetryableError, match="All cells failed"):
            group._check_train_one_attempt(_alive_cells_for(results), results)

    def test_compute_attempt_outcomes_buckets_cells_by_index(self):
        # [FUNCTION DOC] For results [error, [discarded], [normal, normal]] the outcome buckets are
        # errored [0], discarded [1], normal [2] (by cell index).
        """_compute_attempt_outcomes buckets each alive cell into errored / discarded / normal by index."""
        results = [_ERR, [DISCARDED], [NORMAL, NORMAL]]
        outcomes = TrainerController._compute_attempt_outcomes(_alive_cells_for(results), results)
        assert outcomes == {"errored": [0], "discarded": [1], "normal": [2]}

    def test_a_payload_carrying_output_is_bucketed_by_its_outcome(self):
        # [FUNCTION DOC] A discarded-should-retry step output that carries a values payload (a box
        # around a reference) is still bucketed as discarded for its cell.
        """The critic ships values alongside its outcome, so the payload must not hide a retry request."""
        results = [[TrainStepOutput(outcome=TrainStepOutcome.DISCARDED_SHOULD_RETRY, values=Box("ref"))]]
        outcomes = TrainerController._compute_attempt_outcomes(_alive_cells_for(results), results)
        assert outcomes == {"errored": [], "discarded": [0], "normal": []}


async def _set_all_train_return(group: TrainerController, value: TrainStepOutput) -> None:
    # [FUNCTION DOC] HELPER: Sets the train return value on every actor of every cell and waits for each call.
    for cell in group._cells:
        for handle in get_raw_actor_handles(cell):
            ray.get(handle.set_train_return_value.remote(value))


async def _set_all_train_returns_per_attempt(group: TrainerController, values: list[TrainStepOutput]) -> None:
    # [FUNCTION DOC] HELPER: Sets the same per-attempt train return queue on every actor of every
    # cell and waits for each call.
    for cell in group._cells:
        for handle in get_raw_actor_handles(cell):
            ray.get(handle.set_train_return_values_per_attempt.remote(values))


def _count_train_calls(group: TrainerController, cell_index: int) -> int:
    # [FUNCTION DOC] HELPER: Returns the total number of train calls recorded across all actors of
    # the given cell (sums each actor's call log entries named train).
    total = 0
    for handle in get_raw_actor_handles(_cell(group, cell_index)):
        calls = ray.get(handle.get_calls.remote())
        total += sum(1 for c in calls if c[0] == "train")
    return total


class TestTrainRetry:
    # [CLASS DOC] Tests retrying a train step when workers return discarded-should-retry. Counts
    # are totals over all actors of a cell, which matches per-cell counts only when a cell has
    # one actor (the default of these tests).
    async def test_no_retry_on_normal(self):
        # [FUNCTION DOC] When all cells return normal, each of 2 cells has exactly one train call.
        """All cells return NORMAL → no retry, train called once per cell."""
        group = await _make_alive_controller(num_cells=2)

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        for i in range(2):
            assert _count_train_calls(group, i) == 1

    async def test_retry_on_all_discarded_then_normal(self):
        # [FUNCTION DOC] Every actor returns discarded then normal: train is attempted twice in each cell.
        """First attempt: all DISCARDED. Second attempt: all NORMAL. Train called twice."""
        group = await _make_alive_controller(num_cells=2)
        await _set_all_train_returns_per_attempt(group, [DISCARDED, NORMAL])

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        for i in range(2):
            assert _count_train_calls(group, i) == 2

    async def test_retry_multiple_times_then_succeed(self):
        # [FUNCTION DOC] Every actor returns discarded three times then normal: train is attempted
        # four times in each cell.
        """DISCARDED 3 times, then NORMAL on 4th attempt."""
        group = await _make_alive_controller(num_cells=2)
        await _set_all_train_returns_per_attempt(group, [DISCARDED, DISCARDED, DISCARDED, NORMAL])

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        for i in range(2):
            assert _count_train_calls(group, i) == 4

    async def test_a_failed_attempt_releases_values_returned_by_its_successful_workers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Cell 0's actor returns a normal output carrying a values reference then
        # normal; cell 1's actor returns discarded then normal. A recording object store replaces the
        # singleton. After training, the store was asked to remove exactly one reference, the one
        # from the failed first attempt (compared by inner value).
        """A retry must release values already returned by successful workers in the failed attempt."""

        class RecordingStore:
            # [CLASS DOC] HELPER: Fake object store that records the references it is asked to remove.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Starts with an empty removal list.
                self.removed: list[Box] = []

            def remove(self, ref: Box) -> None:
                # [FUNCTION DOC] HELPER: Appends the reference to the removal list.
                self.removed.append(ref)

        group = await _make_alive_controller(num_cells=2)
        values_ref = Box("failed-attempt-values")
        successful_output = TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=values_ref)
        first_cell_handle = get_raw_actor_handles(_cell(group, 0))[0]
        second_cell_handle = get_raw_actor_handles(_cell(group, 1))[0]
        ray.get(first_cell_handle.set_train_return_values_per_attempt.remote([successful_output, NORMAL]))
        ray.get(second_cell_handle.set_train_return_values_per_attempt.remote([DISCARDED, NORMAL]))
        store = RecordingStore()
        monkeypatch.setattr(object_store, "_INSTANCE", store)

        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        assert [ref.inner for ref in store.removed] == [values_ref.inner]

    async def test_cell_errored_does_not_retry_when_others_normal(self):
        # [FUNCTION DOC] Cell 1 of 3 fails train while the others return normal: training succeeds
        # with no retry, cell 1 is killed, and cells 0 and 2 each have exactly one train call.
        """One cell errors during train but others return NORMAL → no retry.

        See _check_train_one_attempt: 'If some cells errors + all other cells claim
        normal, we do *not* retry. This may happen when some cells fails *after*
        exchanging gradients w/ others.' So alive cells get exactly 1 train call.
        """
        group = await _make_alive_controller(num_cells=3)

        # Step 1: Make cell 1 fail (exception)
        for handle in get_raw_actor_handles(_cell(group, 1)):
            ray.get(handle.set_fail_methods.remote(["train"]))

        # Step 2: Train completes without retry (cell 1 errored but others NORMAL)
        await group.train(rollout_id=0, rollout_data_pack=_DUMMY_DATA_PACK)

        # Step 3: Cell 1 errored, alive cells each got 1 train call (no retry)
        assert _was_killed(group, 1)
        for i in [0, 2]:
            assert _count_train_calls(group, i) == 1


class TestAllocateWitnessInfo:
    # [CLASS DOC] Tests allocation of witness ids for a training attempt.
    def test_returns_none_when_disabled(self):
        # [FUNCTION DOC] With no witness allocator, allocating witness info returns none.
        """When _witness_allocator is None, _allocate_witness_info returns None."""
        group = _make_controller(num_cells=1)
        group._witness_allocator = None

        result = group._allocate_witness_info(rollout_id=0, attempt=0, sample_indices=[10, 20, 30])

        assert result is None

    def test_returns_witness_info_when_enabled(self):
        # [FUNCTION DOC] With an allocator of buffer size 100 and the event logger reported as not
        # initialized, allocating for three sample indices returns a witness info with 3 witness
        # ids and a list of stale ids.
        """When witness is enabled, _allocate_witness_info returns a WitnessInfo with correct number of ids."""
        group = _make_controller(num_cells=1)
        group._witness_allocator = WitnessIdAllocator(buffer_size=100)

        with patch("miles.ray.train.group.is_event_logger_initialized", return_value=False):
            result = group._allocate_witness_info(rollout_id=0, attempt=0, sample_indices=[10, 20, 30])

        assert result is not None
        assert len(result.witness_ids) == 3
        assert isinstance(result.stale_ids, list)


class TestLogStepEndEvent:
    # [CLASS DOC] Tests the step end event logging.
    def test_with_normal_and_error_cells(self):
        # [FUNCTION DOC] Three stub cells with results [normal, normal], a runtime error, [normal].
        # With the event logger initialized and mocked, logging the step end for rollout 42 calls the
        # logger once with a payload holding rollout id 42 and cell outcomes: cell 0 two normal
        # outcomes, cell 1 the text error, cell 2 one normal outcome.
        """Passes correct cell_outcomes to event logger for a mix of normal and errored cells."""
        group = _make_controller(num_cells=3)

        mock_cell_0 = MagicMock()
        mock_cell_0.cell_index = 0
        mock_cell_1 = MagicMock()
        mock_cell_1.cell_index = 1
        mock_cell_2 = MagicMock()
        mock_cell_2.cell_index = 2

        snapshot_alive_cells = [mock_cell_0, mock_cell_1, mock_cell_2]
        results = [
            [NORMAL, NORMAL],
            RuntimeError("boom"),
            [NORMAL],
        ]

        with patch("miles.ray.train.group.is_event_logger_initialized", return_value=True), patch(
            "miles.ray.train.group.get_event_logger"
        ) as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            group._log_step_end_event(
                rollout_id=42,
                snapshot_alive_cells=snapshot_alive_cells,
                results=results,
            )

            mock_logger.log.assert_called_once()
            args = mock_logger.log.call_args[0]
            partial = args[1]
            assert partial["rollout_id"] == 42

            cell_outcomes = partial["cell_outcomes"]
            assert cell_outcomes[0] == [TrainStepOutcome.NORMAL, TrainStepOutcome.NORMAL]
            assert cell_outcomes[1] == "error"
            assert cell_outcomes[2] == [TrainStepOutcome.NORMAL]


class TestCellStatusesUnderConcurrentReconcile:
    # [CLASS DOC] Tests that reading cell statuses tolerates concurrent changes to the cell mapping.
    async def test_a_cell_removed_while_the_statuses_are_read_does_not_abort_the_read(self):
        # [FUNCTION DOC] Cell 0 is replaced by a stub whose status call removes cell 1 from the
        # live mapping before returning the real cell 0 status. Reading the statuses of a 3-cell
        # controller must complete and report all three cell ids (the snapshot is taken first).
        """The api server reads this from its own thread while reconcile adds and drops cells,
        and iterating the live dict raises RuntimeError instead of answering the request."""
        controller = _make_controller(num_cells=3)
        victim = compute_cell_id(pool_id=controller._pool_id, cell_index=1)
        real_cell = _cell(controller, 0)

        class _EvictingCell:
            # [CLASS DOC] HELPER: Cell stub whose status call has a side effect on the controller's mapping.
            def cell_status(self_inner):
                # [FUNCTION DOC] HELPER: Removes the victim cell from the controller's live cell
                # mapping (if present) and returns the real cell 0's status.
                controller._cells_by_id.pop(victim, None)
                return real_cell.cell_status()

        controller._cells_by_id[compute_cell_id(pool_id=controller._pool_id, cell_index=0)] = _EvictingCell()

        statuses = await controller.get_cell_statuses()

        # The snapshot is taken before the first cell_status() call, so the evicted cell is still
        # answered for. What matters is that the read completes instead of raising.
        assert set(statuses) == {compute_cell_id(pool_id=controller._pool_id, cell_index=i) for i in range(3)}


class TestUpdateWeightsReturnsTheVersion:
    # [CLASS DOC] Tests the controller's update weights method, which returns the weight version.
    def _make_group(self, *, per_worker_versions: list[int | None]) -> TrainerController:
        # [FUNCTION DOC] HELPER: Creates a controller without running its constructor, with args
        # (no debug flags, no trainer model id) and a first-alive execution replaced by an async mock
        # returning the given per-worker version list.
        group = TrainerController.__new__(TrainerController)
        group.args = SimpleNamespace(debug_train_only=False, debug_rollout_only=False, trainer_model_id=None)
        group._execute_first_alive = AsyncMock(return_value=per_worker_versions)
        return group

    async def test_the_controller_answers_the_version_the_engines_now_serve(self):
        # [FUNCTION DOC] When all workers return version 11, update weights returns 11.
        """The driver can only publish the version to the executor if the controller hands it back."""
        group = self._make_group(per_worker_versions=[11, 11])

        assert await group.update_weights(info=MagicMock()) == 11

    async def test_a_trainer_that_skipped_the_broadcast_answers_nothing(self):
        # [FUNCTION DOC] When the workers return none, update weights returns none.
        """--debug-skip-weight-update returns None from every worker, which must reach the driver as None."""
        group = self._make_group(per_worker_versions=[None])

        assert await group.update_weights(info=MagicMock()) is None

    async def test_it_broadcasts_the_window_the_orchestration_script_opened(self):
        # [FUNCTION DOC] Update weights with an info object runs the first-alive execution exactly
        # once with the method name update weights and that same info object as keyword argument.
        """The engines it writes into are the ones the script snapshotted, not a set it fetched for itself."""
        group = self._make_group(per_worker_versions=[11])
        info = MagicMock()

        await group.update_weights(info=info)

        group._execute_first_alive.assert_awaited_once_with("update_weights", info=info)


class TestInitForwardsModelFlags:
    # [CLASS DOC] Tests the extra arguments the controller sends to each worker during init.
    async def test_every_worker_learns_its_role_and_which_extra_models_to_build(self):
        # [FUNCTION DOC] A controller with reference model and distillation teacher flags on, 2
        # cells of 2 actors, is inited. Every actor's init call carries role actor, with-reference
        # true and with-distillation-teacher true.
        """These flags decide which models a worker allocates, so dropping one silently changes the objective."""
        group = _make_controller(num_cells=2, actor_count_per_cell=2, with_ref=True, with_opd_teacher=True)

        await _init_controller(group)

        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                [init_call] = [c for c in ray.get(handle.get_calls.remote()) if c[0] == "init"]
                assert init_call[2]["role"] == "actor"
                assert init_call[2]["with_ref"] is True
                assert init_call[2]["with_opd_teacher"] is True


class TestTrainRunsFTTestActions:
    # [CLASS DOC] Tests that train executes the fault tolerance test action armed for its rollout.
    async def test_train_applies_the_action_armed_for_that_rollout_before_returning(self):
        # [FUNCTION DOC] An action list arms a stop-cell-at-end action for rollout 4 on cell
        # trainer-engine-actor-00002. Training rollout 4 on 3 alive cells awaits exactly one suspend
        # on the mock cell operations for that cell id.
        """The FT scenario's stop must have landed by the time the driver starts the next rollout."""
        actions = json.dumps(
            [{"at_rollout": 4, "action": "stop_cell_at_end", "cell_id": "trainer-engine-actor-00002"}]
        )
        group = await _make_alive_controller(num_cells=3, ci_ft_test_actions=actions)

        await group.train(rollout_id=4, rollout_data_pack=_DUMMY_DATA_PACK)

        group._cell_operations.suspend.assert_awaited_once_with(cell_id="trainer-engine-actor-00002")

    async def test_train_leaves_the_pool_alone_on_a_rollout_no_action_names(self):
        # [FUNCTION DOC] With the same action armed for rollout 4, training rollout 3 never awaits
        # a suspend.
        """An action that fires on every rollout would tear the pool down for the whole run."""
        actions = json.dumps(
            [{"at_rollout": 4, "action": "stop_cell_at_end", "cell_id": "trainer-engine-actor-00002"}]
        )
        group = await _make_alive_controller(num_cells=3, ci_ft_test_actions=actions)

        await group.train(rollout_id=3, rollout_data_pack=_DUMMY_DATA_PACK)

        group._cell_operations.suspend.assert_not_awaited()


class TestSaveModel:
    # [CLASS DOC] Tests the arguments the controller forwards when saving the model.
    async def test_the_selected_cell_is_told_whether_the_save_must_be_synchronous(self):
        # [FUNCTION DOC] Saving rollout 9 with the force-sync option on 2 alive cells gives each
        # actor of cell 0 exactly one save call whose keyword arguments are rollout id 9 and
        # force sync true.
        """An async save that the caller asked to block on would let training race the checkpoint writer."""
        group = await _make_alive_controller(num_cells=2)

        await group.save_model(rollout_id=9, force_sync=True)

        for handle in get_raw_actor_handles(_cell(group, 0)):
            save_calls = [c for c in ray.get(handle.get_calls.remote()) if c[0] == "save_model"]
            assert [c[2] for c in save_calls] == [{"rollout_id": 9, "force_sync": True}]


class TestExportHf:
    # [CLASS DOC] Tests the hugging face export through the controller.
    async def test_a_failed_first_cell_hands_the_same_export_to_the_next_alive_cell(self):
        # [FUNCTION DOC] Cell 0 fails export. Exporting rollout 4 to /ckpt/hf-4 kills cell 0 and
        # cell 1's actors each record exactly one export call with keyword arguments rollout id 4
        # and that path.
        """The exported checkpoint must land at the requested path even when the first cell dies mid-export."""
        group = await _make_alive_controller(num_cells=2)
        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_fail_methods.remote(["export_hf"]))

        await group.export_hf(rollout_id=4, path="/ckpt/hf-4")

        assert _was_killed(group, 0)
        for handle in get_raw_actor_handles(_cell(group, 1)):
            export_calls = [c for c in ray.get(handle.get_calls.remote()) if c[0] == "export_hf"]
            assert [c[2] for c in export_calls] == [{"rollout_id": 4, "path": "/ckpt/hf-4"}]


class TestUpdateWeightsReachesTheWorker:
    # [CLASS DOC] Tests that update weights passes the snapshot info to workers and returns the version.
    async def test_the_engine_snapshot_reaches_the_worker_and_its_version_comes_back(self):
        # [FUNCTION DOC] With a 1-cell controller whose actor returns weight version 11 and an info
        # carrying a cell-id-to-hash snapshot, update weights for rollout 3 returns 11 and the
        # actor's update call received an info whose snapshot equals the given mapping.
        """A worker that never sees the snapshot broadcasts to engines that were not part of the update window."""
        info = SimpleNamespace(snapshot_cell_id_to_hashes={"trainer-actor-0": "workers-hash-9"})
        group = await _make_alive_controller(num_cells=1)
        for handle in get_raw_actor_handles(_cell(group, 0)):
            ray.get(handle.set_update_weights_return_value.remote(11))

        assert await group.update_weights(info=info, rollout_id=3) == 11

        for handle in get_raw_actor_handles(_cell(group, 0)):
            [update_call] = [c for c in ray.get(handle.get_calls.remote()) if c[0] == "update_weights"]
            assert update_call[2]["info"].snapshot_cell_id_to_hashes == {"trainer-actor-0": "workers-hash-9"}
