# [MODULE DOC]
# Unit tests for the trainer cell monitor: the computation of a cell status (phase, allocation and
# health conditions, workers hash stamp) from a cell state and a health tri-state, and the cell
# health checker whose liveness probe is the heartbeat rpc (healthy whenever the rpc returns,
# unhealthy when it raises, skipped when the cell is not alive), including its checker name.
# Cells and workers are mocks; no ray is used.
from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.ray.train.cell_monitor import compute_cell_status, create_trainer_cell_health_checker
from miles.ray.train.cell_state import StateAllocatedAlive, StateAllocatedErrored, StateAllocatedUninitialized
from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.ft_utils.health_checker import ActiveAndEpoch, SimpleHealthCheckerConfig
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError


def _make_worker_handle_mock() -> MagicMock:
    # [FUNCTION DOC] HELPER: Returns a mock restricted to the attributes of the worker handle base class.
    return MagicMock(spec=BaseWorkerHandle)


def _make_indep_dp_info() -> IndepDPInfo:
    # [FUNCTION DOC] HELPER: Builds a one-cell independent dp info (cell 0 of 1, alive rank 0 of
    # size 1, quorum 1, alive cells [0]).
    return IndepDPInfo(
        cell_index=0,
        num_cells=1,
        alive_rank=0,
        alive_size=1,
        quorum_id=1,
        alive_cell_indices=[0],
    )


def _make_alive_state() -> StateAllocatedAlive:
    # [FUNCTION DOC] HELPER: Builds an alive cell state with one mock worker handle and the one-cell info.
    return StateAllocatedAlive(worker_handles=[_make_worker_handle_mock()], indep_dp_info=_make_indep_dp_info())


def _find_condition(status, type_: str):
    # [FUNCTION DOC] HELPER: Returns the single condition of the given type from a status;
    # asserts (with a message giving the count) that exactly one such condition exists.
    matches = [c for c in status.conditions if c.type == type_]
    assert len(matches) == 1, f"expected exactly one {type_!r} condition, got {len(matches)}"
    return matches[0]


class TestComputeCellStatusAlive:
    # [CLASS DOC] Tests the status of an alive cell for each health tri-state value.
    def test_health_true_reports_healthy_true(self):
        # [FUNCTION DOC] Health true gives phase Running and a Healthy condition that is true with no reason.
        result = compute_cell_status(_make_alive_state(), TriState.TRUE, workers_hash="pseudo-hash-0")

        assert result.phase == "Running"
        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.TRUE
        assert healthy.reason is None

    def test_health_false_reports_healthy_false_with_failed_reason(self):
        # [FUNCTION DOC] Health false gives a Healthy condition that is false with reason HealthCheckFailed.
        result = compute_cell_status(_make_alive_state(), TriState.FALSE, workers_hash="pseudo-hash-0")

        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.FALSE
        assert healthy.reason == "HealthCheckFailed"

    def test_health_unknown_reports_healthy_unknown_not_translated_to_true(self):
        # [FUNCTION DOC] Health unknown gives a Healthy condition that is unknown with reason
        # HealthCheckUnknown (regression guard: unknown must not become true).
        """Regression: paused health checker reports UNKNOWN; previously this was
        silently translated to Healthy=TRUE, hiding the transient state from observers."""
        result = compute_cell_status(_make_alive_state(), TriState.UNKNOWN, workers_hash="pseudo-hash-0")

        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.UNKNOWN
        assert healthy.reason == "HealthCheckUnknown"


class TestComputeCellStatusOtherStates:
    # [CLASS DOC] Tests the status of uninitialized and errored cells, across all health values.
    @pytest.mark.parametrize("health_status", [TriState.TRUE, TriState.FALSE, TriState.UNKNOWN])
    def test_uninitialized_ignores_health_checker(self, health_status: TriState):
        # [FUNCTION DOC] For an uninitialized cell and any health value, the phase is Running and
        # the Healthy condition is true.
        state = StateAllocatedUninitialized(worker_handles=[_make_worker_handle_mock()])

        result = compute_cell_status(state, health_status, workers_hash="pseudo-hash-0")

        assert result.phase == "Running"
        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.TRUE

    @pytest.mark.parametrize("health_status", [TriState.TRUE, TriState.FALSE, TriState.UNKNOWN])
    def test_errored_always_reports_unhealthy(self, health_status: TriState):
        # [FUNCTION DOC] For an errored cell and any health value, the Healthy condition is false
        # with reason ExecutionErrored.
        state = StateAllocatedErrored(worker_handles=[_make_worker_handle_mock()], indep_dp_info=_make_indep_dp_info())

        result = compute_cell_status(state, health_status, workers_hash="pseudo-hash-0")

        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.FALSE
        assert healthy.reason == "ExecutionErrored"


class TestComputeCellStatusForEveryTrainerState:
    # [CLASS DOC] Tests phase and allocation invariants of the status in every cell state.
    @pytest.mark.parametrize(
        "state",
        [
            StateAllocatedUninitialized(worker_handles=[]),
            StateAllocatedAlive(worker_handles=[], indep_dp_info=_make_indep_dp_info()),
            StateAllocatedErrored(worker_handles=[], indep_dp_info=_make_indep_dp_info()),
        ],
    )
    def test_a_trainer_cell_is_always_running_and_allocated(self, state):
        # [FUNCTION DOC] Parametrized over uninitialized, alive and errored states (no workers).
        # With health false, the phase is Running and the Allocated condition is true.
        """Trainer cells are never pending or stopped, so the FT controller must never see them lose their slots."""
        result = compute_cell_status(state, TriState.FALSE, workers_hash="pseudo-hash-0")

        assert result.phase == "Running"
        assert _find_condition(result, "Allocated").status == TriState.TRUE


def _make_cell_mock(*, is_alive: bool, execute: AsyncMock, cell_id: str = "trainer-actor-0") -> MagicMock:
    # [FUNCTION DOC] HELPER: Builds a mock cell with the given alive flag, cell index 0, the given
    # cell id (default trainer-actor-0) and the given async execute mock.
    cell = MagicMock()
    cell.is_alive = is_alive
    cell.cell_index = 0
    cell.cell_id = cell_id
    cell.execute = execute
    return cell


class TestTrainerCellHealthCheckerName:
    # [CLASS DOC] Tests the naming of cell health checkers.
    def test_the_checker_is_named_after_the_cell_id_not_its_index(self):
        # [FUNCTION DOC] For a mock cell with id trainer-critic-2, the created checker's name is
        # trainer-cell- followed by that cell id.
        """Two pools both have a cell 0, so an index-based name makes their health logs indistinguishable."""
        cell = _make_cell_mock(is_alive=True, execute=AsyncMock(), cell_id="trainer-critic-2")

        checker = create_trainer_cell_health_checker(
            cell=cell,
            config=SimpleHealthCheckerConfig(interval=10.0, timeout=10.0, first_wait=0.0, failure_threshold=3),
            get_activeness=lambda: ActiveAndEpoch(active=True, epoch=0),
        )

        assert checker._name == "trainer-cell-trainer-critic-2"


class TestTrainerCellHealthCheckLiveness:
    # [CLASS DOC] Tests the check function of the cell health checker, which probes liveness by
    # executing the heartbeat status call on the cell with kill-on-failure off.
    """Cell health is defined as process liveness (RPC reachability), not training
    progress: the check must succeed whenever the heartbeat RPC returns, and fail only
    when the actor is dead/unresponsive."""

    @pytest.mark.asyncio
    async def test_rpc_returns_means_healthy_regardless_of_progress(self):
        # [FUNCTION DOC] With an alive cell whose heartbeat call returns a stale status (last
        # active timestamp 0), the check returns without error and the heartbeat status call was
        # executed once with kill-on-failure off.
        """A returned heartbeat (even a stale one) proves liveness, so _check passes."""
        execute = AsyncMock(return_value=[MagicMock(last_active_timestamp=0.0, bump_count=1)])
        cell = _make_cell_mock(is_alive=True, execute=execute)

        checker = create_trainer_cell_health_checker(
            cell=cell,
            config=SimpleHealthCheckerConfig(interval=10.0, timeout=10.0, first_wait=0.0, failure_threshold=3),
            get_activeness=lambda: ActiveAndEpoch(active=True, epoch=0),
        )

        await checker._check_fn()
        execute.assert_awaited_once_with("get_heartbeat_status", kill_on_failure=False)

    @pytest.mark.asyncio
    async def test_rpc_error_propagates_as_unhealthy(self):
        # [FUNCTION DOC] When the heartbeat call raises a worker-unreachable error, the check
        # propagates that error.
        """A dead actor makes the heartbeat RPC raise; _check propagates it so the
        checker reports unhealthy."""
        execute = AsyncMock(side_effect=WorkerUnreachableError("worker gone"))
        cell = _make_cell_mock(is_alive=True, execute=execute)

        checker = create_trainer_cell_health_checker(
            cell=cell,
            config=SimpleHealthCheckerConfig(interval=10.0, timeout=10.0, first_wait=0.0, failure_threshold=3),
            get_activeness=lambda: ActiveAndEpoch(active=True, epoch=0),
        )

        with pytest.raises(WorkerUnreachableError):
            await checker._check_fn()

    @pytest.mark.asyncio
    async def test_not_alive_cell_skips_rpc(self):
        # [FUNCTION DOC] For a cell that is not alive, the check returns without error and the
        # execute call is never awaited.
        """A not-yet-alive cell is not probed and is not reported unhealthy."""
        execute = AsyncMock()
        cell = _make_cell_mock(is_alive=False, execute=execute)

        checker = create_trainer_cell_health_checker(
            cell=cell,
            config=SimpleHealthCheckerConfig(interval=10.0, timeout=10.0, first_wait=0.0, failure_threshold=3),
            get_activeness=lambda: ActiveAndEpoch(active=True, epoch=0),
        )

        await checker._check_fn()
        execute.assert_not_awaited()


class TestComputeCellStatusGeneration:
    # [CLASS DOC] Tests that the status carries the workers hash identifying its generation.
    @pytest.mark.parametrize(
        "state_factory",
        [
            _make_alive_state,
            lambda: StateAllocatedUninitialized(worker_handles=[_make_worker_handle_mock()]),
            lambda: StateAllocatedErrored(
                worker_handles=[_make_worker_handle_mock()], indep_dp_info=_make_indep_dp_info()
            ),
        ],
    )
    def test_a_status_is_stamped_with_the_generation_it_describes(self, state_factory):
        # [FUNCTION DOC] Parametrized over alive, uninitialized and errored state factories. The
        # status computed with workers hash hash-7 carries that hash.
        """The api server joins this with a separately polled listing, so every branch must name its own process."""
        result = compute_cell_status(state_factory(), TriState.TRUE, workers_hash="hash-7")

        assert result.workers_hash == "hash-7"
