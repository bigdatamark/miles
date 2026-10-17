# [MODULE DOC] Health checking and status reporting for trainer cells: a factory for a periodic
# liveness checker bound to a cell, and a pure function mapping a cell state plus the health checker's
# tri-state status to the externally visible cell status.
from collections.abc import Callable
from typing import TYPE_CHECKING

from miles.ray.train.cell_state import (
    CellState,
    StateAllocatedAlive,
    StateAllocatedErrored,
    StateAllocatedUninitialized,
)
from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus, TriState
from miles.utils.ft_utils.health_checker import ActiveAndEpoch, SimpleHealthChecker, SimpleHealthCheckerConfig

if TYPE_CHECKING:
    from miles.ray.train.cell import TrainerCell


def create_trainer_cell_health_checker(
    *,
    cell: "TrainerCell",
    config: SimpleHealthCheckerConfig,
    get_activeness: Callable[[], ActiveAndEpoch],
) -> SimpleHealthChecker:
    # [FUNCTION DOC] Build a simple periodic health checker named after the cell id, with the given
    # configuration and activeness getter. The probe: skips when the cell is not alive; otherwise
    # issues the heartbeat-status RPC to all workers WITHOUT killing workers on failure (the checker
    # decides what failure means). Returns the checker; it is not started here.
    async def _check() -> None:
        # [FUNCTION DOC] HELPER: One probe. Liveness only, not training progress: the heartbeat runs
        # on a dedicated concurrency group so it answers while training is blocked in a collective;
        # a returned result proves liveness, while an unreachable-worker error or RPC timeout
        # (propagated to the checker) proves death.
        # Cell health is liveness, not training progress: the heartbeat RPC runs on
        # a dedicated concurrency group and returns even while the training thread is
        # blocked in a (legitimately waiting) cross-cell collective. A returned result
        # proves the process is alive; a WorkerUnreachableError or RPC timeout proves it is not.
        if not cell.is_alive:
            return

        await cell.execute("get_heartbeat_status", kill_on_failure=False)

    return SimpleHealthChecker(
        name=f"trainer-cell-{cell.cell_id}",
        check_fn=_check,
        get_activeness=get_activeness,
        config=config,
    )


def compute_cell_status(state: CellState, health_checker_status: TriState, *, workers_hash: str) -> CellStatus:
    # [FUNCTION DOC] Map a cell state to a status; every status has phase Running, the given workers
    # hash, and allocated true. Alive: adds the condition derived from the health checker's status.
    # Uninitialized: healthy true (nothing to be unhealthy yet; the health checker is ignored).
    # Errored: healthy false with reason ExecutionErrored. Any other state raises not-implemented
    # error.
    match state:
        case StateAllocatedAlive():
            return CellStatus(
                phase="Running",
                conditions=[
                    CellCondition.allocated(TriState.TRUE),
                    CellCondition.from_health_checker_status(health_checker_status),
                ],
                workers_hash=workers_hash,
            )

        case StateAllocatedUninitialized():
            return CellStatus(
                phase="Running",
                conditions=[
                    CellCondition.allocated(TriState.TRUE),
                    CellCondition.healthy(TriState.TRUE),
                ],
                workers_hash=workers_hash,
            )

        case StateAllocatedErrored():
            return CellStatus(
                phase="Running",
                conditions=[
                    CellCondition.allocated(TriState.TRUE),
                    CellCondition.healthy(TriState.FALSE, reason="ExecutionErrored"),
                ],
                workers_hash=workers_hash,
            )

        case _:
            raise NotImplementedError(f"Unknown state: {state}")
