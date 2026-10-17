# [MODULE DOC] Trainer cell: one replica (a set of worker actors, one per training rank) of a
# trainer. The cell wrapper fans every call out to all of its workers concurrently, tracks the
# cell's lifecycle state (uninitialized -> alive; any -> errored), kills all workers on a failed
# call and waits until they are confirmed dead, and offers the cooperative preparation steps used
# by independent data parallel fault tolerance. Constants: kill RPC timeout 10 s, death confirmation
# timeout 120 s. Structured fault tolerance logs are written for state changes and executions.
import asyncio
import logging
import time

from miles.ray.train.cell_monitor import compute_cell_status
from miles.ray.train.cell_state import (
    CellState,
    StateAllocatedAlive,
    StateAllocatedBase,
    StateAllocatedErrored,
    StateAllocatedUninitialized,
)
from miles.utils.ft_utils.api_server.models import CellStatus
from miles.utils.ft_utils.health_checker import BaseHealthChecker
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.retry_utils import NonRetryableError
from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import MASTER_PORT_NAME, HostAndPort

logger = logging.getLogger(__name__)

KILL_RPC_TIMEOUT_S = 10.0
CONFIRM_DEAD_TIMEOUT_S = 120.0


class TrainerCell:
    # [CLASS DOC] Wrapper around the workers of one trainer cell. Public attributes: args, cell id,
    # cell index, workers hash, role, flags for reference model and OPD teacher, and the health
    # checker (started after a successful init, not here). The state is private and must only change
    # through the change-state helper, which asserts the allowed previous state classes. Workers are
    # addressed by their rank order in the cell. Not safe for concurrent calls that mutate state.

    def __init__(
        self,
        *,
        args,
        role: str,
        with_ref: bool,
        with_opd_teacher: bool = False,
        cell_id: str,
        cell_index: int,
        workers_hash: str,
        health_checker: BaseHealthChecker,
        provider: BaseWorkerProvider,
    ) -> None:
        # [FUNCTION DOC] Store the configuration, then ask the provider for the cell's single list of
        # worker infos (unpacking asserts exactly one cell is returned). The master address is the
        # master-port address of the FIRST worker. Gets callable handles for all workers (assertion
        # error if some worker cannot be called) and starts in the allocated-uninitialized state with
        # those handles in order. No network calls to workers.
        self.args = args
        self.cell_id = cell_id
        self.cell_index = cell_index
        self.workers_hash = workers_hash
        self.role = role
        self.with_ref = with_ref
        self.with_opd_teacher = with_opd_teacher
        self.health_checker = health_checker

        (worker_infos,) = provider.get_worker_infos(cell_ids=[cell_id])
        self._master_addr: HostAndPort = worker_infos[0].self_addrs[MASTER_PORT_NAME]
        worker_handles = provider.get_handles_of_worker_infos(worker_infos)
        assert len(worker_handles) == len(worker_infos), f"cell {cell_id} holds workers that cannot be called"

        # NOTE: do *NOT* directly modify `self._state`, but instead use `self._change_state`
        self._state: CellState = StateAllocatedUninitialized(worker_handles=list(worker_handles.values()))

    # ------------------------ API ------------------------

    async def init(
        self,
        *,
        indep_dp_info: IndepDPInfo,
        indep_dp_store_addr: str | None,
        recv_ckpt_src_rank: int | None = None,
    ):
        # [FUNCTION DOC] Initialize all workers of the cell. First tells every worker the master address
        # (host with any IPv6 brackets stripped) and port; then calls the workers' init with args, role,
        # reference-model and OPD teacher flags, the independent-DP info, the independent-DP store
        # address and, when healing, the rank to receive the checkpoint from. Any failure kills the
        # workers, marks the cell errored and re-raises. On success marks the cell alive (requires the
        # uninitialized state), starts the health checker, yields to the event loop once, and returns
        # the list of per-worker init results in rank order.
        await self.execute(
            "configure_master_addr_and_port",
            master_addr=self._master_addr.host.strip("[]"),
            master_port=self._master_addr.port,
        )
        results = await self.execute(
            "init",
            args=self.args,
            role=self.role,
            with_ref=self.with_ref,
            with_opd_teacher=self.with_opd_teacher,
            indep_dp_info=indep_dp_info,
            indep_dp_store_addr=indep_dp_store_addr,
            recv_ckpt_src_rank=recv_ckpt_src_rank,
        )
        self._mark_as_alive(indep_dp_info=indep_dp_info)
        self.health_checker.start()
        await asyncio.sleep(0)
        return results

    async def load_state(self) -> list:
        # [FUNCTION DOC] Ask every worker to load its saved state; returns the per-worker results.
        return await self.execute("load_state")

    async def train(
        self,
        *,
        rollout_id: int,
        rollout_data_ref,
        witness_info,
        attempt: int,
        external_data: list | None = None,
    ) -> list:
        # [FUNCTION DOC] Run one training step on all workers concurrently. Each worker gets the rollout
        # id, the rollout data reference, the witness info and the attempt number; if external data is
        # given, it must hold exactly one payload per worker (otherwise raises a non-retryable error
        # before any call) and worker i receives payload i. Failure handling follows the raw execute
        # helper (workers killed, cell errored). Returns per-worker results in rank order.
        if external_data is not None and len(external_data) != len(self._get_worker_handles()):
            raise NonRetryableError("external_data must contain one payload per train worker")

        return await self._execute_raw(
            "train",
            compute_kwargs=lambda i: dict(
                rollout_id=rollout_id,
                rollout_data_ref=rollout_data_ref,
                witness_info=witness_info,
                attempt=attempt,
                **({} if external_data is None else dict(external_data=external_data[i])),
            ),
        )

    # ------------------------ API :: cooperatively prepare ------------------------

    async def prepare_indep_dp_mode_alive(
        self,
        indep_dp_info: IndepDPInfo,
        indep_dp_store_addr: str | None,
        send_ckpt_dst_ranks: list[int],
    ):
        # [FUNCTION DOC] Prepare a healthy (alive) cell to cooperate with a healing peer: reconfigure
        # all workers with the new independent-DP info and store address, record the info in the cell
        # state (alive to alive), then for each destination rank in order ask all workers to send a
        # checkpoint to that rank (sequentially, one destination at a time).
        await self.execute(
            "reconfigure_indep_dp", indep_dp_info=indep_dp_info, indep_dp_store_addr=indep_dp_store_addr
        )
        self._update_indep_dp_info(indep_dp_info)

        for dst_rank in send_ckpt_dst_ranks:
            await self.execute("send_ckpt", dst_rank=dst_rank)

    async def prepare_indep_dp_mode_healing(
        self,
        indep_dp_info: IndepDPInfo,
        indep_dp_store_addr: str | None,
        recv_ckpt_src_rank: int | None,
    ):
        # [FUNCTION DOC] Prepare a recovering cell: simply run the full init, passing the source rank
        # from which workers receive the checkpoint. The cell must be in the uninitialized state.
        await self.init(
            indep_dp_info=indep_dp_info,
            indep_dp_store_addr=indep_dp_store_addr,
            recv_ckpt_src_rank=recv_ckpt_src_rank,
        )

    # ------------------------ state transition ------------------------

    async def _kill_workers_and_confirm_dead(self) -> None:
        # [FUNCTION DOC] HELPER: If allocated, ask every worker to kill itself (concurrently; see kill
        # helper for tolerated errors), then wait concurrently until each worker is confirmed dead,
        # up to 120 s each (the wait raises on timeout). Logs start and end with elapsed seconds.
        # Does not change the cell state.
        handles = self._get_worker_handles() if self.is_allocated else []

        log_structured(
            logger.info, tag="ft", op="confirm_dead", phase="start", cell=self.cell_id, n_actors=len(handles)
        )
        start = time.monotonic()
        await asyncio.gather(*[_kill_worker(handle) for handle in handles])
        await asyncio.gather(*[handle.wait_dead(timeout=CONFIRM_DEAD_TIMEOUT_S) for handle in handles])
        log_structured(
            logger.info,
            tag="ft",
            op="confirm_dead",
            phase="end",
            cell=self.cell_id,
            elapsed_s=round(time.monotonic() - start, 1),
        )

    def _mark_as_alive(self, indep_dp_info: IndepDPInfo) -> None:
        # [FUNCTION DOC] HELPER: Transition uninitialized -> alive, carrying over the worker handles and
        # recording the independent-DP info.
        self._change_state(
            "_mark_as_alive",
            StateAllocatedUninitialized,
            StateAllocatedAlive(worker_handles=self._state.worker_handles, indep_dp_info=indep_dp_info),
        )

    def _update_indep_dp_info(self, indep_dp_info: IndepDPInfo) -> None:
        # [FUNCTION DOC] HELPER: Replace the alive state by a new alive state with the new
        # independent-DP info (same handles). Requires the alive state.
        self._change_state(
            "_update_indep_dp_info",
            StateAllocatedAlive,
            StateAllocatedAlive(worker_handles=self._state.worker_handles, indep_dp_info=indep_dp_info),
        )

    def _mark_as_errored(self) -> None:
        # [FUNCTION DOC] HELPER: Transition from uninitialized, alive or errored (idempotent) to
        # errored, keeping the worker handles and the independent-DP info (none if the cell had never
        # been initialized). Asserts the previous state.
        assert isinstance(
            self._state, (StateAllocatedUninitialized, StateAllocatedAlive, StateAllocatedErrored)
        ), f"{self.cell_id=} {self._state=}"
        indep_dp_info = None if isinstance(self._state, StateAllocatedUninitialized) else self._state.indep_dp_info
        self._change_state(
            "_mark_as_errored",
            (StateAllocatedUninitialized, StateAllocatedAlive, StateAllocatedErrored),
            StateAllocatedErrored(worker_handles=self._state.worker_handles, indep_dp_info=indep_dp_info),
        )

    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[CellState] | tuple[type[CellState], ...],
        new_state: CellState,
    ) -> None:
        # [FUNCTION DOC] HELPER: The only place that writes the state. Logs a structured start record
        # (transition name, cell, source state class name), asserts the current state is an instance
        # of the allowed old state class(es) (assertion error with cell id and state), installs the new
        # state, and logs a structured end record with the new state class name.
        log_structured(
            logger.info,
            tag="ft",
            op="state",
            phase="start",
            name=debug_name,
            cell=self.cell_id,
            from_state=type(self._state).__name__,
        )
        assert isinstance(self._state, old_state_cls), f"{self.cell_id=} {self._state=}"
        self._state = new_state
        log_structured(
            logger.info,
            tag="ft",
            op="state",
            phase="end",
            name=debug_name,
            cell=self.cell_id,
            to_state=type(self._state).__name__,
        )

    # ------------------------ API :: directly forward calls to actors ------------------------

    async def execute(self, fn_name: str, *, kill_on_failure: bool = True, **kwargs) -> list:
        # [FUNCTION DOC] Call the named method with identical keyword arguments on every worker handle
        # concurrently. Returns results in rank order. On exception, behavior is that of the raw
        # execute helper: with kill-on-failure (default) the cell is marked errored and all workers
        # are killed before the error is re-raised.
        return await self._execute_raw(
            fn_name,
            compute_kwargs=lambda _: kwargs,
            kill_on_failure=kill_on_failure,
        )

    async def _execute_raw(
        self,
        fn_name: str,
        compute_kwargs,
        kill_on_failure: bool = True,
    ) -> list:
        # [FUNCTION DOC] HELPER: Core fan-out. Requires an allocated cell (assertion error otherwise).
        # Calls the named method on each handle, with keyword arguments computed per worker index by
        # the supplied function, gathers all calls concurrently, logs structured start/end (with
        # elapsed seconds) or failure records. On an ordinary exception (cancellation is not caught):
        # logs the failure, and if kill-on-failure then marks the cell errored and kills all workers
        # and waits for confirmed death; then re-raises the original error. Returns the result list.
        handles = self._get_worker_handles()
        log_structured(
            logger.info, tag="ft", op="execute", phase="start", cell=self.cell_id, fn=fn_name, n_actors=len(handles)
        )
        start = time.monotonic()
        try:
            result = await asyncio.gather(
                *[getattr(handle, fn_name)(**compute_kwargs(i)) for i, handle in enumerate(handles)]
            )
            log_structured(
                logger.info,
                tag="ft",
                op="execute",
                phase="end",
                cell=self.cell_id,
                fn=fn_name,
                ok=True,
                elapsed_s=round(time.monotonic() - start, 1),
            )
            return result
        except Exception:
            log_structured(
                logger.error,
                tag="ft",
                op="execute",
                phase="fail",
                cell=self.cell_id,
                fn=fn_name,
                elapsed_s=round(time.monotonic() - start, 1),
                exc_info=True,
            )
            if kill_on_failure:
                self._mark_as_errored()
                await self._kill_workers_and_confirm_dead()
            raise

    # ------------------------ state and misc queries ------------------------

    @property
    def is_allocated(self) -> bool:
        # [FUNCTION DOC] Property: whether the state is any allocated state (has worker handles).
        return isinstance(self._state, StateAllocatedBase)

    @property
    def is_alive(self) -> bool:
        # [FUNCTION DOC] Property: whether the cell is initialized and healthy (alive state).
        return isinstance(self._state, StateAllocatedAlive)

    @property
    def is_errored(self) -> bool:
        # [FUNCTION DOC] Property: whether a call failed and the cell is in the errored state.
        return isinstance(self._state, StateAllocatedErrored)

    @property
    def is_uninitialized(self) -> bool:
        # [FUNCTION DOC] Property: whether the cell has not yet been initialized.
        return isinstance(self._state, StateAllocatedUninitialized)

    @property
    def state_name(self) -> str:
        # [FUNCTION DOC] Property: class name of the current state, for logs and display.
        return type(self._state).__name__

    def cell_status(self) -> CellStatus:
        # [FUNCTION DOC] Externally visible status computed from the state, the health checker's
        # current status and the workers hash (see status helper).
        return compute_cell_status(self._state, self.health_checker.status, workers_hash=self.workers_hash)

    @property
    def indep_dp_info(self) -> IndepDPInfo | None:
        # [FUNCTION DOC] Property: the recorded independent-DP info. Only valid in the alive or errored
        # states (assertion error when uninitialized); an errored cell that was never initialized
        # reports none.
        assert isinstance(self._state, (StateAllocatedAlive, StateAllocatedErrored))
        return self._state.indep_dp_info

    def _get_worker_handles(self) -> list[BaseWorkerHandle]:
        # [FUNCTION DOC] HELPER: The worker handles of the current state, in rank order; assertion
        # error (naming the state) if the cell is not allocated.
        assert isinstance(
            self._state, StateAllocatedBase
        ), f"Cell {self.cell_id} is not allocated (state={type(self._state).__name__})"
        return self._state.worker_handles


async def _kill_worker(handle: BaseWorkerHandle) -> None:
    # [FUNCTION DOC] HELPER: Ask one worker to kill itself, waiting at most 10 s for the RPC. If the
    # worker is already unreachable or its server restarted, returns silently (it is gone). On
    # timeout logs a warning and returns, leaving the later death confirmation probe to decide. Other
    # errors propagate.
    try:
        await asyncio.wait_for(handle.kill_self(), timeout=KILL_RPC_TIMEOUT_S)
    except (WorkerUnreachableError, ServerRestartedError):
        return
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning("Timed out asking a worker to kill itself; falling back to the death confirmation probe")
