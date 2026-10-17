# [MODULE DOC] Trainer controller: the actor-like component that owns all cells (independent data
# parallel replicas) of one trainer (a policy or its critic). It watches the provider for cells
# appearing/disappearing, creates cell wrappers with optional heartbeat health checkers, initializes
# cells, runs training steps across all alive cells with retries and fault-tolerance bookkeeping
# (witness ids, step outcomes, events), heals recovering cells by cooperatively copying a checkpoint
# from a healthy cell and reconfiguring the independent-DP quorum, and forwards administrative calls
# (save, export, weight update, offload/onload, multi-LoRA slot commands) to the first or all alive
# cells. Constants: up to 30 retry attempts; cells-ready timeout 3600 s.
import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.ray.specs.train import compute_trainer_num_cells, compute_trainer_pool_id
from miles.ray.train.cell import TrainerCell
from miles.ray.train.cell_monitor import create_trainer_cell_health_checker
from miles.utils import object_store
from miles.utils.async_utils import AsyncioGatherUtils, gather_and_raise_first
from miles.utils.audit_utils.event_analyzer import analyzer as event_analyzer
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import (
    CellReconfigureEvent,
    TrainGroupStepEndEvent,
    WitnessAllocateIdEvent,
)
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity
from miles.utils.audit_utils.witness.allocator import WitnessIdAllocator, read_persisted_witness_counter
from miles.utils.data import RolloutDataPack, remove_train_output_refs
from miles.utils.ft_utils.api_server.models import CellStatus
from miles.utils.ft_utils.health_checker import ActivenessTracker, NoopHealthChecker, SimpleHealthCheckerConfig
from miles.utils.ft_utils.indep_dp import IndepDPInfo, create_tcp_store
from miles.utils.init_once import InitOnce, init_once
from miles.utils.logging_utils import configure_logger
from miles.utils.retry_utils import NonRetryableError, retry, retry_until_deadline
from miles.utils.test_utils.ft_test_actions import FTTestActionControllerExecutor
from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.rpc.common.wire_types import Pickled
from miles.utils.workers.types import DeploymentIdentity
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.utils import apply_cell_observation

logger = logging.getLogger(__name__)


_RETRY_MAX_ATTEMPTS = 30
_CELLS_READY_TIMEOUT_SECONDS = 3600.0


def compute_trainer_health_checker_config(args, *, expected_num_cells: int) -> SimpleHealthCheckerConfig | None:
    # [FUNCTION DOC] Heartbeat checker configuration for trainer cells: none when only one cell is
    # expected (a single cell has no peers to heal, so no checking); otherwise the config read from
    # args options prefixed trainer_heartbeat_checker.
    if expected_num_cells == 1:
        return None
    return SimpleHealthCheckerConfig.from_args(args, prefix="trainer_heartbeat_checker")


class TrainerController:
    # [CLASS DOC] One controller per trainer. Construction-time state: deployment identity, trainer
    # id, role, reference-model and OPD teacher flags, pool id, cell provider and operations, the
    # watcher disposer, the independent-DP quorum id (starts 0, increments on each reconfiguration),
    # the independent-DP TCP store and its address (only created when more than one cell is
    # expected), a health checker activeness tracker (initially active) shared by all cell checkers,
    # and a mapping from cell id to cell. KNOWN ISSUE: several attributes (args, health checker
    # config, witness allocator, test action executor) are only created by init, so methods that
    # use them (including the expected-cells properties) raise attribute errors if called earlier.
    # Methods are RPC-called; the cell list is always ordered by cell index.

    def __init__(
        self,
        *,
        deployment_identity: DeploymentIdentity,
        cell_provider: BaseWorkerProvider,
        cell_operations: BaseCellOperations,
        trainer_id: str,
        role: str,
        with_ref: bool,
        with_opd_teacher: bool = False,
    ) -> None:
        # [FUNCTION DOC] Store the arguments, create the init-once guard, compute the pool id from the
        # trainer id, and set up empty cell/quorum/store state as described in the class doc. No I/O.
        self._init_once = InitOnce(type(self).__name__)
        self._deployment_identity = deployment_identity
        self._trainer_id = trainer_id
        self._role = role
        self._with_ref = with_ref
        self._with_opd_teacher = with_opd_teacher
        self._pool_id = compute_trainer_pool_id(trainer_id)
        self._provider = cell_provider
        self._cell_operations = cell_operations
        self._watcher_disposer: StopWatchFn | None = None

        self._indep_dp_quorum_id = 0
        self._indep_dp_store: Any | None = None
        self._indep_dp_store_addr: str | None = None

        self._health_checker_activeness = ActivenessTracker(active=True)

        self._cells_by_id: dict[str, TrainerCell] = {}

    @property
    def pool_id(self) -> str:
        # [FUNCTION DOC] Property: the trainer pool id this controller manages.
        return self._pool_id

    @property
    def expected_num_cells(self) -> int:
        # [FUNCTION DOC] Property: public alias of the expected cell count. Requires init to have set
        # args.
        return self._expected_num_cells

    @property
    def _expected_num_cells(self) -> int:
        # [FUNCTION DOC] HELPER: Property: number of cells the trainer pool should have, computed from
        # args and the role (independent DP replicas, else 1) on every access.
        return compute_trainer_num_cells(self.args, role=self._role)

    @property
    def _cells(self) -> list[TrainerCell]:
        # [FUNCTION DOC] HELPER: Property: all known cells sorted ascending by cell index (a new list on
        # each access).
        return sorted(self._cells_by_id.values(), key=lambda cell: cell.cell_index)

    def has_errored_cell(self) -> bool:
        # [FUNCTION DOC] True if any known cell is in the errored state.
        return any(cell.is_errored for cell in self._cells)

    @property
    def cell_ids(self) -> list[str]:
        # [FUNCTION DOC] Property: ids of all known cells in cell index order.
        return [cell.cell_id for cell in self._cells]

    async def _wait_expected_num_cells(self, timeout: float = _CELLS_READY_TIMEOUT_SECONDS) -> None:
        # [FUNCTION DOC] HELPER: Retry a check until at least the expected number of cells have been
        # observed or the timeout (default 3600 s) expires, with backoff from 1 s up to 5 s, retrying
        # only on timeout errors (the deadline helper raises when exhausted). Note it waits for
        # "at least" the count, so extra cells are tolerated.
        async def _check(_remaining: float) -> None:
            # [FUNCTION DOC] HELPER: One attempt: raise a timeout error with observed/expected counts if
            # fewer cells than expected are known; else return.
            expected = self._expected_num_cells
            if len(self._cells_by_id) < expected:
                raise TimeoutError(f"only {len(self._cells_by_id)} of {expected} trainer cells observed")

        await retry_until_deadline(
            _check,
            total_seconds=timeout,
            retry_on=TimeoutError,
            initial_delay=1.0,
            max_delay=5.0,
            log_fields=dict(tag="ft", spec=self._pool_id),
        )

    async def _reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: Callback from the provider's cell watch: compare the observed cell info
        # (none = gone) with the known cell (using its workers hash) via the shared observation helper,
        # which calls the add or remove helper as needed (e.g. a changed hash means replacement).
        actual = self._cells_by_id.get(cell_id)
        await apply_cell_observation(
            cell_id=cell_id,
            observed=observed,
            actual_workers_hash=actual.workers_hash if actual is not None else None,
            add=self._add_cell,
            remove=self._remove_cell,
        )

    async def _add_cell(self, cell_id: str, observed: CellInfo) -> None:
        # [FUNCTION DOC] HELPER: Create and register a cell using the cell index from the observed
        # info's metadata (key error if missing) and its workers hash.
        self._cells_by_id[cell_id] = self._create_cell(
            cell_id, cell_index=observed.meta["cell_index"], workers_hash=observed.workers_hash
        )

    async def _remove_cell(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Forget the cell (key error if unknown) and stop its health checker. It
        # does not kill workers; the platform already removed them.
        cell = self._cells_by_id.pop(cell_id)
        cell.health_checker.stop()

    def _create_cell(self, cell_id: str, *, cell_index: int, workers_hash: str) -> TrainerCell:
        # [FUNCTION DOC] HELPER: Build a cell wrapper with the controller's role and flags, starting
        # with a no-op health checker. If a heartbeat checker config exists (more than one expected
        # cell), replace the checker with a real one bound to this cell and the shared activeness
        # tracker. The checker is only started when the cell is initialized.
        cell = TrainerCell(
            args=self.args,
            role=self._role,
            with_ref=self._with_ref,
            with_opd_teacher=self._with_opd_teacher,
            cell_id=cell_id,
            cell_index=cell_index,
            workers_hash=workers_hash,
            health_checker=NoopHealthChecker(),
            provider=self._provider,
        )

        if self._health_checker_config is not None:
            cell.health_checker = create_trainer_cell_health_checker(
                cell=cell,
                config=self._health_checker_config,
                get_activeness=self._health_checker_activeness.get,
            )

        return cell

    async def dispose(self) -> None:
        # [FUNCTION DOC] Stop watching provider cells (call and clear the watcher disposer if any). Does
        # not stop cell health checkers or workers.
        if (disposer := self._watcher_disposer) is not None:
            await disposer()
            self._watcher_disposer = None

    # ------------------------ API :: train ------------------------

    async def train(
        self,
        rollout_id: int,
        rollout_data_pack: RolloutDataPack,
        external_data: list[TrainStepOutput] | None = None,
    ) -> list[TrainStepOutput]:
        # [FUNCTION DOC] Train on one rollout with fault-tolerant retries. Asserts external data is only
        # given with exactly one cell. Runs the event analysis in a thread, then retries (up to 30
        # attempts) an attempt function that: allocates witness info; refreshes cells (heal and
        # reconfigure); runs the cell train step on every alive cell concurrently capturing exceptions
        # per cell (no recoverability check here); flattens successful cells' per-worker results; checks
        # the attempt (on any failure releases the object-store references held by the results and
        # re-raises); logs a step-end event; and returns the flat list of worker step outputs. After
        # success runs the test-action hooks for the rollout id and returns the outputs.
        """Do one rollout training"""

        assert (
            external_data is None or len(self._cells) == 1
        ), "external_data is only supported for a single cell, i.e. without independent DP"

        await asyncio.to_thread(event_analyzer.run_analysis_from_args, self.args)

        async def _fn(attempt: int) -> list[TrainStepOutput]:
            # [FUNCTION DOC] HELPER: One training attempt, as described by the enclosing function.
            # Returns the flattened step outputs; raises whatever the check raises to trigger a retry.
            witness_info = self._allocate_witness_info(
                rollout_id=rollout_id,
                attempt=attempt,
                sample_indices=rollout_data_pack.sample_indices,
            )

            log_structured(logger.info, tag="ft", op="train", phase="start", rollout=rollout_id, attempt=attempt)
            await self._refresh_cells(rollout_id=rollout_id)
            snapshot_alive_cells, results = await self._gather_all_alive_and_catch(
                lambda cell: cell.train(
                    rollout_id=rollout_id,
                    rollout_data_ref=rollout_data_pack.data_ref,
                    witness_info=witness_info,
                    attempt=attempt,
                    external_data=external_data,
                ),
                debug_name="execute_all_alive_and_catch#train",
                check_recoverable=False,
            )
            worker_results = [
                worker_result
                for cell_results in results
                if not isinstance(cell_results, BaseException)
                for worker_result in cell_results
            ]

            try:
                self._check_train_one_attempt(snapshot_alive_cells, results)
            except Exception:
                remove_train_output_refs(worker_results)
                raise

            self._log_step_end_event(
                rollout_id=rollout_id,
                snapshot_alive_cells=snapshot_alive_cells,
                results=results,
            )

            return worker_results

        worker_results = await retry(_fn, max_attempts=_RETRY_MAX_ATTEMPTS)

        await self._test_action_executor.run_after_step(rollout_id=rollout_id)

        return worker_results

    def _allocate_witness_info(self, *, rollout_id: int, attempt: int, sample_indices):
        # [FUNCTION DOC] HELPER: When witnessing is disabled returns none. Otherwise allocates one
        # witness id per sample index from the allocator and, if the event logger is initialized, logs
        # an allocation event (rollout id, attempt, mapping witness id to sample index, the allocator's
        # counter afterwards). Returns the witness info.
        if self._witness_allocator is None:
            return None

        witness_info = self._witness_allocator.allocate(num_ids=len(sample_indices))

        if is_event_logger_initialized():
            get_event_logger().log(
                WitnessAllocateIdEvent,
                dict(
                    rollout_id=rollout_id,
                    attempt=attempt,
                    witness_id_to_sample_index=dict(zip(witness_info.witness_ids, sample_indices, strict=True)),
                    counter_after=self._witness_allocator.counter,
                ),
            )

        return witness_info

    def _log_step_end_event(self, *, rollout_id: int, snapshot_alive_cells: list, results: list):
        # [FUNCTION DOC] HELPER: If the event logger is initialized, log a group step-end event with a
        # mapping from cell index to either the text "error" (the cell raised) or the list of
        # per-worker outcomes. Cells and results are paired strictly.
        if is_event_logger_initialized():
            cell_outcomes = {
                cell.cell_index: (
                    "error" if isinstance(cell_results, BaseException) else [r.outcome for r in cell_results]
                )
                for cell, cell_results in zip(snapshot_alive_cells, results, strict=True)
            }
            get_event_logger().log(
                TrainGroupStepEndEvent,
                dict(rollout_id=rollout_id, cell_outcomes=cell_outcomes),
            )

    def _check_train_one_attempt(self, snapshot_alive_cells, results):
        # [FUNCTION DOC] HELPER: Decide whether an attempt succeeded. Computes outcome groups (errored,
        # discarded, normal cell indices). If no cell is normal and none is discarded (all errored):
        # builds an all-cells-failed error (non-retryable if no cell can recover or the first cause was
        # non-retryable, else a retryable runtime error), logs the decision (give up vs retry) and
        # raises it chained from the first exception. If some cells discarded: logs a warning and
        # raises a value error to force a retry. KNOWN ISSUE: the logged decision may say give up when
        # nothing is recoverable, yet the raised value error is the same either way. Otherwise (some
        # normal, none discarded, even if some errored) logs success without retry: erroring cells
        # may have failed after exchanging gradients, so survivors' results are accepted.
        outcomes = TrainerController._compute_attempt_outcomes(snapshot_alive_cells, results)
        if not outcomes["normal"] and not outcomes["discarded"]:
            cause = _first_exception(results)
            error = self._make_all_cells_failed_error("All cells failed in this training attempt", cause=cause)
            log_structured(
                logger.error,
                tag="ft",
                op="check",
                **outcomes,
                decision="give_up" if isinstance(error, NonRetryableError) else "retry",
                reason="all alive cells failed",
            )
            raise error from cause

        # NOTE: If some cells errors + all other cells claim normal, we do *not* retry
        #       This may happen when some cells fails *after* exchanging gradients w/ others
        if outcomes["discarded"]:
            log_structured(
                logger.warning,
                tag="ft",
                op="check",
                **outcomes,
                decision="retry" if self._is_recoverable() else "give_up",
                reason="discarded_should_retry",
            )
            raise ValueError("Exists DISCARDED_SHOULD_RETRY, thus need retry")

        log_structured(
            logger.info,
            tag="ft",
            op="check",
            **outcomes,
            decision="no_retry",
            reason="survivors normal, gradients valid",
        )

    @staticmethod
    def _compute_attempt_outcomes(snapshot_alive_cells, results) -> dict[str, list[int]]:
        # [FUNCTION DOC] HELPER: Static. Pair cells with results strictly and return a mapping with
        # three lists of cell indices: errored (result is an exception), discarded (no exception but
        # any worker outcome is discarded-should-retry), and normal (all the rest).
        paired = list(zip(snapshot_alive_cells, results, strict=True))
        errored = [c.cell_index for c, r in paired if isinstance(r, BaseException)]
        discarded = [
            c.cell_index
            for c, r in paired
            if not isinstance(r, BaseException)
            and any(o.outcome == TrainStepOutcome.DISCARDED_SHOULD_RETRY for o in r)
        ]
        normal = [c.cell_index for c, r in paired if c.cell_index not in errored and c.cell_index not in discarded]
        return {"errored": errored, "discarded": discarded, "normal": normal}

    # ------------------------ API :: others ------------------------

    @init_once
    async def init(self, args: Pickled) -> list[Any]:
        # [FUNCTION DOC] One-time startup (repeat calls are no-ops). Stores args; configures process
        # logging with the trainer controller identity (trainer id, model id); initializes the object
        # store without contributing a segment; if more than one cell is expected, creates the TCP
        # store used for independent-DP rendezvous and remembers its address; computes the heartbeat
        # checker config; creates the witness allocator when enabled (resuming its counter from the
        # persisted value in the debug event directory if configured); creates the fault-tolerance test
        # action executor; starts watching provider cells; waits for the expected cell count; then
        # initializes ALL cells concurrently, each with independent-DP info assuming every cell is
        # alive and the store address. Returns the per-worker init results of all cells flattened in
        # cell order (worker values are restored rollout ids).
        """
        Observe the controller's cells, then allocate GPU resources and initialize
        model, optimzier, local ckpt, etc.
        """
        self.args = args
        configure_logger(
            args, source=TrainerControllerProcessIdentity(trainer_id=self._trainer_id, model_id=args.trainer_model_id)
        )
        object_store.init_instance(args, contribute_segment=False)

        if self._expected_num_cells > 1:
            self._indep_dp_store, self._indep_dp_store_addr = create_tcp_store()

        self._health_checker_config = compute_trainer_health_checker_config(
            args, expected_num_cells=self._expected_num_cells
        )

        self._witness_allocator: WitnessIdAllocator | None = (
            WitnessIdAllocator(buffer_size=args.witness_buffer_size) if args.enable_witness else None
        )
        if self._witness_allocator is not None and args.save_debug_event_data is not None:
            self._witness_allocator.resume(read_persisted_witness_counter(Path(args.save_debug_event_data)))

        self._test_action_executor = FTTestActionControllerExecutor.from_args(
            args, controller=self, cell_operations=self._cell_operations
        )

        self._watcher_disposer = await self._provider.watch_cells(self._reconcile)
        await self._wait_expected_num_cells()

        cell_results = await asyncio.gather(
            *[
                cell.init(
                    indep_dp_info=self._compute_indep_dp_info(
                        cell_index=cell.cell_index,
                        # all cells will be alive for this first initialization
                        alive_cell_indices=list(range(len(self._cells))),
                    ),
                    indep_dp_store_addr=self._indep_dp_store_addr,
                )
                for cell in self._cells
            ]
        )
        return [item for sublist in cell_results for item in sublist]

    async def is_initialized(self) -> bool:
        # [FUNCTION DOC] Whether init has completed (used by a new orchestrator to detect a controller
        # left over from an earlier orchestration, i.e. take over).
        return self._init_once.is_initialized()

    async def load_state(self) -> list[Any]:
        # [FUNCTION DOC] Reload saved state on a controller that is already initialized (assertion error
        # otherwise). Waits for the expected cells, asserts every cell is alive (assertion error listing
        # the others), then loads state on all cells concurrently, raising the first failure. Returns
        # the per-worker results flattened in cell order.
        assert self._init_once.is_initialized()

        await self._wait_expected_num_cells(timeout=_CELLS_READY_TIMEOUT_SECONDS)

        not_alive = [cell.cell_id for cell in self._cells if not cell.is_alive]
        assert not not_alive, f"a reload does not support cells that are not alive: {not_alive}"

        cell_results = await gather_and_raise_first([cell.load_state() for cell in self._cells])
        return [item for sublist in cell_results for item in sublist]

    async def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        # [FUNCTION DOC] Save the model through a single cell to avoid write conflicts: tries the first
        # alive cell, and if it fails (it is then marked errored) retries up to 30 attempts which pick
        # the next alive cell. KNOWN ISSUE: docstring says cell 0, but the code uses the first ALIVE
        # cell, which is not necessarily cell 0.
        """Save actor model. Only cell 0 saves to avoid file write conflicts."""
        # Catch with vanilla retry: cells w/ exceptions are auto marked errored, thus retry will find the next one
        await retry(
            lambda _: self._execute_first_alive("save_model", rollout_id=rollout_id, force_sync=force_sync),
            max_attempts=_RETRY_MAX_ATTEMPTS,
        )

    async def export_hf(self, rollout_id: int, path: str) -> None:
        # [FUNCTION DOC] Export current weights as a Hugging Face checkpoint to the path through the
        # first alive cell with up to 30 retries over successive alive cells. KNOWN ISSUE: docstring
        # says cell 0 but the first alive cell is used.
        """Export current weights as an HF checkpoint. Only cell 0 exports to avoid file write conflicts."""
        await retry(
            lambda _: self._execute_first_alive("export_hf", rollout_id=rollout_id, path=path),
            max_attempts=_RETRY_MAX_ATTEMPTS,
        )

    async def update_weights(self, info: UpdatableEngines, rollout_id: int | None = None) -> int | None:
        # [FUNCTION DOC] Push weights to the engines described by the updatable-engines info using the
        # first alive cell (workers of that cell cooperate), with up to 30 retries over successive
        # alive cells (TODO in code: use all cells). Returns the weight version reported by the first
        # worker of the cell that succeeded (may be none when no version applies).
        """Broadcast weights to rollout engines and answer the version they now serve."""
        log_structured(logger.info, tag="ft", op="update_weights", phase="start", rollout=rollout_id)
        # TODO: allow using all cells to update weights (instead of first alive cell)
        # Catch with vanilla retry: cells w/ exceptions are auto marked errored, thus retry will find the next one
        weight_versions = await retry(
            lambda _: self._execute_first_alive("update_weights", info=info),
            max_attempts=_RETRY_MAX_ATTEMPTS,
        )
        return weight_versions[0]

    async def get_deployment_identity(self) -> DeploymentIdentity:
        # [FUNCTION DOC] Return the identity (run, deploy component, instance, trainer id) given at
        # construction; used by orchestrators to verify they reach the right deployment.
        return self._deployment_identity

    async def onload(self) -> None:
        # [FUNCTION DOC] Wake all alive cells' workers (bring GPU memory back) without retry (failing
        # cells are marked errored and then skipped), then re-activate the health checkers. Raises
        # non-retryable error if no cell is alive, or when no cell can recover.
        # Catch *without* retry: cells w/ exceptions are auto marked errored, and will not be used
        await self._execute_all_alive_and_catch("wake_up")
        self._health_checker_activeness.bump_active(True)

    @contextmanager
    def _paused_health_checkers(self) -> Iterator[None]:
        # [FUNCTION DOC] HELPER: Context manager: mark health checkers inactive on entry (bumping the
        # epoch so in-flight checks are discarded) and active again on exit, even on error.
        self._health_checker_activeness.bump_active(False)
        try:
            yield
        finally:
            self._health_checker_activeness.bump_active(True)

    async def offload(self) -> None:
        # [FUNCTION DOC] Deactivate health checkers (workers will not answer while asleep), then put all
        # alive cells' workers to sleep (offloading GPU memory) without retry. Checkers are re-activated
        # by the onload call, not here.
        self._health_checker_activeness.bump_active(False)
        # Catch *without* retry: cells w/ exceptions are auto marked errored, and will not be used
        await self._execute_all_alive_and_catch("sleep")

    async def clear_memory(self) -> None:
        # [FUNCTION DOC] Ask all alive cells' workers to clear cached GPU memory, without retry.
        # Catch *without* retry: cells w/ exceptions are auto marked errored, and will not be used
        await self._execute_all_alive_and_catch("clear_memory")

    async def offload_grad_buffer(self) -> None:
        # [FUNCTION DOC] Ask all alive cells' workers to offload their gradient buffers, without retry.
        # Catch *without* retry: cells w/ exceptions are auto marked errored, and will not be used
        await self._execute_all_alive_and_catch("offload_grad_buffer")

    # ------------------------ API :: multi-LoRA slot commands ------------------------

    async def _execute_slots(self, fn_name: str, **kwargs) -> list:
        # [FUNCTION DOC] HELPER: Forward a multi-LoRA slot command to the only cell. Unpacking raises a
        # value error unless there is exactly one cell; asserts the cell is alive. Returns per-worker
        # results.
        (cell,) = self._cells
        assert cell.is_alive, "the Tinker trainer cell is unavailable"
        return await cell.execute(fn_name, **kwargs)

    async def forward_backward(self, batch_id: int, data_ref: object_store.StoreObjectRef) -> list:
        # [FUNCTION DOC] Run forward and backward for a batch id on the data referenced in the object
        # store, on all workers of the single cell. Returns per-worker results.
        return await self._execute_slots("forward_backward", batch_id=batch_id, rollout_data_ref=data_ref)

    async def optim_step(self, adam_params_by_slot: dict[int, dict]) -> list:
        # [FUNCTION DOC] Run an optimizer step with per-slot Adam parameters (mapping from slot index to
        # parameter mapping) on the single cell's workers. Returns per-worker results.
        return await self._execute_slots("optim_step", adam_params_by_slot=adam_params_by_slot)

    async def forward_only(self, batch_id: int, data_ref: object_store.StoreObjectRef) -> list:
        # [FUNCTION DOC] Run a forward-only pass (no gradients) for a batch id on the referenced data
        # on the single cell's workers. Returns per-worker results.
        return await self._execute_slots("forward_only", batch_id=batch_id, rollout_data_ref=data_ref)

    async def load_slot(
        self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> list:
        # [FUNCTION DOC] Load a LoRA adapter into a slot with the given rank and alpha scaling, from an
        # optional checkpoint path, optionally restoring optimizer state (default yes). Returns
        # per-worker results.
        return await self._execute_slots(
            "load_slot", slot=slot, rank=rank, alpha=alpha, ckpt_path=ckpt_path, load_optimizer=load_optimizer
        )

    async def save_slot(self, slot: int, path: str, metadata: dict | None = None) -> list:
        # [FUNCTION DOC] Save a slot's adapter (and state) to a path with optional metadata mapping.
        # Returns per-worker results.
        return await self._execute_slots("save_slot", slot=slot, path=path, metadata=metadata)

    async def export_slot(self, slot: int, rank: int, alpha: float, path: str, metadata: dict | None = None) -> list:
        # [FUNCTION DOC] Export a slot's adapter for serving, with rank and alpha, to a path with
        # optional metadata. Returns per-worker results.
        return await self._execute_slots(
            "export_slot", slot=slot, rank=rank, alpha=alpha, path=path, metadata=metadata
        )

    async def unload_slot(self, slot: int) -> list:
        # [FUNCTION DOC] Free the adapter in a slot on the single cell's workers. Returns per-worker
        # results.
        return await self._execute_slots("unload_slot", slot=slot)

    async def get_train_parallel_config(self) -> dict[str, Any]:
        # [FUNCTION DOC] Parallelism configuration mapping (e.g. data-parallel size) reported by the
        # first worker of the first alive cell, with the usual no-retry failure handling.
        return (await self._execute_first_alive("get_train_parallel_config"))[0]

    async def get_cell_statuses(self) -> dict[str, CellStatus]:
        # [FUNCTION DOC] Mapping from cell id to the cell's externally visible status, computed over a
        # copy of the cell items so concurrent reconcile is safe.
        return {cell_id: cell.cell_status() for cell_id, cell in list(self._cells_by_id.items())}

    # ------------------------ utils to forward calls to cells ------------------------

    def _is_recoverable(self) -> bool:
        # [FUNCTION DOC] HELPER: True while at least one cell is alive (others can be healed from it).
        return any(cell.is_alive for cell in self._cells)

    def _make_all_cells_failed_error(self, message: str, *, cause: BaseException | None = None) -> Exception:
        # [FUNCTION DOC] HELPER: Choose the error type for total failure: non-retryable error if the
        # cause itself is non-retryable; otherwise a plain runtime error (retryable) while some cell is
        # alive, else non-retryable error. Returns (does not raise) the error with the message.
        if isinstance(cause, NonRetryableError):
            return NonRetryableError(message)
        return RuntimeError(message) if self._is_recoverable() else NonRetryableError(message)

    def _raise_if_no_cell_can_recover(self, *, debug_name: str, cause: BaseException | None) -> None:
        # [FUNCTION DOC] HELPER: Returns if any cell is alive; else raises the all-cells-failed error
        # (see error helper) naming the debug operation, chained from the cause.
        if self._is_recoverable():
            return
        raise self._make_all_cells_failed_error(f"All cells failed during {debug_name}", cause=cause) from cause

    async def _execute_all_alive_and_catch(self, fn_name: str, *, check_recoverable: bool = True, **kwargs):
        # [FUNCTION DOC] HELPER: Call the named method with the same keyword arguments on every alive
        # cell concurrently, capturing per-cell exceptions (see gather helper). Returns the gather
        # helper's pair (snapshot of alive cells, outputs).
        return await self._gather_all_alive_and_catch(
            lambda cell: cell.execute(fn_name, **kwargs),
            debug_name=f"execute_all_alive_and_catch#{fn_name}",
            check_recoverable=check_recoverable,
        )

    async def _gather_all_alive_and_catch(self, compute_coroutine, *, debug_name: str, check_recoverable: bool = True):
        # [FUNCTION DOC] HELPER: Snapshot the alive cells (raises non-retryable error "No alive cells"
        # if none), call the supplied coroutine factory on each concurrently with exceptions returned
        # instead of raised, log errors, and, if requested (default), raise when no cell remains alive
        # (see recoverability helper). Returns (alive cells snapshot, outputs list in the same order,
        # each a result or an exception object). Deliberately has no timeout: a hung cell is detected
        # externally by stale heartbeats and suspended, which unblocks this gather.
        snapshot_alive_cells = [c for c in self._cells if c.is_alive]
        if not snapshot_alive_cells:
            raise NonRetryableError("No alive cells")
        # NOTE: no timeout here. If a cell hangs, the external FT controller
        # detects stale heartbeat via cell_status() and suspends the cell through
        # the worker manager, which unblocks this gather with ActorDiedError.
        outputs = await asyncio.gather(
            *[compute_coroutine(cell) for cell in snapshot_alive_cells],
            return_exceptions=True,
        )
        AsyncioGatherUtils.log_error(outputs, debug_name=debug_name)
        if check_recoverable:
            self._raise_if_no_cell_can_recover(debug_name=debug_name, cause=_first_exception(outputs))
        return snapshot_alive_cells, outputs

    async def _execute_first_alive(self, fn_name: str, **kwargs):
        # [FUNCTION DOC] HELPER: Call the named method on the alive cell with the lowest index. Raises
        # non-retryable error if none is alive. On an ordinary failure (the cell is then marked
        # errored by the cell wrapper): if no cell can recover raises the all-cells-failed error,
        # otherwise re-raises the original error so a retry loop can pick the next alive cell. Returns
        # the per-worker result list.
        alive_cells = [c for c in self._cells if c.is_alive]
        if not alive_cells:
            raise NonRetryableError("No alive cells, therefore cannot heal anymore")
        try:
            return await alive_cells[0].execute(fn_name, **kwargs)
        except Exception as cause:
            self._raise_if_no_cell_can_recover(debug_name=f"execute_first_alive#{fn_name}", cause=cause)
            raise

    # ------------------------ internals for stop/start ------------------------

    async def _refresh_cells(self, *, rollout_id: int) -> None:
        # [FUNCTION DOC] HELPER: Heal and reconfigure independent-DP cells before a training attempt.
        # Snapshot healing (uninitialized) and alive cell indices; the future alive set is their sorted
        # union. Raises non-retryable error if no cell is alive. Reconfiguration is needed if any cell
        # is healing or any currently alive cell's recorded alive set differs from the future set;
        # otherwise returns after logging. If needed: increment the quorum id; pick the lowest alive
        # cell as the checkpoint source (TODO: balance); compute the source's rank within the future
        # alive set and the destination ranks of healing cells; with health checkers paused, run
        # concurrently over cells in the future alive set: alive cells reconfigure (the source also
        # sends checkpoints to all destination ranks), healing cells init and receive the checkpoint.
        # Exceptions are collected, logged and not raised (failing cells mark themselves errored). If
        # none failed, asserts that exactly the future alive set is alive and logs a reconfigure event
        # (source cell only when something healed); otherwise logs an error and leaves it to the
        # attempt check. Each cell's new info is computed with the helper using the new quorum id.
        snapshotted_healing_indices = [c.cell_index for c in self._cells if c.is_uninitialized]
        snapshotted_alive_indices = [c.cell_index for c in self._cells if c.is_alive]
        will_alive_indices = sorted(list(set(snapshotted_healing_indices + snapshotted_alive_indices)))
        all_states = [(c.cell_index, c.state_name) for c in self._cells]
        log_structured(
            logger.info,
            tag="ft",
            op="refresh",
            phase="start",
            rollout=rollout_id,
            alive=snapshotted_alive_indices,
            healing=snapshotted_healing_indices,
            all_states=all_states,
            quorum=self._indep_dp_quorum_id,
        )
        if not snapshotted_alive_indices:
            raise NonRetryableError("Cannot recover when all cells are dead")

        # Step 0: Determine whether need to reconfigure
        exists_alive_cell_changed_config = any(
            cell.indep_dp_info.alive_cell_indices != will_alive_indices
            for cell in self._cells
            if cell.cell_index in snapshotted_alive_indices
        )
        exists_healing_cell = len(snapshotted_healing_indices) != 0
        needs_reconfigure = exists_healing_cell or exists_alive_cell_changed_config
        if not needs_reconfigure:
            log_structured(
                logger.info,
                tag="ft",
                op="refresh",
                phase="decision",
                rollout=rollout_id,
                needs_reconfigure=False,
                reason="alive_config_unchanged,no_healing",
                quorum=self._indep_dp_quorum_id,
            )
            return
        reason = "+".join(
            r
            for r, on in [
                ("healing_cell", exists_healing_cell),
                ("alive_config_changed", exists_alive_cell_changed_config),
            ]
            if on
        )
        log_structured(
            logger.info,
            tag="ft",
            op="refresh",
            phase="decision",
            rollout=rollout_id,
            needs_reconfigure=True,
            reason=reason,
            will_alive=will_alive_indices,
            quorum_from=self._indep_dp_quorum_id,
            quorum_to=self._indep_dp_quorum_id + 1,
        )

        # Step 1: Bump states
        self._indep_dp_quorum_id += 1

        # Step 2: Cooperatively prepare
        src_cell_index = snapshotted_alive_indices[0]  # TODO make it balanced, and support multi-src-to-one-dst
        src_alive_rank = will_alive_indices.index(src_cell_index)
        ckpt_dst_alive_ranks = [will_alive_indices.index(x) for x in snapshotted_healing_indices]

        with self._paused_health_checkers():
            coop_prepare_outputs = await asyncio.gather(
                *[
                    (
                        c.prepare_indep_dp_mode_alive(
                            indep_dp_info=self._compute_indep_dp_info(
                                c.cell_index, alive_cell_indices=will_alive_indices
                            ),
                            indep_dp_store_addr=self._indep_dp_store_addr,
                            send_ckpt_dst_ranks=ckpt_dst_alive_ranks if c.cell_index == src_cell_index else [],
                        )
                        if c.cell_index in snapshotted_alive_indices
                        else c.prepare_indep_dp_mode_healing(
                            indep_dp_info=self._compute_indep_dp_info(
                                c.cell_index, alive_cell_indices=will_alive_indices
                            ),
                            indep_dp_store_addr=self._indep_dp_store_addr,
                            recv_ckpt_src_rank=src_alive_rank if c.cell_index in snapshotted_healing_indices else None,
                        )
                    )
                    for c in self._cells
                    if c.cell_index in will_alive_indices
                ],
                return_exceptions=True,
            )
        # No need to do anything else - cells with exceptions will auto mark itself as errored
        AsyncioGatherUtils.log_error(coop_prepare_outputs, debug_name="refresh_cells#cooperatively_prepare")

        if not AsyncioGatherUtils.has_error(coop_prepare_outputs):
            assert [c.cell_index for c in self._cells if c.is_alive] == will_alive_indices
            log_structured(
                logger.info,
                tag="ft",
                op="refresh",
                phase="end",
                rollout=rollout_id,
                quorum=self._indep_dp_quorum_id,
                alive=will_alive_indices,
                healed=snapshotted_healing_indices,
                reconfigured=True,
            )
            self._log_reconfigure_event(
                rollout_id=rollout_id,
                src_cell_index=src_cell_index if snapshotted_healing_indices else None,
                healed_cell_indices=snapshotted_healing_indices,
                alive_cell_indices_after=will_alive_indices,
            )
        else:
            log_structured(
                logger.error,
                tag="ft",
                op="refresh",
                phase="end",
                rollout=rollout_id,
                reconfigured=False,
                quorum=self._indep_dp_quorum_id,
                reason="cooperative_prepare_raised",
            )

    def _log_reconfigure_event(
        self,
        *,
        rollout_id: int,
        src_cell_index: int | None,
        healed_cell_indices: list[int],
        alive_cell_indices_after: list[int],
    ) -> None:
        # [FUNCTION DOC] HELPER: If the event logger is initialized, log a cell reconfigure event with
        # rollout id, current quorum id, source cell index (or none), healed cell indices and alive cell
        # indices after.
        if is_event_logger_initialized():
            get_event_logger().log(
                CellReconfigureEvent,
                dict(
                    rollout_id=rollout_id,
                    quorum_id=self._indep_dp_quorum_id,
                    src_cell_index=src_cell_index,
                    healed_cell_indices=healed_cell_indices,
                    alive_cell_indices_after=alive_cell_indices_after,
                ),
            )

    def _compute_indep_dp_info(self, cell_index: int, alive_cell_indices: list[int]) -> IndepDPInfo:
        # [FUNCTION DOC] HELPER: Build the independent-DP info for one cell: its index, the current
        # known cell count, its rank (position) within the given alive set (value error if it is not
        # in it), the alive set size, the current quorum id and the alive set itself.
        return IndepDPInfo(
            cell_index=cell_index,
            num_cells=len(self._cells),
            alive_rank=alive_cell_indices.index(cell_index),
            alive_size=len(alive_cell_indices),
            quorum_id=self._indep_dp_quorum_id,
            alive_cell_indices=alive_cell_indices,
        )

    # ------------------------ misc states and utils ------------------------

    @property
    def num_cells(self) -> int:
        # [FUNCTION DOC] Property: number of cells currently known (any state).
        return len(self._cells)


def _first_exception(results) -> BaseException | None:
    # [FUNCTION DOC] HELPER: The first element of the results that is an exception object, or none.
    return next((result for result in results if isinstance(result, BaseException)), None)
