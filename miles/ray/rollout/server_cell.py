# [MODULE DOC] One reconcile unit of the rollout side: a server cell wraps a single SGLang engine
# (possibly multi-node) with its lifecycle state machine (uninitialized -> initializing -> pending
# weights -> serving -> disposed), router registration, health checking, env reporting, offload /
# onload, abort and weight checks. Also holds the cell metadata model and helpers that build the
# health checker and the user-visible cell status. Constants: shutdown timeout 30 s, initializing
# (startup) deadline 1800 s, abort request timeout 30 s.
import asyncio
import dataclasses
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient, probe_server_healthy
from miles.backends.sglang_utils.sglang_engine import build_server_url
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient, use_legacy_router_api
from miles.ray.rollout.cell_state import (
    CellAddrInfo,
    CellState,
    StateDisposed,
    StateInitializing,
    StatePendingWeights,
    StateServing,
    StateUninitialized,
)
from miles.ray.rollout.engine_env_reporter import EngineEnvReporter
from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus, TriState
from miles.utils.ft_utils.health_checker import (
    ActiveAndEpoch,
    BaseHealthChecker,
    NoopHealthChecker,
    SimpleHealthChecker,
    SimpleHealthCheckerConfig,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.launch_gate import GATE_PORT_NAME, activate_launch_gate
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30
INITIALIZING_TIMEOUT_SECONDS = 1800.0
ABORT_REQUEST_TIMEOUT_SECONDS = 30.0


class ServerCellMetadata(FrozenStrictBaseModel):
    # [CLASS DOC] Frozen identity and configuration of a cell: served model id; worker type (regular,
    # prefill or decode); cell id; GPUs per engine; first GPU index of the engine (offset);
    # optional engine API key; name of the cell's primary worker used for address lookup; whether
    # the engine must be offloaded when colocated with training; whether its weights are updated
    # by training; and a hash of the cell's worker set (changes when workers are replaced).
    model_id: str
    worker_type: Literal["regular", "prefill", "decode"]
    cell_id: str
    num_gpus_per_engine: int
    gpu_offset: int
    sglang_api_key: str | None
    worker_name: str
    needs_offload: bool
    update_weights: bool
    workers_hash: str


@dataclass
class ServerCell:
    # [CLASS DOC] Mutable cell. Fields: args; metadata; router API client used to register or
    # unregister the engine; worker provider used to look up addresses; a callable giving the
    # controller-level health activeness and epoch (default always active at epoch zero); the health
    # checker and env reporter (created in post-init, not constructor arguments); the current
    # lifecycle state (default uninitialized). Transitions go through the change-state helper which
    # asserts the previous state class. Must be disposed before garbage collection.
    args: Any
    meta: ServerCellMetadata
    router_api_client: SGLangRouterApiClient
    provider: BaseWorkerProvider
    health_checker_activeness: Callable[[], ActiveAndEpoch] = lambda: ActiveAndEpoch(active=True, epoch=0)
    _health_checker: BaseHealthChecker = dataclasses.field(init=False)
    _env_reporter: EngineEnvReporter = dataclasses.field(init=False)
    _state: CellState = dataclasses.field(default_factory=StateUninitialized)

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Create the env reporter with the configured report interval, create the
        # health checker named after the cell (its client getter builds the API client lazily from the
        # current state, and its activeness getter combines state and controller activeness), and
        # start it immediately (background task). Requires a running event loop for real checkers.
        self._env_reporter = EngineEnvReporter(interval_seconds=self.args.env_report_interval_seconds)
        self._health_checker = create_rollout_cell_health_checker(
            args=self.args,
            name=f"rollout-cell-{self.meta.cell_id}",
            get_api_client=lambda: self.api_client,
            get_activeness=self._get_health_checker_active_and_epoch,
        )
        self._health_checker.start()

    def _get_health_checker_active_and_epoch(self) -> ActiveAndEpoch:
        # [FUNCTION DOC] HELPER: Health checking is active only when the cell is pending weights or
        # serving AND the controller says active; the epoch is passed through from the controller.
        controller_active_and_epoch = self.health_checker_activeness()
        cell_active = isinstance(self._state, (StatePendingWeights, StateServing))
        return ActiveAndEpoch(
            active=cell_active and controller_active_and_epoch.active, epoch=controller_active_and_epoch.epoch
        )

    def __del__(self) -> None:
        # [FUNCTION DOC] HELPER: Finalizer guard: asserts the cell reached the disposed state before
        # being garbage collected (otherwise its health checker task would leak). An assertion in a
        # finalizer is only reported, not propagated.
        assert isinstance(self._state, StateDisposed), (
            f"ServerCell {self.meta.cell_id} was garbage collected without dispose() ({self._state=}); "
            "every cell must be disposed so its health checker task is stopped"
        )

    def cell_status(self) -> CellStatus:
        # [FUNCTION DOC] Build the externally visible status from the state. Uninitialized or
        # initializing: a pending status (flagging startup deadline exceeded only when initializing for
        # at least 1800 s). Pending weights or serving: phase Running with conditions allocated true,
        # the health checker's condition, and serving true/false. Disposed: phase Suspended with
        # allocated false. Every status carries the cell's workers hash. Unknown state raises not
        # implemented error.
        match self._state:
            case StateUninitialized() | StateInitializing():
                return compute_pending_rollout_cell_status(
                    past_startup_deadline=self.is_initializing_past_deadline, workers_hash=self.meta.workers_hash
                )

            case StatePendingWeights() | StateServing():
                return CellStatus(
                    phase="Running",
                    conditions=[
                        CellCondition.allocated(TriState.TRUE),
                        CellCondition.from_health_checker_status(self._health_checker.status),
                        CellCondition.serving(TriState.TRUE if self.is_serving else TriState.FALSE),
                    ],
                    workers_hash=self.meta.workers_hash,
                )

            case StateDisposed():
                return CellStatus(
                    phase="Suspended",
                    conditions=[CellCondition.allocated(TriState.FALSE)],
                    workers_hash=self.meta.workers_hash,
                )

            case _:
                raise NotImplementedError(f"Unknown state: {self._state}")

    @property
    def is_uninitialized(self) -> bool:
        # [FUNCTION DOC] Property: whether the state is uninitialized.
        return isinstance(self._state, StateUninitialized)

    @property
    def is_initializing(self) -> bool:
        # [FUNCTION DOC] Property: whether the state is initializing (engine launched, not yet healthy).
        return isinstance(self._state, StateInitializing)

    @property
    def is_pending_weights_or_serving(self) -> bool:
        # [FUNCTION DOC] Property: whether the engine is up (pending weights or serving), i.e. has
        # addresses and can be sent requests.
        return isinstance(self._state, (StatePendingWeights, StateServing))

    @property
    def is_pending_weights(self) -> bool:
        # [FUNCTION DOC] Property: whether the state is pending weights.
        return isinstance(self._state, StatePendingWeights)

    @property
    def is_serving(self) -> bool:
        # [FUNCTION DOC] Property: whether the state is serving (registered with the router).
        return isinstance(self._state, StateServing)

    @property
    def is_initializing_past_deadline(self) -> bool:
        # [FUNCTION DOC] Property: true if initializing and at least 1800 seconds of monotonic time
        # have elapsed since the initializing state began.
        return self.is_initializing and time.monotonic() - self._state.start_time >= INITIALIZING_TIMEOUT_SECONDS

    @property
    def addr_info(self) -> CellAddrInfo:
        # [FUNCTION DOC] Property: the address info of the current state; assertion error if the state
        # is uninitialized or disposed (no addresses).
        assert isinstance(self._state, (StateInitializing, StatePendingWeights, StateServing))
        return self._state.addr_info

    @property
    def server_url(self) -> str:
        # [FUNCTION DOC] Property: the engine's HTTP server URL (requires an addressable state).
        return self.addr_info.server_url

    @property
    def api_client(self) -> SGLangApiClient:
        # [FUNCTION DOC] Property: a fresh engine API client for the server URL with the metadata API
        # key; a new object per access (requires an addressable state).
        return SGLangApiClient(server_url=self.server_url, api_key=self.meta.sglang_api_key)

    async def init(self) -> None:
        # [FUNCTION DOC] Move from uninitialized to initializing. Asserts the current state is
        # uninitialized. Looks up the engine's addresses from the provider; if the engine has a launch
        # gate URL, activates the gate (letting the launch proceed); then switches to the initializing
        # state recording the addresses and the monotonic start time.
        assert isinstance(self._state, StateUninitialized), f"{self._state=}"
        addr_info = await self._compute_addr_info()
        if (gate_url := addr_info.gate_url) is not None:
            await activate_launch_gate(gate_url=gate_url)
        self._change_state(
            "init", StateUninitialized, StateInitializing(addr_info=addr_info, start_time=time.monotonic())
        )

    async def tick(self) -> None:
        # [FUNCTION DOC] Periodic progress step called by the controller. If initializing, runs the
        # initializing tick (which may advance the state). Then reports the engine env if due.
        if isinstance(self._state, StateInitializing):
            await self._tick_when_initializing()
        await self._report_env_if_due()

    async def _report_env_if_due(self) -> None:
        # [FUNCTION DOC] HELPER: Only when the engine is up (pending weights or serving), ask the env
        # reporter to report this cell's server info if its schedule says it is due.
        if not self.is_pending_weights_or_serving:
            return
        await self._env_reporter.report_if_due(
            cell_id=self.meta.cell_id, server_url=self.server_url, api_client=self.api_client
        )

    async def _tick_when_initializing(self) -> None:
        # [FUNCTION DOC] HELPER: Try to finish startup. Returns doing nothing if the engine's health
        # probe fails. Once healthy: (1) if weight-update equality checking is enabled and this model
        # is updatable, take a weight snapshot via the check action "snapshot"; (2) if the cell needs
        # offload, release all GPU memory occupation then resume only the weights memory (so the
        # engine sits with just weights loaded); (3) decide whether to serve without a weight update:
        # true when the model is frozen or in debug rollout-only mode; (4) if it will receive weights
        # and equality checking is on, reset tensors via the check action "reset_tensors" with the
        # configured skip list (so the later update is verified from scratch); (5) if serving without
        # weight update, register with the router now; (6) switch state to pending weights; (7) if
        # serving without weight update, immediately switch to serving. KNOWN ISSUE: the API client
        # created for the offload step omits the metadata API key, unlike the api client property.
        addr_info = self._state.addr_info
        if not await probe_server_healthy(server_url=addr_info.server_url, api_key=self.meta.sglang_api_key):
            return

        if self.args.check_weight_update_equal and self.meta.update_weights:
            await self.check_weights(action="snapshot", allow_quant_error=False, selector="all", skip_list=None)

        if self.meta.needs_offload:
            api_client = SGLangApiClient(server_url=addr_info.server_url)
            await api_client.release_memory_occupation()
            await api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])

        serve_without_weight_update: bool = not self.meta.update_weights or self.args.debug_rollout_only
        if not serve_without_weight_update and self.args.check_weight_update_equal:
            await self.check_weights(
                action="reset_tensors",
                allow_quant_error=False,
                selector="all",
                skip_list=self.args.check_weight_update_skip_list,
            )

        if serve_without_weight_update:
            await self._register_with_router(addr_info=addr_info)

        self._change_state("mark_pending_weights", StateInitializing, StatePendingWeights(addr_info=addr_info))

        if serve_without_weight_update:
            self._mark_serving()

    async def mark_weights_ready(self) -> None:
        # [FUNCTION DOC] Called after a weight update reached this cell: asserts the state is pending
        # weights, registers the engine with the router, and moves to serving.
        assert isinstance(self._state, StatePendingWeights), f"{self._state=}"
        await self._register_with_router(addr_info=self._state.addr_info)
        self._mark_serving()

    async def _register_with_router(self, addr_info: CellAddrInfo) -> None:
        # [FUNCTION DOC] HELPER: Add the engine's URL to the router as a worker of this cell's type
        # (regular, prefill or decode), choosing the legacy or current router API per args, and
        # passing the prefill/decode bootstrap port (may be none).
        await self.router_api_client.add_worker(
            worker_url=addr_info.server_url,
            worker_type=self.meta.worker_type,
            use_legacy_api=use_legacy_router_api(self.args),
            bootstrap_port=addr_info.bootstrap_port,
        )

    async def dispose(self) -> None:
        # [FUNCTION DOC] Tear the cell down. Stops the health checker first. If serving, unregisters
        # from the router (failures only warn). Other known states need no cleanup (the engine process
        # itself is managed by the provider, not here); an unrecognized state raises value error.
        # Then moves to disposed from any state (disposing twice is allowed).
        self._health_checker.stop()

        match self._state:
            case StateServing():
                await self._unregister_from_router()
            case StateUninitialized() | StateInitializing() | StatePendingWeights() | StateDisposed():
                pass
            case _:
                raise ValueError(f"{self._state=}")

        self._change_state(
            "dispose",
            (StateUninitialized, StateInitializing, StatePendingWeights, StateServing, StateDisposed),
            StateDisposed(),
        )

    async def _unregister_from_router(self) -> None:
        # [FUNCTION DOC] HELPER: Remove the engine from the router with a 30 s timeout. Any exception
        # (including timeout) is logged as a warning and swallowed so teardown continues.
        try:
            await asyncio.wait_for(
                self.router_api_client.remove_worker(
                    worker_url=self.server_url,
                    use_legacy_api=use_legacy_router_api(self.args),
                ),
                timeout=SHUTDOWN_TIMEOUT,
            )
        except Exception as e:
            logger.warning(f"Unregistering cell {self.meta.cell_id} from the router failed, tearing down anyway ({e})")

    async def _compute_addr_info(self) -> CellAddrInfo:
        # [FUNCTION DOC] HELPER: Ask the provider for the named addresses of the cell's worker. The
        # primary entry (required: key error otherwise) gives the server URL; an optional
        # disaggregation bootstrap entry gives the bootstrap port; an optional launch-gate entry gives
        # the gate URL. Absent optional entries become none.
        master_addrs = await self.provider.get_addrs(worker_name=self.meta.worker_name)
        primary = master_addrs["primary"]
        gate = master_addrs.get(GATE_PORT_NAME)
        return CellAddrInfo(
            server_url=build_server_url(host=primary.host, port=primary.port),
            bootstrap_port=x.port if (x := master_addrs.get("disaggregation_bootstrap")) else None,
            gate_url=build_server_url(host=gate.host, port=gate.port) if gate else None,
        )

    def _mark_serving(self) -> None:
        # [FUNCTION DOC] HELPER: Transition pending weights -> serving, keeping the address info.
        self._change_state("mark_serving", StatePendingWeights, StateServing(addr_info=self.addr_info))

    # TODO: unify w/ trainer `change_state`
    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[CellState] | tuple[type[CellState], ...],
        new_state: CellState,
    ) -> None:
        # [FUNCTION DOC] HELPER: Log the start of a named transition, assert the current state is an
        # instance of the allowed old state class(es) (assertion error otherwise), install the new
        # state, and log the end. Synchronous, so the swap is atomic with respect to the event loop.
        logger.info(f"Cell {self.meta.cell_id} {debug_name} start old={self._state}")
        assert isinstance(self._state, old_state_cls), f"{self._state=}"
        self._state = new_state
        logger.info(f"Cell {self.meta.cell_id} {debug_name} end new={self._state}")

    async def offload(self, tags: list[str] | None):
        # [FUNCTION DOC] Ask the engine to release GPU memory for the given memory tags (all kinds if
        # none). Returns the API client's result. Requires an addressable state.
        return await self.api_client.release_memory_occupation(tags=tags)

    async def onload(self, tags: list[str] | None):
        # [FUNCTION DOC] Ask the engine to resume GPU memory occupation for the given tags (all kinds
        # if none). Returns the API client's result.
        return await self.api_client.resume_memory_occupation(tags=tags)

    async def abort_all(self):
        # [FUNCTION DOC] Ask the engine to abort all in-flight requests, with a 30 s request timeout.
        return await self.api_client.abort_all_requests(timeout=ABORT_REQUEST_TIMEOUT_SECONDS)

    async def check_weights(self, action: str, allow_quant_error: bool, selector: str, skip_list: list[str] | None):
        # [FUNCTION DOC] Forward a weight check to the engine: the action (such as snapshot, reset
        # tensors, compare or checksum), whether quantization error is tolerated, the tensor selector
        # and an optional list of names to skip. Returns the engine's result.
        return await self.api_client.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )


def compute_nodes_per_engine(*, num_gpus_per_engine: int, num_gpus_per_node: int) -> int:
    # [FUNCTION DOC] HELPER: Nodes one engine spans: GPUs per engine divided by GPUs per node using
    # integer division, but at least 1. Fractional leftovers are floored (not rounded up).
    return max(1, num_gpus_per_engine // num_gpus_per_node)


def create_rollout_cell_health_checker(
    *,
    args: Any,
    name: str,
    get_api_client: Callable[[], SGLangApiClient],
    get_activeness: Callable[[], ActiveAndEpoch],
) -> BaseHealthChecker:
    # [FUNCTION DOC] Factory for a cell's health checker. If rollout is not among the fault-tolerance
    # components in args, returns a checker that does nothing. Otherwise reads the checker config
    # from args options prefixed rollout_health_check and returns a simple periodic checker with the
    # given name whose probe asks the engine (via the getter) for a health-generate with the
    # config's timeout, and whose activeness comes from the getter.
    if "rollout" not in args.ft_components:
        return NoopHealthChecker()

    config = SimpleHealthCheckerConfig.from_args(args, prefix="rollout_health_check")

    async def _check() -> None:
        # [FUNCTION DOC] HELPER: One probe: a health generate call on a fresh client with the timeout.
        await get_api_client().health_generate(timeout=config.timeout)

    return SimpleHealthChecker(name=name, check_fn=_check, get_activeness=get_activeness, config=config)


def compute_pending_rollout_cell_status(*, workers_hash: str, past_startup_deadline: bool = False) -> CellStatus:
    # [FUNCTION DOC] Status for a cell that is not yet up: phase Pending, condition allocated true,
    # and, when the startup deadline has passed, an additional condition healthy false with reason
    # StartupDeadlineExceeded. Carries the given workers hash. Public so that other components
    # (e.g. providers) can report the same pending status.
    return CellStatus(
        phase="Pending",
        conditions=[
            CellCondition.allocated(TriState.TRUE),
            *(
                [CellCondition.healthy(TriState.FALSE, reason="StartupDeadlineExceeded")]
                if past_startup_deadline
                else []
            ),
        ],
        workers_hash=workers_hash,
    )
