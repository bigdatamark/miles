# [MODULE DOC] The rollout server abstraction: one served model behind a single router, holding a
# mapping from cell id to server cell. Provides a factory that builds one server per model from the
# sglang config, a rule for how many cells a server expects at startup, and cell management
# (add, remove, dispose, offload/onload, abort all, weight check), all guarded by the shared
# context lock. Constants: expected cells default 1; wait-for-cells retry delay 1 s growing to 5 s;
# abort-all timeout 60 s per cell.
import asyncio
import dataclasses
import logging
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils import async_utils
from miles.utils.context_lock import ContextLock, enforce_lock_discipline, lock_exempt, requires_lock
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)

_DEFAULT_INIT_EXPECTED_NUM_CELLS = 1

WAIT_CELLS_INITIAL_DELAY_SECONDS = 1.0
WAIT_CELLS_MAX_DELAY_SECONDS = 5.0

ABORT_ALL_TIMEOUT_SECONDS = 60.0


async def create_rollout_servers(
    args,
    context_lock: ContextLock,
    *,
    engine_provider: BaseWorkerProvider,
    router_addrs: dict[str, HostAndPort],
) -> dict[str, "RolloutServer"]:
    # [FUNCTION DOC] Build an empty server for each model in the resolved sglang config, keyed by
    # model name. Each server takes the shared args, context lock and engine provider, its model's
    # router host and port from the router address mapping (key error if a model has no router),
    # the model's name and update-weights flag, and its expected initial cell count from the
    # expected-cells rule. No cells are added here (they arrive through reconcile).
    """Create rollout servers: one per model, each with its own router."""
    config = resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}

    for model_cfg in config.models:
        router_addr = router_addrs[model_cfg.name]

        servers[model_cfg.name] = RolloutServer(
            server_cells={},
            args=args,
            context_lock=context_lock,
            engine_provider=engine_provider,
            router_ip=router_addr.host,
            router_port=router_addr.port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
            init_expected_num_cells=_compute_init_expected_num_cells(args, engine_provider, model_cfg=model_cfg),
        )

    return servers


def _compute_init_expected_num_cells(args, engine_provider: BaseWorkerProvider, *, model_cfg) -> int:
    # [FUNCTION DOC] HELPER: Number of cells to wait for before the server counts as initialized.
    # Priority: (1) an explicit launch arg value if set; (2) the engine provider's own answer for the
    # model name if it gives one; (3) if the deploy component deploys its own inference engines, the
    # model's configured number of server cells; (4) otherwise the default of 1.
    if (declared := args.init_expected_num_cells) is not None:
        return declared
    if (answered := engine_provider.expected_num_cells(group_id=model_cfg.name)) is not None:
        return answered
    if DeployComponent(args.deploy_component).deploys_own_inference_engines():
        return model_cfg.num_server_cells
    return _DEFAULT_INIT_EXPECTED_NUM_CELLS


@dataclasses.dataclass
@enforce_lock_discipline
class RolloutServer:
    # [CLASS DOC] Mutable dataclass (lock discipline enforced: most methods require the shared
    # context lock held). Fields: mapping of cell id to server cell; args; the shared context lock;
    # the engine provider; router host and port (may be unset); model name (default "default"); a
    # flag whether this model's weights are updated (frozen models set it false); a health checker
    # activeness tracker (initially active) read by cells; and the expected initial cell count
    # (default 0). Cell order for the per-engine accessors is by GPU offset ascending.
    """A model served behind a shared router, as a dict of cell id -> cell.

    Each RolloutServer represents one model deployed behind a single router.
    """

    server_cells: dict[str, ServerCell]
    args: Any
    context_lock: ContextLock
    engine_provider: BaseWorkerProvider
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True
    health_checker_activeness: ActivenessTracker = dataclasses.field(
        default_factory=lambda: ActivenessTracker(active=True)
    )
    init_expected_num_cells: int = 0

    @property
    @requires_lock
    def api_clients(self) -> list[SGLangApiClient]:
        # [FUNCTION DOC] Property: the engine API client of every cell, ordered by GPU offset.
        """One client per cell, talking to its primary (node-0) engine."""
        return [cell.api_client for cell in self._cells_by_gpu_offset()]

    @property
    @requires_lock
    def engine_gpu_counts(self) -> list[int]:
        # [FUNCTION DOC] Property: GPUs per engine of every cell, in the same GPU-offset order as the
        # API clients. KNOWN ISSUE: docstring mentions a parallel list named engines, which does not
        # exist; it is parallel to the API clients list. It covers all cells, not only node-0 ones.
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [cell.meta.num_gpus_per_engine for cell in self._cells_by_gpu_offset()]

    @property
    @requires_lock
    def engine_gpu_offsets(self) -> list[int]:
        # [FUNCTION DOC] Property: first GPU index of every cell's engine, ascending order (the
        # sort key itself), parallel to the API clients and GPU counts.
        return [cell.meta.gpu_offset for cell in self._cells_by_gpu_offset()]

    @requires_lock
    def _cells_by_gpu_offset(self) -> list[ServerCell]:
        # [FUNCTION DOC] HELPER: All cells sorted ascending by their metadata GPU offset (stable for
        # ties).
        return sorted(self.server_cells.values(), key=lambda cell: cell.meta.gpu_offset)

    @requires_lock
    async def add_cell(self, cell_meta: ServerCellMetadata):
        # [FUNCTION DOC] Register a new cell from its metadata (assertion error if the id already
        # exists). Creates a server cell wired with args, the server's router API client, the
        # metadata, the engine provider and the health-checker activeness getter, and stores it. Unless
        # running colocated with a cell that needs offload (such cells start later, offloaded), it
        # initializes the cell immediately; if initialization fails, the cell is removed from the
        # mapping, disposed, and the error re-raised.
        cell_id = cell_meta.cell_id
        assert cell_id not in self.server_cells
        cell = ServerCell(
            args=self.args,
            router_api_client=self._router_api_client,
            meta=cell_meta,
            provider=self.engine_provider,
            health_checker_activeness=self.health_checker_activeness.get,
        )
        self.server_cells[cell_id] = cell
        if not (self.args.colocate and cell_meta.needs_offload):
            try:
                await cell.init()
            except Exception:
                del self.server_cells[cell_id]
                await cell.dispose()
                raise

    @requires_lock
    async def remove_cell(self, cell_id: str):
        # [FUNCTION DOC] Dispose the cell (stops its engine; key error if unknown) and then delete it
        # from the mapping. If disposal raises, the cell stays in the mapping.
        logger.info(f"Killing server {cell_id=}...")
        await self.server_cells[cell_id].dispose()
        del self.server_cells[cell_id]

    @requires_lock
    async def dispose(self) -> None:
        # [FUNCTION DOC] Remove every cell sequentially (over a snapshot of the ids).
        for cell_id in list(self.server_cells.keys()):
            await self.remove_cell(cell_id)

    @requires_lock
    async def offload(self, tags: list[str] | None = None):
        # [FUNCTION DOC] Concurrently offload the given memory tags on every addressable cell (pending
        # weights or serving) whose metadata says it needs offload. Returns the list of results.
        return await asyncio.gather(
            *[cell.offload(tags=tags) for cell in self._addressable_cells() if cell.meta.needs_offload]
        )

    @requires_lock
    async def onload(self, tags: list[str] | None = None):
        # [FUNCTION DOC] Concurrently onload the given memory tags on every addressable cell that
        # needs offload. Returns the list of results. Note cells not yet initialized (colocated,
        # offloaded) are not addressable and are initialized elsewhere.
        return await asyncio.gather(
            *[cell.onload(tags=tags) for cell in self._addressable_cells() if cell.meta.needs_offload]
        )

    @requires_lock
    async def abort_all(self) -> None:
        # [FUNCTION DOC] Abort all running generations on every addressable cell concurrently, each
        # limited to 60 s. All outcomes are awaited and the first failure is raised with a message
        # naming the cell and model, warning that a previous orchestration script's request may still
        # be running there.
        cells = self._addressable_cells()
        await async_utils.gather_and_raise_first(
            [asyncio.wait_for(cell.abort_all(), timeout=ABORT_ALL_TIMEOUT_SECONDS) for cell in cells],
            describe_failure=lambda index: (
                f"Aborting the generations of cell {cells[index].meta.cell_id} of {self.model_name} failed, so a "
                f"request of the previous orchestration script may still be running on it"
            ),
        )

    @requires_lock
    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        # [FUNCTION DOC] Concurrently run the weight check on every addressable cell, forwarding the
        # action, quantization-error tolerance, selector and skip list. Returns the list of per-cell
        # results in cell iteration order.
        return await asyncio.gather(
            *[
                cell.check_weights(
                    action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
                )
                for cell in self._addressable_cells()
            ]
        )

    @requires_lock
    def _addressable_cells(self) -> list[ServerCell]:
        # [FUNCTION DOC] HELPER: Cells that are pending weights or serving (i.e. have a live engine).
        return [cell for cell in self.server_cells.values() if cell.is_pending_weights_or_serving]

    @lock_exempt
    async def wait_init_expected_num_cells(self, timeout: float = 3600):
        # [FUNCTION DOC] Not lock-guarded (so it can run while reconcile adds cells). Retries a check
        # until the number of startable cells reaches the expected count or the timeout (seconds,
        # default 3600) expires, with backoff starting at 1 s and capped at 5 s, retrying on any
        # exception. The deadline helper raises when the time is exhausted.
        async def _check(remaining_seconds: float) -> None:
            # [FUNCTION DOC] HELPER: One attempt: raises a generic exception stating ready/expected
            # counts if fewer than expected cells are startable; otherwise returns. The remaining
            # time argument is ignored.
            count = self._count_startable_cells()
            if count < self.init_expected_num_cells:
                raise Exception(f"Only {count}/{self.init_expected_num_cells} cells of {self.model_name} are ready")

        await retry_until_deadline(
            _check,
            total_seconds=timeout,
            retry_on=Exception,
            initial_delay=WAIT_CELLS_INITIAL_DELAY_SECONDS,
            max_delay=WAIT_CELLS_MAX_DELAY_SECONDS,
            log_fields=dict(op="wait_init_expected_num_cells", model_name=self.model_name),
        )

    @lock_exempt
    def _count_startable_cells(self) -> int:
        # [FUNCTION DOC] HELPER: Count cells that are ready for the server to be considered up: cells
        # that are pending weights or serving, plus, when colocated, cells that need offload (which
        # are intentionally not started yet).
        return sum(
            1
            for cell in self.server_cells.values()
            if (self.args.colocate and cell.meta.needs_offload) or cell.is_pending_weights_or_serving
        )

    @property
    @requires_lock
    def _router_api_client(self) -> SGLangRouterApiClient:
        # [FUNCTION DOC] HELPER: Property building a fresh router API client for the http URL made of
        # the server's router ip and port on every access.
        return SGLangRouterApiClient(router_url=f"http://{self.router_ip}:{self.router_port}")
