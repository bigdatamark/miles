# [MODULE DOC]
# Ray backend worker manager: a single named Ray actor that owns every pool, cell and worker actor
# of a run. It creates worker actors on placement-group slots, allocates their network ports,
# starts them (either as a command subprocess or as a served Python worker, reachable over the
# rpc backend or plain Ray calls), watches their liveness, and stops or restarts whole cells.
# Structure: the manager holds pool managers, each holding cell managers, each holding actor
# managers (one per worker). Cell membership changes (start, stop, liveness teardown) are
# serialized by one lock. A cell's generation counter increases on every launch and identifies a
# particular incarnation of its workers.
from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.function_registry import load_function
from miles.utils.http_utils import wrap_ipv6
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import NodeProbeMixin
from miles.utils.ray_utils import compute_ray_pin_head_options
from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.backend_capability.base import BackendCapability, DeferredBackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.rpc.common.metadata import declared_concurrency_groups
from miles.utils.workers.serving.serve_actor import ServeActor
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    BaseWorkerSpec,
    CommandWorkerSpec,
    HostAndPort,
    LaunchCommandContext,
    NamedHostAndPorts,
    ServeWorkerSpec,
    WorkerCtorContext,
    WorkerLaunchContext,
    WorkerMetaContext,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from miles.ray.placement_group import PlacementGroupInfo

# TODO: unique name, maybe with args.run_uuid
_ACTOR_NAME = "ray_worker_manager"

_LIVENESS_SCAN_INTERVAL_SECONDS = 10.0


class RayWorkerManager:
    # [CLASS DOC]
    # The manager actor body (it is wrapped into a Ray actor by the launch function). Lifecycle:
    # construct, then call the init method once, after which cells are started automatically.
    # State: the port allocator, the communication backend, the placement groups by name, the
    # pool managers keyed by pool name (spec name) and the membership lock.
    def __init__(self):
        # [FUNCTION DOC]
        # Create the port allocator only; the rest of the state is set by the init method.
        self.port_allocator = PortAllocator()

    @staticmethod
    def launch(
        args, specs: list[BaseWorkerSpec], pgs: dict[str, PlacementGroupInfo], *, comm_backend: WorkerCommBackend
    ):
        # [FUNCTION DOC]
        # Static method run by the driver. Creates the manager as a named detached-by-name Ray actor
        # (the fixed actor name, so later lookups can find it), then blocks until its init call
        # finishes (which starts every cell). Returns the actor handle. Raises whatever init raises
        # (for example a failed cell start).
        obj = ray.remote(RayWorkerManager).options(name=_ACTOR_NAME).remote()
        ray.get(obj.init.remote(args, specs, pgs, comm_backend=comm_backend))
        return obj

    @staticmethod
    def get_handle() -> ray.actor.ActorHandle:
        # [FUNCTION DOC]
        # Static method. Look up the manager actor by its fixed name in the Ray cluster and return
        # its handle; Ray raises a value error if no such actor exists.
        return ray.get_actor(_ACTOR_NAME)

    async def init(
        self, args, specs: list[BaseWorkerSpec], pgs: dict[str, PlacementGroupInfo], *, comm_backend: WorkerCommBackend
    ):
        # [FUNCTION DOC]
        # Actor-side setup. Configures logging with a worker-manager process identity, records the
        # communication backend and the placement groups (by name), builds one pool manager per
        # spec keyed by the spec name (asserting that spec names are unique), creates the
        # membership lock, and then starts every cell of every pool. Failure to start any cell
        # propagates.
        configure_logger(args, source=SimpleProcessIdentity(component="worker_manager"))

        self.comm_backend = comm_backend
        self.pgs = pgs
        self._pools = {spec.name: _PoolManager.initial(spec, self) for spec in specs}
        assert len(self._pools) == len(specs)
        self._membership_lock = asyncio.Lock()

        await self.start_cells([c.cell_id for c in self._all_cells()])

    async def start_cells(self, cell_ids: list[str]) -> None:
        # [FUNCTION DOC]
        # Start the given cells under the membership lock, skipping cells that are already running
        # (an unknown cell id fails an assertion). In three phases, each running concurrently over
        # all selected cells and failing fast: launch the worker actors, allocate ports, run the
        # post-setup step (which actually runs the worker). If any phase fails, logs the error,
        # stops all selected cells (ignoring errors while stopping) so nothing is half-started, and
        # re-raises the original error.
        async with self._membership_lock:
            cells = [cell for cell_id in cell_ids if (cell := self._find_cell(cell_id)).actors is None]
            try:
                await _gather_or_raise([c.launch_actors() for c in cells])
                await _gather_or_raise([c.alloc_ports() for c in cells])
                await _gather_or_raise([c.post_setup() for c in cells])
            except Exception:
                logger.error(f"Starting cells {[c.cell_id for c in cells]} failed, rolling back", exc_info=True)
                await asyncio.gather(*[c.stop() for c in cells], return_exceptions=True)
                raise

    async def stop_cells(self, cell_ids: list[str]) -> None:
        # [FUNCTION DOC]
        # Stop the given cells concurrently under the membership lock. Stopping a cell that is
        # already stopped is a no-op; an unknown cell id fails an assertion.
        async with self._membership_lock:
            await asyncio.gather(*[self._find_cell(cell_id).stop() for cell_id in cell_ids])

    async def shutdown(self) -> None:
        # [FUNCTION DOC]
        # Stop every cell of every pool concurrently under the membership lock.
        async with self._membership_lock:
            await asyncio.gather(*[cell.stop() for cell in self._all_cells()])

    def inject_fault(self, cell_id: str, *, mode: str, worker_in_cell_index: int) -> None:
        # [FUNCTION DOC]
        # Ask one worker actor of a running cell to inject a fault of the given mode (a string).
        # Raises a runtime error if the cell is not running and an index error if the worker index
        # is out of range. Fire-and-forget: the remote call is issued without waiting for it or its
        # outcome, so the call returns before the fault happens and errors in the worker are not
        # reported here.
        cell = self._find_cell(cell_id)
        if not cell.alive:
            raise RuntimeError(f"Cell {cell_id} is not alive, cannot inject fault")
        if not 0 <= worker_in_cell_index < len(cell.actors):
            raise IndexError(
                f"worker_in_cell_index {worker_in_cell_index} out of range for cell {cell_id} "
                f"(has {len(cell.actors)} workers)"
            )
        cell.actors[worker_in_cell_index].actor_handle.inject_fault.remote(mode)

    def get_worker_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC]
        # Return the named endpoints of one worker in a running cell (found by worker name; fails
        # an assertion if there is not exactly one match). Asserts the worker has already been
        # given its ports.
        addrs = self._find_actor(worker_name).self_addrs
        assert addrs is not None, (
            f"{worker_name} has not been given its ports yet; a caller reading them now would take the "
            f"endpoints it cannot find for endpoints the worker does not have"
        )
        return addrs

    def get_addrs(self) -> dict[str, list[NamedHostAndPorts]]:
        # [FUNCTION DOC]
        # Return, for each pool name, the list of endpoint mappings of all workers in running
        # cells, in cell then worker order. A worker that has no ports yet contributes an empty
        # mapping; stopped cells contribute nothing.
        return {
            name: [a.self_addrs or {} for c in g.cells if c.alive for a in c.actors] for name, g in self._pools.items()
        }

    def get_worker_infos(self, cell_id: str) -> list[WorkerInfo]:
        # [FUNCTION DOC]
        # Return the worker info records (name, generation, endpoints, gpu ids, worker class when
        # served over rpc) of all workers of the cell; an empty list if the cell is stopped.
        # Asserts the cell id is known.
        cell = self._find_cell(cell_id)
        return [self._compute_worker_info(actor) for actor in (cell.actors if cell.actors is not None else [])]

    def get_cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        # [FUNCTION DOC]
        # Return a mapping from cell id to cell info for every cell of the requested pools.
        # Asserts all pool ids are known. Does not take the membership lock.
        # TODO: about `get_worker_infos` (which is only used by dashboard)
        unknown = set(pool_ids) - set(self._pools)
        assert not unknown, f"{unknown=} {sorted(self._pools)=}"
        infos = [c.get_info() for name in pool_ids for c in self._pools[name].cells]
        return {info.cell_id: info for info in infos}

    def get_actor_handle(self, worker_name: str, *, expected_generation: int) -> ray.actor.ActorHandle:
        # [FUNCTION DOC]
        # Return the Ray actor handle of the named worker, but only if its cell is still at the
        # expected generation; otherwise fails an assertion telling the caller to refetch worker
        # infos. This prevents using a handle that was described for a previous incarnation.
        actor = self._find_actor(worker_name)
        assert actor.generation == expected_generation, (
            f"{worker_name} is now generation {actor.generation}, not the {expected_generation} it was described as; "
            f"ask for its worker infos again"
        )
        return actor.actor_handle

    def _compute_worker_info(self, actor: _BaseActorManager) -> WorkerInfo:
        # [FUNCTION DOC] HELPER: Build the worker info for an actor: name, generation, endpoints
        # (empty if none yet), gpu ids, and the worker class path only when the worker is a served
        # worker and the communication backend is rpc (otherwise absent).
        served_over_rpc = isinstance(actor.spec, ServeWorkerSpec) and self.comm_backend == WorkerCommBackend.RPC
        return WorkerInfo(
            name=actor.name,
            generation=actor.generation,
            self_addrs=actor.self_addrs or {},
            gpu_ids=actor.gpu_ids,
            worker_class=actor.spec.worker_class if served_over_rpc else None,
        )

    def _find_actor(self, worker_name: str) -> _BaseActorManager:
        # [FUNCTION DOC] HELPER: Find the actor manager with the given worker name among running
        # cells; asserts exactly one match (so a stopped cell's workers are not found).
        matches = [a for c in self._all_cells() if c.alive for a in c.actors if a.name == worker_name]
        assert len(matches) == 1, f"{matches=}"
        return matches[0]

    def _find_cell(self, cell_id: str) -> _CellManager:
        # [FUNCTION DOC] HELPER: Find the cell manager with the given cell id; asserts exactly one
        # match (running or not).
        matches = [c for c in self._all_cells() if c.cell_id == cell_id]
        assert len(matches) == 1, f"{cell_id=} {matches=}"
        return matches[0]

    def _all_cells(self) -> list[_CellManager]:
        # [FUNCTION DOC] HELPER: All cell managers across all pools, pool by pool.
        return [c for g in self._pools.values() for c in g.cells]


@dataclass(kw_only=True)
class _PoolManager:
    # [CLASS DOC] HELPER: Keyword-only record of one pool: its worker spec and its cell managers.
    spec: BaseWorkerSpec
    cells: list[_CellManager]

    @classmethod
    def initial(cls, spec: BaseWorkerSpec, manager: RayWorkerManager) -> _PoolManager:
        # [FUNCTION DOC] HELPER: Build a pool with one stopped cell manager (no actors yet) for each
        # cell index from zero up to the spec's number of cells, all bound to the given manager.
        return cls(
            spec=spec,
            cells=[
                _CellManager(
                    manager=manager,
                    cell_index=cell_index,
                    spec=spec,
                    actors=None,
                )
                for cell_index in range(spec.scheduling.num_cells)
            ],
        )


SpecT = TypeVar("SpecT", bound=BaseWorkerSpec)


def _actor_manager_cls(spec: BaseWorkerSpec, *, comm_backend: WorkerCommBackend) -> type[_BaseActorManager]:
    # [FUNCTION DOC] HELPER: Choose the actor manager class for a spec: command specs use the
    # command manager regardless of backend; served specs use the rpc manager for the rpc backend
    # and the plain Ray manager for the Ray backend. Any other combination fails an assertion.
    match spec, comm_backend:
        case CommandWorkerSpec(), _:
            return _CommandActorManager
        case ServeWorkerSpec(), WorkerCommBackend.RPC:
            return _ServeActorRpcCommManager
        case ServeWorkerSpec(), WorkerCommBackend.RAY:
            return _ServeActorRayCommManager
    raise AssertionError(f"{spec.name} is neither served nor launched as a command")


@dataclass(kw_only=True)
class _CellManager(Generic[SpecT]):
    # [CLASS DOC] HELPER: Keyword-only record and lifecycle for one cell. Fields: the owning
    # manager, the cell index in its pool, the spec, the list of actor managers (none while the
    # cell is stopped, which is also the definition of not alive), the generation (incremented on
    # each launch, starts at zero) and the background liveness-scan task of the current generation.
    manager: RayWorkerManager
    cell_index: int
    spec: SpecT
    actors: list[_BaseActorManager] | None
    generation: int = 0
    liveness_scan_task: asyncio.Task | None = None

    async def launch_actors(self):
        # [FUNCTION DOC]
        # Asserts the cell is stopped, increments the generation, creates one actor manager of the
        # kind matching the spec and communication backend per worker in the cell, then launches all
        # their actors concurrently and starts the liveness scan task for this generation. The gpu
        # slot of worker w in cell c is the pool's placement-group slot offset plus (c times
        # workers per cell plus w) times gpu slots per worker, or none if the spec has no placement
        # group.
        assert self.actors is None
        self.generation += 1
        scheduling = self.spec.scheduling
        actor_manager_cls = _actor_manager_cls(self.spec, comm_backend=self.manager.comm_backend)
        self.actors = [
            actor_manager_cls(
                manager=self.manager,
                parent=self,
                worker_in_cell_index=worker_in_cell_index,
                spec=self.spec,
                actor_handle=None,
                gpu_slot_index=(
                    scheduling.pg_slot_offset
                    + (self.cell_index * scheduling.num_workers_per_cell + worker_in_cell_index)
                    * scheduling.num_gpu_slots_per_worker
                    if scheduling.pg_name is not None
                    else None
                ),
            )
            for worker_in_cell_index in range(scheduling.num_workers_per_cell)
        ]
        await self._for_all_actors(lambda a: a.launch_actor())
        self.liveness_scan_task = asyncio.create_task(self._scan_liveness_forever(self.generation))

    async def alloc_ports(self) -> None:
        # [FUNCTION DOC] Allocate ports for all workers of the cell concurrently.
        await self._for_all_actors(lambda a: a.alloc_ports())

    async def post_setup(self) -> None:
        # [FUNCTION DOC] Run the post-setup step of all workers of the cell concurrently.
        await self._for_all_actors(lambda a: a.post_setup())

    async def stop(self) -> None:
        # [FUNCTION DOC]
        # Stop the cell: no-op if already stopped. Otherwise stops all worker actors concurrently
        # and then marks the cell stopped by clearing the actors (the generation is unchanged; the
        # running liveness scan ends itself on its next wake-up). If stopping a worker raises, the
        # actors stay set.
        if self.actors is None:
            return
        await self._for_all_actors(lambda a: a.stop())
        self.actors = None

    async def _scan_liveness_forever(self, generation: int) -> None:
        # [FUNCTION DOC] HELPER: Background loop for one generation: while the cell is still at that
        # generation and running, sleep for the scan interval (10 seconds) and scan once; errors
        # from a scan are logged and the loop continues. Ends when the cell stops or relaunches.
        while self.generation == generation and self.actors is not None:
            await asyncio.sleep(_LIVENESS_SCAN_INTERVAL_SECONDS)
            try:
                await self._scan_liveness_once()
            except Exception:
                logger.error(f"Scanning liveness of cell {self.cell_id} failed, will scan again", exc_info=True)

    async def _scan_liveness_once(self) -> None:
        # [FUNCTION DOC] HELPER: Probe all workers for death without the lock. If any is dead,
        # take the membership lock, re-check that the cell is still running at the same generation
        # (it may have been stopped or restarted meanwhile, in which case do nothing), log an
        # error naming the dead workers and stop the whole cell, so it is reported as not alive.
        generation = self.generation
        dead_worker_names = await self._find_dead_worker_names()
        if not dead_worker_names:
            return

        async with self.manager._membership_lock:
            if self.actors is None or self.generation != generation:
                return
            logger.error(
                f"Cell {self.cell_id} lost workers {dead_worker_names} without being stopped, "
                f"so the whole cell is torn down and reported as not alive"
            )
            await self.stop()

    async def _find_dead_worker_names(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Probe all workers of the cell concurrently and return the names of
        # those found dead; an empty list if the cell is stopped.
        if (actors := self.actors) is None:
            return []
        probes = await asyncio.gather(*[a.probe_is_dead() for a in actors])
        return [actor.name for actor, is_dead in zip(actors, probes, strict=True) if is_dead]

    async def _for_all_actors(self, fn: Callable[[_BaseActorManager], Any]):
        # [FUNCTION DOC] HELPER: Apply an async function to every actor manager of the cell
        # concurrently and wait for all; the first error propagates. Requires the cell to be
        # running (iterating over none would fail).
        await asyncio.gather(*[fn(a) for a in self.actors])

    def get_info(self) -> CellInfo:
        # [FUNCTION DOC]
        # Snapshot of the cell as a cell info: cell id, pool id (spec name), alive only if running
        # and every worker already has its endpoints, worker names (empty when stopped), a
        # placeholder workers hash made from the generation number, and the metadata computed by the
        # spec's optional meta function from the cell index (empty mapping if the spec has none).
        return CellInfo(
            cell_id=self.cell_id,
            pool_id=self.spec.name,
            alive=self.alive and self._all_workers_have_addrs,
            worker_names=[a.name for a in self.actors] if self.actors is not None else [],
            workers_hash=f"pseudo-hash-{self.generation}",
            meta=f(WorkerMetaContext(cell_index=self.cell_index)) if (f := self.spec.meta) is not None else {},
        )

    @property
    def cell_id(self) -> str:
        # [FUNCTION DOC] Property: the cell id built from the pool (spec) name and the cell index.
        return compute_cell_id(pool_id=self.spec.name, cell_index=self.cell_index)

    @property
    def alive(self) -> bool:
        # [FUNCTION DOC] Property: true iff worker actors currently exist for this cell.
        return self.actors is not None

    @property
    def _all_workers_have_addrs(self) -> bool:
        # [FUNCTION DOC] HELPER: Property: true iff every worker has been given its endpoints
        # (vacuously true when the cell is stopped).
        return all(a.self_addrs is not None for a in self.actors or [])


_SHUTDOWN_TIMEOUT = 30


@dataclass(kw_only=True)
class _BaseActorManager(Generic[SpecT]):
    # [CLASS DOC] HELPER: Keyword-only base for the manager of one worker actor. Fields: owning
    # manager, parent cell, index of the worker in its cell, spec, the Ray actor handle (none until
    # launched), the endpoints (none until ports are allocated), the actor-local gpu ids, and the
    # index of the placement-group gpu slot (none when no placement group). Subclasses implement
    # how the actor is created and set up for each worker kind.
    manager: RayWorkerManager
    parent: _CellManager
    worker_in_cell_index: int
    spec: SpecT
    actor_handle: ray.actor.ActorHandle | None
    self_addrs: NamedHostAndPorts | None = None
    local_gpu_ids: list[int] | None = None
    gpu_slot_index: int | None

    async def launch_actor(self) -> None:
        # [FUNCTION DOC] Abstract: create the Ray actor and store its handle. Subclasses override.
        raise NotImplementedError

    async def post_setup(self) -> None:
        # [FUNCTION DOC] Abstract: start the actual worker once ports are known. Subclasses override.
        raise NotImplementedError

    async def alloc_ports(self) -> None:
        # [FUNCTION DOC]
        # Query the actor (concurrently) for its node ip, node external ip and the mapping of its
        # global gpu ids to local ones (stored as the local gpu ids). Then, for each port declared
        # by the spec, in order: skip master-mode ports on workers other than index zero; for
        # dynamic ports allocate (consecutive) ports on that node through the manager's port
        # allocator; for static ports use the static port, plus the cell index if offset by cell,
        # and assert it is free. Each result records the ip-wrapped host (ipv6 bracketed), the
        # port and the external host if any. Stores the mapping from port name to endpoint as the
        # worker's endpoints.
        allocated: NamedHostAndPorts = {}

        node_ip, external_ip, self.local_gpu_ids = await asyncio.gather(
            self.actor_handle._get_node_ip.remote(),
            self.actor_handle._get_node_external_ip.remote(),
            self.actor_handle._to_local_gpu_ids.remote(gpu_ids=self.gpu_ids),
        )
        for port_info in self.spec.port_infos:
            if self.worker_in_cell_index != 0 and port_info.mode == "master":
                continue
            if port_info.allow_dynamic:
                port = await self.manager.port_allocator.alloc(
                    self.actor_handle, node_ip=node_ip, consecutive=port_info.num_consecutive
                )
            else:
                port = port_info.static_port + (self.parent.cell_index if port_info.offset_by_cell else 0)
                await self._assert_static_port_is_free(port=port, port_name=port_info.name, node_ip=node_ip)
            allocated[port_info.name] = HostAndPort(
                host=wrap_ipv6(node_ip),
                port=port,
                external_host=wrap_ipv6(external_ip) if external_ip else None,
            )

        self.self_addrs = allocated

    async def _assert_static_port_is_free(self, *, port: int, port_name: str, node_ip: str) -> None:
        # [FUNCTION DOC] HELPER: Ask the actor whether the port is available on its node and fail an
        # assertion with an explanatory message (a stale process is the usual cause) if not.
        free = await self.actor_handle._is_port_available.remote(port=port)
        assert free, (
            f"Port {port} on {node_ip} is already in use, so {self.name} cannot serve its {port_name!r} "
            f"endpoint there; a stale process from an earlier run is the usual cause"
        )

    @property
    def launch_context(self) -> WorkerLaunchContext:
        # [FUNCTION DOC] Property: the launch context (cell index, worker index in cell, gpu ids)
        # describing this worker.
        return WorkerLaunchContext(
            cell_index=self.parent.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
            gpu_ids=self.gpu_ids,
        )

    def _compute_remote_options(self) -> dict:
        # [FUNCTION DOC] HELPER: Extra options for Ray's actor class decoration; none by default.
        return {}

    def _create_actor(self, actor_class: type, **ctor_kwargs) -> ray.actor.ActorHandle:
        # [FUNCTION DOC] HELPER: Create a Ray actor of the given class and return its handle. If the
        # spec has a placement group name, schedule it into that group at the bundle for this
        # worker's gpu slot (via the group's reordered bundle indices), capturing child tasks.
        # Applies the class decoration options, then the cpu and gpu counts from the spec, and the
        # spec's environment variables for this launch context as the actor runtime environment;
        # optionally pins to the head node when the spec asks. The constructor arguments are passed
        # through.
        scheduling_strategy = None
        if (pg_name := self.spec.scheduling.pg_name) is not None:
            pg = self.manager.pgs[pg_name]
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg.pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=pg.pg_reordered_bundle_indices[self.gpu_slot_index],
            )

        remote_options = self._compute_remote_options()
        remote_class = ray.remote(**remote_options)(actor_class) if remote_options else ray.remote(actor_class)

        return remote_class.options(
            num_cpus=self.spec.scheduling.num_cpus_per_worker,
            num_gpus=self.spec.scheduling.num_gpus_per_worker,
            **(dict(scheduling_strategy=s) if (s := scheduling_strategy) is not None else {}),
            runtime_env={"env_vars": self.spec.env_var(self.launch_context)},
            **(compute_ray_pin_head_options() if self.spec.scheduling.pin_to_head else {}),
        ).remote(**ctor_kwargs)

    async def probe_is_dead(self) -> bool:
        # [FUNCTION DOC] Probe liveness through the Ray worker handle; false if there is no actor
        # yet, otherwise true only when the actor is definitively dead.
        if self.actor_handle is None:
            return False
        return await RayWorkerHandle(self.actor_handle).probe_is_dead()

    async def stop(self) -> None:
        # [FUNCTION DOC] No-op if there is no actor. Otherwise runs the subclass graceful shutdown
        # and then force-kills the Ray actor, logging (not raising) a failure to kill. The actor
        # handle is kept after stopping.
        if self.actor_handle is None:
            return

        await self._shutdown_gracefully()

        try:
            ray.kill(self.actor_handle)
            logger.info(f"Killed actor at {self=}")
        except Exception as e:
            logger.warning(f"Failed to kill actor at {self=} ({e})")

    async def _shutdown_gracefully(self) -> None:
        # [FUNCTION DOC] HELPER: Hook for subclasses to stop their workload before the actor is
        # killed; does nothing by default.
        pass

    @property
    def name(self) -> str:
        # [FUNCTION DOC] Property: worker name from pool (spec) name, cell index and worker index.
        return compute_worker_name(
            pool_id=self.spec.name,
            cell_index=self.parent.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
        )

    @property
    def generation(self) -> int:
        # [FUNCTION DOC] Property: the generation of the parent cell.
        return self.parent.generation

    @property
    def gpu_ids(self) -> list[int]:
        # [FUNCTION DOC] Property: global gpu ids of this worker. Empty if the spec has no placement
        # group; otherwise a consecutive range of gpu slots-per-worker ids starting at the base gpu
        # id of the worker's slot in the group's reordered gpu id list.
        if (pg_name := self.spec.scheduling.pg_name) is None:
            return []
        pg = self.manager.pgs[pg_name]
        base_gpu_id = int(pg.pg_reordered_gpu_ids[self.gpu_slot_index])
        return list(range(base_gpu_id, base_gpu_id + self.spec.scheduling.num_gpu_slots_per_worker))

    @property
    def master_mode_addrs(self) -> NamedHostAndPorts:
        # [FUNCTION DOC] Property: the subset of this worker's endpoints whose port declaration is
        # master mode, keyed by port name. Requires endpoints to be allocated.
        return {info.name: self.self_addrs[info.name] for info in self.spec.port_infos if info.mode == "master"}


@dataclass
class _CommandActorManager(_BaseActorManager[CommandWorkerSpec]):
    # [CLASS DOC] HELPER: Actor manager for command workers: the actor runs a shell command in a
    # subprocess.
    async def launch_actor(self) -> None:
        # [FUNCTION DOC] Create the command actor with no constructor arguments.
        self.actor_handle = self._create_actor(CommandActor)

    async def post_setup(self) -> None:
        # [FUNCTION DOC]
        # Build the launch command context: launch context fields, this worker's endpoints merged
        # with the master-mode endpoints of worker zero of the cell (worker zero's override),
        # the endpoints of all pools, and the local gpu ids. Render the command from the spec and
        # start it on the actor prefixed with exec so the command replaces the shell (so killing
        # the actor kills the command itself, not just a shell leaving it orphaned). The remote run
        # call is fire-and-forget (not awaited); no extra environment variables are passed.
        ctx = LaunchCommandContext(
            **dict(self.launch_context),
            self_addrs={
                **self.self_addrs,
                **self.parent.actors[0].master_mode_addrs,
            },
            pool_addrs=self.manager.get_addrs(),
            local_gpu_ids=self.local_gpu_ids,
        )
        launch_cmd = self.spec.launch_command(ctx)
        # exec lets the command replace /bin/sh, so parent death and Ray's child cleanup kill the
        # command itself rather than a shell that would leave it orphaned with its ports.
        self.actor_handle.run.remote(cmd=f"exec {launch_cmd}", envs={})

    async def _shutdown_gracefully(self) -> None:
        # [FUNCTION DOC] HELPER: Ask the actor to shut down its subprocess, waiting up to the module
        # shutdown timeout (30 seconds); any failure is logged as a warning and ignored.
        try:
            await asyncio.wait_for(self.actor_handle.shutdown.remote(), timeout=_SHUTDOWN_TIMEOUT)
        except Exception as e:
            logger.warning(f"Graceful shutdown of {self=} failed ({e})")


@dataclass
class _ServeActorRayCommManager(_BaseActorManager[ServeWorkerSpec]):
    # [CLASS DOC] HELPER: Actor manager for served Python workers called directly through Ray
    # (the Ray communication backend): the actor is the worker class itself, bootstrapped.
    def _compute_remote_options(self) -> dict:
        # [FUNCTION DOC] HELPER: If the spec declares concurrency groups, pass them as the actor's
        # concurrency groups option; otherwise no options.
        groups = self.spec.concurrency_groups
        return {} if groups is None else dict(concurrency_groups=groups)

    async def launch_actor(self) -> None:
        # [FUNCTION DOC] Create the actor from the computed worker class, passing the spec's
        # constructor-argument function and the launch context as constructor arguments.
        self.actor_handle = self._create_actor(
            self._compute_actor_class(),
            ctor_kwargs=self.spec.ctor_kwargs,
            context=self.launch_context,
        )

    def _compute_actor_class(self) -> type:
        # [FUNCTION DOC] HELPER: Load the bootstrapped worker class for the spec's class path. If
        # some methods are routed to concurrency groups, return a dynamically created subclass in
        # which each such method is overridden by a wrapper tagged with its group; otherwise return
        # the class itself.
        actor_class = bootstrapped_worker_class(self.spec.worker_class)
        method_groups = self._compute_method_concurrency_groups(actor_class)
        if not method_groups:
            return actor_class
        return type(
            f"{actor_class.__name__}WithConcurrencyGroups",
            (actor_class,),
            {
                name: _route_method_to_concurrency_group(getattr(actor_class, name), group=group)
                for name, group in method_groups.items()
            },
        )

    def _compute_method_concurrency_groups(self, actor_class: type) -> dict[str, str]:
        # [FUNCTION DOC] HELPER: Return a mapping from method name to concurrency group name for
        # the worker class, as declared by its rpc annotations. Empty if the spec declares no
        # groups. Asserts that, when groups are declared, at least one method is annotated and
        # that every group used by a method is among the declared ones.
        if (groups := self.spec.concurrency_groups) is None:
            return {}

        method_groups = declared_concurrency_groups(actor_class)
        assert method_groups, (
            f"Worker {self.spec.name!r} declares concurrency groups {sorted(groups)} but no method of "
            f"{actor_class.__name__} is annotated with @rpc(concurrency_group=...): threading the actor "
            f"while every method stays in the default group buys nothing"
        )
        undeclared = sorted(set(method_groups.values()) - set(groups))
        assert not undeclared, f"Worker {self.spec.name!r} routes methods to undeclared groups: {undeclared}"
        return method_groups

    async def post_setup(self) -> None:
        # [FUNCTION DOC] Nothing to do: the worker is fully built by its constructor.
        pass


def _route_method_to_concurrency_group(method: Callable, *, group: str) -> Callable:
    # [FUNCTION DOC] HELPER: Wrap a method in a thin pass-through (keeping its metadata) and tag it
    # as belonging to the named Ray concurrency group; returns the tagged wrapper.
    @functools.wraps(method)
    def routed(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the original method with the same arguments.
        return method(self, *args, **kwargs)

    return ray.method(concurrency_group=group)(routed)


@dataclass
class _ServeActorRpcCommManager(_BaseActorManager[ServeWorkerSpec]):
    # [CLASS DOC] HELPER: Actor manager for served Python workers reached over the rpc server: the
    # actor is a generic serve actor that builds the worker lazily and exposes it over rpc.
    async def launch_actor(self) -> None:
        # [FUNCTION DOC] Create a serve actor, giving it a partial builder that later constructs the
        # worker from the spec's class path, constructor-argument function and the launch context.
        self.actor_handle = self._create_actor(
            ServeActor,
            build_worker=partial(
                _build_serve_worker,
                worker_class_path=self.spec.worker_class,
                ctor_kwargs=self.spec.ctor_kwargs,
                context=self.launch_context,
            ),
        )

    async def post_setup(self) -> None:
        # [FUNCTION DOC] Tell the serve actor to start its rpc server on the port allocated under
        # the well-known rpc port name, and wait for that to finish.
        await self.actor_handle.start_rpc_server.remote(port=self.self_addrs[RPC_PORT_NAME].port)


def _build_serve_worker(
    *, worker_class_path: str, ctor_kwargs: Callable[[WorkerCtorContext], dict[str, Any]], context: WorkerLaunchContext
) -> Any:
    # [FUNCTION DOC] HELPER: Load the bootstrapped worker class for the class path and instantiate
    # it with the constructor-argument function and launch context; returns the worker instance.
    return bootstrapped_worker_class(worker_class_path)(ctor_kwargs=ctor_kwargs, context=context)


@functools.cache
def bootstrapped_worker_class(worker_class_path: str) -> type:
    # [FUNCTION DOC]
    # Load the worker class by its registered path and return a cached subclass of it that also
    # mixes in node probing, and whose constructor takes a constructor-argument function and a
    # launch context instead of the original arguments: it calls the function with a context
    # derived from the launch context and passes the resulting keyword arguments to the original
    # constructor. The subclass copies the original name, qualified name and module so it appears
    # as the original class. Results are cached per class path.
    worker_class = load_function(worker_class_path)

    class BootstrappedWorker(worker_class, NodeProbeMixin):
        # [CLASS DOC] HELPER: The loaded worker class extended with node probing; see the enclosing
        # function.
        def __init__(
            self, *, ctor_kwargs: Callable[[WorkerCtorContext], dict[str, Any]], context: WorkerLaunchContext
        ) -> None:
            # [FUNCTION DOC] HELPER: Compute the original constructor's keyword arguments by
            # calling the function with a constructor context, and call the original constructor.
            super().__init__(**ctor_kwargs(_ctor_context(context)))

    BootstrappedWorker.__name__ = worker_class.__name__
    BootstrappedWorker.__qualname__ = worker_class.__qualname__
    BootstrappedWorker.__module__ = worker_class.__module__
    return BootstrappedWorker


def _ctor_context(launch_context: WorkerLaunchContext) -> WorkerCtorContext:
    # [FUNCTION DOC] HELPER: Build the constructor context from a launch context: copies the cell
    # index, worker index and gpu ids, and supplies a lazily created Ray backend capability.
    return WorkerCtorContext(
        cell_index=launch_context.cell_index,
        worker_in_cell_index=launch_context.worker_in_cell_index,
        gpu_ids=launch_context.gpu_ids,
        capability=DeferredBackendCapability(create=_create_ray_backend_capability),
    )


def _create_ray_backend_capability() -> BackendCapability:
    # [FUNCTION DOC] HELPER: Build a Ray backend capability around the handle of the (named)
    # worker manager actor, looked up in the cluster at call time.
    return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())


async def _gather_or_raise(coros: list[Coroutine[Any, Any, None]]) -> None:
    # [FUNCTION DOC] HELPER: Run all coroutines concurrently and wait for all of them to finish
    # (unlike plain fail-fast gathering, none is abandoned midway), then raise the first exception
    # found in argument order, if any.
    results = await asyncio.gather(*coros, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
