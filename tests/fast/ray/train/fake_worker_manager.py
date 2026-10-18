# [MODULE DOC]
# Provides an in-process fake of the ray worker manager actor handle that trainer cells and the
# ray worker provider talk to. Each manager method that real callers invoke with a remote call is
# wrapped so that calling its remote form runs the function synchronously and returns a ray object
# reference (the result put in the object store), so callers can resolve it with ray get or await
# it. Cells and their dummy trainer actors are created lazily, one cell at a time, and are killed
# when the cell is stopped or the fake is torn down. A global generation number is derived from
# how many start-cells calls were made.
"""In-process stand-in for the RayWorkerManager actor handle used by trainer cells."""

import logging

import ray
from tests.fast.ray.train.dummy_actor import DummyTrainActor

from miles.utils.workers.naming import compute_cell_id, compute_worker_name, parse_cell_id, parse_worker_name
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import MASTER_PORT_NAME, HostAndPort

logger = logging.getLogger(__name__)


class _FakeRemoteMethod:
    # [CLASS DOC] HELPER: Mimics a ray actor method object: it exposes a remote call that executes
    # the wrapped function immediately in the caller's process.
    def __init__(self, fn):
        # [FUNCTION DOC] HELPER: Stores the function to run.
        self._fn = fn

    def remote(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Runs the wrapped function with the given arguments right away and
        # returns a ray object reference holding its return value (ray must be initialized).
        # Exceptions raised by the function propagate immediately at the call, not at resolution.
        return ray.put(self._fn(*args, **kwargs))


class FakeWorkerManager:
    # [CLASS DOC] Fake worker manager. Configuration: the number of cells reported per pool, and
    # the number of dummy actors per cell (both mutable by tests; the actor count is read when a
    # cell's actors are first created). Records every start cells call (list of cell id lists) and
    # every stop cells call. The generation reported for workers is one plus the number of start
    # cells calls so far. Cells whose index is registered as failing get dummy actors preset to
    # fail their init call. Remote style methods: get cell infos, get worker infos, get actor
    # handle, start cells, stop cells.
    def __init__(self, *, num_cells: int = 1, actor_count_per_cell: int = 1):
        # [FUNCTION DOC] Stores the cell and actor counts (defaults 1 and 1), empty start and stop
        # records, an empty mapping from cell id to actor handles, an empty set of cell indices
        # failing init, no per-worker master address override, and the remote-style wrappers of
        # the manager methods.
        self.num_cells = num_cells
        self.actor_count_per_cell = actor_count_per_cell
        self.started_cell_ids: list[list[str]] = []
        self.stopped_cell_ids: list[list[str]] = []
        self._handles: dict[str, list] = {}
        self._cell_indices_failing_init: set[int] = set()
        self.master_addr_per_worker: list[HostAndPort] | None = None

        self.get_cell_infos = _FakeRemoteMethod(self._get_cell_infos)
        self.get_worker_infos = _FakeRemoteMethod(self._get_worker_infos)
        self.get_actor_handle = _FakeRemoteMethod(self._get_actor_handle)
        self.start_cells = _FakeRemoteMethod(self.started_cell_ids.append)
        self.stop_cells = _FakeRemoteMethod(self._stop_cells)

    def fail_init_for_cell(self, cell_index: int) -> None:
        # [FUNCTION DOC] Registers a cell index whose dummy actors will be told to fail their init
        # call. Only affects actors created after this call (cells not yet materialized).
        self._cell_indices_failing_init.add(cell_index)

    def _get_cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        # [FUNCTION DOC] HELPER: Builds cell infos for every cell index (0 up to the configured
        # cell count minus one) of every given pool, keyed by cell id. Each is alive, lists
        # actor-count worker names, carries a workers hash made of the text pseudo-hash- plus one
        # plus the number of start calls so far, and has a metadata mapping holding the cell
        # index. Creates no actors.
        infos: dict[str, CellInfo] = {}
        for pool_id in pool_ids:
            for cell_index in range(self.num_cells):
                cell_id = compute_cell_id(pool_id=pool_id, cell_index=cell_index)
                infos[cell_id] = CellInfo(
                    cell_id=cell_id,
                    pool_id=pool_id,
                    alive=True,
                    worker_names=[
                        compute_worker_name(pool_id=pool_id, cell_index=cell_index, worker_in_cell_index=worker_index)
                        for worker_index in range(self.actor_count_per_cell)
                    ],
                    workers_hash=f"pseudo-hash-{1 + len(self.started_cell_ids)}",
                    meta={"cell_index": cell_index},
                )
        return infos

    def _get_worker_infos(self, cell_id: str) -> list[WorkerInfo]:
        # [FUNCTION DOC] HELPER: Returns worker infos for a cell, one per actor. On first request
        # for a cell id it creates the configured number of dummy trainer actors and remembers
        # them (if the cell index is registered as failing, each actor is told, and the call
        # waits until done, to fail its init); later requests reuse them. Each info has the
        # standard worker name, a generation of one plus the number of start calls so far, an
        # address mapping with the master port name set to the computed master address for the
        # worker index, and the worker index as its only gpu id.
        parsed = parse_cell_id(cell_id)
        if cell_id not in self._handles:
            handles = [DummyTrainActor.remote() for _ in range(self.actor_count_per_cell)]
            if parsed.cell_index in self._cell_indices_failing_init:
                ray.get([handle.set_fail_methods.remote(["init"]) for handle in handles])
            self._handles[cell_id] = handles
        return [
            WorkerInfo(
                name=compute_worker_name(
                    pool_id=parsed.pool_id, cell_index=parsed.cell_index, worker_in_cell_index=worker_index
                ),
                generation=1 + len(self.started_cell_ids),
                self_addrs={MASTER_PORT_NAME: self._compute_master_addr(worker_index)},
                gpu_ids=[worker_index],
            )
            for worker_index, handle in enumerate(self._handles[cell_id])
        ]

    def _get_actor_handle(self, worker_name: str, *, expected_generation: int):
        # [FUNCTION DOC] HELPER: Returns the dummy actor handle for a worker name. Asserts that the
        # current generation (one plus the number of start calls) equals the expected generation,
        # with a message naming both; raises a key or index error if the cell's actors were never
        # created or the worker index is out of range.
        generation = 1 + len(self.started_cell_ids)
        assert (
            generation == expected_generation
        ), f"{worker_name} is generation {generation}, not {expected_generation}"
        pool_id, cell_index, worker_in_cell_index = parse_worker_name(worker_name)
        return self._handles[compute_cell_id(pool_id=pool_id, cell_index=cell_index)][worker_in_cell_index]

    def _compute_master_addr(self, worker_index: int) -> HostAndPort:
        # [FUNCTION DOC] HELPER: Returns the master address for a worker: the fixed 10.0.0.1:20000
        # unless a per-worker override list is set, in which case the override at that worker
        # index is returned (an index error if the list is too short).
        if self.master_addr_per_worker is None:
            return HostAndPort(host="10.0.0.1", port=20000)
        return self.master_addr_per_worker[worker_index]

    def _stop_cells(self, cell_ids: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Appends the cell id list to the stop record, then for each cell id
        # forgets its actor handles and kills those actors (unknown cell ids are ignored). A later
        # request for the same cell creates fresh actors.
        self.stopped_cell_ids.append(cell_ids)
        for cell_id in cell_ids:
            self._kill(self._handles.pop(cell_id, []))

    def kill_all_actors(self) -> None:
        # [FUNCTION DOC] Kills every dummy actor of every remembered cell and clears the mapping.
        # Does not touch the start or stop records. Used by test teardown.
        for handles in self._handles.values():
            self._kill(handles)
        self._handles.clear()

    @staticmethod
    def _kill(handles: list) -> None:
        # [FUNCTION DOC] HELPER: Static. Kills each given ray actor handle; a failure to kill one is
        # logged as a warning with the traceback and does not stop the others.
        # The real manager kills the actor when it stops a cell. Dropping the handle
        # instead leaves the actor alive for as long as a cell still references it,
        # which accumulates one process per cell across a module.
        for handle in handles:
            try:
                ray.kill(handle)
            except Exception:
                logger.warning(f"Failed to kill {handle}", exc_info=True)
