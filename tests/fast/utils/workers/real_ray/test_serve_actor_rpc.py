# [MODULE DOC]
# Tests of a worker launched by the real Ray worker manager but driven over the rpc backend
# (an http server inside the Ray actor). Shared pools serve read-only checks (handle type,
# allocated port, reachability, running inside an actor, the full conformance checks); a private
# pool per test is used where the cell gets stopped, to test death confirmation and the dead
# probe.
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import pytest
import ray
from tests.fast.utils.workers.conformance import (
    CHECK_IDS,
    CHECKS,
    POOL_ID,
    READY_TIMEOUT_SECONDS,
    HandleCheck,
    compute_spec,
)

from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.real_ray.conftest import (
    kill_named_worker_manager,
    kill_quietly,
    wait_until_named_manager_is_gone,
)

from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

CELL_ID = compute_cell_id(pool_id=POOL_ID, cell_index=0)

CONFIRM_DEAD_TIMEOUT_SECONDS = 60.0


@dataclass
class _LaunchedPool:
    # [CLASS DOC] HELPER: A launched manager together with the worker infos of the pool's cell
    # and one handle per worker, in the same order.
    manager: ray.actor.ActorHandle
    infos: list[WorkerInfo]
    handles: list[BaseWorkerHandle]


# every actor the manager launches is a fresh process that imports miles, so the checks that
# only read from the pool share one, and only the class that stops the cell rebuilds it
@pytest.fixture(autouse=True, scope="class")
def clean_named_worker_manager(ray_local_mode) -> Iterator[None]:
    # [FUNCTION DOC] HELPER: Class-scoped autouse fixture freeing the well-known manager name
    # before and after each test class.
    _free_well_known_name()
    yield
    _free_well_known_name()


def _free_well_known_name() -> None:
    # [FUNCTION DOC] HELPER: Kills the fixed-name manager actor if present and waits until its
    # name is unregistered.
    kill_named_worker_manager()
    wait_until_named_manager_is_gone()


def _launch_rpc_pool() -> _LaunchedPool:
    # [FUNCTION DOC] HELPER: Frees the manager name, launches the manager with minimal args, the
    # conformance spec on a dynamic rpc port, no placement groups and the rpc comm backend. Via a
    # Ray worker provider, reads the worker infos of the pool's cell (exactly one cell expected)
    # and builds the handles for them. Returns the manager, infos and handles.
    _free_well_known_name()
    manager = RayWorkerManager.launch(
        worker_manager_args(env_report_interval_seconds=0.0),
        [compute_spec(rpc_port=0)],
        {},
        comm_backend=WorkerCommBackend.RPC,
    )
    provider = RayWorkerProvider(worker_manager_handle=manager, pool_ids=[POOL_ID])
    (infos,) = provider.get_worker_infos(cell_ids=[CELL_ID])
    handles = provider.get_handles_of_worker_infos(infos)
    return _LaunchedPool(manager=manager, infos=infos, handles=[handles[info.name] for info in infos])


@pytest.fixture(scope="class")
def shared_rpc_pool(ray_local_mode) -> Iterator[_LaunchedPool]:
    # [FUNCTION DOC] HELPER: Class-scoped fixture yielding a launched rpc pool; the manager is
    # killed quietly afterwards.
    pool = _launch_rpc_pool()
    yield pool
    kill_quietly(pool.manager)


@pytest.fixture
def rpc_pool(ray_local_mode) -> Iterator[_LaunchedPool]:
    # [FUNCTION DOC] HELPER: Per-test fixture yielding a fresh launched rpc pool; the manager is
    # killed quietly afterwards.
    pool = _launch_rpc_pool()
    yield pool
    kill_quietly(pool.manager)


@pytest.fixture
async def shared_rpc_handle(shared_rpc_pool: _LaunchedPool) -> AsyncIterator[BaseWorkerHandle]:
    # [FUNCTION DOC] HELPER: Yields the first handle of the shared pool after waiting up to the
    # ready timeout for readiness.
    handle = shared_rpc_pool.handles[0]
    await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
    yield handle


@pytest.fixture
async def rpc_handle(rpc_pool: _LaunchedPool) -> AsyncIterator[BaseWorkerHandle]:
    # [FUNCTION DOC] HELPER: Yields the first handle of the private pool after waiting up to the
    # ready timeout for readiness.
    handle = rpc_pool.handles[0]
    await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
    yield handle


class TestARayLaunchedWorkerServedOverRpc:
    # [CLASS DOC] Read-only checks on a shared rpc pool.
    def test_the_launcher_answers_with_an_rpc_handle(self, shared_rpc_pool: _LaunchedPool):
        # [FUNCTION DOC] Asserts the first handle is an rpc worker handle.
        """Under rpc comm the driver must not be handed an actor handle it would call over ray."""
        assert isinstance(shared_rpc_pool.handles[0], RpcWorkerHandle)

    def test_the_worker_serves_on_the_port_the_launcher_allocated(self, shared_rpc_pool: _LaunchedPool):
        # [FUNCTION DOC] Asserts the worker info's "rpc" address has a positive port (the spec
        # asked for a dynamic port).
        """A dynamically allocated port is the only thing that lets two workers share one node."""
        assert shared_rpc_pool.infos[0].self_addrs["rpc"].port > 0

    async def test_the_worker_is_reachable(self, shared_rpc_handle: BaseWorkerHandle):
        # [FUNCTION DOC] Calls the conformance worker's sum with 2 and 5 over http; asserts 7.
        """This is the end to end claim of the mode: ray started the worker, http drives it."""
        assert await shared_rpc_handle.add(a=2, b=5) == 7

    async def test_the_worker_runs_inside_a_ray_actor(self, shared_rpc_handle: BaseWorkerHandle):
        # [FUNCTION DOC] Asserts the worker reports a non-empty Ray actor id.
        """RDT and the rest of the ray ecosystem need the worker in the actor, not in a child process of it."""
        assert await shared_rpc_handle.report_ray_actor_id() is not None

    @pytest.mark.parametrize("check", CHECKS, ids=CHECK_IDS)
    async def test_the_handle_contract_holds(self, shared_rpc_handle: BaseWorkerHandle, check: HandleCheck):
        # [FUNCTION DOC] Parametrized over all conformance checks (shared and rpc-only); runs
        # each against the shared handle.
        """The same contract as the serve-subprocess column, now over a worker that ray launched."""
        await check(shared_rpc_handle)


class TestWhenTheLauncherStopsTheCell:
    # [CLASS DOC] Death confirmation after the manager stops the cell; uses a private pool per
    # test.
    async def test_the_worker_is_confirmed_dead(self, rpc_pool: _LaunchedPool, rpc_handle: BaseWorkerHandle):
        # [FUNCTION DOC] Stops the cell through the manager and asserts the handle's death
        # wait (60 s limit) completes.
        """Fault tolerance kills a cell and then waits for this confirmation before healing it."""
        await rpc_pool.manager.stop_cells.remote([CELL_ID])

        await rpc_handle.wait_dead(timeout=CONFIRM_DEAD_TIMEOUT_SECONDS)

    async def test_the_probe_that_confirms_it_reads_a_refused_connection(
        self, rpc_pool: _LaunchedPool, rpc_handle: BaseWorkerHandle
    ):
        # [FUNCTION DOC] Stops the cell, waits for death confirmation, then asserts the dead
        # probe returns true.
        """Killing the actor takes the server down with it, which is what makes the probe conclusive."""
        await rpc_pool.manager.stop_cells.remote([CELL_ID])
        await rpc_handle.wait_dead(timeout=CONFIRM_DEAD_TIMEOUT_SECONDS)

        assert await rpc_handle.probe_is_dead() is True
