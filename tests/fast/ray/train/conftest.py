# [MODULE DOC]
# Shared fixtures and factories for the trainer cell and trainer controller tests in this
# directory. A module-level fake worker manager (set up fresh for every test by an autouse
# fixture) stands in for the ray worker manager actor handle, serving cells whose workers are
# dummy ray actors. Autouse fixtures also start a local ray (module scope), reset the object store
# singleton, replace the independent-dp quorum store creation with a fixed address, and remove
# retry backoff sleeps. Factories build deployment identities, worker providers, independent dp
# infos and trainer cells (optionally already alive), and a recording health checker lets tests
# observe when and how a cell starts and stops health checking.
import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
import ray
from tests.fast.ray.train.fake_worker_manager import FakeWorkerManager

import miles.ray.train.group as group_module
from miles.ray.specs.train import compute_trainer_pool_id
from miles.ray.train.cell import TrainerCell
from miles.utils import object_store
from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.ft_utils.health_checker import BaseHealthChecker, NoopHealthChecker
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.retry_utils import retry
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import DeploymentIdentity
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

fake_worker_manager: FakeWorkerManager | None = None

FAKE_STORE_ADDR = "10.0.0.7:29500"


def make_deployment_identity(**overrides: Any) -> DeploymentIdentity:
    # [FUNCTION DOC] Builds a deployment identity with run uuid 0123456789abcdef and the trainer
    # deploy component, with any given fields replaced. Returns it.
    defaults: dict[str, Any] = dict(run_uuid="0123456789abcdef", deploy_component="trainer")
    return DeploymentIdentity(**{**defaults, **overrides})


@pytest.fixture(autouse=True)
def _patch_worker_backends(monkeypatch: pytest.MonkeyPatch):
    # [FUNCTION DOC] HELPER: Autouse fixture. Creates a fresh fake worker manager into the module
    # global, makes the real worker manager's handle lookup return it (patched through the
    # monkeypatch fixture so teardown order stays last-in-first-out), yields to the test, and
    # afterwards kills every dummy actor the fake created.
    """Patching through the same monkeypatch a test may reuse keeps the undo order last-in-first-out.

    A `mock.patch` context manager here is torn down before that monkeypatch, so a test that
    repatches `create` restores this fixture's stand-in and leaks it into every later test.
    """
    global fake_worker_manager
    fake_worker_manager = FakeWorkerManager()
    monkeypatch.setattr(RayWorkerManager, "get_handle", lambda: fake_worker_manager)
    yield
    fake_worker_manager.kill_all_actors()


@pytest.fixture(scope="module", autouse=True)
def ray_env(ray_local_mode):
    # [FUNCTION DOC] HELPER: Module scoped autouse fixture that depends on the ray local mode
    # fixture (defined elsewhere) so a local ray runtime exists for every test of the module; it
    # does nothing else.
    yield


@pytest.fixture(autouse=True)
def _fresh_object_store_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture. Resets the object store module's singleton instance
    # to none for the test, so each test starts without an initialized object store.
    monkeypatch.setattr(object_store, "_INSTANCE", None)


@pytest.fixture(autouse=True)
def _fake_indep_dp_store(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture. Replaces the group module's quorum store creation
    # with one returning a plain object and the fixed address 10.0.0.7:29500, so no real tcp store
    # is bound.
    monkeypatch.setattr(group_module, "create_tcp_store", lambda: (object(), FAKE_STORE_ADDR))


@pytest.fixture(autouse=True)
def instant_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture. Replaces the retry function used by the group module
    # with a wrapper that runs the real retry but with a sleep function that returns immediately,
    # so retry backoff never delays tests. Other retry options pass through unchanged.
    async def _no_sleep(_seconds: float) -> None:
        # [FUNCTION DOC] HELPER: Sleep replacement that returns at once.
        return None

    async def _retry_without_sleeping(fn: Callable[[int], Awaitable[Any]], **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Calls the real retry with the given function and options, forcing
        # the sleep function option to the instant one, and returns its result.
        return await retry(fn, **{**kwargs, "sleep_fn": _no_sleep})

    monkeypatch.setattr(group_module, "retry", _retry_without_sleeping)


class RecordingHealthChecker(BaseHealthChecker):
    # [CLASS DOC] Health checker double that never judges health: its status is always unknown. It
    # records how many times it was started, whether it was stopped, whether its background task
    # actually began running, and (through an optional observer callback) whether the owning cell
    # was alive at the moment of starting.
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the counters and flags (zero starts, not stopped, task not
        # started, no observation), the optional observer callback (none) and the task handle (none).
        self.start_count: int = 0
        self.stopped: bool = False
        self.task_started: bool = False
        self.alive_when_started: bool | None = None
        self.observe_alive: Callable[[], bool] | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def status(self) -> TriState:
        # [FUNCTION DOC] Property; always reports the unknown tri-state.
        return TriState.UNKNOWN

    def start(self) -> None:
        # [FUNCTION DOC] Increments the start count; if an observer callback is set, stores its
        # result as the alive-when-started value; then schedules the background run as an asyncio
        # task (requires a running event loop). The task is stored.
        self.start_count += 1
        if self.observe_alive is not None:
            self.alive_when_started = self.observe_alive()
        self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        # [FUNCTION DOC] Marks the checker stopped; cancels and forgets the background task if one exists.
        self.stopped = True
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        # [FUNCTION DOC] HELPER: Background body; only sets the task-started flag and returns.
        self.task_started = True


def make_provider(trainer_id: str = "actor") -> BaseWorkerProvider:
    # [FUNCTION DOC] Creates a ray worker provider over the current module-level fake worker
    # manager, watching the pool of the given trainer id (default actor). The fake manager must
    # already exist (set by the autouse fixture).
    return RayWorkerProvider(worker_manager_handle=fake_worker_manager, pool_ids=[compute_trainer_pool_id(trainer_id)])


def get_raw_actor_handles(cell: TrainerCell) -> list[ray.actor.ActorHandle]:
    # [FUNCTION DOC] Returns the underlying raw ray actor handles of a cell's workers, in worker
    # order, by unwrapping the cell's worker handle wrappers (reads a private attribute of each).
    return [handle._actor_handle for handle in cell._get_worker_handles()]


def make_indep_dp_info(
    *,
    cell_index: int = 0,
    alive_cell_indices: list[int] | None = None,
    quorum_id: int = 1,
) -> IndepDPInfo:
    # [FUNCTION DOC] Builds an independent dp info for a three-cell world. The alive cell indices
    # default to just cell 0; the alive rank is the position of the given cell index in that list
    # (raises a value error if the cell is not in the list) and the alive size is the list length.
    # Quorum id defaults to 1.
    if alive_cell_indices is None:
        alive_cell_indices = [0]
    return IndepDPInfo(
        cell_index=cell_index,
        num_cells=3,
        alive_rank=alive_cell_indices.index(cell_index),
        alive_size=len(alive_cell_indices),
        quorum_id=quorum_id,
        alive_cell_indices=alive_cell_indices,
    )


def make_cell(
    cell_index: int = 0,
    *,
    actor_count: int = 2,
    health_checker: BaseHealthChecker | None = None,
) -> TrainerCell:
    # [FUNCTION DOC] Builds an actor-role trainer cell for the given cell index in the actor pool,
    # with mock args, no reference model, workers hash pseudo-hash-1, the given health checker
    # (default a no-op checker) and a provider over the fake worker manager. Side effect: sets the
    # fake manager's actors-per-cell count to the given actor count (default 2) before building,
    # so the cell observes that many dummy actors.
    fake_worker_manager.actor_count_per_cell = actor_count
    return TrainerCell(
        args=MagicMock(),
        role="actor",
        with_ref=False,
        cell_id=compute_cell_id(pool_id="trainer-engine-actor", cell_index=cell_index),
        cell_index=cell_index,
        workers_hash="pseudo-hash-1",
        health_checker=health_checker if health_checker is not None else NoopHealthChecker(),
        provider=make_provider(),
    )


def make_alive_cell(cell_index: int, *, alive_cell_indices: list[int], quorum_id: int = 0) -> TrainerCell:
    # [FUNCTION DOC] Builds a default cell (see the cell factory) for the cell index and marks it
    # alive directly with an independent dp info made from the given alive cell indices and quorum
    # id (default 0), without dispatching any init call to the workers.
    """Create a cell and transition it to Alive state."""
    cell = make_cell(cell_index)
    cell._mark_as_alive(
        indep_dp_info=make_indep_dp_info(
            cell_index=cell_index,
            alive_cell_indices=alive_cell_indices,
            quorum_id=quorum_id,
        )
    )
    return cell
