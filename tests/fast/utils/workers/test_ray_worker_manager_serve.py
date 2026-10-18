# [MODULE DOC] Tests for RayWorkerManager with serve worker specs (workers named by class import
# path instead of a shell command) on the fake ray cluster. Covers: how the worker class is
# wrapped in a bootstrapped actor class that evaluates the spec's constructor-argument recipe
# inside the actor with the rank's launch context; scheduling options (cpus, gpu ids, environment,
# concurrency groups); validation and annotation of concurrency groups derived from rpc method
# markers (declared once, outermost marker wins, original class never mutated); stop behavior
# (kill without graceful shutdown); the rpc communication mode (a serve actor hosting an rpc
# server on an allocated port, worker class path reported so clients can be built); and the
# cloudpickle round trip, including rebuilding the class in a fresh interpreter.
from __future__ import annotations

import functools
import inspect
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from ray import cloudpickle
from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.fake_ray import EVENT_KILL, FakeRayCluster

from miles.ray.placement_group import PlacementGroupInfo
from miles.utils.workers import ray_worker_manager as rwm
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager, _build_serve_worker, bootstrapped_worker_class
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs, rpc
from miles.utils.workers.serving.serve_actor import ServeActor
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_provider.utils import build_rpc_handle_of_worker_info
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec, WorkerLaunchContext

pytestmark = pytest.mark.asyncio


def _passthrough(fn):
    # [FUNCTION DOC] HELPER: Decorator that wraps a function in a thin wrapper preserving its
    # metadata and forwarding all arguments and the return value; used to test markers declared
    # above a wrapper.
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the wrapped function with the same arguments and returns its
        # result.
        return fn(*args, **kwargs)

    return wrapper


class _GroupedWorker:
    # [CLASS DOC] HELPER: Fake worker with rpc methods in different concurrency groups: isolated
    # (group kill_self), wrapped_isolated (group fault_injector declared above a wrapper),
    # outer_declaration_wins (two markers, outer kill_self over inner heartbeat_status),
    # isolated_echo (kill_self, returns the text repeated) and an unmarked plain method.
    # [FUNCTION DOC] HELPER: Rpc method with no body, declared in concurrency group kill_self.
    @rpc(concurrency_group="kill_self")
    def isolated(self) -> None: ...

    # [FUNCTION DOC] HELPER: Rpc method with no body, declared in group fault_injector above a
    # pass-through wrapper, so the group must be found through the wrapper.
    @rpc(concurrency_group="fault_injector")
    @_passthrough
    def wrapped_isolated(self) -> None: ...

    # [FUNCTION DOC] HELPER: Rpc method with no body carrying two group markers around a
    # pass-through wrapper; the outer one (kill_self) must win over the inner (heartbeat_status).
    @rpc(concurrency_group="kill_self")
    @_passthrough
    @rpc(concurrency_group="heartbeat_status")
    def outer_declaration_wins(self) -> None: ...

    @rpc(concurrency_group="kill_self")
    def isolated_echo(self, value: str, *, times: int = 1) -> str:
        # [FUNCTION DOC] HELPER: Returns the text repeated the given number of times (default 1).
        return value * times

    # [FUNCTION DOC] HELPER: Method with no body and no rpc marker (no concurrency group).
    def plain(self) -> None: ...


class DemoServeWorker:
    # [CLASS DOC] HELPER: Fake serve worker taking only keyword arguments (kept in kwargs), with a
    # ping method and an echo method. Imported by path from a fresh interpreter in one test.
    def __init__(self, **kwargs):
        # [FUNCTION DOC] HELPER: Stores the keyword arguments on the instance as a mapping.
        self.kwargs = kwargs

    def ping(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the text "pong".
        return "pong"

    def echo(self, value: str, *, times: int = 1) -> str:
        # [FUNCTION DOC] HELPER: Returns the text repeated the given number of times (default 1).
        return value * times


_WORKER_CLASS_PATH = f"{DemoServeWorker.__module__}.{DemoServeWorker.__qualname__}"
_GROUPED_WORKER_CLASS_PATH = f"{_GroupedWorker.__module__}.{_GroupedWorker.__qualname__}"
_GROUPED_WORKER_GROUPS = {"kill_self": 1, "fault_injector": 1, "default": 1}

_REPO_ROOT = Path(__file__).resolve().parents[4]

_REBUILD_IN_CHILD = """
import json
import sys

from ray import cloudpickle

from tests.fast.utils.workers.test_ray_worker_manager_serve import DemoServeWorker

actor_class, ctor_kwargs, context = cloudpickle.loads(sys.stdin.buffer.read())
worker = actor_class(ctor_kwargs=ctor_kwargs, context=context)
print(json.dumps(dict(**worker.kwargs, rebuilt=actor_class is not DemoServeWorker, name=actor_class.__name__)))
"""


def _make_spec(
    name: str = "trainer",
    *,
    num_cells: int = 1,
    num_workers_per_cell: int = 1,
    ctor_kwargs=None,
    concurrency_groups: dict[str, int] | None = None,
    num_gpus_per_worker: float = 0,
    num_cpus_per_worker: float = 0.2,
    num_gpu_slots_per_worker: int = 0,
    pg_name: str | None = None,
    env_var=None,
    worker_class: str = _WORKER_CLASS_PATH,
) -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a serve worker spec (default name trainer) with the given cell
    # shape, gpu fraction, cpu request (default 0.2), gpu slots, placement group name, environment
    # callable (default empty), worker class path (default DemoServeWorker), constructor-argument
    # callable (default returning empty) and optional concurrency groups. One master-mode dynamic
    # port named master is declared.
    return ServeWorkerSpec(
        name=name,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master", allow_dynamic=True)],
        env_var=env_var if env_var is not None else (lambda _ctx: {}),
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=num_gpus_per_worker,
            num_cpus_per_worker=num_cpus_per_worker,
            num_gpu_slots_per_worker=num_gpu_slots_per_worker,
            pg_name=pg_name,
        ),
        worker_class=worker_class,
        ctor_kwargs=ctor_kwargs if ctor_kwargs is not None else (lambda _ctx: {}),
        concurrency_groups=concurrency_groups,
    )


@dataclass
class _CtorKwargsProbe:
    # [CLASS DOC] HELPER: Constructor-argument recipe that records the contexts it is called with.
    contexts: list[Any] = field(default_factory=list)

    def __call__(self, context: Any) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Records the context and returns a mapping with the worker-in-cell
        # index as rank and the fixed role "actor".
        self.contexts.append(context)
        return dict(rank=context.worker_in_cell_index, role="actor")


class _RecordingCapability(BackendCapability):
    # [CLASS DOC] HELPER: Backend capability that records which pool ids were requested for dynamic
    # and static providers and returns plain objects.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a plain object as the operations handle and two empty lists
        # for the requested dynamic pool id lists and requested static pool ids.
        self.operations = object()
        self.requested_pool_ids: list[list[str]] = []
        self.requested_static_pool_ids: list[str] = []

    def dynamic_worker_provider(self, *, pool_ids):
        # [FUNCTION DOC] HELPER: Records a copy of the pool ids and returns a plain object.
        self.requested_pool_ids.append(list(pool_ids))
        return object()

    def static_worker_provider(self, *, pool_id: str):
        # [FUNCTION DOC] HELPER: Records the pool id and returns a plain object.
        self.requested_static_pool_ids.append(pool_id)
        return object()

    def cell_operations(self):
        # [FUNCTION DOC] HELPER: Returns the single operations object created at construction.
        return self.operations


def _launch_context(*, worker_in_cell_index: int = 0) -> WorkerLaunchContext:
    # [FUNCTION DOC] HELPER: Builds a launch context for cell 0 with the given worker index
    # (default 0) and no gpu ids.
    return WorkerLaunchContext(cell_index=0, worker_in_cell_index=worker_in_cell_index, gpu_ids=[])


def _make_pgs(num_slots: int = 8) -> dict[str, PlacementGroupInfo]:
    # [FUNCTION DOC] HELPER: Builds a mapping with one placement group named actor having
    # identity bundle indices and gpu ids 0 to num_slots minus one (default 8 slots).
    return {
        "actor": PlacementGroupInfo(
            pg="fake-pg",
            pg_reordered_bundle_indices=list(range(num_slots)),
            pg_reordered_gpu_ids=list(range(num_slots)),
        )
    }


async def _launch(specs, pgs=None, comm_backend: WorkerCommBackend = WorkerCommBackend.RAY) -> RayWorkerManager:
    # [FUNCTION DOC] HELPER: Creates a manager and runs init with the specs, the placement groups
    # (empty by default) and the communication backend (ray by default); returns the manager.
    manager = RayWorkerManager()
    await manager.init(worker_manager_args(), specs, pgs if pgs is not None else {}, comm_backend=comm_backend)
    return manager


def _actor_classes(cluster: FakeRayCluster) -> list[type]:
    # [FUNCTION DOC] HELPER: Returns the actor class of each fake handle, in creation order.
    return [handle.actor_class for handle in cluster.handles]


def _options(cluster: FakeRayCluster) -> list[dict]:
    # [FUNCTION DOC] HELPER: Returns the creation options of each fake handle, in creation order.
    return [handle.options for handle in cluster.handles]


class TestServeWorkersAreLaunched:
    # [CLASS DOC] Tests of the basic launch of serve workers.
    async def test_the_declared_worker_class_is_instantiated(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Two workers; each created actor class is a subclass of the declared worker
        # class.
        """A serve spec names its worker class instead of running a shell command."""
        await _launch([_make_spec(num_workers_per_cell=2)])

        assert [issubclass(cls, DemoServeWorker) for cls in _actor_classes(fake_ray_cluster)] == [True, True]

    async def test_no_launch_command_is_ever_sent(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] No run call is made on any actor.
        """Serve workers start with their constructor, so post_setup must stay silent."""
        await _launch([_make_spec()])

        assert fake_ray_cluster.calls_of("run") == []

    async def test_the_manager_never_evaluates_the_ctor_kwargs_of_a_spec(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A recipe that fails an assertion if called; launching 2 workers succeeds
        # and creates 2 handles, proving the recipe is not evaluated in the manager.
        """ctor kwargs may hold a live provider, which cannot be shipped from here to the actor."""

        def explode(_ctx) -> dict:
            # [FUNCTION DOC] HELPER: Recipe that always fails an assertion.
            raise AssertionError("ctor kwargs were computed in the manager process")

        await _launch([_make_spec(num_workers_per_cell=2, ctor_kwargs=explode)])

        assert len(fake_ray_cluster.handles) == 2

    async def test_each_worker_is_handed_the_spec_s_ctor_kwargs_fn_and_its_own_launch_context(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] 3 workers with a probe recipe; each actor is constructed with exactly the
        # keyword arguments recipe and context, the recipe is the probe itself, and the contexts'
        # worker indices are 0, 1, 2.
        """What ships is the recipe and the rank's identity; the evaluated kwargs never leave the actor."""
        probe = _CtorKwargsProbe()
        await _launch([_make_spec(num_workers_per_cell=3, ctor_kwargs=probe)])

        assert [list(kwargs) for kwargs in fake_ray_cluster.ctor_kwargs] == [["ctor_kwargs", "context"]] * 3
        assert all(kwargs["ctor_kwargs"] is probe for kwargs in fake_ray_cluster.ctor_kwargs)
        assert [kwargs["context"].worker_in_cell_index for kwargs in fake_ray_cluster.ctor_kwargs] == [0, 1, 2]

    async def test_gpu_ids_reach_the_actor(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] 2 workers with one gpu slot each in the actor group; the contexts given to
        # the actors carry gpu ids [0] and [1].
        """A serve worker cannot ask ray for its slot, so the manager must tell it."""
        spec = _make_spec(
            num_workers_per_cell=2,
            num_gpu_slots_per_worker=1,
            num_gpus_per_worker=0.4,
            pg_name="actor",
        )

        await _launch([spec], _make_pgs())

        assert [kwargs["context"].gpu_ids for kwargs in fake_ray_cluster.ctor_kwargs] == [[0], [1]]

    async def test_env_vars_are_computed_per_worker(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] An environment callable producing a per-rank directory; the two actors'
        # runtime environment variables are the rank 0 and rank 1 values.
        """Per-rank paths such as the offload directory live in the runtime env."""
        spec = _make_spec(num_workers_per_cell=2, env_var=lambda ctx: {"RANK_DIR": f"/d/{ctx.worker_in_cell_index}"})

        await _launch([spec])

        env_vars = [options["runtime_env"]["env_vars"] for options in _options(fake_ray_cluster)]
        assert env_vars == [{"RANK_DIR": "/d/0"}, {"RANK_DIR": "/d/1"}]


class TestServeWorkerClassFailures:
    # [CLASS DOC] Tests of a worker class path that cannot be imported.
    async def test_an_unloadable_worker_class_rolls_back_the_serve_cell(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A spec naming a nonexistent class; init raises an exception mentioning the
        # class name, no actor is created and the cell is not alive.
        """A cell left alive around a class that cannot be imported would never be retried nor serve."""
        spec = _make_spec(worker_class=f"{_WORKER_CLASS_PATH}Missing")
        manager = RayWorkerManager()

        with pytest.raises(Exception, match="DemoServeWorkerMissing"):
            await manager.init(worker_manager_args(), [spec], {}, comm_backend=WorkerCommBackend.RAY)

        assert fake_ray_cluster.handles == []
        assert not manager.get_cell_infos(pool_ids=["trainer"])["trainer-00000"].alive


class TestServeSchedulingOptions:
    # [CLASS DOC] Tests of actor creation options and class annotation for serve workers.
    async def test_concurrency_groups_reach_ray(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The declared concurrency groups appear unchanged in the actor options.
        """The trainer heartbeat rpc must not queue behind a running train step."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH, concurrency_groups=_GROUPED_WORKER_GROUPS)])

        assert _options(fake_ray_cluster)[0]["concurrency_groups"] == _GROUPED_WORKER_GROUPS

    async def test_absent_concurrency_groups_are_not_passed_to_ray(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Without groups the actor options have no concurrency-groups key.
        """Passing an empty group mapping would change how ray schedules the actor."""
        await _launch([_make_spec()])

        assert "concurrency_groups" not in _options(fake_ray_cluster)[0]

    async def test_the_routed_methods_are_annotated_on_a_subclass(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] With groups, the actor class differs from the plain bootstrapped class and
        # its isolated method carries the kill_self concurrency group marker.
        """A declared group nobody is routed to leaves the isolated rpc queued behind the default group."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH, concurrency_groups=_GROUPED_WORKER_GROUPS)])

        actor_class = _actor_classes(fake_ray_cluster)[0]
        assert actor_class is not bootstrapped_worker_class(_GROUPED_WORKER_CLASS_PATH)
        assert actor_class.isolated.__ray_concurrency_group__ == "kill_self"

    async def test_a_worker_without_groups_reaches_ray_unannotated(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Without groups the actor class is the plain bootstrapped class and its ping
        # method carries no group marker.
        """Ray refuses to build an actor whose method names a group the class never declares."""
        await _launch([_make_spec()])

        actor_class = _actor_classes(fake_ray_cluster)[0]
        assert actor_class is bootstrapped_worker_class(_WORKER_CLASS_PATH)
        assert not hasattr(actor_class.ping, "__ray_concurrency_group__")

    async def test_the_cpu_request_comes_from_the_spec(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A cpu request of 0.4 per worker appears as the actor's cpu option.
        """Trainer actors reserve a whole slot, unlike the small command workers."""
        await _launch([_make_spec(num_cpus_per_worker=0.4)])

        assert _options(fake_ray_cluster)[0]["num_cpus"] == 0.4


class TestServeConcurrencyGroupRouting:
    # [CLASS DOC] Tests that group annotation happens on a derived class, per method, without
    # mutating the declared class.
    async def test_the_declared_worker_class_stays_unannotated(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] After a launch with groups the original grouped worker's isolated method
        # has no group marker.
        """Annotating the class itself would follow every later non-fault-tolerant run of that class."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH, concurrency_groups=_GROUPED_WORKER_GROUPS)])

        assert not hasattr(_GroupedWorker.isolated, "__ray_concurrency_group__")

    async def test_a_later_launch_without_groups_gets_a_class_no_earlier_launch_annotated(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] Launches the grouped worker with groups, then a second spec for the same
        # class without groups; the original class, the second actor class and the unwrapped
        # wrapped method of the second class all have no group marker.
        """Ray rejects a plain actor whose method still names a group, so a fault-tolerant run must leave no mark."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH, concurrency_groups=_GROUPED_WORKER_GROUPS)])
        await _launch([_make_spec(name="plain-trainer", worker_class=_GROUPED_WORKER_CLASS_PATH)])

        plain_actor_class = _actor_classes(fake_ray_cluster)[-1]
        assert not hasattr(_GroupedWorker.isolated, "__ray_concurrency_group__")
        assert not hasattr(plain_actor_class.isolated, "__ray_concurrency_group__")
        assert not hasattr(inspect.unwrap(plain_actor_class.wrapped_isolated), "__ray_concurrency_group__")

    async def test_each_routed_method_lands_in_its_own_group(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The isolated method is in kill_self and the wrapped one in fault_injector.
        """Collapsing every routed method into one group serializes the heartbeat with the fault injector."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH, concurrency_groups=_GROUPED_WORKER_GROUPS)])

        actor_class = _actor_classes(fake_ray_cluster)[0]
        assert (
            actor_class.isolated.__ray_concurrency_group__,
            actor_class.wrapped_isolated.__ray_concurrency_group__,
        ) == ("kill_self", "fault_injector")

    async def test_a_routed_method_still_runs_the_original_body(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Builds an instance of the actor class directly (recipe returning nothing)
        # and asserts the echo method with arguments returns the repeated text "abab".
        """A wrapper that swallowed the arguments or the return value would break every isolated rpc."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH, concurrency_groups=_GROUPED_WORKER_GROUPS)])

        worker = _actor_classes(fake_ray_cluster)[0](ctor_kwargs=lambda _ctx: {}, context=_launch_context())
        assert worker.isolated_echo("ab", times=2) == "abab"


class TestConcurrencyGroupsAreValidatedAtLaunch:
    # [CLASS DOC] Tests of consistency checks between declared groups and method markers.
    async def test_groups_without_an_annotated_method_are_rejected(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Groups declared for a class with no marked method; init raises an assertion
        # error containing "buys nothing" and creates no actor.
        """Threading the actor while every method stays in the default group buys nothing."""
        spec = _make_spec(concurrency_groups={"heartbeat_status": 1, "default": 1})
        manager = RayWorkerManager()

        with pytest.raises(AssertionError, match="buys nothing"):
            await manager.init(worker_manager_args(), [spec], {}, comm_backend=WorkerCommBackend.RAY)

        assert fake_ray_cluster.handles == []

    async def test_a_method_annotated_with_an_undeclared_group_is_rejected(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Only the default group is declared for the grouped worker; init raises an
        # assertion error naming the worker (trainer) and the missing groups fault_injector and
        # kill_self (sorted).
        """Ray rejects the actor at creation time; the message must name the worker and the missing groups."""
        spec = _make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH, concurrency_groups={"default": 1})
        manager = RayWorkerManager()

        with pytest.raises(AssertionError, match=r"'trainer'.*\['fault_injector', 'kill_self'\]"):
            await manager.init(worker_manager_args(), [spec], {}, comm_backend=WorkerCommBackend.RAY)

    async def test_a_declared_group_nobody_is_annotated_with_is_allowed(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] An extra unused group named spare is accepted and passed to the actor
        # options along with the others.
        """The trainer declares a default group precisely because no method is routed to it."""
        groups = {**_GROUPED_WORKER_GROUPS, "spare": 1}

        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH, concurrency_groups=groups)])

        assert _options(fake_ray_cluster)[0]["concurrency_groups"] == groups


class TestConcurrencyGroupsAreDeclaredOnce:
    # [CLASS DOC] Tests that the group written on an rpc method marker is the single source for both
    # the rpc server and ray.
    async def test_the_group_an_rpc_method_declares_reaches_ray(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The isolated method's group on the actor class is kill_self.
        """A method both wires isolate is declared once, and the launcher is what tells ray about it."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH, concurrency_groups=_GROUPED_WORKER_GROUPS)])

        assert _actor_classes(fake_ray_cluster)[0].isolated.__ray_concurrency_group__ == "kill_self"

    async def test_a_group_declared_above_a_wrapper_still_reaches_ray(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The wrapped method's group on the actor class is fault_injector.
        """A group read off the wrapper alone would leave the wrapped method silently in the default group."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH, concurrency_groups=_GROUPED_WORKER_GROUPS)])

        assert _actor_classes(fake_ray_cluster)[0].wrapped_isolated.__ray_concurrency_group__ == "fault_injector"

    async def test_a_default_group_method_is_left_undeclared(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The unmarked plain method has no group marker and is the very same function
        # object as on the declared class.
        """Ray rejects an actor naming a group its class never declares, and most methods name none."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH, concurrency_groups=_GROUPED_WORKER_GROUPS)])

        actor_class = _actor_classes(fake_ray_cluster)[0]
        assert not hasattr(actor_class.plain, "__ray_concurrency_group__")
        assert actor_class.plain is _GroupedWorker.plain

    async def test_both_wires_end_up_with_the_same_group(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] For every rpc method spec collected from the declared class, the group on
        # the ray actor class (default when unmarked) equals the group in the rpc spec.
        """This is the whole point of declaring once: the two wires must not schedule a method differently."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH, concurrency_groups=_GROUPED_WORKER_GROUPS)])

        actor_class = _actor_classes(fake_ray_cluster)[0]
        specs = collect_rpc_method_specs(_GroupedWorker)
        told_to_ray = {
            name: getattr(getattr(actor_class, name), "__ray_concurrency_group__", "default") for name in specs
        }

        assert told_to_ray == {name: spec.concurrency_group for name, spec in specs.items()}

    async def test_the_outermost_declaration_is_the_one_ray_hears(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] For the method with two markers, both the ray actor class and the rpc spec
        # report the outer group kill_self.
        """Two markers on one method must not resolve differently per wire, whichever one is meant to win."""
        await _launch([_make_spec(worker_class=_GROUPED_WORKER_CLASS_PATH, concurrency_groups=_GROUPED_WORKER_GROUPS)])

        actor_class = _actor_classes(fake_ray_cluster)[0]
        assert actor_class.outer_declaration_wins.__ray_concurrency_group__ == "kill_self"
        assert collect_rpc_method_specs(_GroupedWorker)["outer_declaration_wins"].concurrency_group == "kill_self"


class TestServeWorkersAreStopped:
    # [CLASS DOC] Tests of stopping serve cells.
    async def test_stopping_kills_the_actor_without_a_graceful_shutdown(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Stopping the cell makes no shutdown call and exactly one kill.
        """Serve workers expose no shutdown rpc, so asking for one only logs noise."""
        manager = await _launch([_make_spec()])

        await manager.stop_cells(["trainer-00000"])

        assert fake_ray_cluster.calls_of("shutdown") == []
        assert fake_ray_cluster.events.count(EVENT_KILL) == 1


class TestServeAndCommandSpecsCoexist:
    # [CLASS DOC] Tests that serve cells go through the common port allocation.
    async def test_ports_are_allocated_for_serve_cells_too(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A 2-worker serve cell: worker 0 lists the master port and worker 1 does not.
        # KNOWN ISSUE: the class name speaks of command specs coexisting, but only a serve spec is
        # launched here.
        """The trainer master port is allocated by the same path as engine ports."""
        manager = await _launch([_make_spec(num_workers_per_cell=2)])

        addrs = manager.get_addrs()["trainer"]
        assert "master" in addrs[0]
        assert "master" not in addrs[1]


class TestTheBootstrappedClass:
    # [CLASS DOC] Tests of the class built by the bootstrapped-worker-class function: it wraps the
    # worker class, evaluates the recipe with the context, offers a lazily created backend
    # capability, keeps the worker's name and module, and survives pickling.
    async def test_evaluates_the_handed_ctor_kwargs_with_the_handed_context(self):
        # [FUNCTION DOC] Instantiating with a probe recipe and a context for worker index 2 calls the
        # recipe once with a context whose worker index is 2.
        """The class captures nothing: the recipe and the identity both arrive as constructor arguments."""
        probe = _CtorKwargsProbe()
        actor_class = bootstrapped_worker_class(_WORKER_CLASS_PATH)

        actor_class(ctor_kwargs=probe, context=_launch_context(worker_in_cell_index=2))

        assert probe.contexts[0].worker_in_cell_index == 2

    async def test_builds_the_context_with_a_backend_capability_of_its_own_process(self, monkeypatch):
        # [FUNCTION DOC] Replaces the module's ray backend capability factory with one returning a
        # recording capability; after instantiation the context given to the recipe exposes cell
        # operations that are that capability's operations object.
        """A spec that asks for its engines is answered by the backend this process sees, not the launcher's."""
        built = _RecordingCapability()
        monkeypatch.setattr(rwm, "_create_ray_backend_capability", lambda: built)
        probe = _CtorKwargsProbe()
        actor_class = bootstrapped_worker_class(_WORKER_CLASS_PATH)

        actor_class(ctor_kwargs=probe, context=_launch_context())

        assert probe.contexts[0].capability.cell_operations() is built.operations

    async def test_the_capability_costs_nothing_until_the_spec_asks(self, monkeypatch):
        # [FUNCTION DOC] The capability factory counts creations. After instantiation none has
        # happened; after the recipe's context is asked for cell operations and a dynamic provider,
        # exactly one capability was created (created once, lazily).
        """Reaching for the worker manager at construction time would make every gpu-less worker pay for it."""
        creations: list[str] = []

        def _create():
            # [FUNCTION DOC] HELPER: Records a creation and returns a recording capability.
            creations.append("created")
            return _RecordingCapability()

        monkeypatch.setattr(rwm, "_create_ray_backend_capability", _create)
        probe = _CtorKwargsProbe()
        actor_class = bootstrapped_worker_class(_WORKER_CLASS_PATH)

        actor_class(ctor_kwargs=probe, context=_launch_context())
        assert creations == []

        capability = probe.contexts[0].capability
        capability.cell_operations()
        capability.dynamic_worker_provider(pool_ids=["trainer-engine-actor"])

        assert creations == ["created"]

    async def test_the_capability_forwards_what_the_spec_asked_for(self, monkeypatch):
        # [FUNCTION DOC] Asks the context's capability for a dynamic provider with one pool id and a
        # static provider for one pool; the recording capability saw exactly those ids.
        """The pool ids a spec names are the ones its provider must watch; dropping them would watch everything."""
        built = _RecordingCapability()
        monkeypatch.setattr(rwm, "_create_ray_backend_capability", lambda: built)
        probe = _CtorKwargsProbe()
        actor_class = bootstrapped_worker_class(_WORKER_CLASS_PATH)

        actor_class(ctor_kwargs=probe, context=_launch_context())
        capability = probe.contexts[0].capability
        capability.dynamic_worker_provider(pool_ids=["trainer-engine-actor"])
        capability.static_worker_provider(pool_id="rollout-executor")

        assert built.requested_pool_ids == [["trainer-engine-actor"]]
        assert built.requested_static_pool_ids == ["rollout-executor"]

    async def test_passes_the_computed_keywords_to_the_wrapped_constructor(self):
        # [FUNCTION DOC] The wrapped worker's keyword arguments equal the recipe's result (rank 2,
        # role actor).
        """The worker class is keyword-only, exactly as it is when a pod builds it in serve_inner."""
        actor_class = bootstrapped_worker_class(_WORKER_CLASS_PATH)

        worker = actor_class(ctor_kwargs=_CtorKwargsProbe(), context=_launch_context(worker_in_cell_index=2))

        assert worker.kwargs == dict(rank=2, role="actor")

    async def test_keeps_the_name_of_the_class_it_wraps(self):
        # [FUNCTION DOC] The built class has the wrapped class's name and module.
        """Ray names actors and their errors after the class, and 'BootstrappedWorker' would name them all alike."""
        assert bootstrapped_worker_class(_WORKER_CLASS_PATH).__name__ == DemoServeWorker.__name__
        assert bootstrapped_worker_class(_WORKER_CLASS_PATH).__module__ == DemoServeWorker.__module__

    async def test_survives_cloudpickle_together_with_the_spec_s_ctor_kwargs(self):
        # [FUNCTION DOC] Round-trips the class and a probe recipe through cloudpickle, instantiates
        # with worker index 1 and asserts the keyword arguments are rank 1, role actor.
        """Ray ships the class and the constructor arguments to the actor process, recipe included."""
        actor_class = bootstrapped_worker_class(_WORKER_CLASS_PATH)

        rebuilt_class, rebuilt_probe = cloudpickle.loads(cloudpickle.dumps((actor_class, _CtorKwargsProbe())))
        worker = rebuilt_class(ctor_kwargs=rebuilt_probe, context=_launch_context(worker_in_cell_index=1))

        assert worker.kwargs == dict(rank=1, role="actor")

    async def test_a_fresh_interpreter_rebuilds_the_class_and_still_evaluates_the_recipe(self):
        # [FUNCTION DOC] Pickles the class, a probe recipe and a context for worker 1, and feeds it
        # to a fresh python process (repository root on the path) running a snippet that unpickles,
        # constructs and prints a JSON summary. Asserts a zero return code and that the last output
        # line shows rank 1, role actor, a rebuilt class distinct from the original and the name
        # DemoServeWorker.
        """Unpickling in this process hands back the very same class, so only another interpreter rebuilds it."""
        payload = cloudpickle.dumps(
            (
                bootstrapped_worker_class(_WORKER_CLASS_PATH),
                _CtorKwargsProbe(),
                _launch_context(worker_in_cell_index=1),
            )
        )

        completed = subprocess.run(
            [sys.executable, "-c", _REBUILD_IN_CHILD],
            input=payload,
            capture_output=True,
            env=dict(
                os.environ,
                PYTHONPATH=os.pathsep.join(filter(None, [str(_REPO_ROOT), os.environ.get("PYTHONPATH", "")])),
            ),
        )

        assert completed.returncode == 0, completed.stderr.decode()
        rebuilt = json.loads(completed.stdout.decode().strip().splitlines()[-1])
        assert rebuilt == dict(rank=1, role="actor", rebuilt=True, name="DemoServeWorker")


class DemoRpcServeWorker:
    # [CLASS DOC] HELPER: Fake serve worker for the rpc mode; keeps its keyword arguments.
    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Stores the keyword arguments on the instance as a mapping.
        self.kwargs = kwargs

    def echo(self, *, value: int) -> int:
        # [FUNCTION DOC] HELPER: Returns the given integer unchanged.
        return value


_RPC_WORKER_CLASS_PATH = f"{DemoRpcServeWorker.__module__}.{DemoRpcServeWorker.__qualname__}"


async def _launch_rpc(spec: ServeWorkerSpec) -> RayWorkerManager:
    # [FUNCTION DOC] HELPER: Launches the single spec with the rpc communication backend and
    # returns the manager.
    return await _launch([spec], comm_backend=WorkerCommBackend.RPC)


class TestServeWorkersUnderRpcComm:
    # [CLASS DOC] Tests of serve cells launched with the rpc communication backend.
    async def test_the_actor_is_a_serve_actor_rather_than_the_worker_itself(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The only created actor class is the serve actor.
        """The worker still lives in the actor process, but the actor is the one running the rpc server."""
        await _launch_rpc(_make_spec())

        assert _actor_classes(fake_ray_cluster) == [ServeActor]

    async def test_the_worker_recipe_is_handed_to_the_serve_actor(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The actor is constructed with a single keyword argument, build_worker, and
        # the spec's recipe was not evaluated in the launcher.
        """Evaluating the recipe here would build the worker in the launcher instead of on its gpu."""
        probe = _CtorKwargsProbe()
        await _launch_rpc(_make_spec(ctor_kwargs=probe))

        (ctor_kwargs,) = fake_ray_cluster.ctor_kwargs
        assert list(ctor_kwargs) == ["build_worker"]
        assert probe.contexts == []

    async def test_the_recipe_builds_the_declared_worker_when_the_actor_runs_it(self):
        # [FUNCTION DOC] Builds a partial of the serve-worker builder (class path, probe recipe,
        # context for worker 1), round-trips it through cloudpickle and calls it; asserts the result
        # is a DemoServeWorker with rank 1, role actor.
        """What ships must rebuild the very class the spec names, with the rank's own identity."""
        probe = _CtorKwargsProbe()

        worker = cloudpickle.loads(
            cloudpickle.dumps(
                partial(
                    _build_serve_worker,
                    worker_class_path=_WORKER_CLASS_PATH,
                    ctor_kwargs=probe,
                    context=_launch_context(worker_in_cell_index=1),
                )
            )
        )()

        assert isinstance(worker, DemoServeWorker) and worker.kwargs == dict(rank=1, role="actor")

    async def test_the_server_starts_on_the_allocated_rpc_port(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] One start-rpc-server call, made with the port equal to the worker's rpc
        # address port from the manager.
        """The port is only known after allocation, so the server can only be told about it afterwards."""
        manager = await _launch_rpc(_make_spec())

        (call,) = fake_ray_cluster.calls_of("start_rpc_server")
        assert call.kwargs == dict(port=manager.get_addrs()["trainer"][0]["rpc"].port)

    async def test_the_server_starts_after_the_port_is_allocated(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] In the event log the last free-port-block request precedes the first
        # start-rpc-server call.
        """Starting first would bind a port nobody advertised, or none at all."""
        await _launch_rpc(_make_spec())

        assert fake_ray_cluster.last_event_index("_get_free_port_block") < fake_ray_cluster.first_event_index(
            "start_rpc_server"
        )

    async def test_every_worker_of_a_cell_serves_its_own_port(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Two workers; the two start-rpc-server calls use two different ports.
        """Two workers on one node would otherwise be handed the same port and one would fail to bind."""
        await _launch_rpc(_make_spec(num_workers_per_cell=2))

        ports = [call.kwargs["port"] for call in fake_ray_cluster.calls_of("start_rpc_server")]
        assert len(set(ports)) == 2

    async def test_ray_concurrency_groups_are_not_declared(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Even when groups are declared, the actor options omit concurrency groups.
        """Under rpc the groups are the server's, and ray would schedule methods the actor does not have."""
        await _launch_rpc(_make_spec(concurrency_groups={"heartbeat_status": 1, "default": 1}))

        assert "concurrency_groups" not in _options(fake_ray_cluster)[0]

    async def test_no_server_is_started_under_ray_communication(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] With the ray backend no start-rpc-server call is made.
        """The two modes coexist, so the ray mode must keep launching the worker as the actor itself."""
        await _launch([_make_spec()])

        assert fake_ray_cluster.calls_of("start_rpc_server") == []


class TestTheHandleAWorkerIsCalledThrough:
    # [CLASS DOC] Tests of the worker description's worker-class field and the rpc handle built
    # from it.
    async def test_ray_communication_still_answers_with_a_ray_handle(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Under the ray backend the worker info has no worker class.
        """Nothing about the existing mode changes while both are supported."""
        manager = await _launch([_make_spec()])

        (info,) = manager.get_worker_infos("trainer-00000")
        assert info.worker_class is None

    async def test_rpc_communication_answers_with_the_class_to_build_a_client_for(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] Under the rpc backend the worker info's worker class is the declared class
        # path.
        """An rpc handle holds pydantic validators that no cluster can ship, so it is built by its caller."""
        manager = await _launch_rpc(_make_spec())

        (info,) = manager.get_worker_infos("trainer-00000")
        assert info.worker_class == _WORKER_CLASS_PATH

    async def test_the_provider_turns_that_answer_into_an_rpc_handle(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Builds an rpc handle from the worker info with the provider utility and
        # asserts it is an rpc worker handle.
        """The driver ends up calling the worker over http, which is the whole point of the mode."""
        manager = await _launch_rpc(_make_spec(worker_class=_RPC_WORKER_CLASS_PATH))

        (info,) = manager.get_worker_infos("trainer-00000")
        handle = build_rpc_handle_of_worker_info(info)

        assert isinstance(handle, RpcWorkerHandle)

    async def test_the_handle_points_at_the_address_the_launcher_advertised(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The handle's transport server url (a private attribute) equals the worker
        # info's rpc address.
        """A handle aimed anywhere else silently drives another worker on the same node."""
        manager = await _launch_rpc(_make_spec(worker_class=_RPC_WORKER_CLASS_PATH))

        (info,) = manager.get_worker_infos("trainer-00000")
        handle = build_rpc_handle_of_worker_info(info)

        assert handle._transport._server_url == info.self_addrs["rpc"].addr

    async def test_a_worker_that_is_not_served_names_no_class_to_call_it_by(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] With the ray backend the worker info has no worker class.
        # KNOWN ISSUE: duplicates test_ray_communication_still_answers_with_a_ray_handle, which
        # makes the same assertion on the same setup; despite its name no command worker is used.
        """Command workers are called as the actors they are, and an rpc client for them cannot be built."""
        manager = await _launch([_make_spec()])

        (info,) = manager.get_worker_infos("trainer-00000")
        assert info.worker_class is None
