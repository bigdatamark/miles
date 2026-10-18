# [MODULE DOC] Tests for RayWorkerManager with command workers, run against a fake in-memory ray
# cluster (fixture fake_ray_cluster from the shared fake_ray helper) that records actor creation,
# actor method calls, kills and ordering of events. Covered behavior: the launch entry point and
# its well-known actor name; the init pipeline (create actors, look up node addresses, allocate
# ports, probe static ports, resolve local gpu ids, start commands) with global barriers between
# phases and rollback on failure; port allocation rules (dynamic blocks, static ports, per-cell
# offsets, master-mode ports owned by worker 0 of a cell, per-node tracking, ipv6 bracketing,
# external addresses); address publication (a worker or cell is unreadable or not alive until all
# its ports are assigned); gpu and placement-group scheduling; cell lifecycle (stop, start,
# idempotence, generations and worker hashes, rollback of failed starts, bounded graceful
# shutdown followed by a kill); queries (worker infos, cell infos, addresses); and fault injection.
# Tests are async and rely on the project's asyncio test mode; the file's helpers build specs,
# placement-group info, a launch-command recorder and gates that pause port allocation.
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.fake_ray import EVENT_CREATE, EVENT_KILL, FakeRayCluster

from miles.ray.placement_group import PlacementGroupInfo
from miles.utils.workers import ray_worker_manager
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.ray_worker_manager import RayWorkerManager, _BaseActorManager, _CommandActorManager
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_spec import CommandWorkerSpec, LaunchCommandContext, PortInfo, SchedulingSpec


@dataclass
class _LaunchRecorder:
    # [CLASS DOC] HELPER: Collects the launch contexts handed to a spec's launch-command callable,
    # so tests can inspect what each worker was told (indices, addresses, gpu ids).
    contexts: list[LaunchCommandContext] = field(default_factory=list)

    def command(self, ctx: LaunchCommandContext) -> str:
        # [FUNCTION DOC] HELPER: Launch-command callable for a spec. Stores the context and returns
        # the text "run-<cell index>-<worker index>".
        self.contexts.append(ctx)
        return f"run-{ctx.cell_index}-{ctx.worker_in_cell_index}"

    def context_of(self, *, cell_index: int, worker_in_cell_index: int) -> LaunchCommandContext:
        # [FUNCTION DOC] HELPER: Returns the single recorded context for the given cell and worker
        # index; fails an assertion if zero or several match.
        matches = [
            ctx
            for ctx in self.contexts
            if ctx.cell_index == cell_index and ctx.worker_in_cell_index == worker_in_cell_index
        ]
        assert len(matches) == 1, f"{matches=}"
        return matches[0]


def _make_spec(
    name: str,
    *,
    num_cells: int = 1,
    num_workers_per_cell: int = 1,
    port_infos: list[PortInfo] | None = None,
    env_var: dict[str, str] | None = None,
    launch_command: Callable[[LaunchCommandContext], str] | None = None,
    num_gpus_per_worker: float = 0,
    num_gpu_slots_per_worker: int = 0,
    pg_name: str | None = None,
    pg_slot_offset: int = 0,
    pin_to_head: bool = False,
) -> CommandWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a command worker spec with the given name and scheduling shape
    # (cell count, workers per cell, gpu fraction and slot count, placement group name and slot
    # offset, head pinning). Defaults: one dynamic-capable port named primary with static port 8000
    # when no port list is given (an explicit empty list means no ports); an environment callable
    # returning a copy of the given variables (empty by default); a launch command returning a
    # fixed sleep command unless a callable is given.
    return CommandWorkerSpec(
        name=name,
        port_infos=(
            port_infos if port_infos is not None else [PortInfo(name="primary", static_port=8000, allow_dynamic=True)]
        ),
        env_var=lambda _ctx: dict(env_var or {}),
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=num_gpus_per_worker,
            num_gpu_slots_per_worker=num_gpu_slots_per_worker,
            pg_name=pg_name,
            pg_slot_offset=pg_slot_offset,
            pin_to_head=pin_to_head,
        ),
        launch_command=launch_command if launch_command is not None else (lambda ctx: "sleep 600"),
    )


def _make_pgs(*, num_slots: int = 8, first_gpu_id: int = 0) -> dict[str, PlacementGroupInfo]:
    # [FUNCTION DOC] HELPER: Builds a mapping with one placement group named rollout, with a fake
    # group object, a scrambled bundle-index order (slot times 3 plus 1, modulo the slot count)
    # and consecutive gpu ids starting at first_gpu_id (default 0), for num_slots slots (default 8).
    # KNOWN ISSUE: the scramble is only a permutation when the slot count is not a multiple of 3;
    # for 3, 6 or 9 slots indices repeat. Current callers use 8 or 10 slots.
    return {
        "rollout": PlacementGroupInfo(
            pg="fake-pg",
            pg_reordered_bundle_indices=[(slot * 3 + 1) % num_slots for slot in range(num_slots)],
            pg_reordered_gpu_ids=[first_gpu_id + slot for slot in range(num_slots)],
        )
    }


async def _launch(
    specs: list[CommandWorkerSpec], pgs: dict[str, PlacementGroupInfo] | None = None
) -> RayWorkerManager:
    # [FUNCTION DOC] HELPER: Creates a manager and runs its init with the shared test arguments, the
    # given specs, the given placement groups (empty mapping by default) and the ray communication
    # backend. Returns the initialised manager; init errors propagate.
    manager = RayWorkerManager()
    await manager.init(
        worker_manager_args(), specs, pgs if pgs is not None else {}, comm_backend=WorkerCommBackend.RAY
    )
    return manager


_GATE_MAX_YIELDS = 200


@dataclass
class _AllocPortsGate:
    # [CLASS DOC] HELPER: Pauses port allocation for the workers selected by a predicate until the
    # test sets the release event, and records which workers finished allocating. Events: blocked
    # is set when a selected worker reaches the gate; release lets blocked workers continue.
    should_block: Callable[[_BaseActorManager], bool]
    blocked: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    allocated_names: list[str] = field(default_factory=list)

    def install(self, patched: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] HELPER: Replaces the command actor manager's port allocation method (through
        # the given monkeypatch) with a gated wrapper; returns nothing.
        original_alloc_ports = _CommandActorManager.alloc_ports

        async def gated_alloc_ports(actor: _BaseActorManager) -> None:
            # [FUNCTION DOC] HELPER: If the predicate selects this worker, sets the blocked event and
            # waits (5 second timeout, raising a timeout error) for release. Then runs the original
            # allocation and appends the worker's name to the allocated list.
            if self.should_block(actor):
                self.blocked.set()
                await asyncio.wait_for(self.release.wait(), timeout=5)
            await original_alloc_ports(actor)
            self.allocated_names.append(actor.name)

        patched.setattr(_CommandActorManager, "alloc_ports", gated_alloc_ports)

    async def wait_until_allocated(self, *, worker_names: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Waits (5 seconds) until some worker is blocked at the gate, then
        # yields to the event loop up to 200 times until every named worker has finished allocating.
        # Raises an assertion error listing the names and the allocated ones if that never happens.
        await asyncio.wait_for(self.blocked.wait(), timeout=5)
        for _ in range(_GATE_MAX_YIELDS):
            if set(worker_names) <= set(self.allocated_names):
                return
            await asyncio.sleep(0)
        raise AssertionError(f"{worker_names=} were never allocated, only {self.allocated_names=}")


class TestLaunchEntryPoint:
    # [CLASS DOC] Tests of the manager's static launch and lookup entry points.
    async def test_the_manager_is_registered_under_its_well_known_name(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Launches with no specs; asserts the fake actor was created with the name
        # "ray_worker_manager", the actor class is the manager, and the only call made was init.
        """Consumers find the manager by a fixed actor name, so it must be launched under that name."""
        handle = RayWorkerManager.launch(worker_manager_args(), [], {}, comm_backend=WorkerCommBackend.RAY)

        assert handle.options["name"] == "ray_worker_manager"
        assert handle.actor_class is RayWorkerManager
        assert [call.method for call in fake_ray_cluster.calls] == ["init"]

    async def test_launch_waits_for_init_to_finish(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Launches with one spec; asserts init was called once with exactly the
        # arguments passed (args, specs, placement groups) and that its result reference was
        # resolved before launch returned.
        """Returning before init completes would expose a manager whose workers have no addresses yet."""
        specs = [_make_spec("router")]
        pgs: dict = {}
        args = worker_manager_args()

        RayWorkerManager.launch(args, specs, pgs, comm_backend=WorkerCommBackend.RAY)

        init_calls = fake_ray_cluster.calls_of("init")
        assert len(init_calls) == 1
        assert init_calls[0].args == (args, specs, pgs)
        assert fake_ray_cluster.resolved_refs == ["init"]

    async def test_launch_propagates_an_init_failure(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Makes the fake init call fail with a runtime error and asserts launch
        # re-raises it.
        """A pool that failed to come up must not look like a successful launch."""
        fake_ray_cluster.method_errors["init"] = RuntimeError("init exploded")

        with pytest.raises(RuntimeError, match="init exploded"):
            RayWorkerManager.launch(
                worker_manager_args(), [_make_spec("router")], {}, comm_backend=WorkerCommBackend.RAY
            )

    async def test_get_handle_resolves_the_same_well_known_name(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] After a launch, asserts get_handle returns the very handle registered under
        # the well-known name in the fake cluster.
        """The lookup helper and the launcher must agree on the actor name."""
        RayWorkerManager.launch(worker_manager_args(), [], {}, comm_backend=WorkerCommBackend.RAY)

        assert RayWorkerManager.get_handle() is fake_ray_cluster.named_actors["ray_worker_manager"]


class TestInitLaunchesWorkers:
    # [CLASS DOC] Tests of the actor-creation part of init and the phase ordering.
    async def test_creates_one_command_actor_per_worker_of_every_cell(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Specs: engine with 2 cells of 2 workers and a single-worker router. Asserts
        # 5 actors, all of the command actor class, each constructed without extra keyword args.
        """Every cell of every spec gets its own actor per worker slot."""
        await _launch([_make_spec("engine", num_cells=2, num_workers_per_cell=2), _make_spec("router")])

        assert len(fake_ray_cluster.handles) == 5
        assert {handle.actor_class for handle in fake_ray_cluster.handles} == {CommandActor}
        assert fake_ray_cluster.ctor_kwargs == [{} for _ in range(5)]

    async def test_a_spec_without_cells_launches_nothing(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A spec with zero cells creates no actor.
        """A disabled spec contributes no workers instead of an idle one."""
        await _launch([_make_spec("session-server", num_cells=0)])

        assert fake_ray_cluster.handles == []

    async def test_duplicate_pool_names_are_rejected(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Two specs with the same name make init fail with an assertion error.
        """Two specs sharing a name would collide in the worker registry, so init fails fast."""
        with pytest.raises(AssertionError):
            await _launch([_make_spec("router"), _make_spec("router")])

    async def test_the_specs_env_vars_become_the_actors_runtime_env(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A spec with one environment variable; asserts the actor's creation options
        # carry a runtime environment whose environment-variable mapping equals it.
        """The spec's env vars are handed to ray as the actor's runtime env."""
        await _launch([_make_spec("router", env_var={"MILES_TEST_VAR": "7"})])

        assert fake_ray_cluster.handles[0].options["runtime_env"] == {"env_vars": {"MILES_TEST_VAR": "7"}}

    async def test_each_phase_completes_for_all_workers_before_the_next_starts(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Asserts in the fake cluster's event log that the last actor creation
        # precedes the first node-address lookup, and the last free-port-block request precedes the
        # first command run (global barriers between phases).
        """Launching, port allocation and command start are global barriers, so every worker sees complete state."""
        await _launch([_make_spec("engine", num_cells=2, num_workers_per_cell=2), _make_spec("router")])

        assert fake_ray_cluster.last_event_index(EVENT_CREATE) < fake_ray_cluster.first_event_index("_get_node_ip")
        assert fake_ray_cluster.last_event_index("_get_free_port_block") < fake_ray_cluster.first_event_index("run")

    async def test_a_failing_phase_stops_the_pipeline(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Patches port allocation to raise a runtime error "no ports" for a 2-worker
        # spec; asserts init raises it, both actors were created, and no command was ever run.
        """A worker that cannot allocate its ports must not leave other workers starting their commands."""
        spec = _make_spec("engine", num_workers_per_cell=2)

        async def failing_alloc(self) -> None:
            # [FUNCTION DOC] HELPER: Replacement allocation that always raises a runtime error.
            raise RuntimeError("no ports")

        manager = RayWorkerManager()
        with pytest.raises(RuntimeError, match="no ports"):
            with pytest.MonkeyPatch.context() as patched:
                patched.setattr(ray_worker_manager._CommandActorManager, "alloc_ports", failing_alloc)
                await manager.init(worker_manager_args(), [spec], {}, comm_backend=WorkerCommBackend.RAY)

        assert len(fake_ray_cluster.handles) == 2
        assert fake_ray_cluster.calls_of("run") == []


class TestInitAllocatesPorts:
    # [CLASS DOC] Tests of port assignment during init: dynamic and static ports, blocks, per-cell
    # offsets, per-node tracking and the shape of advertised addresses.
    async def test_dynamic_ports_of_one_node_never_overlap(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] 2 cells of 2 workers on one node; asserts the four primary ports are
        # distinct.
        """Workers sharing a node must be handed distinct ports."""
        manager = await _launch([_make_spec("engine", num_cells=2, num_workers_per_cell=2)])

        ports = [
            manager.get_worker_addrs(
                compute_worker_name(pool_id="engine", cell_index=cell_index, worker_in_cell_index=worker_in_cell_index)
            )["primary"].port
            for cell_index in range(2)
            for worker_in_cell_index in range(2)
        ]
        assert len(set(ports)) == 4

    async def test_a_consecutive_port_block_is_reserved_as_a_whole(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Two workers each ask for a block of 5 consecutive ports; asserts the second
        # worker's port is at least 5 above the first's and both block requests asked for count 5.
        """A worker asking for a port block leaves the whole block out of the next worker's reach."""
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            port_infos=[PortInfo(name="primary", static_port=8000, allow_dynamic=True, num_consecutive=5)],
        )
        manager = await _launch([spec])

        first = manager.get_worker_addrs("engine-00000-00000")["primary"].port
        second = manager.get_worker_addrs("engine-00000-00001")["primary"].port
        assert second >= first + 5
        assert [call.kwargs["count"] for call in fake_ray_cluster.calls_of("_get_free_port_block")] == [5, 5]

    async def test_static_ports_bypass_the_allocator(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A spec with one fixed port 7777 and one dynamic port; asserts the fixed
        # port is used verbatim, only one free-port-block request happened (for the dynamic one) and
        # the fixed port was probed for availability exactly once with port 7777.
        """A port the worker cannot choose is taken from the spec verbatim, without asking the allocator."""
        spec = _make_spec(
            "router",
            port_infos=[
                PortInfo(name="primary", static_port=7777, allow_dynamic=False),
                PortInfo(name="prometheus", static_port=9000, allow_dynamic=True),
            ],
        )
        manager = await _launch([spec])

        assert manager.get_worker_addrs("router-00000-00000")["primary"].port == 7777
        assert len(fake_ray_cluster.calls_of("_get_free_port_block")) == 1
        assert [call.kwargs["port"] for call in fake_ray_cluster.calls_of("_is_port_available")] == [7777]

    async def test_a_static_port_already_taken_on_the_node_fails_the_launch(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Marks port 7777 as occupied on node 10.0.0.1 and launches a spec pinned to
        # it; expects an assertion error saying 7777 on 10.0.0.1 is already in use.
        """A stale listener on the pinned port would otherwise be mistaken for our own worker."""
        fake_ray_cluster.occupy_ports("10.0.0.1", 7777)
        spec = _make_spec("router", port_infos=[PortInfo(name="primary", static_port=7777, allow_dynamic=False)])

        with pytest.raises(AssertionError, match="7777 on 10.0.0.1 is already in use"):
            await _launch([spec])

    async def test_static_ports_marked_offset_by_cell_advance_per_cell(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] 3 cells with a fixed port 7000 flagged offset-by-cell; asserts the cells'
        # primary ports are 7000, 7001, 7002.
        """Cells sharing one pinned base port would bind the same address twice on a node."""
        spec = _make_spec(
            "session-server",
            num_cells=3,
            port_infos=[PortInfo(name="primary", static_port=7000, allow_dynamic=False, offset_by_cell=True)],
        )
        manager = await _launch([spec])

        assert [
            manager.get_worker_addrs(compute_worker_name(pool_id="session-server", cell_index=cell_index))[
                "primary"
            ].port
            for cell_index in range(3)
        ] == [7000, 7001, 7002]

    async def test_ports_are_tracked_per_node(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Two workers land on different nodes (10.0.0.1 and 10.0.0.2); asserts their
        # hosts differ accordingly while their port numbers are equal (reuse across nodes is fine).
        """Workers on different nodes may reuse the same port number."""
        fake_ray_cluster.use_node_ips("10.0.0.1", "10.0.0.2")
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])

        first = manager.get_worker_addrs("engine-00000-00000")["primary"]
        second = manager.get_worker_addrs("engine-00000-00001")["primary"]
        assert (first.host, second.host) == ("10.0.0.1", "10.0.0.2")
        assert first.port == second.port

    async def test_the_worker_addr_host_is_the_node_the_actor_landed_on(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The fake node address is 10.1.2.3; asserts the worker's advertised host
        # equals it.
        """Addresses advertise the actor's own node, not the driver's."""
        fake_ray_cluster.use_node_ips("10.1.2.3")
        manager = await _launch([_make_spec("router")])

        assert manager.get_worker_addrs("router-00000-00000")["primary"].host == "10.1.2.3"

    async def test_the_node_reported_external_address_reaches_the_worker_addr(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Two cells on two nodes; only the first node reports an external address
        # 100.64.0.1. Asserts the first worker's address has that external host and an external
        # network location of host:port, while the second's external host is unset and its external
        # network location equals its ordinary one.
        """Each worker carries the external address its own node reported; a node that
        reported none leaves the field unset and the placed address serves both views."""
        fake_ray_cluster.use_node_ips("10.0.0.1", "10.0.0.2")
        fake_ray_cluster.node_external_ips = {"10.0.0.1": "100.64.0.1"}
        manager = await _launch([_make_spec("session-server", num_cells=2)])

        with_external = manager.get_worker_addrs(compute_worker_name(pool_id="session-server", cell_index=0))[
            "primary"
        ]
        without_external = manager.get_worker_addrs(compute_worker_name(pool_id="session-server", cell_index=1))[
            "primary"
        ]
        assert with_external.external_host == "100.64.0.1"
        assert with_external.external_netloc == f"100.64.0.1:{with_external.port}"
        assert without_external.external_host is None
        assert without_external.external_netloc == without_external.netloc

    async def test_ipv6_hosts_are_bracketed(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A node reporting an ipv6 address; asserts the advertised host is that
        # address wrapped in square brackets.
        """An ipv6 node address is advertised in url-safe bracketed form."""
        fake_ray_cluster.use_node_ips("2001:db8::7")
        manager = await _launch([_make_spec("router")])

        assert manager.get_worker_addrs("router-00000-00000")["primary"].host == "[2001:db8::7]"


class TestStaticPortsAreProbedBeforeUse:
    # [CLASS DOC] Tests that fixed ports are checked for availability on the worker's own node
    # before use, which ports are checked, and that a refusal rolls the launch back.
    async def test_the_probe_asks_for_the_cell_offset_port_rather_than_the_spec_base(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] 3 cells with an offset-by-cell fixed base port 7000; asserts the probes are
        # made for ports 7000, 7001, 7002 (the offset ports, not the base three times).
        """Probing the spec's base port would leave every cell but the first checking an address it never binds."""
        spec = _make_spec(
            "session-server",
            num_cells=3,
            port_infos=[PortInfo(name="primary", static_port=7000, allow_dynamic=False, offset_by_cell=True)],
        )
        await _launch([spec])

        assert [call.kwargs["port"] for call in fake_ray_cluster.calls_of("_is_port_available")] == [7000, 7001, 7002]

    async def test_an_occupied_cell_offset_port_refuses_the_launch(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Port 7002 occupied on the node while 3 cells use offsets from 7000; expects
        # an assertion error stating port 7002 on 10.0.0.1 is already in use.
        """A stale listener on a later cell's offset port must be caught too, not only one on the base port."""
        fake_ray_cluster.occupy_ports("10.0.0.1", 7002)
        spec = _make_spec(
            "session-server",
            num_cells=3,
            port_infos=[PortInfo(name="primary", static_port=7000, allow_dynamic=False, offset_by_cell=True)],
        )

        with pytest.raises(AssertionError, match="Port 7002 on 10.0.0.1 is already in use"):
            await _launch([spec])

    async def test_the_probe_runs_on_the_node_the_worker_landed_on(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Two workers on nodes .1 and .2 with port 7777 occupied only on .2; expects
        # the assertion error naming node .2, and asserts the probes were issued through the two
        # workers' own actors, whose node addresses are .1 then .2.
        """A port is only contended on the node that binds it, so the probe must go through the worker's own actor."""
        fake_ray_cluster.use_node_ips("10.0.0.1", "10.0.0.2")
        fake_ray_cluster.occupy_ports("10.0.0.2", 7777)
        spec = _make_spec(
            "router",
            num_workers_per_cell=2,
            port_infos=[PortInfo(name="primary", static_port=7777, allow_dynamic=False)],
        )

        with pytest.raises(AssertionError, match="Port 7777 on 10.0.0.2 is already in use"):
            await _launch([spec])

        assert [call.handle.node_ip for call in fake_ray_cluster.calls_of("_is_port_available")] == [
            "10.0.0.1",
            "10.0.0.2",
        ]

    async def test_a_port_occupied_on_an_unrelated_node_does_not_block_the_launch(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] Port 7777 occupied only on 10.0.0.2 while the worker runs on 10.0.0.1; the
        # launch succeeds and the worker gets 7777.
        """Refusing a port because another machine uses it would make every run hostage to unrelated nodes."""
        fake_ray_cluster.occupy_ports("10.0.0.2", 7777)
        spec = _make_spec("router", port_infos=[PortInfo(name="primary", static_port=7777, allow_dynamic=False)])

        manager = await _launch([spec])

        assert manager.get_worker_addrs("router-00000-00000")["primary"].port == 7777

    async def test_a_master_static_port_is_probed_only_by_the_worker_that_reserves_it(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] 2 cells of 3 workers with a fixed master-mode port 9000 and a dynamic
        # primary port; asserts only two probes happened (one per cell, by worker 0), both for 9000.
        """Peers that never bind the cell's master port would otherwise refuse a port their own worker 0 is serving."""
        spec = _make_spec(
            "engine",
            num_cells=2,
            num_workers_per_cell=3,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=9000, mode="master", allow_dynamic=False),
            ],
        )
        await _launch([spec])

        assert [call.kwargs["port"] for call in fake_ray_cluster.calls_of("_is_port_available")] == [9000, 9000]

    async def test_dynamic_ports_are_never_probed_by_the_static_check(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Only dynamic ports are declared; asserts no availability probe was made.
        """The allocator already hands out a free port, and re-probing its own reservation would refuse it."""
        spec = _make_spec(
            "engine",
            num_cells=2,
            num_workers_per_cell=2,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="nccl", static_port=10000, allow_dynamic=True),
            ],
        )
        await _launch([spec])

        assert fake_ray_cluster.calls_of("_is_port_available") == []

    async def test_the_refusal_names_the_worker_and_the_endpoint_it_could_not_serve(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] An occupied fixed port for an endpoint named prometheus; expects an
        # assertion error mentioning worker router-00000-00000 and its prometheus endpoint.
        """An operator hunting the stale process needs to know which worker wanted which endpoint."""
        fake_ray_cluster.occupy_ports("10.0.0.1", 7777)
        spec = _make_spec("router", port_infos=[PortInfo(name="prometheus", static_port=7777, allow_dynamic=False)])

        with pytest.raises(AssertionError, match="router-00000-00000 cannot serve its 'prometheus' endpoint"):
            await _launch([spec])

    async def test_a_refused_port_rolls_the_cell_back_before_any_command_runs(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A 2-worker cell whose fixed port is occupied; expects an assertion error,
        # no command run, and both created actors killed.
        """A refusal that let the commands start would add our own processes to the stale ones already there."""
        fake_ray_cluster.occupy_ports("10.0.0.1", 7777)
        spec = _make_spec(
            "router",
            num_workers_per_cell=2,
            port_infos=[PortInfo(name="primary", static_port=7777, allow_dynamic=False)],
        )

        with pytest.raises(AssertionError):
            await _launch([spec])

        assert fake_ray_cluster.calls_of("run") == []
        assert [handle.killed for handle in fake_ray_cluster.handles] == [True, True]


class TestPortAllocationDetails:
    # [CLASS DOC] Finer points of port allocation: where blocks are requested, ordering of the
    # declared ports and uniqueness across specs.
    async def test_the_allocator_probes_the_workers_own_actor_on_its_own_node(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Two workers on two nodes; asserts the free-port-block requests went to the
        # workers' own actor handles, in order, on nodes .1 and .2.
        """Ports must be probed on the node that will bind them, through that worker's own actor."""
        fake_ray_cluster.use_node_ips("10.0.0.1", "10.0.0.2")
        await _launch([_make_spec("engine", num_workers_per_cell=2)])

        probes = fake_ray_cluster.calls_of("_get_free_port_block")
        assert [probe.handle for probe in probes] == fake_ray_cluster.handles
        assert [probe.handle.node_ip for probe in probes] == ["10.0.0.1", "10.0.0.2"]

    async def test_every_declared_port_is_allocated_in_spec_order(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Three dynamic ports are declared (primary, nccl, engine_info_bootstrap);
        # asserts the worker's launch context lists its addresses in that order with three distinct
        # port numbers.
        """A worker is launched with an address for every port name its spec declares."""
        recorder = _LaunchRecorder()
        spec = _make_spec(
            "engine",
            launch_command=recorder.command,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="nccl", static_port=10000, allow_dynamic=True),
                PortInfo(name="engine_info_bootstrap", static_port=12000, allow_dynamic=True),
            ],
        )
        await _launch([spec])

        addrs = recorder.context_of(cell_index=0, worker_in_cell_index=0).self_addrs
        assert list(addrs) == ["primary", "nccl", "engine_info_bootstrap"]
        assert len({addr.port for addr in addrs.values()}) == 3

    async def test_workers_of_different_specs_never_share_a_port(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A router spec with 2 workers and an engine spec with 2 cells; asserts the
        # four primary ports are all distinct.
        """One allocator serves the whole pool, so specs cannot hand out the same port twice."""
        manager = await _launch([_make_spec("router", num_workers_per_cell=2), _make_spec("engine", num_cells=2)])

        ports = [
            manager.get_worker_addrs(name)["primary"].port
            for name in ["router-00000-00000", "router-00000-00001", "engine-00000-00000", "engine-00001-00000"]
        ]
        assert len(set(ports)) == 4


class TestInitStartsCommands:
    # [CLASS DOC] Tests of the command-start phase of init.
    async def test_every_worker_runs_the_command_rendered_for_it(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] An engine with 2 cells and a recording launch command; asserts the run
        # calls carry the commands "exec run-0-0" and "exec run-1-0" (the manager prefixes exec) and
        # an empty environment mapping each.
        """Each worker's actor execs the command its own launch context rendered, so no shell sits in between."""
        recorder = _LaunchRecorder()
        await _launch([_make_spec("engine", num_cells=2, launch_command=recorder.command)])

        run_calls = fake_ray_cluster.calls_of("run")
        assert [call.kwargs["cmd"] for call in run_calls] == ["exec run-0-0", "exec run-1-0"]
        assert [call.kwargs["envs"] for call in run_calls] == [{}, {}]

    async def test_the_launch_context_carries_the_workers_own_indices_and_addrs(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] 2 cells of 2 workers; asserts the recorded contexts cover exactly the four
        # (cell, worker) index pairs and each context's own primary address equals the address the
        # manager reports for that worker.
        """A worker's launch context describes that worker: its cell, its slot and its own addresses."""
        recorder = _LaunchRecorder()
        spec = _make_spec("engine", num_cells=2, num_workers_per_cell=2, launch_command=recorder.command)
        manager = await _launch([spec])

        assert {(ctx.cell_index, ctx.worker_in_cell_index) for ctx in recorder.contexts} == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }
        for cell_index in range(2):
            for worker_in_cell_index in range(2):
                ctx = recorder.context_of(cell_index=cell_index, worker_in_cell_index=worker_in_cell_index)
                expected = manager.get_worker_addrs(
                    compute_worker_name(
                        pool_id="engine", cell_index=cell_index, worker_in_cell_index=worker_in_cell_index
                    )
                )["primary"]
                assert ctx.self_addrs["primary"] == expected


class TestSpecEnvVars:
    # [CLASS DOC] Tests of how a spec's environment callable becomes each actor's runtime env.
    async def test_each_spec_contributes_its_own_env_to_its_own_workers(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A router and a 2-cell engine with different variables; asserts the three
        # actors' runtime environments, in creation order, hold only their own spec's variables.
        """Env vars are per spec, so one spec's variables must not leak into another spec's workers."""
        await _launch(
            [
                _make_spec("router", env_var={"ROUTER_ONLY": "1"}),
                _make_spec("engine", num_cells=2, env_var={"ENGINE_ONLY": "2"}),
            ]
        )

        assert [handle.options["runtime_env"] for handle in fake_ray_cluster.handles] == [
            {"env_vars": {"ROUTER_ONLY": "1"}},
            {"env_vars": {"ENGINE_ONLY": "2"}},
            {"env_vars": {"ENGINE_ONLY": "2"}},
        ]

    async def test_the_env_of_a_spec_is_resolved_once_per_worker(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Uses a spec whose environment callable counts its calls and returns the
        # running count; with 2 cells asserts it was called exactly twice and the two actors got the
        # values "1" and "2".
        """The spec's env callable is what the manager stores, evaluated per worker rather than cached globally."""
        calls: list[int] = []

        def _env(_ctx) -> dict[str, str]:
            # [FUNCTION DOC] HELPER: Records the call and returns a mapping with the call count so
            # far as the value of a CALL_INDEX variable.
            calls.append(len(calls))
            return {"CALL_INDEX": str(len(calls))}

        spec = CommandWorkerSpec(
            name="engine",
            port_infos=[PortInfo(name="primary", static_port=8000, allow_dynamic=True)],
            env_var=_env,
            scheduling=SchedulingSpec(num_cells=2, num_workers_per_cell=1, num_gpus_per_worker=0),
            launch_command=lambda ctx: "sleep 600",
        )
        await _launch([spec])

        assert len(calls) == 2
        assert [handle.options["runtime_env"]["env_vars"]["CALL_INDEX"] for handle in fake_ray_cluster.handles] == [
            "1",
            "2",
        ]


class TestActorResources:
    # [CLASS DOC] Tests of the cpu request of command actors.
    async def test_every_worker_reserves_a_fraction_of_a_cpu(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Asserts the actor's creation options request 0.2 cpus.
        """Workers are launchers, not compute, so they must not each claim a whole cpu."""
        await _launch([_make_spec("router")])

        assert fake_ray_cluster.handles[0].options["num_cpus"] == 0.2


class TestFailureModes:
    # [CLASS DOC] Tests that an early failure prevents later init phases.
    async def test_a_failed_worker_launch_stops_before_any_port_is_allocated(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Patches actor launching to raise a runtime error "no capacity" for a
        # 2-cell spec; asserts init raises it and neither a port block request nor a command run
        # happened.
        """Nothing downstream may run when the pool could not even be created."""
        from miles.utils.workers.ray_worker_manager import _CommandActorManager

        async def failing_launch(self) -> None:
            # [FUNCTION DOC] HELPER: Replacement actor launch that always raises a runtime error.
            raise RuntimeError("no capacity")

        with pytest.raises(RuntimeError, match="no capacity"):
            with pytest.MonkeyPatch.context() as patched:
                patched.setattr(_CommandActorManager, "launch_actor", failing_launch)
                await _launch([_make_spec("engine", num_cells=2)])

        assert fake_ray_cluster.calls_of("_get_free_port_block") == []
        assert fake_ray_cluster.calls_of("run") == []

    async def test_a_command_that_cannot_be_rendered_starts_nothing(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A 2-cell spec whose launch command raises a value error; asserts init
        # raises it and no command was run on any worker.
        """A spec whose launch command raises must not leave half of the pool running."""

        def _explode(ctx: LaunchCommandContext) -> str:
            # [FUNCTION DOC] HELPER: Launch-command callable that always raises a value error.
            raise ValueError("cannot render")

        with pytest.raises(ValueError, match="cannot render"):
            await _launch([_make_spec("engine", num_cells=2, launch_command=_explode)])

        assert fake_ray_cluster.calls_of("run") == []


class TestGetWorkerAddrs:
    # [CLASS DOC] Tests of the per-worker address lookup.
    async def test_names_are_the_pool_with_the_cell_and_worker_index(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] For 2 cells of 2 workers, each name of the form pool-cell-worker (five-digit
        # zero-padded indices) resolves to an address map whose primary port is positive.
        """Workers are addressable under ``<spec>-<cell>-<worker>``."""
        manager = await _launch([_make_spec("engine", num_cells=2, num_workers_per_cell=2)])

        for name in ["engine-00000-00000", "engine-00000-00001", "engine-00001-00000", "engine-00001-00001"]:
            assert manager.get_worker_addrs(name)["primary"].port > 0

    async def test_an_unknown_worker_name_fails_loudly(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Looking up a worker in a cell index that does not exist raises an assertion
        # error.
        """Looking up a worker that does not exist must not silently return an arbitrary address."""
        manager = await _launch([_make_spec("engine", num_cells=2)])

        with pytest.raises(AssertionError):
            manager.get_worker_addrs("engine-00002-00000")["primary"]

    async def test_nothing_is_published_until_every_port_is_allocated(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Patches the static-port availability check with an observer that records
        # the worker's published address map (or none) each time it runs; the spec has two fixed
        # ports, so it runs twice. Asserts both observations were "nothing published yet", and that
        # afterwards the worker's map holds both ports.
        """Allocation crosses an await per port and the manager answers reads in between, so a map
        published as it fills lets a reader take the endpoints not there yet for endpoints the
        worker does not have -- which is how a restarting engine is read as having no primary."""
        observed: list[dict | None] = []

        async def observe(self, port: int, *, port_name: str, node_ip: str) -> None:
            # [FUNCTION DOC] HELPER: Replacement availability check; appends a copy of the actor's
            # current published address map, or none, and does no probing.
            observed.append(None if self.self_addrs is None else dict(self.self_addrs))

        spec = _make_spec(
            "engine",
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=False),
                PortInfo(name="rpc", static_port=9000, allow_dynamic=False),
            ],
        )
        with patch.object(ray_worker_manager._BaseActorManager, "_assert_static_port_is_free", observe):
            manager = await _launch([spec])

        assert observed == [None, None]
        assert set(manager.get_worker_addrs("engine-00000-00000")) == {"primary", "rpc"}

    async def test_a_worker_without_its_ports_yet_is_refused_rather_than_answered(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] Resets one launched worker's published address map to none; asserts the
        # lookup raises an assertion error saying it has not been given its ports yet.
        """Returning nothing would be read as a worker that has no endpoints at all."""
        manager = await _launch([_make_spec("engine")])
        manager._find_actor("engine-00000-00000").self_addrs = None

        with pytest.raises(AssertionError, match="has not been given its ports yet"):
            manager.get_worker_addrs("engine-00000-00000")


class TestPinToHead:
    # [CLASS DOC] Tests of the head-node pinning option of a spec.
    async def test_a_pinned_worker_keeps_its_resources_and_gains_head_affinity(
        self, fake_ray_cluster: FakeRayCluster, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] Replaces the head-pinning options function with one returning a marker
        # scheduling strategy; for a pinned router with one environment variable asserts the actor
        # options contain that strategy, the 0.2 cpu request and the runtime environment.
        """Pinning adds the head-affinity strategy without dropping the other actor options."""
        monkeypatch.setattr(
            ray_worker_manager,
            "compute_ray_pin_head_options",
            lambda: {"scheduling_strategy": "head-affinity"},
        )
        await _launch([_make_spec("router", pin_to_head=True, env_var={"A": "1"})])

        options = fake_ray_cluster.handles[0].options
        assert options["scheduling_strategy"] == "head-affinity"
        assert options["num_cpus"] == 0.2
        assert options["runtime_env"] == {"env_vars": {"A": "1"}}

    async def test_an_unpinned_worker_never_resolves_the_head_node(
        self, fake_ray_cluster: FakeRayCluster, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] Replaces the head-pinning options function with one that fails an
        # assertion if called; launches an unpinned router and asserts it was never called and the
        # actor options have no scheduling strategy.
        """Unpinned workers must not even look the head node up, let alone all land on it."""

        def _fail() -> dict:
            # [FUNCTION DOC] HELPER: Replacement head-pinning options function that always fails an
            # assertion.
            raise AssertionError("the head node must not be resolved for unpinned workers")

        monkeypatch.setattr(ray_worker_manager, "compute_ray_pin_head_options", _fail)
        await _launch([_make_spec("router")])

        assert "scheduling_strategy" not in fake_ray_cluster.handles[0].options


class TestSpecAddrs:
    # [CLASS DOC] Tests of the pool-wide address book given to launch commands.
    async def test_every_worker_sees_the_addresses_of_all_specs(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] An inference router and a 2-cell session server (recording launch commands);
        # asserts every recorded context's pool address book has exactly both pool names, the
        # router's single entry equals the manager's address for it, and the session server's
        # entries are its two cells' addresses in cell order.
        """Cross-spec wiring works because each command is rendered with the whole pool's addresses."""
        recorder = _LaunchRecorder()
        manager = await _launch(
            [
                _make_spec("inference-router-0"),
                _make_spec("session-server", num_cells=2, launch_command=recorder.command),
            ]
        )

        for ctx in recorder.contexts:
            assert sorted(ctx.pool_addrs) == ["inference-router-0", "session-server"]
            assert (
                ctx.pool_addrs["inference-router-0"][0]["primary"]
                == manager.get_worker_addrs("inference-router-0-00000-00000")["primary"]
            )
            assert [addr["primary"] for addr in ctx.pool_addrs["session-server"]] == [
                manager.get_worker_addrs("session-server-00000-00000")["primary"],
                manager.get_worker_addrs("session-server-00001-00000")["primary"],
            ]


class TestGetAddrs:
    # [CLASS DOC] Tests of the manager's full address listing, a mapping from pool name to a list
    # of per-worker address maps.
    async def test_groups_are_listed_in_cell_then_worker_order(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] For 2 cells of 2 workers, asserts the engine list's primary addresses
        # equal the per-worker lookups in cell-major, worker-minor order.
        """Consumers index this map positionally, so the order must follow cells and then workers."""
        recorder = _LaunchRecorder()
        manager = await _launch(
            [_make_spec("engine", num_cells=2, num_workers_per_cell=2, launch_command=recorder.command)]
        )

        expected = [
            manager.get_worker_addrs(
                compute_worker_name(pool_id="engine", cell_index=cell_index, worker_in_cell_index=worker_in_cell_index)
            )["primary"]
            for cell_index in range(2)
            for worker_in_cell_index in range(2)
        ]
        assert [addr["primary"] for addr in manager.get_addrs()["engine"]] == expected

    async def test_a_disabled_group_is_listed_as_empty(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A spec with zero cells appears with an empty list while the router has one
        # entry.
        """A spec with no cells is still visible to consumers, with no addresses."""
        manager = await _launch([_make_spec("router"), _make_spec("session-server", num_cells=0)])

        assert manager.get_addrs()["session-server"] == []
        assert len(manager.get_addrs()["router"]) == 1


class TestMasterPorts:
    # [CLASS DOC] Tests of master-mode ports, which exist once per cell, owned by worker 0, and are
    # shared with the cell's other workers through their launch contexts.
    async def test_a_master_port_is_allocated_once_per_cell(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] 2 cells of 3 workers with a master-mode dynamic block of 5 ports; asserts
        # exactly two block requests of count 5 and that only each cell's worker 0 lists the
        # dist_init port in the published address listing.
        """Only worker 0 reserves the cell's master port, so peers cannot each take their own."""
        spec = _make_spec(
            "engine",
            num_cells=2,
            num_workers_per_cell=3,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=9000, mode="master", allow_dynamic=True, num_consecutive=5),
            ],
        )
        manager = await _launch([spec])

        assert [call.kwargs["count"] for call in fake_ray_cluster.calls_of("_get_free_port_block")].count(5) == 2
        addrs = manager.get_addrs()["engine"]
        assert [sorted(addr) for addr in addrs] == [
            ["dist_init", "primary"],
            ["primary"],
            ["primary"],
            ["dist_init", "primary"],
            ["primary"],
            ["primary"],
        ]

    async def test_a_static_master_port_is_recorded_only_on_worker_zero(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A 2-worker cell with a fixed master port 9123; asserts worker 0 lists
        # dist_init with port 9123, worker 1 does not list it, and there were two block requests
        # (one dynamic primary port per worker).
        """A pinned master port is not allocated, and still belongs to worker 0 alone."""
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=9123, mode="master", allow_dynamic=False),
            ],
        )
        manager = await _launch([spec])

        addrs = manager.get_addrs()["engine"]
        assert addrs[0]["dist_init"].port == 9123
        assert "dist_init" not in addrs[1]
        assert len(fake_ray_cluster.calls_of("_get_free_port_block")) == 2

    async def test_every_worker_of_a_cell_launches_with_that_cells_master_addr(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] 2 cells of 2 workers; from the recorded launch contexts asserts both
        # workers of a cell see the same dist_init address while the two cells' addresses differ.
        """All ranks of a cell must be told the same master endpoint, and never another cell's."""
        recorder = _LaunchRecorder()
        spec = _make_spec(
            "engine",
            num_cells=2,
            num_workers_per_cell=2,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=9000, mode="master", allow_dynamic=True),
            ],
            launch_command=recorder.command,
        )
        await _launch([spec])

        masters = {
            (cell_index, worker_in_cell_index): recorder.context_of(
                cell_index=cell_index, worker_in_cell_index=worker_in_cell_index
            ).self_addrs["dist_init"]
            for cell_index in range(2)
            for worker_in_cell_index in range(2)
        }
        assert masters[(0, 0)] == masters[(0, 1)]
        assert masters[(1, 0)] == masters[(1, 1)]
        assert masters[(0, 0)] != masters[(1, 0)]

    async def test_a_master_addr_keeps_rank_zeros_host_on_another_node(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Two workers on nodes .1 and .2; asserts worker 1's launch context has its
        # primary host .2 but its dist_init host .1 (rank 0's node).
        """A peer on a second node must dial rank 0's host, not its own, or torch.distributed hangs."""
        fake_ray_cluster.use_node_ips("10.0.0.1", "10.0.0.2")
        recorder = _LaunchRecorder()
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=9000, mode="master", allow_dynamic=True),
            ],
            launch_command=recorder.command,
        )
        await _launch([spec])

        peer = recorder.context_of(cell_index=0, worker_in_cell_index=1).self_addrs
        assert peer["primary"].host == "10.0.0.2"
        assert peer["dist_init"].host == "10.0.0.1"

    async def test_a_master_addr_does_not_overwrite_a_workers_own_ports(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] 2-worker cell with a master port; asserts each worker's launch-context
        # primary address equals the manager's address for that worker.
        """Merging the cell's master addr must leave each worker's own addresses intact."""
        recorder = _LaunchRecorder()
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=9000, mode="master", allow_dynamic=True),
            ],
            launch_command=recorder.command,
        )
        manager = await _launch([spec])

        for worker_in_cell_index in range(2):
            ctx = recorder.context_of(cell_index=0, worker_in_cell_index=worker_in_cell_index)
            assert (
                ctx.self_addrs["primary"]
                == manager.get_worker_addrs(
                    compute_worker_name(pool_id="engine", worker_in_cell_index=worker_in_cell_index)
                )["primary"]
            )


class TestConcurrentPhases:
    # [CLASS DOC] Tests that within a phase cells and workers are processed concurrently, using a
    # barrier that only opens once three allocations are in flight.
    async def test_all_cells_of_a_phase_run_concurrently(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Wraps port allocation so it records the cell index and waits for a release
        # event that is set when the third allocation arrives (5 second wait; whole launch limited
        # to 10 seconds). With 3 cells asserts the entered cells are 0, 1, 2 (so none was serialised
        # behind another) and the three primary ports are distinct.
        """Cells are configured concurrently, so a large pool does not start up one cell at a time."""
        from miles.utils.workers.ray_worker_manager import _CommandActorManager

        original_alloc_ports = _CommandActorManager.alloc_ports
        entered: list[int] = []
        release = asyncio.Event()

        async def gated_alloc(self) -> None:
            # [FUNCTION DOC] HELPER: Allocation wrapper; records the cell index, opens the barrier on
            # the third arrival, waits for it, then runs the original allocation.
            entered.append(self.parent.cell_index)
            if len(entered) == 3:
                release.set()
            await asyncio.wait_for(release.wait(), timeout=5)
            await original_alloc_ports(self)

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(_CommandActorManager, "alloc_ports", gated_alloc)
            manager = await asyncio.wait_for(_launch([_make_spec("engine", num_cells=3)]), timeout=10)

        assert sorted(entered) == [0, 1, 2]
        assert (
            len(
                {
                    manager.get_worker_addrs(compute_worker_name(pool_id="engine", cell_index=index))["primary"].port
                    for index in range(3)
                }
            )
            == 3
        )

    async def test_workers_within_one_cell_run_each_phase_concurrently(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Same barrier technique as the cells test, but with one cell of 3 workers;
        # asserts worker indices 0, 1, 2 all entered allocation before any finished, and their
        # three primary ports are distinct.
        """The ranks of one engine configure together; serialising them adds a full setup per rank."""
        from miles.utils.workers.ray_worker_manager import _CommandActorManager

        original_alloc_ports = _CommandActorManager.alloc_ports
        entered: list[int] = []
        release = asyncio.Event()

        async def gated_alloc(self) -> None:
            # [FUNCTION DOC] HELPER: Allocation wrapper; records the worker index, opens the barrier
            # on the third arrival, waits for it, then runs the original allocation.
            entered.append(self.worker_in_cell_index)
            if len(entered) == 3:
                release.set()
            await asyncio.wait_for(release.wait(), timeout=5)
            await original_alloc_ports(self)

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(_CommandActorManager, "alloc_ports", gated_alloc)
            manager = await asyncio.wait_for(_launch([_make_spec("engine", num_workers_per_cell=3)]), timeout=10)

        assert sorted(entered) == [0, 1, 2]
        ports = {
            manager.get_worker_addrs(compute_worker_name(pool_id="engine", worker_in_cell_index=index))["primary"].port
            for index in range(3)
        }
        assert len(ports) == 3


class TestAddressPublication:
    # [CLASS DOC] Tests that a worker's address map and its cell's aliveness are published only
    # when every port of the worker (and every worker of the cell) has been assigned.
    async def test_reading_the_addrs_of_a_worker_mid_allocation_is_refused(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Pauses allocation of a single worker, starts init as a task, and while it
        # is paused asserts the address lookup raises an assertion error (ports not given yet). After
        # releasing and finishing init asserts the lookup includes the primary port.
        """A half-filled map reads like a worker that simply has no such endpoint, so it must not be observable."""
        original_alloc_ports = _CommandActorManager.alloc_ports
        entered = asyncio.Event()
        release = asyncio.Event()

        async def gated_alloc(self) -> None:
            # [FUNCTION DOC] HELPER: Sets the entered event, waits (5 seconds) for release, then runs
            # the original allocation.
            entered.set()
            await asyncio.wait_for(release.wait(), timeout=5)
            await original_alloc_ports(self)

        manager = RayWorkerManager()
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(_CommandActorManager, "alloc_ports", gated_alloc)
            task = asyncio.create_task(
                manager.init(worker_manager_args(), [_make_spec("engine")], {}, comm_backend=WorkerCommBackend.RAY)
            )
            await asyncio.wait_for(entered.wait(), timeout=5)

            with pytest.raises(AssertionError, match="has not been given its ports yet"):
                manager.get_worker_addrs("engine-00000-00000")

            release.set()
            await asyncio.wait_for(task, timeout=5)

        assert "primary" in manager.get_worker_addrs("engine-00000-00000")

    async def test_a_read_between_two_ports_of_one_worker_is_refused(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A router with one dynamic and two fixed ports; wraps the static-port check
        # so each time it runs it tries to read the worker's own addresses and records either the
        # result or the error text. Asserts two observations, both the "not given its ports yet"
        # error, and that afterwards the worker lists all three ports.
        """A worker caught between two of its ports refuses the read rather than answering with the ports so far."""
        spec = _make_spec(
            "router",
            port_infos=[
                PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                PortInfo(name="prometheus", static_port=7001, allow_dynamic=False),
                PortInfo(name="debug", static_port=7002, allow_dynamic=False),
            ],
        )
        original_assert_free = _CommandActorManager._assert_static_port_is_free
        observed: list[object] = []

        async def probing_assert_free(self, *, port: int, port_name: str, node_ip: str) -> None:
            # [FUNCTION DOC] HELPER: Records the outcome of reading this worker's addresses through
            # the manager (the map or the assertion message), then runs the original check.
            try:
                observed.append(self.manager.get_worker_addrs(self.name))
            except AssertionError as error:
                observed.append(str(error))
            await original_assert_free(self, port=port, port_name=port_name, node_ip=node_ip)

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(_CommandActorManager, "_assert_static_port_is_free", probing_assert_free)
            manager = await _launch([spec])

        assert len(observed) == 2
        assert all(isinstance(seen, str) and "has not been given its ports yet" in seen for seen in observed)
        assert sorted(manager.get_worker_addrs("router-00000-00000")) == ["debug", "primary", "prometheus"]

    async def test_a_worker_that_declares_no_ports_publishes_an_empty_map(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A spec with no ports; its worker's address lookup returns an empty mapping.
        """Owning no endpoint is a finished answer, so an empty map must read as published, not as unallocated."""
        manager = await _launch([_make_spec("controller", port_infos=[])])

        assert manager.get_worker_addrs("controller-00000-00000") == {}

    async def test_a_settled_worker_stays_readable_while_a_peer_allocates(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Launches a router and an engine with a fixed port, stops the engine cell,
        # then restarts it with the static-port check wrapped to read the router's primary port
        # during the engine's allocation. Asserts that one read succeeded with the router's
        # original port and the engine came back on its fixed port 7001.
        """Only the worker still being given its ports is unreadable; one that already has all of its own answers."""
        specs = [
            _make_spec("router"),
            _make_spec("engine", port_infos=[PortInfo(name="primary", static_port=7001, allow_dynamic=False)]),
        ]
        manager = await _launch(specs)
        router_port = manager.get_worker_addrs("router-00000-00000")["primary"].port
        await manager.stop_cells(["engine-00000"])

        original_assert_free = _CommandActorManager._assert_static_port_is_free
        observed: list[int] = []

        async def probing_assert_free(self, *, port: int, port_name: str, node_ip: str) -> None:
            # [FUNCTION DOC] HELPER: Appends the router worker's primary port as read through the
            # manager, then runs the original check.
            observed.append(self.manager.get_worker_addrs("router-00000-00000")["primary"].port)
            await original_assert_free(self, port=port, port_name=port_name, node_ip=node_ip)

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(_CommandActorManager, "_assert_static_port_is_free", probing_assert_free)
            await manager.start_cells(["engine-00000"])

        assert observed == [router_port]
        assert manager.get_worker_addrs("engine-00000-00000")["primary"].port == 7001

    async def test_a_cell_is_published_only_once_every_worker_is_addressed(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A 2-worker cell where worker 0 allocates immediately and worker 1 waits for
        # release. While waiting, asserts the cell is not alive and the address listing's entry for
        # worker 1 is empty; after release and completion asserts the cell is alive.
        """One unaddressed rank makes the whole cell unusable, so the cell must not be observed as alive yet."""
        original_alloc_ports = _CommandActorManager.alloc_ports
        first_done = asyncio.Event()
        release = asyncio.Event()

        async def gated_alloc(self) -> None:
            # [FUNCTION DOC] HELPER: Worker 0 runs the original allocation and sets first_done; any
            # other worker waits (5 seconds) for release before allocating.
            if self.worker_in_cell_index == 0:
                await original_alloc_ports(self)
                first_done.set()
                return
            await asyncio.wait_for(release.wait(), timeout=5)
            await original_alloc_ports(self)

        manager = RayWorkerManager()
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(_CommandActorManager, "alloc_ports", gated_alloc)
            task = asyncio.create_task(
                manager.init(
                    worker_manager_args(),
                    [_make_spec("engine", num_workers_per_cell=2)],
                    {},
                    comm_backend=WorkerCommBackend.RAY,
                )
            )
            await asyncio.wait_for(first_done.wait(), timeout=5)

            assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive is False
            assert manager.get_addrs()["engine"][1] == {}

            release.set()
            await asyncio.wait_for(task, timeout=5)

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive is True

    async def test_a_worker_still_being_addressed_is_described_with_an_empty_addr_map(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] Gate blocks worker 1 of a 2-worker cell. While blocked, the worker infos
        # list both workers in order, worker 0 with its primary address and worker 1 with an empty
        # address map. After release, worker 1's infos include the primary port.
        """A missing addr map, unlike an empty one, cannot be described at all and loses the cell's description."""
        gate = _AllocPortsGate(should_block=lambda actor: actor.worker_in_cell_index == 1)

        manager = RayWorkerManager()
        with pytest.MonkeyPatch.context() as patched:
            gate.install(patched)
            task = asyncio.create_task(
                manager.init(
                    worker_manager_args(),
                    [_make_spec("engine", num_workers_per_cell=2)],
                    {},
                    comm_backend=WorkerCommBackend.RAY,
                )
            )
            await gate.wait_until_allocated(worker_names=["engine-00000-00000"])

            infos = manager.get_worker_infos("engine-00000")

            assert [info.name for info in infos] == ["engine-00000-00000", "engine-00000-00001"]
            assert "primary" in infos[0].self_addrs
            assert infos[1].self_addrs == {}

            gate.release.set()
            await asyncio.wait_for(task, timeout=5)

        assert "primary" in manager.get_worker_infos("engine-00000")[1].self_addrs

    async def test_a_cell_still_being_addressed_stays_listed_with_all_of_its_workers(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] Gate blocks worker 1; while blocked the cell info is not alive but still
        # lists both worker names; after release the cell is alive.
        """A cell being addressed is only unpublished, never hidden, so an observer can still track it."""
        gate = _AllocPortsGate(should_block=lambda actor: actor.worker_in_cell_index == 1)

        manager = RayWorkerManager()
        with pytest.MonkeyPatch.context() as patched:
            gate.install(patched)
            task = asyncio.create_task(
                manager.init(
                    worker_manager_args(),
                    [_make_spec("engine", num_workers_per_cell=2)],
                    {},
                    comm_backend=WorkerCommBackend.RAY,
                )
            )
            await gate.wait_until_allocated(worker_names=["engine-00000-00000"])

            info = manager.get_cell_infos(pool_ids=["engine"])["engine-00000"]

            assert info.alive is False
            assert info.worker_names == ["engine-00000-00000", "engine-00000-00001"]

            gate.release.set()
            await asyncio.wait_for(task, timeout=5)

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive is True

    async def test_a_fully_addressed_cell_is_published_while_a_sibling_still_waits_for_ports(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] Gate blocks every worker of cell 1 in a 2-cell spec; once cell 0's workers
        # are allocated, cell 0 is reported alive and cell 1 not. After release both are alive.
        """Addressing is judged per cell, so one slow cell must not withhold the cells that are ready to serve."""
        gate = _AllocPortsGate(should_block=lambda actor: actor.parent.cell_index == 1)

        manager = RayWorkerManager()
        with pytest.MonkeyPatch.context() as patched:
            gate.install(patched)
            task = asyncio.create_task(
                manager.init(
                    worker_manager_args(),
                    [_make_spec("engine", num_cells=2, num_workers_per_cell=2)],
                    {},
                    comm_backend=WorkerCommBackend.RAY,
                )
            )
            await gate.wait_until_allocated(worker_names=["engine-00000-00000", "engine-00000-00001"])

            infos = manager.get_cell_infos(pool_ids=["engine"])

            assert infos["engine-00000"].alive is True
            assert infos["engine-00001"].alive is False

            gate.release.set()
            await asyncio.wait_for(task, timeout=5)

        assert [info.alive for info in manager.get_cell_infos(pool_ids=["engine"]).values()] == [True, True]

    async def test_a_resumed_cell_is_unpublished_again_until_its_new_ports_are_handed_out(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] Stops a 2-worker cell, then restarts it as a task with the gate blocking
        # worker 1. While blocked the cell is not alive and worker 1's listing is empty; after
        # release and completion the cell is alive again.
        """A resumed cell hands out fresh ports, so having been addressed once must not vouch for it again."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        await manager.stop_cells(["engine-00000"])
        gate = _AllocPortsGate(should_block=lambda actor: actor.worker_in_cell_index == 1)

        with pytest.MonkeyPatch.context() as patched:
            gate.install(patched)
            task = asyncio.create_task(manager.start_cells(["engine-00000"]))
            await gate.wait_until_allocated(worker_names=["engine-00000-00000"])

            assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive is False
            assert manager.get_addrs()["engine"][1] == {}

            gate.release.set()
            await asyncio.wait_for(task, timeout=5)

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive is True


class TestGpuPlacement:
    # [CLASS DOC] Tests of how gpu slots from a placement group become gpu ids, local gpu ids and
    # actor scheduling options.
    async def test_gpu_slots_expand_over_cells_and_workers(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] 2 cells of 2 workers, 2 slots per worker, group offset 2, 10 slots; asserts
        # the workers' gpu ids are [2,3], [4,5], [6,7], [8,9] in cell-major order.
        """Each worker occupies its own contiguous slot span after its group's base offset."""
        recorder = _LaunchRecorder()
        spec = _make_spec(
            "engine",
            num_cells=2,
            num_workers_per_cell=2,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=2,
            pg_name="rollout",
            pg_slot_offset=2,
            launch_command=recorder.command,
        )
        await _launch([spec], _make_pgs(num_slots=10, first_gpu_id=0))

        assert [
            recorder.context_of(cell_index=cell_index, worker_in_cell_index=worker_in_cell_index).gpu_ids
            for cell_index in range(2)
            for worker_in_cell_index in range(2)
        ] == [[2, 3], [4, 5], [6, 7], [8, 9]]

    async def test_gpu_ids_come_from_the_reordered_slot_mapping(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The group's first gpu id is 4; a 2-worker cell with 2 slots each gets gpu
        # ids [4,5] and [6,7].
        """A worker's gpu ids start at the node-local id its slot maps to, not at the slot index."""
        recorder = _LaunchRecorder()
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=2,
            pg_name="rollout",
            launch_command=recorder.command,
        )
        await _launch([spec], _make_pgs(num_slots=8, first_gpu_id=4))

        assert recorder.context_of(cell_index=0, worker_in_cell_index=0).gpu_ids == [4, 5]
        assert recorder.context_of(cell_index=0, worker_in_cell_index=1).gpu_ids == [6, 7]

    async def test_local_gpu_ids_are_resolved_by_the_worker_itself_before_its_command_is_built(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] Asserts each worker was asked (to-local-gpu-ids call) with its global ids
        # [4,5] and [6,7], both launch contexts then carry local ids [0,1], and the last such call
        # precedes the first command run in the event log.
        """The manager has no visibility mask of its own, so only the worker can turn its ids into local ones."""
        recorder = _LaunchRecorder()
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=2,
            pg_name="rollout",
            launch_command=recorder.command,
        )
        await _launch([spec], _make_pgs(num_slots=8, first_gpu_id=4))

        assert [call.kwargs["gpu_ids"] for call in fake_ray_cluster.calls_of("_to_local_gpu_ids")] == [[4, 5], [6, 7]]
        assert recorder.context_of(cell_index=0, worker_in_cell_index=0).local_gpu_ids == [0, 1]
        assert recorder.context_of(cell_index=0, worker_in_cell_index=1).local_gpu_ids == [0, 1]
        assert fake_ray_cluster.last_event_index("_to_local_gpu_ids") < fake_ray_cluster.first_event_index("run")

    async def test_each_worker_is_asked_about_its_own_gpu_ids(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Asserts the to-local-gpu-ids calls map worker handle index 0 to [4,5] and
        # index 1 to [6,7].
        """Probing one worker on behalf of the whole cell would describe every engine with that worker's mask."""
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=2,
            pg_name="rollout",
        )
        await _launch([spec], _make_pgs(num_slots=8, first_gpu_id=4))

        asked = {call.handle.index: call.kwargs["gpu_ids"] for call in fake_ray_cluster.calls_of("_to_local_gpu_ids")}
        assert asked == {0: [4, 5], 1: [6, 7]}

    async def test_a_worker_that_owns_no_gpu_is_launched_with_an_empty_local_gpu_id_list(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] A gpu-less router is still asked once with an empty id list, and its
        # launch context has empty local gpu ids.
        """A gpu-less worker is probed like any other, and its command still has to receive a list."""
        recorder = _LaunchRecorder()
        spec = _make_spec("router", launch_command=recorder.command)
        await _launch([spec])

        assert [call.kwargs["gpu_ids"] for call in fake_ray_cluster.calls_of("_to_local_gpu_ids")] == [[]]
        assert recorder.context_of(cell_index=0, worker_in_cell_index=0).local_gpu_ids == []

    async def test_a_pg_worker_is_scheduled_on_the_bundle_its_slot_maps_to(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] 2 workers, 1 slot each, group offset 3; asserts their placement strategies
        # use bundle indices at positions 3 and 4 of the reordered list, the fake group, and a gpu
        # request of 0.2 each.
        """The actor is placed on the reordered bundle of its slot, not on the raw slot index."""
        pgs = _make_pgs(num_slots=8)
        spec = _make_spec(
            "engine",
            num_workers_per_cell=2,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=1,
            pg_name="rollout",
            pg_slot_offset=3,
        )
        await _launch([spec], pgs)

        strategies = [handle.options["scheduling_strategy"] for handle in fake_ray_cluster.handles]
        assert [strategy.placement_group_bundle_index for strategy in strategies] == [
            pgs["rollout"].pg_reordered_bundle_indices[3],
            pgs["rollout"].pg_reordered_bundle_indices[4],
        ]
        assert all(strategy.placement_group == "fake-pg" for strategy in strategies)
        assert all(handle.options["num_gpus"] == 0.2 for handle in fake_ray_cluster.handles)

    async def test_a_worker_without_a_pg_asks_for_no_placement(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A router without a group has no scheduling strategy, requests 0 gpus and
        # gets empty gpu ids.
        """Specs that do not own gpus are scheduled anywhere and get no gpu ids."""
        recorder = _LaunchRecorder()
        await _launch([_make_spec("router", launch_command=recorder.command)])

        assert "scheduling_strategy" not in fake_ray_cluster.handles[0].options
        assert fake_ray_cluster.handles[0].options["num_gpus"] == 0
        assert recorder.context_of(cell_index=0, worker_in_cell_index=0).gpu_ids == []


class TestPgActorResources:
    # [CLASS DOC] Tests of gpu resources of placement-group actors.
    async def test_the_gpu_request_comes_from_the_spec(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A spec with gpu fraction 0.2 yields an actor option of 0.2 gpus.
        """A spec that declares a gpu fraction gets exactly that, so co-located engines still fit."""
        await _launch([_make_spec("engine", num_gpus_per_worker=0.2)])

        assert fake_ray_cluster.handles[0].options["num_gpus"] == 0.2

    async def test_pg_workers_capture_their_child_tasks_in_the_group(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Asserts the placement strategy of a group worker has capture-child-tasks
        # set to true.
        """Child tasks must stay inside the placement group, or they escape the reserved gpus."""
        spec = _make_spec(
            "engine",
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=1,
            pg_name="rollout",
        )
        await _launch([spec], _make_pgs())

        assert fake_ray_cluster.handles[0].options["scheduling_strategy"].placement_group_capture_child_tasks is True


class TestPgFailureModes:
    # [CLASS DOC] Tests of misconfigured placement groups.
    async def test_a_spec_pointing_at_an_unknown_placement_group_fails(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A spec naming a group that is not in the mapping makes init raise a key
        # error, and no command is run.
        """Mistyping the placement group name must fail instead of silently ignoring placement."""
        spec = _make_spec("engine", num_gpus_per_worker=0.2, num_gpu_slots_per_worker=1, pg_name="typo")

        with pytest.raises(KeyError):
            await _launch([spec], _make_pgs())

        assert fake_ray_cluster.calls_of("run") == []


class TestCellLifecycle:
    # [CLASS DOC] Tests of cell-level launch behavior, reaching into the manager's internal pool
    # and cell objects.
    async def test_a_cell_launches_its_workers_only_once(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Calling the actor-launch method again on a populated cell raises an
        # assertion error and creates no new actor (still 2 handles).
        """Launching an already-populated cell again would orphan the first generation of actors."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        cell = manager._pools["engine"].cells[0]

        with pytest.raises(AssertionError):
            await cell.launch_actors()

        assert len(fake_ray_cluster.handles) == 2

    async def test_the_stages_of_a_cell_fan_out_to_every_worker(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A 3-worker cell sees 3 node-address lookups, 3 command runs and 3 actors.
        """Each stage covers all workers of the cell, so no rank is left half-configured."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=3)])

        assert len(fake_ray_cluster.calls_of("_get_node_ip")) == 3
        assert len(fake_ray_cluster.calls_of("run")) == 3
        assert len(manager._pools["engine"].cells[0].actors) == 3


class TestCellStop:
    # [CLASS DOC] Tests of stopping cells and shutting the manager down.
    async def test_shutting_down_the_manager_stops_every_cell_it_owns(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Manager shutdown with 3 workers in total kills all 3 and reports no cell
        # alive.
        """A finished run releases its own workers without selecting unrelated host processes by name."""
        manager = await _launch([_make_spec("engine", num_cells=2), _make_spec("router")])

        await manager.shutdown()

        assert fake_ray_cluster.events.count(EVENT_KILL) == 3
        assert not any(info.alive for info in manager.get_cell_infos(pool_ids=["engine", "router"]).values())

    async def test_stopping_a_cell_shuts_down_and_kills_every_worker(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Stopping a 2-worker cell issues 2 shutdown calls and 2 kills, every handle
        # ends up killed, and the cell's actor list becomes none.
        """A stopped cell releases all of its workers, not just the first one."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        cell = manager._pools["engine"].cells[0]

        await cell.stop()

        assert [call.method for call in fake_ray_cluster.calls if call.method == "shutdown"] == ["shutdown"] * 2
        assert fake_ray_cluster.events.count(EVENT_KILL) == 2
        assert all(handle.killed for handle in fake_ray_cluster.handles)
        assert cell.actors is None

    async def test_stopping_one_cell_leaves_the_others_running(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Stopping cell 0 of 2 kills only its actor; cell 1 keeps its actors.
        """Stopping a cell must not touch the workers of its siblings."""
        manager = await _launch([_make_spec("engine", num_cells=2)])

        await manager._pools["engine"].cells[0].stop()

        assert [handle.killed for handle in fake_ray_cluster.handles] == [True, False]
        assert manager._pools["engine"].cells[1].actors is not None

    async def test_a_worker_that_cannot_shut_down_gracefully_is_still_killed(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The worker's shutdown call raises a runtime error; stop still kills it
        # once and clears the cell's actors.
        """A hung or broken worker is exactly the one that must be force-killed."""
        manager = await _launch([_make_spec("engine")])
        cell = manager._pools["engine"].cells[0]
        fake_ray_cluster.handles[0].failing_methods["shutdown"] = RuntimeError("shutdown timed out")

        await cell.stop()

        assert fake_ray_cluster.events.count(EVENT_KILL) == 1
        assert cell.actors is None

    async def test_a_failing_kill_does_not_abort_the_teardown_of_other_workers(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The fake kill raises a runtime error; stop of a 2-worker cell still attempts
        # both kills and clears the cell's actors without raising.
        """One worker ray cannot kill must not stop the cell from tearing the rest down."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        cell = manager._pools["engine"].cells[0]
        fake_ray_cluster.kill_error = RuntimeError("kill failed")

        await cell.stop()

        assert fake_ray_cluster.events.count(EVENT_KILL) == 2
        assert cell.actors is None


class TestStopDetails:
    # [CLASS DOC] Finer points of stop: ordering, timeouts and concurrency.
    async def test_each_worker_is_asked_to_shut_down_before_it_is_killed(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The last two recorded events of a stop are shutdown then kill.
        """Graceful shutdown first gives the subprocess a chance to exit cleanly."""
        manager = await _launch([_make_spec("engine")])

        await manager._pools["engine"].cells[0].stop()

        assert fake_ray_cluster.events[-2:] == ["shutdown", EVENT_KILL]

    async def test_graceful_shutdown_is_bounded_by_a_timeout(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The shutdown call hangs for an hour; with the module's shutdown timeout
        # patched to 0.01 seconds, stop completes and the worker is killed once.
        """A hung worker must not block teardown forever."""
        manager = await _launch([_make_spec("engine")])
        fake_ray_cluster.handles[0].hanging_methods["shutdown"] = 3600

        with patch.object(ray_worker_manager, "_SHUTDOWN_TIMEOUT", 0.01):
            await manager._pools["engine"].cells[0].stop()

        assert fake_ray_cluster.events.count(EVENT_KILL) == 1

    async def test_shutdown_leaves_the_managers_event_loop_free(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Shutdown takes 0.05 seconds; a concurrent polling task keeps reading cell
        # infos and yielding. After stop completes the poller is cancelled and the test asserts it
        # ran more than 20 polls during the stop.
        """A slow shutdown must not block the manager loop, which still has to answer cell polls meanwhile."""
        manager = await _launch([_make_spec("engine")])
        fake_ray_cluster.handles[0].hanging_methods["shutdown"] = 0.05
        polls: int = 0

        async def poll_cells() -> None:
            # [FUNCTION DOC] HELPER: Loops forever; asserts the cell is listed, increments the poll
            # counter of the enclosing test and yields to the event loop. Ends only by cancellation.
            nonlocal polls
            while True:
                assert "engine-00000" in manager.get_cell_infos(pool_ids=["engine"])
                polls += 1
                await asyncio.sleep(0)

        stop_task = asyncio.create_task(manager._pools["engine"].cells[0].stop())
        poll_task = asyncio.create_task(poll_cells())
        await stop_task
        polls_during_stop: int = polls
        poll_task.cancel()
        await asyncio.gather(poll_task, return_exceptions=True)

        assert polls_during_stop > 20

    async def test_a_hung_shutdown_does_not_delay_killing_peer_workers(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Worker 0's shutdown hangs; with shutdown timeout 0.5 seconds, after 0.05
        # seconds worker 1 is already killed while stop is still running; stop then finishes within
        # 5 seconds with 2 kills in total.
        """One rank ignoring shutdown must not keep its peers holding gpus for the whole grace period."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        fake_ray_cluster.handles[0].hanging_methods["shutdown"] = 3600

        with patch.object(ray_worker_manager, "_SHUTDOWN_TIMEOUT", 0.5):
            stop_task = asyncio.create_task(manager._pools["engine"].cells[0].stop())
            await asyncio.sleep(0.05)
            assert fake_ray_cluster.handles[1].killed
            assert not stop_task.done()
            await asyncio.wait_for(stop_task, timeout=5)

        assert fake_ray_cluster.events.count(EVENT_KILL) == 2

    async def test_all_workers_are_killed_even_when_one_shutdown_hangs(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Worker 1 of 3 fails its shutdown with a timeout error; all 3 are killed.
        # KNOWN ISSUE: the name says "hangs" but the setup makes the call fail immediately.
        """One broken worker must not save its peers from teardown."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=3)])
        fake_ray_cluster.handles[1].failing_methods["shutdown"] = TimeoutError("hung")

        await manager._pools["engine"].cells[0].stop()

        assert fake_ray_cluster.events.count(EVENT_KILL) == 3
        assert all(handle.killed for handle in fake_ray_cluster.handles)


class TestGetWorkerInfos:
    # [CLASS DOC] Tests of the worker description listing for one cell.
    async def test_a_worker_still_being_given_its_ports_is_described_as_holding_none(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] Resets worker 1's address map to none; asserts the infos still list both
        # workers in order, worker 1 described with an empty map, and the full address listing
        # shows an empty entry for it.
        """A description is taken of whatever exists at the time, and a cell mid-start is part of that;
        it has to render as a worker holding no endpoints rather than fail the whole description."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        manager._find_actor("engine-00000-00001").self_addrs = None

        infos = manager.get_worker_infos("engine-00000")

        assert [info.name for info in infos] == ["engine-00000-00000", "engine-00000-00001"]
        assert infos[1].self_addrs == {}
        assert manager.get_addrs()["engine"][1] == {}

    async def test_describes_every_worker_of_only_the_requested_cell(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] 2 cells of 2 workers with 2 gpu slots each; for cell 1 asserts names in
        # rank order, generation 1 for both, gpu ids [4,5] and [6,7], self addresses equal to the
        # last two entries of the address listing, and actor handles (looked up with the expected
        # generation) equal to the last two fake handles.
        """A consumer asking about one cell gets that cell's workers, in rank order, fully described."""
        spec = _make_spec(
            "engine",
            num_cells=2,
            num_workers_per_cell=2,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=2,
            pg_name="rollout",
        )
        manager = await _launch([spec], _make_pgs(num_slots=8))

        infos = manager.get_worker_infos("engine-00001")

        assert [info.name for info in infos] == ["engine-00001-00000", "engine-00001-00001"]
        assert [info.generation for info in infos] == [1, 1]
        assert [info.gpu_ids for info in infos] == [[4, 5], [6, 7]]
        assert [info.self_addrs for info in infos] == manager.get_addrs()["engine"][2:]
        assert [
            manager.get_actor_handle(info.name, expected_generation=info.generation) for info in infos
        ] == fake_ray_cluster.handles[2:]

    async def test_a_stopped_cell_reports_no_workers_instead_of_raising(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] After stopping the cell, the worker infos are an empty list.
        """A cell stopped between snapshot and round-trip must report an empty worker list, not crash."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-00000"])

        infos = manager.get_worker_infos("engine-00000")

        assert infos == []


class TestGetActorHandle:
    # [CLASS DOC] Tests of generation checking when resolving an actor handle.
    async def test_a_cell_restarted_after_its_info_was_read_refuses_the_new_generation_handle(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] Reads a worker's info (generation 1), stops and restarts its cell, then
        # asks for the handle with the old generation; expects an assertion error saying the worker
        # is now generation 2, not the 1 it was described as.
        """A stale worker description must not resolve to the replacement generation's actor."""
        manager = await _launch([_make_spec("engine")])
        cell_id = next(iter(manager.get_cell_infos(pool_ids=["engine"])))
        worker_info = manager.get_worker_infos(cell_id)[0]

        await manager.stop_cells([cell_id])
        await manager.start_cells([cell_id])

        with pytest.raises(AssertionError, match="is now generation 2, not the 1 it was described as"):
            manager.get_actor_handle(worker_info.name, expected_generation=worker_info.generation)


class TestGetWorkerInfosErrors:
    # [CLASS DOC] Tests of invalid arguments to the worker description listing.
    async def test_an_unknown_pool_is_reported_instead_of_guessed(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A cell id from a pool never launched raises an assertion error.
        """Asking about a spec the manager never launched must fail loudly."""
        manager = await _launch([_make_spec("engine")])

        with pytest.raises(AssertionError):
            manager.get_worker_infos("router-00000")

    async def test_a_cell_index_beyond_the_group_is_reported(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Cell index 2 of a 2-cell pool raises an assertion error.
        """Asking about a cell that does not exist must fail rather than return another cell."""
        manager = await _launch([_make_spec("engine", num_cells=2)])

        with pytest.raises(AssertionError):
            manager.get_worker_infos("engine-00002")


class TestGetCellInfos:
    # [CLASS DOC] Tests of the cell description listing, a mapping from cell id to info.
    async def test_only_the_asked_for_specs_are_reported(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Asking for the engine pool among engine and router returns only the two
        # engine cell ids.
        """The controller owns engines only; reconciling a router would try to serve from it."""
        manager = await _launch([_make_spec("engine", num_cells=2), _make_spec("router")])

        infos = manager.get_cell_infos(pool_ids=["engine"])

        assert sorted(infos) == ["engine-00000", "engine-00001"]

    async def test_every_info_carries_the_spec_it_came_from(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The info's pool id equals the spec name.
        """The spec name is what makes a cell's role explicit instead of sniffed from its meta."""
        manager = await _launch([_make_spec("engine")])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].pool_id == "engine"

    async def test_an_unknown_pool_is_reported_instead_of_silently_empty(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Asking for a pool list including an unknown name raises an assertion error.
        """A renamed spec would otherwise make every consumer see zero cells and blame something else."""
        manager = await _launch([_make_spec("engine")])

        with pytest.raises(AssertionError):
            manager.get_cell_infos(pool_ids=["engine", "router"])

    async def test_asking_for_nothing_reports_nothing(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] An empty pool list returns an empty mapping.
        """A run with no engines of its own must not fall back to seeing everyone else's."""
        manager = await _launch([_make_spec("engine")])

        assert manager.get_cell_infos(pool_ids=[]) == {}

    async def test_every_info_says_whether_its_cell_still_has_a_process(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A freshly launched cell's info is marked alive.
        """A suspended cell must stay listed, or nothing could ever ask for it to be resumed."""
        manager = await _launch([_make_spec("engine")])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive

    async def test_each_cell_meta_is_computed_from_its_own_cell_index(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A spec copy with a meta callable returning twice the cell index; 3 cells
        # report meta values 0, 2, 4 in cell order.
        """Meta carries per-cell placement facts, so a shared index would mislabel every cell but one."""
        spec = _make_spec("engine", num_cells=3).model_copy(
            update={"meta": lambda ctx: {"gpu_offset": ctx.cell_index * 2}}
        )
        manager = await _launch([spec])

        infos = manager.get_cell_infos(pool_ids=["engine"])

        assert [infos[compute_cell_id(pool_id="engine", cell_index=cell_index)].meta for cell_index in range(3)] == [
            {"gpu_offset": 0},
            {"gpu_offset": 2},
            {"gpu_offset": 4},
        ]


class TestStartAndStopCells:
    # [CLASS DOC] Tests of the public stop and start operations by cell id, including idempotence,
    # errors and rollback of failed starts.
    async def test_stopping_by_id_releases_only_the_named_cell(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Stopping cell 0 of 2 leaves it not alive, cell 1 alive, and only the first
        # actor killed.
        """Suspending one engine must leave its siblings serving."""
        manager = await _launch([_make_spec("engine", num_cells=2)])

        await manager.stop_cells(["engine-00000"])

        infos = manager.get_cell_infos(pool_ids=["engine"])
        assert not infos["engine-00000"].alive
        assert infos["engine-00001"].alive
        assert [handle.killed for handle in fake_ray_cluster.handles] == [True, False]

    async def test_a_stopped_cell_comes_back_on_start(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Stop then start cell 0; all cells report alive.
        """Resume is a fresh launch of the same cell, so the cell reappears to the reconciler."""
        manager = await _launch([_make_spec("engine", num_cells=2)])
        await manager.stop_cells(["engine-00000"])

        await manager.start_cells(["engine-00000"])

        assert all(info.alive for info in manager.get_cell_infos(pool_ids=["engine"]).values())

    async def test_a_restarted_cell_reports_a_new_workers_hash(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The workers hash after stop plus start differs from the hash before.
        """The hash is what tells the trainer to rebuild its connection to the new process."""
        manager = await _launch([_make_spec("engine")])
        before = manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].workers_hash

        await manager.stop_cells(["engine-00000"])
        await manager.start_cells(["engine-00000"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].workers_hash != before

    async def test_a_restarted_cell_runs_its_command_again(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] After stop and start, the command was run twice in total.
        """Resume must relaunch the subprocess, not merely re-register the old actors."""
        manager = await _launch([_make_spec("engine")])

        await manager.stop_cells(["engine-00000"])
        await manager.start_cells(["engine-00000"])

        assert len(fake_ray_cluster.calls_of("run")) == 2

    async def test_starting_a_running_cell_leaves_it_alone(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Starting a running cell creates no new handles.
        """Relaunching a live cell would orphan its current actors, so a repeated resume is a no-op."""
        manager = await _launch([_make_spec("engine")])
        handles_before = list(fake_ray_cluster.handles)

        await manager.start_cells(["engine-00000"])

        assert fake_ray_cluster.handles == handles_before

    async def test_stopping_an_already_stopped_cell_is_a_noop(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Stopping twice produces a single kill event.
        """Heal loops retry, so a redundant suspend must not blow up on missing actors."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-00000"])

        await manager.stop_cells(["engine-00000"])

        assert fake_ray_cluster.events.count(EVENT_KILL) == 1

    async def test_starting_an_already_running_cell_is_a_noop(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Starting a running cell adds no creation event and the cell stays alive.
        # KNOWN ISSUE: near-duplicate of test_starting_a_running_cell_leaves_it_alone above.
        """A repeated resume must not relaunch a live cell, nor fail the request that asked for it."""
        manager = await _launch([_make_spec("engine")])
        created_before = fake_ray_cluster.events.count(EVENT_CREATE)

        await manager.start_cells(["engine-00000"])

        assert fake_ray_cluster.events.count(EVENT_CREATE) == created_before
        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive

    async def test_a_cell_whose_actor_is_already_dead_still_stops(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The worker's shutdown call fails as if the actor died; stopping the cell
        # still succeeds and the cell is not alive.
        """Suspend is step one of healing a crashed cell, so a dead actor is the normal case."""
        manager = await _launch([_make_spec("engine")])
        fake_ray_cluster.handles[0].failing_methods["shutdown"] = RuntimeError("actor died")

        await manager.stop_cells(["engine-00000"])

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive

    async def test_several_cells_are_stopped_together(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] One stop call naming cells 0 and 2 of 3 leaves both not alive.
        """A multi-cell suspend is one request, not a caller-side loop."""
        manager = await _launch([_make_spec("engine", num_cells=3)])

        await manager.stop_cells(["engine-00000", "engine-00002"])

        assert not any(manager.get_cell_infos(pool_ids=["engine"])[c].alive for c in ["engine-00000", "engine-00002"])

    async def test_a_cell_is_not_reported_alive_until_every_worker_has_its_ports(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] A 2-worker cell is alive; after clearing worker 1's address map to none it
        # is reported not alive.
        """An observer builds cells out of this report, so a worker it could not address must not appear in one."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        actors = manager._pools["engine"].cells[0].actors

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive

        actors[1].self_addrs = None

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive

    async def test_a_half_started_cell_is_withheld_without_withholding_its_healthy_siblings(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] Cell 1's worker 0 loses its address map; only cell 0 is reported alive, and
        # each worker info of cell 0 holds a primary address.
        """An observer must still reconcile the rest of the round, which one unaddressable worker used to abort."""
        manager = await _launch([_make_spec("engine", num_cells=2, num_workers_per_cell=2)])
        manager._pools["engine"].cells[1].actors[0].self_addrs = None

        alive_cell_ids = [
            cell_id for cell_id, info in manager.get_cell_infos(pool_ids=["engine"]).items() if info.alive
        ]

        assert alive_cell_ids == ["engine-00000"]
        for info in manager.get_worker_infos("engine-00000"):
            assert "primary" in info.self_addrs, f"{info.name} is reported alive but describes no ports"

    async def test_an_unknown_cell_id_fails_loudly(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Stopping a nonexistent cell id raises an assertion error.
        """A typo'd cell id must not silently suspend nothing."""
        manager = await _launch([_make_spec("engine")])

        with pytest.raises(AssertionError):
            await manager.stop_cells(["engine-00007"])

    async def test_starting_an_unknown_cell_id_fails_loudly(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Starting a nonexistent cell id raises an assertion error.
        """A typo'd cell id must not silently start nothing while the caller believes it resumed."""
        manager = await _launch([_make_spec("engine")])

        with pytest.raises(AssertionError):
            await manager.start_cells(["engine-00007"])

    async def test_a_cell_that_fails_while_allocating_ports_is_rolled_back_and_can_be_retried(
        self, fake_ray_cluster: FakeRayCluster
    ):
        # [FUNCTION DOC] After stopping the cell, makes the node-address lookup fail; a start raises
        # that error, leaves the cell not alive and kills the newly created actor. Removing the
        # failure and starting again brings the cell alive, with 2 command runs in total.
        """A resume dying mid-setup must leave the cell stopped, else every later resume is skipped as a no-op."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-00000"])
        fake_ray_cluster.method_errors["_get_node_ip"] = RuntimeError("node gone")

        with pytest.raises(RuntimeError, match="node gone"):
            await manager.start_cells(["engine-00000"])

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive
        assert fake_ray_cluster.handles[-1].killed

        del fake_ray_cluster.method_errors["_get_node_ip"]
        await manager.start_cells(["engine-00000"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive
        assert len(fake_ray_cluster.calls_of("run")) == 2

    async def test_a_cell_that_fails_its_post_setup_is_rolled_back(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The launch command renders normally at init, then is flipped to raise a
        # runtime error before a restart; the start raises it, the cell is not alive and the newest
        # actor is killed.
        """Failing after the actors exist must still release them, or the cell stays alive but never serves."""
        command_fails = False

        def launch_command(ctx: LaunchCommandContext) -> str:
            # [FUNCTION DOC] HELPER: Raises a runtime error when the enclosing flag is set,
            # otherwise returns a fixed sleep command.
            if command_fails:
                raise RuntimeError("cannot render command")
            return "sleep 600"

        manager = await _launch([_make_spec("engine", launch_command=launch_command)])
        await manager.stop_cells(["engine-00000"])
        command_fails = True

        with pytest.raises(RuntimeError, match="cannot render command"):
            await manager.start_cells(["engine-00000"])

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive
        assert fake_ray_cluster.handles[-1].killed

    async def test_a_failed_start_leaves_the_running_cells_alone(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Cell 0 stopped and its restart made to fail; afterwards cell 0 is not
        # alive and cell 1 is still alive.
        """Rollback must be scoped to the cell that failed, not to the siblings that are already serving."""
        manager = await _launch([_make_spec("engine", num_cells=2)])
        await manager.stop_cells(["engine-00000"])
        fake_ray_cluster.method_errors["_get_node_ip"] = RuntimeError("node gone")

        with pytest.raises(RuntimeError, match="node gone"):
            await manager.start_cells(["engine-00000"])

        infos = manager.get_cell_infos(pool_ids=["engine"])
        assert not infos["engine-00000"].alive
        assert infos["engine-00001"].alive

    async def test_a_restarted_cell_allocates_ports_and_addrs_again(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] After stop and start the worker has an address and two node-address lookups
        # happened in total.
        """The new process needs its own addresses; a stale addr book would point at the dead one."""
        manager = await _launch([_make_spec("engine")])

        await manager.stop_cells(["engine-00000"])
        await manager.start_cells(["engine-00000"])

        assert manager.get_worker_addrs("engine-00000-00000")["primary"] is not None
        assert len(fake_ray_cluster.calls_of("_get_node_ip")) == 2

    async def test_start_runs_every_phase_before_the_next_one(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Clears the call log after a stop, restarts a 2-worker cell and asserts the
        # first command run comes after the last node-address lookup.
        """A worker must be addressable before any worker of the cell is handed its command."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        await manager.stop_cells(["engine-00000"])
        fake_ray_cluster.calls.clear()

        await manager.start_cells(["engine-00000"])

        methods = [call.method for call in fake_ray_cluster.calls]
        assert methods.index("run") > max(i for i, m in enumerate(methods) if m == "_get_node_ip")

    async def test_a_suspended_sibling_does_not_break_the_addr_book(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] After stopping cell 0 of 2, the address listing's lists have lengths [1]
        # (one pool, one live cell).
        """Launch commands are rendered from every spec's addrs, so a suspended cell must drop out of it."""
        manager = await _launch([_make_spec("engine", num_cells=2)])

        await manager.stop_cells(["engine-00000"])

        assert [len(addrs) for addrs in manager.get_addrs().values()] == [1]

    async def test_a_cell_restarts_while_a_sibling_stays_suspended(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Of 3 cells, stops 0 and 1, restarts 1; cell 0 stays not alive, cells 1 and
        # 2 are alive.
        """Healing one cell must not depend on every other cell being up."""
        manager = await _launch([_make_spec("engine", num_cells=3)])
        await manager.stop_cells(["engine-00000", "engine-00001"])

        await manager.start_cells(["engine-00001"])

        infos = manager.get_cell_infos(pool_ids=["engine"])
        assert not infos["engine-00000"].alive
        assert infos["engine-00001"].alive and infos["engine-00002"].alive

    async def test_worker_addrs_ignore_suspended_cells(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] With cell 0 stopped, the address lookup for a cell 1 worker still works.
        """Resolving a live worker must not walk into a suspended cell's missing actors."""
        manager = await _launch([_make_spec("engine", num_cells=2)])

        await manager.stop_cells(["engine-00000"])

        assert manager.get_worker_addrs("engine-00001-00000")["primary"] is not None


class TestStartCellsRollback:
    # [CLASS DOC] Tests that a start request is transactional: on failure the cells it touched are
    # rolled back and killed, and a later start works.
    async def test_a_resume_that_fails_midway_leaves_the_cell_suspended(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Port allocation is patched to fail during a restart; the start raises and
        # the cell is reported not alive.
        """A half-started cell that still looked alive would never be retried by the heal loop."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-00000"])

        async def failing_alloc(self) -> None:
            # [FUNCTION DOC] HELPER: Replacement allocation that always raises a runtime error.
            raise RuntimeError("no ports")

        with pytest.raises(RuntimeError, match="no ports"):
            with pytest.MonkeyPatch.context() as patched:
                patched.setattr(ray_worker_manager._CommandActorManager, "alloc_ports", failing_alloc)
                await manager.start_cells(["engine-00000"])

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive

    async def test_a_resume_that_fails_midway_kills_the_actors_it_created(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Post-setup is patched to fail during a restart; the start raises and both
        # handles (original and new) are killed.
        """Leaked actors keep holding their GPUs, so the next resume would find no capacity."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-00000"])

        async def failing_post_setup(self) -> None:
            # [FUNCTION DOC] HELPER: Replacement post-setup that always raises a runtime error.
            raise RuntimeError("cannot render")

        with pytest.raises(RuntimeError, match="cannot render"):
            with pytest.MonkeyPatch.context() as patched:
                patched.setattr(ray_worker_manager._CommandActorManager, "post_setup", failing_post_setup)
                await manager.start_cells(["engine-00000"])

        assert [handle.killed for handle in fake_ray_cluster.handles] == [True, True]

    async def test_the_resume_after_a_failed_one_starts_the_cell_for_real(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] After one failed restart (patch removed), a second start succeeds: the cell
        # is alive and the command has run twice overall.
        """The whole point of rolling back is that the very next resume attempt is not skipped."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-00000"])

        async def failing_alloc(self) -> None:
            # [FUNCTION DOC] HELPER: Replacement allocation that always raises a runtime error.
            raise RuntimeError("no ports")

        with pytest.raises(RuntimeError, match="no ports"):
            with pytest.MonkeyPatch.context() as patched:
                patched.setattr(ray_worker_manager._CommandActorManager, "alloc_ports", failing_alloc)
                await manager.start_cells(["engine-00000"])

        await manager.start_cells(["engine-00000"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].alive
        assert len(fake_ray_cluster.calls_of("run")) == 2

    async def test_a_failed_start_leaves_no_late_sibling_actor_alive(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] During init of 2 cells, launching cell 0's actor raises "no capacity" while
        # cell 1 launches 50 ms after that. After init fails and 0.1 seconds pass, asserts every
        # created actor was killed and no cell is alive.
        """A sibling still launching when the request failed must not survive the rollback holding its gpus."""
        from miles.utils.workers.ray_worker_manager import _CommandActorManager

        original_launch_actor = _CommandActorManager.launch_actor
        first_failed = asyncio.Event()

        async def staggered_launch(self) -> None:
            # [FUNCTION DOC] HELPER: Cell 0 signals first_failed and raises a runtime error; other
            # cells wait for that signal, sleep 50 ms, then run the original launch.
            if self.parent.cell_index == 0:
                first_failed.set()
                raise RuntimeError("no capacity")
            await first_failed.wait()
            await asyncio.sleep(0.05)
            await original_launch_actor(self)

        manager = RayWorkerManager()
        with pytest.raises(RuntimeError, match="no capacity"):
            with pytest.MonkeyPatch.context() as patched:
                patched.setattr(_CommandActorManager, "launch_actor", staggered_launch)
                await manager.init(
                    worker_manager_args(), [_make_spec("engine", num_cells=2)], {}, comm_backend=WorkerCommBackend.RAY
                )
        await asyncio.sleep(0.1)

        assert all(handle.killed for handle in fake_ray_cluster.handles)
        assert not any(info.alive for info in manager.get_cell_infos(pool_ids=["engine"]).values())

    async def test_a_failed_start_rolls_back_the_siblings_of_the_failing_cell(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Post-setup is patched to fail during init of 2 cells; init raises and no
        # cell is alive afterwards.
        """One request is one transaction, so no cell of it may survive half configured."""
        spec = _make_spec("engine", num_cells=2)

        async def failing_post_setup(self) -> None:
            # [FUNCTION DOC] HELPER: Replacement post-setup that always raises a runtime error.
            raise RuntimeError("cannot render")

        manager = RayWorkerManager()
        with pytest.raises(RuntimeError, match="cannot render"):
            with pytest.MonkeyPatch.context() as patched:
                patched.setattr(ray_worker_manager._CommandActorManager, "post_setup", failing_post_setup)
                await manager.init(worker_manager_args(), [spec], {}, comm_backend=WorkerCommBackend.RAY)

        assert not any(info.alive for info in manager.get_cell_infos(pool_ids=["engine"]).values())


class TestSuspendedCellInfos:
    # [CLASS DOC] Tests of how a stopped cell appears in descriptions.
    async def test_a_suspended_cell_is_still_described(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] After stopping cell 0 of 2, both cells are listed; 0 not alive, 1 alive.
        """The api server must still list a suspended cell, or it could never be resumed."""
        manager = await _launch([_make_spec("engine", num_cells=2)])
        await manager.stop_cells(["engine-00000"])

        infos = manager.get_cell_infos(pool_ids=["engine"])

        assert sorted(infos) == ["engine-00000", "engine-00001"]
        assert not infos["engine-00000"].alive
        assert infos["engine-00001"].alive

    async def test_the_meta_of_a_suspended_cell_is_still_known(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A spec with a constant meta callable; after stop the cell info still has
        # that meta.
        """Meta comes from the spec, so suspending a cell must not make it unidentifiable."""
        spec = _make_spec("engine").model_copy(update={"meta": lambda ctx: {"model_id": "default"}})
        manager = await _launch([spec])
        await manager.stop_cells(["engine-00000"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].meta == {"model_id": "default"}

    async def test_a_suspended_cell_reports_no_workers(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] A stopped cell's info has an empty worker-name list.
        """Its actors are gone, so anything reading worker names off it must get an empty list."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-00000"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-00000"].worker_names == []


class TestInjectFault:
    # [CLASS DOC] Tests of the fire-and-forget fault injection into one worker of a cell.
    async def test_the_fault_reaches_the_selected_worker(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Injecting a sigkill into worker 1 of a 2-worker cell sends one inject call
        # with the mode as the only argument, to the second handle.
        """A multi-node engine is crashed by crashing one of its node ranks."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])

        manager.inject_fault("engine-00000", mode="sigkill", worker_in_cell_index=1)

        calls = fake_ray_cluster.calls_of("inject_fault")
        assert [call.args for call in calls] == [("sigkill",)]
        assert calls[0].handle is fake_ray_cluster.handles[1]

    async def test_injection_does_not_wait_for_the_worker_to_answer(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] The worker's inject call is set to fail; the manager call still returns
        # without raising (only absence of an error is checked).
        """The worker is about to die, so waiting for its reply would hang the caller."""
        manager = await _launch([_make_spec("engine")])
        fake_ray_cluster.handles[0].failing_methods["inject_fault"] = RuntimeError("actor died")

        manager.inject_fault("engine-00000", mode="sigkill", worker_in_cell_index=0)

    async def test_injecting_into_a_suspended_cell_is_rejected(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] After stopping the cell, injection raises a runtime error mentioning "not
        # alive".
        """A suspended cell has no worker to crash."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-00000"])

        with pytest.raises(RuntimeError, match="not alive"):
            manager.inject_fault("engine-00000", mode="sigkill", worker_in_cell_index=0)

    async def test_a_worker_index_beyond_the_cell_is_rejected(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Worker index 1 in a one-worker cell raises an index error mentioning "out
        # of range".
        """Injecting into a neighbouring cell by accident would corrupt the test's premise."""
        manager = await _launch([_make_spec("engine", num_cells=2, num_workers_per_cell=1)])

        with pytest.raises(IndexError, match="out of range"):
            manager.inject_fault("engine-00000", mode="sigkill", worker_in_cell_index=1)

    async def test_a_negative_worker_index_is_rejected(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] Worker index minus one raises an index error mentioning "out of range".
        """Negative indexing would silently select the last worker instead of failing."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])

        with pytest.raises(IndexError, match="out of range"):
            manager.inject_fault("engine-00000", mode="sigkill", worker_in_cell_index=-1)

    async def test_an_unknown_cell_is_rejected(self, fake_ray_cluster: FakeRayCluster):
        # [FUNCTION DOC] An unknown cell id raises an assertion error.
        """A typo must not silently inject nothing."""
        manager = await _launch([_make_spec("engine")])

        with pytest.raises(AssertionError):
            manager.inject_fault("engine-00007", mode="sigkill", worker_in_cell_index=0)
