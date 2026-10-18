# [MODULE DOC]
# Tests of the Ray worker manager actor against a real local Ray runtime. Workers are probe
# processes (see the sibling conftest) that record their launch context. Covered: launch of every
# worker with its own context and advertised address, port binding, spec environment variables,
# static ports and conflicts, disabled specs, named manager lookup, shutdown, scale, relaunch,
# cross-spec address wiring, per-cell master ports, placement groups, per-cell stop and restart,
# worker info retrieval, and death of the actor and its command.
from __future__ import annotations

import asyncio
import os
import socket
import sys
import time

import pytest
import ray
from tests.fast.utils.workers.real_ray.conftest import (
    is_process_running,
    kill_quietly,
    make_command_spec,
    wait_until_named_manager_is_gone,
)

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.ray.wiring import shutdown_worker_manager
from miles.utils.http_utils import wait_tcp_ready_async
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort, PortInfo


class TestLaunchOnRealRay:
    # [CLASS DOC] Launch behavior of the manager.
    def test_every_worker_of_every_cell_ends_up_running_its_own_command(self, manager_factory, worker_probe_factory):
        # [FUNCTION DOC] Launches a pool of 2 cells with 2 workers each; waits for 4 records.
        # Asserts record keys 0-0, 0-1, 1-0, 1-1; all four pids are running and distinct; each
        # recorded context has the right cell and worker indexes; and the recorded primary
        # address equals (host, port, external host) advertised by the manager for that worker.
        """The manager starts one live subprocess per worker, each with its own launch context."""
        probe = worker_probe_factory()
        handle = manager_factory(
            [make_command_spec("engine", num_cells=2, num_workers_per_cell=2, launch_command=probe.launch_command)]
        )

        records = probe.wait_for_records(4)

        assert sorted(records) == ["0-0", "0-1", "1-0", "1-1"]
        assert all(is_process_running(record["pid"]) for record in records.values())
        assert len({record["pid"] for record in records.values()}) == 4
        for name, record in records.items():
            cell_index, worker_in_cell_index = (int(part) for part in name.split("-"))
            assert record["context"]["cell_index"] == cell_index
            assert record["context"]["worker_in_cell_index"] == worker_in_cell_index
            advertised = ray.get(
                handle.get_worker_addrs.remote(
                    compute_worker_name(
                        pool_id="engine",
                        cell_index=cell_index,
                        worker_in_cell_index=worker_in_cell_index,
                    )
                )
            )["primary"]
            assert record["context"]["self_addrs"]["primary"] == {
                "host": advertised.host,
                "port": advertised.port,
                "external_host": advertised.external_host,
            }

    async def test_the_advertised_address_is_one_the_worker_can_serve_on(self, manager_factory, worker_probe_factory):
        # [FUNCTION DOC] Launches 3 probe workers in one cell that bind their primary port.
        # Asserts the three advertised host and port pairs are distinct and that each accepts a
        # tcp connection within 30 s.
        """A worker can bind the port allocated for it, and that endpoint is what the manager advertises."""
        probe = worker_probe_factory(bind_primary=True)
        handle = manager_factory(
            [make_command_spec("engine", num_workers_per_cell=3, launch_command=probe.launch_command)]
        )

        probe.wait_for_records(3)
        addrs = [
            ray.get(handle.get_worker_addrs.remote(compute_worker_name(pool_id="engine", worker_in_cell_index=index)))[
                "primary"
            ]
            for index in range(3)
        ]

        assert len({(addr.host, addr.port) for addr in addrs}) == 3
        for addr in addrs:
            await wait_tcp_ready_async(addr.host, addr.port, timeout=30)

    def test_the_worker_process_gets_the_env_vars_declared_by_its_spec(self, manager_factory, worker_probe_factory):
        # [FUNCTION DOC] Launches a spec declaring one environment variable; asserts the probe
        # recorded exactly that variable with its value.
        """Env vars from the spec are visible inside the launched process."""
        probe = worker_probe_factory(env_names=("MILES_REAL_RAY_PROBE_VAR",))
        manager_factory(
            [
                make_command_spec(
                    "router",
                    launch_command=probe.launch_command,
                    env_var={"MILES_REAL_RAY_PROBE_VAR": "from-spec"},
                )
            ]
        )

        records = probe.wait_for_records(1)

        assert records["0-0"]["env"] == {"MILES_REAL_RAY_PROBE_VAR": "from-spec"}

    def test_a_static_port_reaches_the_worker_unchanged(self, manager_factory, worker_probe_factory):
        # [FUNCTION DOC] Launches a spec pinning primary port 21987 with dynamic allocation off;
        # asserts both the recorded context and the manager's advertised address use that port.
        """A spec that pins its port keeps it instead of being handed an allocated one."""
        probe = worker_probe_factory()
        handle = manager_factory(
            [
                make_command_spec(
                    "router",
                    launch_command=probe.launch_command,
                    port_infos=[PortInfo(name="primary", static_port=21987, allow_dynamic=False)],
                )
            ]
        )

        records = probe.wait_for_records(1)

        assert records["0-0"]["context"]["self_addrs"]["primary"]["port"] == 21987
        assert ray.get(handle.get_worker_addrs.remote("router-00000-00000"))["primary"].port == 21987

    def test_a_static_port_a_stale_process_still_holds_is_refused(self, manager_factory, worker_probe_factory):
        # [FUNCTION DOC] Opens a listening socket on an ephemeral port to act as a stale
        # process, then launches a spec pinning that port without dynamic fallback; asserts the
        # launch raises an error matching "Port <n> ... is already in use".
        """Readiness is a bare connect probe, which a stale listener satisfies, so a run that
        launched anyway would silently drive the router an earlier run left behind."""
        probe = worker_probe_factory()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squatter:
            squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            squatter.bind(("", 0))
            squatter.listen(1)
            taken = squatter.getsockname()[1]

            with pytest.raises(Exception, match=f"Port {taken} .* is already in use"):
                manager_factory(
                    [
                        make_command_spec(
                            "router",
                            launch_command=probe.launch_command,
                            port_infos=[PortInfo(name="primary", static_port=taken, allow_dynamic=False)],
                        )
                    ]
                )

    def test_a_spec_without_cells_launches_no_worker(self, manager_factory, worker_probe_factory):
        # [FUNCTION DOC] Launches a zero-cell spec next to a normal one; asserts the disabled
        # spec produced no records and the normal one's worker has a positive advertised port.
        """A disabled spec is accepted and simply contributes no workers."""
        disabled_probe = worker_probe_factory()
        enabled_probe = worker_probe_factory()
        handle = manager_factory(
            [
                make_command_spec("session-server", num_cells=0, launch_command=disabled_probe.launch_command),
                make_command_spec("router", launch_command=enabled_probe.launch_command),
            ]
        )

        enabled_probe.wait_for_records(1)

        assert disabled_probe.read_records() == {}
        assert ray.get(handle.get_worker_addrs.remote("router-00000-00000"))["primary"].port > 0


class TestNamedManagerActor:
    # [CLASS DOC] Lookup of the manager by its well-known name.
    async def test_a_driver_knowing_only_the_worker_name_reaches_its_live_endpoint(
        self, manager_factory, worker_probe_factory
    ):
        # [FUNCTION DOC] Launches one binding probe, builds a provider on the named manager,
        # resolves the worker's primary address by name; asserts it is a host-and-port value
        # equal to the recorded one and accepts tcp connections within 30 s.
        """The provider resolves a worker's real endpoint through the named manager actor."""
        probe = worker_probe_factory(bind_primary=True)
        manager_factory([make_command_spec("router", launch_command=probe.launch_command)])
        records = probe.wait_for_records(1)

        provider = RayWorkerProvider(worker_manager_handle=RayWorkerManager.get_handle())
        addr = (await provider.get_addrs(worker_name="router-00000-00000"))["primary"]

        assert isinstance(addr, HostAndPort)
        assert records["0-0"]["context"]["self_addrs"]["primary"] == {
            "host": addr.host,
            "port": addr.port,
            "external_host": addr.external_host,
        }
        await wait_tcp_ready_async(addr.host, addr.port, timeout=30)

    def test_an_unknown_worker_name_is_not_answered_with_another_workers_address(
        self, manager_factory, worker_probe_factory
    ):
        # [FUNCTION DOC] Launches one worker, then asks the named manager for an unknown worker
        # name; asserts the call raises a Ray task error.
        """The lookup forwards the requested name and fails when nothing matches it."""
        probe = worker_probe_factory()
        manager_factory([make_command_spec("router", launch_command=probe.launch_command)])
        probe.wait_for_records(1)

        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(RayWorkerManager.get_handle().get_worker_addrs.remote("router-00009-00009"))


class TestRunShutdownOnRealRay:
    # [CLASS DOC] Driver-side shutdown of the manager.
    async def test_shutdown_terminates_every_owned_worker_process(self, manager_factory, worker_probe_factory):
        # [FUNCTION DOC] Launches 2 cells, calls the driver shutdown function on the manager,
        # waits for every recorded pid to vanish, and for the named actor to be unregistered.
        """Normal driver completion must reclaim the exact subprocesses its manager launched."""
        probe = worker_probe_factory()
        handle = manager_factory([make_command_spec("engine", num_cells=2, launch_command=probe.launch_command)])
        records = probe.wait_for_records(2)

        await shutdown_worker_manager(handle)

        probe.wait_until_gone([record["pid"] for record in records.values()])
        wait_until_named_manager_is_gone()


class TestScaleOnRealRay:
    # [CLASS DOC] Port allocation across many workers.
    def test_a_larger_pool_still_gets_disjoint_port_blocks(self, manager_factory, worker_probe_factory):
        # [FUNCTION DOC] Launches 3 cells of 2 workers, each with a dynamic primary port and a
        # dynamic 3-port nccl block; collects the primary port and the three nccl ports of every
        # worker (6 workers) and asserts no port appears twice.
        """Six workers with multi-port specs must not overlap, including inside reserved blocks."""
        probe = worker_probe_factory()
        manager_factory(
            [
                make_command_spec(
                    "engine",
                    num_cells=3,
                    num_workers_per_cell=2,
                    launch_command=probe.launch_command,
                    port_infos=[
                        PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                        PortInfo(name="nccl", static_port=10000, allow_dynamic=True, num_consecutive=3),
                    ],
                )
            ]
        )

        records = probe.wait_for_records(6)

        reserved: list[int] = []
        for record in records.values():
            addrs = record["context"]["self_addrs"]
            reserved.append(addrs["primary"]["port"])
            reserved.extend(range(addrs["nccl"]["port"], addrs["nccl"]["port"] + 3))
        assert len(reserved) == len(set(reserved))


class TestManagerRelaunchOnRealRay:
    # [CLASS DOC] Reusing the manager name.
    def test_the_well_known_name_can_be_reused_after_the_manager_is_gone(self, manager_factory, worker_probe_factory):
        # [FUNCTION DOC] Launches a manager, kills it and waits for the name to free up, then
        # launches a second one with a new probe; asserts the second probe's pid differs from
        # the first probe's recorded pid.
        """A restarted driver must be able to claim the manager name again."""
        first_probe = worker_probe_factory()
        first_handle = manager_factory([make_command_spec("router", launch_command=first_probe.launch_command)])
        first_probe.wait_for_records(1)

        kill_quietly(first_handle)
        wait_until_named_manager_is_gone()

        second_probe = worker_probe_factory()
        manager_factory([make_command_spec("router", launch_command=second_probe.launch_command)])

        assert second_probe.wait_for_records(1)["0-0"]["pid"] != first_probe.read_records()["0-0"]["pid"]


class TestCrossSpecWiringOnRealRay:
    # [CLASS DOC] Visibility of other pools' addresses and isolation of environments.
    def test_a_dependent_workers_command_sees_the_addresses_of_the_spec_it_depends_on(
        self, manager_factory, worker_probe_factory
    ):
        # [FUNCTION DOC] Launches a one-cell router spec and a two-cell session-server spec.
        # Asserts each session-server record's pool address table lists exactly both pools, the
        # router's first-cell primary address equals the manager's advertised one, and the
        # session-server table has 2 entries.
        """A session-server-shaped worker can only find its router because the manager renders after all allocation."""
        router_probe = worker_probe_factory()
        session_probe = worker_probe_factory()
        handle = manager_factory(
            [
                make_command_spec("inference-router-0", launch_command=router_probe.launch_command),
                make_command_spec("session-server", num_cells=2, launch_command=session_probe.launch_command),
            ]
        )

        router_probe.wait_for_records(1)
        session_records = session_probe.wait_for_records(2)
        addrs = ray.get(handle.get_addrs.remote())

        router_addr = addrs["inference-router-0"][0]["primary"]
        for record in session_records.values():
            pool_addrs = record["context"]["pool_addrs"]
            assert sorted(pool_addrs) == ["inference-router-0", "session-server"]
            assert pool_addrs["inference-router-0"][0]["primary"] == router_addr.model_dump()
            assert len(pool_addrs["session-server"]) == 2

    def test_each_spec_only_gets_its_own_env(self, manager_factory, worker_probe_factory):
        # [FUNCTION DOC] Launches router and engine specs with different variables; asserts each
        # probe sees only its own variable and nothing for the other.
        """Env vars of one spec must not reach another spec's workers."""
        router_probe = worker_probe_factory(env_names=("ROUTER_ONLY", "ENGINE_ONLY"))
        engine_probe = worker_probe_factory(env_names=("ROUTER_ONLY", "ENGINE_ONLY"))
        manager_factory(
            [
                make_command_spec("router", launch_command=router_probe.launch_command, env_var={"ROUTER_ONLY": "r"}),
                make_command_spec("engine", launch_command=engine_probe.launch_command, env_var={"ENGINE_ONLY": "e"}),
            ]
        )

        router_record = router_probe.wait_for_records(1)["0-0"]
        engine_record = engine_probe.wait_for_records(1)["0-0"]

        assert router_record["env"] == {"ROUTER_ONLY": "r", "ENGINE_ONLY": None}
        assert engine_record["env"] == {"ROUTER_ONLY": None, "ENGINE_ONLY": "e"}


class TestMasterPortsOnRealRay:
    # [CLASS DOC] Master-mode ports are allocated once per cell by rank 0.
    def test_all_ranks_of_a_cell_are_launched_with_their_cells_master_endpoint(
        self, manager_factory, worker_probe_factory
    ):
        # [FUNCTION DOC] Launches 2 cells of 2 workers with a primary port and a 4-port master
        # block named dist_init. Asserts ranks in a cell share the same master endpoint and the
        # two cells differ; the advertised address key sets per worker alternate between
        # primary plus dist_init (rank 0) and primary only (rank 1); the four primary ports are
        # distinct; and no master block port collides with any primary port.
        """Every worker of a cell receives the same master endpoint, allocated once by rank 0."""
        probe = worker_probe_factory()
        handle = manager_factory(
            [
                make_command_spec(
                    "engine",
                    num_cells=2,
                    num_workers_per_cell=2,
                    launch_command=probe.launch_command,
                    port_infos=[
                        PortInfo(name="primary", static_port=8000, allow_dynamic=True),
                        PortInfo(
                            name="dist_init", static_port=9000, mode="master", allow_dynamic=True, num_consecutive=4
                        ),
                    ],
                )
            ]
        )

        records = probe.wait_for_records(4)
        masters = {name: record["context"]["self_addrs"]["dist_init"] for name, record in records.items()}

        assert masters["0-0"] == masters["0-1"]
        assert masters["1-0"] == masters["1-1"]
        assert masters["0-0"] != masters["1-0"]
        addrs = ray.get(handle.get_addrs.remote())["engine"]
        assert [sorted(addr) for addr in addrs] == [
            ["dist_init", "primary"],
            ["primary"],
            ["dist_init", "primary"],
            ["primary"],
        ]
        primary_ports = [addr["primary"].port for addr in addrs]
        assert len(set(primary_ports)) == 4
        for master in [masters["0-0"], masters["1-0"]]:
            reserved = range(master["port"], master["port"] + 4)
            assert not set(reserved) & set(primary_ports)


class TestPlacementOnRealRay:
    # [CLASS DOC] Placement group binding.
    def test_workers_run_inside_their_placement_group_bundles_with_their_gpu_slice(
        self, manager_factory, worker_probe_factory, placement_group_factory
    ):
        # [FUNCTION DOC] Creates a 4-bundle placement group whose gpu ids start at 10, launches
        # 2 cells with two gpu slots per worker (visibility-masking variables disabled).
        # Asserts the cells' gpu ids are [10, 11] and [12, 13], every worker sees the same
        # visible-devices value as the test process, and the manager advertises 2 workers.
        """A pg-bound spec starts every worker on its own bundle and hands it the gpu ids of its slots."""
        probe = worker_probe_factory(env_names=("CUDA_VISIBLE_DEVICES",))
        pgs = {"rollout": placement_group_factory(num_bundles=4, first_gpu_id=10)}
        handle = manager_factory(
            [
                make_command_spec(
                    "engine",
                    num_cells=2,
                    launch_command=probe.launch_command,
                    num_gpus_per_worker=0.5,
                    num_gpu_slots_per_worker=2,
                    pg_name="rollout",
                    env_var={name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
                )
            ],
            pgs,
        )

        records = probe.wait_for_records(2)

        assert records["0-0"]["context"]["gpu_ids"] == [10, 11]
        assert records["1-0"]["context"]["gpu_ids"] == [12, 13]
        assert {record["env"]["CUDA_VISIBLE_DEVICES"] for record in records.values()} == {
            os.environ.get("CUDA_VISIBLE_DEVICES")
        }
        assert len(ray.get(handle.get_addrs.remote())["engine"]) == 2


class TestStopCellOnRealRay:
    # [CLASS DOC] Per-cell stop and restart using the test-only stoppable manager.
    def test_stopping_a_cell_ends_its_worker_processes_and_leaves_the_others_alone(
        self, cell_stoppable_manager_factory, worker_probe_factory
    ):
        # [FUNCTION DOC] Launches 2 cells of 2 workers, stops cell 1; waits until its two pids
        # are gone and asserts the two pids of cell 0 are still running.
        """Tearing down one cell kills exactly that cell's worker processes."""
        probe = worker_probe_factory()
        manager_handle = cell_stoppable_manager_factory(
            [make_command_spec("engine", num_cells=2, num_workers_per_cell=2, launch_command=probe.launch_command)]
        )
        records = probe.wait_for_records(4)
        stopped_pids = [records["1-0"]["pid"], records["1-1"]["pid"]]
        surviving_pids = [records["0-0"]["pid"], records["0-1"]["pid"]]

        ray.get(manager_handle.stop_cell.remote("engine", 1))

        probe.wait_until_gone(stopped_pids)
        assert all(is_process_running(pid) for pid in surviving_pids)

    def test_a_restarted_cell_gets_new_actors_running_the_command_again(
        self, cell_stoppable_manager_factory, worker_probe_factory
    ):
        # [FUNCTION DOC] Launches 2 single-worker cells and reads cell 1's worker info; stops
        # cell 1, waits for its pid to vanish, starts it again. Asserts the worker name is
        # unchanged and its generation grew; waits up to 60 s for a record with a different pid
        # and asserts that pid and the surviving cell's pid are running.
        """Healing is only ever exercised against the fake cluster, and a fake handle cannot show
        that the dead slot is refilled by a genuinely new actor rather than the corpse of the old one."""
        probe = worker_probe_factory()
        manager_handle = cell_stoppable_manager_factory(
            [make_command_spec("engine", num_cells=2, num_workers_per_cell=1, launch_command=probe.launch_command)]
        )
        probe.wait_for_records(2)
        provider = RayWorkerProvider(worker_manager_handle=RayWorkerManager.get_handle())
        (before,) = provider.get_worker_infos(cell_ids=["engine-00001"])
        stopped_pid = probe.read_records()["1-0"]["pid"]
        survivor_pid = probe.read_records()["0-0"]["pid"]

        ray.get(manager_handle.stop_cell.remote("engine", 1))
        probe.wait_until_gone([stopped_pid])
        ray.get(manager_handle.start_cells.remote(["engine-00001"]))

        (after,) = provider.get_worker_infos(cell_ids=["engine-00001"])
        assert after[0].name == before[0].name
        assert after[0].generation > before[0].generation
        deadline = time.monotonic() + 60
        while (restarted_pid := probe.read_records()["1-0"]["pid"]) == stopped_pid:
            assert time.monotonic() < deadline, "the restarted worker never recorded a new process"
            time.sleep(0.2)
        assert is_process_running(restarted_pid)
        assert is_process_running(survivor_pid)


class TestWorkerInfosOnRealRay:
    # [CLASS DOC] Worker info retrieval by a driver.
    async def test_a_driver_can_describe_and_reach_the_workers_of_one_cell(
        self, manager_factory, worker_probe_factory
    ):
        # [FUNCTION DOC] Launches 2 cells of 2 workers; reads infos and handles for cell
        # engine-00001. Asserts names, generation 1 for both, empty gpu ids, that each info's
        # primary address equals the one recorded by the probe, and that the host of the
        # address (brackets stripped) equals the node ip reported by the actor handle (private
        # method).
        """Worker infos survive the trip to the driver, including usable actor handles."""
        probe = worker_probe_factory()
        manager_factory(
            [make_command_spec("engine", num_cells=2, num_workers_per_cell=2, launch_command=probe.launch_command)]
        )
        records = probe.wait_for_records(4)

        provider = RayWorkerProvider(worker_manager_handle=RayWorkerManager.get_handle())
        (infos,) = provider.get_worker_infos(cell_ids=["engine-00001"])
        handles = provider.get_handles_of_worker_infos(infos)

        assert [info.name for info in infos] == ["engine-00001-00000", "engine-00001-00001"]
        assert [info.generation for info in infos] == [1, 1]
        assert [info.gpu_ids for info in infos] == [[], []]
        for worker_in_cell_index, info in enumerate(infos):
            recorded = records[f"1-{worker_in_cell_index}"]["context"]["self_addrs"]["primary"]
            assert info.self_addrs["primary"].model_dump() == recorded
            node_ip = await handles[info.name]._get_node_ip()
            assert info.self_addrs["primary"].host.strip("[]") == node_ip


class TestWorkerDeathOnRealRay:
    # [CLASS DOC] Death propagation between worker commands and their actors.
    async def test_an_actor_dies_with_the_command_it_babysits(self, manager_factory, worker_probe_factory):
        # [FUNCTION DOC] Launches 2 workers in one cell, asks the first to kill its subprocess
        # (an unreachable error from the lost reply is tolerated). Polls up to 60 s until
        # asking the first actor for its node ip fails as unreachable or times out (5 s per
        # try), then asserts the second actor still answers within 30 s.
        """The actor's whole reason to exist is its subprocess, so its death must be visible to the driver."""
        probe = worker_probe_factory()
        manager_factory([make_command_spec("engine", num_workers_per_cell=2, launch_command=probe.launch_command)])
        probe.wait_for_records(2)
        provider = RayWorkerProvider(worker_manager_handle=RayWorkerManager.get_handle())
        (infos,) = provider.get_worker_infos(cell_ids=["engine-00000"])
        handles = [provider.get_handle(info.name) for info in infos]

        # The babysit thread may reach os._exit before this reply is sent, so losing the reply is
        # the death under test arriving early rather than a failure; the loop below still has to
        # observe it, so nothing is taken on faith here.
        try:
            await handles[0].kill_subprocess()
        except WorkerUnreachableError:
            pass

        deadline = time.monotonic() + 60
        while True:
            try:
                await asyncio.wait_for(handles[0]._get_node_ip(), timeout=5)
            except (WorkerUnreachableError, asyncio.TimeoutError):
                break
            assert time.monotonic() < deadline, "the actor of a dead command is still alive"
            await asyncio.sleep(0.5)
        assert await asyncio.wait_for(handles[1]._get_node_ip(), timeout=30)

    @pytest.mark.skipif(sys.platform != "linux", reason="needs linux parent-death signals")
    def test_killing_the_manager_frees_the_ports_its_workers_commands_held(
        self, manager_factory, worker_probe_factory
    ):
        # [FUNCTION DOC] Linux only. Launches 2 binding probes, records their ports, kills the
        # manager, and polls up to 60 s until every port can be bound again; fails an assertion
        # naming the ports still held.
        """Owner death force-kills the worker actors, and the commands they launched must not keep their ports."""
        probe = worker_probe_factory(bind_primary=True)
        manager_handle = manager_factory(
            [make_command_spec("engine", num_workers_per_cell=2, launch_command=probe.launch_command)]
        )
        records = probe.wait_for_records(2)
        ports = [record["context"]["self_addrs"]["primary"]["port"] for record in records.values()]

        kill_quietly(manager_handle)

        deadline = time.monotonic() + 60
        while held := [port for port in ports if not _can_bind(port)]:
            assert time.monotonic() < deadline, f"ports {held} are still held after their manager was killed"
            time.sleep(0.2)


def _can_bind(port: int) -> bool:
    # [FUNCTION DOC] HELPER: Tries to bind a tcp socket (address reuse on) to all interfaces on
    # the port; true on success, false if the operating system refuses.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            candidate.bind(("", port))
        except OSError:
            return False
    return True
