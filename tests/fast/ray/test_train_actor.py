# [MODULE DOC]
# Unit tests for the base train actor class (TrainRayActor) and related driver wiring, run without
# gpus or ray: constructor signature, master address proposal and configuration, self kill,
# in-place state reload refusal, the run-once initialization guard, local gpu id discovery with and
# without ray, NUMA cpu affinity chosen from the masked physical gpu, and passing the actor's train
# parallel config to the rollout executor. Actors are built without running the constructor, and
# torch, distributed, NVML and ray calls are replaced through the monkeypatch fixture.
import os
import socket
import sys
from types import SimpleNamespace

import pytest

from miles.ray import placement_group, train_actor
from miles.ray.train_actor import TrainRayActor
from miles.utils.init_once import InitOnce
from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR, SUBPROCESS_INDEX_ENV_VAR


def _unimplemented_actor_operation(*args: object, **kwargs: object) -> None:
    # [FUNCTION DOC] HELPER: Accepts any arguments and always raises a not-implemented error; used
    # to fill the abstract operations of a minimal actor subclass.
    raise NotImplementedError


class _ActorWithoutReloadSupport(TrainRayActor):
    # [CLASS DOC] HELPER: Minimal subclass of the train actor whose abstract operations (init,
    # sleep, wake up, train, save model, update weights, parallel config) all raise not-implemented
    # errors. It does not override in-place state loading, so it exercises the base behavior.
    init = _unimplemented_actor_operation
    sleep = _unimplemented_actor_operation
    wake_up = _unimplemented_actor_operation
    train = _unimplemented_actor_operation
    save_model = _unimplemented_actor_operation
    update_weights = _unimplemented_actor_operation
    _get_parallel_config = _unimplemented_actor_operation


def _inited_guard() -> InitOnce:
    # [FUNCTION DOC] HELPER: Returns a run-once initialization guard (named after the actor class)
    # that has already been entered once, i.e. is in the initialized state.
    guard = InitOnce("TrainRayActor")
    with guard.guarding():
        pass
    return guard


class TestConstructorSignature:
    # [CLASS DOC] Tests the train actor constructor signature.
    def test_positional_constructor_arguments_are_rejected(self):
        # [FUNCTION DOC] Constructing the actor with positional arguments (args object, world size,
        # rank, address, role and so on) raises a type error: only keyword arguments are accepted.
        """Workers are built from a spec's kwargs, so silently shifted positional args must not construct one."""
        with pytest.raises(TypeError):
            TrainRayActor(SimpleNamespace(), 2, 1, "10.0.0.1:1234", "actor", 0)


class TestProposeMasterAddrAndPort:
    # [CLASS DOC] Tests the rendezvous address proposal of an actor.
    def test_the_proposal_steps_past_a_port_that_is_already_taken(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] The node ip lookup returns 10.0.0.3. A real socket is bound and listening
        # on a free port found from 20500, and the random port choice is replaced to return exactly
        # that busy port. The proposal (called on an actor built without its constructor) returns
        # the node ip and a port strictly greater than the busy one, and that proposed port can be
        # bound afterwards (it is actually free). Side effect: opens real local sockets.
        """A cell rendezvouses on the proposing worker's own node, on a port no other process already holds."""
        monkeypatch.setattr(train_actor, "get_current_node_ip", lambda: "10.0.0.3")

        with socket.socket() as occupied:
            occupied.bind(("", train_actor.get_free_port(start_port=20500)))
            occupied.listen(1)
            taken_port = occupied.getsockname()[1]
            monkeypatch.setattr(train_actor.random, "randint", lambda _low, _high: taken_port)

            addr, port = TrainRayActor.__new__(TrainRayActor).propose_master_addr_and_port()

        assert addr == "10.0.0.3"
        assert port > taken_port
        with socket.socket() as probe:
            probe.bind(("", port))


class TestKillSelf:
    # [CLASS DOC] Tests the self kill operation.
    def test_kill_self_exits_with_a_failure_status(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] The process exit call is replaced by a recorder; calling self kill records
        # exactly one exit with status 1 (non-zero, immediate exit).
        """A worker asked to die must leave no survivor and must not look like a clean shutdown."""
        exit_statuses: list[int] = []
        monkeypatch.setattr(train_actor.os, "_exit", exit_statuses.append)

        TrainRayActor.__new__(TrainRayActor).kill_self()

        assert exit_statuses == [1]


class TestLoadState:
    # [CLASS DOC] Tests the default in-place state loading of the train actor.
    def test_an_actor_without_reload_support_refuses_in_place_state_loading(self):
        # [FUNCTION DOC] Calling state loading on the minimal subclass raises a not-implemented
        # error whose message names the subclass and says it cannot reload its state without
        # restarting.
        """An unsupported backend must require a restart instead of pretending to reload in place."""
        actor = _ActorWithoutReloadSupport.__new__(_ActorWithoutReloadSupport)

        with pytest.raises(
            NotImplementedError,
            match="_ActorWithoutReloadSupport cannot reload its state without restarting",
        ):
            actor.load_state()


class TestConfigureMasterAddrAndPort:
    # [CLASS DOC] Tests how the rendezvous address and port are written to the environment.
    def _make_actor(self) -> TrainRayActor:
        # [FUNCTION DOC] HELPER: Creates an actor instance without running its constructor.
        return TrainRayActor.__new__(TrainRayActor)

    def test_the_master_addr_and_port_land_in_the_environment(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With both variables removed, configuring address 10.0.0.1 and port 20001
        # sets MASTER_ADDR to that address and MASTER_PORT to the port as text.
        """The driver-assigned addr/port must reach the env vars that torch's env:// init reads."""
        monkeypatch.delenv("MASTER_ADDR", raising=False)
        monkeypatch.delenv("MASTER_PORT", raising=False)

        self._make_actor().configure_master_addr_and_port(master_addr="10.0.0.1", master_port=20001)

        assert os.environ["MASTER_ADDR"] == "10.0.0.1"
        assert os.environ["MASTER_PORT"] == "20001"

    def test_a_stale_master_addr_and_port_are_overwritten(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With preexisting values 127.0.0.1 and 1, configuring 10.0.0.2 and 20002
        # overwrites both variables.
        """A worker inheriting another run's env must end up on the addr/port the driver assigned."""
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "1")

        self._make_actor().configure_master_addr_and_port(master_addr="10.0.0.2", master_port=20002)

        assert os.environ["MASTER_ADDR"] == "10.0.0.2"
        assert os.environ["MASTER_PORT"] == "20002"


class TestTrainParallelConfigWiring:
    # [CLASS DOC] Tests the driver side training model creation passing the actor's train
    # parallel config on to the rollout executor.
    async def test_the_driver_passes_the_resolved_actor_config_to_the_rollout_executor(self, monkeypatch):
        # [FUNCTION DOC] The placement group module's trainer configs, handle creation, take-over
        # (reports not resumed), argument computation and model creation functions are replaced by
        # fakes for a single actor trainer whose handle returns a nested configuration mapping.
        # After creating the training models, the fake executor must have received exactly that
        # mapping object (not the handle). The executor's load is a no-op.
        """The driver resolves the actor config before handing it to the rollout executor."""
        train_parallel_config = {"dp_size": 4, "topology": {"tp_size": 2}}
        trainer_config = SimpleNamespace(role="actor", trainer_id="actor")

        class FakeActorHandle:
            # [CLASS DOC] HELPER: Fake trainer handle that returns the prepared config mapping.
            async def get_train_parallel_config(self):
                # [FUNCTION DOC] HELPER: Returns the prepared train parallel config mapping.
                return train_parallel_config

        class FakeRolloutExecutor:
            # [CLASS DOC] HELPER: Fake rollout executor that remembers the config it receives.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Starts with no received config.
                self.received_config = None

            async def set_train_parallel_config(self, config):
                # [FUNCTION DOC] HELPER: Asserts the config is not a trainer handle, then stores it.
                assert not isinstance(config, FakeActorHandle)
                self.received_config = config

            async def load(self, rollout_id):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

        actor_handle = FakeActorHandle()
        rollout_executor = FakeRolloutExecutor()
        monkeypatch.setattr(placement_group, "compute_trainer_configs", lambda args: [trainer_config])
        monkeypatch.setattr(
            placement_group,
            "create_trainer_handles",
            lambda args, *, trainer_configs: {trainer_config.trainer_id: actor_handle},
        )
        monkeypatch.setattr(placement_group, "take_over_trainers", lambda args, *, handles: _return(False))
        monkeypatch.setattr(placement_group, "compute_trainer_args", lambda args, config: args)
        monkeypatch.setattr(
            placement_group,
            "create_training_model",
            lambda args, *, handle, trainer_id, resumed: _return(
                placement_group.TrainerInfo(handle=actor_handle, restored_rollout_id=0, start_rollout_id=0)
            ),
        )

        await placement_group.create_training_models(
            SimpleNamespace(use_critic=False, start_rollout_id=None), rollout_executor=rollout_executor
        )

        assert rollout_executor.received_config is train_parallel_config


async def _return(value):
    # [FUNCTION DOC] HELPER: Coroutine that returns its argument; used to give plain lambdas an
    # awaitable result.
    return value


def _actor_with(guard: InitOnce) -> TrainRayActor:
    # [FUNCTION DOC] HELPER: Creates an actor instance without running its constructor and
    # attaches the given run-once guard to it.
    actor = TrainRayActor.__new__(TrainRayActor)
    actor._init_once = guard
    return actor


class TestInitRunsExactlyOnce:
    # [CLASS DOC] Tests the run-once guard and the common initialization sequence.
    def test_init_rebinds_reporting_after_distributed_rank_is_final(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] On a minimal actor with a fresh guard and a heartbeat stub, with cuda
        # device selection, process group init, gloo group init and the hip version patched and
        # the distributed rank and world size faked as 3 and 8, the common init writes rank 3 and
        # world size 8 onto the args and calls the environment reporting rebind exactly once with
        # those args.
        """The reporter must receive the rank and world size discovered by process-group initialization."""
        args = SimpleNamespace(
            debug_deterministic_collective=False,
            distributed_backend="nccl",
            distributed_timeout_minutes=1,
            fsdp_cpu_offload=False,
        )
        actor = _ActorWithoutReloadSupport.__new__(_ActorWithoutReloadSupport)
        actor._init_once = InitOnce("TrainRayActor")
        actor._heartbeat = SimpleNamespace(bump=lambda: None)
        rebound: list[object] = []
        monkeypatch.setattr(train_actor.torch.cuda, "set_device", lambda _device: None)
        monkeypatch.setattr(train_actor.dist, "init_process_group", lambda **_kwargs: None)
        monkeypatch.setattr(train_actor.dist, "get_rank", lambda: 3)
        monkeypatch.setattr(train_actor.dist, "get_world_size", lambda: 8)
        monkeypatch.setattr(train_actor, "init_gloo_group", lambda: None)
        monkeypatch.setattr(train_actor, "rebind_env_reporting", rebound.append)
        monkeypatch.setattr(train_actor.torch.version, "hip", "test")

        actor._init_common(args=args, role="actor")

        assert (args.rank, args.world_size) == (3, 8)
        assert rebound == [args]

    def test_a_second_init_is_refused(self):
        # [FUNCTION DOC] An actor whose guard is already entered raises an assertion error
        # mentioning a stale worker when the common init is called again.
        """A worker that already initialized is a stale process; reusing it must fail loudly, not train on."""
        actor = _actor_with(_inited_guard())

        with pytest.raises(AssertionError, match="stale worker"):
            actor._init_common(None, "actor")

    def test_a_worker_that_never_ran_init_reports_itself_uninitialized(self):
        # [FUNCTION DOC] With a fresh guard, the initialized query returns false.
        """A restarted script asks a worker it found running whether to initialize it or to resume it."""
        assert _actor_with(InitOnce("TrainRayActor")).is_initialized() is False

    def test_a_worker_that_ran_init_reports_itself_initialized(self):
        # [FUNCTION DOC] With an entered guard, the initialized query returns true.
        """The take-over path has to see the worker the previous script built as built."""
        assert _actor_with(_inited_guard()).is_initialized() is True


class TestTheLocalGpuIsFoundWithoutRay:
    # [CLASS DOC] Tests local gpu id discovery: supervised ranks use environment indices, ray
    # placed actors use the ray assignment translated through the visible device mask.
    def test_a_supervised_rank_reads_its_index_rather_than_asking_ray(self, monkeypatch):
        # [FUNCTION DOC] With the cell index variable "0", the subprocess index variable "3", no
        # visible device mask and ray reporting no gpus, the local gpu id is 3.
        """A served worker is a plain process, not an actor, so ray owns no assignment and answers []."""
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        monkeypatch.setenv(SUBPROCESS_INDEX_ENV_VAR, "3")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [])

        assert train_actor.get_local_gpu_id() == 3

    def test_a_pod_running_one_worker_needs_no_supervisor_to_know_its_card(self, monkeypatch):
        # [FUNCTION DOC] With the cell index variable set but the subprocess index variable absent,
        # no mask and ray reporting no gpus, the local gpu id is 0.
        """A cell one worker wide is launched without the supervisor, so the index it would set is absent."""
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        monkeypatch.delenv(SUBPROCESS_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [])

        assert train_actor.get_local_gpu_id() == 0

    def test_a_ray_placed_actor_still_reads_its_assignment_from_ray(self, monkeypatch):
        # [FUNCTION DOC] With neither supervisor variable, a visible device mask 4,5,6,7 and ray
        # assigning physical gpu 6, the local gpu id is 2 (the position of 6 in the mask).
        """Every existing run takes this path, where ray does own the assignment."""
        monkeypatch.delenv(CELL_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SUBPROCESS_INDEX_ENV_VAR, raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [6])

        assert train_actor.get_local_gpu_id() == 2

    def test_a_ray_actor_without_a_visible_device_list_still_reads_its_assignment(self, monkeypatch):
        # [FUNCTION DOC] With neither supervisor variable, no mask and ray assigning gpu 5, the
        # local gpu id is 5 itself.
        """The other ray shape: no mask, so the assignment is the id itself."""
        monkeypatch.delenv(CELL_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SUBPROCESS_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(train_actor.ray, "get_gpu_ids", lambda: [5])

        assert train_actor.get_local_gpu_id() == 5


# PCI bus numbers of the eight GPUs on an HGX H200 board, in NVML (physical) index order.
_PHYSICAL_GPU_PCI_BUSES = [0x18, 0x2A, 0x3A, 0x5D, 0x9A, 0xAB, 0xBA, 0xDB]


class _FakeCuda:
    # [CLASS DOC] HELPER: Fake torch cuda module state for a device mask. Local device i maps to the
    # i-th entry of the visible physical gpu list; the PCI bus of a device comes from the shared
    # table of physical buses (domain 0, device 0).
    """torch.cuda under a device mask: local device i is the i-th physical GPU of the mask."""

    def __init__(self, *, visible_physical_gpus: list[int]) -> None:
        # [FUNCTION DOC] HELPER: Stores the visible physical gpu indices; no device is selected yet.
        self._visible_physical_gpus = visible_physical_gpus
        self._current_device: int | None = None

    def set_device(self, device: str) -> None:
        # [FUNCTION DOC] HELPER: Parses a device string of the form cuda:N (strips the prefix) and
        # stores N as the current device.
        self._current_device = int(device.removeprefix("cuda:"))

    def current_device(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the stored current device (none if never set).
        return self._current_device

    def get_device_properties(self, device: int) -> SimpleNamespace:
        # [FUNCTION DOC] HELPER: Returns properties for the local device: PCI domain 0, the bus of
        # the corresponding physical gpu from the table, and PCI device 0.
        bus = _PHYSICAL_GPU_PCI_BUSES[self._visible_physical_gpus[device]]
        return SimpleNamespace(pci_domain_id=0, pci_bus_id=bus, pci_device_id=0)


class _FakeNvml:
    # [CLASS DOC] HELPER: Fake NVML module exposing only the calls used by NUMA affinity setting.
    # Handles are plain physical indices; bus ids are the canonical uppercase form with an
    # 8-digit zero domain. Records the handles for which cpu affinity was set.
    """NVML indexes every GPU on the host physically, whatever CUDA_VISIBLE_DEVICES says.

    Bus ids are matched in the canonical form NVML itself reports (nvmlPciInfo_t.busId).
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the physical bus id strings from the shared bus table and
        # starts with no recorded affinity settings.
        self._bus_ids = [f"00000000:{bus:02X}:00.0" for bus in _PHYSICAL_GPU_PCI_BUSES]
        self.affinity_set_for: list[int] = []

    def nvmlInit(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        pass

    def nvmlShutdown(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        pass

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        # [FUNCTION DOC] HELPER: Returns the physical index itself as the handle.
        return index

    def nvmlDeviceGetHandleByPciBusId(self, bus_id: str) -> int:
        # [FUNCTION DOC] HELPER: Returns the physical index whose canonical bus id equals the given
        # string; raises a value error if the string is not in the table.
        return self._bus_ids.index(bus_id)

    def nvmlDeviceGetIndex(self, handle: int) -> int:
        # [FUNCTION DOC] HELPER: Returns the handle, which already is the physical index.
        return handle

    def nvmlDeviceSetCpuAffinity(self, handle: int) -> None:
        # [FUNCTION DOC] HELPER: Records the handle for which affinity was set.
        self.affinity_set_for.append(handle)


class TestNumaAffinity:
    # [CLASS DOC] Tests the cpu affinity chosen during common init.
    def test_a_masked_rank_takes_the_cpus_of_the_gpu_it_runs_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With the device mask 4,5,6,7, local and global rank 0, a fake cuda module
        # (local device 0 is physical gpu 4), a fake NVML module installed in the module table, the
        # hip version cleared, and distributed rank 0 of world size 4 faked, the common init sets
        # cpu affinity exactly once, for physical gpu 4 (not gpu 0). Process group, gloo group and
        # reporting rebind calls are stubbed.
        """Under mask 4,5,6,7 local rank 0 runs on GPU 4, so it must take GPU 4's NUMA node, not GPU 0's."""
        args = SimpleNamespace(
            debug_deterministic_collective=False,
            distributed_backend="nccl",
            distributed_timeout_minutes=1,
            fsdp_cpu_offload=False,
            num_gpus_per_node=4,
        )
        actor = _ActorWithoutReloadSupport.__new__(_ActorWithoutReloadSupport)
        actor._init_once = InitOnce("TrainRayActor")
        actor._heartbeat = SimpleNamespace(bump=lambda: None)
        cuda = _FakeCuda(visible_physical_gpus=[4, 5, 6, 7])
        nvml = _FakeNvml()
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setitem(sys.modules, "pynvml", nvml)
        monkeypatch.setattr(train_actor.torch.version, "hip", None)
        monkeypatch.setattr(train_actor.torch.cuda, "set_device", cuda.set_device)
        monkeypatch.setattr(train_actor.torch.cuda, "current_device", cuda.current_device)
        monkeypatch.setattr(train_actor.torch.cuda, "get_device_properties", cuda.get_device_properties)
        monkeypatch.setattr(train_actor.dist, "init_process_group", lambda **_kwargs: None)
        monkeypatch.setattr(train_actor.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(train_actor.dist, "get_world_size", lambda: 4)
        monkeypatch.setattr(train_actor, "init_gloo_group", lambda: None)
        monkeypatch.setattr(train_actor, "rebind_env_reporting", lambda _args: None)

        actor._init_common(args=args, role="actor")

        assert nvml.affinity_set_for == [4]
