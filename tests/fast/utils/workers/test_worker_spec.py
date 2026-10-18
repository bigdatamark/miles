# [MODULE DOC] Tests for the worker spec models: PortInfo (modes, consecutive block size,
# frozen/strict validation, effective static port per worker, cell-offset rules), BaseWorkerSpec
# (deploy component must be concrete, stored uncalled env callable, frozen/strict), the launch
# command context (local gpu ids and pool-keyed address map are required), CommandWorkerSpec and
# ServeWorkerSpec (callables stored uncalled; serve specs get an rpc port injected unless one is
# declared, including when ports are given as plain mappings; concurrency groups; ctor kwargs per
# worker), SchedulingSpec (pin to head default, single-worker shortcut, derivation of pods per cell
# and workers per pod from gpu node size with assertion errors for impossible shapes) and the helm
# builder's check that ranks sharing a pod do not climb their rpc port into another declared port.
import pytest
from pydantic import ValidationError
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability

from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import _assert_worker_ports_fit
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_spec import (
    DEFAULT_RPC_PORT,
    RPC_PORT_NAME,
    BaseWorkerSpec,
    CommandWorkerSpec,
    HostAndPort,
    LaunchCommandContext,
    PortInfo,
    SchedulingSpec,
    ServeWorkerSpec,
    WorkerCtorContext,
    WorkerLaunchContext,
)


def _make_launch_context(**overrides) -> WorkerLaunchContext:
    # [FUNCTION DOC] HELPER: Builds a worker launch context (cell 0, worker 0, no gpu ids) with
    # keyword overrides applied.
    kwargs = dict(cell_index=0, worker_in_cell_index=0, gpu_ids=[])
    kwargs.update(overrides)
    return WorkerLaunchContext(**kwargs)


def _make_launch_command_context(**overrides) -> LaunchCommandContext:
    # [FUNCTION DOC] HELPER: Builds a launch-command context (cell 0, worker 0, empty gpu and local
    # gpu ids, one http address 127.0.0.1:8000, empty pool address map) with overrides applied.
    kwargs = dict(
        cell_index=0,
        worker_in_cell_index=0,
        gpu_ids=[],
        local_gpu_ids=[],
        self_addrs={"http": HostAndPort(host="127.0.0.1", port=8000)},
        pool_addrs={},
    )
    kwargs.update(overrides)
    return LaunchCommandContext(**kwargs)


def _make_ctor_context(**overrides) -> WorkerCtorContext:
    # [FUNCTION DOC] HELPER: Builds a constructor context (cell 0, worker 0, no gpu ids, a fake
    # backend capability) with overrides applied.
    kwargs = dict(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=FakeBackendCapability())
    kwargs.update(overrides)
    return WorkerCtorContext(**kwargs)


def _make_port_info(**overrides) -> PortInfo:
    # [FUNCTION DOC] HELPER: Builds a port description named http with static port 8080, per-worker
    # mode and no dynamic allocation, with overrides applied.
    kwargs = dict(name="http", static_port=8080, mode="per_worker", allow_dynamic=False)
    kwargs.update(overrides)
    return PortInfo(**kwargs)


def _make_base_kwargs(**overrides) -> dict:
    # [FUNCTION DOC] HELPER: Returns constructor keyword arguments shared by all spec types: name
    # demo-worker, one default port description, an environment callable returning DEMO=1 and a
    # scheduling spec of 2 cells with 4 workers each at 0.4 gpu per worker; overrides replace
    # entries.
    kwargs = dict(
        name="demo-worker",
        port_infos=[_make_port_info()],
        env_var=lambda _ctx: {"DEMO": "1"},
        scheduling=SchedulingSpec(num_cells=2, num_workers_per_cell=4, num_gpus_per_worker=0.4),
    )
    kwargs.update(overrides)
    return kwargs


class TestPortInfo:
    # [CLASS DOC] Tests of the port description model's validation.
    def test_accepts_both_modes(self):
        # [FUNCTION DOC] The per-worker and master mode values are both kept as given.
        """Both per_worker and master are valid modes."""
        assert _make_port_info(mode="per_worker").mode == "per_worker"
        assert _make_port_info(mode="master").mode == "master"

    def test_rejects_unknown_mode(self):
        # [FUNCTION DOC] The mode "broadcast" raises a validation error.
        """An unknown mode literal is rejected."""
        with pytest.raises(ValidationError):
            _make_port_info(mode="broadcast")

    def test_num_consecutive_defaults_to_one(self):
        # [FUNCTION DOC] The consecutive count is 1 by default and 32 when given.
        """A port reserves a single slot unless a block is requested."""
        assert _make_port_info().num_consecutive == 1
        assert _make_port_info(num_consecutive=32).num_consecutive == 32

    def test_rejects_extra_field(self):
        # [FUNCTION DOC] An unknown field raises a validation error.
        """Unknown fields are forbidden."""
        with pytest.raises(ValidationError):
            _make_port_info(unknown_field=1)

    def test_is_frozen(self):
        # [FUNCTION DOC] Assigning to a field of a built instance raises a validation error.
        """Field assignment after construction is rejected."""
        port_info = _make_port_info()
        with pytest.raises(ValidationError):
            port_info.static_port = 9000


class TestPortInfoEffectiveStaticPort:
    # [CLASS DOC] Tests of the per-worker fixed port computation within a pod.
    def test_offsets_a_per_worker_port_by_the_whole_block_of_the_workers_before_it(self):
        # [FUNCTION DOC] Base 8080, per-worker, block of 4, worker index 2 in the pod gives 8088
        # (index times block size added to the base).
        """Offsetting by the bare index hands a worker an address inside the previous worker's block."""
        port_info = _make_port_info(static_port=8080, mode="per_worker", num_consecutive=4)

        assert port_info.effective_static_port(worker_in_pod_index=2) == 8088

    def test_leaves_a_master_port_where_every_worker_of_the_pod_expects_it(self):
        # [FUNCTION DOC] A master-mode port stays at its base 8080 for worker index 2.
        """A master port names one endpoint the whole pod talks to, so shifting it per worker would split them."""
        port_info = _make_port_info(static_port=8080, mode="master", num_consecutive=4)

        assert port_info.effective_static_port(worker_in_pod_index=2) == 8080


class TestPortInfoCellOffset:
    # [CLASS DOC] Tests of the offset-by-cell flag's validation.
    def test_a_dynamically_allocated_port_cannot_be_offset_by_cell(self):
        # [FUNCTION DOC] Combining offset-by-cell with dynamic allocation raises a validation error
        # saying it "cannot be offset by cell index".
        """A cell offset applied to a port whose number is chosen at runtime would point at an unrelated socket."""
        with pytest.raises(ValidationError, match="cannot be offset by cell index"):
            _make_port_info(offset_by_cell=True, allow_dynamic=True)

    def test_a_pinned_port_may_be_offset_by_cell(self):
        # [FUNCTION DOC] A fixed port 5100 with offset-by-cell and no dynamic allocation is accepted
        # and keeps both values.
        """Pinned ports are the only ones a cell offset can meaningfully shift, so that combination stays legal."""
        port_info = _make_port_info(static_port=5100, offset_by_cell=True, allow_dynamic=False)

        assert (port_info.static_port, port_info.offset_by_cell) == (5100, True)


class TestBaseWorkerSpec:
    # [CLASS DOC] Tests of the base spec model.
    def test_the_all_selector_cannot_be_stored_as_a_pool_component(self) -> None:
        # [FUNCTION DOC] The trainer deploy component is stored; the all-components selector raises
        # a validation error saying it must name the one component, not the selector.
        """A worker pool must name one concrete deployment component rather than the all-components selector."""
        concrete = BaseWorkerSpec(**_make_base_kwargs(deploy_component=DeployComponent.TRAINER))

        assert concrete.deploy_component is DeployComponent.TRAINER
        with pytest.raises(ValidationError, match="must name the one component.*not the selector"):
            BaseWorkerSpec(**_make_base_kwargs(deploy_component=DeployComponent.ALL))

    def test_constructs_and_exposes_fields(self):
        # [FUNCTION DOC] The built spec exposes its name, first port's fixed port 8080 and 2 cells.
        """A spec keeps its name, ports, and scheduling as provided."""
        spec = BaseWorkerSpec(**_make_base_kwargs())
        assert spec.name == "demo-worker"
        assert spec.port_infos[0].static_port == 8080
        assert spec.scheduling.num_cells == 2

    def test_env_var_is_stored_uncalled(self):
        # [FUNCTION DOC] Constructing a spec does not call the environment callable (no recorded
        # calls); calling it later with a launch context returns its mapping.
        """The env_var callable is stored as-is and only evaluated on demand."""
        calls = []

        def env_var(_ctx) -> dict[str, str]:
            # [FUNCTION DOC] HELPER: Records a call and returns A set to b.
            calls.append(1)
            return {"A": "b"}

        spec = BaseWorkerSpec(**_make_base_kwargs(env_var=env_var))
        assert calls == []
        assert spec.env_var(_make_launch_context()) == {"A": "b"}

    def test_rejects_extra_field(self):
        # [FUNCTION DOC] An unknown field raises a validation error.
        """Unknown fields are forbidden."""
        with pytest.raises(ValidationError):
            BaseWorkerSpec(**_make_base_kwargs(unknown_field=1))

    def test_is_frozen(self):
        # [FUNCTION DOC] Assigning to the name of a built spec raises a validation error.
        """Field assignment after construction is rejected."""
        spec = BaseWorkerSpec(**_make_base_kwargs())
        with pytest.raises(ValidationError):
            spec.name = "other"


class TestLaunchCommandContext:
    # [CLASS DOC] Tests of required fields on the launch-command context.
    def test_the_context_refuses_to_be_built_without_local_gpu_ids(self):
        # [FUNCTION DOC] Omitting the local gpu ids raises a validation error.
        """A default here would let a manager that never probed the worker launch it against the wrong devices."""
        kwargs = dict(
            cell_index=0,
            worker_in_cell_index=0,
            gpu_ids=[],
            self_addrs={"http": HostAndPort(host="127.0.0.1", port=8000)},
            pool_addrs={},
        )

        with pytest.raises(ValidationError):
            LaunchCommandContext(**kwargs)


class TestCommandWorkerSpec:
    # [CLASS DOC] Tests of the command spec.
    def test_constructs_with_launch_command(self):
        # [FUNCTION DOC] A command spec is a base spec and its launch command, called with a
        # context, returns the configured command text.
        """A command spec carries the launch command callable besides base fields."""
        spec = CommandWorkerSpec(**_make_base_kwargs(), launch_command=lambda ctx: "python -m sglang.launch_server")
        ctx = _make_launch_command_context()
        assert spec.launch_command(ctx) == "python -m sglang.launch_server"
        assert isinstance(spec, BaseWorkerSpec)

    def test_launch_command_is_stored_uncalled(self):
        # [FUNCTION DOC] Construction makes no call; later calling with a context holding host
        # 10.0.0.1 port 9001 yields "serve --host 10.0.0.1 --port 9001" and records that context.
        """The launch_command callable is only evaluated once a context is available."""
        calls: list[LaunchCommandContext] = []

        def launch_command(ctx: LaunchCommandContext) -> str:
            # [FUNCTION DOC] HELPER: Records the context and renders a serve command from the http
            # address host and port.
            calls.append(ctx)
            http = ctx.self_addrs["http"]
            return f"serve --host {http.host} --port {http.port}"

        spec = CommandWorkerSpec(**_make_base_kwargs(), launch_command=launch_command)
        assert calls == []

        ctx = _make_launch_command_context(self_addrs={"http": HostAndPort(host="10.0.0.1", port=9001)})
        assert spec.launch_command(ctx) == "serve --host 10.0.0.1 --port 9001"
        assert calls == [ctx]


class TestServeWorkerSpec:
    # [CLASS DOC] Tests of the serve spec.
    def test_constructs_with_worker_class(self):
        # [FUNCTION DOC] A serve spec keeps its worker class path and is a base spec.
        """A serve spec carries the worker class path besides base fields."""
        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.ray.rollout.inference_controller.InferenceController",
            ctor_kwargs=lambda _ctx: {},
        )
        assert spec.worker_class == "miles.ray.rollout.inference_controller.InferenceController"
        assert isinstance(spec, BaseWorkerSpec)

    def test_ctor_kwargs_is_stored_uncalled(self):
        # [FUNCTION DOC] Construction makes no call; calling it later with a constructor context
        # returns its mapping.
        """The ctor_kwargs callable is stored as-is and only evaluated on demand."""
        calls = []

        def ctor_kwargs(_ctx) -> dict:
            # [FUNCTION DOC] HELPER: Records a call and returns x set to 1.
            calls.append(1)
            return {"x": 1}

        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.demo.Worker",
            ctor_kwargs=ctor_kwargs,
        )
        assert calls == []
        assert spec.ctor_kwargs(_make_ctor_context()) == {"x": 1}


class TestServeWorkerSpecRpcPortInjection:
    # [CLASS DOC] Tests of the rpc port that serve specs add automatically.
    def _make_spec(self, **overrides) -> ServeWorkerSpec:
        # [FUNCTION DOC] HELPER: Builds a serve spec from the shared base arguments with overrides,
        # a demo worker class path and an empty constructor-argument callable.
        return ServeWorkerSpec(
            **_make_base_kwargs(**overrides),
            worker_class="miles.demo.Worker",
            ctor_kwargs=lambda _ctx: {},
        )

    def test_rpc_port_is_injected_by_default(self):
        # [FUNCTION DOC] Exactly one port named like the rpc port exists, with the default rpc
        # fixed port, per-worker mode and dynamic allocation allowed.
        """Every serve worker automatically exposes an rpc port."""
        spec = self._make_spec()
        (rpc,) = [port_info for port_info in spec.port_infos if port_info.name == RPC_PORT_NAME]
        assert rpc.static_port == DEFAULT_RPC_PORT
        assert rpc.mode == "per_worker"
        assert rpc.allow_dynamic is True

    def test_injection_keeps_declared_ports(self):
        # [FUNCTION DOC] Port names are the declared http port followed by the rpc port.
        """The injected rpc port is appended after the declared ports."""
        spec = self._make_spec()
        assert [port_info.name for port_info in spec.port_infos] == ["http", RPC_PORT_NAME]

    def test_explicit_rpc_port_is_not_duplicated(self):
        # [FUNCTION DOC] Declaring an rpc port (fixed 9999, not dynamic) leaves the port list equal
        # to just that port.
        """An explicitly declared rpc port wins over the injected default."""
        explicit = PortInfo(name=RPC_PORT_NAME, static_port=9999, mode="per_worker", allow_dynamic=False)
        spec = self._make_spec(port_infos=[explicit])
        assert spec.port_infos == [explicit]

    def test_an_explicit_rpc_port_given_as_a_dict_is_not_duplicated(self):
        # [FUNCTION DOC] An rpc port declared as a plain mapping is converted and not duplicated;
        # the port list equals one port description with that name and port 9999.
        """Callers may declare ports as raw dicts, and such a declaration must still suppress the injected default."""
        spec = self._make_spec(port_infos=[dict(name=RPC_PORT_NAME, static_port=9999)])

        assert spec.port_infos == [PortInfo(name=RPC_PORT_NAME, static_port=9999)]

    def test_ports_given_as_dicts_still_receive_the_injected_rpc_port(self):
        # [FUNCTION DOC] An http port given as a mapping still gets the rpc port appended.
        """Reading a dict port's name must not be confused with reading the rpc name, or the rpc port goes missing."""
        spec = self._make_spec(port_infos=[dict(name="http", static_port=8080)])

        assert [port_info.name for port_info in spec.port_infos] == ["http", RPC_PORT_NAME]

    def test_base_and_command_specs_get_no_rpc_port(self):
        # [FUNCTION DOC] Neither a base spec nor a command spec lists an rpc-named port.
        """Only serve workers run the rpc server, so only they get the port."""
        base = BaseWorkerSpec(**_make_base_kwargs())
        command = CommandWorkerSpec(**_make_base_kwargs(), launch_command=lambda ctx: "sleep 1")
        assert RPC_PORT_NAME not in [port_info.name for port_info in base.port_infos]
        assert RPC_PORT_NAME not in [port_info.name for port_info in command.port_infos]


class TestSchedulingSpecPinToHead:
    # [CLASS DOC] Tests of the pin-to-head flag.
    def test_workers_are_not_pinned_to_the_head_node_by_default(self):
        # [FUNCTION DOC] Both the direct constructor and the single-worker shortcut default to not
        # pinned.
        """Pinning is opt-in, otherwise every worker of every spec would crowd onto the head node."""
        assert SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0).pin_to_head is False
        assert SchedulingSpec.single(num_gpus_per_worker=0).pin_to_head is False

    def test_the_single_worker_shortcut_forwards_the_pin_flag(self):
        # [FUNCTION DOC] The shortcut with 0.5 gpu and pin set gives 1 cell, 1 worker per cell, 0.5
        # gpu per worker and pinned.
        """The convenience constructor must not silently drop the pin request."""
        scheduling = SchedulingSpec.single(num_gpus_per_worker=0.5, pin_to_head=True)

        assert (scheduling.num_cells, scheduling.num_workers_per_cell) == (1, 1)
        assert scheduling.num_gpus_per_worker == 0.5
        assert scheduling.pin_to_head is True


class TestSchedulingSpecPodPacking:
    # [CLASS DOC] Tests of deriving pods per cell and workers per pod from the node gpu count.
    def test_a_cell_a_node_can_hold_rides_in_one_pod(self):
        # [FUNCTION DOC] 8 workers on 8-gpu nodes: 1 pod per cell with 8 workers per pod.
        """A cell no bigger than a node must not be spread, however many workers it holds."""
        scheduling = _gpu_scheduling(num_workers_per_cell=8, num_gpus_per_node=8)

        assert (scheduling.pods_per_cell(), scheduling.workers_per_pod()) == (1, 8)

    def test_a_cell_spanning_several_nodes_is_tiled_by_them(self):
        # [FUNCTION DOC] 16 workers on 8-gpu nodes: 2 pods per cell with 8 workers per pod.
        """This is the whole point of the derivation: 16 gpus on 8-gpu nodes are two equal pods."""
        scheduling = _gpu_scheduling(num_workers_per_cell=16, num_gpus_per_node=8)

        assert (scheduling.pods_per_cell(), scheduling.workers_per_pod()) == (2, 8)

    def test_a_cell_that_claims_no_gpu_rides_in_one_pod(self):
        # [FUNCTION DOC] 4 gpu-less workers: 1 pod per cell with 4 workers per pod.
        """A cpu spec has no node shape to tile, so its whole cell travels together."""
        scheduling = SchedulingSpec(num_cells=1, num_workers_per_cell=4, num_gpus_per_worker=0)

        assert (scheduling.pods_per_cell(), scheduling.workers_per_pod()) == (1, 4)

    def test_rejects_a_gpu_cell_that_never_says_how_big_a_node_is(self):
        # [FUNCTION DOC] A gpu cell with node gpu count 0 makes the pods-per-cell derivation raise
        # an assertion error matching "divide 8 by zero".
        """Forgetting the node shape used to pack one rank per pod in silence."""
        scheduling = _gpu_scheduling(num_workers_per_cell=8, num_gpus_per_node=0)

        with pytest.raises(AssertionError, match="divide 8 by zero"):
            scheduling.pods_per_cell()

    def test_rejects_a_cell_that_is_not_a_whole_number_of_nodes(self):
        # [FUNCTION DOC] 12 workers on 8-gpu nodes make the derivation raise an assertion error
        # matching "12 is not a whole number of 8".
        """A trailing partial node would leave the last pod fewer gpus than its ranks need."""
        scheduling = _gpu_scheduling(num_workers_per_cell=12, num_gpus_per_node=8)

        with pytest.raises(AssertionError, match="12 is not a whole number of 8"):
            scheduling.pods_per_cell()

    def test_rejects_a_cell_whose_workers_cannot_tile_its_pods(self):
        # [FUNCTION DOC] 2 workers of 12 slots each on 8-gpu nodes: the workers-per-pod derivation
        # raises an assertion error matching "2 is not a whole number of 3".
        """A trailing partial pod would shift every later worker's name and rpc port."""
        scheduling = SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=2,
            num_gpus_per_worker=1,
            num_gpu_slots_per_worker=12,
            num_gpus_per_node=8,
        )

        with pytest.raises(AssertionError, match="2 is not a whole number of 3"):
            scheduling.workers_per_pod()


def _gpu_scheduling(*, num_workers_per_cell: int, num_gpus_per_node: int) -> SchedulingSpec:
    # [FUNCTION DOC] HELPER: Builds a one-cell scheduling spec where each worker takes 1 gpu and 1
    # slot, with the given workers per cell and gpus per node.
    return SchedulingSpec(
        num_cells=1,
        num_workers_per_cell=num_workers_per_cell,
        num_gpus_per_worker=1,
        num_gpu_slots_per_worker=1,
        num_gpus_per_node=num_gpus_per_node,
    )


class TestAssertRankPortsFit:
    # [CLASS DOC] Tests of the helm builder's worker-port fit assertion, using 8-worker serve specs
    # with rpc and other fixed ports.
    def test_ranks_sharing_a_pod_may_climb_up_to_the_next_port_block(self):
        # [FUNCTION DOC] 4 ranks per pod from rpc port 8000 occupy 8000-8003 and the other port is
        # 8004; the check passes (no error).
        """The ports a pod hands its ranks are free, so the spec must be accepted."""
        spec = _serve_spec(
            num_gpus_per_node=4,
            port_infos=[PortInfo(name=RPC_PORT_NAME, static_port=8000), PortInfo(name="master", static_port=8004)],
        )

        _assert_worker_ports_fit(spec)

    def test_rejects_rank_ports_reaching_into_another_port(self):
        # [FUNCTION DOC] 4 ranks per pod with another port at 8002 inside the rank range raises an
        # assertion error matching "reaches into".
        """Rank 2 would bind the master port and every collective would rendezvous on nothing."""
        spec = _serve_spec(
            num_gpus_per_node=4,
            port_infos=[PortInfo(name=RPC_PORT_NAME, static_port=8000), PortInfo(name="master", static_port=8002)],
        )

        with pytest.raises(AssertionError, match="reaches into"):
            _assert_worker_ports_fit(spec)

    def test_rejects_rank_ports_reaching_into_a_consecutive_port_block(self):
        # [FUNCTION DOC] 8 ranks per pod and a 30-port block starting at 8003 overlapping the rank
        # range raises an assertion error matching "reaches into".
        """A block claims num_consecutive ports, so the collision test must span all of them."""
        spec = _serve_spec(
            num_gpus_per_node=8,
            port_infos=[
                PortInfo(name=RPC_PORT_NAME, static_port=8000),
                PortInfo(name="dist_init", static_port=8003, num_consecutive=30),
            ],
        )

        with pytest.raises(AssertionError, match="reaches into"):
            _assert_worker_ports_fit(spec)

    def test_a_port_below_the_rpc_port_is_untouched(self):
        # [FUNCTION DOC] Another port at 7000 below the rpc port 8000 passes the check.
        """Ranks climb upwards only, so a lower port can never be reached."""
        spec = _serve_spec(
            num_gpus_per_node=8,
            port_infos=[PortInfo(name=RPC_PORT_NAME, static_port=8000), PortInfo(name="master", static_port=7000)],
        )

        _assert_worker_ports_fit(spec)

    def test_a_pod_of_one_rank_needs_only_its_own_rpc_port(self):
        # [FUNCTION DOC] One gpu per node (one rank per pod) with another port at 8001 passes.
        """Nodes as wide as a cell put one rank in each pod, which must not be constrained by neighbours."""
        spec = _serve_spec(
            num_gpus_per_node=1,
            port_infos=[PortInfo(name=RPC_PORT_NAME, static_port=8000), PortInfo(name="master", static_port=8001)],
        )

        _assert_worker_ports_fit(spec)


def _serve_spec(*, num_gpus_per_node: int, **overrides) -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a serve spec with one cell of 8 workers (1 gpu, 1 slot each) on
    # nodes of the given gpu count; other base arguments (such as ports) come from overrides.
    scheduling = SchedulingSpec(
        num_cells=1,
        num_workers_per_cell=8,
        num_gpus_per_worker=1,
        num_gpu_slots_per_worker=1,
        num_gpus_per_node=num_gpus_per_node,
    )
    return ServeWorkerSpec(
        **_make_base_kwargs(scheduling=scheduling, **overrides),
        worker_class="miles.demo.Worker",
        ctor_kwargs=lambda _ctx: {},
    )


class TestServeWorkerSpecExtraScheduling:
    # [CLASS DOC] Tests of optional serve spec scheduling inputs.
    def test_concurrency_groups_default_to_absent(self):
        # [FUNCTION DOC] Without groups the field is none.
        """Most workers need no concurrency groups, so the field stays optional."""
        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.demo.Worker",
            ctor_kwargs=lambda _ctx: {},
        )

        assert spec.concurrency_groups is None

    def test_concurrency_groups_are_carried_on_the_spec(self):
        # [FUNCTION DOC] A mapping of two groups to sizes is kept unchanged.
        """The trainer needs its heartbeat rpc served outside the default group."""
        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.demo.Worker",
            ctor_kwargs=lambda _ctx: {},
            concurrency_groups={"heartbeat_status": 1, "default": 1},
        )

        assert spec.concurrency_groups == {"heartbeat_status": 1, "default": 1}

    def test_ctor_kwargs_receive_the_worker_position(self):
        # [FUNCTION DOC] A callable building rank and gpu ids from the context, called with worker
        # index 3 and gpu ids [2], returns rank 3 and gpu ids [2].
        """Each worker needs its own rank, so the callable is per worker."""
        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.demo.Worker",
            ctor_kwargs=lambda ctx: {"rank": ctx.worker_in_cell_index, "gpu_ids": ctx.gpu_ids},
        )

        kwargs = spec.ctor_kwargs(_make_ctor_context(worker_in_cell_index=3, gpu_ids=[2]))

        assert kwargs == {"rank": 3, "gpu_ids": [2]}


class TestLaunchCommandContextPoolAddrs:
    # [CLASS DOC] Tests of the pool-keyed address map on the launch-command context.
    def test_a_launch_command_reads_a_peer_address_out_of_the_pool_keyed_map(self):
        # [FUNCTION DOC] A command rendering the first worker's primary address of the
        # inference-router-0 pool produces "serve --backend http://10.0.0.1:3000".
        """A command renders a peer's address by looking that peer's pool id up in pool_addrs."""
        spec = CommandWorkerSpec(
            **_make_base_kwargs(),
            launch_command=lambda ctx: f"serve --backend {ctx.pool_addrs['inference-router-0'][0]['primary'].addr}",
        )
        ctx = _make_launch_command_context(
            pool_addrs={"inference-router-0": [{"primary": HostAndPort(host="10.0.0.1", port=3000)}]}
        )

        assert spec.launch_command(ctx) == "serve --backend http://10.0.0.1:3000"

    def test_the_map_of_a_pool_with_several_workers_keeps_every_worker_under_that_one_key(self):
        # [FUNCTION DOC] A map with one pool of two workers has only that key and the list is kept
        # intact and in order.
        """One key per pool, listing all of its workers, is what lets a command address a whole pool."""
        workers = [
            {"primary": HostAndPort(host="10.0.0.1", port=3000)},
            {"primary": HostAndPort(host="10.0.0.2", port=3000)},
        ]
        ctx = _make_launch_command_context(pool_addrs={"session-server": workers})

        assert list(ctx.pool_addrs) == ["session-server"]
        assert ctx.pool_addrs["session-server"] == workers

    def test_the_legacy_spec_keyed_name_is_not_accepted_beside_the_pool_keyed_one(self):
        # [FUNCTION DOC] Passing the retired spec_addrs field raises a validation error.
        """Two accepted names for one map would let the retired spec_addrs vocabulary creep back in unnoticed."""
        with pytest.raises(ValidationError):
            _make_launch_command_context(spec_addrs={})

    def test_a_context_without_the_pool_map_is_rejected(self):
        # [FUNCTION DOC] A context built without the pool address map (and without local gpu ids)
        # raises a validation error.
        # KNOWN ISSUE: the context also lacks the local gpu ids, so the error does not isolate the
        # missing pool map as the test name claims.
        """Making the map optional would let a caller that forgot to wire it render commands against nothing."""
        with pytest.raises(ValidationError):
            LaunchCommandContext(
                cell_index=0,
                worker_in_cell_index=0,
                gpu_ids=[],
                self_addrs={"http": HostAndPort(host="127.0.0.1", port=8000)},
            )
