# [MODULE DOC]
# Integration style tests of the kubernetes driver assembly, run in process without a cluster. A
# fake pod api serves fake pods; the helm backend capability is built from real trainer, rollout
# executor, inference controller and trainer controller specs, whose worker classes are replaced
# by in-memory fakes. A per-host transport routes http calls straight into rpc applications built
# around those fakes. The tests show that provider scoping, observed worker infos and handles,
# trainer cell rpc execution, cell healing by pod deletion, and the rollout and trainer
# controller handles all work over rpc with no ray actors behind them.
from __future__ import annotations

import asyncio
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from tests.fast.utils.workers.worker_provider.kubernetes import fake_pod_api
from tests.fast.utils.workers.worker_provider.kubernetes.core.test_pod_view import make_pod

from miles.ray import placement_group
from miles.ray.placement_group import create_rollout_components
from miles.ray.rollout.eval_fleet import EvalFleetInfo
from miles.ray.specs import inference as specs_inference
from miles.ray.specs import rollout as specs_rollout
from miles.ray.specs import train as specs_train
from miles.ray.specs.train import POOL_CATEGORY_TRAINER_ENGINE
from miles.ray.train.cell import TrainerCell
from miles.utils import http_utils
from miles.utils.data import RolloutDataPack
from miles.utils.ft_utils.api_server.models import CellStatus
from miles.utils.init_once import InitState
from miles.utils.object_store import _MooncakeStoreObjectRef
from miles.utils.workers.cell_operations import kubernetes as cell_operations_kubernetes
from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.reconcile.k8s_api import PodListPage
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.common.wire_types import Pickled
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.worker_provider.kubernetes.core import provider as core_provider
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm import env, naming
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability
from miles.utils.workers.worker_provider.kubernetes.helm.env import NAMESPACE_ENV_VAR, RELEASE_ENV_VAR
from miles.utils.workers.worker_spec import HostAndPort, PortInfo, SchedulingSpec, ServeWorkerSpec

NAMESPACE = "rl"
_RELEASE = "miles-run-260805"

STATIC_HOSTS = {
    pool_id: naming.static_worker_host(_RELEASE, pool_id, 0)
    for pool_id in ("rollout-executor", "inference-controller", "trainer-controller-actor")
}
POOL = "trainer-engine-actor"
CELL_ID = compute_cell_id(pool_id=POOL, cell_index=0)


class FakeTrainWorker:
    # [CLASS DOC] HELPER: Fake trainer rank exposing the two calls used by the tests; records every
    # master address configuration.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty list of configured address and port pairs.
        self.configured: list[tuple[str, int]] = []

    def configure_master_addr_and_port(self, master_addr: str, master_port: int) -> int:
        # [FUNCTION DOC] HELPER: Records the address and port pair and returns the port.
        self.configured.append((master_addr, master_port))
        return master_port

    def kill_self(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing and returns none (the fake never dies).
        return None


def _data_pack(rollout_id: int) -> RolloutDataPack:
    # [FUNCTION DOC] HELPER: Builds a rollout data pack with the rollout id as the only sample
    # index and a mooncake store object reference whose payload is the text ref- plus the id.
    return RolloutDataPack(sample_indices=[rollout_id], data_ref=_MooncakeStoreObjectRef(payload=f"ref-{rollout_id}"))


class FakeRolloutExecutor:
    # [CLASS DOC] HELPER: In-memory stand-in for the rollout executor worker exposing the rpc
    # methods the driver uses. Records initialization, loads, the train parallel config and the
    # eval fleet info it is given; it returns canned data (a data pack per rollout id, 7 rollouts
    # per epoch).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts uninitialized with no recorded loads, config or fleet info.
        self.initialized = False
        self.loaded: list[int | None] = []
        self.train_parallel_config: dict[str, Any] | None = None
        self.eval_fleet_info: EvalFleetInfo | None = None

    async def init(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the executor initialized.
        self.initialized = True

    async def get_init_state(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the string value of the inited state if init ran, else of
        # the not-inited state.
        return InitState.INITED.value if self.initialized else InitState.NOT_INITED.value

    def dispose(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None

    async def get(self, rollout_id: int) -> RolloutDataPack:
        # [FUNCTION DOC] HELPER: Returns the canned data pack for the rollout id.
        return _data_pack(rollout_id)

    async def eval(self, rollout_id: int) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None

    def save(self, rollout_id: int) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None

    def load(self, rollout_id: int | None = None) -> None:
        # [FUNCTION DOC] HELPER: Appends the rollout id (possibly none) to the recorded loads.
        self.loaded.append(rollout_id)

    def get_num_rollout_per_epoch(self) -> int:
        # [FUNCTION DOC] HELPER: Always returns 7.
        return 7

    def set_train_parallel_config(self, config: dict[str, Any]) -> None:
        # [FUNCTION DOC] HELPER: Stores the given train parallel config.
        self.train_parallel_config = config

    async def set_eval_fleet_info(self, eval_fleet_info: EvalFleetInfo | None) -> None:
        # [FUNCTION DOC] HELPER: Stores the given eval fleet info (possibly none).
        self.eval_fleet_info = eval_fleet_info


class FakeInferenceController:
    # [CLASS DOC] HELPER: In-memory stand-in for the inference controller worker. Tracks
    # initialization and prepared rollouts; every other call is a no-op. The eval fleet info it
    # reports is fixed at construction.
    def __init__(self, eval_fleet_info: EvalFleetInfo | None = None) -> None:
        # [FUNCTION DOC] HELPER: Starts uninitialized with no prepared rollouts and remembers the
        # eval fleet info to report.
        self.initialized = False
        self.prepared: list[int] = []
        self._eval_fleet_info = eval_fleet_info

    async def init(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the controller initialized.
        self.initialized = True

    async def is_initialized(self) -> bool:
        # [FUNCTION DOC] HELPER: Returns whether init ran.
        return self.initialized

    async def prepare_rollout(self, rollout_id: int) -> None:
        # [FUNCTION DOC] HELPER: Appends the rollout id to the prepared list.
        self.prepared.append(rollout_id)

    async def prepare_eval(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None

    async def dispose(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None

    async def offload(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None

    async def onload(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None

    async def get_cell_statuses(self) -> dict[str, CellStatus]:
        # [FUNCTION DOC] HELPER: Returns an empty mapping of cell statuses.
        return {}

    async def get_eval_fleet_info(self) -> EvalFleetInfo | None:
        # [FUNCTION DOC] HELPER: Returns the eval fleet info given at construction (possibly none).
        return self._eval_fleet_info


class FakeTrainerController:
    # [CLASS DOC] HELPER: In-memory stand-in for the trainer controller worker. Records the init
    # argument and the train calls; reports a canned restore position and parallel config.
    # KNOWN ISSUE: the initialized attribute starts as a boolean but init overwrites it with the
    # (unpickled) args object, so it is not a flag after init; tests rely on that to read the args.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with initialized false and no recorded train calls.
        self.initialized = False
        self.trained: list[tuple[int, RolloutDataPack]] = []

    async def init(self, args: Pickled) -> list[Any]:
        # [FUNCTION DOC] HELPER: Stores the received args in the initialized attribute and returns
        # the canned restored rollout list [5].
        self.initialized = args
        return [5]

    async def train(
        self, rollout_id: int, rollout_data_pack: RolloutDataPack, external_data: list[Any] | None = None
    ) -> list[Any]:
        # [FUNCTION DOC] HELPER: Records the rollout id and data pack pair and returns an empty
        # list. The external data argument is ignored.
        self.trained.append((rollout_id, rollout_data_pack))
        return []

    async def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None

    async def update_weights(self, rollout_id: int | None = None) -> None:
        # [FUNCTION DOC] HELPER: Does nothing and returns none.
        return None

    async def onload(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None

    async def offload(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None

    async def clear_memory(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None

    async def get_train_parallel_config(self) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Returns a fixed mapping with data parallel size 2.
        return {"dp_size": 2}

    async def get_cell_statuses(self) -> dict[str, CellStatus]:
        # [FUNCTION DOC] HELPER: Returns an empty mapping of cell statuses.
        return {}

    async def dispose(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None


class FakePodApi:
    # [CLASS DOC] HELPER: Fake kubernetes pod api serving a fixed pod list and recording every
    # (namespace, label selector) pair it is asked for. The watch stream never delivers events.
    def __init__(self, pods: list[Pod]) -> None:
        # [FUNCTION DOC] HELPER: Stores the pods to serve and starts with no recorded selectors.
        self.pods = pods
        self.selectors: list[tuple[str, str]] = []

    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage:
        # [FUNCTION DOC] HELPER: Records the namespace and selector pair, then returns a page with
        # a copy of all stored pods (no filtering) and resource version "1".
        self.selectors.append((namespace, label_selector))
        return PodListPage(pods=list(self.pods), resource_version="1")

    async def stream_pods(self, *, namespace, label_selector, resource_version, timeout_seconds):
        # [FUNCTION DOC] HELPER: Fake watch stream; sleeps for an hour and then yields once with no
        # event, so effectively no event is ever delivered. All arguments are ignored.
        await asyncio.sleep(3600)
        yield None


class _PerHostTransport(httpx.AsyncBaseTransport):
    # [CLASS DOC] HELPER: Asynchronous http transport that dispatches each request to an in-process
    # application chosen by the request host name, and records every host that was called.
    def __init__(self, apps_by_host: dict[str, Any]) -> None:
        # [FUNCTION DOC] HELPER: Builds one in-process application transport per host and starts
        # with an empty list of called hosts.
        self._transports = {host: httpx.ASGITransport(app=app) for host, app in apps_by_host.items()}
        self.hosts_called: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC] HELPER: Records the request host. If an application is registered for that
        # host, forwards the request to it and returns its response; otherwise raises a connect
        # error saying nothing listens on the host.
        host = request.url.host
        self.hosts_called.append(host)
        transport = self._transports.get(host)
        if transport is None:
            raise httpx.ConnectError(f"nothing listens on {host}", request=request)
        return await transport.handle_async_request(request)


def trainer_spec(*, num_workers_per_cell: int, num_gpus_per_node: int) -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a hand-written trainer serve spec for the test pool: one cell
    # of the given workers per cell, one gpu and one gpu slot per worker, the given gpus per node,
    # a master port 9000 declared in master mode, an empty environment, and the fake train worker
    # (defined in this module) as the worker class with empty constructor kwargs.
    return ServeWorkerSpec(
        name=POOL,
        category=POOL_CATEGORY_TRAINER_ENGINE,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master")],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=1,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=num_gpus_per_node,
        ),
        worker_class=f"{__name__}.FakeTrainWorker",
        ctor_kwargs=lambda context: {},
    )


def cell_pods(count: int) -> list[Pod]:
    # [FUNCTION DOC] HELPER: Builds the given number of fake pods of cell 0 of the trainer pool,
    # named pool-0-index with pod-in-cell index equal to the position and pod ip 10.0.0.(index+1).
    return [
        make_pod(
            name=f"{POOL}-0-{index}",
            pool_id=POOL,
            cell_id_suffix="0",
            pod_in_cell_index=str(index),
            pod_ip=f"10.0.0.{index + 1}",
        )
        for index in range(count)
    ]


def rollout_executor_args() -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Returns a minimal argument object for building the rollout executor,
    # inference controller and trainer controller specs: kubernetes cluster backend, rollout
    # manager not pinned, train-only debug on (so no engines), no critic, no kl or distillation.
    return SimpleNamespace(
        cluster_backend="kubernetes",
        pin_rollout_manager_to_head=False,
        debug_train_only=True,
        use_critic=False,
        kl_coef=0,
        use_kl_loss=False,
        use_opd=False,
        opd_type="megatron",
        megatron_config=None,
    )


@pytest.fixture
def deleted(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    # [FUNCTION DOC] HELPER: Fixture. Replaces the pod deletion call of the kubernetes cell
    # operations with a recorder and returns the list that collects the pod name lists deleted.
    recorded: list[list[str]] = []

    async def fake_delete_pods(*, namespace: str, pod_names: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Appends a copy of the pod names to the record; deletes nothing.
        recorded.append(list(pod_names))

    monkeypatch.setattr(cell_operations_kubernetes, "_delete_pods", fake_delete_pods)
    return recorded


def install(monkeypatch: pytest.MonkeyPatch, *, pods: list[Pod], workers_per_pod: int = 1):
    # [FUNCTION DOC] HELPER: Assembles the kubernetes driver environment and returns the helm
    # backend capability. Sets the namespace and release environment variables, resets and installs
    # a fake pod api serving the given pods (the provider is pointed at it), replaces the worker
    # class constants of the rollout executor, inference controller and trainer controller specs
    # with the in-memory fakes of this module, and builds the specs: the hand-written trainer spec
    # (workers per cell equals pods times workers per pod), the rollout executor spec, the
    # inference controller spec and the trainer controller specs. Side effects: environment
    # variables and module attributes patched through the monkeypatch fixture.
    monkeypatch.setenv(NAMESPACE_ENV_VAR, NAMESPACE)
    monkeypatch.setenv(RELEASE_ENV_VAR, _RELEASE)
    fake_pod_api.reset()
    fake_pod_api.install(FakePodApi(pods))
    monkeypatch.setattr(core_provider, "_kubernetes_pod_api", fake_pod_api.installed)
    monkeypatch.setattr(specs_rollout, "ROLLOUT_EXECUTOR_WORKER_CLASS", f"{__name__}.FakeRolloutExecutor")
    monkeypatch.setattr(specs_inference, "INFERENCE_CONTROLLER_WORKER_CLASS", f"{__name__}.FakeInferenceController")
    monkeypatch.setattr(specs_train, "TRAINER_CONTROLLER_WORKER_CLASS", f"{__name__}.FakeTrainerController")

    specs = [
        trainer_spec(num_workers_per_cell=len(pods) * workers_per_pod, num_gpus_per_node=workers_per_pod),
        specs_rollout.spec_rollout_executor(rollout_executor_args()),
        specs_inference.spec_inference_controller(rollout_executor_args()),
        *specs_train.specs_trainer_controller(rollout_executor_args()),
    ]
    return compute_helm_backend_capability(specs=specs)


async def _ignore_cell(cell_id, info) -> None:
    # [FUNCTION DOC] HELPER: Cell reconcile callback that does nothing.
    return None


def installed_cells_provider(capability):
    # [FUNCTION DOC] HELPER: Returns the capability's dynamic worker provider over the trainer pool.
    return capability.dynamic_worker_provider(pool_ids=[POOL])


class TestKubernetesDriverAssembly:
    # [CLASS DOC] Tests the kubernetes driver assembly end to end with fakes: capability sharing,
    # watch scoping, worker observation, rpc driven trainer cells, pod deletion for healing, and the
    # rollout executor, rollout component and trainer controller handles over rpc.
    def test_the_installed_capability_answers_every_component_of_the_process(self, monkeypatch):
        # [FUNCTION DOC] Two dynamic providers requested from one capability share the same run
        # object (internal state), and the provider is a kubernetes worker provider.
        """A run announces its backend once, and every later component has to see that answer."""
        capability = install(monkeypatch, pods=cell_pods(2))

        provider = installed_cells_provider(capability)

        assert isinstance(provider, KubernetesWorkerProvider)
        assert provider._run is installed_cells_provider(capability)._run

    def test_the_watch_is_scoped_to_this_release_and_its_pools(self, monkeypatch):
        # [FUNCTION DOC] Starting and stopping a cell watch makes the first pod listing use the
        # configured namespace and a label selector requiring the instance label to equal the
        # release and the pool id label to be in the watched pool list.
        """The selector is the only thing keeping one run from healing the cells of another run's release."""
        provider = installed_cells_provider(install(monkeypatch, pods=cell_pods(2)))
        api = fake_pod_api.current()

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the cell watch and immediately stops it.
            stop = await provider.watch_cells(_ignore_cell)
            await stop()

        asyncio.run(scenario())

        namespace, selector = api.selectors[0]
        assert namespace == NAMESPACE
        assert selector == f"{env.INSTANCE_LABEL}={_RELEASE},{env.DEFAULT_LABEL_KEYS.pool_id} in ({POOL})"

    def test_refuses_to_hand_out_a_provider_for_a_pool_it_does_not_watch(self, monkeypatch):
        # [FUNCTION DOC] Asking the capability for a dynamic provider over an unknown pool id
        # raises an assertion error saying those ids are not pool ids.
        """A provider that silently watches nothing would leave those cells unhealed forever."""
        capability = install(monkeypatch, pods=cell_pods(2))

        with pytest.raises(AssertionError, match="are not pool_ids"):
            capability.dynamic_worker_provider(pool_ids=["some-other-pool_id"])

    def test_observing_a_cell_yields_rank_ordered_workers_with_handles(self, monkeypatch):
        # [FUNCTION DOC] With 3 pods in one cell, the observed worker infos are ordered by worker
        # index, named by the standard worker naming, carry rpc addresses with the pod ips in order,
        # and every info has an rpc worker handle available from the provider.
        """This is what a trainer cell is built from, so the order and the handles are the whole product."""
        provider = installed_cells_provider(install(monkeypatch, pods=cell_pods(3)))

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the cell watch, returns the worker infos of the cell
            # (the first and only entry for the cell id), and always stops the watch afterwards.
            stop = await provider.watch_cells(_ignore_cell)
            try:
                return provider.get_worker_infos(cell_ids=[CELL_ID])[0]
            finally:
                await stop()

        infos = asyncio.run(scenario())

        assert [info.name for info in infos] == [
            compute_worker_name(pool_id=POOL, worker_in_cell_index=index) for index in range(3)
        ]
        assert [info.self_addrs["rpc"].host for info in infos] == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
        handles = provider.get_handles_of_worker_infos(infos)
        assert all(isinstance(handles[info.name], RpcWorkerHandle) for info in infos)

    def test_one_pod_serving_several_ranks_yields_one_worker_per_rank(self, monkeypatch):
        # [FUNCTION DOC] One pod hosting 2 workers yields 2 worker infos named for worker indices 0
        # and 1. Both have the pod ip; their rpc ports are 8000 and 8001; the handles' server urls
        # match; and both report the same master port 9000.
        """A trainer pod supervises one process per gpu, and a cell that saw one of them would hang the collective."""
        provider = installed_cells_provider(install(monkeypatch, pods=cell_pods(1), workers_per_pod=2))

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the cell watch, returns the worker infos of the cell,
            # and always stops the watch afterwards.
            stop = await provider.watch_cells(_ignore_cell)
            try:
                return provider.get_worker_infos(cell_ids=[CELL_ID])[0]
            finally:
                await stop()

        infos = asyncio.run(scenario())

        assert [info.name for info in infos] == [
            compute_worker_name(pool_id=POOL, worker_in_cell_index=index) for index in range(2)
        ]
        assert [(info.self_addrs["rpc"].host, info.self_addrs["rpc"].port) for info in infos] == [
            ("10.0.0.1", 8000),
            ("10.0.0.1", 8001),
        ]
        handles = provider.get_handles_of_worker_infos(infos)
        assert [handles[info.name]._transport._server_url for info in infos] == [
            "http://10.0.0.1:8000",
            "http://10.0.0.1:8001",
        ]
        assert [info.self_addrs["master"].port for info in infos] == [9000, 9000]

    def test_a_trainer_cell_builds_and_drives_its_ranks_over_rpc(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Two pods whose ips each route to an rpc application around a fake train
        # worker. A trainer cell (without ref, fake health checker) is built over the observed
        # workers and told to execute the master address configuration call. Both results are the
        # port 9000, each fake worker recorded exactly that address and port, and both pod hosts
        # were called. Side effect: the shared http client provider is replaced with one using the
        # per-host transport.
        """The point of the whole assembly: training over workers the platform created, not ones miles launched."""
        provider = installed_cells_provider(install(monkeypatch, pods=cell_pods(2)))

        workers = [FakeTrainWorker(), FakeTrainWorker()]
        apps = {f"10.0.0.{index + 1}": create_rpc_app(worker) for index, worker in enumerate(workers)}
        transport = _PerHostTransport(apps)

        async def scenario():
            # [FUNCTION DOC] HELPER: Inside an http client over the per-host transport, makes the
            # shared client provider return that client, enters each application's lifespan, starts
            # the cell watch, builds the trainer cell and returns it with the result of executing
            # the master address call (master address 10.0.0.1, port 9000); stops the watch at the end.
            # The application lifespans are entered but never exited.
            async with httpx.AsyncClient(transport=transport) as client:
                monkeypatch.setattr(http_utils.GeneralHttpClientProvider, "client", classmethod(lambda cls: client))
                for app in apps.values():
                    await app.router.lifespan_context(app).__aenter__()
                stop = await provider.watch_cells(_ignore_cell)
                try:
                    cell = TrainerCell(
                        args=SimpleNamespace(),
                        role="actor",
                        with_ref=False,
                        cell_id=CELL_ID,
                        cell_index=0,
                        workers_hash="hash-1",
                        health_checker=SimpleNamespace(start=lambda: None, status=None),
                        provider=provider,
                    )
                    return cell, await cell.execute(
                        "configure_master_addr_and_port", master_addr="10.0.0.1", master_port=9000
                    )
                finally:
                    await stop()

        cell, results = asyncio.run(scenario())

        assert results == [9000, 9000]
        assert [worker.configured for worker in workers] == [[("10.0.0.1", 9000)], [("10.0.0.1", 9000)]]
        assert set(transport.hosts_called) == {"10.0.0.1", "10.0.0.2"}

    def test_healing_a_cell_deletes_its_pods(self, monkeypatch, deleted):
        # [FUNCTION DOC] Suspending the cell through the capability's cell operations deletes
        # exactly the two pods of the cell, in one call, in pod order.
        """Under Kubernetes a cell heals because its workload recreates the pods somebody deleted."""
        capability = install(monkeypatch, pods=cell_pods(2))

        operations = capability.cell_operations()
        asyncio.run(operations.suspend(cell_id=CELL_ID))

        assert deleted == [[f"{POOL}-0-0", f"{POOL}-0-1"]]

    def test_the_rollout_executor_answers_over_rpc_with_no_actor_behind_it(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] The fake executor is served at the static host of the executor pool. A
        # handle created from the capability is driven: set the train parallel config, load rollout
        # 11, fetch the data of rollout 3. The handle is an rpc worker handle, the returned data pack
        # equals the canned one (including its store reference type), the fake recorded the config
        # and the load, and only the executor host was called.
        """Under Kubernetes the executor is a pod in the address book, not an actor the driver creates."""
        capability = install(monkeypatch, pods=cell_pods(2))

        executor = FakeRolloutExecutor()
        host = STATIC_HOSTS[specs_rollout.ROLLOUT_EXECUTOR_POOL_ID]
        app = create_rpc_app(executor)
        transport = _PerHostTransport({host: app})

        async def scenario():
            # [FUNCTION DOC] HELPER: Under a per-host http client (installed as the shared client),
            # enters the application lifespan, creates the executor handle, sets the parallel config,
            # loads rollout 11, and returns the handle with the data fetched for rollout 3.
            async with httpx.AsyncClient(transport=transport) as client:
                monkeypatch.setattr(http_utils.GeneralHttpClientProvider, "client", classmethod(lambda cls: client))
                await app.router.lifespan_context(app).__aenter__()
                handle = specs_rollout.create_rollout_executor_handle(capability=capability)
                await handle.set_train_parallel_config(config={"dp_size": 4})
                await handle.load(rollout_id=11)
                return handle, await handle.get(rollout_id=3)

        handle, rollout_data = asyncio.run(scenario())

        assert isinstance(handle, RpcWorkerHandle)
        assert rollout_data == _data_pack(3)
        assert isinstance(rollout_data.data_ref, _MooncakeStoreObjectRef)
        assert executor.train_parallel_config == {"dp_size": 4}
        assert executor.loaded == [11]
        assert set(transport.hosts_called) == {host}

    def test_the_rollout_components_assemble_around_those_handles(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Serves a fake inference controller (reporting an eval fleet of 2 gpus at 1
        # per engine behind a router at 10.0.0.9:31000) and a fake executor at their static hosts,
        # and points the placement group module's capability lookup at the installed capability.
        # With train-only args (no epoch count given, 3 epochs), the rollout components call returns
        # rpc handles for both; both fakes end up initialized; the number of rollouts per epoch is
        # 7 and the number of rollouts in args becomes 21; and the executor received the eval fleet
        # info unchanged.
        """create_rollout_components is the driver's door into rollout, so it too must open over rpc."""
        capability = install(monkeypatch, pods=cell_pods(2))

        eval_fleet_info = EvalFleetInfo(
            router=HostAndPort(host="10.0.0.9", port=31000), num_gpus=2, num_gpus_per_engine=1
        )
        controller = FakeInferenceController(eval_fleet_info)
        executor = FakeRolloutExecutor()
        executor_host = STATIC_HOSTS[specs_rollout.ROLLOUT_EXECUTOR_POOL_ID]
        controller_host = STATIC_HOSTS[specs_inference.INFERENCE_CONTROLLER_POOL_ID]
        apps = {
            executor_host: create_rpc_app(executor),
            controller_host: create_rpc_app(controller),
        }
        transport = _PerHostTransport(apps)
        args = SimpleNamespace(
            cluster_backend="kubernetes",
            pin_rollout_manager_to_head=False,
            num_rollout=None,
            num_epoch=3,
            debug_train_only=True,
            eval_num_gpus=0,
        )
        monkeypatch.setattr(placement_group, "get_backend_capability", lambda _args: capability)

        async def scenario():
            # [FUNCTION DOC] HELPER: Under a per-host http client (installed as the shared client),
            # enters every application lifespan and awaits the rollout components creation.
            async with httpx.AsyncClient(transport=transport) as client:
                monkeypatch.setattr(http_utils.GeneralHttpClientProvider, "client", classmethod(lambda cls: client))
                for app in apps.values():
                    await app.router.lifespan_context(app).__aenter__()
                return await create_rollout_components(args)

        result = asyncio.run(scenario())

        assert isinstance(result.inference_controller, RpcWorkerHandle)
        assert isinstance(result.rollout_executor, RpcWorkerHandle)
        assert controller.initialized
        assert executor.initialized
        assert result.num_rollout_per_epoch == 7
        assert args.num_rollout == 21
        # The fleet is only knowable through a call, and it must arrive at the executor intact.
        assert executor.eval_fleet_info == eval_fleet_info

    def test_the_trainer_controller_answers_over_rpc_with_no_actor_behind_it(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] The fake trainer controller is served at the static host of the actor
        # controller pool. A handle created with no external addresses is driven: init (with args
        # holding 7 rollouts) returns the canned [5], one train call for rollout 3, and the parallel
        # config fetch. The handle is an rpc worker handle, the fake received the args (7
        # rollouts), recorded the train call, the config is data parallel size 2, and only the
        # controller host was called.
        """Under Kubernetes the trainer controller is a pod the driver addresses, not an object it builds."""
        capability = install(monkeypatch, pods=cell_pods(2))

        controller = FakeTrainerController()
        host = STATIC_HOSTS[specs_train.compute_trainer_controller_pool_id("actor")]
        app = create_rpc_app(controller)
        transport = _PerHostTransport({host: app})

        async def scenario():
            # [FUNCTION DOC] HELPER: Under a per-host http client (installed as the shared client),
            # enters the application lifespan, creates the trainer controller handle for trainer
            # actor, asserts init returns [5], trains rollout 3, and returns the handle with the
            # fetched parallel config.
            async with httpx.AsyncClient(transport=transport) as client:
                monkeypatch.setattr(http_utils.GeneralHttpClientProvider, "client", classmethod(lambda cls: client))
                await app.router.lifespan_context(app).__aenter__()
                handle = specs_train.create_trainer_controller_handle(
                    Namespace(trainer_controller_addrs=None, megatron_config=None),
                    capability=capability,
                    trainer_id="actor",
                )
                assert await handle.init(Namespace(num_rollout=7)) == [5]
                await handle.train(rollout_id=3, rollout_data_pack=_data_pack(3))
                return handle, await handle.get_train_parallel_config()

        handle, parallel_config = asyncio.run(scenario())

        assert isinstance(handle, RpcWorkerHandle)
        assert controller.initialized.num_rollout == 7
        assert controller.trained == [(3, _data_pack(3))]
        assert parallel_config == {"dp_size": 2}
        assert set(transport.hosts_called) == {host}
