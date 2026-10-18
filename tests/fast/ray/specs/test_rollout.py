# [MODULE DOC]
# Unit tests for the rollout executor worker spec: scheduling shape, worker class resolution,
# constructor keyword arguments built from a fake backend capability, stable names and rendering
# into helm static worker values. CPU only; no executor is actually started.
from __future__ import annotations

from argparse import Namespace

from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.ray.specs.rollout import (
    ROLLOUT_EXECUTOR_POOL_ID,
    ROLLOUT_EXECUTOR_WORKER_CLASS,
    rollout_executor_cell_id,
    rollout_executor_worker_name,
    spec_rollout_executor,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import SECTION_OF_CATEGORY, LaunchPlan
from miles.utils.function_registry import load_function
from miles.utils.misc import NodeProbeMixin
from miles.utils.workers.ray_worker_manager import bootstrapped_worker_class
from miles.utils.workers.worker_spec import WorkerCtorContext


def _args(*, debug_train_only: bool = False, use_session_server: bool = False, **overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds a test argument namespace with the rollout manager not pinned
    # to the head, the given train-only and session server flags, and any extra overrides.
    return make_args(
        pin_rollout_manager_to_head=False,
        debug_train_only=debug_train_only,
        use_session_server=use_session_server,
        **overrides,
    )


def _layout() -> LaunchPlan:
    # [FUNCTION DOC] HELPER: Returns a fixed launch plan (run id, state file, release, namespace,
    # orchestrator command and worker argv) used to render the spec into helm values.
    return LaunchPlan(
        run_id="260101-000000-000",
        state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
        release="miles-run-260101",
        namespace="rl",
        orchestrator_command=["python", "/repo/train.py"],
        worker_argv=["--rollout-num-gpus", "8"],
    )


class TestRolloutExecutorSpec:
    # [CLASS DOC] Tests the spec produced for the rollout executor worker.
    def test_a_run_asks_for_exactly_one_gpuless_worker(self):
        # [FUNCTION DOC] The spec is named with the executor pool id, has 1 cell of 1 worker, and
        # requests zero gpus per worker.
        """One executor per run, and it must claim no gpu or the scheduler would reserve a whole slot."""
        spec = spec_rollout_executor(_args())

        assert spec.name == ROLLOUT_EXECUTOR_POOL_ID
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (1, 1)
        assert spec.scheduling.num_gpus_per_worker == 0

    def test_the_worker_class_is_the_executor_itself(self):
        # [FUNCTION DOC] The worker class path in the spec loads to the real rollout executor class.
        """The spec names the class a pod or actor constructs, so it must resolve to the real implementation."""
        assert load_function(spec_rollout_executor(_args()).worker_class) is RolloutExecutor

    def test_the_worker_class_answers_the_managers_node_probe(self):
        # [FUNCTION DOC] The bootstrapped version of the worker class is a subclass of both the
        # rollout executor and the node probe mixin, so the manager can probe the node before
        # reading port information.
        """alloc_ports() probes the node before it reads port_infos, so a worker without the probe dies at launch."""
        bootstrapped = bootstrapped_worker_class(ROLLOUT_EXECUTOR_WORKER_CLASS)

        assert issubclass(bootstrapped, RolloutExecutor)
        assert issubclass(bootstrapped, NodeProbeMixin)

    def test_the_ctor_kwargs_hand_the_worker_the_providers_it_resolves_with(self):
        # [FUNCTION DOC] With a session server enabled and a fake capability whose static provider
        # is a plain object, the constructor kwargs have exactly the keys args, inference controller
        # provider, router providers and session server provider. The router providers are a one
        # element list with the static provider, the other two providers are the static provider,
        # and the static pool ids requested were the model 0 router, the session server and the
        # inference controller, in that order.
        """The executor resolves its own addresses in init(), so its spec names exactly what that takes."""
        capability = FakeBackendCapability(static_provider=object())
        context = WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)

        kwargs = spec_rollout_executor(_args(use_session_server=True)).ctor_kwargs(context)

        assert sorted(kwargs) == [
            "args",
            "inference_controller_provider",
            "router_providers",
            "session_server_provider",
        ]
        assert kwargs["router_providers"] == [capability.static_provider]
        assert kwargs["session_server_provider"] is capability.static_provider
        assert kwargs["inference_controller_provider"] is capability.static_provider
        assert capability.requested_static_pool_ids == [
            "inference-router-0",
            "session-server",
            "inference-controller",
        ]

    def test_a_run_without_session_servers_is_given_no_session_provider(self):
        # [FUNCTION DOC] With the session server disabled, the session server provider kwarg is
        # none.
        """Nothing is deployed to wait for, and a provider would make the executor wait for it anyway."""
        capability = FakeBackendCapability(static_provider=object())
        context = WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)

        kwargs = spec_rollout_executor(_args(use_session_server=False)).ctor_kwargs(context)

        assert kwargs["session_server_provider"] is None

    def test_the_executor_is_given_the_way_to_reach_the_inference_controller(self):
        # [FUNCTION DOC] The inference controller provider kwarg is the capability's static
        # provider, and the controller pool id is among the requested static pool ids.
        """Pinning the eval fleet is a call on the controller, so the executor must be able to address it."""
        capability = FakeBackendCapability(static_provider=object())
        context = WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)

        kwargs = spec_rollout_executor(_args()).ctor_kwargs(context)

        assert kwargs["inference_controller_provider"] is capability.static_provider
        assert "inference-controller" in capability.requested_static_pool_ids

    def test_the_worker_and_cell_names_are_stable(self):
        # [FUNCTION DOC] The executor worker name is rollout-executor-00000-00000 and the cell id
        # is rollout-executor-00000.
        """The driver looks the executor up by name, so these names are part of the release's contract."""
        assert rollout_executor_worker_name() == "rollout-executor-00000-00000"
        assert rollout_executor_cell_id() == "rollout-executor-00000"

    def test_it_renders_into_static_workers_with_its_rpc_port(self):
        # [FUNCTION DOC] Renders the spec into helm values. Exactly one static worker entry results;
        # the spec category maps to the static workers section; the entry is named after the executor
        # pool id, exposes an rpc port 8000, passes the pool id on its command line, and requests no
        # resources; the spec worker class is the executor worker class constant.
        """The release has to contain the executor pod, or the address book would point at nothing."""
        spec = spec_rollout_executor(_args())

        values = build_values([spec], _layout()).as_values()

        (entry,) = values["run"]["staticWorkers"]
        assert SECTION_OF_CATEGORY[spec.category] == "staticWorkers"
        assert entry["name"] == ROLLOUT_EXECUTOR_POOL_ID
        assert entry["ports"] == [{"name": "rpc", "port": 8000}]
        assert entry["command"][entry["command"].index("--pool-id") + 1] == ROLLOUT_EXECUTOR_POOL_ID
        assert spec.worker_class == ROLLOUT_EXECUTOR_WORKER_CLASS
        assert "resources" not in entry
