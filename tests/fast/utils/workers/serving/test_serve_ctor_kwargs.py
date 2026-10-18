# [MODULE DOC] Unit tests for how the inner serve entrypoint builds a worker from a serve worker
# spec: constructor keyword arguments computed by the spec are passed by keyword, an unknown pool id
# is refused, the dynamic worker provider capability handed to specs is built lazily and only once,
# and the rpc port of a worker depends on its index inside the pod (worker zero keeps the declared
# static port). The spec table and worker class are registered temporarily in the function registry
# under test names, and every test sets the cell index environment variable to zero.
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.fixtures.kubernetes_fixtures import _RELEASE, NAMESPACE, install_workers

from miles.utils.function_registry import function_registry
from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR
from miles.utils.workers.serving import serve_inner
from miles.utils.workers.serving.utils import compute_serve_worker_spec
from miles.utils.workers.serving.worker_identity import SUBPROCESS_INDEX_ENV_VAR, read_worker_in_pod_index
from miles.utils.workers.worker_provider.kubernetes.helm.env import NAMESPACE_ENV_VAR, RELEASE_ENV_VAR
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

SPECS_FN = "test:specs"
WORKER_FN = "test:worker"

POOL_ID = "trainer-engine-actor"
RPC_PORT = 8000


class KeywordOnlyWorker:
    # [CLASS DOC] HELPER: Worker whose constructor accepts only a keyword argument named args; records it.
    def __init__(self, *, args: str) -> None:
        # [FUNCTION DOC] HELPER: Stores the args string.
        self.args = args


def compute_specs(worker_argv: list[str]) -> list[ServeWorkerSpec]:
    # [FUNCTION DOC] HELPER: Spec table function registered under the test name. Returns one serve
    # spec named by the pool id with a static rpc port of 8000, an empty environment, one cell of one
    # cpu-only worker, the registered keyword-only worker class, and constructor keyword arguments
    # holding args equal to the pool id, a colon, and the worker argv joined by spaces.
    return [
        ServeWorkerSpec(
            name=POOL_ID,
            port_infos=[PortInfo(name="rpc", static_port=RPC_PORT)],
            env_var=lambda context: {},
            scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
            worker_class=WORKER_FN,
            ctor_kwargs=lambda context: dict(args=f"{POOL_ID}:{' '.join(worker_argv)}"),
        )
    ]


@pytest.fixture(autouse=True)
def pod_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture setting the cell index environment variable to "0" for
    # every test (undone afterwards).
    """Every pod is told which cell of its pool it serves before serve runs."""
    monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")


@pytest.fixture
def registered_functions():
    # [FUNCTION DOC] HELPER: Fixture that temporarily registers the spec table function and the
    # keyword-only worker class in the function registry under their test names, for the duration of
    # the test.
    with function_registry.temporary(SPECS_FN, compute_specs):
        with function_registry.temporary(WORKER_FN, KeywordOnlyWorker):
            yield


def worker_of(worker_argv: list[str]) -> Any:
    # [FUNCTION DOC] HELPER: Computes the serve spec for the test pool from the registered spec table
    # and the given worker argv, then creates the worker with the inner entrypoint's create function.
    # Returns the worker.
    spec = compute_serve_worker_spec(specs_fn=SPECS_FN, pool_id=POOL_ID, worker_argv=worker_argv)
    return serve_inner.create_worker(spec, specs_fn=SPECS_FN, worker_argv=worker_argv)


class TestCreateWorker:
    # [CLASS DOC] Tests worker creation from specs.
    def test_builds_a_keyword_only_worker_from_the_computed_kwargs(self, registered_functions):
        # [FUNCTION DOC] Builds the worker with argv "--rollout-num-gpus 8"; asserts it is the
        # keyword-only worker and its args string is the pool id, colon and joined argv.
        """Every real served worker takes keyword arguments, so handing it the argv positionally is a TypeError."""
        worker = worker_of(["--rollout-num-gpus", "8"])

        assert isinstance(worker, KeywordOnlyWorker)
        assert worker.args == "trainer-engine-actor:--rollout-num-gpus 8"

    def test_refuses_a_pool_the_run_does_not_describe(self, registered_functions):
        # [FUNCTION DOC] Asking for a pool id not in the spec table raises an assertion error saying
        # there is not one spec with that name.
        """The pod and the launcher would otherwise disagree silently about what this process serves."""
        with pytest.raises(AssertionError, match="not one spec named"):
            compute_serve_worker_spec(specs_fn=SPECS_FN, pool_id="no-such-pool", worker_argv=[])


class TestDeferredCapability:
    # [CLASS DOC] Tests laziness of the cluster backend capability handed to constructor contexts.
    def test_the_backend_is_built_only_when_a_spec_asks_for_a_provider(
        self, registered_functions, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Sets the namespace and release environment variables and replaces the
        # inner module's backend capability factory with one recording the spec count and returning a
        # fake install, and the argument parser with one returning the ray backend. Creates a worker
        # from a spec whose constructor kwargs function captures the context's capability. Asserts no
        # backend was built yet; then requests the dynamic worker provider twice from the captured
        # capability and asserts the backend was built exactly once (with one spec).
        """Every served worker is handed this capability, and most specs never look at it."""
        built: list[int] = []
        captured: dict[str, Any] = {}

        def _install(*, specs, cluster_backend):
            # [FUNCTION DOC] HELPER: Records how many specs it was given and returns the fixture's
            # fake worker installation; the cluster backend argument is ignored.
            built.append(len(specs))
            return install_workers()

        monkeypatch.setenv(NAMESPACE_ENV_VAR, NAMESPACE)
        monkeypatch.setenv(RELEASE_ENV_VAR, _RELEASE)
        monkeypatch.setattr(serve_inner, "get_backend_capability", _install)
        monkeypatch.setattr(serve_inner, "parse_args", lambda: SimpleNamespace(cluster_backend="ray"))

        serve_inner.create_worker(_capturing_spec(captured), specs_fn=SPECS_FN, worker_argv=[])
        capability = captured["capability"]
        assert built == []

        capability.dynamic_worker_provider(pool_ids=["engine"])
        capability.dynamic_worker_provider(pool_ids=["engine"])

        assert built == [1]


class TestRpcPortOfARank:
    # [CLASS DOC] Tests the effective rpc port per worker index inside a pod.
    def test_the_workers_of_one_pod_listen_on_different_ports(self, registered_functions):
        # [FUNCTION DOC] For subprocess indices 0 to 3, the effective static port equals the declared
        # port plus the index (8000 to 8003).
        """The supervisor runs them all in one network namespace, so a shared port is a bind failure."""
        spec = compute_serve_worker_spec(specs_fn=SPECS_FN, pool_id=POOL_ID, worker_argv=[])
        ports = [
            serve_inner._rpc_port_of(spec).effective_static_port(
                worker_in_pod_index=read_worker_in_pod_index({SUBPROCESS_INDEX_ENV_VAR: str(index)})
            )
            for index in range(4)
        ]

        assert ports == [8000, 8001, 8002, 8003]

    def test_the_first_worker_keeps_the_port_the_address_book_predicts(self, registered_functions):
        # [FUNCTION DOC] With an empty environment (no supervisor index), the effective port is the
        # declared static port.
        """The provider addresses a pod at the spec's static rpc port, which worker zero has to answer on."""
        spec = compute_serve_worker_spec(specs_fn=SPECS_FN, pool_id=POOL_ID, worker_argv=[])

        assert (
            serve_inner._rpc_port_of(spec).effective_static_port(worker_in_pod_index=read_worker_in_pod_index({}))
            == RPC_PORT
        )


def _capturing_spec(captured: dict[str, Any]) -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a serve spec like the registered one (static rpc port 8000, empty
    # environment, one cpu worker, the registered worker class) whose constructor kwargs function
    # stores the context's capability into the given mapping under the key capability and returns
    # args "captured".
    def ctor_kwargs(context) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Saves the context capability and returns the constructor arguments.
        captured["capability"] = context.capability
        return dict(args="captured")

    return ServeWorkerSpec(
        name=POOL_ID,
        port_infos=[PortInfo(name="rpc", static_port=RPC_PORT)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        worker_class=WORKER_FN,
        ctor_kwargs=ctor_kwargs,
    )
