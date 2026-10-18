# [MODULE DOC] Tests for the kubernetes assembly of worker capabilities: after installing the
# kubernetes workers, components get kubernetes providers (observers of the same run), static
# workers resolve to launcher-computed addresses, undeployed static pools are refused, suspending
# a cell deletes its pods, listing cells starts the watch it needs, and the Ray worker manager
# module is never imported. The pod API is replaced by a fake and the namespace and release come
# from environment variables set per test (autouse fixtures). Constant: the module path of the
# Ray worker manager.
from __future__ import annotations

import asyncio
import sys

import pytest
from tests.fast.fixtures.kubernetes_fixtures import _RELEASE, NAMESPACE, ROUTER_HOST, install_workers
from tests.fast.utils.workers.worker_provider.kubernetes import fake_pod_api
from tests.fast.utils.workers.worker_provider.kubernetes.core.test_pod_view import make_pod

from miles.utils.workers.cell_operations import kubernetes as cell_operations_kubernetes
from miles.utils.workers.worker_provider.kubernetes.core import provider as core_provider
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm.env import NAMESPACE_ENV_VAR, RELEASE_ENV_VAR
from miles.utils.workers.worker_provider.static import StaticWorkerProvider

RAY_WORKER_MANAGER_MODULE = "miles.utils.workers.ray_worker_manager"


@pytest.fixture
def deleted(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    # [FUNCTION DOC] HELPER: Fixture replacing the kubernetes cell operations module's pod delete
    # function with a recorder (reverted at teardown); returns the list of (namespace, pod names)
    # pairs it was called with.
    recorded: list[tuple[str, list[str]]] = []

    async def fake_delete_pods(*, namespace: str, pod_names: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Records the namespace and a copy of the pod names.
        recorded.append((namespace, list(pod_names)))

    monkeypatch.setattr(cell_operations_kubernetes, "_delete_pods", fake_delete_pods)
    return recorded


@pytest.fixture(autouse=True)
def _fake_pod_api(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture: resets the fake pod API and installs it in place of
    # the provider's pod API factory (reverted at teardown).
    fake_pod_api.reset()
    monkeypatch.setattr(core_provider, "_kubernetes_pod_api", fake_pod_api.installed)


@pytest.fixture(autouse=True)
def _release_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture setting the namespace and release environment
    # variables to the shared test values (reverted at teardown).
    monkeypatch.setenv(NAMESPACE_ENV_VAR, NAMESPACE)
    monkeypatch.setenv(RELEASE_ENV_VAR, _RELEASE)


class TestKubernetesAssembly:
    # [CLASS DOC] Tests of the capability installed by the shared install-workers fixture helper.
    def test_components_of_the_process_then_see_a_kubernetes_provider(self) -> None:
        # [FUNCTION DOC] The dynamic provider for pool engine is a kubernetes worker provider.
        """The whole point of the assembly: the capability must stop answering with Ray."""
        capability = install_workers()

        assert isinstance(capability.dynamic_worker_provider(pool_ids=["engine"]), KubernetesWorkerProvider)

    def test_a_static_worker_resolves_to_the_address_the_launcher_decided(self) -> None:
        # [FUNCTION DOC] The static provider for the router resolves the router worker's primary
        # address (run in a fresh event loop) to the shared router host and port 8000.
        """A router has no cell to observe, so its address is recomputed from the release that deployed it."""
        capability = install_workers()

        provider = capability.static_worker_provider(pool_id="inference-router-0")
        addr = asyncio.run(provider.get_addrs("inference-router-0-00000-00000"))["primary"]

        assert addr.host == ROUTER_HOST
        assert addr.port == 8000

    def test_refuses_a_static_worker_the_run_never_deployed(self) -> None:
        # [FUNCTION DOC] A static pool inference-router-9 raises an assertion error saying it is
        # not a static pool.
        """Answering with an invented address would send the caller at nothing at all."""
        capability = install_workers()

        with pytest.raises(AssertionError, match="not a static pool"):
            capability.static_worker_provider(pool_id="inference-router-9")

    def test_the_static_scope_answers_with_the_address_book_rather_than_with_the_watcher(self) -> None:
        # [FUNCTION DOC] The static provider is the static worker provider type, and watching its
        # cells raises a not-implemented error saying it does not observe cells.
        """Statically addressed components need no watch, and a watch would never report them anyway."""
        capability = install_workers()

        static = capability.static_worker_provider(pool_id="inference-router-0")

        assert isinstance(static, StaticWorkerProvider)
        with pytest.raises(NotImplementedError, match="does not observe cells"):
            asyncio.run(static.watch_cells(lambda cell_id, info: None))

    def test_every_component_observes_the_same_run(self) -> None:
        # [FUNCTION DOC] Two dynamic providers for engine share the same private run object
        # (identity).
        """Each component watches for itself, but they must all be told about the same pods."""
        capability = install_workers()

        first = capability.dynamic_worker_provider(pool_ids=["engine"])
        second = capability.dynamic_worker_provider(pool_ids=["engine"])

        assert first._run is second._run

    def test_suspending_a_cell_deletes_its_pods(self, deleted) -> None:
        # [FUNCTION DOC] With one engine pod installed, suspending cell engine-00000 through the
        # cell operations deletes that pod in the test namespace (recorded by the fixture).
        """Kubernetes has no suspend: a cell heals because its workload recreates deleted pods."""
        capability = install_workers(pods=[make_pod(name="engine-0-0", pool_id="engine", cell_id_suffix="0")])
        operations = capability.cell_operations()

        asyncio.run(operations.suspend(cell_id="engine-00000"))

        assert deleted == [(NAMESPACE, ["engine-0-0"])]

    def test_listing_cells_starts_the_watch_it_needs(self) -> None:
        # [FUNCTION DOC] Asking the cell operations for cell infos of pool engine without having
        # started a watch returns the single installed cell engine-00000.
        """The api server asks for cells without knowing that observation has to be started first."""
        capability = install_workers(pods=[make_pod(name="engine-0-0", pool_id="engine", cell_id_suffix="0")])
        operations = capability.cell_operations()

        infos = asyncio.run(operations.cell_infos(pool_ids=["engine"]))

        assert list(infos) == ["engine-00000"]

    def test_never_reaches_for_the_ray_worker_manager(self, monkeypatch: pytest.MonkeyPatch, deleted) -> None:
        # [FUNCTION DOC] Marks the Ray worker manager module as unimportable (a none entry in the
        # module table, reverted at teardown), installs workers, suspends a cell, and asserts
        # nothing failed: the dynamic provider is still a kubernetes one and the pod was deleted.
        """A namespace has no Ray cluster, so touching the manager would fail the run there."""
        monkeypatch.setitem(sys.modules, RAY_WORKER_MANAGER_MODULE, None)
        capability = install_workers(pods=[make_pod(name="engine-0-0", pool_id="engine", cell_id_suffix="0")])

        operations = capability.cell_operations()
        asyncio.run(operations.suspend(cell_id="engine-00000"))

        assert isinstance(capability.dynamic_worker_provider(pool_ids=["engine"]), KubernetesWorkerProvider)
        assert deleted == [(NAMESPACE, ["engine-0-0"])]
