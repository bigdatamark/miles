# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC] HELPER: Pytest fixtures for the kind suite: a session-scoped cluster (either the
# existing cluster that a kubeconfig environment variable points at, never deleted, or a freshly
# created kind cluster that is deleted at the end unless the keep switch is set), a per-test core
# API client built from the cluster's kubeconfig, and a per-test namespace deleted after the test.

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator

import pytest
from kubernetes_asyncio import client as kubernetes_client
from tests.e2e.k8s_apiserver.environment import existing_kubeconfig, keep_environment, new_run_id, require_docker
from tests.e2e.k8s_apiserver.utils import create_namespace
from tests.e2e.k8s_kind.kind_cluster import KindCluster, create_cluster, delete_cluster
from tests.e2e.k8s_kind.utils import build_kubeconfig_api_client

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def kind_cluster(tmp_path_factory: pytest.TempPathFactory) -> Iterator[KindCluster]:
    # [FUNCTION DOC] HELPER: Fixture (session scope). If the kubeconfig environment variable is set,
    # logs a warning and yields a cluster record named reused with that kubeconfig (no docker check,
    # no cleanup). Otherwise requires docker (skip or error), creates a kind cluster named by a new
    # run id with its kubeconfig written under a temp directory, yields it, and at teardown deletes
    # it unless the keep switch is set (then logs a warning).
    reused = existing_kubeconfig()
    if reused is not None:
        logger.warning(f"reusing the cluster the environment points at {reused=}")
        yield KindCluster(name="reused", kubeconfig=reused)
        return

    require_docker()
    cluster = create_cluster(run_id=new_run_id(), kubeconfig=tmp_path_factory.mktemp("kind") / "kubeconfig")
    try:
        yield cluster
    finally:
        if keep_environment():
            logger.warning(f"leaving the cluster up as requested {cluster=}")
        else:
            delete_cluster(cluster)


@pytest.fixture
async def cluster_core_v1(kind_cluster: KindCluster) -> AsyncIterator[kubernetes_client.CoreV1Api]:
    # [FUNCTION DOC] HELPER: Fixture (async, per test). Builds an API client from the cluster's
    # kubeconfig, yields a core API wrapper, and closes the client at teardown.
    api_client = await build_kubeconfig_api_client(kubeconfig=kind_cluster.kubeconfig)
    try:
        yield kubernetes_client.CoreV1Api(api_client)
    finally:
        await api_client.close()


@pytest.fixture
async def cluster_namespace(cluster_core_v1: kubernetes_client.CoreV1Api) -> AsyncIterator[str]:
    # [FUNCTION DOC] HELPER: Fixture (async, per test). Creates a uniquely named namespace, yields
    # its name, and at teardown deletes it with a zero grace period (without waiting for the
    # deletion to complete).
    namespace = await create_namespace(cluster_core_v1)
    try:
        yield namespace
    finally:
        await cluster_core_v1.delete_namespace(name=namespace, grace_period_seconds=0)
