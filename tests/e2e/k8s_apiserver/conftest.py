# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC] HELPER: Pytest fixtures for the API server end-to-end tests. Session-scoped
# fixtures provide a run id (requires docker, skipping or failing otherwise) and two disposable
# API server environments: a normal one and an "expiring" one with the watch cache disabled (so
# watches can be made to expire). Per-test async fixtures provide a core API client authenticated
# with the environment's token (certificate checks off) and a freshly created namespace. When the
# keep-environment switch is set, the servers are left running at teardown.

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator

import pytest
from kubernetes_asyncio import client as kubernetes_client
from tests.e2e.k8s_apiserver.apiserver import (
    ApiserverEnvironment,
    log_apiserver_diagnostics,
    start_apiserver,
    stop_apiserver,
)
from tests.e2e.k8s_apiserver.environment import keep_environment, new_run_id, require_docker
from tests.e2e.k8s_apiserver.utils import build_insecure_api_client, create_namespace, wait_until_serving

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def k8s_run_id() -> str:
    # [FUNCTION DOC] HELPER: Fixture (session scope). Requires a working docker daemon (skips the
    # tests, or raises when docker is mandatory) and returns a fresh unique run id.
    require_docker()
    return new_run_id()


@pytest.fixture(scope="session")
def apiserver_environment(k8s_run_id: str, tmp_path_factory: pytest.TempPathFactory) -> Iterator[ApiserverEnvironment]:
    # [FUNCTION DOC] HELPER: Fixture (session scope). Starts an API server with the run id plus an
    # api suffix and a temp work directory, waits until it serves (running a private event loop),
    # and yields the environment. At teardown stops it, unless the keep switch is set, in which
    # case it only logs a warning and leaves it running.
    environment = start_apiserver(run_id=f"{k8s_run_id}-api", work_dir=tmp_path_factory.mktemp("apiserver"))
    try:
        asyncio.run(_wait_until_apiserver_serves(environment))
        yield environment
    finally:
        if keep_environment():
            logger.warning(f"leaving the apiserver running as requested {environment=}")
        else:
            stop_apiserver(environment)


@pytest.fixture
async def apiserver_core_v1(
    apiserver_environment: ApiserverEnvironment,
) -> AsyncIterator[kubernetes_client.CoreV1Api]:
    # [FUNCTION DOC] HELPER: Fixture (async, per test). Yields a core API client for the normal
    # API server, built with the environment endpoint and token without TLS verification; closes
    # the underlying client at teardown.
    api_client = build_insecure_api_client(endpoint=apiserver_environment.endpoint, token=apiserver_environment.token)
    try:
        yield kubernetes_client.CoreV1Api(api_client)
    finally:
        await api_client.close()


@pytest.fixture
async def apiserver_namespace(apiserver_core_v1: kubernetes_client.CoreV1Api) -> str:
    # [FUNCTION DOC] HELPER: Fixture (async, per test). Creates a new namespace on the normal API
    # server and returns its name. It is not deleted at teardown.
    return await create_namespace(apiserver_core_v1)


@pytest.fixture(scope="session")
def expiring_apiserver_environment(
    k8s_run_id: str, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[ApiserverEnvironment]:
    # [FUNCTION DOC] HELPER: Fixture (session scope). Like the normal API server fixture but with a
    # separate expiry-suffixed run id and the watch cache disabled, so resource-version expiry
    # (gone responses) can be provoked. Waits until serving, yields, and tears down the same way.
    environment = start_apiserver(
        run_id=f"{k8s_run_id}-expiry", work_dir=tmp_path_factory.mktemp("apiserver-expiry"), watch_cache=False
    )
    try:
        asyncio.run(_wait_until_apiserver_serves(environment))
        yield environment
    finally:
        if keep_environment():
            logger.warning(f"leaving the apiserver running as requested {environment=}")
        else:
            stop_apiserver(environment)


@pytest.fixture
async def expiring_apiserver_core_v1(
    expiring_apiserver_environment: ApiserverEnvironment,
) -> AsyncIterator[kubernetes_client.CoreV1Api]:
    # [FUNCTION DOC] HELPER: Fixture (async, per test). Yields a core API client for the expiring
    # API server (no TLS verification); closes the client at teardown.
    api_client = build_insecure_api_client(
        endpoint=expiring_apiserver_environment.endpoint, token=expiring_apiserver_environment.token
    )
    try:
        yield kubernetes_client.CoreV1Api(api_client)
    finally:
        await api_client.close()


@pytest.fixture
async def expiring_apiserver_namespace(expiring_apiserver_core_v1: kubernetes_client.CoreV1Api) -> str:
    # [FUNCTION DOC] HELPER: Fixture (async, per test). Creates a new namespace on the expiring API
    # server and returns its name.
    return await create_namespace(expiring_apiserver_core_v1)


async def _wait_until_apiserver_serves(environment: ApiserverEnvironment) -> None:
    # [FUNCTION DOC] HELPER: Builds a temporary insecure client and waits until the API server
    # answers. On any failure (including cancellation) logs the containers' recent logs and
    # re-raises; always closes the client.
    api_client = build_insecure_api_client(endpoint=environment.endpoint, token=environment.token)
    try:
        await wait_until_serving(kubernetes_client.CoreV1Api(api_client))
    except BaseException:
        log_apiserver_diagnostics(environment)
        raise
    finally:
        await api_client.close()
