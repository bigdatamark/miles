# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC] HELPER: Generic helpers for the API server tests: constants (the cell label key used
# as the label selector, and the pause and busybox image names), unique name generation, pod
# manifest building, an insecure token-authenticated API client builder, namespace creation, and
# two polling waiters (server readiness and arbitrary predicate).

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable

from kubernetes_asyncio import client as kubernetes_client

logger = logging.getLogger(__name__)

CELL_LABEL = "miles-cell"
PAUSE_IMAGE = "registry.k8s.io/pause:3.9"
BUSYBOX_IMAGE = "busybox:1.36"


def unique_name(prefix: str) -> str:
    # [FUNCTION DOC] HELPER: The prefix, a dash, and 8 random hex characters.
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def pod_body(
    *,
    name: str,
    cell: str,
    image: str = PAUSE_IMAGE,
    command: list[str] | None = None,
    restart_policy: str = "Never",
    grace_period_seconds: int = 0,
) -> kubernetes_client.V1Pod:
    # [FUNCTION DOC] HELPER: Builds a pod manifest object (not created on a server). Inputs: pod
    # name; the cell label value (stored under the cell label key); image (default the pause
    # image); optional command; restart policy (default never); termination grace period seconds
    # (default 0 so deletion is immediate). The pod has one container named main.
    return kubernetes_client.V1Pod(
        metadata=kubernetes_client.V1ObjectMeta(name=name, labels={CELL_LABEL: cell}),
        spec=kubernetes_client.V1PodSpec(
            restart_policy=restart_policy,
            termination_grace_period_seconds=grace_period_seconds,
            containers=[kubernetes_client.V1Container(name="main", image=image, command=command)],
        ),
    )


def build_insecure_api_client(*, endpoint: str, token: str) -> kubernetes_client.ApiClient:
    # [FUNCTION DOC] HELPER: Builds an API client for the endpoint with TLS verification disabled
    # (the test server has a self-signed certificate) and a default authorization header carrying
    # the bearer token. The caller must close the client.
    configuration = kubernetes_client.Configuration(host=endpoint)
    configuration.verify_ssl = False
    api_client = kubernetes_client.ApiClient(configuration=configuration)
    api_client.set_default_header("Authorization", f"Bearer {token}")
    return api_client


async def create_namespace(core_v1_api: kubernetes_client.CoreV1Api) -> str:
    # [FUNCTION DOC] HELPER: Creates a namespace with a unique name (test prefix plus random hex)
    # and returns the name. Raises the client error if creation fails.
    namespace = unique_name("miles-test")
    await core_v1_api.create_namespace(
        body=kubernetes_client.V1Namespace(metadata=kubernetes_client.V1ObjectMeta(name=namespace))
    )
    return namespace


async def wait_until_serving(core_v1_api: kubernetes_client.CoreV1Api, *, timeout: float = 180.0) -> None:
    # [FUNCTION DOC] HELPER: Polls the server by listing one namespace until it succeeds, retrying
    # every 0.5 seconds on any error. After the timeout (default 180 seconds) logs an error and
    # re-raises the last error.
    deadline = time.monotonic() + timeout
    while True:
        try:
            await core_v1_api.list_namespace(limit=1)
            return
        except Exception:
            if time.monotonic() >= deadline:
                logger.error(f"the apiserver never started serving within {timeout=}", exc_info=True)
                raise
            await asyncio.sleep(0.5)


async def wait_until(
    predicate: Callable[[], bool], *, description: str, timeout: float = 120.0, interval: float = 0.2
) -> None:
    # [FUNCTION DOC] HELPER: Async wait until a synchronous predicate returns true, checking first
    # immediately and then every interval (default 0.2 seconds). Past the timeout (default 120
    # seconds) fails an assertion whose message includes the description.
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out after {timeout}s waiting until {description}"
        await asyncio.sleep(interval)
