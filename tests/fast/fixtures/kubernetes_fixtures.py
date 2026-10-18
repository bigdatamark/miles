# [MODULE DOC]
# Fixtures for kubernetes worker provider tests: installs a fake pod api and builds the helm backend
# capability for one router spec and one engine spec. Constants: the namespace name and the static
# worker host of the first router, computed from the shared test release name.
from __future__ import annotations

from typing import Any

from tests.fast.utils.workers.worker_provider.kubernetes import fake_pod_api
from tests.fast.utils.workers.worker_provider.kubernetes.core.test_provider import FakePodApi
from tests.fast.utils.workers.worker_provider.kubernetes.run_specs import _RELEASE, make_engine_spec, make_router_spec

from miles.utils.workers.backend_capability.kubernetes import KubernetesBackendCapability
from miles.utils.workers.worker_provider.kubernetes.helm import naming
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability

NAMESPACE = "team-a"
ROUTER_HOST = naming.static_worker_host(_RELEASE, "inference-router-0", 0)


def install_workers(*, pods: list[Any] | None = None) -> KubernetesBackendCapability:
    # [FUNCTION DOC] Installs (replacing any previous one, globally) a fake pod api seeded with a copy
    # of the given pods (default none), then returns the helm backend capability computed from a router
    # spec and an engine spec.
    fake_pod_api.install(FakePodApi(pods=list(pods or [])))

    return compute_helm_backend_capability(specs=[make_router_spec(), make_engine_spec()])
