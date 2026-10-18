# [MODULE DOC] HELPER: Test helper deciding whether a cluster backend (Ray or Kubernetes) is usable
# from the current environment, and creating a command backend for a run. Kubernetes availability
# requires the kubectl and helm executables, a namespace (from the test-namespace environment
# variable, or, for a run config, from the config; the variable names used in messages differ), a
# reachable cluster (version endpoint answers), the leader-worker-set API served, and a set of
# namespaced permissions (list and delete pods, create pod exec, list leader-worker-sets), each
# checked with kubectl permission queries with a sixty second timeout. Ray availability only
# requires that the ray package imports. Unavailability is expressed as a result with a reason, which
# tests turn into a skip (require_backend) or a failure (create_backend_for_run).
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass

import pytest

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend
from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.workers.types import ClusterBackend

NAMESPACE_ENV_VAR = "MILES_TEST_K8S_NAMESPACE"

RUN_NAMESPACE_ENV_VAR = "MILES_SCRIPT_NAMESPACE"

KUBECTL_TIMEOUT_SECONDS: float = 60.0

LEADER_WORKER_SET_API_PATH: str = "/apis/leaderworkerset.x-k8s.io/v1"

logger = logging.getLogger(__name__)

REQUIRED_NAMESPACED_PERMISSIONS: tuple[tuple[str, str], ...] = (
    ("list", "pods"),
    ("delete", "pods"),
    ("create", "pods/exec"),
    ("list", "leaderworkersets.leaderworkerset.x-k8s.io"),
)


@dataclass(frozen=True)
class BackendAvailability:
    # [CLASS DOC] HELPER: Immutable result of an availability probe: a flag saying whether the
    # backend is usable and a human readable reason (why not, or what will be used).
    available: bool
    reason: str


def kubernetes_availability() -> BackendAvailability:
    # [FUNCTION DOC] HELPER: Kubernetes availability for the test lane: reads the namespace from the
    # test namespace environment variable (empty if unset) and delegates to the namespace-specific
    # probe, naming that variable in the missing-namespace message. Runs kubectl subprocesses.
    return kubernetes_availability_of_namespace(
        os.environ.get(NAMESPACE_ENV_VAR, ""), namespace_source=NAMESPACE_ENV_VAR
    )


def kubernetes_availability_of_namespace(namespace: str, *, namespace_source: str) -> BackendAvailability:
    # [FUNCTION DOC] HELPER: Probes Kubernetes in the namespace, returning the first failing
    # condition as an unavailable result. Steps in order: kubectl then helm must be on the path; the
    # namespace must be non-empty (the message names namespace_source as where to set it); a raw
    # version request must succeed (else the reason carries up to 200 characters of its stderr,
    # claiming the credentials were refused); a raw request for the leader-worker-set API path must
    # succeed; then for each required (verb, resource) pair a permission query in the namespace must
    # print "yes". If all pass, returns available with a reason naming the namespace. Spawns
    # subprocesses; no exceptions are raised for failed commands (a timeout error from the runner may
    # still propagate).
    for tool, why in (("kubectl", "so no cluster can be reached"), ("helm", "and a run is installed as a release")):
        if shutil.which(tool) is None:
            return BackendAvailability(False, f"{tool} is not installed, {why}")

    if not namespace:
        return BackendAvailability(False, compute_missing_namespace_reason(namespace_source))

    reachable = _run_kubectl(["get", "--raw", "/version"])
    if reachable.returncode != 0:
        return BackendAvailability(False, f"the cluster refused the credentials: {reachable.stderr.strip()[:200]}")

    served = _run_kubectl(["get", "--raw", LEADER_WORKER_SET_API_PATH])
    if served.returncode != 0:
        return BackendAvailability(False, "LeaderWorkerSet is not served by this cluster; an admin has to add it")

    for verb, resource in REQUIRED_NAMESPACED_PERMISSIONS:
        allowed = _run_kubectl(["auth", "can-i", verb, *_compute_can_i_target(resource), "--namespace", namespace])
        if allowed.stdout.strip() != "yes":
            return BackendAvailability(False, f"this account may not {verb} {resource} in namespace {namespace}")

    return BackendAvailability(True, f"using namespace {namespace}")


def compute_missing_namespace_reason(namespace_source: str) -> str:
    # [FUNCTION DOC] HELPER: Builds the message telling the user to set the named source (an
    # environment variable name) to a namespace of their own, with a pointer to the cluster backend
    # documentation page.
    return f"set {namespace_source} to a namespace of your own; see docs/advanced/cluster-backend.md"


def ray_availability() -> BackendAvailability:
    # [FUNCTION DOC] HELPER: Available if the ray package can be imported (the import is only a
    # probe); otherwise unavailable with the reason that ray is not installed.
    try:
        import ray  # noqa: F401
    except ImportError:
        return BackendAvailability(False, "ray is not installed")
    return BackendAvailability(True, "ray is importable")


def _compute_can_i_target(resource: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Splits a resource string at the first slash. Returns a one-item list of
    # the resource, or, when a subresource is present, a two-item list of the resource and the
    # subresource option for permission queries.
    target, _, subresource = resource.partition("/")
    return [target, f"--subresource={subresource}"] if subresource else [target]


def _run_kubectl(args: list[str]) -> subprocess.CompletedProcess[str]:
    # [FUNCTION DOC] HELPER: Runs kubectl with the arguments via the shared process runner, capturing
    # output, not raising on a nonzero exit, with a sixty second timeout. Returns the completed
    # process.
    return run_process(["kubectl", *args], capture_output=True, check=False, timeout=KUBECTL_TIMEOUT_SECONDS)


_AVAILABILITY = {
    ClusterBackend.RAY.value: ray_availability,
    ClusterBackend.KUBERNETES.value: kubernetes_availability,
}


def create_backend_for_run(config: command_utils.ExecuteTrainConfig) -> BaseCommandBackend:
    # [FUNCTION DOC] HELPER: Asserts the run config's backend is available (a failure, not a skip,
    # raising an assertion error with the reason) and then returns the command backend created from
    # the config.
    _require_backend_available(config=config)
    return config.create_backend()


def _require_backend_available(*, config: command_utils.ExecuteTrainConfig) -> None:
    # [FUNCTION DOC] HELPER: Computes availability for the config's backend and asserts it is
    # available; the assertion message says the lane asked for that backend so an unreachable one is
    # a failure rather than a skip, followed by the reason. On success logs the backend and reason at
    # info level.
    backend = config.cluster_backend
    availability = availability_of_run(config=config)
    assert availability.available, (
        f"this lane asked for the {backend.value} backend, so a backend it cannot reach is a failure and not a "
        f"skip: {availability.reason}"
    )
    logger.info(f"Running on the {backend.value} backend: {availability.reason}")


def availability_of_run(*, config: command_utils.ExecuteTrainConfig) -> BackendAvailability:
    # [FUNCTION DOC] HELPER: Dispatches on the config's cluster backend: Ray uses the Ray probe;
    # Kubernetes probes the config's namespace, naming the run namespace environment variable as the
    # place to set it. A backend value matching neither case falls through and returns nothing.
    match config.cluster_backend:
        case ClusterBackend.RAY:
            return ray_availability()
        case ClusterBackend.KUBERNETES:
            return kubernetes_availability_of_namespace(config.namespace, namespace_source=RUN_NAMESPACE_ENV_VAR)


def require_backend(backend: str) -> str:
    # [FUNCTION DOC] HELPER: Looks up the availability probe for the backend value string (raises a
    # key error for an unknown name), runs it, and skips the current test with the reason if it is
    # unavailable. Returns the test namespace environment variable's value (empty string if unset),
    # also for the Ray backend.
    availability = _AVAILABILITY[backend]()
    if not availability.available:
        pytest.skip(f"{backend} backend unavailable: {availability.reason}")
    return os.environ.get(NAMESPACE_ENV_VAR, "")


def both_backends(test):
    # [FUNCTION DOC] HELPER: Decorator parametrizing a test over the cluster backend argument with
    # every known backend value (sorted by name), using the values as test ids. Returns the
    # decorated test.
    return pytest.mark.parametrize("cluster_backend", sorted(_AVAILABILITY), ids=sorted(_AVAILABILITY))(test)
