# [PACKAGE DOC] Observability for a launcher that is following a run: a context manager that
# reports pod phases and warning events and follows pod logs in the background, and the farewell
# text printed when the launcher stops watching (how to keep reading logs, when the release
# uninstalls itself, how to tear down). Submodules: cluster_info (pod/event reporting),
# log_follower, polling, pod_facts (pod and event readers) and diagnosis (failure snapshot).
from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.cluster_info import with_cluster_info
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.log_follower import (
    with_log_following,
)

_MAX_LOG_REQUESTS = 100

_OBSERVABILITY_BOUNDARY = (
    "this launcher follows the pods of this run while it is watching, and prints their phases and warning "
    "events; anything beyond that -- metrics, history, logs of pods that already went away -- belongs to your "
    "cluster's own observability stack, which miles deliberately does not replace"
)


@contextmanager
def with_observability(*, namespace: str, selector: str) -> Iterator[None]:
    # [FUNCTION DOC] Context manager (keyword-only inputs: namespace and label selector of the
    # pods). On entry starts cluster-info reporting and then log following for those pods; both
    # are stopped, in reverse order, when the block exits (also on error). Yields nothing.
    with ExitStack() as stack:
        stack.enter_context(with_cluster_info(namespace=namespace, selector=selector))
        stack.enter_context(with_log_following(namespace=namespace, selector=selector))
        yield


def farewell(*, namespace: str, release: str, workload: str) -> str:
    # [FUNCTION DOC] Keyword-only. Returns a multi-line message: the run continues after the
    # launcher exits; a command to follow the orchestrator workload's log; a command to follow all
    # pods of the release (all containers, prefixed, up to 100 concurrent log requests); a note
    # that the release uninstalls itself about two minutes after the run's verdict while logs and
    # diagnosis stay in the run directory; the helm uninstall command for earlier teardown; and the
    # observability boundary statement.
    return "\n".join(
        [
            "the run keeps going after this launcher exits",
            f"  orchestrator log: {_orchestrator_log_command(namespace=namespace, workload=workload)}",
            f"  every pod of the run: {_release_log_command(namespace=namespace, release=release)}",
            "  this release uninstalls itself about two minutes after the run's verdict; the logs and the "
            "diagnosis of the run stay in its run directory",
            f"  tear down earlier: helm uninstall -n {namespace} {release}",
            f"  {_OBSERVABILITY_BOUNDARY}",
        ]
    )


def _orchestrator_log_command(*, namespace: str, workload: str) -> str:
    # [FUNCTION DOC] HELPER: Text of a kubectl logs command following the "orchestrator" container
    # of the stateful set workload.
    return f"kubectl logs --follow --namespace {namespace} statefulset/{workload} -c orchestrator"


def _release_log_command(*, namespace: str, release: str) -> str:
    # [FUNCTION DOC] HELPER: Text of a kubectl logs command following every container of every pod
    # selected by the release label, with pod prefixes and a limit of 100 concurrent log requests.
    return (
        f"kubectl logs --follow --namespace {namespace} --selector {Kubectl.release_selector(release)} "
        f"--all-containers --prefix --max-log-requests {_MAX_LOG_REQUESTS}"
    )
