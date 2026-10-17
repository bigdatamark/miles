# [MODULE DOC] HELPER: Read-only kubectl queries for objects belonging to one helm release, used by
# Kubernetes-based tests. Runs the command-line tool as a subprocess.
from collections.abc import Sequence

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL

KUBECTL_TIMEOUT_SECONDS: float = 60.0


def read_objects_of_release(
    *, kind: str, release: str, namespace: str, output: str, extra_labels: Sequence[str] = ()
) -> str:
    # [FUNCTION DOC] HELPER: Runs a "get" for the given resource kind in the namespace, filtered by
    # the release label selector (plus any extra label selectors), with the requested output
    # format (for example json or a name listing). Runs through the shared process runner with
    # captured output, a failure check and a 60 second timeout; so a nonzero exit or a timeout
    # raises (exception types come from the runner). Returns the captured standard output text.
    result = run_process(
        [
            "kubectl",
            "get",
            kind,
            "--namespace",
            namespace,
            "--selector",
            compute_release_selector(release=release, extra_labels=extra_labels),
            "--output",
            output,
        ],
        capture_output=True,
        check=True,
        timeout=KUBECTL_TIMEOUT_SECONDS,
    )
    return result.stdout


def compute_release_selector(*, release: str, extra_labels: Sequence[str] = ()) -> str:
    # [FUNCTION DOC] HELPER: Builds a comma-joined label selector string: the helm instance label
    # equal to the release name, followed by the extra label selector terms in order (all are
    # AND-ed by kubectl).
    return ",".join([f"{INSTANCE_LABEL}={release}", *extra_labels])
