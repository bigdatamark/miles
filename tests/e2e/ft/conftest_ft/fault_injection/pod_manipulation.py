# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

# [MODULE DOC] kubectl helpers that crash a cell's pods in a Kubernetes-backed FT test: find the
# pods of a cell within a helm release (by pool id and cell index labels derived from the cell
# id), delete one of them, or send a forced kill to matching processes inside a pod container.
# All kubectl calls are bounded by the shared kubectl timeout.

import logging
import random

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS, read_objects_of_release
from miles.utils.workers.naming import parse_cell_id
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS

logger = logging.getLogger(__name__)


def delete_one_pod_of_cell(*, namespace: str, release: str, cell_id: str, rng: random.Random) -> str:
    # [FUNCTION DOC] Deletes one randomly chosen pod of the cell. Lists the cell's pods in the
    # release and namespace (assertion error if there are none), picks one with the generator, and
    # runs the kubectl delete without waiting for termination (check enabled, so a failing command
    # raises). Logs and returns the deleted pod's name.
    pod_names = list_pod_names_of_cell(namespace=namespace, release=release, cell_id=cell_id)
    assert pod_names, f"Release {release} has no pod of cell {cell_id} in {namespace}, so there is nothing to delete"

    pod_name = rng.choice(pod_names)
    run_process(
        ["kubectl", "delete", "pod", "--namespace", namespace, "--wait=false", pod_name],
        capture_output=True,
        check=True,
        timeout=KUBECTL_TIMEOUT_SECONDS,
    )
    logger.info(f"Deleted pod {pod_name} of cell {cell_id}, one of {pod_names}")
    return pod_name


def list_pod_names_of_cell(*, namespace: str, release: str, cell_id: str) -> list[str]:
    # [FUNCTION DOC] Returns the names of pods in the release and namespace that carry the labels
    # identifying the cell (pool id and cell index). Reads pod names via kubectl and splits the
    # whitespace-separated output; empty list when none match.
    return read_objects_of_release(
        kind="pods",
        release=release,
        namespace=namespace,
        output="jsonpath={.items[*].metadata.name}",
        extra_labels=_compute_cell_labels(cell_id),
    ).split()


def _compute_cell_labels(cell_id: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Parses the cell id into pool id and cell index and returns two label
    # selector strings of the form key equals value, using the default label keys.
    parsed = parse_cell_id(cell_id)
    return [f"{DEFAULT_LABEL_KEYS.pool_id}={parsed.pool_id}", f"{DEFAULT_LABEL_KEYS.cell_index}={parsed.cell_index}"]


def sigkill_process_patterns_in_pod(*, namespace: str, pod_name: str, container: str, process_pattern: str) -> None:
    # [FUNCTION DOC] Executes a forced-kill (signal 9) of every process whose full command line
    # matches the pattern, inside the named container of the pod, via kubectl exec with a timeout.
    # The exit status is checked manually: a nonzero status (typically no process matched)
    # raises an assertion error including the status and captured output, so a crash that did not
    # happen is never counted as one that did. Logs on success.
    result = run_process(
        [
            "kubectl",
            "exec",
            "--namespace",
            namespace,
            pod_name,
            "--container",
            container,
            "--",
            "pkill",
            "-9",
            "-f",
            process_pattern,
        ],
        capture_output=True,
        check=False,
        timeout=KUBECTL_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"No process matching {process_pattern!r} was killed inside {pod_name} (exit "
        f"{result.returncode}): {result.stderr.strip() or result.stdout.strip()}. A crash nobody caused would "
        f"otherwise be counted as one that happened"
    )

    logger.info(f"Sigkilled a {process_pattern} process inside pod {pod_name}")
