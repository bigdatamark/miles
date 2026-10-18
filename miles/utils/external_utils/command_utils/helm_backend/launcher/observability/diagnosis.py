# [MODULE DOC] Snapshot of a failed run for later reading. Creates a timestamped directory under a
# given output directory and fills it with: the namespace's events sorted by last timestamp, and,
# for every pod matching the selector, its logs (all containers), previous-instance logs and
# describe output; optionally a copy of the run state file. Each capture is a kubectl subprocess;
# failures are collected into a list of missing items instead of raising.
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.k8s_types import PodList

logger = logging.getLogger(__name__)

_NO_PREVIOUS_CONTAINER = "previous terminated container"


class Diagnosis(FrozenStrictBaseModel):
    # [CLASS DOC] Result: the directory holding the snapshot and a tuple of descriptions of items
    # that could not be captured (default empty).
    directory: Path
    missing: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        # [FUNCTION DOC] Property: true when nothing is missing.
        return not self.missing


def collect_diagnosis(
    *, namespace: str, output_dir: Path, selector: str | None = None, state_file: Path | None = None
) -> Diagnosis:
    # [FUNCTION DOC] Keyword-only: namespace, output directory, optional pod label selector, and
    # optional state file. Creates "miles-diagnosis-<namespace>-<local timestamp to microseconds>"
    # under the output directory (parents created; an existing directory raises). Writes events.txt
    # (the namespace's events sorted by last timestamp). Lists pods: a listing failure records
    # "pod listing in namespace ..." as missing; an empty listing records "pods of the run in
    # namespace ..." as missing; for each pod captures its files and collects what is missing. If a
    # state file is given, copies its text (or a "does not exist" note) to a file of the same name
    # in the diagnosis directory. Returns the directory and the missing items.
    directory = output_dir / f"miles-diagnosis-{namespace}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    directory.mkdir(parents=True)

    missing: list[str] = []
    if not _capture(
        path=directory / "events.txt",
        command=["kubectl", "get", "events", "-n", namespace, "--sort-by=.lastTimestamp"],
    ):
        missing.append("events")

    pods = _pod_names(namespace=namespace, selector=selector)
    if pods is None:
        missing.append(f"pod listing in namespace {namespace}")
    elif not pods:
        missing.append(f"pods of the run in namespace {namespace}")
    for pod in pods or []:
        missing += _capture_pod(pod, namespace=namespace, directory=directory)

    if state_file is not None:
        text = state_file.read_text() if state_file.is_file() else f"{state_file} does not exist\n"
        (directory / state_file.name).write_text(text)

    return Diagnosis(directory=directory, missing=tuple(missing))


def _capture_pod(pod: str, *, namespace: str, directory: Path) -> list[str]:
    # [FUNCTION DOC] HELPER: Writes <pod>.log (all containers), <pod>.previous.log (previous
    # instances; the failure "previous terminated container" is tolerated and counted as success
    # without writing a file) and <pod>.describe.txt into the directory. Returns the list of
    # descriptions of captures that failed (logs, previous logs, describe).
    missing = []
    if not _capture(
        path=directory / f"{pod}.log", command=["kubectl", "logs", pod, "-n", namespace, "--all-containers"]
    ):
        missing.append(f"logs of {pod}")
    if not _capture(
        path=directory / f"{pod}.previous.log",
        command=["kubectl", "logs", pod, "-n", namespace, "--all-containers", "--previous"],
        ignored_failure=_NO_PREVIOUS_CONTAINER,
    ):
        missing.append(f"previous logs of {pod}")
    if not _capture(
        path=directory / f"{pod}.describe.txt", command=["kubectl", "describe", "pod", pod, "-n", namespace]
    ):
        missing.append(f"describe of {pod}")
    return missing


def _pod_names(*, namespace: str, selector: str | None) -> list[str] | None:
    # [FUNCTION DOC] HELPER: Names of the pods in the namespace matching the selector (all pods if
    # none), empty if the listing is absent. Any error is logged as a warning and gives none (to
    # distinguish a failed listing from an empty one).
    try:
        listed = Kubectl.get_json("pods", return_type=PodList, namespace=namespace, selector=selector)
        return [pod.metadata.name for pod in listed.items] if listed is not None else []
    except Exception:
        logger.warning(f"Could not list the pods of namespace {namespace}", exc_info=True)
        return None


def _capture(*, path: Path, command: list[str], ignored_failure: str | None = None) -> bool:
    # [FUNCTION DOC] HELPER: Runs the command capturing output without raising. If it failed and
    # its error output contains the given tolerated text, returns true without writing anything.
    # Otherwise writes standard output followed by error output to the path and returns whether the
    # exit status was zero (a failed command's output is still saved).
    result = run_process(command, capture_output=True, check=False)
    if result.returncode != 0 and ignored_failure is not None and ignored_failure in result.stderr:
        return True
    path.write_text(result.stdout + result.stderr)
    return result.returncode == 0
