# [MODULE DOC]
# Read-only helpers that query pods and pod warning events through kubectl and derive facts from
# typed pod objects: phase, readiness, restart count, scheduling gate, startup failure reason, and
# human-readable status descriptions and diffs between two pod listings. Also defines the container
# identity tuples used by the log follower.
from __future__ import annotations

from typing import NamedTuple

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.observer import ObservedPod
from miles.utils.workers.k8s_types import Event, EventList, Pod, PodList

_SCHEDULING_GATED_REASON = "SchedulingGated"
_POD_WARNINGS_SELECTOR = "involvedObject.kind=Pod,type=Warning"
_STARTUP_FAILURE_REASONS = frozenset(
    {
        "CrashLoopBackOff",
        "ImagePullBackOff",
        "ErrImagePull",
        "InvalidImageName",
        "CreateContainerConfigError",
        "CreateContainerError",
        "RunContainerError",
    }
)


class ContainerKey(NamedTuple):
    # [CLASS DOC] Identity of one followable log source: pod name, container name, and a flag saying
    # whether it refers to the previous (terminated) instance of the container.
    pod: str
    container: str
    previous: bool


class ContainerRun(NamedTuple):
    # [CLASS DOC] One container instance: its key, a container id made of pod uid and runtime
    # container id (unique per restart), and whether the instance is currently running.
    key: ContainerKey
    container_id: str
    running: bool


def selected_pods(namespace: str, selector: str) -> list[Pod]:
    # [FUNCTION DOC] Lists pods in the namespace matching the label selector via kubectl (network
    # call, subprocess). Returns the typed pods, or an empty list when kubectl returns nothing.
    listed = Kubectl.get_json("pods", return_type=PodList, namespace=namespace, selector=selector)
    return listed.items if listed is not None else []


def pod_events(namespace: str, pods: list[Pod]) -> list[Event]:
    # [FUNCTION DOC] Returns warning events whose involved object is one of the given pods. Returns
    # an empty list without calling kubectl if no pods are given. Otherwise lists all pod-kind
    # warning events in the namespace via kubectl and filters them by matching the involved object
    # uid against the pods' uids.
    wanted = {pod.metadata.uid for pod in pods}
    if not wanted:
        return []

    listed = Kubectl.get_json(
        "events", return_type=EventList, namespace=namespace, field_selector=_POD_WARNINGS_SELECTOR
    )
    events = listed.items if listed is not None else []
    return [event for event in events if event.involved_object.uid in wanted]


def observed_pod(namespace: str, workload: str) -> ObservedPod | None:
    # [FUNCTION DOC] Fetches the first replica pod of the workload (named workload plus suffix
    # "-0") via kubectl. Returns none if it does not exist; otherwise an observed-pod record with
    # its phase and startup failure reason (if any).
    described = Kubectl.get_json("pod", return_type=Pod, name=f"{workload}-0", namespace=namespace)
    if described is None:
        return None
    return ObservedPod(phase=phase_of_pod(described), startup_failure=startup_failure_of_pod(described))


def status_changes(previous: list[Pod], current: list[Pod]) -> list[str]:
    # [FUNCTION DOC] Compares two pod listings by pod name and returns human-readable change lines.
    # First, one "gone" line per vanished pod (sorted by name); then, in sorted name order, an
    # "appeared" line with the status description for new pods or an "is now" line for pods whose
    # description (phase plus restart info) changed. Unchanged pods produce nothing.
    before = {pod.metadata.name: pod for pod in previous}
    after = {pod.metadata.name: pod for pod in current}

    lines = [f"pod {name} is gone" for name in sorted(set(before) - set(after))]
    for name in sorted(after):
        described = _described(after[name])
        if (was := before.get(name)) is None:
            lines.append(f"pod {name} appeared: {described}")
        elif _described(was) != described:
            lines.append(f"pod {name} is now {described}")
    return lines


def container_runs(pods: list[Pod]) -> dict[ContainerKey, ContainerRun]:
    # [FUNCTION DOC] Expands pods into followable container runs, returned as a mapping from
    # container key to run. For each container status of each pod, up to two runs are produced: the
    # current instance (if it has a container id; marked running only when its state is running) and
    # the previous instance (if the last state is terminated with a container id; never marked
    # running). The run's container id is the pod uid and runtime container id joined by a slash.
    runs = [
        ContainerRun(
            key=ContainerKey(pod=pod.metadata.name, container=container.name, previous=previous),
            container_id=f"{pod.metadata.uid}/{container_id}",
            running=container.state.running is not None and not previous,
        )
        for pod in pods
        for container in pod.status.container_statuses
        for previous, container_id in (
            (False, container.container_id),
            (True, container.last_state.terminated.container_id if container.last_state.terminated else None),
        )
        if container_id
    ]
    return {run.key: run for run in runs}


def phase_of_pod(pod: Pod) -> str:
    # [FUNCTION DOC] Returns the pod status phase, or the text "Unknown" when it is empty.
    return pod.status.phase or "Unknown"


def is_pod_ready(pod: Pod) -> bool:
    # [FUNCTION DOC] True if any pod condition of type Ready has status "True".
    return any(condition.type == "Ready" and condition.status == "True" for condition in pod.status.conditions)


def restarts_of_pod(pod: Pod) -> int:
    # [FUNCTION DOC] Total restart count summed over all container statuses of the pod.
    return sum(container.restart_count for container in pod.status.container_statuses)


def startup_failure_of_pod(pod: Pod) -> str | None:
    # [FUNCTION DOC] Returns the waiting reason of the first container whose waiting reason is one
    # of the known startup failures (crash loop, image pull problems, invalid image name, container
    # creation or run errors); none if no container is in such a state.
    return next(
        (
            waiting.reason
            for container in pod.status.container_statuses
            if (waiting := container.state.waiting) is not None and waiting.reason in _STARTUP_FAILURE_REASONS
        ),
        None,
    )


def is_pod_scheduling_gated(pod: Pod) -> bool:
    # [FUNCTION DOC] True if any pod condition of type PodScheduled has the reason SchedulingGated.
    return any(
        condition.type == "PodScheduled" and condition.reason == _SCHEDULING_GATED_REASON
        for condition in pod.status.conditions
    )


def _described(pod: Pod) -> str:
    # [FUNCTION DOC] HELPER: Status text of a pod: the phase description, followed by ", restarted N
    # times" when the total restart count is non-zero.
    restarts = f", restarted {count} times" if (count := restarts_of_pod(pod)) else ""
    return f"{_phase_described(pod)}{restarts}"


def _phase_described(pod: Pod) -> str:
    # [FUNCTION DOC] HELPER: Phase text refined with detail: "Pending (scheduling gated)" for gated
    # pods, "Running (not ready yet)" for running pods that are not ready, else the plain phase.
    if is_pod_scheduling_gated(pod):
        return "Pending (scheduling gated)"
    phase = phase_of_pod(pod)
    if phase == "Running" and not is_pod_ready(pod):
        return "Running (not ready yet)"
    return phase
