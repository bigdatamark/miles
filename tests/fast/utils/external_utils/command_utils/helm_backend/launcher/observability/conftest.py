# [MODULE DOC] HELPER: Shared builders for the observability tests of the Helm launcher: factory
# functions that create fake Kubernetes container status, pod and event objects using the worker
# utilities' typed Kubernetes models, and a polling helper for background-thread tests. No
# fixtures are defined despite the module name.
import time

from miles.utils.workers.k8s_types import (
    ContainerState,
    ContainerStateTerminated,
    ContainerStatus,
    Event,
    ObjectReference,
    Pod,
    PodCondition,
    PodMetadata,
    PodStatus,
)


def make_container(name="app", container_id="", previous_container_id="", running=False, restarts=0):
    # [FUNCTION DOC] HELPER: Builds a container status with the given name, container id,
    # restart count, and a current state that is running (an empty running record) only when the
    # running flag is true. The last state carries a terminated record with the previous
    # container id when that id is non-empty, otherwise no terminated record.
    return ContainerStatus(
        name=name,
        container_id=container_id,
        restart_count=restarts,
        state=ContainerState(running={} if running else None),
        last_state=ContainerState(
            terminated=ContainerStateTerminated(container_id=previous_container_id) if previous_container_id else None
        ),
    )


def make_pod(name="p", uid="u", phase="Running", ready=True, restarts=0, scheduling_gated=False, containers=()) -> Pod:
    # [FUNCTION DOC] HELPER: Builds a pod with the given name, uid and phase. Conditions always
    # include a Ready condition (status "True" when ready is true, else "False"); when the
    # scheduling-gated flag is set a PodScheduled condition with status "False" and reason
    # "SchedulingGated" is added. Container statuses are the given containers; if none are given
    # and the restart count is non-zero, one default container with that restart count is
    # created; otherwise the list is empty. Returns the pod.
    conditions = [PodCondition(type="Ready", status="True" if ready else "False")]
    if scheduling_gated:
        conditions.append(PodCondition(type="PodScheduled", status="False", reason="SchedulingGated"))
    containers = containers or ((make_container(restarts=restarts),) if restarts else ())
    return Pod(
        metadata=PodMetadata(name=name, uid=uid),
        status=PodStatus(phase=phase, conditions=conditions, container_statuses=list(containers)),
    )


def make_event(pod_name="p", reason="FailedScheduling", message="no node", count=1, event_type="Warning", uid=None):
    # [FUNCTION DOC] HELPER: Builds a Kubernetes event about a pod: the involved object is a
    # Pod named pod_name with the given uid (default "uid-" followed by the pod name), plus the
    # given reason, message, occurrence count and event type (defaults: failed scheduling warning
    # with the message "no node" and count 1).
    return Event(
        involved_object=ObjectReference(name=pod_name, kind="Pod", uid=uid or f"uid-{pod_name}"),
        reason=reason,
        message=message,
        count=count,
        type=event_type,
    )


def wait_for(condition, timeout: float = 5.0) -> None:
    # [FUNCTION DOC] HELPER: Polls the given zero-argument condition function every 0.01 seconds
    # until it returns a true value or the timeout (default 5 seconds, measured on the monotonic
    # clock) passes. Returns nothing on success; raises an assertion error if the condition never
    # became true.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("the condition the test waited for never happened")
