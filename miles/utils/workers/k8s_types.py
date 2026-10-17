# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC]
# Minimal, immutable, partial models of the Kubernetes objects the reconcile code reads (pods,
# jobs, events, list wrappers and watch frames). Only the needed fields are modeled. Each field
# accepts both snake_case and camel_case spellings, and fields that the API may send as null are
# normalized to empty containers or defaults, so the same models parse REST JSON and client objects.
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import AliasChoices, BeforeValidator, Field

from miles.utils.pydantic_utils import FrozenPartialBaseModel


def _absent_as_empty_map(value: Any) -> Any:
    # [FUNCTION DOC] HELPER: Before-validator turning a missing (null) value into an empty mapping;
    # any other value passes through unchanged.
    return {} if value is None else value


def _absent_as_empty_list(value: Any) -> Any:
    # [FUNCTION DOC] HELPER: Before-validator turning a missing (null) value into an empty list;
    # any other value passes through unchanged.
    return [] if value is None else value


def _absent_as_one(value: Any) -> Any:
    # [FUNCTION DOC] HELPER: Before-validator turning a missing (null) value into the integer one
    # (used for event counts); any other value passes through unchanged.
    return 1 if value is None else value


StringMap = Annotated[dict[str, str], BeforeValidator(_absent_as_empty_map)]


class PodMetadata(FrozenPartialBaseModel):
    # [CLASS DOC]
    # Pod identity and bookkeeping: name, unique id, optional resource version and deletion
    # timestamp (set once the pod is terminating), and label and annotation maps (default empty).
    name: str
    uid: str
    resource_version: str | None = Field(
        default=None, validation_alias=AliasChoices("resource_version", "resourceVersion")
    )
    deletion_timestamp: datetime | None = Field(
        default=None, validation_alias=AliasChoices("deletion_timestamp", "deletionTimestamp")
    )
    labels: StringMap = {}
    annotations: StringMap = {}


class PodSchedulingGate(FrozenPartialBaseModel):
    # [CLASS DOC]
    # One scheduling gate on a pod; a pod with any gate is held back from scheduling. Only the
    # gate name is modeled.
    name: str


class PodSpec(FrozenPartialBaseModel):
    # [CLASS DOC]
    # The subset of a pod spec used here: assigned node name, subdomain, node selector map and the
    # list of scheduling gates (both default empty when absent).
    node_name: str | None = Field(default=None, validation_alias=AliasChoices("node_name", "nodeName"))
    subdomain: str | None = None
    node_selector: StringMap = Field(default={}, validation_alias=AliasChoices("node_selector", "nodeSelector"))
    scheduling_gates: Annotated[list[PodSchedulingGate], BeforeValidator(_absent_as_empty_list)] = Field(
        default=[], validation_alias=AliasChoices("scheduling_gates", "schedulingGates")
    )


class PodCondition(FrozenPartialBaseModel):
    # [CLASS DOC]
    # One pod condition entry: condition type, status string and optional reason.
    type: str
    status: str
    reason: str | None = None


class ContainerStateTerminated(FrozenPartialBaseModel):
    # [CLASS DOC]
    # Terminated container state; only the optional container id is modeled.
    container_id: str | None = Field(default=None, validation_alias=AliasChoices("container_id", "containerID"))


class ContainerStateWaiting(FrozenPartialBaseModel):
    # [CLASS DOC]
    # Waiting container state; only the optional reason is modeled.
    reason: str | None = None


class ContainerStateRunning(FrozenPartialBaseModel):
    # [CLASS DOC]
    # Running container state; only the optional start time is modeled.
    started_at: datetime | None = Field(default=None, validation_alias=AliasChoices("started_at", "startedAt"))


class ContainerState(FrozenPartialBaseModel):
    # [CLASS DOC]
    # Container state: at most one of the waiting, running and terminated parts is expected to be
    # present; each defaults to absent.
    waiting: ContainerStateWaiting | None = None
    running: ContainerStateRunning | None = None
    terminated: ContainerStateTerminated | None = None


class ContainerStatus(FrozenPartialBaseModel):
    # [CLASS DOC]
    # Status of one container in a pod: name (default empty), optional container id, required
    # restart count, current state and last state (a missing state becomes an empty state).
    name: str = ""
    container_id: str | None = Field(default=None, validation_alias=AliasChoices("container_id", "containerID"))
    restart_count: int = Field(validation_alias=AliasChoices("restart_count", "restartCount"))
    state: Annotated[ContainerState, BeforeValidator(_absent_as_empty_map)] = ContainerState()
    last_state: Annotated[ContainerState, BeforeValidator(_absent_as_empty_map)] = Field(
        default=ContainerState(), validation_alias=AliasChoices("last_state", "lastState")
    )


class PodStatus(FrozenPartialBaseModel):
    # [CLASS DOC]
    # Pod status: optional phase and pod ip, list of conditions and list of container statuses
    # (both default empty when absent or null).
    phase: str | None = None
    pod_ip: str | None = Field(default=None, validation_alias=AliasChoices("pod_ip", "podIP"))
    conditions: Annotated[list[PodCondition], BeforeValidator(_absent_as_empty_list)] = []
    container_statuses: Annotated[list[ContainerStatus], BeforeValidator(_absent_as_empty_list)] = Field(
        default=[], validation_alias=AliasChoices("container_statuses", "containerStatuses")
    )


class Pod(FrozenPartialBaseModel):
    # [CLASS DOC]
    # A pod: required metadata, plus spec and status that default to empty ones when missing.
    metadata: PodMetadata
    spec: Annotated[PodSpec, BeforeValidator(_absent_as_empty_map)] = PodSpec()
    status: Annotated[PodStatus, BeforeValidator(_absent_as_empty_map)] = PodStatus()


class PodList(FrozenPartialBaseModel):
    # [CLASS DOC]
    # Result of listing pods: the items (empty if absent or null).
    items: Annotated[list[Pod], BeforeValidator(_absent_as_empty_list)] = []


class JobCondition(FrozenPartialBaseModel):
    # [CLASS DOC]
    # One job condition: type and status strings.
    type: str
    status: str


class JobStatus(FrozenPartialBaseModel):
    # [CLASS DOC]
    # Job status: the list of conditions (empty if absent or null).
    conditions: Annotated[list[JobCondition], BeforeValidator(_absent_as_empty_list)] = []


class Job(FrozenPartialBaseModel):
    # [CLASS DOC]
    # A job, modeled by its status only (empty status when missing).
    status: Annotated[JobStatus, BeforeValidator(_absent_as_empty_map)] = JobStatus()


class ObjectReference(FrozenPartialBaseModel):
    # [CLASS DOC]
    # Reference to a cluster object: optional name, kind and uid.
    name: str | None = None
    kind: str | None = None
    uid: str | None = None


class Event(FrozenPartialBaseModel):
    # [CLASS DOC]
    # A cluster event: the object it involves, optional reason, message and type, and an
    # occurrence count that defaults to one when missing or null.
    involved_object: Annotated[ObjectReference, BeforeValidator(_absent_as_empty_map)] = Field(
        default=ObjectReference(), validation_alias=AliasChoices("involved_object", "involvedObject")
    )
    reason: str | None = None
    message: str | None = None
    count: Annotated[int, BeforeValidator(_absent_as_one)] = 1
    type: str | None = None


class EventList(FrozenPartialBaseModel):
    # [CLASS DOC]
    # Result of listing events: the items (empty if absent or null).
    items: Annotated[list[Event], BeforeValidator(_absent_as_empty_list)] = []


class WatchFrameMetadata(FrozenPartialBaseModel):
    # [CLASS DOC]
    # Metadata of a watch frame; only the optional resource version is modeled.
    resource_version: str | None = Field(
        default=None, validation_alias=AliasChoices("resource_version", "resourceVersion")
    )


class WatchFrame(FrozenPartialBaseModel):
    # [CLASS DOC]
    # A status-style frame from a watch stream: metadata (carrying the resource version), and an
    # optional error code and reason, used to detect expired or failed watches.
    metadata: Annotated[WatchFrameMetadata, BeforeValidator(_absent_as_empty_map)] = WatchFrameMetadata()
    code: int | None = None
    reason: str | None = None
