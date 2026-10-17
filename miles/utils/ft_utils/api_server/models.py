# [MODULE DOC]
# Wire models of the fault tolerance api server, shaped like kubernetes resources: cell (metadata,
# spec, status), cell list, patch bodies, a fault injection request, a status error body and an
# ok response. Strict models (extra fields rejected). Field names like api version use camel case
# to match the wire format.
from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum
from typing import Literal

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.test_utils.fault_injector import FailureMode


class TriState(StrEnum):
    # [CLASS DOC] String enumeration for condition status with values "True", "False", "Unknown".
    """K8s condition status: ``"True"``, ``"False"``, or ``"Unknown"``."""

    TRUE = "True"
    FALSE = "False"
    UNKNOWN = "Unknown"


class _OkResponse(StrictBaseModel):
    # [CLASS DOC] HELPER: Generic success body with status text "ok".
    status: str = "ok"


class CellCondition(StrictBaseModel):
    # [CLASS DOC] A cell condition: type (Allocated, Healthy or Serving), tri-state status, and
    # optional reason, message and last transition time (text). Constructors below build the
    # common conditions.
    type: Literal["Allocated", "Healthy", "Serving"]
    status: TriState
    reason: str | None = None
    message: str | None = None
    lastTransitionTime: str | None = None

    @classmethod
    def allocated(cls, status: TriState) -> CellCondition:
        # [FUNCTION DOC] Condition of type Allocated with the given status.
        return cls(type="Allocated", status=status)

    @classmethod
    def healthy(cls, status: TriState, *, reason: str | None = None) -> CellCondition:
        # [FUNCTION DOC] Condition of type Healthy with the given status and optional reason.
        return cls(type="Healthy", status=status, reason=reason)

    @classmethod
    def serving(cls, status: TriState) -> CellCondition:
        # [FUNCTION DOC] Condition of type Serving with the given status.
        return cls(type="Serving", status=status)

    @classmethod
    def from_health_checker_status(cls, status: TriState) -> CellCondition:
        # [FUNCTION DOC] Health condition from a health checker result: false gets reason
        # "HealthCheckFailed", unknown gets "HealthCheckUnknown", true has no reason.
        match status:
            case TriState.FALSE:
                return cls.healthy(TriState.FALSE, reason="HealthCheckFailed")
            case TriState.UNKNOWN:
                return cls.healthy(TriState.UNKNOWN, reason="HealthCheckUnknown")
            case TriState.TRUE:
                return cls.healthy(TriState.TRUE)


class CellStatus(StrictBaseModel):
    # [CLASS DOC] Observed cell state: phase (Pending, Running or Suspended), list of conditions,
    # and the hash identifying the workers generation the status refers to.
    phase: Literal["Pending", "Running", "Suspended"]
    conditions: list[CellCondition]
    workers_hash: str


class CellSpec(StrictBaseModel):
    # [CLASS DOC] Desired cell state: whether the cell is suspended (default false).
    suspend: bool = False


class CellMetadata(StrictBaseModel):
    # [CLASS DOC] Cell identity: name and string labels.
    name: str
    labels: dict[str, str]


class Cell(StrictBaseModel):
    # [CLASS DOC] Cell resource with fixed api version "miles.io/v1" and kind "Cell", plus metadata,
    # spec and status.
    apiVersion: Literal["miles.io/v1"] = "miles.io/v1"
    kind: Literal["Cell"] = "Cell"
    metadata: CellMetadata
    spec: CellSpec
    status: CellStatus


class CellList(StrictBaseModel):
    # [CLASS DOC] List resource (kind "CellList") holding cell items.
    apiVersion: Literal["miles.io/v1"] = "miles.io/v1"
    kind: Literal["CellList"] = "CellList"
    items: list[Cell]


class CellPatchSpec(StrictBaseModel):
    # [CLASS DOC] Patchable spec: optional suspend flag (none means leave unchanged).
    suspend: bool | None = None


class CellPatch(StrictBaseModel):
    # [CLASS DOC] Patch body: optional spec patch.
    spec: CellPatchSpec | None = None


class FaultInjection(StrictBaseModel):
    # [CLASS DOC] Fault injection request: failure mode and index of the target sub-component within
    # the cell (default 0).
    mode: FailureMode
    sub_index: int = 0


class K8sStatus(StrictBaseModel):
    # [CLASS DOC] Error body in kubernetes Status style: fixed version "v1", kind "Status", status
    # "Failure", plus message, machine reason and numeric http code.
    apiVersion: Literal["v1"] = "v1"
    kind: Literal["Status"] = "Status"
    status: Literal["Failure"] = "Failure"
    message: str
    reason: str
    code: int
