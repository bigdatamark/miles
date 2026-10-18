# [MODULE DOC]
# The http protocol between rpc client and server: header names, the status code for a boot uuid
# mismatch (412), url path templates (health, in-flight listing, call status by id, submit by
# method name), long-poll timeout defaults (30 seconds) and cap (300 seconds), and the strict
# request and response models.
from __future__ import annotations

from typing import Any, Literal

from miles.utils.pydantic_utils import StrictBaseModel

EXPECTED_BOOT_UUID_HEADER = "x-miles-expected-boot-uuid"
BOOT_UUID_HEADER = "x-miles-boot-uuid"

BOOT_UUID_MISMATCH_STATUS = 412

HEALTH_PATH = "/v1/health"
IN_FLIGHT_PATH = "/v1/in-flight"
CALL_STATUS_PATH = "/v1/calls/{call_id}"
SUBMIT_PATH = "/v1/{method_name}"

DEFAULT_POLL_TIMEOUT_SECONDS = 30.0

MAX_POLL_TIMEOUT_SECONDS = 300.0


class SubmitRequest(StrictBaseModel):
    # [CLASS DOC] Body of a submit: the client-chosen call id and the encoded arguments mapping.
    call_id: str
    query: dict[str, Any]


class SubmitResponse(StrictBaseModel):
    # [CLASS DOC] Acknowledgement of a submit; the status is always the text submitted.
    status: Literal["submitted"] = "submitted"


class CallStatusResponse(StrictBaseModel):
    # [CLASS DOC] State of a call: pending, success (with the encoded result) or failed (with the
    # error text).
    status: Literal["pending", "success", "failed"]
    result: Any = None
    error: str | None = None


class HealthResponse(StrictBaseModel):
    # [CLASS DOC] Health answer; the status is always ok.
    status: Literal["ok"] = "ok"


class InFlightResponse(StrictBaseModel):
    # [CLASS DOC] The ids of calls currently running on the server.
    call_ids: list[str]
