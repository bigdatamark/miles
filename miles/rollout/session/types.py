# [MODULE DOC]
# Pydantic data models of the session server API: the create-session request body, the published
# description of a session-server instance, a recorded turn, and the get-session reply.
from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, field_validator, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel, StrictBaseModel


class CreateSessionRequest(StrictBaseModel):
    # [CLASS DOC]
    # Body of the create-session call (strict, extra fields rejected). Fields: evaluation flag
    # (strict boolean, default false); optional sampling defaults temperature, top-p (strict
    # floats) and top-k (strict integer) that the session applies to requests omitting them.
    evaluation: StrictBool = False
    # the caller resolves rollout/eval/dataset values; the session only fills fields a request omits
    temperature: StrictFloat | None = None
    top_p: StrictFloat | None = None
    top_k: StrictInt | None = None

    @field_validator("top_k", mode="before")
    @classmethod
    def _integral_float_as_int(cls, value):
        # [FUNCTION DOC] HELPER: Before-validation hook for top-k: converts a float with an
        # integral value (such as 40.0 from YAML) into an integer; other values, including
        # fractional floats, pass through unchanged (and then fail strict integer validation).
        # an eval dataset YAML may spell top_k as 40.0; a fractional top_k stays an error
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value


class SessionServerInstance(FrozenStrictBaseModel):
    # [CLASS DOC]
    # Immutable record of a session server as published by the driver: the address the driver
    # dials (host and port), the address external peers dial (defaults to the driver address when
    # empty or missing), and an optional instance id. Provides http urls for both addresses.
    """One session-server instance as the driver published it."""

    # ``host:port`` the driver dials.
    addr: str
    # ``host:port`` a peer outside the cluster dials; defaults to ``addr`` (filled in below).
    external_addr: str = ""
    instance_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_external_addr_to_addr(cls, values: dict) -> dict:
        # [FUNCTION DOC] HELPER: Before-validation hook: if the raw input is a mapping whose
        # external address is missing or empty, return a copy with it set to the address value
        # (which may itself be missing, leaving validation to fail); otherwise return unchanged.
        if isinstance(values, dict) and not values.get("external_addr"):
            values = {**values, "external_addr": values.get("addr")}
        return values

    @property
    def url(self) -> str:
        # [FUNCTION DOC] Property: plain-http url of the driver-facing address.
        return f"http://{self.addr}"

    @property
    def external_url(self) -> str:
        # [FUNCTION DOC] Property: plain-http url of the external address.
        return f"http://{self.external_addr}"


class SessionRecord(BaseModel):
    # [CLASS DOC]
    # One recorded backend exchange of a session: completion time, optional request start time,
    # HTTP method, request path, the request body mapping sent to the backend (includes token
    # ids), the full response mapping, and the HTTP status code.
    timestamp: float
    request_timestamp: float | None = None
    method: str
    path: str
    request: dict
    response: dict
    status_code: int


class GetSessionResponse(BaseModel):
    # [CLASS DOC] Reply of the get-session call: session id, the list of recorded turns, and a
    # free-form metadata mapping (default empty).
    session_id: str
    records: list[SessionRecord]
    metadata: dict = Field(default_factory=dict)
