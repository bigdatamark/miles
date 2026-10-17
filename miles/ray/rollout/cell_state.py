# [MODULE DOC] Immutable state models for one inference cell's lifecycle, plus a union type of all
# states. Lifecycle: uninitialized -> initializing -> pending weights -> serving -> disposed.
# All models are frozen and reject unknown fields.
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class CellAddrInfo(FrozenStrictBaseModel):
    # [CLASS DOC] Network addresses of a cell's engine: the server URL, an optional bootstrap port
    # (used for prefill/decode style bootstrap; may be nothing) and an optional gate URL.
    server_url: str
    bootstrap_port: int | None
    gate_url: str | None


class StateUninitialized(FrozenStrictBaseModel):
    # [CLASS DOC] Cell has not been started; carries no data.
    pass


class StateInitializing(FrozenStrictBaseModel):
    # [CLASS DOC] Cell engine has been launched and is starting up; holds its addresses and the
    # start time as a floating point timestamp (used to detect startup timeouts).
    addr_info: CellAddrInfo
    start_time: float


class StatePendingWeights(FrozenStrictBaseModel):
    # [CLASS DOC] Engine is up but has not yet received current weights, so it must not serve.
    addr_info: CellAddrInfo


class StateServing(FrozenStrictBaseModel):
    # [CLASS DOC] Engine is up with weights loaded and can serve requests.
    addr_info: CellAddrInfo


class StateDisposed(FrozenStrictBaseModel):
    # [CLASS DOC] Cell has been torn down permanently; carries no data.
    pass


# Type alias for any of the five lifecycle state models above.
CellState = StateUninitialized | StateInitializing | StatePendingWeights | StateServing | StateDisposed
