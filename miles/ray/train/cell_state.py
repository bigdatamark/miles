# [MODULE DOC] Immutable state models for a trainer cell. All allocated states hold the worker
# handles (rank order); states differ in whether the cell is initialized and whether it errored. The
# union type CellState lists the three states.
from pydantic import BaseModel, ConfigDict

from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.workers.worker_handle import BaseWorkerHandle


class StateBase(BaseModel):
    # [CLASS DOC] Base of all states: frozen (hashing/mutation disallowed), forbids unknown fields and
    # allows arbitrary (non-pydantic) field types such as worker handles.
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class StateAllocatedBase(StateBase):
    # [CLASS DOC] Common base of states where workers are allocated: the list of worker handles in
    # rank order.
    worker_handles: list[BaseWorkerHandle]


class StateAllocatedUninitialized(StateAllocatedBase):
    # [CLASS DOC] Workers exist but the cell's init has not completed successfully; no extra data.
    pass


class StateAllocatedAlive(StateAllocatedBase):
    # [CLASS DOC] Initialized and serving; records the current independent data parallel info.
    indep_dp_info: IndepDPInfo


# TODO may remove this state
class StateAllocatedErrored(StateAllocatedBase):
    # [CLASS DOC] A call to the workers failed; workers are being or have been killed. Records the
    # last independent DP info, or none if the cell never became alive.
    indep_dp_info: IndepDPInfo | None


CellState = StateAllocatedUninitialized | StateAllocatedAlive | StateAllocatedErrored
