# [MODULE DOC]
# Wire models for cell registration, shared by reporters (senders) and the hub (receiver). Both
# are immutable and reject unknown fields.
from __future__ import annotations


from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo


class RegisteredCellInfo(FrozenStrictBaseModel):
    # [CLASS DOC]
    # One cell as announced by a reporter: the id of the reporter that owns it, the cell info, and
    # the info of each of its workers (name, endpoints, and so on).
    reporter_id: str
    info: CellInfo
    workers: list[WorkerInfo]


class RegistrationSnapshot(FrozenStrictBaseModel):
    # [CLASS DOC]
    # The complete set of cells one reporter currently announces. Carries the run uuid (must match
    # the hub's run), the reporter id, a sequence number that increases with each snapshot of that
    # reporter (used by the hub to discard stale ones) and the list of cells. A snapshot replaces
    # everything the reporter announced before.
    run_uuid: str
    reporter_id: str
    sequence_number: int
    cells: list[RegisteredCellInfo]
