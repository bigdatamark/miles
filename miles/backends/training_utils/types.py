# [MODULE DOC]
# Small shared result types for a training step: an outcome enum and an immutable output record.
from dataclasses import dataclass
from enum import auto

from miles.utils.object_store import StoreObjectRef

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


class TrainStepOutcome(StrEnum):
    # [CLASS DOC] String enum of how a training step ended: normal (completed), or discarded and
    # should be retried (the step's results were thrown away, e.g. after a fault).
    NORMAL = auto()
    DISCARDED_SHOULD_RETRY = auto()


@dataclass(frozen=True)
class TrainStepOutput:
    # [CLASS DOC] Immutable result of a training step: the outcome, and an optional object store
    # reference to produced values (such as critic values handed to another role; default none).
    outcome: TrainStepOutcome
    values: StoreObjectRef | None = None
