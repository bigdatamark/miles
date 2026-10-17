# [MODULE DOC] Immutable, strictly validated identity records for the processes that emit audit
# events, with a readable name for each. Records are distinguished by the component field (a
# tagged union), exported as the ProcessIdentity alias. Variants: simple singleton components,
# trainer controllers (named by trainer id) and training worker processes (named by model id,
# cell index and rank within the cell).

from typing import Annotated, Literal

from pydantic import Discriminator, NonNegativeInt

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.naming import format_name_index


class _ProcessIdentityBase(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: Base of all identity variants; holds the component string and a default
    # name equal to it.
    component: str

    def to_name(self) -> str:
        # [FUNCTION DOC] Returns the component string as the process name.
        return self.component


class SimpleProcessIdentity(_ProcessIdentityBase):
    # [CLASS DOC] Identity of a singleton process; the component must be one of: main,
    # rollout_executor, inference_controller, multi_lora_controller, worker_manager,
    # registration_reporter. The name is the component.
    component: Literal[
        "main",
        "rollout_executor",
        "inference_controller",
        "multi_lora_controller",
        "worker_manager",
        "registration_reporter",
    ]


class TrainerControllerProcessIdentity(_ProcessIdentityBase):
    # [CLASS DOC] Identity of a trainer controller; the component defaults to trainer_controller.
    # Fields: trainer id (required) and an optional model id.
    component: Literal["trainer_controller"] = "trainer_controller"
    trainer_id: str
    model_id: str | None = None

    def to_name(self) -> str:
        # [FUNCTION DOC] Returns the component, an underscore and the trainer id. The model id is
        # not part of the name.
        return f"{self.component}_{self.trainer_id}"


class TrainProcessIdentity(_ProcessIdentityBase):
    # [CLASS DOC] Identity of one training worker rank; the component is actor or critic. Fields:
    # optional model id, non-negative cell index and non-negative rank within the cell.
    component: Literal["actor", "critic"]
    model_id: str | None = None
    cell_index: NonNegativeInt
    rank_within_cell: NonNegativeInt

    def to_name(self) -> str:
        # [FUNCTION DOC] Builds the name as: the model id and an underscore (only if a non-empty
        # model id is set), the component, "_cell" plus the formatted cell index, and "_rank" plus
        # the formatted rank within the cell. Index formatting is delegated to the shared naming
        # helper.
        return (
            f"{f'{x}_' if (x := self.model_id) else ''}{self.component}"
            f"_cell{format_name_index(self.cell_index)}_rank{format_name_index(self.rank_within_cell)}"
        )


ProcessIdentity = Annotated[
    SimpleProcessIdentity | TrainerControllerProcessIdentity | TrainProcessIdentity,
    Discriminator("component"),
]
