# [MODULE DOC]
# The orchestrator state file: a small json document written by the orchestrator wrapper pod and
# read by the launcher. It records whether the orchestration script started or exited, the exit
# code (mandatory when exited) and a timestamp. Writes are atomic; reads tolerate missing or
# corrupt files. Also defines the command line flag name used to pass the state file path.
from __future__ import annotations

import time
from enum import auto
from pathlib import Path

from pydantic import Field, ValidationError, model_validator

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from miles.utils.file_utils import atomic_write_text
from miles.utils.pydantic_utils import FrozenStrictBaseModel

STATE_FILE_FLAG = "--state-file"


class OrchestratorStatus(StrEnum):
    # [CLASS DOC] Lifecycle of the orchestration script: started (running) or exited (verdict
    # known). Values are the lower case member names.
    STARTED = auto()
    EXITED = auto()


class OrchestratorState(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable state record: status, optional exit code and a unix timestamp (seconds)
    # defaulting to the creation time. Validation requires an exit code when the status is exited.
    status: OrchestratorStatus
    exit_code: int | None = None
    timestamp: float = Field(default_factory=time.time)

    @property
    def is_terminal(self) -> bool:
        # [FUNCTION DOC] Property: true when the status is exited.
        return self.status is OrchestratorStatus.EXITED

    @model_validator(mode="after")
    def _check_exit_code(self) -> OrchestratorState:
        # [FUNCTION DOC] HELPER: After-validator asserting that an exited state carries an exit
        # code; returns the model.
        assert self.status is not OrchestratorStatus.EXITED or self.exit_code is not None, (
            f"A terminal state is the verdict of the whole run, so it has to carry the exit code the launcher "
            f"passes back to its caller, but {self} names none"
        )
        return self

    @classmethod
    def read(cls, path: str | Path) -> OrchestratorState | None:
        # [FUNCTION DOC] Reads and parses the state file. Returns none if the file does not exist or
        # its content fails validation (invalid json, wrong fields, or an exited state without exit
        # code). Other I/O errors propagate.
        try:
            payload = Path(path).read_text()
        except FileNotFoundError:
            return None

        try:
            return cls.model_validate_json(payload)
        except ValidationError:
            return None

    def write(self, path: str | Path) -> None:
        # [FUNCTION DOC] Creates parent directories as needed and atomically writes the json dump of
        # the state to the path (so readers never see partial content).
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, self.model_dump_json())
