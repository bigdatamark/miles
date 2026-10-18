# [MODULE DOC] Tests for strict validation of the WorkerInfo model: unknown fields are rejected
# and the worker class field must be a string (an import path), not a live object.
from typing import Any

import pytest
from pydantic import ValidationError

from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_spec import HostAndPort


def _fields(**overrides: Any) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns valid constructor fields for a worker info (name engine-0-0,
    # generation 1, one primary address 10.0.0.1:30000, gpu ids [0]) merged with the overrides,
    # which win.
    return (
        dict(
            name="engine-0-0",
            generation=1,
            self_addrs={"primary": HostAndPort(host="10.0.0.1", port=30000)},
            gpu_ids=[0],
        )
        | overrides
    )


class TestWorkerInfoValidation:
    # [CLASS DOC] Tests of validation errors raised by the worker info model.
    def test_an_unknown_field_is_rejected(self):
        # [FUNCTION DOC] Passing a misspelled gpu_id field raises a validation error mentioning
        # gpu_id.
        """A misspelled field must fail loudly instead of silently dropping out of the description."""
        with pytest.raises(ValidationError, match="gpu_id"):
            WorkerInfo(**_fields(gpu_id=[0]))

    def test_a_non_string_worker_class_is_rejected(self):
        # [FUNCTION DOC] Passing a plain object as the worker class raises a validation error
        # mentioning worker_class.
        """The class crosses the wire as an import path, so a live object must not pass for it."""
        with pytest.raises(ValidationError, match="worker_class"):
            WorkerInfo(**_fields(worker_class=object()))
