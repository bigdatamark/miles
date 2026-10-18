# [MODULE DOC] Tests for the audit event models (miles.utils.audit_utils.event_logger.models),
# currently the validation of the inference-engine weight checksum event.
import pytest
from pydantic import ValidationError

from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent


class TestInferenceEngineWeightChecksumEvent:
    # [CLASS DOC] Validation tests of the weight checksum event model.
    def test_none_rollout_id_is_rejected(self) -> None:
        # [FUNCTION DOC] Validates an event payload (timestamp, source component "main", engine
        # checksums with one rank-to-hash mapping) whose rollout_id is null, and asserts a model
        # validation error whose message names the rollout_id field.
        """A checksum event rejects the former null startup rollout identifier."""
        data = {
            "timestamp": "2026-01-01T00:00:00Z",
            "source": {"component": "main"},
            "rollout_id": None,
            "engine_checksums": [{"rank0/embed.weight": "aaa"}],
        }

        with pytest.raises(ValidationError, match="rollout_id"):
            InferenceEngineWeightChecksumEvent.model_validate(data)
