# [MODULE DOC] Unit test for the rpc protocol response models: checks that each response model
# (submit, call status, health) refuses a status value outside its declared set of literals.
import pytest
from pydantic import ValidationError

from miles.utils.workers.rpc.common.protocol import CallStatusResponse, HealthResponse, SubmitResponse


@pytest.mark.parametrize("response_model", [SubmitResponse, CallStatusResponse, HealthResponse])
def test_response_models_reject_unknown_status_values(
    response_model: type[SubmitResponse] | type[CallStatusResponse] | type[HealthResponse],
) -> None:
    # [FUNCTION DOC] Parametrized over the three response models. Validating a mapping whose status is
    # the string "unknown" must raise a pydantic validation error for each model.
    """Every response model rejects status values outside its protocol literal."""
    with pytest.raises(ValidationError):
        response_model.model_validate({"status": "unknown"})
