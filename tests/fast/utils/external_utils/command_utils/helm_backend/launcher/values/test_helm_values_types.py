# [MODULE DOC] Tests for the infra values model of the helm launcher: validation of the shared
# memory (dev shm) volume declaration. Only one case is covered: the model must reject a dev shm
# entry that declares no source or two competing sources.
import pytest
from pydantic import ValidationError

from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import InfraValues


class TestDevShm:
    # [CLASS DOC] Validation tests for the dev shm field of the infra values.
    @pytest.mark.parametrize(
        "dev_shm",
        [
            {"mountPath": "/dev/shm"},
            {
                "mountPath": "/dev/shm",
                "hostPath": {"path": "/dev/shm"},
                "emptyDir": {"medium": "Memory"},
            },
        ],
        ids=["without-a-source", "with-two-sources"],
    )
    def test_dev_shm_requires_exactly_one_source(self, dev_shm: dict[str, object]) -> None:
        # [FUNCTION DOC] Parametrized over a dev shm with only a mount path (no source) and one
        # with both a host path and an in-memory empty dir source. Validating infra values with
        # an image, no volumes and that dev shm must raise a validation error whose message says
        # exactly one source is required.
        """Shared memory must declare one source rather than none or two competing sources."""
        values = {
            "image": {"repository": "radixark/miles", "tag": "dev"},
            "volumes": [],
            "devShm": dev_shm,
        }

        with pytest.raises(ValidationError, match="exactly one"):
            InfraValues.model_validate(values)
