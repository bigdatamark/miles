# [MODULE DOC] Tests for the workbench release naming helper: a release name built for a run id,
# deploy component and instance id can be parsed back by the helm backend's release name model
# into the same identity.
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.external_utils.miles_workbench.naming import run_release_name
from miles.utils.workers.types import DeployComponent


class TestRunReleaseName:
    # [CLASS DOC] Test of the run release name function.
    def test_run_release_name_preserves_the_component_and_instance_id(self) -> None:
        # [FUNCTION DOC] Builds a release for run demo, inference component and instance east,
        # parses it back, and asserts the parse is not none and the run id, component and instance
        # id all equal the inputs.
        """A release generated for an instance parses back to the same deployment identity."""
        release = run_release_name(
            run_id="demo", deploy_component=DeployComponent.INFERENCE, deploy_instance_id="east"
        )

        parsed = ReleaseName.parse(release)

        assert parsed is not None
        assert parsed.run_id == "demo"
        assert parsed.deploy_component is DeployComponent.INFERENCE
        assert parsed.deploy_instance_id == "east"
