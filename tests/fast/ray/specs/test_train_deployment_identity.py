# [MODULE DOC]
# Unit test checking that the deployment identity handed to a trainer controller by its spec
# records which trainer entry the controller serves. Reuses the argument and context helpers of
# the trainer spec test module.
from tests.fast.ray.specs.test_train import _controller_context, _controller_providers, _make_args

from miles.ray.specs.train import specs_trainer_controller


class TestTrainerControllerDeploymentIdentity:
    # [CLASS DOC] Tests the deployment identity in the trainer controller constructor kwargs.
    def test_the_identity_records_the_declared_trainer_id(self) -> None:
        # [FUNCTION DOC] For a run with a critic, the second controller spec (the critic one) builds
        # constructor kwargs whose deployment identity has trainer id "critic".
        """A split trainer controller must identify which trainer entry it serves."""
        capability = _controller_providers()
        spec = specs_trainer_controller(_make_args(use_critic=True))[1]

        identity = spec.ctor_kwargs(_controller_context(capability))["deployment_identity"]

        assert identity.trainer_id == "critic"
