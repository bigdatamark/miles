# [MODULE DOC]
# Unit test that a trainer controller reports back, unchanged, the deployment identity it was
# constructed with. Its dependencies (cell provider and cell operations) are specification-bound
# mocks, so no cluster or ray is involved.
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from miles.ray.train.group import TrainerController
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.types import DeploymentIdentity
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

pytestmark = pytest.mark.asyncio


def _controller(identity: DeploymentIdentity) -> TrainerController:
    # [FUNCTION DOC] HELPER: Builds an actor-role trainer controller with the given identity, mock
    # cell provider and cell operations restricted to their base classes, and no reference model.
    return TrainerController(
        deployment_identity=identity,
        cell_provider=MagicMock(spec=BaseWorkerProvider),
        cell_operations=MagicMock(spec=BaseCellOperations),
        trainer_id="actor",
        role="actor",
        with_ref=False,
    )


class TestDeploymentIdentity:
    # [CLASS DOC] Tests the identity getter of the trainer controller.
    async def test_the_identity_names_the_launch_this_controller_was_started_by(self):
        # [FUNCTION DOC] A controller built with an identity (run uuid 0123456789abcdef, trainer
        # deploy component) returns that very identity object from its identity query.
        """A trainer answers for the deployment that launched it, not for the arguments a script hands it later."""
        identity = DeploymentIdentity(run_uuid="0123456789abcdef", deploy_component="trainer")

        assert await _controller(identity).get_deployment_identity() is identity
