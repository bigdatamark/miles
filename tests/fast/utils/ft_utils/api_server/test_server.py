# [MODULE DOC] Test for the fault tolerance api server startup: when the rollout component is
# enabled but no local inference controller is given, starting the server must fail with an
# assertion error before anything starts. Includes a fake cell operations object whose methods
# do nothing.
from types import SimpleNamespace

import pytest

from miles.utils.ft_utils.api_server import server
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.worker_provider.base import CellInfo


class FakeCellOperations:
    # [CLASS DOC] HELPER: Inert stand-in for cell operations; every method is an async no-op.
    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        # [FUNCTION DOC] HELPER: Returns an empty mapping regardless of the pool ids.
        return {}

    async def suspend(self, *, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        pass

    async def resume(self, *, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        pass

    async def inject_fault(self, *, cell_id: str, mode: FailureMode, sub_index: int) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        pass


class TestStartApiServer:
    # [CLASS DOC] Test of api server startup validation.
    def test_rollout_ft_requires_a_local_inference_controller(self) -> None:
        # [FUNCTION DOC] Starting with empty args, no trainer models, no inference controller,
        # port 1234, the rollout fault tolerance component and fake cell operations raises an
        # assertion error saying rollout cells are suspended and resumed through the inference
        # controller.
        """Rollout fault tolerance fails before startup when no local inference controller exists."""

        with pytest.raises(
            AssertionError,
            match="rollout cells are suspended and resumed through the inference controller",
        ):
            server.start_api_server(
                args=SimpleNamespace(),
                trainer_models={},
                inference_controller=None,
                port=1234,
                ft_components=["rollout"],
                cell_operations=FakeCellOperations(),
            )
