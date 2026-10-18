# [MODULE DOC]
# Unit tests for the rollout executor's init lifecycle: train-only eval resolves session server
# addresses inside the executor process, a freshly constructed executor reports not initialized,
# one that ran init reports initialized, and a second init is refused. Network address resolution,
# session server readiness and tracking init are replaced by monkeypatched stand-ins. All tests in
# the module are marked as asynchronous.
from unittest.mock import MagicMock

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import rollout_executor as executor_module
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.utils.init_once import InitOnce, InitState

pytestmark = pytest.mark.asyncio


def _inited_executor() -> RolloutExecutor:
    # [FUNCTION DOC] HELPER: Creates an init-once guard named after the executor and enters its
    # guarding block once (so the guard counts as initialized), then builds an executor without its
    # constructor and attaches that guard. Returns the executor.
    guard = InitOnce("RolloutExecutor")
    with guard.guarding():
        pass
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor._init_once = guard
    return executor


class TestInitRunsExactlyOnce:
    # [CLASS DOC] Tests the executor's init-once guard and the init sequence's address resolution.
    async def test_train_only_eval_resolves_addresses_in_the_executor_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Constructs a real executor for a train-only run with one eval GPU and the
        # session server enabled. Patches router address resolution (returns empty), session server
        # readiness (writes a session server address list onto the arguments it is given) and
        # tracking init (raises a private stop exception). Runs init, expects the stop exception
        # (so the test ends right after address resolution), and asserts the executor's own
        # arguments object now holds the resolved session server address.
        """The executor resolves session addresses on its own copy of train-only evaluation args."""
        args = make_args(debug_train_only=True, eval_num_gpus=1, use_session_server=True)
        executor = RolloutExecutor(
            args=args,
            router_providers=[],
            session_server_provider=MagicMock(),
            inference_controller_provider=MagicMock(),
        )

        async def resolve_router(args, *, router_providers) -> dict:
            # [FUNCTION DOC] HELPER: Replacement router address resolver; returns an empty mapping.
            return {}

        async def resolve_session(args, *, provider) -> None:
            # [FUNCTION DOC] HELPER: Replacement session server wait; sets the session server
            # address list on the given arguments to a single fake address "eval-session:5000".
            args.session_server_addrs = ["eval-session:5000"]

        class StopAfterAddressResolution(Exception):
            # [CLASS DOC] HELPER: Marker exception used to end init early.
            pass

        def stop_tracking(*args, **kwargs) -> None:
            # [FUNCTION DOC] HELPER: Replacement tracking init; always raises the marker exception.
            raise StopAfterAddressResolution

        monkeypatch.setattr(executor_module, "resolve_router_addrs", resolve_router)
        monkeypatch.setattr(executor_module, "wait_session_server_ready", resolve_session)
        monkeypatch.setattr(executor_module, "init_tracking", stop_tracking)

        with pytest.raises(StopAfterAddressResolution):
            await executor.init()

        assert args.session_server_addrs == ["eval-session:5000"]

    async def test_a_constructed_executor_reports_itself_uninitialized(self):
        # [FUNCTION DOC] A really constructed executor (train-only, no session server) reports the
        # not-initialized state value from its get-init-state method.
        """The constructor the run really uses is what has to leave the guard clear."""
        executor = RolloutExecutor(
            args=make_args(debug_train_only=True),
            router_providers=[],
            session_server_provider=None,
            inference_controller_provider=MagicMock(),
        )

        assert await executor.get_init_state() == InitState.NOT_INITED.value

    async def test_an_executor_that_ran_init_reports_itself_initialized(self):
        # [FUNCTION DOC] An executor with an entered guard reports the initialized state value.
        """The wait at the start of the rollout components only ends once this answer flips back."""
        assert await _inited_executor().get_init_state() == InitState.INITED.value

    async def test_a_second_init_is_refused(self):
        # [FUNCTION DOC] Calling init on an executor whose guard was already entered raises an
        # assertion error mentioning a stale worker.
        """An executor process the previous script initialized is about to be replaced, not re-initialized."""
        with pytest.raises(AssertionError, match="stale worker"):
            await _inited_executor().init()
