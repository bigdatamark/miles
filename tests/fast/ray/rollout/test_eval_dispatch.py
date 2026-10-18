# [MODULE DOC]
# Tests of how the eval dispatcher settles one eval point (the settle step): a crashed eval is
# reported to the rollout manager as a skipped point with reason crashed and, under the CI test flag,
# the failure of that report fails the run; a cleanly finished eval reports nothing; in all cases the
# exported snapshot directory is handed back to the dispatcher's exported list. Registered as a CPU CI
# test in the stage-a-cpu suite with an estimated time of 60 seconds. Uses a fake rollout manager whose
# remote method calls are recorded.
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.eval_dispatch import EvalDispatcher


class FakeRemoteMethod:
    # [CLASS DOC] HELPER: Stand-in for a remote method handle. Calling it records the call and returns
    # an awaitable that is already completed with no result.
    def __init__(self, log: list, name: str):
        # [FUNCTION DOC] HELPER: Stores the shared call log and the method name used in log entries.
        self.log = log
        self.name = name

    def __call__(self, *args):
        # [FUNCTION DOC] HELPER: Appends a pair of method name and positional arguments to the log,
        # creates a future on the current event loop, completes it with no result, and returns it.
        self.log.append((self.name, args))
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(None)
        return fut


class FakeRolloutManager:
    # [CLASS DOC] HELPER: Fake rollout manager exposing only a recorded report-eval-skip remote method.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates an empty shared call log and the report-eval-skip fake bound to
        # it.
        self.log: list = []
        self.report_eval_skip = FakeRemoteMethod(self.log, "report_eval_skip")


def make_dispatcher(**overrides) -> EvalDispatcher:
    # [FUNCTION DOC] HELPER: Builds test args keeping two eval snapshots plus any overrides, and returns
    # an eval dispatcher with no actor model and a fake rollout manager.
    args = make_args(eval_keep_snapshots=2, **overrides)
    return EvalDispatcher(args, actor_model=None, rollout_manager=FakeRolloutManager())


def failed_ref(error: Exception):
    # [FUNCTION DOC] HELPER: Returns a future on the current event loop that is already failed with the
    # given exception, standing in for a crashed eval task reference.
    fut = asyncio.get_event_loop().create_future()
    fut.set_exception(error)
    return fut


class TestSettle:
    # [CLASS DOC] Checks the settle step of the dispatcher for crashed and clean eval tasks.
    async def test_a_crashed_eval_fails_when_reporting_the_ci_skip(self, tmp_path):
        # [FUNCTION DOC] Under the CI test flag, replaces the manager's report-eval-skip with a fake
        # whose returned future fails with a runtime error saying the CI eval was skipped. Settling
        # eval point 3 with a failed task reference must propagate that error. Asserts the fake was
        # called once with point 3 and reason crashed, and that the dispatcher's exported list holds the
        # exported directory (the snapshot is still retired).
        """A crashed eval fails CI through the centralized skipped-point contract."""
        dispatcher = make_dispatcher(ci_test=True)
        exported_dir = str(tmp_path / "step_3")

        class FailingRemoteMethod(FakeRemoteMethod):
            # [CLASS DOC] HELPER: Remote method fake whose returned future is failed instead of
            # completed.
            def __call__(self, *args):
                # [FUNCTION DOC] HELPER: Records the call like the base fake, then returns a future
                # failed with a runtime error whose message says CI eval 3 was skipped as crashed.
                self.log.append((self.name, args))
                fut = asyncio.get_event_loop().create_future()
                fut.set_exception(RuntimeError("CI eval 3 skipped: crashed"))
                return fut

        dispatcher.rollout_manager.report_eval_skip = FailingRemoteMethod(
            dispatcher.rollout_manager.log, "report_eval_skip"
        )

        with pytest.raises(RuntimeError, match="CI eval 3 skipped: crashed"):
            await dispatcher._settle(3, failed_ref(RuntimeError("engine gone")), exported_dir)

        assert dispatcher.rollout_manager.log == [("report_eval_skip", (3, "crashed"))]
        assert dispatcher._exported == [exported_dir]

    async def test_a_crashed_eval_degrades_to_a_skipped_point_outside_ci(self, tmp_path):
        # [FUNCTION DOC] Without the CI test flag, settling point 3 with a failed task reference must
        # not raise. Asserts the manager was told about a skipped point 3 with reason crashed and the
        # exported directory is in the dispatcher's exported list.
        """Without --ci-test a raised eval only logs a crashed skip and training continues."""
        dispatcher = make_dispatcher(ci_test=False)
        exported_dir = str(tmp_path / "step_3")

        await dispatcher._settle(3, failed_ref(RuntimeError("engine gone")), exported_dir)

        assert dispatcher.rollout_manager.log == [("report_eval_skip", (3, "crashed"))]
        assert dispatcher._exported == [exported_dir]

    async def test_a_settled_eval_reports_nothing_and_retires_its_snapshot(self, tmp_path):
        # [FUNCTION DOC] Under the CI test flag, settles point 3 with a task reference already completed
        # successfully. Asserts nothing was reported to the manager, nothing was raised, and the exported
        # directory is in the dispatcher's exported list.
        """A point that finished cleanly is neither reported as skipped nor re-raised."""
        dispatcher = make_dispatcher(ci_test=True)
        exported_dir = str(tmp_path / "step_3")
        ref = asyncio.get_event_loop().create_future()
        ref.set_result(None)

        await dispatcher._settle(3, ref, exported_dir)

        assert dispatcher.rollout_manager.log == []
        assert dispatcher._exported == [exported_dir]
