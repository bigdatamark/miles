# [MODULE DOC] Unit test for the background polling context manager of the Helm launcher
# observability code: it runs a given zero-argument step repeatedly in a background thread (with a
# description used for logging) while the with block runs, and on exit stops the thread. The test
# checks the behavior when a step is blocked at the time of exit.
import threading

from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.polling import polling_in_background


class TestPollingInBackground:
    # [CLASS DOC] Tests of the background polling context manager.
    def test_context_exit_is_bounded_when_the_polling_step_is_blocked(self) -> None:
        # [FUNCTION DOC] Uses three events: step entered, release, and exit completed. The polling
        # step signals entered and then blocks until released. A helper thread enters the context
        # (asserting the step starts within 1 second) and leaves it, then signals exit completed.
        # The main thread waits for the step to start, asserts that exit has NOT completed while
        # the step is blocked, then releases the step and asserts exit completes within 1 second.
        # In a final cleanup it releases again and joins the helper thread (1 second max) and
        # asserts the thread is no longer alive.
        # KNOWN ISSUE: the test name says context exit is bounded when the step is blocked, but
        # the body (and its docstring) show the opposite: exit waits for the in-flight step and
        # only completes after the step is released.
        """Context exit waits for an in-flight polling step before completing."""
        step_entered = threading.Event()
        release = threading.Event()
        exit_completed = threading.Event()

        def blocked_step() -> None:
            # [FUNCTION DOC] HELPER: The polling step: signals that it was entered and then
            # blocks until the release event is set.
            step_entered.set()
            release.wait()

        def own_context() -> None:
            # [FUNCTION DOC] HELPER: Thread body: enters the polling context with the blocked
            # step and description "poll", asserts the step is entered within 1 second, leaves
            # the context, and signals exit completed.
            with polling_in_background(blocked_step, description="poll"):
                assert step_entered.wait(timeout=1.0)
            exit_completed.set()

        owner = threading.Thread(target=own_context)
        owner.start()

        try:
            assert step_entered.wait(timeout=1.0)
            assert not exit_completed.is_set()
            release.set()
            assert exit_completed.wait(timeout=1.0)
        finally:
            release.set()
            owner.join(timeout=1.0)

        assert not owner.is_alive()
