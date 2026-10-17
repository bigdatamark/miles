# [MODULE DOC] HELPER: A small daemon-thread worker for background polling in tests, with
# cooperative stopping through an event and propagation of a crash in the thread to whoever joins
# it, plus a ready-made polling loop that calls a tick function at a fixed interval until stopped.
import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

RunUntilStoppedFn = Callable[[threading.Event], None]
TickFn = Callable[[], None]


class PollingWorker:
    # [CLASS DOC] HELPER: Wraps a function that takes a stop event and runs until the event is set.
    # The thread is a daemon with the given name, not started until start is called. If the function
    # raises anything (including system-exit and keyboard-interrupt style exceptions), the
    # exception is logged at error level and remembered, to be re-raised from join once the thread
    # has finished.
    def __init__(self, *, name: str, run: RunUntilStoppedFn) -> None:
        # [FUNCTION DOC] HELPER: Creates the stop event, an empty failure slot and the (unstarted)
        # daemon thread whose body runs the function with the stop event inside a catch-all that logs
        # "Polling worker <name> died" with the traceback and stores the exception.
        self._stop_event = threading.Event()
        self._exception: BaseException | None = None

        def run_and_record_failure() -> None:
            # [FUNCTION DOC] HELPER: Thread body: runs the user function with the stop event;
            # on any exception logs it and stores it for join to re-raise.
            try:
                run(self._stop_event)
            except BaseException as err:
                logger.error("Polling worker %s died", name, exc_info=err)
                self._exception = err

        self._thread = threading.Thread(target=run_and_record_failure, daemon=True, name=name)

    @property
    def is_running(self) -> bool:
        # [FUNCTION DOC] HELPER: True while the thread is alive (false before start and after it
        # ends).
        return self._thread.is_alive()

    def start(self) -> None:
        # [FUNCTION DOC] HELPER: Starts the thread; raises a runtime error if started twice.
        self._thread.start()

    def stop(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the stop event, asking the function to return; does not wait.
        self._stop_event.set()

    def join(self, *, timeout_seconds: float) -> None:
        # [FUNCTION DOC] HELPER: Waits up to the timeout for the thread to end. If it has ended and
        # had recorded an exception, re-raises that exception; if the timeout expires with the thread
        # still alive, returns silently (use the running check to detect that). Joining a never
        # started thread raises a runtime error.
        self._thread.join(timeout=timeout_seconds)
        if not self._thread.is_alive() and self._exception is not None:
            raise self._exception

    def stop_and_join(self, *, timeout_seconds: float) -> None:
        # [FUNCTION DOC] HELPER: Sets the stop event, then joins with the timeout (with the join
        # semantics above, including re-raising a recorded failure).
        self.stop()
        self.join(timeout_seconds=timeout_seconds)

    def assert_not_running(self, *, message: str) -> None:
        # [FUNCTION DOC] HELPER: Asserts the thread is not alive, failing with the given message.
        assert not self.is_running, message


def poll_until_stopped(stop_event: threading.Event, *, tick: TickFn, poll_interval_seconds: float) -> None:
    # [FUNCTION DOC] HELPER: Loop suitable as the worker function: while the stop event is not set,
    # calls the tick function (an exception from it ends the loop and propagates) and then waits on
    # the stop event for up to the poll interval, so a stop request wakes it immediately. The tick
    # always runs at least once if the event was not set beforehand.
    while not stop_event.is_set():
        tick()
        stop_event.wait(timeout=poll_interval_seconds)
