# [MODULE DOC]
# Background polling helper: runs a zero-argument step function repeatedly on a daemon thread at a
# fixed interval (5 seconds, the module constant) for the lifetime of a context manager. Failures
# in the step are logged and retried rather than propagated.
from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5.0


@contextmanager
def polling_in_background(step: Callable[[], None], *, description: str) -> Iterator[None]:
    # [FUNCTION DOC] Context manager that starts a daemon thread calling the step repeatedly while
    # the with-body runs. The description is used in the retry warning message. On exit (also on
    # exception) signals the thread to stop and joins it, waiting for any in-flight step to finish
    # (no timeout). Yields nothing.
    stop = threading.Event()
    thread = threading.Thread(target=_poll_until_stopped, args=(step, description, stop), daemon=True)
    thread.start()

    try:
        yield
    finally:
        stop.set()
        thread.join()


def _poll_until_stopped(step: Callable[[], None], description: str, stop: threading.Event) -> None:
    # [FUNCTION DOC] HELPER: Thread body. Until the stop event is set: run the step once (any
    # exception is logged as a warning with traceback, saying it could not do the description and
    # will retry), then wait for the poll interval or until stop is set, whichever comes first. The
    # step runs immediately on start, before the first wait.
    while not stop.is_set():
        try:
            step()
        except Exception:
            logger.warning(f"Could not {description}; retrying", exc_info=True)
        stop.wait(POLL_INTERVAL_SECONDS)
