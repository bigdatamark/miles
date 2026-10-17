# [MODULE DOC] Process-wide named wall-clock timers. A singleton registry accumulates elapsed seconds
# per name across start/end pairs, notifies optional event sinks (used by the dashboard) on each
# completed interval, and logs start/end on distributed rank zero. Offers a decorator/context-manager
# front end and an inverse context manager that pauses a running timer. Only the timer class and the
# front-end function are exported by the module's public names; the inverse timer is importable
# directly. Not thread-safe.
import logging
from contextlib import contextmanager
from functools import wraps
from time import time

import torch.distributed

from .misc import SingletonMeta

__all__ = ["Timer", "timer"]

logger = logging.getLogger(__name__)


class Timer(metaclass=SingletonMeta):
    # [CLASS DOC] Singleton timer registry (the metaclass returns the same instance per process).
    # State: accumulated seconds per name, start timestamps of currently running names, and a list of
    # event sinks. A sink is a callable taking name, start time and end time (epoch seconds), called at
    # every end; if it also has a callable begin attribute, that is called with name and start time on
    # every start.
    def __init__(self):
        # [FUNCTION DOC] Initializes empty accumulated-time and running-start mappings and an empty sink
        # list.
        self.timers = {}
        self.start_time = {}
        # observers called sink(name, t0, t1) on end(); an optional
        # sink.begin(name, t0) also hears starts. Registered by the dashboard.
        self.event_sinks = []

    def start(self, name):
        # [FUNCTION DOC] Starts the named timer: asserts it is not already running, records the current
        # epoch time, notifies each sink's optional begin hook (sink exceptions propagate), and on
        # distributed rank zero logs an info line. Timers with different names may overlap.
        assert name not in self.start_time, f"Timer {name} already started."
        self.start_time[name] = time()
        for sink in self.event_sinks:
            begin = getattr(sink, "begin", None)
            if callable(begin):
                begin(name, self.start_time[name])
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"Timer {name} start")

    def end(self, name):
        # [FUNCTION DOC] Ends the named timer: asserts it is running, computes elapsed seconds, calls
        # every sink with name, start and end times, adds the elapsed seconds to the accumulated total,
        # removes the running entry, and on rank zero logs the elapsed time with one decimal. If a sink
        # raises, the running entry stays and nothing is accumulated.
        assert name in self.start_time, f"Timer {name} not started."
        start_time = self.start_time[name]
        elapsed_time = time() - start_time
        for sink in self.event_sinks:
            sink(name, start_time, start_time + elapsed_time)
        self.add(name, elapsed_time)
        del self.start_time[name]
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"Timer {name} end (elapsed: {elapsed_time:.1f}s)")

    def reset(self, name=None):
        # [FUNCTION DOC] With no name, clears all accumulated totals; with a name, removes that total if
        # present. Running timers are not affected.
        if name is None:
            self.timers = {}
        elif name in self.timers:
            del self.timers[name]

    def add(self, name, elapsed_time):
        # [FUNCTION DOC] Adds the given seconds to the named total (starting from zero). Does not
        # notify sinks.
        self.timers[name] = self.timers.get(name, 0) + elapsed_time

    def log_dict(self):
        # [FUNCTION DOC] Returns the live accumulated totals mapping (not a copy) from name to seconds.
        return self.timers

    @contextmanager
    def context(self, name):
        # [FUNCTION DOC] Context manager: starts the named timer on entry and always ends it on exit,
        # including on exceptions.
        self.start(name)
        try:
            yield
        finally:
            self.end(name)


def timer(name_or_func):
    # [FUNCTION DOC] Dual-purpose front end on the singleton. Given a string, returns a context manager
    # timing the block under that name. Given a callable, returns a wrapper that times each call under
    # the function's own name and returns its result (exceptions propagate after the timer ends).
    # Reentrant use of the same name raises the already-started assertion.
    """
    Can be used either as a decorator or a context manager:

    @timer
    def func():
        ...

    or

    with timer("block_name"):
        ...
    """
    # When used as a context manager
    if isinstance(name_or_func, str):
        name = name_or_func
        return Timer().context(name)

    func = name_or_func

    @wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Runs the wrapped function inside a timer context named after it and
        # returns its result.
        with Timer().context(func.__name__):
            return func(*args, **kwargs)

    return wrapper


@contextmanager
def inverse_timer(name):
    # [FUNCTION DOC] Context manager that pauses a running timer: ends the named timer on entry
    # (assertion error if not running), and on exit, even after exceptions, starts it again (a new
    # interval, so sinks see two separate intervals).
    Timer().end(name)
    try:
        yield
    finally:
        Timer().start(name)
