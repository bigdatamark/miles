# [MODULE DOC] Append-only JSON-lines audit event logger. An EventLogger instance (one per
# process, held in a module global set via a setter) stamps each event with a UTC timestamp, the
# process identity and any ambient context fields, appends it as one JSON line to a file, and
# optionally emits a structured log line (tag "audit"). Also provides a decorator that wraps
# methods with ambient context when a logger is installed, and a reader that parses every
# JSON-lines file under a directory back into typed events. EVENTS_DIRNAME is the conventional
# subdirectory name "events".

import contextvars
import functools
import inspect
import logging
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from miles.utils.audit_utils.event_logger.models import Event, EventBase
from miles.utils.audit_utils.process_identity import ProcessIdentity
from miles.utils.tracking_utils.structured_log import log_structured, prune_for_log

logger = logging.getLogger(__name__)

_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)

EVENTS_DIRNAME: str = "events"


class EventLogger:
    # [CLASS DOC] Writer for one process. Thread-safe for writes (a lock guards file appends);
    # the ambient context is stored in a context variable so it is isolated per thread and per
    # asyncio task.
    def __init__(self, *, log_dir: Path | str, file_name: str = "events.jsonl", source: ProcessIdentity) -> None:
        # [FUNCTION DOC] Keyword-only inputs: log directory (created with parents if missing; side
        # effect), file name (default events.jsonl) and the source process identity stamped on
        # every event. Creates the write lock and the context variable (no default value).
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._path = self._log_dir / file_name
        self._source = source
        self._context_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
            "event_logger_context",
        )

    @property
    def source(self) -> ProcessIdentity:
        # [FUNCTION DOC] Returns the process identity given at construction.
        return self._source

    @contextmanager
    def with_context(self, ctx: dict[str, Any]) -> Generator[None, None, None]:
        # [FUNCTION DOC] Context manager: merges the given fields over the current ambient context
        # (new values win) for the duration of the block, then resets the context variable to its
        # previous state. On exit it asserts that the context is still the merged one (nested
        # scopes must be properly nested). Yields nothing.
        """Temporarily merge extra fields into every event logged within this scope.

        Safe for both threads and asyncio tasks (uses contextvars).
        """
        prev = self._context_var.get({})
        merged = {**prev, **ctx}
        token = self._context_var.set(merged)
        try:
            yield
        finally:
            assert self._context_var.get() == merged
            self._context_var.reset(token)

    def log(self, event_cls: type[EventBase], partial: dict[str, Any], *, print_log: bool = True) -> None:
        # [FUNCTION DOC] Builds and records one event. Inputs: the event class, a mapping of its
        # fields, and a keyword flag (default true) for also printing. The event is constructed
        # from the partial fields, then the current UTC time as timestamp, then the logger's source
        # identity, then the ambient context; later sources win on key clashes, so the ambient
        # context can override even the timestamp and source and the partial fields. Validation
        # errors from the event class propagate. The event is serialized to one JSON line and
        # appended under the lock, reopening the file for each write so it may be swapped between
        # events. When the flag is true, also emits an info-level structured log record (tag
        # "audit", operation "event", event class name) with the pruned fields excluding timestamp
        # and source.
        event = event_cls(
            **{
                **partial,
                "timestamp": datetime.now(timezone.utc),
                "source": self._source,
                **self._context_var.get({}),
            }
        )
        line = event.model_dump_json() + "\n"
        with self._lock:
            # Opened per write so the file can be replaced (e.g. restored from a
            # checkpoint snapshot) at any point between events.
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)
        if print_log:
            payload = prune_for_log(event.model_dump(mode="json", exclude={"timestamp", "source"}))
            log_structured(logger.info, tag="audit", op="event", event=type(event).__name__, **payload)

    def close(self) -> None:
        # [FUNCTION DOC] No-op; present for interface completeness since files are opened per
        # write.
        pass


_event_logger: EventLogger | None = None


def set_event_logger(event_logger: EventLogger | None) -> None:
    # [FUNCTION DOC] Sets (or clears with None) the process-global event logger.
    global _event_logger
    _event_logger = event_logger


def get_event_logger() -> EventLogger:
    # [FUNCTION DOC] Returns the global event logger; raises a runtime error if none was set.
    if _event_logger is None:
        raise RuntimeError("EventLogger not initialized. Call set_event_logger() first.")
    return _event_logger


def is_event_logger_initialized() -> bool:
    # [FUNCTION DOC] True when a global event logger is currently set.
    return _event_logger is not None


def event_logger_context(ctx_fn: Callable[..., dict[str, Any]]) -> Callable:
    # [FUNCTION DOC] Decorator factory. Input: a function mapping the decorated method's
    # arguments to a context mapping. Returns a decorator that wraps either a coroutine function
    # (async wrapper that awaits it) or a plain function, preserving metadata. At each call the
    # wrapper evaluates the context function and runs the method inside the logger's context only
    # if a logger is installed at that moment; otherwise runs it plainly without evaluating the
    # context function.
    """Decorator that wraps a method with EventLogger.with_context if initialized.

    ``ctx_fn`` receives the same arguments as the decorated method and returns
    the context dict.  If the event logger is not initialized, the method runs
    without any context.
    """

    def decorator(method: Callable) -> Callable:
        # [FUNCTION DOC] HELPER: Chooses the async or sync wrapper depending on whether the method
        # is a coroutine function.
        if inspect.iscoroutinefunction(method):

            @functools.wraps(method)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # [FUNCTION DOC] HELPER: Enters the optional context, then awaits the method with
                # the same arguments and returns its result.
                with _maybe_with_context(ctx_fn, args=args, kwargs=kwargs):
                    return await method(*args, **kwargs)

            return async_wrapper

        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Enters the optional context, then calls the method with the
            # same arguments and returns its result.
            with _maybe_with_context(ctx_fn, args=args, kwargs=kwargs):
                return method(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def _maybe_with_context(
    ctx_fn: Callable[..., dict[str, Any]], *, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager. If no global logger is installed, yields without
    # doing anything. Otherwise calls the context function with the given positional and keyword
    # arguments and holds the logger's context scope with the result while yielding.
    if not is_event_logger_initialized():
        yield
        return

    with get_event_logger().with_context(ctx_fn(*args, **kwargs)):
        yield


def read_events(log_dir: Path) -> list[Event]:
    # [FUNCTION DOC] Recursively finds all files with the JSON-lines extension under the
    # directory, sorted by path, and parses every non-blank line as a typed event (discriminated
    # by its type field). Lines that fail to parse are logged as warnings with file and line
    # number and skipped. If no file is found, logs a warning and returns an empty list. Returns
    # events in file order, then line order. Events are not sorted by time across files.
    """Read all JSONL event files from a directory and return parsed events."""
    events: list[Event] = []

    jsonl_files = sorted(log_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        logger.warning("No JSONL files found in %s", log_dir)
        return events

    for jsonl_path in jsonl_files:
        with open(jsonl_path, encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    event = _event_adapter.validate_json(raw_line)
                    events.append(event)
                except Exception:
                    logger.warning(
                        "Failed to parse event at %s:%d",
                        jsonl_path,
                        line_num,
                        exc_info=True,
                    )

    return events
