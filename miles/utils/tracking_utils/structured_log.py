# [MODULE DOC] Structured logging helpers producing logfmt-style key=value messages: a logger front
# end that attributes the log record to the caller's frame, a decorator that logs start, end, failure
# or cancellation with elapsed time for sync and async functions, value formatting/quoting helpers,
# and a pruner that shrinks large values for logging.
import asyncio
import functools
import inspect
import json
import logging
import time
from collections.abc import Callable
from typing import Any

_PRUNE_CAP = 160


def log_structured(log_fn: Callable[..., None], *, tag: str = "", exc_info: bool = False, **fields: Any) -> None:
    # [FUNCTION DOC] Formats the keyword fields as logfmt (space-separated key=value in insertion
    # order), prefixes the tag and a space when the tag is non-empty, and calls the given logger method
    # with the message, a stack level of 2 (so the record shows the caller of this function) and the
    # exception-info flag. The log method must accept stack level and exception info keywords.
    message = _to_logfmt(fields)
    log_fn(f"{tag} {message}" if tag else message, stacklevel=2, exc_info=exc_info)


def with_logs(func: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] Decorator for functions or coroutine functions (metadata preserved). Each call logs
    # at info level, tagged "actor", the class name of the first positional argument (empty if none),
    # function name and phase "start"; on success logs phase "end", ok true and elapsed seconds
    # (monotonic, one decimal); on any exception (including interrupts) logs at error level with ok
    # false, elapsed time and traceback, and re-raises; for async functions a cancellation is logged at
    # info level with ok false and cancelled true and re-raised. The logger is the one named after the
    # function's module. The first argument is assumed to be the instance for methods; for plain
    # functions the class name is that of the first argument's type.
    fn_name = func.__name__
    method_logger = logging.getLogger(func.__module__)

    def log_start(args: tuple[Any, ...]) -> tuple[str, float]:
        # [FUNCTION DOC] HELPER: Logs the start line and returns the class name and monotonic start time.
        cls = type(args[0]).__name__ if args else ""
        log_structured(method_logger.info, tag="actor", cls=cls, fn=fn_name, phase="start")
        return cls, time.monotonic()

    def log_end(cls: str, start: float) -> None:
        # [FUNCTION DOC] HELPER: Logs the success end line with elapsed seconds.
        log_structured(
            method_logger.info, tag="actor", cls=cls, fn=fn_name, phase="end", ok=True, elapsed_s=_elapsed(start)
        )

    def log_fail(cls: str, start: float) -> None:
        # [FUNCTION DOC] HELPER: Logs the failure end line at error level with traceback; must be
        # called inside an exception handler.
        log_structured(
            method_logger.error,
            tag="actor",
            cls=cls,
            fn=fn_name,
            phase="end",
            ok=False,
            elapsed_s=_elapsed(start),
            exc_info=True,
        )

    def log_cancelled(cls: str, start: float) -> None:
        # [FUNCTION DOC] HELPER: Logs the cancellation end line at info level.
        log_structured(
            method_logger.info,
            tag="actor",
            cls=cls,
            fn=fn_name,
            phase="end",
            ok=False,
            elapsed_s=_elapsed(start),
            cancelled=True,
        )

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Async wrapper implementing the logging described above.
            cls, start = log_start(args)
            try:
                result = await func(*args, **kwargs)
            except asyncio.CancelledError:
                log_cancelled(cls, start)
                raise
            except BaseException:
                log_fail(cls, start)
                raise
            log_end(cls, start)
            return result

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Sync wrapper implementing the logging described above.
        cls, start = log_start(args)
        try:
            result = func(*args, **kwargs)
        except BaseException:
            log_fail(cls, start)
            raise
        log_end(cls, start)
        return result

    return wrapper


def _elapsed(start: float) -> float:
    # [FUNCTION DOC] HELPER: Seconds since the monotonic start time, rounded to one decimal.
    return round(time.monotonic() - start, 1)


def _to_logfmt(fields: dict[str, Any]) -> str:
    # [FUNCTION DOC] HELPER: Joins key=formatted-value pairs with single spaces.
    return " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())


def _format_value(value: Any) -> str:
    # [FUNCTION DOC] HELPER: none becomes an empty string; booleans "true"/"false"; integers and floats
    # their plain string form; lists and tuples a comma-joined string of scalar forms, quoted if
    # needed; mappings compact JSON (no spaces, unserializable values stringified) always quoted;
    # everything else its string form, quoted if needed.
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return _maybe_quote(",".join(_format_scalar(item) for item in value))
    if isinstance(value, dict):
        return _quote(json.dumps(value, separators=(",", ":"), default=str))
    return _maybe_quote(str(value))


def _format_scalar(value: Any) -> str:
    # [FUNCTION DOC] HELPER: Booleans as "true"/"false", anything else as its string form.
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _maybe_quote(text: str) -> str:
    # [FUNCTION DOC] HELPER: Quotes (with escaping) non-empty text containing a space, equals sign,
    # double quote, backslash, newline, carriage return or tab; otherwise returns it unchanged.
    if text and any(ch in text for ch in (" ", "=", '"', "\\", "\n", "\r", "\t")):
        return _quote(text)
    return text


def _quote(text: str) -> str:
    # [FUNCTION DOC] HELPER: Escapes backslash, double quote, newline, carriage return and tab with
    # backslash sequences and wraps the result in double quotes.
    escaped = (
        text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    )
    return f'"{escaped}"'


def prune_for_log(value: Any, cap: int = _PRUNE_CAP) -> Any:
    # [FUNCTION DOC] Returns the value unchanged if its compact JSON form (unserializable parts
    # stringified) is at most cap characters (default 160). Otherwise: mappings are pruned recursively
    # per value with the same cap (each value judged independently, so the result can still be long);
    # lists and tuples become a "<list len=N>" placeholder string; strings "<str N chars>"; any other
    # type "<TypeName>". Raises if the value cannot be JSON-encoded even with stringification (for
    # example circular references or non-string mapping keys of unsortable kinds).
    if len(_compact_json(value)) <= cap:
        return value
    if isinstance(value, dict):
        return {key: prune_for_log(item, cap) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return f"<list len={len(value)}>"
    if isinstance(value, str):
        return f"<str {len(value)} chars>"
    return f"<{type(value).__name__}>"


def _compact_json(value: Any) -> str:
    # [FUNCTION DOC] HELPER: JSON text with no spaces after separators, using string conversion for
    # unserializable values.
    return json.dumps(value, separators=(",", ":"), default=str)
