# [MODULE DOC]
# Process logging setup. Configures root logging format (timestamp with milliseconds and a process
# name), optionally an event logger and environment reporting, and installs a strict hook that turns
# "coroutine was never awaited" warnings into a hard process exit, since they would otherwise be
# swallowed in garbage collection.
import logging
import os
import re
import sys
import warnings
from miles.utils.audit_utils.event_logger.logger import EventLogger, is_event_logger_initialized, set_event_logger
from miles.utils.audit_utils.process_identity import ProcessIdentity
from miles.utils.env_report.reporter import EnvReporter, start_env_reporting

_STRICT_ASYNC_WARNINGS_INSTALLED = False
_ENV_REPORTER: EnvReporter | None = None

logger = logging.getLogger(__name__)

_FATAL_ASYNC_PATTERN = "coroutine .* was never awaited"


def configure_logger(args, *, source: ProcessIdentity, report_env: bool = True) -> None:
    # [FUNCTION DOC] Configures raw logging with the process identity's name. If args has a debug
    # event data directory and no event logger exists yet, creates one writing a jsonl file named
    # after the process. Unless disabled, starts environment reporting once per process (stores the
    # reporter in a module global).
    name = source.to_name()
    configure_logger_raw(name)

    if (event_dir := getattr(args, "save_debug_event_data", None)) is not None:
        if not is_event_logger_initialized():
            set_event_logger(EventLogger(log_dir=event_dir, file_name=f"{name}.jsonl", source=source))

    global _ENV_REPORTER
    if report_env and _ENV_REPORTER is None:
        _ENV_REPORTER = start_env_reporting(args)


def rebind_env_reporting(args) -> None:
    # [FUNCTION DOC] If an env reporter was started, rebinds it to the new args object; otherwise
    # does nothing.
    # TODO: stop rebinding here; the args object should not be overwritten in place all over the codebase.
    if _ENV_REPORTER is not None:
        _ENV_REPORTER.rebind(args)


# ref: SGLang
def configure_logger_raw(name: str = "") -> None:
    # [FUNCTION DOC] Sets root logging at info level with format "[time.millis name] file:line -
    # message", replacing existing handlers. On the first call in the process also installs strict
    # async warnings.
    global _STRICT_ASYNC_WARNINGS_INSTALLED

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s.%(msecs)03d {name}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    if not _STRICT_ASYNC_WARNINGS_INSTALLED:
        _STRICT_ASYNC_WARNINGS_INSTALLED = True
        configure_strict_async_warnings()


def configure_strict_async_warnings() -> None:
    # [FUNCTION DOC] Makes never-awaited-coroutine runtime warnings raise as errors, and wraps the
    # interpreter's unraisable hook: if the unraisable exception is such a warning, logs and prints a
    # fatal message to standard error and exits the process immediately with status 1 (no cleanup);
    # otherwise delegates to the previous hook. Calling it twice stacks wrappers.
    """Turn unawaited-coroutine warnings into fatal errors.

    Python emits RuntimeWarning when a coroutine is called but never awaited.
    The warning fires inside __del__, so the resulting exception is swallowed
    by sys.unraisablehook. We override the hook to hard-exit the process.
    """
    warnings.filterwarnings("error", category=RuntimeWarning, message=_FATAL_ASYNC_PATTERN)

    _original_hook = sys.unraisablehook

    def _crash_on_async_misuse(unraisable):
        # [FUNCTION DOC] HELPER: The replacement hook described above.
        if isinstance(unraisable.exc_value, RuntimeWarning) and re.search(
            _FATAL_ASYNC_PATTERN, str(unraisable.exc_value)
        ):
            msg = f"Fatal async misuse, aborting: {unraisable.exc_value}"
            logger.error(msg)
            print(msg, file=sys.stderr, flush=True)
            os._exit(1)
        _original_hook(unraisable)

    sys.unraisablehook = _crash_on_async_misuse
