# [MODULE DOC]
# Container init-style supervisor, run as a module with a command line: starts several copies of
# one command (using torch elastic's multiprocessing launcher), gives each its index through an
# environment variable, ties their lifetimes together (first failure tears down all others and
# determines the exit code), and forwards termination signals to them. Usage is the options, then
# a double-dash separator, then the supervised command.
from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
from types import FrameType

from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, Std, start_processes
from torch.distributed.elastic.multiprocessing.api import RunProcsResult, SignalException

from miles.utils.workers.env_vars import SUBPROCESS_INDEX_ENV_VAR

logger = logging.getLogger(__name__)

_DEFAULT_TERMINATION_GRACE_PERIOD_SECONDS = 20.0
_FORWARDED_SIGNALS = (signal.SIGTERM, signal.SIGINT)
_SIGNALS_TO_HANDLE_ENV_VAR = "TORCHELASTIC_SIGNALS_TO_HANDLE"
_SUPERVISED_SUBPROCESS_NAME = "miles_supervised_subprocess"


def main(argv: list[str] | None = None) -> int:
    # [FUNCTION DOC]
    # Program entry. Uses the process command-line arguments when none are given, parses them,
    # configures root logging at info level with a supervisor-tagged format, and runs the
    # supervision. Returns the supervisor's exit code.
    argv = sys.argv[1:] if argv is None else argv
    args, command = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [process_supervisor] %(levelname)s %(message)s",
    )

    return supervise(
        command=command,
        num_subprocesses=args.num_subprocesses,
        termination_grace_period_seconds=args.termination_grace_period_seconds,
    )


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    # [FUNCTION DOC] HELPER: Split the arguments at the first double-dash: the part before is parsed
    # as options, the part after is the supervised command. Calls the parser's error exit (usage
    # message and process exit status 2) if the separator is missing, the command is empty, the
    # subprocess count is below one, or the grace period is not finite or is negative. Returns the
    # parsed options and the command list.
    parser = _build_parser()
    if "--" not in argv:
        parser.parse_args(argv)
        parser.error("the supervised command must be given after '--'")

    separator_index = argv.index("--")
    args = parser.parse_args(argv[:separator_index])
    command = argv[separator_index + 1 :]
    if not command:
        parser.error("the supervised command after '--' must not be empty")
    if args.num_subprocesses < 1:
        parser.error("--num-subprocesses must be at least 1")
    if not math.isfinite(args.termination_grace_period_seconds) or args.termination_grace_period_seconds < 0:
        parser.error("--termination-grace-period-seconds must be finite and non-negative")

    return args, command


def _build_parser() -> argparse.ArgumentParser:
    # [FUNCTION DOC] HELPER: Build the option parser: a required integer count of subprocesses and
    # an optional float grace period in seconds (default 20) before remaining subprocesses are
    # force-killed after a forwarded signal. The program name is derived from the module spec when
    # available.
    spec = sys.modules[__name__].__spec__

    parser = argparse.ArgumentParser(
        prog=f"python -m {spec.name}" if spec is not None else None,
        description=(
            "Run several copies of one command as the PID 1 of a container and tie their lifetimes together. "
            f"Every copy runs in its own session and gets {SUBPROCESS_INDEX_ENV_VAR} set to its index. "
            "The first copy to fail takes down all the others, and the supervisor exits with its exit code. "
            "When the supervisor itself dies, the container's PID namespace takes every remaining process with it."
        ),
    )
    parser.add_argument("--num-subprocesses", type=int, required=True, help="How many copies of the command to run")
    parser.add_argument(
        "--termination-grace-period-seconds",
        type=float,
        default=_DEFAULT_TERMINATION_GRACE_PERIOD_SECONDS,
        help="How long subprocesses may take to exit after a forwarded signal before they are SIGKILLed",
    )
    return parser


def supervise(
    *,
    command: list[str],
    num_subprocesses: int,
    termination_grace_period_seconds: float = _DEFAULT_TERMINATION_GRACE_PERIOD_SECONDS,
) -> int:
    # [FUNCTION DOC]
    # Run the command as the given number of subprocess copies and block until they finish.
    # Raises a value error if the count is below one. Side effect: sets the environment variable
    # that tells torch elastic to handle terminate and interrupt signals. Starts the copies with
    # each one's index in its environment, all output teed with a rank prefix per line. If a
    # signal arrives while spawning, returns 128 plus the signal number and leaves cleanup to the
    # container's pid namespace. If a signal arrives while waiting, further signals are only
    # reported, the signal is forwarded to all copies, which get the grace period before being
    # force-killed, and 128 plus the signal number is returned. On any other failure of the
    # supervisor, all copies are terminated with the grace period and the error is re-raised.
    # Otherwise returns zero if all copies succeeded or the converted exit code of the first
    # failure.
    if num_subprocesses < 1:
        raise ValueError(f"num_subprocesses must be at least 1, got {num_subprocesses}")

    os.environ[_SIGNALS_TO_HANDLE_ENV_VAR] = ",".join(forwarded.name for forwarded in _FORWARDED_SIGNALS)

    indices = range(num_subprocesses)
    try:
        context = start_processes(
            name=_SUPERVISED_SUBPROCESS_NAME,
            entrypoint=command[0],
            args={index: tuple(command[1:]) for index in indices},
            envs={index: {SUBPROCESS_INDEX_ENV_VAR: str(index)} for index in indices},
            logs_specs=DefaultLogsSpecs(tee=Std.ALL),
            log_line_prefixes={index: f"[rank{index}] " for index in indices},
        )
    except SignalException as received:
        _stop_reacting_to_forwarded_signals()
        logger.info("Received %s while still spawning, leaving the rest to the pid namespace", received.sigval.name)
        return 128 + received.sigval

    try:
        for index, pid in context.pids().items():
            logger.info("Spawned subprocess index=%s pid=%s", index, pid)
        result = context.wait()
    except SignalException as received:
        _stop_reacting_to_forwarded_signals()
        logger.info("Received %s, forwarding it to every subprocess", received.sigval.name)
        context.close(death_sig=received.sigval, timeout=termination_grace_period_seconds)
        return 128 + received.sigval
    except BaseException:
        _stop_reacting_to_forwarded_signals()
        logger.exception("Tearing every subprocess down after an unexpected supervisor failure")
        context.close(death_sig=signal.SIGTERM, timeout=termination_grace_period_seconds)
        raise

    assert result is not None
    return _exit_code_from_result(result)


def _stop_reacting_to_forwarded_signals() -> None:
    # [FUNCTION DOC] HELPER: Replace the handlers of the forwarded signals (terminate, interrupt)
    # with one that only prints a notice, so repeated signals during teardown do not interrupt it.
    for forwarded in _FORWARDED_SIGNALS:
        signal.signal(forwarded, _report_signal_received_during_teardown)


def _report_signal_received_during_teardown(signum: int, frame: FrameType | None) -> None:
    # [FUNCTION DOC] HELPER: Signal handler used during teardown: writes a one-line notice naming
    # the received signal directly to the standard error file descriptor (a raw write, safe in a
    # handler). Ignores the frame argument.
    notice = f"[process_supervisor] Received {signal.Signals(signum).name} while already tearing down\n"
    os.write(sys.stderr.fileno(), notice.encode())


def _exit_code_from_result(result: RunProcsResult) -> int:
    # [FUNCTION DOC] HELPER: Map the launcher result to an exit code: zero when nothing failed.
    # Otherwise logs a warning per failure in index order, then returns the converted exit code of
    # the first failure in the result's own iteration order (not necessarily the lowest index; it
    # is the first to be recorded).
    if not result.is_failed():
        return 0

    for index, failure in sorted(result.failures.items()):
        logger.warning("Subprocess index=%s pid=%s exited with %s", index, failure.pid, failure.exitcode)

    first_failure = next(iter(result.failures.values()))
    return _exit_code_from_wait_result(first_failure.exitcode)


def _exit_code_from_wait_result(wait_result: int) -> int:
    # [FUNCTION DOC] HELPER: Convert a wait status to a shell-style exit code: a negative value
    # (death by signal) becomes 128 plus the signal number, any other value is returned as is.
    return 128 - wait_result if wait_result < 0 else wait_result


if __name__ == "__main__":
    sys.exit(main())
