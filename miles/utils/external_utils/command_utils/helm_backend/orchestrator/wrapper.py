# [MODULE DOC]
# Entry point of the orchestrator pod. Runs the orchestration script given after a separator,
# publishes its lifecycle and exit code to the state file, optionally creates a kubernetes job that
# uninstalls the release when the verdict is in, and then stays alive forever (so logs remain
# readable) until the pod is killed. Restart-safe: if the pod restarts, a recorded terminal state is
# replayed instead of rerunning, and a recorded in-progress state is reported as failure. Signals
# (terminate, interrupt) publish a verdict of 128 plus the signal number before exiting.
from __future__ import annotations

import argparse
import logging
import signal
import sys
from dataclasses import dataclass
from time import sleep
from types import FrameType

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.naming import RunFiles
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import (
    OrchestratorState,
    OrchestratorStatus,
)
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.workers.process_utils import launch_bound_subprocess
from miles.utils.workers.serving.utils import split_worker_argv

logger = logging.getLogger(__name__)

_KEEP_ALIVE_POLL_SECONDS = 60
_UNINSTALL_JOB_RETRY_SLEEPS = (5, 15, 45, 90)


def main(argv: list[str] | None = None) -> int:
    # [FUNCTION DOC] Program entry. Configures logging, splits the argument vector (defaulting to the
    # process arguments) into own options and the script command at the separator. Options: required
    # state file path and optional uninstall manifest path. Asserts a command was given. Runs it via
    # the runner, logs the exit code, then never returns normally: it blocks in the keep-alive loop
    # (the return is only reachable if that loop ends, which it does not). Raises SystemExit from
    # the signal handler.
    configure_logger_raw("orchestrator_wrapper")
    own_argv, command = split_worker_argv(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(description="Run the orchestration script and publish its exit code")
    parser.add_argument("--state-file", required=True, help="Path of the orchestrator exit file to write")
    parser.add_argument(
        "--uninstall-manifest",
        default=None,
        help="Path of a rendered job manifest that uninstalls this run's release once its verdict is in",
    )
    args = parser.parse_args(own_argv)
    assert command, "Pass the orchestration script after a -- separator"

    runner = _Runner(state_file=args.state_file, uninstall_manifest=args.uninstall_manifest)
    exit_code = runner.run(command)

    logger.info(f"exit_code={exit_code}, staying alive so the logs remain readable")
    _keep_alive()
    return exit_code


def _keep_alive() -> None:
    # [FUNCTION DOC] HELPER: Sleeps forever in 60 second steps; never returns.
    while True:
        sleep(_KEEP_ALIVE_POLL_SECONDS)


@dataclass(frozen=True)
class _Runner:
    # [CLASS DOC] HELPER: Frozen holder of the state file path and optional uninstall manifest path,
    # with the logic to run the script and publish verdicts.
    state_file: str
    uninstall_manifest: str | None

    def run(self, command: list[str]) -> int:
        # [FUNCTION DOC] Runs the orchestration script once per launch and returns the exit code.
        # Steps: install signal handlers; if a previous state decides the outcome, return that
        # code; publish the started state; launch the command as a bound subprocess (no extra
        # environment) and wait; convert a negative return code to 128 plus its absolute value; then
        # conclude (publish exited state and create the uninstall job). If the signal handler's
        # system exit occurs, it is re-raised without publishing again. Any other exception
        # (including keyboard interrupt) is logged, concluded with exit code 1 and re-raised.
        self._install_signal_verdict()

        if (decided := self._handle_previous_state()) is not None:
            return decided

        self._publish(OrchestratorStatus.STARTED, exit_code=None)
        logger.info(f"Running {command}")

        try:
            process = launch_bound_subprocess(command, envs={})
            exit_code = _exit_code_of_returncode(process.wait())
        except SystemExit:
            logger.info("The signal handler already published this run's verdict; leaving it alone")
            raise
        except BaseException as exception:
            logger.error(f"Failed to run the orchestration script: {exception!r}", exc_info=True)
            self._conclude(exit_code=1)
            raise

        return self._conclude(exit_code=exit_code)

    def _install_signal_verdict(self) -> None:
        # [FUNCTION DOC] HELPER: Registers one handler for terminate and interrupt signals (global
        # process side effect).
        def handle(signal_number: int, frame: FrameType | None) -> None:
            # [FUNCTION DOC] HELPER: Signal handler. If a terminal verdict with an exit code is
            # already published, only logs; otherwise publishes an exited state with exit code 128
            # plus the signal number. Always raises a system exit with that same code.
            if self._is_verdict_published():
                logger.info(f"Received signal {signal_number} after the run reported its exit code; exiting")
            else:
                logger.info(f"Received signal {signal_number}; publishing a verdict before exiting")
                self._publish(OrchestratorStatus.EXITED, exit_code=128 + signal_number)
            raise SystemExit(128 + signal_number)

        for signal_number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signal_number, handle)

    def _handle_previous_state(self) -> int | None:
        # [FUNCTION DOC] HELPER: Inspects the state file left by an earlier run of this launch. No
        # state: none (run normally). Terminal with exit code: create the uninstall job again (it
        # may not have been created) and return that code. Non-terminal (script was running when
        # the pod restarted): log an error and conclude with exit code 1. Terminal state without an
        # exit code cannot occur (rejected by state validation, which reads as no state); the final
        # fallthrough returns none.
        """The exit code this pod is already bound to, or None when it is the first to run this launch."""
        previous = OrchestratorState.read(self.state_file)
        if previous is None:
            return None

        if previous.is_terminal and previous.exit_code is not None:
            logger.info(f"This run already reported exit_code={previous.exit_code}; not running it again")
            self._create_uninstall_job()
            return previous.exit_code

        if not previous.is_terminal:
            logger.error(
                f"This pod restarted while the orchestration script was still running "
                f"(status {previous.status.value}); reporting the run as failed instead of training a second time"
            )
            return self._conclude(exit_code=1)

        return None

    def _is_verdict_published(self) -> bool:
        # [FUNCTION DOC] HELPER: True if the state file holds a terminal state with an exit code.
        published = OrchestratorState.read(self.state_file)
        return published is not None and published.is_terminal and published.exit_code is not None

    def _conclude(self, *, exit_code: int) -> int:
        # [FUNCTION DOC] HELPER: Publishes the exited state with the code, creates the uninstall job
        # (if configured), and returns the code.
        self._publish(OrchestratorStatus.EXITED, exit_code=exit_code)
        self._create_uninstall_job()
        return exit_code

    def _create_uninstall_job(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing without an uninstall manifest. Otherwise makes up to
        # four attempts (sleeping 5, 15, 45, then 90 seconds after failed attempts). Before each
        # attempt checks for the superseded marker next to the state file; if present, a later
        # launch owns the uninstall and this returns. Gives up with an error log after the last
        # attempt, leaving the release installed.
        # KNOWN ISSUE: it also sleeps after the final failed attempt before logging the give-up.
        if self.uninstall_manifest is None:
            return

        marker = RunFiles.superseded_marker(state_file=self.state_file)
        for attempt, sleep_seconds in enumerate(_UNINSTALL_JOB_RETRY_SLEEPS, start=1):
            if marker.exists():
                logger.info(
                    f"{marker} says a later launch took this run over, so this generation leaves the uninstall to it"
                )
                return
            if self._create_uninstall_job_once(attempt=attempt):
                return
            sleep(sleep_seconds)

        logger.error(
            f"Gave up creating the uninstall job of {self.uninstall_manifest} after "
            f"{len(_UNINSTALL_JOB_RETRY_SLEEPS)} attempts, so this release stays installed until it is "
            f"uninstalled by hand"
        )

    def _create_uninstall_job_once(self, *, attempt: int) -> bool:
        # [FUNCTION DOC] HELPER: One attempt via kubectl: create the job from the manifest if absent;
        # if it already exists and its jobs have finished, replace it so it runs again. Any
        # exception is logged and returns false. Returns true on success (including when an
        # unfinished job already exists and is left alone).
        try:
            created = Kubectl.create_if_absent(self.uninstall_manifest)
            if not created and Kubectl.jobs_finished(self.uninstall_manifest):
                Kubectl.replace(self.uninstall_manifest)
                created = True
        except Exception:
            logger.error(
                f"Attempt {attempt} at creating the uninstall job of {self.uninstall_manifest} failed", exc_info=True
            )
            return False

        if created:
            logger.info(f"This run's release uninstalls itself through the job created from {self.uninstall_manifest}")
        else:
            logger.info("This run's uninstall job is already there and has not run yet, so it still uninstalls it")
        return True

    def _publish(self, status: OrchestratorStatus, *, exit_code: int | None) -> None:
        # [FUNCTION DOC] HELPER: Writes a new state record with the status and exit code to the
        # state file.
        OrchestratorState(status=status, exit_code=exit_code).write(self.state_file)


def _exit_code_of_returncode(returncode: int) -> int:
    # [FUNCTION DOC] HELPER: Shell convention: a negative return code (killed by signal N) becomes
    # 128 plus N; non-negative codes are unchanged.
    return 128 + abs(returncode) if returncode < 0 else returncode


if __name__ == "__main__":
    raise SystemExit(main())
