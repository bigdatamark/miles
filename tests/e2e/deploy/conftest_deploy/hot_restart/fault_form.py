# [MODULE DOC]
# HELPER: A fault-injection "form" that performs a hot restart (take-over of the orchestration
# script of a live deployed run) as one kind of injected fault for the randomized fault-tolerance
# soak framework. Injecting relaunches the training command with the hot-restart configuration
# on a background thread, then waits until the cluster shows the take-over landed: the restart
# stamps of the two rolled workloads all changed from their pre-injection values. Records one
# take-over record per injection (progress at trigger). Constants: the form name
# "hot_restart", take-over timeout 1800 s, poll interval 10 s, relaunch join timeout 1800 s, and
# 10 attempts to read the baseline stamps.
import logging
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import (
    compute_hot_restart_workloads,
    read_restart_stamp_of_workload,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import (
    compute_hot_restart_config,
    compute_release_of_config,
    is_replaced_launch_exit,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord, read_run_progress
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import BaseFaultForm

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig

logger = logging.getLogger(__name__)

HOT_RESTART_FORM_NAME: str = "hot_restart"
TAKE_OVER_TIMEOUT_SECONDS: float = 1800.0
TAKE_OVER_POLL_INTERVAL_SECONDS: float = 10.0
RELAUNCH_JOIN_TIMEOUT_SECONDS: float = 1800.0
BASELINE_STAMP_READ_ATTEMPTS: int = 10


class HotRestartFaultForm(BaseFaultForm):
    # [CLASS DOC] HELPER: Fault form whose injection is a hot restart. Holds the launch callable
    # (takes an execute-train configuration), the run configuration, the checkpoint and event
    # directories, the highest rollout id after which injections are still allowed, and tuning
    # (poll interval, timeout, baseline read attempts). Derived at construction: the helm release
    # and namespace, and the set of workload names that a hot restart restamps. Accumulates relaunch
    # threads, failures (pairs of take-over index and exception) and take-over records. The form
    # does not harm any training cell.
    def __init__(
        self,
        *,
        launch: Callable[[ExecuteTrainConfig], None],
        config: ExecuteTrainConfig,
        checkpoint_dir: Path,
        events_dir: Path,
        max_allowed_rollout_id: int,
        poll_interval_seconds: float = TAKE_OVER_POLL_INTERVAL_SECONDS,
        timeout_seconds: float = TAKE_OVER_TIMEOUT_SECONDS,
        baseline_read_attempts: int = BASELINE_STAMP_READ_ATTEMPTS,
    ) -> None:
        # [FUNCTION DOC] HELPER: Store the arguments, compute the release name from the
        # configuration, take the namespace from it, compute the set of workloads a hot restart
        # restamps for that release, and start with empty thread, failure and record lists.
        self._launch = launch
        self._config = config
        self._release = compute_release_of_config(config)
        self._namespace = config.namespace
        self._checkpoint_dir = checkpoint_dir
        self._events_dir = events_dir
        self._max_allowed_rollout_id = max_allowed_rollout_id
        self._poll_interval_seconds = poll_interval_seconds
        self._timeout_seconds = timeout_seconds
        self._baseline_read_attempts = baseline_read_attempts
        self._stamped_workloads = compute_hot_restart_workloads(self._release)
        self._threads: list[threading.Thread] = []
        self._failures: list[tuple[int, BaseException]] = []
        self._records: list[HotRestartRecord] = []

    @property
    def name(self) -> str:
        # [FUNCTION DOC] The form's name: "hot_restart".
        return HOT_RESTART_FORM_NAME

    @property
    def records(self) -> tuple[HotRestartRecord, ...]:
        # [FUNCTION DOC] A tuple copy of the take-over records made so far.
        return tuple(self._records)

    def join_relaunches(self, *, timeout_seconds: float = RELAUNCH_JOIN_TIMEOUT_SECONDS) -> None:
        # [FUNCTION DOC] Wait for each relaunch thread in turn, up to the timeout (default 1800
        # seconds) per thread. Does not report threads still alive.
        for thread in self._threads:
            thread.join(timeout=timeout_seconds)

    def assert_take_overs_installed_cleanly(self) -> None:
        # [FUNCTION DOC] Assert that no unexplained relaunch failure was collected (message lists
        # each by take-over index) and that no relaunch thread is still alive. A failure is
        # explained when it is the replaced-launch exit of a take-over that a later take-over
        # (with a record) superseded.
        failures = self._unexplained_failures()
        assert not failures, "a hot restart of this run did not install cleanly:\n" + "\n".join(
            f"  - take-over {at}: {failure!r}" for at, failure in failures
        )
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        assert not alive, (
            f"{alive} are still installing a hot restart, so this run may still be replaced under the dumps that "
            f"are about to be read"
        )

    @property
    def harms_cell(self) -> bool:
        # [FUNCTION DOC] Always false: a hot restart does not harm a training cell.
        return False

    def is_within_injection_window(self) -> bool:
        # [FUNCTION DOC] Read the run progress; true when no step has finished yet or the last
        # finished rollout id is below the maximum allowed rollout id (leaving quiet steps at the
        # end of the run).
        progress = read_run_progress(checkpoint_dir=self._checkpoint_dir, events_dir=self._events_dir)
        return (
            progress.last_finished_rollout_id is None
            or progress.last_finished_rollout_id < self._max_allowed_rollout_id
        )

    def inject(self, cell: dict, rng: random.Random) -> None:
        # [FUNCTION DOC] Perform one take-over. The cell and random generator arguments are unused.
        # Reads the current run progress and the restart stamps of the workloads to be replaced
        # (retrying), logs the action, starts the relaunch on a new thread, waits until the
        # stamps change (see the wait helper, which can raise assertion errors on refusal, early
        # return of the relaunch or timeout), then appends a record with the take-over index, the
        # last saved iteration at trigger, and the last finished rollout id (minus one if none).
        # Note the record's frozen rollout id field holds the progress at trigger rather than a
        # deliberate freeze point.
        progress = read_run_progress(checkpoint_dir=self._checkpoint_dir, events_dir=self._events_dir)
        stamps_before = self._read_stamps_to_replace()

        logger.info(f"Hot restarting {self._release} of a run that stands at {progress}")
        index = len(self._threads)
        self._relaunch_on_thread()
        self._wait_until_take_over_reached_run(index=index, stamps_before=stamps_before)
        self._records.append(
            HotRestartRecord(
                index=index,
                saved_iteration_at_trigger=progress.last_saved_iteration,
                frozen_rollout_id=(
                    -1 if progress.last_finished_rollout_id is None else progress.last_finished_rollout_id
                ),
            )
        )

    def _relaunch_on_thread(self) -> None:
        # [FUNCTION DOC] HELPER: Start a daemon thread named "take-over-<index>" (index = number of
        # threads so far) running the relaunch body, and remember it.
        index = len(self._threads)
        thread = threading.Thread(target=self._relaunch, args=(index,), daemon=True, name=f"take-over-{index}")
        self._threads.append(thread)
        thread.start()

    def _relaunch(self, index: int) -> None:
        # [FUNCTION DOC] HELPER: Thread body. Call the launch callable with the hot-restart
        # configuration derived from the run configuration for the installed release. Any
        # exception (broadest class) is logged as a warning and appended to the failures with the
        # take-over index.
        try:
            self._launch(compute_hot_restart_config(self._config, installed_release=self._release))
        except BaseException as e:
            logger.warning(f"The hot restart of {self._release} failed to install", exc_info=True)
            self._failures.append((index, e))

    def _wait_until_take_over_reached_run(self, *, index: int, stamps_before: dict[str, str | None]) -> None:
        # [FUNCTION DOC] HELPER: Poll until the take-over landed or fail. Until the timeout
        # (default 1800 s): sleep one poll interval (so the first check is delayed); read the
        # current stamps (missing on failure) and return when every replaced workload has a stamp
        # that differs from before. Otherwise assert that this take-over has no unexplained failure
        # (the cluster refused the relaunch) and that its relaunch thread is still alive (it
        # returning without restamping means nothing took over). After the deadline raises an
        # assertion error saying the stamps never changed.
        deadline = time.monotonic() + self._timeout_seconds
        relaunch = self._threads[index]

        while time.monotonic() < deadline:
            time.sleep(self._poll_interval_seconds)
            if restamped_replaced_workloads(
                before=stamps_before, after=self._read_restart_stamps_or_none(), workloads=self._stamped_workloads
            ):
                logger.info(f"The take-over of {self._release} landed; it restamped {sorted(self._stamped_workloads)}")
                return

            assert not self._failures_of(index), (
                f"the hot restart of {self._release} was refused rather than installed, so the run keeps training "
                f"under the script this injection meant to replace: {self._failures_of(index)}"
            )
            assert relaunch.is_alive(), (
                f"the relaunch of {self._release} returned without restamping {sorted(self._stamped_workloads)}, so "
                f"nothing took its orchestration script over"
            )

        raise AssertionError(
            f"the relaunch of {self._release} was installed {self._timeout_seconds}s ago and "
            f"{sorted(self._stamped_workloads)} still carry the stamps they carried before it ({stamps_before}), "
            f"so a take-over cannot be told from a relaunch that hung"
        )

    def _read_stamps_to_replace(self) -> dict[str, str | None]:
        # [FUNCTION DOC] HELPER: Read the baseline restart stamps of the release's workloads, trying
        # up to the configured number of attempts (default 10) and sleeping one poll interval after
        # each failed read. Returns the first successful mapping; raises an assertion error if all
        # attempts fail.
        for _ in range(self._baseline_read_attempts):
            if (stamps := self._read_restart_stamps_or_none()) is not None:
                return stamps
            time.sleep(self._poll_interval_seconds)

        raise AssertionError(
            f"the workloads of {self._release} could not be read in {self._baseline_read_attempts} attempt(s), and "
            f"a take-over is told from a relaunch that hung by the stamps it replaces, so this draw has nothing to "
            f"compare against"
        )

    def _failures_of(self, index: int) -> list[BaseException]:
        # [FUNCTION DOC] HELPER: The unexplained failures that belong to the given take-over index.
        return [failure for at, failure in self._unexplained_failures() if at == index]

    def _unexplained_failures(self) -> list[tuple[int, BaseException]]:
        # [FUNCTION DOC] HELPER: Collected (index, exception) failures minus those explained as the
        # exit of a launch replaced by a later take-over.
        return [
            (index, failure)
            for index, failure in self._failures
            if not self._is_replacement_exit_explained(index=index, failure=failure)
        ]

    def _is_replacement_exit_explained(self, *, index: int, failure: BaseException) -> bool:
        # [FUNCTION DOC] HELPER: True when the failure is the replaced-launch exit (code 143), a
        # later take-over thread exists, and a record exists for the take-over with index plus
        # one (so the replacement actually happened).
        return (
            is_replaced_launch_exit(failure)
            and index + 1 < len(self._threads)
            and any(record.index == index + 1 for record in self._records)
        )

    def _read_restart_stamps_or_none(self) -> dict[str, str | None] | None:
        # [FUNCTION DOC] HELPER: Read the restart stamps of the release's workloads; any exception
        # is logged as a warning and turned into a missing result (a failed listing also gives a
        # missing result).
        try:
            return read_restart_stamp_of_workload(release=self._release, namespace=self._namespace)
        except Exception:
            logger.warning(f"Failed to read the restart stamps of {self._release}", exc_info=True)
            return None


def restamped_replaced_workloads(
    *, before: dict[str, str | None], after: dict[str, str | None] | None, workloads: frozenset[str]
) -> bool:
    # [FUNCTION DOC] True when the later stamp mapping exists and, for every workload in the set,
    # has a non-missing stamp that differs from the earlier mapping's stamp for it (a workload
    # absent from the earlier mapping counts as a changed stamp). False if the later mapping is
    # missing. With an empty workload set it would be true.
    if after is None:
        return False
    return all((stamp := after.get(one)) is not None and stamp != before.get(one) for one in workloads)
