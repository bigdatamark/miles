# [MODULE DOC]
# HELPER: Drives the "hot restart" of a deployed training run for deploy end-to-end tests. A hot
# restart upgrades the already-installed release with a hot-restart argument so that only the
# orchestration script and rollout executor workloads are replaced while the trainers keep
# running. The driver, in a background thread, waits until the run is frozen at each scheduled
# rollout step (a fault injection puts the run to sleep forever at the end of that step),
# validates the checkpoint state, records the trigger, and relaunches the training command on
# its own thread (the previous launch is expected to be killed by the take-over and exit with
# the termination-signal exit code 143). It also feeds a cluster observer. Provides: constants
# for polling, timeouts and limits; the checkpoint directory name; the immutable scheduled
# freeze record; helpers computing the freeze plan, release name and hot-restart configuration;
# the driver class; and a context manager running the driver around a launch.
import dataclasses
import logging
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import ClusterObserver
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import (
    HotRestartEvidence,
    HotRestartRecord,
    RunProgress,
    read_run_progress,
)
from tests.e2e.ft.conftest_ft.execution import run_training
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.launcher.entrypoint import RunExitedError
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.test_utils.ft_test_actions import SLEEP_FOREVER_AT_END_ACTION, read_frozen_rollout_id
from miles.utils.test_utils.polling_worker import PollingWorker
from miles.utils.workers.types import HOT_RESTART_SEPARATOR, HotRestartComponent


# ================================= constants ==================================


logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS: float = 5.0
FREEZE_TIMEOUT_SECONDS: float = 3600.0
RELAUNCH_JOIN_TIMEOUT_SECONDS: float = 1800.0
CONSECUTIVE_READ_FAILURE_LIMIT: int = 20
TRACKER_SETTLE_ATTEMPTS: int = 3
TRACKER_SETTLE_INTERVAL_SECONDS: float = 2.0
HOT_RESTART_ARG: str = HOT_RESTART_SEPARATOR.join(one.value for one in HotRestartComponent)
REPLACED_LAUNCH_EXIT_CODE: int = 128 + signal.SIGTERM


# ============================== run directories ===============================


CHECKPOINT_DIRNAME: str = "checkpoints"


def compute_checkpoint_dir(dump_dir: str) -> Path:
    # [FUNCTION DOC] Return the checkpoint directory of a run: a folder named "checkpoints" inside
    # the dump directory. Pure path computation.
    return Path(dump_dir) / CHECKPOINT_DIRNAME


# ============================== the freeze plan ===============================


@dataclass(frozen=True)
class ScheduledFreeze:
    # [CLASS DOC] HELPER: Immutable description of one planned take-over: the rollout id after
    # which the run is frozen, and the checkpoint iteration the run is pinned to hold at that
    # moment (missing when no checkpoint exists yet). The take-over resumes from the saved
    # iteration (or from scratch), so the steps after it up to the frozen step get redone.
    frozen_rollout_id: int
    saved_iteration: int | None

    def __post_init__(self) -> None:
        # [FUNCTION DOC] HELPER: Validate on construction: the frozen rollout id must be
        # non-negative, and a saved iteration, if present, must be strictly smaller than the
        # frozen rollout id (otherwise nothing would be redone). Raises assertion errors.
        assert self.frozen_rollout_id >= 0, f"a run starts at step 0 and cannot freeze after {self.frozen_rollout_id}"
        assert self.saved_iteration is None or self.saved_iteration < self.frozen_rollout_id, (
            f"a run frozen after step {self.frozen_rollout_id} resumes from iteration {self.saved_iteration}, "
            f"which already covers that step, so nothing would be redone"
        )


def compute_freeze_plan(frozen_rollout_id: int | None) -> list[dict]:
    # [FUNCTION DOC] Build the fault-injection plan for freezing a run. With no rollout id returns
    # an empty list; otherwise a one-element list holding a mapping with the rollout id to act
    # at and the "sleep forever at the end" action.
    if frozen_rollout_id is None:
        return []
    return [{"at_rollout": frozen_rollout_id, "action": SLEEP_FOREVER_AT_END_ACTION}]


# ============================ relaunching a release ===========================


def compute_release_of_config(config: ExecuteTrainConfig) -> str:
    # [FUNCTION DOC] Return the serialized helm release name for the configuration, built from its
    # run id, deploy component and deploy instance id.
    return ReleaseName(
        run_id=config.run_id,
        deploy_component=config.deploy_component,
        deploy_instance_id=config.deploy_instance_id,
    ).serialize()


def relaunch_with_hot_restart(
    *, train_args: str, mode: FTTestMode, config: ExecuteTrainConfig, installed_release: str
) -> None:
    # [FUNCTION DOC] Run the training command again as a hot restart of the installed release:
    # build the hot-restart configuration (which asserts the release matches) and call the
    # fault-tolerance training runner with the same arguments and mode. Blocks until that run
    # ends; side effects: cluster deployment upgrade.
    run_training(
        train_args=train_args,
        mode=mode,
        config=compute_hot_restart_config(config, installed_release=installed_release),
    )


def compute_hot_restart_config(config: ExecuteTrainConfig, *, installed_release: str) -> ExecuteTrainConfig:
    # [FUNCTION DOC] Assert that the release derived from the configuration equals the installed
    # release (a hot restart must upgrade the release already up, else it would leave the
    # trainers behind), then return a copy of the configuration with its hot-restart field set
    # to the components joined by the hot-restart separator (all components of the hot restart
    # enumeration).
    assert (relaunched := compute_release_of_config(config)) == installed_release, (
        f"a hot restart upgrades the release that is already up: this relaunch would install {relaunched}, not the "
        f"watched {installed_release}, so it built a config with a run id of its own and would leave the trainers behind"
    )
    return dataclasses.replace(config, hot_restart=HOT_RESTART_ARG)


# ============================== the take-over loop =============================


@dataclass
class HotRestartDriver:
    # [CLASS DOC] HELPER: Background driver performing scheduled take-overs of a frozen run.
    # Fields: a relaunch callable taking the next rollout id to freeze at (or nothing for the last
    # one); the checkpoint and event-log directories; the release, namespace and trainer id; the
    # path of the freeze plan file (marked as an ad hoc hack to revert after an arguments
    # refactor); the tuple of scheduled freezes (must be non-empty); tuning (poll interval 5 s,
    # freeze timeout 3600 s per take-over, consecutive read failure limit 20, checkpoint tracker
    # settle attempts 3 at 2 s); and the list of records of triggered take-overs. Lifecycle: start,
    # run the launch under the context manager, stop collecting, assert nothing is running, assert
    # all restarts happened. Private state: failures list, relaunch threads, a schedule-finished
    # event, the polling worker, the highest finished rollout id seen, the count of take-overs
    # already caught up after, and a cluster observer.
    relaunch: Callable[[int | None], None]
    checkpoint_dir: Path
    events_dir: Path
    release: str
    namespace: str
    trainer_id: str
    # TODO ad hoc hack: revert after the args refactor
    freeze_plan_path: Path
    schedule: tuple[ScheduledFreeze, ...]
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS
    freeze_timeout_seconds: float = FREEZE_TIMEOUT_SECONDS
    read_failure_limit: int = CONSECUTIVE_READ_FAILURE_LIMIT
    tracker_settle_attempts: int = TRACKER_SETTLE_ATTEMPTS
    tracker_settle_interval_seconds: float = TRACKER_SETTLE_INTERVAL_SECONDS
    records: list[HotRestartRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        # [FUNCTION DOC] HELPER: Asserts the schedule is non-empty, then creates the private state:
        # empty failure and thread lists, the schedule-finished event, the polling worker (named
        # "hot-restart-driver", running the drive loop), no maximum finished rollout id yet, zero
        # caught-up take-overs, and a cluster observer for the release, namespace and trainer.
        assert (
            self.schedule
        ), "an empty schedule describes a run nothing freezes, and this driver only takes over a frozen run"
        self._failures: list[BaseException] = []
        self._relaunch_threads: list[threading.Thread] = []
        self._schedule_finished = threading.Event()
        self._worker = PollingWorker(name="hot-restart-driver", run=self._drive)
        self._max_finished_rollout_id: int | None = None
        self._num_take_overs_caught_up: int = 0
        self._observer = ClusterObserver(release=self.release, namespace=self.namespace, trainer_id=self.trainer_id)

    @property
    def num_restarts(self) -> int:
        # [FUNCTION DOC] HELPER: Number of scheduled take-overs (length of the schedule).
        return len(self.schedule)

    @property
    def evidence(self) -> HotRestartEvidence:
        # [FUNCTION DOC] Build an immutable evidence object from the current records, the observer's
        # snapshots, the release name and the observer's attempt and failure counts. Takes copies as
        # tuples, so later changes do not affect it.
        return HotRestartEvidence(
            records=tuple(self.records),
            snapshots=tuple(self._observer.snapshots),
            release=self.release,
            observation_attempts=self._observer.attempts,
            observation_failures=self._observer.failures,
        )

    def start(self) -> None:
        # [FUNCTION DOC] Assert the checkpoint directory and the events directory do not exist yet
        # (a previous run would be resumed and compared with a baseline starting from nothing),
        # then start the background worker thread.
        for path in (self.checkpoint_dir, self.events_dir):
            assert not path.exists(), (
                f"{path} exists before this run was even installed: it would resume from a previous run and be compared "
                f"against a baseline that started from nothing"
            )
        self._worker.start()

    def stop_collecting(self) -> None:
        # [FUNCTION DOC] Stop the driver worker and wait up to 1800 seconds, take one warn-only
        # observation, wait up to 1800 seconds for each relaunch thread, then take a final warn-only
        # observation. Does not itself assert that everything stopped.
        self._worker.stop_and_join(timeout_seconds=RELAUNCH_JOIN_TIMEOUT_SECONDS)
        self._observer.observe_once_or_warn()
        for thread in self._relaunch_threads:
            thread.join(timeout=RELAUNCH_JOIN_TIMEOUT_SECONDS)
        self._observer.observe_once_or_warn()

    def wait_until_all_restarts_triggered(self) -> None:
        # [FUNCTION DOC] Called after a launch exited with the replaced-launch code (143). Asserts at
        # least one take-over record exists (otherwise nothing explains the signal), then waits for
        # the schedule-finished event up to freeze timeout times the number of restarts, and asserts
        # it was set in time.
        assert self.records, (
            f"a launch exited {REPLACED_LAUNCH_EXIT_CODE} before the driver triggered any hot restart, so no "
            f"scheduled replacement explains its SIGTERM"
        )
        timeout_seconds = self.freeze_timeout_seconds * self.num_restarts
        assert self._schedule_finished.wait(timeout=timeout_seconds), (
            f"the replaced launch exited {REPLACED_LAUNCH_EXIT_CODE}, but the driver did not trigger all "
            f"{self.num_restarts} hot restart(s) within {timeout_seconds}s"
        )

    def assert_nothing_running(self) -> None:
        # [FUNCTION DOC] Assert the worker thread and every relaunch thread have terminated
        # (messages say reading what was collected would race them).
        self._worker.assert_not_running(
            message=(
                f"the hot restart driver was still working {RELAUNCH_JOIN_TIMEOUT_SECONDS}s after being asked to "
                f"stop, so reading what it collected would race it"
            )
        )
        for thread in self._relaunch_threads:
            assert not thread.is_alive(), (
                f"{thread.name} is still installing a hot restart {RELAUNCH_JOIN_TIMEOUT_SECONDS}s after the run "
                f"ended, so the dumps about to be compared may still be replaced under it"
            )

    def assert_all_restarts_happened(self) -> None:
        # [FUNCTION DOC] Assert that no failure was collected (message lists each failure's repr)
        # and that the number of take-over records equals the scheduled number.
        assert not self._failures, "the hot restart driver failed:\n" + "\n".join(
            f"  - {one!r}" for one in self._failures
        )
        assert len(self.records) == self.num_restarts, (
            f"the run ended after {len(self.records)} of {self.num_restarts} hot restart(s), so what a take-over "
            f"of a still-training run's trainers costs was never measured"
        )

    def _drive(self, stop_event: threading.Event) -> None:
        # [FUNCTION DOC] HELPER: Worker body. Runs the take-over loop; any exception (broadest base
        # class) is logged and appended to the failure list; in all cases sets the
        # schedule-finished event at the end.
        try:
            self._take_over_at_scheduled_freezes(stop_event)
        except BaseException as e:
            logger.warning("The hot restart driver stopped before every take-over had landed", exc_info=True)
            self._failures.append(e)
        finally:
            self._schedule_finished.set()

    def _take_over_at_scheduled_freezes(self, stop_event: threading.Event) -> None:
        # [FUNCTION DOC] HELPER: For each scheduled freeze, in order: wait until the run is frozen at
        # that step (return immediately if stopped meanwhile); build and store the take-over record
        # (which validates the checkpoint state); log it; and start the relaunch on a new thread
        # that is told the next scheduled frozen step (or nothing after the last). After all are
        # triggered, set the schedule-finished event and keep observing the cluster until
        # stopped.
        for index, scheduled in enumerate(self.schedule):
            progress = self._wait_until_run_frozen(stop_event, index=index, scheduled=scheduled)
            if progress is None:
                return

            record = self._compute_record(index=index, scheduled=scheduled, progress=progress)
            self.records.append(record)
            logger.info(f"Hot restart {record.index} is due against a frozen run: {record}")
            self._relaunch_on_thread(index)

        self._schedule_finished.set()
        self._observe_until_stopped(stop_event)

    def _observe_until_stopped(self, stop_event: threading.Event) -> None:
        # [FUNCTION DOC] HELPER: Until the stop event is set, take a warn-only cluster observation
        # and wait one poll interval (or until the event is set).
        while not stop_event.is_set():
            self._observer.observe_once_or_warn()
            stop_event.wait(timeout=self.poll_interval_seconds)

    def _wait_until_run_frozen(
        self, stop_event: threading.Event, *, index: int, scheduled: ScheduledFreeze
    ) -> RunProgress | None:
        # [FUNCTION DOC] HELPER: Poll until the run stands frozen at the scheduled step. Loop until
        # the stop event is set: read the run progress (observing the cluster too); a failed read
        # increments a consecutive failure count and asserts it stays below the failure limit (20);
        # a good read resets the count, asserts no step was lost outside a take-over, and returns
        # the progress if the run stands frozen at the scheduled step. Each iteration asserts the
        # freeze deadline (timeout 3600 s from the start of the wait) has not passed, then waits one
        # poll interval. Returns nothing if stopped.
        deadline = time.monotonic() + self.freeze_timeout_seconds
        failures = 0

        while not stop_event.is_set():
            if (progress := self._read_progress_and_observe_cluster()) is None:
                failures += 1
                assert failures < self.read_failure_limit, (
                    f"hot restart {index} failed to read how far the run had come {failures} time(s) in a row, so "
                    f"it cannot tell a run frozen after step {scheduled.frozen_rollout_id} from one still training"
                )
            else:
                failures = 0
                self._assert_no_step_lost_outside_take_over(progress)
                if self._stands_frozen_at(scheduled, progress=progress):
                    return progress

            assert time.monotonic() < deadline, (
                f"hot restart {index} waited {self.freeze_timeout_seconds}s for a run frozen after step "
                f"{scheduled.frozen_rollout_id}, and the run only reached {progress}"
            )
            stop_event.wait(timeout=self.poll_interval_seconds)

        return None

    def _read_progress_and_observe_cluster(self) -> RunProgress | None:
        # [FUNCTION DOC] HELPER: Read run progress from the checkpoint and event directories; on any
        # exception log a warning and return nothing. If at least one step has finished, also take
        # a warn-only cluster observation. Returns the progress.
        try:
            progress = read_run_progress(checkpoint_dir=self.checkpoint_dir, events_dir=self.events_dir)
        except Exception:
            logger.warning("Failed to read how far the run being hot restarted has come", exc_info=True)
            return None

        if progress.last_finished_rollout_id is not None:
            self._observer.observe_once_or_warn()
        return progress

    def _stands_frozen_at(self, scheduled: ScheduledFreeze, *, progress: RunProgress) -> bool:
        # [FUNCTION DOC] HELPER: If a finished step is known, assert it does not exceed the
        # scheduled frozen step (else the freezing injection never fired). Returns whether the
        # frozen rollout id read from the freeze plan file equals the scheduled frozen rollout id.
        if (finished := progress.last_finished_rollout_id) is not None:
            assert finished <= scheduled.frozen_rollout_id, (
                f"the run was armed to sleep after step {scheduled.frozen_rollout_id} and finished step {finished}: "
                f"the injection that freezes it never fired, so a take-over would race the run"
            )

        return read_frozen_rollout_id(self.freeze_plan_path) == scheduled.frozen_rollout_id

    def _compute_record(self, *, index: int, scheduled: ScheduledFreeze, progress: RunProgress) -> HotRestartRecord:
        # [FUNCTION DOC] HELPER: Read the settled saved iteration (waiting for the checkpoint tracker
        # to reach the pinned value) and assert it equals the scheduled saved iteration (else the
        # run saves at a different cadence than pinned). Returns a record with the take-over index,
        # the scheduled saved iteration and the scheduled frozen rollout id.
        saved = self._read_settled_saved_iteration(on=scheduled.saved_iteration, progress=progress)

        assert saved == scheduled.saved_iteration, (
            f"the run frozen after step {scheduled.frozen_rollout_id} holds iteration {saved}, not the pinned "
            f"{scheduled.saved_iteration}: it saves at another save cadence than it was installed with, so the "
            f"take-over resumes from a checkpoint nobody pinned"
        )
        return HotRestartRecord(
            index=index,
            saved_iteration_at_trigger=scheduled.saved_iteration,
            frozen_rollout_id=scheduled.frozen_rollout_id,
        )

    def _read_settled_saved_iteration(self, *, on: int | None, progress: RunProgress) -> int | None:
        # [FUNCTION DOC] HELPER: Start from the progress's last saved iteration. Up to the settle
        # attempts (3): if it equals the wanted value return it; otherwise sleep the settle
        # interval (2 s) and re-read the tracker, keeping the new value if the re-read succeeded.
        # Returns the last value seen (which may differ from the wanted one).
        saved = progress.last_saved_iteration
        for _ in range(self.tracker_settle_attempts):
            if saved == on:
                return saved
            time.sleep(self.tracker_settle_interval_seconds)
            if (reread := self._reread_saved_iteration()) is not None:
                saved = reread
        return saved

    def _reread_saved_iteration(self) -> int | None:
        # [FUNCTION DOC] HELPER: Read the run progress again and return only its last saved
        # iteration; on any exception log a warning and return nothing.
        try:
            return read_run_progress(
                checkpoint_dir=self.checkpoint_dir, events_dir=self.events_dir
            ).last_saved_iteration
        except Exception:
            logger.warning("Failed to re-read the checkpoint tracker of a run being hot restarted", exc_info=True)
            return None

    def _assert_no_step_lost_outside_take_over(self, progress: RunProgress) -> None:
        # [FUNCTION DOC] HELPER: Monotonicity guard on the event log. Does nothing if no step has
        # finished, or if the run is currently catching up after a take-over (its log was rolled
        # back). Otherwise asserts the finished step is not lower than the highest seen so far, and
        # then updates the highest seen. Mutates that state.
        finished = progress.last_finished_rollout_id
        if finished is None:
            return
        if self._is_catching_up_after_take_over(finished):
            return

        assert self._max_finished_rollout_id is None or finished >= self._max_finished_rollout_id, (
            f"the run had finished step {self._max_finished_rollout_id} and now reports {finished}: outside a "
            f"take-over's rollback an event log only grows, so this run lost work nobody asked it to"
        )
        self._max_finished_rollout_id = finished

    def _is_catching_up_after_take_over(self, finished: int) -> bool:
        # [FUNCTION DOC] HELPER: True when the finished step is explained by the rollback of the
        # latest take-over. False if there are no records or all take-overs are already counted as
        # caught up. If the finished step has reached or passed the highest seen, mark all current
        # take-overs as caught up and return false. Otherwise true exactly when the finished step
        # is at most the latest record's frozen rollout id. Mutates the caught-up count.
        if not self.records or self._num_take_overs_caught_up >= len(self.records):
            return False
        if self._max_finished_rollout_id is not None and finished >= self._max_finished_rollout_id:
            self._num_take_overs_caught_up = len(self.records)
            return False
        return finished <= self.records[-1].frozen_rollout_id

    def _relaunch_on_thread(self, index: int) -> None:
        # [FUNCTION DOC] HELPER: Start a daemon thread named "hot-restart-relaunch-<index>" that
        # calls the relaunch wrapper with the next scheduled frozen rollout id; keep the thread in
        # the list.
        thread = threading.Thread(
            target=self._relaunch,
            args=(self._compute_next_frozen_rollout_id(index),),
            daemon=True,
            name=f"hot-restart-relaunch-{index}",
        )
        self._relaunch_threads.append(thread)
        thread.start()

    def _compute_next_frozen_rollout_id(self, index: int) -> int | None:
        # [FUNCTION DOC] HELPER: The frozen rollout id of the schedule entry after the given index,
        # or nothing if the index is the last one.
        if index + 1 >= len(self.schedule):
            return None
        return self.schedule[index + 1].frozen_rollout_id

    def _relaunch(self, frozen_rollout_id: int | None) -> None:
        # [FUNCTION DOC] HELPER: Thread body. Call the relaunch callable with the next frozen
        # rollout id. If it raises: when a next freeze is scheduled and the error is the
        # replaced-launch exit (a run exited with code 143, because a later take-over replaced
        # it), log it and treat as expected; any other exception (broadest class) is logged as a
        # warning and appended to the failure list.
        try:
            self.relaunch(frozen_rollout_id)
        except BaseException as e:
            if frozen_rollout_id is not None and is_replaced_launch_exit(e):
                logger.info(f"The launch replaced by the next take-over exited {REPLACED_LAUNCH_EXIT_CODE}")
                return
            logger.warning("A hot restart relaunch failed", exc_info=True)
            self._failures.append(e)


def is_replaced_launch_exit(error: BaseException) -> bool:
    # [FUNCTION DOC] True when the error is a run-exited error whose exit code equals 143 (128 plus
    # the termination signal), the code of a launch killed by a take-over.
    return isinstance(error, RunExitedError) and error.exit_code == REPLACED_LAUNCH_EXIT_CODE


@contextmanager
def driving_hot_restarts(driver: HotRestartDriver, *, dump_dir: str) -> Iterator[HotRestartDriver]:
    # [FUNCTION DOC] Context manager wrapping the initial launch. Starts the driver (asserting
    # clean directories) and yields it. If the body raises a run-exited error that is the
    # replaced-launch exit and at least one take-over has been recorded, the error is swallowed and
    # the manager waits until all restarts are triggered; any other error propagates. Always
    # afterwards (even on error): stop collecting, write the evidence file under the dump
    # directory, and assert nothing is still running (this last assertion may mask the original
    # error).
    driver.start()
    try:
        try:
            yield driver
        except RunExitedError as e:
            if not is_replaced_launch_exit(e) or not driver.records:
                raise
            driver.wait_until_all_restarts_triggered()
    finally:
        driver.stop_collecting()
        driver.evidence.write(dump_dir=dump_dir)
        driver.assert_nothing_running()
