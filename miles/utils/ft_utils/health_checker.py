# [MODULE DOC]
# Health checking for fault tolerance. Provides a config model with command line registration,
# an "activeness" tracker (active flag plus an epoch counter bumped on each change), an abstract
# checker interface, a periodic asynchronous checker that probes a coroutine function with a timeout
# and debounces failures, and a no-op checker. Status is tri-state (true, false, unknown): unknown
# whenever the target is inactive or the activeness epoch changed since the status was computed.
from __future__ import annotations

import abc
import argparse
import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any, NamedTuple

from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.test_utils.clock import Clock, RealClock
from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)


class SimpleHealthCheckerConfig(StrictBaseModel):
    # [CLASS DOC] Settings of the simple checker: seconds between checks, per-check timeout in
    # seconds, initial grace period in seconds after each (re)activation, and the number of
    # consecutive failures required before reporting unhealthy.
    interval: float
    timeout: float
    first_wait: float
    failure_threshold: int

    @staticmethod
    def add_arguments(
        parser: argparse.ArgumentParser,
        *,
        prefix: str,
        interval_default: float = 10.0,
        timeout_default: float = 10.0,
        first_wait_default: float = 300.0,
        failure_threshold_default: int = 3,
    ) -> None:
        # [FUNCTION DOC] Registers four command line options on the parser, named with the given
        # prefix: interval, timeout, first-wait and failure-threshold (float, float, float, int),
        # with the given defaults (10, 10, 300 seconds, and 3 failures). Mutates the parser.
        parser.add_argument(
            f"--{prefix}-interval",
            type=float,
            default=interval_default,
            help=f"Interval in seconds between {prefix} health checks.",
        )
        parser.add_argument(
            f"--{prefix}-timeout",
            type=float,
            default=timeout_default,
            help=f"Timeout in seconds for a single {prefix} health check RPC.",
        )
        parser.add_argument(
            f"--{prefix}-first-wait",
            type=float,
            default=first_wait_default,
            help=(
                f"Initial grace period (seconds) before starting {prefix} health checks, re-armed on every "
                "resume. This allows time for model compilation and initialization; increase it "
                "significantly when using deepgemm."
            ),
        )
        parser.add_argument(
            f"--{prefix}-failure-threshold",
            type=int,
            default=failure_threshold_default,
            help=(
                f"Number of consecutive failed {prefix} checks before reporting unhealthy. "
                "Debounces transient RPC blips so a single hiccup does not recycle a live cell."
            ),
        )

    @staticmethod
    def from_args(args: object, *, prefix: str) -> SimpleHealthCheckerConfig:
        # [FUNCTION DOC] Builds the config from a parsed arguments object by reading the four
        # attributes named after the prefix (dashes converted to underscores). Raises an attribute
        # error if one is missing.
        attr_prefix = prefix.replace("-", "_")
        return SimpleHealthCheckerConfig(
            interval=getattr(args, f"{attr_prefix}_interval"),
            timeout=getattr(args, f"{attr_prefix}_timeout"),
            first_wait=getattr(args, f"{attr_prefix}_first_wait"),
            failure_threshold=getattr(args, f"{attr_prefix}_failure_threshold"),
        )


class ActiveAndEpoch(NamedTuple):
    # [CLASS DOC] Snapshot of activeness: whether active, and the epoch (number of changes so far).
    active: bool
    epoch: int


class ActivenessTracker:
    # [CLASS DOC] Holds the current activeness snapshot; the epoch grows by one every time the
    # active flag actually changes, letting observers detect that a pause/resume cycle happened.
    def __init__(self, *, active: bool) -> None:
        # [FUNCTION DOC] Starts with the given flag and epoch zero.
        self._state = ActiveAndEpoch(active=active, epoch=0)

    def get(self) -> ActiveAndEpoch:
        # [FUNCTION DOC] Returns the current snapshot.
        return self._state

    def bump_active(self, active: bool) -> None:
        # [FUNCTION DOC] Sets the flag; no change (and no epoch increase) if it equals the current
        # flag, otherwise stores the new flag with the epoch increased by one.
        if active == self._state.active:
            return
        self._state = ActiveAndEpoch(active=active, epoch=self._state.epoch + 1)


class BaseHealthChecker(abc.ABC):
    # [CLASS DOC] Abstract health checker interface: a status property plus start and stop.
    @property
    @abc.abstractmethod
    def status(self) -> TriState:
        # [FUNCTION DOC] Abstract property: current tri-state health.
        ...

    @abc.abstractmethod
    def start(self) -> None:
        # [FUNCTION DOC] Abstract: begin checking.
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        # [FUNCTION DOC] Abstract: stop checking.
        ...


class SimpleHealthChecker(BaseHealthChecker):
    # [CLASS DOC] Runs an asyncio task loop. Each iteration reads the activeness snapshot; when the
    # snapshot changed since the previous iteration it handles pause or resume (resume re-arms the
    # first wait, clears status to unknown and failure count). While active and the first wait is
    # pending, sleeps for the first-wait period (via the injected clock) and re-evaluates. While
    # active otherwise, runs one probe, and publishes its result unless activeness changed
    # (epoch differs or became inactive) during the probe, in which case the result is discarded.
    # Then sleeps for the interval. Status becomes true on any success, and false only after the
    # configured number of consecutive failures. All transitions are logged in structured form.
    # KNOWN ISSUE: the class docstring says the first wait applies after each transition back to
    # active; the code also applies it for the first activation at start, which is consistent, but
    # the stop method does not re-arm it, so a stop followed by start keeps the previous state.
    """Periodic async health checker. Calls *check_fn*; reports result via *on_result*.

    Probing is driven by *get_activeness*, read once per loop before deciding to probe.
    After each transition back to active, waits ``first_wait`` seconds before the first check.
    """

    def __init__(
        self,
        *,
        name: str,
        check_fn: Callable[[], Coroutine[Any, Any, None]],
        get_activeness: Callable[[], ActiveAndEpoch],
        on_result: Callable[[bool], None] | None = None,
        config: SimpleHealthCheckerConfig,
        clock: Clock | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores: a name for logs; the async check function (raises on failure); the
        # activeness getter; an optional callback receiving each published success flag; the
        # config; and a clock (default real clock; injectable for tests). Initial state: status
        # unknown, last seen activeness inactive at epoch zero, first wait pending, zero failures,
        # no tasks.
        self._name = name
        self._check_fn = check_fn
        self._get_activeness = get_activeness
        self._on_result = on_result
        self._config = config
        self._clock = clock or RealClock()

        self._status = TriState.UNKNOWN
        self._active_and_epoch = ActiveAndEpoch(active=False, epoch=0)
        self._need_first_wait: bool = True
        self._consecutive_failures: int = 0
        self._task: asyncio.Task[None] | None = None
        self._probe_task: asyncio.Task[None] | None = None

    @property
    def status(self) -> TriState:
        # [FUNCTION DOC] Property: unknown if the target is currently inactive or the current
        # activeness snapshot differs from the one the loop last processed (loop has not yet caught
        # up); otherwise the stored status.
        active_and_epoch = self._get_activeness()
        if not active_and_epoch.active or active_and_epoch != self._active_and_epoch:
            return TriState.UNKNOWN
        return self._status

    def start(self) -> None:
        # [FUNCTION DOC] Idempotent: if no loop task exists, logs and creates the loop task on the
        # running event loop (must be called from a running loop).
        if self._task is not None:
            return
        log_structured(logger.info, tag="ft", op="health", phase="start", name=self._name)
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        # [FUNCTION DOC] Cancels the loop task and any in-flight probe task (each at most once),
        # clears them, and sets the status to unknown. Safe to call repeatedly.
        if self._task is not None:
            log_structured(logger.info, tag="ft", op="health", phase="stop", name=self._name)
            self._task.cancel()
            self._task = None
        if self._probe_task is not None:
            self._probe_task.cancel()
            self._probe_task = None
        self._status = TriState.UNKNOWN

    def _on_paused(self) -> None:
        # [FUNCTION DOC] HELPER: Logs and resets status to unknown.
        log_structured(logger.info, tag="ft", op="health", phase="pause", name=self._name)
        self._status = TriState.UNKNOWN

    def _on_resumed(self) -> None:
        # [FUNCTION DOC] HELPER: Logs, re-arms the first wait, resets status to unknown and the
        # consecutive failure count to zero.
        log_structured(logger.info, tag="ft", op="health", phase="resume", name=self._name)
        self._need_first_wait = True
        self._status = TriState.UNKNOWN
        self._consecutive_failures = 0

    async def _loop(self) -> None:
        # [FUNCTION DOC] HELPER: Infinite loop described in the class documentation. Note the
        # first-wait branch skips the interval sleep (it continues straight to re-read activeness),
        # and a probe result is discarded when activeness changed during the probe.
        while True:
            active_and_epoch = self._get_activeness()
            active = active_and_epoch.active
            if active_and_epoch != self._active_and_epoch:
                self._active_and_epoch = active_and_epoch
                if active:
                    self._on_resumed()
                else:
                    self._on_paused()

            if active and self._need_first_wait:
                self._need_first_wait = False
                log_structured(
                    logger.info,
                    tag="ft",
                    op="health",
                    phase="first_wait",
                    name=self._name,
                    wait_s=self._config.first_wait,
                )
                await self._clock.sleep(self._config.first_wait)
                continue

            if active:
                success = await self._run_probe()
                active_and_epoch_now = self._get_activeness()
                if not active_and_epoch_now.active or active_and_epoch_now.epoch != active_and_epoch.epoch:
                    log_structured(logger.info, tag="ft", op="health", phase="probe_discarded", name=self._name)
                else:
                    self._publish_result(success=success)

            await self._clock.sleep(self._config.interval)

    async def _run_probe(self) -> bool:
        # [FUNCTION DOC] HELPER: Runs the check function as a task with the configured timeout
        # (the task is cancelled on timeout). Returns true if it completed without error; any
        # ordinary exception (including timeout) is logged as an error and returns false. The probe
        # task reference is cleared in all cases. Cancellation of the loop itself propagates.
        self._probe_task = asyncio.create_task(self._check_fn())

        try:
            await asyncio.wait_for(self._probe_task, timeout=self._config.timeout)
            return True
        except Exception:
            log_structured(logger.error, tag="ft", op="health", phase="check_failed", name=self._name, exc_info=True)
            return False
        finally:
            self._probe_task = None

    def _publish_result(self, *, success: bool) -> None:
        # [FUNCTION DOC] HELPER: Success sets status true and clears the failure count. Failure
        # increments the count and sets status false only once the count reaches the threshold
        # (below it the previous status is retained). Logs the poll and any status change, then
        # calls the result callback with the raw success flag (not the debounced status); callback
        # exceptions are logged and swallowed.
        prev_status = self._status
        if success:
            self._consecutive_failures = 0
            self._status = TriState.TRUE
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._config.failure_threshold:
                self._status = TriState.FALSE

        log_structured(
            logger.info,
            tag="ft",
            op="health",
            phase="poll",
            name=self._name,
            ok=success,
            status=self._status.value,
            consecutive_failures=self._consecutive_failures,
        )

        if prev_status != self._status:
            log_structured(
                logger.info,
                tag="ft",
                op="health",
                phase="status_change",
                name=self._name,
                from_status=prev_status.value,
                to_status=self._status.value,
                consecutive_failures=self._consecutive_failures,
            )

        if self._on_result is not None:
            try:
                self._on_result(success)
            except Exception:
                log_structured(
                    logger.error,
                    tag="ft",
                    op="health",
                    phase="on_result_failed",
                    name=self._name,
                    exc_info=True,
                )


class NoopHealthChecker(BaseHealthChecker):
    # [CLASS DOC] Checker that never checks: status always unknown; records that stop was called.
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the stopped flag to false.
        self.stopped: bool = False

    @property
    def status(self) -> TriState:
        # [FUNCTION DOC] Property: always unknown.
        return TriState.UNKNOWN

    def start(self) -> None:
        # [FUNCTION DOC] Does nothing.
        pass

    def stop(self) -> None:
        # [FUNCTION DOC] Sets the stopped flag to true.
        self.stopped = True
