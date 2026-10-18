# [MODULE DOC]
# Unit tests for the health checker wired into rollout server cells: when a cell gets a real simple
# checker versus a no-op one (rollout fault tolerance), its configuration (no grace period, failure
# threshold taken from the flag whose shipped default is read from the real argument parser), when
# it is active (cell state and the global activeness flag), which endpoint it probes, how pausing
# the controller discards in-flight probe results and resets failure counters through an epoch,
# how a pause clears a failed health condition at once, and that disposal stops the checker task
# while a cell dropped without disposal complains on collection. A fake clock drives timing; engine
# api clients are fakes. All tests are marked with a fixture that disposes tracked cells.
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Coroutine
from typing import Any
from unittest.mock import MagicMock

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import CellAddrInfo, StatePendingWeights, StateServing, StateUninitialized
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.arguments import get_miles_extra_args_provider
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.api_server.models import CellCondition, TriState
from miles.utils.ft_utils.health_checker import (
    ActiveAndEpoch,
    NoopHealthChecker,
    SimpleHealthChecker,
    SimpleHealthCheckerConfig,
)
from miles.utils.test_utils.clock import FakeClock
from miles.utils.workers.worker_spec import HostAndPort

pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")

_ENDPOINT_CALLS: list[tuple[str, str]] = []


def _make_meta(*, needs_offload: bool = False) -> ServerCellMetadata:
    # [FUNCTION DOC] HELPER: Builds cell metadata for model default, regular worker type, cell id
    # inference-engine-0-0-0, worker inference-engine-0-0-0-0, one GPU, offset 0, no api key,
    # weight updates on, hash pseudo-hash-0, with the given offload flag (default false).
    return ServerCellMetadata(
        model_id="default",
        worker_type="regular",
        cell_id="inference-engine-0-0-0",
        num_gpus_per_engine=1,
        gpu_offset=0,
        sglang_api_key=None,
        worker_name="inference-engine-0-0-0-0",
        needs_offload=needs_offload,
        update_weights=True,
        workers_hash="pseudo-hash-0",
    )


class _StubProvider:
    # [CLASS DOC] HELPER: Stub worker provider answering every lookup with primary 10.0.0.1:30000
    # and gate 10.0.0.1:31000.
    async def get_addrs(self, worker_name: str) -> dict[str, HostAndPort]:
        # [FUNCTION DOC] HELPER: Ignores the worker name and returns the fixed address mapping.
        return dict(
            primary=HostAndPort(host="10.0.0.1", port=30000),
            gate=HostAndPort(host="10.0.0.1", port=31000),
        )


def _make_cell(*, ft_components: list[str], global_activeness: bool = True, **arg_overrides: Any) -> ServerCell:
    # [FUNCTION DOC] HELPER: Builds a real server cell with test arguments carrying the given fault
    # tolerance components and overrides, default metadata, a mock router client, the stub provider
    # and a global activeness callable that always reports the given flag (default true) at epoch 0.
    # The cell is registered with the tracking helper so it is disposed after the test.
    return track_server_cell(
        ServerCell(
            args=make_args(ft_components=ft_components, **arg_overrides),
            meta=_make_meta(),
            router_api_client=MagicMock(),
            provider=_StubProvider(),
            health_checker_activeness=lambda: ActiveAndEpoch(active=global_activeness, epoch=0),
        )
    )


def _addr_info() -> CellAddrInfo:
    # [FUNCTION DOC] HELPER: Returns fixed address info: server url http://10.0.0.1:30000, no
    # bootstrap port, gate url http://10.0.0.1:31000.
    return CellAddrInfo(server_url="http://10.0.0.1:30000", bootstrap_port=None, gate_url="http://10.0.0.1:31000")


class _RecordingApiClient:
    # [CLASS DOC] HELPER: Fake engine api client whose health probe appends the endpoint name and
    # server url to the module-level call list and reports healthy.
    def __init__(self, server_url: str, api_key: str | None = None) -> None:
        # [FUNCTION DOC] HELPER: Stores the server url and optional api key.
        self.server_url = server_url
        self.api_key = api_key

    async def health_generate(self, timeout: float = 5.0) -> bool:
        # [FUNCTION DOC] HELPER: Records the probe (endpoint name health_generate and server url)
        # in the module-level list and returns true. The timeout is ignored.
        _ENDPOINT_CALLS.append(("health_generate", self.server_url))
        return True


class _NoopEngineApiClient:
    # [CLASS DOC] HELPER: Fake engine api client whose health probe, memory release and memory
    # resume all succeed without doing anything.
    def __init__(self, server_url: str, api_key: str | None = None) -> None:
        # [FUNCTION DOC] HELPER: Stores the server url and optional api key.
        self.server_url = server_url
        self.api_key = api_key

    async def health_generate(self, timeout: float = 5.0) -> bool:
        # [FUNCTION DOC] HELPER: Returns true.
        return True

    async def release_memory_occupation(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        pass

    async def resume_memory_occupation(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        pass


class _NoopRouterApiClient:
    # [CLASS DOC] HELPER: Fake router client whose add and remove worker calls do nothing.
    async def add_worker(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Accepts and ignores any keywords.
        pass

    async def remove_worker(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Accepts and ignores any keywords.
        pass


class TestRolloutCellHealthCheckerGating:
    # [CLASS DOC] Tests which health checker a cell is built with and how it is configured.
    async def test_a_cell_gets_no_checker_when_rollout_ft_is_off(self):
        # [FUNCTION DOC] With only the train fault tolerance component, the cell's checker is a
        # no-op checker.
        """Probing engines nobody will heal only produces noise and load."""
        cell = _make_cell(ft_components=["train"])
        assert isinstance(cell._health_checker, NoopHealthChecker)

    async def test_a_cell_gets_a_real_checker_when_rollout_ft_is_on(self):
        # [FUNCTION DOC] With the rollout component, the cell's checker is a simple health checker.
        """Rollout healing needs liveness, so the checker must actually be wired up."""
        cell = _make_cell(ft_components=["rollout"])
        assert isinstance(cell._health_checker, SimpleHealthChecker)

    async def test_the_checker_never_waits_a_grace_period(self):
        # [FUNCTION DOC] The checker's config has a first wait of 0.0 seconds.
        """Activeness flips every weight update window, so a grace period would restart forever."""
        cell = _make_cell(ft_components=["rollout"])
        assert cell._health_checker._config.first_wait == 0.0

    async def test_the_checker_honours_a_configured_failure_threshold(self):
        # [FUNCTION DOC] An argument value of 3 for the rollout health check failure threshold
        # becomes the checker's configured failure threshold.
        """The rollout default is one failed probe, but the flag must still be able to add debouncing."""
        cell = _make_cell(ft_components=["rollout"], rollout_health_check_failure_threshold=3)
        assert cell._health_checker._config.failure_threshold == 3

    async def test_the_checker_debounces_exactly_as_much_as_the_shipped_flag_default(self):
        # [FUNCTION DOC] Builds the real extra-arguments parser, parses a minimal command line (a
        # rollout batch size of 64) to read the shipped default failure threshold, builds a cell from
        # the test arguments' default and asserts the checker uses the same threshold.
        """A fake args default drifting from the parser would let these tests bless a threshold nobody runs."""
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        shipped = parser.parse_args(["--rollout-batch-size", "64"]).rollout_health_check_failure_threshold

        cell = _make_cell(ft_components=["rollout"])

        assert cell._health_checker._config.failure_threshold == shipped


class TestRolloutCellHealthCheckerActiveness:
    # [CLASS DOC] Tests the activeness predicate of the cell's checker.
    @pytest.mark.parametrize(
        "state, expected",
        [
            (StateUninitialized(), False),
            (StatePendingWeights(addr_info=_addr_info()), True),
            (StateServing(addr_info=_addr_info()), True),
        ],
    )
    async def test_only_a_started_engine_is_probed(self, state, expected):
        # [FUNCTION DOC] Parametrized over uninitialized (inactive), pending weights (active) and
        # serving (active). Forces the cell's private state and asserts the checker's activeness
        # predicate reports the expected flag.
        """An engine whose process is not up yet would fail every probe and look unhealthy."""
        cell = _make_cell(ft_components=["rollout"])
        cell._state = state
        assert cell._health_checker._get_activeness().active is expected

    async def test_the_global_flag_can_silence_a_serving_cell(self):
        # [FUNCTION DOC] A serving cell whose global activeness reports false is inactive.
        """During a weight update the engine is offloaded, so probing it would kill a healthy cell."""
        cell = _make_cell(ft_components=["rollout"], global_activeness=False)
        cell._state = StateServing(addr_info=_addr_info())
        assert cell._health_checker._get_activeness().active is False


class TestRolloutCellHealthCheckerProbeEndpoint:
    # [CLASS DOC] Tests which endpoint the checker's probe function calls.
    async def test_the_probe_hits_health_generate_on_the_cells_own_engine(self, monkeypatch):
        # [FUNCTION DOC] Clears the module-level call list, replaces the engine api client class with
        # the recording fake, builds a serving cell, runs the checker's check function once, and
        # asserts exactly one recorded call: health_generate on http://10.0.0.1:30000.
        """A probe aimed at another endpoint or another engine would never notice this engine dying."""
        _ENDPOINT_CALLS.clear()
        monkeypatch.setattr(server_cell_module, "SGLangApiClient", _RecordingApiClient)
        cell = _make_cell(ft_components=["rollout"])
        cell._state = StateServing(addr_info=_addr_info())

        await cell._health_checker._check_fn()

        assert _ENDPOINT_CALLS == [("health_generate", "http://10.0.0.1:30000")]


class TestRolloutCellHealthCheckerPause:
    # [CLASS DOC] Tests how pausing the inference controller (through offload and resume through
    # prepare eval) affects the cell's checker, which uses an epoch to tell stale probe results.
    async def test_pausing_the_controller_makes_the_predicate_false_for_every_cell(self, monkeypatch):
        # [FUNCTION DOC] With a controller and a serving cell, the activeness predicate is true;
        # after the controller offloads it is false.
        """The whole point of the pause is that no cell is probed inside the protected window."""
        controller, cell = await _make_controller_with_serving_cell(monkeypatch)
        assert cell._health_checker._get_activeness().active is True

        await controller.offload()

        assert cell._health_checker._get_activeness().active is False

    async def test_a_probe_that_lands_after_the_pause_is_discarded(self, monkeypatch):
        # [FUNCTION DOC] Builds a fake-clock checker for the cell whose check function blocks until
        # released and then fails, with results collected in a list. Starts the checker, lets the
        # first wait elapse (100 s) and waits until the probe started. Offloads through the
        # controller (pause), then releases the probe. Asserts no result was reported and the
        # consecutive failure counter is still zero. Stops the checker.
        """A probe launched before the pause must not publish a failure about the protected window."""
        results: list[bool] = []
        probe_started = asyncio.Event()
        release_probe = asyncio.Event()

        controller, cell = await _make_controller_with_serving_cell(monkeypatch)
        checker, clock = _make_fake_clock_checker(
            cell, check_fn=_gated_check_fn(probe_started, release_probe), on_result=results.append
        )
        checker.start()
        await _settle(clock)
        await clock.elapse(100.0)
        await asyncio.wait_for(probe_started.wait(), timeout=1)

        await controller.offload()
        release_probe.set()
        await _settle(clock)

        assert results == []
        assert checker._consecutive_failures == 0
        checker.stop()

    async def test_a_probe_that_spans_a_whole_pause_resume_window_is_discarded(self, monkeypatch):
        # [FUNCTION DOC] Same setup as the pause test, but between starting and releasing the probe
        # the controller both offloads (pause) and prepares an eval (resume), so the cell is active
        # again when the probe lands. Asserts the result is still discarded (no results, zero failure
        # count) because the epoch changed. Stops the checker.
        """The window is closed again when the probe lands, so only the epoch tells the result is stale."""
        results: list[bool] = []
        probe_started = asyncio.Event()
        release_probe = asyncio.Event()

        controller, cell = await _make_controller_with_serving_cell(monkeypatch)
        checker, clock = _make_fake_clock_checker(
            cell, check_fn=_gated_check_fn(probe_started, release_probe), on_result=results.append
        )
        checker.start()
        await _settle(clock)
        await clock.elapse(100.0)
        await asyncio.wait_for(probe_started.wait(), timeout=1)

        await controller.offload()
        await controller.prepare_eval()
        release_probe.set()
        await _settle(clock)

        assert results == []
        assert checker._consecutive_failures == 0
        checker.stop()


class TestRolloutCellActiveAndEpoch:
    # [CLASS DOC] Tests the cell's private active-and-epoch reading used by its checker.
    async def test_reading_the_active_and_epoch_twice_returns_the_very_same_value(self, monkeypatch):
        # [FUNCTION DOC] With a serving cell under a controller, two consecutive reads are equal and
        # equal active true at epoch 0.
        """A pull that samples into a tracker of its own degrades the epoch back into a boolean read."""
        _, cell = await _make_controller_with_serving_cell(monkeypatch)

        first = cell._get_health_checker_active_and_epoch()
        second = cell._get_health_checker_active_and_epoch()

        assert first == second == ActiveAndEpoch(active=True, epoch=0)

    async def test_a_pause_resume_window_completed_between_two_polls_resets_the_failure_counter(self, monkeypatch):
        # [FUNCTION DOC] A fake-clock checker with the default always-failing probe (failure
        # threshold 3, interval 10 s, first wait 100 s). After the first wait and one more interval,
        # two failures are counted. The controller then offloads and prepares an eval, and after one
        # more interval the failure counter is back to zero. Stops the checker.
        """The controller opens and closes the whole window while the loop sleeps, so only the epoch reveals it."""
        controller, cell = await _make_controller_with_serving_cell(monkeypatch)
        checker, clock = _make_fake_clock_checker(cell)

        checker.start()
        await _settle(clock)
        await clock.elapse(100.0)
        await clock.elapse(10.0)
        assert checker._consecutive_failures == 2

        await controller.offload()
        await controller.prepare_eval()
        await clock.elapse(10.0)

        assert checker._consecutive_failures == 0
        checker.stop()


class TestRolloutCellHealthConditionDuringPause:
    # [CLASS DOC] Tests the Healthy condition reported in the cell's status while paused.
    async def test_pausing_the_controller_clears_a_failed_health_condition_at_once(self, monkeypatch):
        # [FUNCTION DOC] Installs a fake-clock checker as the cell's checker and lets it fail three
        # times, so the cell's Healthy condition is false with reason HealthCheckFailed. Then, under
        # the controller's lock, pauses health monitoring (no model id) and asserts the Healthy
        # condition becomes unknown with reason HealthCheckUnknown. Stops the checker.
        """A cell still advertised as HealthCheckFailed while it is being offloaded gets recycled a second time."""
        controller, cell = await _make_controller_with_serving_cell(monkeypatch)
        checker, clock = _make_fake_clock_checker(cell)
        cell._health_checker = checker
        checker.start()
        await _settle(clock)
        await clock.elapse(100.0)
        await clock.elapse(10.0)
        await clock.elapse(10.0)
        assert _healthy_condition(cell) == CellCondition.healthy(TriState.FALSE, reason="HealthCheckFailed")

        async with controller.context_lock:
            await controller._health_monitoring_pause(None)

        assert _healthy_condition(cell) == CellCondition.healthy(TriState.UNKNOWN, reason="HealthCheckUnknown")
        checker.stop()


class TestRolloutCellHealthCheckerDisposal:
    # [CLASS DOC] Tests the cell's disposal and collection behavior regarding its checker.
    async def test_disposing_a_cell_stops_its_checker(self):
        # [FUNCTION DOC] A new rollout cell has a running checker task; after dispose the task
        # reference is cleared.
        """A removed cell whose loop keeps polling leaks the task and the whole cell it closes over."""
        cell = _make_cell(ft_components=["rollout"])
        assert cell._health_checker._task is not None

        await cell.dispose()

        assert cell._health_checker._task is None

    async def test_a_cell_dropped_without_dispose_complains_on_collection(self):
        # [FUNCTION DOC] Calling the cell's finalizer directly on an undisposed cell raises an
        # assertion error mentioning "without dispose".
        """Silently dropping a cell leaks its checker task forever, so the mistake must be loud."""
        cell = _make_cell(ft_components=["rollout"])

        with pytest.raises(AssertionError, match="without dispose"):
            cell.__del__()

    async def test_a_disposed_cell_is_collected_quietly(self):
        # [FUNCTION DOC] After dispose, calling the finalizer directly raises nothing.
        """The collection guard must not fire on the normal teardown path."""
        cell = _make_cell(ft_components=["rollout"])
        await cell.dispose()

        cell.__del__()


async def _make_controller_with_serving_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[InferenceController, ServerCell]:
    # [FUNCTION DOC] HELPER: Replaces the engine api client class with the no-op fake. Builds a
    # controller without its constructor (colocated arguments with rollout fault tolerance, a named
    # context lock), a real rollout server on that lock with the stub provider, and registers it as
    # the default server. Under the lock adds an offloaded cell; takes that cell, registers it with
    # the tracking helper, gives it a no-op router client and forces its state to serving with the
    # fixed address info. Returns the controller and the cell.
    monkeypatch.setattr(server_cell_module, "SGLangApiClient", _NoopEngineApiClient)
    args: Any = make_args(ft_components=["rollout"], colocate=True)
    controller = InferenceController.__new__(InferenceController)
    controller.args = args
    controller.context_lock = ContextLock("InferenceController")

    srv = RolloutServer(
        server_cells={},
        args=args,
        context_lock=controller.context_lock,
        engine_provider=_StubProvider(),
    )
    controller.servers = {"default": srv}

    async with controller.context_lock:
        await srv.add_cell(_make_meta(needs_offload=True))

    cell: ServerCell = track_server_cell(srv.server_cells["inference-engine-0-0-0"])
    cell.router_api_client = _NoopRouterApiClient()
    cell._state = StateServing(addr_info=_addr_info())
    return controller, cell


def _healthy_condition(cell: ServerCell) -> CellCondition:
    # [FUNCTION DOC] HELPER: Returns the single Healthy condition from the cell's status; fails
    # (unpacking error) if there is not exactly one.
    (condition,) = [condition for condition in cell.cell_status().conditions if condition.type == "Healthy"]
    return condition


def _make_fake_clock_checker(
    cell: ServerCell, *, check_fn: Any = None, on_result: Any = None
) -> tuple[SimpleHealthChecker, FakeClock]:
    # [FUNCTION DOC] HELPER: Stops the cell's own checker, then builds a new simple health checker
    # named after the cell id, using the given check function (default one that always raises
    # "engine down"), the cell's active-and-epoch reading, the optional result callback, a config of
    # interval 10 s, timeout 5 s, first wait 100 s and failure threshold 3, and a new fake clock.
    # The checker is returned unstarted together with the clock. The cell is not given the new
    # checker unless the caller assigns it.
    cell._health_checker.stop()

    async def _failing_check() -> None:
        # [FUNCTION DOC] HELPER: Always raises a runtime error "engine down".
        raise RuntimeError("engine down")

    clock = FakeClock()
    checker = SimpleHealthChecker(
        name=f"rollout-cell-{cell.meta.cell_id}",
        check_fn=check_fn or _failing_check,
        get_activeness=cell._get_health_checker_active_and_epoch,
        on_result=on_result,
        config=SimpleHealthCheckerConfig(interval=10.0, timeout=5.0, first_wait=100.0, failure_threshold=3),
        clock=clock,
    )
    return checker, clock


def _gated_check_fn(started: asyncio.Event, release: asyncio.Event) -> Callable[[], Coroutine[Any, Any, None]]:
    # [FUNCTION DOC] HELPER: Returns a check function that signals the started event, waits for the
    # release event, and then raises a runtime error "engine down".
    async def check_fn() -> None:
        # [FUNCTION DOC] HELPER: See the enclosing function.
        started.set()
        await release.wait()
        raise RuntimeError("engine down")

    return check_fn


async def _settle(clock: FakeClock) -> None:
    # [FUNCTION DOC] HELPER: Yields to the event loop up to 1000 times until the fake clock has at
    # least one pending sleeper, then returns. Returns silently after 1000 yields even if none is
    # pending.
    for _ in range(1000):
        if clock.pending_count >= 1:
            return
        await asyncio.sleep(0)
