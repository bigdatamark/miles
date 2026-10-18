# [MODULE DOC] Unit tests for the registration reporter (RegistrationReporter, its worker wrapper, the
# debounced interval trigger and the exit-on-stop callback). All tests run in-process with fakes: a
# fake worker provider that reports a fixed set of engine cells, fake hub endpoints that record the
# registration snapshots they receive (some hang, fail once, or behave like a restarted hub), and a
# fake trigger that fires after a one millisecond sleep. Covered behaviors: snapshot contents (whole
# membership, names, addresses, run uuid, generation, worker type), strictly increasing sequence
# numbers, wake-ups on cell changes, the reporter worker's construction and process-exit behavior,
# the run lifecycle ordering (provider init, hub readiness wait, then watching cells), ingest
# timeout and error handling, and how the reporter drops the hub's boot-uuid pin only after the hub
# signalled that it restarted.
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import random
import threading
from argparse import Namespace
from unittest.mock import patch

import pytest

from miles.utils.workers.naming import _worker_name_of_cell, compute_cell_id
from miles.utils.workers.registration.models import RegistrationSnapshot
from miles.utils.workers.registration.reporter import (
    SNAPSHOT_INTERVAL_SECONDS,
    RegistrationReporter,
    RegistrationReporterWorker,
    _DebouncedIntervalTrigger,
    _exit_on_reporter_stop,
)
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

_POOL_ID = "inference-engine-0-0"
_REPORTER_ID = "miles-run-r1-inference"
_RUN_UUID = "run-uuid-1"


def _cell_id(cell_index: int) -> str:
    # [FUNCTION DOC] HELPER: Builds the cell id of the given cell index inside the test pool by
    # delegating to the shared cell id computation with the fixed test pool id. Returns a string.
    return compute_cell_id(pool_id=_POOL_ID, cell_index=cell_index)


class _FakeEngineProvider(BaseWorkerProvider):
    # [CLASS DOC] HELPER: A worker provider fake that serves a fixed list of engine cells. It remembers
    # the reconcile callback it was given and whether its watch was stopped, so tests can drive further
    # cell changes and check cleanup. Every cell has exactly one worker with a fixed address.
    def __init__(self, *, cell_indices: list[int], worker_type: str = "regular", generation: int = 0) -> None:
        # [FUNCTION DOC] HELPER: Stores a copy of the cell indices, the worker type that goes into every
        # cell's meta, and the generation reported for every worker. Initializes the reconcile callback
        # to nothing and the stopped flag to false. Does not call the base constructor.
        self.cell_indices = list(cell_indices)
        self.worker_type = worker_type
        self.generation = generation
        self.reconcile: CellReconcileFn | None = None
        self.stopped = False

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] HELPER: Not used by the reporter; always raises a not-implemented error.
        raise NotImplementedError

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        # [FUNCTION DOC] HELPER: Returns, for each requested cell id in order, a list holding one worker
        # info. The worker is named from the cell id, carries the provider's generation, one "primary"
        # address at host 10.0.0.5 port 8000, gpu id 0 and no worker class.
        return [
            [
                WorkerInfo(
                    name=_worker_name_of_cell(cell_id),
                    generation=self.generation,
                    self_addrs={"primary": HostAndPort(host="10.0.0.5", port=8000)},
                    gpu_ids=[0],
                    worker_class=None,
                )
            ]
            for cell_id in cell_ids
        ]

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        # [FUNCTION DOC] HELPER: Saves the reconcile callback, then awaits it once per configured cell
        # index with that cell's id and a cell info built with the provider's worker type. Returns an
        # async stop function that only sets the stopped flag.
        self.reconcile = reconcile
        for cell_index in self.cell_indices:
            await reconcile(_cell_id(cell_index), _cell_info(cell_index, worker_type=self.worker_type))

        async def _stop() -> None:
            # [FUNCTION DOC] HELPER: Marks the provider as stopped; no other effect.
            self.stopped = True

        return _stop


class _FakeHub:
    # [CLASS DOC] HELPER: A hub endpoint fake that records everything the reporter sends: every
    # registration snapshot, the timeout of every readiness wait, and whether each readiness wait
    # allowed the server uuid to change (that is, whether the pin was dropped).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with three empty recording lists: snapshots, ready timeouts and
        # pin drop flags.
        self.snapshots: list[RegistrationSnapshot] = []
        self.ready_timeouts: list[float] = []
        self.pin_drops: list[bool] = []

    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None:
        # [FUNCTION DOC] HELPER: Records the timeout and the allow-server-uuid-change flag and returns
        # immediately, so the hub is always ready.
        self.ready_timeouts.append(timeout)
        self.pin_drops.append(allow_server_uuid_change)

    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        # [FUNCTION DOC] HELPER: Appends the received snapshot to the recorded list and returns.
        self.snapshots.append(snapshot)


class _HungHub(_FakeHub):
    # [CLASS DOC] HELPER: A hub whose ingest call never answers. It exposes an event that is set when
    # the hung call is cancelled, so a test can verify the reporter cancels instead of abandoning it.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Runs the base constructor and adds the cancelled event (initially unset).
        super().__init__()
        self.cancelled = asyncio.Event()

    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        # [FUNCTION DOC] HELPER: Waits forever on a future that is never resolved; when the wait ends
        # (only by cancellation) it sets the cancelled event. The snapshot is not recorded.
        try:
            await asyncio.Future()
        finally:
            self.cancelled.set()


class _FailOnceHub(_FakeHub):
    # [CLASS DOC] HELPER: A hub that rejects the first snapshot with an error and accepts later ones. It
    # exposes an event that is set when a second snapshot has been accepted.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Runs the base constructor and adds the accepted event (initially unset).
        super().__init__()
        self.accepted = asyncio.Event()

    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        # [FUNCTION DOC] HELPER: Records the snapshot, then raises a runtime error saying the first
        # ingest failed if it was the first snapshot; for every later snapshot it sets the accepted
        # event instead.
        self.snapshots.append(snapshot)
        if len(self.snapshots) == 1:
            raise RuntimeError("first ingest failed")
        self.accepted.set()


def _cell_info(cell_index: int, *, workers_hash: str = "hash-1", worker_type: str = "regular") -> CellInfo:
    # [FUNCTION DOC] HELPER: Builds an alive cell info for the given cell index in the test pool, with
    # one worker name derived from the cell id, the given workers hash, and a meta mapping holding the
    # model id "default" and the given worker type.
    return CellInfo(
        cell_id=_cell_id(cell_index),
        pool_id=_POOL_ID,
        alive=True,
        worker_names=[_worker_name_of_cell(_cell_id(cell_index))],
        workers_hash=workers_hash,
        meta=dict(model_id="default", worker_type=worker_type),
    )


class _FakeTrigger:
    # [CLASS DOC] HELPER: A trigger replacement that counts notifications and fires almost
    # immediately, so a reporter test never waits on a real interval. It exposes the same interval
    # attribute as the real trigger (set to the default snapshot interval).
    """Fires as soon as anything asks, so a reporter test never waits on a real interval."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the interval to the default snapshot interval and the notification
        # counter to zero.
        self.interval_seconds = SNAPSHOT_INTERVAL_SECONDS
        self.notified = 0

    def notify(self) -> None:
        # [FUNCTION DOC] HELPER: Increments the notification counter.
        self.notified += 1

    async def wait(self) -> None:
        # [FUNCTION DOC] HELPER: Sleeps for one millisecond and returns. The real timer turn is needed
        # because returning without yielding would starve the event loop: the reporter would send in a
        # tight cycle and the cancellation that ends the test would never get a turn.
        # answering without a timer turn starves the loop this runs in: the reporter sends in a tight
        # cycle and nothing else gets a turn, including the cancellation that ends the test.
        await asyncio.sleep(0.001)


def _reporter(*, provider: _FakeEngineProvider, hub_endpoint: _FakeHub):
    # [FUNCTION DOC] HELPER: Builds a registration reporter with the fixed test run uuid and reporter
    # id, the given hub endpoint and worker provider, and a fake trigger injected through the private
    # trigger parameter. Returns the reporter; nothing is started.
    return RegistrationReporter(
        run_uuid=_RUN_UUID,
        reporter_id=_REPORTER_ID,
        hub_endpoint=hub_endpoint,
        worker_provider=provider,
        _trigger=_FakeTrigger(),
    )


async def _synced(
    *,
    cell_indices: tuple[int, ...] = (0,),
    worker_type: str = "regular",
    hub_endpoint: _FakeHub | None = None,
    generation: int = 0,
) -> tuple[RegistrationReporter, _FakeEngineProvider, _FakeHub]:
    # [FUNCTION DOC] HELPER: Builds a fake provider (given cell indices, worker type, generation), a
    # fresh fake hub unless one is passed, and a reporter over them. Then lets the provider watch
    # cells with the reporter's private observe callback, so the reporter knows all cells without
    # running its main loop. Returns the reporter, the provider and the hub endpoint as a triple.
    provider = _FakeEngineProvider(cell_indices=list(cell_indices), worker_type=worker_type, generation=generation)
    hub_endpoint = hub_endpoint if hub_endpoint is not None else _FakeHub()
    reporter = _reporter(provider=provider, hub_endpoint=hub_endpoint)
    await provider.watch_cells(reporter._observe)
    return reporter, provider, hub_endpoint


class TestSnapshotContents:
    # [CLASS DOC] Tests what a single sent snapshot contains, using a synced reporter and a recording
    # fake hub; each test sends once through the private send-once method and inspects the first
    # recorded snapshot.
    async def test_a_snapshot_carries_every_cell_of_this_deployment(self):
        # [FUNCTION DOC] Syncs a reporter with cells 0 and 1, sends one snapshot, and asserts the single
        # received snapshot lists both cell ids in index order.
        """The run replaces this deployment's membership with the snapshot, so it has to be the whole one."""
        reporter, _provider, hub_endpoint = await _synced(cell_indices=(0, 1))

        await reporter._send_once()

        (snapshot,) = hub_endpoint.snapshots
        assert [cell.info.cell_id for cell in snapshot.cells] == [_cell_id(0), _cell_id(1)]

    async def test_it_reports_the_names_its_own_deployment_gave_its_cells(self):
        # [FUNCTION DOC] Syncs one cell, sends once, and asserts the reported cell keeps the pool id
        # of the deployment and its one worker keeps the name derived from the cell id (no renaming).
        """Those pool ids are born naming this deployment, so a name rewritten here would name nothing."""
        reporter, _provider, hub_endpoint = await _synced()

        await reporter._send_once()

        (cell,) = hub_endpoint.snapshots[0].cells
        assert cell.info.pool_id == _POOL_ID
        assert [worker.name for worker in cell.workers] == [_worker_name_of_cell(_cell_id(0))]

    async def test_it_reports_the_addresses_its_own_deployment_observed(self):
        # [FUNCTION DOC] Syncs one cell, sends once, and asserts the worker's primary self address
        # equals the host and port the provider observed.
        """The run calls the engine there, so an address computed anywhere else would be a guess."""
        reporter, _provider, hub_endpoint = await _synced()

        await reporter._send_once()

        (cell,) = hub_endpoint.snapshots[0].cells
        assert cell.workers[0].self_addrs["primary"] == HostAndPort(host="10.0.0.5", port=8000)

    async def test_it_names_the_run_it_was_deployed_for(self):
        # [FUNCTION DOC] Syncs one cell, sends once, and asserts the snapshot carries the run uuid the
        # reporter was constructed with.
        """The deployment script gives every component the same --run-uuid, and the run refuses any other."""
        reporter, _provider, hub_endpoint = await _synced()

        await reporter._send_once()

        assert hub_endpoint.snapshots[0].run_uuid == _RUN_UUID

    async def test_it_reports_the_generation_its_own_deployment_observed(self):
        # [FUNCTION DOC] Syncs one cell on a provider with generation 3, sends once, and asserts the
        # reported worker carries generation 3.
        """A restarted engine is a new engine to the run, and only its own deployment counts the restarts."""
        reporter, _provider, hub_endpoint = await _synced(generation=3)

        await reporter._send_once()

        (cell,) = hub_endpoint.snapshots[0].cells
        assert cell.workers[0].generation == 3

    async def test_it_reports_a_prefill_or_decode_deployment_like_any_other(self):
        # [FUNCTION DOC] Syncs one cell whose worker type is prefill, sends once, and asserts the
        # worker type in the cell's meta mapping reaches the snapshot unchanged.
        """A deployment of split engines announces them the same way, and the run addresses them the same way."""
        reporter, _provider, hub_endpoint = await _synced(worker_type="prefill")

        await reporter._send_once()

        (cell,) = hub_endpoint.snapshots[0].cells
        assert cell.info.meta["worker_type"] == "prefill"


class TestSnapshotSequencing:
    # [CLASS DOC] Tests the sequence numbering and whole-membership semantics across consecutive
    # snapshots from one synced reporter.
    async def test_every_snapshot_carries_a_higher_sequence_than_the_last(self):
        # [FUNCTION DOC] Sends two snapshots and asserts the recorded sequence numbers are exactly 1 then
        # 2 (starting at one, increasing by one per send).
        """The run drops a snapshot that arrived late, and only the sequence number tells it which one that is."""
        reporter, _provider, hub_endpoint = await _synced()

        await reporter._send_once()
        await reporter._send_once()

        assert [snapshot.sequence_number for snapshot in hub_endpoint.snapshots] == [1, 2]

    async def test_an_unchanged_membership_is_sent_whole_again(self):
        # [FUNCTION DOC] Sends twice without changing cells and asserts both snapshots list the full
        # cell membership (cell 0), proving there is no delta encoding.
        """Every tick declares the whole membership, so the run never depends on a message it may have missed."""
        reporter, _provider, hub_endpoint = await _synced()

        await reporter._send_once()
        await reporter._send_once()

        assert [cell.info.cell_id for cell in hub_endpoint.snapshots[0].cells] == [_cell_id(0)]
        assert [cell.info.cell_id for cell in hub_endpoint.snapshots[1].cells] == [_cell_id(0)]

    async def test_a_changed_membership_is_sent_whole(self):
        # [FUNCTION DOC] Sends once, then reconciles a new cell 1 through the provider's saved callback
        # and sends again; asserts the second snapshot lists both cells 0 and 1.
        """A cell that appeared or went away has to reach the run in the very next snapshot."""
        reporter, provider, hub_endpoint = await _synced()
        await reporter._send_once()

        await provider.reconcile(_cell_id(1), _cell_info(1))
        await reporter._send_once()

        assert [cell.info.cell_id for cell in hub_endpoint.snapshots[1].cells] == [
            _cell_id(0),
            _cell_id(1),
        ]


class TestObservingItsOwnCells:
    # [CLASS DOC] Tests how the reporter reacts to cell reconcile events it observes.
    async def test_a_cell_that_came_or_went_wakes_the_reporter(self):
        # [FUNCTION DOC] After the initial sync (one notification for the first cell), reconciles cell 1
        # as added and then as removed (cell info of nothing). Asserts the trigger was notified three
        # times in total and the reporter's per-cell info mapping holds only cell 0 afterwards.
        """Waiting out the whole period would leave the run without engines its deployment already has."""
        reporter, provider, _hub_endpoint = await _synced()
        assert reporter._trigger.notified == 1

        await provider.reconcile(_cell_id(1), _cell_info(1))
        await provider.reconcile(_cell_id(1), None)

        assert reporter._trigger.notified == 3
        assert sorted(reporter._info_of_cell_id) == [_cell_id(0)]


class TestReporterWorker:
    # [CLASS DOC] Tests the reporter worker wrapper, which starts the reporter loop on construction.
    def test_construction_starts_reporting_and_a_stopped_reporter_exits_the_process(self) -> None:
        # [FUNCTION DOC] Uses a fake reporter whose run coroutine only sets a started event and then
        # returns. Patches the process exit function to set an exit event. Constructs the worker with
        # an args namespace (report interval 0) and asserts within five seconds each that reporting
        # started and that exit was requested, and that exit was called exactly once with status 1.
        """Construction starts reporting, and an ended reporting loop makes the deployment exit."""
        reporting_started = threading.Event()
        exit_called = threading.Event()

        class FakeReporter:
            # [CLASS DOC] HELPER: Reporter stand-in whose run finishes at once after signalling start.
            async def run(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the reporting started event and returns, which ends the loop.
                reporting_started.set()

        args = Namespace(env_report_interval_seconds=0)
        with patch("os._exit", side_effect=lambda _code: exit_called.set()) as exit_process:
            RegistrationReporterWorker(args=args, reporter=FakeReporter())

            assert reporting_started.wait(timeout=5.0)
            assert exit_called.wait(timeout=5.0)
            exit_process.assert_called_once_with(1)

    def test_the_worker_the_engine_release_serves_answers_no_call(self):
        # [FUNCTION DOC] Asserts that collecting the remote call specs of the worker class gives an
        # empty mapping, i.e. the worker exposes no callable method.
        """Its whole value is the reporting loop its constructor starts, so it exposes nothing to call."""
        assert collect_rpc_method_specs(RegistrationReporterWorker) == {}


class _RecordingProvider(_FakeEngineProvider):
    # [CLASS DOC] HELPER: Fake provider that appends "init" and "watch" markers to a shared event log,
    # to let tests check the call order of provider initialization and watching.
    def __init__(self, events: list[str], **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Forwards the keyword arguments to the fake provider and keeps the
        # shared event list.
        super().__init__(**kwargs)
        self.events = events

    async def init(self) -> None:
        # [FUNCTION DOC] HELPER: Appends the marker "init" to the shared event list.
        self.events.append("init")

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        # [FUNCTION DOC] HELPER: Appends the marker "watch" to the event list, then behaves like the
        # fake provider and returns its stop function.
        self.events.append("watch")
        return await super().watch_cells(reconcile)


class _InitRequiringProvider(_FakeEngineProvider):
    # [CLASS DOC] HELPER: Fake provider that refuses to be watched before its init ran, mimicking a
    # provider (such as the static external-engine one) that discovers its cells in init.
    def __init__(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Forwards the keyword arguments and sets the initialized flag to false.
        super().__init__(**kwargs)
        self.initialized = False

    async def init(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the initialized flag.
        self.initialized = True

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        # [FUNCTION DOC] HELPER: Asserts init already ran (raising an assertion error with an
        # explanatory message otherwise), then behaves like the fake provider.
        assert self.initialized, "the engines this provider serves only exist once init() discovered them"
        return await super().watch_cells(reconcile)


class _RecordingHub(_FakeHub):
    # [CLASS DOC] HELPER: Fake hub that appends a "wait_ready" marker to a shared event log whenever
    # the reporter waits for readiness.
    def __init__(self, events: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Runs the base constructor and keeps the shared event list.
        super().__init__()
        self.events = events

    async def wait_ready(self, *, timeout: float) -> None:
        # [FUNCTION DOC] HELPER: Appends the marker "wait_ready" and delegates to the base with the
        # timeout only.
        # KNOWN ISSUE: unlike the base fake and the real hub, this override does not accept the
        # allow-server-uuid-change keyword, so it would fail if the reporter ever passed that keyword
        # on the first (ordinary) readiness wait.
        self.events.append("wait_ready")
        await super().wait_ready(timeout=timeout)


async def _run_until_watching(reporter: RegistrationReporter, provider: _FakeEngineProvider) -> None:
    # [FUNCTION DOC] HELPER: Starts the reporter's run as a task, yields to the event loop up to 100
    # times until the provider has received its reconcile callback (that is, the reporter started
    # watching), then cancels the run and awaits it, asserting it ends with a cancellation error.
    run = asyncio.create_task(reporter.run())
    for _ in range(100):
        await asyncio.sleep(0)
        if provider.reconcile is not None:
            break

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run


class TestReporterLifecycle:
    # [CLASS DOC] Tests the ordering and failure handling of the reporter's main run loop.
    async def test_it_waits_for_the_hub_before_it_watches_its_own_cells(self):
        # [FUNCTION DOC] Runs the reporter until it is watching, then cancels it. Asserts the hub saw at
        # least one readiness wait and the provider's watch was stopped on cancellation.
        """The hub_endpoint comes up with the run, and a snapshot into nothing is a wasted period."""
        provider = _FakeEngineProvider(cell_indices=[0])
        hub_endpoint = _FakeHub()
        run = asyncio.create_task(_reporter(provider=provider, hub_endpoint=hub_endpoint).run())
        for _ in range(100):
            await asyncio.sleep(0)
            if provider.reconcile is not None:
                break

        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        assert hub_endpoint.ready_timeouts
        assert provider.stopped

    async def test_it_initializes_the_worker_provider_before_anything_else(self):
        # [FUNCTION DOC] Uses a shared event log recorded by the provider and hub, runs the reporter
        # until watching, and asserts the first three events are init, wait_ready, watch in that order.
        """InferenceController.init() does the same, and a provider is not usable before it."""
        events: list[str] = []
        provider = _RecordingProvider(events, cell_indices=[0])

        await _run_until_watching(_reporter(provider=provider, hub_endpoint=_RecordingHub(events)), provider)

        assert events[:3] == ["init", "wait_ready", "watch"]

    async def test_a_provider_whose_cells_init_discovers_is_watched_without_tripping_its_own_check(self):
        # [FUNCTION DOC] Runs the reporter over a provider that asserts it was initialized before being
        # watched, and asserts the reporter reached the watching state (the reconcile callback exists).
        """The static external-engine provider asserted this and took the whole reporter down with it."""
        provider = _InitRequiringProvider(cell_indices=[0])

        await _run_until_watching(_reporter(provider=provider, hub_endpoint=_FakeHub()), provider)

        assert provider.reconcile is not None

    async def test_the_cells_init_discovered_reach_the_hub(self):
        # [FUNCTION DOC] Runs the reporter until watching over an init-requiring provider, sends one
        # more snapshot explicitly, and asserts the last recorded snapshot lists cell 0.
        """A split run on --rollout-external-engine-addrs registered no engine at all, and then timed out."""
        provider = _InitRequiringProvider(cell_indices=[0])
        hub_endpoint = _FakeHub()
        reporter = _reporter(provider=provider, hub_endpoint=hub_endpoint)

        await _run_until_watching(reporter, provider)
        await reporter._send_once()

        assert [cell.info.cell_id for cell in hub_endpoint.snapshots[-1].cells] == [_cell_id(0)]

    async def test_a_hung_ingest_is_cancelled_at_the_timeout(self):
        # [FUNCTION DOC] Uses a hub whose ingest never answers and patches the reporter's ingest
        # timeout constant to one millisecond. Asserts a send raises a timeout error and that the hung
        # ingest call observed its own cancellation.
        """A hub call that never answers is cancelled when its ingest deadline expires."""
        hub_endpoint = _HungHub()
        reporter, _provider, _hub_endpoint = await _synced(hub_endpoint=hub_endpoint)

        with patch("miles.utils.workers.registration.reporter.INGEST_TIMEOUT_SECONDS", 0.001):
            with pytest.raises(TimeoutError):
                await reporter._send_once()

        assert hub_endpoint.cancelled.is_set()

    async def test_a_failed_ingest_is_logged_and_the_next_snapshot_is_sent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # [FUNCTION DOC] Runs the reporter in the background against a hub that fails the first ingest.
        # Waits up to five seconds for the second snapshot to be accepted, with warning level log
        # capture on. Asserts the run task is still alive, the first two snapshots have sequence
        # numbers 1 and 2, and a warning naming the reporter id and the failure was logged. Then
        # cancels the run, expects a cancellation error, and asserts the provider watch was stopped.
        """A transient ingest failure is logged while reporting continues with a newer snapshot sequence."""
        provider = _FakeEngineProvider(cell_indices=[0])
        hub_endpoint = _FailOnceHub()
        reporter = _reporter(provider=provider, hub_endpoint=hub_endpoint)

        with caplog.at_level(logging.WARNING):
            run = asyncio.create_task(reporter.run())
            await asyncio.wait_for(hub_endpoint.accepted.wait(), timeout=5.0)

        assert not run.done()
        assert [snapshot.sequence_number for snapshot in hub_endpoint.snapshots[:2]] == [1, 2]
        assert f"Reporting the cells of {_REPORTER_ID} failed" in caplog.text

        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        assert provider.stopped


class TestReporterFailure:
    # [CLASS DOC] Tests the done-callback that turns any ended reporting loop into a process exit.
    def test_a_stopped_reporter_forces_the_deployment_to_exit(self):
        # [FUNCTION DOC] Builds one future completed normally and one completed with an exception,
        # patches the process exit function in the reporter module, calls the exit-on-stop callback
        # with each future, and asserts the exit function was called twice, each time with status 1.
        """Any stopped reporter makes its deployment exit unsuccessfully instead of appearing healthy."""
        completed: concurrent.futures.Future[None] = concurrent.futures.Future()
        completed.set_result(None)
        failed: concurrent.futures.Future[None] = concurrent.futures.Future()
        failed.set_exception(RuntimeError("reporting stopped"))

        with patch("miles.utils.workers.registration.reporter.os._exit") as exit_process:
            _exit_on_reporter_stop(completed)
            _exit_on_reporter_stop(failed)

        assert [call.args for call in exit_process.call_args_list] == [(1,), (1,)]


def _trigger(*, interval_seconds: float = 0.05, jitter_ratio: float = 0.0, debounce_seconds: float = 0.0):
    # [FUNCTION DOC] HELPER: Builds a real debounced interval trigger with the given interval (default
    # 0.05 seconds), jitter ratio (default none) and debounce time (default none), using a random
    # generator seeded with zero so jitter is reproducible.
    return _DebouncedIntervalTrigger(
        interval_seconds=interval_seconds,
        jitter_ratio=jitter_ratio,
        debounce_seconds=debounce_seconds,
        rng=random.Random(0),
    )


class TestDebouncedIntervalTrigger:
    # [CLASS DOC] Tests the real debounced interval trigger: periodic firing, early firing on
    # notification, debounce settling, and period jitter.
    async def test_it_fires_on_its_own_when_nothing_notified_it(self):
        # [FUNCTION DOC] Waits (bounded to five seconds) on a trigger with a short interval and no
        # notification; the test passes if the wait returns by itself.
        """A membership nobody changed still has to be re-announced, or the run forgets this reporter."""
        await asyncio.wait_for(_trigger().wait(), timeout=5.0)

    async def test_a_notification_fires_it_before_the_interval_is_up(self):
        # [FUNCTION DOC] Notifies a trigger whose interval is 1000 seconds and asserts the wait still
        # returns within five seconds.
        """A cell that came or went has to reach the run now, not up to a whole period later."""
        trigger = _trigger(interval_seconds=1000.0)
        trigger.notify()

        await asyncio.wait_for(trigger.wait(), timeout=5.0)

    async def test_it_settles_before_firing_so_a_burst_of_changes_sends_once(self):
        # [FUNCTION DOC] Notifies a trigger with a 0.05 second debounce, waits for it to fire, and
        # asserts the internal changed event is cleared afterwards (the change was consumed).
        """Cells arrive in bursts, and one snapshot per cell would be a snapshot storm."""
        trigger = _trigger(interval_seconds=1000.0, debounce_seconds=0.05)
        trigger.notify()

        await asyncio.wait_for(trigger.wait(), timeout=5.0)

        assert not trigger._changed.is_set()

    async def test_a_change_during_the_settling_window_is_not_carried_into_the_next_wait(self):
        # [FUNCTION DOC] Notifies, starts a wait, yields once so it enters the debounce window,
        # notifies again, then waits for completion. Asserts the changed event is cleared, so the
        # second notification did not leak into the next wait.
        """It is already part of the snapshot this wait is about to send, so firing again would send a duplicate."""
        trigger = _trigger(interval_seconds=1000.0, debounce_seconds=0.05)
        trigger.notify()

        waiting = asyncio.ensure_future(trigger.wait())
        await asyncio.sleep(0)
        trigger.notify()
        await asyncio.wait_for(waiting, timeout=5.0)

        assert not trigger._changed.is_set()

    async def test_the_period_it_waits_is_jittered_around_the_interval(self):
        # [FUNCTION DOC] Draws 20 next-interval values from a trigger with interval 10 and jitter ratio
        # 0.2; asserts more than one distinct value appears and all lie within 8 to 12 seconds.
        """Every reporter of a run wakes on its own schedule, so they never pile onto the hub_endpoint together."""
        trigger = _trigger(interval_seconds=10.0, jitter_ratio=0.2)

        periods = {trigger._compute_next_interval_seconds() for _ in range(20)}

        assert len(periods) > 1
        assert all(8.0 <= period <= 12.0 for period in periods)

    async def test_no_jitter_asks_for_exactly_the_interval(self):
        # [FUNCTION DOC] Asserts that with a zero jitter ratio the computed next interval equals the
        # configured interval of 10 seconds exactly.
        """The jitter is a spread around the configured period, not a replacement for it."""
        assert _trigger(interval_seconds=10.0, jitter_ratio=0.0)._compute_next_interval_seconds() == 10.0


class _FailingHub(_FakeHub):
    # [CLASS DOC] HELPER: Fake hub whose ingest raises a configured error for the first N calls and
    # then behaves like the recording fake hub.
    def __init__(self, failures: int, error: BaseException) -> None:
        # [FUNCTION DOC] HELPER: Runs the base constructor and stores the remaining failure count and
        # the error to raise.
        super().__init__()
        self.failures = failures
        self.error = error

    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        # [FUNCTION DOC] HELPER: While failures remain, decrements the count and raises the stored
        # error without recording the snapshot; afterwards records the snapshot as the base does.
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        await super().registration_ingest(snapshot=snapshot)


class _RestartedHub(_FakeHub):
    # [CLASS DOC] HELPER: Fake hub that behaves like a hub whose process was replaced: it refuses
    # ingest with a server-restarted error until a readiness wait has dropped the pin.
    """Answers like a hub whose process was replaced: it refuses every call made on the old pin."""

    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        # [FUNCTION DOC] HELPER: If no readiness wait so far allowed a server uuid change, raises a
        # server-restarted error saying the answering hub is a different process. Otherwise records
        # the snapshot like the base fake.
        if True not in self.pin_drops:
            raise ServerRestartedError("the hub answering now is a different process")
        await super().registration_ingest(snapshot=snapshot)


async def _report_until(reporter: RegistrationReporter, condition) -> None:
    # [FUNCTION DOC] HELPER: Starts the reporter's run as a task and polls the given zero-argument
    # condition callable every millisecond for up to 1000 rounds, stopping early when it is true.
    # Then cancels the run, expects it to end with a cancellation error, and finally asserts the
    # condition holds.
    run = asyncio.create_task(reporter.run())
    for _ in range(1000):
        await asyncio.sleep(0.001)
        if condition():
            break

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert condition()


class TestAHubThatRestartedUnderTheReporter:
    # [CLASS DOC] Tests how the reporter reacts when the hub process is replaced: it must wait for
    # readiness again with the pin dropped, only after a server-restarted error.
    async def test_it_awaits_the_new_hub_with_the_pin_dropped(self):
        # [FUNCTION DOC] Runs the reporter against a restarted hub until a snapshot is accepted, and
        # asserts the first two readiness waits had the allow-server-uuid-change flag false then true.
        """Dropping the pin is the only way the handle stops talking to a process that is gone."""
        hub_endpoint = _RestartedHub()
        provider = _FakeEngineProvider(cell_indices=[0])

        await _report_until(_reporter(provider=provider, hub_endpoint=hub_endpoint), lambda: hub_endpoint.snapshots)

        assert hub_endpoint.pin_drops[:2] == [False, True]

    async def test_it_goes_on_reporting_into_the_process_that_answers_now(self):
        # [FUNCTION DOC] Same setup as the previous test; asserts the first accepted snapshot still
        # lists cell 0, i.e. reporting resumed with the full membership.
        """The run waits for engines that are running until this deployment reaches the new hub."""
        hub_endpoint = _RestartedHub()
        provider = _FakeEngineProvider(cell_indices=[0])

        await _report_until(_reporter(provider=provider, hub_endpoint=hub_endpoint), lambda: hub_endpoint.snapshots)

        assert [cell.info.cell_id for cell in hub_endpoint.snapshots[0].cells] == [_cell_id(0)]

    async def test_an_ordinary_failure_leaves_the_pin_where_it_was(self):
        # [FUNCTION DOC] Uses a hub whose first ingest raises a generic runtime error, runs until a
        # snapshot is accepted, and asserts only one readiness wait happened, without dropping the pin.
        """Re-awaiting readiness on every hiccup would accept whatever process answered next."""
        hub_endpoint = _FailingHub(failures=1, error=RuntimeError("the hub was busy"))
        provider = _FakeEngineProvider(cell_indices=[0])

        await _report_until(_reporter(provider=provider, hub_endpoint=hub_endpoint), lambda: hub_endpoint.snapshots)

        assert hub_endpoint.pin_drops == [False]

    async def test_a_hub_that_never_restarted_is_awaited_once(self):
        # [FUNCTION DOC] Uses a healthy hub, runs until a snapshot arrives, and asserts exactly one
        # readiness wait took place and it did not drop the pin.
        """Every reporter of every run takes this path, and it may not wait for readiness twice."""
        hub_endpoint = _FakeHub()
        provider = _FakeEngineProvider(cell_indices=[0])

        await _report_until(_reporter(provider=provider, hub_endpoint=hub_endpoint), lambda: hub_endpoint.snapshots)

        assert hub_endpoint.pin_drops == [False]
