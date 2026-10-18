# [MODULE DOC]
# Unit tests for the registration hub, which lets deployments report the cells they own through
# whole-membership snapshots. The hub validates snapshots (run identity, reporter ownership,
# duplicates, cell and worker naming consistency, sequence numbers), keeps a cell table partitioned
# by reporter, announces additions, changes and removals to exactly one watcher, serves worker
# addresses and infos, retries cells whose reconcile failed, and drops reporters that stopped
# reporting after a time-to-live measured on an injectable clock.
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from miles.utils.workers.naming import _worker_name_of_cell, compute_cell_id
from miles.utils.workers.registration.hub import REPORTER_TTL_SECONDS, RegistrationHub
from miles.utils.workers.registration.models import RegisteredCellInfo, RegistrationSnapshot
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import HostAndPort

_REPORTER = "miles-run-r1-inference"
_POOL_ID = f"{_REPORTER}-inference-engine-0-0"
_OTHER_REPORTER = "miles-run-r1-inference-b"
_OTHER_POOL_ID = f"{_OTHER_REPORTER}-inference-engine-0-0"
_PROVIDER_MODULE = "miles.utils.workers.registration.hub"
_RUN_UUID = "run-uuid-1"
_POLL_INTERVAL_SECONDS = 0.001


def _cell_id(cell_index: int, *, pool_id: str = _POOL_ID) -> str:
    # [FUNCTION DOC] HELPER: Computes the cell id for the pool (default the first reporter's
    # pool) and cell index.
    return compute_cell_id(pool_id=pool_id, cell_index=cell_index)


def _cell(
    cell_index: int,
    *,
    host: str = "10.0.0.5",
    model_id: str = "default",
    generation: int = 0,
    reporter_id: str = _REPORTER,
    pool_id: str = _POOL_ID,
) -> RegisteredCellInfo:
    # [FUNCTION DOC] HELPER: Builds a registered cell owned by the reporter: an alive cell info
    # for the pool and index (workers hash derived from the host, metadata with the model id and
    # regular worker type, one worker name derived from the cell id) and one worker with the
    # given generation, a primary address at the host on port 8000, and gpu ids [0].
    cell_id = _cell_id(cell_index, pool_id=pool_id)
    return RegisteredCellInfo(
        reporter_id=reporter_id,
        info=CellInfo(
            cell_id=cell_id,
            pool_id=pool_id,
            alive=True,
            worker_names=[_worker_name_of_cell(cell_id)],
            workers_hash=f"hash-{host}",
            meta=dict(model_id=model_id, worker_type="regular"),
        ),
        workers=[
            WorkerInfo(
                name=_worker_name_of_cell(cell_id),
                generation=generation,
                self_addrs={"primary": HostAndPort(host=host, port=8000)},
                gpu_ids=[0],
            )
        ],
    )


def _other_cell(cell_index: int, *, host: str = "10.0.0.5") -> RegisteredCellInfo:
    # [FUNCTION DOC] HELPER: Same as the standard cell builder but owned by the second reporter
    # and its pool.
    return _cell(cell_index, host=host, reporter_id=_OTHER_REPORTER, pool_id=_OTHER_POOL_ID)


def _snapshot(
    cells: list[RegisteredCellInfo], *, reporter_id: str = _REPORTER, sequence_number: int = 1
) -> RegistrationSnapshot:
    # [FUNCTION DOC] HELPER: Builds a snapshot of the test run carrying the cells, from the given
    # reporter (default the first) with the given sequence number (default 1).
    return RegistrationSnapshot(
        run_uuid=_RUN_UUID, reporter_id=reporter_id, sequence_number=sequence_number, cells=cells
    )


@pytest.fixture(autouse=True)
def _poll_without_delay():
    # [FUNCTION DOC] HELPER: Autouse fixture patching the hub module's polling interval for
    # registered cells to 1 ms for the duration of each test.
    with patch(f"{_PROVIDER_MODULE}.REGISTERED_CELLS_POLL_INTERVAL_SECONDS", _POLL_INTERVAL_SECONDS):
        yield


class _FakeClock:
    # [CLASS DOC] HELPER: Manually advanced clock; calling it returns the current time.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts at time zero.
        self.now = 0.0

    def __call__(self) -> float:
        # [FUNCTION DOC] HELPER: Returns the current fake time in seconds.
        return self.now


class _Watcher:
    # [CLASS DOC] HELPER: Reconcile callback object recording (cell id, observed info or nothing)
    # pairs; cells in the failing set make it raise instead of recording.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with no observations and no failing cells.
        self.observations: list[tuple[str, CellInfo | None]] = []
        self.failing_cell_ids: set[str] = set()

    async def reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        # [FUNCTION DOC] HELPER: Raises a runtime error for failing cells; otherwise records the
        # cell id and observed info (nothing means the cell was removed).
        if cell_id in self.failing_cell_ids:
            raise RuntimeError(f"cell {cell_id} refuses to be reconciled")
        self.observations.append((cell_id, observed))


async def _watched(**kwargs) -> tuple[RegistrationHub, _Watcher]:
    # [FUNCTION DOC] HELPER: Creates a hub for the test run (extra keywords such as a clock are
    # passed through), starts its watch with a new watcher, and returns both. The stop function
    # of the watch is discarded.
    watcher = _Watcher()
    provider = RegistrationHub(run_uuid=_RUN_UUID, **kwargs)
    await _start_watch(provider, watcher)
    return provider, watcher


async def _start_watch(provider: RegistrationHub, watcher: _Watcher):
    # [FUNCTION DOC] HELPER: Starts the hub's cell watch with the watcher's reconcile callback;
    # returns the async stop function.
    return await provider.watch_cells(watcher.reconcile)


async def _apply(provider: RegistrationHub, snapshot: RegistrationSnapshot) -> None:
    # [FUNCTION DOC] HELPER: Ingests the snapshot and then lets the poller run (drain).
    await provider.ingest(snapshot)
    await _drain()


async def _drain() -> None:
    # [FUNCTION DOC] HELPER: Sleeps 50 poll intervals (about 50 ms of real time) so background
    # polling and reconciliation can run.
    for _ in range(50):
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)


class TestSnapshotMembership:
    # [CLASS DOC] Snapshot ingestion: what is announced, kept, replaced, and refused.
    async def test_a_first_snapshot_announces_every_cell_it_carries(self):
        # [FUNCTION DOC] A snapshot with cells 0 and 1: the hub's private cell table has both
        # and the watcher observed both ids in order.
        """A snapshot is the whole membership of its deployment, so the run takes all of it in at once."""
        provider, watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0), _cell(1)]))

        assert sorted(provider._cell_of_id) == [_cell_id(0), _cell_id(1)]
        assert [cell_id for cell_id, _observed in watcher.observations] == [_cell_id(0), _cell_id(1)]

    async def test_an_unchanged_cell_is_not_announced_twice(self):
        # [FUNCTION DOC] Two snapshots (sequence 1 and 2) with the same cell yield one
        # observation.
        """The run adds a cell once; announcing it again would tear down a serving engine to rebuild it."""
        provider, watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0)], sequence_number=1))
        await _apply(provider, _snapshot([_cell(0)], sequence_number=2))

        assert len(watcher.observations) == 1

    async def test_a_cell_that_changed_is_announced_anew(self):
        # [FUNCTION DOC] The second snapshot has the cell at another host; the second observation
        # is for that cell id with the new workers hash.
        """A cell rebuilt on another host serves from another address, and the old one answers nothing."""
        provider, watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0)], sequence_number=1))
        await _apply(provider, _snapshot([_cell(0, host="10.0.0.6")], sequence_number=2))

        (_first, (cell_id, observed)) = watcher.observations
        assert cell_id == _cell_id(0)
        assert observed.workers_hash == "hash-10.0.0.6"

    async def test_a_cell_the_snapshot_stops_naming_is_removed(self):
        # [FUNCTION DOC] A later snapshot omitting cell 1 removes it from the table and the
        # watcher's last observation is cell 1 with nothing observed.
        """Membership is level, so omission is how a deployment says a cell is gone; there is no death message."""
        provider, watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence_number=1))
        await _apply(provider, _snapshot([_cell(0)], sequence_number=2))

        assert sorted(provider._cell_of_id) == [_cell_id(0)]
        assert watcher.observations[-1] == (_cell_id(1), None)

    async def test_a_late_snapshot_is_ignored(self):
        # [FUNCTION DOC] Sequence 5 with two cells, then sequence 4 with one: the table still has
        # both cells.
        """A snapshot that crossed the wan slowly would otherwise resurrect cells the run already dropped."""
        provider, _watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence_number=5))
        await _apply(provider, _snapshot([_cell(0)], sequence_number=4))

        assert sorted(provider._cell_of_id) == [_cell_id(0), _cell_id(1)]

    async def test_a_cell_reported_by_two_deployments_is_refused(self):
        # [FUNCTION DOC] A second reporter sending an already owned cell id makes ingest fail an
        # assertion matching "is reported by both".
        """One cell id names one cell, and the second owner could remove the cell of the first."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)]))

        with pytest.raises(AssertionError, match="is reported by both"):
            await provider.ingest(_snapshot([_cell(0, reporter_id="other")], reporter_id="other"))

    async def test_a_snapshot_of_another_run_is_refused(self):
        # [FUNCTION DOC] A snapshot with a different run uuid fails an assertion matching
        # "carries run_uuid".
        """A resume reusing a run id leaves the reporter of the previous training announcing into this one."""
        provider, _watcher = await _watched()

        with pytest.raises(AssertionError, match="carries run_uuid"):
            await provider.ingest(_snapshot([_cell(0)]).model_copy(update=dict(run_uuid="another-run")))

    async def test_a_snapshot_carrying_a_cell_of_another_reporter_is_refused(self):
        # [FUNCTION DOC] A snapshot from the first reporter carrying a cell owned by another
        # reporter fails an assertion matching "snapshot of reporter".
        """A snapshot is the membership of one deployment, so a cell of another one was assembled in by mistake."""
        provider, _watcher = await _watched()

        with pytest.raises(AssertionError, match="snapshot of reporter"):
            await provider.ingest(_snapshot([_cell(0, reporter_id="other")]))

    async def test_a_cell_carried_twice_by_one_snapshot_is_refused(self):
        # [FUNCTION DOC] Two entries for cell 0 fail an assertion matching "more than once" and
        # the table stays empty.
        """Either entry could be the truth, so holding one would hold a membership nobody announced."""
        provider, _watcher = await _watched()

        with pytest.raises(AssertionError, match="more than once"):
            await provider.ingest(_snapshot([_cell(0), _cell(0, host="10.0.0.6")]))

        assert sorted(provider._cell_of_id) == []

    async def test_a_cell_that_does_not_name_its_own_pool_is_refused(self):
        # [FUNCTION DOC] A cell whose id is "not-a-cell-id" fails an assertion matching "does not
        # read as".
        """The run parses a cell id to address the workers of that cell."""
        provider, _watcher = await _watched()
        cell = _cell(0)
        cell = cell.model_copy(update=dict(info=cell.info.model_copy(update=dict(cell_id="not-a-cell-id"))))

        with pytest.raises(AssertionError, match="does not read as"):
            await provider.ingest(_snapshot([cell]))

    async def test_a_cell_id_that_names_another_pool_is_refused(self):
        # [FUNCTION DOC] A cell whose info declares another pool than its id names fails an
        # assertion matching "its cell id names pool".
        """A parseable cell id must name the pool declared by its cell information."""
        provider, _watcher = await _watched()
        cell = _cell(0)
        cell = cell.model_copy(update=dict(info=cell.info.model_copy(update=dict(pool_id=_OTHER_POOL_ID))))

        with pytest.raises(AssertionError, match="its cell id names pool"):
            await provider.ingest(_snapshot([cell]))

    async def test_a_cell_without_workers_is_refused(self):
        # [FUNCTION DOC] A cell with an empty worker list fails an assertion matching "it
        # carries no worker".
        """A registered cell must carry at least one addressable worker."""
        provider, _watcher = await _watched()
        cell = _cell(0).model_copy(update=dict(workers=[]))

        with pytest.raises(AssertionError, match="it carries no worker"):
            await provider.ingest(_snapshot([cell]))

    async def test_a_cell_with_an_unparseable_worker_name_is_refused(self):
        # [FUNCTION DOC] A cell whose worker is named "not-a-worker-name" fails an assertion
        # matching "does not read as".
        """A cell with a malformed worker name must be rejected as unaddressable."""
        provider, _watcher = await _watched()
        cell = _cell(0)
        worker = cell.workers[0].model_copy(update=dict(name="not-a-worker-name"))
        cell = cell.model_copy(update=dict(workers=[worker]))

        with pytest.raises(AssertionError, match="does not read as"):
            await provider.ingest(_snapshot([cell]))

    async def test_a_cell_whose_workers_belong_to_another_cell_is_refused(self):
        # [FUNCTION DOC] Cell 0 carrying cell 1's workers fails an assertion matching "name
        # cells".
        """The run addresses a worker by parsing its name, so a name from another cell would route nowhere."""
        provider, _watcher = await _watched()
        cell = _cell(0).model_copy(update=dict(workers=_cell(1).workers))

        with pytest.raises(AssertionError, match="name cells"):
            await provider.ingest(_snapshot([cell]))


class TestPartitioningCellsByReporter:
    # [CLASS DOC] Snapshots only govern the cells of their own reporter, with independent
    # sequence numbering.
    async def test_a_snapshot_replaces_only_the_cells_of_the_reporter_that_sent_it(self):
        # [FUNCTION DOC] First reporter has cells 0 and 1, second has 0 and 1, then the second
        # sends a snapshot with only its cell 0: the table keeps the first reporter's two cells
        # and the second's cell 0.
        """A snapshot declares one deployment's membership, so taking it as the run's would drop every other engine."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0), _cell(1)]))
        await _apply(provider, _snapshot([_other_cell(0), _other_cell(1)], reporter_id=_OTHER_REPORTER))

        await _apply(provider, _snapshot([_other_cell(0)], reporter_id=_OTHER_REPORTER, sequence_number=2))

        assert sorted(provider._cell_of_id) == sorted([_cell_id(0), _cell_id(1), _cell_id(0, pool_id=_OTHER_POOL_ID)])

    async def test_each_reporter_is_sequenced_on_its_own(self):
        # [FUNCTION DOC] The first reporter is at sequence 7; the second sends sequence 1 and is
        # accepted: the table has both reporters' cells.
        """Deployments count their own snapshots, so one that has run for longer must not silence a fresh one."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)], sequence_number=7))

        await _apply(provider, _snapshot([_other_cell(0)], reporter_id=_OTHER_REPORTER, sequence_number=1))

        assert sorted(provider._cell_of_id) == sorted([_cell_id(0), _cell_id(0, pool_id=_OTHER_POOL_ID)])


class TestResendingTheSameMembership:
    # [CLASS DOC] Idempotence of repeated identical snapshots.
    async def test_the_same_snapshot_sent_again_leaves_the_membership_as_it_was(self):
        # [FUNCTION DOC] Two snapshots (sequences 1 and 2) with the same cell leave the table
        # with that one cell.
        """Every tick carries the whole membership, so the steady state has to be idempotent."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)], sequence_number=1))

        await _apply(provider, _snapshot([_cell(0)], sequence_number=2))

        assert sorted(provider._cell_of_id) == [_cell_id(0)]

    async def test_resending_it_announces_no_change_to_the_watcher(self):
        # [FUNCTION DOC] After the first snapshot and a drain, clears the watcher's observations;
        # resending the same membership produces no new observation.
        """A membership that did not move must not churn the cells the run reconciles."""
        provider, watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)], sequence_number=1))
        await _drain()
        watcher.observations.clear()

        await _apply(provider, _snapshot([_cell(0)], sequence_number=2))
        await _drain()

        assert watcher.observations == []


class TestAddressingRegisteredCells:
    # [CLASS DOC] Address and info lookups for registered cells and late watchers.
    async def test_the_addresses_of_a_registered_worker_are_the_ones_reported(self):
        # [FUNCTION DOC] The addresses of a registered worker include its primary address as
        # reported (10.0.0.5, port 8000).
        """The run calls the engine at the address its own deployment observed, never at one derived here."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)]))

        addrs = await provider.get_addrs(_worker_name_of_cell(_cell_id(0)))

        assert addrs["primary"] == HostAndPort(host="10.0.0.5", port=8000)

    async def test_a_registered_worker_keeps_the_generation_its_own_deployment_gave_it(self):
        # [FUNCTION DOC] A cell registered with generation 3 yields a worker info with
        # generation 3.
        """The dashboard identifies an engine by its generation, so a restarted pod has to look like a new one."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0, generation=3)]))

        ((worker_info,),) = provider.get_worker_infos(cell_ids=[_cell_id(0)])

        assert worker_info.generation == 3

    async def test_an_unannounced_worker_of_a_registered_cell_is_refused(self):
        # [FUNCTION DOC] Asking for addresses of worker index 1 of a cell that announced only
        # worker 0 fails an assertion matching "is not among the workers".
        """A cell registration authorizes only the exact workers its deployment announced."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)]))

        with pytest.raises(AssertionError, match="is not among the workers"):
            await provider.get_addrs(f"{_POOL_ID}-0-1")

    async def test_a_watcher_that_starts_late_is_replayed_the_cells_already_reported(self):
        # [FUNCTION DOC] A snapshot is ingested before any watch; starting the watch afterwards
        # delivers that cell to the watcher.
        """A snapshot may land before the run watches, and nothing announces that cell a second time."""
        provider = RegistrationHub(run_uuid=_RUN_UUID)
        await _apply(provider, _snapshot([_cell(0)]))

        watcher = _Watcher()
        await _start_watch(provider, watcher)

        assert [cell_id for cell_id, _observed in watcher.observations] == [_cell_id(0)]


class TestWatcherOwnership:
    # [CLASS DOC] A hub supports a single watcher.
    async def test_a_second_watcher_is_refused(self):
        # [FUNCTION DOC] Starting a second watch while one is active fails an assertion matching
        # "exactly one watcher"; the first watch is stopped in a cleanup block.
        """One hub has one watcher, so a second consumer cannot race its reconciliation state."""
        provider = RegistrationHub(run_uuid=_RUN_UUID)
        stop_watch = await _start_watch(provider, _Watcher())

        try:
            with pytest.raises(AssertionError, match="exactly one watcher"):
                await _start_watch(provider, _Watcher())
        finally:
            await stop_watch()


class TestFailedReconciliation:
    # [CLASS DOC] Retry of cells whose reconcile raised.
    async def test_a_cell_the_run_could_not_take_in_is_offered_again_on_the_next_poll(self):
        # [FUNCTION DOC] The watcher fails for cell 0 while a snapshot is ingested: no
        # observation but the cell stays in the table. After the failure is cleared and time
        # passes, the watcher observes the cell.
        """A cell whose reconcile raised must be retried from the membership this run already holds."""
        provider, watcher = await _watched()
        watcher.failing_cell_ids = {_cell_id(0)}

        await _apply(provider, _snapshot([_cell(0)], sequence_number=1))

        assert watcher.observations == []
        assert sorted(provider._cell_of_id) == [_cell_id(0)]

        watcher.failing_cell_ids = set()
        await _drain()

        assert [cell_id for cell_id, _observed in watcher.observations] == [_cell_id(0)]


class TestDroppingAReporterThatStoppedReporting:
    # [CLASS DOC] Time-to-live expiry of reporters, measured on a manually advanced clock.
    async def test_a_reporter_that_stopped_reporting_loses_every_cell_it_announced(self):
        # [FUNCTION DOC] After one snapshot, advances the clock past the reporter time-to-live
        # and drains: the table is empty and the watcher's last observation is the cell removed.
        """Its deployment is gone or unreachable, and a request routed to those engines answers nothing."""
        clock = _FakeClock()
        provider, watcher = await _watched(clock=clock)
        await _apply(provider, _snapshot([_cell(0)]))

        clock.now += REPORTER_TTL_SECONDS + 1.0
        await _drain()

        assert sorted(provider._cell_of_id) == []
        assert watcher.observations[-1] == (_cell_id(0), None)

    async def test_a_reporter_still_inside_its_deadline_keeps_its_cells(self):
        # [FUNCTION DOC] Advancing to one second before the time-to-live keeps the cell.
        """A reporter reports far more often than this, so dropping one early would flap a healthy deployment."""
        clock = _FakeClock()
        provider, _watcher = await _watched(clock=clock)
        await _apply(provider, _snapshot([_cell(0)]))

        clock.now += REPORTER_TTL_SECONDS - 1.0
        await _drain()

        assert sorted(provider._cell_of_id) == [_cell_id(0)]

    async def test_an_accepted_snapshot_renews_the_reporter_deadline(self) -> None:
        # [FUNCTION DOC] No watcher. A newer snapshot arrives just before expiry and then time
        # passes beyond the original deadline; after calling the private stale-reporter sweep
        # directly, the reporter and its cell are still present.
        """A newer accepted snapshot keeps its reporter and cells alive past the original deadline."""
        clock = _FakeClock()
        provider = RegistrationHub(run_uuid=_RUN_UUID, clock=clock)
        await _apply(provider, _snapshot([_cell(0)], sequence_number=1))

        clock.now += REPORTER_TTL_SECONDS - 1.0
        await provider.ingest(_snapshot([_cell(0)], sequence_number=2))
        clock.now += 2.0
        provider._remove_stale_reporters()

        assert sorted(provider._state_of_reporter_id) == [_REPORTER]
        assert sorted(provider._cell_of_id) == [_cell_id(0)]

    async def test_a_reporter_that_comes_back_is_taken_in_again(self):
        # [FUNCTION DOC] After expiry drops the reporter, a snapshot with sequence 9 is accepted
        # and the cell is back in the table.
        """Being dropped is not a verdict on the deployment; it announces itself anew like any first time."""
        clock = _FakeClock()
        provider, _watcher = await _watched(clock=clock)
        await _apply(provider, _snapshot([_cell(0)]))

        clock.now += REPORTER_TTL_SECONDS + 1.0
        await _drain()
        await _apply(provider, _snapshot([_cell(0)], sequence_number=9))

        assert sorted(provider._cell_of_id) == [_cell_id(0)]

    async def test_a_snapshot_dropped_as_late_does_not_keep_its_reporter_alive(self):
        # [FUNCTION DOC] A reporter at sequence 5 sends sequence 1 just before expiry (ignored as
        # late); after time passes beyond the original deadline and the private sweep runs,
        # the reporter state and cells are gone.
        """A reporter that restarted resends from sequence 1, and only the deadline ends that stalemate."""
        clock = _FakeClock()
        provider = RegistrationHub(run_uuid=_RUN_UUID, clock=clock)
        await _apply(provider, _snapshot([_cell(0)], sequence_number=5))

        clock.now += REPORTER_TTL_SECONDS - 1.0
        await provider.ingest(_snapshot([_cell(0)], sequence_number=1))
        clock.now += 2.0
        provider._remove_stale_reporters()

        assert sorted(provider._state_of_reporter_id) == []
        assert sorted(provider._cell_of_id) == []

    async def test_dropping_a_reporter_forgets_the_reporter_itself(self):
        # [FUNCTION DOC] After expiry and the private sweep, the private per-reporter state table
        # no longer contains the reporter (it held one entry before).
        """A state left behind keeps the reporter's sequence number, which would refuse its fresh snapshots."""
        clock = _FakeClock()
        provider = RegistrationHub(run_uuid=_RUN_UUID, clock=clock)
        await _apply(provider, _snapshot([_cell(0)]))
        assert sorted(provider._state_of_reporter_id) == [_REPORTER]

        clock.now += REPORTER_TTL_SECONDS + 1.0
        provider._remove_stale_reporters()

        assert sorted(provider._state_of_reporter_id) == []

    async def test_the_sweep_stops_with_the_watcher_it_was_started_for(self):
        # [FUNCTION DOC] Starts and then stops the watch, advances the clock past the
        # time-to-live, and ingests a snapshot: the cell is held, showing no sweep removed it.
        """A hub nobody watches serves nobody, and a task outliving it would keep the whole hub alive."""
        clock = _FakeClock()
        provider = RegistrationHub(run_uuid=_RUN_UUID, clock=clock)
        stop_watch = await _start_watch(provider, _Watcher())

        await stop_watch()
        clock.now += REPORTER_TTL_SECONDS + 1.0
        await _apply(provider, _snapshot([_cell(0)]))

        assert sorted(provider._cell_of_id) == [_cell_id(0)]


def _announcing(cell: RegisteredCellInfo, worker_names: list[str]) -> RegisteredCellInfo:
    # [FUNCTION DOC] HELPER: Returns a copy of the cell whose info announces the given worker
    # names (the carried workers are unchanged).
    return cell.model_copy(update={"info": cell.info.model_copy(update={"worker_names": worker_names})})


class TestACellAnnouncingTheWorkersItCarries:
    # [CLASS DOC] A cell's announced worker names must equal the workers it carries.
    async def test_a_cell_that_announces_no_worker_at_all_is_refused(self):
        # [FUNCTION DOC] An empty announced list fails an assertion matching "announces the
        # workers".
        """The rollout reconciliation indexes worker_names[0], and an empty list is what it fails on."""
        provider, _watcher = await _watched()

        with pytest.raises(AssertionError, match="announces the workers"):
            await provider.ingest(_snapshot([_announcing(_cell(0), [])]))

    async def test_a_cell_that_announces_a_worker_it_does_not_carry_is_refused(self):
        # [FUNCTION DOC] An announced name that no carried worker has fails an assertion
        # matching "announces the workers".
        """Addresses are looked up by the announced names, and a name nothing answers to is a dead lookup."""
        provider, _watcher = await _watched()

        with pytest.raises(AssertionError, match="announces the workers"):
            await provider.ingest(_snapshot([_announcing(_cell(0), ["some-other-worker-0"])]))

    async def test_a_cell_that_carries_one_worker_twice_is_refused(self):
        # [FUNCTION DOC] A cell whose worker list repeats its worker fails an assertion matching
        # "one of them twice".
        """Two entries of one name make the run's view of the fleet count an engine that is not there."""
        provider, _watcher = await _watched()
        cell = _cell(0)
        twice = cell.model_copy(update={"workers": [*cell.workers, *cell.workers]})

        with pytest.raises(AssertionError, match="one of them twice"):
            await provider.ingest(_snapshot([twice]))

    async def test_a_cell_that_announces_exactly_what_it_carries_is_taken_in(self):
        # [FUNCTION DOC] A consistent cell is accepted and appears in the table.
        """Every healthy engine reports this shape, and the check may not stand in its way."""
        provider, _watcher = await _watched()

        await _apply(provider, _snapshot([_cell(0)]))

        assert sorted(provider._cell_of_id) == [_cell_id(0)]

    async def test_a_refused_snapshot_leaves_the_membership_it_found(self):
        # [FUNCTION DOC] After accepting a snapshot with cell 0, a second snapshot with an
        # inconsistent cell 1 is refused with an assertion error and the table still holds only
        # cell 0.
        """The hub had already taken such a snapshot in, and the run then failed on what it held."""
        provider, _watcher = await _watched()
        await _apply(provider, _snapshot([_cell(0)]))

        with pytest.raises(AssertionError):
            await provider.ingest(_snapshot([_announcing(_cell(1), [])], sequence_number=2))

        assert sorted(provider._cell_of_id) == [_cell_id(0)]


async def _hub_with_two_reporters(clock: _FakeClock) -> RegistrationHub:
    # [FUNCTION DOC] HELPER: Builds a watched hub on the given clock and applies one snapshot
    # from each reporter, each with its cell 0. Returns the hub.
    provider, _watcher = await _watched(clock=clock)
    await _apply(provider, _snapshot([_cell(0)]))
    await _apply(provider, _snapshot([_other_cell(0)], reporter_id=_OTHER_REPORTER))
    return provider


class TestRenewingAReporterOnTheSnapshotsTakenIn:
    # [CLASS DOC] Only snapshots that were accepted renew a reporter's deadline.
    async def test_a_snapshot_claiming_another_reporters_cells_does_not_renew_its_sender(self):
        # [FUNCTION DOC] Near expiry, the second reporter sends a snapshot claiming the first
        # reporter's cell (refused with "is reported by both"); after the deadline passes and the
        # private sweep runs, the second reporter is gone.
        """A reporter that keeps sending such snapshots was never dropped, so its stale cells stayed."""
        clock = _FakeClock()
        provider = await _hub_with_two_reporters(clock)

        clock.now += REPORTER_TTL_SECONDS - 1.0
        with pytest.raises(AssertionError, match="is reported by both"):
            await provider.ingest(
                _snapshot([_cell(0, reporter_id=_OTHER_REPORTER)], reporter_id=_OTHER_REPORTER, sequence_number=2)
            )
        clock.now += 2.0
        provider._remove_stale_reporters()

        assert _OTHER_REPORTER not in provider._state_of_reporter_id

    async def test_the_cells_such_a_reporter_registered_before_are_dropped_with_it(self):
        # [FUNCTION DOC] Same scenario; after the sweep, the second reporter's earlier cell is
        # no longer in the table.
        """Those engines may long be gone, and the run keeps routing requests to them until they are."""
        clock = _FakeClock()
        provider = await _hub_with_two_reporters(clock)

        clock.now += REPORTER_TTL_SECONDS - 1.0
        with pytest.raises(AssertionError, match="is reported by both"):
            await provider.ingest(
                _snapshot([_cell(0, reporter_id=_OTHER_REPORTER)], reporter_id=_OTHER_REPORTER, sequence_number=2)
            )
        clock.now += 2.0
        provider._remove_stale_reporters()

        assert _cell_id(0, pool_id=_OTHER_POOL_ID) not in provider._cell_of_id

    async def test_a_snapshot_that_was_taken_in_renews_its_sender(self):
        # [FUNCTION DOC] Near expiry the second reporter sends an accepted snapshot (sequence 2);
        # after the original deadline and a sweep, it is still known.
        """A healthy reporter is kept alive by exactly this stamp, and dropping it would flap the fleet."""
        clock = _FakeClock()
        provider = await _hub_with_two_reporters(clock)

        clock.now += REPORTER_TTL_SECONDS - 1.0
        await _apply(provider, _snapshot([_other_cell(0)], reporter_id=_OTHER_REPORTER, sequence_number=2))
        clock.now += 2.0
        provider._remove_stale_reporters()

        assert _OTHER_REPORTER in provider._state_of_reporter_id

    async def test_a_snapshot_refused_as_late_still_leaves_the_reporter_to_its_deadline(self):
        # [FUNCTION DOC] Near expiry the second reporter resends sequence 1 (ignored as late);
        # after the original deadline and a sweep, the reporter is gone.
        """Only the deadline ends the stalemate with a reporter that restarted and resends from sequence 1."""
        clock = _FakeClock()
        provider = await _hub_with_two_reporters(clock)

        clock.now += REPORTER_TTL_SECONDS - 1.0
        await provider.ingest(_snapshot([_other_cell(0)], reporter_id=_OTHER_REPORTER, sequence_number=1))
        clock.now += 2.0
        provider._remove_stale_reporters()

        assert _OTHER_REPORTER not in provider._state_of_reporter_id
