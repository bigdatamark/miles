# [MODULE DOC]
# Unit tests for the reconcile loop, which feeds a keyed object store from a source event stream
# and calls a user reconcile function once per parent key through a single worker with a work
# queue, per-key failure backoff, periodic resync and source retry. Uses a fake source, a fake
# clock, and a recording reconcile callable. Covers cadence validation, the initial sync barrier,
# incremental upserts and deletes, work queue discipline, failure backoff, resync, the lister,
# stop semantics, in-stream relists and relist replacement after reconnects.
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.fast.utils.workers.reconcile.utils import (
    FakeSource,
    StreamEnd,
    StreamError,
    make_pod,
    pod_cell,
    replace_of,
    settle,
)

from miles.utils.test_utils.clock import FakeClock

from miles.utils.workers.reconcile.loop import DEFAULT_RESYNC_PERIOD, ReconcileLoop
from miles.utils.workers.reconcile.retry_scheduler import POLL_INTERVAL
from miles.utils.workers.reconcile.source_event import DeleteEvent, UpsertEvent


class Recorder:
    # [CLASS DOC] HELPER: Reconcile callable that records calls. Tracks the keys in call order, a
    # snapshot of the store's object names per call (when a loop is attached), the current and
    # maximum concurrency, an optional gate event that blocks calls until opened, and a set of
    # keys whose calls fail with a runtime error.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with empty records, zero concurrency, no gate, no failing
        # keys and no loop attached.
        self.parent_keys: list[str] = []
        self.snapshots: list[list[str]] = []
        self.concurrency = 0
        self.max_concurrency = 0
        self.gate: asyncio.Event | None = None
        self.fail_parent_keys: set[str] = set()
        self.loop: ReconcileLoop | None = None

    async def __call__(self, parent_key: str) -> None:
        # [FUNCTION DOC] HELPER: Increments concurrency and updates the maximum; records the key;
        # if a loop is attached records the names of the objects the loop's store holds for that
        # key; if a gate exists waits for it; raises a runtime error if the key is marked failing.
        # Always decrements concurrency at the end.
        self.concurrency += 1
        self.max_concurrency = max(self.max_concurrency, self.concurrency)
        try:
            self.parent_keys.append(parent_key)
            if self.loop is not None:
                self.snapshots.append([pod.metadata.name for pod in self.loop.get_by_parent(parent_key)])
            if self.gate is not None:
                await self.gate.wait()
            if parent_key in self.fail_parent_keys:
                raise RuntimeError(f"reconcile failed for {parent_key}")
        finally:
            self.concurrency -= 1

    def counts(self) -> dict[str, int]:
        # [FUNCTION DOC] HELPER: Returns a mapping from each reconciled key to how many times it
        # was reconciled.
        return {parent_key: self.parent_keys.count(parent_key) for parent_key in set(self.parent_keys)}


async def make_loop(
    *,
    source: FakeSource | None = None,
    recorder: Recorder | None = None,
    key_map: Any = pod_cell,
    clock: FakeClock | None = None,
    start: bool = True,
    initial: list[Any] | None = None,
    fail_parent_keys: set[str] | None = None,
    gated: bool = False,
    **kwargs: Any,
) -> tuple[ReconcileLoop, FakeSource, Recorder, FakeClock]:
    # [FUNCTION DOC] HELPER: Builds a reconcile loop around the given or new fake source,
    # recorder and fake clock, with the key map (default: the pod's cell label) and extra
    # options. Attaches the loop to the recorder, sets its failing keys, and creates the gate
    # if gated. If start is true (default), starts the loop in a task, emits a replace event
    # holding the initial objects, lets things settle, asserts the start completed, and awaits
    # it. Returns the loop, source, recorder and clock.
    source = source or FakeSource()
    recorder = recorder or Recorder()
    clock = clock or FakeClock()
    loop = ReconcileLoop(source=source, reconcile=recorder, key_map=key_map, clock=clock, **kwargs)
    recorder.loop = loop
    recorder.fail_parent_keys = fail_parent_keys or set()
    if gated:
        recorder.gate = asyncio.Event()
    if start:
        start_task = asyncio.create_task(loop.start())
        await settle()
        source.emit(replace_of(*(initial or [])))
        await settle()
        assert start_task.done()
        await start_task
    return loop, source, recorder, clock


class TestCadenceValidation:
    # [CLASS DOC] Validation of timing options at construction.
    @pytest.mark.parametrize(
        "kwargs", [dict(resync_period=0.0), dict(resync_period=-1.0), dict(source_retry_delay=0.0)]
    )
    async def test_a_non_positive_cadence_is_rejected(self, kwargs):
        # [FUNCTION DOC] Parametrized over zero or negative resync period and zero source retry
        # delay; asserts construction fails an assertion.
        """A zero delay would turn a retry or resync loop into a hot loop, so it never reaches construction."""
        with pytest.raises(AssertionError):
            ReconcileLoop(source=FakeSource(), reconcile=Recorder(), clock=FakeClock(), **kwargs)

    async def test_the_default_period_is_on_and_far_from_any_test_horizon(self):
        # [FUNCTION DOC] After the initial reconcile, advances 10000 s and asserts no reconcile;
        # advancing the default resync period then triggers exactly one reconcile of cell-a.
        """Resync defaults on, so a caller that never thinks about it still gets the backstop."""
        loop, _, recorder, clock = await make_loop(initial=[make_pod("pod-0")])
        recorder.parent_keys.clear()

        await clock.elapse(10_000.0)
        await settle()
        assert recorder.parent_keys == []

        await clock.elapse(DEFAULT_RESYNC_PERIOD)
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_resync_period_none_is_accepted(self):
        # [FUNCTION DOC] Constructing a loop with resync disabled raises nothing.
        """None is how resync is disabled, and stays distinct from an invalid zero."""
        ReconcileLoop(source=FakeSource(), reconcile=Recorder(), clock=FakeClock(), resync_period=None)


class TestInitialSync:
    # [CLASS DOC] Start-up behavior: the barrier until the first replace event, reconcile
    # dispatch from the initial objects, and source retries before the first replace.
    @pytest.mark.parametrize("context_entry", [False, True])
    async def test_cancelled_initial_sync_leaves_no_background_tasks_without_explicit_stop(
        self, context_entry: bool
    ) -> None:
        # [FUNCTION DOC] Parametrized over starting via start or via async context entry. Starts
        # in a task, checks timers are pending, cancels it (asserts cancellation), then asserts
        # the source was closed once, no timers remain, and after advancing 10 s the source was
        # not reopened.
        """Failed startup releases its source and retry timer without caller cleanup."""
        source = FakeSource()
        clock = FakeClock()
        loop = ReconcileLoop(source=source, reconcile=Recorder(), clock=clock)
        startup = asyncio.create_task(loop.__aenter__() if context_entry else loop.start())
        await settle()
        assert clock.pending_count > 0

        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup

        assert source.closed_count == 1
        assert clock.pending_count == 0
        await clock.elapse(10.0)
        await settle()
        assert source.open_count == 1
        assert clock.pending_count == 0

    async def test_start_blocks_until_the_initial_replace(self):
        # [FUNCTION DOC] Start does not finish until the source emits a replace event, then it
        # finishes.
        """start() returns only once the initial LIST has landed in the store."""
        source = FakeSource()
        loop = ReconcileLoop(source=source, reconcile=Recorder(), clock=FakeClock())
        start_task = asyncio.create_task(loop.start())
        await settle()
        assert not start_task.done()

        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        assert start_task.done()
        await loop.stop()

    async def test_no_reconcile_before_the_initial_replace(self):
        # [FUNCTION DOC] No reconcile occurs before the replace; after it exactly one reconcile
        # of cell-a.
        """No reconcile is dispatched while the initial LIST is still outstanding."""
        source = FakeSource()
        recorder = Recorder()
        loop = ReconcileLoop(source=source, reconcile=recorder, key_map=pod_cell, clock=FakeClock())
        start_task = asyncio.create_task(loop.start())
        await settle()
        assert recorder.parent_keys == []

        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        await start_task
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_initial_objects_reconcile_once_per_parent_key(self):
        # [FUNCTION DOC] Eight pods in one cell give exactly one reconcile of that cell.
        """Eight pods of one cell collapse into a single reconcile of that cell."""
        pods = [make_pod(f"pod-{i}", cell="cell-a") for i in range(8)]
        loop, _, recorder, _ = await make_loop(initial=pods)
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_initial_objects_reconcile_each_distinct_parent(self):
        # [FUNCTION DOC] Pods in two cells give one reconcile each.
        """Distinct cells each get exactly one initial reconcile."""
        pods = [make_pod("pod-0", cell="cell-a"), make_pod("pod-1", cell="cell-b")]
        loop, _, recorder, _ = await make_loop(initial=pods)
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-b"]
        await loop.stop()

    async def test_store_is_populated_before_first_reconcile(self):
        # [FUNCTION DOC] With three pods in one cell, the single snapshot taken in the first
        # reconcile lists all three names in order.
        """The first reconcile already sees the full initial LIST in the store."""
        pods = [make_pod(f"pod-{i}", cell="cell-a") for i in range(3)]
        loop, _, recorder, _ = await make_loop(initial=pods)
        assert recorder.snapshots == [["pod-0", "pod-1", "pod-2"]]
        await loop.stop()

    async def test_without_key_map_object_key_is_the_reconcile_key(self):
        # [FUNCTION DOC] With no key map, the reconciled keys are the pod names themselves.
        """key_map=None means the object key is used verbatim as the reconcile key."""
        loop, _, recorder, _ = await make_loop(key_map=None, initial=[make_pod("pod-0"), make_pod("pod-1")])
        assert sorted(recorder.parent_keys) == ["pod-0", "pod-1"]
        await loop.stop()

    async def test_start_retries_when_the_first_stream_fails(self):
        # [FUNCTION DOC] The first stream fails on open with retry delay 5 s. Start stays
        # pending after one open; after 5 s the source was opened twice; a replace then lets
        # start finish with one reconcile of cell-a.
        """A source that fails before delivering its initial LIST is reopened after the retry delay."""
        source = FakeSource(fail_opens=1)
        recorder = Recorder()
        clock = FakeClock()
        loop = ReconcileLoop(source=source, reconcile=recorder, key_map=pod_cell, clock=clock, source_retry_delay=5.0)
        start_task = asyncio.create_task(loop.start())
        await settle()
        assert not start_task.done()
        assert source.open_count == 1

        await clock.elapse(5.0)
        await settle()
        assert source.open_count == 2

        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        await start_task
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_a_stream_ending_before_its_replace_is_retried(self):
        # [FUNCTION DOC] The stream ends before delivering a replace (retry delay 2 s): start is
        # pending and nothing reconciled. After the delay a fresh replace with "pod-fresh" lets
        # start finish; asserts one reconcile of cell-a, the store holds pod-fresh, and the
        # source was opened twice.
        """A stream that dies before delivering the initial LIST leaves the barrier closed and nothing cached."""
        source = FakeSource()
        recorder = Recorder()
        clock = FakeClock()
        loop = ReconcileLoop(source=source, reconcile=recorder, key_map=pod_cell, clock=clock, source_retry_delay=2.0)
        recorder.loop = loop
        start_task = asyncio.create_task(loop.start())
        await settle()
        source.emit(StreamEnd())
        await settle()
        assert not start_task.done()
        assert recorder.parent_keys == []

        await clock.elapse(2.0)
        await settle()
        source.emit(replace_of(make_pod("pod-fresh")))
        await settle()
        await start_task

        assert recorder.parent_keys == ["cell-a"]
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-fresh"]
        assert source.open_count == 2
        await loop.stop()

    async def test_source_factory_raising_is_retried(self):
        # [FUNCTION DOC] The first source call raises immediately, retry delay 3 s. Asserts one
        # open at first and still one after 2.9 s; two opens after 3 s; an empty replace then
        # lets start finish.
        """A source callable that raises synchronously is retried after the delay."""
        source = FakeSource(fail_calls=1)
        recorder = Recorder()
        clock = FakeClock()
        loop = ReconcileLoop(source=source, reconcile=recorder, key_map=pod_cell, clock=clock, source_retry_delay=3.0)
        start_task = asyncio.create_task(loop.start())
        await settle()
        assert source.open_count == 1

        await clock.elapse(2.9)
        await settle()
        assert source.open_count == 1

        await clock.elapse(0.1)
        await settle()
        assert source.open_count == 2
        source.emit(replace_of())
        await settle()
        await start_task
        await loop.stop()

    async def test_store_is_updated_before_the_key_is_enqueued(self):
        # [FUNCTION DOC] Wraps the loop's private queue add (replacing it) to snapshot the store
        # for the key whenever a key is enqueued. An upsert of pod-1 must show both pods in the
        # snapshot; a delete of pod-1 must show only pod-0.
        """The invariant that a handler never reads a world older than the event it woke on."""
        loop, source, _, _ = await make_loop(initial=[make_pod("pod-0")])
        seen_at_enqueue: list[list[str]] = []
        original_add = loop._queue.add

        def recording_add(key: str) -> None:
            # [FUNCTION DOC] HELPER: Records the names of the objects the store holds for the key
            # and then calls the original queue add.
            seen_at_enqueue.append([pod.metadata.name for pod in loop.get_by_parent(key)])
            original_add(key)

        loop._queue.add = recording_add
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1")))
        await settle()
        assert seen_at_enqueue == [["pod-0", "pod-1"]]

        source.emit(DeleteEvent(key="pod-1", last_obj=make_pod("pod-1")))
        await settle()
        assert seen_at_enqueue[-1] == ["pod-0"]
        await loop.stop()

    async def test_empty_initial_list_reconciles_nothing(self):
        # [FUNCTION DOC] An empty initial replace causes no reconcile.
        """An empty cluster produces no reconcile calls."""
        loop, _, recorder, _ = await make_loop(initial=[])
        assert recorder.parent_keys == []
        await loop.stop()


class TestIncrementalEvents:
    # [CLASS DOC] Effects of upsert and delete events on the store and on reconcile dispatch.
    async def test_upsert_enqueues_parent_key(self):
        # [FUNCTION DOC] An upsert of a pod in cell-a causes one reconcile of cell-a.
        """A new pod wakes its cell."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_delete_removes_from_store_and_wakes_parent(self):
        # [FUNCTION DOC] Deleting the only pod of cell-a gives a second reconcile of cell-a and
        # leaves the store empty for it.
        """A deleted pod leaves the store and wakes its cell."""
        pod = make_pod("pod-0", cell="cell-a")
        loop, source, recorder, _ = await make_loop(initial=[pod])
        source.emit(DeleteEvent(key="pod-0", last_obj=pod))
        await settle()
        assert recorder.parent_keys == ["cell-a", "cell-a"]
        assert loop.get_by_parent("cell-a") == []
        await loop.stop()

    async def test_known_delete_trusts_the_store_over_a_stale_tombstone(self):
        # [FUNCTION DOC] A delete whose tombstone names cell-stale for an object stored in
        # cell-a reconciles cell-a only. A following empty replace adds no reconcile, showing the
        # entry was really dropped.
        """For a known object the stored copy decides the parent, and the store entry is really dropped."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0", cell="cell-a")])
        recorder.parent_keys.clear()
        source.emit(DeleteEvent(key="pod-0", last_obj=make_pod("pod-0", cell="cell-stale")))
        await settle()

        assert recorder.parent_keys == ["cell-a"]

        source.emit(replace_of())
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_delete_of_unknown_key_falls_back_to_tombstone(self):
        # [FUNCTION DOC] Deleting an unknown key with a tombstone in cell-z reconciles cell-z.
        """A delete for an object never seen still wakes the cell via its tombstone."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(DeleteEvent(key="pod-9", last_obj=make_pod("pod-9", cell="cell-z")))
        await settle()
        assert recorder.parent_keys == ["cell-z"]
        await loop.stop()

    async def test_delete_without_tombstone_is_ignored(self):
        # [FUNCTION DOC] Deleting an unknown key with no tombstone causes no reconcile, and a
        # later upsert still works (one reconcile of cell-a).
        """A delete with no last_obj for an unknown key cannot be attributed and is dropped."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(DeleteEvent(key="pod-9", last_obj=None))
        await settle()
        assert recorder.parent_keys == []

        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0")))
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_delete_with_an_unmappable_tombstone_is_dropped(self):
        # [FUNCTION DOC] Deleting an unknown key whose tombstone lacks a cell label causes no
        # reconcile and the key is not in the (private) store; a later upsert reconciles cell-a.
        """An unknown delete whose tombstone has no parent is discarded, not queued under its own key."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(DeleteEvent(key="pod-orphan", last_obj=make_pod("pod-orphan", cell=None)))
        await settle()

        assert recorder.parent_keys == []
        assert "pod-orphan" not in loop._store

        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0")))
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_reparenting_upsert_wakes_old_and_new_parent(self):
        # [FUNCTION DOC] An upsert moving a pod from cell-a to cell-b reconciles both cells and
        # moves the pod in the lister view.
        """A pod whose cell label changed wakes both the old and the new cell."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0", cell="cell-a")])
        recorder.parent_keys.clear()
        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-b")))
        await settle()
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-b"]
        assert loop.get_by_parent("cell-a") == []
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-b")] == ["pod-0"]
        await loop.stop()

    async def test_repeated_upsert_keeps_one_store_entry(self):
        # [FUNCTION DOC] Upserting the same pod with a newer version leaves one entry carrying
        # the new version.
        """Re-delivery of the same object does not duplicate store membership."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0", resource_version="1")])
        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0", resource_version="2")))
        await settle()
        pods = loop.get_by_parent("cell-a")
        assert len(pods) == 1
        assert pods[0].metadata.resource_version == "2"
        await loop.stop()

    async def test_reconcile_reads_store_snapshot_not_event_payload(self):
        # [FUNCTION DOC] Three upserts emitted in a burst fold into one reconcile whose snapshot
        # contains all three pods.
        """Reconcile observes the store, so a burst of pods folds into one complete view."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(
            UpsertEvent(key="pod-0", obj=make_pod("pod-0")),
            UpsertEvent(key="pod-1", obj=make_pod("pod-1")),
            UpsertEvent(key="pod-2", obj=make_pod("pod-2")),
        )
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        assert recorder.snapshots[-1] == ["pod-0", "pod-1", "pod-2"]
        await loop.stop()

    async def test_an_unmappable_object_is_dropped_not_stored(self):
        # [FUNCTION DOC] An upsert of a pod without a cell label is not stored and triggers no
        # reconcile, the source is not reopened, and a later valid upsert reconciles cell-a.
        """A key_map that raises drops that one object; the stream keeps running."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0")])
        recorder.parent_keys.clear()
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1", cell=None)))
        await settle()

        assert "pod-1" not in loop._store
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-0"]
        assert source.open_count == 1

        source.emit(UpsertEvent(key="pod-2", obj=make_pod("pod-2")))
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_an_object_that_becomes_unmappable_leaves_its_parent(self):
        # [FUNCTION DOC] An upsert removing the cell label of a stored pod reconciles its old
        # cell once, empties that cell's view and removes the entry from the store.
        """Losing the label is observed as the object leaving the cell."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0")])
        recorder.parent_keys.clear()
        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell=None)))
        await settle()

        assert recorder.parent_keys == ["cell-a"]
        assert loop.get_by_parent("cell-a") == []
        assert "pod-0" not in loop._store
        await loop.stop()


class TestWorkQueueDiscipline:
    # [CLASS DOC] Serialization, collapsing and ordering of reconcile dispatch (single worker).
    async def test_a_busy_key_is_never_reconciled_twice_at_once(self):
        # [FUNCTION DOC] With a gated reconcile, emits upserts for two cells; asserts maximum
        # concurrency 1 before and after opening the gate and that both cells get reconciled.
        """The single worker serializes everything, so a key can never overlap itself."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0")], gated=True)
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1")))
        await settle()
        source.emit(UpsertEvent(key="pod-2", obj=make_pod("pod-2", cell="cell-b")))
        await settle()
        assert recorder.max_concurrency == 1

        assert recorder.gate is not None
        recorder.gate.set()
        await settle()
        assert recorder.max_concurrency == 1
        assert sorted(set(recorder.parent_keys)) == ["cell-a", "cell-b"]
        await loop.stop()

    async def test_repeated_events_while_busy_collapse_into_one_requeue(self):
        # [FUNCTION DOC] Gates the reconcile, then emits one upsert and five more for the same
        # cell while it is busy; after opening the gate the cell is reconciled exactly twice
        # (the busy run plus one follow-up).
        """Many events for a busy key produce exactly one follow-up reconcile."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0")])
        recorder.parent_keys.clear()
        recorder.gate = asyncio.Event()
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1")))
        await settle()
        for index in range(5):
            source.emit(UpsertEvent(key=f"pod-extra-{index}", obj=make_pod(f"pod-extra-{index}")))
            await settle()

        recorder.gate.set()
        await settle()
        assert recorder.parent_keys == ["cell-a", "cell-a"]
        await loop.stop()

    async def test_distinct_keys_are_serialized_by_default(self):
        # [FUNCTION DOC] With two cells and a gated reconcile, concurrency stays 1; after the
        # gate opens both cells were reconciled.
        """The default single worker mirrors controller-runtime's MaxConcurrentReconciles=1."""
        pods = [make_pod("pod-a", cell="cell-a"), make_pod("pod-b", cell="cell-b")]
        loop, _, recorder, _ = await make_loop(initial=pods, gated=True)
        assert recorder.max_concurrency == 1

        assert recorder.gate is not None
        recorder.gate.set()
        await settle()
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-b"]
        await loop.stop()

    async def test_events_before_worker_pickup_collapse(self):
        # [FUNCTION DOC] Two upserts for the same pod emitted back to back give one reconcile.
        """Two events for the same key arriving back-to-back yield one reconcile."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(
            UpsertEvent(key="pod-0", obj=make_pod("pod-0")),
            UpsertEvent(key="pod-0", obj=make_pod("pod-0", resource_version="2")),
        )
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_keys_are_dispatched_in_arrival_order(self):
        # [FUNCTION DOC] Gated reconcile; cells b and c arrive after the running cell a; after
        # opening the gate the order is cell-a, cell-b, cell-c.
        """The queue is FIFO, not a stack."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0")], gated=True)
        source.emit(UpsertEvent(key="pod-b", obj=make_pod("pod-b", cell="cell-b")))
        await settle()
        source.emit(UpsertEvent(key="pod-c", obj=make_pod("pod-c", cell="cell-c")))
        await settle()

        assert recorder.gate is not None
        recorder.gate.set()
        await settle()
        assert recorder.parent_keys == ["cell-a", "cell-b", "cell-c"]
        await loop.stop()


class TestFailureBackoff:
    # [CLASS DOC] Per-key exponential backoff after failed reconciles, on the fake clock.
    async def test_failed_reconcile_is_retried_only_after_the_backoff_delay(self):
        # [FUNCTION DOC] With base delay 2 s and a failing cell: one call at first and after
        # 1 s; two calls after 2 s.
        """A raising reconcile is requeued when — and not before — the delay elapses."""
        loop, _, recorder, clock = await make_loop(
            initial=[make_pod("pod-0")], fail_parent_keys={"cell-a"}, failure_base_delay=2.0
        )
        assert recorder.counts() == {"cell-a": 1}

        await clock.elapse(1.0)
        await settle()
        assert recorder.counts() == {"cell-a": 1}

        await clock.elapse(1.0)
        await settle()
        assert recorder.counts() == {"cell-a": 2}
        await loop.stop()

    async def test_backoff_grows_exponentially_and_resets_on_success(self):
        # [FUNCTION DOC] Base delay 1 s, max 8 s, failing cell. Asserts call counts across the
        # sequence: retry after 1 s (2 calls), the next retry needs 2 s (3 calls after another
        # 2 s), then with failures cleared a retry after 4 s succeeds (4 calls), and a new
        # failure after a fresh event restarts at the base delay (5 calls, then 6 after 1 s).
        """Retry delay doubles per consecutive failure and resets after a success."""
        source = FakeSource()
        recorder = Recorder()
        clock = FakeClock()
        loop = ReconcileLoop(
            source=source,
            reconcile=recorder,
            key_map=pod_cell,
            clock=clock,
            failure_base_delay=1.0,
            failure_max_delay=8.0,
        )
        recorder.loop = loop
        recorder.fail_parent_keys = {"cell-a"}
        start_task = asyncio.create_task(loop.start())
        await settle()
        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        await start_task
        assert recorder.counts() == {"cell-a": 1}

        await clock.elapse(1.0)
        await settle()
        assert recorder.counts() == {"cell-a": 2}

        await clock.elapse(1.0)
        await settle()
        assert recorder.counts() == {"cell-a": 2}
        await clock.elapse(1.0)
        await settle()
        assert recorder.counts() == {"cell-a": 3}

        recorder.fail_parent_keys = set()
        await clock.elapse(4.0)
        await settle()
        assert recorder.counts() == {"cell-a": 4}

        recorder.fail_parent_keys = {"cell-a"}
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1")))
        await settle()
        assert recorder.counts() == {"cell-a": 5}
        await clock.elapse(1.0)
        await settle()
        assert recorder.counts() == {"cell-a": 6}
        await loop.stop()

    async def test_backoff_is_capped(self):
        # [FUNCTION DOC] Base 1 s, max 2 s, always failing. Advancing 2 s at a time yields one
        # more call each time, up to 7 calls total.
        """The retry delay never exceeds failure_max_delay."""
        source = FakeSource()
        recorder = Recorder()
        clock = FakeClock()
        loop = ReconcileLoop(
            source=source,
            reconcile=recorder,
            key_map=pod_cell,
            clock=clock,
            failure_base_delay=1.0,
            failure_max_delay=2.0,
        )
        recorder.loop = loop
        recorder.fail_parent_keys = {"cell-a"}
        start_task = asyncio.create_task(loop.start())
        await settle()
        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        await start_task

        for expected in range(2, 8):
            await clock.elapse(2.0)
            await settle()
            assert recorder.counts() == {"cell-a": expected}
        await loop.stop()

    async def test_a_fresh_failure_installs_its_own_delay(self):
        # [FUNCTION DOC] After a first failure, an event at 0.5 s causes a second call (a second
        # failure with a 2 s delay replacing the pending 1 s timer). Nothing happens at 1 s; the
        # third call occurs after 3 s in total from the second failure.
        """Retry is latest-wins: a new failure replaces the pending timer, even with a longer delay."""
        loop, source, recorder, clock = await make_loop(
            initial=[make_pod("pod-0")], fail_parent_keys={"cell-a"}, failure_base_delay=1.0, failure_max_delay=64.0
        )
        assert recorder.counts() == {"cell-a": 1}

        await clock.elapse(0.5)
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1")))
        await settle()
        assert recorder.counts() == {"cell-a": 2}

        await clock.elapse(0.5)
        await settle()
        assert recorder.counts() == {"cell-a": 2}

        await clock.elapse(1.5)
        await settle()
        assert recorder.counts() == {"cell-a": 3}
        await loop.stop()

    async def test_a_success_cancels_the_pending_retry(self):
        # [FUNCTION DOC] After a failure with a 10 s delay, failures are cleared and an event
        # triggers a successful reconcile (2 calls). Advancing 100 s causes no further call and
        # the loop's private retry table is empty.
        """A key that recovers must not be woken again by the timer its failure installed."""
        loop, source, recorder, clock = await make_loop(
            initial=[make_pod("pod-0")], fail_parent_keys={"cell-a"}, failure_base_delay=10.0, failure_max_delay=10.0
        )
        recorder.fail_parent_keys = set()
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1")))
        await settle()
        assert recorder.counts() == {"cell-a": 2}

        await clock.elapse(100.0)
        await settle()
        assert recorder.counts() == {"cell-a": 2}
        assert loop._retry._infos == {}
        await loop.stop()

    async def test_endless_failures_do_not_kill_the_worker(self):
        # [FUNCTION DOC] Sets the private failure count for the cell to 5000 and advances time
        # in 2 s steps (max delay 2 s); asserts calls continue (2 then 3), i.e. no overflow.
        """A very large consecutive-failure count must not overflow the backoff computation."""
        loop, _, recorder, clock = await make_loop(
            initial=[make_pod("pod-0")], fail_parent_keys={"cell-a"}, failure_base_delay=1.0, failure_max_delay=2.0
        )
        loop._retry._infos["cell-a"].failures = 5000

        await clock.elapse(2.0)
        await settle()
        assert recorder.counts() == {"cell-a": 2}

        await clock.elapse(2.0)
        await settle()
        assert recorder.counts() == {"cell-a": 3}
        await loop.stop()

    async def test_another_keys_success_does_not_reset_a_failing_keys_backoff(self):
        # [FUNCTION DOC] Cell-a fails, cell-b succeeds. After 1 s cell-a has been retried once;
        # a second event for cell-b reconciles it again without resetting cell-a's counter.
        # After 1.9 s cell-a is still at 2 calls; after a further 0.1 s plus the poll interval
        # it reaches 3 (the second delay is 2 s).
        """Failure counters are per key."""
        pods = [make_pod("pod-a", cell="cell-a"), make_pod("pod-b", cell="cell-b")]
        loop, source, recorder, clock = await make_loop(
            initial=pods, fail_parent_keys={"cell-a"}, failure_base_delay=1.0, failure_max_delay=64.0
        )
        await clock.elapse(1.0)
        await settle()
        assert recorder.counts() == {"cell-a": 2, "cell-b": 1}

        source.emit(UpsertEvent(key="pod-b2", obj=make_pod("pod-b2", cell="cell-b")))
        await settle()
        assert recorder.counts() == {"cell-a": 2, "cell-b": 2}

        await clock.elapse(1.9)
        await settle()
        assert recorder.counts()["cell-a"] == 2

        await clock.elapse(0.1 + POLL_INTERVAL)
        await settle()
        assert recorder.counts()["cell-a"] == 3
        await loop.stop()

    async def test_failing_keys_keep_independent_retry_deadlines(self):
        # [FUNCTION DOC] Cells a and b both fail with base delay 2 s. Cell-b first fails at 1 s;
        # asserts cell-a is retried at 2 s and cell-b at 3 s, each by its own deadline.
        """One key rescheduling its retry must not cancel another key's pending timer."""
        loop, source, recorder, clock = await make_loop(
            initial=[make_pod("pod-0")], fail_parent_keys={"cell-a", "cell-b"}, failure_base_delay=2.0
        )
        assert recorder.counts() == {"cell-a": 1}

        await clock.elapse(1.0)
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1", cell="cell-b")))
        await settle()
        assert recorder.counts() == {"cell-a": 1, "cell-b": 1}

        await clock.elapse(1.0)
        await settle()
        assert recorder.counts() == {"cell-a": 2, "cell-b": 1}

        await clock.elapse(1.0)
        await settle()
        assert recorder.counts() == {"cell-a": 2, "cell-b": 2}
        await loop.stop()

    async def test_other_keys_progress_while_one_key_backs_off(self):
        # [FUNCTION DOC] With cell-a failing under a 100 s backoff, events for cell-a and cell-b
        # are reconciled in that order without waiting.
        """A failing key does not block reconciles of other keys."""
        loop, source, recorder, clock = await make_loop(
            initial=[make_pod("pod-0", cell="cell-a")], failure_base_delay=100.0, failure_max_delay=100.0
        )
        recorder.fail_parent_keys = {"cell-a"}
        recorder.parent_keys.clear()
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1", cell="cell-a")))
        await settle()
        source.emit(UpsertEvent(key="pod-2", obj=make_pod("pod-2", cell="cell-b")))
        await settle()
        assert recorder.parent_keys == ["cell-a", "cell-b"]
        await loop.stop()


class TestResync:
    # [CLASS DOC] Periodic resync of every known parent key.
    async def test_resync_reenqueues_all_known_keys(self):
        # [FUNCTION DOC] Period 30 s with two cells: each 30 s step reconciles both cells again.
        """The resync tick re-reconciles every key currently in the store."""
        pods = [make_pod("pod-0", cell="cell-a"), make_pod("pod-1", cell="cell-b")]
        loop, _, recorder, clock = await make_loop(initial=pods, resync_period=30.0)
        recorder.parent_keys.clear()

        await clock.elapse(30.0)
        await settle()
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-b"]

        await clock.elapse(30.0)
        await settle()
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-a", "cell-b", "cell-b"]
        await loop.stop()

    async def test_resync_waits_a_full_period_and_covers_parents_added_later(self):
        # [FUNCTION DOC] After adding a cell, nothing happens at 29.9 s; at 30 s both the old
        # and the new cell are reconciled.
        """The first tick is a full period away, and it uses the current parent set."""
        loop, source, recorder, clock = await make_loop(initial=[make_pod("pod-0")], resync_period=30.0)
        recorder.parent_keys.clear()
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1", cell="cell-new")))
        await settle()
        recorder.parent_keys.clear()

        await clock.elapse(29.9)
        await settle()
        assert recorder.parent_keys == []

        await clock.elapse(0.1)
        await settle()
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-new"]
        await loop.stop()

    async def test_no_resync_when_period_is_none(self):
        # [FUNCTION DOC] With resync disabled, 10000 s pass with no reconcile.
        """resync_period=None opts out of the backstop, leaving the loop purely event driven."""
        loop, _, recorder, clock = await make_loop(initial=[make_pod("pod-0")], resync_period=None)
        recorder.parent_keys.clear()
        await clock.elapse(10_000.0)
        await settle()
        assert recorder.parent_keys == []
        await loop.stop()

    async def test_resync_drives_a_key_that_is_waiting_on_a_retry_timer(self):
        # [FUNCTION DOC] A failing cell parked on a 1000 s backoff is reconciled again by the
        # resync at 10 s (2 calls).
        """A key parked on a long backoff is still re-driven by the resync tick."""
        loop, _, recorder, clock = await make_loop(
            initial=[make_pod("pod-0")],
            fail_parent_keys={"cell-a"},
            resync_period=10.0,
            failure_base_delay=1000.0,
            failure_max_delay=1000.0,
        )
        assert recorder.counts() == {"cell-a": 1}

        await clock.elapse(10.0)
        await settle()
        assert recorder.counts() == {"cell-a": 2}
        await loop.stop()

    async def test_a_resync_failure_reinstalls_the_backoff_and_a_success_clears_it(self):
        # [FUNCTION DOC] Resync 10 s, backoff 4 s, failing cell: retry at 4 s makes 2 calls and a
        # private failure count of 2. With failures cleared, the resync at 10 s makes call 3 and
        # a further second makes call 4; the private retry table no longer holds the cell.
        """Retries driven by resync keep the latest-wins timer discipline."""
        loop, _, recorder, clock = await make_loop(
            initial=[make_pod("pod-0")],
            fail_parent_keys={"cell-a"},
            resync_period=10.0,
            failure_base_delay=4.0,
            failure_max_delay=4.0,
        )
        await clock.elapse(4.0)
        await settle()
        assert recorder.counts() == {"cell-a": 2}
        assert loop._retry._infos["cell-a"].failures == 2

        recorder.fail_parent_keys = set()
        await clock.elapse(5.0)
        await settle()
        assert recorder.counts() == {"cell-a": 3}

        await clock.elapse(1.0)
        await settle()
        assert recorder.counts() == {"cell-a": 4}
        assert "cell-a" not in loop._retry._infos
        await loop.stop()

    async def test_resync_skips_deleted_keys(self):
        # [FUNCTION DOC] After the only pod is deleted, a resync tick reconciles nothing.
        """Keys whose objects are gone are not resynced."""
        pod = make_pod("pod-0", cell="cell-a")
        loop, source, recorder, clock = await make_loop(initial=[pod], resync_period=30.0)
        source.emit(DeleteEvent(key="pod-0", last_obj=pod))
        await settle()
        recorder.parent_keys.clear()

        await clock.elapse(30.0)
        await settle()
        assert recorder.parent_keys == []
        await loop.stop()


class TestLister:
    # [CLASS DOC] The read-only view of the store by parent key.
    async def test_get_by_parent_returns_members_sorted_by_key(self):
        # [FUNCTION DOC] Three pods inserted out of order come back sorted by key.
        """get_by_parent is the Lister equivalent and is deterministic."""
        pods = [make_pod("pod-2"), make_pod("pod-0"), make_pod("pod-1")]
        loop, _, _, _ = await make_loop(initial=pods)
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-0", "pod-1", "pod-2"]
        await loop.stop()

    async def test_get_by_parent_unknown_key_is_empty(self):
        # [FUNCTION DOC] An unknown parent key yields an empty list.
        """Unknown parents read as empty, not as an error."""
        loop, _, _, _ = await make_loop(initial=[])
        assert loop.get_by_parent("nope") == []
        await loop.stop()


class TestStop:
    # [CLASS DOC] Shutdown semantics of the loop.
    async def test_stop_halts_reconciles(self):
        # [FUNCTION DOC] After stop, an emitted upsert causes no reconcile.
        """After stop() no further events are processed."""
        loop, source, recorder, _ = await make_loop(initial=[])
        await loop.stop()
        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0")))
        await settle()
        assert recorder.parent_keys == []

    async def test_stop_closes_the_source_stream(self):
        # [FUNCTION DOC] The source stream is closed once, and only by stop.
        """Cancelling the driver closes the stream it was iterating."""
        loop, source, _, _ = await make_loop(initial=[make_pod("pod-0")])
        assert source.closed_count == 0

        await loop.stop()
        assert source.closed_count == 1

    async def test_a_second_stop_is_a_no_op(self):
        # [FUNCTION DOC] Calling stop twice raises nothing and closes the source only once.
        """Teardown paths overlap, and a stop() that raises would replace whatever error is unwinding."""
        loop, source, _, _ = await make_loop(initial=[])
        await loop.stop()

        await loop.stop()

        assert source.closed_count == 1

    async def test_stop_before_start_is_rejected(self):
        # [FUNCTION DOC] Stop on a never-started loop fails an assertion.
        """stop() has nothing to wait for before start(); calling it early is a caller error."""
        loop = ReconcileLoop(source=FakeSource(), reconcile=Recorder(), clock=FakeClock())
        with pytest.raises(AssertionError):
            await loop.stop()

    async def test_stop_after_an_aborted_start_lets_the_real_failure_through(self):
        # [FUNCTION DOC] Cancels a pending start, then stop raises nothing and the start task
        # stays cancelled.
        """A caller unwinding a cancelled start() calls stop(); asserting there would hide the CancelledError."""
        source = FakeSource()
        loop = ReconcileLoop(source=source, reconcile=Recorder(), key_map=pod_cell, clock=FakeClock())
        start_task = asyncio.create_task(loop.start())
        await settle()
        start_task.cancel()
        await asyncio.gather(start_task, return_exceptions=True)

        await loop.stop()

        assert start_task.cancelled()

    async def test_start_twice_is_rejected(self):
        # [FUNCTION DOC] A second start on a running loop fails an assertion; stop afterwards.
        """start() is single-shot."""
        loop, _, _, _ = await make_loop(initial=[make_pod("pod-0")])
        with pytest.raises(AssertionError):
            await loop.start()
        await loop.stop()

    async def test_stop_cancels_every_background_timer(self):
        # [FUNCTION DOC] With a failing cell, resync and retry timers pending, stop leaves no
        # pending timer on the fake clock and the cell was reconciled only once.
        """No source, resync, or retry sleeper survives stop()."""
        loop, _, recorder, clock = await make_loop(
            initial=[make_pod("pod-0")], fail_parent_keys={"cell-a"}, resync_period=30.0, failure_base_delay=5.0
        )
        assert clock.pending_count > 0

        await loop.stop()
        assert clock.pending_count == 0
        assert recorder.counts() == {"cell-a": 1}

    async def test_awaiting_stop_inside_reconcile_is_rejected(self):
        # [FUNCTION DOC] A reconcile that awaits stop of its own loop gets an assertion error
        # (recorded); asserts exactly one rejection, then stops normally.
        """stop() waits for the worker, so awaiting it from inside reconcile is a caller error, not a hang."""
        source = FakeSource()
        rejected: list[BaseException] = []
        loop: ReconcileLoop | None = None

        async def reconcile(key: str) -> None:
            # [FUNCTION DOC] HELPER: Awaits the loop's stop and records the assertion error it
            # raises.
            assert loop is not None
            try:
                await loop.stop()
            except AssertionError as error:
                rejected.append(error)

        loop = ReconcileLoop(source=source, reconcile=reconcile, key_map=pod_cell, clock=FakeClock())
        start_task = asyncio.create_task(loop.start())
        await settle()
        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        await start_task

        assert len(rejected) == 1
        await loop.stop()

    async def test_a_reconcile_shuts_the_loop_down_by_spawning_a_task(self):
        # [FUNCTION DOC] A reconcile spawns a task calling stop (once); awaiting that task works
        # and the source ends closed once.
        """The sanctioned way for reconcile to request shutdown is a task that outlives the worker."""
        source = FakeSource()
        loop: ReconcileLoop | None = None
        shutdown: asyncio.Task[None] | None = None

        async def reconcile(key: str) -> None:
            # [FUNCTION DOC] HELPER: On the first call creates a task that stops the loop and
            # remembers it; later calls do nothing.
            nonlocal shutdown
            assert loop is not None
            if shutdown is None:
                shutdown = asyncio.create_task(loop.stop())

        loop = ReconcileLoop(source=source, reconcile=reconcile, key_map=pod_cell, clock=FakeClock())
        start_task = asyncio.create_task(loop.start())
        await settle()
        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        await start_task

        assert shutdown is not None
        await shutdown
        assert source.closed_count == 1

    async def test_stop_cancels_pending_backoff_retries(self):
        # [FUNCTION DOC] After stop, the retry scheduler's private poller task is cancelled and
        # advancing 100 s causes no extra reconcile.
        """Pending retry timers do not survive stop()."""
        loop, _, recorder, clock = await make_loop(
            initial=[make_pod("pod-0")], fail_parent_keys={"cell-a"}, failure_base_delay=5.0
        )
        await loop.stop()

        assert loop._retry._poller.cancelled()
        await clock.elapse(100.0)
        await settle()
        assert recorder.counts() == {"cell-a": 1}


class TestInStreamRelist:
    # [CLASS DOC] Replace events arriving after the initial one.
    async def test_relist_replaces_the_store(self):
        # [FUNCTION DOC] A replace holding only pod-0 removes pod-1 from the store and
        # reconciles both cells.
        """A ReplaceEvent mid-stream swaps the whole world, not just the keys it mentions."""
        pods = [make_pod("pod-0", cell="cell-a"), make_pod("pod-1", cell="cell-b")]
        loop, source, recorder, _ = await make_loop(initial=pods)
        recorder.parent_keys.clear()

        source.emit(replace_of(pods[0]))
        await settle()
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-0"]
        assert loop.get_by_parent("cell-b") == []
        assert sorted(set(recorder.parent_keys)) == ["cell-a", "cell-b"]
        await loop.stop()

    async def test_relist_to_empty_deletes_everything_and_wakes_every_parent(self):
        # [FUNCTION DOC] An empty replace empties the store and reconciles every known cell.
        """A relist that returns nothing must retire every known object."""
        pods = [make_pod("pod-0", cell="cell-a"), make_pod("pod-1", cell="cell-b")]
        loop, source, recorder, _ = await make_loop(initial=pods)
        recorder.parent_keys.clear()

        source.emit(replace_of())
        await settle()
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-b"]
        assert loop.get_by_parent("cell-a") == []
        assert loop.get_by_parent("cell-b") == []
        await loop.stop()

    async def test_relist_refreshes_and_reparents_a_survivor(self):
        # [FUNCTION DOC] A replace moving a pod from cell-a to cell-b with a newer version
        # reconciles both cells and updates the stored copy.
        """A relisted object that moved cells updates both the store copy and both parents."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0", cell="cell-a", resource_version="1")])
        recorder.parent_keys.clear()

        source.emit(replace_of(make_pod("pod-0", cell="cell-b", resource_version="2")))
        await settle()
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-b"]
        assert loop.get_by_parent("cell-a") == []
        assert [pod.metadata.resource_version for pod in loop.get_by_parent("cell-b")] == ["2"]
        await loop.stop()

    async def test_incremental_events_resume_after_a_relist(self):
        # [FUNCTION DOC] After an empty replace, an upsert still reconciles its cell.
        """The stream returns to incremental mode once the relist has been applied."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(replace_of())
        await settle()
        recorder.parent_keys.clear()

        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0")))
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_a_relist_skips_unmappable_objects_and_applies_the_rest(self):
        # [FUNCTION DOC] A replace holding a pod moved to cell-c and a pod without a label: the
        # unlabeled one is skipped, the source is not reopened, cell-c holds pod-a, cells a and b
        # are empty and cells a, b, c are reconciled.
        """One poison object must not stop the whole pool from being updated."""
        pods = [make_pod("pod-a", cell="cell-a"), make_pod("pod-b", cell="cell-b")]
        loop, source, recorder, _ = await make_loop(initial=pods)
        recorder.parent_keys.clear()

        source.emit(replace_of(make_pod("pod-a", cell="cell-c"), make_pod("pod-bad", cell=None)))
        await settle()

        assert source.open_count == 1
        assert "pod-bad" not in loop._store
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-c")] == ["pod-a"]
        assert loop.get_by_parent("cell-a") == []
        assert loop.get_by_parent("cell-b") == []
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-b", "cell-c"]
        await loop.stop()

    async def test_a_relist_removes_a_known_object_that_becomes_unmappable(self):
        # [FUNCTION DOC] A replace in which a stored pod lost its label removes it from the
        # store; only the other pod remains in cell-a.
        """An object still listed but no longer attributable must leave the store, not linger."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0"), make_pod("pod-1")])
        recorder.parent_keys.clear()

        source.emit(replace_of(make_pod("pod-0", cell=None), make_pod("pod-1")))
        await settle()

        assert "pod-0" not in loop._store
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-1"]
        await loop.stop()

    async def test_consecutive_relists_replace_overlapping_then_disjoint_sets(self):
        # [FUNCTION DOC] Two replaces in a row: an overlapping set (pod-1, pod-2) then a disjoint
        # one (pod-3 in cell-b). Asserts the lister views after each and the reconcile order
        # cell-a, cell-a, cell-b.
        """Two relists in a row apply a survivor plus a newcomer, then a completely new world."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0"), make_pod("pod-1")])
        recorder.parent_keys.clear()

        source.emit(replace_of(make_pod("pod-1"), make_pod("pod-2")))
        await settle()
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-1", "pod-2"]

        source.emit(replace_of(make_pod("pod-3", cell="cell-b")))
        await settle()

        assert loop.get_by_parent("cell-a") == []
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-b")] == ["pod-3"]
        assert recorder.parent_keys == ["cell-a", "cell-a", "cell-b"]
        await loop.stop()

    async def test_stream_not_opening_with_replace_is_retried(self):
        # [FUNCTION DOC] A stream whose first event is an upsert leaves start pending; after the
        # 1 s retry delay the source is opened again and an empty replace completes start.
        """A stream whose first event is not ReplaceEvent cannot be trusted to hold the whole world."""
        source = FakeSource()
        recorder = Recorder()
        clock = FakeClock()
        loop = ReconcileLoop(source=source, reconcile=recorder, key_map=pod_cell, clock=clock, source_retry_delay=1.0)
        start_task = asyncio.create_task(loop.start())
        await settle()
        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0")))
        await settle()
        assert not start_task.done()

        await clock.elapse(1.0)
        await settle()
        assert source.open_count == 2
        source.emit(replace_of())
        await settle()
        await start_task
        await loop.stop()


class TestRelistReplace:
    # [CLASS DOC] Reopening the source after failures and the resulting replace.
    async def test_reopened_stream_deletes_objects_missing_from_the_new_list(self):
        # [FUNCTION DOC] A stream error, a reopen after 1 s, and a replace without pod-1:
        # pod-1's cell is emptied and both cells were reconciled.
        """A relist that drops a pod synthesizes its deletion (no ghost objects)."""
        pods = [make_pod("pod-0", cell="cell-a"), make_pod("pod-1", cell="cell-b")]
        loop, source, recorder, clock = await make_loop(initial=pods, source_retry_delay=1.0)
        recorder.parent_keys.clear()

        source.emit(StreamError(RuntimeError("connection reset")))
        await settle()
        await clock.elapse(1.0)
        await settle()
        assert source.open_count == 2

        source.emit(replace_of(pods[0]))
        await settle()
        assert loop.get_by_parent("cell-b") == []
        assert sorted(set(recorder.parent_keys)) == ["cell-a", "cell-b"]
        await loop.stop()

    async def test_reopened_stream_rewakes_surviving_objects(self):
        # [FUNCTION DOC] After a normal stream end and reopen, a replace holding the same pod
        # reconciles its cell once and keeps the pod in the lister view.
        """Surviving objects are reconciled again after a relist (at-least-once)."""
        pod = make_pod("pod-0", cell="cell-a")
        loop, source, recorder, clock = await make_loop(initial=[pod], source_retry_delay=1.0)
        recorder.parent_keys.clear()

        source.emit(StreamEnd())
        await settle()
        await clock.elapse(1.0)
        await settle()
        source.emit(replace_of(pod))
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        assert [p.metadata.name for p in loop.get_by_parent("cell-a")] == ["pod-0"]
        await loop.stop()

    async def test_stream_error_closes_the_old_stream(self):
        # [FUNCTION DOC] After a stream error and the retry delay, the source was closed once and
        # opened twice.
        """The failed stream is closed before a new one is opened."""
        loop, source, _, clock = await make_loop(initial=[], source_retry_delay=1.0)
        source.emit(StreamError(RuntimeError("boom")))
        await settle()
        await clock.elapse(1.0)
        await settle()
        assert source.closed_count == 1
        assert source.open_count == 2
        await loop.stop()

    async def test_relist_carries_objects_added_while_disconnected(self):
        # [FUNCTION DOC] After a reconnect, a replace with a new pod in cell-new reconciles
        # cell-new.
        """Objects created during the outage appear after the relist."""
        loop, source, recorder, clock = await make_loop(initial=[], source_retry_delay=1.0)
        source.emit(StreamEnd())
        await settle()
        await clock.elapse(1.0)
        await settle()
        source.emit(replace_of(make_pod("pod-new", cell="cell-new")))
        await settle()
        assert recorder.parent_keys == ["cell-new"]
        await loop.stop()

    async def test_cancelling_start_aborts_the_initial_sync(self):
        # [FUNCTION DOC] Cancels a pending start; asserts the source is closed once and stays so,
        # the task is cancelled, and a later emitted replace causes no reconcile.
        """A start() hung on the initial LIST is aborted by cancelling its task, which closes the source."""
        source = FakeSource()
        recorder = Recorder()
        loop = ReconcileLoop(source=source, reconcile=recorder, key_map=pod_cell, clock=FakeClock())
        start_task = asyncio.create_task(loop.start())
        await settle()
        assert not start_task.done()

        start_task.cancel()
        await asyncio.gather(start_task, return_exceptions=True)
        assert source.closed_count == 1
        await settle()

        assert start_task.cancelled()
        assert source.closed_count == 1
        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        assert recorder.parent_keys == []

    async def test_a_non_positive_source_retry_delay_is_rejected(self):
        # [FUNCTION DOC] A zero source retry delay fails an assertion at construction (duplicates
        # the cadence validation case).
        """A zero delay would turn the reopen loop into a hot loop, so it never reaches construction."""
        with pytest.raises(AssertionError):
            ReconcileLoop(source=FakeSource(), reconcile=Recorder(), clock=FakeClock(), source_retry_delay=0.0)
