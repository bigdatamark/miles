# [MODULE DOC] Scenario tests for the context lock module, built from a controller-like object
# (owns the lock, exposes locked entry points and a cross-call update-weights window) and a
# server-like collaborator (handed the controller's lock, all methods require it). They check that
# accidental concurrent callers are serialized instead of interleaving, that mistakes around the
# window (double end, end without start, forgotten end, failing start) are caught, that bypassing
# the owner is rejected, and that typical mistakes when extending a guarded class are detected.
# Journal objects record events so ordering can be asserted. CPU only, async tests.
import asyncio
import logging

import pytest

from miles.utils import context_lock
from miles.utils.context_lock import (
    ContextLock,
    acquires_lock,
    enforce_lock_discipline,
    lock_exempt,
    releases_lock,
    requires_lock,
    with_lock,
)


class _Journal:
    # [CLASS DOC] HELPER: Event recorder. Keeps an ordered list of event strings plus a count of
    # currently open sections and its maximum, so tests can tell interleaving from serialization.
    """Records who ran when, so a test can tell interleaving apart from serialization."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with no events and zero open sections (current and maximum).
        self.events: list[str] = []
        self.max_open_sections = 0
        self._open_sections = 0

    def record(self, event: str) -> None:
        # [FUNCTION DOC] HELPER: Appends the event text to the event list.
        self.events.append(event)

    def enter_section(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Increments the open section count, raises the maximum if exceeded
        # and records "<name>-enter".
        self._open_sections += 1
        self.max_open_sections = max(self.max_open_sections, self._open_sections)
        self.record(f"{name}-enter")

    def leave_section(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Decrements the open section count and records "<name>-leave".
        self._open_sections -= 1
        self.record(f"{name}-leave")

    def events_between(self, start: str, end: str) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns the events strictly between the first occurrence of the
        # start event and the first occurrence of the end event. Raises a value error if either is
        # missing.
        return self.events[self.events.index(start) + 1 : self.events.index(end)]


@enforce_lock_discipline
class _Server:
    # [CLASS DOC] HELPER: Collaborator modelled on a rollout server. Fully guarded: every method
    # requires the owner's lock. Receives the lock object from its owner and holds a mapping from
    # cell id to a fake engine name ("engine-<cell id>"), plus the shared journal.
    """Shaped like RolloutServer: fully guarded, and handed its owner's lock."""

    @lock_exempt
    def __init__(self, journal: _Journal, lock: ContextLock, cell_ids: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Exempt constructor: stores the journal and the given lock and
        # builds the cell mapping from the cell ids.
        self.journal = journal
        self.context_lock = lock
        self.cells = {cell_id: f"engine-{cell_id}" for cell_id in cell_ids}

    @requires_lock
    async def offload(self) -> None:
        # [FUNCTION DOC] HELPER: Requires the lock; touches every cell with the action "offload".
        await self._touch_every_cell("offload")

    @requires_lock
    async def onload(self) -> None:
        # [FUNCTION DOC] HELPER: Requires the lock; touches every cell with the action "onload".
        await self._touch_every_cell("onload")

    @requires_lock
    async def mark_weights_ready(self) -> None:
        # [FUNCTION DOC] HELPER: Requires the lock; touches every cell with the action
        # "mark-weights-ready".
        await self._touch_every_cell("mark-weights-ready")

    @requires_lock
    async def remove_cell(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Requires the lock; records "remove:<cell id>", yields once, then
        # removes the cell from the mapping if present.
        self.journal.record(f"remove:{cell_id}")
        await asyncio.sleep(0)
        self.cells.pop(cell_id, None)

    @property
    @requires_lock
    def engines(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Requires the lock; returns a snapshot list of the engine names.
        return list(self.cells.values())

    @requires_lock
    async def _touch_every_cell(self, action: str) -> None:
        # [FUNCTION DOC] HELPER: Requires the lock; for each cell (iterating over a copy of the cell
        # ids) records "<action>:<cell id>" and yields once, so concurrent callers could interleave
        # if the lock were not held.
        for cell_id in list(self.cells):
            self.journal.record(f"{action}:{cell_id}")
            await asyncio.sleep(0)


@enforce_lock_discipline
class _Controller:
    # [CLASS DOC] HELPER: Owner modelled on an inference controller. Creates the lock named
    # "InferenceController" and a server collaborator that shares it. Its public methods hold the
    # lock for their whole call; the weight update is a cross-call window opened by one method and
    # closed by another.
    """Shaped like InferenceController: owns the lock and hands it to its server."""

    @lock_exempt
    def __init__(self, journal: _Journal, cell_ids: list[str] | None = None) -> None:
        # [FUNCTION DOC] HELPER: Exempt constructor: stores the journal, creates the lock and builds
        # the server with the shared lock and the cell ids (default "cell-0" and "cell-1").
        self.journal = journal
        self.context_lock = ContextLock("InferenceController")
        self.server = _Server(journal, self.context_lock, cell_ids or ["cell-0", "cell-1"])

    @with_lock
    async def prepare_rollout(self) -> None:
        # [FUNCTION DOC] HELPER: Under the lock: onloads the server.
        await self.server.onload()

    @with_lock
    async def offload(self) -> None:
        # [FUNCTION DOC] HELPER: Under the lock: offloads the server.
        await self.server.offload()

    @with_lock
    async def onload_weights(self) -> None:
        # [FUNCTION DOC] HELPER: Under the lock: delegates to the private lock-requiring onload
        # helper (the working shape for sibling entry points).
        await self._onload()

    @with_lock
    async def onload_kv(self) -> None:
        # [FUNCTION DOC] HELPER: Under the lock: delegates to the private onload helper.
        await self._onload()

    @with_lock
    async def reconcile(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Under the lock: removes the named cell from the server.
        await self.server.remove_cell(cell_id)

    @acquires_lock
    async def start_update_weights(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Opens the weight-update window (lock stays locked afterwards);
        # returns the server's engine snapshot taken while the lock is held.
        return self.server.engines

    @releases_lock
    async def end_update_weights(self, engines: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Closes the window: marks the server's weights ready, then the lock
        # is released. The engines argument is accepted but not used in the body.
        await self.server.mark_weights_ready()

    @requires_lock
    async def _onload(self) -> None:
        # [FUNCTION DOC] HELPER: Requires the lock; onloads the server.
        await self.server.onload()


async def _update_weights_flow(controller: _Controller, label: str) -> None:
    # [FUNCTION DOC] HELPER: Models the trainer: opens the window to snapshot engines, enters a
    # journal section named by the label, yields twice (the broadcast), leaves the section, and
    # closes the window with the snapshot.
    """Shaped like the trainer: snapshot the engines, broadcast, then mark them ready."""
    engines = await controller.start_update_weights()
    controller.journal.enter_section(label)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    controller.journal.leave_section(label)
    await controller.end_update_weights(engines)


class TestCallersRacingByAccident:
    # [CLASS DOC] Tests that concurrently fired controller calls serialize on the lock.
    @pytest.mark.asyncio
    async def test_two_overlapping_update_weights_flows_never_interleave(self):
        # [FUNCTION DOC] Gathers two update flows; asserts at most one journal section is open at a
        # time and that no event occurs between each flow's enter and leave.
        """A double-fired weight update (a retry, two trainers) must serialize, not interleave."""
        journal = _Journal()
        controller = _Controller(journal)

        await asyncio.gather(_update_weights_flow(controller, "first"), _update_weights_flow(controller, "second"))

        assert journal.max_open_sections == 1
        assert journal.events_between("first-enter", "first-leave") == []
        assert journal.events_between("second-enter", "second-leave") == []

    @pytest.mark.asyncio
    async def test_a_reconcile_cannot_land_inside_an_update_weights_window(self):
        # [FUNCTION DOC] Gathers an update flow with a reconcile removing "cell-0"; asserts nothing is
        # journaled between the update's enter and leave, and that the removal did happen.
        """A cell vanishing mid-update must not be torn down under the trainer's snapshot."""
        journal = _Journal()
        controller = _Controller(journal)

        await asyncio.gather(_update_weights_flow(controller, "update"), controller.reconcile("cell-0"))

        assert journal.events_between("update-enter", "update-leave") == []
        assert "remove:cell-0" in journal.events

    @pytest.mark.asyncio
    async def test_an_offload_cannot_land_inside_an_update_weights_window(self):
        # [FUNCTION DOC] Gathers an update flow with an offload; asserts no offload event is journaled
        # between the update's enter and leave.
        """Offloading engines whose weights are being written would corrupt the update."""
        journal = _Journal()
        controller = _Controller(journal)

        await asyncio.gather(_update_weights_flow(controller, "update"), controller.offload())

        assert not [event for event in journal.events_between("update-enter", "update-leave") if "offload" in event]

    @pytest.mark.asyncio
    async def test_offload_and_onload_fired_together_do_not_interleave_per_cell(self):
        # [FUNCTION DOC] Gathers an offload and a prepare-rollout (onload) on two cells; the journal
        # equals either all offloads then all onloads, or all onloads then all offloads (whole-call
        # serialization, either order).
        """A forgotten await upstream can fire both at once; each engine still sees them in order."""
        journal = _Journal()
        controller = _Controller(journal, cell_ids=["cell-0", "cell-1"])

        await asyncio.gather(controller.offload(), controller.prepare_rollout())

        assert journal.events in (
            ["offload:cell-0", "offload:cell-1", "onload:cell-0", "onload:cell-1"],
            ["onload:cell-0", "onload:cell-1", "offload:cell-0", "offload:cell-1"],
        )

    @pytest.mark.asyncio
    async def test_a_whole_burst_of_lifecycle_calls_is_fully_serialized(self):
        # [FUNCTION DOC] Gathers two update flows, an offload, an onload of weights, a prepare-rollout
        # and a reconcile of "cell-1" all at once; asserts a single open journal section at most and
        # that the lock is free afterwards.
        """Whatever mix of calls arrives at once, exactly one critical section runs at a time."""
        journal = _Journal()
        controller = _Controller(journal)

        await asyncio.gather(
            _update_weights_flow(controller, "update-a"),
            _update_weights_flow(controller, "update-b"),
            controller.offload(),
            controller.onload_weights(),
            controller.prepare_rollout(),
            controller.reconcile("cell-1"),
        )

        assert journal.max_open_sections == 1
        assert not controller.context_lock.locked

    @pytest.mark.asyncio
    async def test_the_engine_snapshot_cannot_change_while_the_window_is_open(self):
        # [FUNCTION DOC] Opens the window and gets the snapshot; starts a reconcile removing "cell-0"
        # as a task and yields five times. Asserts both cells are still present and the snapshot is
        # the two engine names. After closing the window and awaiting the reconcile, only "cell-1"
        # remains.
        """The whole point of the window: the engine list handed to the trainer stays valid."""
        journal = _Journal()
        controller = _Controller(journal)

        engines = await controller.start_update_weights()
        reconcile = asyncio.create_task(controller.reconcile("cell-0"))
        for _ in range(5):
            await asyncio.sleep(0)
        assert set(controller.server.cells) == {"cell-0", "cell-1"}
        assert engines == ["engine-cell-0", "engine-cell-1"]

        await controller.end_update_weights(engines)
        await reconcile
        assert set(controller.server.cells) == {"cell-1"}


class TestMistakesAroundTheUpdateWeightsWindow:
    # [CLASS DOC] Tests of misuse of the open/close window: double end, end without start, a
    # forgotten end (blocks later callers, and is logged), and a start that fails.
    @pytest.mark.asyncio
    async def test_ending_the_window_twice_is_rejected(self):
        # [FUNCTION DOC] After a start and a first end, a second end raises an assertion error
        # matching "was not detached".
        """A retry loop calling end twice must not release a window it no longer owns."""
        controller = _Controller(_Journal())
        engines = await controller.start_update_weights()
        await controller.end_update_weights(engines)

        with pytest.raises(AssertionError, match="was not detached"):
            await controller.end_update_weights(engines)

    @pytest.mark.asyncio
    async def test_ending_a_window_that_was_never_started_is_rejected(self):
        # [FUNCTION DOC] An end call without any start raises an assertion error matching "was not
        # detached".
        """An early return that skips start must not let end unlock the controller."""
        controller = _Controller(_Journal())

        with pytest.raises(AssertionError, match="was not detached"):
            await controller.end_update_weights([])

    @pytest.mark.asyncio
    async def test_a_forgotten_end_blocks_the_next_caller_instead_of_running_unguarded(self):
        # [FUNCTION DOC] After a start with no end, an offload call waited on with a 0.05 s timeout
        # raises a timeout error (it is blocked by the open window).
        """Losing the end call stalls the next rollout rather than letting it run unprotected."""
        controller = _Controller(_Journal())
        await controller.start_update_weights()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(controller.offload(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_a_stalled_window_is_reported_in_the_log(self, monkeypatch, caplog):
        # [FUNCTION DOC] Sets the reminder interval to 0.01 s, opens a window without closing it,
        # starts an offload task and observes for 0.05 s at info level, then cancels the blocked task
        # (asserting cancellation). Asserts at least one logged message mentions "Still waiting for
        # lock 'InferenceController'".
        """A trainer that died mid-update leaves the lock held, so the waiter must say so."""
        monkeypatch.setattr(context_lock, "WAIT_LOG_INTERVAL_SECONDS", 0.01)
        controller = _Controller(_Journal())
        await controller.start_update_weights()

        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            blocked = asyncio.create_task(controller.offload())
            await asyncio.sleep(0.05)
        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked

        assert any("Still waiting for lock 'InferenceController'" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_a_start_that_raises_does_not_wedge_the_controller(self):
        # [FUNCTION DOC] A controller subclass whose window start raises "engines never became ready":
        # the error propagates, the lock is left free, and a later offload call succeeds.
        """When taking the snapshot fails, the next rollout must still be able to proceed."""

        @enforce_lock_discipline
        class _FailingController(_Controller):
            # [CLASS DOC] HELPER: Enforced controller subclass with a failing window start.
            @acquires_lock
            async def start_update_weights(self) -> list[str]:
                # [FUNCTION DOC] HELPER: Always raises a runtime error "engines never became ready".
                raise RuntimeError("engines never became ready")

        controller = _FailingController(_Journal())
        with pytest.raises(RuntimeError, match="engines never became ready"):
            await controller.start_update_weights()

        assert not controller.context_lock.locked
        await controller.offload()


class TestBypassingTheOwner:
    # [CLASS DOC] Tests that calling the collaborator without the owner's lock is rejected.
    @pytest.mark.asyncio
    async def test_driving_the_server_directly_is_rejected(self):
        # [FUNCTION DOC] Calling the server's offload directly raises an assertion error matching
        # "must be called with".
        """New code reaching into controller.server instead of going through the controller."""
        controller = _Controller(_Journal())

        with pytest.raises(AssertionError, match="must be called with"):
            await controller.server.offload()

    def test_reading_the_engine_snapshot_directly_is_rejected(self):
        # [FUNCTION DOC] Reading the server's engines property without the lock raises an assertion
        # error matching "must be called with".
        """A snapshot read outside the lock may already be stale by the time it is used."""
        controller = _Controller(_Journal())

        with pytest.raises(AssertionError, match="must be called with"):
            _ = controller.server.engines

    @pytest.mark.asyncio
    async def test_holding_some_other_lock_does_not_authorize_the_server(self):
        # [FUNCTION DOC] Holding an unrelated lock named "SomeOtherController" does not satisfy the
        # server's requirement: still "must be called with".
        """Grabbing a nearby lock instead of the owning controller's is still a bypass."""
        controller = _Controller(_Journal())

        async with ContextLock("SomeOtherController"):
            with pytest.raises(AssertionError, match="must be called with"):
                await controller.server.offload()

    @pytest.mark.asyncio
    async def test_a_server_built_with_its_own_fresh_lock_is_rejected(self):
        # [FUNCTION DOC] Replaces the controller's server with one built on a new lock object of the
        # same name; an offload through the controller (which holds its own lock) fails with "must be
        # called with" because the identity differs.
        """The classic wiring mistake: constructing a lock instead of accepting the owner's."""
        journal = _Journal()
        controller = _Controller(journal)
        controller.server = _Server(journal, ContextLock("InferenceController"), ["cell-0"])

        with pytest.raises(AssertionError, match="must be called with"):
            await controller.offload()


class TestMistakesWhenExtendingAGuardedClass:
    # [CLASS DOC] Tests of typical errors when subclassing the controller: missing discipline
    # decorator, missing class decorator, delegating to a locked sibling, and the corrected shape.
    def test_adding_a_method_and_forgetting_the_decorator_is_rejected_at_class_creation(self):
        # [FUNCTION DOC] Subclassing with an undecorated new method fails at class creation with an
        # assertion error matching "drain_engines must be decorated".
        """The common mistake: a new method whose author never thought about the lock."""
        with pytest.raises(AssertionError, match="drain_engines must be decorated"):

            @enforce_lock_discipline
            class _Extended(_Controller):
                # [CLASS DOC] HELPER: Invalid subclass with an undecorated method.
                async def drain_engines(self) -> None:
                    # [FUNCTION DOC] HELPER: Undecorated no-op async method.
                    pass

    @pytest.mark.asyncio
    async def test_decorating_the_method_but_forgetting_the_class_is_rejected_at_call_time(self):
        # [FUNCTION DOC] A subclass with a locked method but no class decorator: calling it raises an
        # assertion error matching "not decorated with @enforce_lock_discipline".
        """The subtler mistake: the method states its discipline but nothing ever checks the class."""

        class _Extended(_Controller):
            # [CLASS DOC] HELPER: Unenforced subclass with a locked method.
            @with_lock
            async def drain_engines(self) -> None:
                # [FUNCTION DOC] HELPER: Locked no-op async method.
                pass

        with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
            await _Extended(_Journal()).drain_engines()

    @pytest.mark.asyncio
    async def test_a_locked_method_calling_a_sibling_locked_method_is_rejected(self):
        # [FUNCTION DOC] A subclass locked method that awaits another locked method fails with an
        # assertion error matching "already held" when called.
        """The reentrancy mistake: delegating to a public sibling instead of a private helper."""

        @enforce_lock_discipline
        class _Extended(_Controller):
            # [CLASS DOC] HELPER: Enforced subclass with a locked method that calls a locked sibling.
            @with_lock
            async def onload_everything(self) -> None:
                # [FUNCTION DOC] HELPER: Under the lock, awaits the locked onload-weights method
                # (reentrancy mistake).
                await self.onload_weights()

        with pytest.raises(AssertionError, match="already held"):
            await _Extended(_Journal()).onload_everything()

    @pytest.mark.asyncio
    async def test_delegating_to_a_private_helper_instead_is_the_shape_that_works(self):
        # [FUNCTION DOC] Calling two sibling entry points that share the private helper works, and the
        # lock is free afterwards.
        """Regression for that fix: sibling entry points share one lock-requiring private helper."""
        controller = _Controller(_Journal())

        await controller.onload_weights()
        await controller.onload_kv()

        assert not controller.context_lock.locked

    @pytest.mark.asyncio
    async def test_a_guarded_helper_calling_another_guarded_helper_is_fine(self):
        # [FUNCTION DOC] Inside async-with on the controller's lock, the private onload helper and the
        # server's offload (both lock-requiring) run without errors.
        """requires_lock nests freely; only re-acquiring the lock is the error."""
        controller = _Controller(_Journal())

        async with controller.context_lock:
            await controller._onload()
            await controller.server.offload()
