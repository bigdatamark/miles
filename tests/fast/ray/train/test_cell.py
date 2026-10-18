# [MODULE DOC]
# Tests for the trainer cell (the unit of independent-dp training made of several workers) against
# dummy ray actors served by a fake worker manager. Covers construction, the state machine
# (uninitialized, alive with an independent dp info, errored) and its guards, killing workers and
# confirming death, init dispatch order (configure the rendezvous address, then init), health
# checking start, concurrent dispatch of worker calls, failure handling that kills the whole cell
# (or not, when asked), cooperative preparation in alive and healing modes, and a full
# kill-and-replace lifecycle. Cells are created by factories in the shared conftest.
# KNOWN ISSUE: the module wide asyncio marker also applies to the synchronous tests of this
# module, which are not coroutines.
import asyncio

import pytest
import ray
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import (
    RecordingHealthChecker,
    get_raw_actor_handles,
    make_alive_cell,
    make_cell,
    make_indep_dp_info,
)

from miles.ray.train import cell as cell_module
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

pytestmark = pytest.mark.asyncio


class TestInitialState:
    # [CLASS DOC] Tests the state of a freshly constructed cell.
    def test_starts_as_uninitialized_after_init(self):
        # [FUNCTION DOC] A new cell is allocated and uninitialized (its actors exist but were not
        # inited) and is not alive.
        """After __init__, cell is allocated (uninitialized) — actors created but not init'd."""
        cell = make_cell()

        assert cell.is_allocated
        assert not cell.is_alive
        assert cell.is_uninitialized

    def test_worker_handles_wrap_real_ray_actors(self):
        # [FUNCTION DOC] A 3-actor cell exposes 3 worker handles that are all worker handle
        # objects, and the raw handles beneath them are real ray actor handles.
        cell = make_cell(actor_count=3)

        handles = cell._get_worker_handles()
        assert len(handles) == 3
        assert all(isinstance(h, BaseWorkerHandle) for h in handles)
        assert all(isinstance(h, ray.actor.ActorHandle) for h in get_raw_actor_handles(cell))

    def test_a_cell_refuses_worker_infos_without_callable_handles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] When the ray provider's handle construction is replaced by one returning
        # no handles, building a 2-actor cell raises an assertion error saying it holds workers
        # that cannot be called.
        """Every described trainer rank must have a handle or the collective would silently omit it."""

        def _no_handles(_provider: RayWorkerProvider, _infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
            # [FUNCTION DOC] HELPER: Replacement handle builder that returns an empty mapping.
            return {}

        monkeypatch.setattr(RayWorkerProvider, "get_handles_of_worker_infos", _no_handles)

        with pytest.raises(AssertionError, match="holds workers that cannot be called"):
            make_cell(actor_count=2)


class TestKillWorkers:
    # [CLASS DOC] Tests killing a cell's workers and confirming their death.
    async def test_a_restarted_worker_is_already_gone_for_killing(self):
        # [FUNCTION DOC] Killing through a handle whose self kill raises a server-restarted error
        # completes without raising (the worker is treated as already gone).
        """A stale handle that reaches a new server must not make cell teardown fail."""

        class RestartedHandle:
            # [CLASS DOC] HELPER: Fake worker handle whose self kill always raises a
            # server-restarted error.
            async def kill_self(self) -> None:
                # [FUNCTION DOC] HELPER: Always raises the server-restarted error.
                raise ServerRestartedError

        await cell_module._kill_worker(RestartedHandle())

    async def test_killing_reaches_every_worker(self):
        # [FUNCTION DOC] After killing an alive cell's workers, every raw actor handle raises an
        # actor death error on a later call.
        """The dead workers must not linger in a cross-cell collective."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        handles = get_raw_actor_handles(cell)

        await cell._kill_workers_and_confirm_dead()

        for handle in handles:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())

    async def test_killing_does_not_involve_the_worker_manager(self):
        # [FUNCTION DOC] Killing an alive cell's workers records no stop-cells call on the fake manager.
        """The manager keeps reporting the cell alive so its errored status stays visible."""
        cell = make_alive_cell(0, alive_cell_indices=[0])

        await cell._kill_workers_and_confirm_dead()

        assert train_conftest.fake_worker_manager.stopped_cell_ids == []

    async def test_teardown_leaves_every_worker_handle_confirmed_dead(self):
        # [FUNCTION DOC] For a 2-actor cell, after kill-and-confirm each wrapped handle's wait-dead
        # (30 second timeout, bounded by an outer 35 second wait) completes and a call on the raw
        # actor raises an actor death error.
        """Teardown really kills its workers, so every handle is confirmed dead and rejects new calls."""
        cell = make_cell(actor_count=2)
        wrapped_handles = cell._get_worker_handles()

        await cell._kill_workers_and_confirm_dead()

        for wrapped in wrapped_handles:
            await asyncio.wait_for(wrapped.wait_dead(timeout=30.0), timeout=35.0)
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(wrapped._actor_handle.get_calls.remote())


class TestMarkAsAlive:
    # [CLASS DOC] Tests the uninitialized-to-alive state transition.
    def test_transitions_uninitialized_to_alive(self):
        # [FUNCTION DOC] Marking a fresh cell alive with an independent dp info makes it alive and
        # stores that info.
        cell = make_cell()
        info = make_indep_dp_info(alive_cell_indices=[0, 1, 2])

        cell._mark_as_alive(indep_dp_info=info)

        assert cell.is_alive
        assert cell.indep_dp_info == info

    def test_preserves_actor_handles(self):
        # [FUNCTION DOC] Marking a 3-actor cell alive leaves its raw actor handles unchanged.
        cell = make_cell(actor_count=3)
        handles_before = get_raw_actor_handles(cell)

        cell._mark_as_alive(indep_dp_info=make_indep_dp_info())

        assert get_raw_actor_handles(cell) == handles_before

    def test_rejects_from_alive(self):
        # [FUNCTION DOC] Marking an already alive cell alive raises an assertion error.
        cell = make_alive_cell(0, alive_cell_indices=[0])

        with pytest.raises(AssertionError):
            cell._mark_as_alive(indep_dp_info=make_indep_dp_info())


class TestUpdateIndepDPInfo:
    # [CLASS DOC] Tests replacing the independent dp info of an alive cell.
    def test_updates_stored_info(self):
        # [FUNCTION DOC] Updating an alive cell with a new info (alive cells 0 and 2, quorum 2)
        # replaces the stored info.
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        cell._update_indep_dp_info(new_info)

        assert cell.indep_dp_info == new_info

    def test_preserves_actor_handles(self):
        # [FUNCTION DOC] Updating the info of an alive cell leaves its raw actor handles unchanged.
        cell = make_alive_cell(0, alive_cell_indices=[0])
        handles = get_raw_actor_handles(cell)

        cell._update_indep_dp_info(make_indep_dp_info(quorum_id=5))

        assert get_raw_actor_handles(cell) == handles

    def test_rejects_from_uninitialized(self):
        # [FUNCTION DOC] Updating the info of a not-yet-alive cell raises an assertion error.
        cell = make_cell()

        with pytest.raises(AssertionError):
            cell._update_indep_dp_info(make_indep_dp_info())


class TestMarkAsErrored:
    # [CLASS DOC] Tests the transition to the errored state.
    def test_transitions_alive_to_errored(self):
        # [FUNCTION DOC] Marking an alive cell errored makes it errored, not alive, still
        # allocated, and keeps its independent dp info.
        cell = make_alive_cell(0, alive_cell_indices=[0])
        info = cell.indep_dp_info

        cell._mark_as_errored()

        assert cell.is_errored
        assert not cell.is_alive
        assert cell.is_allocated
        assert cell.indep_dp_info == info

    def test_errored_is_idempotent(self):
        # [FUNCTION DOC] Marking an errored cell errored again keeps it errored with no error.
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._mark_as_errored()

        cell._mark_as_errored()

        assert cell.is_errored

    def test_transitions_uninitialized_to_errored_without_info(self):
        # [FUNCTION DOC] A fresh (never alive) cell can be marked errored and then has no
        # independent dp info.
        """A cell whose init never completed can still be marked errored; its indep_dp_info is None."""
        cell = make_cell()

        cell._mark_as_errored()

        assert cell.is_errored
        assert cell.indep_dp_info is None


class TestErroredCellTeardown:
    # [CLASS DOC] Tests tearing down errored cells.
    async def test_kill_from_errored_reaches_the_workers(self):
        # [FUNCTION DOC] After an alive cell is marked errored and its workers are killed, every
        # raw actor raises an actor death error on a later call.
        """An errored cell is torn down by killing its own workers."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._mark_as_errored()
        assert cell.is_errored
        handles = get_raw_actor_handles(cell)

        await cell._kill_workers_and_confirm_dead()

        for handle in handles:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())

    async def test_the_replacement_cell_recovers_the_lifecycle(self):
        # [FUNCTION DOC] An alive cell is marked errored and its workers killed; the fake manager
        # then stops the cell, a replacement cell for the same index is built, and marked alive
        # with quorum 99. The replacement is alive and keeps that quorum.
        """Errored → kill → heal restarts the cell → reconcile builds a fresh one → alive."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._mark_as_errored()
        await cell._kill_workers_and_confirm_dead()

        train_conftest.fake_worker_manager._stop_cells([cell.cell_id])
        replacement = make_cell(cell.cell_index)
        replacement._mark_as_alive(indep_dp_info=make_indep_dp_info(quorum_id=99))

        assert replacement.is_alive
        assert replacement.indep_dp_info.quorum_id == 99


class TestAsyncInit:
    # [CLASS DOC] Tests the asynchronous init of a cell.
    async def test_dispatches_init_and_marks_alive(self):
        # [FUNCTION DOC] Initing a 2-actor cell with an info and a store address returns two
        # results, makes the cell alive with that info, and each actor's call log is exactly
        # configure-master-address then init: the first with master address 10.0.0.1 and port
        # 20000, the second with the info, the store address 10.0.0.9:1234 and no checkpoint
        # source rank.
        """Init configures the rendezvous address on every worker before dispatching init itself."""
        cell = make_cell(actor_count=2)
        info = make_indep_dp_info()

        results = await cell.init(indep_dp_info=info, indep_dp_store_addr="10.0.0.9:1234")

        assert len(results) == 2
        assert cell.is_alive
        assert cell.indep_dp_info == info

        for handle in get_raw_actor_handles(cell):
            calls = ray.get(handle.get_calls.remote())
            assert [call[0] for call in calls] == ["configure_master_addr_and_port", "init"]
            assert calls[0][2] == {"master_addr": "10.0.0.1", "master_port": 20000}
            kwargs = calls[1][2]
            assert kwargs["indep_dp_info"] == info
            assert kwargs["indep_dp_store_addr"] == "10.0.0.9:1234"
            assert kwargs["recv_ckpt_src_rank"] is None

    async def test_health_checking_starts_on_an_alive_cell_and_is_running_when_init_returns(self):
        # [FUNCTION DOC] With a recording checker observing whether the cell is alive, init starts
        # the checker exactly once, the cell was already alive at that moment, and the checker's
        # background task has begun running by the time init returns.
        """A checker that is only scheduled never probes until the next await, which may be a whole train step away."""
        checker = RecordingHealthChecker()
        cell = make_cell(actor_count=1, health_checker=checker)
        checker.observe_alive = lambda: cell.is_alive

        await cell.init(indep_dp_info=make_indep_dp_info(), indep_dp_store_addr=None)

        assert checker.start_count == 1
        assert checker.alive_when_started is True
        assert checker.task_started


class _EntryBarrier:
    # [CLASS DOC] HELPER: Asynchronous barrier for a fixed number of parties. Each party enters
    # once and waits until all have entered (at most 10 seconds), proving that the parties ran
    # concurrently.
    def __init__(self, party_size: int) -> None:
        # [FUNCTION DOC] HELPER: Stores the number of parties still to enter and creates the event
        # that is set when all have entered.
        self._remaining = party_size
        self._all_entered = asyncio.Event()

    async def enter(self) -> None:
        # [FUNCTION DOC] HELPER: Decrements the remaining count; the last party to enter sets the
        # event. Then waits for the event with a 10 second timeout (raises a timeout error if not
        # all parties enter in time).
        self._remaining -= 1
        if self._remaining == 0:
            self._all_entered.set()
        await asyncio.wait_for(self._all_entered.wait(), timeout=10.0)


class _BarrierWorkerHandle:
    # [CLASS DOC] HELPER: Fake worker handle whose train call passes through a shared entry barrier.
    def __init__(self, barrier: _EntryBarrier) -> None:
        # [FUNCTION DOC] HELPER: Stores the shared barrier.
        self._barrier = barrier

    async def train(self, **_kwargs) -> str:
        # [FUNCTION DOC] HELPER: Enters the barrier (waiting for the other handles) and then
        # returns the string done. Keyword arguments are ignored.
        await self._barrier.enter()
        return "done"


class TestExecuteConcurrency:
    # [CLASS DOC] Tests that a cell dispatches a call to all its workers concurrently.
    async def test_every_worker_rpc_is_entered_before_any_of_them_returns(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With the cell's worker handles replaced by three barrier handles (barrier
        # of three parties), executing the train call returns three done results; this only
        # succeeds if all three calls were started before any finished, otherwise the barrier
        # times out.
        """Worker methods enter a collective, so dispatching them one at a time deadlocks the cell."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        barrier = _EntryBarrier(3)
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: [_BarrierWorkerHandle(barrier) for _ in range(3)])

        results = await cell.execute("train", rollout_id=0)

        assert results == ["done", "done", "done"]


class TestPartialWorkerFailure:
    # [CLASS DOC] Tests how a cell reacts when some worker call fails.
    async def test_one_failing_rank_errors_the_cell_and_kills_every_rank(self):
        # [FUNCTION DOC] In an alive 3-actor cell where actor 1 is set to fail train, executing
        # train raises the injected runtime error; afterwards the cell is errored and every raw
        # actor raises an actor death error on a later call.
        """A surviving rank of a broken cell would sit in the collective and stall the cells that are still healthy."""
        cell = make_cell(actor_count=3)
        cell._mark_as_alive(indep_dp_info=make_indep_dp_info())
        handles = get_raw_actor_handles(cell)
        ray.get(handles[1].set_fail_methods.remote(["train"]))

        with pytest.raises(RuntimeError, match="Injected failure"):
            await cell.execute("train", rollout_id=0)

        assert cell.is_errored
        for handle in handles:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())

    async def test_a_failure_that_must_not_kill_leaves_the_cell_alive_and_reachable(self):
        # [FUNCTION DOC] Executing train with the kill-on-failure option off on a cell whose
        # actor fails train raises the injected error but leaves the cell alive and not errored,
        # and every actor still answers a call log query.
        """The heartbeat probe rides on execute; recycling a cell because one probe raised would kill live training."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        handles = get_raw_actor_handles(cell)
        ray.get(handles[0].set_fail_methods.remote(["train"]))

        with pytest.raises(RuntimeError, match="Injected failure"):
            await cell.execute("train", kill_on_failure=False, rollout_id=0)

        assert cell.is_alive
        assert not cell.is_errored
        for handle in handles:
            assert ray.get(handle.get_calls.remote())


class TestAsyncInitFailure:
    # [CLASS DOC] Tests a failing cell init.
    async def test_init_failure_leaves_cell_not_alive(self):
        # [FUNCTION DOC] With the single actor set to fail init, initing the cell raises the
        # injected runtime error, the cell is not alive, and the actor has been killed (later calls
        # raise actor death errors).
        """A failed remote init marks the cell errored and tears it down; it is never reported alive."""
        cell = make_cell(actor_count=1)
        for handle in get_raw_actor_handles(cell):
            ray.get(handle.set_fail_methods.remote(["init"]))

        with pytest.raises(RuntimeError, match="Injected failure"):
            await cell.init(indep_dp_info=make_indep_dp_info(), indep_dp_store_addr=None)

        assert not cell.is_alive
        for handle in get_raw_actor_handles(cell):
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())


class TestPrepareIndepDPModeAlive:
    # [CLASS DOC] Tests cooperative preparation of an already alive cell for a new quorum.
    async def test_reconfigure_and_update_info(self):
        # [FUNCTION DOC] Preparing an alive cell with a new info (alive cells 0 and 2, quorum 2),
        # a store address and no checkpoint destinations updates the stored info, keeps the cell
        # alive, and sends each actor exactly one independent dp reconfiguration carrying the new
        # info and store address.
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        await cell.prepare_indep_dp_mode_alive(
            indep_dp_info=new_info, indep_dp_store_addr="10.0.0.9:1234", send_ckpt_dst_ranks=[]
        )

        assert cell.indep_dp_info == new_info
        assert cell.is_alive

        for handle in get_raw_actor_handles(cell):
            calls = ray.get(handle.get_calls.remote())
            reconfig_calls = [c for c in calls if c[0] == "reconfigure_indep_dp"]
            assert len(reconfig_calls) == 1
            assert reconfig_calls[0][2]["indep_dp_info"] == new_info
            assert reconfig_calls[0][2]["indep_dp_store_addr"] == "10.0.0.9:1234"

    async def test_sends_ckpt_to_correct_dst_ranks(self):
        # [FUNCTION DOC] Preparing with checkpoint destination ranks 1 and 2 makes the first actor
        # record exactly two checkpoint send calls, in order, with destination ranks 1 and 2.
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 1, 2], quorum_id=2)
        await cell.prepare_indep_dp_mode_alive(
            indep_dp_info=new_info, indep_dp_store_addr=None, send_ckpt_dst_ranks=[1, 2]
        )

        handle = get_raw_actor_handles(cell)[0]
        calls = ray.get(handle.get_calls.remote())
        send_calls = [c for c in calls if c[0] == "send_ckpt"]
        assert len(send_calls) == 2
        assert send_calls[0][2]["dst_rank"] == 1
        assert send_calls[1][2]["dst_rank"] == 2


class TestPrepareIndepDPModeHealing:
    # [CLASS DOC] Tests cooperative preparation of a cell that is being healed.
    async def test_healing_inits_and_marks_alive(self):
        # [FUNCTION DOC] Preparing a fresh single-actor cell in healing mode (no store address, no
        # checkpoint source) makes the cell alive with the given info and the actor records an init call.
        cell = make_cell(actor_count=1)
        info = make_indep_dp_info()

        await cell.prepare_indep_dp_mode_healing(indep_dp_info=info, indep_dp_store_addr=None, recv_ckpt_src_rank=None)

        assert cell.is_alive
        assert cell.indep_dp_info == info

        handle = get_raw_actor_handles(cell)[0]
        calls = ray.get(handle.get_calls.remote())
        assert any(c[0] == "init" for c in calls)


class TestStatePredicates:
    # [CLASS DOC] Tests the state predicates (allocated, uninitialized, alive, errored) in each state.
    def test_uninitialized(self):
        # [FUNCTION DOC] A fresh cell is allocated and uninitialized, and neither alive nor errored.
        cell = make_cell()

        assert cell.is_allocated
        assert cell.is_uninitialized
        assert not cell.is_alive
        assert not cell.is_errored

    def test_alive(self):
        # [FUNCTION DOC] An alive cell is allocated and alive, and neither uninitialized nor errored.
        cell = make_alive_cell(0, alive_cell_indices=[0])

        assert cell.is_allocated
        assert not cell.is_uninitialized
        assert cell.is_alive
        assert not cell.is_errored

    def test_errored(self):
        # [FUNCTION DOC] An errored cell is allocated and errored, and neither uninitialized nor alive.
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._mark_as_errored()

        assert cell.is_allocated
        assert not cell.is_uninitialized
        assert not cell.is_alive
        assert cell.is_errored


class TestFullLifecycle:
    # [CLASS DOC] Tests a whole cell lifecycle through the state machine.
    async def test_full_kill_and_replacement_cycle(self):
        # [FUNCTION DOC] Walks a 2-actor cell through: created (uninitialized), marked alive with
        # quorum 1, workers killed directly, the fake manager stopping the cell and a replacement
        # cell being built (uninitialized again), and the replacement being marked alive with a new
        # info of quorum 2. The comments inside the body label these five steps.
        """Full lifecycle: attach → alive → kill → heal restarts → reconcile replaces the object → alive again."""
        # Step 1: Create (attaches to the manager's workers)
        cell = make_cell(actor_count=2)
        assert cell.is_uninitialized and not cell.is_alive

        # Step 2: Alive
        info_v1 = make_indep_dp_info(alive_cell_indices=[0, 1, 2], quorum_id=1)
        cell._mark_as_alive(indep_dp_info=info_v1)
        assert cell.is_alive

        # Step 3: Kill the workers directly
        await cell._kill_workers_and_confirm_dead()

        # Step 4: The ft controller heals it and reconcile builds a fresh object on the new workers
        train_conftest.fake_worker_manager._stop_cells([cell.cell_id])
        cell = make_cell(actor_count=2)
        assert cell.is_uninitialized and not cell.is_alive

        # Step 5: Alive again with new config
        info_v2 = make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        cell._mark_as_alive(indep_dp_info=info_v2)
        assert cell.is_alive
        assert cell.indep_dp_info.quorum_id == 2
