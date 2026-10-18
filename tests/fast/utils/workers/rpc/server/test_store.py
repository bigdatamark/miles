# [MODULE DOC] Unit tests for the rpc server call store, which tracks calls by id: begin registers a
# pending call (duplicates are rejected unless purged), finish records one outcome, wait long-polls
# for the outcome and marks the call as retrieved, and expired records are purged lazily whenever a
# new call begins. Two time to live values apply to finished records: a retrieved ttl counted from
# the first retrieval, and a finished ttl counted from finishing for never-retrieved ones; unfinished
# records are never purged. Time is controlled by replacing the store module's monotonic clock with
# a settable value. Also tests default ttl values and the sorted in-flight call id list.
import asyncio

import pytest

from miles.utils.workers.rpc.common.protocol import CallStatusResponse
from miles.utils.workers.rpc.server import store as store_module
from miles.utils.workers.rpc.server.store import FINISHED_TTL_SECONDS, CallStore, DuplicateCallError


def _make_store(*, ttl: float = 300.0, finished_ttl: float = FINISHED_TTL_SECONDS) -> CallStore:
    # [FUNCTION DOC] HELPER: Creates a call store with the given retrieved ttl (default 300 seconds)
    # and finished ttl (default the module constant).
    return CallStore(retrieved_ttl_seconds=ttl, finished_ttl_seconds=finished_ttl)


async def _retrieved_call(
    *, ttl: float, outcome: CallStatusResponse, finished_ttl: float = FINISHED_TTL_SECONDS
) -> CallStore:
    # [FUNCTION DOC] HELPER: Builds a store with the given ttls, begins call c1, finishes it with the
    # given outcome, and waits on it once (0.01 second timeout) so it counts as retrieved at the
    # current clock time. Returns the store.
    store = _make_store(ttl=ttl, finished_ttl=finished_ttl)
    store.begin(call_id="c1")
    store.finish(call_id="c1", outcome=outcome)
    await store.wait(call_id="c1", timeout=0.01)
    return store


class TestBegin:
    # [CLASS DOC] Tests registration of calls and duplicate detection.
    async def test_first_begin_registers_call(self) -> None:
        # [FUNCTION DOC] After begin, the store contains the call id.
        """The first submission registers its call id."""
        store = _make_store()

        store.begin(call_id="c1")

        assert store.contains("c1")

    async def test_duplicate_pending_call_rejected(self) -> None:
        # [FUNCTION DOC] A second begin of a pending id raises a duplicate call error mentioning
        # "already submitted".
        """Reusing a pending call id fails loudly."""
        store = _make_store()
        store.begin(call_id="c1")

        with pytest.raises(DuplicateCallError, match="already submitted"):
            store.begin(call_id="c1")

    async def test_duplicate_finished_call_rejected(self) -> None:
        # [FUNCTION DOC] A begin of an id whose call finished also raises the duplicate call error.
        """Reusing a finished call id fails loudly."""
        store = _make_store()
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        with pytest.raises(DuplicateCallError, match="already submitted"):
            store.begin(call_id="c1")

    async def test_expired_call_id_can_be_reused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With a fake clock at 10 and retrieved ttl 5, a retrieved call is expired at
        # 15.01. Beginning the same id again succeeds (purge happens before the duplicate check),
        # finishing it with a new outcome works, and a wait returns the new outcome.
        """An expired call id is purged before duplicate detection, so it is accepted again as a fresh call."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = await _retrieved_call(
            ttl=5.0,
            outcome=CallStatusResponse(status="success", result=1),
        )

        now[0] = 15.01
        store.begin(call_id="c1")
        reused_outcome = CallStatusResponse(status="success", result=2)
        store.finish(call_id="c1", outcome=reused_outcome)

        assert store.contains("c1")
        assert await store.wait(call_id="c1", timeout=0.01) == reused_outcome


class TestFinish:
    # [CLASS DOC] Tests recording of outcomes.
    async def test_finish_unknown_call_raises_key_error(self) -> None:
        # [FUNCTION DOC] Finishing an id never begun raises a key error and does not create a record.
        """Finishing a call id that was never begun raises KeyError instead of creating a record."""
        store = _make_store()

        with pytest.raises(KeyError):
            store.finish(call_id="missing", outcome=CallStatusResponse(status="success", result=1))

        assert not store.contains("missing")

    async def test_double_finish_rejected_and_outcome_preserved(self) -> None:
        # [FUNCTION DOC] A second finish raises a runtime error mentioning "finished twice"; a later
        # wait still returns the first outcome.
        """Finishing a call twice raises and preserves the first outcome."""
        store = _make_store()
        expected = CallStatusResponse(status="success", result=1)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=expected)

        with pytest.raises(RuntimeError, match="finished twice"):
            store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=2))

        assert await store.wait(call_id="c1", timeout=0.01) == expected


class TestWait:
    # [CLASS DOC] Tests the long-poll wait for an outcome.
    async def test_wait_returns_finished_outcome(self) -> None:
        # [FUNCTION DOC] Waiting on a finished call returns its outcome.
        """Waiting on a finished call returns its outcome immediately."""
        store = _make_store()
        expected = CallStatusResponse(status="success", result=42)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=expected)

        outcome = await store.wait(call_id="c1", timeout=0.01)

        assert outcome == expected

    async def test_wait_times_out_pending(self) -> None:
        # [FUNCTION DOC] Waiting on a pending call for 0.01 seconds returns nothing.
        """Waiting on a pending call returns None after the poll timeout."""
        store = _make_store()
        store.begin(call_id="c1")

        assert await store.wait(call_id="c1", timeout=0.01) is None

    async def test_wait_unblocked_by_finish(self) -> None:
        # [FUNCTION DOC] A background task finishes the call (after one event loop turn) while the test
        # waits with a 1 second timeout; the wait returns the failed outcome that was recorded.
        """A concurrent finish unblocks an in-flight wait."""
        store = _make_store()
        expected = CallStatusResponse(status="failed", error="boom")
        store.begin(call_id="c1")

        async def finisher() -> None:
            # [FUNCTION DOC] HELPER: Yields once to the event loop and then finishes call c1 with the
            # expected outcome.
            await asyncio.sleep(0)
            store.finish(call_id="c1", outcome=expected)

        task = asyncio.create_task(finisher())
        outcome = await store.wait(call_id="c1", timeout=1.0)
        await task

        assert outcome == expected

    async def test_wait_unknown_call_raises(self) -> None:
        # [FUNCTION DOC] Waiting on an unknown id raises a key error.
        """Waiting on an unknown call id raises KeyError."""
        store = _make_store()

        with pytest.raises(KeyError):
            await store.wait(call_id="missing", timeout=0.01)

    async def test_wait_after_retrieval_still_returns(self) -> None:
        # [FUNCTION DOC] A second wait after a first retrieval still returns the same outcome.
        """A retrieved outcome remains available before its TTL expires."""
        store = _make_store()
        expected = CallStatusResponse(status="success", result=1)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=expected)
        await store.wait(call_id="c1", timeout=0.01)

        assert await store.wait(call_id="c1", timeout=0.01) == expected


class TestPurge:
    # [CLASS DOC] Tests lazy purging of expired records. Each test replaces the store module clock
    # with a mutable fake starting at 10, advances it, then begins another call to trigger the purge
    # and asserts whether call c1 is still contained. Boundaries are exclusive: a record expires only
    # when the clock is strictly past the ttl deadline.
    async def test_retrieved_record_purged_after_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Retrieved at 10 with ttl 5; at 15.01 it is purged.
        """A retrieved record is purged after its TTL."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = await _retrieved_call(
            ttl=5.0,
            outcome=CallStatusResponse(status="success", result=1),
        )

        now[0] = 15.01
        store.begin(call_id="other")

        assert not store.contains("c1")

    async def test_retrieved_record_kept_within_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Retrieved at 10 with ttl 5; at exactly 15 it is still kept.
        """A retrieved record remains tracked within its TTL."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = await _retrieved_call(
            ttl=5.0,
            outcome=CallStatusResponse(status="success", result=1),
        )

        now[0] = 15.0
        store.begin(call_id="other")

        assert store.contains("c1")

    async def test_failed_outcome_also_purged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Same as the success case with a failed outcome; purged at 15.01.
        """Failed outcomes follow the same retrieval TTL."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = await _retrieved_call(
            ttl=5.0,
            outcome=CallStatusResponse(status="failed", error="boom"),
        )

        now[0] = 15.01
        store.begin(call_id="other")

        assert not store.contains("c1")

    async def test_later_retrieval_does_not_extend_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Retrieved at 10 and again at 14; ttl 5 still counts from the first retrieval,
        # so it is purged at 15.01.
        """Later retrievals do not extend the first retrieval TTL."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = await _retrieved_call(
            ttl=5.0,
            outcome=CallStatusResponse(status="success", result=1),
        )

        now[0] = 14.0
        await store.wait(call_id="c1", timeout=0.01)
        now[0] = 15.01
        store.begin(call_id="other")

        assert not store.contains("c1")

    async def test_never_retrieved_record_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Finished but never retrieved with the default (long) finished ttl; still
        # kept at 20.
        """A finished but never retrieved record is retained."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 20.0
        store.begin(call_id="other")

        assert store.contains("c1")

    async def test_never_retrieved_record_purged_after_finished_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Finished at 10 with finished ttl 100; at 110.01 it is purged.
        """A finished but never retrieved record is purged once the finished TTL passes."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0, finished_ttl=100.0)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 110.01
        store.begin(call_id="other")

        assert not store.contains("c1")

    async def test_never_retrieved_record_kept_within_finished_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Same setup; at exactly 110 it is still kept.
        """A finished but never retrieved record survives while inside the finished TTL."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0, finished_ttl=100.0)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 110.0
        store.begin(call_id="other")

        assert store.contains("c1")

    async def test_never_retrieved_failed_outcome_expires_after_finished_ttl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] A failed, never retrieved record is kept at 110 and purged at 110.01 (finished
        # ttl 100, finished at 10).
        """A failed outcome nobody collected expires on the finished TTL just like a successful one."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0, finished_ttl=100.0)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="failed", error="boom"))

        now[0] = 110.0
        store.begin(call_id="other")
        assert store.contains("c1")

        now[0] = 110.01
        store.begin(call_id="another")
        assert not store.contains("c1")

    async def test_never_retrieved_records_do_not_accumulate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With both ttls zero, 50 calls are begun and finished one clock second apart,
        # never retrieved. After one more begin, only the last call remains of the sampled ids
        # (first, middle, final and the new one).
        """Many finished calls whose results are never collected do not grow the store."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=0.0, finished_ttl=0.0)

        for index in range(50):
            now[0] += 1.0
            store.begin(call_id=f"c{index}")
            store.finish(call_id=f"c{index}", outcome=CallStatusResponse(status="success", result=index))

        now[0] += 1.0
        store.begin(call_id="last")

        assert [call_id for call_id in ("c0", "c25", "c49", "last") if store.contains(call_id)] == ["last"]

    async def test_unfinished_record_never_purged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] A pending call survives a clock jump to a billion seconds even with zero ttls.
        """A still running call is retained however old it gets, even with both TTLs at zero."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=0.0, finished_ttl=0.0)
        store.begin(call_id="c1")

        now[0] = 1e9
        store.begin(call_id="other")

        assert store.contains("c1")

    async def test_retrieved_record_purged_despite_long_finished_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Retrieved at 10 with retrieved ttl 5 and a huge finished ttl; purged at 15.01.
        """Once retrieved, a record expires on the retrieved TTL and a long finished TTL does not save it."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = await _retrieved_call(
            ttl=5.0,
            finished_ttl=1e6,
            outcome=CallStatusResponse(status="success", result=1),
        )

        now[0] = 15.01
        store.begin(call_id="other")

        assert not store.contains("c1")

    async def test_late_retrieval_keeps_record_for_retrieved_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Finished at 10 (finished ttl 100), retrieved at 100 (retrieved ttl 5); at 105
        # it is still kept, since the retrieved ttl counts from retrieval.
        """A record retrieved long after finishing is kept for the retrieved TTL counted from that retrieval."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0, finished_ttl=100.0)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 100.0
        await store.wait(call_id="c1", timeout=0.01)
        now[0] = 105.0
        store.begin(call_id="other")

        assert store.contains("c1")

    async def test_finished_ttl_runs_from_finishing_not_from_submission(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Begun at 10, finished at 500 (finished ttl 100); at 599 it is kept, showing
        # the finished ttl starts at finishing.
        """A long running call still gets its whole finished TTL, counted from when it finished."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0, finished_ttl=100.0)
        store.begin(call_id="c1")

        now[0] = 500.0
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 599.0
        store.begin(call_id="other")

        assert store.contains("c1")

    async def test_retrieval_just_before_the_finished_deadline_wins_the_retrieved_ttl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Finished at 10 (deadline 110), retrieved at 109 (retrieved ttl 5); at 112,
        # past the finished deadline but inside the retrieved ttl, it is kept.
        """Once retrieved, the retrieved TTL alone decides, so a record may outlive its finished deadline."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0, finished_ttl=100.0)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 109.0
        await store.wait(call_id="c1", timeout=0.01)
        now[0] = 112.0
        store.begin(call_id="other")

        assert store.contains("c1")


class TestDefaultTtls:
    # [CLASS DOC] Tests the default ttl values.
    async def test_finished_ttl_default_is_twelve_hours(self) -> None:
        # [FUNCTION DOC] The finished ttl constant equals twelve hours in seconds.
        """The finished TTL constant is twelve hours."""
        assert FINISHED_TTL_SECONDS == 12 * 3600.0

    async def test_default_store_uses_five_minute_retrieved_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] A default store, retrieved at 10, still contains the call at 310 and no longer
        # at 310.01.
        """A default store keeps a retrieved outcome for five minutes and purges it right after."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = CallStore()
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))
        await store.wait(call_id="c1", timeout=0.01)

        now[0] = 10.0 + 300.0
        store.begin(call_id="other")
        assert store.contains("c1")

        now[0] = 10.0 + 300.01
        store.begin(call_id="another")
        assert not store.contains("c1")

    async def test_default_store_purges_never_retrieved_after_finished_ttl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] A default store, finished at 10 and never retrieved, still contains the call
        # exactly one finished ttl later and no longer 0.01 seconds after that.
        """A default store keeps a never retrieved record for the default finished TTL and no longer."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = CallStore()
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 10.0 + FINISHED_TTL_SECONDS
        store.begin(call_id="other")
        assert store.contains("c1")

        now[0] = 10.0 + FINISHED_TTL_SECONDS + 0.01
        store.begin(call_id="another")
        assert not store.contains("c1")


class TestInFlightCallIds:
    # [CLASS DOC] Tests the sorted list of ids of calls begun but not yet finished.
    def test_a_fresh_store_is_idle(self):
        # [FUNCTION DOC] A new store lists no call ids.
        """A store that was never asked to run anything has nothing for a caller to wait out."""
        assert _make_store().in_flight_call_ids() == []

    def test_a_submitted_call_is_in_flight(self):
        # [FUNCTION DOC] After begin, the list is exactly that id.
        """A call still running is exactly what a caller waiting the worker out has to see."""
        store = _make_store()
        store.begin(call_id="c1")

        assert store.in_flight_call_ids() == ["c1"]

    def test_a_finished_call_is_no_longer_in_flight(self):
        # [FUNCTION DOC] After finish, the list is empty.
        """Waiting on a call whose result is already recorded would never let the caller proceed."""
        store = _make_store()
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success"))

        assert store.in_flight_call_ids() == []

    def test_the_call_ids_are_sorted(self):
        # [FUNCTION DOC] Ids begun in reverse order (c2 then c1) are listed sorted ascending.
        """They are logged and compared, so an arbitrary dict order would make the wait look flaky."""
        store = _make_store()
        for call_id in ("c2", "c1"):
            store.begin(call_id=call_id)

        assert store.in_flight_call_ids() == ["c1", "c2"]
