# [MODULE DOC]
# Tests for the simple heartbeat in the fault-tolerance utilities. A heartbeat holds an immutable
# status snapshot with a bump count (starts at zero, grows by one per bump, never resets) and the
# last-active wall-clock timestamp (taken from the time module at construction and at each bump;
# tests replace that clock). Checks count behaviour, the frozen status dataclass, snapshot identity
# and isolation, and timestamp tracking.
import miles.utils.ft_utils.heartbeat_utils as heartbeat_utils
from miles.utils.ft_utils.heartbeat_utils import HeartbeatStatus, SimpleHeartbeat


class TestSimpleHeartbeatBumpCount:
    # [CLASS DOC] Tests of the bump counter.
    def test_initial_bump_count_is_zero(self):
        # [FUNCTION DOC] A new heartbeat reports a bump count of zero.
        """A fresh SimpleHeartbeat reports a bump_count of 0."""
        heartbeat = SimpleHeartbeat()
        assert heartbeat.status().bump_count == 0

    def test_single_bump_increments_count(self):
        # [FUNCTION DOC] One bump raises the count from zero to one.
        """One bump() raises bump_count from 0 to 1."""
        heartbeat = SimpleHeartbeat()
        heartbeat.bump()
        assert heartbeat.status().bump_count == 1

    def test_bump_count_increments_monotonically(self):
        # [FUNCTION DOC] Five consecutive bumps give counts one through five.
        """Repeated bump() calls increment bump_count by exactly one each time."""
        heartbeat = SimpleHeartbeat()
        for expected in range(1, 6):
            heartbeat.bump()
            assert heartbeat.status().bump_count == expected

    def test_bump_count_never_resets(self):
        # [FUNCTION DOC] Twenty bumps each raise the count by exactly one, ending at twenty.
        """bump_count keeps growing and never drops back across many bumps."""
        heartbeat = SimpleHeartbeat()
        previous = heartbeat.status().bump_count
        for _ in range(20):
            heartbeat.bump()
            current = heartbeat.status().bump_count
            assert current == previous + 1
            previous = current
        assert heartbeat.status().bump_count == 20


class TestSimpleHeartbeatStatusObject:
    # [CLASS DOC] Tests of the status snapshot object's type, immutability and identity.
    def test_status_returns_heartbeat_status_instance(self):
        # [FUNCTION DOC] The status is an instance of the heartbeat status class.
        """status() yields a HeartbeatStatus dataclass instance."""
        heartbeat = SimpleHeartbeat()
        assert isinstance(heartbeat.status(), HeartbeatStatus)

    def test_status_is_frozen(self):
        # [FUNCTION DOC] Assigning to a field of the returned status must raise the frozen-instance
        # error of dataclasses; if no error occurs the test raises an assertion error.
        """The returned HeartbeatStatus is frozen and rejects attribute mutation."""
        import dataclasses

        heartbeat = SimpleHeartbeat()
        status = heartbeat.status()
        try:
            status.bump_count = 999
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError("expected FrozenInstanceError on frozen dataclass")

    def test_status_returns_new_object_each_bump(self):
        # [FUNCTION DOC] The status object obtained before a bump is a different object from the
        # one obtained after.
        """Each bump() replaces the status with a distinct object identity."""
        heartbeat = SimpleHeartbeat()
        before = heartbeat.status()
        heartbeat.bump()
        after = heartbeat.status()
        assert before is not after

    def test_status_stable_between_bumps(self):
        # [FUNCTION DOC] Two status reads with no bump between them return the identical object.
        """status() returns the same object identity when no bump occurs in between."""
        heartbeat = SimpleHeartbeat()
        assert heartbeat.status() is heartbeat.status()

    def test_old_status_snapshot_unaffected_by_later_bump(self):
        # [FUNCTION DOC] A snapshot taken at count zero keeps zero after two bumps while a fresh
        # status shows two.
        """A captured status snapshot keeps its bump_count after subsequent bumps."""
        heartbeat = SimpleHeartbeat()
        snapshot = heartbeat.status()
        heartbeat.bump()
        heartbeat.bump()
        assert snapshot.bump_count == 0
        assert heartbeat.status().bump_count == 2


class TestSimpleHeartbeatTimestamp:
    # [CLASS DOC] Tests of the last-active timestamp, with the time function monkeypatched.
    def test_initial_timestamp_from_time_time(self, monkeypatch):
        # [FUNCTION DOC] With the clock fixed at 1000.0, a new heartbeat's timestamp is 1000.0.
        """Construction stamps last_active_timestamp from time.time()."""
        monkeypatch.setattr(heartbeat_utils.time, "time", lambda: 1000.0)
        heartbeat = SimpleHeartbeat()
        assert heartbeat.status().last_active_timestamp == 1000.0

    def test_bump_updates_timestamp_from_time_time(self, monkeypatch):
        # [FUNCTION DOC] Construct at clock 1000.0, move the clock to 2500.0 and bump; the status
        # then shows timestamp 2500.0 and count one.
        """bump() refreshes last_active_timestamp to the current time.time() value."""
        clock = {"now": 1000.0}
        monkeypatch.setattr(heartbeat_utils.time, "time", lambda: clock["now"])
        heartbeat = SimpleHeartbeat()
        assert heartbeat.status().last_active_timestamp == 1000.0

        clock["now"] = 2500.0
        heartbeat.bump()
        status = heartbeat.status()
        assert status.last_active_timestamp == 2500.0
        assert status.bump_count == 1

    def test_consecutive_bumps_track_timestamp(self, monkeypatch):
        # [FUNCTION DOC] Bump at clock 200.0 then at 350.0; after each bump the timestamp equals the
        # clock value at that bump, and the final count is two.
        """Each bump() captures the time.time() value observed at that call."""
        clock = {"now": 100.0}
        monkeypatch.setattr(heartbeat_utils.time, "time", lambda: clock["now"])
        heartbeat = SimpleHeartbeat()

        clock["now"] = 200.0
        heartbeat.bump()
        assert heartbeat.status().last_active_timestamp == 200.0

        clock["now"] = 350.0
        heartbeat.bump()
        assert heartbeat.status().last_active_timestamp == 350.0
        assert heartbeat.status().bump_count == 2
