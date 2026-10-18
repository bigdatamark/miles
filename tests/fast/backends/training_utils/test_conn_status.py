# [MODULE DOC] CPU tests for the weight-update connection status manager, which decides whether the
# trainer must reconnect to rollout cells given a snapshot (mapping from cell id to a hash). It
# reconnects before the first connect (even for an empty snapshot), after any change (cell added,
# removed, replaced, id swapped, all gone), after the trainer is marked stale, and not otherwise;
# an empty snapshot is a valid recorded state; and the stored snapshot is copied.
import pytest

from miles.backends.training_utils.weight_update.conn_status import ConnStatusManager

_INITIAL_SNAPSHOT: dict[str, str] = {"cell-a": "hash-a", "cell-b": "hash-b"}


@pytest.mark.parametrize(
    ("next_snapshot", "expected_needs_reconnect"),
    [
        pytest.param(dict(_INITIAL_SNAPSHOT), False, id="unchanged"),
        pytest.param({**_INITIAL_SNAPSHOT, "cell-c": "hash-c"}, True, id="cell-added"),
        pytest.param({"cell-a": "hash-a"}, True, id="cell-removed"),
        pytest.param({"cell-a": "hash-a", "cell-b": "hash-b-restarted"}, True, id="cell-replaced"),
        pytest.param({"cell-a": "hash-a", "cell-c": "hash-b"}, True, id="cell-id-swapped"),
        pytest.param({}, True, id="all-cells-gone"),
    ],
)
def test_needs_reconnect_tracks_snapshot_changes(
    next_snapshot: dict[str, str], expected_needs_reconnect: bool
) -> None:
    # [FUNCTION DOC] Test (parametrized over six next-snapshot variants): after marking the initial
    # two-cell snapshot as reconnected, needs-reconnect is false only for an equal snapshot and
    # true for an added cell, a removed cell, a cell with a changed hash, a swapped cell id, and an
    # empty snapshot.
    """After a successful connect, only a changed rollout cell snapshot forces another reconnect."""
    manager = ConnStatusManager()
    manager.mark_reconnected(_INITIAL_SNAPSHOT)

    assert manager.needs_reconnect(next_snapshot) is expected_needs_reconnect


@pytest.mark.parametrize(
    "snapshot",
    [
        pytest.param({}, id="empty-snapshot"),
        pytest.param({"cell-a": "hash-a"}, id="non-empty-snapshot"),
    ],
)
def test_needs_reconnect_is_true_before_first_connect(snapshot: dict[str, str]) -> None:
    # [FUNCTION DOC] Test (parametrized: empty and one-cell snapshot): a fresh manager reports that
    # a reconnect is needed.
    """A freshly created manager always reconnects, including for a defensive empty snapshot."""
    manager = ConnStatusManager()

    assert manager.needs_reconnect(snapshot) is True


def test_empty_snapshot_stops_reconnecting_once_marked() -> None:
    # [FUNCTION DOC] Test: after marking an empty snapshot as reconnected, an equal empty snapshot
    # no longer needs a reconnect.
    """An empty snapshot is a real state: after marking it connected, an equal empty snapshot needs no reconnect."""
    manager = ConnStatusManager()
    manager.mark_reconnected({})

    assert manager.needs_reconnect({}) is False


def test_trainer_stale_forces_reconnect_on_unchanged_snapshot() -> None:
    # [FUNCTION DOC] Test: after a reconnect is recorded and the trainer is marked stale, an
    # identical snapshot still needs a reconnect; recording the reconnect again clears the stale
    # flag so the identical snapshot no longer needs one.
    """A stale trainer reconnects even when the rollout cell snapshot is identical, and clears the flag afterwards."""
    manager = ConnStatusManager()
    manager.mark_reconnected(_INITIAL_SNAPSHOT)
    manager.mark_trainer_stale()

    assert manager.needs_reconnect(dict(_INITIAL_SNAPSHOT)) is True

    manager.mark_reconnected(_INITIAL_SNAPSHOT)

    assert manager.needs_reconnect(dict(_INITIAL_SNAPSHOT)) is False


def test_reconnect_decisions_are_repeatable_across_updates() -> None:
    # [FUNCTION DOC] Test: recording a two-cell snapshot then checking a one-cell snapshot needs a
    # reconnect; after recording the one-cell snapshot it does not, and going back to the two-cell
    # snapshot needs one again.
    """Scale-down then scale-up each trigger exactly one reconnect, and the steady state stays connected."""
    manager = ConnStatusManager()
    scaled_up: dict[str, str] = {"cell-a": "hash-a", "cell-b": "hash-b"}
    scaled_down: dict[str, str] = {"cell-a": "hash-a"}

    manager.mark_reconnected(scaled_up)

    assert manager.needs_reconnect(scaled_down) is True

    manager.mark_reconnected(scaled_down)

    assert manager.needs_reconnect(scaled_down) is False
    assert manager.needs_reconnect(scaled_up) is True


def test_stored_snapshot_is_isolated_from_caller_mutation() -> None:
    # [FUNCTION DOC] Test: after recording a snapshot dictionary and then mutating the caller's
    # dictionary (changing a hash), checking the mutated dictionary reports a needed reconnect
    # (the manager stored a copy).
    """The manager copies the snapshot it was given, so mutating the caller's dict still reads as a change."""
    manager = ConnStatusManager()
    snapshot: dict[str, str] = {"cell-a": "hash-a"}
    manager.mark_reconnected(snapshot)

    snapshot["cell-a"] = "hash-a-restarted"

    assert manager.needs_reconnect(snapshot) is True
