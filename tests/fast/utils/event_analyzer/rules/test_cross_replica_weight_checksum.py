# [MODULE DOC] Unit tests for the cross-replica weight checksum rule of the event analyzer. The
# rule takes logged training-engine weight checksum events (each from a training process
# identified by cell index and rank within the cell, at a rollout id, carrying parameter hashes,
# buffer hashes and optimizer state hashes), groups them by rollout id and rank within the cell,
# compares every cell's event against the first cell's, and returns a list of mismatches. Tests
# also cover flattening one event into a flat dictionary and the nested flattening helper
# (imported from the shared comparison module). Events are built with a fixed timestamp.
# KNOWN ISSUE: the existing module docstring says these are tests for "rules/weight_checksum",
# but the module under test is the cross replica weight checksum rule.
"""Tests for event_analyzer rules/weight_checksum."""

from datetime import datetime, timezone

from miles.utils.audit_utils.event_analyzer.rules.checksum_compare import flatten_nested as _flatten_nested
from miles.utils.audit_utils.event_analyzer.rules.cross_replica_weight_checksum import _flatten_event, check
from miles.utils.audit_utils.event_logger.models import (
    OptimizerStateInfo,
    TrainEngineLocalWeightChecksumEvent,
    TrainEngineLocalWeightChecksumState,
)
from miles.utils.audit_utils.process_identity import TrainProcessIdentity

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_event(
    rollout_id: int,
    cell_index: int = 0,
    rank_within_cell: int = 0,
    param_hashes: dict[str, str] | None = None,
    buffer_hashes: dict[str, str] | None = None,
    optimizer_state_dict: dict | None = None,
) -> TrainEngineLocalWeightChecksumEvent:
    # [FUNCTION DOC] HELPER: Builds a weight checksum event with the fixed timestamp, a source
    # identity of component "actor" with the given cell index and rank within cell, the given
    # rollout id, and a state holding the parameter hashes and buffer hashes (each defaulting to
    # an empty mapping). The optimizer hashes list is empty when no optimizer state dictionary is
    # given (none); otherwise it holds one optimizer info with parameter names mapping index 0 to
    # "pp0.weight" and the given state dictionary (an empty dictionary stays an empty state).
    return TrainEngineLocalWeightChecksumEvent(
        timestamp=_FIXED_TS,
        source=TrainProcessIdentity(component="actor", cell_index=cell_index, rank_within_cell=rank_within_cell),
        rollout_id=rollout_id,
        state=TrainEngineLocalWeightChecksumState(
            param_hashes=param_hashes or {},
            buffer_hashes=buffer_hashes or {},
            optimizer_hashes=(
                [
                    OptimizerStateInfo(
                        param_names={0: "pp0.weight"},
                        state_dict=optimizer_state_dict or {},
                    ),
                ]
                if optimizer_state_dict is not None
                else []
            ),
        ),
    )


class TestCheck:
    # [CLASS DOC] Tests for the rule's check function, which takes a list of events and returns a
    # list of mismatch items (key, values of both sides, labels containing the rollout id in the
    # form "rollout_N/..."). Comparison is within a rollout id and a rank within cell, across
    # cells, always against the first cell.
    def test_matching_replicas_no_mismatches(self) -> None:
        # [FUNCTION DOC] Three cells with identical parameter hashes give an empty result.
        events = [
            _make_event(rollout_id=0, cell_index=0, param_hashes={"pp0.weight": "aaa"}),
            _make_event(rollout_id=0, cell_index=1, param_hashes={"pp0.weight": "aaa"}),
            _make_event(rollout_id=0, cell_index=2, param_hashes={"pp0.weight": "aaa"}),
        ]
        assert check(events) == []

    def test_param_hash_mismatch_detected(self) -> None:
        # [FUNCTION DOC] Two cells with different hashes for one parameter give at least one
        # mismatch whose key contains "param_hashes.pp0.weight".
        events = [
            _make_event(rollout_id=5, cell_index=0, param_hashes={"pp0.weight": "aaa"}),
            _make_event(rollout_id=5, cell_index=1, param_hashes={"pp0.weight": "zzz"}),
        ]
        mismatches = check(events)

        assert len(mismatches) >= 1
        keys = [m.key for m in mismatches]
        assert any("param_hashes.pp0.weight" in k for k in keys)

    def test_missing_key_in_one_replica_detected(self) -> None:
        # [FUNCTION DOC] When the second cell lacks a parameter the first has, a mismatch for
        # that key is reported and the second side's value contains the marker "<missing>".
        events = [
            _make_event(rollout_id=0, cell_index=0, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"}),
            _make_event(rollout_id=0, cell_index=1, param_hashes={"pp0.weight": "aaa"}),
        ]
        mismatches = check(events)

        assert len(mismatches) >= 1
        keys = [m.key for m in mismatches]
        assert any("pp0.bias" in k for k in keys)
        assert any("<missing>" in m.value_b for m in mismatches)

    def test_multiple_steps_only_mismatched_step_reported(self) -> None:
        # [FUNCTION DOC] Three rollouts with two cells each, only rollout 1 differing: every
        # reported mismatch has a label (on either side) containing "rollout_1/".
        events = [
            # Step 0: match
            _make_event(rollout_id=0, cell_index=0, param_hashes={"pp0.w": "aaa"}),
            _make_event(rollout_id=0, cell_index=1, param_hashes={"pp0.w": "aaa"}),
            # Step 1: mismatch
            _make_event(rollout_id=1, cell_index=0, param_hashes={"pp0.w": "aaa"}),
            _make_event(rollout_id=1, cell_index=1, param_hashes={"pp0.w": "zzz"}),
            # Step 2: match
            _make_event(rollout_id=2, cell_index=0, param_hashes={"pp0.w": "aaa"}),
            _make_event(rollout_id=2, cell_index=1, param_hashes={"pp0.w": "aaa"}),
        ]
        mismatches = check(events)

        assert len(mismatches) >= 1
        for m in mismatches:
            assert "rollout_1/" in m.label_a or "rollout_1/" in m.label_b

    def test_empty_events_no_mismatches(self) -> None:
        # [FUNCTION DOC] An empty event list gives an empty result.
        assert check([]) == []

    def test_single_replica_no_comparison(self) -> None:
        # [FUNCTION DOC] A single event has nothing to compare against and gives an empty
        # result.
        events = [_make_event(rollout_id=0, cell_index=0, param_hashes={"pp0.w": "aaa"})]
        assert check(events) == []

    def test_buffer_mismatch_detected(self) -> None:
        # [FUNCTION DOC] Differing buffer hashes between two cells give a mismatch whose key
        # contains "buffer_hashes".
        events = [
            _make_event(rollout_id=0, cell_index=0, buffer_hashes={"pp0.running_mean": "aaa"}),
            _make_event(rollout_id=0, cell_index=1, buffer_hashes={"pp0.running_mean": "bbb"}),
        ]
        mismatches = check(events)

        assert len(mismatches) >= 1
        assert any("buffer_hashes" in m.key for m in mismatches)

    def test_optimizer_state_mismatch_detected(self) -> None:
        # [FUNCTION DOC] Differing hashes of an optimizer state entry ("exp_avg") between two
        # cells give a mismatch whose key contains "exp_avg".
        events = [
            _make_event(rollout_id=3, cell_index=0, optimizer_state_dict={"state": {0: {"exp_avg": "aaa"}}}),
            _make_event(rollout_id=3, cell_index=1, optimizer_state_dict={"state": {0: {"exp_avg": "bbb"}}}),
        ]
        mismatches = check(events)

        assert len(mismatches) >= 1
        assert any("exp_avg" in m.key for m in mismatches)

    def test_non_tensor_state_mismatch_detected(self) -> None:
        # [FUNCTION DOC] Differing plain numbers in the optimizer state (a step counter, 10
        # versus 20) are also compared and reported, with a key containing "step".
        events = [
            _make_event(rollout_id=0, cell_index=0, optimizer_state_dict={"state": {0: {"step": 10}}}),
            _make_event(rollout_id=0, cell_index=1, optimizer_state_dict={"state": {0: {"step": 20}}}),
        ]
        mismatches = check(events)

        assert len(mismatches) >= 1
        assert any("step" in m.key for m in mismatches)

    def test_different_ranks_not_compared(self) -> None:
        # [FUNCTION DOC] Two events in the same cell but different ranks within the cell (as
        # with pipeline sharding, differing parameter names and hashes) are not compared against
        # each other and give an empty result.
        """Events from different rank_within_cell should not be compared (PP sharding)."""
        events = [
            _make_event(rollout_id=0, cell_index=0, rank_within_cell=0, param_hashes={"pp0.w": "aaa"}),
            _make_event(rollout_id=0, cell_index=0, rank_within_cell=1, param_hashes={"pp1.w": "bbb"}),
        ]
        assert check(events) == []

    def test_same_rank_different_cells_mismatch(self) -> None:
        # [FUNCTION DOC] Same rank within cell in two different cells with different hashes
        # gives at least one mismatch.
        """Same rank_within_cell across different cells must be compared."""
        events = [
            _make_event(rollout_id=0, cell_index=0, rank_within_cell=0, param_hashes={"pp0.w": "aaa"}),
            _make_event(rollout_id=0, cell_index=1, rank_within_cell=0, param_hashes={"pp0.w": "zzz"}),
        ]
        mismatches = check(events)
        assert len(mismatches) >= 1

    def test_same_rank_different_cells_match(self) -> None:
        # [FUNCTION DOC] Two cells with two ranks each, where each rank has the same hashes in
        # both cells (and ranks have different parameter names), give an empty result.
        """Same rank_within_cell across different cells, identical hashes → no mismatch."""
        events = [
            _make_event(rollout_id=0, cell_index=0, rank_within_cell=0, param_hashes={"pp0.w": "aaa"}),
            _make_event(rollout_id=0, cell_index=1, rank_within_cell=0, param_hashes={"pp0.w": "aaa"}),
            _make_event(rollout_id=0, cell_index=0, rank_within_cell=1, param_hashes={"pp1.w": "bbb"}),
            _make_event(rollout_id=0, cell_index=1, rank_within_cell=1, param_hashes={"pp1.w": "bbb"}),
        ]
        assert check(events) == []

    def test_multi_rank_multi_cell_only_mismatched_rank_reported(self) -> None:
        # [FUNCTION DOC] Two cells with two ranks, where only rank 1 differs: every reported
        # mismatch key refers to that rank's parameter "pp1.w".
        """Only the rank with mismatch should produce issues."""
        events = [
            # rank 0: match
            _make_event(rollout_id=0, cell_index=0, rank_within_cell=0, param_hashes={"pp0.w": "aaa"}),
            _make_event(rollout_id=0, cell_index=1, rank_within_cell=0, param_hashes={"pp0.w": "aaa"}),
            # rank 1: mismatch
            _make_event(rollout_id=0, cell_index=0, rank_within_cell=1, param_hashes={"pp1.w": "bbb"}),
            _make_event(rollout_id=0, cell_index=1, rank_within_cell=1, param_hashes={"pp1.w": "zzz"}),
        ]
        mismatches = check(events)
        assert len(mismatches) >= 1
        assert all("pp1.w" in m.key for m in mismatches)

    def test_three_cells_first_vs_rest(self) -> None:
        # [FUNCTION DOC] With three cells where only the third differs from the first, at least
        # one mismatch is reported (cells after the first are compared against the first).
        """With 3 cells, all are compared against the first."""
        events = [
            _make_event(rollout_id=0, cell_index=0, param_hashes={"w": "aaa"}),
            _make_event(rollout_id=0, cell_index=1, param_hashes={"w": "aaa"}),
            _make_event(rollout_id=0, cell_index=2, param_hashes={"w": "bbb"}),
        ]
        mismatches = check(events)
        assert len(mismatches) >= 1


class TestFlattenEvent:
    # [CLASS DOC] Tests for flattening one weight checksum event into a flat dictionary from
    # dotted key to value.
    def test_excludes_metadata_fields(self) -> None:
        # [FUNCTION DOC] The flat dictionary has no key starting with "step", "rank" or "type"
        # (event metadata is left out) but has the parameter hash key "param_hashes.pp0.w".
        event = _make_event(rollout_id=0, cell_index=0, param_hashes={"pp0.w": "aaa"})
        flat = _flatten_event(event)

        assert not any(k.startswith("step") or k.startswith("rank") or k.startswith("type") for k in flat.keys())
        assert "param_hashes.pp0.w" in flat

    def test_includes_optimizer_hashes(self) -> None:
        # [FUNCTION DOC] An event with an optimizer state entry "exp_avg" produces at least one
        # key containing "exp_avg".
        event = _make_event(rollout_id=0, cell_index=0, optimizer_state_dict={"state": {0: {"exp_avg": "hash1"}}})
        flat = _flatten_event(event)

        assert any("exp_avg" in k for k in flat.keys())


class TestFlattenNested:
    # [CLASS DOC] Tests for the nested flattening helper as used by this rule (imported under a
    # private alias): dotted keys for dictionaries, bracket indices for lists, numbers kept as
    # leaves, and an optional prefix.
    # KNOWN ISSUE: this class overlaps with the flatten tests in the checksum compare test module,
    # which test the same function.
    def test_flat_dict_with_string_values(self) -> None:
        # [FUNCTION DOC] A flat dictionary of two string values is prefixed with "root.".
        result = _flatten_nested({"a": "hash1", "b": "hash2"}, prefix="root")
        assert result == {"root.a": "hash1", "root.b": "hash2"}

    def test_nested_dict(self) -> None:
        # [FUNCTION DOC] A nested dictionary with an integer key flattens to a single dotted
        # key under the prefix.
        result = _flatten_nested({"state": {0: {"exp_avg": "h1"}}}, prefix="opt0")
        assert result == {"opt0.state.0.exp_avg": "h1"}

    def test_list_values(self) -> None:
        # [FUNCTION DOC] A list flattens to bracket-indexed keys.
        result = _flatten_nested({"params": ["a", "b"]}, prefix="opt0")
        assert result == {"opt0.params[0]": "a", "opt0.params[1]": "b"}

    def test_keeps_int_and_float_leaves(self) -> None:
        # [FUNCTION DOC] Float and integer leaves are kept with their original types next to a
        # string leaf.
        result = _flatten_nested({"lr": 0.001, "step": 42, "hash": "abc"}, prefix="root")
        assert result == {"root.hash": "abc", "root.lr": 0.001, "root.step": 42}

    def test_empty_prefix(self) -> None:
        # [FUNCTION DOC] With an empty prefix the keys carry no leading dot.
        result = _flatten_nested({"a": "x"}, prefix="")
        assert result == {"a": "x"}

    def test_empty_dict(self) -> None:
        # [FUNCTION DOC] An empty dictionary flattens to an empty dictionary.
        result = _flatten_nested({}, prefix="root")
        assert result == {}
