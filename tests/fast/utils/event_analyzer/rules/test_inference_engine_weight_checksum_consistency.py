# [MODULE DOC] Unit tests for the inference engine weight checksum consistency rule of the event
# analyzer. The rule takes events that each carry, for one rollout id, a list with one mapping per
# inference engine from tensor name to checksum string; it compares every engine against engine 0
# of the same event and returns mismatch items labelled "rollout_<id>/engine_<index>" for both
# sides. Events are built with a fixed timestamp and a main-component source.
"""Tests for event_analyzer rules/inference_engine_weight_checksum_consistency."""

from datetime import datetime, timezone

from miles.utils.audit_utils.event_analyzer.rules.inference_engine_weight_checksum_consistency import check
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_event(*, rollout_id: int, engine_checksums: list[dict[str, str]]) -> InferenceEngineWeightChecksumEvent:
    # [FUNCTION DOC] HELPER: Builds an inference engine checksum event with the fixed timestamp,
    # a main-component source, the given rollout id and the given per-engine checksum mappings.
    return InferenceEngineWeightChecksumEvent(
        timestamp=_FIXED_TS,
        source=SimpleProcessIdentity(component="main"),
        rollout_id=rollout_id,
        engine_checksums=engine_checksums,
    )


class TestCheck:
    # [CLASS DOC] Tests for the rule's check function (list of events in, list of mismatch items
    # out).
    def test_empty_events_no_mismatches(self) -> None:
        # [FUNCTION DOC] An empty event list gives an empty result.
        """No engine checksum events means nothing to check."""
        assert check([]) == []

    def test_single_engine_no_comparison(self) -> None:
        # [FUNCTION DOC] An event with a single engine gives an empty result.
        """A single engine has no peer to compare against."""
        events = [_make_event(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}])]
        assert check(events) == []

    def test_matching_engines_no_mismatches(self) -> None:
        # [FUNCTION DOC] Three engines with identical checksums give an empty result.
        """All engines holding identical checksums produce no issues."""
        events = [
            _make_event(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}, {"rank0/w": "aaa"}])
        ]
        assert check(events) == []

    def test_tensor_mismatch_reports_engine_and_tensor(self) -> None:
        # [FUNCTION DOC] Two engines differing in one tensor give exactly one mismatch whose key
        # is the tensor name and whose labels are "rollout_5/engine_0" and "rollout_5/engine_1".
        """A single differing tensor on one engine is reported with engine and tensor labels."""
        events = [_make_event(rollout_id=5, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "zzz"}])]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].key == "rank0/w"
        assert mismatches[0].label_a == "rollout_5/engine_0"
        assert mismatches[0].label_b == "rollout_5/engine_1"

    def test_missing_tensor_on_one_engine_detected(self) -> None:
        # [FUNCTION DOC] A tensor present on engine 0 and absent on engine 1 is reported with
        # the second value containing "<missing>".
        """A tensor present on engine 0 but absent on engine 1 is a mismatch."""
        events = [
            _make_event(rollout_id=0, engine_checksums=[{"rank0/w": "aaa", "rank0/b": "bbb"}, {"rank0/w": "aaa"}])
        ]
        mismatches = check(events)
        assert any(m.key == "rank0/b" and "<missing>" in m.value_b for m in mismatches)

    def test_all_engines_compared_against_first(self) -> None:
        # [FUNCTION DOC] With three engines where only the third differs, exactly one mismatch
        # is reported, labelled engine 0 versus engine 2 (the comparison is always against the
        # first engine).
        """Every engine is compared against engine 0; a later engine's diff names engine_2."""
        events = [
            _make_event(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}, {"rank0/w": "bbb"}])
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].label_a == "rollout_0/engine_0"
        assert mismatches[0].label_b == "rollout_0/engine_2"

    def test_startup_sync_mismatch_labelled_with_the_id_before_the_first_rollout(self) -> None:
        # [FUNCTION DOC] An event with rollout id -1 (the startup sync before any rollout) is
        # checked like any other, and its labels read "rollout_-1/engine_0" and
        # "rollout_-1/engine_1".
        """The out-of-loop startup sync is stamped -1, and its engines are checked like any other rollout's."""
        events = [_make_event(rollout_id=-1, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "zzz"}])]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].label_a == "rollout_-1/engine_0"
        assert mismatches[0].label_b == "rollout_-1/engine_1"

    def test_only_mismatched_rollout_reported(self) -> None:
        # [FUNCTION DOC] Two events, one consistent (rollout 0) and one inconsistent (rollout
        # 1): exactly one mismatch is returned and its first label starts with "rollout_1/".
        """Each rollout is its own event; only the inconsistent rollout yields issues."""
        events = [
            _make_event(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}]),
            _make_event(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "zzz"}]),
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert "rollout_1/" in mismatches[0].label_a
