# [MODULE DOC] Unit tests for the inference engine checksum comparison helpers: comparing a
# baseline and a target run (internal agreement among engines, equality of representative
# checksums, matching sets of model and rollout ids, multi-policy handling, logs written before
# the trainer model id field existed), the weight movement assertion and the engine count
# assertion. Each side is a temporary directory holding event log files written with the real
# event logger (or by hand for legacy logs). A checksum payload is a list with one mapping per
# engine from tensor key to checksum string; rollout id -1 denotes the startup sync.
"""Tests for test_utils.comparisons.inference_engine_checksums.compare_inference_engine_checksums."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, EventLogger
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainerControllerProcessIdentity
from miles.utils.test_utils.comparisons.inference_engine_checksums import (
    assert_engine_count,
    assert_engine_weights_moved,
    compare_inference_engine_checksums,
)


def _write_inference_engine_events(
    side_dir: Path, partials: list[dict[str, Any]], *, model_id: str | None = None
) -> None:
    # [FUNCTION DOC] HELPER: Writes the given partial event payloads as engine weight checksum
    # events, using the real event logger, into the events subdirectory of the side directory.
    # The writer identity comes from the source helper for the model id, and the file is named
    # after that identity. Several calls with different model ids to the same side add separate
    # files. Logger is closed afterwards; nothing is printed.
    events_dir = side_dir / EVENTS_DIRNAME
    source = _source_of(model_id)
    event_logger = EventLogger(log_dir=events_dir, source=source, file_name=f"{source.to_name()}.jsonl")
    for partial in partials:
        event_logger.log(InferenceEngineWeightChecksumEvent, partial, print_log=False)
    event_logger.close()


def _write_events_without_trainer_model_id(side_dir: Path, partials: list[dict[str, Any]]) -> None:
    # [FUNCTION DOC] HELPER: Simulates an old log format. Creates the events directory if
    # needed, builds each event with the current UTC time and the default main source, serializes
    # it to plain JSON data, deletes the trainer model id key, and writes the lines (one JSON
    # document per line) into a file named after the source identity.
    events_dir = side_dir / EVENTS_DIRNAME
    events_dir.mkdir(parents=True, exist_ok=True)
    source = _source_of(None)
    lines: list[str] = []
    for partial in partials:
        dumped = InferenceEngineWeightChecksumEvent(
            **partial, timestamp=datetime.now(timezone.utc), source=source
        ).model_dump(mode="json")
        del dumped["trainer_model_id"]
        lines.append(json.dumps(dumped))
    (events_dir / f"{source.to_name()}.jsonl").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _source_of(model_id: str | None) -> SimpleProcessIdentity | TrainerControllerProcessIdentity:
    # [FUNCTION DOC] HELPER: Returns the main component process identity when no model id is
    # given, otherwise a trainer controller identity whose trainer id is the model id followed by
    # the word actor and whose model id is the given one.
    if model_id is None:
        return SimpleProcessIdentity(component="main")
    return TrainerControllerProcessIdentity(trainer_id=f"{model_id}-actor", model_id=model_id)


def _partial(
    *, rollout_id: int, engine_checksums: list[dict[str, str]], trainer_model_id: str | None = None
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Builds the partial event payload mapping with rollout id, trainer
    # model id (default none) and the per-engine checksum list.
    return dict(rollout_id=rollout_id, trainer_model_id=trainer_model_id, engine_checksums=engine_checksums)


class TestCompareInferenceEngineChecksums:
    # [CLASS DOC] Tests of the main comparison function, which takes baseline and target
    # directory path strings and raises an assertion error on any disagreement.
    def test_identical_passes(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Both sides have one rollout with two agreeing engines; no error.
        """Internally-consistent sides with equal representative checksums pass."""
        partials = [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}])]
        _write_inference_engine_events(tmp_path / "baseline", partials)
        _write_inference_engine_events(tmp_path / "target", partials)

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_differing_engine_counts_still_pass(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Baseline has one engine and target three, all with the same checksum;
        # the comparison passes since only internal agreement and representative equality count.
        """Engine count may differ between sides; only internal agreement + representative equality matter."""
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}, {"rank0/w": "aaa"}])],
        )

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_the_fresh_startup_sync_is_compared_like_any_other_rollout(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Both sides have a startup sync (rollout id -1) with different checksums
        # and a matching rollout 1; asserts an assertion error is raised.
        """The startup push is stamped -1 rather than skipped, so two runs that boot from different weights fail."""
        _write_inference_engine_events(
            tmp_path / "baseline",
            [
                _partial(rollout_id=-1, engine_checksums=[{"rank0/w": "init_baseline"}]),
                _partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}]),
            ],
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=-1, engine_checksums=[{"rank0/w": "init_target"}]),
                _partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}]),
            ],
        )

        with pytest.raises(AssertionError):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_baseline_engines_disagree_fails(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Baseline's two engines hold different checksums; asserts an assertion
        # error whose message says baseline engines disagree.
        """If baseline's own engines disagree, the comparison fails (caught by the consistency rule)."""
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "zzz"}])]
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )

        with pytest.raises(AssertionError, match="Baseline engines disagree"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_target_engines_disagree_fails(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Target's two engines hold different checksums; asserts an assertion
        # error whose message says target engines disagree.
        """If target's own engines disagree, the comparison fails."""
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "zzz"}])]
        )

        with pytest.raises(AssertionError, match="Target engines disagree"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_representative_mismatch_fails(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Each side is self-consistent but the checksums differ; asserts an
        # assertion error whose message names the tensor key.
        """Internally-consistent sides whose representatives differ fail and name the tensor."""
        _write_inference_engine_events(
            tmp_path / "baseline", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "zzz"}])]
        )

        with pytest.raises(AssertionError, match=r"key rank0/w"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_missing_rollout_fails(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Baseline has rollouts 1 and 2, target only 1; asserts an assertion error
        # saying the sets of model and rollout id pairs differ.
        """A rollout present only on one side fails closed."""
        _write_inference_engine_events(
            tmp_path / "baseline",
            [
                _partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}]),
                _partial(rollout_id=2, engine_checksums=[{"rank0/w": "ccc"}]),
            ],
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )

        with pytest.raises(AssertionError, match=r"\(model_id, rollout_id\) sets differ"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_empty_baseline_fails(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Baseline writes no events at all (an empty event file), target has one;
        # asserts an assertion error saying no engine checksum events were found in baseline.
        """No baseline events fails closed rather than vacuously passing."""
        _write_inference_engine_events(tmp_path / "baseline", [])
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )

        with pytest.raises(AssertionError, match="No InferenceEngineWeightChecksumEvents found in baseline"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))


class TestSeveralPolicies:
    # [CLASS DOC] Tests of runs training several policies (model ids a and b), where events are
    # keyed by model id and rollout id together.
    def test_the_same_rollout_id_of_two_policies_is_not_a_duplicate(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] On both sides, policies a and b each log rollout 1 from their own writer
        # identity with different checksums; the comparison passes (no duplicate error).
        """Every policy counts its own rollouts, so keying by rollout id alone rejects a legal multi policy run."""
        for side in ("baseline", "target"):
            _write_inference_engine_events(
                tmp_path / side,
                [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}], trainer_model_id="a")],
                model_id="a",
            )
            _write_inference_engine_events(
                tmp_path / side,
                [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "bbb"}], trainer_model_id="b")],
                model_id="b",
            )

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_a_policy_whose_weights_differ_is_reported(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Policy a agrees across sides but policy b differs (bbb versus ccc);
        # asserts an assertion error naming the baseline and target entries for policy b at
        # rollout 1.
        """Comparing only one of the two policies would hide exactly the drift this comparison exists to catch."""
        _write_inference_engine_events(
            tmp_path / "baseline",
            [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}], trainer_model_id="a")],
            model_id="a",
        )
        _write_inference_engine_events(
            tmp_path / "baseline",
            [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "bbb"}], trainer_model_id="b")],
            model_id="b",
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}], trainer_model_id="a")],
            model_id="a",
        )
        _write_inference_engine_events(
            tmp_path / "target",
            [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "ccc"}], trainer_model_id="b")],
            model_id="b",
        )

        with pytest.raises(AssertionError, match=r"baseline/b/rollout_1 vs target/b/rollout_1"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_the_writers_identity_no_longer_decides_which_policy_an_event_belongs_to(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Both policies' events are written by a single main writer identity (no
        # model id) and only the payload names the policy; the comparison passes.
        """The orchestration script writes every policy's event, so only the payload can name the policy."""
        for side in ("baseline", "target"):
            _write_inference_engine_events(
                tmp_path / side,
                [
                    _partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}], trainer_model_id="a"),
                    _partial(rollout_id=1, engine_checksums=[{"rank0/w": "bbb"}], trainer_model_id="b"),
                ],
            )

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))


class TestLogsPredatingTheTrainerModelIdField:
    # [CLASS DOC] Tests of reading legacy event logs that lack the trainer model id field.
    def test_events_without_the_field_read_back_as_one_unnamed_policy(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Baseline is written in the legacy format and target in the current
        # format with identical payloads; the comparison passes since a missing field reads as
        # the unnamed policy.
        """A log written before the event named its policy is a single policy run, so it compares under None."""
        partials = [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        _write_events_without_trainer_model_id(tmp_path / "baseline", partials)
        _write_inference_engine_events(tmp_path / "target", partials)

        compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_a_field_less_event_still_compares_its_checksums(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Legacy baseline checksum differs from the target; asserts an assertion
        # error naming the tensor key, i.e. legacy events are still compared.
        """Reading the missing field as None must not turn the old side into an unchecked one."""
        _write_events_without_trainer_model_id(
            tmp_path / "baseline", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "aaa"}])]
        )
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_checksums=[{"rank0/w": "zzz"}])]
        )

        with pytest.raises(AssertionError, match=r"key rank0/w"):
            compare_inference_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))


class TestAssertEngineWeightsMoved:
    # [CLASS DOC] Tests of the weight movement assertion, called with a side label and a dump
    # directory path string.
    def test_fewer_than_two_updates_do_not_prove_that_engine_weights_moved(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] One update only; asserts an assertion error saying there is no pair to
        # compare.
        """A lone weight update provides no pair from which movement can be established."""
        _write_inference_engine_events(
            tmp_path / "target", [_partial(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}])]
        )

        with pytest.raises(AssertionError, match="there is no pair to compare"):
            assert_engine_weights_moved(side="target", dump_dir=str(tmp_path / "target"))

    def test_identical_updates_do_not_prove_that_engine_weights_moved(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two updates whose checksum mappings have the same entries in different
        # key order; asserts an assertion error mentioning byte-identical engine weights.
        """Reordered checksum items remain the same weights and cannot establish movement."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=0, engine_checksums=[{"rank0/a": "aaa", "rank0/b": "bbb"}]),
                _partial(rollout_id=1, engine_checksums=[{"rank0/b": "bbb", "rank0/a": "aaa"}]),
            ],
        )

        with pytest.raises(AssertionError, match="byte-identical engine weights"):
            assert_engine_weights_moved(side="target", dump_dir=str(tmp_path / "target"))

    def test_distinct_updates_prove_that_engine_weights_moved(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two updates with different checksums; the assertion passes silently.
        """Two distinct checksum dictionaries establish that engine weights changed."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}]),
                _partial(rollout_id=1, engine_checksums=[{"rank0/w": "bbb"}]),
            ],
        )

        assert_engine_weights_moved(side="target", dump_dir=str(tmp_path / "target"))


class TestAssertEngineCount:
    # [CLASS DOC] Tests of the engine count assertion, which checks that every logged update
    # (including the startup sync) reached the expected number of engines.
    def test_a_run_every_update_of_which_covered_both_engines_passes(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Startup sync and rollout 0 both reached two engines, expected count 2;
        # passes silently.
        """The happy path has to stay reachable, or the refusals below prove nothing."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=-1, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}]),
                _partial(rollout_id=0, engine_checksums=[{"rank0/w": "bbb"}, {"rank0/w": "bbb"}]),
            ],
        )

        assert_engine_count(side="target", dump_dir=str(tmp_path / "target"), expected=2)

    def test_an_update_that_reached_one_engine_of_two_is_caught(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Rollout 0 reached two engines but rollout 1 only one; asserts an
        # assertion error saying weights were pushed to a set of one and two engines.
        """An engine that dropped out mid-run still leaves a run that trains, so nothing else notices."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}]),
                _partial(rollout_id=1, engine_checksums=[{"rank0/w": "bbb"}]),
            ],
        )

        with pytest.raises(AssertionError, match=r"pushed to \[1, 2\] engine"):
            assert_engine_count(side="target", dump_dir=str(tmp_path / "target"), expected=2)

    def test_a_startup_sync_that_reached_one_engine_of_two_is_caught(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] The startup sync reached one engine while rollout 0 reached two; asserts
        # the same kind of assertion error as above, so the startup sync is counted.
        """The engines are synced before the first rollout, and one that joined late missed those weights."""
        _write_inference_engine_events(
            tmp_path / "target",
            [
                _partial(rollout_id=-1, engine_checksums=[{"rank0/w": "aaa"}]),
                _partial(rollout_id=0, engine_checksums=[{"rank0/w": "bbb"}, {"rank0/w": "bbb"}]),
            ],
        )

        with pytest.raises(AssertionError, match=r"pushed to \[1, 2\] engine"):
            assert_engine_count(side="target", dump_dir=str(tmp_path / "target"), expected=2)

    def test_a_side_that_pushed_weights_to_no_engine_at_all_is_caught(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Only creates an empty events directory; asserts an assertion error
        # saying no engine ever took weights.
        """A side whose engines never registered would otherwise report every count it was asked for."""
        (tmp_path / "target" / EVENTS_DIRNAME).mkdir(parents=True)

        with pytest.raises(AssertionError, match="no engine ever took weights"):
            assert_engine_count(side="target", dump_dir=str(tmp_path / "target"), expected=2)
