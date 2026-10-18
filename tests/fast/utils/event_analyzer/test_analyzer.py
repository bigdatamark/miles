# [MODULE DOC] Tests for the event analyzer driver: running all registered rules over the event
# log files in a directory (run_analysis), partitioning events by trainer model id so different
# models are never compared against each other, wiring of the inference engine checksum rule, and
# the argument-driven wrapper (run_analysis_from_args) that is skipped when disabled or when no
# event directory is set, logs its duration, and raises a value error when issues are found.
# Events are written with real event loggers into temporary directories, one log file per logger.
"""Tests for event_analyzer/analyzer.py."""

import logging
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.audit_utils.event_analyzer.analyzer import (
    _partition_by_model_id,
    run_analysis,
    run_analysis_from_args,
)
from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import (
    InferenceEngineWeightChecksumEvent,
    TrainEngineLocalWeightChecksumEvent,
    TrainEngineLocalWeightChecksumState,
)
from miles.utils.audit_utils.process_identity import (
    SimpleProcessIdentity,
    TrainerControllerProcessIdentity,
    TrainProcessIdentity,
)


def _log_checksum_event(
    event_logger: EventLogger,
    *,
    rollout_id: int,
    param_hashes: dict[str, str] | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Logs one training-engine local weight checksum event through the
    # given event logger, with the rollout id and a state holding the given parameter hashes
    # (default empty), empty buffer hashes and no optimizer hashes. Side effect: appends to the
    # logger's file.
    event_logger.log(
        TrainEngineLocalWeightChecksumEvent,
        dict(
            rollout_id=rollout_id,
            state=TrainEngineLocalWeightChecksumState(
                param_hashes=param_hashes or {},
                buffer_hashes={},
                optimizer_hashes=[],
            ),
        ),
    )


def _make_source(*, cell_index: int = 0, rank: int = 0) -> TrainProcessIdentity:
    # [FUNCTION DOC] HELPER: Builds a training process identity of component "actor" with the
    # given cell index and rank within cell (both default 0).
    return TrainProcessIdentity(component="actor", cell_index=cell_index, rank_within_cell=rank)


class TestRunAnalysis:
    # [CLASS DOC] Tests for the directory-based analysis function, which returns the list of
    # issues found by all rules.
    def test_empty_directory_returns_no_issues(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A directory with no event files gives an empty list.
        assert run_analysis(event_dir=tmp_path) == []

    def test_delegates_to_rules_and_returns_issues(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two loggers (cell 0 and cell 1, same rank 0, separate files) record
        # different parameter hashes for the same rollout; the analysis returns exactly one
        # issue. Cells must differ because the cross-replica rule compares the same rank across
        # different cells.
        # Cross-replica rule compares same rank across DIFFERENT cells, so use cell_index 0/1.
        logger_a = EventLogger(log_dir=tmp_path, file_name="a.jsonl", source=_make_source(cell_index=0, rank=0))
        _log_checksum_event(logger_a, rollout_id=0, param_hashes={"pp0.w": "aaa"})
        logger_a.close()

        logger_b = EventLogger(log_dir=tmp_path, file_name="b.jsonl", source=_make_source(cell_index=1, rank=0))
        _log_checksum_event(logger_b, rollout_id=0, param_hashes={"pp0.w": "zzz"})
        logger_b.close()

        issues = run_analysis(event_dir=tmp_path)
        assert len(issues) == 1


class TestSeveralModelIds:
    # [CLASS DOC] Tests that events carrying different model ids (multi-model training) are
    # analyzed in separate partitions.
    def test_trainer_controller_events_are_partitioned_by_model_id(self) -> None:
        # [FUNCTION DOC] Three fake events (objects with only a source): a verifier controller,
        # a model-neutral main process, and a solver controller, given in that order. The
        # partitioning returns two lists ordered by model id: first the solver partition (neutral
        # then solver event), then the verifier partition (verifier then neutral event); the
        # neutral event appears in every partition and the original relative order is kept.
        """Controller events stay with their model while model-neutral events accompany every partition."""
        solver_event = SimpleNamespace(
            source=TrainerControllerProcessIdentity(trainer_id="solver-actor", model_id="solver")
        )
        verifier_event = SimpleNamespace(
            source=TrainerControllerProcessIdentity(trainer_id="verifier-actor", model_id="verifier")
        )
        neutral_event = SimpleNamespace(source=SimpleProcessIdentity(component="main"))

        partitions = _partition_by_model_id([verifier_event, neutral_event, solver_event])

        assert partitions == [[neutral_event, solver_event], [verifier_event, neutral_event]]

    def test_two_model_ids_are_not_compared_against_each_other(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] For model ids "solver" (hash aaa) and "verifier" (hash zzz), writes two
        # cells each with identical hashes inside the model; the analysis returns no issues
        # because models are not compared with each other.
        """Two model ids mean two different models, so their weights differ by design, not by fault."""
        for model_id, param_hash in (("solver", "aaa"), ("verifier", "zzz")):
            for cell_index in (0, 1):
                event_logger = EventLogger(
                    log_dir=tmp_path,
                    file_name=f"{model_id}-{cell_index}.jsonl",
                    source=TrainProcessIdentity(
                        component="actor", model_id=model_id, cell_index=cell_index, rank_within_cell=0
                    ),
                )
                _log_checksum_event(event_logger, rollout_id=0, param_hashes={"pp0.w": param_hash})
                event_logger.close()

        assert run_analysis(event_dir=tmp_path) == []

    def test_two_cells_of_one_model_id_are_still_compared(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two "solver" cells with different hashes plus one "verifier" cell:
        # exactly one issue is returned (inside the solver model); the verifier cell is alone in
        # its partition.
        """Partitioning by model id must not disable the check inside one model id, which is what it exists for."""
        for cell_index, param_hash in ((0, "aaa"), (1, "zzz")):
            event_logger = EventLogger(
                log_dir=tmp_path,
                file_name=f"solver-{cell_index}.jsonl",
                source=TrainProcessIdentity(
                    component="actor", model_id="solver", cell_index=cell_index, rank_within_cell=0
                ),
            )
            _log_checksum_event(event_logger, rollout_id=0, param_hashes={"pp0.w": param_hash})
            event_logger.close()

        other = EventLogger(
            log_dir=tmp_path,
            file_name="verifier.jsonl",
            source=TrainProcessIdentity(component="actor", model_id="verifier", cell_index=0, rank_within_cell=0),
        )
        _log_checksum_event(other, rollout_id=0, param_hashes={"pp0.w": "bbb"})
        other.close()

        assert len(run_analysis(event_dir=tmp_path)) == 1


def _log_inference_engine_checksum_event(
    event_logger: EventLogger,
    *,
    rollout_id: int,
    engine_checksums: list[dict[str, str]],
) -> None:
    # [FUNCTION DOC] HELPER: Logs one inference engine weight checksum event through the given
    # logger with the rollout id and the per-engine checksum mappings. Side effect: appends to
    # the logger's file.
    event_logger.log(
        InferenceEngineWeightChecksumEvent,
        dict(rollout_id=rollout_id, engine_checksums=engine_checksums),
    )


class TestInferenceEngineChecksumRuleWiredIn:
    # [CLASS DOC] Tests that the inference engine checksum consistency rule is among the rules run
    # by the analysis function.
    def test_engine_inconsistency_reported(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] One logged event with two engines having different checksums for the
        # same tensor yields exactly one issue.
        """run_analysis surfaces engine-to-engine checksum mismatches via the registered rule."""
        event_logger = EventLogger(
            log_dir=tmp_path, file_name="e.jsonl", source=SimpleProcessIdentity(component="main")
        )
        _log_inference_engine_checksum_event(
            event_logger, rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "zzz"}]
        )
        event_logger.close()

        issues = run_analysis(event_dir=tmp_path)
        assert len(issues) == 1

    def test_consistent_engines_no_issue(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two engines with identical checksums yield no issues.
        """Identical engine checksums produce no issue."""
        event_logger = EventLogger(
            log_dir=tmp_path, file_name="e.jsonl", source=SimpleProcessIdentity(component="main")
        )
        _log_inference_engine_checksum_event(
            event_logger, rollout_id=0, engine_checksums=[{"rank0/w": "aaa"}, {"rank0/w": "aaa"}]
        )
        event_logger.close()

        assert run_analysis(event_dir=tmp_path) == []


class TestRunAnalysisFromArgs:
    # [CLASS DOC] Tests for the wrapper that reads an argument namespace: it does nothing unless
    # the analyzer is enabled and a debug event data directory is set, logs how long the scan
    # took, and raises a value error mentioning "issues" when any issue is found.
    def test_skips_when_disabled(self) -> None:
        # [FUNCTION DOC] With the analyzer disabled the call returns without error even though
        # the directory (/tmp/whatever) is not touched. No explicit assertion beyond not
        # raising.
        args = Namespace(enable_event_analyzer=False, save_debug_event_data="/tmp/whatever")
        run_analysis_from_args(args)

    def test_skips_when_no_event_dir(self) -> None:
        # [FUNCTION DOC] With the analyzer enabled but no event data directory attribute on the
        # namespace at all, the call returns without error. No explicit assertion.
        args = Namespace(enable_event_analyzer=True)
        run_analysis_from_args(args)

    def test_logs_analysis_duration(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        # [FUNCTION DOC] Runs the wrapper on an empty temporary directory with info-level log
        # capture and asserts the log contains "Event analysis of <dir> took ".
        """Enabled analysis reports how long its event-log scan takes."""
        caplog.set_level(logging.INFO)

        run_analysis_from_args(Namespace(enable_event_analyzer=True, save_debug_event_data=str(tmp_path)))

        assert f"Event analysis of {tmp_path} took " in caplog.text

    def test_raises_on_mismatch(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two cells (0 and 1, rank 0) record different parameter hashes; the
        # wrapper raises a value error whose message matches "issues".
        logger_a = EventLogger(log_dir=tmp_path, file_name="a.jsonl", source=_make_source(cell_index=0, rank=0))
        _log_checksum_event(logger_a, rollout_id=0, param_hashes={"pp0.w": "aaa"})
        logger_a.close()

        logger_b = EventLogger(log_dir=tmp_path, file_name="b.jsonl", source=_make_source(cell_index=1, rank=0))
        _log_checksum_event(logger_b, rollout_id=0, param_hashes={"pp0.w": "zzz"})
        logger_b.close()

        args = Namespace(enable_event_analyzer=True, save_debug_event_data=str(tmp_path))
        with pytest.raises(ValueError, match="issues"):
            run_analysis_from_args(args)

    def test_passes_when_all_match(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two loggers record the same hashes for the same rollout and the wrapper
        # returns without raising.
        # KNOWN ISSUE: the two loggers use the same cell (default 0) with ranks 0 and 1, and the
        # cross-replica rule only compares the same rank across different cells, so nothing is
        # actually compared; the test would also pass with different hashes.
        logger_a = EventLogger(log_dir=tmp_path, file_name="a.jsonl", source=_make_source(rank=0))
        _log_checksum_event(logger_a, rollout_id=0, param_hashes={"pp0.w": "aaa"})
        logger_a.close()

        logger_b = EventLogger(log_dir=tmp_path, file_name="b.jsonl", source=_make_source(rank=1))
        _log_checksum_event(logger_b, rollout_id=0, param_hashes={"pp0.w": "aaa"})
        logger_b.close()

        args = Namespace(enable_event_analyzer=True, save_debug_event_data=str(tmp_path))
        run_analysis_from_args(args)
