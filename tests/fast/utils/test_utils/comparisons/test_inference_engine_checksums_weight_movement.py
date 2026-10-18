# [MODULE DOC] Unit tests for the assertion that inference engine weights moved between weight
# updates, based on logged engine weight checksum events. Events are written with the real event
# logger into a temporary directory and the assertion reads them back.
from pathlib import Path

import pytest

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, EventLogger
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.comparisons.inference_engine_checksums import assert_engine_weights_moved


def _write_checksums(side_dir: Path, checksums: list[str]) -> None:
    # [FUNCTION DOC] HELPER: Writes one engine weight checksum event per given checksum string
    # into the events subdirectory of the side directory, as the main component process. Rollout
    # ids are the list positions starting at 0, there is no trainer model id, and each event has a
    # single engine whose one tensor key maps to the checksum. File output only; the logger is
    # closed afterwards and nothing is printed.
    source = SimpleProcessIdentity(component="main")
    event_logger = EventLogger(
        log_dir=side_dir / EVENTS_DIRNAME,
        source=source,
        file_name=f"{source.to_name()}.jsonl",
    )
    for rollout_id, checksum in enumerate(checksums):
        event_logger.log(
            InferenceEngineWeightChecksumEvent,
            dict(rollout_id=rollout_id, trainer_model_id=None, engine_checksums=[{"rank0/w": checksum}]),
            print_log=False,
        )
    event_logger.close()


class TestAssertEngineWeightsMoved:
    # [CLASS DOC] Tests of the refusal cases of the weight movement assertion.
    def test_one_recorded_push_cannot_prove_weights_moved(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With a single recorded update, asserts an assertion error saying there is
        # no pair to compare.
        """A single checksum has no earlier or later update against which movement can be established."""
        _write_checksums(tmp_path, ["aaa"])

        with pytest.raises(AssertionError, match="no pair to compare"):
            assert_engine_weights_moved(side="target", dump_dir=str(tmp_path))

    def test_repeated_identical_pushes_cannot_prove_weights_moved(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With two updates carrying the same checksum, asserts an assertion error
        # saying the optimizer moved nothing.
        """Several byte-identical pushes must not make a vacuous comparison side look trained."""
        _write_checksums(tmp_path, ["aaa", "aaa"])

        with pytest.raises(AssertionError, match="optimizer moved nothing"):
            assert_engine_weights_moved(side="target", dump_dir=str(tmp_path))
