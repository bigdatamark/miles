# [MODULE DOC]
# HELPER: Data model and readers for the evidence of hot restarts. Reads how far a training run
# has come from files: the checkpoint tracker file (a single text file named
# "latest_checkpointed_iteration.txt" anywhere under the checkpoint directory, holding the last
# saved iteration as digits) and the audit event logs (metric events carrying the training-step
# metric key, the gradient norm, identify finished rollout steps). Also defines the immutable
# record of one triggered take-over and the evidence bundle (records, cluster snapshots, release,
# observation attempt and failure counts) with writing to and loading from a JSON file at
# "hot_restart/evidence.json" inside the dump directory.
import logging
from dataclasses import dataclass
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import ClusterSnapshot

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.comparisons.metrics import read_metric_events

logger = logging.getLogger(__name__)

CHECKPOINT_TRACKER_FILENAME: str = "latest_checkpointed_iteration.txt"
TRAIN_STEP_METRIC_KEY: str = "train/grad_norm"


# =========================== how far a run has come ===========================


@dataclass(frozen=True)
class RunProgress:
    # [CLASS DOC] HELPER: Immutable progress of a run: the last saved checkpoint iteration (missing
    # if none) and the highest rollout id with a finished training step in the event log (missing
    # if none).
    last_saved_iteration: int | None
    last_finished_rollout_id: int | None


def read_run_progress(*, checkpoint_dir: Path, events_dir: Path) -> RunProgress:
    # [FUNCTION DOC] Combine the last saved iteration from the checkpoint directory and the last
    # finished rollout id from the events directory into one progress object. Exceptions from the
    # underlying readers propagate.
    return RunProgress(
        last_saved_iteration=read_last_saved_iteration(checkpoint_dir),
        last_finished_rollout_id=read_last_finished_rollout_id(events_dir),
    )


def read_last_saved_iteration(checkpoint_dir: Path) -> int | None:
    # [FUNCTION DOC] Return the last saved iteration, or nothing. If the directory does not exist
    # return nothing. Search it recursively for tracker files; asserts at most one exists (several
    # would mean several policies saving at their own pace). With none, return nothing. Read the
    # tracker's text (missing or unreadable file counts as empty); return its integer value if the
    # text is all digits, else nothing (e.g. still being written).
    if not checkpoint_dir.is_dir():
        return None

    trackers = sorted(checkpoint_dir.glob(f"**/{CHECKPOINT_TRACKER_FILENAME}"))
    assert len(trackers) <= 1, (
        f"{checkpoint_dir} holds the trackers {[str(one) for one in trackers]}: this run trains several policies, "
        f"and one restart window cannot be read off trainers that each save at their own pace"
    )
    if not trackers:
        return None

    text = _read_text_or_empty(trackers[0])
    return int(text) if text.isdigit() else None


def read_last_finished_rollout_id(events_dir: Path) -> int | None:
    # [FUNCTION DOC] Return the highest finished rollout id found in the events directory, or
    # nothing if there are none.
    return max(read_finished_rollout_ids(events_dir), default=None)


def read_finished_rollout_ids(events_dir: Path) -> list[int]:
    # [FUNCTION DOC] Return the sorted distinct rollout ids of metric events in the events
    # directory that have a rollout id and contain the training-step metric key. Returns an empty
    # list if the directory does not exist.
    if not events_dir.is_dir():
        return []

    return sorted(
        {
            event.rollout_id
            for event in read_metric_events(events_dir)
            if event.rollout_id is not None and TRAIN_STEP_METRIC_KEY in event.metrics
        }
    )


def _read_text_or_empty(path: Path) -> str:
    # [FUNCTION DOC] HELPER: Return the stripped text of the file; on an operating-system error log
    # an info message (treated as not yet written) and return an empty string.
    try:
        return path.read_text().strip()
    except OSError:
        logger.info(f"Failed to read {path}; treating it as not written yet", exc_info=True)
        return ""


# ========================== what the take-overs left ==========================


class HotRestartRecord(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: Immutable record of one triggered take-over: its zero-based index, the
    # checkpoint iteration the run held when the take-over was triggered (missing if none), and
    # the rollout id the run was frozen after.
    index: int
    saved_iteration_at_trigger: int | None
    frozen_rollout_id: int


class HotRestartEvidence(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: Immutable bundle of everything a hot-restart test observed: the take-over
    # records, the cluster snapshots, the release name, and counts of observation attempts and of
    # failed observation reads (both default to zero). Serialized to JSON with indentation of 2.
    records: tuple[HotRestartRecord, ...]
    snapshots: tuple[ClusterSnapshot, ...]
    release: str
    observation_attempts: int = 0
    observation_failures: int = 0

    def write(self, *, dump_dir: str) -> None:
        # [FUNCTION DOC] Write the evidence as indented JSON to the evidence path under the dump
        # directory, creating parent directories and overwriting any existing file; logs the
        # location. Side effect: file I/O.
        path = evidence_path(dump_dir=dump_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        logger.info(f"Wrote what {len(self.records)} hot restart(s) left behind to {path}")

    @classmethod
    def load(cls, *, dump_dir: str) -> "HotRestartEvidence":
        # [FUNCTION DOC] Load evidence from the evidence path under the dump directory. Asserts the
        # file exists (message tells to run the target side before comparing), then validates and
        # returns the parsed object; invalid content raises a validation error.
        path = evidence_path(dump_dir=dump_dir)
        assert path.is_file(), (
            f"{path} does not exist, so nothing recorded which steps this run redid and which pods outlived its "
            f"orchestration script; run the target side before comparing"
        )
        return cls.model_validate_json(path.read_text())


def evidence_path(*, dump_dir: str) -> Path:
    # [FUNCTION DOC] Return the path "hot_restart/evidence.json" inside the dump directory. Pure
    # path computation.
    return Path(dump_dir) / "hot_restart" / "evidence.json"
