# [MODULE DOC]
# Persistent record of where each policy stood when a multi-policy checkpoint was written, stored as
# json files named by the leader policy's rollout id under a subdirectory of the checkpoint
# directory.
import logging
from pathlib import Path

from miles.utils.file_utils import atomic_write_text
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)

MULTI_POLICY_STATE_DIRNAME = "multi_policy_state"


class MultiPolicyCheckpointState(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable record: the leader model id and a mapping from every model id to its
    # rollout id at checkpoint time.
    leader_model_id: str
    rollout_ids: dict[str, int]

    def save(self, save_dir: Path) -> None:
        # [FUNCTION DOC] Writes the record as json (atomically, creating directories) to the file
        # named after the leader's rollout id; raises a key error if the leader is not in the
        # mapping. Logs the path.
        path = _state_path(save_dir, leader_rollout_id=self.rollout_ids[self.leader_model_id])
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, self.model_dump_json())
        logger.info(f"Saved multi policy checkpoint state {self.rollout_ids} to {path}")

    @classmethod
    def load(cls, save_dir: Path, *, leader_rollout_id: int) -> "MultiPolicyCheckpointState | None":
        # [FUNCTION DOC] Reads the record for the leader rollout id; none if the file does not exist.
        # Parses it (validation errors propagate) and asserts that the recorded leader rollout id
        # equals the requested one (guards against a misplaced or renamed file).
        path = _state_path(save_dir, leader_rollout_id=leader_rollout_id)
        if not path.exists():
            return None
        ans = cls.model_validate_json(path.read_text(encoding="utf-8"))
        recorded = ans.rollout_ids[ans.leader_model_id]
        assert recorded == leader_rollout_id, (
            f"{path} records its leader policy {ans.leader_model_id!r} at rollout {recorded}, but it is the "
            f"record of rollout {leader_rollout_id}"
        )
        return ans


def _state_path(save_dir: Path, *, leader_rollout_id: int) -> Path:
    # [FUNCTION DOC] HELPER: Path of the record file: save dir, state subdirectory, then the leader
    # rollout id plus the json extension.
    return Path(save_dir) / MULTI_POLICY_STATE_DIRNAME / f"{leader_rollout_id}.json"
