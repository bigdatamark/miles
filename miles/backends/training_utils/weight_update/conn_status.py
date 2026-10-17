# [MODULE DOC]
# Tracks whether the trainer's connections to the rollout engines must be (re)established before the
# next weight update.
from copy import deepcopy


class ConnStatusManager:
    # [CLASS DOC] Small state holder: whether a connection was ever established, whether the trainer
    # side was marked stale (e.g. after a trainer restart), and a copy of the last seen mapping from
    # rollout cell id to snapshot hash string. Not thread-safe; no I/O.
    def __init__(self) -> None:
        # [FUNCTION DOC] Start uninitialized, not stale, with an empty snapshot mapping.
        self._initialized: bool = False
        self._trainer_stale: bool = False
        self._rollout_snapshot_cell_id_to_hashes: dict[str, str] = {}

    def needs_reconnect(self, new_rollout_snapshot_cell_id_to_hashes: dict[str, str]) -> bool:
        # [FUNCTION DOC] True if never connected, or the trainer was marked stale, or the given
        # mapping of cell id to snapshot hash differs from the one recorded at the last reconnect
        # (any added, removed or changed cell). Pure query.
        return (
            (not self._initialized)
            or self._trainer_stale
            or (new_rollout_snapshot_cell_id_to_hashes != self._rollout_snapshot_cell_id_to_hashes)
        )

    def mark_trainer_stale(self) -> None:
        # [FUNCTION DOC] Flag the trainer side as stale so the next check demands a reconnect.
        self._trainer_stale = True

    def mark_reconnected(self, new_rollout_snapshot_cell_id_to_hashes: dict[str, str]) -> None:
        # [FUNCTION DOC] Record a successful reconnect: mark initialized, clear the stale flag and
        # store a deep copy of the given cell-id-to-hash mapping for later comparison.
        self._initialized = True
        self._trainer_stale = False
        self._rollout_snapshot_cell_id_to_hashes = deepcopy(new_rollout_snapshot_cell_id_to_hashes)
