# [MODULE DOC] Reads Megatron-style checkpoint tracker files: a small text file in a checkpoint
# root directory (named by the constant below) that holds the latest saved iteration number.
from __future__ import annotations

from pathlib import Path

CHECKPOINT_TRACKER_FILENAME = "latest_checkpointed_iteration.txt"


def read_checkpoint_tracker_iteration(checkpoint_root: str | Path | None) -> int | None:
    # [FUNCTION DOC] Returns the iteration number stored in the tracker file under the checkpoint
    # root, or none if the root is none, the file does not exist, or its stripped content is not made
    # only of decimal digits (so values such as "release" or negative numbers give none). Reads the
    # file as text; no other side effects.
    if checkpoint_root is None:
        return None

    tracker = Path(checkpoint_root) / CHECKPOINT_TRACKER_FILENAME
    if not tracker.is_file():
        return None

    content = tracker.read_text().strip()
    if not content.isdigit():
        return None
    return int(content)
