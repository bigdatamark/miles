# [MODULE DOC] Keeps the audit event directory in step with model checkpoints. A snapshot copies
# the event directory (args.save_debug_event_data) into the checkpoint tree at
# <save root>/iter_<7-digit zero-padded iteration>/debug_events; restore copies it back when
# loading; discard moves a taken-over run's log aside. Moved-aside directories are siblings named
# ".trash_<timestamp>_<8 hex chars>" and are never deleted. All three entry points read
# attributes of the parsed args and do filesystem I/O.

"""Snapshot/restore the event directory alongside model checkpoints."""

import logging
import shutil
import time
import uuid
from argparse import Namespace
from pathlib import Path

from miles.backends.megatron_utils.megatron_config import compute_trainer_checkpoint_dir, resolve_megatron_config
from miles.backends.training_utils.checkpoint.tracker import read_checkpoint_tracker_iteration

logger = logging.getLogger(__name__)


def snapshot(args: Namespace, iteration: int) -> None:
    # [FUNCTION DOC] Copies the event directory into the checkpoint tree for the given iteration.
    # No-op if the event directory or the checkpoint save root is not configured, or if the event
    # directory does not exist. An existing snapshot for the same iteration is deleted first, then
    # parent directories are created and the whole tree is copied. Logs at info level.
    if args.save_debug_event_data is None or args.save is None:
        return

    src = Path(args.save_debug_event_data)
    if not src.is_dir():
        return

    dst = _snapshot_dir(Path(args.save), iteration)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    logger.info("Snapshotted event dir %s -> %s", src, dst)


def restore(args: Namespace) -> None:
    # [FUNCTION DOC] Restores the event directory from the snapshot matching the checkpoint being
    # loaded. No-op if the event directory or the requested-load path is not configured, if the
    # checkpoint tracker yields no iteration, or if no snapshot directory exists for that
    # iteration under the requested-load path. If the event directory already exists it is first
    # renamed aside (trash sibling, logged); then the snapshot is copied into place.
    if args.save_debug_event_data is None or args.requested_load is None:
        return

    requested_load = Path(args.requested_load)
    iteration = _read_checkpoint_iteration(args)
    if iteration is None:
        return

    src = _snapshot_dir(requested_load, iteration)
    if not src.is_dir():
        return

    dst = Path(args.save_debug_event_data)
    if dst.exists():
        trash = _move_aside(dst)
        logger.info("Moved pre-restore event dir %s -> %s", dst, trash)
    shutil.copytree(src, dst)
    logger.info("Restored event dir %s <- %s", dst, src)


def discard(args: Namespace) -> None:
    # [FUNCTION DOC] Used when a hot restart takes over a run: moves the existing event log aside
    # so the new incarnation starts clean. No-op if no event directory is configured, or if it is
    # missing, not a directory or empty. Asserts that the path is not a symbolic link. Otherwise
    # renames it to a trash sibling, recreates an empty directory at the original path and logs.
    if args.save_debug_event_data is None:
        return

    dst = Path(args.save_debug_event_data)
    assert not dst.is_symlink(), f"a take-over moves the event log aside, and {dst} is a symlink"
    if not dst.is_dir() or not any(dst.iterdir()):
        return

    # TODO: startup events the new incarnation already wrote go into the trash with the abandoned log
    trash = _move_aside(dst)
    dst.mkdir(parents=True)
    logger.info("Moved the log of the run a hot restart takes over %s -> %s", dst, trash)


def _read_checkpoint_iteration(args: Namespace) -> int | None:
    # [FUNCTION DOC] HELPER: Finds the iteration recorded by the checkpoint tracker in the load
    # directory. Resolves the Megatron config from the args and takes its first trainer as leader.
    # If the leader has a model id, the load directory is the per-trainer checkpoint directory
    # under the requested-load path; otherwise the requested-load path itself. Returns the
    # tracker's iteration (None when absent).
    leader = resolve_megatron_config(args).trainers[0]
    load_dir = (
        compute_trainer_checkpoint_dir(base_dir=args.requested_load, trainer_id=leader.trainer_id)
        if leader.model_id is not None
        else args.requested_load
    )
    return read_checkpoint_tracker_iteration(load_dir)


def _move_aside(dst: Path) -> Path:
    # [FUNCTION DOC] HELPER: Renames the directory to a sibling named with a ".trash_" prefix, the
    # local time as year-month-day_hour-minute-second and 8 random hex characters. Returns the new
    # path. Raises an OS error if the rename fails.
    trash = dst.parent / f".trash_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    dst.rename(trash)
    return trash


def _snapshot_dir(checkpoint_root: Path, iteration: int) -> Path:
    # [FUNCTION DOC] HELPER: Returns root/iter_<iteration padded to 7 digits>/debug_events. Pure.
    return checkpoint_root / f"iter_{iteration:07d}" / "debug_events"
