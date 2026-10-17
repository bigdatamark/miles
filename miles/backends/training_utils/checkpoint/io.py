# [MODULE DOC] Writes a checkpoint directory in a distributed-safe way. Rank zero prepares an empty
# directory (optionally replacing an existing one), all ranks write their shards, failures on any
# rank are collected by an all-gather over the CPU (gloo) group, and on success rank zero writes
# optional metadata and an optional empty completion marker file; on any failure rank zero deletes
# the whole directory so no half-written checkpoint remains. Works without distributed too.
"""Checkpoint directory writes with completion metadata and failure cleanup."""

# TODO: isolate checkpoint IO failures in Tinker; they still terminate the trainer cell.

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)


def write_checkpoint_dir(
    path: str | Path,
    write_shards: Callable[[Path], None],
    metadata: dict | None = None,
    *,
    overwrite: bool = True,
    completion_marker: str | None = None,
) -> None:
    # [FUNCTION DOC] Inputs: target directory path; a callback that receives the directory path and
    # writes this rank's shards; optional metadata dictionary; keyword-only overwrite flag (default
    # true) and optional completion marker file name. Returns nothing; raises on any failure.
    # Steps: (1) rank zero (or the only process) checks the directory: if it exists and overwrite is
    # off, records a file-exists error; otherwise removes it recursively; then creates it with
    # parents; any exception is captured. The captured error is broadcast to all ranks over the gloo
    # group and every rank re-raises it, so nobody proceeds into the write phase. (2) Every rank runs
    # the write callback and captures any exception (not raising yet, so collectives inside the
    # callback finish on all ranks). (3) When distributed, all ranks all-gather a text description of
    # their write error (type name and message) or none; this also acts as a barrier. (4) A rank with
    # its own write error re-raises it; otherwise if any rank reported an error, raises a runtime
    # error listing "rank N: error" entries; otherwise rank zero writes the metadata as indented JSON
    # to META.json (if metadata is given) and creates the empty completion marker (if named).
    # (5) On any failure in step 4, rank zero removes the directory (logging, not raising, if the
    # cleanup itself fails with an operating-system error) and the original error propagates.
    # Callers must make sure no readers use the directory concurrently.
    """Replace a checkpoint; callers must exclude concurrent readers.

    All ranks must call and finish weight collectives before raising local write errors.
    """
    checkpoint_dir = Path(path)
    distributed = dist.is_initialized()
    is_rank0 = not distributed or dist.get_rank() == 0
    prepare_error = [None]
    if is_rank0:
        try:
            if checkpoint_dir.exists():
                if not overwrite:
                    raise FileExistsError(f"checkpoint {checkpoint_dir} already exists")
                shutil.rmtree(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True)
        except Exception as exc:
            prepare_error[0] = exc
    if distributed:
        dist.broadcast_object_list(prepare_error, src=0, group=get_gloo_group())
    if prepare_error[0] is not None:
        raise prepare_error[0]

    write_error = None
    try:
        write_shards(checkpoint_dir)
    except Exception as exc:
        write_error = exc
    errors = []
    if distributed:
        # This also waits for every writer; a failed collective must not trigger directory cleanup.
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(
            errors,
            f"{type(write_error).__name__}: {write_error}" if write_error is not None else None,
            group=get_gloo_group(),
        )
    try:
        if write_error is not None:
            raise write_error
        if any(errors):
            raise RuntimeError(
                "Checkpoint write failed: "
                + "; ".join(f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None)
            )
        if is_rank0:
            if metadata is not None:
                (checkpoint_dir / "META.json").write_text(json.dumps(metadata, indent=2))
            if completion_marker is not None:
                (checkpoint_dir / completion_marker).touch()
    except Exception:
        if is_rank0:
            try:
                shutil.rmtree(checkpoint_dir)
            except OSError:
                logger.exception(f"Failed to clean up checkpoint {checkpoint_dir}")
        raise
