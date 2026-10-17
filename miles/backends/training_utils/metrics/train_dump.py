# [MODULE DOC]
# Debug dump of the training data seen by each rank for a rollout, written with torch serialization
# to a user-configured path template. Used for offline replay/tests.
import logging
from pathlib import Path

import torch

from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)


def save_debug_train_data(args, *, rollout_id, rollout_data):
    # [FUNCTION DOC] If the save-debug-train-data option is set, and this process is tensor-parallel
    # rank 0 on the last pipeline stage (TP peers duplicate data; only the last stage has per-token
    # fields), save this rank's rollout data through the per-rank function using the global
    # distributed rank and the context-parallel rank/size. Otherwise does nothing.
    if args.save_debug_train_data is not None:
        parallel_state = get_parallel_state()
        # TP peers duplicate the DP shard and only the last PP stage computes the per-token fields
        if parallel_state.tp.rank != 0 or not parallel_state.is_pp_last_stage:
            return
        save_debug_train_data_for_rank(
            args,
            rollout_id=rollout_id,
            rollout_data=rollout_data,
            rank=torch.distributed.get_rank(),
            cp_rank=parallel_state.cp.rank,
            cp_size=parallel_state.cp.size,
        )


def save_debug_train_data_for_rank(args, *, rollout_id, rollout_data, rank, cp_rank=0, cp_size=1):
    # [FUNCTION DOC] If the path template option is set: format it with the rollout id and rank,
    # log it, create parent directories, and save a mapping (rollout id, rank, rollout data, context
    # parallel rank and size, qkv format) via torch serialization. Rank info is passed explicitly so
    # it works outside process groups (tests, offline tools).
    """Write one rank's slice of a rollout.

    ``cp_rank`` / ``cp_size`` are passed in for the same reason ``rank`` is:
    this runs outside the process group in tests and offline tools.
    """
    if (path_template := args.save_debug_train_data) is not None:
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                rollout_data=rollout_data,
                cp_rank=cp_rank,
                cp_size=cp_size,
                qkv_format=args.qkv_format,
            ),
            path,
        )
