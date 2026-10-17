# [MODULE DOC] Checkpoint save and load for the FSDP backend using torch distributed checkpoint
# (every rank writes/reads its shard). Layout under the save root: a tracker file holding the
# latest step number, and per-step folders named iter plus a seven-digit zero-padded step number,
# each containing separate model, optimizer and lr_scheduler distributed-checkpoint folders, an
# rng file (rank 0's torch and CUDA random states) and a metadata JSON file. The step number is
# the finished rollout id plus one. Three small state-holder classes adapt model, optimizer and
# scheduler to the distributed checkpoint protocol; the actor object is used duck-typed (its args,
# model, optimizer, scheduler and step counters). Loading is split in two: load restores tensors
# and returns a payload; finalize_load (called after the rest of init) applies random state and
# counters.
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from miles.backends.training_utils.checkpoint.tracker import CHECKPOINT_TRACKER_FILENAME

logger = logging.getLogger(__name__)


class ModelState(Stateful):
    # [CLASS DOC] Adapter exposing only a model's weights to the distributed checkpoint protocol.
    """Wrapper for model state only."""

    def __init__(self, model):
        # [FUNCTION DOC] Keep a reference to the model.
        self.model = model

    def state_dict(self):
        # [FUNCTION DOC] Return a mapping with one key holding the model state dict obtained from
        # the distributed state dict API (sharded values), with no optimizers involved.
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        # [FUNCTION DOC] Load the model entry of the mapping into the model in place via the
        # distributed state dict API (no optimizer state).
        set_state_dict(self.model, optimizers=[], model_state_dict=state_dict["model"], optim_state_dict=None)


class OptimizerState(Stateful):
    # [CLASS DOC] Adapter exposing only an optimizer's state (keyed consistently with the model's
    # parameter names) to the distributed checkpoint protocol.
    """Wrapper for optimizer state only."""

    def __init__(self, model, optimizer):
        # [FUNCTION DOC] Keep references to the model (needed to map parameter names) and optimizer.
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # [FUNCTION DOC] Return a mapping with one key holding the optimizer state dict from the
        # distributed state dict API, ignoring the model part.
        _, optimizer_state_dict = get_state_dict(self.model, optimizers=self.optimizer)
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        # [FUNCTION DOC] Load the optimizer entry of the mapping into the optimizer in place (no
        # model weights).
        set_state_dict(
            self.model, optimizers=self.optimizer, model_state_dict=None, optim_state_dict=state_dict["optim"]
        )


class LRSchedulerState(Stateful):
    # [CLASS DOC] Adapter exposing a learning rate scheduler's state to the distributed
    # checkpoint protocol.
    """Wrapper for LR scheduler state only."""

    def __init__(self, lr_scheduler):
        # [FUNCTION DOC] Keep a reference to the scheduler.
        self.lr_scheduler = lr_scheduler

    def state_dict(self):
        # [FUNCTION DOC] Return a mapping with one key holding the scheduler's own state dict.
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        # [FUNCTION DOC] Restore the scheduler from the mapping's scheduler entry.
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Read a JSON metadata file into a mapping; returns an empty mapping if
    # the file is missing or is not valid JSON (the latter logs a warning).
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse checkpoint metadata at {path}")
        return {}


def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    # [FUNCTION DOC] HELPER: Atomically write the mapping as indented, key-sorted JSON: write to a
    # sibling temporary file (original name plus a temp suffix) and then rename it over the target.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    tmp_path.replace(path)


def load(actor: Any) -> dict[str, Any] | None:
    # [FUNCTION DOC] Restore weights (and, when available, optimizer and scheduler state) in place
    # on the actor, then return a payload for the finalize step, or none if nothing was loaded.
    # Steps: the load root comes from the args' load option (none if unset; a missing directory is
    # skipped with a log). The step is the args' checkpoint step option if set, else the integer in
    # the root's tracker file (skip if missing). The step folder is iter plus the zero-padded
    # seven-digit step; if its model folder is missing, skip. Model weights are loaded with the
    # distributed checkpoint API; ANY exception there is logged as an error and the function
    # returns none. Optimizer state is loaded unless the no-load-optim arg is set or the actor has no
    # optimizer, and only if its folder exists (failures are only warnings). Scheduler state is
    # loaded if the actor has a scheduler and its folder exists (failures warn). Random state is read
    # to CPU from the rng file if present. Metadata comes from the metadata JSON (empty if missing
    # or invalid). Returns a mapping with the random state (or none), metadata and the step number
    # under the iteration name.
    # KNOWN ISSUE: a failure to load model weights is swallowed (logged only), so training would
    # continue silently from the freshly initialized weights instead of failing the run.
    """Load checkpoint from disk.

    Loads model weights and optionally optimizer state from separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    load_root = getattr(actor.args, "load", None)
    if load_root is None:
        return None

    root_path = Path(load_root).expanduser()
    if not root_path.exists():
        logger.info(f"[FSDP] Checkpoint directory {root_path} not found; skipping load.")
        return None

    target_step = getattr(actor.args, "ckpt_step", None)
    if target_step is None:
        tracker_file = root_path / CHECKPOINT_TRACKER_FILENAME
        if not tracker_file.exists():
            logger.info(f"[FSDP] No tracker file at {tracker_file}; skipping load.")
            return None
        tracker_text = tracker_file.read_text().strip()
        target_step = int(tracker_text)

    checkpoint_dir = root_path / f"iter_{target_step:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if not model_dir.exists():
        logger.info(f"[FSDP] Model checkpoint {model_dir} not found; skipping load.")
        return None

    # Load model weights (always)
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}

    try:
        dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir))
        logger.info(f"[FSDP] Loaded model from {model_dir}")
    except Exception as e:
        logger.error(f"[FSDP] Failed to load model from {model_dir}: {e}")
        return None

    # Load optimizer state (optional)
    load_optimizer = not getattr(actor.args, "no_load_optim", False) and hasattr(actor, "optimizer")
    if load_optimizer and optimizer_dir.exists():
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        try:
            dcp.load(state_dict=optim_state_dict, checkpoint_id=str(optimizer_dir))
            logger.info(f"[FSDP] Loaded optimizer from {optimizer_dir}")
        except Exception as e:
            logger.warning(f"[FSDP] Failed to load optimizer from {optimizer_dir}: {e}")
    elif load_optimizer:
        logger.info(f"[FSDP] Optimizer checkpoint not found at {optimizer_dir}, skipping optimizer load.")

    # Load LR scheduler state (optional)
    load_lr_scheduler = hasattr(actor, "lr_scheduler") and lr_scheduler_dir.exists()
    if load_lr_scheduler:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        try:
            dcp.load(state_dict=lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))
            logger.info(f"[FSDP] Loaded LR scheduler from {lr_scheduler_dir}")
        except Exception as e:
            logger.warning(f"[FSDP] Failed to load LR scheduler from {lr_scheduler_dir}: {e}")
    elif hasattr(actor, "lr_scheduler"):
        logger.info(f"[FSDP] LR scheduler checkpoint not found at {lr_scheduler_dir}, skipping LR scheduler load.")

    rng_state = None
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")

    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")

    return {
        "rng": rng_state,
        "metadata": metadata,
        "iteration": target_step,
    }


def finalize_load(actor: Any, checkpoint_payload: dict[str, Any] | None) -> None:
    # [FUNCTION DOC] Apply the non-tensor part of a loaded checkpoint, then synchronize. With no
    # payload, just a barrier across ranks. Otherwise: unless the no-load-rng arg is set, restore
    # the CPU random state and, if CUDA is available, the random state of all CUDA devices (only the
    # keys present). If metadata exists: restore the global and micro step counters (keeping
    # current values if absent) and set the args' start rollout id to the recorded next rollout id
    # when present. If there is no metadata but a step number exists: set the start rollout id to
    # the step number only if it is currently unset. Finally synchronize CUDA and barrier.
    # Note: every rank restores the same random state because only rank 0 saved one.
    # KNOWN ISSUE: the actor sets the start rollout id to 0 before loading, so the no-metadata
    # branch (which only fills an unset value) never takes effect and resume without a metadata
    # file restarts from rollout 0.
    if checkpoint_payload is None:
        dist.barrier()
        return

    if checkpoint_payload.get("rng") is not None and not getattr(actor.args, "no_load_rng", False):
        rng_state = checkpoint_payload["rng"]
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    metadata = checkpoint_payload.get("metadata") or {}
    iteration = checkpoint_payload.get("iteration")
    if metadata:
        actor.global_step = int(metadata.get("global_step", actor.global_step))
        actor.micro_step = int(metadata.get("micro_step", actor.micro_step))
        next_rollout = metadata.get("next_rollout_id")
        if next_rollout is not None:
            actor.args.start_rollout_id = next_rollout
    elif iteration is not None:
        if getattr(actor.args, "start_rollout_id", None) is None:
            actor.args.start_rollout_id = iteration

    torch.cuda.synchronize()
    dist.barrier()


def save(actor: Any, iteration: int) -> None:
    # [FUNCTION DOC] Collectively save a checkpoint after the given (zero-based) rollout; all
    # ranks must call it. The step number is iteration plus one and the folder is the save root's
    # iter plus zero-padded step. Steps: synchronize CUDA; rank 0 creates the step, model,
    # optimizer and scheduler folders; barrier; save model weights (distributed save); unless the
    # no-save-optim arg is set, save optimizer state if the actor has a non-none optimizer and
    # scheduler state if it has a non-none scheduler; rank 0 then writes the random states (CPU and
    # all CUDA devices) and an atomically written metadata file (step, rollout id, next rollout id,
    # global and micro step, world size, timestamp) and finally overwrites the tracker file in the
    # save root with the step number (not atomically); final barrier. Logs the location from rank 0.
    """Save checkpoint to disk.

    Saves model weights and optimizer state to separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    torch.cuda.synchronize()

    base_dir = Path(actor.args.save).expanduser()
    step_id = iteration + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if dist.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        optimizer_dir.mkdir(parents=True, exist_ok=True)
        lr_scheduler_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # Save model weights
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}
    dcp.save(state_dict, checkpoint_id=str(model_dir))

    # Save optimizer state (skip if --no-save-optim is set)
    save_optimizer_state = not getattr(actor.args, "no_save_optim", False)
    if save_optimizer_state and hasattr(actor, "optimizer") and actor.optimizer is not None:
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        dcp.save(optim_state_dict, checkpoint_id=str(optimizer_dir))

    # Save LR scheduler state (skip if --no-save-optim is set)
    if save_optimizer_state and hasattr(actor, "lr_scheduler") and actor.lr_scheduler is not None:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        dcp.save(lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))

    if dist.get_rank() == 0:
        rng_state = {"torch": torch.get_rng_state()}
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, checkpoint_dir / "rng.pt")

        metadata = {
            "iteration": step_id,
            "rollout_id": iteration,
            "next_rollout_id": iteration + 1,
            "global_step": actor.global_step,
            "micro_step": actor.micro_step,
            "world_size": dist.get_world_size(),
            "timestamp": time.time(),
        }
        _write_checkpoint_metadata(checkpoint_dir / "meta.json", metadata)

        tracker_file = base_dir / CHECKPOINT_TRACKER_FILENAME
        tracker_file.write_text(str(step_id))
        logger.info(f"[FSDP] Saved checkpoint to {checkpoint_dir}")

    dist.barrier()
