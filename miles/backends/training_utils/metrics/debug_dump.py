# [MODULE DOC]
# Debug dump of policy-loss inputs/outputs to disk. When the dump-details directory option is set,
# each call writes one file per call per rank under a policy-loss-debug subdirectory, containing
# per-sample tensors and finiteness flags. Uses a module-global call counter.
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch

_POLICY_LOSS_DUMP_COUNTER = 0


def maybe_dump_policy_loss_debug(
    *,
    args: Namespace,
    batch: RolloutBatch,
    train_log_probs: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor] | None,
    advantages: list[torch.Tensor],
    local_loss_masks: list[torch.Tensor],
    ppo_kl: torch.Tensor,
    pg_loss: torch.Tensor,
) -> None:
    # [FUNCTION DOC] Keyword-only. No-op when the dump-details option is unset or this process is not
    # tensor-parallel rank 0 (TP peers hold identical data; CP ranks hold different slices so they
    # all dump). Otherwise increments the global call counter and creates (parents included) the
    # file path "<dump dir>/policy_loss_debug/rank_<global rank>_call_<counter>.pt" and saves a
    # mapping with: rank, call counter, a list of per-sample mappings (index, total length, response
    # length, and CPU float32 detached train log probs, old log probs, advantages, local loss mask;
    # plus rollout log probs and, when shapes match, the absolute train-minus-rollout difference if
    # rollout log probs are provided), CPU ppo kl and pg loss, and a mapping of booleans telling
    # whether ppo kl, pg loss, and all train log probs, old log probs and advantages are finite.
    # Side effects: directory creation, file write, global counter increment.
    dump_dir = getattr(args, "dump_details", None)
    if dump_dir is None:
        return
    # TP peers compute the loss on identical data; CP ranks hold distinct token slices
    if get_parallel_state().tp.rank != 0:
        return

    global _POLICY_LOSS_DUMP_COUNTER
    counter = _POLICY_LOSS_DUMP_COUNTER
    _POLICY_LOSS_DUMP_COUNTER += 1

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    path = Path(dump_dir) / "policy_loss_debug" / f"rank_{rank}_call_{counter}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Detach, cast to float32 and copy to CPU.
        return tensor.detach().float().cpu()

    samples = []
    for index, train_lp in enumerate(train_log_probs):
        sample = {
            "index": index,
            "total_length": batch["total_lengths"][index],
            "response_length": batch["response_lengths"][index],
            "train_log_probs": to_cpu(train_lp),
            "old_log_probs": to_cpu(old_log_probs[index]),
            "advantages": to_cpu(advantages[index]),
            "local_loss_mask": to_cpu(local_loss_masks[index]),
        }
        if rollout_log_probs is not None:
            sample["rollout_log_probs"] = to_cpu(rollout_log_probs[index])
            if train_lp.shape == rollout_log_probs[index].shape:
                sample["train_rollout_abs_diff"] = to_cpu((train_lp - rollout_log_probs[index]).abs())
        samples.append(sample)

    torch.save(
        {
            "rank": rank,
            "call": counter,
            "samples": samples,
            "ppo_kl": to_cpu(ppo_kl),
            "pg_loss": to_cpu(pg_loss),
            "finite": {
                "ppo_kl": torch.isfinite(ppo_kl).all().item(),
                "pg_loss": torch.isfinite(pg_loss).all().item(),
                "train_log_probs": all(torch.isfinite(t).all().item() for t in train_log_probs),
                "old_log_probs": all(torch.isfinite(t).all().item() for t in old_log_probs),
                "advantages": all(torch.isfinite(t).all().item() for t in advantages),
            },
        },
        path,
    )
