# [MODULE DOC] Off-policy correction functions for the policy loss, selectable as the importance
# sampling hook: each takes the arguments, the per-token policy gradient loss, the trainer's and the
# rollout engine's per-sample log-probs and the loss masks (plus ignored extras), forms the
# per-token ratio of exponentiated log-prob differences, and returns (corrected per-token loss,
# loss masks, metrics mapping of detached per-token tensors). Hooks with the same signature can be
# supplied from a user module.
from argparse import Namespace
from typing import Any

import torch


def vanilla_tis_function(
    args: Namespace,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    # [FUNCTION DOC] Concatenates rollout and trainer log-probs over samples and forms the ratio as
    # the exponential of trainer minus rollout. Weights are the ratio clamped between the low and
    # high clip bounds from arguments. Metrics (flat per-token, detached): the unclamped ratio, the
    # fraction flag (1.0 where clamping changed the value), and the absolute deviation of the ratio
    # from one. Multiplies the policy loss by the weights and returns it, the loss masks unchanged
    # and the metrics. Extra keyword arguments are ignored.
    """Truncated importance sampling: clamp `exp(train - rollout)` to
    `[tis_clip_low, tis_clip]` and multiply into `pg_loss`. `loss_masks` is
    passed through unchanged; metrics report the pre-clamp ratio.
    """
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args: Namespace,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    # [FUNCTION DOC] Same inputs and outputs as the truncated variant. The ratio is the exponential
    # of trainer minus rollout log-prob; the weight is the ratio itself where it lies within the
    # inclusive range from the low to the high clip bound, and zero elsewhere. The loss is
    # multiplied by that weight (so in-range tokens are scaled by their ratio, not by one). Metrics
    # (detached): the ratio, the flag that is 1.0 where the weight differs from the ratio, and the
    # absolute deviation of the ratio from one. Returns the loss, unchanged masks and metrics.
    # KNOWN ISSUE: the docstring says the in-range ratio passes through "unweighted", but the code
    # multiplies the loss by the ratio.
    """IS clip-or-pop: zero out tokens whose `exp(train - rollout)` is outside
    `[tis_clip_low, tis_clip]` and pass the in-range ratio through unweighted.
    Same return shape as `vanilla_tis_function`.
    """
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics
