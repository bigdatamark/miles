# [MODULE DOC]
# Score-centering estimator: a per-token policy-gradient loss for a policy trained on a truncated
# candidate set ("head": top candidate tokens) where the remaining probability mass is the "tail".
# It combines an importance-weighted sampled-token term with a head correction term built from the
# difference between trainer and sampler head distributions. Sampler quantities are constants:
# weights and head residual are computed without gradient.
"""Score centering from arXiv:2609.20807, Appendix A.

The sampler distribution is fixed data. Both the importance weights and the
head residual must be detached: differentiating either changes the estimator.
"""

from dataclasses import dataclass, fields

import torch

from miles.backends.training_utils.loss.hub.score_centering.importance_sampling import importance_sampling
from miles.backends.training_utils.loss.hub.score_centering.masks import (
    drop_inactive_nan,
    head_probs,
    sanitize_head_log_probs,
    scored_log_probs,
)


@dataclass(frozen=True)
class ScoreCenteringInputs:
    # [CLASS DOC] Immutable bundle of tensors and settings for the loss. Per-token 1-D tensors:
    # trainer log prob of the sampled token, sampler (rollout) log prob of it, and advantage. Head
    # tensors of shape [tokens, candidates]: trainer and sampler log probs of the head candidates
    # and a boolean mask of valid candidate slots. Settings: importance weighting mode (none, tis
    # or mis; default none), truncation clip for tis (default 2.0), lower and upper bounds for mis
    # (defaults 0.5 and 5.0), and a small epsilon (default 1e-6) flooring tail masses. Construction
    # validates shapes and raises a value error on any mismatch.
    """Sample tensors are [tokens]; head tensors and mask are [tokens, candidates]."""

    train_log_probs: torch.Tensor
    train_head_log_probs: torch.Tensor
    rollout_log_probs: torch.Tensor
    rollout_head_log_probs: torch.Tensor
    head_mask: torch.Tensor
    advantages: torch.Tensor
    mode: str = "none"
    tis_clip: float = 2.0
    mis_low: float = 0.5
    mis_high: float = 5.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        # [FUNCTION DOC] HELPER: Validate shapes: the three sample tensors must be 1-D with equal
        # shape; the trainer head log probs must be 2-D with the same number of tokens; sampler head
        # log probs and the head mask must match the trainer head shape. Raises a value error.
        if (
            self.train_log_probs.ndim != 1
            or self.rollout_log_probs.shape != self.train_log_probs.shape
            or self.advantages.shape != self.train_log_probs.shape
            or self.train_head_log_probs.ndim != 2
            or self.train_head_log_probs.shape[0] != self.train_log_probs.shape[0]
            or self.rollout_head_log_probs.shape != self.train_head_log_probs.shape
            or self.head_mask.shape != self.train_head_log_probs.shape
        ):
            raise ValueError("Score-centering sample tensors must be [tokens] and head tensors [tokens, candidates]")


@dataclass(frozen=True)
class ScoreCenteringMetrics:
    # [CLASS DOC] Immutable bundle of per-token diagnostic tensors (all [tokens]): head correction
    # value, trainer head probability mass, sampler head probability mass, tail mass ratio and the
    # sampled-token importance weight.
    """Detached per-token diagnostics before the training loss reducer."""

    correction: torch.Tensor
    train_head_mass: torch.Tensor
    rollout_head_mass: torch.Tensor
    tail_ratio: torch.Tensor
    importance_weight: torch.Tensor

    def as_log_dict(self) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC] Return a mapping from "sc_" plus each field name to the field tensor (same
        # storage, no copy).
        """Map existing log names to the tensors without copying their storage."""
        return {f"sc_{field.name}": getattr(self, field.name) for field in fields(self)}


def score_centering_loss(inputs: ScoreCenteringInputs) -> tuple[torch.Tensor, ScoreCenteringMetrics]:
    # [FUNCTION DOC] Compute the per-token loss and metrics. Steps: build the importance sampling
    # scheme for the mode (validating parameters); mark tokens active where the detached advantage is
    # nonzero; sanitize the head log probs of trainer and sampler (inactive tokens get replaced so no
    # nan leaks). Without gradient: head probabilities p (trainer) and q (sampler) with masses p_mass
    # and q_mass; tail ratio rho = max(1 - q_mass, eps) / max(1 - p_mass, eps); tail scale alpha from
    # the scheme; residual = scheme head mass (from p, q and the log ratio of trainer minus sampler
    # head) minus alpha * p; sampled-token weight from the scheme applied to the log ratio of trainer
    # minus sampler log prob of the sampled token (inactive tokens' nan replaced by 0). With gradient:
    # correction = sum over candidates of residual times the scored (trainer) head log probs; loss =
    # -detached advantage * (weight * scored sampled log prob - correction). Returns (loss [tokens],
    # metrics with detached correction and the no-grad quantities). Raises value errors from the
    # scheme for invalid parameters or nan log ratios.
    """Return unreduced token losses and detached token metrics.

    Inputs have shape [tokens], except head tensors/mask [tokens, candidates].
    Missing head slots have a false mask. Tail masses use the appendix's floor;
    cancellation is exact for a full distribution, approximate for a true tail
    that differs from the modeled, rescaled trainer tail.
    """
    sampling = importance_sampling(
        inputs.mode, tis_clip=inputs.tis_clip, mis_low=inputs.mis_low, mis_high=inputs.mis_high
    )
    active = inputs.advantages.detach() != 0
    train_head = sanitize_head_log_probs(inputs.train_head_log_probs, inputs.head_mask, active)
    rollout_head = sanitize_head_log_probs(inputs.rollout_head_log_probs, inputs.head_mask, active)
    with torch.no_grad():
        p, q = head_probs(train_head, inputs.head_mask), head_probs(rollout_head, inputs.head_mask)
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(inputs.eps) / (1 - p_mass).clamp_min(inputs.eps)
        alpha = sampling.tail_scale(rho, p.dtype)
        residual = sampling.head_mass(p, q, train_head - rollout_head) - alpha.unsqueeze(-1) * p
        weight = sampling.sample_weight(
            drop_inactive_nan(inputs.train_log_probs - inputs.rollout_log_probs, active, 0.0)
        )
    correction = (residual * scored_log_probs(train_head, residual)).sum(-1)
    loss = -inputs.advantages.detach() * (weight * scored_log_probs(inputs.train_log_probs, weight) - correction)
    return loss, ScoreCenteringMetrics(
        correction=correction.detach(),
        train_head_mass=p_mass,
        rollout_head_mass=q_mass,
        tail_ratio=rho,
        importance_weight=weight,
    )
