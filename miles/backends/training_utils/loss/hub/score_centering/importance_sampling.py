# [MODULE DOC]
# Importance-sampling schemes for score centering: three small immutable strategy objects (none,
# truncated, masked) sharing the same three methods, a factory that validates parameters and picks
# one by mode name, and a public helper returning token weights from log ratios. All math is done in
# log space for the ratio so unbounded ratios are never exponentiated.
"""Importance sampling for score centering (arXiv:2609.20807, Appendix A.2/A.3).

Each scheme applies an importance-sampling weight function f to the ratio
r = p / q. It supplies the terms of Eq. 12, whose head coefficient is
q * f(p / q) - alpha * p with tail scale alpha = rho * f(1 / rho), and the
sampled-token weight of the loss:

- ``head_mass(p, q, log_ratio)``: q * f(p / q) on the head candidates;
- ``tail_scale(rho, dtype)``: alpha = rho * f(1 / rho), where rho is the
  sampler-to-trainer tail-mass ratio;
- ``sample_weight(log_ratio)``: f(r) for the sampled token.

No term exponentiates an unbounded ratio.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NoImportanceSampling:
    # [CLASS DOC] Weight function is constant one (no reweighting). Stateless.
    """Vanilla score centering: f(r) = 1."""

    def head_mass(self, p: torch.Tensor, q: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Returns the sampler head probabilities q unchanged (ignores p and the log
        # ratio).
        return q

    def tail_scale(self, rho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # [FUNCTION DOC] Returns the tail mass ratio rho unchanged (dtype ignored).
        return rho

    def sample_weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Returns a tensor of ones shaped like the log ratio.
        return torch.ones_like(log_ratio)


@dataclass(frozen=True)
class TruncatedImportanceSampling:
    # [CLASS DOC] Weight function is the ratio truncated above at a clip constant (the clip field).
    """Truncated importance sampling (TIS): f(r) = min(r, clip)."""

    clip: float

    def head_mass(self, p: torch.Tensor, q: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Elementwise minimum of the trainer probs p and clip times sampler probs q,
        # equal to q times the truncated ratio but safe when q is zero. Log ratio is ignored.
        # q * min(p/q, c), including q=0, without 0 * inf.
        return torch.minimum(p, self.clip * q)

    def tail_scale(self, rho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # [FUNCTION DOC] Returns clip times rho, capped above at 1 (dtype ignored).
        return (self.clip * rho).clamp_max(1)

    def sample_weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Raises a value error if the log ratio contains not-a-number; otherwise
        # returns exp of the log ratio capped above at log of clip.
        if torch.isnan(log_ratio).any():
            raise ValueError("Score-centering importance log-ratio contains NaN")
        return log_ratio.clamp(max=math.log(self.clip)).exp()


@dataclass(frozen=True)
class MaskedImportanceSampling:
    # [CLASS DOC] Weight function keeps the ratio when it lies within [low, high] (fields) and is
    # zero outside.
    """Masked importance sampling (MIS): f(r) = r for low <= r <= high, else 0."""

    low: float
    high: float

    def _inside(self, log_ratio: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Boolean tensor, true where the log ratio lies between log of low
        # and log of high inclusive.
        """Whether low <= r <= high, tested on log(r)."""
        return (log_ratio >= math.log(self.low)) & (log_ratio <= math.log(self.high))

    def head_mass(self, p: torch.Tensor, q: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Trainer probs p where the log ratio is inside the bounds, else zero (q
        # ignored).
        return torch.where(self._inside(log_ratio), p, 0.0)

    def tail_scale(self, rho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # [FUNCTION DOC] Indicator (cast to the given dtype) that the tail ratio rho lies between
        # one over high and one over low inclusive.
        return ((rho >= 1 / self.high) & (rho <= 1 / self.low)).to(dtype)

    def sample_weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Raises a value error on not-a-number input; otherwise exp of the log ratio
        # (capped above at log of high) where inside the bounds, else zero.
        if torch.isnan(log_ratio).any():
            raise ValueError("Score-centering importance log-ratio contains NaN")
        return torch.where(self._inside(log_ratio), log_ratio.clamp(max=math.log(self.high)).exp(), 0.0)


def importance_sampling(
    mode: str, *, tis_clip: float, mis_low: float, mis_high: float
) -> NoImportanceSampling | TruncatedImportanceSampling | MaskedImportanceSampling:
    # [FUNCTION DOC] Factory. Mode "none" returns the no-op scheme, "tis" the truncated scheme (clip
    # must be finite and positive), "mis" the masked scheme (bounds finite with 0 < low <= high).
    # Only the parameters of the chosen mode are validated. Raises a value error for invalid
    # parameters or an unknown mode.
    """Validate the parameters of ``mode`` and return its importance sampling."""
    if mode == "tis" and (not math.isfinite(tis_clip) or tis_clip <= 0):
        raise ValueError("Score-centering TIS clip must be positive and finite")
    if mode == "mis" and (not math.isfinite(mis_low) or not math.isfinite(mis_high) or not 0 < mis_low <= mis_high):
        raise ValueError("Score-centering MIS bounds must be finite with 0 < low <= high")
    if mode == "none":
        return NoImportanceSampling()
    if mode == "tis":
        return TruncatedImportanceSampling(clip=tis_clip)
    if mode == "mis":
        return MaskedImportanceSampling(low=mis_low, high=mis_high)
    raise ValueError(f"Unknown score-centering importance weighting: {mode}")


def importance_weights(
    log_ratio: torch.Tensor,
    mode: str,
    *,
    tis_clip: float = 2.0,
    mis_low: float = 0.5,
    mis_high: float = 5.0,
) -> torch.Tensor:
    # [FUNCTION DOC] Public helper: build the scheme for the mode (defaults clip 2.0, bounds 0.5 and
    # 5.0) and return its per-token sample weights for the given log ratio tensor. Same errors as
    # the factory and the scheme.
    """Evaluate token-level weights without exponentiating unbounded ratios."""
    sampling = importance_sampling(mode, tis_clip=tis_clip, mis_low=mis_low, mis_high=mis_high)
    return sampling.sample_weight(log_ratio)
