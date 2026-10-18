# [MODULE DOC] Vision patch encoder for Inkling: a hierarchical MLP that progressively folds
# space-time blocks of each raw patch (time, height, width, channels) into the channel dimension and
# projects with linear layers (RMS norm and gelu between them) down to the decoder width, yielding
# one embedding per patch. A planner picks the intermediate fold scales from the prime factors of the
# patch and temporal patch sizes so that the information reduction per layer is as even as possible
# in log scale. Written without serving-side dependencies.
from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class InklingVisionConfig:
    # [CLASS DOC] HELPER: Attribute holder built from a mapping: decoder width, channel count (default
    # 3), layer count, spatial patch size, temporal patch size (default 1) and whether to apply the final
    # norm (default true).
    """Duck-typed stand-in for the serving-side config (fields the encoder reads)."""

    def __init__(self, d: dict):
        # [FUNCTION DOC] HELPER: Copies fields; missing required keys (decoder width, layer count,
        # patch size) raise key errors.
        self.decoder_dmodel = d["decoder_dmodel"]
        self.n_channels = d.get("n_channels", 3)
        self.n_layers = d["n_layers"]
        self.patch_size = d["patch_size"]
        self.temporal_patch_size = d.get("temporal_patch_size", 1)
        self.use_vision_norm = d.get("use_vision_norm", True)


class RMSNorm(nn.Module):
    # [CLASS DOC] HELPER: RMS norm over the last dim with a learnable scale (default dtype, ones).
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        # [FUNCTION DOC] HELPER: Stores size and epsilon (default 1e-6); creates the scale of ones.
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Returns empty input unchanged; otherwise normalizes in float32 with
        # the scale cast to float32 and casts back to the input dtype.
        if x.numel() == 0:
            return x
        return F.rms_norm(x.float(), (self.hidden_size,), self.weight.float(), self.variance_epsilon).to(x.dtype)


def _prime_factors(n: int) -> list[int]:
    # [FUNCTION DOC] HELPER: Trial-division prime factorization with multiplicity, ascending (twos
    # first, then odd candidates). Raises a value error for n below 1; returns an empty list for 1.
    """
    Return the prime factors of *n* in ascending order.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    factors: list[int] = []

    while n % 2 == 0:
        factors.append(2)
        n //= 2

    p = 3
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 2

    if n > 1:
        factors.append(n)
    return factors


def plan_out_scales(
    temporal_patch_size: int, patch_size: int, n_layers: int, n_channels: int = 3
) -> list[tuple[int, int, int, int]]:
    # [FUNCTION DOC] HELPER: Plans the fold hierarchy; returns a list of n_layers+1 scale tuples
    # (time fold, height fold, width fold, channel width), first being (1, 1, 1, channels) and last the
    # full patch. Raises a value error if patch size is 1 or less. Candidate scales: start at the raw
    # patch; for each prime factor of the spatial patch size, largest first, multiply the spatial fold
    # (channel width = spatial area times channels rounded up to a multiple of 64); then for each prime
    # factor of the temporal size, largest first, multiply the temporal fold (channel width includes the
    # time fold, rounded up to 64). Ideal log-reduction levels are n_layers+1 evenly spaced values from
    # zero to the log of the full patch volume; cost is the absolute log difference between ideals and
    # candidate volume reductions (product of folds). If there are at least as many candidates as layers
    # picks the nearest candidate for each ideal; otherwise solves a linear assignment (scipy, imported
    # lazily). Forces the first pick to the raw patch and the last to the full patch. Returns the
    # chosen tuples.
    # KNOWN ISSUE: the nearest-candidate branch (used when the layer count is at least the candidate
    # count) can pick the same candidate for several levels, yielding zero-fold layers.
    if patch_size <= 1:
        raise ValueError("patch_size must be greater than 1, otherwise this doesn't make sense")

    def _round_up(x: int) -> int:
        # [FUNCTION DOC] HELPER: Rounds up to the next multiple of 64.
        return int(np.ceil(x / 64)) * 64

    last_h_scale = 1
    scales: list[tuple[int, int, int, int]] = [(1, 1, 1, n_channels)]
    for pscale in _prime_factors(patch_size)[::-1]:
        last_h_scale *= pscale
        scales.append(
            (
                1,
                last_h_scale,
                last_h_scale,
                _round_up((last_h_scale**2) * n_channels),
            )
        )
    last_t_scale = 1
    for tscale in _prime_factors(temporal_patch_size)[::-1]:
        last_t_scale *= tscale
        scales.append(
            (
                last_t_scale,
                last_h_scale,
                last_h_scale,
                _round_up((last_h_scale**2) * n_channels * last_t_scale),
            )
        )

    size_reduction = np.prod(np.array(scales)[:, :-1], 1)

    log_ideal_scales = np.linspace(
        0,
        np.log(patch_size * patch_size * temporal_patch_size * n_channels),
        n_layers + 1,
    )
    cost_matrix = np.abs(log_ideal_scales[:, None] - np.log(size_reduction)[None])

    if n_layers >= len(scales):
        idxs = np.argmin(cost_matrix, axis=1)
    else:
        from scipy.optimize import linear_sum_assignment

        idxs = linear_sum_assignment(cost_matrix)[1]

    assert len(idxs) >= 2
    idxs[0] = 0
    idxs[-1] = len(scales) - 1

    return [scales[i] for i in idxs]


def fold_timespace_to_depth(vision_patches_bthwc: torch.Tensor, t_fold: int, hw_fold: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Space-to-depth on (batch, time, height, width, channels): asserts each
    # spatial dim and time are divisible by the folds (assertion errors with messages); splits time into
    # (blocks, fold), height and width likewise, moves the fold axes (time, height, width order) next
    # to channels, and merges them into the channel axis: channels become fold_t * fold_hw^2 * C with
    # the within-block time, row, column as the slowest-varying part. Returns the new tensor.
    """
    Convert a tensor of shape (B, T, H, W, C) to a tensor of shape (B, T // t, H // hw, W //  hw, C * (t * hw**2))
    """
    B, T, H, W, C = vision_patches_bthwc.shape

    assert T % t_fold == 0, f"Temporal dimension {T} must be divisible by {t_fold}"
    assert H % hw_fold == 0, f"Height dimension {H} must be divisible by {hw_fold}"
    assert W % hw_fold == 0, f"Width dimension {W} must be divisible by {hw_fold}"

    t_new = T // t_fold
    h_new = H // hw_fold
    w_new = W // hw_fold

    x = vision_patches_bthwc.reshape(B, t_new, t_fold, h_new, hw_fold, w_new, hw_fold, C)

    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)

    x = x.reshape(B, t_new, h_new, w_new, t_fold * hw_fold * hw_fold * C)

    return x


class HMLPPatchEncoder(nn.Module):
    # [CLASS DOC] Encoder with n_layers bias-free linear layers named linear_i (all but the last
    # followed by RMS norm named norm_i and gelu) and an optional final RMS norm; module names are
    # checkpoint-visible. Input is (patches, time, height, width, channels) for a whole patch; output is
    # (patches, decoder width).
    def __init__(
        self,
        config: InklingVisionConfig,
    ):
        # [FUNCTION DOC] Plans scales, then for each adjacent scale pair creates the layer: input
        # width is the start channel width times the fold multiplier (ratio of time, height, width
        # between scales); output width is the next scale's channel width, except the last layer which
        # outputs the decoder width and has no norm. Adds the final norm if configured.
        super().__init__()
        self.decoder_dmodel = config.decoder_dmodel
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.n_channels = config.n_channels
        self.n_layers = config.n_layers
        self.use_vision_norm = config.use_vision_norm

        self.scales: list[tuple[int, int, int, int]] = plan_out_scales(
            self.temporal_patch_size, self.patch_size, self.n_layers, self.n_channels
        )
        self.layers: nn.ModuleDict = nn.ModuleDict()
        for i, (start_scale, end_scale) in enumerate(zip(self.scales[:-1], self.scales[1:], strict=False)):
            shuffle_mult = (
                (end_scale[0] // start_scale[0]) * (end_scale[1] // start_scale[1]) * (end_scale[2] // start_scale[2])
            )
            if i == self.n_layers - 1:
                self.layers[f"linear_{i}"] = nn.Linear(start_scale[3] * shuffle_mult, self.decoder_dmodel, bias=False)
            else:
                self.layers[f"linear_{i}"] = nn.Linear(start_scale[3] * shuffle_mult, end_scale[3], bias=False)
                self.layers[f"norm_{i}"] = RMSNorm(end_scale[3])

        self.final_norm: RMSNorm | None = None
        if self.use_vision_norm:
            self.final_norm = RMSNorm(self.decoder_dmodel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] For each layer: folds space-time into channels by the ratio between consecutive
        # scales when either fold exceeds one; asserts the remaining spatial-temporal shape matches the
        # scale; applies the linear layer; for non-final layers applies RMS norm and gelu. Then the
        # optional final norm, and flattens to (patches, decoder width) (the spatial-temporal dims must
        # be 1 by then).
        num_patches, T, H, W, C = x.shape
        for i, (start_scale, end_scale) in enumerate(zip(self.scales[:-1], self.scales[1:], strict=False)):
            t_fold = end_scale[0] // start_scale[0]
            hw_fold = end_scale[1] // start_scale[1]
            if hw_fold > 1 or t_fold > 1:
                x = fold_timespace_to_depth(x, t_fold, hw_fold)
            assert x.shape[1:-1] == (
                T // end_scale[0],
                H // end_scale[1],
                W // end_scale[2],
            )
            x = self.layers[f"linear_{i}"](x)
            if i < self.n_layers - 1:
                norm = cast(RMSNorm, self.layers[f"norm_{i}"])
                x = norm(x)
                x = F.gelu(x)

        if self.final_norm is not None:
            x = self.final_norm(x)

        x = x.reshape(num_patches, -1)
        return x
