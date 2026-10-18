# [MODULE DOC] Small numeric building blocks for Kimi K3: the gated activation used in MLP/experts,
# an RMS norm variant, and the attention-residual aggregation (softmax-weighted mixture of a stack of
# block snapshots and the current prefix sum, followed by an output norm).
import torch
import torch.nn as nn


def situ_and_mul(
    x: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    # [FUNCTION DOC] Gated activation on the fused first-layer output. In float32, splits the last dim
    # in two equal halves (gate first, then linear). Gate becomes beta times tanh(gate over beta)
    # times sigmoid(gate) (a soft-clipped silu with default beta 4); the linear half becomes
    # linear_beta times tanh(linear over linear_beta) (a soft clip, default 25). Returns the product cast
    # to the input dtype, with half the input width.
    gate, linear = torch.chunk(x.float(), 2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    linear = linear_beta * torch.tanh(linear / linear_beta)
    return (gate * linear).to(x.dtype)


class KimiRMSNorm(nn.Module):
    # [CLASS DOC] RMS norm over the last dim with a learnable scale (ones initially, in the given
    # device and dtype). Differs from some other norms in rounding order: normalization is done in
    # float32 and cast back to the input dtype BEFORE multiplying by the scale.
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        device: torch.device | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        # [FUNCTION DOC] Creates the scale parameter and stores epsilon (required, no default).
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Computes in float32 x times the reciprocal root of (mean square plus
        # epsilon), casts to the input dtype, then multiplies by the scale (type promotion follows the
        # scale dtype). Returns the same shape.
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(input_dtype)


def attn_res_aggregate(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_proj: nn.Linear,
    score_norm: KimiRMSNorm,
    output_norm: nn.Module,
) -> torch.Tensor:
    # [FUNCTION DOC] Inputs: prefix sum (..., hidden); block bank (..., rows, hidden) possibly with
    # zero rows; a one-output linear (score projection) and a norm module (used only for its scale and
    # epsilon); and an output norm module. Steps: append the prefix sum as the last row of the bank; in
    # float32 normalize each row by its RMS (epsilon from the score norm); the per-channel score weight
    # is the norm scale times the projection weight; the score per row is the sum over channels of the
    # normalized row times that weight; softmax over rows; mixture is the probability-weighted sum of the
    # unnormalized float32 rows, cast to the row dtype; returns the output norm applied to the mixture
    # (shape (..., hidden)). Rows are weighted per token position.
    rows = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)
    rows_float = rows.float()
    normalized = rows_float * torch.rsqrt(rows_float.square().mean(dim=-1, keepdim=True) + score_norm.eps)
    score_weight = score_norm.weight.float() * score_proj.weight.squeeze(0).float()
    scores = (normalized * score_weight).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    mixed = (probabilities.unsqueeze(-1) * rows_float).sum(dim=-2).to(rows.dtype)
    return output_norm(mixed)
