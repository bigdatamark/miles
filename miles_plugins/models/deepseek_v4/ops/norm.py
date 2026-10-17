# [MODULE DOC] HELPER: Root-mean-square layer normalization used by the DeepSeek-V4 compressor,
# implemented in plain tensor operations with a float32 learnable scale.
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    # [CLASS DOC] HELPER: RMS norm over the last dimension with a learnable per-channel scale held in
    # float32 (initialized to ones). Normalization math is done in float32 and the result is cast back
    # to the input dtype.
    """
    Kept in pure PyTorch with FP32 weights to match SGLang's compressor norm.

    Args:
        dim: Dimension of the input tensor.
        eps: Epsilon for numerical stability. Defaults to ``1e-6``.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        # [FUNCTION DOC] HELPER: Stores the dimension and epsilon and creates the scale parameter, a
        # float32 vector of ones with length dim.
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        # [FUNCTION DOC] HELPER: Upcasts to float32, divides by the root of (mean of squares over the
        # last dim plus epsilon), multiplies by the scale, and casts back to the original dtype.
        # Output has the same shape and dtype as the input. No side effects.
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)
