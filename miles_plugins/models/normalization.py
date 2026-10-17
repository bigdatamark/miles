# [MODULE DOC] HELPER: normalization utilities for model plugins. Currently only an eager RMS norm
# used to recompute, outside the fused kernel, the normalisation that Transformer Engine fuses into a
# layer-norm column-parallel linear.
import torch


def rms_norm(inputs: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: root-mean-square normalisation over the last dimension. Computes in
    # fp32: x times the reciprocal square root of (mean of x squared over the last dim, plus eps),
    # multiplies by gamma (cast to fp32), then casts back to the input's original dtype.
    # Returns a tensor shaped like inputs. No side effects; no bias or mean subtraction.
    """Recompute the RMSNorm fused into TELayerNormColumnParallelLinear (eager, fp32 internals)."""
    x = inputs.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * gamma.float()).to(inputs.dtype)
