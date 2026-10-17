# [MODULE DOC] Grouped RMS normalization with Triton kernels. Each token has n groups (streams) of
# C channels laid out contiguously (row width n times C). Every group is normalized separately:
# normed = x times rstd times (1 + weight), where rstd = 1 / sqrt(mean of squares of the group
# plus eps) and the weight (length n times C, one value per channel, shared by all tokens) is used
# in its "one plus" form. Exposes the forward launcher (returns the float32 normalized values and
# the per-group rstd) and both raw kernels; the backward kernel (input gradient only; no weight
# gradient is computed here) is launched by the calling autograd functions.
"""Grouped RMSNorm over n groups of C channels per token (Triton). grouped_rmsnorm_fwd launches the
forward; the backward kernel is launched inside the callers' autograd functions, fused with their grids."""

import torch
import triton
import triton.language as tl


def block_c(C: int) -> int:
    # [FUNCTION DOC] HELPER: Returns the kernel block width for a group of C channels: the
    # smallest power of two not below C.
    return triton.next_power_of_2(C)


@triton.jit(do_not_specialize=["T"])
def grouped_rmsnorm_fwd_kernel(
    x_ptr,
    w_ptr,
    normed_ptr,  # fp32 out [T, n*C]
    rstd_ptr,  # fp32 out [T, n]
    T,
    N: tl.constexpr,  # streams
    C: tl.constexpr,  # per-stream hidden
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # [FUNCTION DOC] Triton forward kernel with one program per (token, group) pair (grid of
    # T times N; program id divided by N gives the token, remainder the group; out-of-range
    # programs exit). Loads the group's C channels (padded to BLOCK_C) in float32, computes the
    # mean of squares, rstd = 1/sqrt(mean + EPS), writes x times rstd times (1 + weight) to the
    # float32 normed output (token row of N times C) and rstd to the float32 rstd output (T by N).
    # The weight is indexed by group and channel only. T is excluded from specialization.
    pid = tl.program_id(0).to(tl.int64)
    t = pid // N
    c = pid % N
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    base = t * (N * C) + c * C
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + EPS)
    w = tl.load(w_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(normed_ptr + base + offs, x * rstd * (1.0 + w), mask=mask)
    tl.store(rstd_ptr + t * N + c, rstd)


@triton.jit(do_not_specialize=["T"])
def grouped_rmsnorm_bwd_kernel(
    x_ptr,
    w_ptr,
    rstd_ptr,
    dnormed_ptr,  # fp32 in [T, n*C]
    dx_ptr,  # out, x dtype
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # [FUNCTION DOC] Triton backward kernel, one program per (token, group) like the forward.
    # Inputs: original x, weight, saved rstd (T by N) and the float32 gradient of the normed output
    # (read without any cast, so it must already be float32). With g = grad times (1 + weight) and
    # dot = sum over the group of g times x, the input gradient is rstd times g minus x times
    # rstd cubed times dot divided by C, stored cast to dx's dtype. Does not produce a weight
    # gradient.
    pid = tl.program_id(0).to(tl.int64)
    t = pid // N
    c = pid % N
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    base = t * (N * C) + c * C
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dnormed_ptr + base + offs, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + t * N + c)
    g = dy * (1.0 + w)
    dot = tl.sum(g * x, axis=0)
    dx = rstd * g - x * (rstd * rstd * rstd) * (dot / C)
    tl.store(dx_ptr + base + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def grouped_rmsnorm_fwd(x2d: torch.Tensor, weight: torch.Tensor, n: int, eps: float):
    # [FUNCTION DOC] Normalizes a two-dimensional input of T tokens by W channels, split into n
    # groups of C = W // n channels (W is assumed divisible by n; not checked). weight has W
    # elements. Returns (normed, rstd): normed is a new float32 tensor of shape T by W and rstd a
    # new float32 tensor of shape T by n; both are uninitialized when T is zero. Launches T times
    # n programs on the GPU.
    T, W = x2d.shape
    C = W // n
    normed = torch.empty(T, W, dtype=torch.float32, device=x2d.device)
    rstd = torch.empty(T, n, dtype=torch.float32, device=x2d.device)
    if T > 0:
        grouped_rmsnorm_fwd_kernel[(T * n,)](x2d, weight, normed, rstd, T, N=n, C=C, EPS=eps, BLOCK_C=block_c(C))
    return normed, rstd
