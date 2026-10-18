# [MODULE DOC] GPU tests of Triton kernels (grouped RMS norm, hyper-connection gate-multiply-mean
# and combine, per-layer-embedding gate and segmented convolution, and the embedding row gather)
# at token counts large enough that element offsets exceed signed 32-bit range (up to 262,144 tokens
# of width 4 x 2560 in bfloat16). To keep memory modest, autograd references are built only for a
# handful of sampled rows (start, middle, around the 2^31 element boundary, end); input values are
# deterministic functions of row and column so a wrong in-bounds address also fails. Tolerance is
# 2e-2 relative and 2e-3 absolute. Constants: 4 streams, 2560 channels, norm epsilon 1e-6. Registered
# for 2-GPU CUDA and ROCm nightly suites (miles-plugin label). Kernels are launched directly with
# explicit grids and 4096-wide blocks.
"""Exercise HC/PLE forward and backward beyond signed int32 element offsets.

Only sampled rows have autograd references, keeping the test smaller than a full
model. Inputs vary across rows and channels so a wrong in-bounds address also fails.
"""

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(est_time=180, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=40, suite="nightly-stage-c-2-gpu-mi350", labels=["miles-plugin"])

import math

import pytest
import torch
import torch.nn.functional as F

from miles.kernels.embedding import ple_gate_conv as ple
from miles.kernels.embedding.ple_gather import gather_ple_rows
from miles.kernels.hyper_connection import hc
from miles.kernels.norm import grouped_rmsnorm

N, C = 4, 2560
W, EPS = N * C, 1e-6
LONG_LENGTHS = [1024, 262144]


def _input(tokens, width=W, phase=0, dtype=torch.bfloat16):
    # [FUNCTION DOC] HELPER: Builds a deterministic GPU matrix (tokens by width, default width 4 x
    # 2560) whose element is the sum of a row term and a column term: row term = ((row index + phase)
    # mod 17 - 8) / 8; column term = ((column index + 3 x phase) mod 13 - 6) / 7. Computed in float32,
    # then cast to the dtype (default bfloat16). The phase varies inputs between calls.
    rows = (torch.arange(tokens, device="cuda", dtype=torch.float32) + phase).remainder(17).sub_(8).div_(8)
    cols = (torch.arange(width, device="cuda", dtype=torch.float32) + 3 * phase).remainder(13).sub_(6).div_(7)
    return (rows[:, None] + cols[None, :]).to(dtype)


def _rows(tokens):
    # [FUNCTION DOC] HELPER: Returns a sorted GPU tensor of unique sample row indices within range:
    # 0, 1, the middle token and its neighbours, the row index where element offsets reach 2^31 (2^31
    # divided by the stream width, integer division) and its neighbours, and the last token. Indices
    # outside [0, tokens) are dropped.
    boundary = 2**31 // W
    candidates = [
        0,
        1,
        tokens // 2 - 1,
        tokens // 2,
        tokens // 2 + 1,
        boundary - 1,
        boundary,
        boundary + 1,
        tokens - 1,
    ]
    return torch.tensor(sorted({i for i in candidates if 0 <= i < tokens}), device="cuda")


def _check(actual, rows, expected):
    # [FUNCTION DOC] HELPER: Asserts the sampled rows of the actual tensor are close to the expected
    # values (cast to the actual dtype) with relative tolerance 2e-2 and absolute tolerance 2e-3.
    torch.testing.assert_close(actual[rows], expected.to(actual.dtype), rtol=2e-2, atol=2e-3)


def _norm_reference(x, weight):
    # [FUNCTION DOC] HELPER: Differentiable torch reference for the grouped RMS norm. Reshapes x
    # into (rows, 4 streams, 2560), computes the reciprocal root mean square per stream (epsilon 1e-6),
    # normalizes, flattens streams back and multiplies by (1 + weight). Returns (normalized output,
    # reciprocal root mean square of shape rows by streams).
    streams = x.reshape(-1, N, C)
    rstd = torch.rsqrt(streams.square().mean(-1) + EPS)
    return (streams * rstd[..., None]).flatten(1) * (1 + weight), rstd


@pytest.mark.parametrize("tokens", [1024, 209715, 209716, 262144])
def test_grouped_rmsnorm_long_offsets(tokens):
    # [FUNCTION DOC] Parametrized over 1024, 209715, 209716 and 262144 tokens (bracketing the 2^31
    # element boundary). Runs the forward kernel on the full input with a small float32 weight, and
    # compares output and reciprocal-rms to the torch reference computed on sampled rows. Frees the
    # output, then launches the backward kernel (one program per token-stream) with a deterministic
    # upstream gradient and compares the sampled input gradient to autograd of the reference.
    rows = _rows(tokens)
    x = _input(tokens)
    weight = _input(1, phase=1, dtype=torch.float32).flatten() * 0.05
    sample = x[rows].float().requires_grad_()
    expected, expected_rstd = _norm_reference(sample, weight)
    out, rstd = grouped_rmsnorm.grouped_rmsnorm_fwd(x, weight, N, EPS)
    _check(out, rows, expected)
    _check(rstd, rows, expected_rstd)
    del out

    dout = _input(tokens, phase=2)
    expected.backward(dout[rows].float())
    dx = torch.empty_like(x)
    grouped_rmsnorm.grouped_rmsnorm_bwd_kernel[(tokens * N,)](
        x, weight, rstd, dout, dx, tokens, N=N, C=C, BLOCK_C=4096
    )
    _check(dx, rows, sample.grad)


@pytest.mark.parametrize("tokens", LONG_LENGTHS)
def test_gate_mul_mean_long_offsets(tokens):
    # [FUNCTION DOC] Parametrized over 1024 and 262144 tokens. Reference: elementwise product of gate
    # and normed inputs, averaged over the 4 streams. Runs the forward kernel (one program per token) and
    # compares sampled rows; then the backward kernel with a deterministic upstream gradient and compares
    # sampled gate and normed gradients to autograd.
    rows = _rows(tokens)
    gate, normed = _input(tokens), _input(tokens, phase=1)
    gate_ref, normed_ref = [x[rows].float().requires_grad_() for x in (gate, normed)]
    expected = (gate_ref * normed_ref).reshape(-1, N, C).mean(1)
    out = torch.empty(tokens, C, device="cuda", dtype=torch.bfloat16)
    hc._gate_mul_mean_fwd_kernel[(tokens,)](gate, normed, out, tokens, N=N, C=C, BLOCK_C=4096)
    _check(out, rows, expected)
    del out

    dout = _input(tokens, C, phase=2)
    expected.backward(dout[rows].float())
    dgate, dnormed = [torch.empty(tokens, W, device="cuda") for _ in range(2)]
    hc._gate_mul_mean_bwd_kernel[(tokens,)](dout, gate, normed, dgate, dnormed, tokens, N=N, C=C, BLOCK_C=4096)
    _check(dgate, rows, gate_ref.grad)
    _check(dnormed, rows, normed_ref.grad)


@pytest.mark.parametrize("tokens", LONG_LENGTHS)
def test_combine_long_offsets(tokens):
    # [FUNCTION DOC] Parametrized over 1024 and 262144 tokens. Reference for the hyper-connection
    # combine: residual (per stream) plus post weight (per stream, float32) times the layer output y
    # broadcast across streams. Runs the forward kernel (one program per token-stream) and checks
    # sampled rows; then the backward kernel using the residual as a stand-in upstream gradient (to avoid
    # a second large input) and checks the sampled gradients of y and the post weights against autograd.
    rows = _rows(tokens)
    residual, y = _input(tokens), _input(tokens, C, phase=1)
    hpost = _input(tokens, N, phase=2, dtype=torch.float32)
    y_ref, hpost_ref = [x[rows].float().requires_grad_() for x in (y, hpost)]
    expected = (residual[rows].float().reshape(-1, N, C) + hpost_ref[..., None] * y_ref[:, None, :]).flatten(1)
    out = torch.empty_like(residual)
    hc._combine_fwd_kernel[(tokens * N,)](residual, y, hpost, out, tokens, N=N, C=C, BLOCK_C=4096)
    _check(out, rows, expected)
    del out

    # Reuse a varying input as the incoming gradient; no full-size reference graph.
    expected.backward(residual[rows].float())
    dy, dhpost = torch.empty_like(y), torch.empty_like(hpost)
    hc._combine_bwd_kernel[(tokens,)](residual, y, hpost, dy, dhpost, tokens, N=N, C=C, BLOCK_C=4096)
    _check(dy, rows, y_ref.grad)
    _check(dhpost, rows, hpost_ref.grad)


@pytest.mark.parametrize("tokens", LONG_LENGTHS)
def test_ple_gate_long_offsets(tokens):
    # [FUNCTION DOC] Parametrized over 1024 and 262144 tokens. Reference for the per-layer-embedding
    # gate: normalizes key and query streams with their own weights (expanded per sampled row so partial
    # weight gradients are visible), takes the per-stream dot product scaled by the inverse square root
    # of the channel count, applies a signed-square-root transform (sign times square root of the
    # magnitude floored at 1e-6) then a sigmoid to get per-stream gates, and multiplies the value
    # (broadcast to streams) by the gates. Runs the forward kernel and checks gated output, gates and
    # both reciprocal-rms tensors on sampled rows; frees the large outputs; then runs the backward kernel
    # with a deterministic upstream gradient and checks gradients of key, query, value (flattened per
    # stream) and both weights (per-row partials) against autograd.
    rows = _rows(tokens)
    key, query, value = _input(tokens), _input(tokens, phase=3), _input(tokens, C, phase=5)
    wk, wq = [_input(1, phase=p, dtype=torch.float32).flatten() * 0.05 for p in (1, 2)]
    key_ref, query_ref = [x[rows].float().requires_grad_() for x in (key, query)]
    # Independent per-row weights and per-stream values expose the partial gradients.
    wk_ref, wq_ref = [w.expand(len(rows), -1).clone().requires_grad_() for w in (wk, wq)]
    value_ref = value[rows].float()[:, None, :].expand(-1, N, -1).clone().requires_grad_()
    kn, rk = _norm_reference(key_ref, wk_ref)
    qn, rq = _norm_reference(query_ref, wq_ref)
    score = (kn * qn).reshape(-1, N, C).sum(-1) / math.sqrt(C)
    gates_ref = torch.sigmoid(torch.where(score >= 0, 1.0, -1.0) * score.abs().clamp_min(1e-6).sqrt())
    expected = (gates_ref[..., None] * value_ref).flatten(1)

    gated = torch.empty(tokens, W, device="cuda")
    gates, rstdk, rstdq = [torch.empty(tokens, N, device="cuda") for _ in range(3)]
    ple._ple_gate_fwd_kernel[(tokens * N,)](
        key,
        query,
        value,
        wk,
        wq,
        gated,
        gates,
        rstdk,
        rstdq,
        tokens,
        N=N,
        C=C,
        EPS=EPS,
        SQRTC=math.sqrt(C),
        BLOCK_C=4096,
    )
    for actual, ref in [(gated, expected), (gates, gates_ref), (rstdk, rk), (rstdq, rq)]:
        _check(actual, rows, ref)
    del gated, actual

    dout = _input(tokens, phase=6)
    expected.backward(dout[rows].float())
    dkey, dquery = torch.empty_like(key), torch.empty_like(query)
    dvalue, dwk, dwq = [torch.empty(tokens, W, device="cuda") for _ in range(3)]
    ple._ple_gate_bwd_kernel[(tokens * N,)](
        dout,
        key,
        query,
        value,
        wk,
        wq,
        gates,
        rstdk,
        rstdq,
        dkey,
        dquery,
        dvalue,
        dwk,
        dwq,
        tokens,
        N=N,
        C=C,
        SQRTC=math.sqrt(C),
        BLOCK_C=4096,
    )
    for actual, ref in [
        (dkey, key_ref.grad),
        (dquery, query_ref.grad),
        (dvalue, value_ref.grad.flatten(1)),
        (dwk, wk_ref.grad),
        (dwq, wq_ref.grad),
    ]:
        _check(actual, rows, ref)


@pytest.mark.parametrize("tokens", LONG_LENGTHS)
def test_ple_segmented_conv_long_offsets(tokens):
    # [FUNCTION DOC] Parametrized over 1024 and 262144 tokens. Depthwise causal convolution with
    # kernel size 3 and dilation 2 over two packed segments (split at the middle token), followed by
    # silu and added to a gated input. The reference gathers the three dilated source rows for each
    # sampled row, zeroing those before the segment start, and builds the expected output and
    # pre-activation. Runs the forward kernel over a (tokens by width/256) grid and checks sampled rows;
    # then the backward kernel with a sparse upstream gradient (nonzero only at sampled rows) and checks
    # the normed-input gradient on all touched source rows (accumulating contributions from
    # neighbouring sampled rows), the gated-input gradient on those rows, and the full weight gradient.
    rows = _rows(tokens)
    kernel, dilation = 3, 2
    normed = _input(tokens, dtype=torch.float32)
    gated = _input(tokens, phase=1)
    weight = _input(W, kernel, phase=2, dtype=torch.float32) * 0.1
    cu = torch.tensor([0, tokens // 2, tokens], device="cuda", dtype=torch.int32)
    lo, hi = ple._seg_bounds(tokens, cu, "cuda")
    source_rows = rows[:, None] - torch.arange(kernel - 1, -1, -1, device="cuda") * dilation
    valid = source_rows >= lo[rows, None]
    samples = normed[source_rows.clamp_min(0)].detach().requires_grad_()
    weight_ref = weight.clone().requires_grad_()
    pre_ref = (samples * valid[..., None] * weight_ref.T).sum(1)
    expected = gated[rows].float() + F.silu(pre_ref)

    out, pre = torch.empty_like(gated), torch.empty_like(normed)
    grid = (tokens, W // 256)
    ple._ple_conv_fwd_kernel[grid](normed, gated, weight, lo, out, pre, tokens, W, K=kernel, DIL=dilation, BLOCK_W=256)
    _check(out, rows, expected)
    _check(pre, rows, pre_ref)
    del out, gated

    # Sparse upstream gradients make a complete small reference for the weight
    # reduction and neighboring input gradients, including packed boundaries.
    dout = torch.zeros(tokens, W, device="cuda", dtype=torch.bfloat16)
    dout[rows] = _input(len(rows), phase=7)
    expected.backward(dout[rows].float())
    dnormed, dgated = torch.empty_like(normed), torch.empty_like(normed)
    dweight = torch.zeros_like(weight)
    ple._ple_conv_bwd_kernel[grid](
        dout,
        pre,
        normed,
        weight,
        lo,
        hi,
        dnormed,
        dweight,
        dgated,
        tokens,
        W,
        K=kernel,
        DIL=dilation,
        BLOCK_W=256,
    )
    checked_rows, inverse = source_rows.clamp_min(0).flatten().unique(return_inverse=True)
    dx_ref = torch.zeros(len(checked_rows), W, device="cuda")
    dx_ref.index_add_(0, inverse, samples.grad.flatten(0, 1))
    _check(dnormed, checked_rows, dx_ref)
    _check(dgated, checked_rows, dout[checked_rows])
    torch.testing.assert_close(dweight, weight_ref.grad, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("tokens", [1024, 60000])
def test_ple_gather_long_offsets(tokens):
    # [FUNCTION DOC] Parametrized over 1024 and 60000 tokens. Builds a pinned host table of 4096 rows
    # by 2560 channels and pseudo-random row ids (multiples of 7919 modulo the table size) of shape
    # tokens by 16 heads, gathers rows over the full id range via the gather function, and asserts that
    # at sampled tokens (including those around the 2^31 element boundary) the output equals the
    # corresponding table rows exactly.
    heads, rows = 16, 4096
    table = _input(rows, C).cpu().pin_memory()
    ids = (torch.arange(tokens * heads, device="cuda") * 7919).remainder(rows).view(tokens, heads)
    out = gather_ple_rows(table, ids, 0, rows)
    boundary = 2**31 // (heads * C)
    sample = torch.tensor(
        sorted({i for i in (0, boundary - 1, boundary, boundary + 1, tokens - 1) if i < tokens}), device="cuda"
    )
    torch.testing.assert_close(out[sample], table.cuda()[ids[sample]], rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
