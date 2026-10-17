# [MODULE DOC] Triton kernels for the mixing and aggregation steps of multi-stream (hyper-)
# connections where each token keeps N residual streams of d channels (flattened width N times d)
# and per-token coefficients select how streams combine. Layout: sequence by batch by width; each
# (sequence, batch) pair is one "token" row. Two differentiable operations:
# mhc_mix: new stream j = h_post[j] times x + sum over streams i of h_res[i, j] times old stream i,
# i.e. a per-token N-by-N stream mixing matrix applied to the old residual plus a scaled copy of the
# block output x; mhc_aggregate: weighted sum over streams of the stream rows with per-token
# weights ("pre"), producing one d-channel row, without materializing the expanded intermediate.
# Coefficients are held in float32; all arithmetic is float32 with one cast to the output dtype;
# gradients are returned in the original input dtypes. N must be a power of two (it is used as a
# Triton range length, and N times N likewise); d is arbitrary (blocks of 1024 channels).
# KNOWN ISSUE: unlike the sibling hc module, token and stream offsets are computed with the 32-bit
# program index (no widening to 64-bit), so row offsets overflow when tokens times N times d
# exceeds 2 to the power 31.
import torch
import triton
import triton.language as tl


@triton.jit
def _mix_fwd(x_ptr, res_ptr, hres_ptr, hpost_ptr, out_ptr, d, N: tl.constexpr, BLOCK: tl.constexpr):
    # [FUNCTION DOC] HELPER: Triton forward kernel for stream mixing; grid of tokens by channel blocks
    # of BLOCK. For each output stream j: out[token, j, block] = h_post[token, j] times x[token,
    # block] plus the sum over input streams i of h_res[token, i, j] times residual[token, i, block];
    # coefficients are float32 tables (h_res: token by N by N, h_post: token by N), x and residual
    # are loaded as float32, and the result is cast to the output dtype (token by N by d).
    t = tl.program_id(0)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < d
    x = tl.load(x_ptr + t * d + off, mask=mask, other=0.0).to(tl.float32)
    for j in tl.static_range(N):
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for i in tl.static_range(N):
            r = tl.load(res_ptr + (t * N + i) * d + off, mask=mask, other=0.0).to(tl.float32)
            acc += tl.load(hres_ptr + (t * N + i) * N + j) * r
        out = tl.load(hpost_ptr + t * N + j) * x + acc
        tl.store(out_ptr + (t * N + j) * d + off, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mix_bwd(
    g_ptr,
    x_ptr,
    res_ptr,
    hres_ptr,
    hpost_ptr,
    gx_ptr,
    gres_ptr,
    ghres_ptr,
    ghpost_ptr,
    d,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton backward kernel for stream mixing; one program per token that
    # loops over channel blocks of BLOCK (so the per-token coefficient gradients are reduced inside
    # the program, no atomics). Per block: gradient of x = sum over streams j of h_post[j] times
    # incoming gradient of stream j; gradient of old stream i = sum over j of h_res[i, j] times
    # incoming gradient of stream j; accumulates gradient of h_post[j] = sum over channels of
    # incoming gradient j times x and gradient of h_res[i, j] = sum over channels of incoming
    # gradient j times old stream i (float32 vectors of length N and N times N held in registers).
    # Gradients of x and the residual are stored per block in their buffers' dtypes; after the loop
    # the coefficient gradients are stored once as float32.
    t = tl.program_id(0)
    ghpost = tl.zeros([N], dtype=tl.float32)
    ghres = tl.zeros([N * N], dtype=tl.float32)
    for base in range(0, d, BLOCK):
        off = base + tl.arange(0, BLOCK)
        mask = off < d
        x = tl.load(x_ptr + t * d + off, mask=mask, other=0.0).to(tl.float32)
        gx = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.static_range(N):
            g = tl.load(g_ptr + (t * N + j) * d + off, mask=mask, other=0.0).to(tl.float32)
            gx += tl.load(hpost_ptr + t * N + j) * g
            hits = tl.sum(tl.where(mask, g * x, 0.0))
            ghpost += tl.where(tl.arange(0, N) == j, hits, 0.0)
        tl.store(gx_ptr + t * d + off, gx.to(gx_ptr.dtype.element_ty), mask=mask)
        for i in tl.static_range(N):
            r = tl.load(res_ptr + (t * N + i) * d + off, mask=mask, other=0.0).to(tl.float32)
            gres = tl.zeros([BLOCK], dtype=tl.float32)
            for j in tl.static_range(N):
                g = tl.load(g_ptr + (t * N + j) * d + off, mask=mask, other=0.0).to(tl.float32)
                gres += tl.load(hres_ptr + (t * N + i) * N + j) * g
                hit = tl.sum(tl.where(mask, g * r, 0.0))
                ghres += tl.where(tl.arange(0, N * N) == i * N + j, hit, 0.0)
            tl.store(gres_ptr + (t * N + i) * d + off, gres.to(gres_ptr.dtype.element_ty), mask=mask)
    tl.store(ghpost_ptr + t * N + tl.arange(0, N), ghpost)
    tl.store(ghres_ptr + t * N * N + tl.arange(0, N * N), ghres)


class _HCMix(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function for stream mixing; saves contiguous inputs and float32
    # coefficients, plus the original dtypes and the shape, so gradients are cast back.
    @staticmethod
    def forward(ctx, x, residual, h_res, h_post, n):
        # [FUNCTION DOC] Inputs: x (sequence by batch by d), residual (sequence by batch by n times d),
        # h_res (sequence by batch by n by n), h_post (sequence by batch by n), stream count n.
        # Coefficients are converted to contiguous float32. Output has the residual's dtype and shape
        # sequence by batch by n times d. Launches a grid of tokens by ceil(d / 1024) with 4 warps.
        s, b, d = x.shape
        t = s * b
        ctx.dtypes = (x.dtype, residual.dtype, h_res.dtype, h_post.dtype)
        x = x.contiguous()
        residual = residual.contiguous()
        h_res = h_res.contiguous().float()
        h_post = h_post.contiguous().float()
        out = torch.empty(t, n, d, dtype=residual.dtype, device=x.device)
        block = 1024
        _mix_fwd[(t, triton.cdiv(d, block))](x, residual, h_res, h_post, out, d, N=n, BLOCK=block, num_warps=4)
        ctx.save_for_backward(x, residual, h_res, h_post)
        ctx.n, ctx.shape = n, (s, b, d)
        return out.view(s, b, n * d)

    @staticmethod
    def backward(ctx, grad_out):
        # [FUNCTION DOC] Returns gradients for (x, residual, h_res, h_post) cast to the original input
        # dtypes (x and residual shapes restored, coefficient gradients reshaped to sequence by batch
        # by n by n and by n) and none for n. One program per token with 1024-channel blocks.
        x, residual, h_res, h_post = ctx.saved_tensors
        n = ctx.n
        s, b, d = ctx.shape
        t = s * b
        grad_out = grad_out.contiguous()
        gx = torch.empty_like(x)
        gres = torch.empty_like(residual)
        ghres = torch.empty(t, n, n, dtype=torch.float32, device=x.device)
        ghpost = torch.empty(t, n, dtype=torch.float32, device=x.device)
        _mix_bwd[(t,)](
            grad_out,
            x,
            residual,
            h_res,
            h_post,
            gx,
            gres,
            ghres,
            ghpost,
            d,
            N=n,
            BLOCK=1024,
            num_warps=4,
        )
        x_dt, res_dt, hres_dt, hpost_dt = ctx.dtypes
        return (
            gx.view(s, b, d).to(x_dt),
            gres.view(s, b, n * d).to(res_dt),
            ghres.view(s, b, n, n).to(hres_dt),
            ghpost.view(s, b, n).to(hpost_dt),
            None,
        )


def mhc_mix(x, original_residual, h_res, h_post, n):
    # [FUNCTION DOC] Public entry: mixes streams as described in the module doc and returns a tensor
    # of shape sequence by batch by n times d in the residual's dtype; differentiable in all four
    # tensor inputs. x is sequence by batch by d, the residual sequence by batch by n times d.
    """`h_post * x + einsum("sbij,sbid->sbjd", h_res, residual)`, flattened back to [s, b, n*d]."""
    return _HCMix.apply(x, original_residual, h_res, h_post, n)


@triton.jit
def _agg_fwd(x_ptr, pre_ptr, out_ptr, d, N: tl.constexpr, BLOCK: tl.constexpr):
    # [FUNCTION DOC] HELPER: Triton forward kernel for aggregation; grid of tokens by channel blocks.
    # out[token, block] = sum over the N streams of pre[token, stream] (float32) times
    # x[token, stream, block], accumulated in float32 and cast to the output dtype.
    t = tl.program_id(0)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < d
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(N):
        v = tl.load(x_ptr + (t * N + i) * d + off, mask=mask, other=0.0).to(tl.float32)
        acc += tl.load(pre_ptr + t * N + i) * v
    tl.store(out_ptr + t * d + off, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _agg_bwd(g_ptr, x_ptr, pre_ptr, gx_ptr, gpre_ptr, d, N: tl.constexpr, BLOCK: tl.constexpr):
    # [FUNCTION DOC] HELPER: Triton backward kernel for aggregation; one program per token looping over
    # channel blocks. Gradient of stream i row = pre[token, i] times incoming gradient row (cast to
    # the buffer dtype); gradient of pre[token, i] = sum over all channels of incoming gradient times
    # stream i row, reduced within the program in float32 and stored once at the end.
    t = tl.program_id(0)
    gpre = tl.zeros([N], dtype=tl.float32)
    for base in range(0, d, BLOCK):
        off = base + tl.arange(0, BLOCK)
        mask = off < d
        g = tl.load(g_ptr + t * d + off, mask=mask, other=0.0).to(tl.float32)
        for i in tl.static_range(N):
            v = tl.load(x_ptr + (t * N + i) * d + off, mask=mask, other=0.0).to(tl.float32)
            tl.store(
                gx_ptr + (t * N + i) * d + off,
                (tl.load(pre_ptr + t * N + i) * g).to(gx_ptr.dtype.element_ty),
                mask=mask,
            )
            hit = tl.sum(tl.where(mask, g * v, 0.0))
            gpre += tl.where(tl.arange(0, N) == i, hit, 0.0)
    tl.store(gpre_ptr + t * N + tl.arange(0, N), gpre)


class _Aggregate(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function for the weighted stream aggregation; saves contiguous x
    # and float32 weights, plus original dtypes and the shape.
    @staticmethod
    def forward(ctx, x, pre, n):
        # [FUNCTION DOC] Inputs: x (sequence by batch by n times d), pre (sequence by batch by n,
        # converted to float32), stream count n. Returns sequence by batch by d in x's dtype.
        # Launches a grid of tokens by ceil(d / 1024) with 4 warps.
        s, b, nc = x.shape
        d = nc // n
        t = s * b
        ctx.dtypes = (x.dtype, pre.dtype)
        x = x.contiguous()
        pre = pre.contiguous().float()
        out = torch.empty(t, d, dtype=x.dtype, device=x.device)
        block = 1024
        _agg_fwd[(t, triton.cdiv(d, block))](x, pre, out, d, N=n, BLOCK=block, num_warps=4)
        ctx.save_for_backward(x, pre)
        ctx.n, ctx.shape = n, (s, b, d)
        return out.view(s, b, d)

    @staticmethod
    def backward(ctx, grad_out):
        # [FUNCTION DOC] Returns (gradient for x of shape sequence by batch by n times d, gradient for
        # pre of shape sequence by batch by n, both cast to the original dtypes; none for n).
        x, pre = ctx.saved_tensors
        n = ctx.n
        s, b, d = ctx.shape
        t = s * b
        grad_out = grad_out.contiguous()
        gx = torch.empty_like(x)
        gpre = torch.empty(t, n, dtype=torch.float32, device=x.device)
        _agg_bwd[(t,)](grad_out, x, pre, gx, gpre, d, N=n, BLOCK=1024, num_warps=4)
        x_dt, pre_dt = ctx.dtypes
        return gx.view(s, b, n * d).to(x_dt), gpre.view(s, b, n).to(pre_dt), None


def mhc_aggregate(x, pre, n):
    # [FUNCTION DOC] Public entry: weighted sum over the n streams of x (sequence by batch by n times
    # d) with per-token weights pre (sequence by batch by n), returning sequence by batch by d;
    # differentiable in both inputs.
    """`(pre.unsqueeze(-1) * x.view(s, b, n, d)).sum(dim=2)` in fp32, without the [s, b, n, d] temporary."""
    return _Aggregate.apply(x, pre, n)
