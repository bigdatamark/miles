# [MODULE DOC] In-place rotary position embedding (RoPE) with interleaved channel pairs, as a
# differentiable operation backed by a Triton kernel. The last dimension of x holds complex pairs:
# channels (2k, 2k+1) are the real and imaginary parts of complex element k. Each element is
# multiplied by a complex frequency factor chosen by the token's position (the conjugate factor when
# applying the inverse rotation). Public entry point: apply_rotary_emb. Helpers: a stride probe
# and the rotation driver.
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr, f_ptr, n_rows, seqlen, inner, row_stride, HALF: tl.constexpr, INVERSE: tl.constexpr, ROWS: tl.constexpr
):
    # [FUNCTION DOC] HELPER: Triton kernel; each program handles ROWS rows of x (rows are the
    # flattened leading dimensions, each holding 2*HALF channels). Row r has position
    # (r // inner) mod seqlen, where inner is the product of dimensions between sequence and
    # channel (heads for four-dimensional input, 1 otherwise). Rows start row_stride elements
    # apart. Loads the real/imaginary pairs in float32 and the position's frequency pair from f
    # (real view of the complex table, 2*HALF floats per position), conjugates it if INVERSE, and
    # writes the complex product back over x in x's dtype. Rows beyond n_rows are masked. HALF must
    # be a power of two for the Triton range.
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    pos = (rows // inner) % seqlen
    k = tl.arange(0, HALF)
    x_offs = rows[:, None] * row_stride + 2 * k[None, :]
    mask = rmask[:, None] & (k[None, :] < HALF)
    xr = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
    xi = tl.load(x_ptr + x_offs + 1, mask=mask, other=0.0).to(tl.float32)
    f_offs = pos[:, None] * (2 * HALF) + 2 * k[None, :]
    fr = tl.load(f_ptr + f_offs, mask=mask, other=0.0)
    fi = tl.load(f_ptr + f_offs + 1, mask=mask, other=0.0)
    if INVERSE:
        fi = -fi
    outr = xr * fr - xi * fi
    outi = xr * fi + xi * fr
    tl.store(x_ptr + x_offs, outr.to(x_ptr.dtype.element_ty), mask=mask)
    tl.store(x_ptr + x_offs + 1, outi.to(x_ptr.dtype.element_ty), mask=mask)


def row_stride_or_none(x: torch.Tensor):
    # [FUNCTION DOC] HELPER: Returns the common distance between consecutive rows if all rows
    # (leading dimensions flattened) are evenly spaced, else None. Requires a unit last stride; a
    # one-dimensional tensor returns its length. Otherwise the stride of the second-to-last
    # dimension is the row stride and each earlier dimension must have the stride that makes the
    # leading dimensions contiguous with respect to it (this accepts slices of the last dimension).
    """Row stride of a contiguous tensor or a last-dim slice of one, else None."""
    if x.stride(-1) != 1:
        return None
    if x.ndim == 1:
        return x.shape[-1]
    row_stride = x.stride(-2)
    expected = row_stride
    for d in range(x.ndim - 3, -1, -1):
        expected *= x.shape[d + 1]
        if x.stride(d) != expected:
            return None
    return row_stride


def _rotate_(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Rotates x in place and returns it. Asserts the frequency table's last
    # dimension equals half the channel count. If rows are not evenly spaced (see the stride
    # probe), rotates a contiguous copy recursively and copies the result back into x. Otherwise
    # sequence length is dimension 1 of x, inner is dimension 2 for four-dimensional x (else 1),
    # the complex table is made contiguous and viewed as real, and the kernel runs with 32 rows per
    # program. The table must have at least sequence-length rows (not checked); only three- or
    # four-dimensional x is meaningful.
    dim = x.shape[-1]
    half = dim // 2
    assert freqs_cis.shape[-1] == half
    row_stride = row_stride_or_none(x)
    if row_stride is None:
        # rows of a permuted view have no single stride: rotate a contiguous copy, then write it back in place
        x.copy_(_rotate_(x.contiguous(), freqs_cis, inverse))
        return x
    seqlen = x.shape[1]
    inner = x.shape[2] if x.ndim == 4 else 1
    f = torch.view_as_real(freqs_cis.contiguous()).contiguous()
    n_rows = x.numel() // dim
    rows_per_prog = 32
    grid = (triton.cdiv(n_rows, rows_per_prog),)
    _rope_kernel[grid](x, f, n_rows, seqlen, inner, row_stride, HALF=half, INVERSE=inverse, ROWS=rows_per_prog)
    return x


class _RotaryEmb(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function for the in-place rotation. Marks the input as modified
    # in place and returns it. The gradient of a rotation is the opposite rotation, so backward
    # applies the same kernel with the inverse flag flipped.
    @staticmethod
    def forward(ctx, x, freqs_cis, inverse):
        # [FUNCTION DOC] Saves the frequency table and the flag, marks x dirty and rotates it in
        # place. Returns x itself.
        ctx.save_for_backward(freqs_cis)
        ctx.inverse = inverse
        ctx.mark_dirty(x)
        return _rotate_(x, freqs_cis, inverse)

    @staticmethod
    def backward(ctx, grad):
        # [FUNCTION DOC] Clones the incoming gradient, rotates the clone in place with the flipped
        # inverse flag and returns it as the gradient for x; none for the table and the flag.
        (freqs_cis,) = ctx.saved_tensors
        return _rotate_(grad.clone(), freqs_cis, not ctx.inverse), None, None


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    # [FUNCTION DOC] Public entry: rotates the last dimension of x in place (x is mutated and also
    # returned) using the complex table freqs_cis of shape sequence length by half the channels;
    # inverse selects the conjugate rotation. x must be batch by sequence by channels or batch by
    # sequence by heads by channels. Differentiable in x. Runs on GPU.
    """In-place RoPE on the last dim of x ([b, s, d] or [b, s, h, d]); `freqs_cis` is complex [s, d/2]."""
    return _RotaryEmb.apply(x, freqs_cis, inverse)
