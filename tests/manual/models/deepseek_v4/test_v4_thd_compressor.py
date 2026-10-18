# [MODULE DOC] Manual tests that the DeepSeek-V4 token compressor behaves identically under
# packed (THD) layouts and context-parallel (CP) splitting. The compressor is built with a small
# fake config (hidden size 256, head dim 128, fixed rope settings, random small weights from a
# fixed seed) and run on random bf16 inputs. Helpers: a per-segment reference that compresses
# each sample alone, and a process-group-free CP simulation that slices boundary rows from the
# neighbouring rank, compacts each rank's rows, compresses them, and maps the rank-major result
# back to sequence order. Tests are skipped unless the fp8 kernels import and the build is ROCm.
"""Tests for the DeepSeek-V4 compressor under THD packing and Context Parallel.

Compressing a packed stream must equal compressing each sample alone, and compacting it per CP
rank then mapping the gathered rows back must equal compressing it once.

Not in CI: the compressor imports the fp8 kernels, and its bf16-in/fp32-out matmul has no CPU
kernel outside a ROCm build. The CP collectives are slicing and concatenation here; the real
ones are checked by tests/fast-gpu/test_dsv4_thd_cp_correctness.py.
"""

from types import SimpleNamespace

import pytest
import torch

from miles_plugins.models.deepseek_v4.ops.thd_utils import (
    CompressorInputCompact,
    ThdLayout,
    compact_gather_index,
    compact_group_capacity,
    compressed_cu_seqlens,
    compressed_rank_layout,
    compressor_boundary_width,
)

try:
    from miles_plugins.models.deepseek_v4.ops.compressor import DeepSeekV4Compressor
except ImportError:
    DeepSeekV4Compressor = None


def requires_fp8_kernels():
    # [FUNCTION DOC] Returns a pytest skip marker that is active when the compressor class could
    # not be imported (it is set to none in that case), with the reason "fp8 kernels not installed".
    return pytest.mark.skipif(DeepSeekV4Compressor is None, reason="fp8 kernels not installed")


def requires_rocm_build():
    # [FUNCTION DOC] Returns a pytest skip marker active when the torch build has no HIP version
    # (not a ROCm build), because the compressor's bf16-in/fp32-out matmul lacks a CPU kernel.
    return pytest.mark.skipif(
        torch.version.hip is None,
        reason="the compressor's bf16 matmul has no CPU kernel outside a ROCm build",
    )


HEAD_DIM = 128
HIDDEN = 256


def _thd(cu, *, max_seqlen=0, compressed_group_ids=None):
    # [FUNCTION DOC] HELPER: Builds a packed-layout descriptor for a single rank starting at
    # global position zero. Takes the cumulative sequence length tensor, optional maximum
    # sequence length (default zero) and optional precomputed compressed group ids; the
    # per-sample lengths are derived as differences of the cumulative lengths.
    """ThdLayout for a single-rank packed stream."""
    return ThdLayout(
        cu_seqlens=cu,
        seq_lens=tuple(torch.diff(cu).tolist()),
        global_start=0,
        max_seqlen=max_seqlen,
        compressed_group_ids=compressed_group_ids,
    )


def _cu(lens):
    # [FUNCTION DOC] HELPER: Converts a list of sample lengths into a 32-bit integer tensor of
    # cumulative offsets with a leading zero (length = number of samples plus one).
    return torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)


def _pad_to(lens, multiple):
    # [FUNCTION DOC] HELPER: Returns a pair of the (possibly extended) length list and the total
    # token count. If the total is not divisible by the multiple, one extra pad segment of the
    # remainder size is appended; otherwise a plain copy is returned. Input is not mutated.
    """Append a pad segment, the way data.py does, so the stream divides evenly."""
    rem = (multiple - sum(lens) % multiple) % multiple
    lens = list(lens) + [rem] if rem else list(lens)
    return lens, sum(lens)


def _compressor(compress_ratio):
    # [FUNCTION DOC] HELPER: Builds a deterministic compressor module for the given compression
    # ratio. Seeds the global random generator with zero, constructs it from a namespace config
    # (no fp8, rope base 160000, original max positions 65536, scaling factor 4, beta fast 32,
    # beta slow 1, position-embedding head dim 64, layernorm epsilon 1e-6) with no rotation, then
    # fills the position bias and the kv and gate projection weights with small normal values
    # (standard deviation 0.02) without gradient tracking, since uninitialized memory could be NaN.
    # Side effect: reseeds the global torch generator.
    torch.manual_seed(0)
    c = DeepSeekV4Compressor(
        config=SimpleNamespace(
            hidden_size=HIDDEN,
            qk_pos_emb_head_dim=64,
            layernorm_epsilon=1e-6,
            fp8=None,
            csa_compress_rotary_base=160000,
            original_max_position_embeddings=65536,
            rotary_scaling_factor=4,
            beta_fast=32,
            beta_slow=1,
        ),
        head_dim=HEAD_DIM,
        compress_ratio=compress_ratio,
        rotate=False,
    )
    # A checkpoint fills these in production; left alone they are raw memory, i.e. NaN.
    with torch.no_grad():
        c.ape.normal_(0, 0.02)
        c.linear_wkv.weight.normal_(0, 0.02)
        c.linear_wgate.weight.normal_(0, 0.02)
    return c


def _packed_input(lens):
    # [FUNCTION DOC] HELPER: Returns a random bf16 tensor shaped total tokens by one by hidden
    # size, generated after seeding the global generator with one (side effect: reseeds it).
    torch.manual_seed(1)
    return torch.randn(sum(lens), 1, HIDDEN, dtype=torch.bfloat16)


def _per_segment_reference(compressor, x, lens):
    # [FUNCTION DOC] HELPER: Reference output for a packed input. For each segment, keeps only
    # the first (length // ratio) * ratio tokens; if that is non-zero, transposes the slice to
    # batch-first, runs the compressor's raw batch-layout forward, transposes back, and collects
    # it. Offsets advance by the full segment length. Returns the concatenation along the token
    # axis, or none if no segment produced output.
    """Compress each segment alone through the BSHD path, then concatenate.

    Trims each segment to a whole number of groups first, mirroring the THD cutoff rule.
    """
    ratio = compressor.compress_ratio
    out, start = [], 0
    for n in lens:
        cutoff = (n // ratio) * ratio
        if cutoff:
            seg = x[start : start + cutoff].transpose(0, 1)
            out.append(compressor.forward_raw(seg).transpose(0, 1))
        start += n
    return torch.cat(out, dim=0) if out else None


def _cp_compress(compressor, x, cu, cp_size, max_seqlen):
    # [FUNCTION DOC] HELPER: Simulates context-parallel compression in one process. The stream is
    # split evenly into cp_size local chunks (local length = total // cp_size) and a fixed
    # compact group capacity is derived from the local length and ratio. For each rank: build a
    # boundary block of the boundary width (zeros); if the rank start is at least that width, copy
    # the preceding rows, else if start is positive, fill the tail of the block with all rows
    # before start. Then compact the local rows plus boundary (returns compacted rows and
    # compressed group ids) and run the packed-layout forward on them. Finally computes the
    # rank-major row map from the cumulative lengths, compressed cumulative lengths and layout
    # parameters, concatenates per-rank outputs and gathers rows by the map with negative
    # entries clamped to zero. Returns (gathered output, mapping); padded slots map to negative.
    # Differentiable w.r.t. x.
    """Run the CP path without a process group.

    The boundary exchange is a slice of what the neighbour owns; the all-gather is a cat.
    """
    ratio = compressor.compress_ratio
    d_comp = compressor_boundary_width(ratio)
    l_local = x.size(0) // cp_size
    c_cap = compact_group_capacity(l_local, ratio)

    per_rank = []
    for cp_rank in range(cp_size):
        start = cp_rank * l_local
        boundary = x.new_zeros((d_comp,) + tuple(x.shape[1:]))
        if start >= d_comp:
            boundary = x[start - d_comp : start]
        elif start > 0:
            boundary[-start:] = x[:start]
        compact, comp_ids = CompressorInputCompact.apply(x[start : start + l_local], boundary, cu, start, ratio, c_cap)
        per_rank.append(
            compressor._forward_thd(compact, _thd(cu, max_seqlen=max_seqlen, compressed_group_ids=comp_ids))
        )

    mapping = compressed_rank_layout(
        cu, compressed_cu_seqlens(cu, ratio), l_local=l_local, cp_size=cp_size, ratio=ratio, c_cap=c_cap
    )
    rank_major = torch.cat(per_rank, dim=0)
    return rank_major.index_select(0, mapping.clamp(min=0).long()), mapping


PACKED_SHAPES = [
    pytest.param(4, [8, 8], id="r4-exact-multiples"),
    pytest.param(4, [10, 7], id="r4-both-with-a-tail"),
    pytest.param(4, [40, 24, 64], id="r4-three-segments"),
    pytest.param(128, [3 * 128 + 5, 2 * 128, 128 + 127], id="r128-mixed-tails"),
    pytest.param(128, [128, 300], id="r128-exact-then-tail"),
]


@requires_fp8_kernels()
@requires_rocm_build()
@pytest.mark.parametrize("ratio,lens", PACKED_SHAPES)
def test_packed_matches_per_segment(ratio, lens):
    # [FUNCTION DOC] Compresses a packed stream in one call and compares with the per-segment
    # reference using loose tolerances (relative and absolute 0.02 after casting to float32).
    # Parametrized by ratio (4 or 128) and segment length lists with exact multiples, tails and
    # mixed lengths.
    """The oracle is the BSHD path, a separate implementation, so this one needs a tolerance.
    The CP checks below compare two runs of the same path and must agree bit for bit.
    """
    compressor = _compressor(ratio)
    x = _packed_input(lens)
    packed = compressor._forward_thd(x, _thd(_cu(lens)))
    reference = _per_segment_reference(compressor, x, lens)
    torch.testing.assert_close(packed.float(), reference.float(), rtol=2e-2, atol=2e-2)


CP_SHAPES = [
    pytest.param(4, [10, 22], id="r4-none-divisible"),
    pytest.param(4, [3, 29], id="r4-first-shorter-than-ratio"),
    pytest.param(4, [32], id="r4-single-segment-spans-all-ranks"),
    pytest.param(4, [4, 4, 4, 20], id="r4-more-segments-than-ranks"),
    pytest.param(4, [16, 16], id="r4-segment-edge-on-the-cp-split"),
    pytest.param(4, [1, 1, 1, 29], id="r4-several-segments-shorter-than-ratio"),
    pytest.param(4, [31, 1], id="r4-tail-segment-of-one-token"),
    pytest.param(128, [512, 256, 1280], id="r128-uniform"),
    pytest.param(128, [3, 1789], id="r128-first-shorter-than-ratio"),
    pytest.param(128, [1536, 512, 7], id="r128-short-tail"),
]


@requires_fp8_kernels()
@requires_rocm_build()
@pytest.mark.parametrize("ratio,lens", CP_SHAPES)
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_cp_chain_matches_whole_stream(ratio, lens, cp_size):
    # [FUNCTION DOC] Pads the lengths so the total is divisible by cp_size times ratio, computes
    # the whole-stream result, skips the test if that result is none (no segment reaches the
    # ratio), then runs the simulated CP path. Asserts the whole-stream row count equals the
    # total compressed length, and the first that-many gathered rows equal the reference exactly
    # (zero tolerance). Parametrized over CP sizes 1, 2 and 4 and the shape list.
    """Compacting per rank, compressing, then mapping the gathered rows back to sequence order
    must equal compressing the packed stream once.

    Covers the compaction, the fixed capacity, comp_ids driving RoPE and the overlap transform,
    and the rank-major row map. Trailing tokens get no compressed entry, so none may be invented.
    """
    lens, _ = _pad_to(lens, cp_size * ratio)
    cu = _cu(lens)
    compressor = _compressor(ratio)
    x = _packed_input(lens)
    reference = compressor._forward_thd(x, _thd(cu))
    if reference is None:
        pytest.skip("no segment reaches the ratio, nothing to compare")
    got, _ = _cp_compress(compressor, x, cu, cp_size, max(lens))
    total_comp = int(compressed_cu_seqlens(cu, ratio)[-1])
    assert reference.shape[0] == total_comp
    torch.testing.assert_close(got[:total_comp], reference, rtol=0, atol=0)


@requires_fp8_kernels()
@requires_rocm_build()
@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_chain_backward_matches_whole_stream(cp_size):
    # [FUNCTION DOC] Uses ratio 4 and lengths 40, 24, 64 with CP sizes 2 and 4. Builds one float32
    # input, backpropagates the sum of the whole-stream output into a clone, and the sum of the
    # CP-simulated output into another clone, then asserts the two input gradients are close
    # (relative 1e-5, absolute 1e-6).
    """The compaction's scatter must send each source row's gradient back where it came from.

    Unlike the forward, this needs a tolerance: the scatter is an index_add_, and the two paths
    accumulate into a row in a different order.
    """
    ratio, lens = 4, [40, 24, 64]
    cu = _cu(lens)
    compressor = _compressor(ratio)
    base = _packed_input(lens).float()

    ref_x = base.clone().requires_grad_(True)
    compressor._forward_thd(ref_x, _thd(cu)).sum().backward()

    cp_x = base.clone().requires_grad_(True)
    _cp_compress(compressor, cp_x, cu, cp_size, max(lens))[0].sum().backward()

    torch.testing.assert_close(cp_x.grad, ref_x.grad, rtol=1e-5, atol=1e-6)


@requires_fp8_kernels()
@requires_rocm_build()
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_short_segments_produce_no_group(cp_size):
    # [FUNCTION DOC] With ratio 128 and segments of three full groups, one token short of a group,
    # and five tokens (padded to a multiple of cp_size times ratio), asserts the first compressed
    # cumulative offsets are zero, three, three; that the CP mapping has exactly as many
    # non-negative entries as the total compressed length; and that a packed input of two segments
    # (5 and 7 tokens) yields none from the packed forward.
    """A segment shorter than `ratio` contributes nothing, and a rank holding only such
    segments emits zero real groups rather than a padded one.
    """
    ratio = 128
    lens, _ = _pad_to([3 * ratio, ratio - 1, 5], cp_size * ratio)
    cu = _cu(lens)
    cu_comp = compressed_cu_seqlens(cu, ratio)
    assert cu_comp.tolist()[:3] == [0, 3, 3]

    _, mapping = _cp_compress(_compressor(ratio), _packed_input(lens), cu, cp_size, max(lens))
    assert int((mapping >= 0).sum()) == int(cu_comp[-1])

    assert _compressor(ratio)._forward_thd(_packed_input([5, 7]), _thd(_cu([5, 7]))) is None


@requires_fp8_kernels()
@requires_rocm_build()
def test_pre_grouped_requires_max_seqlen():
    # [FUNCTION DOC] Builds an already-compacted zero input and group ids for a one-rank stream
    # (ratio 4, lengths 40 and 24), then asserts the packed forward with those ids but a maximum
    # sequence length of none raises a value error whose message mentions the max sequence length.
    """Pre-grouped input needs max_seqlen: its group ids address positions past the compacted
    row count, so falling back to x.size(0) would index out of the rope table.
    """
    ratio, lens = 4, [40, 24]
    cu = _cu(lens)
    c_cap = compact_group_capacity(sum(lens), ratio)
    _, comp_ids = compact_gather_index(cu, global_start=0, l_local=sum(lens), ratio=ratio, c_cap=c_cap)
    compact = torch.zeros(c_cap * ratio, 1, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="max_seqlen"):
        _compressor(ratio)._forward_thd(compact, _thd(cu, max_seqlen=None, compressed_group_ids=comp_ids))
